import os
import json
import base64
import asyncio
import threading
import logging
from datetime import datetime, timezone
import uuid
import math
import time
import cv2
import torch
import numpy as np
import csv
from typing import Optional
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request, Header, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import live_camera_manager
from detection.yolo_module import YOLODetector
from tracking.deepsort_module import VehicleTracker
from phases.phase_a_proximity import proximity_filter
from phases.phase_b_trajectory import (
    analyze_trajectory_conflict,
    is_stationary,
    was_recently_moving,
    compute_iou,
)
from phases.phase_c_anomaly import analyze_anomaly
from utils.optical_flow import compute_optical_flow, calculate_frame_diff_ratio
from utils.geometry import calculate_bbox_containment_ratio
from fusion.scoring import fuse_scores
from model import model, transform, DEVICE, predict_image, SEQUENCE_LEN, frame_to_tensor_fast
from llm_vision_module import analyze_frame_with_llm
from utils.incident_clip import create_video_writer, transcode_video_for_browser, extract_clip_from_file
from utils import incident_store
from utils.task_pool import submit as submit_bg
from firebase_uploader import FirebaseUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UyirApp")

app = FastAPI()
_firebase = FirebaseUploader()
_firebase.retry_local_events()


def require_api_key(x_api_key: str = Header(default=None)):
    """
    Stopgap auth for mutating endpoints: a shared secret required via the
    "X-API-Key" header, checked against config.API_KEY (UYIR_API_KEY env
    var). This is NOT a real auth system — no identity, no roles, no audit
    trail of who acted. It only stops an anonymous person on the network
    segment from wiping incident history, retraining the refinement model,
    or kicking off arbitrary video-processing jobs. Replace with proper
    login/role-based access before handing this over to police IT staff.
    """
    if not x_api_key or x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return True

# ================= XGBOOST MODEL INTEGRATION =================
CSV_FILE = "accident_features.csv"
XGB_MODEL_PATH = "model_output/accident_xgboost.json"
xgb_clf = None

def init_csv_file():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "proximity", "trajectory", "anomaly", "cnn",
                "occlusion", "merge", "kinetic", "density",
                "avg_speed", "stopped_ratio", "label"
            ])

XGB_MIN_ROWS_PER_CLASS = 50

def load_xgboost_model():
    global xgb_clf
    if os.path.exists(XGB_MODEL_PATH):
        try:
            if os.path.exists(CSV_FILE):
                import pandas as pd
                df = pd.read_csv(CSV_FILE)
                if "label" in df.columns:
                    counts = df["label"].value_counts()
                    n0 = int(counts.get(0, 0))
                    n1 = int(counts.get(1, 0))
                    if n0 < XGB_MIN_ROWS_PER_CLASS or n1 < XGB_MIN_ROWS_PER_CLASS:
                        xgb_clf = None
                        return
            from xgboost import XGBClassifier
            XGBClassifier._estimator_type = "classifier"
            xgb_clf = XGBClassifier()
            xgb_clf.load_model(XGB_MODEL_PATH)
            logger.info(f"Loaded XGBoost classifier from {XGB_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            xgb_clf = None
    else:
        xgb_clf = None

init_csv_file()
load_xgboost_model()


class FeatureLog(BaseModel):
    proximity: float
    trajectory: float
    anomaly: float
    cnn: float
    occlusion: float
    merge: float
    kinetic: float
    density: float
    avg_speed: float
    stopped_ratio: float
    label: int


def _phases_from_details(details):
    phases = []
    if details.get("phase_a_confirmed") or (details.get("proximity_score", 0) or details.get("ttc_score", 0)) > 0.3:
        phases.append("phase_a")
    if details.get("phase_b_confirmed") or (details.get("trajectory_score", 0) or details.get("trajectory_stop_score", 0)) > 0.3:
        phases.append("phase_b")
    if details.get("phase_c_confirmed") or details.get("flow_score", 0) > 0.3:
        phases.append("phase_c")
    return phases or ["phase_a"]


def _save_incident_records(stubs, input_path, source_fps, total_frames, location, source="web"):
    """Save suspicious/suppressed stubs to disk at end of processing."""
    saved_incidents = []
    for stub in stubs:
        incident_id = str(uuid.uuid4())
        clip_fs, snap_fs, clip_url, snap_url = incident_store.build_incident_paths(incident_id)
        cv2.imwrite(snap_fs, stub["snapshot_frame"])
        clip_ok = extract_clip_from_file(
            input_path, stub["source_frame_idx"], source_fps, clip_fs,
            total_frames=total_frames,
        )
        details = stub["details"]
        details.setdefault("lstm_peak", stub["dl_confidence"])
        status = stub.get("status", "confirmed")
        llm_text = analyze_frame_with_llm(snap_fs, details) if status == "confirmed" else None
        record = incident_store.save_incident({
            "id": incident_id,
            "source": source,
            "camera_id": "UPLOAD",
            "location": location,
            "frame_number": stub["source_frame_idx"],
            "time_in_video_sec": round(stub["source_frame_idx"] / source_fps, 2),
            "confidence": stub["confidence"],
            "dl_confidence": stub["dl_confidence"],
            "trigger_phase": stub["trigger_phase"],
            "phases_triggered": _phases_from_details(details),
            "clip_url": clip_url if clip_ok else None,
            "snapshot_url": snap_url,
            "llm_analysis": llm_text,
            "details": details,
            "status": status,
        })
        _firebase.upload_incident_record_async(record)
        saved_incidents.append(record)
    return saved_incidents


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ================= STREAMING STATE =================
_stream_jobs = {}
_stream_lock = threading.Lock()

def _new_job(job_id):
    with _stream_lock:
        _stream_jobs[job_id] = {
            "frames": [],
            "done": False,
            "error": None,
            "result": None,
            "incidents": [],
            "clip_events": [],   # clip_ready notifications pushed after async extraction
        }

def _push_frame(job_id, jpeg_bytes, metrics):
    with _stream_lock:
        if job_id in _stream_jobs:
            _stream_jobs[job_id]["frames"].append((jpeg_bytes, metrics))

def _finish_job(job_id, result=None, error=None):
    with _stream_lock:
        if job_id in _stream_jobs:
            _stream_jobs[job_id]["done"] = True
            _stream_jobs[job_id]["result"] = result
            _stream_jobs[job_id]["error"] = error

def _push_incident(job_id, incident):
    with _stream_lock:
        if job_id in _stream_jobs:
            _stream_jobs[job_id]["incidents"].append(incident)

def _push_clip_ready(job_id, incident_id, clip_url):
    """Notify the SSE client that a clip is ready for a previously pushed incident."""
    with _stream_lock:
        if job_id in _stream_jobs:
            _stream_jobs[job_id]["clip_events"].append({
                "incident_id": incident_id,
                "clip_url": clip_url,
                "_sent": False,
            })

# ================= SSE STREAM ENDPOINT =================
@app.get("/stream/{job_id}")
async def stream_job(job_id: str):
    async def event_generator():
        sent = 0
        while True:
            with _stream_lock:
                job = _stream_jobs.get(job_id)
            if job is None:
                yield "data: {\"error\": \"Job not found\"}\n\n"
                return

            # Push new annotated frames
            frames = job["frames"]
            while sent < len(frames):
                jpeg_bytes, metrics = frames[sent]
                b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                payload = json.dumps({"frame": b64, "metrics": metrics})
                yield f"data: {payload}\n\n"
                sent += 1

            # Push any new incident events (confirmed/suspicious/suppressed)
            with _stream_lock:
                incidents_list = _stream_jobs.get(job_id, {}).get("incidents", [])
            for inc in incidents_list:
                if not inc.get("_sent"):
                    inc["_sent"] = True
                    payload = json.dumps({"incident": inc})
                    yield f"data: {payload}\n\n"

            # Push any new clip_ready events (confirmed clips extracted in background)
            with _stream_lock:
                clip_events = _stream_jobs.get(job_id, {}).get("clip_events", [])
            for evt in clip_events:
                if not evt.get("_sent"):
                    evt["_sent"] = True
                    payload = json.dumps({
                        "clip_ready": {
                            "incident_id": evt["incident_id"],
                            "clip_url": evt["clip_url"],
                        }
                    })
                    yield f"data: {payload}\n\n"

            if job["done"]:
                result = job.get("result") or {}
                payload = json.dumps({"done": True, "result": result})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(30)
                with _stream_lock:
                    _stream_jobs.pop(job_id, None)
                return

            await asyncio.sleep(0.04)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ================= START STREAMING VIDEO JOB =================
@app.post("/start-stream")
async def start_stream(file: UploadFile = File(...), threshold: float = 0.55,
                        _auth: bool = Depends(require_api_key)):
    if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/quicktime"]:
        return JSONResponse(status_code=400, content={"error": "Invalid video format"})

    file_ext = os.path.splitext(file.filename)[1]
    raw_filename = f"{uuid.uuid4()}{file_ext}"
    input_path = os.path.join(UPLOAD_DIR, raw_filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    job_id = str(uuid.uuid4())
    _new_job(job_id)

    thread = threading.Thread(
        target=_process_video_streaming,
        args=(job_id, input_path, file.filename or "video", threshold),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


def _encode_frame_jpeg(frame, quality=70):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _process_video_streaming(job_id, input_path, filename, threshold):
    """
    Background thread: processes video frame-by-frame, pushes JPEG frames
    and detection metrics to the SSE channel, simulating a live camera feed.

    KEY BEHAVIOURS:
      - Confirmed accidents: snapshot saved immediately, incident pushed to SSE
        with real id, clip extracted in a parallel background thread, clip_ready
        event pushed when done. No waiting until video ends.
      - Suspicious/suppressed events: pushed to SSE immediately but saved to disk
        only at the end of the video.
      - SSE status = "confirmed" ONLY when the consecutive-frame gate passes.
        Pipeline evaluation that is building up shows as "pending".
      - dl_raw  = cnn_lstm_prob for the current frame (actual model output).
      - dl_peak = rolling max of last 30 scores (used for gating, shown as gate state).
      - fused_score = the weighted fusion score for the current frame.
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            _finish_job(job_id, error="Cannot open video file")
            return

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        MAX_WIDTH = 800
        if width > MAX_WIDTH:
            scale  = MAX_WIDTH / float(width)
            width  = MAX_WIDTH
            height = int(height * scale)

        PROCESS_EVERY_N_FRAMES = 2
        output_fps = fps / PROCESS_EVERY_N_FRAMES

        processed_filename = f"processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(UPLOAD_DIR, processed_filename)
        try:
            out_writer = create_video_writer(output_path, output_fps, width, height)
        except RuntimeError as e:
            _finish_job(job_id, error=str(e))
            return

        tracker = VehicleTracker()

        prev_gray           = None
        features_buffer     = []
        lstm_scores_history = []
        scene_speed_history = []

        source_frame_idx           = -1
        frame_idx                  = 0
        accident_detected_globally = False
        max_accident_score         = 0.0
        triggering_phase_globally  = "None"
        accident_details           = {}
        max_accident_frame_idx     = 0
        last_alert_source_frame    = -999999
        last_alt_alert_frame       = -999999
        last_alt_status            = None
        consec_count               = 0
        incident_stubs             = []    # suspicious/suppressed only
        all_inline_saved           = []    # confirmed incidents saved immediately
        bg_threads                 = []    # background clip-extraction futures
        source_fps                 = fps
        cooldown_frames            = max(1, int(config.COOLDOWN_SECONDS * source_fps))

        zone_x_min = int(width * 0.25);  zone_x_max = int(width * 0.75)
        zone_y_min = int(height * 0.25); zone_y_max = int(height * 0.75)

        DL_GATE_THRESHOLD = config.DL_GATE_THRESHOLD    # 0.55
        PHASE_SIGNAL_MIN  = config.DL_PHASE_SIGNAL_MIN  # 0.30
        WARMUP_FRAMES     = config.DL_WARMUP_FRAMES     # 16

        t_start          = time.time()
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            source_frame_idx += 1
            if source_frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                continue

            if frame.shape[1] > MAX_WIDTH:
                frame = cv2.resize(frame, (width, height))

            frames_processed += 1
            elapsed  = time.time() - t_start
            proc_fps = frames_processed / elapsed if elapsed > 0.5 else 0.0

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── DL inference ──────────────────────────────────────────
            # frame_to_tensor_fast() replaces the old PIL round-trip
            # (cvtColor -> Image.fromarray -> transform) with a direct
            # cv2-based resize + normalize — same 240x240 ImageNet-normalized
            # input the checkpoint expects, just without the PIL overhead.
            frame_feat_input = frame_to_tensor_fast(frame)
            with torch.no_grad():
                feat = model.cnn(frame_feat_input)

            features_buffer.append(feat)
            if len(features_buffer) > SEQUENCE_LEN:
                features_buffer.pop(0)

            # Pad with the LAST buffered feature (matches model.py training convention)
            if len(features_buffer) < SEQUENCE_LEN:
                last_feat = features_buffer[-1]
                padded = [last_feat] * (SEQUENCE_LEN - len(features_buffer)) + features_buffer
            else:
                padded = features_buffer

            features_tensor = torch.stack(padded, dim=1)
            with torch.no_grad():
                lstm_out, _ = model.bilstm(features_tensor)
                context      = model.attention(lstm_out)
                logits       = model.classifier(context)
                probs        = torch.softmax(logits, dim=1)
                cnn_lstm_prob = float(probs[0, 1].item())   # raw per-frame score

            # Rolling peak with warmup guard
            lstm_scores_history.append(cnn_lstm_prob)
            if len(lstm_scores_history) > 30:
                lstm_scores_history.pop(0)

            warmup_done = len(features_buffer) >= WARMUP_FRAMES
            lstm_peak = max(lstm_scores_history) if warmup_done else cnn_lstm_prob

            # ── YOLO + ByteTrack ──────────────────────────────────────
            active_tracks = tracker.update(frame=frame)

            # ── Optical flow ──────────────────────────────────────────
            flow = (
                compute_optical_flow(prev_gray, frame_gray, scale=config.OPTICAL_FLOW_SCALE)
                if prev_gray is not None else None
            )

            # ── Scene speed ───────────────────────────────────────────
            mean_current_speed = 0.0
            if active_tracks:
                speeds = [
                    math.sqrt(t.velocities[-1][0]**2 + t.velocities[-1][1]**2)
                    for t in active_tracks if t.velocities
                ]
                mean_current_speed = sum(speeds) / len(speeds) if speeds else 0.0
            scene_speed_history.append(mean_current_speed)
            if len(scene_speed_history) > 120:
                scene_speed_history.pop(0)

            scene_interruption_score = 0.0
            if len(scene_speed_history) >= 20:
                bw = max(5, len(scene_speed_history) // 4)
                base_spd   = sum(scene_speed_history[:bw]) / bw
                recent_spd = sum(scene_speed_history[-5:]) / 5
                if base_spd > 1.0:
                    drop = (base_spd - recent_spd) / base_spd
                    scene_interruption_score = min(1.0, max(0.0, drop / 0.5)) if drop > 0.2 else 0.0

            # ── Phase signals ─────────────────────────────────────────
            candidate_pairs = proximity_filter(active_tracks)
            ttc_score = trajectory_stop_score = emergency_stop_score = 0.0
            relative_velocity_score = trajectory_score = anomaly_score = 0.0
            occlusion_score = merge_score = energy_drop_score = spin_score = 0.0
            flow_dispersion_score = diff_burst_score = 0.0
            frame_collision_pairs = []
            is_in_intersection_zone = False

            # Phase C — per track
            for track in active_tracks:
                anom_res = analyze_anomaly(track, flow)
                anomaly_score         = max(anomaly_score, anom_res["anomaly_score"])
                flow_dispersion_score = max(flow_dispersion_score, anom_res["dispersion_val"])
                diff_ratio = calculate_frame_diff_ratio(prev_gray, frame_gray, track.bbox)
                diff_burst_score = max(diff_burst_score, 1.0 if diff_ratio > 0.15 else diff_ratio / 0.15)

            # Phase A + B — per pair
            for t1, t2, dist, pair_ttc in candidate_pairs:
                if t1.age < 2 or t2.age < 2:
                    continue

                # Skip only permanently-parked pairs, not post-crash stopped ones
                both_stationary = is_stationary(t1) and is_stationary(t2)
                if both_stationary:
                    if not was_recently_moving(t1) and not was_recently_moving(t2):
                        continue

                pair_threshold = (
                    config.PROXIMITY_PERSON_THRESHOLD
                    if (t1.label == config.PERSON_CLASS or t2.label == config.PERSON_CLASS)
                    else config.PROXIMITY_THRESHOLD
                )
                prox_s    = max(0.0, 1.0 - (dist / pair_threshold), pair_ttc)
                ttc_score = max(ttc_score, prox_s, pair_ttc)

                traj_res = analyze_trajectory_conflict(t1, t2)
                trajectory_score        = max(trajectory_score, traj_res["score"],
                                              traj_res["trajectory_stop_score"],
                                              traj_res["emergency_stop_score"],
                                              traj_res["relative_velocity_score"])
                trajectory_stop_score   = max(trajectory_stop_score, traj_res["trajectory_stop_score"])
                emergency_stop_score    = max(emergency_stop_score,  traj_res["emergency_stop_score"])
                relative_velocity_score = max(relative_velocity_score, traj_res["relative_velocity_score"])
                occlusion_score = max(
                    occlusion_score,
                    traj_res["containment"] if traj_res["occluded"] else traj_res["containment"] * 0.5
                )
                if traj_res["merged"]:
                    merge_score = 1.0
                energy_drop_score = max(energy_drop_score, traj_res["max_ke_drop"], traj_res["emergency_stop_score"])
                spin_score        = max(spin_score, traj_res["max_spin_var"])

                c1, c2 = t1.get_centroid(), t2.get_centroid()
                cx, cy = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                if zone_x_min <= cx <= zone_x_max and zone_y_min <= cy <= zone_y_max:
                    is_in_intersection_zone = True
                if traj_res["class"] == "Collision" or traj_res["merged"] or traj_res["occluded"]:
                    frame_collision_pairs.append((t1, t2, dist, traj_res["class"]))

            prev_gray = frame_gray

            stopped = sum(
                1 for t in active_tracks
                if t.velocities and math.sqrt(t.velocities[-1][0]**2 + t.velocities[-1][1]**2) < 2.0
            )
            stopped_ratio   = stopped / len(active_tracks) if active_tracks else 0.0
            traffic_density = min(len(active_tracks) / 20.0, 1.0)

            # ── Option 2 pipeline ─────────────────────────────────────
            phase_a_signal = ttc_score
            phase_b_signal = max(trajectory_stop_score, emergency_stop_score, relative_velocity_score)
            phase_c_signal = anomaly_score
            dl_confirmed   = lstm_peak >= DL_GATE_THRESHOLD

            phases_signalling = 0
            phases_detail     = {}
            if dl_confirmed:
                if phase_a_signal >= PHASE_SIGNAL_MIN:
                    phases_signalling += 1; phases_detail["phase_a"] = True
                if phase_b_signal >= PHASE_SIGNAL_MIN:
                    phases_signalling += 1; phases_detail["phase_b"] = True
                if phase_c_signal >= PHASE_SIGNAL_MIN:
                    phases_signalling += 1; phases_detail["phase_c"] = True

            # ── Detection status per frame ────────────────────────────
            # "pending"  = pipeline evaluates as confirmed (DL + 2 phases) but consecutive gate not yet met
            # "confirmed" = only sent in SSE when confirmed_accident fires
            if not dl_confirmed:
                frame_score      = float(cnn_lstm_prob)
                frame_accident   = False
                frame_trigger    = f"DL Gate Not Cleared ({lstm_peak:.2f})"
                detection_status = "scanning"
                fuse_res         = None

            elif phases_signalling >= 2:
                fuse_res = fuse_scores(
                    trajectory_stop=trajectory_stop_score, ttc_critical=ttc_score,
                    emergency_stop=emergency_stop_score, cnn_lstm=lstm_peak,
                    optical_flow=anomaly_score, flow_dispersion=flow_dispersion_score,
                    scene_density=len(active_tracks), avg_scene_speed=mean_current_speed,
                    stopped_ratio=stopped_ratio, threshold=threshold,
                )
                frame_score      = max(fuse_res["score"], lstm_peak * 0.8)
                frame_accident   = True
                frame_trigger    = (
                    f"DL + {' & '.join(p.replace('phase_','Phase ').upper() for p in phases_detail)} Verified"
                )
                # "pending" until consecutive gate confirms — prevents premature SSE "confirmed"
                detection_status = "pending"

            elif phases_signalling == 1:
                fuse_res = fuse_scores(
                    trajectory_stop=trajectory_stop_score, ttc_critical=ttc_score,
                    emergency_stop=emergency_stop_score, cnn_lstm=lstm_peak,
                    optical_flow=anomaly_score, flow_dispersion=flow_dispersion_score,
                    scene_density=len(active_tracks), avg_scene_speed=mean_current_speed,
                    stopped_ratio=stopped_ratio, threshold=threshold,
                )
                frame_score      = fuse_res["score"] * 0.7
                frame_accident   = False
                frame_trigger    = "DL Confirmed, 1 Phase Only"
                detection_status = "suspicious"

            else:
                fuse_res = fuse_scores(
                    trajectory_stop=trajectory_stop_score, ttc_critical=ttc_score,
                    emergency_stop=emergency_stop_score, cnn_lstm=lstm_peak,
                    optical_flow=anomaly_score, flow_dispersion=flow_dispersion_score,
                    scene_density=len(active_tracks), avg_scene_speed=mean_current_speed,
                    stopped_ratio=stopped_ratio, threshold=threshold,
                )
                frame_score      = fuse_res["score"] * 0.5
                frame_accident   = False
                frame_trigger    = "DL Confirmed, No Phase Signal"
                detection_status = "suppressed"

            # XGBoost refinement
            if dl_confirmed and phases_signalling >= 2 and xgb_clf is not None:
                try:
                    feats = [[
                        float(ttc_score), float(trajectory_stop_score), float(anomaly_score),
                        float(lstm_peak), float(occlusion_score), float(merge_score),
                        float(emergency_stop_score), float(traffic_density),
                        float(mean_current_speed), float(stopped_ratio),
                    ]]
                    xgb_prob = float(xgb_clf.predict_proba(feats)[0][1])
                    if xgb_prob < 0.35:
                        frame_score  *= 0.7
                        frame_trigger += " (XGB↓)"
                    else:
                        frame_score   = max(frame_score, xgb_prob)
                        frame_trigger += " & XGB"
                    if frame_score < threshold:
                        frame_accident = False
                except Exception:
                    pass

            if frame_accident and is_in_intersection_zone:
                frame_score   = min(1.0, frame_score * 1.15)
                frame_trigger += " & Risk Zone"

            frame_details = {
                "proximity_score":         float(phase_a_signal),
                "trajectory_score":        float(phase_b_signal),
                "flow_score":              float(phase_c_signal),
                "ttc_score":               float(ttc_score),
                "trajectory_stop_score":   float(trajectory_stop_score),
                "emergency_stop_score":    float(emergency_stop_score),
                "relative_velocity_score": float(relative_velocity_score),
                "energy_drop":             float(energy_drop_score),
                "occlusion_score":         float(occlusion_score),
                "merge_score":             float(merge_score),
                "spin_score":              float(spin_score),
                "lstm_peak":               float(lstm_peak),
                "cnn_lstm_prob":           float(cnn_lstm_prob),
                "traffic_density":         float(traffic_density),
                "avg_speed":               float(mean_current_speed),
                "stopped_ratio":           float(stopped_ratio),
                "scene_interruption":      float(scene_interruption_score),
                "dl_confirmed":            bool(dl_confirmed),
                "phases_signalling":       int(phases_signalling),
                "phase_a_confirmed":       bool(phases_detail.get("phase_a", False)),
                "phase_b_confirmed":       bool(phases_detail.get("phase_b", False)),
                "phase_c_confirmed":       bool(phases_detail.get("phase_c", False)),
                "post_intersect_static":   False,
            }

            # ── Consecutive gate + cooldown ───────────────────────────
            cooldown_ok = (source_frame_idx - last_alert_source_frame) > cooldown_frames
            if frame_accident and cooldown_ok:
                consec_count += 1
            else:
                if not frame_accident:
                    consec_count = max(0, consec_count - 1)

            confirmed_accident = (
                frame_accident
                and consec_count >= config.CONSECUTIVE_FRAMES
                and cooldown_ok
            )

            if frame_score > max_accident_score:
                max_accident_score        = frame_score
                triggering_phase_globally = frame_trigger
                accident_details          = dict(frame_details)
                max_accident_frame_idx    = source_frame_idx

            # ── Handle confirmed accident — save immediately ───────────
            if confirmed_accident:
                accident_detected_globally = True
                last_alert_source_frame    = source_frame_idx
                consec_count               = 0

                # Save snapshot to disk right now
                _inc_id    = str(uuid.uuid4())
                _clip_fs, _snap_fs, _clip_url, _snap_url = incident_store.build_incident_paths(_inc_id)
                cv2.imwrite(_snap_fs, frame.copy())

                _det_copy = dict(frame_details)
                _det_copy.setdefault("lstm_peak", lstm_peak)

                # Persist incident record immediately (clip_url = None yet)
                _record = incident_store.save_incident({
                    "id":               _inc_id,
                    "source":           "web",
                    "camera_id":        "UPLOAD",
                    "location":         filename,
                    "timestamp":        datetime.now(timezone.utc).isoformat(),
                    "frame_number":     source_frame_idx,
                    "time_in_video_sec": round(source_frame_idx / source_fps, 2),
                    "confidence":       float(frame_score),
                    "dl_confidence":    float(lstm_peak),
                    "trigger_phase":    frame_trigger,
                    "phases_triggered": _phases_from_details(_det_copy),
                    "clip_url":         None,
                    "snapshot_url":     _snap_url,
                    "llm_analysis":     None,
                    "details":          _det_copy,
                    "status":           "confirmed",
                })
                all_inline_saved.append(_record)
                _firebase.upload_incident_record_async(_record)

                # Push to SSE immediately with snapshot — clip will follow via clip_ready
                _push_incident(job_id, {
                    "id":            _inc_id,
                    "status":        "confirmed",
                    "confidence":    float(frame_score),
                    "dl_confidence": float(lstm_peak),
                    "trigger_phase": frame_trigger,
                    "frame":         source_frame_idx,
                    "time_sec":      round(source_frame_idx / source_fps, 2),
                    "snapshot_url":  _snap_url,
                    "clip_url":      None,
                    "details":       _det_copy,
                    "_sent":         False,
                })

                # Background: extract clip then run LLM, push clip_ready when done.
                # Routed through the shared bounded pool (config.MAX_BACKGROUND_WORKERS)
                # instead of a raw threading.Thread per incident — see utils/task_pool.py.
                def _bg_clip_llm(jid, iid, snap_fs, details_c,
                                  src, fr_idx, fps_v, tot, clip_fs, clip_url_v):
                    try:
                        _llm = analyze_frame_with_llm(snap_fs, details_c)
                        incident_store.update_incident(iid, {"llm_analysis": _llm})
                    except Exception as _ex:
                        logger.warning(f"[BG LLM] {iid}: {_ex}")
                    try:
                        _ok = extract_clip_from_file(
                            src, fr_idx, fps_v, clip_fs, total_frames=tot
                        )
                        if _ok:
                            incident_store.update_incident(iid, {"clip_url": clip_url_v})
                            _push_clip_ready(jid, iid, clip_url_v)
                    except Exception as _ex:
                        logger.warning(f"[BG clip] {iid}: {_ex}")

                _bg_future = submit_bg(
                    _bg_clip_llm,
                    job_id, _inc_id, _snap_fs, dict(frame_details),
                    input_path, source_frame_idx, source_fps, total_frames,
                    _clip_fs, _clip_url
                )
                bg_threads.append(_bg_future)

            # ── Handle suspicious / suppressed — push to SSE, save later ──
            elif dl_confirmed and not frame_accident and cooldown_ok and detection_status in ("suspicious", "suppressed"):
                alt_cooldown_ok = (source_frame_idx - last_alt_alert_frame) > cooldown_frames
                status_changed  = detection_status != last_alt_status
                if alt_cooldown_ok or status_changed:
                    last_alt_alert_frame = source_frame_idx
                    last_alt_status      = detection_status
                    _push_incident(job_id, {
                        "status":        detection_status,
                        "confidence":    float(frame_score),
                        "dl_confidence": float(lstm_peak),
                        "trigger_phase": frame_trigger,
                        "frame":         source_frame_idx,
                        "time_sec":      round(source_frame_idx / source_fps, 2),
                        "details":       dict(frame_details),
                        "_sent":         False,
                    })
                    incident_stubs.append({
                        "source_frame_idx": source_frame_idx,
                        "confidence":       float(frame_score),
                        "dl_confidence":    float(lstm_peak),
                        "trigger_phase":    frame_trigger,
                        "details":          dict(frame_details),
                        "snapshot_frame":   frame.copy(),
                        "status":           detection_status,
                    })

            # ── Frame annotations ─────────────────────────────────────
            cv2.rectangle(frame, (zone_x_min, zone_y_min), (zone_x_max, zone_y_max), (255, 255, 255), 1)
            cv2.putText(frame, "RISK ZONE", (zone_x_min + 5, zone_y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                color = (0, 255, 0)
                lbl   = f"ID{track.track_id}:{track.label[:3].upper()}"
                if hasattr(track, "anomaly_streak") and track.anomaly_streak >= 3:
                    color = (0, 0, 255); lbl += " ANOM"
                elif hasattr(track, "anomaly_streak") and track.anomaly_streak > 0:
                    color = (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, lbl, (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Show collision markers when pipeline is actively detecting
            if frame_accident and frame_collision_pairs:
                for t1, t2, _, _status in frame_collision_pairs:
                    c1, c2 = t1.get_centroid(), t2.get_centroid()
                    pt1 = (int(c1[0]), int(c1[1])); pt2 = (int(c2[0]), int(c2[1]))
                    cx2, cy2 = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                    cv2.circle(frame, (cx2, cy2), 10, (0, 0, 255), -1)

            # Show banner: "CONFIRMED" in red, "DETECTING" in orange (building up)
            if confirmed_accident:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 55), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, f"ACCIDENT CONFIRMED  {frame_score * 100:.1f}%",
                            (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            elif frame_accident:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 55), (0, 100, 200), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(frame,
                            f"DETECTING... {frame_score * 100:.1f}%  [{consec_count}/{config.CONSECUTIVE_FRAMES}]",
                            (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 100), 2)

            hud_lines = [
                f"DL(frame):{cnn_lstm_prob:.2f}  Peak:{lstm_peak:.2f}{'✓' if dl_confirmed else '✗'}",
                f"A:{phase_a_signal:.2f}  B:{phase_b_signal:.2f}  C:{phase_c_signal:.2f}  Votes:{phases_signalling}/3",
                f"Fused:{frame_score:.2f}  FPS:{proc_fps:.1f}",
                f"Tracks:{len(active_tracks)}  Spd:{mean_current_speed:.1f}",
            ]
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (0, 56), (300, 56 + len(hud_lines) * 18 + 6), (10, 10, 30), -1)
            cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)
            for i, line in enumerate(hud_lines):
                cv2.putText(frame, line, (8, 72 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 230, 255), 1)

            out_writer.write(frame)
            frame_idx += 1

            jpeg = _encode_frame_jpeg(frame, quality=65)
            # SSE metrics: dl_raw = per-frame model output, dl_peak = rolling max for gating
            # status = "confirmed" ONLY when confirmed_accident fires, otherwise pipeline label
            metrics = {
                "frame":        source_frame_idx,
                "total_frames": total_frames,
                "progress":     min(100, int(source_frame_idx / max(total_frames, 1) * 100)),
                "fps":          round(proc_fps, 1),
                "dl_raw":       round(cnn_lstm_prob, 3),   # actual per-frame DL score
                "dl_peak":      round(lstm_peak, 3),        # rolling max (gate value)
                "dl":           round(cnn_lstm_prob, 3),    # backward compat
                "dl_confirmed": bool(dl_confirmed),
                "phase_a":      round(phase_a_signal, 3),
                "phase_b":      round(phase_b_signal, 3),
                "phase_c":      round(phase_c_signal, 3),
                "votes":        phases_signalling,
                "fused_score":  round(frame_score, 3),
                "score":        round(frame_score, 3),      # backward compat
                "status":       "confirmed" if confirmed_accident else detection_status,
                "tracks":       len(active_tracks),
                "speed":        round(mean_current_speed, 2),
                "density":      round(traffic_density, 2),
                "stopped":      round(stopped_ratio, 2),
                "trigger":      frame_trigger,
                "elapsed":      round(elapsed, 1),
                "consec":       consec_count,
            }
            _push_frame(job_id, jpeg, metrics)

        cap.release()
        out_writer.release()
        transcode_video_for_browser(output_path)

        # Save suspicious/suppressed stubs to disk (these still need input_path for clip extraction)
        saved_end = _save_incident_records(
            incident_stubs, input_path, source_fps, total_frames, filename,
        )

        # Wait for all background confirmed-clip futures before removing source file
        for _f in bg_threads:
            try:
                _f.result(timeout=60)
            except Exception as _ex:
                logger.warning(f"[BG wait] incident background task failed or timed out: {_ex}")

        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass

        result = {
            "class":               "ACCIDENT" if accident_detected_globally else "NO ACCIDENT",
            "confidence":          float(max_accident_score * 100),
            "trigger_phase":       triggering_phase_globally,
            "processed_video_url": f"/static/uploads/{processed_filename}",
            "details":             accident_details,
            "incidents":           all_inline_saved + saved_end,
            "incident_count":      len(all_inline_saved) + len(saved_end),
        }
        _finish_job(job_id, result=result)

    except Exception as e:
        logger.exception("Video stream processing failed")
        _finish_job(job_id, error=str(e))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/incidents")
def api_incidents(limit: int = 50):
    return {"incidents": incident_store.list_incidents(limit=limit)}


@app.delete("/api/incidents")
def api_clear_all_incidents(_auth: bool = Depends(require_api_key)):
    """Delete all incidents from the index and their media files."""
    incident_store.clear_all_incidents()
    return {"ok": True, "cleared": True}


@app.delete("/api/incidents/{incident_id}")
def api_delete_incident(incident_id: str, _auth: bool = Depends(require_api_key)):
    if not incident_store.delete_incident(incident_id):
        return JSONResponse(status_code=404, content={"error": "Incident not found"})
    return {"ok": True}


# ================= OPERATOR WORKFLOW (ack / resolve / reopen) =================
# States: new -> acknowledged -> resolved. See utils/incident_store.py for
# the transition logic and RESOLUTION_REASONS. No per-operator identity yet
# (anyone at the shared dashboard can act) — that's a deliberate scope cut
# for now, not an oversight; add real login before treating this as an
# audited trail of who acted.

class ResolveIncidentBody(BaseModel):
    reason: str
    note: Optional[str] = None


@app.get("/api/incidents/resolution-reasons")
def api_resolution_reasons():
    return {"reasons": sorted(incident_store.RESOLUTION_REASONS)}


@app.post("/api/incidents/{incident_id}/acknowledge")
def api_acknowledge_incident(incident_id: str, _auth: bool = Depends(require_api_key)):
    record = incident_store.acknowledge_incident(incident_id)
    if record is None:
        return JSONResponse(status_code=404, content={"error": "Incident not found"})
    return {"ok": True, "incident": record}


@app.post("/api/incidents/{incident_id}/resolve")
def api_resolve_incident(incident_id: str, body: ResolveIncidentBody,
                          _auth: bool = Depends(require_api_key)):
    if body.reason not in incident_store.RESOLUTION_REASONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid reason. Must be one of: {sorted(incident_store.RESOLUTION_REASONS)}"},
        )
    record = incident_store.resolve_incident(incident_id, body.reason, body.note)
    if record is None:
        return JSONResponse(status_code=404, content={"error": "Incident not found"})
    return {"ok": True, "incident": record}


@app.post("/api/incidents/{incident_id}/reopen")
def api_reopen_incident(incident_id: str, _auth: bool = Depends(require_api_key)):
    record = incident_store.reopen_incident(incident_id)
    if record is None:
        return JSONResponse(status_code=404, content={"error": "Incident not found"})
    return {"ok": True, "incident": record}


@app.get("/api/firebase/status")
def api_firebase_status():
    return {
        "connected": _firebase.enabled,
        "storage":   _firebase.storage_enabled,
        "mode":      "firestore+storage" if _firebase.storage_enabled else "firestore_only",
    }


# ================= LIVE IP/RTSP CAMERAS (dashboard-started) =================
# Lets an operator start watching a camera directly from the dashboard by
# pasting its IP/RTSP address — no CLI, no separate stream_processor.py
# process. See live_camera_manager.py for the concurrency caveats.

class LiveCameraStartBody(BaseModel):
    location: str
    source: str                    # IP/RTSP URL, or "0" for the server's webcam
    camera_id: Optional[str] = None


@app.post("/api/live-cameras")
def api_start_live_camera(body: LiveCameraStartBody, _auth: bool = Depends(require_api_key)):
    camera_id = (body.camera_id or "").strip() or f"CAM_{uuid.uuid4().hex[:6].upper()}"
    source_str = (body.source or "").strip()
    if not source_str:
        return JSONResponse(status_code=400, content={"error": "Camera IP/RTSP address is required."})
    source = int(source_str) if source_str.isdigit() else source_str

    try:
        live_camera_manager.start_camera(
            camera_id, body.location or camera_id, source, _firebase
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.exception("Failed to start live camera")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"ok": True, "camera_id": camera_id}


@app.delete("/api/live-cameras/{camera_id}")
def api_stop_live_camera(camera_id: str, _auth: bool = Depends(require_api_key)):
    if not live_camera_manager.stop_camera(camera_id):
        return JSONResponse(status_code=404, content={"error": "Camera not found"})
    return {"ok": True}


@app.get("/api/live-cameras")
def api_list_live_cameras():
    return {"cameras": live_camera_manager.list_cameras()}


@app.get("/api/live-cameras/{camera_id}/frame")
def api_live_camera_frame(camera_id: str):
    session = live_camera_manager.get_camera(camera_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Camera not found"})
    jpeg = session.snapshot_jpeg()
    if jpeg is None:
        return JSONResponse(status_code=404, content={"error": "No frame available yet"})
    return Response(content=jpeg, media_type="image/jpeg")


# ================= IMAGE PREDICTION =================
@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...), threshold: float = 0.50):
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        file_ext     = os.path.splitext(file.filename)[1]
        raw_filename = f"{uuid.uuid4()}{file_ext}"
        file_path    = os.path.join(UPLOAD_DIR, raw_filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        dl_label, dl_conf_pct = predict_image(file_path)
        cnn_lstm_prob = dl_conf_pct / 100.0 if dl_label == "ACCIDENT" else (100.0 - dl_conf_pct) / 100.0

        detector   = YOLODetector()
        frame      = cv2.imread(file_path)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Could not read image file."})

        detections = detector.detect(frame)

        from tracking.deepsort_module import Track
        tracks = [Track(idx + 1, det["bbox"], det["label"], det["confidence"])
                  for idx, det in enumerate(detections)]

        proximity_score = 0.0
        occlusion_score = 0.0
        candidate_pairs = proximity_filter(tracks)

        for t1, t2, dist, ttc_s in candidate_pairs:
            pair_threshold = (
                config.PROXIMITY_PERSON_THRESHOLD
                if (t1.label == config.PERSON_CLASS or t2.label == config.PERSON_CLASS)
                else config.PROXIMITY_THRESHOLD
            )
            prox_s          = max(0.0, 1.0 - (dist / pair_threshold), ttc_s)
            proximity_score = max(proximity_score, prox_s)
            area1 = (t1.bbox[2] - t1.bbox[0]) * (t1.bbox[3] - t1.bbox[1])
            area2 = (t2.bbox[2] - t2.bbox[0]) * (t2.bbox[3] - t2.bbox[1])
            if area1 < area2:
                containment = calculate_bbox_containment_ratio(t1.bbox, t2.bbox)
            else:
                containment = calculate_bbox_containment_ratio(t2.bbox, t1.bbox)
            occlusion_score = max(occlusion_score, containment)

        max_iou = 0.0
        if len(tracks) >= 2:
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    max_iou = max(max_iou, compute_iou(tracks[i].bbox, tracks[j].bbox))

        if len(tracks) >= 2:
            if max_iou < 0.40 and occlusion_score < 0.70:
                proximity_score *= 0.15; occlusion_score *= 0.15
            elif len(tracks) > 4:
                proximity_score *= 0.25; occlusion_score *= 0.25

        vehicle_count   = len(tracks)
        traffic_density = min(vehicle_count / 20.0, 1.0)
        avg_speed       = 0.0
        stopped_ratio   = 1.0 if vehicle_count > 0 else 0.0

        if cnn_lstm_prob < 0.40:
            final_score = 0.0; is_accident = False; final_class = "NO ACCIDENT"
        else:
            final_score = 0.2 * proximity_score + 0.2 * occlusion_score + 0.6 * cnn_lstm_prob
            if xgb_clf is not None:
                feats = [[
                    float(proximity_score), 0.0, 0.0, float(cnn_lstm_prob),
                    float(occlusion_score), 0.0, 0.0,
                    float(traffic_density), float(avg_speed), float(stopped_ratio),
                ]]
                try:
                    final_score = float(xgb_clf.predict_proba(feats)[0][1])
                except Exception as e:
                    logger.warning(f"XGBoost image predict error: {e}")
            is_accident = final_score >= threshold
            final_class = "ACCIDENT" if is_accident else "NO ACCIDENT"

        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{track.label.upper()} {track.confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for t1, t2, dist, _ in candidate_pairs:
            c1, c2 = t1.get_centroid(), t2.get_centroid()
            cv2.line(frame, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 255, 255), 2)
            if is_accident:
                cx, cy = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        if is_accident:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, f"ACCIDENT SUSPECTED ({final_score * 100:.1f}%)",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        processed_filename = f"processed_{uuid.uuid4()}.jpg"
        processed_path     = os.path.join(UPLOAD_DIR, processed_filename)
        cv2.imwrite(processed_path, frame)

        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "class":      final_class,
            "confidence": float(final_score * 100),
            "trigger_phase": (
                "XGBoost Classifier Model" if xgb_clf is not None
                else ("Phase A & Occlusion & CNN-LSTM DL" if is_accident else "None")
            ),
            "processed_image_url": f"/static/uploads/{processed_filename}",
            "details": {
                "proximity_score":  float(proximity_score),
                "occlusion_score":  float(occlusion_score),
                "trajectory_score": 0.0,
                "anomaly_score":    0.0,
                "cnn_lstm_prob":    float(cnn_lstm_prob),
                "traffic_density":  float(traffic_density),
                "avg_speed":        float(avg_speed),
                "stopped_ratio":    float(stopped_ratio),
            },
        }

    except Exception as e:
        logger.exception("predict-image failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= LOG FEATURES =================
@app.post("/log-feature")
async def log_feature(data: FeatureLog, _auth: bool = Depends(require_api_key)):
    try:
        init_csv_file()
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                data.proximity, data.trajectory, data.anomaly, data.cnn,
                data.occlusion, data.merge, data.kinetic, data.density,
                data.avg_speed, data.stopped_ratio, data.label,
            ])
        row_count = 0
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, "r") as f:
                row_count = sum(1 for _ in f) - 1
        return {"success": True, "message": "Features logged successfully", "total_rows": row_count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= TRAIN XGBOOST =================
@app.post("/train-model")
async def train_model(_auth: bool = Depends(require_api_key)):
    try:
        import pandas as pd
        from xgboost import XGBClassifier
        XGBClassifier._estimator_type = "classifier"
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        if not os.path.exists(CSV_FILE):
            return JSONResponse(status_code=400, content={"error": "Dataset not found."})

        df = pd.read_csv(CSV_FILE)
        if len(df) < 5:
            return JSONResponse(status_code=400, content={"error": f"Only {len(df)} rows."})

        X = df.drop("label", axis=1); y = df["label"]
        if len(y.unique()) < 2:
            return JSONResponse(status_code=400, content={"error": "Need both classes."})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss")
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
        clf.save_model(XGB_MODEL_PATH)
        load_xgboost_model()

        return {
            "success":    True,
            "accuracy":   float(acc),
            "train_size": len(X_train),
            "test_size":  len(X_test),
            "total_rows": len(df),
        }
    except Exception as e:
        logger.exception("train-model failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/xgboost/clear-dataset")
async def clear_xgboost_dataset(_auth: bool = Depends(require_api_key)):
    """
    Clears the logged accident_features.csv rows used to train the
    XGBoost refinement model — the "Log for Model Improvement" button's
    accumulated data. Does NOT delete or unload an already-trained model:
    if one is currently active it keeps refining detections exactly as
    before, until you explicitly retrain or restart the server. This is
    purely a clean slate for the next round of labeling.
    """
    try:
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
        init_csv_file()
        logger.info("XGBoost training dataset cleared.")
        return {"ok": True, "message": "Dataset cleared."}
    except Exception as e:
        logger.exception("Failed to clear XGBoost dataset")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= DATASET STATUS =================
@app.get("/dataset-status")
def dataset_status():
    total_rows = class_0 = class_1 = 0
    if os.path.exists(CSV_FILE):
        try:
            import pandas as pd
            df = pd.read_csv(CSV_FILE)
            total_rows = len(df)
            if "label" in df.columns:
                counts  = df["label"].value_counts()
                class_0 = int(counts.get(0, 0))
                class_1 = int(counts.get(1, 0))
        except Exception as e:
            logger.warning(f"Error reading CSV status: {e}")
    return {
        "total_rows":     total_rows,
        "class_0":        class_0,
        "class_1":        class_1,
        "xgboost_active": xgb_clf is not None,
    }


# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
