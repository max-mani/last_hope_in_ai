"""
UYIR Accident Detector — Option 2 pipeline (DL gate + 2-of-3 phase vote).

This detector is used by stream_processor.py for live RTSP/camera feeds.
It now uses exactly the same detection logic as the web pipeline in app.py:

  1. EfficientNet-B0 CNN + BiLSTM + Attention → cnn_lstm_prob
  2. Rolling peak with warmup guard → lstm_peak
  3. DL gate: lstm_peak must be >= DL_GATE_THRESHOLD (0.55) to proceed
  4. Phase A (proximity/TTC), Phase B (trajectory), Phase C (optical flow)
  5. Need >= 2 of 3 phases to signal >= PHASE_SIGNAL_MIN (0.30)
  6. Fuse remaining signals → confidence score
  7. Consecutive-frame gate + cooldown
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

import config
from fusion.scoring import fuse_scores
from model import DEVICE, SEQUENCE_LEN, model, transform, frame_to_tensor_fast
from phases.phase_a_proximity import proximity_filter
from phases.phase_b_trajectory import (
    analyze_trajectory_conflict,
    is_stationary,
    was_recently_moving,
)
from phases.phase_c_anomaly import analyze_anomaly
from tracking.deepsort_module import Track
from utils.calibration import get_calibration
from utils.optical_flow import compute_optical_flow

logger = logging.getLogger("AccidentDetector")


@dataclass
class AccidentEvent:
    camera_id: str
    location: str
    timestamp: float
    frame_num: int
    confidence_score: float
    stage1_confidence: float
    phases_triggered: list
    involved_vehicle_ids: list
    snapshot_frame: object
    trigger_phase: str = "Weighted Fusion"
    fusion_details: dict = field(default_factory=dict)
    cnn_lstm_confidence: float = 0.0
    clip_path: Optional[str] = None


def _tracked_vehicle_to_track(v) -> Track:
    """Convert TrackedVehicle (stream tracker) to Track (web tracker) for shared phase modules."""
    track = Track(v.id, v.bbox, v.class_name, 1.0)
    track.history = list(v.centroid_history)
    track.bbox_history = list(v.bbox_history)
    track.speed_history = list(v.speed_history)
    track.velocities = []
    for i in range(1, len(v.centroid_history)):
        p0 = v.centroid_history[i - 1]
        p1 = v.centroid_history[i]
        track.velocities.append((p1[0] - p0[0], p1[1] - p0[1]))
    if not track.velocities:
        track.velocities = [(0.0, 0.0)]
    track.age = len(v.centroid_history)
    return track


class AccidentDetector:
    def __init__(self, camera_id=None, location=None):
        self.camera_id = camera_id or config.CAMERA_ID
        self.location = location or config.CAMERA_LOCATION
        self.accident_model = None
        self._consec_count = 0
        # Spatial + temporal cooldown: list of (timestamp, (x, y)) for
        # recent confirmed alerts, instead of one single global timestamp.
        # See _in_spatial_cooldown() — a second, physically distinct
        # collision elsewhere in the same camera's view is no longer
        # blanket-suppressed just because another one fired recently.
        self._recent_alerts = []
        self._prev_gray = None

        # Live per-frame telemetry — lets any caller (the dashboard's live
        # camera view, in particular) show the exact same DL-gate/phase/
        # fusion breakdown that the video-upload SSE path already shows,
        # instead of only finding out about fully-confirmed accidents.
        # Updated on every analyze() call, read via get_telemetry().
        self._telemetry_lock = threading.Lock()
        self.last_telemetry = {}

        # DL feature buffer — same as app.py
        self._features_buffer = []
        self._lstm_scores_history = []   # rolling window for peak computation

        # Camera calibration (pixel -> meters). Disabled and None unless
        # config.CALIBRATION_ENABLED is set AND a real calibration file
        # exists for this exact camera_id — see utils/calibration.py.
        self.calibration = None
        if config.CALIBRATION_ENABLED:
            self.calibration = get_calibration(self.camera_id)
            if self.calibration is None:
                logger.warning(
                    f"[Detector] CALIBRATION_ENABLED but no calibration file found "
                    f"for camera '{self.camera_id}' — falling back to pixel thresholds."
                )

        if os.path.exists(config.STAGE1_YOLO_GATE_PATH):
            try:
                from ultralytics import YOLO
                self.accident_model = YOLO(config.STAGE1_YOLO_GATE_PATH)
                logger.info("[Detector] Stage-1 accident model loaded.")
            except Exception as e:
                logger.warning(f"[Detector] Stage-1 model failed to load: {e}")
        else:
            logger.info("[Detector] No Stage-1 model — running DL gate + 3-phase verification only.")

    # ------------------------------------------------------------------
    # DL inference — matches app.py exactly
    # ------------------------------------------------------------------
    def _run_dl(self, frame: np.ndarray):
        """
        Extract per-frame CNN features, maintain rolling sequence buffer,
        run BiLSTM+Attention, and return (cnn_lstm_prob, lstm_peak).

        FIX 1 — Padding direction: the buffer is padded with the LAST frame
                 (features_buffer[-1]), matching model.py's predict_video().
                 The old code padded with features_buffer[0] (first frame),
                 causing the model to see an out-of-distribution input pattern
                 during the warmup period and produce artificially high scores.

        FIX 2 — Warmup guard: lstm_peak (rolling max of 30 scores) is only
                 trusted after DL_WARMUP_FRAMES frames have been processed.
                 Before that, we use the raw per-frame probability so that an
                 early padding-induced spike cannot lock the gate open for 30
                 frames on every video.
        """
        frame_feat_input = frame_to_tensor_fast(frame)

        with torch.no_grad():
            feat = model.cnn(frame_feat_input)

        self._features_buffer.append(feat)
        if len(self._features_buffer) > SEQUENCE_LEN:
            self._features_buffer.pop(0)

        # FIX 1: pad with the LAST frame (was features_buffer[0])
        if len(self._features_buffer) < SEQUENCE_LEN:
            last_feat = self._features_buffer[-1]
            padded = [last_feat] * (SEQUENCE_LEN - len(self._features_buffer)) + self._features_buffer
        else:
            padded = self._features_buffer

        features_tensor = torch.stack(padded, dim=1)
        with torch.no_grad():
            lstm_out, _ = model.bilstm(features_tensor)
            context = model.attention(lstm_out)
            logits = model.classifier(context)
            probs = torch.softmax(logits, dim=1)
            cnn_lstm_prob = float(probs[0, 1].item())

        # FIX 2: rolling peak with warmup guard
        self._lstm_scores_history.append(cnn_lstm_prob)
        if len(self._lstm_scores_history) > 30:
            self._lstm_scores_history.pop(0)

        warmup_done = len(self._features_buffer) >= config.DL_WARMUP_FRAMES
        if warmup_done:
            lstm_peak = max(self._lstm_scores_history)
        else:
            lstm_peak = cnn_lstm_prob   # raw prob during warmup — no locked max

        return cnn_lstm_prob, lstm_peak

    # ------------------------------------------------------------------
    # Main analysis — Option 2 pipeline
    # ------------------------------------------------------------------
    def analyze(self, frame: np.ndarray, vehicles: dict, frame_num: int,
                effective_fps: float = None) -> Optional[AccidentEvent]:
        now = time.time()

        # ── 1. DL inference ─────────────────────────────────────────
        cnn_lstm_prob, lstm_peak = self._run_dl(frame)

        # ── 2. DL gate ───────────────────────────────────────────────
        dl_confirmed = lstm_peak >= config.DL_GATE_THRESHOLD

        # ── 4. Optional Stage-1 YOLO gate ───────────────────────────
        stage1_conf = 0.0
        if self.accident_model is not None:
            stage1_conf, _ = self._run_stage1(frame)
            if stage1_conf < config.STAGE1_GATE_CONFIDENCE:
                self._consec_count = 0
                self._update_flow(frame)
                self._set_telemetry(
                    frame_num=frame_num, dl_raw=cnn_lstm_prob, dl_peak=lstm_peak,
                    dl_confirmed=dl_confirmed, tracks=len(vehicles),
                    status="scanning", trigger=f"Stage-1 gate not cleared ({stage1_conf:.2f})",
                )
                return None

        # ── 5. Track conversion ──────────────────────────────────────
        tracks = [_tracked_vehicle_to_track(v) for v in vehicles.values()]
        if len(tracks) < 2:
            self._update_flow(frame)
            self._set_telemetry(
                frame_num=frame_num, dl_raw=cnn_lstm_prob, dl_peak=lstm_peak,
                dl_confirmed=dl_confirmed, tracks=len(tracks),
                status="scanning", trigger="Not enough tracked vehicles",
            )
            return None

        # ── 6. Optical flow ──────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = (
            compute_optical_flow(self._prev_gray, gray, scale=config.OPTICAL_FLOW_SCALE)
            if self._prev_gray is not None else None
        )
        self._prev_gray = gray.copy()

        # ── 7. Phase signals ─────────────────────────────────────────
        candidate_pairs = proximity_filter(
            tracks, calibration=self.calibration, effective_fps=effective_fps
        )

        ttc_score = 0.0
        trajectory_stop_score = 0.0
        emergency_stop_score = 0.0
        relative_velocity_score = 0.0
        optical_flow_score = 0.0
        flow_dispersion_score = 0.0
        occlusion_score = 0.0
        merge_score = 0.0
        spin_score = 0.0
        energy_drop_score = 0.0
        involved_ids = []

        # Phase C — per-track anomaly (independent of pairs)
        for track in tracks:
            anom = analyze_anomaly(track, flow)
            optical_flow_score = max(optical_flow_score, anom["anomaly_score"])
            flow_dispersion_score = max(flow_dispersion_score, anom["dispersion_val"])

        # Phase A + Phase B — per-pair
        for t1, t2, dist, pair_ttc in candidate_pairs:
            if t1.age < 2 or t2.age < 2:
                continue

            # FIX 3: Don't skip post-crash stopped pairs.
            # Only skip if NEITHER vehicle was recently moving (permanently parked).
            both_stationary = is_stationary(t1) and is_stationary(t2)
            if both_stationary and not was_recently_moving(t1) and not was_recently_moving(t2):
                continue

            pair_threshold = (
                config.PROXIMITY_PERSON_THRESHOLD
                if (t1.label == config.PERSON_CLASS or t2.label == config.PERSON_CLASS)
                else config.PROXIMITY_THRESHOLD
            )
            prox_s = max(0.0, 1.0 - (dist / pair_threshold), pair_ttc)
            ttc_score = max(ttc_score, prox_s, pair_ttc)

            traj = analyze_trajectory_conflict(t1, t2)
            trajectory_stop_score = max(trajectory_stop_score, traj["trajectory_stop_score"])
            emergency_stop_score = max(emergency_stop_score, traj["emergency_stop_score"])
            relative_velocity_score = max(relative_velocity_score, traj["relative_velocity_score"])
            energy_drop_score = max(energy_drop_score, traj["max_ke_drop"], traj["emergency_stop_score"])
            spin_score = max(spin_score, traj["max_spin_var"])

            if traj["occluded"]:
                occlusion_score = max(occlusion_score, traj["containment"])
            else:
                occlusion_score = max(occlusion_score, traj["containment"] * 0.5)
            if traj["merged"]:
                merge_score = 1.0

            involved_ids.extend([t1.track_id, t2.track_id])

        # Scene aggregates
        stopped = sum(
            1 for t in tracks
            if t.velocities and math.sqrt(t.velocities[-1][0]**2 + t.velocities[-1][1]**2) < 2.0
        )
        stopped_ratio = stopped / len(tracks) if tracks else 0.0
        speeds = [
            math.sqrt(t.velocities[-1][0]**2 + t.velocities[-1][1]**2)
            for t in tracks if t.velocities
        ]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        traffic_density = min(len(tracks) / 20.0, 1.0)

        # ── 8. Option 2 decision logic ────────────────────────────────
        # Mirrors app.py's video-upload tiering exactly (scanning / pending
        # / suspicious / suppressed / confirmed) instead of only ever
        # reporting "nothing" until a full confirmation — this is what lets
        # the live-camera dashboard show the same build-up telemetry
        # (DL gate, phase bars, fused score) that the video-upload SSE
        # preview already shows, and makes it obvious when the pipeline is
        # actually seeing partial signal vs. truly seeing nothing.
        phase_a_signal = ttc_score
        phase_b_signal = max(trajectory_stop_score, emergency_stop_score, relative_velocity_score)
        phase_c_signal = optical_flow_score

        phases_signalling = 0
        phases_detail = {}
        if dl_confirmed:
            if phase_a_signal >= config.DL_PHASE_SIGNAL_MIN:
                phases_signalling += 1
                phases_detail["phase_a"] = True
            if phase_b_signal >= config.DL_PHASE_SIGNAL_MIN:
                phases_signalling += 1
                phases_detail["phase_b"] = True
            if phase_c_signal >= config.DL_PHASE_SIGNAL_MIN:
                phases_signalling += 1
                phases_detail["phase_c"] = True

        # Always compute the fused score now, even on weak/no signal, so
        # every tier below has a real number instead of just "not enough".
        fuse_res = fuse_scores(
            trajectory_stop=trajectory_stop_score,
            ttc_critical=ttc_score,
            emergency_stop=emergency_stop_score,
            cnn_lstm=lstm_peak,
            optical_flow=optical_flow_score,
            flow_dispersion=flow_dispersion_score,
            scene_density=len(tracks),
            avg_scene_speed=avg_speed,
            stopped_ratio=stopped_ratio,
        )

        if not dl_confirmed:
            frame_score = float(cnn_lstm_prob)
            frame_accident = False
            detection_status = "scanning"
            frame_trigger = f"DL Gate Not Cleared ({lstm_peak:.2f})"
        elif phases_signalling >= 2:
            frame_score = max(fuse_res["score"], lstm_peak * 0.8)
            frame_accident = True
            detection_status = "pending"
            frame_trigger = (
                f"DL + {' & '.join(p.replace('phase_', 'Phase ').upper() for p in phases_detail)} Verified"
            )
        elif phases_signalling == 1:
            frame_score = fuse_res["score"] * 0.7
            frame_accident = False
            detection_status = "suspicious"
            frame_trigger = "DL Confirmed, 1 Phase Only"
        else:
            frame_score = fuse_res["score"] * 0.5
            frame_accident = False
            detection_status = "suppressed"
            frame_trigger = "DL Confirmed, No Phase Signal"

        # ── 9. Spatial + temporal cooldown ───────────────────────────
        # Computed every frame now (not just on the "pending" tier) so the
        # consecutive-frame counter below can correctly hold steady rather
        # than reset while a genuine candidate event is merely waiting out
        # a nearby recent alert.
        event_location = self._event_location(tracks, involved_ids)
        cooldown_active = self._in_spatial_cooldown(event_location, now)

        # ── 10. Consecutive-frame gate (leaky bucket) ─────────────────
        # A single dropped/noisy frame no longer resets progress to zero —
        # it decays by one instead, same as the video-upload path. This
        # matters a lot more now that CONSECUTIVE_FRAMES defaults to 5:
        # a hard reset-on-any-miss made that much harder to ever satisfy
        # on real, slightly-noisy footage.
        if frame_accident and not cooldown_active:
            self._consec_count += 1
        elif not frame_accident:
            self._consec_count = max(0, self._consec_count - 1)

        confirmed_accident = (
            frame_accident
            and not cooldown_active
            and self._consec_count >= config.CONSECUTIVE_FRAMES
        )

        self._set_telemetry(
            frame_num=frame_num,
            dl_raw=cnn_lstm_prob, dl_peak=lstm_peak, dl_confirmed=dl_confirmed,
            phase_a=phase_a_signal, phase_b=phase_b_signal, phase_c=phase_c_signal,
            votes=phases_signalling,
            fused_score=frame_score,
            status="confirmed" if confirmed_accident else detection_status,
            tracks=len(tracks), speed=avg_speed, density=traffic_density,
            stopped=stopped_ratio, trigger=frame_trigger, consec=self._consec_count,
        )

        if not confirmed_accident:
            return None

        # ── 11. Confirmed accident ────────────────────────────────────
        self._consec_count = 0
        self._record_alert(event_location, now)

        confirmed_phases = list(phases_detail.keys())

        fusion_details = dict(fuse_res["details"])
        fusion_details.update({
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
            "avg_speed":               float(avg_speed),
            "stopped_ratio":           float(stopped_ratio),
            "scene_interruption":      0.0,
            "dl_confirmed":            True,
            "phases_signalling":       int(phases_signalling),
            "phase_a_confirmed":       bool(phases_detail.get("phase_a", False)),
            "phase_b_confirmed":       bool(phases_detail.get("phase_b", False)),
            "phase_c_confirmed":       bool(phases_detail.get("phase_c", False)),
            "post_intersect_static":   False,
        })

        return AccidentEvent(
            camera_id=self.camera_id,
            location=self.location,
            timestamp=now,
            frame_num=frame_num,
            confidence_score=min(1.0, frame_score),
            stage1_confidence=stage1_conf,
            phases_triggered=list(set(confirmed_phases)),
            involved_vehicle_ids=list(set(involved_ids)),
            snapshot_frame=frame.copy(),
            trigger_phase=frame_trigger,
            fusion_details=fusion_details,
            cnn_lstm_confidence=lstm_peak,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_stage1(self, frame):
        results = self.accident_model.predict(frame, conf=config.ACCIDENT_CONF_THRESHOLD, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return 0.0, []
        return float(results[0].boxes.conf.max().cpu().numpy()), []

    def _update_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_gray = gray.copy()

    def _event_location(self, tracks, involved_ids):
        """Rough (x, y) pixel location of a candidate event — the mean
        centroid of the tracks actually involved in the triggering
        pair(s), falling back to the mean of all visible tracks."""
        pts = [t.get_centroid() for t in tracks if t.track_id in involved_ids]
        if not pts:
            pts = [t.get_centroid() for t in tracks]
        if not pts:
            return (0.0, 0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _in_spatial_cooldown(self, location, now):
        """True if a recent alert exists within COOLDOWN_SECONDS *and*
        COOLDOWN_RADIUS_PX of `location`. Also prunes expired alerts."""
        self._recent_alerts = [
            (t, loc) for t, loc in self._recent_alerts
            if now - t < config.COOLDOWN_SECONDS
        ]
        for t, loc in self._recent_alerts:
            if math.hypot(location[0] - loc[0], location[1] - loc[1]) < config.COOLDOWN_RADIUS_PX:
                return True
        return False

    def _record_alert(self, location, now):
        self._recent_alerts.append((now, location))

    def _set_telemetry(self, **kwargs):
        """
        Store the latest per-frame telemetry snapshot. Field names
        deliberately mirror app.py's SSE `metrics` dict (dl_raw, dl_peak,
        phase_a/b/c, votes, fused_score, status, tracks, speed, density,
        stopped, trigger, consec) so the same frontend rendering code can
        be reused for both the video-upload preview and the live-camera
        dashboard view.
        """
        snapshot = {
            "frame": kwargs.get("frame_num", 0),
            "dl_raw": round(float(kwargs.get("dl_raw", 0.0)), 3),
            "dl_peak": round(float(kwargs.get("dl_peak", 0.0)), 3),
            "dl_confirmed": bool(kwargs.get("dl_confirmed", False)),
            "phase_a": round(float(kwargs.get("phase_a", 0.0)), 3),
            "phase_b": round(float(kwargs.get("phase_b", 0.0)), 3),
            "phase_c": round(float(kwargs.get("phase_c", 0.0)), 3),
            "votes": int(kwargs.get("votes", 0)),
            "fused_score": round(float(kwargs.get("fused_score", 0.0)), 3),
            "status": kwargs.get("status", "scanning"),
            "tracks": int(kwargs.get("tracks", 0)),
            "speed": round(float(kwargs.get("speed", 0.0)), 2),
            "density": round(float(kwargs.get("density", 0.0)), 2),
            "stopped": round(float(kwargs.get("stopped", 0.0)), 2),
            "trigger": kwargs.get("trigger", ""),
            "consec": int(kwargs.get("consec", 0)),
            "updated_at": time.time(),
        }
        with self._telemetry_lock:
            self.last_telemetry = snapshot

    def get_telemetry(self):
        with self._telemetry_lock:
            return dict(self.last_telemetry)
