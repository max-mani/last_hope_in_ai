"""
UYIR Live Camera Manager — lets the web dashboard start/stop a live
accident-detection session against an RTSP/IP camera (or webcam) directly,
without needing to run stream_processor.py as a separate CLI process.

Each session runs the exact same Option 2 pipeline (AccidentDetector +
VehicleTracker) as the CLI stream_processor.py and the video-upload SSE
path in app.py — this file is a thin adapter that runs that pipeline in a
background thread per camera and exposes its state (latest annotated
frame, health/status, and confirmed incidents) to the FastAPI routes in
app.py.

CONCURRENCY NOTE: running more than one live camera session at once means
multiple concurrent YOLO + EfficientNet-B0 + BiLSTM inference passes
sharing the same CPU (or GPU). This has not been benchmarked — treat more
than 1-2 concurrent dashboard-started cameras on typical hardware as
experimental until you've confirmed sustained per-camera FPS holds up.
For a real multi-camera deployment, one stream_processor.py process per
camera (per the deployment report's recommendation) still scales more
predictably than stacking sessions inside one web server process.
"""

import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np

import config
from accident_detector import AccidentDetector
from tracking.vehicle_tracker import VehicleTracker
from utils.incident_clip import write_clip_from_frames
from utils import incident_store
from utils.task_pool import submit as submit_bg

logger = logging.getLogger("LiveCameraManager")

_sessions = {}
_sessions_lock = threading.Lock()


class LiveCameraSession:
    """One dashboard-started live camera. Owns its own tracker + detector
    so multiple cameras never share tracking state."""

    def __init__(self, camera_id, location, source, firebase_uploader, on_confirmed=None):
        self.camera_id = camera_id
        self.location = location
        self.source = source
        self.firebase = firebase_uploader
        # Optional callback invoked (via the shared background task pool)
        # with the saved incident record after a confirmed accident is
        # written to disk — used by app.py to run the external LLM
        # verification + auto-labeling + auto-retrain pipeline. Left as
        # None from stream_processor.py/CLI usage, where it doesn't apply.
        self.on_confirmed = on_confirmed

        self.tracker = VehicleTracker()
        self.detector = AccidentDetector(camera_id=camera_id, location=location)

        self._running = False
        self._thread = None
        self._state_lock = threading.Lock()

        self.frame_count = 0
        self.started_at = time.time()
        self._fps_list = []
        self._latest_jpeg = None
        self._status = "connecting"   # connecting | online | offline | error | stopped
        self._error = None
        # Recomputed in _connect() from the camera's own reported FPS —
        # see config.LIVE_TARGET_FPS. Starts at the static fallback so
        # there's always a sane value before the first successful connect.
        self._frame_skip = max(1, config.FRAME_SKIP)

        buffer_len = int(config.CLIP_SECONDS_BEFORE * config.CLIP_BUFFER_FPS)
        after_len = int(config.CLIP_SECONDS_AFTER * config.CLIP_BUFFER_FPS)
        self._clip_buffer = deque(maxlen=max(buffer_len, 1))
        self._pending_clip = None
        self._after_frames_needed = after_len

    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"uyir-cam-{self.camera_id}")
        self._thread.start()

    def stop(self):
        self._running = False

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def snapshot_jpeg(self):
        with self._state_lock:
            return self._latest_jpeg

    def status(self):
        with self._state_lock:
            fps = round(float(np.mean(self._fps_list)), 1) if self._fps_list else 0.0
            base = {
                "camera_id": self.camera_id,
                "location": self.location,
                "source": str(self.source),
                "status": self._status,
                "fps": fps,
                "frame_count": self.frame_count,
                "started_at": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
                "error": self._error,
            }
        # Merged outside the state lock — get_telemetry() has its own lock
        # and reads a fully-independent snapshot, so this can't deadlock.
        # This is what lets the dashboard show the exact same DL gate /
        # phase A/B/C / fused-score breakdown for a live camera that the
        # video-upload SSE preview already shows.
        base["telemetry"] = self.detector.get_telemetry()
        return base

    # ------------------------------------------------------------------
    def _set_status(self, status, error=None):
        with self._state_lock:
            self._status = status
            self._error = error

    def _run(self):
        cap = self._connect(attempts=3)
        if cap is None:
            self._set_status("error", "Could not open source. Check the IP/RTSP address and try again.")
            return

        self._set_status("online")
        raw_frame_count = 0

        while self._running:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                self._set_status("offline")
                logger.warning(f"[{self.camera_id}] Stream lost. Reconnecting in 3s...")
                cap.release()
                time.sleep(3)
                if not self._running:
                    break
                cap = self._connect(attempts=5)
                if cap is None:
                    self._set_status("error", "Lost connection and could not reconnect.")
                    break
                self._set_status("online")
                continue

            raw_frame_count += 1
            if raw_frame_count % self._frame_skip != 0:
                continue

            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            self._clip_buffer.append(frame.copy())
            self._collect_after_frames(frame)

            vehicles = self.tracker.process_frame(frame)
            fps_est = self._get_fps()
            event = self.detector.analyze(
                frame, vehicles, self.frame_count, effective_fps=fps_est or 10.0
            )
            self.frame_count += 1

            if event:
                self._on_accident(event)

            display = self.tracker.draw_tracks(frame.copy(), vehicles)
            self._draw_hud(display, vehicles, event)
            ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok:
                with self._state_lock:
                    self._latest_jpeg = buf.tobytes()

            elapsed = max(time.time() - t_start, 1e-6)
            with self._state_lock:
                self._fps_list.append(1.0 / elapsed)
                if len(self._fps_list) > 30:
                    self._fps_list.pop(0)

        cap.release()
        with self._state_lock:
            if self._status != "error":
                self._status = "stopped"

    def _connect(self, attempts=3):
        for attempt in range(attempts):
            if not self._running:
                return None
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._update_frame_skip(cap)
                logger.info(
                    f"[{self.camera_id}] Connected to {self.source} "
                    f"(frame_skip={self._frame_skip}, target={config.LIVE_TARGET_FPS}fps)"
                )
                return cap
            logger.warning(f"[{self.camera_id}] Connection attempt {attempt + 1}/{attempts} failed...")
            time.sleep(2)
        return None

    def _update_frame_skip(self, cap):
        """
        A fixed frame-skip (the old behavior) caps sustained processed FPS
        at (native_fps / skip) no matter how fast the pipeline itself can
        run — e.g. a 24fps RTSP source with a skip of 3 tops out at 8fps
        even on hardware that could easily process 20. Instead, pick a
        skip factor from the camera's own reported FPS that targets
        config.LIVE_TARGET_FPS processed frames per second directly.
        Many RTSP sources misreport FPS (0, absurdly high, or NaN) — fall
        back to the static config.FRAME_SKIP when that happens rather than
        trust a bogus value.
        """
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        target = max(1, config.LIVE_TARGET_FPS)
        if native_fps and 1.0 <= native_fps <= 120.0:
            self._frame_skip = max(1, round(native_fps / target))
        else:
            self._frame_skip = max(1, config.FRAME_SKIP)

    def _get_fps(self):
        with self._state_lock:
            return round(float(np.mean(self._fps_list)), 1) if self._fps_list else 0.0

    def _draw_hud(self, frame, vehicles, event):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"UYIR | {self.camera_id} | {self.location} | Vehicles:{len(vehicles)}",
            (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        if event:
            h = frame.shape[0]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h // 2 - 40), (frame.shape[1], h // 2 + 40), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, "ACCIDENT DETECTED", (max(frame.shape[1] // 2 - 150, 10), h // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    def _collect_after_frames(self, frame):
        with self._state_lock:
            if self._pending_clip is None:
                return
            self._pending_clip["after_frames"].append(frame.copy())
            ready = len(self._pending_clip["after_frames"]) >= self._after_frames_needed
            pending = self._pending_clip if ready else None
            if ready:
                self._pending_clip = None
        if pending is not None:
            submit_bg(self._finalize_clip, pending)

    def _on_accident(self, event):
        with self._state_lock:
            if self._pending_clip is not None:
                return
            self._pending_clip = {
                "event": event,
                "before_frames": list(self._clip_buffer),
                "after_frames": [],
            }

    def _finalize_clip(self, pending):
        event = pending["event"]
        frames = pending["before_frames"] + pending["after_frames"]
        incident_id = str(uuid.uuid4())
        clip_fs, snap_fs, clip_url, snap_url = incident_store.build_incident_paths(incident_id)

        cv2.imwrite(snap_fs, event.snapshot_frame)
        clip_ok = write_clip_from_frames(frames, config.CLIP_BUFFER_FPS, clip_fs)

        details = dict(event.fusion_details or {})
        record = incident_store.save_incident({
            "id": incident_id,
            "source": "live_ip",
            "timestamp": datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat(),
            "camera_id": event.camera_id,
            "location": event.location,
            "frame_number": event.frame_num,
            "confidence": float(event.confidence_score),
            "dl_confidence": float(event.cnn_lstm_confidence or event.stage1_confidence),
            "trigger_phase": event.trigger_phase,
            "phases_triggered": event.phases_triggered,
            "involved_vehicle_ids": event.involved_vehicle_ids,
            "clip_url": clip_url if clip_ok else None,
            "snapshot_url": snap_url,
            "details": details,
            "status": "confirmed",
        })
        self.firebase.upload_incident_record_async(record)
        if self.on_confirmed is not None:
            submit_bg(self.on_confirmed, record)


# ================= Session registry =================
def start_camera(camera_id, location, source, firebase_uploader, on_confirmed=None):
    with _sessions_lock:
        existing = _sessions.get(camera_id)
        if existing is not None and existing.is_alive():
            raise ValueError(f"Camera '{camera_id}' is already running.")
        session = LiveCameraSession(camera_id, location, source, firebase_uploader, on_confirmed=on_confirmed)
        _sessions[camera_id] = session
    session.start()
    return session


def stop_camera(camera_id):
    with _sessions_lock:
        session = _sessions.get(camera_id)
        if session is None:
            return False
        session.stop()
        del _sessions[camera_id]
    return True


def list_cameras():
    with _sessions_lock:
        sessions = list(_sessions.values())
    return [s.status() for s in sessions]


def get_camera(camera_id):
    with _sessions_lock:
        return _sessions.get(camera_id)
