"""
UYIR Stream Processor — live camera / RTSP / video file pipeline.

Usage:
  python stream_processor.py
  python stream_processor.py --source 0
  python stream_processor.py --source rtsp://192.168.1.5:8080/h264_ulaw.sdp
  python stream_processor.py --source video.mp4 --no_display
"""

import argparse
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np

import config
from accident_detector import AccidentDetector, AccidentEvent
from firebase_uploader import FirebaseUploader
from health_monitor import HealthMonitor
from tracking.vehicle_tracker import VehicleTracker
from utils.incident_clip import write_clip_from_frames
from utils import incident_store
from utils.task_pool import submit as submit_bg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("StreamProcessor")


class StreamProcessor:
    def __init__(self, source=None, show_window=True, camera_id=None, camera_location=None):
        self.source = source if source is not None else config.RTSP_URL
        self.show_window = show_window
        self._running = False
        self._fps_list = []
        self._fps_lock = threading.Lock()

        # Per-camera identity: CLI args (highest priority) > UYIR_CAMERA_ID /
        # UYIR_CAMERA_LOCATION env vars (already resolved into config.CAMERA_ID
        # / config.CAMERA_LOCATION) > hardcoded default. This lets one codebase
        # run N camera processes (e.g. one systemd unit per camera) without
        # editing config.py per camera.
        self.camera_id = camera_id or config.CAMERA_ID
        self.camera_location = camera_location or config.CAMERA_LOCATION

        self.tracker = VehicleTracker()
        self.detector = AccidentDetector(camera_id=self.camera_id, location=self.camera_location)
        self.uploader = FirebaseUploader()
        self.health = HealthMonitor(fps_provider=self._get_fps)

        buffer_len = int(config.CLIP_SECONDS_BEFORE * config.CLIP_BUFFER_FPS)
        after_len = int(config.CLIP_SECONDS_AFTER * config.CLIP_BUFFER_FPS)
        self._clip_buffer = deque(maxlen=max(buffer_len, 1))
        self._pending_clip = None
        self._after_frames_needed = after_len
        self._clip_lock = threading.Lock()

    def start(self):
        self.uploader.retry_local_events()
        self.health.start()

        cap = self._connect(self.source)
        if cap is None:
            logger.error("Could not open stream. Exiting.")
            return

        self._running = True
        frame_count = 0
        processed = 0

        logger.info(f"Pipeline running | camera={self.camera_id} | location={self.camera_location}")
        logger.info("Press Q in the window to stop.\n")

        while self._running:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Stream lost. Reconnecting in 3s...")
                cap.release()
                time.sleep(3)
                cap = self._connect(self.source)
                if cap is None:
                    break
                continue

            frame_count += 1
            if frame_count % config.FRAME_SKIP != 0:
                continue

            processed += 1
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

            self._clip_buffer.append(frame.copy())
            self._collect_after_frames(frame)

            vehicles = self.tracker.process_frame(frame)
            # effective_fps feeds calibrated Phase-A TTC (utils/calibration.py)
            # when calibration is enabled; harmless no-op otherwise.
            event = self.detector.analyze(
                frame, vehicles, processed, effective_fps=self._get_fps() or 10.0
            )

            if event:
                self._on_accident(event)

            if self.show_window:
                display = self._draw_ui(frame, vehicles, event, processed)
                cv2.imshow(f"UYIR | {self.camera_id}", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = max(time.time() - t_start, 1e-6)
            with self._fps_lock:
                self._fps_list.append(1.0 / elapsed)
                if len(self._fps_list) > 30:
                    self._fps_list.pop(0)

        cap.release()
        self.health.stop()
        if self.show_window:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")

    def stop(self):
        self._running = False

    def _connect(self, source):
        for attempt in range(5):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(f"Connected to source: {source}")
                return cap
            logger.warning(f"Connection attempt {attempt + 1}/5 failed...")
            time.sleep(2)
        return None

    def _collect_after_frames(self, frame):
        with self._clip_lock:
            if self._pending_clip is None:
                return
            self._pending_clip["after_frames"].append(frame.copy())
            if len(self._pending_clip["after_frames"]) >= self._after_frames_needed:
                pending = self._pending_clip
                self._pending_clip = None
                submit_bg(self._finalize_clip, pending)

    def _on_accident(self, event: AccidentEvent):
        print("\n" + "=" * 58)
        print("  ACCIDENT DETECTED")
        print(f"  Camera     : {event.camera_id}")
        print(f"  Location   : {event.location}")
        print(f"  Time       : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}")
        print(f"  Confidence : {event.confidence_score:.0%}")
        print(f"  Phases     : {', '.join(event.phases_triggered)}")
        print(f"  Vehicles   : {event.involved_vehicle_ids}")
        print("=" * 58 + "\n")

        with self._clip_lock:
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

        event.clip_path = clip_fs if clip_ok else None
        details = dict(event.fusion_details or {})

        record = incident_store.save_incident({
            "id": incident_id,
            "source": "stream",
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

        self.uploader.upload_incident_record_async(record)

    def _draw_ui(self, frame, vehicles, event, frame_num):
        display = self.tracker.draw_tracks(frame.copy(), vehicles)
        avg_fps = self._get_fps()
        cv2.rectangle(display, (0, 0), (display.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(
            display,
            f"UYIR | {self.camera_id} | {self.camera_location} "
            f"| Frame:{frame_num} | FPS:{avg_fps:.1f} | Vehicles:{len(vehicles)}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1,
        )
        if event:
            h = display.shape[0]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h // 2 - 44), (display.shape[1], h // 2 + 44), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
            cv2.putText(display, "ACCIDENT DETECTED",
                        (display.shape[1] // 2 - 160, h // 2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        return display

    def _get_fps(self):
        with self._fps_lock:
            return round(float(np.mean(self._fps_list)), 1) if self._fps_list else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UYIR Stream Processor")
    parser.add_argument("--source", default=str(config.RTSP_URL),
                        help="Video source: 0=webcam, RTSP URL, or video file path")
    parser.add_argument("--no_display", action="store_true",
                        help="Run without display window")
    # Per-camera overrides — take priority over UYIR_CAMERA_ID / UYIR_CAMERA_LOCATION
    # env vars and the hardcoded config.py defaults. Lets one process-per-camera
    # deployment (e.g. one systemd unit per junction) share a single codebase:
    #   python stream_processor.py --source rtsp://... \
    #       --camera-id CAM_002 --location "Ukkadam Junction"
    parser.add_argument("--camera-id", default=None, help="Override camera ID for this process")
    parser.add_argument("--location", default=None, help="Override camera location label")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    StreamProcessor(
        source=source,
        show_window=not args.no_display,
        camera_id=args.camera_id,
        camera_location=args.location,
    ).start()
