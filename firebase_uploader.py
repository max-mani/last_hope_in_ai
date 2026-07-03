"""UYIR Firebase Uploader — async accident event upload with local fallback."""

import base64
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import cv2

import config
from utils.task_pool import submit as submit_bg

logger = logging.getLogger("FirebaseUploader")

# Firestore document size limit is 1 MB; leave headroom for metadata fields.
_MAX_EMBED_BYTES = 750_000


class FirebaseUploader:
    def __init__(self):
        self._db = None
        self._bucket = None
        self._enabled = False
        self._storage_enabled = False
        self._lock = threading.Lock()
        os.makedirs(config.LOCAL_EVENTS_DIR, exist_ok=True)
        os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
        self._init_firebase()

    def _init_firebase(self):
        if not os.path.exists(config.FIREBASE_KEY_PATH):
            logger.warning("firebase_key.json not found — events saved locally only.")
            return
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            with self._lock:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                    if config.FIREBASE_USE_STORAGE:
                        firebase_admin.initialize_app(
                            cred, {"storageBucket": config.FIREBASE_BUCKET}
                        )
                    else:
                        firebase_admin.initialize_app(cred)

                self._db = firestore.client()
                self._enabled = True

                if config.FIREBASE_USE_STORAGE:
                    from firebase_admin import storage
                    self._bucket = storage.bucket()
                    self._storage_enabled = True
                    logger.info("Firebase connected (Firestore + Storage).")
                else:
                    logger.info(
                        "Firebase connected (Firestore only — Storage disabled, free Spark plan)."
                    )
        except ImportError:
            logger.warning("firebase-admin not installed. Run: pip install firebase-admin")
        except Exception as e:
            logger.error(f"Firebase init failed: {e}")

    def upload_event_async(self, event, record=None):
        # Previously a raw, uncapped threading.Thread per call — a burst of
        # near-simultaneous incidents could spawn unbounded threads doing
        # network I/O at once. Now routed through a shared bounded pool
        # (config.MAX_BACKGROUND_WORKERS) — see utils/task_pool.py.
        submit_bg(self._upload, event, record)

    def upload_incident_record_async(self, record):
        """Upload a dashboard incident record to Firestore (and Storage if enabled)."""
        submit_bg(self._upload_incident_record, record)

    @property
    def enabled(self):
        return self._enabled

    @property
    def storage_enabled(self):
        return self._storage_enabled

    def _embed_snapshot_field(self, snap_path):
        """Return Firestore field dict with base64 JPEG thumbnail when small enough."""
        if not config.FIREBASE_EMBED_SNAPSHOT or not snap_path or not os.path.exists(snap_path):
            return {}
        try:
            size = os.path.getsize(snap_path)
            if size > _MAX_EMBED_BYTES:
                logger.info(
                    f"Snapshot too large to embed ({size} bytes) — skipped Firestore thumbnail."
                )
                return {}
            with open(snap_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            return {
                "snapshot_base64": encoded,
                "snapshot_mime": "image/jpeg",
                "snapshot_embedded": True,
            }
        except Exception as e:
            logger.warning(f"Snapshot embed failed: {e}")
            return {}

    def _upload_to_storage(self, local_path, blob_path, content_type):
        if not self._storage_enabled or not local_path or not os.path.exists(local_path):
            return ""
        try:
            blob = self._bucket.blob(blob_path)
            blob.upload_from_filename(local_path, content_type=content_type)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            logger.error(f"Storage upload failed ({blob_path}): {e}")
            return ""

    def _upload_incident_record(self, record):
        from utils import incident_store

        incident_id = record.get("id", "unknown")
        clip_fs, snap_fs, local_clip_url, local_snap_url = incident_store.build_incident_paths(
            incident_id
        )

        image_url = record.get("snapshot_url") or local_snap_url or ""
        clip_url = record.get("clip_url") or local_clip_url or ""

        if self._storage_enabled:
            image_url = self._upload_to_storage(
                snap_fs, f"incidents/{incident_id}/snapshot.jpg", "image/jpeg"
            ) or image_url
            clip_url = self._upload_to_storage(
                clip_fs, f"incidents/{incident_id}/clip.mp4", "video/mp4"
            ) or clip_url

        doc = {
            "incident_id": incident_id,
            "timestamp": record.get("timestamp"),
            "status": record.get("status", "confirmed"),
            "source": record.get("source"),
            "camera_id": record.get("camera_id"),
            "location": record.get("location"),
            "confidence_score": record.get("confidence"),
            "dl_confidence": record.get("dl_confidence"),
            "trigger_phase": record.get("trigger_phase"),
            "phases_triggered": record.get("phases_triggered"),
            "involved_vehicle_ids": record.get("involved_vehicle_ids"),
            "frame_number": record.get("frame_number"),
            "time_in_video_sec": record.get("time_in_video_sec"),
            "image_url": image_url,
            "clip_url": clip_url,
            "local_snapshot_path": snap_fs if os.path.exists(snap_fs) else "",
            "local_clip_path": clip_fs if os.path.exists(clip_fs) else "",
            "llm_analysis": record.get("llm_analysis"),
            "details": record.get("details") or {},
            "created_at": time.time(),
            "storage_mode": "cloud" if self._storage_enabled else "local_media",
        }

        if not self._storage_enabled:
            doc.update(self._embed_snapshot_field(snap_fs))

        if self._enabled:
            try:
                self._db.collection(config.FIRESTORE_COLLECTION).document(incident_id).set(doc)
                logger.info(f"Incident {incident_id} ({doc['status']}) uploaded to Firestore.")
                return
            except Exception as e:
                logger.error(f"Firestore write failed: {e}")

        path = os.path.join(config.LOCAL_EVENTS_DIR, f"incident_{incident_id}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        logger.info(f"Incident saved locally: {path}")

    def _upload(self, event, record=None):
        ts_str = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()
        ts_int = int(event.timestamp)
        snap_name = f"{event.camera_id}_{ts_int}.jpg"
        snap_path = os.path.join(config.SNAPSHOTS_DIR, snap_name)
        cv2.imwrite(snap_path, event.snapshot_frame)

        image_url = ""
        clip_url = ""
        if self._storage_enabled:
            image_url = self._upload_to_storage(snap_path, f"snapshots/{snap_name}", "image/jpeg")
            clip_path = getattr(event, "clip_path", None) or (record or {}).get("clip_path")
            if clip_path and os.path.exists(clip_path):
                clip_name = f"{event.camera_id}_{ts_int}.mp4"
                clip_url = self._upload_to_storage(
                    clip_path, f"clips/{clip_name}", "video/mp4"
                )

        details = getattr(event, "fusion_details", None) or (record or {}).get("details") or {}
        doc = {
            "timestamp": ts_str,
            "camera_id": event.camera_id,
            "location": event.location,
            "confidence_score": event.confidence_score,
            "stage1_confidence": event.stage1_confidence,
            "dl_confidence": getattr(event, "cnn_lstm_confidence", event.stage1_confidence),
            "trigger_phase": getattr(event, "trigger_phase", "Weighted Fusion"),
            "phases_triggered": event.phases_triggered,
            "involved_vehicle_ids": event.involved_vehicle_ids,
            "image_url": image_url,
            "clip_url": clip_url or (record or {}).get("clip_url", ""),
            "frame_number": event.frame_num,
            "details": details,
            "created_at": time.time(),
            "storage_mode": "cloud" if self._storage_enabled else "local_media",
        }

        if record:
            doc["incident_id"] = record.get("id")

        if not self._storage_enabled:
            doc.update(self._embed_snapshot_field(snap_path))

        if self._enabled:
            try:
                self._db.collection(config.FIRESTORE_COLLECTION).add(doc)
                return
            except Exception as e:
                logger.error(f"Firestore write failed: {e}")

        path = os.path.join(config.LOCAL_EVENTS_DIR, f"event_{ts_int}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        logger.info(f"Event saved locally: {path}")

    def retry_local_events(self):
        if not self._enabled:
            return
        folder = config.LOCAL_EVENTS_DIR
        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            if not (fname.startswith("event_") or fname.startswith("incident_")):
                continue
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath) as f:
                    doc = json.load(f)
                incident_id = doc.get("incident_id")
                if incident_id:
                    self._db.collection(config.FIRESTORE_COLLECTION).document(incident_id).set(doc)
                else:
                    self._db.collection(config.FIRESTORE_COLLECTION).add(doc)
                os.remove(fpath)
            except Exception as e:
                logger.warning(f"Retry failed for {fname}: {e}")
