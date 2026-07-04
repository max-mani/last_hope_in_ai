# ============================================================
# Accident Detection System — Configuration
# Change model paths and thresholds here only.
# ============================================================

import os

# ── Compute Device ───────────────────────────────────────────
# "auto" picks CUDA if available, else CPU. Override with the
# UYIR_DEVICE env var ("cpu" / "cuda") to force a specific device —
# useful for benchmarking sustained FPS on the actual target hardware
# before sizing a multi-camera deployment.
DEVICE_MODE = os.environ.get("UYIR_DEVICE", "auto")

# ── Model Paths ──────────────────────────────────────────────
VEHICLE_MODEL_PATH = "yolov8n.pt"
# Renamed from "accident_model.pt" — that name was one character away
# from model_output/accident_model.pth (the CNN-BiLSTM checkpoint).
# These are two unrelated models; the near-identical names risked an
# operator overwriting/confusing them during a field update.
STAGE1_YOLO_GATE_PATH = "stage1_yolo_gate.pt"  # optional Stage-1 YOLO for stream pipeline

# ── Camera Settings ──────────────────────────────────────────
# Overridable via env vars so one codebase can run N camera processes
# (e.g. one systemd unit per camera) without N code forks.
# stream_processor.py also accepts --camera-id / --location / --source
# on the command line, which take priority over these env defaults.
CAMERA_ID = os.environ.get("UYIR_CAMERA_ID", "CAM_001")
CAMERA_LOCATION = os.environ.get("UYIR_CAMERA_LOCATION", "Gandhipuram Junction")
_rtsp_env = os.environ.get("UYIR_RTSP_URL")
if _rtsp_env is None:
    RTSP_URL = 0
elif _rtsp_env.isdigit():
    RTSP_URL = int(_rtsp_env)
else:
    RTSP_URL = _rtsp_env

FRAME_SKIP = 3
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ── Stage 1 — Accident Model Settings ───────────────────────
ACCIDENT_CONF_THRESHOLD = 0.50
STAGE1_GATE_CONFIDENCE = 0.65

# ── Tracker Settings ─────────────────────────────────────────
VEHICLE_CONF_THRESHOLD = 0.15
TRACK_HISTORY_FRAMES = 30
TRACK_LOST_TIMEOUT = 30

# ── Phase A — Proximity & TTC ────────────────────────────────
PROXIMITY_THRESHOLD = 150
PROXIMITY_PERSON_THRESHOLD = 80
TTC_MAX_FRAMES = 8
TTC_MIN_CLOSING_SPEED = 0.5

PROXIMITY_PX_CITY = 150
PROXIMITY_PX_HIGHWAY = 220

# NOTE: "motorcycle" and "auto" (autorickshaw) are listed here as a
# reminder that the detector does NOT actually distinguish these —
# YOLOv8n is COCO-trained and only produces the 5 TARGET_CLASSES below;
# "bike" covers all two-wheelers and autorickshaws aren't detected as
# their own class at all. Two-wheelers are a large share of Indian road
# fatalities, so this is a real detection gap (needs fine-tuning on an
# Indian traffic dataset), not just an unused constant.
VEHICLE_CLASSES = {"car", "bike", "bus", "truck", "motorcycle", "auto"}
PERSON_CLASS = "person"

TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "bike",
    5: "bus",
    7: "truck",
}

# ── Camera Calibration (pixel → metric conversion) ───────────
# Disabled by default: no physical site survey of any camera location
# has been performed. Only enable a camera after generating a real
# calibration file for that exact camera_id with:
#     python -m utils.calibration --camera <ID> --frame <snapshot.jpg>
# (see utils/calibration.py). Running with a guessed/fabricated
# calibration would produce confidently WRONG metric thresholds —
# worse than the honestly-approximate pixel thresholds used today.
CALIBRATION_ENABLED = os.environ.get("UYIR_CALIBRATION_ENABLED", "false").lower() == "true"
CALIBRATION_DIR = "calibration"

# Metric thresholds used ONLY for a camera with a valid calibration file.
PROXIMITY_THRESHOLD_M = 5.0          # vehicle-vehicle, meters
PROXIMITY_PERSON_THRESHOLD_M = 2.0   # vehicle-person, meters
TTC_MAX_SECONDS = 2.0                # seconds until contact
TTC_MIN_CLOSING_SPEED_MPS = 0.3      # m/s

# ── Phase B — Trajectory Conflict ───────────────────────────
SPEED_DROP_PERCENT = 70.0
ANGLE_DIVERGENCE_DEG = 30.0
VELOCITY_SUM_STOP = 8.0

EMERGENCY_BASELINE_FRAMES = 15
EMERGENCY_RECENT_FRAMES = 3
EMERGENCY_DROP_PERCENT = 75.0
EMERGENCY_SUDDEN_RATIO = 0.65
TRAJECTORY_STOP_PREV_SPEED = 3.0
TRAJECTORY_STOP_RECENT_SPEED = 2.0
TRAJECTORY_STOP_FRAMES = 5

REL_VEL_PREV_DIFF_MIN = 8.0
REL_VEL_CURR_DIFF_MAX = 2.0

# Trajectory deviation threshold — previously hardcoded as 40.0 inside
# threshold_analyzer.py, which broke the project's own "all tunable
# thresholds live in config.py" rule (tech_Des.md).
TRAJECTORY_DEVIATION_THRESHOLD = 40.0

# ── Phase B — Recent-motion guard ───────────────────────────
# A track that had peak speed > this in the last N frames is
# considered "recently moving" (post-crash stop ≠ parked car).
RECENTLY_MOVING_FRAMES = 15
RECENTLY_MOVING_MIN_SPEED = 3.0   # px/frame

# ── Phase C — Anomaly Confirmation ──────────────────────────
OPTICAL_FLOW_SPIKE = 2.5
BBOX_DEFORM_RATIO = 0.25
FLOW_HISTORY_FRAMES = 10

# ── Gate Settings ────────────────────────────────────────────
CONSECUTIVE_FRAMES = 3
COOLDOWN_SECONDS = 20.0
FUSION_THRESHOLD = 0.55

# Cooldown is now spatial, not purely per-camera: a newly confirmed
# accident within COOLDOWN_SECONDS of a previous one is only suppressed
# if it's also within COOLDOWN_RADIUS_PX of that earlier incident's
# location. A second, physically distinct collision elsewhere in the
# same camera's view (e.g. a chain-reaction pileup) can still fire.
COOLDOWN_RADIUS_PX = 220

# ── DL Gate ──────────────────────────────────────────────────
DL_GATE_THRESHOLD = 0.55   # lstm_peak must reach this to open the gate
DL_PHASE_SIGNAL_MIN = 0.30  # a phase must reach this to count as a vote
DL_WARMUP_FRAMES = 16       # SEQUENCE_LEN // 2 — don't trust rolling peak before this

# ── Fusion Weights ────────────────────────────────────────────
# CNN-LSTM acts as a HARD GATE only — its weight is 0.
# The weight that was on cnn_lstm (0.25) is redistributed to
# trajectory_stop (+0.15) and emergency_stop (+0.05) and optical_flow (+0.05).
FUSION_WEIGHTS = {
    "trajectory_stop": 0.45,
    "emergency_stop":  0.25,
    "ttc_critical":    0.15,
    "optical_flow":    0.10,
    "flow_dispersion": 0.05,
    "cnn_lstm":        0.0,   # gate only — not a weighted contributor
}

# Legacy score weights for stream pipeline phase gating (kept for compatibility)
SCORE_PHASE_A = 3
SCORE_PHASE_B = 2
SCORE_PHASE_C = 1
MIN_SCORE_TO_PASS = 4

# ── Firebase Settings ────────────────────────────────────────
FIREBASE_KEY_PATH = "firebase_key.json"
FIREBASE_BUCKET = "kapaan-web.firebasestorage.app"
FIRESTORE_COLLECTION = "accident_events"
HEALTH_COLLECTION = "pi_health"
HEALTH_INTERVAL_SEC = 30
# Storage requires Blaze (billing). Set False to use Firestore-only on free Spark plan.
FIREBASE_USE_STORAGE = False
# When storage is off, optionally embed a JPEG thumbnail in Firestore (max ~1 MB/doc).
FIREBASE_EMBED_SNAPSHOT = True

# ── Local Fallback ───────────────────────────────────────────
LOCAL_EVENTS_DIR = "local_events"
SNAPSHOTS_DIR = "snapshots"

# ── Incident Clips ───────────────────────────────────────────
CLIP_SECONDS_BEFORE = 5
CLIP_SECONDS_AFTER = 5
INCIDENTS_DIR = "static/uploads/incidents"
INCIDENTS_INDEX = "local_events/incidents_index.json"
CLIP_BUFFER_FPS = 10

# ── Data Logger ──────────────────────────────────────────────
DATA_LOG_CSV = "uyir_data_log.csv"

# ── Performance Tuning ───────────────────────────────────────
# Dense Farneback optical flow (Phase C) is the single most expensive
# CPU-bound step per frame — its cost scales directly with pixel count.
# Computing it on a downscaled copy of the frame and upsampling the flow
# field back to full resolution before Phase C samples it (see
# utils/optical_flow.py) is a large speed win for a small precision cost.
# 1.0 = full resolution (original behavior, slowest). Lower = faster.
# Tested range: 0.4–0.6 is usually a good speed/precision balance.
OPTICAL_FLOW_SCALE = float(os.environ.get("UYIR_OPTICAL_FLOW_SCALE", "0.5"))

# Override PyTorch's CPU thread pool size. Unset (None) lets model.py use
# all logical cores, which is usually right, but on a shared box you may
# want to leave headroom for OpenCV/ByteTrack's own threads.
TORCH_CPU_THREADS = os.environ.get("UYIR_TORCH_THREADS")

# ── Background Task Pool ─────────────────────────────────────
# Bounded worker pool shared by firebase_uploader.py, app.py, and
# stream_processor.py for clip extraction / Firebase upload / LLM
# calls, so a burst of near-simultaneous incidents (a multi-vehicle
# pileup, or a mis-tuned camera false-triggering rapidly) can't spawn
# an unbounded number of raw threads. See utils/task_pool.py.
MAX_BACKGROUND_WORKERS = int(os.environ.get("UYIR_MAX_WORKERS", "8"))

# ── API Authentication (stopgap) ─────────────────────────────
# A simple shared-secret key required via the "X-API-Key" header on
# every mutating endpoint in app.py: DELETE /api/incidents[/{id}],
# POST /train-model, POST /log-feature, POST /start-stream.
#
# This is NOT a real auth system — no per-user identity, no roles, no
# audit trail of who acted. It only stops an anonymous person on the
# network segment from wiping incident history or retraining the
# refinement model. Replace with proper login/role-based access before
# handing this over to police IT staff for production use.
#
# CHANGE THIS before deployment — set the UYIR_API_KEY env var, and
# update the matching constant in templates/index.html.
API_KEY = os.environ.get("UYIR_API_KEY", "uyir-dev-key-change-me")
