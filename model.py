import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import cv2

import config

# ================= CONFIG =================
SEQUENCE_LEN = 32
MODEL_PATH = "model_output/accident_model.pth"

CLASS_NAMES = {
    0: "NO ACCIDENT",
    1: "ACCIDENT"
}

NUM_CLASSES = 2


def _resolve_device():
    """
    Was hardcoded to CPU regardless of what hardware the process ran on —
    silently wasting any available GPU. Now honors config.DEVICE_MODE
    ("auto" / "cpu" / "cuda", settable via the UYIR_DEVICE env var).
    """
    mode = getattr(config, "DEVICE_MODE", "auto")
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] UYIR_DEVICE=cuda requested but no CUDA device found — falling back to CPU.")
        return torch.device("cpu")
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


DEVICE = _resolve_device()

# Was left at PyTorch's default thread count, which isn't always the full
# core count in constrained/containerized environments. On CPU this is the
# single biggest lever for sustained FPS on real hardware — explicit is
# safer than relying on the default. Override with UYIR_TORCH_THREADS if a
# specific value benchmarks better than "all cores" (e.g. leaving headroom
# for a concurrent OpenCV/ByteTrack thread pool).
if DEVICE.type == "cpu":
    _cpu_threads = int(getattr(config, "TORCH_CPU_THREADS", None) or os.cpu_count() or 4)
    torch.set_num_threads(_cpu_threads)

# ================= TRANSFORM =================
# Must match training transform (resize size + normalization).
# Note: no random flip / color jitter here - those are train-time only.
transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= MODEL =================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = (weights * x).sum(dim=1)
        return context


class AccidentNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = efficientnet_b0(weights=None)  # weights loaded from checkpoint
        backbone.classifier = nn.Identity()
        self.cnn = backbone

        self.bilstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.attention = Attention(512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        features = self.cnn(x)
        features = features.view(b, t, 1280)

        lstm_out, _ = self.bilstm(features)
        context = self.attention(lstm_out)

        return self.classifier(context)


# ================= LOAD MODEL =================
model = AccidentNet().to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Model Loaded Successfully (device={DEVICE}, mode={config.DEVICE_MODE})")


# ================= FAST FRAME PREPROCESSING (live/video pipelines) =====
# transform() above (PIL-based) stays the source of truth for
# predict_image()/predict_video() so single-shot inference is byte-for-byte
# the same as before. For the per-frame hot path in accident_detector.py
# and app.py's streaming loop, the PIL round-trip (ndarray -> PIL.Image ->
# PIL resize -> ToTensor -> Normalize) was a measurable per-frame cost at
# 8fps. frame_to_tensor_fast() does the same 240x240 resize + ImageNet
# normalize directly on the OpenCV array with cv2.resize (SIMD-accelerated)
# and torch tensor ops, skipping PIL entirely. Output is numerically very
# close (bilinear vs. PIL's bilinear can differ by a fraction of a pixel
# value) but not bit-identical to transform() — if you see any drift in DL
# scores after updating, compare a known test clip's dl_raw trace before
# and after to confirm nothing meaningful changed.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def frame_to_tensor_fast(frame_bgr):
    """Fast equivalent of transform(Image.fromarray(rgb)) for one OpenCV BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (240, 240), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    return tensor.to(DEVICE)


# ================= IMAGE PREDICT =================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    frames = torch.stack([image] * SEQUENCE_LEN)
    frames = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(frames)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return CLASS_NAMES[pred.item()], float(conf.item() * 100)


# ================= VIDEO PREDICT =================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return "ERROR", 0.0

    if len(frames) >= SEQUENCE_LEN:
        frames = frames[-SEQUENCE_LEN:]
    else:
        last = frames[-1]
        while len(frames) < SEQUENCE_LEN:
            frames.append(last)

    frames = torch.stack(frames).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(frames)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return CLASS_NAMES[pred.item()], float(conf.item() * 100)
