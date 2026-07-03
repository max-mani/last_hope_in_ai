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
