import contextlib
import io

from ultralytics import YOLO
import numpy as np
import torch

MODEL_PATH = "yolo26m.pt"
CONF = 0.50
IMGSZ = 640
MAX_DET = 50
SHOW_CLASSES = ("person", "tv", "backpack", "chair", "bench")


def _yolo_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _yolo_device()
QUANT = "fp16" if DEVICE == "cuda" else "fp32"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

model = YOLO(MODEL_PATH)
with contextlib.redirect_stdout(io.StringIO()):
    model.fuse()
SHOW_CLASS_IDS = [
    i for i, name in model.names.items() if name in SHOW_CLASSES
]
print(f"YOLO: {DEVICE} | {MODEL_PATH}")


def process_frame(frame):
    results = model.predict(
        frame,
        device=DEVICE,
        conf=CONF,
        imgsz=IMGSZ,
        quantize=QUANT,
        max_det=MAX_DET,
        classes=SHOW_CLASS_IDS,
        verbose=False,
    )

    frame_detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        frame_detections.append({
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": [x1, y1, x2, y2],
        })

    annotated = np.asarray(results[0].plot(), dtype=np.uint8)
    return annotated, frame_detections
