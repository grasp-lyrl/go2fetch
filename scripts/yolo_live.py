from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch

from go2_interface.camera import make_camera_reader

model = YOLO("yolo26s.pt")
CONF = 0.5

def _yolo_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _yolo_device()
print(f"YOLO device: {DEVICE}")

def process_frame(frame):

    results = model(frame, device=DEVICE, verbose=False)

    frame_detections = []

    for box in results[0].boxes:
        conf = float(box.conf)

        if conf < CONF:
            continue

        class_id = int(box.cls)
        class_name = model.names[class_id]

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

        frame_detections.append({
            "class": class_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    annotated = results[0].plot()
    annotated = np.asarray(annotated, dtype=np.uint8)

    return annotated, frame_detections

"""
index = 0

try:
    while True:
        frame = camera.read()

        if frame is None:
            continue

        annotated_frame, frame_detections = process_frame(frame, index)

        all_frames.append({
            "frame": index,
            "detections": frame_detections
        })

        cv2.imshow("yolo", annotated_frame)

        output.write(annotated_frame)

        for d in frame_detections:
            print(f"{d['class']}, {d['confidence']:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        index += 1

finally: 
    camera.stop()
    output.release()
    cv2.destroyAllWindows()
"""