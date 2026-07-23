from ultralytics import YOLO
import cv2
import numpy as np
import time

from go2_interface.camera import make_camera_reader

model = YOLO("yolo26n.pt")
CONF = 0.8

def process_frame(frame):

    results = model(frame, device="mps", verbose=False)

    frame_detections = []
    annotated = frame.copy()

    for box in results[0].boxes:
        conf = float(box.conf)
        class_id = int(box.cls)
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{class_name} {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        if conf < CONF:
            continue

        frame_detections.append({
            "class": class_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

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