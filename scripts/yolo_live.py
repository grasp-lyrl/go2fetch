from ultralytics import YOLO
#import rerun as rr
import cv2
import numpy as np
import time

from go2_interface.camera import make_camera_reader


model = YOLO("yolo11n.pt")
CONF = 0.5

#rr.init("go2fetch_yolo")

#recording = cv2.VideoCapture("../logs/levine.mp4") #for test recordings

camera = make_camera_reader("en7") 
while camera.read() is None: 
    time.sleep(0.01)

print("camera connected")

first_frame = camera.read()

height, width = first_frame.shape[:2]
fps = 15

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter("data/live_yolo.mp4", fourcc, fps, (width, height))

all_frames = []

def process_frame(frame, index):
    #rr.set_time_sequence("frame", index)

    results = model(frame, verbose=False)

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