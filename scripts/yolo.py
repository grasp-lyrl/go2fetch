#https://rerun.io/docs/getting-started/data-in

from ultralytics import YOLO
import rerun as rr

model = YOLO("yolov8n.pt") #change later
stream_path = "" #change later, do i even need?
CONF = 0.5 #idk change later
all_frames = []

rr.init("go2fetch_yolo")

# *ask how to loard the stream recording 
recording = None #change later

for index, image in enumerate(recording):
    rr.set_time_sequence("frame", index)

    results = model(image)

    frame_detections = []

    for box in results[0].boxes:
        conf = float(box.conf)

        if conf<CONF:
            continue

        class_id = int(box.cls)
        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detect = {
            "class": class_name,
            "confidence": conf, 
            "bbox": [x1, y1, x2, y2]
        }

        frame_detections.append(detect)

    all_frames.append({
        "frame":index,
        "detections":frame_detections
        })

    for d in frame_detections:
        print(f"{d['class']}, {d['confidence']:.2f}")






