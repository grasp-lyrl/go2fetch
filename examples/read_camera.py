import argparse
import time

import cv2
import rerun as rr
import rerun.blueprint as rrb

from go2_interface.camera import make_camera_reader


parser = argparse.ArgumentParser()
parser.add_argument("iface")
parser.add_argument("--hz", type=float, default=15.0)
parser.add_argument("--jpeg-quality", type=int, default=75)
args = parser.parse_args()

blueprint = rrb.Blueprint(rrb.Spatial2DView(name="Camera", origin="/camera"))

rr.init("go2fetch-camera")
rr.spawn()
rr.send_blueprint(blueprint)

read_camera = make_camera_reader(args.iface, hz=args.hz)
last_frame = None

try:
    while True:
        frame = read_camera()

        if frame is None or frame is last_frame:
            time.sleep(0.001)
            continue

        last_frame = frame
        rr.set_time("time", timestamp=time.time())

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("camera/image", rr.Image(rgb).compress(jpeg_quality=args.jpeg_quality))

except KeyboardInterrupt:
    pass

finally:
    read_camera.stop()
    rr.disconnect()
