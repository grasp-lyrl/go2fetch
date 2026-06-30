import threading
import time

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


def make_video_client(iface, timeout=3.0):
    ChannelFactoryInitialize(0, iface)
    client = VideoClient()
    client.SetTimeout(timeout)
    client.Init()
    return client


def decode_image(data):
    image_data = np.frombuffer(bytes(data), dtype=np.uint8)
    return cv2.imdecode(image_data, cv2.IMREAD_COLOR)


class CameraReader:
    def __init__(self, iface, hz=15.0, decode=True):
        self.client = make_video_client(iface)
        self.frame = None
        self.encoded = None
        self.code = None
        self.decode_ok = False
        self.decode = decode
        self.running = True
        self.dt = 1.0 / hz if hz > 0 else 0.0

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            code, data = self.client.GetImageSample()
            self.code = code

            if code == 0:
                self.encoded = bytes(data)

                if self.decode:
                    frame = decode_image(self.encoded)
                    if frame is not None:
                        self.frame = frame
                        self.decode_ok = True
                    else:
                        self.decode_ok = False
                else:
                    self.decode_ok = True

            if self.dt > 0:
                time.sleep(self.dt)

    def read(self):
        return self.frame

    def read_encoded(self):
        return self.encoded

    def status(self):
        return {"code": self.code, "decode_ok": self.decode_ok}

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

    def __call__(self):
        return self.read()


def make_camera_reader(iface, hz=15.0):
    return CameraReader(iface, hz)


def make_encoded_camera_reader(iface, hz=15.0):
    return CameraReader(iface, hz, decode=False)
