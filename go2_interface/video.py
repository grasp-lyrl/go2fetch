import os
import shutil
import signal
import subprocess


MULTICAST_ADDRESS = "230.1.1.1"
VIDEO_PORT = 1720
RTP_H264_CAPS = "application/x-rtp,media=(string)video,encoding-name=(string)H264,payload=(int)96"


def default_video_path(rrd_path):
    root, _ = os.path.splitext(rrd_path)
    return f"{root}.mp4"


def make_h264_pipeline(iface, out=None, live=False, address=MULTICAST_ADDRESS, port=VIDEO_PORT):
    src = [
        "udpsrc",
        f"address={address}",
        f"port={port}",
        f"multicast-iface={iface}",
        f"caps={RTP_H264_CAPS}",
        "!",
        "rtph264depay",
        "!",
        "h264parse",
        "config-interval=-1",
    ]

    if live and out:
        return [
            *src,
            "!",
            "tee",
            "name=t",
            "t.",
            "!",
            "queue",
            "!",
            "mp4mux",
            "faststart=true",
            "!",
            "filesink",
            f"location={out}",
            "t.",
            "!",
            "queue",
            "leaky=downstream",
            "max-size-buffers=1",
            "!",
            "decodebin",
            "!",
            "videoconvert",
            "!",
            "autovideosink",
            "sync=false",
        ]

    if out:
        return [
            *src,
            "!",
            "mp4mux",
            "faststart=true",
            "!",
            "filesink",
            f"location={out}",
        ]

    if live:
        return [
            *src,
            "!",
            "decodebin",
            "!",
            "videoconvert",
            "!",
            "autovideosink",
            "sync=false",
        ]

    raise ValueError("use live=True, out=path, or both")


class H264VideoProcess:
    def __init__(self, iface, out=None, live=False):
        if shutil.which("gst-launch-1.0") is None:
            raise RuntimeError("gst-launch-1.0 not found; install GStreamer tools/plugins")

        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

        self.out = out
        self.cmd = ["gst-launch-1.0", "-e", *make_h264_pipeline(iface, out=out, live=live)]
        self.proc = subprocess.Popen(self.cmd)

    def stop(self):
        if self.proc.poll() is not None:
            return

        self.proc.send_signal(signal.SIGINT)

        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=2.0)


def start_h264_video(iface, out=None, live=False):
    return H264VideoProcess(iface, out=out, live=live)
