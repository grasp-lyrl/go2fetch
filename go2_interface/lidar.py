import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_


TOPIC = "rt/utlidar/cloud"


def make_lidar_reader(iface):
    latest = {"msg": None, "sub": None}

    def on_cloud(msg):
        latest["msg"] = msg

    ChannelFactoryInitialize(0, iface)
    latest["sub"] = ChannelSubscriber(TOPIC, PointCloud2_)
    latest["sub"].Init(on_cloud)

    return lambda: latest["msg"]


def pointcloud_to_xyz(msg):
    fields = {f.name: f for f in msg.fields}
    dtype = ">f4" if msg.is_bigendian else "<f4"
    data = bytes(msg.data)
    n = msg.width * msg.height

    return np.stack(
        [
            np.ndarray(n, dtype=dtype, buffer=data, offset=fields[k].offset, strides=(msg.point_step,))
            for k in ("x", "y", "z")
        ],
        axis=1,
    )


def pointcloud_info(msg):
    return {
        "stamp": [msg.header.stamp.sec, msg.header.stamp.nanosec],
        "frame_id": msg.header.frame_id,
        "width": msg.width,
        "height": msg.height,
        "points": msg.width * msg.height,
        "point_step": msg.point_step,
    }
    