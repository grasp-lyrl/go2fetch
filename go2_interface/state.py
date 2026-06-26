from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


TOPIC = "rt/sportmodestate"


def make_state_reader(iface):
    latest = {"msg": None, "sub": None}

    def on_state(msg):
        latest["msg"] = msg

    ChannelFactoryInitialize(0, iface)
    latest["sub"] = ChannelSubscriber(TOPIC, SportModeState_)
    latest["sub"].Init(on_state)

    return lambda: latest["msg"]


def state_to_dict(msg):
    imu = msg.imu_state
    return {
        "stamp": [msg.stamp.sec, msg.stamp.nanosec],
        "imu_state": {
            "quaternion": list(imu.quaternion),
            "gyroscope": list(imu.gyroscope),
            "accelerometer": list(imu.accelerometer),
            "rpy": list(imu.rpy),
        },
        "position": list(msg.position),
        "velocity": list(msg.velocity),
        "yaw_speed": float(msg.yaw_speed),
    }
    