import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rerun as rr
import rerun.blueprint as rrb

from go2_interface.lidar import make_lidar_reader, pointcloud_to_xyz
from go2_interface.state import make_state_reader, state_to_dict


p = argparse.ArgumentParser()
p.add_argument("iface")
p.add_argument("--out", default=None)
p.add_argument("--live", action="store_true")
p.add_argument("--stride", type=int, default=1)
p.add_argument("--lidar-hz", type=float, default=0.0)
p.add_argument("--window", type=float, default=10.0)
args = p.parse_args()

if args.live == bool(args.out):
    raise SystemExit("use exactly one: --live or --out logs/name.rrd")


def ts(name, origin):
    return rrb.TimeSeriesView(
        name=name,
        origin=origin,
        axis_x=rrb.TimeAxis(
            view_range=rr.TimeRange(
                start=rrb.TimeRangeBoundary.cursor_relative(seconds=-args.window),
                end=rrb.TimeRangeBoundary.cursor_relative(),
            ),
            zoom_lock=True,
        ),
    )


blueprint = rrb.Blueprint(rrb.Tabs(
    rrb.Vertical(
        ts("Position", "/state/position"),
        ts("Velocity", "/state/velocity"),
        name="State",
    ),
    rrb.Vertical(
        ts("Gyroscope", "/imu/gyroscope"),
        ts("Accelerometer", "/imu/accelerometer"),
        ts("RPY", "/imu/rpy"),
        ts("Quaternion", "/imu/quaternion"),
        name="IMU",
    ),
    rrb.Spatial3DView(name="Lidar", origin="/lidar"),
))

rr.init("go2fetch")

if args.live:
    rr.spawn()

if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rr.save(args.out)

rr.send_blueprint(blueprint)

read_state = make_state_reader(args.iface)
read_lidar = make_lidar_reader(args.iface)

last_state_key = None
last_lidar_key = None
last_lidar_log = 0.0
lidar_dt = 0.0 if args.lidar_hz <= 0 else 1.0 / args.lidar_hz


def log_scalar(path, value):
    rr.log(path, rr.Scalars(value))


def log_state(s):
    imu = s["imu_state"]

    log_scalar("state/position/x", s["position"][0])
    log_scalar("state/position/y", s["position"][1])
    log_scalar("state/position/z", s["position"][2])
    log_scalar("state/velocity/x", s["velocity"][0])
    log_scalar("state/velocity/y", s["velocity"][1])
    log_scalar("state/velocity/z", s["velocity"][2])

    log_scalar("imu/gyroscope/x", imu["gyroscope"][0])
    log_scalar("imu/gyroscope/y", imu["gyroscope"][1])
    log_scalar("imu/gyroscope/z", imu["gyroscope"][2])
    log_scalar("imu/accelerometer/x", imu["accelerometer"][0])
    log_scalar("imu/accelerometer/y", imu["accelerometer"][1])
    log_scalar("imu/accelerometer/z", imu["accelerometer"][2])
    log_scalar("imu/rpy/roll", imu["rpy"][0])
    log_scalar("imu/rpy/pitch", imu["rpy"][1])
    log_scalar("imu/rpy/yaw", imu["rpy"][2])
    log_scalar("imu/quaternion/w", imu["quaternion"][0])
    log_scalar("imu/quaternion/x", imu["quaternion"][1])
    log_scalar("imu/quaternion/y", imu["quaternion"][2])
    log_scalar("imu/quaternion/z", imu["quaternion"][3])


try:
    while True:
        now = time.time()
        rr.set_time("time", timestamp=now)

        state_msg = read_state()
        if state_msg is not None:
            state_key = (state_msg.stamp.sec, state_msg.stamp.nanosec)

            if state_key != last_state_key:
                last_state_key = state_key
                log_state(state_to_dict(state_msg))

        lidar_msg = read_lidar()
        if lidar_msg is not None:
            lidar_key = (lidar_msg.header.stamp.sec, lidar_msg.header.stamp.nanosec)

            if (
                lidar_key != last_lidar_key
                and (lidar_dt == 0.0 or now - last_lidar_log >= lidar_dt)
            ):
                last_lidar_key = lidar_key
                last_lidar_log = now

                xyz = pointcloud_to_xyz(lidar_msg)[::args.stride]
                rr.log("lidar/points", rr.Points3D(xyz, radii=0.01))

        time.sleep(0.001)

except KeyboardInterrupt:
    pass

finally:
    rr.disconnect()
