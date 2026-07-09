import rerun as rr
import numpy as np
import time


def test(rrd_path: str):

    server = rr.server.Server(datasets={"datasets": [rrd_path]})
    recording = server.client().get_dataset("datasets")

    query = recording.reader(index="time")
    df = query.to_pandas()

    required = [
        "/state/position/x:Scalars:scalars",
        "/state/position/y:Scalars:scalars",
        "/imu/rpy/yaw:Scalars:scalars",
    ]

    for col in required:
        if col not in df.columns:
            print("Missing:", col)
            return

    for _, row in df.iterrows():

        x = row["/state/position/x:Scalars:scalars"]
        y = row["/state/position/y:Scalars:scalars"]
        z = row["/state/position/z:Scalars:scalars"]

        yaw = row["/imu/rpy/yaw:Scalars:scalars"]

        position = np.array([x.item(), y.item(), z.item()])
        yaw = float(yaw.item())

        if x is None or y is None or yaw is None:
            continue

        print("pos:", (x, y, z))
        print("yaw:", yaw)

        time.sleep(0.05)


def get_state_stream(rrd_path: str):

    server = rr.server.Server(datasets={"datasets": [rrd_path]})
    recording = server.client().get_dataset("datasets")

    query = recording.reader(index="time")
    df = query.to_pandas()

    # Added roll and pitch columns to required list
    required = [
        "/state/position/x:Scalars:scalars",
        "/state/position/y:Scalars:scalars",
        "/state/position/z:Scalars:scalars",
        "/imu/rpy/roll:Scalars:scalars",
        "/imu/rpy/pitch:Scalars:scalars",
        "/imu/rpy/yaw:Scalars:scalars",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    for timestamp, row in df.iterrows():

        x = row["/state/position/x:Scalars:scalars"]
        y = row["/state/position/y:Scalars:scalars"]
        z = row["/state/position/z:Scalars:scalars"]

        roll = row["/imu/rpy/roll:Scalars:scalars"]
        pitch = row["/imu/rpy/pitch:Scalars:scalars"]
        yaw = row["/imu/rpy/yaw:Scalars:scalars"]

        if x is None or y is None or z is None or roll is None or pitch is None or yaw is None:
            continue

        position = np.array([x.item(), y.item(), z.item()])
        rpy = np.array([float(roll.item()), float(pitch.item()), float(yaw.item())])

        yield timestamp, position, rpy