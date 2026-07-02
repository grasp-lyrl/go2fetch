import rerun as rr
import time
import numpy as np

def test(rrd_path: str):
    
    server = rr.server.Server(datasets={"datasets": [rrd_path]})
    recording = server.client().get_dataset("datasets")

    query = recording.reader(index="time")
    df = query.to_pandas()

    column_name = "/lidar/points:Points3D:positions"

    for _, row in df.iterrows():
        xyz = row[column_name]

        if xyz is not None and len(xyz) > 0:
            print(xyz[:3])
            time.sleep(1)


def get_lidar_points(rrd_path: str):

    server = rr.server.Server(datasets={"datasets": [rrd_path]})
    recording = server.client().get_dataset("datasets")

    query = recording.reader(index="time")
    df = query.to_pandas()

    column_name = "/lidar/points:Points3D:positions"

    for timestamp, row in df.iterrows():

        xyz = row[column_name]

        if xyz is None:
            continue

        points = np.vstack(xyz)

        yield timestamp, points