import argparse
import time

from go2_interface.lidar import make_lidar_reader, pointcloud_info, pointcloud_to_xyz


parser = argparse.ArgumentParser()
parser.add_argument("iface")
args = parser.parse_args()

read_lidar = make_lidar_reader(args.iface)
last_msg = None

while True:
    msg = read_lidar()

    if msg is None or msg is last_msg:
        time.sleep(0.001)
        continue

    last_msg = msg
    xyz = pointcloud_to_xyz(msg)

    print(pointcloud_info(msg), xyz[:3])
    time.sleep(1)
    