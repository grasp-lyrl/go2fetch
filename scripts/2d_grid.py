from examples.read_lidar_rrd import get_lidar_points

lidar_stream = get_lidar_points("logs/levine.rrd")

for xyz in lidar_stream:
    print(xyz)