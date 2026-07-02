import numpy as np
import matplotlib.pyplot as plt

from examples.read_lidar_rrd import get_lidar_points
from examples.read_state_rrd import get_state_stream

Z_MIN = 0 #change

RESOLUTION = 0.05        

MAP_WIDTH = 1000      
MAP_HEIGHT = 1000

ORIGIN_X = -10.0          
ORIGIN_Y = -10.0

WORLD_OFFSET_Y = 10
WORLD_OFFSET_X = 0

occupancy_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)

#change
def lidar_to_robot(points):

    points[:, 0] += 0.28945
    points[:, 1] += 0.0
    points[:, 2] += -0.046825

    points[:, 2] *= -1

    return points

def robot_to_world(points, position, yaw):

    c = np.cos(yaw)
    s = np.sin(yaw)

    R = np.array([
        [c, -s],
        [s,  c]
    ])

    xy = points[:, :2]

    xy_world = (R @ xy.T).T

    xy_world[:, 0] += position[0]
    xy_world[:, 1] += position[1]

    return np.column_stack((xy_world, points[:, 2]))


def filter_height(points):

    mask = (
        (points[:, 2] >= Z_MIN) 
    )

    return points[mask]

def world_to_grid(points):

    gx = ((points[:, 0] + WORLD_OFFSET_X - ORIGIN_X) / RESOLUTION).astype(int)
    gy = ((points[:, 1] + WORLD_OFFSET_Y - ORIGIN_Y) / RESOLUTION).astype(int)

    valid = (
        (gx >= 0) &
        (gx < MAP_WIDTH) &
        (gy >= 0) &
        (gy < MAP_HEIGHT)
    )

    return gx[valid], gy[valid]


def update_grid(grid, xy):

    gx, gy = world_to_grid(xy)
    
    if len(gx) > 0:
        grid[gy, gx] = 1


lidar_stream = get_lidar_points("logs/levine.rrd")
state_stream = get_state_stream("logs/levine.rrd")

state_iter = iter(state_stream)

t_state, robot_position, robot_yaw = next(state_iter)

for t_lidar, lidar_points in lidar_stream:

    try:
        while t_state < t_lidar:
            t_next, pos_next, yaw_next = next(state_iter)
            
            if t_next > t_lidar:
                break
                
            t_state = t_next
            robot_position = pos_next
            robot_yaw = yaw_next

    except StopIteration:
        pass

    robot_points = lidar_to_robot(lidar_points)

    world_points = robot_to_world(
        robot_points,
        robot_position,
        robot_yaw
    )

    world_points = filter_height(world_points)

    xy_world = world_points[:, :2]

    update_grid(occupancy_grid, xy_world)

print(occupancy_grid)
print("occupied:", np.sum(occupancy_grid == 1))
print("free:", np.sum(occupancy_grid == 0))


plt.figure(figsize=(8, 8))

img = np.zeros_like(occupancy_grid, dtype=np.uint8)
img[occupancy_grid == 1] = 0      
img[occupancy_grid == 0] = 255    

plt.imshow(img, cmap="gray", vmin=0, vmax=255, origin="lower")
plt.title("Occupancy Grid")
plt.axis("off")
plt.show()
