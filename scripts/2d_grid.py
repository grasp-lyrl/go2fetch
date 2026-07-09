import numpy as np
import matplotlib.pyplot as plt
import time

from examples.read_lidar_rrd import get_lidar_points
from examples.read_state_rrd import get_state_stream

RESOLUTION = 0.05        
MAP_WIDTH = 2000     
MAP_HEIGHT = 1000

ORIGIN_X = -25.0          
ORIGIN_Y = -25.0

Z_MIN = 0.1
Z_MAX = 1.6
EXCLUSION_RADIUS = 0.45
MIN_RANGE = 0.2
MAX_RANGE = 12.0

occupancy_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int16)


def lidar_to_robot(points):
    pitch_lidar = 2.8782
    cp = np.cos(pitch_lidar)
    sp = np.sin(pitch_lidar)

    R_lidar = np.array([
        [ cp,  0.0,  sp],
        [ 0.0, 1.0, 0.0],
        [-sp,  0.0,  cp]
    ])

    xyz = points[:, :3]
    xyz_rotated = (R_lidar @ xyz.T).T

    xyz_rotated[:, 0] += 0.28945
    xyz_rotated[:, 1] += 0.0
    xyz_rotated[:, 2] += -0.046825

    return xyz_rotated


def robot_to_world(points, position, rpy):
    roll, pitch, yaw = rpy

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cr, -sr],
        [0.0,  sr,  cr]
    ])
    R_y = np.array([
        [ cp, 0.0,  sp],
        [ 0.0, 1.0, 0.0],
        [-sp, 0.0,  cp]
    ])
    R_z = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    R_body = R_z @ R_y @ R_x

    xyz = points[:, :3]
    xyz_world = (R_body @ xyz.T).T

    xyz_world[:, 0] += position[0]
    xyz_world[:, 1] += position[1]
    xyz_world[:, 2] += position[2]

    return xyz_world


def filter_height(points, robot_position):
    height_mask = (points[:, 2] >= Z_MIN) & (points[:, 2] <= Z_MAX)
    
    relative_vectors = points[:, :3] - robot_position
    
    distances_3d = np.linalg.norm(relative_vectors, axis=1)
    body_mask = (distances_3d > EXCLUSION_RADIUS)

    distances_2d = np.linalg.norm(relative_vectors[:, :2], axis=1)
    range_mask = (distances_2d >= MIN_RANGE) & (distances_2d <= MAX_RANGE)
    
    final_mask = height_mask & body_mask & range_mask
    return points[final_mask]

def world_to_grid(points):

    gx = ((points[:, 0] - ORIGIN_X) / RESOLUTION).astype(int)
    gy = ((points[:, 1] - ORIGIN_Y) / RESOLUTION).astype(int)

    valid = (
        (gx >= 0) &
        (gx < MAP_WIDTH) &
        (gy >= 0) &
        (gy < MAP_HEIGHT)
    )

    return gx[valid], gy[valid]


def bresenham_ray(x0, y0, x1, y1):
    cells = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:

        cells.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy

    return cells

def update_grid(grid, xy, robot_position):

    gx, gy = world_to_grid(xy)
    rx, ry = world_to_grid(robot_position.reshape(1, 2))

    if len(rx) == 0:
        return

    robot_x = rx[0]
    robot_y = ry[0]

    for x, y in zip(gx, gy):

        ray = bresenham_ray(robot_x, robot_y, x, y)

        for free_x, free_y in ray[:-1]:
            grid[free_y, free_x] -= 1
            if grid[free_y, free_x] < -100:
                grid[free_y, free_x] = -100

        end_x, end_y = ray[-1]
        grid[end_y, end_x] += 4
        if grid[end_y, end_x] > 100:
            grid[end_y, end_x] = 100

def interpolate_state(t_lidar, t1, pos1, rpy1, t2, pos2, rpy2):
    if t2 == t1:
        return pos1, rpy1
    
    lmbda = (t_lidar - t1) / (t2 - t1)
    
    interp_pos = (1 - lmbda) * pos1 + lmbda * pos2
    
    interp_rpy = (1 - lmbda) * rpy1 + lmbda * rpy2
    
    return interp_pos, interp_rpy

lidar_stream = get_lidar_points("logs/levine.rrd")

state_stream = get_state_stream("logs/levine.rrd")
state_iter = iter(state_stream)

t_prev, pos_prev, rpy_prev = next(state_iter)
t_next, pos_next, rpy_next = next(state_iter)

ORIGIN_X = pos_prev[0] - (MAP_WIDTH * RESOLUTION) / 2.0
ORIGIN_Y = pos_prev[1] - (MAP_HEIGHT * RESOLUTION) / 2.0

previous_lidar_time = None
trajectory_points = []

plt.ion()
fig, ax = plt.subplots(figsize=(9, 9))

for t_lidar, lidar_points in lidar_stream:
    
    if previous_lidar_time is not None:
        dt = (t_lidar - previous_lidar_time) / 1000.0
        time.sleep(dt)

    previous_lidar_time = t_lidar

    try:
        while t_next < t_lidar:
            t_prev, pos_prev, rpy_prev = t_next, pos_next, rpy_next
            t_next, pos_next, rpy_next = next(state_iter)

    except StopIteration:
        pass

    robot_position, robot_rpy = interpolate_state(
        t_lidar, 
        t_prev, pos_prev, rpy_prev, 
        t_next, pos_next, rpy_next
    )

    robot_points = lidar_to_robot(lidar_points)

    world_points = robot_to_world(
        robot_points,
        robot_position,
        robot_rpy
    )
    world_points = filter_height(world_points, robot_position)

    xy_world = world_points[:, :2]
    update_grid(
        occupancy_grid,
        xy_world,
        robot_position[:2]
    )
    
    trajectory_points.append(robot_position[:2].copy())


    if len(trajectory_points) % 10 == 0:

        img = np.zeros_like(occupancy_grid, dtype=np.uint8)
        img[:] = 127
        img[occupancy_grid >= 8] = 0
        img[occupancy_grid <= -8] = 255

        ax.clear()

        ax.imshow(
            img,
            cmap="gray",
            vmin=0,
            vmax=255,
            origin="lower",
            extent=[
                ORIGIN_X,
                ORIGIN_X + MAP_WIDTH * RESOLUTION,
                ORIGIN_Y,
                ORIGIN_Y + MAP_HEIGHT * RESOLUTION
            ]
        )

        trajectory_np = np.array(trajectory_points)

        ax.plot(
            trajectory_np[:, 0],
            trajectory_np[:, 1],
            color="blue",
            linewidth=1.5
        )

        ax.scatter(
        trajectory_np[0, 0],
        trajectory_np[0, 1],
        color="green",
        s=50,
        zorder=5
        )

        ax.set_title("Live Occupancy Grid")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        plt.pause(0.001)



print("occupied:", np.sum(occupancy_grid > 0))

x_min = ORIGIN_X
x_max = ORIGIN_X + (MAP_WIDTH * RESOLUTION)
y_min = ORIGIN_Y
y_max = ORIGIN_Y + (MAP_HEIGHT * RESOLUTION)

plt.figure(figsize=(9, 9))

img = np.zeros_like(occupancy_grid, dtype=np.uint8)
img[:] = 127
img[occupancy_grid >= 8] = 0
img[occupancy_grid <= -8] = 255

plt.imshow(img, cmap="gray", vmin=0, vmax=255, origin="lower", extent=[x_min, x_max, y_min, y_max])

trajectory_np = np.array(trajectory_points)
plt.plot(trajectory_np[:, 0], trajectory_np[:, 1], color="blue", linewidth=1.5, label="trajectory")

plt.scatter(trajectory_np[0, 0], trajectory_np[0, 1], color="green", s=50, zorder=5, label="start")
plt.scatter(trajectory_np[-1, 0], trajectory_np[-1, 1], color="red", s=50, zorder=5, label="end")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("grid with trajectory (m)")
plt.grid(True, color="gainsboro", linestyle="--", linewidth=0.5)
plt.legend(loc="upper right")

plt.show()