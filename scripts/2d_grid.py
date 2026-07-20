import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard
import cv2

from examples.read_lidar_rrd import get_lidar_points
from examples.read_state_rrd import get_state_stream
from scripts import exploration

from go2_interface.lidar import make_lidar_reader, pointcloud_to_xyz
from go2_interface.state import make_state_reader
from go2_interface.command import make_sport_client, move, stop

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

FORWARD_SPEED = 0.1
TURN_SPEED = 0.5

def create_grid():
    return np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int16)

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

def plot_grid(grid, trajectory_points, origin_x, origin_y):

    print("occupied:", np.sum(grid > 0))

    x_min = origin_x
    x_max = origin_x + (MAP_WIDTH * RESOLUTION)
    y_min = origin_y
    y_max = origin_y + (MAP_HEIGHT * RESOLUTION)

    plt.figure(figsize=(9, 9))

    img = np.zeros_like(grid, dtype=np.uint8)
    img[:] = 127
    img[grid >= 8] = 0
    img[grid <= -8] = 255

    plt.imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max]
    )

    trajectory_np = np.array(trajectory_points)

    plt.plot(
        trajectory_np[:, 0],
        trajectory_np[:, 1],
        color="blue",
        linewidth=1.5,
        label="trajectory"
    )

    plt.scatter(
        trajectory_np[0, 0],
        trajectory_np[0, 1],
        color="green",
        s=50,
        zorder=5,
        label="start"
    )

    plt.scatter(
        trajectory_np[-1, 0],
        trajectory_np[-1, 1],
        color="red",
        s=50,
        zorder=5,
        label="end"
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("grid with trajectory (m)")
    plt.legend()
    plt.show(block=True)


if __name__ == "__main__":

    USE_CV2 = True
    REALTIME_REPLAY = False

    vx = 0.0
    vy = 0.0
    vyaw = 0.0

    goal_x = None
    goal_y = None

    execute_path = False

    occupancy_grid = create_grid()

    # lidar_stream = get_lidar_points("logs/levine.rrd")

    # state_stream = get_state_stream("logs/levine.rrd")
    # state_iter = iter(state_stream)

    # t_prev, pos_prev, rpy_prev = next(state_iter)
    # t_next, pos_next, rpy_next = next(state_iter)

    get_lidar = make_lidar_reader("en7")
    get_state = make_state_reader("en7")

    client = make_sport_client("en7")

    state_msg = None

    while state_msg is None:
        state_msg = get_state()

    initial_position = np.array(state_msg.position)

    # ORIGIN_X = pos_prev[0] - (MAP_WIDTH * RESOLUTION) / 2.0
    # ORIGIN_Y = pos_prev[1] - (MAP_HEIGHT * RESOLUTION) / 2.0

    ORIGIN_X = initial_position[0] - (MAP_WIDTH * RESOLUTION) / 2.0
    ORIGIN_Y = initial_position[1] - (MAP_HEIGHT * RESOLUTION) / 2.0


    previous_lidar_time = None
    trajectory_points = []
    frame_count = 0


    if not USE_CV2:

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.set_xlim(
            ORIGIN_X,
            ORIGIN_X + (MAP_WIDTH * RESOLUTION)
        )

        ax.set_ylim(
            ORIGIN_Y,
            ORIGIN_Y + (MAP_HEIGHT * RESOLUTION)
        )
        
        display_img = np.zeros(
            (MAP_HEIGHT, MAP_WIDTH, 3),
            dtype=np.uint8
        )

        im_artist = ax.imshow(
            display_img,
            origin="lower",
            extent=[
                ORIGIN_X,
                ORIGIN_X + (MAP_WIDTH * RESOLUTION),
                ORIGIN_Y,
                ORIGIN_Y + (MAP_HEIGHT * RESOLUTION)
            ]
        )
        
        frontiers_scat = ax.scatter(
            [], [],
            c='cyan',
            s=2,
            label='Frontiers',
            zorder=2
        )

        all_centers_scat = ax.scatter(
            [],
            [],
            c='orange',
            marker='o',
            s=10,
            edgecolors='black',
            label='All Cluster Centers',
            zorder=3
        )

        selected_scat = ax.scatter(
            [],
            [],
            c='magenta',
            s=6,
            label='Selected Goal Cluster',
            zorder=4
        )

        goal_scat = ax.scatter(
            [],
            [],
            c='yellow',
            marker='X',
            s=25,
            edgecolors='black',
            label='Goal Centroid',
            zorder=6
        )
        
        trajectory_line, = ax.plot(
            [],
            [],
            c='blue',
            linewidth=1.0,
            alpha=0.7,
            zorder=3,
            label='Traveled Path'
        )

        path_line, = ax.plot(
            [],
            [],
            c='blue',
            linewidth=2,
            zorder=5,
            label='Planned Path'
        )

        robot_marker = ax.scatter(
            [],
            [],
            c='blue',
            marker='s',
            s=20,
            label='Robot',
            zorder=11
        )
        
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Exploration Pipeline: BFS Planning")
        ax.legend(loc='upper right')


    else:

        fig = ax = im_artist = None
        frontiers_scat = all_centers_scat = selected_scat = goal_scat = None
        trajectory_line = path_line = robot_marker = None

    robot_mode = "PLANNING"

    current_goal = None
    current_path = None

    active_goal = None
    active_path = None

    goal_x = None
    goal_y = None

    vx = 0.0
    vy = 0.0
    vyaw = 0.0

    path_index = 10
    start_time = None
    started = False

    # for t_lidar, lidar_points in lidar_stream:
    while True:

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):

            stop(client)

            print("s was pressed, stopped, planning again")

            active_path = None
            active_goal = None

            robot_mode = "PLANNING"

            continue

        lidar_msg = get_lidar()
        state_msg = get_state()

        if lidar_msg is None or state_msg is None:
            continue

        lidar_points = pointcloud_to_xyz(lidar_msg)

        robot_position = np.array(state_msg.position)
        robot_rpy = np.array(state_msg.imu_state.rpy)

        robot_points = lidar_to_robot(lidar_points)

        world_points = robot_to_world(robot_points, robot_position, robot_rpy)

        world_points = filter_height(world_points, robot_position)

        update_grid(occupancy_grid, world_points[:, :2], robot_position[:2])

        trajectory_points.append(robot_position[:2].copy())


        if robot_mode == "PLANNING":
            frame_count += 1

            if frame_count % 15 == 0:

                rx_grid, ry_grid = world_to_grid(robot_position[:2].reshape(1, 2))

                if len(rx_grid) > 0:

                    robot_grid_cell = (ry_grid[0], rx_grid[0])

                    reachable_set, parent_map, cost_map = exploration.compute_reachability(
                        occupancy_grid,
                        robot_grid_cell,
                        resolution=RESOLUTION
                    )

                    frontier_cells = exploration.detect_frontiers_cv2(
                        occupancy_grid,
                        robot_grid_cell,
                        resolution=RESOLUTION
                    )

                    frontier_clusters = exploration.cluster_frontiers_cv2(
                        occupancy_grid,
                        frontier_cells,
                        reachable_set,
                        cost_map,
                        resolution=RESOLUTION
                    )

                    reachable_clusters = [
                        c for c in frontier_clusters if c["reachable"]
                    ]

                    if reachable_clusters:

                        current_goal = max(
                            reachable_clusters,
                            key=lambda c: (0.4 * c["size"] - 0.6 * c["cost"])
                        )

                        goal_row, goal_col = current_goal["center"]

                        goal_x = (goal_col + 0.5) * RESOLUTION + ORIGIN_X
                        goal_y = (goal_row + 0.5) * RESOLUTION + ORIGIN_Y

                        current_path = exploration.plan_path(
                            occupancy_grid,
                            goal=(goal_row, goal_col),
                            parent_map=parent_map,
                            reachable_set=reachable_set
                        )

                        if current_path:

                            if started:

                                active_path = current_path.copy()
                                active_goal = current_goal
                                path_index = min(10, len(active_path)-1)
                                start_time = time.time()

                                print("NEW PATH FOUND - STARTING")

                                robot_mode = "STARTING"

                            else:
                                print("FIRST PATH READY - PRESS W")
                                robot_mode = "WAITING"

        elif robot_mode == "STARTING":
            if time.time() - start_time > 0.1:

                print("EXECUTING")

                robot_mode = "EXECUTING"


        elif robot_mode == "WAITING":
            if key == ord('w'):

                if current_path and goal_x is not None and goal_y is not None:

                    active_path = current_path.copy()
                    active_goal = current_goal

                    path_index = min(10, len(active_path)-1)

                    started = True

                    print("STARTING AUTONOMOUS EXPLORATION")

                    robot_mode = "EXECUTING"

                else:

                    print("NO VALID PATH")



        elif robot_mode == "EXECUTING":

            if active_path:

                next_cell = active_path[
                    min(path_index, len(active_path)-1)
                ]

                target_x = (next_cell[1] + 0.5) * RESOLUTION + ORIGIN_X
                target_y = (next_cell[0] + 0.5) * RESOLUTION + ORIGIN_Y


                dx = target_x - robot_position[0]
                dy = target_y - robot_position[1]

                distance_to_waypoint = np.sqrt(dx**2 + dy**2)


                # advance waypoint
                if distance_to_waypoint < 0.15:

                    if path_index < len(active_path)-1:

                        path_index += 5

                        print("ADVANCING PATH INDEX:", path_index)

                    else:

                        goal_distance = np.sqrt(
                            (goal_x - robot_position[0])**2 +
                            (goal_y - robot_position[1])**2
                        )

                        if goal_distance < 0.30:

                            stop(client)

                            print("GOAL REACHED - REPLANNING")

                            active_path = None
                            active_goal = None
                            path_index = 10

                            robot_mode = "PLANNING"

                            continue


                target_yaw = np.arctan2(dy, dx)

                yaw_error = np.arctan2(
                    np.sin(target_yaw - robot_rpy[2]),
                    np.cos(target_yaw - robot_rpy[2])
                )


                if abs(yaw_error) > 0.3:

                    vx = 0.0
                    vy = 0.0
                    vyaw = np.clip(
                        yaw_error,
                        -TURN_SPEED,
                        TURN_SPEED
                    )

                else:

                    vx = FORWARD_SPEED
                    vy = 0.0
                    vyaw = np.clip(
                        yaw_error,
                        -TURN_SPEED,
                        TURN_SPEED
                    )


                move(client, vx, vy, vyaw)


                print(
                    f"EXEC | INDEX:{path_index} "
                    f"x:{robot_position[0]:.2f} "
                    f"y:{robot_position[1]:.2f} "
                    f"yaw:{robot_rpy[2]:.2f} "
                    f"vx:{vx:.2f} "
                    f"vy:{vy:.2f} "
                    f"vyaw:{vyaw:.2f}"
                )


        if USE_CV2 and current_goal is not None and goal_x is not None and goal_y is not None:

            display_path = active_path if robot_mode == "EXECUTING" else current_path
            display_goal = active_goal if robot_mode == "EXECUTING" else current_goal

            exploration.plot_cv2(display_goal, goal_x, goal_y, display_path, trajectory_points, robot_position, robot_rpy, vx, vy, vyaw, RESOLUTION, ORIGIN_X, ORIGIN_Y, frontier_cells, frontier_clusters, occupancy_grid, path_index)  
       
       #plot_grid(occupancy_grid, trajectory_points, ORIGIN_X, ORIGIN_Y)