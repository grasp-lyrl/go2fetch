import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os

from examples.read_lidar_rrd import get_lidar_points
from examples.read_state_rrd import get_state_stream

from scripts import exploration
from scripts.yolo_live import process_frame

from go2_interface.camera import make_camera_reader
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

FORWARD_SPEED = 0.5
TURN_SPEED = 1.5

def create_grid():
    return np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int16)

def lidar_to_robot(points):
    pitch = 2.8782
    cp = np.cos(pitch)
    sp = np.sin(pitch)

    R = np.array([
        [ cp, 0.0,  sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0,  cp]
    ])

    xyz = (R @ points[:, :3].T).T
    xyz += np.array([0.28945, 0.0, -0.046825])
    return xyz

def camera_to_optical(points):

    R = np.array([
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
        [1.0,  0.0,  0.0]
    ])

    return (R @ points[:, :3].T).T

def optical_to_robot(point):

    R = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ])

    return R @ point

def lidar_to_camera_optical(points):

    T = np.array([
        [0.0,       -1.0,       0.0,       -0.000030],
        [0.2603577,  0.0,       0.9655122,  0.089795],
        [-0.9655122, 0.0,       0.2603577, -0.037700],
        [0.0,        0.0,       0.0,        1.0]
    ])

    xyz = points[:, :3]

    xyz_h = np.hstack([
        xyz,
        np.ones((len(xyz), 1))
    ])

    camera_points = (T @ xyz_h.T).T

    return camera_points[:, :3]

def project_camera_to_pixel(points):

    K = np.array([
        [864.39938,   0.0,      639.19798],
        [0.0,       863.73849,  373.28118],
        [0.0,         0.0,        1.0]
    ])

    valid = points[:, 2] > 0

    xyz = points[valid]

    pixels = (K @ xyz.T).T

    pixels[:, 0] /= pixels[:, 2]
    pixels[:, 1] /= pixels[:, 2]

    return pixels[:, :2], valid

def get_points_in_bbox(points, pixels, valid, bbox):

    x1, y1, x2, y2 = bbox

    inside = (
        (pixels[:, 0] >= x1) &
        (pixels[:, 0] <= x2) &
        (pixels[:, 1] >= y1) &
        (pixels[:, 1] <= y2)
    )

    valid_points = points[valid]

    chair_points = valid_points[inside]

    return chair_points


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

    xyz = np.asarray(points)

    if xyz.ndim == 1:
        xyz_world = R_body @ xyz[:3]
        xyz_world += position
    else:
        xyz = xyz[:, :3]
        xyz_world = (R_body @ xyz.T).T
        xyz_world += position

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

    points = np.asarray(points)

    if points.ndim == 1:
        gx = int((points[0] - ORIGIN_X) / RESOLUTION)
        gy = int((points[1] - ORIGIN_Y) / RESOLUTION)

        return gy, gx

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

    plt.imshow(img, cmap="gray", vmin=0, vmax=255, origin="lower", extent=[x_min, x_max, y_min, y_max])

    trajectory_np = np.array(trajectory_points)

    plt.plot(trajectory_np[:, 0], trajectory_np[:, 1], color="blue", linewidth=1.5, label="trajectory")

    plt.scatter(trajectory_np[0, 0], trajectory_np[0, 1], color="green", s=50, zorder=5, label="start")

    plt.scatter(trajectory_np[-1, 0], trajectory_np[-1, 1], color="red", s=50, zorder=5, label="end")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("grid with trajectory (m)")
    plt.legend()
    plt.show(block=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "iface",
        nargs="?",
        default="en7",
        help="network interface connected to Go2 (default: en7)",
    )
    args = parser.parse_args()

    RUN_NAME = "run002"
    SAVE_DIR = f"data/{RUN_NAME}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TARGET_CLASS = "chair"
    locked_object = None
    lost_counter = 0

    USE_CV2 = True
    REALTIME_REPLAY = False
    YOLO_EVERY = 5
    MAP_EVERY = 5

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

    get_lidar = make_lidar_reader(args.iface)
    get_state = make_state_reader(args.iface)

    client = make_sport_client(args.iface)

    camera = make_camera_reader(args.iface)

    while camera.read() is None:
        time.sleep(0.01)
    print("camera connected")

    print("waiting for state...")
    state_msg = None
    while state_msg is None:
        state_msg = get_state()
        time.sleep(0.01)
    print("state connected")

    print("waiting for lidar...")
    lidar_wait_start = time.time()
    stood_up = False
    while get_lidar() is None:
        if not stood_up and time.time() - lidar_wait_start > 3.0:
            print("no lidar yet; sending StandUp to wake unitree_lidar")
            try:
                client.StandUp()
            except Exception as e:
                print(f"StandUp failed: {e}")
            stood_up = True
        time.sleep(0.05)
    print("lidar connected")

    initial_position = np.array(state_msg.position)
    start_position = initial_position[:2].copy()

    # ORIGIN_X = pos_prev[0] - (MAP_WIDTH * RESOLUTION) / 2.0
    # ORIGIN_Y = pos_prev[1] - (MAP_HEIGHT * RESOLUTION) / 2.0

    ORIGIN_X = initial_position[0] - (MAP_WIDTH * RESOLUTION) / 2.0
    ORIGIN_Y = initial_position[1] - (MAP_HEIGHT * RESOLUTION) / 2.0

    previous_lidar_time = None
    trajectory_points = []
    frame_count = 0
    loop_count = 0


    if not USE_CV2:

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.set_xlim(ORIGIN_X, ORIGIN_X + (MAP_WIDTH * RESOLUTION))

        ax.set_ylim(ORIGIN_Y, ORIGIN_Y + (MAP_HEIGHT * RESOLUTION))
        
        display_img = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)

        im_artist = ax.imshow(display_img, origin="lower", extent=[ORIGIN_X, ORIGIN_X + (MAP_WIDTH * RESOLUTION), ORIGIN_Y, ORIGIN_Y + (MAP_HEIGHT * RESOLUTION)])
        
        frontiers_scat = ax.scatter([], [], c='cyan', s=2, label='Frontiers', zorder=2)

        all_centers_scat = ax.scatter([], [], c='orange', marker='o', s=10, edgecolors='black', label='All Cluster Centers', zorder=3)

        selected_scat = ax.scatter([], [], c='magenta', s=6, label='Selected Goal Cluster', zorder=4)

        goal_scat = ax.scatter([], [], c='yellow', marker='X', s=25, edgecolors='black', label='Goal Centroid', zorder=6)
        
        trajectory_line, = ax.plot([], [], c='blue', linewidth=1.0, alpha=0.7, zorder=3, label='Traveled Path')

        path_line, = ax.plot([], [], c='blue', linewidth=2, zorder=5, label='Planned Path')

        robot_marker = ax.scatter([], [], c='blue', marker='s', s=20, label='Robot', zorder=11)
        
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Exploration Pipeline: BFS Planning")
        ax.legend(loc='upper right')

    else:
        fig = ax = im_artist = None
        frontiers_scat = all_centers_scat = selected_scat = goal_scat = None
        trajectory_line = path_line = robot_marker = None
    

    robot_mode = "PLANNING"
    replan_counter = 0

    current_goal = None
    current_path = None

    active_goal = None
    active_path = None

    object_goal = None

    goal_x = None
    goal_y = None

    vx = 0.0
    vy = 0.0
    vyaw = 0.0

    path_index = 10
    started = False

    display_goal = None
    display_path = None
    frontier_cells = []
    frontier_clusters = []

    robot_position = initial_position.copy()
    robot_rpy = np.zeros(3)

    try:
        while True:
            loop_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):

                stop(client)
                print("s was pressed, stopped, planning again")

                active_path = None
                active_goal = None

                robot_mode = "WAITING"
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

            print("before planning")
            if robot_mode == "PLANNING":
                print("in planning")
                frame_count += 1

                if frame_count % 15 == 0:
                    
                    rx_grid, ry_grid = world_to_grid(robot_position[:2].reshape(1, 2))

                    if len(rx_grid) > 0:
                        robot_grid_cell = (ry_grid[0], rx_grid[0])

                        reachable_set, parent_map, cost_map = exploration.compute_reachability(occupancy_grid, robot_grid_cell, resolution=RESOLUTION)

                        frontier_cells = exploration.detect_frontiers_cv2(occupancy_grid, robot_grid_cell, resolution=RESOLUTION)
                        frontier_clusters = exploration.cluster_frontiers_cv2(occupancy_grid, frontier_cells, reachable_set, cost_map, resolution=RESOLUTION)

                        reachable_clusters = [c for c in frontier_clusters if c["reachable"]]

                        if reachable_clusters:
                            current_goal = max(reachable_clusters, key=lambda c: (0.4 * c["size"] - 0.6 * c["cost"]))

                            goal_row, goal_col = current_goal["center"]

                            goal_x = (goal_col + 0.5) * RESOLUTION + ORIGIN_X
                            goal_y = (goal_row + 0.5) * RESOLUTION + ORIGIN_Y

                            current_path = exploration.plan_path(occupancy_grid, goal=(goal_row, goal_col), parent_map=parent_map, reachable_set=reachable_set)

                            if current_path:
                                if started:
                                    active_path = current_path.copy()
                                    active_goal = current_goal
                                    path_index = min(10, len(active_path)-1)
                                    print("new path found, starting execution")

                                    robot_mode = "EXECUTING"

                                else:
                                    print("press 'w' to start")
                                    robot_mode = "WAITING"

            elif robot_mode == "WAITING":

                if key == ord('w'):
                    if current_path and goal_x is not None and goal_y is not None:
                        active_path = current_path.copy()
                        active_goal = current_goal
                        path_index = min(10, len(active_path)-1)
                        started = True
                        print("staring exploration")

                        robot_mode = "EXECUTING"

                    else:
                        print("no valid path")


            elif robot_mode == "EXECUTING":
                
                replan_counter += 1

                if replan_counter % 20 == 0:
                    rx_grid, ry_grid = world_to_grid(robot_position[:2].reshape(1, 2))

                    if len(rx_grid) > 0:
                        robot_grid_cell = (ry_grid[0], rx_grid[0])

                        reachable_set, parent_map, cost_map = exploration.compute_reachability(occupancy_grid, robot_grid_cell, resolution=RESOLUTION)

                        frontier_cells = exploration.detect_frontiers_cv2(occupancy_grid, robot_grid_cell, resolution=RESOLUTION)

                        frontier_clusters = exploration.cluster_frontiers_cv2(occupancy_grid, frontier_cells, reachable_set, cost_map, resolution=RESOLUTION)

                        reachable_clusters = [c for c in frontier_clusters if c["reachable"]]

                        if reachable_clusters:
                            current_goal_candidate = max(reachable_clusters, key=lambda c: (0.4*c["size"] - 0.6*c["cost"]))

                            goal_row, goal_col = current_goal_candidate["center"]

                            new_path = exploration.plan_path(occupancy_grid, goal=(goal_row, goal_col), parent_map=parent_map, reachable_set=reachable_set)

                            if new_path:
                                current_path = new_path
                                current_goal = current_goal_candidate

                if active_path:

                    next_cell = active_path[min(path_index, len(active_path)-1)]

                    target_x = (next_cell[1] + 0.5) * RESOLUTION + ORIGIN_X
                    target_y = (next_cell[0] + 0.5) * RESOLUTION + ORIGIN_Y

                    dx = target_x - robot_position[0]
                    dy = target_y - robot_position[1]

                    distance_to_waypoint = np.sqrt(dx**2 + dy**2)


                    if distance_to_waypoint < 0.25:
                        remaining_cells = len(active_path) - path_index

                        if remaining_cells > 30:
                            path_index += 5
                            print("path index:", path_index)

                        else:
                            if current_path is not None:
                                active_path = current_path.copy()
                                active_goal = current_goal
                                path_index = 10
                                current_path = None

                                print("switching to new path")

                            else:
                                stop(client)
                                active_path = None
                                active_goal = None
                                robot_mode = "PLANNING"
                                print("no next path ready, replanning")

                    target_yaw = np.arctan2(dy, dx)

                    yaw_error = np.arctan2(np.sin(target_yaw - robot_rpy[2]), np.cos(target_yaw - robot_rpy[2])
                    )

                    if abs(yaw_error) > 0.6:
                        vx = 0.0
                        vy = 0.0
                        vyaw = np.clip(yaw_error, -TURN_SPEED, TURN_SPEED)

                    else:
                        vx = FORWARD_SPEED
                        vy = 0.0
                        vyaw = np.clip(yaw_error, -TURN_SPEED, TURN_SPEED)

                    move(client, vx, vy, vyaw)

                    print(
                        f"executing | "
                        f"path index:{path_index} "
                        f"x:{robot_position[0]:.2f} "
                        f"y:{robot_position[1]:.2f} "
                        f"yaw:{robot_rpy[2]:.2f} "
                        f"vx:{vx:.2f} "
                        f"vy:{vy:.2f} "
                        f"vyaw:{vyaw:.2f}"
                    )
            if loop_count % YOLO_EVERY == 0:
                frame = camera.read()
                if frame is not None:
                    annotated_frame, detections = process_frame(frame)
                    chairs = [d for d in detections if d["class"] == TARGET_CLASS]

                    if locked_object is None:
                        if chairs:
                            locked_object = max(chairs, key=lambda d: d["confidence"])

                            print("Locked onto chair")

                    else:
                        if chairs:
                            old_bbox = locked_object["bbox"]

                            old_center = ((old_bbox[0] + old_bbox[2]) / 2, (old_bbox[1] + old_bbox[3]) / 2)

                            best_score = -float("inf")
                            best_chair = None

                            for chair in chairs:

                                bbox = chair["bbox"]

                                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                                distance = np.sqrt((center[0] - old_center[0])**2 + (center[1] - old_center[1])**2)

                                score = (0.7 * chair["confidence"] - 0.3 * (distance / 500))

                                if score > best_score:
                                    best_score = score
                                    best_chair = chair

                            if best_chair is not None:
                                locked_object = best_chair

                    if locked_object is not None:

                        x1, y1, x2, y2 = map(int, locked_object["bbox"])

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,0,255), 3)

                        cv2.putText(annotated_frame, "LOCKED CHAIR", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                        camera_points = lidar_to_camera_optical(lidar_points)

                        pixels, valid = project_camera_to_pixel(camera_points)

                        chair_points = get_points_in_bbox(camera_points, pixels, valid, locked_object["bbox"])

                        if len(chair_points) > 0:
                            chair_position_camera = np.median(chair_points, axis=0)

                            chair_position_robot = optical_to_robot(chair_position_camera)

                            chair_position_world = robot_to_world(
                                chair_position_robot,
                                robot_position,
                                robot_rpy
                            )

                            new_goal_x = chair_position_world[0]
                            new_goal_y = chair_position_world[1]

                            if goal_x is None:
                                goal_x = new_goal_x
                                goal_y = new_goal_y
                            else:
                                goal_x = 0.9 * goal_x + 0.1 * new_goal_x
                                goal_y = 0.9 * goal_y + 0.1 * new_goal_y

                            goal_grid_x, goal_grid_y = world_to_grid(
                                np.array([[goal_x, goal_y]])
                            )

                            if len(goal_grid_x) > 0:
                                object_goal = (goal_grid_y[0], goal_grid_x[0])
                                print("chair world:", goal_x, goal_y)
                                print("chair grid:", object_goal)
                            else:
                                print("chair is outside map")

                    cv2.imshow("YOLO Camera", annotated_frame)

            if USE_CV2 and current_goal is not None and goal_x is not None and goal_y is not None:

                display_path = active_path if robot_mode == "EXECUTING" else current_path
                display_goal = active_goal if robot_mode == "EXECUTING" else current_goal

                if loop_count % MAP_EVERY == 0:
                    exploration.plot_cv2(display_goal, goal_x, goal_y, display_path, trajectory_points, robot_position, robot_rpy, vx, vy, vyaw, RESOLUTION, ORIGIN_X, ORIGIN_Y, frontier_cells, frontier_clusters, occupancy_grid, path_index, start_position)
        #plot_grid(occupancy_grid, trajectory_points, ORIGIN_X, ORIGIN_Y)

    except KeyboardInterrupt:
        print("interruption")

    finally:
        end_position = robot_position[:2].copy()
        if goal_x is not None and goal_y is not None:
            exploration.plot_cv2(display_goal, goal_x, goal_y, display_path, trajectory_points, robot_position, robot_rpy, vx, vy, vyaw, RESOLUTION, ORIGIN_X, ORIGIN_Y, frontier_cells, frontier_clusters, occupancy_grid, path_index, start_position, end_position, run=SAVE_DIR, save=True)
