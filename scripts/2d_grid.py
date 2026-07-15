import numpy as np
import matplotlib.pyplot as plt
import time

from examples.read_lidar_rrd import get_lidar_points
from examples.read_state_rrd import get_state_stream
from scripts import exploration

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
    REALTIME_REPLAY = True

    occupancy_grid = create_grid()

    lidar_stream = get_lidar_points("logs/levine.rrd")

    state_stream = get_state_stream("logs/levine.rrd")
    state_iter = iter(state_stream)

    t_prev, pos_prev, rpy_prev = next(state_iter)
    t_next, pos_next, rpy_next = next(state_iter)

    ORIGIN_X = pos_prev[0] - (MAP_WIDTH * RESOLUTION) / 2.0
    ORIGIN_Y = pos_prev[1] - (MAP_HEIGHT * RESOLUTION) / 2.0

    previous_lidar_time = None
    trajectory_points = []
    frame_count = 0

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
        
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Exploration Pipeline: High-Performance Persistent BFS")
        ax.legend(loc='upper right')
    else:
        fig = ax = im_artist = None
        frontiers_scat = all_centers_scat = selected_scat = goal_scat = None
        trajectory_line = path_line = robot_marker = None
        
    for t_lidar, lidar_points in lidar_stream:

        if REALTIME_REPLAY and previous_lidar_time is not None:
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

        world_points = filter_height(
            world_points,
            robot_position
        )

        update_grid(
            occupancy_grid,
            world_points[:, :2],
            robot_position[:2]
        )

        trajectory_points.append(robot_position[:2].copy())
        frame_count += 1

        if frame_count % 10 == 0:
            rx_grid, ry_grid = world_to_grid(robot_position[:2].reshape(1, 2))
            
            if len(rx_grid) > 0:
                robot_grid_cell = (ry_grid[0], rx_grid[0]) 

                reachable_set, parent_map, cost_map = exploration.compute_reachability(occupancy_grid,
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

                if not USE_CV2:
                    if frontier_clusters:
                        all_c_xs = [((cl['center'][1] + 0.5) * RESOLUTION) + ORIGIN_X for cl in frontier_clusters]
                        all_c_ys = [((cl['center'][0] + 0.5) * RESOLUTION) + ORIGIN_Y for cl in frontier_clusters]
                        all_centers_scat.set_offsets(np.c_[all_c_xs, all_c_ys])
                    else:
                        all_centers_scat.set_offsets(np.empty((0, 2)))

                    grid_color = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8)
                    grid_color[occupancy_grid >= 8] = [0, 0, 0]        
                    grid_color[occupancy_grid <= -8] = [255, 255, 255] 
                    grid_color[(occupancy_grid > -8) & (occupancy_grid < 8)] = [147, 147, 147] 
                    im_artist.set_data(grid_color)
                    
                    if frontier_cells:
                        f_xs = [((c[1] + 0.5) * RESOLUTION) + ORIGIN_X for c in frontier_cells]
                        f_ys = [((c[0] + 0.5) * RESOLUTION) + ORIGIN_Y for c in frontier_cells]
                        frontiers_scat.set_offsets(np.c_[f_xs, f_ys])
                    else:
                        frontiers_scat.set_offsets(np.empty((0, 2)))

                path_xs, path_ys = [], []
                
                reachable_clusters = [c for c in frontier_clusters if c['reachable']]
                best_goal = None
                
                if reachable_clusters:
                    if len(reachable_clusters) == 1:
                        best_goal = reachable_clusters[0]
                    else:
                        max_size = max(c['size'] for c in reachable_clusters)
                        max_dist = max(c['cost'] for c in reachable_clusters)
                        
                        highest_score = -1.0
                        w_size = 0.4
                        w_dist = 0.6
                        
                        for cluster in reachable_clusters:
                            norm_size = cluster['size'] / max_size if max_size > 0 else 0
                            norm_dist = (max_dist - cluster['cost']) / max_dist if max_dist > 0 else 0               
                            score = (w_size * norm_size) + (w_dist * norm_dist)
                            
                            if score > highest_score:
                                highest_score = score
                                best_goal = cluster

                    goal_row, goal_col = best_goal['center']
                    
                    if occupancy_grid[goal_row, goal_col] >= 8:
                        safe_cells = [cell for cell in best_goal['cells'] if occupancy_grid[cell[0], cell[1]] <= -8]
                        if safe_cells:
                            goal_row, goal_col = min(safe_cells, key=lambda c: (c[0]-goal_row)**2 + (c[1]-goal_col)**2)

                    goal_x = (goal_col + 0.5) * RESOLUTION + ORIGIN_X
                    goal_y = (goal_row + 0.5) * RESOLUTION + ORIGIN_Y
                    
                    path = exploration.plan_path(
                        occupancy_grid,
                        goal=(goal_row, goal_col),
                        parent_map=parent_map,
                        reachable_set=reachable_set
                    )

                    if best_goal is not None and not USE_CV2:
                        exploration.plot_mat(best_goal, goal_x, goal_y, path, path_xs, path_ys, trajectory_points, robot_position, RESOLUTION, ORIGIN_X, ORIGIN_Y, selected_scat, goal_scat, path_line, trajectory_line, robot_marker, occupancy_grid)
                    elif best_goal is not None and USE_CV2:
                        exploration.plot_cv2(best_goal, goal_x, goal_y, path, trajectory_points, robot_position, RESOLUTION, ORIGIN_X, ORIGIN_Y, frontier_cells, frontier_clusters, occupancy_grid)
                    else:
                        selected_scat.set_offsets(np.empty((0, 2)))
                        goal_scat.set_offsets(np.empty((0, 2)))
                        path_line.set_data([], [])
                
                if not USE_CV2:
                    robot_marker.set_offsets(np.c_[[robot_position[0]], [robot_position[1]]])
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

        trajectory_points.append(robot_position[:2].copy())
    
    #plot_grid(occupancy_grid, trajectory_points, ORIGIN_X, ORIGIN_Y)