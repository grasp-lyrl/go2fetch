import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque
import heapq
import cv2
from matplotlib.patches import Rectangle


#outputs list of coordinates [(x,y),(x,y)...] each (x,y) being a frontier cell
def detect_frontiers_cv2(grid_array, robot_grid_cell, resolution=0.05, radius_m=10.0):
    rows, cols = grid_array.shape
    r_robot, c_robot = robot_grid_cell

    radius_pixels = int(radius_m / resolution)
    r_min = max(0, r_robot - radius_pixels)
    r_max = min(rows, r_robot + radius_pixels + 1)
    c_min = max(0, c_robot - radius_pixels)
    c_max = min(cols, c_robot + radius_pixels + 1)

    roi = grid_array[r_min:r_max, c_min:c_max]
    explored_mask = np.where((roi <= -8) | (roi >= 8), 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(explored_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    frontier_cells = set()

    for contour in contours:
        for pt in contour:
            c_local, r_local = int(pt[0][0]), int(pt[0][1])
            r, c = r_local + r_min, c_local + c_min

            if grid_array[r, c] >= 8:
                continue

            if grid_array[r, c] <= -8:
                frontier_cells.add((r, c))

    return frontier_cells


#plt visual stuff
def visualize_grid(grid_array, origin_x, origin_y, frontier_cells, resolution=0.05):
    plt.clf() 
    
    display_img = np.zeros((grid_array.shape[0], grid_array.shape[1], 3), dtype=np.uint8)

    display_img[grid_array > 0] = [0, 0, 0]      
    display_img[grid_array < 0] = [255, 255, 255] 
    display_img[grid_array == 0] = [147, 147, 147]
    
    rows, cols = grid_array.shape
    x_min = origin_x
    x_max = origin_x + (cols * resolution)
    y_min = origin_y
    y_max = origin_y + (rows * resolution)
    
    plt.imshow(display_img, origin="lower", extent=[x_min, x_max, y_min, y_max])
 
    if frontier_cells:
        frontier_xs = [(cell[1] * resolution) + origin_x for cell in frontier_cells]
        frontier_ys = [(cell[0] * resolution) + origin_y for cell in frontier_cells]
        
        plt.scatter(frontier_xs, frontier_ys, c='cyan', s=2, label='Frontiers', zorder=2)
        
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("Exploration Pipeline: Frontier Detection")


def compute_reachability(grid_array, start, inflation_radius=0, resolution=0.05, radius_m=10.0):
    rows, cols = grid_array.shape
    r_start, c_start = start

    radius_pixels = int(radius_m / resolution)

    r_min = max(0, r_start - radius_pixels)
    r_max = min(rows, r_start + radius_pixels + 1)
    c_min = max(0, c_start - radius_pixels)
    c_max = min(cols, c_start + radius_pixels + 1)

    blocked_mask = (grid_array >= 8)

    if inflation_radius > 0:
        inflated_mask = blocked_mask.copy()

        wall_rows, wall_cols = np.where(grid_array >= 8)

        for r_global, c_global in zip(wall_rows, wall_cols):
            r_low = max(0, r_global - inflation_radius)
            r_high = min(rows, r_global + inflation_radius + 1)
            c_low = max(0, c_global - inflation_radius)
            c_high = min(cols, c_global + inflation_radius + 1)

            inflated_mask[r_low:r_high, c_low:c_high] = True
    else:
        inflated_mask = blocked_mask

    inflated_mask[start[0], start[1]] = False

    queue = [(0, start)]

    parent_map = {start: None}
    cost_map = {start: 0}
    reachable_set = {start}

    directions = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    ]

    while queue:
        current_cost, current = heapq.heappop(queue)

        if current not in cost_map:
            continue

        if current_cost > cost_map[current]:
            continue

        for dr, dc in directions:
            nr = current[0] + dr
            nc = current[1] + dc

            if not (r_min <= nr < r_max and c_min <= nc < c_max):
                continue

            if grid_array[nr, nc] > -8:
                continue

            if inflated_mask[nr, nc]:
                continue

            if dr != 0 and dc != 0:
                if grid_array[current[0] + dr, current[1]] > -8:
                    continue
                if grid_array[current[0], current[1] + dc] > -8:
                    continue

            neighbor = (nr, nc)
            obstacle_penalty = 0

            for rr in range(max(0, nr-7), min(rows, nr+6)):
                for cc in range(max(0, nc-7), min(cols, nc+6)):
                    if inflated_mask[rr, cc]:
                        obstacle_penalty += 3

            obstacle_penalty = obstacle_penalty ** 2


            if dr != 0 and dc != 0:
                new_cost = cost_map[current] + 1.414 + obstacle_penalty
            else:
                new_cost = cost_map[current] + 1 + obstacle_penalty

            if neighbor not in cost_map or new_cost < cost_map[neighbor]:

                cost_map[neighbor] = new_cost
                parent_map[neighbor] = current
                reachable_set.add(neighbor)

                heapq.heappush(
                    queue,
                    (new_cost, neighbor)
                )

    return reachable_set, parent_map, cost_map

#outputs list of dictionaries containing attributes of frontier clusters
def cluster_frontiers_cv2(grid_array, frontier_cells, reachable_set, cost_map, resolution=0.05, radius_m=10.0, min_cluster_size=5):

    if not frontier_cells:
        return []

    rs = [r for r, _ in frontier_cells]
    cs = [c for _, c in frontier_cells]
    r_min, r_max = min(rs), max(rs)
    c_min, c_max = min(cs), max(cs)

    frontier_mask = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=np.uint8)

    for r, c in frontier_cells:
        frontier_mask[r - r_min, c - c_min] = 255

    contours, _ = cv2.findContours(
        frontier_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    valid_clusters = []

    for contour in contours:

        cluster_cells = [
            (int(pt[0][1]) + r_min, int(pt[0][0]) + c_min)
            for pt in contour
        ]

        if len(cluster_cells) < min_cluster_size:
            continue

        reachable_cells = [
            cell for cell in cluster_cells
            if cell in reachable_set
        ]

        if not reachable_cells:
            continue

        M = cv2.moments(contour)

        if M["m00"] != 0:
            avg_col = int(M["m10"] / M["m00"]) + c_min
            avg_row = int(M["m01"] / M["m00"]) + r_min
        else:
            avg_row = int(np.mean([cell[0] for cell in cluster_cells]))
            avg_col = int(np.mean([cell[1] for cell in cluster_cells]))

        center_cell = min(
            reachable_cells,
            key=lambda c: (c[0] - avg_row)**2 + (c[1] - avg_col)**2
        )

        valid_clusters.append({
            'cells': cluster_cells,
            'size': len(cluster_cells),
            'center': center_cell,
            'distance': cost_map[center_cell],
            'cost': cost_map[center_cell],
            'reachable': True
        })

    return valid_clusters

#outputs list of coordinates [(start_x, start_y)...(end_x, end_y)] of BFS path to goal cluster
def plan_path(grid_array, goal, parent_map, reachable_set):

    if goal not in reachable_set:
        return None

    path = []

    current = goal

    while current is not None:
        path.append(current)
        current = parent_map[current]

    path.reverse()

    return path

def plot_mat(best_goal, goal_x, goal_y, path, path_xs, path_ys, trajectory_points, robot_position, RESOLUTION, ORIGIN_X, ORIGIN_Y, selected_scat, goal_scat, path_line, trajectory_line, robot_marker, occupancy_grid):
    goal_cells = best_goal['cells']
    cluster_xs = [((cell[1] + 0.5) * RESOLUTION) + ORIGIN_X for cell in goal_cells]
    cluster_ys = [((cell[0] + 0.5) * RESOLUTION) + ORIGIN_Y for cell in goal_cells]
    selected_scat.set_offsets(np.c_[cluster_xs, cluster_ys])
    goal_scat.set_offsets(np.c_[[goal_x], [goal_y]])
    
    if path:
        path_xs = [((wp[1] + 0.5) * RESOLUTION) + ORIGIN_X for wp in path]
        path_ys = [((wp[0] + 0.5) * RESOLUTION) + ORIGIN_Y for wp in path]
    path_line.set_data(path_xs, path_ys)

    if len(trajectory_points) > 0:
        traj_np = np.array(trajectory_points)
        trajectory_line.set_data(traj_np[:, 0], traj_np[:, 1])
    
    robot_marker.set_offsets(np.c_[[robot_position[0]], [robot_position[1]]])

    ax = robot_marker.axes
    box_width = 20.0
    box_height = 20.0
    rect_x = robot_position[0] - (box_width / 2.0)
    rect_y = robot_position[1] - (box_height / 2.0)

    if not hasattr(ax, 'bounds_rect'):
        ax.bounds_rect = Rectangle(
            (rect_x, rect_y), box_width, box_height, 
            linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--', zorder=10
        )
        ax.add_patch(ax.bounds_rect)
    else:
        ax.bounds_rect.set_xy((rect_x, rect_y))

    if len(trajectory_points) > 0:
        traj_np = np.array(trajectory_points)
        dynamic_min_x = np.min(traj_np[:, 0]) - 1.5
    else:
        dynamic_min_x = -1.5

    ax.set_xlim(dynamic_min_x, ORIGIN_X + (occupancy_grid.shape[1] * RESOLUTION))

def plot_cv2(best_goal, goal_x, goal_y, path, trajectory_points, robot_position, robot_rpy, vx, vy, vyaw, RESOLUTION, ORIGIN_X, ORIGIN_Y, frontier_cells, frontier_clusters, occupancy_grid, path_index, start_position, end_position=None, run=None, save=False):

    WINDOW_NAME = "Exploration Pipeline - OpenCV Live View"

    grid = occupancy_grid 
    raw_frontiers = frontier_cells
    clusters = frontier_clusters
    
    grid_color = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    grid_color[grid >= 8] = [0, 0, 0]  
    grid_color[grid <= -8] = [255, 255, 255]   
    grid_color[(grid > -8) & (grid < 8)] = [147, 147, 147] 

    display_img = grid_color.copy()

    def world_to_grid_pixel(x, y):
        col = int((x - ORIGIN_X) / RESOLUTION)
        row = int((y - ORIGIN_Y) / RESOLUTION)
        return col, row

    if len(trajectory_points) > 1:
        for i in range(len(trajectory_points) - 1):
            pt1 = trajectory_points[i]
            pt2 = trajectory_points[i+1]
            c1, r1 = world_to_grid_pixel(pt1[0], pt1[1])
            c2, r2 = world_to_grid_pixel(pt2[0], pt2[1])
            cv2.line(display_img, (c1, r1), (c2, r2), (180, 50, 50), 1)

    if raw_frontiers:
        for cell in raw_frontiers:
            r, c = cell
            display_img[r, c] = [255, 255, 0] 

    if clusters:
        for cl in clusters:
            cr, cc = cl['center']
            cv2.circle(display_img, (cc, cr), 2, (0, 140, 255), -1) 

    if best_goal is not None:
        for cell in best_goal['cells']:
            r, c = cell
            display_img[r, c] = [255, 0, 255] 


        if goal_x is not None and goal_y is not None:
            gc, gr = world_to_grid_pixel(goal_x, goal_y)
            
            cv2.line(display_img, (gc - 5, gr - 5), (gc + 5, gr + 5), (0, 0, 0), 4)
            cv2.line(display_img, (gc - 5, gr + 5), (gc + 5, gr - 5), (0, 0, 0), 4)
            
            cv2.line(display_img, (gc - 4, gr - 4), (gc + 4, gr + 4), (0, 255, 255), 2)
            cv2.line(display_img, (gc - 4, gr + 4), (gc + 4, gr - 4), (0, 255, 255), 2)

    if path and len(path) > 1:
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            cv2.line(display_img, (c1, r1), (c2, r2), (255, 0, 0), 2) 

    if path is not None and len(path) > 0:

        target_row, target_col = path[min(path_index, len(path)-1)]

        cv2.circle(
            display_img,
            (target_col, target_row),
            6,
            (255, 0, 255),
            -1
        )

    rc, rr = world_to_grid_pixel(robot_position[0], robot_position[1])
    cv2.rectangle(display_img, (rc - 3, rr - 3), (rc + 3, rr + 3), (200, 0, 0), -1)

    if start_position is not None:
        sc, sr = world_to_grid_pixel(start_position[0], start_position[1])

        cv2.circle(display_img, (sc, sr), 8, (0, 255, 0), -1)

    if end_position is not None:
        ec, er = world_to_grid_pixel(end_position[0], end_position[1])

        cv2.circle( display_img, (ec, er), 8, (0, 0, 255), -1)

    rc, rr = world_to_grid_pixel(robot_position[0],robot_position[1])

    yaw = robot_rpy[2]

    scale = 1.0 / RESOLUTION

    dx = goal_x - robot_position[0]
    dy = goal_y - robot_position[1]

    distance = np.sqrt(dx**2 + dy**2)

    if distance > 0.01:

        arrow_length = 1.0 / RESOLUTION

        end_x = int(rc + (dx / distance) * arrow_length)
        end_y = int(rr + (dy / distance) * arrow_length)

        cv2.arrowedLine(
            display_img,
            (rc, rr),
            (end_x, end_y),
            (0,255,0),
            2,
            tipLength=0.25
        )

    if abs(vyaw) > 0.01:
        radius = 12

        arc_angle = min(abs(vyaw) * 180, 120)

        if vyaw > 0:
            start_angle = 0
            end_angle = arc_angle
        else:
            start_angle = 0
            end_angle = -arc_angle

        cv2.ellipse(
            display_img,
            (rc, rr),
            (radius, radius),
            0,
            start_angle,
            end_angle,
            (0, 0, 255),
            2
        )

        angle = np.deg2rad(end_angle)

        arrow_x = int(rc + radius * np.cos(angle))
        arrow_y = int(rr + radius * np.sin(angle))

        cv2.arrowedLine(
            display_img,
            (arrow_x - 2, arrow_y - 2),
            (arrow_x, arrow_y),
            (0, 0, 255),
            2,
            tipLength=0.5
        )

    min_x, max_x = robot_position[0] - 10.0, robot_position[0] + 10.0
    min_y, max_y = robot_position[1] - 10.0, robot_position[1] + 10.0
    c_min, r_min = world_to_grid_pixel(min_x, min_y)
    c_max, r_max = world_to_grid_pixel(max_x, max_y)

    def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=4, gap_len=4):
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0: return
        ux, uy = dx / dist, dy / dist
        step = dash_len + gap_len
        for i in range(0, int(dist), step):
            start_dist = i
            end_dist = min(i + dash_len, dist)
            p1 = (int(pt1[0] + start_dist * ux), int(pt1[1] + start_dist * uy))
            p2 = (int(pt1[0] + end_dist * ux), int(pt1[1] + end_dist * uy))
            cv2.line(img, p1, p2, color, thickness)

    red_bgr = (0, 0, 255)
    draw_dashed_line(display_img, (c_min, r_min), (c_max, r_min), red_bgr, 1)
    draw_dashed_line(display_img, (c_max, r_min), (c_max, r_max), red_bgr, 1)
    draw_dashed_line(display_img, (c_max, r_max), (c_min, r_max), red_bgr, 1)
    draw_dashed_line(display_img, (c_min, r_max), (c_min, r_min), red_bgr, 1)

    flipped_display = cv2.flip(display_img, 0)
    cv2.imshow(WINDOW_NAME, flipped_display)

    if save:
        cv2.imwrite(f"{run}/map.png", flipped_display)

    cv2.waitKey(1)