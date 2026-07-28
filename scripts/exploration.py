import numpy as np
import heapq
import cv2


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


def compute_reachability(grid_array, start, resolution=0.05, radius_m=10.0, robot_radius_m=0.45):
    rows, cols = grid_array.shape
    r_start, c_start = start

    radius_pixels = int(radius_m / resolution)

    r_min = max(0, r_start - radius_pixels)
    r_max = min(rows, r_start + radius_pixels + 1)
    c_min = max(0, c_start - radius_pixels)
    c_max = min(cols, c_start + radius_pixels + 1)

    k = max(1, int(round(robot_radius_m / resolution)))
    blocked_mask = cv2.dilate(
        (grid_array >= 8).astype(np.uint8),
        np.ones((2 * k + 1, 2 * k + 1), np.uint8),
    ).astype(bool)
    # Always leave a small disk around the robot so planning can start.
    clear_r = max(2, int(round(0.40 / resolution)))
    r0 = max(0, r_start - clear_r)
    r1 = min(rows, r_start + clear_r + 1)
    c0 = max(0, c_start - clear_r)
    c1 = min(cols, c_start + clear_r + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    blocked_mask[r0:r1, c0:c1] &= ((yy - r_start) ** 2 + (xx - c_start) ** 2) > clear_r ** 2

    occ = blocked_mask.astype(np.int32, copy=False)
    integ = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    integ[1:, 1:] = occ.cumsum(axis=0).cumsum(axis=1)

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

        if current_cost > cost_map[current]:
            continue

        for dr, dc in directions:
            nr = current[0] + dr
            nc = current[1] + dc

            if not (r_min <= nr < r_max and c_min <= nc < c_max):
                continue

            near_robot = (nr - r_start) ** 2 + (nc - c_start) ** 2 <= clear_r ** 2
            if grid_array[nr, nc] > -8 and not near_robot:
                continue

            if blocked_mask[nr, nc]:
                continue

            if dr != 0 and dc != 0 and not near_robot:
                if grid_array[current[0] + dr, current[1]] > -8:
                    continue
                if grid_array[current[0], current[1] + dc] > -8:
                    continue

            neighbor = (nr, nc)
            r0 = max(0, nr - 8)
            r1 = min(rows, nr + 8)
            c0 = max(0, nc - 8)
            c1 = min(cols, nc + 8)
            count = integ[r1, c1] - integ[r0, c1] - integ[r1, c0] + integ[r0, c0]
            obstacle_penalty = (5 * int(count)) ** 2

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
            'cost': cost_map[center_cell],
        })

    return valid_clusters

#outputs list of coordinates [(start_x, start_y)...(end_x, end_y)] of BFS path to goal cluster
def plan_path(goal, parent_map, reachable_set):

    if goal not in reachable_set:
        return None

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent_map[current]
    path.reverse()
    return path


def nearest_reachable(goal, reachable_set):
    if not reachable_set:
        return None
    if goal in reachable_set:
        return goal
    gr, gc = goal
    return min(reachable_set, key=lambda c: (c[0] - gr) ** 2 + (c[1] - gc) ** 2)

def plot_cv2(best_goal, goal_x, goal_y, path, trajectory_points, robot_position, vyaw, RESOLUTION, ORIGIN_X, ORIGIN_Y, frontier_cells, frontier_clusters, occ_canvas, path_index, start_position, end_position=None, run=None, save=False, chair_path=None, stop_radius_m=None, chair_xy=None):
    rows, cols = occ_canvas.shape[:2]
    view = occ_canvas.copy()

    FREE = (255, 255, 255)
    UNKNOWN = (127, 127, 127)
    OCCUPIED = (0, 0, 0)
    FRONTIER = (207, 190, 23)
    CLUSTER = (14, 127, 255)
    GOAL_CLUSTER = (189, 103, 148)
    PATH = (255, 144, 30)
    TRAJECTORY = (180, 119, 31)
    CHAIR_PATH = (194, 119, 227)
    WAYPOINT = (0, 255, 255)
    ROBOT = (255, 128, 0)
    START = (44, 160, 44)
    GOAL = (40, 39, 214)
    CHAIR = (0, 255, 0)

    if frontier_cells:
        rr, cc = zip(*frontier_cells)
        view[rows - 1 - np.fromiter(rr, np.int32, len(rr)), np.fromiter(cc, np.int32, len(cc))] = FRONTIER

    if best_goal is not None and "cells" in best_goal:
        rr, cc = zip(*best_goal["cells"])
        view[rows - 1 - np.fromiter(rr, np.int32, len(rr)), np.fromiter(cc, np.int32, len(cc))] = GOAL_CLUSTER
    elif best_goal is not None and "center" in best_goal:
        r, c = best_goal["center"]
        view[rows - 1 - r, c] = GOAL_CLUSTER

    def world_to_grid(x, y):
        return int((x - ORIGIN_X) / RESOLUTION), int((y - ORIGIN_Y) / RESOLUTION)

    def to_view(c, r):
        return int(c), int(rows - 1 - r)

    if len(trajectory_points) > 1:
        pts = [
            to_view(*world_to_grid(p[0], p[1]))
            for p in trajectory_points
        ]
        cv2.polylines(view, [np.asarray(pts, dtype=np.int32)], False, TRAJECTORY, 2, cv2.LINE_AA)

    if frontier_clusters:
        for cl in frontier_clusters:
            cr, cc = cl["center"]
            cv2.circle(view, to_view(cc, cr), 4, CLUSTER, -1, cv2.LINE_AA)

    if goal_x is not None and goal_y is not None:
        gc, gr = world_to_grid(goal_x, goal_y)
        x, y = to_view(gc, gr)
        if stop_radius_m is not None and stop_radius_m > 0:
            radius_px = max(1, int(round(stop_radius_m / RESOLUTION)))
            cv2.circle(view, (x, y), radius_px, GOAL, 2, cv2.LINE_AA)
        cv2.line(view, (x - 8, y - 8), (x + 8, y + 8), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(view, (x - 8, y + 8), (x + 8, y - 8), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(view, (x - 7, y - 7), (x + 7, y + 7), GOAL, 2, cv2.LINE_AA)
        cv2.line(view, (x - 7, y + 7), (x + 7, y - 7), GOAL, 2, cv2.LINE_AA)

    if path and len(path) > 1:
        pts = np.asarray([to_view(c, r) for r, c in path], dtype=np.int32)
        cv2.polylines(view, [pts], False, PATH, 2, cv2.LINE_AA)

    if path is not None and len(path) > 0:
        tr, tc = path[min(path_index, len(path) - 1)]
        cv2.circle(view, to_view(tc, tr), 7, WAYPOINT, -1, cv2.LINE_AA)

    if chair_path is not None and len(chair_path) > 1:
        pts = np.asarray([to_view(c, r) for r, c in chair_path], dtype=np.int32)
        cv2.polylines(view, [pts], False, CHAIR_PATH, 2, cv2.LINE_AA)

    if chair_xy is not None:
        cc, cr = world_to_grid(float(chair_xy[0]), float(chair_xy[1]))
        cx, cy = to_view(cc, cr)
        cv2.line(view, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(view, (cx - 10, cy + 10), (cx + 10, cy - 10), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(view, (cx - 9, cy - 9), (cx + 9, cy + 9), CHAIR, 2, cv2.LINE_AA)
        cv2.line(view, (cx - 9, cy + 9), (cx + 9, cy - 9), CHAIR, 2, cv2.LINE_AA)

    rc, rr = world_to_grid(robot_position[0], robot_position[1])
    rx, ry = to_view(rc, rr)
    cv2.rectangle(view, (rx - 4, ry - 4), (rx + 4, ry + 4), ROBOT, -1)

    if start_position is not None:
        sc, sr = world_to_grid(start_position[0], start_position[1])
        cv2.circle(view, to_view(sc, sr), 3, START, -1, cv2.LINE_AA)

    if end_position is not None:
        ec, er = world_to_grid(end_position[0], end_position[1])
        cv2.circle(view, to_view(ec, er), 8, GOAL, -1, cv2.LINE_AA)

    dx = goal_x - robot_position[0] if goal_x is not None else 0.0
    dy = goal_y - robot_position[1] if goal_y is not None else 0.0
    distance = float(np.hypot(dx, dy))
    if goal_x is not None and goal_y is not None and distance > 0.01:
        arrow_len = 1.0 / RESOLUTION
        end = to_view(
            int(rc + (dx / distance) * arrow_len),
            int(rr + (dy / distance) * arrow_len),
        )
        cv2.arrowedLine(view, (rx, ry), end, GOAL, 2, tipLength=0.25, line_type=cv2.LINE_AA)

    if abs(vyaw) > 0.01:
        radius = 14
        arc_angle = min(abs(vyaw) * 180, 120)
        end_angle = arc_angle if vyaw > 0 else -arc_angle
        cv2.ellipse(view, (rx, ry), (radius, radius), 0, 0, end_angle, ROBOT, 2, cv2.LINE_AA)

    cv2.rectangle(view, (0, 0), (cols - 1, rows - 1), (80, 80, 80), 2)

    if save and run is not None:
        cv2.imwrite(f"{run}/map.png", view)

    legend = [
        ("patch", FREE, "Free"),
        ("patch", UNKNOWN, "Unknown"),
        ("patch", OCCUPIED, "Occupied"),
        ("patch", FRONTIER, "Frontier"),
        ("marker", CLUSTER, "Cluster"),
        ("patch", GOAL_CLUSTER, "Goal cluster"),
        ("line", PATH, "Path"),
        ("line", TRAJECTORY, "Trajectory"),
        ("line", CHAIR_PATH, "Chair path"),
        ("marker", WAYPOINT, "Waypoint"),
        ("marker", ROBOT, "Robot"),
        ("marker", START, "Start"),
        ("marker", GOAL, "Goal"),
        ("marker", CHAIR, "Target"),
        ("line", GOAL, "Reach radius"),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    row_h = 22
    handle_w = 30
    pad_x, pad_y = 12, 10
    gap = 10
    text_w = max(cv2.getTextSize(label, font, font_scale, thickness)[0][0] for _, _, label in legend)
    box_w = pad_x * 2 + handle_w + gap + text_w
    box_h = pad_y * 2 + row_h * len(legend)
    x0 = view.shape[1] - box_w - 16
    y0 = 16
    overlay = view.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.8, view, 0.2, 0, view)
    cv2.rectangle(view, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 1)
    for i, (kind, bgr, label) in enumerate(legend):
        cy = y0 + pad_y + i * row_h + row_h // 2
        hx0 = x0 + pad_x
        hx1 = hx0 + handle_w
        if kind == "patch":
            cv2.rectangle(view, (hx0, cy - 6), (hx1, cy + 6), bgr, -1)
            cv2.rectangle(view, (hx0, cy - 6), (hx1, cy + 6), (0, 0, 0), 1)
        elif kind == "line":
            cv2.line(view, (hx0, cy), (hx1, cy), bgr, 2, cv2.LINE_AA)
        else:
            cv2.circle(view, ((hx0 + hx1) // 2, cy), 5, bgr, -1, cv2.LINE_AA)
            cv2.circle(view, ((hx0 + hx1) // 2, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(view, label, (hx1 + gap, cy + 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return view
