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

def plot_cv2(occ_canvas, robot_position, robot_yaw, trajectory_points, path,
             RESOLUTION, ORIGIN_X, ORIGIN_Y, target_xy=None,
             frontier_cells=None, goal_cluster=None,
             vx=0.0, vy=0.0, vyaw=0.0, run=None, save=False):
    rows = occ_canvas.shape[0]
    view = occ_canvas.copy()

    # BGR
    PATH = (229, 70, 79)          # indigo
    TRAJECTORY = (105, 150, 5)    # emerald
    ROBOT = (72, 29, 225)         # rose
    ROBOT_EDGE = (55, 19, 136)    # dark rose
    TARGET = (6, 119, 217)        # amber
    HEADING = (178, 145, 8)       # cyan
    YAW = (211, 38, 192)          # fuchsia
    FRONTIER = (53, 230, 163)     # lime
    FRONTIER_GOAL = (18, 98, 63)  # dark lime

    def world_to_grid(x, y):
        return int((x - ORIGIN_X) / RESOLUTION), int((y - ORIGIN_Y) / RESOLUTION)

    def to_view(c, r):
        return int(c), int(rows - 1 - r)

    def paint_cells(cells, color, width):
        mask = np.zeros(view.shape[:2], dtype=np.uint8)
        rr = np.fromiter((r for r, _ in cells), np.int32, len(cells))
        cc = np.fromiter((c for _, c in cells), np.int32, len(cells))
        mask[rows - 1 - rr, cc] = 255
        if width > 1:
            mask = cv2.dilate(mask, np.ones((width, width), np.uint8))
        view[mask > 0] = color

    if frontier_cells:
        paint_cells(frontier_cells, FRONTIER, 2)

    if goal_cluster is not None and goal_cluster.get("cells"):
        paint_cells(goal_cluster["cells"], FRONTIER_GOAL, 3)

    if path and len(path) > 1:
        pts = np.asarray([to_view(c, r) for r, c in path], dtype=np.int32)
        cv2.polylines(view, [pts], False, PATH, 2, cv2.LINE_AA)

    if trajectory_points is not None and len(trajectory_points) > 1:
        pts = np.asarray(
            [to_view(*world_to_grid(p[0], p[1])) for p in trajectory_points],
            dtype=np.int32,
        )
        cv2.polylines(view, [pts], False, TRAJECTORY, 2, cv2.LINE_AA)

    if target_xy is not None:
        tc, tr = world_to_grid(float(target_xy[0]), float(target_xy[1]))
        tx, ty = to_view(tc, tr)
        cv2.line(view, (tx - 10, ty - 10), (tx + 10, ty + 10), TARGET, 3, cv2.LINE_AA)
        cv2.line(view, (tx - 10, ty + 10), (tx + 10, ty - 10), TARGET, 3, cv2.LINE_AA)

    rc, rr = world_to_grid(robot_position[0], robot_position[1])
    rx, ry = to_view(rc, rr)
    cv2.circle(view, (rx, ry), 6, ROBOT, -1, cv2.LINE_AA)
    cv2.circle(view, (rx, ry), 6, ROBOT_EDGE, 1, cv2.LINE_AA)

    speed = float(np.hypot(vx, vy))
    if speed > 0.01:
        cy_yaw, sy_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
        wdx = (cy_yaw * vx - sy_yaw * vy) / speed
        wdy = (sy_yaw * vx + cy_yaw * vy) / speed
        arrow_len = 1.5 / RESOLUTION
        end = to_view(rc + wdx * arrow_len, rr + wdy * arrow_len)
        cv2.arrowedLine(view, (rx, ry), end, HEADING, 2, tipLength=0.25,
                        line_type=cv2.LINE_AA)

    if abs(vyaw) > 0.01:
        radius = 13
        sweep = float(np.clip(abs(vyaw) * 90.0, 60.0, 200.0))
        # cv2 angles grow clockwise on screen, world yaw grows counter-clockwise
        sign = -1.0 if vyaw > 0 else 1.0
        start_deg = -90.0 - sign * sweep * 0.5
        cv2.ellipse(view, (rx, ry), (radius, radius), 0,
                    start_deg, start_deg + sign * sweep, YAW, 2, cv2.LINE_AA)
        a_end = np.radians(start_deg + sign * sweep)
        a_prev = np.radians(start_deg + sign * (sweep - 12.0))
        cv2.arrowedLine(
            view,
            (int(rx + radius * np.cos(a_prev)), int(ry + radius * np.sin(a_prev))),
            (int(rx + radius * np.cos(a_end)), int(ry + radius * np.sin(a_end))),
            YAW, 2, tipLength=2.0, line_type=cv2.LINE_AA,
        )

    if save and run is not None:
        cv2.imwrite(f"{run}/map.png", view)

    return view
