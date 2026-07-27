import argparse
import os
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np


def _ensure_cv2_qt_fonts():
    try:
        qt_fonts = os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts")
    except Exception:
        return
    marker = os.path.join(qt_fonts, "DejaVuSans.ttf")
    if os.path.exists(marker):
        return
    src_dir = "/usr/share/fonts/truetype/dejavu"
    if not os.path.isdir(src_dir):
        return
    os.makedirs(qt_fonts, exist_ok=True)
    for name in os.listdir(src_dir):
        if not name.endswith(".ttf"):
            continue
        dst = os.path.join(qt_fonts, name)
        if not os.path.exists(dst):
            os.symlink(os.path.join(src_dir, name), dst)


_ensure_cv2_qt_fonts()
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from scripts import exploration
from scripts.yolo_live import process_frame

from go2_interface.camera import make_camera_reader
from go2_interface.lidar import make_lidar_reader, pointcloud_to_xyz
from go2_interface.state import make_state_reader
from go2_interface.command import make_sport_client, move, stop, sit, free_avoid


RESOLUTION = 0.05
MAP_WIDTH = 1000
MAP_HEIGHT = 1000

ORIGIN_X = -25.0
ORIGIN_Y = -25.0

Z_MIN = 0.1
Z_MAX = 1.6
EXCLUSION_RADIUS = 0.45
MIN_RANGE = 0.2
MAX_RANGE = 12.0

FORWARD_SPEED = 0.4
TURN_SPEED = 1.5
ROBOT_RADIUS = 0.40  # ~0.8 m opening fits a doorway; larger blocks doors
CLEARANCE_CHECK_M = 1.0
BLOCKED_FRAMES = 4
GOAL_STOP_M = 0.8
GOAL_REPLAN_S = 2.0
GOAL_SLOW_M = 1.0
GOAL_UPDATE_CELLS = 20

_MAP_FREE = (255, 255, 255)
_MAP_UNKNOWN = (127, 127, 127)
_MAP_OCC = (0, 0, 0)

_LIDAR_PITCH = 2.8782
_cp, _sp = np.cos(_LIDAR_PITCH), np.sin(_LIDAR_PITCH)
LIDAR_TO_ROBOT_R = np.array([
    [_cp, 0.0, _sp],
    [0.0, 1.0, 0.0],
    [-_sp, 0.0, _cp],
])
LIDAR_TO_ROBOT_T = np.array([0.28945, 0.0, -0.046825])

OPTICAL_TO_ROBOT_R = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])

LIDAR_TO_CAMERA_T = np.array([
    [0.0,       -1.0,       0.0,       -0.000030],
    [0.2603577,  0.0,       0.9655122,  0.089795],
    [-0.9655122, 0.0,       0.2603577, -0.037700],
    [0.0,        0.0,       0.0,        1.0],
])

CAMERA_K = np.array([
    [864.39938,   0.0,      639.19798],
    [0.0,       863.73849,  373.28118],
    [0.0,         0.0,        1.0],
])

EXCLUSION_RADIUS_SQ = EXCLUSION_RADIUS ** 2
MIN_RANGE_SQ = MIN_RANGE ** 2
MAX_RANGE_SQ = MAX_RANGE ** 2
WAYPOINT_REACH_SQ = 0.25 ** 2
LOOKAHEAD_M = 2.0
EXPLORE_REACH_M = 0.7
MIN_EXPLORE_PATH = 40  # ~2 m — commit to a real trip


def create_grid():
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int16)
    canvas = np.full((MAP_HEIGHT, MAP_WIDTH, 3), _MAP_UNKNOWN, dtype=np.uint8)
    return grid, canvas


def lidar_to_robot(points):
    return (LIDAR_TO_ROBOT_R @ points[:, :3].T).T + LIDAR_TO_ROBOT_T


def optical_to_robot(point):
    return OPTICAL_TO_ROBOT_R @ point


def lidar_to_camera_optical(points):
    xyz = points[:, :3]
    return (xyz @ LIDAR_TO_CAMERA_T[:3, :3].T) + LIDAR_TO_CAMERA_T[:3, 3]


def project_camera_to_pixel(points):
    valid = points[:, 2] > 0
    pixels = (CAMERA_K @ points[valid].T).T
    pixels[:, 0] /= pixels[:, 2]
    pixels[:, 1] /= pixels[:, 2]
    return pixels[:, :2], valid


def get_points_in_bbox(points, pixels, valid, bbox):
    x1, y1, x2, y2 = bbox
    inside = (
        (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) &
        (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
    )
    return points[valid][inside]


def robot_to_world(points, position, rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_body = (
        np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    )

    xyz = np.asarray(points)
    if xyz.ndim == 1:
        return R_body @ xyz[:3] + position
    return (R_body @ xyz[:, :3].T).T + position


def filter_height(points, robot_position):
    height_mask = (points[:, 2] >= Z_MIN) & (points[:, 2] <= Z_MAX)
    relative = points[:, :3] - robot_position
    dist3_sq = np.einsum("ij,ij->i", relative, relative)
    dist2_sq = relative[:, 0] ** 2 + relative[:, 1] ** 2
    body_mask = dist3_sq > EXCLUSION_RADIUS_SQ
    range_mask = (dist2_sq >= MIN_RANGE_SQ) & (dist2_sq <= MAX_RANGE_SQ)
    return points[height_mask & body_mask & range_mask]


def world_to_grid(points):
    points = np.asarray(points)
    if points.ndim == 1:
        gx = int((points[0] - ORIGIN_X) / RESOLUTION)
        gy = int((points[1] - ORIGIN_Y) / RESOLUTION)
        return gy, gx

    gx = ((points[:, 0] - ORIGIN_X) / RESOLUTION).astype(int)
    gy = ((points[:, 1] - ORIGIN_Y) / RESOLUTION).astype(int)
    valid = (gx >= 0) & (gx < MAP_WIDTH) & (gy >= 0) & (gy < MAP_HEIGHT)
    return gx[valid], gy[valid]


def _paint_cell(grid, canvas, x, y):
    v = grid[y, x]
    canvas[grid.shape[0] - 1 - y, x] = (
        _MAP_OCC if v >= 8 else _MAP_FREE if v <= -8 else _MAP_UNKNOWN
    )


def apply_bresenham_ray(grid, canvas, x0, y0, x1, y1):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    x, y = x0, y0

    while True:
        if x == x1 and y == y1:
            grid[y, x] = min(grid[y, x] + 4, 100)
            _paint_cell(grid, canvas, x, y)
            break
        grid[y, x] = max(grid[y, x] - 1, -100)
        _paint_cell(grid, canvas, x, y)
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def update_grid(grid, canvas, xy, robot_position):
    gx, gy = world_to_grid(xy)
    rx, ry = world_to_grid(robot_position.reshape(1, 2))
    if len(rx) == 0:
        return
    robot_x, robot_y = rx[0], ry[0]
    for x, y in zip(gx, gy):
        apply_bresenham_ray(grid, canvas, robot_x, robot_y, x, y)


def bbox_center(bbox):
    return (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5


def robot_cell(position):
    gx, gy = world_to_grid(np.asarray(position[:2]).reshape(1, 2))
    if len(gx) == 0:
        return None
    return gy[0], gx[0]


def cell_to_world(cell):
    row, col = cell
    return (col + 0.5) * RESOLUTION + ORIGIN_X, (row + 0.5) * RESOLUTION + ORIGIN_Y


def pick_frontier(clusters, robot_xy, yaw):
    """Farthest reachable frontier ahead; if none ahead, farthest overall."""
    ahead = []
    for c in clusters:
        gx, gy = cell_to_world(c["center"])
        bearing = np.arctan2(gy - robot_xy[1], gx - robot_xy[0])
        err = abs(np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw)))
        if err < np.pi / 2:
            ahead.append(c)
    pool = ahead if ahead else clusters
    return max(pool, key=lambda c: c["cost"])


def path_blocked(grid, path, idx):
    """True if the path itself is occupied ahead (not door frames beside it)."""
    n_ahead = max(8, int(CLEARANCE_CHECK_M / RESOLUTION))
    for r, c in path[idx:min(len(path), idx + n_ahead)]:
        if grid[r, c] >= 8:
            return True
    return False


def obstacle_ahead(lidar_robot, stop_m=0.55, half_angle=0.85):
    """Stop if anything is in a forward wedge (covers turning into walls)."""
    if lidar_robot is None or len(lidar_robot) == 0:
        return False
    x, y, z = lidar_robot[:, 0], lidar_robot[:, 1], lidar_robot[:, 2]
    r = np.hypot(x, y)
    ang = np.arctan2(y, x)
    return bool(np.any(
        (r > 0.18) & (r < stop_m)
        & (np.abs(ang) < half_angle)
        & (z >= Z_MIN) & (z <= Z_MAX)
    ))


def nearest_path_index(path, robot_xy):
    best_i, best_d = 0, float("inf")
    for i, cell in enumerate(path):
        x, y = cell_to_world(cell)
        d = (x - robot_xy[0]) ** 2 + (y - robot_xy[1]) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def explore_target(path, robot_xy):
    """Pure-pursuit: aim a fixed distance *along the path* (stable, no zigzag)."""
    ex, ey = cell_to_world(path[-1])
    if (ex - robot_xy[0]) ** 2 + (ey - robot_xy[1]) ** 2 < EXPLORE_REACH_M ** 2:
        return None
    i = nearest_path_index(path, robot_xy)
    look = min(i + max(1, int(LOOKAHEAD_M / RESOLUTION)), len(path) - 1)
    tx, ty = cell_to_world(path[look])
    return look, tx, ty


def plan_frontiers(occupancy_grid, position, yaw):
    cell = robot_cell(position)
    if cell is None:
        return None

    # Prefer full clearance only — tight fallbacks are what make it graze walls.
    reachable_set, parent_map, cost_map = exploration.compute_reachability(
        occupancy_grid, cell, resolution=RESOLUTION, robot_radius_m=ROBOT_RADIUS
    )
    frontier_cells = exploration.detect_frontiers_cv2(
        occupancy_grid, cell, resolution=RESOLUTION
    )
    frontier_clusters = exploration.cluster_frontiers_cv2(
        occupancy_grid, frontier_cells, reachable_set, cost_map, resolution=RESOLUTION
    )
    if not frontier_clusters:
        return frontier_cells, [], None, None

    xy = np.asarray(position[:2], dtype=np.float64)
    goal = pick_frontier(frontier_clusters, xy, yaw)
    path = exploration.plan_path(goal["center"], parent_map, reachable_set)
    if path is not None and len(path) >= MIN_EXPLORE_PATH:
        return frontier_cells, frontier_clusters, goal, path
    return frontier_cells, frontier_clusters, None, None


def yaw_command(dx, dy, yaw, turn_thresh, speed=FORWARD_SPEED):
    target_yaw = np.arctan2(dy, dx)
    yaw_error = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    # Turn first on large errors — arcing at full speed is what hits walls
    # after dead-end reversals.
    if abs(yaw_error) > turn_thresh:
        vx = 0.0
    elif abs(yaw_error) > 0.4:
        vx = speed * 0.35
    else:
        vx = speed
    return vx, 0.0, float(np.clip(2.0 * yaw_error, -TURN_SPEED, TURN_SPEED))


DASHBOARD_WINDOW = "Go2Fetch"
CAM_H = 420
MAP_H = 720


def _fit_height(img, height, nearest=False):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((height, height, 3), dtype=np.uint8)
    scale = height / h
    if nearest:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(w * scale)), height), interpolation=interp)


def _window_open(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def show_dashboard(camera_bgr, map_bgr):
    if camera_bgr is None:
        camera_bgr = np.full((CAM_H, int(CAM_H * 16 / 9), 3), 40, dtype=np.uint8)
        cv2.putText(
            camera_bgr, "waiting for camera",
            (24, CAM_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2,
        )
    if map_bgr is None:
        map_bgr = np.full((MAP_H, int(MAP_H * 2), 3), 40, dtype=np.uint8)
        cv2.putText(
            map_bgr, "waiting for map",
            (24, MAP_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2,
        )

    cam = _fit_height(camera_bgr, CAM_H, nearest=False)
    mp = _fit_height(map_bgr, MAP_H, nearest=True)

    gap = 12
    canvas_h = max(cam.shape[0], mp.shape[0])
    canvas_w = cam.shape[1] + gap + mp.shape[1]
    canvas = np.full((canvas_h, canvas_w, 3), 28, dtype=np.uint8)

    canvas[:cam.shape[0], :cam.shape[1]] = cam
    canvas[:mp.shape[0], cam.shape[1] + gap:] = mp
    cv2.imshow(DASHBOARD_WINDOW, canvas)


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
    YOLO_EVERY = 5
    MAP_EVERY = 5

    occupancy_grid, occ_canvas = create_grid()

    get_lidar = make_lidar_reader(args.iface)
    get_state = make_state_reader(args.iface)
    client = make_sport_client(args.iface)
    camera = make_camera_reader(args.iface)

    while camera.read() is None:
        time.sleep(0.01)
    print("Camera: Connected")

    state_msg = None
    while state_msg is None:
        state_msg = get_state()
        time.sleep(0.01)
    print("State: Connected")

    while get_lidar() is None:
        time.sleep(0.01)
    print("Lidar: Connected")

    try:
        stop(client)
        free_avoid(client, True)
    except Exception:
        pass
    print("Press Space to enable/disable control.")

    initial_position = np.array(state_msg.position)
    start_position = initial_position[:2].copy()

    ORIGIN_X = initial_position[0] - (MAP_WIDTH * RESOLUTION) / 2.0
    ORIGIN_Y = initial_position[1] - (MAP_HEIGHT * RESOLUTION) / 2.0

    trajectory_points = []
    frame_count = 14
    loop_count = 0

    robot_mode = "PLANNING"
    initial_path_planned = False
    started = False

    current_goal = None
    current_path = None
    active_goal = None
    active_path = None
    chair_path = None
    object_goal = None
    locked_object = None
    lost_counter = 0
    goal_replan_at = 0.0
    chair_dist = None
    goal_smooth = None
    chair_xy = None
    blocked_streak = 0

    goal_x = None
    goal_y = None
    vx = vy = vyaw = 0.0
    path_index = 10

    display_goal = None
    display_path = None
    frontier_cells = []
    frontier_clusters = []

    latest_camera = None
    latest_map = None

    robot_position = initial_position.copy()
    robot_rpy = np.zeros(3)

    def finish_goal(reason="Goal: reached"):
        global started, active_path, locked_object, chair_dist, robot_mode
        try:
            code = sit(client)
            if code != 0:
                print(f"Sit returned code {code}, trying StandDown")
                client.StandDown()
        except Exception as e:
            print(f"sit failed: {e}")
        started = False
        active_path = locked_object = chair_dist = None
        robot_mode = "DONE"
        print(reason)
        print("Done: sitting (Ctrl+C to quit)")

    def approach_dist():
        if goal_x is None or goal_y is None:
            return None
        return float(np.hypot(goal_x - robot_position[0], goal_y - robot_position[1]))

    cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
    _init_w = int(CAM_H * 16 / 9) + 12 + int(MAP_H * (MAP_WIDTH / MAP_HEIGHT))
    cv2.resizeWindow(DASHBOARD_WINDOW, _init_w, MAP_H)
    show_dashboard(None, None)

    try:
        while True:
            loop_count += 1

            key = cv2.waitKey(1) & 0xFF
            if not _window_open(DASHBOARD_WINDOW):
                break
            if key == ord(" "):
                if robot_mode == "DONE":
                    continue
                if started:
                    stop(client)
                    started = False
                    active_path = active_goal = chair_path = None
                    locked_object = object_goal = None
                    lost_counter = 0
                    chair_dist = goal_smooth = chair_xy = None
                    robot_mode = "WAITING"
                    print("Control: disabled — stopped")
                else:
                    started = True
                    if current_path:
                        active_path = current_path.copy()
                        active_goal = current_goal
                        path_index = 0
                        robot_mode = "EXECUTING"
                        print("Control: enabled")
                    else:
                        robot_mode = "PLANNING"
                        print("Control: enabled — waiting for a path")
                continue

            lidar_msg = get_lidar()
            state_msg = get_state()
            if lidar_msg is None or state_msg is None:
                continue

            lidar_points = pointcloud_to_xyz(lidar_msg)
            robot_position = np.array(state_msg.position)
            robot_rpy = np.array(state_msg.imu_state.rpy)
            lidar_robot = lidar_to_robot(lidar_points)
            update_grid(
                occupancy_grid,
                occ_canvas,
                filter_height(
                    robot_to_world(lidar_robot, robot_position, robot_rpy),
                    robot_position,
                )[:, :2],
                robot_position[:2],
            )
            trajectory_points.append(robot_position[:2].copy())

            if robot_mode == "DONE":
                frame = camera.read()
                if frame is not None and loop_count % YOLO_EVERY == 0:
                    latest_camera, _ = process_frame(frame)
                if loop_count % MAP_EVERY == 0:
                    latest_map = exploration.plot_cv2(
                        None, goal_x, goal_y, None,
                        trajectory_points, robot_position, 0.0,
                        RESOLUTION, ORIGIN_X, ORIGIN_Y,
                        frontier_cells, frontier_clusters, occ_canvas,
                        path_index, start_position, chair_path=chair_path,
                        stop_radius_m=GOAL_STOP_M, chair_xy=chair_xy,
                    )
                show_dashboard(latest_camera, latest_map)
                continue

            if robot_mode == "PLANNING":
                frame_count += 1
                if frame_count % 10 == 0 or (
                    not initial_path_planned and frame_count % 5 == 0
                ):
                    planned = plan_frontiers(
                        occupancy_grid, robot_position, robot_rpy[2]
                    )
                    if planned is not None:
                        frontier_cells, frontier_clusters, goal, path = planned
                        if goal is not None and path and len(path) >= MIN_EXPLORE_PATH:
                            current_goal, current_path = goal, path
                            goal_x, goal_y = cell_to_world(goal["center"])
                            initial_path_planned = True
                            if started:
                                active_path = current_path.copy()
                                active_goal = current_goal
                                path_index = 0
                                print(
                                    f"Control: path ready ({len(path)} cells) — executing"
                                )
                                robot_mode = "EXECUTING"
                            else:
                                print("Press Space to enable control.")
                                robot_mode = "WAITING"
                        elif started and frame_count % 45 == 0:
                            print(
                                f"Explore: waiting for path "
                                f"(frontiers={len(frontier_cells)}, "
                                f"clusters={len(frontier_clusters)})"
                            )

            elif robot_mode == "EXECUTING":
                if not active_path:
                    robot_mode = "PLANNING"
                    continue
                path_index = nearest_path_index(active_path, robot_position[:2])
                close = path_blocked(occupancy_grid, active_path, path_index)
                front = obstacle_ahead(lidar_robot)
                if close or front:
                    blocked_streak += 1
                else:
                    blocked_streak = 0
                if blocked_streak >= BLOCKED_FRAMES:
                    stop(client)
                    active_path = active_goal = None
                    blocked_streak = 0
                    robot_mode = "PLANNING"
                    why = "front lidar" if front else "path clearance"
                    print(f"Explore: obstacle ({why}) — replanning")
                    continue
                target = explore_target(active_path, robot_position[:2])
                if target is None:
                    active_path = active_goal = current_path = None
                    blocked_streak = 0
                    robot_mode = "PLANNING"
                    print("Explore: path done — replanning")
                    continue
                path_index, target_x, target_y = target
                dx = target_x - robot_position[0]
                dy = target_y - robot_position[1]
                vx, vy, vyaw = yaw_command(dx, dy, robot_rpy[2], turn_thresh=0.75)
                move(client, vx, vy, vyaw)

            frame = camera.read()

            if frame is not None and loop_count % YOLO_EVERY == 0:
                annotated_frame, detections = process_frame(frame)
                latest_camera = annotated_frame
                chairs = [d for d in detections if d["class"] == TARGET_CLASS]

                if initial_path_planned and started and not chairs and locked_object is not None:
                    lost_counter += 1
                    if lost_counter > 20:
                        if robot_mode == "GOAL" and object_goal is not None:
                            locked_object = None
                        else:
                            print("Chair lost. Returning to exploration.")
                            stop(client)
                            active_path = current_path = chair_path = None
                            locked_object = object_goal = None
                            goal_x = goal_y = None
                            chair_dist = goal_smooth = chair_xy = None
                            robot_mode = "PLANNING"

                elif initial_path_planned and started and chairs:
                    lost_counter = 0
                    if locked_object is None:
                        locked_object = max(chairs, key=lambda d: d["confidence"])
                        print(f"{TARGET_CLASS.capitalize()} detected - Locking in!")
                    else:
                        old = bbox_center(locked_object["bbox"])

                        def track_score(chair):
                            cx, cy = bbox_center(chair["bbox"])
                            return 0.7 * chair["confidence"] - 0.3 * (
                                np.hypot(cx - old[0], cy - old[1]) / 500
                            )

                        locked_object = max(chairs, key=track_score)

                    if locked_object is not None:
                        x1, y1, x2, y2 = map(int, locked_object["bbox"])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            annotated_frame, "LOCKED CHAIR", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )
                        camera_points = lidar_to_camera_optical(lidar_points)
                        pixels, valid = project_camera_to_pixel(camera_points)
                        w, h = x2 - x1, y2 - y1
                        core = [
                            x1 + w * 0.35, y1 + h * 0.10,
                            x2 - w * 0.35, y1 + h * 0.40,
                        ]
                        cv2.rectangle(
                            annotated_frame,
                            (int(core[0]), int(core[1])),
                            (int(core[2]), int(core[3])),
                            (0, 255, 0), 2,
                        )
                        chair_pts = get_points_in_bbox(
                            camera_points, pixels, valid, core
                        )
                        if len(chair_pts) > 0:
                            cw = robot_to_world(
                                optical_to_robot(np.median(chair_pts, axis=0)),
                                robot_position, robot_rpy,
                            )
                            cx, cy = float(cw[0]), float(cw[1])
                            dxg, dyg = cx - robot_position[0], cy - robot_position[1]
                            dist = float(np.hypot(dxg, dyg))
                            if dist >= 0.5:
                                chair_dist = dist
                                chair_xy = (cx, cy)
                                if dist > GOAL_STOP_M:
                                    s = (dist - GOAL_STOP_M) / dist
                                    gx = robot_position[0] + dxg * s
                                    gy = robot_position[1] + dyg * s
                                else:
                                    gx, gy = cx, cy
                                if goal_smooth is None:
                                    goal_smooth = np.array([gx, gy], dtype=np.float64)
                                else:
                                    goal_smooth = 0.85 * goal_smooth + 0.15 * np.array(
                                        [gx, gy], dtype=np.float64
                                    )
                                goal_x, goal_y = float(goal_smooth[0]), float(goal_smooth[1])
                                ggx, ggy = world_to_grid(np.array([[goal_x, goal_y]]))
                                if len(ggx) > 0:
                                    new_goal = (int(ggy[0]), int(ggx[0]))
                                    if robot_mode != "GOAL":
                                        print("Goal: chasing chair")
                                        robot_mode = "GOAL"
                                        active_path = current_path = None
                                        object_goal = new_goal
                                        goal_replan_at = 0.0
                                    elif (
                                        object_goal is None
                                        or abs(new_goal[0] - object_goal[0])
                                        + abs(new_goal[1] - object_goal[1])
                                        > GOAL_UPDATE_CELLS
                                    ):
                                        object_goal = new_goal
                                        goal_replan_at = 0.0
                                else:
                                    print("Goal: chair outside map")

                latest_camera = annotated_frame

            if robot_mode == "GOAL" and started:
                ad = approach_dist()
                if (ad is not None and ad < GOAL_STOP_M) or (
                    chair_dist is not None and chair_dist < GOAL_STOP_M
                ):
                    finish_goal(
                        f"Goal: reached ({(ad if ad is not None else chair_dist):.2f}m)"
                    )
                    continue

                if obstacle_ahead(lidar_robot):
                    blocked_streak += 1
                else:
                    blocked_streak = 0
                if blocked_streak >= BLOCKED_FRAMES:
                    stop(client)
                    active_path = None
                    blocked_streak = 0
                    goal_replan_at = 0.0
                    print("Goal: obstacle ahead — stopping / replanning")
                    continue

                now = time.time()
                if now - goal_replan_at >= GOAL_REPLAN_S:
                    had_path = active_path is not None
                    old_index, old_len = path_index, (len(active_path) if had_path else 0)
                    goal_replan_at = now
                    cell = robot_cell(robot_position)
                    if cell is not None and object_goal is not None:
                        reachable_set, parent_map, _ = exploration.compute_reachability(
                            occupancy_grid, cell, resolution=RESOLUTION,
                            robot_radius_m=ROBOT_RADIUS,
                        )
                        plan_goal = exploration.nearest_reachable(
                            object_goal, reachable_set
                        )
                        new_path = (
                            exploration.plan_path(plan_goal, parent_map, reachable_set)
                            if plan_goal is not None else None
                        )
                        if new_path and len(new_path) > 1:
                            end = cell_to_world(new_path[-1])
                            d_end = float(np.hypot(
                                end[0] - robot_position[0], end[1] - robot_position[1]
                            ))
                            if d_end >= 0.3:
                                active_path = chair_path = new_path.copy()
                                if had_path and old_len > 1:
                                    path_index = int(np.clip(
                                        (old_index / (old_len - 1)) * (len(active_path) - 1),
                                        0, len(active_path) - 1,
                                    ))
                                else:
                                    path_index = min(5, len(active_path) - 1)
                                if not had_path:
                                    print(
                                        f"Goal: path ready ({len(active_path)} cells, "
                                        f"{d_end:.1f}m)"
                                    )
                            else:
                                active_path = None
                        else:
                            active_path = None

                if active_path:
                    if path_blocked(occupancy_grid, active_path, path_index):
                        active_path = None
                        goal_replan_at = now
                        print("Goal: path blocked — replanning")
                        continue
                    target_x, target_y = cell_to_world(
                        active_path[min(path_index, len(active_path) - 1)]
                    )
                    dx = target_x - robot_position[0]
                    dy = target_y - robot_position[1]
                    if dx * dx + dy * dy < WAYPOINT_REACH_SQ:
                        if path_index < len(active_path) - 1:
                            path_index += 5
                        else:
                            active_path = None
                            goal_replan_at = now
                            if (ad is not None and ad < GOAL_STOP_M * 1.25) or (
                                chair_dist is not None and chair_dist < GOAL_STOP_M
                            ):
                                finish_goal(
                                    f"Goal: reached "
                                    f"({(ad if ad is not None else chair_dist):.2f}m)"
                                )
                            continue
                    ref = ad if ad is not None else (
                        chair_dist if chair_dist is not None else GOAL_SLOW_M
                    )
                    speed = FORWARD_SPEED * min(1.0, max(ref, 0.4) / GOAL_SLOW_M)
                    vx, vy, vyaw = yaw_command(
                        dx, dy, robot_rpy[2], turn_thresh=0.7, speed=speed
                    )
                    move(client, vx, vy, vyaw)

            if loop_count % MAP_EVERY == 0:
                if robot_mode == "GOAL":
                    display_path, display_goal = active_path, None
                elif robot_mode == "EXECUTING":
                    display_path, display_goal = active_path, active_goal
                else:
                    display_path, display_goal = current_path, current_goal
                latest_map = exploration.plot_cv2(
                    display_goal, goal_x, goal_y, display_path,
                    trajectory_points, robot_position, vyaw,
                    RESOLUTION, ORIGIN_X, ORIGIN_Y,
                    frontier_cells, frontier_clusters, occ_canvas,
                    path_index, start_position, chair_path=chair_path,
                    stop_radius_m=GOAL_STOP_M, chair_xy=chair_xy,
                )

            show_dashboard(latest_camera, latest_map)

    except KeyboardInterrupt:
        pass

    finally:
        print("Quitting app!")
        try:
            stop(client)
        except Exception as e:
            print(f"robot stop failed: {e}")

        try:
            camera.stop()
        except Exception as e:
            print(f"camera stop failed: {e}")

        end_position = robot_position[:2].copy()
        if goal_x is not None and goal_y is not None:
            try:
                exploration.plot_cv2(
                    display_goal, goal_x, goal_y, display_path,
                    trajectory_points, robot_position, vyaw,
                    RESOLUTION, ORIGIN_X, ORIGIN_Y,
                    frontier_cells, frontier_clusters, occ_canvas,
                    path_index, start_position, end_position,
                    run=SAVE_DIR, save=True,
                    stop_radius_m=GOAL_STOP_M, chair_xy=chair_xy,
                )
            except Exception as e:
                print(f"map save failed: {e}")

        cv2.destroyAllWindows()
