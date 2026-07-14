import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


#outputs list of coordinates [(x,y),(x,y)...] each (x,y) being a frontier cell
def detect_frontiers(grid_array, robot_grid_cell, resolution=0.05, radius_m=10.0):
    frontier_cells = set()
    rows, cols = grid_array.shape
    r_robot, c_robot = robot_grid_cell
    
    radius_pixels = int(radius_m / resolution)
    r_min = max(0, r_robot - radius_pixels)
    r_max = min(rows, r_robot + radius_pixels + 1)
    c_min = max(0, c_robot - radius_pixels)
    c_max = min(cols, c_robot + radius_pixels + 1)
    
    dr = [-1, 1, 0, 0] 
    dc = [0, 0, -1, 1]
    
    for r in range(r_min, r_max):
        for c in range(c_min, c_max):
            if grid_array[r, c] <= -8:
                for i in range(4):
                    nr, nc = r + dr[i], c + dc[i]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if -8 < grid_array[nr, nc] < 8:
                            frontier_cells.add((r, c))
                            break 
                            
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


#outputs list of dictionaries containing attributes of frontier clusters
def cluster_frontiers(grid_array, frontier_cells, robot_grid_cell, resolution=0.05, radius_m=10.0, min_cluster_size=5):

    rows, cols = grid_array.shape
    r_robot, c_robot = robot_grid_cell
    
    radius_pixels = int(radius_m / resolution)
    r_min = max(0, r_robot - radius_pixels)
    r_max = min(rows, r_robot + radius_pixels + 1)
    c_min = max(0, c_robot - radius_pixels)
    c_max = min(cols, c_robot + radius_pixels + 1)

    reachable_set = {robot_grid_cell}
    queue = deque([robot_grid_cell]) 
    
    dr_path = [-1, 1, 0, 0, -1, -1, 1, 1]
    dc_path = [0, 0, -1, 1, -1, 1, -1, 1]
    
    while queue:
        curr_r, curr_c = queue.popleft()
        for i in range(8):
            nr, nc = curr_r + dr_path[i], curr_c + dc_path[i]
            if r_min <= nr < r_max and c_min <= nc < c_max:
                if (nr, nc) not in reachable_set and grid_array[nr, nc] <= -8: 
                    reachable_set.add((nr, nc))
                    queue.append((nr, nc))

    unvisited = set(frontier_cells)
    valid_clusters = []
    dr_8 = [-1, -1, -1, 0, 0, 1, 1, 1]
    dc_8 = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    while unvisited:
        start_cell = unvisited.pop()
        current_cluster = [start_cell]
        q = deque([start_cell])
        
        while q:
            curr_r, curr_c = q.popleft() 
            for i in range(8):
                neighbor = (curr_r + dr_8[i], curr_c + dc_8[i])
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    current_cluster.append(neighbor)
                    q.append(neighbor)
                    
        cluster_size = len(current_cluster)
        if cluster_size < min_cluster_size: 
            continue 
            
        avg_row = int(np.mean([cell[0] for cell in current_cluster]))
        avg_col = int(np.mean([cell[1] for cell in current_cluster]))
        center_cell = min(current_cluster, key=lambda c: (c[0] - avg_row)**2 + (c[1] - avg_col)**2)
        
        distance = np.sqrt((center_cell[0] - r_robot)**2 + (center_cell[1] - c_robot)**2)
        
        reachable = center_cell in reachable_set
        
        valid_clusters.append({
            'cells': current_cluster,
            'size': cluster_size,
            'center': center_cell,
            'distance': distance,
            'reachable': reachable
        })

    return valid_clusters

#outputs list of coordinates [(start_x, start_y)...(end_x, end_y)] of BFS path to goal cluster
def plan_path(grid_array, start, goal, inflation_radius=3, resolution=0.05, radius_m=10.0):

    for current_radius in range(inflation_radius, -1, -1):
        path = _plan_path_internal(grid_array, start, goal, current_radius, resolution, radius_m)
        if path is not None:
            return path
            
    return None 

def _plan_path_internal(grid_array, start, goal, inflation_radius, resolution, radius_m):
    rows, cols = grid_array.shape
    r_start, c_start = start
    
    radius_pixels = int(radius_m / resolution)
    r_min = max(0, r_start - radius_pixels)
    r_max = min(rows, r_start + radius_pixels + 1)
    c_min = max(0, c_start - radius_pixels)
    c_max = min(cols, c_start + radius_pixels + 1)
    
    blocked_mask = (grid_array >= 8)
    inflated_mask = blocked_mask.copy()
    
    local_slice = grid_array[r_min:r_max, c_min:c_max]
    wall_rows, wall_cols = np.where(local_slice >= 8)
    
    if inflation_radius > 0:
        for r_local, c_local in zip(wall_rows, wall_cols):
            r_global = r_local + r_min
            c_global = c_local + c_min
            
            r_low = max(r_min, r_global - inflation_radius)
            r_high = min(r_max, r_global + inflation_radius + 1)
            c_low = max(c_min, c_global - inflation_radius)
            c_high = min(c_max, c_global + inflation_radius + 1)
            inflated_mask[r_low:r_high, c_low:c_high] = True

    inflated_mask[start[0], start[1]] = False
    inflated_mask[goal[0], goal[1]] = False

    queue = [start]
    parent_map = {start: None} 
    visited = {start}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    found = False
    while queue:
        current = queue.pop(0)
        
        if current == goal:
            found = True
            break
            
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if r_min <= neighbor[0] < r_max and c_min <= neighbor[1] < c_max:
                if neighbor not in visited and not inflated_mask[neighbor[0], neighbor[1]]:
                    visited.add(neighbor)
                    parent_map[neighbor] = current
                    queue.append(neighbor)
                    
    if not found:
        return None 
        
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent_map[curr]
        
    path.reverse() 
    return path