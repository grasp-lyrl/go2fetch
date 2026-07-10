import numpy as np
import matplotlib.pyplot as plt
import numpy as np


#outputs list of coordinates [(x,y),(x,y)...] each (x,y) being a frontier cell
def detect_frontiers(grid_array):
    frontier_cells = set()
    rows, cols = grid_array.shape
    
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]
    
    for r in range(rows):
        for c in range(cols):
            if grid_array[r, c] < 0:
                
                for i in range(4):
                    nr, nc = r + dr[i], c + dc[i]
                    
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid_array[nr, nc] == 0:
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
def cluster_frontiers(grid_array, frontier_cells, robot_grid_cell, min_cluster_size=5):
    unvisited = set(frontier_cells)
    valid_clusters = []
    
    dr_8 = [-1, -1, -1, 0, 0, 1, 1, 1]
    dc_8 = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    dr_4 = [-1, 1, 0, 0]
    dc_4 = [0, 0, -1, 1]
    rows, cols = grid_array.shape
    
    while unvisited:
        start_cell = unvisited.pop()
        current_cluster = [start_cell]
        
        queue = [start_cell]
        while queue:
            curr_r, curr_c = queue.pop() 
            for i in range(8):
                nr, nc = curr_r + dr_8[i], curr_c + dc_8[i]
                neighbor = (nr, nc)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    current_cluster.append(neighbor)
                    queue.append(neighbor)
                    
        cluster_size = len(current_cluster)
        if cluster_size < min_cluster_size:
            continue 
            
        avg_row = int(np.mean([cell[0] for cell in current_cluster]))
        avg_col = int(np.mean([cell[1] for cell in current_cluster]))
        center_cell = (avg_row, avg_col)
        
        distance = np.sqrt((avg_row - robot_grid_cell[0])**2 + (avg_col - robot_grid_cell[1])**2)
        
        reachable = False
        if grid_array[center_cell[0], center_cell[1]] <= 0:  
            path_queue = [robot_grid_cell]
            path_visited = {robot_grid_cell}
            
            while path_queue:
                curr_r, curr_c = path_queue.pop() 
                
                if (curr_r, curr_c) == center_cell:
                    reachable = True
                    break
                    
                for i in range(4):
                    nr, nc = curr_r + dr_4[i], curr_c + dc_4[i]
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid_array[nr, nc] < 0 and (nr, nc) not in path_visited:
                            path_visited.add((nr, nc))
                            path_queue.append((nr, nc))
                            
        valid_clusters.append({
            'cells': current_cluster,
            'size': cluster_size,
            'center': center_cell,
            'distance': distance,
            'reachable': reachable
        })

    return valid_clusters


#outputs list of coordinates [(start_x, start_y)...(end_x, end_y)] of BFS path to goal cluster
def plan_path(grid_array, start, goal, inflation_radius=3):
    rows, cols = grid_array.shape
    
    blocked_mask = (grid_array >= 0) 
    inflated_mask = blocked_mask.copy()
    wall_rows, wall_cols = np.where(grid_array > 0)
    
    for r, c in zip(wall_rows, wall_cols):
        r_min = max(0, r - inflation_radius)
        r_max = min(rows, r + inflation_radius + 1)
        c_min = max(0, c - inflation_radius)
        c_max = min(cols, c + inflation_radius + 1)
        inflated_mask[r_min:r_max, c_min:c_max] = True

    inflated_mask[start[0], start[1]] = False
    inflated_mask[goal[0], goal[1]] = False

    queue = [start]
    parent_map = {start: None} 
    visited = set([start])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    found = False
    while queue:
        current = queue.pop(0)
        
        if current == goal:
            found = True
            break
            
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
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