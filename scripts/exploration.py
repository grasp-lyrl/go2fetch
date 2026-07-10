import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

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

def visualize_grid(grid_array, frontier_cells):

    plt.clf() 
    
    display_img = np.zeros((grid_array.shape[0], grid_array.shape[1], 3), dtype=np.uint8)

    display_img[grid_array > 0] = [0, 0, 0]      
    display_img[grid_array < 0] = [255, 255, 255] 
    display_img[grid_array == 0] = [147, 147, 147]
    
    plt.imshow(display_img, origin="lower")
 
    if frontier_cells:
        rows = [cell[0] for cell in frontier_cells]
        cols = [cell[1] for cell in frontier_cells]
        
        plt.scatter(cols, rows, c='cyan', s=2, label='Frontiers')
        
    plt.title("Exploration Pipeline: Frontier Detection")
    plt.pause(0.005) 

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