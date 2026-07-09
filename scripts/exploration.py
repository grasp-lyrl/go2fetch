import numpy as np
import matplotlib.pyplot as plt

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