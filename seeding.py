import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def size_gradient(d1, d2, x, y, res_x, res_y):
    return (d2 - d1) / res_y * y + d1


if __name__ == "__main__":
    #define region
    max_x = 10
    max_y = 10
    r = 0.5
    d2 = 5
    d1 = 1

    #Rasterized
    dx = r / np.sqrt(2)
    res_x, res_y = int(max_x / dx), int(max_y / dx)
    init_len = res_x * res_y
    grid_ids = np.arange(init_len)
    grid = grid_ids.reshape(res_x, res_y)

    #precompute sizes
    box_x_indices = np.array(grid / res_x, dtype=int)
    box_y_indices = np.array(grid - box_x_indices * res_x, dtype=int)
    ds = size_gradient(d1, d2, box_x_indices, box_y_indices, res_x, res_y)

    #store chosen points
    seeds = []
    while len(grid_ids) / init_len > 0.1:
        #choose random point as initial point
        chosen_id = np.random.choice(grid_ids)
        box_x = int(chosen_id / res_x)
        box_y = chosen_id - box_x * res_x
        seeds.append(np.array([box_x, box_y]))
        #get the size
        d = ds[box_x, box_y]

        #compute forbidden area
        distances = np.sqrt((box_x - box_x_indices)**2 + (box_y - box_y_indices)**2)
        grid[(d + ds - distances) > 0] = -1

        #disregard forbidden area
        grid_ids = grid.flatten()
        grid_ids = grid_ids[grid_ids >= 0]
        print("new len is: ", len(grid_ids))

    seeds = np.array(seeds) * dx

    plt.scatter(seeds[:, 0], seeds[:, 1])
    plt.savefig("seeds.png")

    vor = Voronoi(seeds)
    voronoi_plot_2d(vor, show_vertices=True, show_points=True)
    plt.savefig("grains.png")


