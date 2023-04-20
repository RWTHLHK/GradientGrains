import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


def size_gradient(d1, d2, x, y):
    return (d2 - d1) / 10 * y + d1

def check_collision(p, index) ->bool:
    x, y = index[0, 0], index[0, 1]
    collision = False
    for i in range(max(0, x - 2), min(res_x, x + 3)):
        for j in range(max(0, y - 2), min(res_y, y + 3)):
            if grid[i, j] != -1:
                q = coords[grid[i, j],:]
                if np.linalg.norm(q - p) < radius - 1e-6:
                    collision = True
    return collision

if __name__ == "__main__":

    d1 = 1
    d2 = 10
    n = 0
    coords = np.zeros((5, 2))

    radius = 0.5
    dx = radius / np.sqrt(2)
    res_x = res_y = int(10 / dx) + 1
    res = (res_x, res_y)
    desired_samples = 100000
    grid = np.zeros((res_x, res_y), dtype=int) - 1
    coords[0,:] = 0.5, 0.5
    grid[int(0.5 / dx), int(0.5 / dx)] = 0
    while n < 5:
        coord = np.random.rand(1, 2) * 10
        dg = size_gradient(d1=d1, d2=d2, x=coord[0, 0], y=coord[0, 1])
        alpha = np.random.uniform()
        index = np.array(coord / dx, dtype=int)
        # print(index)
        collision = check_collision(coord, index)
        if not collision:
            if (d1 / dg) ** 2 > alpha:
                coords[n, :] = coord
                grid[index] = n
                n += 1

    vor = Voronoi(coords)
    voronoi_plot_2d(vor, show_vertices=True, show_points=True)
    plt.savefig("test.png")
    plt.close()
    grains = []
    part_grains = []
    vertices = vor.vertices
    # for region in vor.regions:
    #     if len(region) > 0:
    #         if(np.asarray(region) < 0).any():
    #             part_grains.append(region)
    #             # pass
    #         else:
    #             grains.append(vertices[region])

    # vertices ridge_vertices顶点坐标
    # ridge_points ridge 与points连线相垂直
    # print(vor.ridge_vertices)
    # print(vor.ridge_points)
    #
    # print(part_grains)
    print(vor.points)
    print(vertices)
    print(vor.ridge_vertices)
    print(vor.ridge_points)
    print(vor.ridge_dict)
    print(vor.regions)
    print(vor.point_region)
    # to construct a grain we need seed point and ridge_dict
    # to know the ridges, and to which points they are perpendicular to
    # vertices to know the coords of end points of ridges
