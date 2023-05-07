import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.measurement import area
import taichi as ti
import taichi.math as tm
import pyvista as pv

def size_gradient(d1, d2, x, y, res_x, res_y):
    return (d2 - d1) / res_y * y + d1


def seeding(rve_x: float, rve_y: float, res: float, dmin: float, dmax: float) -> np.ndarray:
    # Rasterization
    res_x, res_y = int(rve_x / res), int(rve_y / res)
    init_len = res_x * res_y
    grid_ids = np.arange(init_len)
    grid = grid_ids.reshape(res_x, res_y)

    # precompute sizes
    box_x_indices = np.array(grid / res_x, dtype=int)
    box_y_indices = np.array(grid - box_x_indices * res_x, dtype=int)
    ds = size_gradient(dmin, dmax, box_x_indices, box_y_indices, res_x, res_y)

    # store chosen points
    seeds = []
    while len(grid_ids) / init_len > 0.1:
        # choose random point as initial point
        chosen_id = np.random.choice(grid_ids)
        box_x = int(chosen_id / res_x)
        box_y = chosen_id - box_x * res_x
        seeds.append(np.array([box_x, box_y]))
        # get the size
        d = ds[box_x, box_y]

        # compute forbidden area
        distances = np.sqrt((box_x - box_x_indices) ** 2 + (box_y - box_y_indices) ** 2)
        grid[(d + ds - distances) > 0] = -1

        # disregard forbidden area
        grid_ids = grid.flatten()
        grid_ids = grid_ids[grid_ids >= 0]

    seeds = np.array(seeds) * res
    return seeds


def gen_periodic_voronoi(seeds: np.ndarray, rve_x: float) -> scipy.spatial.Voronoi:
    left_side_seeds = seeds.copy()
    left_side_seeds[:, 0] = left_side_seeds[:, 0] - rve_x
    right_side_seeds = seeds.copy()
    right_side_seeds[:, 0] = right_side_seeds[:, 0] + rve_x
    seeds = np.vstack([left_side_seeds, seeds])
    seeds = np.vstack([seeds, right_side_seeds])
    vor = Voronoi(seeds)
    return vor


def voronoi_finite_polygons_2d(vor, radius=None):
    # @https://github.com/Sklavit
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def get_grains(vor: scipy.spatial.Voronoi, rve_x: float, rve_y: float) -> tuple[list, list]:
    regions, vertices = voronoi_finite_polygons_2d(vor)

    min_x = 0
    max_x = rve_x
    min_y = 0
    max_y = rve_y

    box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    grains = []
    grain_ids = []
    # grain_areas = []
    i = 0
    for region in regions:
        grain_id = i % int(len(vor.points) / 3)
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        poly = poly.intersection(box)  # needs modification
        polygon = [p for p in poly.exterior.coords]
        if len(polygon) > 0:
            grains.append(polygon)
            grain_ids.append(grain_id)
            # grain_areas.append(area(poly))
        i += 1

    return grains, grain_ids


def fix_vor(vor: scipy.spatial.Voronoi, min_dis: float):
    for i, (p1, p2) in enumerate(vor.ridge_vertices):
        if p1 == -1 or p2 == -1:
            # Skip infinite ridges
            continue

        # Get the coordinates of the two vertices of the current ridge
        v1 = vor.vertices[p1]
        v2 = vor.vertices[p2]

        # get the vector that connects vertices v1 and v2
        vec = v2 - v1

        # Compute the length of the current ridge and get unit vector
        length = np.linalg.norm(vec)

        if length < min_dis:
            unit_vec = vec / length
            new_v2 = unit_vec * min_dis + v1
            vor.vertices[p2] = new_v2


if __name__ == "__main__":
    min_x = -10
    min_y = 0
    max_x = 20
    max_y = 10
    rve_x = 10
    seeds = seeding(10, 10, 0.5, 1, 5)
    # vor = Voronoi(seeds)
    # print(vor.vertices)
    left_side_seeds = seeds.copy()
    left_side_seeds[:, 0] = left_side_seeds[:, 0] - rve_x
    right_side_seeds = seeds.copy()
    right_side_seeds[:, 0] = right_side_seeds[:, 0] + rve_x

    periodic_seeds = np.vstack([left_side_seeds, seeds])
    periodic_seeds = np.vstack([periodic_seeds, right_side_seeds])
    fig, ax = plt.subplots()
    # ax1.scatter(left_side_seeds[:, 0], left_side_seeds[:, 1])
    # ax2.scatter(seeds[:, 0], seeds[:, 1])
    # ax3.scatter(right_side_seeds[:, 0], right_side_seeds[:, 1])
    # plt.savefig("periodic_seeds.png")
    ax.set_xticks(np.arange(-10, 22, 2))
    ax.set_yticks(np.arange(0, 12, 2))
    ax.set_xlim(-10, 20)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.scatter(periodic_seeds[:, 0], periodic_seeds[:, 1])
    ax.plot([0, 0], [0, 10], color='r')
    ax.plot([10, 10], [0, 10], color='r')
    ax.plot([-10, -10], [0, 10], color='r')
    ax.plot([20, 20], [0, 10], color='r')
    plt.show()
    plt.close()
    fig, ax = plt.subplots()
    vor = gen_periodic_voronoi(seeds=seeds, rve_x=10)
    ax.set_xticks(np.arange(-10, 22, 2))
    ax.set_yticks(np.arange(0, 12, 2))
    ax.set_xlim(-10, 20)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.plot([0, 0], [-1, 11], color='r')
    ax.plot([10, 10], [-1, 11], color='r')
    ax.plot([-10, -10], [-1, 11], color='r')
    ax.plot([20, 20], [-1, 11], color='r')
    voronoi_plot_2d(vor, ax=ax)
    plt.show()
    # plt.savefig("periodic_voronoi.png")
    # fix_vor(vor=vor, min_dis=1)
    # regions, vertices = voronoi_finite_polygons_2d(vor=vor)
    # region = regions[0]
    # polygon = vertices[region]
    # poly = Polygon(polygon)
    # box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
    # poly = poly.intersection(box)
    # polygon = [p for p in poly.exterior.coords]
    # point = vor.points[0]
    # print(point, polygon)
    #
    # # possion sampling
    # ti.init(arch=ti.cpu)
    # num_mesh_elems = 5
    # grain_vertices = np.array(polygon)
    # xmin, ymin = np.min(grain_vertices, axis=0)
    # xmax, ymax = np.max(grain_vertices, axis=0)
    # width = xmax - xmin
    # height = ymax - ymin
    # grain_area = area(poly)
    # radius = ti.sqrt(grain_area / num_mesh_elems / np.pi)
    # cell_size = radius / ti.sqrt(2)
    # grid_width = int(width / cell_size) + 1
    # grid_height = int(height / cell_size) + 1
    # grid = ti.field(dtype=int, shape=(grid_width, grid_height))
    # num_samples = int((width * height) / (np.pi * radius * radius) + 1) * 10
    # samples = ti.Vector.field(2, float, shape=int(num_samples))
    #
    # grid.fill(-1)
    #
    # @ti.func
    # def check_collision(p, index):
    #     x, y = index
    #     collision = False
    #     for i in range(ti.max(0, x - 2), ti.min(grid_width, x + 3)):
    #         for j in range(ti.max(0, y - 2), ti.min(grid_height, y + 3)):
    #             if grid[i, j] != -1:
    #                 q = samples[grid[i, j]]
    #                 if (q - p).norm() < radius - 1e-6:
    #                     collision = True
    #     return collision
    #
    #
    # @ti.kernel
    # def poisson_disk_sample(desired_samples: int, init_point_x: float, init_point_y: float) -> int:
    #     samples[0] = tm.vec2(init_point_x - xmin, init_point_y - ymin)
    #     grid[int((init_point_x - xmin) / cell_size), int((init_point_y - ymin) / cell_size)] = 0
    #     head, tail = 0, 1
    #     while head < tail and head < desired_samples:
    #         source_x = samples[head]
    #         head += 1
    #
    #         for _ in range(100):
    #             theta = ti.random() * 2 * tm.pi
    #             offset = tm.vec2(tm.cos(theta), tm.sin(theta)) * (1 + ti.random()) * radius
    #             new_x = source_x + offset
    #             new_index = int(new_x / cell_size)
    #
    #             if 0 <= new_x[0] < width and 0 <= new_x[1] < height:
    #                 collision = check_collision(new_x, new_index)
    #                 if not collision and tail < desired_samples:
    #                     samples[tail] = new_x
    #                     grid[new_index] = tail
    #                     tail += 1
    #     return tail
    #
    # poisson_disk_sample(desired_samples=num_samples, init_point_x=point[0], init_point_y=point[1])
    # samples = samples.to_numpy()
    # samples[:, 0] += xmin
    # samples[:, 1] += ymin
    # samples = np.unique(samples, axis=0)
    #
    # #check if the samples are inside polygon
    # points = MultiPoint(samples)
    # intersection = poly.intersection(points)
    # inside_points = np.array([[point.x, point.y] for point in intersection.geoms])
    # poly = np.array(polygon)
    # # plt.plot(poly[:, 0], poly[:, 1])
    # # plt.scatter(inside_points[:, 0], inside_points[:, 1])
    # # plt.show()
    # cloud = np.vstack([poly, inside_points])
    # zeros = np.zeros((cloud.shape[0], 1))
    # cloud = np.hstack([cloud, zeros])
    # cloud = pv.PolyData(cloud)
    # mesh = cloud.delaunay_2d()
    # quality = mesh.compute_cell_quality()
    # print(quality['CellQuality'])
    # mesh.plot(show_edges=True)
    pass