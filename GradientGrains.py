import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from shapely.geometry import Polygon


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


def get_grains(vor: scipy.spatial.Voronoi, rve_x: float, rve_y: float) -> list:
    regions, vertices = voronoi_finite_polygons_2d(vor)

    min_x = 0
    max_x = rve_x
    min_y = 0
    max_y = rve_y

    box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    grains = []
    grain_ids = []
    i = 0
    for region in regions:
        grain_id = i % int(len(vor.points) / 3)
        polygon = vertices[region]
        # polygon = remove_close_vertices(polygon, 0.5)
        # Clipping polygon
        poly = Polygon(polygon)
        poly = poly.intersection(box) # needs modification
        polygon = [p for p in poly.exterior.coords]
        if len(polygon) > 0:
            grains.append(polygon)
            grain_ids.append(grain_id)
        i += 1

    return grains, grain_ids

def fix_vor(vor):
    vertices = vor.vertices
    for i, (p1, p2) in enumerate(vor.ridge_points):
        ridge_vertices = vor.ridge_vertices[i]
        if ridge_vertices[0] != -1 and ridge_vertices[1] != -1:
            length = ((vor.vertices[ridge_vertices[0]] - vor.vertices[ridge_vertices[1]]) ** 2).sum() ** 0.5
            if length < 0.5:
                new_index = min(ridge_vertices[0], ridge_vertices[1])
                vertices[ridge_vertices[0]] = vertices[ridge_vertices[1]] = vertices[new_index]
        
        

if __name__ == "__main__":
    seeds = seeding(10, 10, 0.5, 1, 5)
    vor = Voronoi(seeds)
    print(vor.vertices)
    # voronoi_plot_2d(vor)
    # plt.xlim([0, 10])
    # plt.ylim([0, 10])
    # plt.savefig("old_vor.png")
    # plt.close()
    fix_vor(vor=vor)
    # vor = gen_periodic_voronoi(seeds=seeds, rve_x=10)
    # grains, grain_ids = get_grains(vor=vor, rve_x=10, rve_y=10)

    print(vor.vertices)
    voronoi_plot_2d(vor)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.savefig("new_vor.png")
    #
    # grains = get_grains(vor=vor, rve_x=10, rve_y=10)
    #
    # for grain in grains:
    #     plt.fill(*zip(*grain), alpha=0.4)
    #
    # plt.plot(vor.points[:, 0], vor.points[:, 1], 'ko')
    # plt.axis('equal')
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    #
    # plt.savefig('voronoi.png')
    # plt.show()

