import matplotlib.pyplot as plt
import numpy as np
from GradientGrains import *
import pyvista as pv
import pandas as pd


def mesh_2d(grains: list, grain_ids: list) -> pv.PolyData:
    mesh = pv.PolyData()
    for grain in grains:
        points = np.array(grain)
        zeros = np.zeros((points.shape[0], 1))
        points = np.hstack([points, zeros])
        faces = np.hstack([len(points), np.arange(len(points))])
        mesh = mesh + pv.PolyData(points, faces)

    mesh.cell_data['GrainID'] = grain_ids
    return mesh


def adaptive_refinement(mesh: pv.PolyData, degree: int) -> pv.PolyData:
    mesh.triangulate(inplace=True)
    mesh.subdivide(nsub=degree, subfilter='linear', inplace=True)
    return mesh


if __name__ == "__main__":
    seeds = seeding(10, 10, 0.5, 1, 5)
    vor = gen_periodic_voronoi(seeds=seeds, rve_x=10)
    voronoi_plot_2d(vor, show_vertices=True, show_points=True)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.savefig("test_grains.png")
    plt.close()

    grains, grain_ids = get_grains(vor, 10, 10)
    mesh = mesh_2d(grains=grains, grain_ids=grain_ids)
    mesh.plot(show_edges=True)

    refined_mesh = adaptive_refinement(mesh=mesh, degree=2)
    refined_mesh.plot(show_edges=True)


