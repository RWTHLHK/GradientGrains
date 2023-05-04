import matplotlib.pyplot as plt
import numpy as np
from GradientGrains import *
import pyvista as pv


# import pandas as pd


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
    fix_vor(vor=vor, min_dis=1)
    grains, grain_ids = get_grains(vor=vor, rve_x=10, rve_y=10)

    mesh = mesh_2d(grains, grain_ids)
    # mesh.plot(show_edges=True)

    mesh.triangulate(inplace=True)
    areas = mesh.compute_cell_sizes()['Area']
    sizes = mesh.compute_cell_sizes()

    refined_mesh = pv.UnstructuredGrid()
    for i in range(mesh.n_cells):
        cell = mesh.extract_cells(i)
        triangle = cell.extract_geometry()
        if areas[i] > 1:
            triangle.subdivide(nsub=1, inplace=True)

        refined_mesh = refined_mesh + triangle

    refined_mesh.plot(show_edges=True)
