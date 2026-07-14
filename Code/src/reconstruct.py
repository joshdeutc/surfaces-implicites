"""
Sections 5 et 6 du notebook : évaluation sur une grille 3D puis
extraction de l'iso-surface f = 0 par marching cubes.
"""
import numpy as np
import torch
from skimage import measure

from src.device import get_device


def evaluate_grid(model, box, grid_res=64):
    """Évalue f appris sur une grille régulière grid_res^3."""
    device = get_device()
    model = model.to(device)

    xs = np.linspace(-box, box, grid_res)
    ys = np.linspace(-box, box, grid_res)
    zs = np.linspace(-box, box, grid_res)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

    with torch.no_grad():
        values = model(torch.tensor(grid_points, device=device)).cpu().numpy().reshape(
            grid_res, grid_res, grid_res
        )
    return xs, ys, zs, values


def extract_surface(xs, ys, zs, values):
    """Marching cubes : reconstruit le maillage triangulé de f = 0."""
    spacing = (
        xs[1] - xs[0],
        ys[1] - ys[0],
        zs[1] - zs[0],
    )

    verts, faces, normals, vals = measure.marching_cubes(
        values, level=0.0, spacing=spacing
    )

    # marching_cubes part de 0 : on recale dans [-box, box]^3
    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += zs[0]
    return verts, faces, normals, vals
