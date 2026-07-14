"""
Reproduit le notebook Learn_SurfaceCapsule.ipynb, découpé en modules src/.

Pipeline : forme -> données -> réseau -> entraînement -> grille 3D
           -> marching cubes -> figure PNG.

Lancer :  python main.py

Seule différence avec le notebook : une graine aléatoire fixe (set_seed)
pour rendre les résultats reproductibles d'une exécution à l'autre.
"""
import os
import numpy as np
import torch

from src.shapes import get_surface
from src.data import build_training_data
from src.model import ImplicitMLP
from src.train import train
from src.reconstruct import evaluate_grid, extract_surface
from src.visualize import plot_surface


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(0)

    # 1. Forme test
    surface_points, implicit_exact, box = get_surface("ellipsoid")

    # 2. Données d'entraînement
    X_train_t, y_train_t = build_training_data(surface_points, implicit_exact, box)

    # 3-4. Modèle + entraînement
    model = ImplicitMLP(hidden=128)
    train(model, X_train_t, y_train_t, n_epochs=2000, batch_size=2048, lr=1e-3)

    # 5-6. Reconstruction de la surface
    xs, ys, zs, values = evaluate_grid(model, box, grid_res=64)
    verts, faces, normals, vals = extract_surface(xs, ys, zs, values)
    print(f"Maillage extrait : {len(verts)} sommets, {len(faces)} triangles")

    # 7. Visualisation
    os.makedirs("output", exist_ok=True)
    plot_surface(verts, faces, surface_points, box, out_path="output/surface.png")


if __name__ == "__main__":
    main()
