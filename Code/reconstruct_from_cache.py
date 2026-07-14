"""
Reconstruit la surface + figure + métrique À PARTIR DU CACHE,
sans réentraîner (instantané).

Pré-requis : avoir lancé main_eikonal.py au moins une fois
(qui écrit output/grid_eikonal.npz).

Lancer :  python reconstruct_from_cache.py
"""
import numpy as np

from src.shapes import get_surface
from src.reconstruct import extract_surface
from src.visualize import plot_surface


def main():
    data = np.load("output/grid_eikonal.npz")
    xs, ys, zs, values = data["xs"], data["ys"], data["zs"], data["values"]
    box = float(data["box"])

    verts, faces, normals_m, vals = extract_surface(xs, ys, zs, values)
    print(f"Maillage extrait : {len(verts)} sommets, {len(faces)} triangles")

    # métrique de reconstruction (la formule sert seulement à ÉVALUER)
    surface_points, implicit_exact, _ = get_surface("ellipsoid")
    err = np.abs(implicit_exact(verts.astype(np.float32))).mean()
    grid_res = len(xs)
    maille = (2 * box) / grid_res
    print(f"Erreur moyenne de reconstruction : mean|f_exact| = {err:.4f} "
          f"(maille grille ~ {maille:.4f})")

    plot_surface(verts, faces, surface_points, box,
                 out_path="output/surface_eikonal.png")


if __name__ == "__main__":
    main()
