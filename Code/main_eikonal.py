"""
Étape 1 — version RÉALISTE : apprendre la forme à partir de points seuls.

On n'utilise JAMAIS la valeur exacte de f. Seulement :
  - des points de surface (f = 0) + leurs normales,
  - des points de volume étiquetés dedans / dehors (signe),
  - la loi eikonal ‖∇f‖ = 1.

À la fin, on reconstruit la surface et on MESURE l'erreur en comparant les
sommets reconstruits à l'ellipsoïde vrai (le seul endroit où l'on s'autorise
la formule : pour évaluer, pas pour entraîner).

Lancer :  python main_eikonal.py
"""
import os
import numpy as np
import torch

from src.shapes import get_surface, get_surface_normals
from src.data import build_training_data_realistic
from src.model import IGRNet
from src.train import train_igr
from src.reconstruct import evaluate_grid, extract_surface
from src.visualize import plot_surface


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(0)
    surface_type = "ellipsoid"

    # 1. Forme + normales (données réalistes : points + dedans/dehors)
    surface_points, implicit_exact, box = get_surface(surface_type)
    normals = get_surface_normals(surface_type, surface_points)

    X_surf, normals_t, X_vol, sign_vol = build_training_data_realistic(
        surface_points, implicit_exact, box, normals=normals
    )
    n_in = int((sign_vol < 0).sum())
    n_out = int((sign_vol > 0).sum())
    print(f"Données : {len(X_surf)} points surface, "
          f"{n_in} dedans / {n_out} dehors (signe seulement)")

    # 2. Réseau adapté (init géométrique + Softplus) + entraînement eikonal
    model = IGRNet(radius_init=1.0)
    train_igr(model, X_surf, normals_t, X_vol, sign_vol, box,
              n_epochs=2000, batch_surf=1024, batch_vol=1024, lr=1e-3)

    # 2bis. CACHE : on sauvegarde le modèle entraîné pour ne jamais avoir à
    #       réentraîner si on veut seulement retoucher reconstruction/figure.
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), "output/model_eikonal.pt")
    print("Modèle sauvegardé -> output/model_eikonal.pt")

    # 3. Reconstruction
    grid_res = 64
    xs, ys, zs, values = evaluate_grid(model, box, grid_res=grid_res)
    np.savez("output/grid_eikonal.npz", xs=xs, ys=ys, zs=zs, values=values, box=box)
    verts, faces, normals_m, vals = extract_surface(xs, ys, zs, values)
    print(f"Maillage extrait : {len(verts)} sommets, {len(faces)} triangles")

    # 4. Vérification quantitative : les sommets reconstruits doivent être
    #    proches de la vraie surface (f_exact ≈ 0).
    err = np.abs(implicit_exact(verts.astype(np.float32))).mean()
    maille = (2 * box) / grid_res
    print(f"Erreur moyenne de reconstruction : mean|f_exact| = {err:.4f} "
          f"(maille grille ~ {maille:.4f})")
    if err < 0.03:
        print("OK -> forme reconstruite fidèlement SANS la formule.")
    else:
        print("Erreur encore élevée -> à améliorer (epochs / poids).")

    # 5. Figure
    os.makedirs("output", exist_ok=True)
    plot_surface(verts, faces, surface_points, box,
                 out_path="output/surface_eikonal.png")


if __name__ == "__main__":
    main()
