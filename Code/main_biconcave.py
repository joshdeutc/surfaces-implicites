"""
Étape 3 — forme biconcave (globule rouge), le "test des limites" du mail.

Même méthode RÉALISTE que l'étape 1 (surface + dedans/dehors + eikonal,
réseau IGRNet Softplus). Aucune valeur exacte de f utilisée pour entraîner.
But : voir si/où Softplus arrondit la fossette centrale (biais spectral).

Lancer :  python main_biconcave.py
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
    surface_type = "biconcave"
    tag = "biconcave"

    # 1. Forme + normales (données réalistes : surface + dedans/dehors)
    surface_points, implicit_exact, box = get_surface(surface_type)
    normals = get_surface_normals(surface_type, surface_points)

    X_surf, normals_t, X_vol, sign_vol = build_training_data_realistic(
        surface_points, implicit_exact, box, normals=normals, n_vol=40000
    )
    n_in = int((sign_vol < 0).sum())
    n_out = int((sign_vol > 0).sum())
    print(f"Données : {len(X_surf)} points surface, "
          f"{n_in} dedans / {n_out} dehors (signe seulement)")

    # 2. Réseau IGRNet (Softplus) + entraînement eikonal
    model = IGRNet(radius_init=1.0)
    train_igr(model, X_surf, normals_t, X_vol, sign_vol, box,
              n_epochs=2500, batch_surf=2048, batch_vol=2048, lr=1e-3)

    # 2bis. Cache (pour ne jamais réentraîner si on retouche la suite)
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), f"output/model_{tag}.pt")
    print(f"Modèle sauvegardé -> output/model_{tag}.pt")

    # 3. Reconstruction
    grid_res = 96  # plus fin : la fossette est un détail
    xs, ys, zs, values = evaluate_grid(model, box, grid_res=grid_res)
    np.savez(f"output/grid_{tag}.npz", xs=xs, ys=ys, zs=zs, values=values, box=box)
    verts, faces, normals_m, vals = extract_surface(xs, ys, zs, values)
    print(f"Maillage extrait : {len(verts)} sommets, {len(faces)} triangles")

    # 4. Métrique de reconstruction (formule utilisée SEULEMENT pour évaluer)
    err = np.abs(implicit_exact(verts.astype(np.float32))).mean()
    maille = (2 * box) / grid_res
    print(f"Erreur moyenne de reconstruction : mean|f_exact| = {err:.4f} "
          f"(maille grille ~ {maille:.4f})")

    # 5. Figures : vue 3D + coupe verticale (pour voir la fossette)
    plot_surface(verts, faces, surface_points, box,
                 out_path=f"output/surface_{tag}.png")

    # coupe dans le plan y=0 : profil z(x) reconstruit vs vrai
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mask = np.abs(verts[:, 1]) < 0.03
    plt.figure(figsize=(7, 4))
    plt.scatter(verts[mask, 0], verts[mask, 2], s=3, label="reconstruit")
    xx = np.linspace(-1, 1, 400)
    from src.shapes import biconcave_zfun
    zz = biconcave_zfun(np.abs(xx))
    plt.plot(xx, zz, "r-", lw=1, label="vrai (haut)")
    plt.plot(xx, -zz, "r-", lw=1, label="vrai (bas)")
    plt.gca().set_aspect("equal")
    plt.xlabel("x"); plt.ylabel("z"); plt.legend()
    plt.title("Coupe y=0 — fossette du globule rouge")
    plt.tight_layout()
    plt.savefig(f"output/coupe_{tag}.png", dpi=130)
    plt.close()
    print(f"Coupe sauvegardée -> output/coupe_{tag}.png")


if __name__ == "__main__":
    main()
