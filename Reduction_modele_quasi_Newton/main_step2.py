"""
Étape 2 — comparaison Softplus vs Fourier, SANS normales (données brutes du mail).

Pour les DEUX réseaux : mêmes données (surface + dedans/dehors, pas de normales),
même eikonal. On reconstruit la biconcave et on compare la fossette.

Hypothèse : sans normales, Softplus arrondit la fossette (biais spectral),
l'encodage de Fourier la tient mieux.

Lancer :  python main_step2.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.shapes import get_surface, biconcave_zfun
from src.data import build_training_data_realistic
from src.model import IGRNet, FourierNet
from src.train import train_igr
from src.reconstruct import evaluate_grid, extract_surface


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(model, tag, n_epochs=3000):
    """Entraîne SANS normales sur la biconcave et reconstruit."""
    set_seed(0)  # mêmes données pour les deux réseaux
    surface_points, implicit_exact, box = get_surface("biconcave")

    X_surf, _, X_vol, sign_vol = build_training_data_realistic(
        surface_points, implicit_exact, box, normals=None, n_vol=40000
    )

    train_igr(model, X_surf, None, X_vol, sign_vol, box,
              n_epochs=n_epochs, batch_surf=2048, batch_vol=2048, lr=1e-3)

    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), f"output/model_{tag}.pt")

    xs, ys, zs, values = evaluate_grid(model, box, grid_res=96)
    verts, faces, _, _ = extract_surface(xs, ys, zs, values)
    err = float(np.abs(implicit_exact(verts.astype(np.float32))).mean())
    print(f"[{tag}] sommets={len(verts)}  mean|f_exact|={err:.4f}")
    return verts, err, box


def cross_section(verts, ax, title):
    """Coupe y=0 : profil reconstruit vs vrai."""
    mask = np.abs(verts[:, 1]) < 0.03
    ax.scatter(verts[mask, 0], verts[mask, 2], s=3, label="reconstruit")
    xx = np.linspace(-1, 1, 400)
    zz = biconcave_zfun(np.abs(xx))
    ax.plot(xx, zz, "r-", lw=1)
    ax.plot(xx, -zz, "r-", lw=1, label="vrai")
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8)


def main():
    print("=== 1) Softplus (IGRNet) SANS normales ===")
    v_soft, e_soft, box = run_one(IGRNet(radius_init=1.0), "soft_nonorm")

    print("=== 2) Fourier SANS normales ===")
    v_four, e_four, _ = run_one(FourierNet(n_freqs=6), "fourier_nonorm")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cross_section(v_soft, axes[0], f"Softplus sans normales\nmean|f|={e_soft:.4f}")
    cross_section(v_four, axes[1], f"Fourier sans normales\nmean|f|={e_four:.4f}")
    fig.suptitle("Étape 2 — fossette du globule rouge : Softplus vs Fourier (données brutes)")
    fig.tight_layout()
    fig.savefig("output/compare_step2.png", dpi=130)
    plt.close(fig)
    print("Comparaison sauvegardée -> output/compare_step2.png")
    print(f"RÉSUMÉ : Softplus mean|f|={e_soft:.4f}  |  Fourier mean|f|={e_four:.4f}")


if __name__ == "__main__":
    main()
