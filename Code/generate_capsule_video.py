"""
Script de génération d'animation GIF / Vidéo de la déformation de la capsule.
Génère une séquence comparant :
  1. Données réelles (BMBI maillage de référence)
  2. Reconstruction par le réseau neuronal 4D f(x, y, z, t)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import imageio

from src import capsules
from src.model_temporal import TemporalIGRNet, figer_instant
from src.reconstruct import evaluate_grid, extract_surface

def make_animation(mode="uniforme", out_gif="output/capsule_deformation_60fps.gif", n_frames=60):
    device = torch.device("cpu")
    model_path = f"output/modele_capsule_{mode}.pt"
    if not os.path.exists(model_path):
        print(f"Modèle {model_path} non encore disponible.")
        return

    print("Chargement du modèle entraîné...")
    model = TemporalIGRNet(radius_init=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Chargement des données BMBI...")
    faces, caps = capsules.charger()
    box = 0.6984
    grid_res = 48  # Résolution rapide et nette pour le rendu 3D

    t_values = np.linspace(0.0, 1.0, n_frames)
    pas_ef = (t_values * 4000).astype(int)

    frames = []
    os.makedirs("output/tmp_frames", exist_ok=True)

    print(f"Génération des {n_frames} frames de l'animation...")
    for i, (t_norm, p) in enumerate(zip(t_values, pas_ef)):
        # 1. Vérité terrain BMBI
        verts_ref, _ = capsules.instant(caps, p, recentre=True)

        # 2. Prédiction par le réseau à l'instant t
        vue = figer_instant(model, t_norm, device=device)
        xs, ys, zs, vals = evaluate_grid(vue, box=box, grid_res=grid_res)
        
        # Marching cubes sur la prédiction
        verts_pred, faces_pred, _, _ = extract_surface(xs, ys, zs, vals)

        # Figure comparative
        fig = plt.figure(figsize=(10, 5))
        
        # Subplot 1 : Vérité terrain BMBI
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        mesh1 = Poly3DCollection(verts_ref[faces], alpha=0.7, edgecolor='k', linewidths=0.2)
        mesh1.set_facecolor('royalblue')
        ax1.add_collection3d(mesh1)
        ax1.set_xlim(-box, box); ax1.set_ylim(-box, box); ax1.set_zlim(-box, box)
        ax1.set_title(f"Vérité terrain BMBI (pas {p})", fontsize=10)
        ax1.axis('off')
        ax1.view_init(elev=20, azim=45)

        # Subplot 2 : Réseau de neurones 4D f(x,y,z,t)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        if len(verts_pred) > 0 and len(faces_pred) > 0:
            mesh2 = Poly3DCollection(verts_pred[faces_pred], alpha=0.8, edgecolor='crimson', linewidths=0.1)
            mesh2.set_facecolor('tomato')
            ax2.add_collection3d(mesh2)
        ax2.set_xlim(-box, box); ax2.set_ylim(-box, box); ax2.set_zlim(-box, box)
        is_cache = "Caché (interpolé)" if p not in [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000] else "Observé"
        ax2.set_title(f"Réseau f(x,y,z,t) [{is_cache}]\nt = {t_norm:.2f}", fontsize=10)
        ax2.axis('off')
        ax2.view_init(elev=20, azim=45)

        fig.suptitle("Déformation Espace-Temps de la Capsule", fontsize=12, fontweight='bold')
        fig.tight_layout()

        frame_path = f"output/tmp_frames/frame_{i:03d}.png"
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)

        frames.append(imageio.v2.imread(frame_path))
        print(f"  Frame {i+1}/{n_frames} rendue...")

    imageio.mimsave(out_gif, frames, duration=80, loop=0)
    print(f"Animation sauvegardée avec succès dans : {out_gif}")

if __name__ == "__main__":
    make_animation()
