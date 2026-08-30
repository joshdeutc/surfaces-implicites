"""
La figure pédagogique : quand ça marche, et quand ça casse.

Trois lignes, quatre instants que le réseau n'a JAMAIS vus :

    ligne 1   la vraie capsule (le maillage du BMBI)
    ligne 2   ce que produit un réseau entraîné sur 6 instants  -> ça marche
    ligne 3   ce que produit un réseau entraîné sur 3 instants  -> ça casse

Les surfaces prédites sont COLORÉES PAR LEUR ÉCART à la vraie surface, mesuré au
point le plus proche d'un nuage dense tiré sur le maillage. Vert = la surface est
au bon endroit, rouge = elle s'en écarte. L'écart est exprimé en pourcentage du
diamètre de la capsule, pour être lisible sans connaître les unités.

Les douze vues partagent la même caméra et la même échelle de couleur : une
différence visible est une différence de forme, jamais un effet de cadrage.

Lancer :  /usr/local/bin/python3 main_capsules_figure.py
"""
import os

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree

from src import capsules
from src.campagne import GRILLE, N_REF, mc
from src.device import get_device
from src.model_temporal import TemporalIGRNet, figer_instant
from src.reconstruct import evaluate_grid
from src.viz_pv import to_polydata

OUT = "output"
PAS_MAX = 2000
INSTANTS = [100, 300, 500, 900]        # cachés pour 6 photos ET pour 3 photos
CONFIGS = [("6", "AVEC 6 PHOTOS\nça marche"),
           ("3", "AVEC 3 PHOTOS\nça casse")]
ERREUR_MAX = 5.0                       # % du diamètre, borne de l'échelle


def charger_modele(tag, device):
    m = TemporalIGRNet(radius_init=0.5)
    m.load_state_dict(torch.load(f"{OUT}/modele_evenement_{tag}.pt",
                                 map_location="cpu"))
    return m.to(device).eval()


def main():
    device = get_device()
    faces, caps = capsules.charger()
    evalu = list(range(0, PAS_MAX + 1, 100))
    box = capsules.boite(faces, caps, evalu, marge=1.25)

    # diamètre équivalent de la capsule, pour exprimer l'écart en %
    V0, _ = capsules.instant(caps, 0)
    diametre = 2 * (3 * abs(capsules.geometrie(faces, V0)[0]) / (4 * np.pi)) ** (1 / 3)
    print(f"diamètre équivalent de la capsule : {diametre:.4f}")

    modeles = {tag: charger_modele(tag, device) for tag, _ in CONFIGS}

    lignes = []          # [(titre, [maillage par instant])]
    vrais = []
    for p in INSTANTS:
        Vp, _ = capsules.instant(caps, p)
        axe, val = capsules.champ_reference(Vp, faces, box, grille=GRILLE,
                                            n_echantillons=N_REF)
        v, f = mc(axe, val)
        vrais.append({"mesh": to_polydata(v, f), "nuage":
                      capsules.echantillon_surface(Vp, faces, n=N_REF, seed=3)[0],
                      "axe": axe})
    formes = []
    for p in INSTANTS:
        Vp, _ = capsules.instant(caps, p)
        formes.append(capsules.demi_axes(Vp)[1])
    lignes.append(("LA VÉRITÉ\ndonnées du BMBI",
                   [d["mesh"] for d in vrais], None,
                   [f"allongement : {a:.2f}" for a in formes]))

    for tag, titre in CONFIGS:
        maillages, medians = [], []
        for k, p in enumerate(INSTANTS):
            vue = figer_instant(modeles[tag], p / PAS_MAX, device=device)
            _, _, _, val = evaluate_grid(vue, box, grid_res=GRILLE)
            v, f = mc(vrais[k]["axe"], val)
            ecart = 100.0 * cKDTree(vrais[k]["nuage"]).query(v)[0] / diametre
            maillages.append(to_polydata(v, f, scalars=ecart,
                                         scalar_name="ecart"))
            medians.append(float(np.median(ecart)))
            print(f"  {tag} photos, pas {p:5d} : écart médian "
                  f"{medians[-1]:.2f} %, 99e centile "
                  f"{np.percentile(ecart, 99):.2f} %")
        lignes.append((titre, maillages, "ecart", medians))

    rendre(lignes, f"{OUT}/capsules_quand_ca_casse.png")


def vue_png(mesh, chemin, champ=None, cam=None):
    """Rend UNE vue dans son propre PNG, avec une caméra imposée."""
    p = pv.Plotter(off_screen=True, window_size=[700, 700])
    p.set_background("white")
    if champ is None:
        p.add_mesh(mesh, color="#9fb6c9", smooth_shading=True, specular=0.35)
    else:
        p.add_mesh(mesh, scalars=champ, cmap="RdYlGn_r",
                   clim=[0.0, ERREUR_MAX], smooth_shading=True,
                   specular=0.35, show_scalar_bar=False)
    if cam is None:
        p.camera_position = "iso"
        p.camera.zoom(1.35)
        cam = p.camera_position
    else:
        p.camera_position = cam
    p.screenshot(chemin)
    p.close()
    return cam


def rendre(lignes, chemin):
    """
    Compose la grille avec matplotlib : PyVista rend chaque vue dans son coin,
    matplotlib place les titres et la barre de couleur. C'est le seul moyen
    d'avoir des légendes qui ne se chevauchent pas, et une caméra identique
    partout — donc des panneaux réellement comparables.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    tmp = f"{OUT}/_vues"
    os.makedirs(tmp, exist_ok=True)
    n_l, n_c = len(lignes), len(INSTANTS)

    cam = None
    chemins = []
    for i, (_, maillages, champ, _) in enumerate(lignes):
        ligne = []
        for j, mesh in enumerate(maillages):
            f = f"{tmp}/v{i}{j}.png"
            cam = vue_png(mesh, f, champ=champ, cam=cam)
            ligne.append(f)
        chemins.append(ligne)

    fig, axes = plt.subplots(n_l, n_c, figsize=(4.0 * n_c, 4.3 * n_l))
    for i, (titre, _, champ, notes) in enumerate(lignes):
        for j in range(n_c):
            a = axes[i, j]
            a.imshow(plt.imread(chemins[i][j]))
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if i == 0:
                a.set_title(f"instant caché n\u00b0 {INSTANTS[j]}", fontsize=15,
                            pad=10, color="#222222")
            if champ is None:
                a.set_xlabel(notes[j], fontsize=13, color="#555555", labelpad=6)
            else:
                c = "#1a7a3a" if notes[j] < 0.8 else ("#b06a00" if notes[j] < 2.0
                                                      else "#a01010")
                a.set_xlabel(f"écart médian  {notes[j]:.1f} %", fontsize=15,
                             color=c, labelpad=6, weight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.89, bottom=0.10,
                        wspace=0.02, hspace=0.18)

    couleurs_ligne = ["#333333", "#1a7a3a", "#a01010"]
    for i, (titre, _, _, _) in enumerate(lignes):
        bb = axes[i, 0].get_position()
        fig.text(0.045, (bb.y0 + bb.y1) / 2, titre, rotation=90,
                 va="center", ha="center", fontsize=16, weight="bold",
                 color=couleurs_ligne[i], linespacing=1.5)
    cax = fig.add_axes([0.22, 0.045, 0.56, 0.018])
    sm = ScalarMappable(norm=Normalize(0, ERREUR_MAX), cmap="RdYlGn_r")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("écart à la vraie surface, en % du diamètre de la capsule   "
                 "—   vert = juste, rouge = faux", fontsize=13)
    fig.suptitle("Un seul réseau f(x,y,z,t) doit deviner les instants qu'on ne "
                 "lui a jamais montrés", fontsize=18, y=0.97)
    fig.savefig(chemin, dpi=110)
    plt.close(fig)
    print(f"\n-> {chemin}  ({os.path.getsize(chemin)/1024:.0f} Ko)")


if __name__ == "__main__":
    main()
