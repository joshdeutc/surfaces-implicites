"""
Comparaison d'architectures : Softplus / Fourier / SIREN, SANS normales.

Le projet a déjà comparé Softplus et Fourier (main_step2.py). SIREN (Sitzmann
et al. 2020) est la troisième famille couramment citée dans la littérature des
surfaces implicites neuronales ; sans elle la comparaison reste incomplète.

Le protocole est celui de main_step2.py, volontairement :

  - même forme : le globule biconcave, dont la FOSSETTE est le seul détail à
    forte courbure — c'est là que les architectures se départagent ;
  - même perte, MÊMES DONNÉES, même graine, même nombre d'époques ;
  - SANS normales. C'est le cas défavorable, et c'est le cas réaliste : sur des
    données brutes, les normales doivent être estimées et sont bruitées. Avec
    normales, tous les réseaux réussissent et la comparaison ne dit rien.

À la différence de main_step2.py, on ne se contente pas de mean|f_exact| : on
utilise les métriques de src/metrics.py, qui ne dépendent d'aucune formule et
sont donc celles qu'on pourra appliquer aux capsules du BMBI.

`omega0` est balayé dans {10, 15, 30} : c'est le seul réglage propre à SIREN,
il fixe la fréquence spatiale que le réseau manipule naturellement.

Lancer :  /usr/local/bin/python3 main_siren.py
"""
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.shapes import get_surface, biconcave_zfun, sample_biconcave_surface
from src.data import build_training_data_realistic
from src.model import IGRNet, FourierNet
from src.model_siren import SirenNet, nb_poids
from src.train import train_igr
from src.reconstruct import evaluate_grid, extract_surface
from src.metrics import chamfer_distance, iou_from_model, eikonal_error
from src.invariants import euler_characteristic, mesh_volume

EPOCHS = 3000
RES = 96


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def reference():
    """Nuage dense et test dedans/dehors de référence — pour MESURER seulement."""
    _, implicit_exact, box = get_surface("biconcave")
    P_ref = sample_biconcave_surface(n=40000).astype(np.float64)
    dedans = lambda P: np.asarray(implicit_exact(np.asarray(P, dtype=np.float32))) < 0.0
    return P_ref, dedans, implicit_exact, box


def _cache(tag):
    return f"output/cache_{tag}.npz"


def run(model, tag, P_ref, dedans, implicit_exact, box, n_epochs=EPOCHS):
    """
    Entraîne puis mesure — sauf si le résultat est déjà en cache.

    Le cache existe pour une raison pratique : l'ensemble du balayage dépasse
    largement la durée d'un appel unique. Chaque variante est donc mesurée une
    fois puis figée, et relancer le script reprend là où il s'était arrêté au
    lieu de tout refaire.
    """
    chemin = _cache(tag)
    if os.path.exists(chemin):
        d = np.load(chemin, allow_pickle=False)
        r = {k: (float(d[k]) if d[k].ndim == 0 else d[k]) for k in d.files}
        r["tag"] = tag
        r["chi"] = int(r["chi"]); r["poids"] = int(r["poids"])
        r["echec"] = int(r.get("echec", 0))
        r.setdefault("verts", None)
        if r["echec"]:
            print("    -> [cache] ÉCHEC : le champ ne croise pas zéro", flush=True)
        else:
            print(f"    -> [cache] Chamfer={r['chamfer']:.5f}  IoU={r['iou']:.5f}  "
                  f"eikonal={r['eikonal']:.4f}  chi={r['chi']}  "
                  f"mean|f|={r['mfe']:.5f}", flush=True)
        return r

    set_seed(0)                       # mêmes données pour toutes les variantes
    surface_points, _, _ = get_surface("biconcave")
    X_surf, _, X_vol, sign_vol = build_training_data_realistic(
        surface_points, implicit_exact, box, normals=None, n_vol=40000)

    t0 = time.time()
    train_igr(model, X_surf, None, X_vol, sign_vol, box,
              n_epochs=n_epochs, batch_surf=2048, batch_vol=2048, lr=1e-3)
    dt = time.time() - t0

    xs, ys, zs, values = evaluate_grid(model, box, grid_res=RES)

    # Un champ appris peut ne PLUS CROISER ZÉRO sur la grille : marching cubes
    # n'a alors aucune surface à extraire. Ce n'est pas un incident technique,
    # c'est le résultat — la variante a divergé. On le consigne au lieu de faire
    # tomber tout le balayage, sans quoi une seule variante ratée effacerait les
    # quatre autres.
    if not (values.min() < 0.0 < values.max()):
        print(f"    -> ÉCHEC : le champ ne croise pas zéro sur la grille "
              f"(min={values.min():.3e}, max={values.max():.3e}) — "
              f"pas de surface à extraire ({dt:.0f}s)", flush=True)
        r = dict(tag=tag, verts=None, chamfer=np.nan, hausdorff=np.nan,
                 iou=np.nan, eikonal=eikonal_error(model, box, n=50000),
                 chi=0, mfe=np.nan, volume=np.nan, poids=nb_poids(model),
                 duree=dt, echec=1)
        np.savez(chemin, **{k: v for k, v in r.items()
                            if k not in ("tag", "verts")})
        return r

    verts, faces, _, _ = extract_surface(xs, ys, zs, values)

    ch = chamfer_distance(verts, P_ref)
    iou = iou_from_model(model, dedans, box, n_samples=200000)
    eik = eikonal_error(model, box, n=50000)
    topo = euler_characteristic(verts, faces)
    # mean|f_exact| : gardé uniquement pour comparer à main_step2.py
    mfe = float(np.abs(implicit_exact(verts.astype(np.float32))).mean())

    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), f"output/model_{tag}.pt")

    r = dict(tag=tag, verts=verts, chamfer=ch["chamfer"], hausdorff=ch["hausdorff"],
             iou=iou, eikonal=eik, chi=topo["chi"], mfe=mfe,
             volume=mesh_volume(verts, faces), poids=nb_poids(model), duree=dt,
             echec=0)
    np.savez(chemin, **{k: v for k, v in r.items() if k != "tag"})
    print(f"    -> Chamfer={r['chamfer']:.5f}  IoU={r['iou']:.5f}  "
          f"eikonal={r['eikonal']:.4f}  chi={r['chi']}  mean|f|={r['mfe']:.5f}  "
          f"({dt:.0f}s)", flush=True)
    return r


def coupe(ax, verts, titre):
    """Coupe y = 0 : le profil reconstruit contre le vrai profil."""
    m = np.abs(verts[:, 1]) < 0.03
    ax.scatter(verts[m, 0], verts[m, 2], s=3, label="reconstruit")
    xx = np.linspace(-1, 1, 400)
    zz = biconcave_zfun(np.abs(xx))
    ax.plot(xx, zz, "r-", lw=1)
    ax.plot(xx, -zz, "r-", lw=1, label="vrai")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_title(titre, fontsize=9); ax.legend(loc="upper right", fontsize=7)


def main():
    P_ref, dedans, implicit_exact, box = reference()
    print("=" * 104)
    print("COMPARAISON D'ARCHITECTURES SUR LE GLOBULE BICONCAVE, SANS NORMALES")
    print(f"boîte = {box}  |  {EPOCHS} époques  |  grille {RES}^3  |  "
          f"nuage de référence {len(P_ref)} points")
    print("=" * 104)

    res = []
    print("\n--- Softplus (IGRNet) ---", flush=True)
    res.append(run(IGRNet(radius_init=1.0), "cmp_softplus", P_ref, dedans,
                   implicit_exact, box))

    print("\n--- Fourier (FourierNet, 6 fréquences) ---", flush=True)
    res.append(run(FourierNet(n_freqs=6), "cmp_fourier", P_ref, dedans,
                   implicit_exact, box))

    for w in (10.0, 15.0, 30.0):
        print(f"\n--- SIREN omega0 = {w:g} ---", flush=True)
        r = run(SirenNet(omega0=w), f"cmp_siren_w{int(w)}", P_ref, dedans,
                implicit_exact, box)
        r["tag"] = f"SIREN omega0={w:g}"
        res.append(r)
    res[0]["tag"] = "Softplus"
    res[1]["tag"] = "Fourier"

    print("\n" + "=" * 104)
    print(f"{'architecture':<22}{'poids':>9}{'Chamfer':>10}{'Hausdorff':>11}"
          f"{'IoU':>9}{'eikonal':>9}{'mean|f|':>10}{'chi':>6}{'volume':>9}{'durée':>8}")
    print("-" * 104)
    for r in res:
        if r.get("echec"):
            print(f"{r['tag']:<22}{r['poids']:>9d}{'—':>10}{'—':>11}{'—':>9}"
                  f"{r['eikonal']:>9.4f}{'—':>10}{'—':>6}{'—':>9}"
                  f"{r['duree']:>7.0f}s   DIVERGÉ, aucune surface")
            continue
        print(f"{r['tag']:<22}{r['poids']:>9d}{r['chamfer']:>10.5f}"
              f"{r['hausdorff']:>11.5f}{r['iou']:>9.5f}{r['eikonal']:>9.4f}"
              f"{r['mfe']:>10.5f}{r['chi']:>6d}{r['volume']:>9.4f}{r['duree']:>7.0f}s")
    print("-" * 104)

    ok = [r for r in res if not r.get("echec")]
    best_ch = min(ok, key=lambda r: r["chamfer"])
    best_iou = max(ok, key=lambda r: r["iou"])
    best_eik = min(ok, key=lambda r: r["eikonal"])
    print(f"meilleur Chamfer : {best_ch['tag']} ({best_ch['chamfer']:.5f})")
    print(f"meilleure IoU    : {best_iou['tag']} ({best_iou['iou']:.5f})")
    print(f"meilleur eikonal : {best_eik['tag']} ({best_eik['eikonal']:.4f})")
    print("chi = 2 attendu (genre 0). Toute autre valeur signale une "
          "reconstruction topologiquement fausse.")

    fig, axes = plt.subplots(1, len(res), figsize=(3.4 * len(res), 3.6))
    for ax, r in zip(np.atleast_1d(axes), res):
        if r.get("echec") or r.get("verts") is None:
            ax.text(0.5, 0.5, "DIVERGÉ\naucune surface", ha="center",
                    va="center", color="red", fontsize=11,
                    transform=ax.transAxes)
            ax.set_title(r["tag"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        coupe(ax, r["verts"], f"{r['tag']}\nChamfer={r['chamfer']:.4f}  "
                              f"IoU={r['iou']:.4f}")
    fig.suptitle("Fossette du globule biconcave, sans normales — "
                 "Softplus / Fourier / SIREN", fontsize=11)
    fig.tight_layout()
    fig.savefig("output/compare_siren.png", dpi=130)
    plt.close(fig)
    print("\nFigure -> output/compare_siren.png "
          f"({os.path.getsize('output/compare_siren.png')/1024:.1f} Ko)")


if __name__ == "__main__":
    main()
