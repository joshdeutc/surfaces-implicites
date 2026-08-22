"""
Les courbures d'une surface implicite APPRISE par un réseau.

Question posée : un réseau qui reconstruit une forme avec une excellente
précision de position est-il pour autant juste sur ses dérivées SECONDES ?

  - le Chamfer et mean|f| mesurent l'ordre 0 (où est la surface) ;
  - le terme eikonal mesure l'ordre 1 (le gradient) ;
  - la courbure fait intervenir la hessienne, donc l'ordre 2.

Le contrôle employé est Gauss-Bonnet : pour TOUTE surface fermée de genre 0,
∫K_G dA = 4π ≈ 12.566, quelle que soit sa forme. Aucune vérité de référence
n'est donc nécessaire — le théorème suffit.

Trois mesures sont produites :
  1. contrôle : réseau entraîné sur une SPHÈRE (courbure constante, cas le plus
     facile qui soit) — donne le plancher d'erreur de la méthode ;
  2. la surface analytique du globule biconcave, maillée puis mesurée par
     différences finies — dit si la difficulté vient de la FORME ;
  3. le globule appris par le réseau — le résultat cherché.

Les outils sont validés indépendamment par validate_geometry.py (28/28).

Lancer :  /usr/local/bin/python3 main_curvature.py
"""
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.shapes import get_surface, get_surface_normals
from src.data import build_training_data_realistic
from src.model import IGRNet
from src.train import train_igr
from src.reconstruct import extract_surface
from src.device import get_device
from src.curvature import curvature_numpy, curvature_autograd
from src.invariants import (euler_characteristic, is_watertight, mesh_area,
                            mesh_volume, gauss_bonnet, triangle_areas)

GRID_RES = 128
QUATRE_PI = 4.0 * np.pi


def fmt(v):
    """Format compact : notation scientifique dès que la valeur déraille."""
    return f"{v:+.4f}" if abs(v) < 1e4 else f"{v:+.3e}"


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------
# Évaluation de grille par paquets
# ------------------------------------------------------------------

def evaluate_grid_chunked(model, box, grid_res=128, chunk=65536):
    """
    Comme src.reconstruct.evaluate_grid, mais par paquets.

    À 128³ la grille fait 2,1 million de points ; les évaluer d'un bloc dans un
    réseau à 256 neurones demanderait plusieurs gigaoctets d'activations.
    """
    device = get_device()
    model = model.to(device).eval()
    lin = np.linspace(-box, box, grid_res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

    out = np.empty(len(pts), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(pts), chunk):
            block = torch.tensor(pts[s:s + chunk], device=device)
            out[s:s + chunk] = model(block).cpu().numpy().ravel()
    return lin, lin, lin, out.reshape(grid_res, grid_res, grid_res)


def grid_values_numpy(f, box, grid_res=128, chunk=200000):
    """Même chose pour une fonction implicite numpy."""
    lin = np.linspace(-box, box, grid_res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    out = np.empty(len(pts), dtype=np.float64)
    for s in range(0, len(pts), chunk):
        out[s:s + chunk] = np.asarray(f(pts[s:s + chunk])).ravel()
    return lin, lin, lin, out.reshape(grid_res, grid_res, grid_res)


# ------------------------------------------------------------------
# Entraînement (avec cache : l'expérience se relance en quelques secondes)
# ------------------------------------------------------------------

def train_or_load(surface_type, tag, n_epochs, n_vol, batch):
    """Entraîne le réseau IGR sur la forme, ou recharge le cache."""
    path = f"output/model_curv_{tag}.pt"
    os.makedirs("output", exist_ok=True)

    set_seed(0)
    surface_points, implicit_exact, box = get_surface(surface_type)
    normals = get_surface_normals(surface_type, surface_points)
    model = IGRNet(radius_init=1.0)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"  modèle rechargé depuis {path}")
        return model, surface_points, implicit_exact, box

    X_surf, normals_t, X_vol, sign_vol = build_training_data_realistic(
        surface_points, implicit_exact, box, normals=normals, n_vol=n_vol
    )
    print(f"  entraînement sur '{surface_type}' "
          f"({len(X_surf)} points de surface, {n_epochs} époques)...")
    t0 = time.time()
    train_igr(model, X_surf, normals_t, X_vol, sign_vol, box,
              n_epochs=n_epochs, batch_surf=batch, batch_vol=batch, lr=1e-3)
    print(f"  entraîné en {time.time() - t0:.0f} s")
    torch.save(model.state_dict(), path)
    return model, surface_points, implicit_exact, box


# ------------------------------------------------------------------
# Mesure
# ------------------------------------------------------------------

def bilan_maillage(verts, faces, K_gauss, titre):
    """Affiche topologie, métrique et Gauss-Bonnet d'un maillage."""
    topo = euler_characteristic(verts, faces)
    gb = gauss_bonnet(verts, faces, K_gauss)
    ecart_pc = 100.0 * (gb["integral"] - QUATRE_PI) / QUATRE_PI

    print(f"\n  --- {titre} ---")
    print(f"  maillage    : {len(verts)} sommets, {len(faces)} triangles, "
          f"étanche={is_watertight(verts, faces)}")
    print(f"  topologie   : chi={topo['chi']}  genre={topo['genus']:.1f}")
    print(f"  métrique    : aire={mesh_area(verts, faces):.4f}  "
          f"volume={mesh_volume(verts, faces):.4f}")
    print(f"  courbure    : K_gauss dans [{fmt(K_gauss.min())}, {fmt(K_gauss.max())}]  "
          f"médiane={np.median(K_gauss):+.4f}")
    print(f"  GAUSS-BONNET: ∫K dA = {fmt(gb['integral'])}   "
          f"attendu 4π = {QUATRE_PI:.4f}   ÉCART = {fmt(ecart_pc)} %")

    # D'où vient l'intégrale ? Si 1 % des sommets en portent l'essentiel,
    # c'est que le champ a des pics de courbure aberrants.
    aires = triangle_areas(verts, faces)
    contrib = K_gauss[faces].mean(axis=1) * aires
    ordre = np.argsort(-np.abs(contrib))
    top1 = max(1, len(contrib) // 100)
    part = 100.0 * np.abs(contrib[ordre[:top1]]).sum() / np.abs(contrib).sum()
    print(f"  diagnostic  : les 1 % de triangles les plus courbés portent "
          f"{part:.1f} % de ∫|K| dA")
    return {"chi": topo["chi"], "integral": gb["integral"], "ecart_pc": ecart_pc,
            "part_top1": part}


def maillage_du_reseau(model, box, res=GRID_RES):
    xs, ys, zs, vals = evaluate_grid_chunked(model, box, res)
    return extract_surface(xs, ys, zs, vals)


# ------------------------------------------------------------------
# Vérité analytique LOCALE pour le globule
# ------------------------------------------------------------------

def biconcave_curvature_analytic(r, h=1e-5):
    """
    Courbures exactes de la face supérieure du globule, vue comme une surface
    de RÉVOLUTION z = h(r) — sans passer par la fonction implicite définie par
    morceaux de shapes.py.

        κ_méridien = h'' / (1+h'²)^{3/2}
        κ_parallèle = h' / (r √(1+h'²))
        K_G = κ_m · κ_p        K_M = (κ_m + κ_p)/2

    Avec la normale dirigée vers le haut (donc sortante sur la face
    supérieure), ces formules suivent la MÊME convention de signe que
    src/curvature.py : une calotte sphérique convexe y donne K_M = −1/R.

    Contrôle analytique au centre du creux : près de r=0,
    z ≈ a0 + (a1 − a0/2) r², donc h' = 2c·r et h'' = 2c avec c = a1 − a0/2.
    D'où K_G(0) = 4c² et K_M(0) = 2c, valeurs exactes indépendantes de tout
    calcul numérique.
    """
    from src.shapes import biconcave_zfun

    r = np.asarray(r, dtype=np.float64)
    z_p = biconcave_zfun(r + h)
    z_m = biconcave_zfun(r - h)
    z_0 = biconcave_zfun(r)
    d1 = (z_p - z_m) / (2 * h)
    d2 = (z_p - 2 * z_0 + z_m) / (h * h)

    denom = 1.0 + d1 * d1
    kappa_m = d2 / denom ** 1.5
    kappa_p = d1 / (np.maximum(r, 1e-9) * np.sqrt(denom))
    # Sur l'axe, κ_parallèle est une forme 0/0 : sa limite est κ_méridien.
    kappa_p = np.where(r < 1e-6, kappa_m, kappa_p)
    return {"K_gauss": kappa_m * kappa_p, "K_mean": 0.5 * (kappa_m + kappa_p)}


def validation_locale_globule(verts, curv, r_max=0.9, z_min=0.05):
    """
    Compare, point par point, la courbure du RÉSEAU à la formule exacte de la
    surface de révolution, sur la face supérieure et loin du bord.

    Gauss-Bonnet est un contrôle GLOBAL : une intégrale peut tomber juste avec
    des courbures localement fausses qui se compensent. Cette comparaison-ci
    est le contrôle LOCAL qui manque.
    """
    r = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
    masque = (verts[:, 2] > z_min) & (r < r_max)
    if masque.sum() < 100:
        return None

    exact = biconcave_curvature_analytic(r[masque])
    kg_res, kg_ref = curv["K_gauss"][masque], exact["K_gauss"]
    km_res, km_ref = curv["K_mean"][masque], exact["K_mean"]

    return {
        "n": int(masque.sum()),
        "kg_erreur_abs": float(np.abs(kg_res - kg_ref).mean()),
        "kg_amplitude": float(np.abs(kg_ref).mean()),
        "km_erreur_abs": float(np.abs(km_res - km_ref).mean()),
        "km_amplitude": float(np.abs(km_ref).mean()),
        "correlation": float(np.corrcoef(kg_res, kg_ref)[0, 1]),
    }


# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def figure_coupe(verts, K_gauss, K_mean, implicit_exact, out_path):
    """Coupe y=0 du globule, colorée par la courbure."""
    masque = np.abs(verts[:, 1]) < 0.03
    x, z = verts[masque, 0], verts[masque, 2]
    kg, km = K_gauss[masque], K_mean[masque]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, champ, nom in ((axes[0], kg, "courbure gaussienne $K_G$"),
                           (axes[1], km, "courbure moyenne $K_M$")):
        # Échelle robuste : les pics isolés écraseraient la palette.
        v = np.quantile(np.abs(champ), 0.98)
        s = ax.scatter(x, z, c=champ, s=6, cmap="coolwarm", vmin=-v, vmax=v)
        fig.colorbar(s, ax=ax, label=nom)
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_title(f"{nom}\n(échelle coupée au quantile 98 %)", fontsize=10)

    fig.suptitle("Globule rouge appris — courbures du champ neuronal, coupe y=0")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n  figure -> {out_path}")


# ------------------------------------------------------------------

def main():
    os.makedirs("output", exist_ok=True)
    print("=" * 78)
    print("COURBURES D'UN CHAMP IMPLICITE APPRIS — contrôle par Gauss-Bonnet")
    print("=" * 78)
    print(f"Pour toute surface fermée de genre 0 : ∫K_G dA = 4π = {QUATRE_PI:.4f}")
    print("Ce test ne demande aucune vérité de référence, seulement le théorème.")

    bilans = {}

    # --- 1. Contrôle : réseau entraîné sur une sphère -------------------
    print("\n" + "=" * 78)
    print("1. CONTRÔLE — réseau entraîné sur une SPHÈRE")
    print("=" * 78)
    print("  La sphère a une courbure constante : c'est le cas le plus facile.")
    print("  L'écart obtenu ici est le PLANCHER d'erreur de la méthode.")
    m_sph, _, _, box_sph = train_or_load("sphere", "sphere", 2000, 20000, 1024)
    v_sph, f_sph, _, _ = maillage_du_reseau(m_sph, box_sph)
    c_sph = curvature_autograd(m_sph, v_sph)
    print(f"  device autograd : {c_sph['device']}")
    bilans["sphère apprise"] = bilan_maillage(v_sph, f_sph, c_sph["K_gauss"],
                                              "sphère apprise par le réseau")
    # Contrôle LOCAL : sur une sphère de rayon 1, K_G doit valoir 1 partout.
    err_kg = np.abs(c_sph["K_gauss"] - 1.0)
    err_km = np.abs(np.abs(c_sph["K_mean"]) - 1.0)
    print(f"  contrôle local (K_G doit valoir 1 en tout point) : "
          f"erreur moyenne={err_kg.mean():.4f}  médiane={np.median(err_kg):.4f}  "
          f"q95={np.quantile(err_kg, 0.95):.4f}")
    print(f"  contrôle local (|K_M| doit valoir 1)             : "
          f"erreur moyenne={err_km.mean():.4f}  médiane={np.median(err_km):.4f}")

    # --- 2. Référence : la surface ANALYTIQUE du globule ----------------
    print("\n" + "=" * 78)
    print("2. RÉFÉRENCE — surface analytique du globule (sans réseau)")
    print("=" * 78)
    print("  Même mesure, sur la formule exacte maillée. Dit si la difficulté")
    print("  vient de la FORME (creux central, bord fin) ou du réseau.")
    _, implicit_bicon, box_bic = get_surface("biconcave")
    xa, ya, za, va = grid_values_numpy(implicit_bicon, box_bic, GRID_RES)
    v_ana, f_ana, _, _ = extract_surface(xa, ya, za, va)
    c_ana = curvature_numpy(implicit_bicon, v_ana)
    bilans["globule analytique"] = bilan_maillage(v_ana, f_ana, c_ana["K_gauss"],
                                                  "globule analytique (diff. finies)")
    # Où le calcul explose-t-il ? La réponse attendue est : au bord r=1, là où
    # implicit_biconcave change de branche (np.where(r < 1, ...)).
    r_ana = np.sqrt(v_ana[:, 0] ** 2 + v_ana[:, 1] ** 2)
    aberrants = np.abs(c_ana["K_gauss"]) > 100.0
    if aberrants.any():
        print(f"  localisation des aberrations : {aberrants.sum()} sommets sur "
              f"{len(v_ana)} ont |K_G| > 100")
        print(f"    leur rayon r est dans [{r_ana[aberrants].min():.4f}, "
              f"{r_ana[aberrants].max():.4f}]  (médiane {np.median(r_ana[aberrants]):.4f})")
        print(f"    à comparer au reste du maillage : r dans "
              f"[{r_ana[~aberrants].min():.4f}, {r_ana[~aberrants].max():.4f}]")
    print("  DIAGNOSTIC : implicit_biconcave est définie par morceaux")
    print("  (np.where(r < 1, ...)), donc son gradient est discontinu en r=1.")
    print("  Ses dérivées SECONDES y divergent : la formule analytique du dépôt")
    print("  est inutilisable pour de l'analyse différentielle.")

    # --- 2bis. La même forme, mais par une formule LISSE ----------------
    print("\n" + "=" * 78)
    print("2 bis. CONTRE-ÉPREUVE — le globule par sa QUARTIQUE (Kuchel)")
    print("=" * 78)
    print("  Même objet géométrique, mais décrit par un polynôme de degré 4")
    print("  lisse partout, au lieu d'une formule définie par morceaux.")
    print("  Si le diagnostic ci-dessus est juste, l'intégrale doit redevenir 4π.")
    try:
        from src.shapes import implicit_rbc_unit
        xq, yq, zq, vq = grid_values_numpy(implicit_rbc_unit, 1.3, GRID_RES)
        v_q, f_q, _, _ = extract_surface(xq, yq, zq, vq)
        c_q = curvature_numpy(implicit_rbc_unit, v_q)
        bilans["globule quartique"] = bilan_maillage(
            v_q, f_q, c_q["K_gauss"], "globule quartique (analytique, lisse)")
        print("  NB : la quartique n'est PAS une distance signée (‖∇F‖ ≠ 1).")
        print("  Que le résultat soit juste quand même vérifie au passage que les")
        print("  formules de courbure sont bien invariantes par reparamétrage de f.")
    except ImportError:
        print("  (implicit_rbc_unit pas encore disponible — cas ignoré)")

    # --- 3. Le résultat : le globule appris ----------------------------
    print("\n" + "=" * 78)
    print("3. RÉSULTAT — globule rouge appris par le réseau")
    print("=" * 78)
    m_bic, pts_bic, implicit_bic, box_b = train_or_load("biconcave", "biconcave",
                                                        2500, 40000, 2048)
    v_bic, f_bic, _, _ = maillage_du_reseau(m_bic, box_b)
    err_pos = float(np.abs(implicit_bic(v_bic.astype(np.float32))).mean())
    print(f"  précision de POSITION : mean|f_exact| = {err_pos:.4f} "
          f"(maille de grille {2*box_b/GRID_RES:.4f})")
    c_bic = curvature_autograd(m_bic, v_bic)
    print(f"  device autograd : {c_bic['device']}")
    bilans["globule appris"] = bilan_maillage(v_bic, f_bic, c_bic["K_gauss"],
                                              "globule appris par le réseau")

    # Contrôle LOCAL contre la formule exacte de la surface de révolution.
    print("\n  Contrôle LOCAL — courbure du réseau contre la formule exacte")
    print("  (face supérieure, r < 0.9, hors du bord) :")
    loc = validation_locale_globule(v_bic, c_bic)
    if loc:
        print(f"    {loc['n']} sommets comparés")
        print(f"    K_gauss : erreur absolue moyenne = {loc['kg_erreur_abs']:.4f} "
              f"pour une amplitude de {loc['kg_amplitude']:.4f}  "
              f"({100*loc['kg_erreur_abs']/loc['kg_amplitude']:.1f} %)")
        print(f"    K_moyenne : erreur absolue moyenne = {loc['km_erreur_abs']:.4f} "
              f"pour une amplitude de {loc['km_amplitude']:.4f}  "
              f"({100*loc['km_erreur_abs']/loc['km_amplitude']:.1f} %)")
        print(f"    corrélation K_gauss réseau / exact = {loc['correlation']:.4f}")

    # Valeur exacte au centre du creux, calculée à la main (aucun numérique) :
    # z ≈ a0 + (a1 − a0/2) r² près de r=0, donc K_G(0) = 4(a1 − a0/2)².
    from src.shapes import _BC
    c_axe = _BC["a1"] - _BC["a0"] / 2.0
    kg_axe_exact, km_axe_exact = 4.0 * c_axe ** 2, 2.0 * c_axe
    r_bic = np.sqrt(v_bic[:, 0] ** 2 + v_bic[:, 1] ** 2)
    proche_axe = (r_bic < 0.08) & (v_bic[:, 2] > 0.05)
    if proche_axe.any():
        print(f"\n    Au centre du creux (r < 0.08, {proche_axe.sum()} sommets) :")
        print(f"      K_gauss   réseau = {c_bic['K_gauss'][proche_axe].mean():+.3f}"
              f"   exact = {kg_axe_exact:+.3f}")
        print(f"      K_moyenne réseau = {c_bic['K_mean'][proche_axe].mean():+.3f}"
              f"   exact = {km_axe_exact:+.3f}")

    figure_coupe(v_bic, c_bic["K_gauss"], c_bic["K_mean"], implicit_bic,
                 "output/courbure_globule.png")

    # --- Synthèse ------------------------------------------------------
    print("\n" + "=" * 78)
    print("SYNTHÈSE — ∫K_G dA comparé à 4π")
    print("=" * 78)
    print(f"  {'cas':<26s} {'chi':>4s} {'∫K dA':>14s} {'écart':>14s} {'top 1%':>8s}")
    for nom, b in bilans.items():
        print(f"  {nom:<26s} {b['chi']:>4d} {fmt(b['integral']):>14s} "
              f"{fmt(b['ecart_pc'])+' %':>14s} {b['part_top1']:>7.1f}%")
    print(f"  {'valeur théorique':<26s} {2:>4d} {fmt(QUATRE_PI):>14s} "
          f"{'0.00 %':>14s}")
    print("=" * 78)
    print("""
LECTURE

  Gauss-Bonnet est un contrôle NÉCESSAIRE, pas suffisant : c'est une intégrale,
  donc des erreurs locales de signes opposés pourraient s'y compenser. C'est
  pourquoi il est doublé ici d'un contrôle LOCAL point par point (sphère : K_G
  doit valoir 1 partout ; globule : comparaison à la formule de la surface de
  révolution). Les deux doivent passer pour conclure.

  Le cas "globule analytique" n'est PAS un échec de la méthode : c'est un
  défaut de la formule implicite du dépôt, définie par morceaux, dont les
  dérivées secondes divergent au bord. C'est l'argument le plus direct pour
  lui substituer une forme algébrique lisse partout.""")


if __name__ == "__main__":
    main()
