"""
Une surface implicite qui évolue dans le temps, apprise depuis six instants.

C'est le « Graal » du mail de recommandations et la formulation même de la fiche
d'enregistrement : « condenser la forme mobile en une surface implicite
espace-temps ». Le protocole imite ce que fournira le code Éléments Finis du
BMBI — non pas une fonction du temps, mais une SUITE DE FICHIERS.

    GÉNÉRATION                    |  EXPLOITATION
    la quartique de Kuchel est    |  on relit six fichiers .npz et on
    étirée à volume constant,     |  n'y voit plus que des points et des
    maillée, et six instants      |  normales. Un seul réseau f(x,y,z,t)
    sont écrits sur disque        |  est appris sur les six.

Quatre mesures sont affichées :

  1. fidélité aux six instants OBSERVÉS ;
  2. INTERPOLATION aux cinq instants JAMAIS VUS — le test central : il décide
     si le réseau a condensé la séquence ou s'il l'a mémorisée ;
  3. conservation du VOLUME, invariant que l'entraînement ignore ;
  4. SUIVI LAGRANGIEN d'un point de membrane par la vitesse normale.

Lancer :  python main_temporal.py
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import temporal as T
from src.model_temporal import TemporalIGRNet, figer_instant, n_parametres
from src.train_temporal import (derivees, en_tenseurs, jeu_depuis_fichiers,
                                train_temporel)
from src.reconstruct import evaluate_grid, extract_surface
from src.metrics import chamfer_distance, iou_occupancy, model_inside
from src.invariants import euler_characteristic, mesh_volume
from src.device import get_device

OUT = "output"
DOSSIER = os.path.join(OUT, "temporel")
GRILLE = 96                 # résolution de la grille de marching cubes


def titre(n, texte):
    print("\n" + "=" * 78)
    print(f"BLOC {n} — {texte}")
    print("=" * 78)


class Compteur:
    """Enveloppe une fonction et compte ses appels. Sert de preuve, pas de calcul."""

    def __init__(self, fonction):
        self.fonction = fonction
        self.appels = 0

    def __call__(self, *a, **kw):
        self.appels += 1
        return self.fonction(*a, **kw)


# ============================================================
# Préparation : écrire les six instants, puis les relire
# ============================================================

def preparer(n_points=6000):
    print("=" * 78)
    print("SURFACE IMPLICITE ESPACE-TEMPS — APPRISE DEPUIS SIX INSTANTS DISCRETS")
    print("=" * 78)
    print("\nDéformation : étirement selon z à VOLUME CONSTANT,")
    print("    A(t) = diag(xi^-1/2, xi^-1/2, xi),  xi(t) = 1 + t/2,  det A = 1")
    print("appliqué à la quartique de Kuchel (globule rouge), bord ramené à r = 1.")
    print("\nÉcriture des fichiers d'échange (points + normales de maillage) :\n")

    seq = T.ecrire_sequence(DOSSIER, n_points=n_points)

    print(f"  {'t':>5} {'xi':>7} {'fichier':>22} {'points':>7} "
          f"{'V maillage':>11} {'chi':>5} {'rayon max':>10} {'|z| max':>8}")
    print("  " + "-" * 80)
    for t, chemin, info in seq:
        V, F, _, _, _ = T.maillage_instant(t)
        e = euler_characteristic(V, F)
        r = float(np.hypot(V[:, 0], V[:, 1]).max())
        print(f"  {t:>5.2f} {info['xi']:>7.3f} {os.path.basename(chemin):>22} "
              f"{info['n_points']:>7} {mesh_volume(V, F):>11.6f} {e['chi']:>5} "
              f"{r:>10.4f} {float(np.abs(V[:, 2]).max()):>8.4f}")

    print(f"\n  volume exact de la quartique (quadrature) : {T.volume_exact():.6f}")
    print(f"  volume des maillages, à tout instant      : {mesh_volume(*T.maillage_instant(0.0)[:2]):.6f}"
          f"   (det A = 1, donc constant par construction)")
    print("\n  Les six nuages sont tirés avec des graines DIFFÉRENTES : aucune")
    print("  correspondance point à point d'un instant à l'autre.")
    return seq


def apprendre(seq, n_epochs, n_vol, k, seed, device):
    print("\n" + "-" * 78)
    print("APPRENTISSAGE — un seul champ f(x,y,z,t) pour les six instants")
    print("-" * 78)

    # Preuve, et non déclaration : on met la formule sous compteur AVANT même la
    # préparation des données. Tout ce qui suit — signe dedans/dehors compris —
    # doit sortir des seuls fichiers.
    compteur = Compteur(T.implicit_rbc_unit)
    vraie_f, T.implicit_rbc_unit = T.implicit_rbc_unit, compteur

    instants = [(t, chemin) for t, chemin, _ in seq]
    jeu = jeu_depuis_fichiers(instants, k=k, n_vol_par_instant=n_vol, seed=seed)
    appels_donnees = compteur.appels

    print(f"\n  {'t':>5} {'fichier relu':>22} {'surface':>8} {'dedans':>8} "
          f"{'dehors':>8}   provenance déclarée")
    print("  " + "-" * 90)
    for d in jeu["detail"]:
        print(f"  {d['t']:>5.2f} {os.path.basename(d['fichier']):>22} "
              f"{d['n_surface']:>8} {d['dedans']:>8} {d['dehors']:>8}   {d['source']}")

    print(f"\n  points de surface, tous instants confondus : {len(jeu['X_surf'])}")
    print(f"  points de volume,  tous instants confondus : {len(jeu['X_vol'])}")
    print(f"  signe obtenu par produit scalaire, vote sur k = {k} voisins")
    print(f"  boîte d'étude : ±{jeu['box']}   temps : [{jeu['t_min']}, {jeu['t_max']}]")

    tenseurs = en_tenseurs(jeu)
    torch.manual_seed(seed)
    np.random.seed(seed)
    modele = TemporalIGRNet(t_min=jeu["t_min"], t_max=jeu["t_max"])
    print(f"\n  réseau : entrées (x,y,z,t), Softplus, init géométrique, "
          f"{n_parametres(modele)} poids")

    # vérification de l'initialisation : le champ de départ ignore le temps
    with torch.no_grad():
        q = (torch.rand(4000, 3) * 2 - 1) * jeu["box"]
        ref = modele(torch.cat([q, torch.zeros(4000, 1)], 1))
        pire = max(float((modele(torch.cat([q, torch.full((4000, 1), t)], 1))
                          - ref).abs().max())
                   for t in (0.25, 0.5, 0.75, 1.0))
    print(f"  à l'initialisation, max |f(x,t) − f(x,0)| = {pire:.1e}  "
          f"(le champ de départ ignore le temps)")

    print(f"\n  {n_epochs} époques, eikonal tiré à t CONTINU dans "
          f"[{jeu['t_min']}, {jeu['t_max']}]\n")
    t0 = time.time()
    modele, hist = train_temporel(
        modele, tenseurs["X_surf"], tenseurs["N_surf"],
        tenseurs["X_vol"], tenseurs["s_vol"], jeu["box"],
        t_min=jeu["t_min"], t_max=jeu["t_max"],
        n_epochs=n_epochs, device=device, seed=seed)
    duree = time.time() - t0

    appels_entrainement = compteur.appels - appels_donnees
    T.implicit_rbc_unit = vraie_f
    print(f"\n  durée : {duree:.0f} s sur {device}")
    print(f"  appels à la formule exacte pendant la préparation des données : "
          f"{appels_donnees}   (attendu 0)")
    print(f"  appels à la formule exacte pendant l'entraînement            : "
          f"{appels_entrainement}   (attendu 0)")
    return modele, jeu, hist, duree


# ============================================================
# Reconstruction et mesures à un instant donné
# ============================================================

def reconstruire(modele, t, box, grille=GRILLE, device=None):
    """Marching cubes sur le champ appris figé à l'instant t."""
    vue = figer_instant(modele, t, device=device)
    xs, ys, zs, valeurs = evaluate_grid(vue, box, grid_res=grille)
    sommets, faces, _, _ = extract_surface(xs, ys, zs, valeurs)
    return sommets, faces, vue


def mesurer(modele, t, box, grille=GRILLE, device=None, n_iou=100000):
    """Chamfer, IoU, volume et topologie de la reconstruction à l'instant t."""
    sommets, faces, vue = reconstruire(modele, t, box, grille, device)
    if len(sommets) == 0:
        return {"t": t, "chamfer": np.nan, "iou": 0.0, "volume": 0.0,
                "chi": None, "n_sommets": 0}
    ch = chamfer_distance(sommets, T.surface_dense(t))
    iou = iou_occupancy(model_inside(vue, device=device), T.dedans_exact(t),
                        box, n_samples=n_iou, seed=7)
    e = euler_characteristic(sommets, faces)
    return {"t": t, "chamfer": ch["chamfer"], "hausdorff": ch["hausdorff"],
            "iou": iou, "volume": mesh_volume(sommets, faces),
            "chi": e["chi"], "n_sommets": len(sommets)}


def volume_plancher(t, box, grille=GRILLE):
    """
    Volume que donne le marching cubes appliqué au champ EXACT, même grille.

    C'est le plancher de discrétisation : en dessous, aucun réseau ne peut
    faire mieux. Sans ce repère on attribuerait au réseau une erreur qui
    n'appartient qu'à la grille.
    """
    axe = np.linspace(-box, box, grille)
    X, Y, Z = np.meshgrid(axe, axe, axe, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    vals = T.f_exact(pts, t).reshape(grille, grille, grille)
    from skimage import measure
    pas = axe[1] - axe[0]
    v, f, _, _ = measure.marching_cubes(vals, level=0.0, spacing=(pas, pas, pas))
    return mesh_volume(v + axe[0], f)


# ============================================================
# BLOCS 1 et 2 — instants observés puis instants jamais vus
# ============================================================

def blocs12(modele, box, device):
    titre(1, "fidélité aux six instants OBSERVÉS")
    print("\n  Ce sont les instants dont les fichiers ont servi à l'entraînement.")
    print("  Chamfer en unités d'espace (le bord du globule est à r = 1), IoU par")
    print(f"  Monte-Carlo sur 100 000 points. Grille de reconstruction {GRILLE}³,")
    print(f"  soit un pas de {2 * box / (GRILLE - 1):.4f}.\n")

    vus = [mesurer(modele, t, box, device=device) for t in T.T_OBSERVES]
    entete = (f"  {'t':>6} {'Chamfer':>10} {'Hausdorff':>11} {'IoU':>9} "
              f"{'chi':>5} {'sommets':>9}")
    print(entete)
    print("  " + "-" * 55)
    for r in vus:
        print(f"  {r['t']:>6.2f} {r['chamfer']:>10.5f} {r['hausdorff']:>11.5f} "
              f"{r['iou']:>9.5f} {str(r['chi']):>5} {r['n_sommets']:>9}")

    titre(2, "INTERPOLATION — les instants JAMAIS VUS (le test central)")
    print("\n  Le réseau n'a reçu aucun fichier à ces instants. Rien dans la perte")
    print("  ne lui a dit quelle forme y mettre : le seul terme qui parle des")
    print("  instants intermédiaires est l'eikonal, et il n'exige que de rester")
    print("  une distance signée — une contrainte purement géométrique.")
    print("\n  Si la qualité s'effondre ici, le réseau a MÉMORISÉ six photos.")
    print("  Si elle tient, il a CONDENSÉ la séquence.\n")

    caches = [mesurer(modele, t, box, device=device) for t in T.T_CACHES]

    tout = sorted(vus + caches, key=lambda r: r["t"])
    marques = {t: "vu" for t in T.T_OBSERVES}
    marques.update({t: "  JAMAIS VU" for t in T.T_CACHES})

    print(f"  {'t':>6} {'statut':>12} {'Chamfer':>10} {'Hausdorff':>11} "
          f"{'IoU':>9} {'chi':>5} {'sommets':>9}")
    print("  " + "-" * 68)
    for r in tout:
        print(f"  {r['t']:>6.2f} {marques[r['t']]:>12} {r['chamfer']:>10.5f} "
              f"{r['hausdorff']:>11.5f} {r['iou']:>9.5f} {str(r['chi']):>5} "
              f"{r['n_sommets']:>9}")

    cv = np.array([r["chamfer"] for r in vus])
    cc = np.array([r["chamfer"] for r in caches])
    iv = np.array([r["iou"] for r in vus])
    ic = np.array([r["iou"] for r in caches])

    print(f"\n  {'':>14} {'Chamfer moyen':>15} {'IoU moyen':>12}")
    print("  " + "-" * 44)
    print(f"  {'instants vus':>14} {cv.mean():>15.5f} {iv.mean():>12.5f}")
    print(f"  {'jamais vus':>14} {cc.mean():>15.5f} {ic.mean():>12.5f}")
    print(f"  {'rapport':>14} {cc.mean() / cv.mean():>15.3f} "
          f"{ic.mean() / iv.mean():>12.3f}")

    degradation = 100.0 * (cc.mean() / cv.mean() - 1.0)
    print(f"\n  VERDICT — le Chamfer aux instants jamais vus est "
          f"{degradation:+.1f} % par rapport\n  aux instants observés, l'IoU y "
          f"vaut {100.0 * ic.mean() / iv.mean():.1f} % de celui des instants observés.")
    if degradation < 25.0 and ic.mean() > 0.95 * iv.mean():
        print("  Le réseau RESTITUE des instants qu'on ne lui a jamais montrés : il a")
        print("  condensé la séquence en un champ continu, pas mémorisé six photos.")
    elif degradation < 100.0:
        print("  La qualité se dégrade nettement hors des instants observés sans")
        print("  s'effondrer : le champ interpole, mais imparfaitement.")
    else:
        print("  La qualité s'effondre hors des instants observés : le réseau a")
        print("  MÉMORISÉ les six photos au lieu de condenser le mouvement.")
    return vus, caches, tout


# ============================================================
# BLOC 3 — conservation du volume
# ============================================================

def bloc3(tout, box, marques):
    titre(3, "conservation du VOLUME — un invariant que l'entraînement ignore")
    print("\n  La déformation a det A = 1 : le volume est le même à tout instant.")
    print("  Rien dans la perte ne le dit au réseau. C'est donc un contrôle")
    print("  entièrement extérieur à l'apprentissage.")
    print("\n  Le marching cubes commet lui-même une erreur de discrétisation. On")
    print(f"  la mesure en appliquant la MÊME grille {GRILLE}³ au champ EXACT :")
    print("  c'est le plancher, en dessous duquel aucun réseau ne peut aller.\n")

    v_exact = T.volume_exact()
    print(f"  volume exact (quadrature) : {v_exact:.6f}\n")
    print(f"  {'t':>6} {'statut':>12} {'V plancher':>11} {'écart':>9} "
          f"{'V appris':>10} {'écart':>9} {'net du plancher':>16}")
    print("  " + "-" * 78)

    lignes = []
    for r in tout:
        vp = volume_plancher(r["t"], box)
        e_plancher = 100.0 * (vp - v_exact) / v_exact
        e_appris = 100.0 * (r["volume"] - v_exact) / v_exact
        net = 100.0 * (r["volume"] - vp) / vp
        lignes.append((r["t"], vp, e_plancher, r["volume"], e_appris, net))
        print(f"  {r['t']:>6.2f} {marques[r['t']]:>12} {vp:>11.6f} "
              f"{e_plancher:>+8.3f}% {r['volume']:>10.6f} {e_appris:>+8.3f}% "
              f"{net:>+15.3f}%")

    v = np.array([l[3] for l in lignes])
    ecarts = np.array([l[4] for l in lignes])
    nets = np.array([l[5] for l in lignes])
    print(f"\n  volume appris : moyenne {v.mean():.6f}, écart-type {v.std():.6f} "
          f"({100 * v.std() / v.mean():.3f} % de la moyenne)")
    print(f"  écart au volume exact  : de {ecarts.min():+.3f} % à {ecarts.max():+.3f} % "
          f"(moyenne {ecarts.mean():+.3f} %)")
    print(f"  écart net du plancher  : de {nets.min():+.3f} % à {nets.max():+.3f} % "
          f"(moyenne {nets.mean():+.3f} %)")
    print(f"\n  La dispersion du volume appris au fil du temps ({100 * v.std() / v.mean():.3f} %)")
    print("  est ce qui compte : elle mesure une DÉRIVE, alors qu'un biais constant")
    print("  ne serait qu'un décalage de l'isosurface.")
    return lignes


# ============================================================
# BLOC 4 — suivi lagrangien par la vitesse normale
# ============================================================

def vitesse_normale_exacte(P, t):
    """v_n et n du champ EXACT — la référence du même opérateur."""
    g = T.grad_f_exact(P, t)
    dfdt = T.dfdt_exact(P, t)
    norme = np.maximum(np.linalg.norm(g, axis=1), 1e-12)
    return -dfdt / norme, g / norme[:, None]


def vitesse_normale_apprise(modele, P, t, device):
    _, g, dfdt = derivees(modele, P, t, device=device)
    norme = np.maximum(np.linalg.norm(g, axis=1), 1e-12)
    return -dfdt / norme, g / norme[:, None]


def advecter(champ_vitesse, P0, t0, t1, n_pas):
    """
    Intègre ẋ = v_n(x,t)·n(x,t) par Runge-Kutta 4.

    Renvoie (temps, trajectoires) de formes (n_pas+1,) et (n_pas+1, N, 3).
    """
    dt = (t1 - t0) / n_pas
    P = np.asarray(P0, dtype=np.float64).reshape(-1, 3).copy()
    temps = [t0]
    traj = [P.copy()]

    def champ(X, t):
        v, n = champ_vitesse(X, t)
        return v[:, None] * n

    for i in range(n_pas):
        t = t0 + i * dt
        k1 = champ(P, t)
        k2 = champ(P + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = champ(P + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = champ(P + dt * k3, t + dt)
        P = P + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        temps.append(t + dt)
        traj.append(P.copy())
    return np.array(temps), np.array(traj)


def distance_a_la_surface(P, t):
    """
    Distance approchée du point à la vraie surface : |f| / ‖∇f‖.

    Estimation au premier ordre, largement suffisante près de la surface, et
    qui a l'avantage de se lire dans les mêmes unités que le Chamfer.
    """
    f = T.f_exact(P, t)
    g = np.linalg.norm(T.grad_f_exact(P, t), axis=1)
    return np.abs(f) / np.maximum(g, 1e-12)


def bloc4(modele, box, device, n_pas=200):
    titre(4, "SUIVI LAGRANGIEN — advecter un point de membrane")
    print("\n  Sur la surface, f(x(t), t) = 0 à tout instant. En dérivant le long")
    print("  du mouvement :   ∇_x f · ẋ + ∂f/∂t = 0.")
    print("  Un point qui suit la normale, ẋ = v_n·n, avance donc à la vitesse")
    print("\n      v_n = − (∂f/∂t) / ‖∇_x f‖")
    print("\n  et cette quantité se lit DIRECTEMENT sur le réseau : ∂f/∂t est sa")
    print("  quatrième dérivée partielle. C'est ce que la représentation")
    print("  espace-temps apporte et qu'une suite de réseaux statiques ne donne pas.")

    # trois points de départ caractéristiques, dans le plan méridien y = 0
    from src.shapes import _rbc_z_half, rbc_coeffs, rbc_scale, rbc_radius_max
    Pc, Qc, Rc = rbc_coeffs()
    L = rbc_scale()
    r_bord = rbc_radius_max() / L

    def z_surface(r):
        return float(np.sqrt(_rbc_z_half((r * L) ** 2, Pc, Qc, Rc)) / L)

    depart = np.array([
        [0.0, 0.0, z_surface(0.0)],              # fond de la fossette
        [0.55, 0.0, z_surface(0.55)],            # flanc
        [r_bord * 0.999, 0.0, z_surface(r_bord * 0.999)],   # bord
    ])
    noms = ["fossette (r=0)", "flanc (r=0.55)", "bord (r≈1)"]

    print(f"\n  Trois points de départ sur la membrane à t = 0, plan méridien y = 0 :")
    for nom, p in zip(noms, depart):
        print(f"    {nom:<18} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})   "
              f"distance à la surface {distance_a_la_surface(p[None, :], 0.0)[0]:.2e}")

    print(f"\n  Intégration RK4, {n_pas} pas de t=0 à t=1, sur le champ APPRIS et")
    print("  sur le champ EXACT — même équation, même schéma, seul le champ change.\n")

    t0 = time.time()
    temps, tr_appris = advecter(
        lambda X, t: vitesse_normale_apprise(modele, X, t, device),
        depart, 0.0, 1.0, n_pas)
    _, tr_exact = advecter(vitesse_normale_exacte, depart, 0.0, 1.0, n_pas)
    print(f"  ({time.time() - t0:.0f} s d'intégration)")

    # trajectoire MATÉRIELLE : le point de matière transporté par A(t)
    tr_materiel = np.array([T.deformer(depart, t) for t in temps])

    ecart = np.linalg.norm(tr_appris - tr_exact, axis=2)          # (n_pas+1, 3)
    d_appris = np.array([distance_a_la_surface(tr_appris[i], temps[i])
                         for i in range(len(temps))])
    d_exact = np.array([distance_a_la_surface(tr_exact[i], temps[i])
                        for i in range(len(temps))])

    print(f"\n  {'point':<18} {'écart moyen':>12} {'écart final':>12} "
          f"{'dérive appris':>14} {'dérive exact':>13}")
    print("  " + "-" * 74)
    for j, nom in enumerate(noms):
        print(f"  {nom:<18} {ecart[:, j].mean():>12.5f} {ecart[-1, j]:>12.5f} "
              f"{d_appris[:, j].mean():>14.5f} {d_exact[:, j].mean():>13.5f}")
    print("  " + "-" * 74)
    print(f"  {'ensemble':<18} {ecart.mean():>12.5f} {ecart[-1].mean():>12.5f} "
          f"{d_appris.mean():>14.5f} {d_exact.mean():>13.5f}")

    print("\n  « écart » = distance entre la trajectoire issue du champ appris et")
    print("  celle issue du champ exact. « dérive » = distance du point advecté à")
    print("  la vraie membrane : elle dit si le point RESTE sur la surface.")
    print("  La dérive du champ exact est le plancher du schéma d'intégration.")

    ecart_materiel = np.linalg.norm(tr_exact - tr_materiel, axis=2)
    print(f"\n  Repère utile : la trajectoire MATÉRIELLE (le point de matière")
    print(f"  transporté par A(t)) s'écarte de la trajectoire normale exacte de")
    print(f"  {ecart_materiel[-1].mean():.5f} en moyenne à t = 1. Les deux sont")
    print("  justes : un point de matière glisse aussi TANGENTIELLEMENT, et ce")
    print("  glissement est invisible pour une surface de niveau. Une surface")
    print("  implicite donne le mouvement de la SURFACE, pas celui de la matière.")

    figure_trajectoires(temps, tr_appris, tr_exact, tr_materiel, noms, ecart)
    return {"temps": temps, "ecart": ecart, "d_appris": d_appris,
            "d_exact": d_exact, "ecart_materiel": ecart_materiel}


def meridien(t, n=600):
    """Demi-méridien (r, z) de la vraie surface à l'instant t, tracé propre."""
    from src.shapes import _rbc_z_half, rbc_coeffs, rbc_scale, rbc_radius_max
    Pc, Qc, Rc = rbc_coeffs()
    L = rbc_scale()
    r = np.linspace(0.0, rbc_radius_max() / L, n)
    z = np.sqrt(_rbc_z_half((r * L) ** 2, Pc, Qc, Rc)) / L
    a = T.diag_A(t)
    return r * a[0], z * a[2]


def figure_trajectoires(temps, tr_appris, tr_exact, tr_materiel, noms, ecart):
    """Trajectoires dans le demi-plan méridien + écart en fonction du temps."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    for t, gris in ((0.0, "0.78"), (0.5, "0.58"), (1.0, "0.35")):
        r, z = meridien(t)
        ax.plot(r, z, "-", lw=1.1, color=gris, label=f"membrane t={t:g}")
    couleurs = ["#c0392b", "#2980b9", "#27ae60"]
    for j, nom in enumerate(noms):
        r_ex = np.hypot(tr_exact[:, j, 0], tr_exact[:, j, 1])
        r_ap = np.hypot(tr_appris[:, j, 0], tr_appris[:, j, 1])
        r_ma = np.hypot(tr_materiel[:, j, 0], tr_materiel[:, j, 1])
        ax.plot(r_ex, tr_exact[:, j, 2], "-", lw=3.2, color=couleurs[j],
                alpha=0.35, solid_capstyle="round", label=f"exact — {nom}")
        ax.plot(r_ap, tr_appris[:, j, 2], "--", lw=1.6, color=couleurs[j],
                label=f"appris — {nom}")
        ax.plot(r_ma, tr_materiel[:, j, 2], ":", lw=1.3, color=couleurs[j],
                alpha=0.9, label=f"matériel — {nom}")
        ax.plot(r_ex[0], tr_exact[0, j, 2], "o", ms=6, color=couleurs[j],
                mec="k", mew=0.6, zorder=5)
        ax.plot(r_ex[-1], tr_exact[-1, j, 2], "s", ms=6, color=couleurs[j],
                mec="k", mew=0.6, zorder=5)
    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.02, 0.45)
    ax.set_aspect("equal")
    ax.set_title("Trajectoires par la vitesse normale (demi-méridien)\n"
                 "rond = départ t=0 · carré = arrivée t=1", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", framealpha=0.9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for j, nom in enumerate(noms):
        ax.plot(temps, ecart[:, j], lw=1.9, color=couleurs[j], label=nom)
    ax.set_xlabel("t")
    ax.set_ylabel("‖x_appris(t) − x_exact(t)‖")
    ax.set_title("Écart entre la trajectoire apprise et la trajectoire exacte",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    chemin = os.path.join(OUT, "temporel_trajectoires.png")
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    print(f"\n  figure : {chemin}")


# ============================================================
# Figure d'ensemble : les profils reconstruits, vus et non vus
# ============================================================

def figure_profils(modele, box, device, tous_les_t):
    """Coupe méridienne du champ appris à chaque instant, vraie surface en fond."""
    n = len(tous_les_t)
    cols = 6
    lignes = int(np.ceil(n / cols))
    fig, axes = plt.subplots(lignes, cols, figsize=(2.5 * cols, 2.7 * lignes),
                             squeeze=False)

    xr = np.linspace(-box, box, 260)
    zr = np.linspace(-0.65, 0.65, 160)
    XX, ZZ = np.meshgrid(xr, zr, indexing="ij")
    plat = np.column_stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()])

    for i, t in enumerate(tous_les_t):
        ax = axes[i // cols][i % cols]
        vue = figer_instant(modele, t, device=device)
        with torch.no_grad():
            v = vue(torch.as_tensor(plat.astype(np.float32),
                                    device=device)).cpu().numpy().reshape(XX.shape)
        ve = T.f_exact(plat, t).reshape(XX.shape)
        ax.contourf(XX, ZZ, v, levels=[-10, 0], colors=["#5dade2"], alpha=0.5)
        ax.contour(XX, ZZ, v, levels=[0.0], colors=["#1a5276"], linewidths=1.8)
        ax.contour(XX, ZZ, ve, levels=[0.0], colors=["#c0392b"],
                   linewidths=1.0, linestyles="--")
        vu = t in T.T_OBSERVES
        ax.set_title(f"t = {t:.2f}" + ("" if vu else "   JAMAIS VU"),
                     fontsize=9, color="k" if vu else "#c0392b",
                     fontweight="normal" if vu else "bold")
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.62, 0.62)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, lignes * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Coupe méridienne du champ espace-temps appris — "
                 "bleu = appris, tirets rouges = vraie surface", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    chemin = os.path.join(OUT, "temporel_profils.png")
    fig.savefig(chemin, dpi=140)
    plt.close(fig)
    print(f"\n  figure : {chemin}")



# ============================================================
# BLOC 5 — les contrôles : que valent vraiment ces chiffres ?
# ============================================================

def _sdf_exacte_sur(points, t, arbres):
    """Distance signée exacte à la surface de l'instant t (KD-tree + signe)."""
    d, _ = arbres[t].query(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    return np.sign(T.f_exact(points, t)) * d


def bloc5(modele, box, device, vus, caches):
    titre(5, "CONTRÔLES — le Chamfer mesure-t-il le réseau, ou la grille ?")

    ch_vus = np.mean([r["chamfer"] for r in vus])
    ch_caches = {r["t"]: r["chamfer"] for r in caches}
    ch_caches_moy = np.mean(list(ch_caches.values()))

    # --- 5.1 plancher de mesure -------------------------------------------
    print("\n  5.1 — PLANCHER DE MESURE")
    print("  On applique le marching cubes au champ EXACT, aux mêmes résolutions,")
    print("  et on mesure le même Chamfer contre le même nuage de référence.")
    print("  En dessous de ce chiffre, aucun réseau ne peut descendre.\n")
    print(f"  {'grille':>7} {'pas':>8} {'Chamfer champ EXACT':>21} {'Chamfer/pas':>12}")
    print("  " + "-" * 52)
    plancher = None
    for G in (64, 96, 128, 160):
        axe = np.linspace(-box, box, G)
        pas = axe[1] - axe[0]
        XX, YY, ZZ = np.meshgrid(axe, axe, axe, indexing="ij")
        vals = T.f_exact(np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]),
                         0.5).reshape(G, G, G)
        from skimage import measure
        v, f, _, _ = measure.marching_cubes(vals, level=0.0, spacing=(pas, pas, pas))
        c = chamfer_distance(v + axe[0], T.surface_dense(0.5))["chamfer"]
        if G == GRILLE:
            plancher = c
        print(f"  {G:>7} {pas:>8.4f} {c:>21.5f} {c/pas:>12.3f}")
    print(f"\n  À la grille {GRILLE}³ des blocs 1 à 3, le plancher vaut {plancher:.5f}.")
    print(f"  Le réseau donne {ch_vus:.5f} aux instants vus et {ch_caches_moy:.5f} aux")
    print(f"  instants cachés, soit {ch_caches_moy/plancher:.2f} fois le plancher.")
    print("  Le Chamfer est donc SATURÉ : il dit « aussi bon que la grille le")
    print("  permet », et il ne peut pas départager deux méthodes proches.")

    # --- 5.2 étalon image gelée -------------------------------------------
    print("\n  5.2 — ÉTALON « IMAGE GELÉE »")
    print("  Un réseau paresseux qui se contenterait de rendre l'instant observé le")
    print("  plus proche : voilà l'erreur qu'il commettrait. Si elle était du même")
    print("  ordre que celle du réseau, le test d'interpolation ne prouverait rien.\n")
    ref = {t: T.surface_dense(t) for t in list(T.T_OBSERVES) + list(T.T_CACHES)}
    print(f"  {'t caché':>8} {'obs. le + proche':>17} {'Chamfer gelé':>14} "
          f"{'réseau':>9} {'rapport':>9}")
    print("  " + "-" * 62)
    gel = []
    for t in T.T_CACHES:
        proche = min(T.T_OBSERVES, key=lambda o: abs(o - t))
        c = chamfer_distance(ref[t], ref[proche])["chamfer"]
        gel.append(c)
        print(f"  {t:>8.2f} {proche:>17.2f} {c:>14.5f} {ch_caches[t]:>9.5f} "
              f"{c/ch_caches[t]:>8.2f}x")
    gel = np.array(gel)
    print("  " + "-" * 62)
    print(f"  {'moyenne':>8} {'':>17} {gel.mean():>14.5f} {ch_caches_moy:>9.5f} "
          f"{gel.mean()/ch_caches_moy:>8.2f}x")
    ampli = chamfer_distance(ref[0.0], ref[1.0])["chamfer"]
    print(f"\n  Amplitude totale du mouvement (Chamfer t=0 contre t=1) : {ampli:.5f}")
    print(f"  Le réseau se trompe de {100*ch_caches_moy/ampli:.1f} % de cette amplitude.")

    # --- 5.3 étalon morphing par paires -----------------------------------
    print("\n  5.3 — ÉTALON « MORPHING ENTRE LES DEUX INSTANTS ENCADRANTS »")
    print("  La vraie objection : pourquoi un réseau à quatre entrées, plutôt que")
    print("  d'interpoler linéairement entre les deux instants observés voisins ?")
    print("  On construit donc la distance signée EXACTE aux deux bouts et on les")
    print("  morphe (src/csg.py, theta = 1/2). Étalon très favorable au morphing :")
    print("  il part des surfaces exactes, le réseau n'a eu que 6000 points bruts.\n")

    from scipy.spatial import cKDTree
    from skimage import measure
    arbres = {t: cKDTree(ref[t]) for t in T.T_OBSERVES}
    axe = np.linspace(-box, box, GRILLE)
    pas = axe[1] - axe[0]
    XX, YY, ZZ = np.meshgrid(axe, axe, axe, indexing="ij")
    grille = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    sdf_grille = {t: _sdf_exacte_sur(grille, t, arbres) for t in T.T_OBSERVES}

    print(f"  {'t caché':>8} {'encadré par':>13} {'Chamfer morphing':>17} "
          f"{'Chamfer réseau':>15} {'rapport':>9}")
    print("  " + "-" * 66)
    mor = []
    for t in T.T_CACHES:
        ta = max(o for o in T.T_OBSERVES if o < t)
        tb = min(o for o in T.T_OBSERVES if o > t)
        th = (t - ta) / (tb - ta)
        champ = ((1 - th) * sdf_grille[ta] + th * sdf_grille[tb]).reshape(
            GRILLE, GRILLE, GRILLE)
        v, f, _, _ = measure.marching_cubes(champ, level=0.0, spacing=(pas, pas, pas))
        c = chamfer_distance(v + axe[0], ref[t])["chamfer"]
        mor.append(c)
        print(f"  {t:>8.2f} {f'{ta:.1f}–{tb:.1f}':>13} {c:>17.5f} "
              f"{ch_caches[t]:>15.5f} {c/ch_caches[t]:>8.2f}x")
    mor = np.array(mor)
    print("  " + "-" * 66)
    print(f"  {'moyenne':>8} {'':>13} {mor.mean():>17.5f} {ch_caches_moy:>15.5f} "
          f"{mor.mean()/ch_caches_moy:>8.2f}x")
    print("\n  Les deux se valent — et pour cause : tous deux sont au plancher de")
    print("  la grille (5.1). Sur la POSITION de la surface, le réseau 4D n'apporte")
    print("  rien de plus qu'un morphing par paires. Il faut le dire.")

    # --- 5.4 ce qui départage : la vitesse normale ------------------------
    print("\n  5.4 — CE QUI DÉPARTAGE : LA VITESSE NORMALE")
    print("  On mesure la grandeur qui ne sature pas, et qui est le vrai apport de")
    print("  la représentation espace-temps : v_n = −(∂f/∂t)/‖∇_x f‖.")
    print("  Le morphing par paires a bien une dérivée en temps, mais CONSTANTE PAR")
    print("  MORCEAUX : le champ est affine en t entre deux instants observés, donc")
    print("  ∂f/∂t y est figée et saute à chaque instant observé. Le réseau apprend")
    print("  une fonction lisse des quatre variables.\n")

    def vn_morphing(X, t, h=1e-3):
        ta = max(o for o in T.T_OBSERVES if o < t)
        tb = min(o for o in T.T_OBSERVES if o > t)
        th = (t - ta) / (tb - ta)
        g = lambda P: ((1 - th) * _sdf_exacte_sur(P, ta, arbres)
                       + th * _sdf_exacte_sur(P, tb, arbres))
        dfdt = (_sdf_exacte_sur(X, tb, arbres)
                - _sdf_exacte_sur(X, ta, arbres)) / (tb - ta)
        grad = np.empty_like(X)
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            grad[:, j] = (g(X + e) - g(X - e)) / (2 * h)
        return -dfdt / np.maximum(np.linalg.norm(grad, axis=1), 1e-12)

    print(f"  {'t caché':>8} {'|v_n| exact':>13} {'erreur RÉSEAU':>15} "
          f"{'erreur MORPHING':>17} {'rapport':>9}")
    print("  " + "-" * 68)
    er_res, er_mor = [], []
    for t in T.T_CACHES:
        P = T.surface_dense(t, n=4000, seed=23)
        g = T.grad_f_exact(P, t)
        vn_ex = -T.dfdt_exact(P, t) / np.maximum(np.linalg.norm(g, axis=1), 1e-12)
        _, gr, dfdt = derivees(modele, P, t, device=device)
        vn_re = -dfdt / np.maximum(np.linalg.norm(gr, axis=1), 1e-12)
        vn_mo = vn_morphing(P, t)
        a, b = np.abs(vn_re - vn_ex).mean(), np.abs(vn_mo - vn_ex).mean()
        er_res.append(a)
        er_mor.append(b)
        print(f"  {t:>8.2f} {np.abs(vn_ex).mean():>13.5f} {a:>15.5f} "
              f"{b:>17.5f} {b/a:>8.2f}x")
    er_res, er_mor = np.array(er_res), np.array(er_mor)
    print("  " + "-" * 68)
    print(f"  {'moyenne':>8} {'':>13} {er_res.mean():>15.5f} "
          f"{er_mor.mean():>17.5f} {er_mor.mean()/er_res.mean():>8.2f}x")

    print(f"\n  VERDICT — sur la vitesse normale, le réseau se trompe de "
          f"{er_res.mean():.5f},")
    print(f"  le morphing de {er_mor.mean():.5f} : le réseau est "
          f"{er_mor.mean()/er_res.mean():.1f} fois plus juste.")
    print("  C'est LÀ que la représentation espace-temps gagne — pas sur la position")
    print("  de la surface, où la grille sature la mesure, mais sur les DÉRIVÉES.")
    print("  Ce qu'apporte un champ f(x,y,z,t), ce n'est pas de mieux placer les")
    print("  points : c'est de posséder un champ de vitesse dérivable partout, qu'on")
    print("  peut interroger à n'importe quel instant et intégrer (bloc 4).")
    return {"plancher": plancher, "gel": gel, "morphing": mor,
            "vn_reseau": er_res, "vn_morphing": er_mor, "amplitude": ampli}



# ============================================================
# Figures de présentation
# ============================================================

def _chemin_instant(t):
    return os.path.join(DOSSIER, f"instant_t{int(round(t * 100)):03d}.npz")


def figure_donnees_vs_appris(modele, box, device, tous_les_t):
    """
    Ce qu'on a donné au réseau, et ce qu'il a produit — la planche de synthèse.

    Le globule est une surface de RÉVOLUTION : sa coupe méridienne contient
    toute l'information de forme, et c'est la seule vue où l'on puisse à la fois
    montrer les données d'entrée et la surface apprise sans que l'une cache
    l'autre. Les vues 3D véritables sont produites par figure_3d().
    """
    from src.cloud_io import load_cloud
    groupes = [tous_les_t[:6], tous_les_t[6:]]
    xr = np.linspace(-1.12, 1.12, 460)
    zr = np.linspace(-0.52, 0.52, 240)
    XX, ZZ = np.meshgrid(xr, zr, indexing="ij")
    plat = np.column_stack([XX.ravel(), np.zeros(XX.size),
                            ZZ.ravel()]).astype(np.float32)
    LX, LZ = 1.14, 0.55
    fig, axes = plt.subplots(4, 6, figsize=(27, 13.2), squeeze=False)

    for gi, groupe in enumerate(groupes):
        for ci, t in enumerate(groupe):
            vu = t in T.T_OBSERVES
            haut, bas = axes[2 * gi][ci], axes[2 * gi + 1][ci]
            coul = "#154360" if vu else "#c0392b"

            if vu:
                d = load_cloud(_chemin_instant(t))
                P = d["points"]
                r = np.hypot(P[:, 0], P[:, 1])
                haut.plot(np.concatenate([r, -r]),
                          np.concatenate([P[:, 2], P[:, 2]]), ".",
                          ms=3.0, color="#2c3e50", alpha=0.8, mew=0)
                haut.text(0.5, 0.055, f"{len(P)} points de surface + normales",
                          transform=haut.transAxes, ha="center", fontsize=12.5,
                          color="#2c3e50")
            else:
                haut.set_facecolor("#fdf0ee")
                haut.text(0.5, 0.5, "AUCUNE DONNÉE", transform=haut.transAxes,
                          ha="center", va="center", fontsize=21,
                          fontweight="bold", color="#c0392b")
            haut.set_title(f"t = {t:.2f}" + ("" if vu else "        JAMAIS MONTRÉ"),
                           fontsize=21, fontweight="bold", color=coul, pad=10)

            vue = figer_instant(modele, t, device=device)
            with torch.no_grad():
                v = vue(torch.as_tensor(plat, device=device)
                        ).cpu().numpy().reshape(XX.shape)
            ve = T.f_exact(plat, t).reshape(XX.shape)
            bas.contourf(XX, ZZ, v, levels=[-10, 0], colors=["#aed6f1"], alpha=0.75)
            bas.contour(XX, ZZ, v, levels=[0.0], colors=["#154360"], linewidths=3.4)
            bas.contour(XX, ZZ, ve, levels=[0.0], colors=["#c0392b"],
                        linewidths=1.7, linestyles="--")
            r = mesurer(modele, t, box, device=device, n_iou=60000)
            bas.text(0.5, 0.055,
                     f"IoU {100 * r['iou']:.2f} %      Chamfer {r['chamfer']:.4f}",
                     transform=bas.transAxes, ha="center", fontsize=14,
                     bbox=dict(fc="white", ec="0.7", alpha=0.94, pad=4))

            for ax in (haut, bas):
                ax.set_aspect("equal")
                ax.set_xlim(-LX, LX)
                ax.set_ylim(-LZ, LZ)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(coul)
                    sp.set_linewidth(3.0 if not vu else 1.8)

    for r, (txt, c) in enumerate([("CE QU'ON\nLUI DONNE", "#2c3e50"),
                                  ("CE QU'IL\nPRODUIT", "#154360")] * 2):
        axes[r][0].set_ylabel(txt, fontsize=16, fontweight="bold", color=c,
                              labelpad=18)
    for ligne in (2, 3):
        ax = axes[ligne][5]
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[3][5].text(0.5, 0.62,
                    "bleu   la surface apprise, f = 0\n"
                    "tirets rouges   la VRAIE surface\n\n"
                    "Les points sont sur une surface de\n"
                    "révolution : on les rabat dans le plan\n"
                    "méridien, d'où le trait épais.",
                    transform=axes[3][5].transAxes, ha="center", va="center",
                    fontsize=14, bbox=dict(fc="#fbfbfb", ec="0.65", pad=13))

    fig.suptitle(
        "UN SEUL réseau f(x,y,z,t) apprend toute la séquence à partir de six instants\n"
        "les cinq instants encadrés de rouge ne lui ont jamais été montrés — "
        "il les reconstruit quand même", fontsize=23, y=0.982)
    fig.tight_layout(rect=[0, 0.005, 1, 0.925])
    chemin = os.path.join(OUT, "temporel_donnees_vs_appris.png")
    fig.savefig(chemin, dpi=105, facecolor="white")
    plt.close(fig)
    print(f"  figure : {chemin}")


def _cadrer_3d(ax):
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(-0.95, 0.95)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-62)
    ax.set_axis_off()


def figure_3d(modele, box, device, tous_les_t):
    """Les surfaces reconstruites, en vraie 3D : ce sont des maillages triangulés."""
    fig = plt.figure(figsize=(26, 11.5))
    for i, t in enumerate(tous_les_t):
        ax = fig.add_subplot(2, 6, i + 1, projection="3d")
        sommets, faces, _ = reconstruire(modele, t, box, device=device)
        vu = t in T.T_OBSERVES
        ax.plot_trisurf(sommets[:, 0], sommets[:, 1], faces, sommets[:, 2],
                        color="#5dade2" if vu else "#ec7063", edgecolor="none",
                        linewidth=0, antialiased=True, shade=True)
        e = euler_characteristic(sommets, faces)
        ax.set_title(f"t = {t:.2f}" + ("" if vu else "  —  JAMAIS MONTRÉ"),
                     fontsize=15, fontweight="bold",
                     color="#154360" if vu else "#c0392b", y=0.94)
        ax.text2D(0.5, 0.06, f"V = {mesh_volume(sommets, faces):.4f}    "
                             f"χ = {e['chi']}    {len(faces)} triangles",
                  transform=ax.transAxes, ha="center", fontsize=11)
        _cadrer_3d(ax)
    fig.suptitle("La surface APPRISE, reconstruite en 3D par marching cubes "
                 f"sur une grille {GRILLE}³\n"
                 "bleu = instant fourni au réseau   ·   rouge = instant jamais montré",
                 fontsize=21, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    chemin = os.path.join(OUT, "temporel_3d.png")
    fig.savefig(chemin, dpi=105, facecolor="white")
    plt.close(fig)
    print(f"  figure : {chemin}")


def figure_3d_nuage(modele, box, device):
    """Les données d'entrée sont un vrai nuage 3D orienté, pas une coupe."""
    from src.cloud_io import load_cloud
    fig = plt.figure(figsize=(21, 7.6))
    for i, (t, lib) in enumerate([(0.0, "instant FOURNI"),
                                  (0.5, "instant JAMAIS MONTRÉ"),
                                  (1.0, "instant FOURNI")]):
        vu = t in T.T_OBSERVES
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        if vu:
            d = load_cloud(_chemin_instant(t))
            P, Nn = d["points"], d["normals"]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1.4, c="#34495e",
                       alpha=0.30, depthshade=False)
            k = np.random.default_rng(0).choice(len(P), 120, replace=False)
            ax.quiver(P[k, 0], P[k, 1], P[k, 2], Nn[k, 0], Nn[k, 1], Nn[k, 2],
                      length=0.13, color="#e67e22", linewidth=1.0,
                      arrow_length_ratio=0.35)
        sommets, faces, _ = reconstruire(modele, t, box, device=device)
        ax.plot_trisurf(sommets[:, 0], sommets[:, 1], faces, sommets[:, 2],
                        color="#5dade2" if vu else "#ec7063", edgecolor="none",
                        shade=True, alpha=0.62 if vu else 1.0)
        ax.set_title(f"t = {t:.2f} — {lib}\n" +
                     ("points 3D + normales, puis surface apprise" if vu else
                      "rien en entrée — surface produite par le réseau"),
                     fontsize=15, fontweight="bold",
                     color="#154360" if vu else "#c0392b")
        _cadrer_3d(ax)
    fig.suptitle("Les données sont un vrai NUAGE 3D (points + normales sortantes), "
                 "pas une coupe", fontsize=20, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    chemin = os.path.join(OUT, "temporel_3d_nuage.png")
    fig.savefig(chemin, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"  figure : {chemin}")


# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--points", type=int, default=6000)
    ap.add_argument("--volume", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="forcer le CPU")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cpu") if args.cpu else get_device()

    seq = preparer(n_points=args.points)
    modele, jeu, hist, duree = apprendre(seq, args.epochs, args.volume,
                                         args.k, args.seed, device)
    box = jeu["box"]

    vus, caches, tout = blocs12(modele, box, device)
    marques = {t: "vu" for t in T.T_OBSERVES}
    marques.update({t: "  JAMAIS VU" for t in T.T_CACHES})
    bloc3(tout, box, marques)
    bloc4(modele, box, device)
    bloc5(modele, box, device, vus, caches)
    tous_les_t = [r["t"] for r in tout]
    print("\n" + "-" * 78)
    print("FIGURES")
    print("-" * 78)
    figure_profils(modele, box, device, tous_les_t)
    figure_donnees_vs_appris(modele, box, device, tous_les_t)
    figure_3d(modele, box, device, tous_les_t)
    figure_3d_nuage(modele, box, device)

    torch.save(modele.state_dict(), os.path.join(OUT, "modele_temporel.pt"))
    print("\n" + "=" * 78)
    print("FIN — les quatre blocs sont affichés, plus les contrôles")
    print("=" * 78)


if __name__ == "__main__":
    main()
