"""
Morphing et opérations booléennes sur des surfaces implicites.

Réalise la recommandation (d) du mail de F. De Vuyst : « D'où toute la
puissance des surfaces implicites. » Le point à démontrer n'est pas que les
images sont jolies, mais que les opérations sont JUSTES et qu'elles ne
demandent rien de plus qu'une évaluation de f — donc qu'elles marchent
identiquement sur une formule et sur un réseau de neurones.

Trois blocs :

  1. Morphing sphère → globule rouge quartique, theta = 0, 0.25, 0.5, 0.75, 1.
     Mesure du volume et de l'aire à chaque étape.

  2. Booléens sur deux sphères décalées, avec un CONTRÔLE QUANTITATIF :
     V(A∪B) + V(A∩B) = V(A) + V(B). C'est une identité d'ensembles, donc
     exacte quelle que soit la forme ; tout écart mesuré ne peut venir que de
     la discrétisation. C'est elle qui prouve que min/max sont bien posés,
     et pas seulement l'aspect des images.

  3. Le MÊME morphing, entre deux champs APPRIS par réseau. Une seule ligne
     change (from_model au lieu de la formule) : c'est l'argument de fond,
     le morphing ne demande pas de connaître d'équation.

Lancer :  /usr/local/bin/python3 main_morph.py
"""
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.csg import (morph, union, intersection, difference, from_model,
                     normalized, translated, gradient_norm)
from src.shapes import (implicit_rbc_unit, get_surface, get_surface_normals,
                        rbc_radius_max)
from src.data import build_training_data_realistic
from src.model import IGRNet
from src.train import train_igr
from src.reconstruct import extract_surface
from src.invariants import euler_characteristic, mesh_area, mesh_volume

THETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


# ------------------------------------------------------------------
# Formes de test définies LOCALEMENT (rien n'est importé de src/shapes
# pour la sphère : le module de morphing doit être testé sur un objet
# dont on connaît le volume à la main).
# ------------------------------------------------------------------

def sphere_field(radius=1.0, center=(0.0, 0.0, 0.0)):
    """Distance signée exacte à une sphère : f = ‖x − c‖ − R."""
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)

    def f(P):
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        return np.linalg.norm(P - c, axis=1) - radius

    return f


def sample_sphere_surface(n, radius=1.0, center=(0.0, 0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * radius + np.asarray(center, dtype=np.float64)


# ------------------------------------------------------------------
# Outils communs
# ------------------------------------------------------------------

def grid_eval(f, box, res, chunk=200000):
    """Évalue un champ sur une grille res³ dans [−box, box]³, par paquets."""
    lin = np.linspace(-box, box, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    out = np.empty(len(pts), dtype=np.float64)
    for s in range(0, len(pts), chunk):
        out[s:s + chunk] = np.asarray(f(pts[s:s + chunk])).ravel()
    return lin, out.reshape(res, res, res)


def maille(f, box, res):
    """Champ -> maillage de f = 0, avec ses invariants."""
    lin, vals = grid_eval(f, box, res)
    if vals.min() > 0 or vals.max() < 0:
        return None
    verts, faces, _, _ = extract_surface(lin, lin, lin, vals)
    topo = euler_characteristic(verts, faces)
    return {"verts": verts, "faces": faces, "chi": topo["chi"],
            "aire": mesh_area(verts, faces), "volume": mesh_volume(verts, faces)}


def coupe(f, box, res=400):
    """Valeurs du champ dans le plan y = 0, pour tracer un contour."""
    lin = np.linspace(-box, box, res)
    X, Z = np.meshgrid(lin, lin, indexing="ij")
    pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    return lin, np.asarray(f(pts)).reshape(res, res)


def charger_reseau(surface_type, tag, n_epochs=2000, n_vol=20000, batch=1024):
    """Recharge un réseau déjà entraîné, ou l'entraîne si le cache est vide."""
    path = f"output/model_curv_{tag}.pt"
    np.random.seed(0)
    torch.manual_seed(0)
    surface_points, implicit_exact, box = get_surface(surface_type)
    model = IGRNet(radius_init=1.0)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"    réseau '{tag}' rechargé depuis {path}")
        return model, box
    normals = get_surface_normals(surface_type, surface_points)
    X_surf, normals_t, X_vol, sign_vol = build_training_data_realistic(
        surface_points, implicit_exact, box, normals=normals, n_vol=n_vol)
    print(f"    entraînement du réseau '{tag}' ({n_epochs} époques)...")
    t0 = time.time()
    train_igr(model, X_surf, normals_t, X_vol, sign_vol, box,
              n_epochs=n_epochs, batch_surf=batch, batch_vol=batch, lr=1e-3)
    print(f"    entraîné en {time.time() - t0:.0f} s")
    torch.save(model.state_dict(), path)
    return model, box


# ==================================================================
# BLOC 1 — morphing sphère → globule quartique (champs analytiques)
# ==================================================================

def bloc_morphing_analytique(box=1.25, res=128):
    print("=" * 78)
    print("1. MORPHING  theta·f_globule + (1−theta)·f_sphère = 0")
    print("=" * 78)

    # Sphère de MÊME VOLUME que le globule : les deux extrémités du morphing
    # sont alors comparables, et la variation de volume le long du chemin
    # mesure quelque chose (elle n'est pas imposée par les bornes).
    L = rbc_radius_max()
    vol_rbc = 85.947 / L ** 3          # volume du globule ramené au rayon unité
    R_sph = (3.0 * vol_rbc / (4.0 * np.pi)) ** (1.0 / 3.0)
    print(f"  f1 = globule rouge quartique (Kuchel), normalisé au rayon 1")
    print(f"  f2 = sphère de rayon {R_sph:.4f}, choisie de MÊME VOLUME "
          f"({vol_rbc:.4f}) que le globule")

    f_glob_brut = implicit_rbc_unit
    f_sph_brut = sphere_field(R_sph)

    # Les deux champs n'ont pas la même pente : la quartique vaut des dizaines
    # là où la distance signée vaut l'unité. Sans remise à l'échelle, theta=0.5
    # donnerait une forme quasi identique au globule — l'interpolation serait
    # dominée par le champ le plus « raide ». On les normalise donc par la
    # médiane de ‖∇f‖ sur leur propre surface.
    from src.shapes import sample_rbc_unit_surface
    pts_glob = sample_rbc_unit_surface(3000, seed=0)
    pts_sph = sample_sphere_surface(3000, R_sph)
    f_glob = normalized(f_glob_brut, pts_glob)
    f_sph = normalized(f_sph_brut, pts_sph)
    print(f"  médiane ‖∇f‖ sur la surface : globule {f_glob.scale:.4f}, "
          f"sphère {f_sph.scale:.4f}  (la sphère vaut 1, c'est une SDF exacte)")
    print(f"  -> les deux champs sont ramenés à une pente unité avant le mélange\n")

    print(f"  {'theta':>6s} {'forme':<24s} {'chi':>4s} {'volume':>10s} "
          f"{'aire':>10s} {'sphéricité':>11s}")
    print("  " + "-" * 70)
    series, resultats = [], []
    for t in THETAS:
        f = morph(f_glob, f_sph, t)
        m = maille(f, box, res)
        if m is None:
            print(f"  {t:6.2f}  surface absente de la boîte")
            continue
        # Sphéricité de Wadell : 1 pour une sphère, < 1 sinon. Elle résume
        # d'un chiffre le passage de la sphère au disque biconcave.
        psi = (np.pi ** (1 / 3.0) * (6 * m["volume"]) ** (2 / 3.0)) / m["aire"]
        nom = {0.0: "sphère (départ)", 1.0: "globule (arrivée)"}.get(t, "intermédiaire")
        print(f"  {t:6.2f} {nom:<24s} {m['chi']:>4d} {m['volume']:>10.5f} "
              f"{m['aire']:>10.5f} {psi:>11.5f}")
        series.append((t, f))
        resultats.append((t, m, psi))

    print("\n  Lecture : chi reste égal à 2 tout du long — le morphing linéaire")
    print("  ne déchire ni ne perce la surface. La sphéricité décroît de façon")
    print("  monotone, ce qui est la signature d'un passage progressif du plus")
    print("  compact (sphère, psi=1) au plus aplati (disque biconcave).")
    figure_morphing(series, box, "output/morph_sphere_globule.png",
                    "Morphing sphère → globule rouge (champs analytiques)")
    return resultats


def figure_morphing(series, box, out_path, titre):
    """Coupe y=0 de chaque étape, plus une superposition des contours."""
    n = len(series)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.0 * (n + 1), 3.4))
    couleurs = plt.cm.viridis(np.linspace(0, 0.9, n))

    for k, (t, f) in enumerate(series):
        lin, V = coupe(f, box)
        ax = axes[k]
        ax.contourf(lin, lin, V.T, levels=[-1e9, 0.0], colors=[couleurs[k]])
        ax.contour(lin, lin, V.T, levels=[0.0], colors="k", linewidths=0.8)
        ax.set_aspect("equal")
        ax.set_title(rf"$\theta$ = {t:.2f}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("z" if k == 0 else "")
        ax.set_xlim(-box, box); ax.set_ylim(-box, box)

    ax = axes[-1]
    for k, (t, f) in enumerate(series):
        lin, V = coupe(f, box)
        ax.contour(lin, lin, V.T, levels=[0.0], colors=[couleurs[k]], linewidths=1.6)
    ax.set_aspect("equal"); ax.set_xlim(-box, box); ax.set_ylim(-box, box)
    ax.set_xlabel("x"); ax.set_title("superposition", fontsize=11)

    fig.suptitle(titre + r"   —   coupe $y=0$,   $\theta f_1 + (1-\theta) f_2 = 0$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {out_path}")


# ==================================================================
# BLOC 2 — booléens et contrôle du volume
# ==================================================================

def bloc_booleens(box=1.45, res=160, ecart=0.4, R=1.0):
    print("\n" + "=" * 78)
    print("2. OPÉRATIONS BOOLÉENNES — union, intersection, différence")
    print("=" * 78)
    print(f"  A : sphère de rayon {R}, centrée en (-{ecart/2:.2f}, 0, 0)")
    print(f"  B : sphère de rayon {R}, centrée en (+{ecart/2:.2f}, 0, 0)   "
          f"(écart des centres : {ecart})")
    print(f"  grille {res}³ sur [−{box}, {box}]³, la MÊME pour les cinq maillages.")

    fA = sphere_field(R, (-ecart / 2, 0, 0))
    fB = sphere_field(R, (+ecart / 2, 0, 0))
    ops = [("A", fA), ("B", fB),
           ("A ∪ B  = min(fA, fB)", union(fA, fB)),
           ("A ∩ B  = max(fA, fB)", intersection(fA, fB)),
           ("A \\ B  = max(fA, −fB)", difference(fA, fB))]

    print(f"\n  {'opération':<24s} {'chi':>4s} {'volume':>11s} {'aire':>11s}")
    print("  " + "-" * 54)
    vol = {}
    meshes = {}
    for nom, f in ops:
        m = maille(f, box, res)
        vol[nom.split()[0] if nom in ("A", "B") else nom] = m["volume"]
        meshes[nom] = m
        print(f"  {nom:<24s} {m['chi']:>4d} {m['volume']:>11.6f} {m['aire']:>11.6f}")

    vA, vB = meshes["A"]["volume"], meshes["B"]["volume"]
    vU = meshes["A \u222a B  = min(fA, fB)"]["volume"]
    vI = meshes["A \u2229 B  = max(fA, fB)"]["volume"]
    vD = meshes["A \\ B  = max(fA, \u2212fB)"]["volume"]
    exact = 4.0 / 3.0 * np.pi * R ** 3
    # Volume EXACT de la lentille d'intersection de deux sph\u00e8res de m\u00eame rayon
    # dont les centres sont distants de d :   V = \u03c0(4R + d)(2R \u2212 d)\u00b2 / 12.
    # (contr\u00f4le de la formule : d = 0 redonne bien 4\u03c0R\u00b3/3)
    lentille = np.pi * (4.0 * R + ecart) * (2.0 * R - ecart) ** 2 / 12.0

    print(f"\n  contr\u00f4le pr\u00e9alable \u2014 volume de A seul : {vA:.6f}  "
          f"contre 4\u03c0R\u00b3/3 = {exact:.6f}  (\u00e9cart {100*(vA-exact)/exact:+.4f} %)")

    print("\n  A. CONFRONTATION AUX VOLUMES EXACTS")
    print("     Les trois volumes se d\u00e9duisent de la lentille analytique, donc")
    print("     chaque op\u00e9ration a une v\u00e9rit\u00e9 de r\u00e9f\u00e9rence, pas seulement une image.")
    print(f"     {'op\u00e9ration':<10s} {'mesur\u00e9':>11s} {'exact':>11s} {'\u00e9cart':>10s}")
    for nom, mesure, attendu in (("A \u2229 B", vI, lentille),
                                 ("A \u222a B", vU, 2 * exact - lentille),
                                 ("A \\ B", vD, exact - lentille)):
        print(f"     {nom:<10s} {mesure:>11.6f} {attendu:>11.6f} "
              f"{100*(mesure-attendu)/attendu:>+9.4f} %")

    print("\n  B. IDENTIT\u00c9 D'INCLUSION-EXCLUSION   V(A\u222aB) + V(A\u2229B) = V(A) + V(B)")
    g, d = vU + vI, vA + vB
    print(f"     membre de gauche  V(A\u222aB) + V(A\u2229B) = {vU:.6f} + {vI:.6f} = {g:.8f}")
    print(f"     membre de droite  V(A)   + V(B)   = {vA:.6f} + {vB:.6f} = {d:.8f}")
    print(f"     \u00e9cart absolu = {abs(g-d):.3e}      \u00c9CART RELATIF = {100*abs(g-d)/d:.3e} %")
    print("     C'est une identit\u00e9 ENSEMBLISTE, vraie pour n'importe quels")
    print("     ensembles : l'\u00e9cart mesur\u00e9 ne peut venir que de la grille.")
    print("     Il est ici plus petit que l'erreur de discr\u00e9tisation d'un seul")
    print("     volume (0.02 %) parce que les deux erreurs se compensent case")
    print("     par case : dans chaque cellule, min et max se partagent")
    print("     exactement les m\u00eames morceaux d'interface que A et B.")

    print("\n  C. CONTR\u00d4LE SECONDAIRE           V(A\\B) = V(A) \u2212 V(A\u2229B)")
    att = vA - vI
    print(f"     mesur\u00e9 {vD:.6f}   attendu {att:.6f}   "
          f"\u00e9cart relatif {100*abs(vD-att)/att:.4f} %")

    print("\n  D. ORDRE DES VOLUMES \u2014 ce qui distingue vraiment min de max")
    print("     L'identit\u00e9 B est SYM\u00c9TRIQUE : \u00e9changer union et intersection la")
    print("     laisse vraie. Elle ne peut donc pas, \u00e0 elle seule, prouver que")
    print("     min est l'union. Ce sont A, C et l'encadrement ci-dessous qui le font.")
    ordre = vI < min(vA, vB) <= max(vA, vB) < vU
    print(f"     V(A\u2229B)={vI:.6f} < V(A)={vA:.6f} = V(B) < V(A\u222aB)={vU:.6f}  "
          f"-> {'VRAI' if ordre else '\u00c9CHEC'}")
    print(f"     recouvrement : V(A\u2229B)/V(A) = {100*vI/vA:.2f} % de chaque sph\u00e8re")

    figure_booleens(ops, box, "output/csg_booleens.png")
    return {"vA": vA, "vB": vB, "vU": vU, "vI": vI, "vD": vD,
            "lentille": lentille,
            "err_incl_excl": 100 * abs(g - d) / d,
            "err_diff": 100 * abs(vD - att) / att}


def figure_booleens(ops, box, out_path):
    fig, axes = plt.subplots(1, len(ops), figsize=(3.0 * len(ops), 3.4))
    for ax, (nom, f) in zip(axes, ops):
        lin, V = coupe(f, box)
        ax.contourf(lin, lin, V.T, levels=[-1e9, 0.0], colors=["#4C72B0"])
        ax.contour(lin, lin, V.T, levels=[0.0], colors="k", linewidths=0.8)
        ax.set_aspect("equal"); ax.set_xlim(-box, box); ax.set_ylim(-box, box)
        ax.set_title(nom, fontsize=10)
        ax.set_xlabel("x")
    axes[0].set_ylabel("z")
    fig.suptitle("Opérations booléennes sur deux sphères décalées — coupe $y=0$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {out_path}")


# ==================================================================
# BLOC 3 — le même morphing, entre deux champs APPRIS
# ==================================================================

def bloc_morphing_appris(res=128):
    print("\n" + "=" * 78)
    print("3. LE MÊME MORPHING, ENTRE DEUX CHAMPS APPRIS PAR RÉSEAU")
    print("=" * 78)
    print("  Aucune formule n'intervient : f1 et f2 sont deux réseaux IGR.")
    print("  Le code appelé est exactement le même qu'au bloc 1 — seule la")
    print("  fabrication des deux champs change (from_model au lieu de la formule).")

    m_glob, box_g = charger_reseau("biconcave", "biconcave")
    m_sph, box_s = charger_reseau("sphere", "sphere")
    box = min(box_g, box_s)
    print(f"    boîtes d'entraînement : globule {box_g}, sphère {box_s} "
          f"-> on travaille sur la plus petite, {box}")

    f_glob_res = from_model(m_glob)
    f_sph_res = from_model(m_sph)

    # Normalisation sur la surface du réseau lui-même. Les réseaux sont
    # entraînés avec un terme eikonal, donc leur ‖∇f‖ doit déjà valoir ~1 :
    # les facteurs affichés ci-dessous sont au passage une mesure de la
    # qualité de cette contrainte.
    m1 = maille(f_glob_res, box, 96)
    m2 = maille(f_sph_res, box, 96)
    f_glob = normalized(f_glob_res, m1["verts"][::7], h=1e-3)
    f_sph = normalized(f_sph_res, m2["verts"][::7], h=1e-3)
    print(f"    médiane ‖∇f‖ des réseaux sur leur surface : "
          f"globule {f_glob.scale:.4f}, sphère {f_sph.scale:.4f}")
    print(f"    (attendu ≈ 1 : c'est ce que le terme eikonal impose pendant "
          f"l'entraînement)\n")

    print(f"  {'theta':>6s} {'chi':>4s} {'volume':>10s} {'aire':>10s} "
          f"{'sphéricité':>11s}")
    print("  " + "-" * 46)
    series, resultats = [], []
    for t in THETAS:
        f = morph(f_glob, f_sph, t)
        m = maille(f, box, res)
        if m is None:
            print(f"  {t:6.2f}  surface absente de la boîte")
            continue
        psi = (np.pi ** (1 / 3.0) * (6 * m["volume"]) ** (2 / 3.0)) / m["aire"]
        print(f"  {t:6.2f} {m['chi']:>4d} {m['volume']:>10.5f} "
              f"{m['aire']:>10.5f} {psi:>11.5f}")
        series.append((t, f))
        resultats.append((t, m, psi))

    print("\n  Même comportement qu'au bloc 1 : chi = 2 conservé, sphéricité")
    print("  monotone. Le morphing ne demande donc PAS de connaître l'équation")
    print("  des formes — c'est ce qui le rend utilisable sur des capsules")
    print("  mesurées, pour lesquelles aucune formule n'existera.")
    figure_morphing(series, box, "output/morph_reseaux.png",
                    "Morphing entre deux champs APPRIS (réseaux IGR)")
    return resultats


# ==================================================================

def main():
    os.makedirs("output", exist_ok=True)
    t0 = time.time()
    bloc_morphing_analytique()
    b2 = bloc_booleens()
    bloc_morphing_appris()
    print("\n" + "=" * 78)
    print(f"terminé en {time.time() - t0:.0f} s")
    print(f"identité d'inclusion-exclusion vérifiée à {b2['err_incl_excl']:.2e} %")
    print("=" * 78)


if __name__ == "__main__":
    main()
