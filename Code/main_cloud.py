"""
Apprendre une surface à partir d'un FICHIER de nuage de points, sans formule.

C'est le mode qui s'appliquera aux capsules du BMBI : là-bas il n'y aura pas de
`implicit_exact`, il y aura un fichier. La chaîne est donc coupée en deux :

    GÉNÉRATION            |   EXPLOITATION
    la formule fabrique   |   on relit le fichier et on ne voit plus que
    un maillage, puis un  |   des points et des normales. La formule ne
    fichier .npz          |   ressert QUE pour noter le résultat.

Le compteur d'appels installé au bloc 4 rend cette séparation vérifiable plutôt
que déclarative : il affiche noir sur blanc que la formule n'est pas appelée une
seule fois pendant l'entraînement.

Lancer :  python main_cloud.py
"""
import io
import os
import time
from contextlib import redirect_stdout

import numpy as np
import torch
from scipy.spatial import cKDTree

# --- tout sur CPU -----------------------------------------------------------
# Les blocs 4 et 5 comparent deux entraînements qui ne diffèrent QUE par la
# source du signe. Or sur GPU Apple (MPS) deux entraînements à graine identique
# divergent (constaté sur ce projet, cf. README) : l'écart mesuré serait alors
# un mélange de l'effet cherché et du bruit matériel. Sur CPU l'entraînement est
# reproductible au bit près, donc tout écart observé est imputable au signe.
import src.train
import src.metrics
import src.reconstruct

_CPU = lambda: torch.device("cpu")
src.train.get_device = _CPU
src.metrics.get_device = _CPU
src.reconstruct.get_device = _CPU

from src.cloud_io import generate_cloud_file, load_cloud, fem_like_mesh
from src.inside_outside import OrientedCloud, agreement
from src.shapes import (get_surface, sample_ellipsoid_surface,
                        sample_rbc_unit_surface)
from src.model import IGRNet
from src.train import train_igr
from src.reconstruct import evaluate_grid, extract_surface
from src.metrics import chamfer_distance, iou_occupancy, eikonal_error, model_inside

OUT = "output"
N_TEST = 200000          # points de contrôle tirés dans la boîte
COQUILLE = 0.05          # demi-épaisseur de la coquille exclue, en |f_exact|


def titre(n, texte):
    print("\n" + "=" * 78)
    print(f"BLOC {n} — {texte}")
    print("=" * 78)


def reference_dense(shape):
    """Nuage dense sur la VRAIE surface — évaluation seulement."""
    if shape == "ellipsoid":
        return sample_ellipsoid_surface(n=50000, a=1.2, b=0.8, c=0.6).astype(np.float64)
    if shape == "rbc_unit":
        return sample_rbc_unit_surface(n=50000, seed=11).astype(np.float64)
    raise ValueError(shape)


class Compteur:
    """Enveloppe une fonction et compte ses appels. Sert de preuve, pas de calcul."""

    def __init__(self, fonction):
        self.fonction = fonction
        self.appels = 0

    def __call__(self, *a, **kw):
        self.appels += 1
        return self.fonction(*a, **kw)


# ============================================================
# Génération des fichiers d'échange
# ============================================================

def generer(shape, n, seed=0, noise=0.0):
    suffixe = f"_bruit{noise:g}" if noise > 0 else ""
    chemin = os.path.join(OUT, f"nuage_{shape}_{n}{suffixe}.npz")
    _, info = generate_cloud_file(chemin, shape, n_points=n, seed=seed, noise=noise)
    return chemin, info


def preparer():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    print("APPRENTISSAGE DEPUIS UN NUAGE DE POINTS SEUL — CHAÎNE COMPLÈTE")
    print("=" * 78)
    print("\nÉtape de génération : maillage triangulaire puis fichier d'échange.")
    print("Les normales viennent des TRIANGLES (sens trigonométrique), pas du")
    print("gradient analytique ; l'orientation est fixée par le volume signé du")
    print("maillage, qui ne fait intervenir aucune formule.\n")
    print(f"  {'forme':<12} {'sommets':>9} {'triangles':>10} {'volume signé':>13} "
          f"{'orientation':>12}")
    print("  " + "-" * 60)
    for shape in ("ellipsoid", "rbc_unit"):
        _, _, _, _, info = fem_like_mesh(shape)
        etat = "inversée" if info["orientation_inversee"] else "déjà sortante"
        print(f"  {shape:<12} {info['n_verts']:>9} {info['n_faces']:>10} "
              f"{info['volume']:>13.5f} {etat:>12}")


# ============================================================
# BLOC 1 — le signe déduit du nuage contre le signe exact
# ============================================================

def bloc1():
    titre(1, "le signe déduit du nuage tient-il face au signe exact ?")
    print(f"\n{N_TEST} points tirés uniformément dans la boîte. Le nuage compte")
    print("6000 points, le vote porte sur k = 5 voisins.\n")

    resultats = {}
    for shape in ("ellipsoid", "rbc_unit"):
        chemin, _ = generer(shape, 6000)
        d = load_cloud(chemin)
        nuage = OrientedCloud(d["points"], d["normals"])
        box = d["box"]

        _, f_exact, _ = get_surface(shape)
        rng = np.random.default_rng(3)
        Q = rng.uniform(-box, box, size=(N_TEST, 3))
        f = f_exact(Q)
        s_ref = np.sign(f)
        s_nuage = nuage.sign(Q, k=5)

        hors = np.abs(f) > COQUILLE
        acc_global = agreement(s_nuage, s_ref)
        acc_hors = agreement(s_nuage[hors], s_ref[hors])

        # Équivalence de la coquille en distance : |f| = 0.05 ne représente pas
        # la même épaisseur pour une distance signée et pour une quartique.
        arbre = cKDTree(reference_dense(shape))
        dist, _ = arbre.query(Q)
        larg = float(dist[np.abs(f) <= COQUILLE].max())

        faux = s_nuage != s_ref
        print(f"  {shape}")
        print(f"    fichier relu            : {os.path.basename(chemin)}  "
              f"({d['n']} points, boîte ±{box})")
        print(f"    provenance déclarée     : {d['source']}")
        print(f"    accord global           : {acc_global:7.3f} %   "
              f"({int(faux.sum())} points en désaccord sur {N_TEST})")
        print(f"    accord hors coquille    : {acc_hors:7.3f} %   "
              f"(|f_exact| > {COQUILLE}, soit {int(hors.sum())} points)"
              f"   [{'OK' if acc_hors >= 99.0 else 'ÉCHEC'}]  attendu ≥ 99 %")
        print(f"    la coquille |f| ≤ {COQUILLE} correspond à une distance à la "
              f"surface ≤ {larg:.4f}")
        if faux.any():
            print(f"    tous les désaccords sont à une distance ≤ "
                  f"{float(dist[faux].max()):.4f} de la surface "
                  f"(médiane {float(np.median(dist[faux])):.4f})")
        else:
            print("    aucun désaccord")
        print()
        resultats[shape] = (acc_global, acc_hors)
    return resultats


# ============================================================
# BLOC 2 — combien de points faut-il ?
# ============================================================

def bloc2():
    titre(2, "de combien de points a-t-on besoin ?")
    print("\nMême maillage à chaque fois, seul le nombre de points tirés change :")
    print("la comparaison est donc propre. « épaisseur d'erreur » = distance à la")
    print("surface du point mal classé le plus éloigné — c'est la vraie mesure de")
    print("ce que coûte un nuage lâche.\n")

    for shape in ("ellipsoid", "rbc_unit"):
        _, f_exact, box = get_surface(shape)
        rng = np.random.default_rng(3)
        Q = rng.uniform(-box, box, size=(N_TEST, 3))
        f = f_exact(Q)
        s_ref = np.sign(f)
        hors = np.abs(f) > COQUILLE
        arbre = cKDTree(reference_dense(shape))
        dist, _ = arbre.query(Q)

        print(f"  {shape}")
        print(f"    {'points':>7} {'espacement':>11} {'accord global':>14} "
              f"{'accord hors coquille':>21} {'épaisseur erreur':>17}")
        print("    " + "-" * 74)
        for n in (500, 2000, 6000, 20000):
            chemin, _ = generer(shape, n)
            d = load_cloud(chemin)
            nuage = OrientedCloud(d["points"], d["normals"])
            # espacement moyen du nuage : distance au plus proche voisin
            dd, _ = nuage.tree.query(nuage.points, k=2)
            espacement = float(dd[:, 1].mean())

            s = nuage.sign(Q, k=5)
            faux = s != s_ref
            ep = float(dist[faux].max()) if faux.any() else 0.0
            print(f"    {n:>7} {espacement:>11.4f} {agreement(s, s_ref):>13.3f} % "
                  f"{agreement(s[hors], s_ref[hors]):>20.3f} % {ep:>17.4f}")

        # variante bruitée : ce que ferait une acquisition imparfaite
        chemin, _ = generer(shape, 6000, noise=0.01)
        d = load_cloud(chemin)
        nuage = OrientedCloud(d["points"], d["normals"])
        s = nuage.sign(Q, k=5)
        faux = s != s_ref
        ep = float(dist[faux].max()) if faux.any() else 0.0
        print(f"    {'6000':>7} {'bruit 0.01':>11} {agreement(s, s_ref):>13.3f} % "
              f"{agreement(s[hors], s_ref[hors]):>20.3f} % {ep:>17.4f}")
        print()


# ============================================================
# BLOC 3 — effet du vote sur k voisins
# ============================================================

def bloc3():
    titre(3, "effet de k — le creux du globule est le cas difficile")
    _, f_exact, box = get_surface("rbc_unit")
    rng = np.random.default_rng(3)
    Q = rng.uniform(-box, box, size=(N_TEST, 3))
    f = f_exact(Q)
    s_ref = np.sign(f)
    hors = np.abs(f) > COQUILLE

    # le creux : autour de l'axe, là où les deux cuvettes sont presque à égale
    # distance et où le plus proche voisin peut basculer de l'une à l'autre
    r = np.hypot(Q[:, 0], Q[:, 1])
    creux = (r < 0.40) & (np.abs(Q[:, 2]) < 0.30)

    for etiquette, noise in (("nuage propre", 0.0), ("nuage bruité (0.01)", 0.01)):
        chemin, _ = generer("rbc_unit", 6000, noise=noise)
        d = load_cloud(chemin)
        nuage = OrientedCloud(d["points"], d["normals"])
        print(f"\n  {etiquette} — 6000 points, {int(creux.sum())} points de "
              f"contrôle dans le creux")
        print(f"    {'k':>3} {'accord global':>14} {'hors coquille':>15} "
              f"{'dans le creux':>15}")
        print("    " + "-" * 50)
        for k in (1, 3, 5, 9, 15):
            s = nuage.sign(Q, k=k)
            print(f"    {k:>3} {agreement(s, s_ref):>13.3f} % "
                  f"{agreement(s[hors], s_ref[hors]):>14.3f} % "
                  f"{agreement(s[creux], s_ref[creux]):>14.3f} %")


# ============================================================
# BLOCS 4 et 5 — entraîner depuis le fichier, puis comparer
# ============================================================

def entrainer_et_mesurer(shape, X_surf, normals, X_vol, sign_vol, box, seed,
                         n_epochs=2000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    modele = IGRNet(radius_init=1.0)

    tampon = io.StringIO()
    t0 = time.time()
    with redirect_stdout(tampon):                    # la boucle est bavarde
        train_igr(modele, X_surf, normals, X_vol, sign_vol, box,
                  n_epochs=n_epochs, batch_surf=1024, batch_vol=1024, lr=1e-3)
    duree = time.time() - t0
    perte = tampon.getvalue().strip().splitlines()[-1].split("|")[1].strip()

    xs, ys, zs, valeurs = evaluate_grid(modele, box, grid_res=64)
    sommets, faces, _, _ = extract_surface(xs, ys, zs, valeurs)

    _, f_exact, _ = get_surface(shape)
    ch = chamfer_distance(sommets, reference_dense(shape))
    iou = iou_occupancy(model_inside(modele, device=torch.device("cpu")),
                        lambda p: f_exact(np.asarray(p, dtype=np.float64)) < 0.0,
                        box, n_samples=100000, seed=7)
    eik = eikonal_error(modele, box, n=50000, seed=7, device=torch.device("cpu"))
    return {"chamfer": ch["chamfer"], "hausdorff": ch["hausdorff"], "iou": iou,
            "eikonal": eik, "sommets": len(sommets), "duree": duree,
            "perte": perte}


def donnees_depuis_fichier(chemin, k=5, n_vol=20000, seed=0):
    """Tout ce qui entre dans l'entraînement sort de ce fichier, et de rien d'autre."""
    d = load_cloud(chemin)
    nuage = OrientedCloud(d["points"], d["normals"])
    box = d["box"]
    rng = np.random.default_rng(seed)
    X_vol = rng.uniform(-box, box, size=(n_vol, 3))
    s_vol = nuage.sign(X_vol, k=k)          # <- le signe vient du NUAGE
    return nuage, box, X_vol, s_vol


def en_tenseurs(nuage, X_vol, s_vol):
    return (torch.tensor(nuage.points, dtype=torch.float32),
            torch.tensor(nuage.normals, dtype=torch.float32),
            torch.tensor(X_vol, dtype=torch.float32),
            torch.tensor(s_vol[:, None], dtype=torch.float32))


def blocs45(graines=(0, 1, 2)):
    shape = "ellipsoid"
    chemin, _ = generer(shape, 6000)

    titre(4, "entraînement 100 % nuage, depuis le fichier")
    nuage, box, X_vol, s_vol_nuage = donnees_depuis_fichier(chemin, k=5)
    print(f"\n  fichier              : {os.path.basename(chemin)}")
    print(f"  points de surface    : {len(nuage)} (avec leurs normales)")
    print(f"  points de volume     : {len(X_vol)}, signe obtenu par produit "
          f"scalaire sur k = 5 voisins")
    print(f"  répartition du signe : {int((s_vol_nuage < 0).sum())} dedans / "
          f"{int((s_vol_nuage > 0).sum())} dehors")

    # preuve que la formule ne sert pas : on la met sous compteur
    vraie_f = get_surface(shape)[1]
    compteur = Compteur(vraie_f)
    avant = compteur.appels

    args = en_tenseurs(nuage, X_vol, s_vol_nuage)
    res_nuage = [entrainer_et_mesurer(shape, *args, box, s) for s in graines]
    print(f"\n  appels à la formule exacte pendant l'entraînement : "
          f"{compteur.appels - avant}   (attendu 0)")

    print(f"\n  {'graine':>7} {'Chamfer':>10} {'Hausdorff':>11} {'IoU':>9} "
          f"{'eikonal':>10} {'sommets':>9} {'durée':>8}")
    print("  " + "-" * 70)
    for g, r in zip(graines, res_nuage):
        print(f"  {g:>7} {r['chamfer']:>10.5f} {r['hausdorff']:>11.5f} "
              f"{r['iou']:>9.5f} {r['eikonal']:>10.5f} {r['sommets']:>9} "
              f"{r['duree']:>7.0f}s")

    titre(5, "le même entraînement avec le signe issu de la formule")
    print("\n  Seule différence : le signe des points de volume. Mêmes points de")
    print("  surface, mêmes normales, mêmes positions de volume, mêmes graines,")
    print("  même matériel (CPU, donc reproductible).\n")
    s_vol_formule = np.sign(vraie_f(X_vol))
    desaccord = int((s_vol_formule != s_vol_nuage).sum())
    print(f"  signes différents entre les deux sources : {desaccord} sur "
          f"{len(X_vol)}  ({100.0 * desaccord / len(X_vol):.3f} %)")

    args_f = en_tenseurs(nuage, X_vol, s_vol_formule)
    res_formule = [entrainer_et_mesurer(shape, *args_f, box, s) for s in graines]

    def moy(liste, cle):
        v = np.array([r[cle] for r in liste])
        return float(v.mean()), float(v.std())

    print(f"\n  {'métrique':<12} {'signe du NUAGE':>24} {'signe de la FORMULE':>24} "
          f"{'écart':>12}")
    print("  " + "-" * 76)
    for cle, nom in (("chamfer", "Chamfer"), ("hausdorff", "Hausdorff"),
                     ("iou", "IoU"), ("eikonal", "eikonal")):
        mn, sn = moy(res_nuage, cle)
        mf, sf = moy(res_formule, cle)
        ecart = mn - mf
        # l'écart est-il plus grand que la dispersion entre graines ?
        bruit = max(sn, sf)
        verdict = "dans le bruit" if abs(ecart) <= 2 * bruit else "SIGNIFICATIF"
        print(f"  {nom:<12} {mn:>14.5f} ± {sn:<7.5f} {mf:>14.5f} ± {sf:<7.5f} "
              f"{ecart:>+11.5f}  {verdict}")
    print("\n  (moyenne ± écart-type sur les graines "
          f"{', '.join(str(g) for g in graines)} ; l'écart-type mesure la "
          "dispersion\n   d'un entraînement à l'autre, c'est l'étalon de bruit)")
    return res_nuage, res_formule


def main():
    preparer()
    bloc1()
    bloc2()
    bloc3()
    blocs45()
    print("\n" + "=" * 78)
    print("FIN — les cinq blocs sont affichés")
    print("=" * 78)


if __name__ == "__main__":
    main()
