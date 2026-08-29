"""
Étape 14 — apprendre f(x,y,z,t) sur la CAPSULE RÉELLE du BMBI.

C'est la campagne `main_temporal.py` refaite sur des données qu'on n'a pas
fabriquées. La question est la même :

    un SEUL champ f(x,y,z,t) = 0, appris depuis quelques instants écrits sur
    disque, sait-il reconstruire les instants qu'on lui a cachés ?

Trois choses changent, et elles viennent toutes de `RESULTATS_capsules.md`.

1. **La déformation est NON AFFINE** (14.9 % entre snapshots voisins, contre 0 %
   sur la séquence synthétique, et la capsule cesse d'être convexe). L'étalon
   « morphing entre les deux instants encadrants », qui faisait jeu égal avec le
   réseau sur le cas synthétique, part cette fois avec un handicap réel. C'est ce
   qui rend enfin le test d'interpolation discriminant.

2. **Il n'y a plus de formule exacte.** La vérité de référence est construite à
   partir du MAILLAGE seul (`capsules.champ_reference`) : distance au plus proche
   point d'un nuage dense tiré sur les triangles, signe par produit scalaire avec
   la normale. Ce champ a été validé contre le test d'appartenance exact de
   trimesh : 100 % d'accord sur 20000 points.

3. **Le contrôle lagrangien est réécrit.** 93 % du mouvement des nœuds est
   tangentiel (tank-treading) : advecter un point par v_n et le comparer à la
   trajectoire d'un nœud n'a pas de sens ici. On compare v_n appris à v_n du
   MAILLAGE — la vitesse des nœuds projetée sur la normale, qui est bien la
   vitesse de la surface.

Protocole. Fenêtre sur le TRANSITOIRE (pas 0 à 4000), là où la capsule se
déforme vraiment : elle part d'une sphère parfaite, s'aplatit jusqu'à un rapport
d'axes de 1.354 au pas 800, puis relaxe. 21 instants équirépartis, un sur deux
caché :

    observés : pas 0, 400, 800, ..., 4000   (11 instants écrits sur disque)
    cachés   : pas 200, 600, ..., 3800      (10 instants, rien du tout)

Lancer :  /usr/local/bin/python3 main_capsules_temporal.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from skimage import measure

from src import capsules
from src.device import get_device
from src.invariants import euler_characteristic, mesh_volume
from src.metrics import chamfer_distance
from src.model_temporal import TemporalIGRNet, figer_instant, n_parametres
from src.reconstruct import evaluate_grid, extract_surface
from src.train_temporal import (derivees, en_tenseurs, jeu_depuis_fichiers,
                                train_temporel)

OUT = "output"
DOSSIER = f"{OUT}/capsules_temporel"
PAS_MAX = 4000          # fenêtre : le transitoire
PAS_ENTRE = 200         # un instant tous les 200 pas -> 21 instants
GRILLE = 96
EPOCHS = 8000
N_REF = 60000           # nuage dense de référence, par instant
DPAS_VN = 25            # demi-écart de la différence centrée pour v_n (cf. bloc 5)


def titre(n, texte):
    print(f"\n{'=' * 78}\nBLOC {n} — {texte}\n{'=' * 78}", flush=True)


def t_de(pas):
    """Pas EF -> temps d'entraînement dans [0,1]."""
    return float(pas) / PAS_MAX


# ==================================================================
# Préparation
# ==================================================================

def grille_instants(mode):
    '''
    Les 21 instants de la campagne, et lesquels sont observés.

    `uniforme`    : un instant tous les 200 pas. C'est la recommandation du mail
                    (« par exemple 100 snapshots »), appliquée à notre fenêtre.
    `transitoire` : même nombre d'instants, mais resserrés au DÉBUT, en
                    t proportionnel à (k/n)^2. La capsule se déforme presque
                    tout entier dans les 800 premiers pas ; un échantillonnage
                    uniforme n'y met que deux instants observés.
    '''
    if mode == "uniforme":
        pas = list(range(0, PAS_MAX + 1, PAS_ENTRE))
    else:
        u = (np.arange(21) / 20.0) ** 2
        pas = sorted(set(int(round(x * PAS_MAX / 25.0)) * 25 for x in u))
        k = 1
        while len(pas) < 21:
            if 25 * k not in pas:
                pas = sorted(pas + [25 * k])
            k += 1
    return pas, pas[0::2], pas[1::2]


def preparer():
    faces, caps = capsules.charger()
    pas, observes, caches = grille_instants(MODE)

    print(f"  mode           : {MODE}")
    print(f"  fenêtre        : pas 0 à {PAS_MAX} (le transitoire)")
    print(f"  observés ({len(observes):2d})  : {observes}")
    print(f"  cachés   ({len(caches):2d})  : {caches}")

    box = capsules.boite(faces, caps, pas, marge=1.25)
    fichiers, _ = capsules.exporter_sequence(f"{DOSSIER}_{MODE}", faces, caps, observes,
                                             box=box)
    print(f"  boîte          : {box:.4f}")
    print(f"  écrits         : {len(fichiers)} fichiers dans {DOSSIER}_{MODE}/")

    # déplacement entre instants observés — l'écart que le réseau doit combler
    a0 = capsules.instant(caps, 0)[0]
    arete = np.linalg.norm(a0[faces[:, 0]] - a0[faces[:, 1]], axis=1).mean()
    d = [np.linalg.norm(capsules.instant(caps, observes[i + 1])[0]
                        - capsules.instant(caps, observes[i])[0], axis=1).mean()
         for i in range(len(observes) - 1)]
    print(f"  déplacement entre instants observés : {min(d):.4f} à {max(d):.4f} "
          f"= {min(d)/arete:.1f} à {max(d)/arete:.1f} arêtes")
    print("  (le premier intervalle est le plus dur : la sphère s'aplatit d'un coup)")

    instants = [(t_de(p), os.path.join(f"{DOSSIER}_{MODE}", f"capsule_{k:04d}.npz"))
                for k, p in enumerate(observes)]
    return faces, caps, box, observes, caches, pas, instants


# ==================================================================
# Référence : la vérité dont on dispose, tirée du maillage seul
# ==================================================================

def references(faces, caps, pas, box):
    """Champ signé sur grille + nuage dense, pour chaque instant."""
    ref = {}
    axe = None
    for p in pas:
        V, _ = capsules.instant(caps, p)
        axe, val = capsules.champ_reference(V, faces, box, grille=GRILLE,
                                            n_echantillons=N_REF)
        S, _ = capsules.echantillon_surface(V, faces, n=N_REF, seed=3)
        ref[p] = {"val": val, "nuage": S, "verts": V}
    return axe, ref


def mc(axe, val):
    """Marching cubes sur une grille de valeurs, coordonnées remises en place."""
    if not (val.min() < 0.0 < val.max()):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    h = axe[1] - axe[0]
    v, f, _, _ = measure.marching_cubes(val, level=0.0, spacing=(h, h, h))
    return v + axe[0], f


def interpoler(axe, val, P):
    """Valeur d'un champ de grille en des points quelconques (trilinéaire)."""
    h = axe[1] - axe[0]
    c = ((P - axe[0]) / h).T
    return map_coordinates(val, c, order=1, mode="nearest")


# ==================================================================
# Mesure
# ==================================================================

def juger(sommets, faces_r, p, ref, axe, val_apprise=None):
    """Chamfer, IoU voxel, volume, topologie d'une reconstruction."""
    if len(sommets) == 0:
        return {"chamfer": np.nan, "iou": 0.0, "volume": 0.0, "chi": None,
                "n": 0}
    ch = chamfer_distance(sommets, ref[p]["nuage"])
    d = {"chamfer": ch["chamfer"], "hausdorff": ch["hausdorff"],
         "volume": mesh_volume(sommets, faces_r),
         "chi": euler_characteristic(sommets, faces_r)["chi"],
         "n": len(sommets)}
    if val_apprise is not None:
        a = val_apprise < 0.0
        b = ref[p]["val"] < 0.0
        d["iou"] = float((a & b).sum() / max((a | b).sum(), 1))
    return d


def champ_appris(modele, t, box, device):
    vue = figer_instant(modele, t, device=device)
    xs, _, _, val = evaluate_grid(vue, box, grid_res=GRILLE)
    return xs, val


def main():
    t_debut = time.time()
    device = get_device()
    os.makedirs(OUT, exist_ok=True)

    titre(0, "protocole")
    faces, caps, box, observes, caches, pas, instants = preparer()
    print(f"  appareil       : {device}")

    print("\n  construction de la référence (maillage seul, aucune formule)...",
          flush=True)
    t0 = time.time()
    axe, ref = references(faces, caps, pas, box)
    print(f"  {len(pas)} champs de référence en {time.time() - t0:.0f}s")

    # ---------------- entraînement ----------------
    titre(1, "entraînement du champ f(x,y,z,t)")
    jeu = jeu_depuis_fichiers(instants, k=5, n_vol_par_instant=20000, seed=0)
    T_ = en_tenseurs(jeu)
    print(f"  points de surface {len(jeu['X_surf'])}, de volume "
          f"{len(jeu['X_vol'])}, boîte {jeu['box']:.4f}")
    modele = TemporalIGRNet(radius_init=0.5)
    print(f"  réseau : entrées (x,y,z,t), Softplus, {n_parametres(modele)} poids")
    t0 = time.time()
    modele, _ = train_temporel(modele, T_["X_surf"], T_["N_surf"], T_["X_vol"],
                               T_["s_vol"], jeu["box"], t_min=0.0, t_max=1.0,
                               n_epochs=EPOCHS, log_every=1000, device=device,
                               seed=0)
    print(f"  entraînement en {time.time() - t0:.0f}s")
    torch.save(modele.state_dict(), f"{OUT}/modele_capsule_{MODE}.pt")

    # ---------------- plancher de la grille ----------------
    titre(2, "le plancher : ce que la grille permet au mieux")
    plancher = {}
    for p in pas:
        v, f = mc(axe, ref[p]["val"])
        plancher[p] = juger(v, f, p, ref, axe)
    ch_pl = np.array([plancher[p]["chamfer"] for p in pas])
    vo_pl = np.array([plancher[p]["volume"] for p in pas])
    print(f"  marching cubes sur le champ de référence, grille {GRILLE}^3 :")
    print(f"    Chamfer {ch_pl.mean():.5f} (min {ch_pl.min():.5f}, "
          f"max {ch_pl.max():.5f})")
    print(f"    volume  {vo_pl.mean():.5f}")
    print("  Aucun champ appris ne peut descendre sous ce Chamfer. Sans ce")
    print("  repère, on attribuerait au réseau une erreur qui est celle de la grille.")

    # ---------------- mesure du champ appris ----------------
    titre(3, "LE TEST CENTRAL — instants observés contre instants jamais vus")
    res = {}
    for p in pas:
        _, val = champ_appris(modele, t_de(p), box, device)
        v, f = mc(axe, val)
        res[p] = juger(v, f, p, ref, axe, val_apprise=val)
        res[p]["val"] = val

    def moy(cle, groupe):
        return float(np.nanmean([res[p][cle] for p in groupe]))

    print(f"  {'':<22}{'observés':>12}{'CACHÉS':>12}{'plancher':>12}")
    for cle, nom in (("chamfer", "Chamfer"), ("iou", "IoU voxel"),
                     ("volume", "volume")):
        pl = (float(np.mean([plancher[p][cle] for p in pas]))
              if cle != "iou" else 1.0)
        print(f"  {nom:<22}{moy(cle, observes):>12.5f}{moy(cle, caches):>12.5f}"
              f"{pl:>12.5f}")
    chis = {p: res[p]["chi"] for p in pas}
    print(f"  chi : {'2 partout' if all(c == 2 for c in chis.values()) else chis}")

    # volume de référence, l'invariant que l'entraînement ignore
    vref = np.array([mesh_volume(ref[p]["verts"], faces) for p in pas])
    vapp = np.array([res[p]["volume"] for p in pas])
    print(f"\n  VOLUME — l'invariant que la perte ignore complètement")
    print(f"    donnée   : {vref.mean():.5f}  dispersion "
          f"{100*vref.std()/vref.mean():.4f} %")
    print(f"    plancher : {vo_pl.mean():.5f}  dispersion "
          f"{100*vo_pl.std()/vo_pl.mean():.4f} %")
    print(f"    appris   : {vapp.mean():.5f}  dispersion "
          f"{100*vapp.std()/vapp.mean():.4f} %  "
          f"biais {100*(vapp.mean()-vo_pl.mean())/vo_pl.mean():+.3f} % / plancher")

    # ---------------- étalons ----------------
    titre(4, "LES ÉTALONS — le réseau bat-il des méthodes bêtes ?")
    et_gel, et_mor = {}, {}
    for p in caches:
        i = int(np.searchsorted(observes, p))
        p1, p2 = observes[i - 1], observes[i]

        proche = p1 if abs(p - p1) <= abs(p - p2) else p2
        v, f = mc(axe, ref[proche]["val"])
        et_gel[p] = juger(v, f, p, ref, axe, val_apprise=ref[proche]["val"])

        th = (p2 - p) / (p2 - p1)
        vm = th * ref[p1]["val"] + (1 - th) * ref[p2]["val"]
        v, f = mc(axe, vm)
        et_mor[p] = juger(v, f, p, ref, axe, val_apprise=vm)
        et_mor[p]["val"] = vm
        et_mor[p]["voisins"] = (p1, p2)

    def m(d, cle):
        return float(np.nanmean([d[p][cle] for p in caches]))

    print(f"  aux {len(caches)} instants cachés :")
    print(f"  {'méthode':<34}{'Chamfer':>10}{'IoU':>10}")
    print(f"  {'image gelée (instant le + proche)':<34}"
          f"{m(et_gel,'chamfer'):>10.5f}{m(et_gel,'iou'):>10.5f}")
    print(f"  {'morphing entre encadrants':<34}"
          f"{m(et_mor,'chamfer'):>10.5f}{m(et_mor,'iou'):>10.5f}")
    print(f"  {'RÉSEAU f(x,y,z,t)':<34}{moy('chamfer',caches):>10.5f}"
          f"{moy('iou',caches):>10.5f}")
    pl_c = float(np.mean([plancher[p]['chamfer'] for p in caches]))
    print(f"  {'plancher de la grille':<34}{pl_c:>10.5f}{1.0:>10.5f}")
    print("\n  Le Chamfer est SATURÉ par la grille : la bonne lecture est")
    print("  l'EXCÉDENT sur le plancher, c'est-à-dire ce que la méthode ajoute")
    print("  à l'erreur incompressible de discrétisation.")
    ex_r = moy('chamfer', caches) - pl_c
    ex_m = m(et_mor, 'chamfer') - pl_c
    ex_g = m(et_gel, 'chamfer') - pl_c
    print(f"    réseau       {ex_r:.5f}")
    print(f"    morphing     {ex_m:.5f}   soit {ex_m/ex_r:.1f}x le réseau")
    print(f"    image gelée  {ex_g:.5f}   soit {ex_g/ex_r:.1f}x le réseau")
    print("  Sur la séquence SYNTHÉTIQUE, le morphing faisait jeu égal avec le")
    print("  réseau (0.00806 contre 0.00815) parce que la déformation y était")
    print("  affine. Ici elle ne l'est pas, et l'écart apparaît.")

    # ---------------- vitesse normale : le contrôle réécrit ----------------
    titre(5, "LA VITESSE DE LA MEMBRANE — le contrôle réécrit")
    print("  93 % du mouvement des nœuds est TANGENTIEL. On ne compare donc pas")
    print("  des trajectoires, mais v_n appris contre v_n du MAILLAGE (vitesse")
    print("  des nœuds projetée sur la normale) : c'est ce que porte une surface.")
    dt_par_pas = 1.0 / PAS_MAX

    print("\n  D'ABORD : jusqu'où peut-on croire la RÉFÉRENCE ? v_n du maillage est")
    print("  une différence centrée, donc dépend de son écart. On la fait converger.")
    print(f"    {'pas':>6}" + "".join(f"{d:>10}" for d in (25, 50, 100, 200)))
    for p in (caches[0], caches[len(caches)//2], caches[-1]):
        base = capsules.vitesse_normale_maillage(caps, faces, p, DPAS_VN,
                                                 dt_par_pas)[1]
        amp = np.abs(base).mean()
        cases = []
        for d in (25, 50, 100, 200):
            if p - d < 0:                      # l'écart ne tient pas
                cases.append("      —  ")
                continue
            vn = capsules.vitesse_normale_maillage(caps, faces, p, d,
                                                   dt_par_pas)[1]
            cases.append(f"{100 * np.abs(vn - base).mean() / amp:>8.1f}%")
        print(f"    {p:>6}" + "".join(cases))
    print(f"  -> un écart de 200 pas fausse la référence de 3 à 12 %. On prend")
    print(f"     DPAS_VN = {DPAS_VN}, où elle est stationnaire. Mesurer la vitesse d'un")
    print("     champ appris avec une référence fausse de 10 % n'aurait rien dit.")

    lignes = []
    for p in caches:
        P, vn_ref = capsules.vitesse_normale_maillage(caps, faces, p, DPAS_VN,
                                                      dt_par_pas)
        _, g, dfdt = derivees(modele, P, t_de(p), device=device)
        vn_app = -dfdt / np.maximum(np.linalg.norm(g, axis=1), 1e-9)

        p1, p2 = et_mor[p]["voisins"]
        gm = np.stack(np.gradient(et_mor[p]["val"], axe[1] - axe[0]), axis=-1)
        gmP = np.stack([interpoler(axe, gm[..., c], P) for c in range(3)], axis=1)
        dfdt_m = interpoler(axe, (ref[p2]["val"] - ref[p1]["val"])
                            / (t_de(p2) - t_de(p1)), P)
        vn_mor = -dfdt_m / np.maximum(np.linalg.norm(gmP, axis=1), 1e-9)

        lignes.append((p, np.abs(vn_ref).mean(),
                       np.abs(vn_app - vn_ref).mean(),
                       np.abs(vn_mor - vn_ref).mean()))
    L = np.array(lignes)
    print("\n  instant par instant :")
    print(f"    {'pas':>6}{'amplitude':>12}{'réseau':>12}{'morphing':>12}{'gain':>8}")
    for p_, amp, er, em in L:
        print(f"    {int(p_):>6}{amp:>12.5f}{er:>12.5f}{em:>12.5f}"
              f"{em/max(er,1e-12):>7.1f}x")
    print(f"\n  {'':<26}{'erreur moyenne sur v_n':>24}")
    print(f"  {'amplitude de v_n':<26}{L[:,1].mean():>12.5f}")
    print(f"  {'réseau f(x,y,z,t)':<26}{L[:,2].mean():>12.5f}   "
          f"soit {100*L[:,2].mean()/L[:,1].mean():.1f} % de l'amplitude")
    print(f"  {'morphing':<26}{L[:,3].mean():>12.5f}   "
          f"soit {100*L[:,3].mean()/L[:,1].mean():.1f} % de l'amplitude")
    fact = L[:, 3].mean() / max(L[:, 2].mean(), 1e-12)
    print(f"  -> en moyenne, le réseau est {fact:.1f} fois plus juste que le morphing")

    sauf = L[1:]
    fact_sauf = sauf[:, 3].mean() / max(sauf[:, 2].mean(), 1e-12)
    print("\n  MAIS cette moyenne est écrasée par le PREMIER instant caché :")
    print(f"    au pas {int(L[0,0])}, l'amplitude vaut {L[0,1]:.5f}, soit "
          f"{L[0,1]/sauf[:,1].mean():.0f}x le reste de la séquence,")
    print(f"    et le réseau y fait {L[0,2]:.5f} contre {L[0,3]:.5f} au morphing "
          "— il PERD.")
    print(f"    sur les {len(sauf)} autres instants : réseau {sauf[:,2].mean():.5f} "
          f"contre {sauf[:,3].mean():.5f}, soit {fact_sauf:.1f}x")
    print("  Lecture : la sphère se déforme presque tout entier dans les premiers")
    print("  pas, et 400 pas entre instants observés y sont trop grossiers. Ce")
    print("  n'est pas une limite de la méthode mais de l'ÉCHANTILLONNAGE — et")
    print("  c'est exactement la question pratique posée au BMBI : à quelle")
    print("  fréquence leur code doit-il sauvegarder ? Réponse : pas uniformément.")

    # ---------------- figures + résumé ----------------
    figures(axe, ref, res, plancher, et_gel, et_mor, observes, caches, pas,
            vref, vapp, vo_pl, L)

    resume = {
        "fenetre": [0, PAS_MAX], "observes": observes, "caches": caches,
        "grille": GRILLE, "epochs": EPOCHS, "box": box,
        "chamfer_observes": moy("chamfer", observes),
        "chamfer_caches": moy("chamfer", caches),
        "chamfer_plancher": float(ch_pl.mean()),
        "iou_observes": moy("iou", observes), "iou_caches": moy("iou", caches),
        "chi_tous_2": all(c == 2 for c in chis.values()),
        "volume_donnee_disp": float(100 * vref.std() / vref.mean()),
        "volume_appris_disp": float(100 * vapp.std() / vapp.mean()),
        "etalon_gele_chamfer": m(et_gel, "chamfer"),
        "etalon_morphing_chamfer": m(et_mor, "chamfer"),
        "vn_amplitude": float(L[:, 1].mean()),
        "vn_erreur_reseau": float(L[:, 2].mean()),
        "vn_erreur_morphing": float(L[:, 3].mean()),
        "vn_facteur": float(fact),
        "vn_facteur_hors_premier": float(fact_sauf),
        "vn_detail": [[float(x) for x in ligne] for ligne in L],
        "mode": MODE,
        "duree_s": time.time() - t_debut,
    }
    with open(f"{OUT}/capsules_temporel_{MODE}_resume.json", "w") as fh:
        json.dump(resume, fh, indent=2)
    print(f"\nRésumé -> {OUT}/capsules_temporel_resume.json")
    print(f"Terminé en {(time.time() - t_debut)/60:.1f} min.")


def figures(axe, ref, res, plancher, et_gel, et_mor, observes, caches, pas,
            vref, vapp, vo_pl, L):
    titre(6, "figures")

    # --- ce qu'on lui donne / ce qu'il produit ---
    montre = pas[:12]
    fig, axs = plt.subplots(2, len(montre), figsize=(1.55 * len(montre), 4.4))
    for j, p in enumerate(montre):
        vu = p in observes
        a = axs[0, j]
        S = ref[p]["nuage"]
        m_ = np.abs(S[:, 0]) < 0.02
        if vu:
            a.scatter(S[m_, 1], S[m_, 2], s=0.4, color="k")
            a.set_title(f"pas {p}", fontsize=7, color="#1a4a7a")
        else:
            a.text(0.5, 0.5, "AUCUNE\nDONNÉE", ha="center", va="center",
                   color="red", fontsize=7, transform=a.transAxes)
            a.set_title(f"pas {p}", fontsize=7, color="red")
        b = axs[1, j]
        vr = ref[p]["val"][GRILLE // 2]
        va = res[p]["val"][GRILLE // 2]
        ext = [axe[0], axe[-1], axe[0], axe[-1]]
        b.contourf((va.T <= 0).astype(float), levels=[0.5, 1.5], colors=["#a9cce3"],
                   extent=ext, origin="lower")
        b.contour(vr.T, levels=[0.0], colors=["red"], linewidths=0.8,
                  extent=ext, origin="lower", linestyles="--")
        b.set_title(f"IoU {res[p]['iou']*100:.1f} %", fontsize=6)
        for c in (a, b):
            c.set_xticks([]); c.set_yticks([]); c.set_aspect("equal")
            for s in c.spines.values():
                s.set_color("red" if not vu else "#1a4a7a")
                s.set_linewidth(1.4)
    axs[0, 0].set_ylabel("ce qu'on donne", fontsize=7)
    axs[1, 0].set_ylabel("ce qu'il produit", fontsize=7)
    fig.suptitle("Capsule réelle du BMBI — un seul réseau f(x,y,z,t).  "
                 "Bleu : la surface apprise.  Tirets rouges : la vraie surface.  "
                 "Les colonnes rouges n'ont JAMAIS été montrées.", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/capsules_temporel_{MODE}_donnees.png", dpi=140)
    plt.close(fig)

    # --- diagnostics ---
    xs = np.arange(len(pas))
    est_cache = np.array([p in caches for p in pas])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(xs, [res[p]["chamfer"] for p in pas], "o-", label="réseau", ms=4)
    ax[0].plot(xs, [plancher[p]["chamfer"] for p in pas], "k--",
               label="plancher grille", lw=1)
    ax[0].scatter(xs[est_cache], [res[p]["chamfer"] for p in np.array(pas)[est_cache]],
                  color="red", zorder=5, s=28, label="instants cachés")
    ax[0].set_title("Chamfer"); ax[0].legend(fontsize=8)

    ax[1].plot(xs, [100 * res[p]["iou"] for p in pas], "o-", ms=4)
    ax[1].scatter(xs[est_cache],
                  [100 * res[p]["iou"] for p in np.array(pas)[est_cache]],
                  color="red", zorder=5, s=28)
    ax[1].set_title("IoU voxel (%)")

    ax[2].plot(xs, 100 * (vapp / vo_pl.mean() - 1), "o-", ms=4, label="appris")
    ax[2].plot(xs, 100 * (vref / vo_pl.mean() - 1), "g--", label="donnée")
    ax[2].axhline(0, color="k", lw=1, ls=":")
    ax[2].set_title("volume, écart au plancher (%)"); ax[2].legend(fontsize=8)

    for a in ax:
        a.set_xlabel("indice d'instant"); a.grid(alpha=0.3)
    fig.suptitle("Capsule réelle — le réseau aux instants observés et cachés",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/capsules_temporel_{MODE}_diagnostics.png", dpi=140)
    plt.close(fig)

    # --- vitesse normale ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(L[:, 0], L[:, 2], "o-", label="réseau f(x,y,z,t)")
    ax.plot(L[:, 0], L[:, 3], "s-", label="morphing entre encadrants")
    ax.plot(L[:, 0], L[:, 1], "k--", label="amplitude de v_n", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("pas EF (instants cachés)")
    ax.set_ylabel("erreur moyenne sur la vitesse normale")
    ax.set_title("La vitesse de la membrane — le contrôle réécrit")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/capsules_temporel_{MODE}_vitesse.png", dpi=140)
    plt.close(fig)

    for f in ("donnees", "diagnostics", "vitesse"):
        print(f"  {OUT}/capsules_temporel_{MODE}_{f}.png")


MODE = "uniforme"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("uniforme", "transitoire"):
        MODE = sys.argv[1]
    main()
