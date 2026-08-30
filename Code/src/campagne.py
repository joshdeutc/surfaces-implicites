"""
La campagne « combien d'instants faut-il ? », partagée par les deux régimes.

Ce module existe pour une raison de méthode, pas de confort : les deux campagnes
(l'événement de déformation, puis le tank-treading) ne valent que si elles
suivent le MÊME protocole. Le seul moyen d'en être sûr est qu'elles exécutent le
même code. Tout ce qui les distingue est donc un paramètre — la fenêtre de pas de
temps — et rien d'autre ne peut diverger par accident.

Le protocole, identique pour toute fenêtre :

  - 21 instants d'ÉVALUATION équirépartis dans la fenêtre. C'est sur eux, et sur
    eux seuls, que toutes les configurations sont jugées.
  - une configuration = un nombre d'instants OBSERVÉS, pris dans cette grille.
    Les autres sont cachés : le réseau n'en reçoit rien.
  - la vérité de référence vient du MAILLAGE seul (`capsules.champ_reference`),
    jamais d'une formule. Elle donne aussi le PLANCHER : marching cubes appliqué
    dessus, c'est-à-dire l'erreur de discrétisation pure, sous laquelle aucun
    champ appris ne peut descendre.
  - deux étalons aux mêmes instants cachés : l'image gelée (rendre l'instant
    observé le plus proche) et le morphing entre les deux instants encadrants.
  - la vitesse normale est comparée à celle du MAILLAGE, par différence centrée
    étroite. Jamais à une trajectoire de nœud : le mouvement est à 93 %
    tangentiel, et une surface implicite ne porte que le mouvement normal.
"""
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import measure

from src import capsules
from src.device import get_device
from src.invariants import euler_characteristic, mesh_volume
from src.metrics import chamfer_distance
from src.model_temporal import TemporalIGRNet, figer_instant
from src.reconstruct import evaluate_grid
from src.train_temporal import (derivees, en_tenseurs, jeu_depuis_fichiers,
                                train_temporel)

OUT = "output"
N_EVAL = 21             # instants d'évaluation
GRILLE = 96
EPOCHS = 8000
N_REF = 60000
DPAS_VN = 25            # demi-écart de la différence centrée (cf. RESULTATS_capsules_temporel §5)
CONFIGS = [11, 6, 5, 3]


def titre(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def mc(axe, val):
    """Marching cubes sur une grille de valeurs, coordonnées remises en place."""
    if not (val.min() < 0.0 < val.max()):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    h = axe[1] - axe[0]
    v, f, _, _ = measure.marching_cubes(val, level=0.0, spacing=(h, h, h))
    return v + axe[0], f


def juger(sommets, faces_r, p, ref, val=None):
    if len(sommets) == 0:
        return {"chamfer": np.nan, "iou": 0.0, "volume": 0.0, "chi": None}
    ch = chamfer_distance(sommets, ref[p]["nuage"])
    d = {"chamfer": ch["chamfer"], "volume": mesh_volume(sommets, faces_r),
         "chi": euler_characteristic(sommets, faces_r)["chi"]}
    if val is not None:
        a, b = val < 0.0, ref[p]["val"] < 0.0
        d["iou"] = float((a & b).sum() / max((a | b).sum(), 1))
    return d


def campagne(nom, pas_min, pas_max, configs=None, epochs=EPOCHS, grille=GRILLE):
    """
    Exécute la campagne complète sur une fenêtre, et renvoie le résumé.

    `nom` sert de préfixe aux sorties ; `pas_min`/`pas_max` bornent la fenêtre.
    Le temps d'entraînement est normalisé dans [0,1] sur la fenêtre.
    """
    configs = configs or CONFIGS
    t0_global = time.time()
    device = get_device()
    os.makedirs(OUT, exist_ok=True)
    faces, caps = capsules.charger()

    etendue = pas_max - pas_min
    pas_eval = int(round(etendue / (N_EVAL - 1)))
    evalu = [pas_min + k * pas_eval for k in range(N_EVAL)]

    def t_de(p):
        return float(p - pas_min) / etendue

    titre(f"CAMPAGNE « {nom} » — fenêtre pas {pas_min} à {pas_max}")
    for p in (evalu[0], evalu[N_EVAL // 2], evalu[-1]):
        V, _ = capsules.instant(caps, p)
        print(f"  pas {p:6d} : rapport d'axes {capsules.demi_axes(V)[1]:.3f}")
    a0 = capsules.instant(caps, evalu[0])[0]
    arete = np.linalg.norm(a0[faces[:, 0]] - a0[faces[:, 1]], axis=1).mean()
    dep = [np.linalg.norm(capsules.instant(caps, evalu[i + 1])[0]
                          - capsules.instant(caps, evalu[i])[0], axis=1).mean()
           for i in range(len(evalu) - 1)]
    print(f"  {N_EVAL} instants d'évaluation, un tous les {pas_eval} pas")
    print(f"  déplacement entre eux : {min(dep):.4f} à {max(dep):.4f} = "
          f"{min(dep)/arete:.2f} à {max(dep)/arete:.2f} arêtes")

    box = capsules.boite(faces, caps, evalu, marge=1.25)
    print(f"  boîte : {box:.4f}")

    titre("LA RÉFÉRENCE — depuis le maillage seul, une fois pour toutes")
    t0 = time.time()
    ref, axe = {}, None
    for p in evalu:
        V, _ = capsules.instant(caps, p)
        axe, val = capsules.champ_reference(V, faces, box, grille=grille,
                                            n_echantillons=N_REF)
        S, _ = capsules.echantillon_surface(V, faces, n=N_REF, seed=3)
        ref[p] = {"val": val, "nuage": S, "verts": V}
    plancher = {p: juger(*mc(axe, ref[p]["val"]), p, ref, ref[p]["val"])
                for p in evalu}
    pl = float(np.mean([plancher[p]["chamfer"] for p in evalu]))
    print(f"  {len(evalu)} champs en {time.time()-t0:.0f}s")
    print(f"  PLANCHER de la grille {grille}^3 : Chamfer {pl:.5f}")

    resultats = []
    for n_obs in configs:
        pas_obs = evalu[::(N_EVAL - 1) // (n_obs - 1)]
        caches = [p for p in evalu if p not in pas_obs]
        titre(f"[{nom}] {n_obs} INSTANTS OBSERVÉS (un tous les "
              f"{pas_obs[1]-pas_obs[0]} pas) -> {len(caches)} cachés")
        print(f"  observés : {pas_obs}")

        dossier = f"{OUT}/campagne_{nom}_{n_obs}"
        capsules.exporter_sequence(dossier, faces, caps, pas_obs, box=box)
        instants = [(t_de(p), os.path.join(dossier, f"capsule_{k:04d}.npz"))
                    for k, p in enumerate(pas_obs)]

        jeu = jeu_depuis_fichiers(instants, k=5, n_vol_par_instant=20000, seed=0)
        T_ = en_tenseurs(jeu)
        modele = TemporalIGRNet(radius_init=0.5)
        t0 = time.time()
        modele, _ = train_temporel(modele, T_["X_surf"], T_["N_surf"],
                                   T_["X_vol"], T_["s_vol"], jeu["box"],
                                   n_epochs=epochs, log_every=4000,
                                   device=device, seed=0)
        duree = time.time() - t0
        torch.save(modele.state_dict(), f"{OUT}/modele_{nom}_{n_obs}.pt")

        res = {}
        for p in evalu:
            vue = figer_instant(modele, t_de(p), device=device)
            _, _, _, val = evaluate_grid(vue, box, grid_res=grille)
            res[p] = juger(*mc(axe, val), p, ref, val)

        gel, mor = {}, {}
        for p in caches:
            i = int(np.searchsorted(pas_obs, p))
            p1, p2 = pas_obs[i - 1], pas_obs[i]
            proche = p1 if abs(p - p1) <= abs(p - p2) else p2
            gel[p] = juger(*mc(axe, ref[proche]["val"]), p, ref,
                           ref[proche]["val"])
            th = (p2 - p) / (p2 - p1)
            vm = th * ref[p1]["val"] + (1 - th) * ref[p2]["val"]
            mor[p] = juger(*mc(axe, vm), p, ref, vm)

        vn = []
        for p in caches:
            if p - DPAS_VN < 0 or p + DPAS_VN > capsules.N_PAS - 1:
                continue
            P, vr = capsules.vitesse_normale_maillage(caps, faces, p, DPAS_VN,
                                                      1.0 / etendue)
            _, g, dfdt = derivees(modele, P, t_de(p), device=device)
            va = -dfdt / np.maximum(np.linalg.norm(g, axis=1), 1e-9)
            vn.append((np.abs(vr).mean(), np.abs(va - vr).mean()))
        vn = np.array(vn)

        def m(d, cle, ou=None):
            return float(np.nanmean([d[p][cle] for p in (ou or caches)]))

        r = {"n_obs": n_obs, "n_caches": len(caches), "pas_obs": pas_obs,
             "chamfer_obs": m(res, "chamfer", pas_obs),
             "chamfer_cache": m(res, "chamfer"), "iou_cache": m(res, "iou"),
             "chamfer_gel": m(gel, "chamfer"), "iou_gel": m(gel, "iou"),
             "chamfer_mor": m(mor, "chamfer"), "iou_mor": m(mor, "iou"),
             "plancher": pl,
             "chi_ok": all(res[p]["chi"] == 2 for p in evalu),
             "vn_amplitude": float(vn[:, 0].mean()),
             "vn_erreur": float(vn[:, 1].mean()),
             "duree": duree}
        resultats.append(r)

        print(f"\n  {'':<26}{'Chamfer':>10}{'excédent':>11}{'IoU':>10}")
        print(f"  {'plancher de la grille':<26}{pl:>10.5f}{0.0:>11.5f}{1.0:>10.4f}")
        print(f"  {'RÉSEAU (instants cachés)':<26}{r['chamfer_cache']:>10.5f}"
              f"{r['chamfer_cache']-pl:>11.5f}{r['iou_cache']:>10.4f}")
        print(f"  {'morphing':<26}{r['chamfer_mor']:>10.5f}"
              f"{r['chamfer_mor']-pl:>11.5f}{r['iou_mor']:>10.4f}")
        print(f"  {'image gelée':<26}{r['chamfer_gel']:>10.5f}"
              f"{r['chamfer_gel']-pl:>11.5f}{r['iou_gel']:>10.4f}")
        print(f"  topologie correcte partout : {r['chi_ok']}")
        print(f"  vitesse normale : erreur {r['vn_erreur']:.5f} / amplitude "
              f"{r['vn_amplitude']:.5f} = "
              f"{100*r['vn_erreur']/r['vn_amplitude']:.1f} %")
        print(f"  ({duree:.0f}s d'entraînement)")

    titre(f"SYNTHÈSE « {nom} » — jusqu'où peut-on enlever des photos ?")
    entete(pl)
    for r in resultats:
        ligne(r, pl)

    resume = {"nom": nom, "fenetre": [pas_min, pas_max], "plancher": pl,
              "evaluation": evalu, "pas_eval": pas_eval, "grille": grille,
              "epochs": epochs, "resultats": resultats,
              "duree_min": (time.time() - t0_global) / 60}
    with open(f"{OUT}/campagne_{nom}_resume.json", "w") as fh:
        json.dump(resume, fh, indent=2)
    figure(nom, resultats, pl)
    print(f"\n« {nom} » terminée en {resume['duree_min']:.1f} min.")
    return resume


def entete(pl):
    print(f"  plancher de la grille : {pl:.5f}")
    print(f"  {'observés':>9}{'cachés':>8}{'Chamfer':>10}{'excédent':>10}"
          f"{'x morphing':>12}{'IoU':>9}{'erreur v_n':>12}{'topo':>7}")


def ligne(r, pl):
    ex_r = r["chamfer_cache"] - pl
    ex_m = r["chamfer_mor"] - pl
    print(f"  {r['n_obs']:>9}{r['n_caches']:>8}{r['chamfer_cache']:>10.5f}"
          f"{ex_r:>10.5f}{ex_m/max(ex_r,1e-12):>11.1f}x{r['iou_cache']:>9.4f}"
          f"{100*r['vn_erreur']/r['vn_amplitude']:>11.1f}%"
          f"{'  ok' if r['chi_ok'] else '  FAUX':>7}")


def figure(nom, resultats, pl):
    n = [r["n_obs"] for r in resultats]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(n, [max(r["chamfer_cache"] - pl, 1e-7) for r in resultats], "o-",
               label="réseau f(x,y,z,t)")
    ax[0].plot(n, [max(r["chamfer_mor"] - pl, 1e-7) for r in resultats], "s-",
               label="morphing")
    ax[0].plot(n, [max(r["chamfer_gel"] - pl, 1e-7) for r in resultats], "^-",
               label="image gelée")
    ax[0].set_yscale("log"); ax[0].set_title("Excédent de Chamfer sur le plancher")
    ax[1].plot(n, [100 * r["iou_cache"] for r in resultats], "o-", label="réseau")
    ax[1].plot(n, [100 * r["iou_mor"] for r in resultats], "s-", label="morphing")
    ax[1].plot(n, [100 * r["iou_gel"] for r in resultats], "^-", label="image gelée")
    ax[1].set_title("IoU aux instants cachés (%)")
    ax[2].plot(n, [100 * r["vn_erreur"] / r["vn_amplitude"] for r in resultats],
               "o-", color="C3", label="réseau")
    ax[2].set_title("Erreur sur la vitesse de la membrane (% de l'amplitude)")
    for a in ax:
        a.set_xlabel("nombre d'instants fournis au réseau")
        a.invert_xaxis(); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"Campagne « {nom} » — combien de photos faut-il ?", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/campagne_{nom}.png", dpi=140)
    plt.close(fig)
    print(f"  {OUT}/campagne_{nom}.png")
