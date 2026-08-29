"""
Les vraies données du BMBI : carte d'identité du dataset, et ce qu'il change.

Étape 11 de PLAN.md, débloquée le 26/08/2026. Ce script ne fait PAS
d'apprentissage : il répond d'abord aux questions qu'il faut avoir tranchées
avant d'entraîner quoi que ce soit sur des données qu'on n'a pas fabriquées.

  1. Le maillage est-il exploitable par notre chaîne ? (fermé, manifold,
     orienté, genre 0)
  2. Que fait la capsule, physiquement ?
  3. Notre protocole synthétique était-il représentatif ? En particulier :
     la déformation est-elle affine, comme la nôtre l'était ?
  4. À quelle fréquence faut-il échantillonner le temps ?
  5. Et on écrit la séquence au format d'échange du projet.

Aucune formule implicite n'intervient : tout est mesuré sur le maillage.

Lancer :  /usr/local/bin/python3 main_capsules.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, cKDTree

from src import capsules
from src.invariants import euler_characteristic, is_watertight

OUT = "output"
N_SNAP = 101


def section(titre):
    print(f"\n{'=' * 74}\n{titre}\n{'=' * 74}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    faces, caps = capsules.charger()
    idx = capsules.instants_reguliers(N_SNAP)

    # ---------------------------------------------------------------
    section("1. LE MAILLAGE EST-IL EXPLOITABLE PAR NOTRE CHAÎNE ?")
    V0, _ = capsules.instant(caps, 0)
    top = euler_characteristic(V0, faces)
    etanche = is_watertight(V0, faces)
    vol0, aire0 = capsules.geometrie(faces, V0)
    print(f"  sommets {top['V']}   arêtes {top['E']}   triangles {top['F']}")
    print(f"  chi = {top['chi']}   genre = {top['genus']:.0f}   étanche = {etanche}")
    print(f"  volume signé au premier pas = {vol0:+.6f}  -> normales "
          f"{'SORTANTES' if vol0 > 0 else 'RENTRANTES'}")
    print(f"  aire = {aire0:.6f}")
    print("  la connectivité ne change jamais : les 2562 noeuds sont MATÉRIELS.")
    print("  -> le maillage entre tel quel dans src/cloud_io.py, sans retournement.")

    # ---------------------------------------------------------------
    section("2. QUE FAIT LA CAPSULE ?")
    P = np.stack([capsules.instant(caps, int(t))[0] for t in idx])
    C = np.stack([capsules.instant(caps, int(t), recentre=False)[0].mean(0)
                  for t in idx])
    vols = np.array([capsules.geometrie(faces, p)[0] for p in P])
    aires = np.array([capsules.geometrie(faces, p)[1] for p in P])
    apl = np.array([capsules.demi_axes(p)[1] for p in P])

    k_max = int(apl.argmax())
    print(f"  MIGRATION   |centre| : {np.linalg.norm(C[0]):.4f} -> "
          f"{np.linalg.norm(C[-1]):.4f}  (elle part décentrée et rejoint l'axe)")
    print(f"  FORME       sphère parfaite au départ (rapport d'axes "
          f"{apl[0]:.4f}), maximum {apl.max():.4f} au pas t={idx[k_max]}, "
          f"puis relaxation jusqu'à {apl[-1]:.4f}")
    print(f"  VOLUME      {vols.mean():.6f}  dispersion "
          f"{100 * vols.std() / abs(vols.mean()):.4f} %  dérive totale "
          f"{100 * (vols[-1] - vols[0]) / vols[0]:+.4f} %")
    print(f"  AIRE        {aires.min():.5f} -> {aires.max():.5f}, soit "
          f"{100 * (aires.max() - aires.min()) / aires.mean():.2f} % d'étirement")
    print("  -> membrane incompressible qui s'étire : le VOLUME est un invariant")
    print("     que l'entraînement ignorera, donc un juge gratuit, comme sur la")
    print("     séquence synthétique.")

    # ---------------------------------------------------------------
    section("3. TANK-TREADING : LA MEMBRANE TOURNE-T-ELLE ?")
    print("  On compare le déplacement d'un noeud (mouvement MATÉRIEL) à sa")
    print("  distance à la surface de départ (mouvement NORMAL). L'écart est la")
    print("  part tangentielle, invisible pour une surface implicite.")
    print("  L'écart doit valoir plusieurs arêtes, sinon la mesure sature.")
    a0 = capsules.instant(caps, 0)[0]
    arete = np.linalg.norm(a0[faces[:, 0]] - a0[faces[:, 1]], axis=1).mean()
    print(f"  arête moyenne du maillage : {arete:.5f}")
    tang = []
    for t0, dt in ((1000, 1000), (5000, 2000), (10000, 4000), (15000, 5000)):
        a = capsules.instant(caps, t0)[0]
        b = capsules.instant(caps, t0 + dt)[0]
        mat = float(np.linalg.norm(b - a, axis=1).mean())
        nor = float(cKDTree(a).query(b)[0].mean())
        tang.append(1 - nor / mat)
        print(f"    t {t0:5d} -> {t0 + dt:5d} : matériel {mat:.5f} "
              f"({mat / arete:.1f} arêtes)   normal {nor:.5f}   "
              f"-> TANGENTIEL {100 * tang[-1]:.1f} %")
    print(f"  -> {100 * np.mean(tang):.0f} % du mouvement est du glissement le long")
    print("     de la surface. CONSÉQUENCE DE PROTOCOLE : on ne peut pas valider")
    print("     un suivi lagrangien en advectant par v_n et en comparant à la")
    print("     trajectoire d'un noeud. Ce n'est pas la même chose, et ce n'est")
    print("     pas un défaut du champ implicite : v_n EST tout ce qu'une surface")
    print("     peut porter. Le contrôle du bloc 4 de RESULTATS_temporel.md ne se")
    print("     transpose donc pas tel quel.")

    # ---------------------------------------------------------------
    section("4. LA DÉFORMATION EST-ELLE AFFINE ?  (le point qui change tout)")
    print("  Notre séquence synthétique était affine PAR CONSTRUCTION : un simple")
    print("  morphing entre les deux instants voisins y faisait jeu égal avec le")
    print("  réseau 4D, et le test d'interpolation ne départageait rien.")
    na_ref = [capsules.part_non_affine(P[0], P[k])[2] for k in range(1, len(P))]
    na_voisin = [capsules.part_non_affine(P[k - 1], P[k])[2]
                 for k in range(1, len(P))]
    print(f"  par rapport au premier instant : {100 * min(na_ref):.1f} à "
          f"{100 * max(na_ref):.1f} % non affine")
    print(f"  ENTRE DEUX SNAPSHOTS VOISINS   : {100 * np.mean(na_voisin):.1f} % "
          f"en moyenne (min {100 * min(na_voisin):.1f}, max {100 * max(na_voisin):.1f})")
    print("  -> c'est CE chiffre que voit l'étalon morphing. Il valait 0 % sur la")
    print("     séquence synthétique. Le test d'interpolation devient discriminant :")
    print("     la limite n°1 de RESULTATS_temporel.md tombe d'elle-même.")

    print("\n  Et il y a mieux qu'un pourcentage — une PREUVE. L'image affine d'un")
    print("  convexe est convexe. La capsule part d'une sphère ; si elle cesse")
    print("  d'être convexe, aucune application affine ne peut l'y amener.")
    print("  Défaut de convexité = (volume de l'enveloppe convexe / volume) - 1 :")
    for t in (0, 800, 1500, 3000, 8000, 20000):
        v, _ = capsules.instant(caps, t)
        vol = capsules.geometrie(faces, v)[0]
        hull = float(ConvexHull(v).volume)
        d = 100 * (hull - vol) / vol
        marque = "  <- CONCAVE" if d > 0.5 else ""
        print(f"    pas {t:6d} : {d:5.2f} %{marque}")
    print("  -> la capsule devient franchement concave dès le pas 1500 (creux du")
    print("     côté de l'axe du canal, forme dite en parachute). CQFD.")

    # ---------------------------------------------------------------
    section("5. À QUELLE FRÉQUENCE ÉCHANTILLONNER LE TEMPS ?")
    for s in (10, 50, 100, 200, 400):
        d = float(np.linalg.norm(capsules.instant(caps, 10000 + s)[0]
                                 - capsules.instant(caps, 10000)[0], axis=1).mean())
        print(f"  1 instant tous les {s:4d} pas : déplacement moyen {d:.5f} = "
              f"{d / arete:.2f} arête")
    dep = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    print(f"  Avec {len(idx)} instants uniformes (la recommandation du mail), le")
    print(f"  déplacement entre snapshots passe de {dep[0]:.4f} à {dep[-1]:.4f} :")
    print(f"  un facteur {dep[0] / dep[-1]:.0f}. La moitié des instants a un rapport")
    print(f"  d'axes sous {np.median(apl):.3f} alors que le maximum est {apl.max():.3f} :")
    print("  un échantillonnage UNIFORME dépense ses instants dans la relaxation.")
    print("  -> pour la première campagne, resserrer le début "
          "(capsules.instants_transitoire).")

    # ---------------------------------------------------------------
    section("6. EXPORT AU FORMAT D'ÉCHANGE DU PROJET")
    dossier = os.path.join(OUT, "capsules_bmbi")
    fichiers, box = capsules.exporter_sequence(dossier, faces, caps, idx)
    print(f"  {len(fichiers)} fichiers écrits dans {dossier}/  (box = {box:.4f})")
    print("  Format identique à celui de la campagne synthétique : points de")
    print("  surface + normales sortantes. Tout l'aval du projet s'applique sans")
    print("  une ligne de changement.")

    # ---------------------------------------------------------------
    figures(faces, caps, idx, P, C, vols, aires, apl, na_voisin)
    print("\nTerminé.")


def figures(faces, caps, idx, P, C, vols, aires, apl, na_voisin):
    section("7. FIGURES")

    montre = [0, 4, 15, 35, 65, 100]
    fig = plt.figure(figsize=(16, 3.2))
    for j, k in enumerate(montre):
        ax = fig.add_subplot(1, len(montre), j + 1, projection="3d")
        p = P[k]
        ax.plot_trisurf(p[:, 0], p[:, 1], p[:, 2], triangles=faces,
                        color="#7fb3d5", edgecolor="k", linewidth=0.05, alpha=0.9)
        ax.set_title(f"pas {idx[k]}\nrapport d'axes {apl[k]:.3f}", fontsize=9)
        for s in "xyz":
            getattr(ax, f"set_{s}lim")(-0.55, 0.55)
        ax.set_axis_off()
    fig.suptitle("Capsule du BMBI — la sphère se déforme puis relaxe "
                 "(recentrée ; la migration est tracée à part)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/capsules_formes.png", dpi=130)
    plt.close(fig)

    t = np.arange(len(idx))
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax[0, 0].plot(t, np.linalg.norm(C, axis=1), color="C0")
    ax[0, 0].set_title("Migration vers l'axe du canal")
    ax[0, 0].set_ylabel("distance du centre à l'axe")

    ax[0, 1].plot(t, apl, color="C1")
    ax[0, 1].axvline(int(apl.argmax()), ls=":", color="k")
    ax[0, 1].set_title(f"Déformation — maximum au pas {idx[int(apl.argmax())]}")
    ax[0, 1].set_ylabel("grand axe / petit axe")

    ax[0, 2].plot(t, 100 * (vols / vols[0] - 1), color="C2", label="volume")
    ax[0, 2].plot(t, 100 * (aires / aires[0] - 1), color="C3", label="aire")
    ax[0, 2].legend()
    ax[0, 2].set_title("Volume conservé, aire étirée")
    ax[0, 2].set_ylabel("écart au premier instant (%)")

    ax[1, 0].plot(t[1:], 100 * np.asarray(na_voisin), color="C4")
    ax[1, 0].axhline(0, ls="--", color="k", lw=1)
    ax[1, 0].set_title("Part NON AFFINE entre snapshots voisins")
    ax[1, 0].set_ylabel("% du déplacement")
    ax[1, 0].text(0.5, 0.08, "séquence synthétique : 0 %",
                  transform=ax[1, 0].transAxes, ha="center", color="red", fontsize=9)

    ax[1, 1].semilogy(t[1:], np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1),
                      color="C5")
    ax[1, 1].set_title("Déplacement entre snapshots voisins")
    ax[1, 1].set_ylabel("déplacement moyen (échelle log)")

    ax[1, 2].plot(t, C[:, 1], label="y")
    ax[1, 2].plot(t, C[:, 2], label="z")
    ax[1, 2].plot(t, C[:, 0], label="x")
    ax[1, 2].legend()
    ax[1, 2].set_title("Trajectoire du centre, composante par composante")

    for a in ax.ravel():
        a.set_xlabel("indice du snapshot")
        a.grid(alpha=0.3)
    fig.suptitle("Ce que fait la capsule du BMBI sur les 20001 pas de temps",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/capsules_diagnostics.png", dpi=130)
    plt.close(fig)
    print(f"  {OUT}/capsules_formes.png")
    print(f"  {OUT}/capsules_diagnostics.png")


if __name__ == "__main__":
    main()
