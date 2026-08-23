"""
Rendu PyVista et export vers les formats d'échange.

La fiche d'enregistrement de la TZ nomme PyVista deux fois — dans la prise en
main des bibliothèques (« pytorch, skimage, trimesh, pyvista ») puis pour
« l'affichage pyvista (format vtk) » du réseau temporel. Le code affichait
jusqu'ici en matplotlib : suffisant pour contrôler, insuffisant pour une figure
de rapport, et incapable de TRANSMETTRE un maillage à quelqu'un d'autre.

Trois choses sont produites et vérifiées ici :

  1. la figure du globule rouge coloré par sa COURBURE GAUSSIENNE, refaite en
     PyVista, à comparer à output/courbure_globule.png (matplotlib) ;
  2. les exports .obj / .vtk / .vtp, chacun RELU après écriture — un fichier
     de taille non nulle peut parfaitement être vide de géométrie, donc on ne
     se fie pas à l'octet, on recompte les sommets ;
  3. la série temporelle .pvd des six instants du champ f(x,y,z,t), le format
     que ParaView lit comme une animation.

Aucune fenêtre interactive n'est ouverte (pv.OFF_SCREEN = True).

Lancer :  /usr/local/bin/python3 main_render.py
"""
import os

import numpy as np

from src.shapes import (implicit_rbc_quartic, rbc_radius_max,
                        implicit_rbc_unit, sample_rbc_unit_surface)
from src.curvature import curvature_numpy
from src.reconstruct import extract_surface
from src.invariants import (euler_characteristic, is_watertight, mesh_area,
                            mesh_volume, gauss_bonnet)
from src import temporal
from src import viz_pv
from skimage import measure

SORTIE = "output"


def maille_champ(f, bornes, res=160):
    """Marching cubes sur un champ analytique, dans une boîte anisotrope."""
    (xa, xb), (ya, yb), (za, zb) = bornes
    xs = np.linspace(xa, xb, res)
    ys = np.linspace(ya, yb, res)
    zs = np.linspace(za, zb, res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    V = np.asarray(f(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
                   ).reshape(res, res, res)
    verts, faces, _, _ = measure.marching_cubes(
        V, 0.0, spacing=(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]))
    verts[:, 0] += xa
    verts[:, 1] += ya
    verts[:, 2] += zb * 0 + za
    return verts, faces


# ==================================================================
# 1. Le globule rouge coloré par sa courbure gaussienne
# ==================================================================

def globule_courbure():
    print("=" * 92)
    print("1. GLOBULE ROUGE COLORÉ PAR SA COURBURE GAUSSIENNE (PyVista)")
    print("=" * 92)

    r = rbc_radius_max()
    bornes = ((-r * 1.02, r * 1.02), (-r * 1.02, r * 1.02), (-1.3, 1.3))
    verts, faces = maille_champ(implicit_rbc_quartic, bornes, res=180)

    K = curvature_numpy(implicit_rbc_quartic, verts)["K_gauss"]
    topo = euler_characteristic(verts, faces)
    gb = gauss_bonnet(verts, faces, K)

    print(f"  maillage    : V={topo['V']}  F={topo['F']}  chi={topo['chi']}  "
          f"étanche={is_watertight(verts, faces)}")
    print(f"  géométrie   : aire={mesh_area(verts, faces):.3f} µm²   "
          f"volume={mesh_volume(verts, faces):.3f} µm³")
    print(f"  Gauss-Bonnet: ∫K dA={gb['integral']:+.4f}  attendu 4π={gb['expected']:+.4f}"
          f"  écart={100*gb['relative_error']:.3f} %")
    print(f"  courbure    : K_G min={K.min():+.5f}  médiane={np.median(K):+.5f}  "
          f"max={K.max():+.5f} µm⁻²")

    # Bornes de couleur symétriques et robustes : les extrêmes numériques du
    # marching cubes écraseraient l'échelle et rendraient la figure illisible.
    lim = float(np.quantile(np.abs(K), 0.98))
    mesh = viz_pv.prepare(viz_pv.to_polydata(verts, faces, K, "K_gauss"))

    p1 = viz_pv.render(
        mesh, f"{SORTIE}/pv_globule_courbure.png", scalar_name="K_gauss",
        cmap="RdBu_r", clim=(-lim, lim), direction="trois-quarts", distance=2.4,
        scalar_bar_title="courbure gaussienne K_G  (µm⁻²)",
        title="Globule rouge quartique (Kuchel 2021) — courbure gaussienne")
    p2 = viz_pv.render(
        mesh, f"{SORTIE}/pv_globule_courbure_coupe.png", scalar_name="K_gauss",
        cmap="RdBu_r", clim=(-lim, lim), direction="face", distance=2.2,
        scalar_bar_title="courbure gaussienne K_G  (µm⁻²)",
        title="Vue de profil : la fossette est le seul lieu de K_G maximale")

    ref = f"{SORTIE}/courbure_globule.png"
    print(f"\n  {p1}  {viz_pv.taille_ko(p1):8.1f} Ko")
    print(f"  {p2}  {viz_pv.taille_ko(p2):8.1f} Ko")
    if os.path.exists(ref):
        print(f"  {ref} (matplotlib, référence)  {viz_pv.taille_ko(ref):8.1f} Ko")
    else:
        print(f"  {ref} absent — relancer main_curvature.py pour le régénérer")
    return verts, faces, K, mesh


# ==================================================================
# 2. Exports, chacun relu
# ==================================================================

def exports(verts, faces, K, mesh):
    print("\n" + "=" * 92)
    print("2. EXPORTS — chaque fichier est RELU après écriture")
    print("=" * 92)

    fichiers = []
    fichiers.append(viz_pv.export_obj(verts, faces, f"{SORTIE}/globule.obj"))
    fichiers.append(viz_pv.export_vtk(mesh, f"{SORTIE}/globule.vtk"))
    fichiers.append(viz_pv.export_vtk(mesh, f"{SORTIE}/globule.vtp"))

    print(f"{'fichier':<32}{'taille':>12}{'sommets relus':>15}"
          f"{'faces relues':>14}   champs conservés")
    print("-" * 92)
    for f in fichiers:
        nv, nf, champs = viz_pv.relire(f)
        marque = "OK" if nv == len(verts) else f"ÉCART ({len(verts)} attendus)"
        print(f"{f:<32}{viz_pv.taille_ko(f):>10.1f} Ko{nv:>15d}{nf:>14d}"
              f"   {champs if champs else '— (OBJ ne porte aucun champ)'}  {marque}")
    print("-" * 92)
    print(f"Le maillage de départ a {len(verts)} sommets et {len(faces)} faces.")
    print("OBJ perd la courbure : c'est le format à donner pour la GÉOMÉTRIE seule.")
    print("VTK/VTP la conservent : c'est le format à donner pour une MESURE.")
    return fichiers


# ==================================================================
# 3. Série temporelle .pvd
# ==================================================================

def serie_temporelle():
    print("\n" + "=" * 92)
    print("3. SÉRIE TEMPORELLE .pvd — les six instants du champ f(x,y,z,t)")
    print("=" * 92)

    dossier = f"{SORTIE}/serie_pvd"
    meshes, instants = [], list(temporal.T_OBSERVES)
    for t in instants:
        v, f, _, _, _ = temporal.maillage_instant(t, grid_res=120)
        vol = mesh_volume(v, f)
        m = viz_pv.prepare(viz_pv.to_polydata(v, f, v[:, 2], "z"))
        meshes.append(m)
        print(f"  t = {t:.1f}  xi = {temporal.xi(t):.3f}  sommets = {len(v):6d}  "
              f"volume = {vol:.5f}")

    pvd, vtps = viz_pv.export_time_series(meshes, instants, dossier, "globule")
    print(f"\n{'fichier':<40}{'taille':>12}{'sommets relus':>15}")
    print("-" * 92)
    for f in vtps:
        nv, nf, _ = viz_pv.relire(f)
        print(f"{f:<40}{viz_pv.taille_ko(f):>10.1f} Ko{nv:>15d}")
    print(f"{pvd:<40}{viz_pv.taille_ko(pvd):>10.1f} Ko{'index XML':>15}")
    print("-" * 92)
    print(open(pvd).read().strip()[:400])

    p = viz_pv.render_comparison(
        [meshes[0], meshes[2], meshes[-1]],
        [f"t = {instants[0]:.1f}  (xi = {temporal.xi(instants[0]):.2f})",
         f"t = {instants[2]:.1f}  (xi = {temporal.xi(instants[2]):.2f})",
         f"t = {instants[-1]:.1f}  (xi = {temporal.xi(instants[-1]):.2f})"],
        f"{SORTIE}/pv_serie_temporelle.png", color="indianred",
        direction="trois-quarts", distance=2.5,
        title="Étirement à volume constant — caméras liées, l'écart visible est un écart de FORME")
    print(f"\n  {p}  {viz_pv.taille_ko(p):8.1f} Ko")
    return pvd


def main():
    os.makedirs(SORTIE, exist_ok=True)
    verts, faces, K, mesh = globule_courbure()
    exports(verts, faces, K, mesh)
    serie_temporelle()
    print("\nTerminé — aucune fenêtre interactive n'a été ouverte.")


if __name__ == "__main__":
    main()
