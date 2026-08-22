"""
Validation de src/curvature.py et src/invariants.py sur des cas dont la réponse
est connue d'avance.

Aucun réseau n'intervient ici : on vérifie d'abord que les outils sont justes
sur des formes analytiques, AVANT de les pointer sur un champ appris
(main_curvature.py). Si le tore ne passe pas, rien de ce qui suit n'a de valeur.

Les formes de test (sphère, tore) sont définies localement dans ce fichier.

Lancer :  /usr/local/bin/python3 validate_geometry.py
"""
import numpy as np
import torch

from src.curvature import curvature_numpy, curvature_autograd
from src.invariants import (euler_characteristic, is_watertight, mesh_area,
                            mesh_volume, gauss_bonnet, triangle_areas)
from src.reconstruct import extract_surface


# ------------------------------------------------------------------
# Formes de test analytiques (locales à ce fichier)
# ------------------------------------------------------------------

def sphere_sdf(radius=1.0):
    """Distance signée exacte à la sphère : f = ‖x‖ − R."""
    def f(P):
        P = np.asarray(P, dtype=np.float64)
        return np.linalg.norm(P, axis=1) - radius
    return f


def sample_sphere(n, radius=1.0, seed=0):
    """Points uniformes sur la sphère."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * radius


def torus_sdf(R=1.0, r=0.35):
    """
    Distance signée exacte au tore d'axe z :
        f = √((√(x²+y²) − R)² + z²) − r
    Grand rayon R, petit rayon r. Genre 1, donc χ = 0.
    """
    def f(P):
        P = np.asarray(P, dtype=np.float64)
        q = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2) - R
        return np.sqrt(q * q + P[:, 2] ** 2) - r
    return f


def torus_curvature_exact(P, R=1.0, r=0.35):
    """
    Courbure gaussienne exacte du tore en ses points de surface :
        K = cos θ / (r (R + r cos θ))
    où θ est l'angle dans le tube. On retrouve cos θ depuis la position :
    cos θ = (√(x²+y²) − R) / r.
    """
    P = np.asarray(P, dtype=np.float64)
    rho = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    cos_t = np.clip((rho - R) / r, -1.0, 1.0)
    return cos_t / (r * (R + r * cos_t))


def grid_values(f, box, res):
    """Évalue une fonction implicite numpy sur une grille res³ dans [-box,box]³."""
    lin = np.linspace(-box, box, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    values = np.asarray(f(pts)).reshape(res, res, res)
    return lin, lin, lin, values


class SphereSDFModule(torch.nn.Module):
    """Module torch qui calcule EXACTEMENT ‖x‖ − R : l'oracle de l'autograd."""

    def __init__(self, radius=1.0):
        super().__init__()
        self.radius = radius
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return x.norm(dim=-1, keepdim=True) - self.radius


# ------------------------------------------------------------------
# Vérifications
# ------------------------------------------------------------------

resultats = []


def check(nom, obtenu, attendu, tol, unite=""):
    """Compare, affiche, et mémorise le verdict."""
    ecart = abs(obtenu - attendu)
    ok = ecart <= tol
    resultats.append(ok)
    statut = "OK  " if ok else "ÉCHEC"
    print(f"  [{statut}] {nom:<46s} obtenu={obtenu:+.6f}{unite}  "
          f"attendu={attendu:+.6f}  écart={ecart:.2e}  (tol {tol:.0e})")
    return ok


print("=" * 78)
print("1. COURBURES ANALYTIQUES SUR LA SPHÈRE (différences finies)")
print("=" * 78)
print("  Convention : f = ‖x‖ − R (f<0 dedans), donc K_moyenne NÉGATIVE.")

for R in (1.0, 2.0):
    pts = sample_sphere(2000, radius=R, seed=1)
    c = curvature_numpy(sphere_sdf(R), pts)
    print(f"\n  Sphère R={R}  (attendu K_gauss=1/R²={1/R**2:.4f}, K_moyenne=-1/R={-1/R:.4f})")
    check(f"K_gauss moyen (R={R})", float(c["K_gauss"].mean()), 1.0 / R ** 2, 1e-3)
    check(f"|K_moyenne| moyen (R={R})", float(np.abs(c["K_mean"]).mean()), 1.0 / R, 1e-3)
    check(f"K_moyenne moyen signé (R={R})", float(c["K_mean"].mean()), -1.0 / R, 1e-3)
    check(f"k1 moyen (R={R}, ombilic)", float(c["k1"].mean()), -1.0 / R, 1e-3)
    check(f"k2 moyen (R={R}, ombilic)", float(c["k2"].mean()), -1.0 / R, 1e-3)
    print(f"         dispersion : K_gauss écart-type={c['K_gauss'].std():.2e}, "
          f"K_moyenne écart-type={c['K_mean'].std():.2e}")

print()
print("=" * 78)
print("2. SPHÈRE MAILLÉE PAR MARCHING CUBES 128³")
print("=" * 78)

BOX_S, RES_S = 1.5, 128
xs, ys, zs, vals = grid_values(sphere_sdf(1.0), BOX_S, RES_S)
verts_s, faces_s, _, _ = extract_surface(xs, ys, zs, vals)
topo_s = euler_characteristic(verts_s, faces_s)
brut_s = euler_characteristic(verts_s, faces_s, weld=False)
print(f"  maillage : {len(verts_s)} sommets bruts, {len(faces_s)} triangles")
print(f"  après soudure des sommets : V={topo_s['V']}  E={topo_s['E']}  F={topo_s['F']}")
print(f"  (sans soudure on aurait chi={brut_s['chi']}, avec V={brut_s['V']})")
check("chi de la sphère", topo_s["chi"], 2, 0)
check("genre de la sphère", topo_s["genus"], 0.0, 0)
print(f"  [{'OK  ' if is_watertight(verts_s, faces_s) else 'ÉCHEC'}] "
      f"{'étanchéité (chaque arête vue 2 fois)':<46s}")
resultats.append(is_watertight(verts_s, faces_s))

aire_s = mesh_area(verts_s, faces_s)
vol_s = mesh_volume(verts_s, faces_s)
check("aire (attendu 4π)", aire_s, 4 * np.pi, 0.01 * 4 * np.pi)
check("volume (attendu 4π/3)", vol_s, 4 * np.pi / 3, 0.01 * 4 * np.pi / 3)

K_s = curvature_numpy(sphere_sdf(1.0), verts_s)["K_gauss"]
gb_s = gauss_bonnet(verts_s, faces_s, K_s)
check("Gauss-Bonnet ∫K dA (attendu 4π)", gb_s["integral"], 4 * np.pi, 0.01 * 4 * np.pi)

print()
print("=" * 78)
print("3. TORE (grand rayon 1, petit rayon 0.35) — LE TEST SÉVÈRE")
print("=" * 78)
print("  chi=0 ne sort que si les arêtes sont comptées une seule fois.")
print("  ∫K dA=0 exige que les courbures positives (extérieur du tube)")
print("  annulent exactement les négatives (intérieur) : aucun biais possible.")

R_MAJ, R_MIN = 1.0, 0.35
BOX_T, RES_T = 1.6, 128
xt, yt, zt, valt = grid_values(torus_sdf(R_MAJ, R_MIN), BOX_T, RES_T)
verts_t, faces_t, _, _ = extract_surface(xt, yt, zt, valt)
topo_t = euler_characteristic(verts_t, faces_t)
brut_t = euler_characteristic(verts_t, faces_t, weld=False)
print(f"\n  maillage : {len(verts_t)} sommets bruts, {len(faces_t)} triangles")
print(f"  après soudure : V={topo_t['V']}  E={topo_t['E']}  F={topo_t['F']}")
print(f"  (sans soudure on aurait chi={brut_t['chi']})")
check("chi du tore", topo_t["chi"], 0, 0)
check("genre du tore", topo_t["genus"], 1.0, 0)
print(f"  [{'OK  ' if is_watertight(verts_t, faces_t) else 'ÉCHEC'}] "
      f"{'étanchéité':<46s}")
resultats.append(is_watertight(verts_t, faces_t))

check("aire (attendu 4π²Rr)", mesh_area(verts_t, faces_t),
      4 * np.pi ** 2 * R_MAJ * R_MIN, 0.02 * 4 * np.pi ** 2 * R_MAJ * R_MIN)
check("volume (attendu 2π²Rr²)", mesh_volume(verts_t, faces_t),
      2 * np.pi ** 2 * R_MAJ * R_MIN ** 2, 0.02 * 2 * np.pi ** 2 * R_MAJ * R_MIN ** 2)

# Courbure du tore : d'abord la formule exacte, puis nos différences finies.
K_t_exact = torus_curvature_exact(verts_t, R_MAJ, R_MIN)
K_t_fd = curvature_numpy(torus_sdf(R_MAJ, R_MIN), verts_t)["K_gauss"]
print(f"\n  courbure gaussienne aux sommets : "
      f"exacte [{K_t_exact.min():+.3f}, {K_t_exact.max():+.3f}], "
      f"différences finies [{K_t_fd.min():+.3f}, {K_t_fd.max():+.3f}]")
check("écart max formule exacte / diff. finies", float(np.abs(K_t_exact - K_t_fd).max()),
      0.0, 1e-2)

gb_t_exact = gauss_bonnet(verts_t, faces_t, K_t_exact)
gb_t_fd = gauss_bonnet(verts_t, faces_t, K_t_fd)
check("Gauss-Bonnet ∫K dA, courbure exacte", gb_t_exact["integral"], 0.0, 0.1)
check("Gauss-Bonnet ∫K dA, diff. finies", gb_t_fd["integral"], 0.0, 0.1)
ampl = float(np.sum(np.abs(K_t_exact[faces_t].mean(axis=1)) * triangle_areas(verts_t, faces_t)))
print(f"         (pour référence ∫|K| dA = {ampl:.3f} — l'annulation se fait sur cette amplitude)")

print()
print("=" * 78)
print("4. AUTOGRAD D'ORDRE 2 SUR UN ORACLE (module torch qui calcule ‖x‖−1)")
print("=" * 78)

oracle = SphereSDFModule(radius=1.0)
pts_a = sample_sphere(4000, radius=1.0, seed=2)
c_auto = curvature_autograd(oracle, pts_a)
print(f"  device utilisé : {c_auto['device']}")
check("K_gauss moyen (autograd)", float(c_auto["K_gauss"].mean()), 1.0, 1e-3)
check("|K_moyenne| moyen (autograd)", float(np.abs(c_auto["K_mean"]).mean()), 1.0, 1e-3)
check("K_moyenne signé (autograd)", float(c_auto["K_mean"].mean()), -1.0, 1e-3)

c_fd = curvature_numpy(sphere_sdf(1.0), pts_a)
check("accord autograd / différences finies (K_gauss)",
      float(np.abs(c_auto["K_gauss"] - c_fd["K_gauss"]).max()), 0.0, 1e-2)

print()
print("=" * 78)
n_ok, n_tot = sum(resultats), len(resultats)
print(f"BILAN : {n_ok}/{n_tot} vérifications passées")
print("=" * 78)
if n_ok != n_tot:
    raise SystemExit(1)
