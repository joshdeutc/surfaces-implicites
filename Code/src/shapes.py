"""
Section 1 du notebook : formes test et échantillonnage de leur surface.

Chaque forme est décrite par une fonction implicite EXACTE f(x,y,z) :
    f < 0  -> intérieur,  f = 0  -> surface,  f > 0  -> extérieur.
"""
import numpy as np


def implicit_sphere(x, radius=1.0):
    """f(x,y,z) = sqrt(x^2+y^2+z^2) - R."""
    return np.linalg.norm(x, axis=1) - radius


def implicit_ellipsoid(x, a=1.2, b=0.8, c=0.6):
    """f(x,y,z) = sqrt((x/a)^2+(y/b)^2+(z/c)^2) - 1."""
    y = np.column_stack([x[:, 0] / a, x[:, 1] / b, x[:, 2] / c])
    return np.linalg.norm(y, axis=1) - 1.0


def sample_sphere_surface(n=3000, radius=1.0):
    """n points répartis uniformément sur la sphère."""
    u = np.random.rand(n)
    v = np.random.rand(n)

    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return np.column_stack([x, y, z])


def sample_ellipsoid_surface(n=3000, a=1.2, b=0.8, c=0.6):
    """Points sur la sphère unité, étirés en ellipsoïde."""
    pts = sample_sphere_surface(n, radius=1.0)
    pts[:, 0] *= a
    pts[:, 1] *= b
    pts[:, 2] *= c
    return pts


# ------------------------------------------------------------
# Globule rouge biconcave (étape 3) — profil de révolution d'Evans-Fung,
# normalisé pour un rayon = 1. C'est l'exemple "test des limites" du mail.
# ------------------------------------------------------------

_BC = dict(a0=0.207, a1=2.003, a2=-1.123)


def biconcave_zfun(r):
    """Demi-épaisseur z(r) du globule rouge (r dans [0,1]), 0 au-delà."""
    r = np.asarray(r, dtype=np.float64)
    rr = np.clip(r, 0.0, 1.0)
    s = np.sqrt(np.clip(1.0 - rr ** 2, 0.0, None))
    p = _BC["a0"] + _BC["a1"] * rr ** 2 + _BC["a2"] * rr ** 4
    return np.where(r <= 1.0, s * p, 0.0)


def biconcave_zfun_deriv(r):
    """Dérivée z'(r) (sert au calcul des normales)."""
    r = np.asarray(r, dtype=np.float64)
    rr = np.clip(r, 1e-8, 1.0 - 1e-9)
    s = np.sqrt(1.0 - rr ** 2)
    p = _BC["a0"] + _BC["a1"] * rr ** 2 + _BC["a2"] * rr ** 4
    dp = 2 * _BC["a1"] * rr + 4 * _BC["a2"] * rr ** 3
    dz = (-rr / s) * p + s * dp
    return np.where((r > 1e-8) & (r < 1.0), dz, 0.0)


def implicit_biconcave(P):
    """
    Fonction signée : < 0 dedans, > 0 dehors, ~ 0 sur la surface.
    Approximation par l'écart vertical à la surface (suffisant pour le SIGNE
    en entraînement et comme métrique de reconstruction).
    """
    P = np.asarray(P, dtype=np.float64)
    r = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    zf = biconcave_zfun(r)
    val_in = np.abs(P[:, 2]) - zf                      # r<1 : <0 dedans
    val_out = np.maximum(np.abs(P[:, 2]), r - 1.0)     # r>=1 : >0 dehors
    return np.where(r < 1.0, val_in, val_out)


def sample_biconcave_surface(n=6000):
    """Échantillonne la surface (faces haute et basse) du globule rouge."""
    m = n // 2
    r = np.random.rand(m)                  # rayon
    theta = 2 * np.pi * np.random.rand(m)
    z = biconcave_zfun(r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    top = np.column_stack([x, y, z])
    bottom = np.column_stack([x, y, -z])
    return np.vstack([top, bottom]).astype(np.float32)


def biconcave_normals(P):
    """Normales sortantes analytiques de la surface du globule rouge."""
    P = np.asarray(P, dtype=np.float64)
    r = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    rr = np.maximum(r, 1e-8)
    d = biconcave_zfun_deriv(r)
    nx = -d * P[:, 0] / rr
    ny = -d * P[:, 1] / rr
    nz = np.sign(P[:, 2])
    nz = np.where(nz == 0, 1.0, nz)
    n = np.column_stack([nx, ny, nz])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return n.astype(np.float32)


def get_surface(surface_type="ellipsoid"):
    """
    Renvoie (surface_points, implicit_exact, box) pour la forme choisie.
    `box` = demi-taille de la boîte 3D d'étude.
    """
    if surface_type == "sphere":
        surface_points = sample_sphere_surface(n=4000, radius=1.0)
        implicit_exact = lambda x: implicit_sphere(x, radius=1.0)
        box = 1.5
    elif surface_type == "biconcave":
        surface_points = sample_biconcave_surface(n=6000)
        implicit_exact = implicit_biconcave
        box = 1.3
    else:
        surface_points = sample_ellipsoid_surface(n=4000, a=1.2, b=0.8, c=0.6)
        implicit_exact = lambda x: implicit_ellipsoid(x, a=1.2, b=0.8, c=0.6)
        box = 1.6
    return surface_points, implicit_exact, box


# ------------------------------------------------------------
# Normales de surface (souvent estimables sur de vraies données ;
# utiles au terme "normales" de la perte IGR).
# ------------------------------------------------------------

def sphere_normals(pts, radius=1.0):
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def ellipsoid_normals(pts, a=1.2, b=0.8, c=0.6):
    # gradient de f(x) = sqrt((x/a)^2+(y/b)^2+(z/c)^2) - 1, puis normalisé
    g = np.column_stack([pts[:, 0] / a**2, pts[:, 1] / b**2, pts[:, 2] / c**2])
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def get_surface_normals(surface_type, pts):
    if surface_type == "sphere":
        return sphere_normals(pts, radius=1.0).astype("float32")
    if surface_type == "biconcave":
        return biconcave_normals(pts)
    return ellipsoid_normals(pts, a=1.2, b=0.8, c=0.6).astype("float32")
