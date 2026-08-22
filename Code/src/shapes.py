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
    elif surface_type == "rbc":
        # globule rouge quartique, unités physiques de l'article (µm)
        surface_points = sample_rbc_quartic_surface(n=6000).astype(np.float32)
        implicit_exact = implicit_rbc_quartic
        box = 4.5
    elif surface_type == "rbc_unit":
        # même forme, bord ramené à r = 1 (adapté aux grilles du projet)
        surface_points = sample_rbc_unit_surface(n=6000).astype(np.float32)
        implicit_exact = implicit_rbc_unit
        box = 1.3
    elif surface_type == "torus":
        surface_points = sample_torus_surface(n=6000).astype(np.float32)
        implicit_exact = implicit_torus
        box = 1.5
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
    if surface_type == "rbc":
        return rbc_quartic_normals(pts).astype("float32")
    if surface_type == "rbc_unit":
        return rbc_unit_normals(pts).astype("float32")
    if surface_type == "torus":
        return torus_normals(pts).astype("float32")
    return ellipsoid_normals(pts, a=1.2, b=0.8, c=0.6).astype("float32")


# ============================================================
# Globule rouge QUARTIQUE (Kuchel et al., Sci. Rep. 11:13712, 2021)
# ============================================================
#
# Contrairement au profil d'Evans-Fung ci-dessus, qui est défini par morceaux
# (np.where sur r < 1) et dont le gradient est mal défini à la coupure, celui-ci
# est une surface implicite ALGÉBRIQUE, lisse partout :
#
#     F(x,y,z) = (x²+y²+z²)² + P·(x²+y²) + Q·z² + R = 0
#
# avec, pour un diamètre d, une épaisseur b au creux central et une épaisseur
# maximale h au bord :
#
#     P = −d²/2 + (h²/2)(d²/b² − 1) − (h²/2)(d²/b² − 1)·√(1 − b²/h²)
#     Q = P·d²/b² + (b²/4)(d⁴/b⁴ − 1)
#     R = −P·d²/4 − d⁴/16
#
# Trois conséquences utiles pour ce projet :
#   - le gradient est analytique partout, donc l'entraînement eikonal travaille
#     sur un champ propre ;
#   - les courbures ont une forme close (c'est l'objet de l'article) ;
#   - la forme entière tient en TROIS coefficients (P, Q, R) — l'argument de
#     compression des surfaces implicites, sur un cas biologique réel.
#
# Valeurs de référence de l'article : d = 8 µm, b = 1 µm, h = 2.12 µm
# (volume ≈ 86 fL, conforme au globule rouge humain).

RBC_DEFAULTS = dict(d=8.0, b=1.0, h=2.12)


def rbc_coeffs(d=8.0, b=1.0, h=2.12):
    """Coefficients (P, Q, R) de la quartique, éq. (2)-(4) de l'article."""
    d2, b2, h2 = d ** 2, b ** 2, h ** 2
    ratio = d2 / b2
    P = -d2 / 2 + h2 / 2 * (ratio - 1.0) - h2 / 2 * (ratio - 1.0) * np.sqrt(1 - b2 / h2)
    Q = P * ratio + b2 / 4 * (d ** 4 / b ** 4 - 1.0)
    R = -P * d2 / 4 - d ** 4 / 16.0
    return float(P), float(Q), float(R)


def implicit_rbc_quartic(Pts, d=8.0, b=1.0, h=2.12):
    """F(x,y,z) de la quartique : < 0 dedans, 0 sur la surface, > 0 dehors."""
    Pts = np.asarray(Pts, dtype=np.float64).reshape(-1, 3)
    P, Q, R = rbc_coeffs(d, b, h)
    x, y, z = Pts[:, 0], Pts[:, 1], Pts[:, 2]
    rho2 = x * x + y * y + z * z
    return rho2 ** 2 + P * (x * x + y * y) + Q * z * z + R


def rbc_quartic_normals(Pts, d=8.0, b=1.0, h=2.12):
    """
    Normales sortantes ANALYTIQUES : ∇F / ‖∇F‖, avec

        ∂F/∂x = 4x·(x²+y²+z²) + 2P·x
        ∂F/∂y = 4y·(x²+y²+z²) + 2P·y
        ∂F/∂z = 4z·(x²+y²+z²) + 2Q·z

    F croît vers l'extérieur, donc ∇F pointe bien vers le dehors.
    """
    Pts = np.asarray(Pts, dtype=np.float64).reshape(-1, 3)
    P, Q, _ = rbc_coeffs(d, b, h)
    x, y, z = Pts[:, 0], Pts[:, 1], Pts[:, 2]
    rho2 = x * x + y * y + z * z
    g = np.column_stack([4 * x * rho2 + 2 * P * x,
                         4 * y * rho2 + 2 * P * y,
                         4 * z * rho2 + 2 * Q * z])
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-300)


def _rbc_z_half(s, P, Q, R):
    """
    Demi-épaisseur z(r) au carré, pour s = r² donné, par résolution EXACTE.

    En posant u = z², F = 0 devient une équation du second degré en u :
        u² + (2s + Q)·u + (s² + P·s + R) = 0
    On garde la racine positive, écrite sous forme numériquement stable
    (u₊ = −2c / (β + √Δ)) pour éviter l'annulation catastrophique près du bord,
    là où u₊ tend vers 0.
    """
    s = np.asarray(s, dtype=np.float64)
    beta = 2.0 * s + Q
    c = s * s + P * s + R
    disc = np.maximum(beta * beta - 4.0 * c, 0.0)
    u = -2.0 * c / (beta + np.sqrt(disc))
    return np.maximum(u, 0.0)


def rbc_radius_max(d=8.0, b=1.0, h=2.12):
    """Rayon du bord (z = 0) : plus grande racine de s² + P·s + R = 0."""
    P, _, R = rbc_coeffs(d, b, h)
    s = (-P + np.sqrt(P * P - 4.0 * R)) / 2.0
    return float(np.sqrt(s))


def sample_rbc_quartic_surface(n=6000, d=8.0, b=1.0, h=2.12, seed=None,
                               n_grid=40000):
    """
    Points sur la surface F = 0, répartis proportionnellement à l'AIRE.

    La surface est de révolution : deux nappes z = ±z(r) qui se rejoignent au
    bord. On construit le méridien sur une grille resserrée aux deux extrémités
    (le creux central et le bord, où la courbure est forte), on pondère chaque
    segment par 2π·r·(longueur d'arc) — théorème de Pappus — et on tire par
    inversion de la fonction de répartition. Le z final est RECALCULÉ exactement
    à partir du r tiré, donc les points sont sur la surface à la précision
    machine, pas à celle de l'interpolation.
    """
    rng = np.random.default_rng(seed)
    P, Q, R = rbc_coeffs(d, b, h)
    s_max = rbc_radius_max(d, b, h) ** 2

    # grille en s resserrée aux deux bouts : s = s_max·sin²(πt/2)
    t = np.linspace(0.0, 1.0, n_grid + 1)
    s_grid = s_max * np.sin(0.5 * np.pi * t) ** 2
    r_grid = np.sqrt(s_grid)
    z_grid = np.sqrt(_rbc_z_half(s_grid, P, Q, R))

    seg_len = np.hypot(np.diff(r_grid), np.diff(z_grid))
    r_mid = 0.5 * (r_grid[:-1] + r_grid[1:])
    w = r_mid * seg_len                      # 2π constant : inutile de le porter
    cdf = np.cumsum(w)
    cdf /= cdf[-1]

    m = n // 2                               # nappe supérieure, puis miroir
    k = np.searchsorted(cdf, rng.random(m), side="right")
    k = np.clip(k, 0, len(seg_len) - 1)
    frac = rng.random(m)
    r = r_grid[k] + frac * (r_grid[k + 1] - r_grid[k])
    z = np.sqrt(_rbc_z_half(r * r, P, Q, R))

    theta = 2 * np.pi * rng.random(m)
    top = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    bottom = top.copy()
    bottom[:, 2] *= -1.0
    return np.vstack([top, bottom])


# --- variante normalisée (bord ramené à r = 1) -------------------------------
#
# La quartique physique s'étend sur 8 µm : mal adaptée aux grilles et aux
# réseaux du projet, qui travaillent dans une boîte de l'ordre de l'unité.
# En posant x = L·x', F(L·x')/L⁴ reste une quartique de MÊME forme, avec
#     P' = P/L² ,  Q' = Q/L² ,  R' = R/L⁴
# ce qui permet de comparer directement les coefficients ajustés aux vrais.

# Remarque de précision : à l'échelle physique (bord à r = 4, ‖∇F‖ ≈ 134), une
# position stockée en float32 suffit à décaler F de ~2e-5. La quartique n'est pas
# en cause, la représentation si — d'où l'intérêt de la variante normalisée
# ci-dessous pour tout ce qui passe par le pipeline d'apprentissage.

def rbc_scale(d=8.0, b=1.0, h=2.12):
    """Facteur d'échelle qui ramène le bord du globule à r = 1."""
    return rbc_radius_max(d, b, h)


def rbc_coeffs_scaled(d=8.0, b=1.0, h=2.12):
    """Coefficients (P', Q', R') de la quartique une fois normalisée."""
    P, Q, R = rbc_coeffs(d, b, h)
    L = rbc_scale(d, b, h)
    return P / L ** 2, Q / L ** 2, R / L ** 4


def implicit_rbc_unit(Pts, d=8.0, b=1.0, h=2.12):
    """Quartique normalisée : même forme, bord à r = 1."""
    L = rbc_scale(d, b, h)
    return implicit_rbc_quartic(np.asarray(Pts, dtype=np.float64) * L, d, b, h) / L ** 4


def rbc_unit_normals(Pts, d=8.0, b=1.0, h=2.12):
    """Normales de la quartique normalisée (invariantes par homothétie)."""
    L = rbc_scale(d, b, h)
    return rbc_quartic_normals(np.asarray(Pts, dtype=np.float64) * L, d, b, h)


def sample_rbc_unit_surface(n=6000, d=8.0, b=1.0, h=2.12, seed=None):
    return sample_rbc_quartic_surface(n, d, b, h, seed=seed) / rbc_scale(d, b, h)


# ============================================================
# Tore — forme de contrôle à topologie non triviale (genre 1)
# ============================================================
#
# Sert de cas test : sa caractéristique d'Euler vaut 0 (et non 2), et
# l'intégrale de sa courbure de Gauss vaut 0 par Gauss-Bonnet. C'est le
# contre-exemple qui valide un calcul d'invariants — une sphère seule ne
# distingue pas un comptage d'arêtes correct d'un comptage fautif.

def implicit_torus(Pts, R_maj=1.0, R_min=0.35):
    """f = (√(x²+y²) − R_maj)² + z² − R_min² : < 0 dedans, > 0 dehors."""
    Pts = np.asarray(Pts, dtype=np.float64).reshape(-1, 3)
    rho = np.sqrt(Pts[:, 0] ** 2 + Pts[:, 1] ** 2)
    return (rho - R_maj) ** 2 + Pts[:, 2] ** 2 - R_min ** 2


def torus_normals(Pts, R_maj=1.0, R_min=0.35):
    """Normales sortantes analytiques du tore."""
    Pts = np.asarray(Pts, dtype=np.float64).reshape(-1, 3)
    x, y, z = Pts[:, 0], Pts[:, 1], Pts[:, 2]
    rho = np.maximum(np.sqrt(x * x + y * y), 1e-12)
    a = 2.0 * (rho - R_maj)
    g = np.column_stack([a * x / rho, a * y / rho, 2.0 * z])
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)


def sample_torus_surface(n=6000, R_maj=1.0, R_min=0.35, seed=None):
    """
    Points sur le tore, répartis proportionnellement à l'aire.

    L'élément d'aire vaut R_min·(R_maj + R_min·cos φ)·dφ·dθ : la densité en φ
    n'est pas uniforme (le bord extérieur est plus grand que l'intérieur), d'où
    un tirage par rejet sur φ.
    """
    rng = np.random.default_rng(seed)
    phi = np.empty(0)
    while len(phi) < n:
        cand = 2 * np.pi * rng.random(2 * n)
        keep = rng.random(2 * n) < (R_maj + R_min * np.cos(cand)) / (R_maj + R_min)
        phi = np.concatenate([phi, cand[keep]])
    phi = phi[:n]
    theta = 2 * np.pi * rng.random(n)
    rho = R_maj + R_min * np.cos(phi)
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta),
                            R_min * np.sin(phi)])
