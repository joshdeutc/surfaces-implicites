"""
Surfaces implicites complexes : gyroïde, Chmutov, sextique de Barth, metaballs.

POURQUOI CES FORMES-LÀ
----------------------
La fiche d'enregistrement de la TZ leur consacre une ligne explicite (« 3h :
exploration de surfaces complexes : surfaces de Chmutov, gyroïdes, metaballs,
surfaces de Barth ») et l'énoncé du sujet s'ouvre sur l'argument qu'elles
servent à vérifier : « les équations implicites ont la capacité de représenter
des surfaces très complexes en termes de forme ET DE TOPOLOGIE ».

Or tout ce que le projet a validé jusqu'ici — sphère, ellipsoïde, globule
rouge — est de GENRE 0 : trois ballons sans trou, topologiquement le même
objet. L'argument central du sujet n'avait donc encore jamais été mis à
l'épreuve. C'est le seul endroit du code où la topologie n'est pas triviale.

Chaque forme apporte une difficulté différente :

  gyroïde   — surface périodique NON FERMÉE, sans intérieur borné. Elle casse
              une hypothèse du pipeline, et c'est justement l'intérêt : voir
              `RESULTATS_rendu_formes.md` pour ce que cela invalide.
  Chmutov   — algébrique de degré 4, plusieurs composantes connexes, genre
              élevé. Teste la capacité du maillage à compter juste.
  Barth     — sextique à 65 points doubles : la surface est SINGULIÈRE, le
              gradient s'y annule. C'est le pire cas possible pour une méthode
              qui suppose ‖∇f‖ = 1.
  metaballs — champ lisse dont la topologie DÉPEND D'UN SEUIL : les mêmes
              boules donnent un tore ou une sphère selon l'iso-valeur. C'est le
              cas qui ressemble le plus aux capsules du BMBI, où deux capsules
              qui se touchent puis fusionnent changent de topologie en cours
              de déformation.

Convention du projet respectée partout : f < 0 dedans, f = 0 surface,
f > 0 dehors — sauf pour la gyroïde, où « dedans » n'existe pas (voir plus bas).
"""
import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0     # nombre d'or, coefficient de la Barth


def _as_points(P):
    A = np.asarray(P, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"points doit être de forme (N,3), reçu {A.shape}")
    return A[:, 0], A[:, 1], A[:, 2]


def _normalise(G, eps=1e-12):
    n = np.linalg.norm(G, axis=1, keepdims=True)
    return G / np.maximum(n, eps)


# ==================================================================
# 1. Gyroïde  —  sin x cos y + sin y cos z + sin z cos x = 0
# ==================================================================

def implicit_gyroid(P, periode=1.0):
    """
    Gyroïde de Schoen, approximation nodale classique. Période spatiale 2π/`periode`.

    ATTENTION — cette fonction n'a PAS de « dedans ». La gyroïde est une
    surface triplement périodique de courbure moyenne quasi nulle qui sépare
    l'espace en deux labyrinthes IMBRIQUÉS et non bornés, images l'un de
    l'autre. Le signe de f distingue bien les deux labyrinthes, mais aucun des
    deux n'est un intérieur : il n'y a pas de volume fini à remplir.
    """
    x, y, z = _as_points(P)
    k = float(periode)
    return (np.sin(k * x) * np.cos(k * y)
            + np.sin(k * y) * np.cos(k * z)
            + np.sin(k * z) * np.cos(k * x))


def gyroid_normals(P, periode=1.0):
    x, y, z = _as_points(P)
    k = float(periode)
    gx = k * (np.cos(k * x) * np.cos(k * y) - np.sin(k * z) * np.sin(k * x))
    gy = k * (np.cos(k * y) * np.cos(k * z) - np.sin(k * x) * np.sin(k * y))
    gz = k * (np.cos(k * z) * np.cos(k * x) - np.sin(k * y) * np.sin(k * z))
    return _normalise(np.column_stack([gx, gy, gz]))


# ==================================================================
# 2. Chmutov  —  T_n(x) + T_n(y) + T_n(z) = 0
# ==================================================================

def _chebyshev(t, n):
    """T_n et T_n' par la récurrence, valable hors de [-1,1] contrairement au cosinus."""
    T0 = np.ones_like(t)
    T1 = t
    D0 = np.zeros_like(t)
    D1 = np.ones_like(t)
    if n == 0:
        return T0, D0
    for _ in range(2, n + 1):
        T2 = 2.0 * t * T1 - T0
        D2 = 2.0 * T1 + 2.0 * t * D1 - D0
        T0, T1 = T1, T2
        D0, D1 = D1, D2
    return T1, D1


def implicit_chmutov(P, degre=4):
    """
    Surface de Chmutov de degré `degre` : somme des polynômes de Tchebychev.

    Le degré fixe le nombre de bosses par axe, donc le genre. Le degré 4 donne
    déjà plusieurs composantes connexes — de quoi vérifier que le comptage
    d'Euler du projet ne suppose nulle part la connexité.

    La surface est contenue dans [-1.06, 1.06]^3 : |T_n| >= 1 hors de [-1,1] et
    croît vite, donc la somme ne peut s'annuler que près du cube unité.
    """
    x, y, z = _as_points(P)
    return (_chebyshev(x, degre)[0] + _chebyshev(y, degre)[0]
            + _chebyshev(z, degre)[0])


def chmutov_normals(P, degre=4):
    x, y, z = _as_points(P)
    return _normalise(np.column_stack([_chebyshev(x, degre)[1],
                                       _chebyshev(y, degre)[1],
                                       _chebyshev(z, degre)[1]]))


# ==================================================================
# 3. Sextique de Barth  —  65 points doubles
# ==================================================================

def implicit_barth(P, w=1.0):
    """
    4(φ²x²−y²)(φ²y²−z²)(φ²z²−x²) − (1+2φ)(x²+y²+z²−w²)² w² = 0

    Sextique de Barth (1996) : c'est la surface de degré 6 qui possède le plus
    grand nombre possible de points doubles ordinaires, 65. En ces points le
    gradient s'ANNULE.

    C'est ce qui en fait le cas limite du projet. La perte IGR impose
    ‖∇f‖ = 1 partout ; ici la vraie fonction viole cette condition en 65
    points, et son gradient varie de plusieurs ordres de grandeur ailleurs.
    Aucune fonction de distance signée lisse ne peut représenter exactement
    cette forme.
    """
    x, y, z = _as_points(P)
    p2 = PHI * PHI
    terme = 4.0 * (p2 * x * x - y * y) * (p2 * y * y - z * z) * (p2 * z * z - x * x)
    sphere = (x * x + y * y + z * z - w * w)
    return terme - (1.0 + 2.0 * PHI) * sphere * sphere * w * w


def barth_normals(P, w=1.0):
    x, y, z = _as_points(P)
    p2 = PHI * PHI
    A = p2 * x * x - y * y
    B = p2 * y * y - z * z
    C = p2 * z * z - x * x
    s = (x * x + y * y + z * z - w * w)
    k = 4.0 * (1.0 + 2.0 * PHI) * w * w * s
    gx = 4.0 * (2.0 * p2 * x * B * C + A * B * (-2.0 * x)) - k * x
    gy = 4.0 * (-2.0 * y * B * C + A * (2.0 * p2 * y) * C) - k * y
    gz = 4.0 * (A * (-2.0 * z) * C + A * B * (2.0 * p2 * z)) - k * z
    return _normalise(np.column_stack([gx, gy, gz]))


# ==================================================================
# 4. Metaballs  —  topologie pilotée par un seuil
# ==================================================================

METABALL_PRESETS = {
    # six boules en anneau : elles fusionnent deux à deux mais le trou central
    # reste ouvert -> TORE, genre 1, chi = 0
    "anneau": dict(centres=None, n=6, rayon_anneau=1.0, sigma=0.35, iso=0.40),
    # deux boules qui se recouvrent : une seule masse -> genre 0, chi = 2.
    # C'est le cas « deux capsules qui fusionnent » du labo.
    "fusion": dict(centres=np.array([[-0.45, 0.0, 0.0], [0.45, 0.0, 0.0]]),
                   sigma=0.45, iso=0.55),
}


def metaball_centres(preset="anneau"):
    cfg = dict(METABALL_PRESETS[preset])
    if cfg.get("centres") is not None:
        return np.asarray(cfg["centres"], dtype=np.float64), cfg["sigma"], cfg["iso"]
    n, a = cfg["n"], cfg["rayon_anneau"]
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    C = np.column_stack([a * np.cos(th), a * np.sin(th), np.zeros(n)])
    return C, cfg["sigma"], cfg["iso"]


def implicit_metaballs(P, centres, sigma=0.35, iso=0.40):
    """
    f(P) = iso − Σ_i exp(−‖P−c_i‖² / 2σ²)

    Noyau gaussien plutôt que le noyau polynomial de Blinn : il est C^∞, donc
    ses dérivées secondes existent partout et les courbures du projet restent
    calculables. f < 0 là où l'influence cumulée dépasse le seuil : la
    convention dedans/dehors du projet est respectée.

    `iso` est le paramètre intéressant : à sigma et centres fixés, le faire
    varier fait passer la forme de N boules séparées à une masse unique, en
    traversant des changements de TOPOLOGIE. Une surface implicite les encaisse
    sans rien changer à sa représentation ; un maillage, non.
    """
    A = np.asarray(P, dtype=np.float64)
    C = np.asarray(centres, dtype=np.float64)
    d2 = ((A[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return float(iso) - np.exp(-d2 / (2.0 * sigma * sigma)).sum(axis=1)


def metaballs_normals(P, centres, sigma=0.35, iso=0.40):
    A = np.asarray(P, dtype=np.float64)
    C = np.asarray(centres, dtype=np.float64)
    diff = A[:, None, :] - C[None, :, :]
    w = np.exp(-(diff ** 2).sum(axis=2) / (2.0 * sigma * sigma))
    G = (w[:, :, None] * diff).sum(axis=1) / (sigma * sigma)
    return _normalise(G)


# ==================================================================
# Échantillonnage de surface par projection de Newton
# ==================================================================

def newton_projection(f, grad, points, n_iter=40, tol=1e-7, pas_max=0.25):
    """
    Ramène des points quelconques sur la surface f = 0.

    Itération : P <- P − f(P) · n(P) / ‖∇f(P)‖, avec `grad` renvoyant la
    normale UNITAIRE ; le déplacement vaut donc f/‖∇f‖, l'estimation au premier
    ordre de la distance à la surface.

    `pas_max` borne chaque déplacement. Sans cette borne la sextique de Barth
    diverge : son gradient s'annule aux 65 points doubles, et un pas en 1/‖∇f‖
    y devient infini. C'est le même phénomène qui rendra son apprentissage
    difficile.
    """
    P = np.array(points, dtype=np.float64, copy=True)
    for _ in range(int(n_iter)):
        val = np.asarray(f(P), dtype=np.float64).ravel()
        if np.max(np.abs(val)) < tol:
            break
        G = grad(P)                                  # normales unitaires
        gn = np.linalg.norm(_gradient_brut(f, P), axis=1)
        pas = np.clip(val / np.maximum(gn, 1e-9), -pas_max, pas_max)
        P = P - pas[:, None] * G
    return P


def _gradient_brut(f, P, h=1e-5):
    """Norme du gradient par différences finies — `grad` ne renvoie que la direction."""
    e = np.eye(3) * h
    return np.column_stack([
        (np.asarray(f(P + e[i])).ravel() - np.asarray(f(P - e[i])).ravel()) / (2 * h)
        for i in range(3)])


def sample_surface(f, grad, box, n=6000, seed=0, tol=1e-4, sur_tirage=8):
    """
    Nuage de points sur f = 0, par tirage uniforme puis projection de Newton.

    On sur-tire (`sur_tirage`×) parce que la projection échoue pour une part
    des germes : ceux qui partent d'une zone où le gradient est plat, et ceux
    qui convergent hors de la boîte. On ne garde que les points effectivement
    convergés ET restés dans la boîte, puis on complète si besoin.

    `box` : scalaire (cube) ou triplet de demi-tailles.
    """
    rng = np.random.default_rng(seed)
    half = np.asarray(box, dtype=np.float64)
    if half.ndim == 0:
        half = np.repeat(half, 3)

    gardes = []
    total = 0
    for _ in range(6):
        germes = rng.uniform(-half, half, size=(int(n * sur_tirage), 3))
        P = newton_projection(f, grad, germes)
        val = np.abs(np.asarray(f(P)).ravel())
        dans = np.all(np.abs(P) <= half * 1.001, axis=1)
        ok = np.isfinite(P).all(axis=1) & (val < tol) & dans
        gardes.append(P[ok])
        total += int(ok.sum())
        if total >= n:
            break
    P = np.vstack(gardes) if gardes else np.zeros((0, 3))
    if len(P) > n:
        P = P[rng.choice(len(P), n, replace=False)]
    return P


# ==================================================================
# Catalogue
# ==================================================================

def catalogue():
    """
    Les cinq surfaces avec leur boîte d'étude et ce que la théorie en dit.

    `fermee` est le champ décisif : il dit si le pipeline du projet — qui
    suppose partout une surface FERMÉE dotée d'un intérieur borné — s'applique
    à cette forme. Deux des quatre surfaces demandées par la fiche ne sont pas
    fermées, et `main_shapes.py` le vérifie au lieu de le croire.

    « attendu » est ce que dit la théorie ; le tableau affiché par
    `main_shapes.py` donne ce qui est MESURÉ sur le maillage. L'écart entre les
    deux est l'information, pas l'inverse.
    """
    C_ann, s_ann, i_ann = metaball_centres("anneau")
    C_fus, s_fus, i_fus = metaball_centres("fusion")
    return [
        dict(nom="gyroïde", box=np.pi, fermee=False, chi_attendu=None,
             f=lambda P: implicit_gyroid(P),
             n=lambda P: gyroid_normals(P),
             note="périodique : sort de la boîte, aucun intérieur borné"),
        dict(nom="Chmutov degré 4", box=1.25, fermee=True, chi_attendu=None,
             f=lambda P: implicit_chmutov(P, degre=4),
             n=lambda P: chmutov_normals(P, degre=4),
             note="algébrique compacte, genre élevé — le seul cas fermé non trivial"),
        dict(nom="Barth sextique", box=2.0, fermee=False, chi_attendu=None,
             f=lambda P: implicit_barth(P),
             n=lambda P: barth_normals(P),
             note="65 points doubles ET branches non bornées : jamais close"),
        dict(nom="metaballs anneau", box=1.7, fermee=True, chi_attendu=0,
             f=lambda P: implicit_metaballs(P, C_ann, s_ann, i_ann),
             n=lambda P: metaballs_normals(P, C_ann, s_ann, i_ann),
             note="6 boules fusionnées en tore, genre 1 attendu"),
        dict(nom="metaballs fusion", box=1.4, fermee=True, chi_attendu=2,
             f=lambda P: implicit_metaballs(P, C_fus, s_fus, i_fus),
             n=lambda P: metaballs_normals(P, C_fus, s_fus, i_fus),
             note="2 boules fusionnées, genre 0 — cas « capsules qui se touchent »"),
    ]
