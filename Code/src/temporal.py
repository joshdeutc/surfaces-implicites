"""
Séquence d'INSTANTS DISCRETS d'un globule rouge en déformation.

Pourquoi ce module existe
-------------------------
Le « Graal » du sujet, c'est la surface qui évolue dans le temps : condenser une
forme mobile en UNE seule surface implicite espace-temps f(x,y,z,t) = 0.

Un code Éléments Finis ne rend pas une fonction du temps. Il rend une SUITE DE
FICHIERS, un par pas de temps sauvegardé — typiquement quelques dizaines pour un
calcul qui en a fait des milliers. C'est exactement ce que ce module fabrique :

    six fichiers .npz, aux instants t = 0, 0.2, 0.4, 0.6, 0.8, 1.0,
    chacun contenant des points de surface et leurs normales sortantes.

Le réseau ne verra QUE ces six fichiers. Tout ce qu'il saura du mouvement, il
devra l'avoir déduit de six photographies.

La déformation choisie
----------------------
Un étirement selon z, à VOLUME CONSTANT :

    A(t) = diag( xi^(-1/2), xi^(-1/2), xi ),      xi(t) = 1 + t/2

    det A = xi^(-1/2) · xi^(-1/2) · xi = 1   pour tout t.

C'est le geste élémentaire d'une capsule qui s'allonge dans un écoulement
élongationnel : elle s'étire dans une direction, se contracte dans les deux
autres, et son volume ne bouge pas (la membrane est incompressible et le fluide
intérieur aussi). xi va de 1.0 (globule au repos) à 1.5 (allongé de 50 %).

Le volume constant n'est pas un détail esthétique : il donne un INVARIANT que le
réseau ignore complètement pendant l'apprentissage et qu'on peut donc lui
opposer à la fin. Si le champ appris ne conserve pas le volume, ce n'est pas la
mesure qui est en cause, c'est le champ.

Quelle quartique
----------------
`implicit_rbc_unit` de src/shapes.py, c'est-à-dire la quartique de Kuchel
`implicit_rbc_quartic` composée avec l'homothétie qui ramène le bord à r = 1
(la fonction normalisée appelle littéralement la fonction physique). On préfère
la variante normalisée pour une raison numérique documentée dans le README : à
l'échelle physique (bord à 4 µm, ‖∇F‖ ≈ 134), une position stockée en float32 —
ce qu'utilise tout le pipeline d'apprentissage — décale déjà F de ~2e-5. Le
profil biconcave d'Evans-Fung, lui, est écarté : ses dérivées secondes divergent
au bord (cf. RESULTATS_courbures.md).

D'où viennent les normales
--------------------------
Du maillage triangulaire, pas du gradient analytique : on déforme les sommets du
maillage de référence, on réoriente les triangles par le volume signé, et on
moyenne les normales de face pondérées par l'aire. C'est la procédure décrite
par F. De Vuyst et déjà implémentée dans src/cloud_io.py.

Un point de méthode : les six nuages sont tirés avec des GRAINES DIFFÉRENTES.
Il n'y a donc aucune correspondance point à point d'un instant à l'autre. Un
calcul lagrangien en offrirait une, mais s'en passer est le cas défavorable, et
c'est celui qu'on veut éprouver.
"""
import os

import numpy as np

from src.cloud_io import (fem_like_mesh, orient_outward, save_cloud,
                          vertex_normals)
from src.shapes import (implicit_rbc_unit, rbc_coeffs, rbc_scale,
                        sample_rbc_unit_surface)


# ------------------------------------------------------------------
# La déformation
# ------------------------------------------------------------------

XI_DEBUT = 1.0
XI_FIN = 1.5

T_OBSERVES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)      # les six fichiers écrits
T_CACHES = (0.1, 0.3, 0.5, 0.7, 0.9)             # jamais montrés au réseau


def xi(t):
    """Facteur d'élongation à l'instant t ∈ [0,1]."""
    return XI_DEBUT + (XI_FIN - XI_DEBUT) * np.asarray(t, dtype=np.float64)


def dxi_dt(t=0.0):
    """dxi/dt — constant ici, mais nommé pour que la chaîne dérivée soit lisible."""
    return XI_FIN - XI_DEBUT


def diag_A(t):
    """Les trois coefficients diagonaux de A(t) : (xi^-1/2, xi^-1/2, xi)."""
    x = xi(t)
    return np.array([x ** -0.5, x ** -0.5, x], dtype=np.float64)


def diag_A_inv(t):
    """Les trois coefficients diagonaux de A(t)^-1 : (xi^1/2, xi^1/2, xi^-1)."""
    x = xi(t)
    return np.array([x ** 0.5, x ** 0.5, x ** -1.0], dtype=np.float64)


def d_diag_A_inv_dt(t):
    """
    Dérivée en temps des coefficients de A^-1 — sert au terme ∂f/∂t exact.

        d(xi^1/2)/dt = (1/2)·xi^(-1/2)·dxi/dt
        d(xi^-1 )/dt = −xi^(-2)·dxi/dt
    """
    x = xi(t)
    d = dxi_dt(t)
    return np.array([0.5 * x ** -0.5 * d, 0.5 * x ** -0.5 * d, -(x ** -2.0) * d],
                    dtype=np.float64)


def deformer(P, t):
    """Applique A(t) à un jeu de points (N,3) : configuration de référence -> instant t."""
    return np.asarray(P, dtype=np.float64).reshape(-1, 3) * diag_A(t)[None, :]


def dedeformer(P, t):
    """Applique A(t)^-1 : instant t -> configuration de référence."""
    return np.asarray(P, dtype=np.float64).reshape(-1, 3) * diag_A_inv(t)[None, :]


def volume_exact():
    """
    Volume enclos par la quartique normalisée, invariant du problème.

    Obtenu par quadrature du solide de révolution, V = 2·∫ 2πr·z(r) dr, avec
    z(r) la demi-épaisseur exacte. Ne dépend pas de t puisque det A(t) = 1.
    """
    from src.shapes import _rbc_z_half, rbc_radius_max
    P, Q, R = rbc_coeffs()
    L = rbc_scale()
    r_max = rbc_radius_max() / L
    # grille resserrée au bord, où z(r) a une tangente verticale
    u = np.linspace(0.0, 1.0, 400001)
    r = r_max * np.sin(0.5 * np.pi * u) ** 2
    z = np.sqrt(_rbc_z_half((r * L) ** 2, P, Q, R)) / L
    return float(2.0 * np.trapz(2.0 * np.pi * r * z, r))


# ------------------------------------------------------------------
# Le champ exact à l'instant t — RÉSERVÉ À LA MESURE
# ------------------------------------------------------------------
#
# Tout ce qui suit utilise la formule. Rien de tout cela n'entre dans
# l'entraînement : le réseau ne voit que les fichiers .npz. On s'en sert
# exclusivement pour NOTER le résultat, et pour fournir la trajectoire de
# référence du suivi lagrangien.

def f_exact(Pts, t):
    """f_t(x) = f_0(A(t)^-1 x) — la quartique transportée par la déformation."""
    return implicit_rbc_unit(dedeformer(Pts, t))


def grad_f_unit(Pts):
    """
    ∇ de la quartique NORMALISÉE, analytique.

    Pour la quartique physique F(x) = ρ⁴ + P(x²+y²) + Qz² + R :
        ∂F/∂x = 4x·ρ² + 2P·x,  ∂F/∂y = 4y·ρ² + 2P·y,  ∂F/∂z = 4z·ρ² + 2Q·z.
    La version normalisée vaut F(L·x)/L⁴, donc son gradient vaut ∇F(L·x)/L³.
    """
    Pts = np.asarray(Pts, dtype=np.float64).reshape(-1, 3)
    L = rbc_scale()
    Q3 = Pts * L
    P, Qc, _ = rbc_coeffs()
    x, y, z = Q3[:, 0], Q3[:, 1], Q3[:, 2]
    rho2 = x * x + y * y + z * z
    g = np.column_stack([4 * x * rho2 + 2 * P * x,
                         4 * y * rho2 + 2 * P * y,
                         4 * z * rho2 + 2 * Qc * z])
    return g / L ** 3


def grad_f_exact(Pts, t):
    """
    ∇_x f_t(x) = A^-1 · ∇f_0(A^-1 x).

    A est diagonale, donc A^-T = A^-1 et la règle de chaîne se réduit à une
    multiplication terme à terme.
    """
    ref = dedeformer(Pts, t)
    return grad_f_unit(ref) * diag_A_inv(t)[None, :]


def dfdt_exact(Pts, t):
    """
    ∂f_t/∂t (x) = ∇f_0(A^-1 x) · (dA^-1/dt) x.

    C'est la partie qui n'existe pas en statique : elle mesure à quelle vitesse
    la valeur du champ change en un point FIXE de l'espace, et c'est elle qui
    porte toute l'information de mouvement.
    """
    Pts = np.asarray(Pts, dtype=np.float64).reshape(-1, 3)
    ref = dedeformer(Pts, t)
    return np.einsum("ij,ij->i", grad_f_unit(ref), Pts * d_diag_A_inv_dt(t)[None, :])


def dedans_exact(t):
    """Test dedans/dehors de référence à l'instant t, pour l'IoU."""
    return lambda Pts: f_exact(Pts, t) < 0.0


def surface_dense(t, n=60000, seed=11):
    """
    Nuage dense sur la VRAIE surface à l'instant t — évaluation seulement.

    Les points sont tirés proportionnellement à l'aire dans la configuration de
    référence puis transportés. La déformation étant anisotrope, la densité qui
    en résulte n'est plus exactement proportionnelle à l'aire déformée ; à
    60 000 points l'effet sur une distance de Chamfer est négligeable devant le
    pas de la grille de reconstruction.
    """
    return deformer(sample_rbc_unit_surface(n=n, seed=seed), t)


# ------------------------------------------------------------------
# Écriture des six instants
# ------------------------------------------------------------------

def maillage_instant(t, grid_res=160):
    """
    Maillage triangulaire de la surface à l'instant t, normales sortantes.

    On maille UNE fois la configuration de référence par marching cubes, puis on
    transporte les sommets. C'est ce que fait un calcul lagrangien : la
    connectivité ne change pas, seules les positions bougent.

    Renvoie (sommets, triangles, normales, boîte, info).
    """
    verts, faces, _, box, info = fem_like_mesh("rbc_unit", grid_res=grid_res)
    V = deformer(verts, t)
    F, volume, inverse = orient_outward(V, faces)
    N = vertex_normals(V, F)
    info = dict(info, t=float(t), xi=float(xi(t)), volume_maillage=volume,
                orientation_inversee=inverse)
    return V, F, N, box, info


def ecrire_instant(chemin, t, n_points=6000, grid_res=160, seed=0):
    """Tire n_points sur le maillage de l'instant t et écrit le fichier .npz."""
    V, F, N, box, info = maillage_instant(t, grid_res=grid_res)
    if n_points > len(V):
        raise ValueError(f"{n_points} points demandés, le maillage n'en a que {len(V)}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), size=n_points, replace=False)
    save_cloud(chemin, V[idx], N[idx], box,
               source=f"globule quartique étiré, t={t:g}, xi={xi(t):.4f}, "
                      f"{n_points} points de maillage")
    return chemin, dict(info, n_points=n_points, seed=seed, chemin=chemin)


def ecrire_sequence(dossier, instants=T_OBSERVES, n_points=6000, grid_res=160):
    """
    Écrit la séquence complète. Renvoie la liste des (t, chemin, info).

    Une graine différente par instant : aucune correspondance point à point
    d'un fichier à l'autre.
    """
    os.makedirs(dossier, exist_ok=True)
    sortie = []
    for i, t in enumerate(instants):
        nom = f"instant_t{t:.2f}.npz".replace(".", "p", 1)
        chemin = os.path.join(dossier, f"instant_t{int(round(t * 100)):03d}.npz")
        _, info = ecrire_instant(chemin, t, n_points=n_points,
                                 grid_res=grid_res, seed=100 + i)
        sortie.append((float(t), chemin, info))
    return sortie
