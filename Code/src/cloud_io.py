"""
Format d'échange d'un nuage de points de surface, et fabrication d'un faux
nuage « sortie Éléments Finis » pour éprouver la chaîne avant l'arrivée des
données du BMBI.

Pourquoi ce module existe
-------------------------
Tout le code du projet part jusqu'ici d'une forme ANALYTIQUE : on appelle
`get_surface(...)` qui renvoie une fonction `implicit_exact`. Sur les vraies
capsules il n'y aura pas de fonction — il y aura un FICHIER, produit par un
calcul Éléments Finis ou par une acquisition, contenant :

    - des points échantillonnant la surface,
    - un vecteur normal en chacun d'eux, sortant du volume,
    - la boîte d'étude.

C'est exactement le contenu du .npz défini ici. Une fois ce fichier écrit, la
suite de la chaîne (signe dedans/dehors, entraînement, reconstruction) ne doit
plus jamais toucher à la formule : `load_cloud` est la seule porte d'entrée.

D'où viennent les normales
--------------------------
Elles ne sont PAS le gradient analytique. On suit le paragraphe « Plus
difficile » du mail de F. De Vuyst : on maille la surface par des triangles, on
numérote les sommets de chaque triangle dans le sens trigonométrique, et la
normale du triangle K est alors

    n_K = (v1 − v0) × (v2 − v0),   normalisée,

qui pointe vers l'extérieur. Les normales aux sommets s'obtiennent en moyennant
les normales des triangles adjacents, pondérées par leur aire.

L'orientation globale du maillage se règle SANS aucune formule, par le volume
signé du maillage fermé :

    V = (1/6) Σ_K  v0 · (v1 × v2)

qui est positif si et seulement si les normales sortent. C'est le théorème de la
divergence appliqué au champ x ↦ x/3 ; il ne demande que la géométrie du
maillage. Si V < 0, on inverse l'ordre des sommets de tous les triangles.
"""
import numpy as np
from skimage import measure

from src.shapes import get_surface


# ------------------------------------------------------------
# Écriture / lecture du format d'échange
# ------------------------------------------------------------

def save_cloud(path, points, normals, box, source="inconnu"):
    """
    Écrit un nuage orienté au format .npz.

    points  : (N,3) points de surface
    normals : (N,3) normales unitaires sortantes
    box     : demi-taille de la boîte d'étude (scalaire)
    source  : étiquette libre, tracée dans le fichier (provenance des données)
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    Nn = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(P) != len(Nn):
        raise ValueError(f"{len(P)} points mais {len(Nn)} normales")
    norms = np.linalg.norm(Nn, axis=1)
    if not np.all(norms > 1e-12):
        raise ValueError("au moins une normale est nulle")
    Nn = Nn / norms[:, None]

    np.savez(path, points=P, normals=Nn,
             box=np.float64(box), source=np.array(str(source)))
    return path


def load_cloud(path):
    """
    Relit un nuage. Seule porte d'entrée autorisée pour l'entraînement.

    Renvoie un dict {points, normals, box, source, n}. Le chargement se fait
    avec allow_pickle=False : rien d'exécutable ne peut entrer par ce fichier.
    """
    with np.load(path, allow_pickle=False) as d:
        points = np.asarray(d["points"], dtype=np.float64)
        normals = np.asarray(d["normals"], dtype=np.float64)
        box = float(d["box"])
        source = str(d["source"]) if "source" in d.files else "inconnu"

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points de forme {points.shape}, attendu (N,3)")
    if normals.shape != points.shape:
        raise ValueError("points et normales de tailles incompatibles")
    if not np.all(np.abs(points) <= box + 1e-9):
        raise ValueError("des points sortent de la boîte annoncée")

    return {"points": points, "normals": normals, "box": box,
            "source": source, "n": len(points)}


# ------------------------------------------------------------
# Maillage triangulaire : normales et orientation
# ------------------------------------------------------------

def face_normals(verts, faces, normalise=True):
    """
    Normale de chaque triangle : (v1−v0) × (v2−v0).

    Sans normalisation, la longueur du produit vectoriel vaut deux fois l'aire
    du triangle — c'est la pondération que l'on veut pour les normales aux
    sommets, et c'est pourquoi `normalise=False` existe.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    if not normalise:
        return n
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def mesh_signed_volume(verts, faces):
    """
    Volume signé du maillage fermé, (1/6) Σ v0 · (v1 × v2).

    Positif si les triangles sont orientés dans le sens trigonométrique vu de
    l'extérieur, donc si les normales sortent. Sert de test d'orientation sans
    la moindre référence à une fonction implicite.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def orient_outward(verts, faces):
    """
    Renumérote les sommets pour que toutes les normales sortent.

    Renvoie (faces_orientées, volume, a_ete_inverse).
    """
    vol = mesh_signed_volume(verts, faces)
    if vol < 0:
        faces = faces[:, [0, 2, 1]].copy()
        vol = -vol
        return faces, vol, True
    return faces, vol, False


def vertex_normals(verts, faces):
    """
    Normale en chaque sommet : moyenne des normales des triangles adjacents,
    pondérée par leur aire (pondération naturelle, obtenue en sommant les
    produits vectoriels NON normalisés).
    """
    contrib = face_normals(verts, faces, normalise=False)
    acc = np.zeros_like(verts)
    for c in range(3):
        np.add.at(acc, faces[:, c], contrib)
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-300)


# ------------------------------------------------------------
# Faux nuage « sortie Éléments Finis »
# ------------------------------------------------------------

_MESH_CACHE = {}


def fem_like_mesh(shape="ellipsoid", grid_res=160, chunk=500000):
    """
    Maillage surfacique triangulaire d'une forme analytique.

    C'est l'ÉTAPE DE GÉNÉRATION DE DONNÉES : la formule n'intervient qu'ici,
    pour fabriquer le maillage qui tient lieu de sortie Éléments Finis. Tout ce
    qui suit ne verra que des sommets et des triangles.

    Renvoie (verts, faces, normals, box, info).
    """
    cle = (shape, grid_res)
    if cle in _MESH_CACHE:
        return _MESH_CACHE[cle]

    _, implicit_exact, box = get_surface(shape)

    axe = np.linspace(-box, box, grid_res)
    X, Y, Z = np.meshgrid(axe, axe, axe, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    vals = np.empty(len(pts), dtype=np.float64)
    for i in range(0, len(pts), chunk):
        vals[i:i + chunk] = implicit_exact(pts[i:i + chunk])
    vals = vals.reshape(grid_res, grid_res, grid_res)

    pas = axe[1] - axe[0]
    verts, faces, _, _ = measure.marching_cubes(vals, level=0.0,
                                                spacing=(pas, pas, pas))
    verts = verts + axe[0]          # marching cubes part de l'origine

    faces, volume, inverse = orient_outward(verts, faces)
    normals = vertex_normals(verts, faces)

    info = {"grid_res": grid_res, "n_verts": len(verts), "n_faces": len(faces),
            "volume": volume, "orientation_inversee": inverse, "box": box}
    _MESH_CACHE[cle] = (verts, faces, normals, box, info)
    return _MESH_CACHE[cle]


def make_fem_cloud(shape="ellipsoid", n_points=6000, grid_res=160, seed=0,
                   noise=0.0):
    """
    Nuage orienté de `n_points` points, tiré sur le maillage triangulaire.

    noise : écart-type d'un bruit gaussien ajouté aux POSITIONS, pour imiter
        l'imprécision d'une mesure réelle. Les normales, elles, restent celles
        du maillage — c'est le cas favorable, à garder en tête.

    Renvoie (points, normals, box, info).
    """
    verts, faces, normals, box, info = fem_like_mesh(shape, grid_res=grid_res)
    if n_points > len(verts):
        raise ValueError(f"{n_points} points demandés mais le maillage n'en a "
                         f"que {len(verts)} (augmenter grid_res)")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(verts), size=n_points, replace=False)
    P = verts[idx].copy()
    Nn = normals[idx].copy()
    if noise > 0:
        P = P + noise * rng.standard_normal(P.shape)

    info = dict(info, n_points=n_points, seed=seed, noise=noise)
    return P, Nn, box, info


def generate_cloud_file(path, shape="ellipsoid", n_points=6000, grid_res=160,
                        seed=0, noise=0.0):
    """Fabrique le nuage et l'écrit sur disque. Renvoie (chemin, info)."""
    P, Nn, box, info = make_fem_cloud(shape, n_points=n_points,
                                      grid_res=grid_res, seed=seed, noise=noise)
    etiquette = f"maillage triangulaire de '{shape}', {n_points} points"
    if noise > 0:
        etiquette += f", bruit {noise:g}"
    save_cloud(path, P, Nn, box, source=etiquette)
    return path, info
