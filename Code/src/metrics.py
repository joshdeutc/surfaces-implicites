"""
Métriques de qualité de reconstruction.

Point important du projet : aucune de ces mesures n'a besoin de la VALEUR de la
fonction implicite exacte. Elles se calculent à partir d'un nuage de points de
référence et d'un test dedans/dehors — c'est-à-dire de ce dont on disposera sur
les vraies capsules du BMBI, où aucune formule n'existera.

À comparer avec `mean|f_exact|`, utilisée jusqu'ici, qui exige la formule et
n'est donc pas transposable aux données réelles.

Conventions retenues (à citer telles quelles dans le rapport) :

- Chamfer : moyenne symétrique des distances au plus proche voisin, dans les
  UNITÉS DE L'ESPACE. La littérature diverge — DeepSDF somme des carrés sur
  30 000 points puis multiplie par 10³, Occupancy Networks fait une moyenne L1
  sur 100 000 points normalisée par la boîte englobante. On prend la moyenne L1
  symétrique non normalisée : elle se lit directement (0.015 = 1.5 centième
  d'unité d'écart moyen) et se compare à la maille de la grille.

- IoU : rapport des volumes intersection/union, estimé par Monte-Carlo sur des
  points tirés uniformément dans la boîte (protocole Occupancy Networks).
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

from src.device import get_device


# ------------------------------------------------------------
# Chamfer / Hausdorff — ne demandent qu'un nuage de référence
# ------------------------------------------------------------

def chamfer_distance(points_a, points_b):
    """
    Distance de Chamfer symétrique entre deux nuages (N,3) et (M,3).

    Aucune formule implicite n'intervient : c'est une comparaison nuage à nuage.

    Renvoie un dict :
      a_to_b    : distance moyenne de A vers son plus proche voisin dans B,
      b_to_a    : l'inverse,
      chamfer   : demi-somme des deux (mesure symétrique, unités de l'espace),
      hausdorff : pire écart des deux sens (épingle les points aberrants, par
                  exemple les zéros parasites d'un champ mal régularisé).
    """
    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("chamfer_distance : nuage vide")

    d_ab, _ = cKDTree(b).query(a)
    d_ba, _ = cKDTree(a).query(b)
    a_to_b, b_to_a = float(d_ab.mean()), float(d_ba.mean())
    return {
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "chamfer": 0.5 * (a_to_b + b_to_a),
        "hausdorff": float(max(d_ab.max(), d_ba.max())),
    }


# ------------------------------------------------------------
# IoU d'occupation — ne demande qu'un test dedans/dehors
# ------------------------------------------------------------

def iou_occupancy(inside_pred, inside_ref, box, n_samples=100000, seed=0,
                  chunk=50000):
    """
    IoU volumétrique par Monte-Carlo, sur des points uniformes dans [-box, box]^3.

    `inside_pred` et `inside_ref` sont des CALLABLES (N,3) -> booléen (N,).
    Sur les vraies données, `inside_ref` viendrait de l'occupation du maillage
    Éléments Finis ; ici on la fabrique avec la formule, mais UNIQUEMENT pour
    évaluer.

    `box` peut être un scalaire (boîte cubique) ou un triplet de demi-tailles.
    """
    rng = np.random.default_rng(seed)
    half = np.asarray(box, dtype=np.float64)
    if half.ndim == 0:
        half = np.repeat(half, 3)

    inter = 0
    union = 0
    remaining = int(n_samples)
    while remaining > 0:
        n = min(chunk, remaining)
        pts = rng.uniform(-half, half, size=(n, 3)).astype(np.float32)
        p = np.asarray(inside_pred(pts)).ravel().astype(bool)
        r = np.asarray(inside_ref(pts)).ravel().astype(bool)
        inter += int(np.logical_and(p, r).sum())
        union += int(np.logical_or(p, r).sum())
        remaining -= n

    return float(inter) / float(union) if union > 0 else 0.0


def model_inside(model, device=None, chunk=50000):
    """
    Enveloppe un réseau en test dedans/dehors (dedans = f < 0), utilisable
    directement comme `inside_pred` de `iou_occupancy`.
    """
    device = device or get_device()
    model = model.to(device).eval()

    def _inside(points):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        out = np.empty(len(pts), dtype=bool)
        with torch.no_grad():
            for i in range(0, len(pts), chunk):
                block = torch.as_tensor(pts[i:i + chunk], device=device)
                out[i:i + chunk] = model(block).cpu().numpy().ravel() < 0.0
        return out

    return _inside


def iou_from_model(model, inside_ref, box, n_samples=100000, seed=0):
    """Raccourci : IoU entre un réseau appris et une occupation de référence."""
    return iou_occupancy(model_inside(model), inside_ref, box,
                         n_samples=n_samples, seed=seed)


# ------------------------------------------------------------
# Cohérence des normales et qualité du champ
# ------------------------------------------------------------

def normal_consistency(verts_a, normals_a, verts_b, normals_b):
    """
    Moyenne de |n_a · n_b| entre plus proches voisins, dans les deux sens.

    La valeur absolue absorbe l'ambiguïté d'orientation des normales de maillage
    (convention DeepSDF). 1.0 = normales parfaitement alignées.
    """
    va = np.asarray(verts_a, dtype=np.float64).reshape(-1, 3)
    vb = np.asarray(verts_b, dtype=np.float64).reshape(-1, 3)
    na = _unit(np.asarray(normals_a, dtype=np.float64).reshape(-1, 3))
    nb = _unit(np.asarray(normals_b, dtype=np.float64).reshape(-1, 3))

    _, i_ab = cKDTree(vb).query(va)
    _, i_ba = cKDTree(va).query(vb)
    c_ab = np.abs(np.einsum("ij,ij->i", na, nb[i_ab])).mean()
    c_ba = np.abs(np.einsum("ij,ij->i", nb, na[i_ba])).mean()
    return float(0.5 * (c_ab + c_ba))


def _unit(v, eps=1e-12):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), eps)


def eikonal_error(model, box, n=100000, seed=0, chunk=20000, device=None):
    """
    Écart moyen |‖∇f‖ − 1| sur des points tirés dans la boîte.

    Mesure la qualité du champ appris en tant que distance signée, sans aucune
    référence extérieure : c'est la seule métrique ici qui ne regarde même pas
    la forme, seulement la régularité de f.
    """
    device = device or get_device()
    model = model.to(device).eval()

    half = float(box)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    total, count = 0.0, 0
    remaining = int(n)
    while remaining > 0:
        m = min(chunk, remaining)
        pts = (torch.rand(m, 3, generator=gen) * 2 - 1) * half
        x = pts.to(device).requires_grad_(True)
        f = model(x)
        g = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=False)[0]
        total += float((g.norm(dim=-1) - 1.0).abs().sum().item())
        count += m
        remaining -= m
    return total / count


# ------------------------------------------------------------
# Agrégation
# ------------------------------------------------------------

def evaluate_reconstruction(model, verts, surface_ref, inside_ref, box,
                            n_samples=100000, seed=0):
    """
    Les métriques d'un modèle appris et de son maillage reconstruit, réunies.

    verts       : sommets du maillage marching cubes (N,3),
    surface_ref : nuage dense de référence sur la vraie surface (M,3),
    inside_ref  : test dedans/dehors de référence (callable) pour l'IoU.
    """
    ch = chamfer_distance(verts, surface_ref)
    return {
        "chamfer": ch["chamfer"],
        "hausdorff": ch["hausdorff"],
        "iou": iou_from_model(model, inside_ref, box,
                              n_samples=n_samples, seed=seed),
        "eikonal": eikonal_error(model, box, seed=seed),
    }


def format_metrics(m):
    """Ligne lisible pour la console."""
    parts = []
    for key, label in (("chamfer", "Chamfer"), ("hausdorff", "Hausdorff"),
                       ("iou", "IoU"), ("eikonal", "eikonal")):
        if key in m and m[key] is not None:
            parts.append(f"{label}={m[key]:.4f}")
    return "  ".join(parts)
