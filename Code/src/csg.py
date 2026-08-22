"""
Morphing et opérations booléennes sur des surfaces implicites.

C'est la recommandation (d) du mail de F. De Vuyst
(Encadrement/mail_recommandations_surfaces-implicites.md) :

    « les surfaces implicites sont très pratiques pour faire du morphing de
      formes : pour une forme f1 = 0 et une autre f2 = 0, une forme interpolée
      sera donnée par theta·f1 + (1−theta)·f2 = 0, avec theta ∈ [0,1]. On peut
      aussi définir des opérations d'union, d'intersection ou de différences de
      formes, très facilement. D'où toute la puissance des surfaces implicites. »

Tout le module manipule des CHAMPS, c'est-à-dire des fonctions
(N,3) -> (N,) qui rendent f < 0 dedans, f > 0 dehors. Il n'a jamais besoin de
savoir d'où vient le champ : formule analytique ou réseau de neurones, c'est le
même code. `from_model` fait le pont — il emballe un module torch dans un champ
numpy — et c'est la seule ligne qui change entre les deux démonstrations.

C'est là l'intérêt de la représentation implicite : combiner deux maillages
demanderait un algorithme de découpe de triangles ; ici une union est un
min(), et elle coûte le même prix sur un réseau que sur un polynôme.

CONVENTIONS
-----------
    union(f1, f2)        = min(f1, f2)     dedans si dedans l'un OU l'autre
    intersection(f1, f2) = max(f1, f2)     dedans si dedans l'un ET l'autre
    difference(f1, f2)   = max(f1, −f2)    dedans f1 mais PAS dans f2
    complement(f)        = −f

Ces règles sont exactes au sens ENSEMBLISTE : l'ensemble {min(f1,f2) < 0} est
rigoureusement la réunion de {f1 < 0} et {f2 < 0}, quel que soit le signe des
valeurs ailleurs. C'est ce qui rend le test de volume du fichier main_morph.py
concluant : V(A∪B) + V(A∩B) = V(A) + V(B) est une identité d'ensembles, donc
tout écart mesuré ne peut venir que de la discrétisation, pas des formules.

En revanche min et max ne préservent PAS le caractère de distance signée : le
champ résultant a un gradient discontinu le long de l'intersection des deux
surfaces. C'est sans conséquence pour le marching cubes (qui ne lit que le
signe et interpole linéairement), mais cela interdit d'en calculer des dérivées
secondes sur l'arête — c'est exactement le défaut relevé sur `implicit_biconcave`
dans RESULTATS_courbures.md.
"""
import numpy as np


# ------------------------------------------------------------------
# Passerelle : un réseau de neurones vu comme un champ implicite
# ------------------------------------------------------------------

def from_model(model, device=None, chunk=65536):
    """
    Emballe un module torch (N,3)->(N,1) en un champ numpy (N,3)->(N,).

    Une fois passé ici, un champ APPRIS est indiscernable d'une formule pour
    le reste du module : morph, union, intersection s'y appliquent tels quels.

    L'évaluation se fait par paquets car les grilles de marching cubes font
    couramment 128³ = 2,1 millions de points.
    """
    import torch
    from src.device import get_device

    dev = torch.device(device) if device is not None else get_device()
    model = model.to(dev).eval()

    def f(P):
        P = np.asarray(P, dtype=np.float32).reshape(-1, 3)
        out = np.empty(len(P), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(P), chunk):
                block = torch.tensor(P[s:s + chunk], device=dev)
                out[s:s + chunk] = model(block).cpu().numpy().ravel()
        return out

    return f


# ------------------------------------------------------------------
# Morphing
# ------------------------------------------------------------------

def morph(f1, f2, theta):
    """
    Champ interpolé theta·f1 + (1−theta)·f2, formule du mail.

    theta = 0 rend f2, theta = 1 rend f1. La surface obtenue passe
    continûment de l'une à l'autre.

    ATTENTION — l'interpolation dépend de la NORMALISATION des deux champs :
    remplacer f1 par 10·f1 décrit la même surface mais déplace toutes les
    formes intermédiaires (theta = 0.5 donnerait alors la même chose que
    theta ≈ 0.09 avant). Seuls theta = 0 et theta = 1 sont invariants.
    C'est pourquoi `normalized` existe ci-dessous : elle remet les deux champs
    à une pente comparable au voisinage de leur surface, ce qui rend la
    trajectoire du morphing géométriquement sensée plutôt qu'arbitraire.
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta doit être dans [0,1], reçu {theta}")

    def f(P):
        return theta * np.asarray(f1(P), dtype=np.float64).ravel() \
            + (1.0 - theta) * np.asarray(f2(P), dtype=np.float64).ravel()

    return f


def morph_series(f1, f2, thetas):
    """Liste des champs interpolés pour une suite de theta."""
    return [morph(f1, f2, float(t)) for t in thetas]


# ------------------------------------------------------------------
# Opérations booléennes (CSG)
# ------------------------------------------------------------------

def union(*fields):
    """min des champs : dedans si dedans au moins un."""
    if len(fields) < 2:
        raise ValueError("union demande au moins deux champs")

    def f(P):
        vals = [np.asarray(g(P), dtype=np.float64).ravel() for g in fields]
        return np.min(np.column_stack(vals), axis=1)

    return f


def intersection(*fields):
    """max des champs : dedans si dedans tous."""
    if len(fields) < 2:
        raise ValueError("intersection demande au moins deux champs")

    def f(P):
        vals = [np.asarray(g(P), dtype=np.float64).ravel() for g in fields]
        return np.max(np.column_stack(vals), axis=1)

    return f


def complement(f1):
    """−f : échange l'intérieur et l'extérieur."""
    def f(P):
        return -np.asarray(f1(P), dtype=np.float64).ravel()
    return f


def difference(f1, f2):
    """max(f1, −f2) : dedans f1 mais dehors f2. C'est f1 ∩ complément(f2)."""
    return intersection(f1, complement(f2))


# ------------------------------------------------------------------
# Mise à l'échelle des champs
# ------------------------------------------------------------------

def gradient_norm(f, points, h=1e-4):
    """‖∇f‖ par différences finies centrées aux points donnés."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    e = np.eye(3) * h
    g = np.column_stack([
        (np.asarray(f(P + e[i])).ravel() - np.asarray(f(P - e[i])).ravel()) / (2.0 * h)
        for i in range(3)
    ])
    return np.linalg.norm(g, axis=1)


def normalized(f, surface_points, h=1e-4):
    """
    Renvoie f divisé par la valeur médiane de ‖∇f‖ sur sa propre surface.

    Après ce traitement, f varie d'environ 1 par unité de longueur au
    voisinage de la surface, comme le ferait une distance signée. Deux champs
    ainsi traités sont comparables terme à terme, ce qui donne un sens au
    theta du morphing. La surface f = 0 est évidemment inchangée : diviser par
    une constante positive ne déplace aucun point.

    Médiane plutôt que moyenne : sur une quartique ‖∇f‖ varie d'un facteur 10
    entre le creux et le bord, et la moyenne serait tirée par le bord.
    """
    s = float(np.median(gradient_norm(f, surface_points, h)))
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"norme de gradient médiane invalide : {s}")

    def g(P):
        return np.asarray(f(P), dtype=np.float64).ravel() / s

    g.scale = s
    return g


def translated(f, offset):
    """Champ translaté : f(x − offset)."""
    off = np.asarray(offset, dtype=np.float64).reshape(1, 3)

    def g(P):
        return np.asarray(f(np.asarray(P, dtype=np.float64).reshape(-1, 3) - off)).ravel()

    return g
