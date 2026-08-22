"""
Dedans / dehors à partir du SEUL nuage de points orienté.

Méthode décrite par F. De Vuyst (mail de recommandations, paragraphe « Plus
difficile ») :

    « par des considérations de produit scalaire, en cherchant le point de la
      surface de minimum de distance, on peut déterminer si un point est à
      l'intérieur ou à l'extérieur. »

Pour un point P de l'espace :

  1. on cherche le point S du nuage le plus proche de P (arbre k-d, coût
     O(log N) au lieu des N comparaisons de la recherche naïve — c'est ce qui
     rend le procédé utilisable sur les M points d'un maillage volumique) ;
  2. on regarde le signe du produit scalaire

         s(P) = (P − S) · n(S)

     où n(S) est la normale sortante en S. Si P est du côté vers lequel pointe
     la normale, il est dehors ; sinon il est dedans.

Là où ça se gâte, et pourquoi le vote sur k voisins
---------------------------------------------------
La règle est exacte tant que le plus proche point de surface est bien le pied
de la projection orthogonale. Deux situations la mettent en défaut :

  - P est très près de la surface : (P − S) est court, le produit scalaire est
    petit devant le bruit d'échantillonnage et le signe devient un tirage ;
  - P est près de l'axe médian de la forme, là où plusieurs points de surface
    sont à égale distance. C'est précisément le cas du creux du globule rouge :
    depuis le centre, les deux cuvettes sont à peu près équidistantes, et le
    voisin le plus proche peut basculer de l'une à l'autre.

D'où le vote sur les k plus proches voisins : on interroge k points de surface
au lieu d'un seul et on tranche à la majorité. Un voisin isolé mal orienté ne
suffit plus à faire basculer la décision.

Convention du projet : dedans = f < 0, donc `sign` renvoie −1 dedans, +1 dehors.
"""
import numpy as np
from scipy.spatial import cKDTree


VOTES = ("majorite", "somme", "poids")


class OrientedCloud:
    """
    Nuage de points de surface muni de ses normales sortantes.

    Ne connaît aucune fonction implicite : tout ce qu'il sait faire, il le fait
    à partir de `points` et `normals`.
    """

    def __init__(self, points, normals):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if len(n) != len(self.points):
            raise ValueError("autant de normales que de points, sinon rien")
        self.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True),
                                      1e-300)
        self.tree = cKDTree(self.points)

    def __len__(self):
        return len(self.points)

    @classmethod
    def from_file(cls, path):
        """Construit le nuage depuis un .npz — sans passer par aucune formule."""
        from src.cloud_io import load_cloud
        d = load_cloud(path)
        obj = cls(d["points"], d["normals"])
        obj.box = d["box"]
        obj.source = d["source"]
        return obj

    # --------------------------------------------------------
    # Le produit scalaire, cœur de la méthode
    # --------------------------------------------------------

    def dots(self, query, k=1):
        """
        Produits scalaires (P − S_i) · n(S_i) pour les k plus proches voisins.

        Renvoie (d, s) de forme (M, k) : distances et produits scalaires,
        voisins triés du plus proche au plus lointain.
        """
        P = np.asarray(query, dtype=np.float64).reshape(-1, 3)
        k = int(k)
        if k < 1:
            raise ValueError("k doit valoir au moins 1")
        k = min(k, len(self.points))

        d, idx = self.tree.query(P, k=k)
        if k == 1:                       # cKDTree aplatit le cas k = 1
            d = d[:, None]
            idx = idx[:, None]

        ecart = P[:, None, :] - self.points[idx]        # (M, k, 3)
        s = np.einsum("mkj,mkj->mk", ecart, self.normals[idx])
        return d, s

    def sign(self, query, k=1, vote="majorite"):
        """
        +1 dehors, −1 dedans, décidé sur les k plus proches voisins.

        vote = "majorite" : chaque voisin apporte une voix ±1 (le vote demandé) ;
               "somme"    : on somme les produits scalaires, ce qui donne plus
                            de poids aux voisins qui tranchent nettement ;
               "poids"    : somme des voix pondérées par 1/distance, pour que le
                            voisin le plus proche pèse davantage.

        En cas d'égalité parfaite, on retombe sur la décision du voisin le plus
        proche — c'est-à-dire sur la règle brute k = 1.
        """
        if vote not in VOTES:
            raise ValueError(f"vote inconnu : {vote!r}, attendu parmi {VOTES}")
        d, s = self.dots(query, k=k)

        if vote == "majorite":
            score = np.sign(s).sum(axis=1)
        elif vote == "somme":
            score = s.sum(axis=1)
        else:
            score = (np.sign(s) / np.maximum(d, 1e-12)).sum(axis=1)

        out = np.sign(score)
        indecis = out == 0
        if np.any(indecis):                       # égalité : on tranche au plus proche
            secours = np.sign(s[indecis, 0])
            secours[secours == 0] = 1.0           # pile sur la surface : dehors
            out[indecis] = secours
        return out

    def inside(self, query, k=1, vote="majorite"):
        """Test booléen dedans/dehors, prêt à servir d'`inside_pred` à l'IoU."""
        return self.sign(query, k=k, vote=vote) < 0

    def distance(self, query):
        """Distance non signée au nuage (plus proche voisin)."""
        d, _ = self.tree.query(np.asarray(query, dtype=np.float64).reshape(-1, 3))
        return d

    def pseudo_sdf(self, query, k=1, vote="majorite"):
        """
        Distance signée approchée : signe déduit du nuage × distance au nuage.

        Ce n'est PAS la distance à la surface continue mais la distance au nuage
        échantillonné : elle la surestime d'autant que le nuage est lâche. Utile
        comme champ de référence grossier, pas comme cible d'apprentissage.
        """
        return self.sign(query, k=k, vote=vote) * self.distance(query)


def sign_from_cloud(query, points, normals, k=1, vote="majorite"):
    """Version fonctionnelle, quand on n'a pas besoin de garder l'arbre k-d."""
    return OrientedCloud(points, normals).sign(query, k=k, vote=vote)


def agreement(sign_pred, sign_ref):
    """Taux d'accord entre deux jeux de signes, en pourcentage."""
    a = np.sign(np.asarray(sign_pred, dtype=np.float64)).ravel()
    b = np.sign(np.asarray(sign_ref, dtype=np.float64)).ravel()
    if a.shape != b.shape:
        raise ValueError("tailles incompatibles")
    return 100.0 * float(np.mean(a == b))
