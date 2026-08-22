"""
Invariants géométriques et topologiques d'un maillage triangulaire.

Ce sont les contrôles qui disent si l'on peut CROIRE une surface reconstruite.
Une reconstruction peut avoir un excellent Chamfer (les sommets sont au bon
endroit) tout en ayant une topologie fausse ou des dérivées secondes aberrantes.

Trois niveaux de contrôle :
  - topologique  : caractéristique d'Euler χ = V − E + F, étanchéité ;
  - métrique     : aire et volume, comparables aux valeurs analytiques ;
  - différentiel : Gauss-Bonnet, ∫K_G dA = 2πχ — le plus sévère, car il fait
                   intervenir à la fois la topologie ET la courbure.

PIÈGE PRINCIPAL — le comptage des arêtes
----------------------------------------
Un maillage fermé de F triangles a E = 3F/2 arêtes, pas 3F : chaque arête est
partagée par exactement deux triangles. Compter 3F donne χ = V − 2F, soit une
valeur absurde. Il faut donc dédupliquer les arêtes après les avoir orientées
de façon canonique (indices triés).

Second piège : le marching cubes peut produire des sommets DUPLIQUÉS à la même
position (lorsque l'isosurface passe très près d'un nœud de la grille). V est
alors gonflé et χ est faux. On soude donc les sommets coïncidents avant tout
comptage — c'est ce que fait `weld_vertices`.
"""
import numpy as np


# ------------------------------------------------------------------
# Préparation du maillage
# ------------------------------------------------------------------

def weld_vertices(verts, faces, decimals=9):
    """
    Fusionne les sommets géométriquement confondus et supprime les triangles
    devenus dégénérés.

    `decimals` : nombre de décimales utilisé pour reconnaître deux sommets
    comme identiques. 9 est très serré (1e-9) : on ne fusionne que de vrais
    doublons, jamais deux sommets distincts.

    Renvoie (verts, faces) — le maillage est géométriquement inchangé, seule
    la connectivité est nettoyée.
    """
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if len(F) == 0:
        return V, F

    key = np.round(V, decimals)
    # +0.0 évite que -0.0 et +0.0 soient vus comme deux clés différentes
    key = key + 0.0
    _, first, inverse = np.unique(key, axis=0, return_index=True,
                                  return_inverse=True)
    inverse = np.asarray(inverse).ravel()

    new_verts = V[first]
    # `first` est trié par ordre de clé ; inverse pointe déjà vers ces indices
    new_faces = inverse[F]

    ok = ((new_faces[:, 0] != new_faces[:, 1])
          & (new_faces[:, 1] != new_faces[:, 2])
          & (new_faces[:, 0] != new_faces[:, 2]))
    return new_verts, new_faces[ok]


def _unique_edges(faces):
    """Arêtes uniques (non orientées) d'un maillage triangulaire, et leur multiplicité."""
    F = np.asarray(faces, dtype=np.int64)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)                     # orientation canonique
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq, counts


# ------------------------------------------------------------------
# Topologie
# ------------------------------------------------------------------

def euler_characteristic(verts, faces, weld=True, decimals=9):
    """
    χ = V − E + F, avec E = nombre d'arêtes UNIQUES.

    Pour une surface fermée orientable : χ = 2 − 2g, donc
      sphère (genre 0) → χ = 2
      tore   (genre 1) → χ = 0

    Renvoie {"V","E","F","chi","genus"}.
    """
    if weld:
        verts, faces = weld_vertices(verts, faces, decimals)
    V = int(len(verts))
    F = int(len(faces))
    uniq, _ = _unique_edges(faces)
    E = int(len(uniq))
    chi = V - E + F
    return {"V": V, "E": E, "F": F, "chi": int(chi), "genus": (2 - chi) / 2.0}


def is_watertight(verts, faces, weld=True, decimals=9):
    """Vrai si chaque arête appartient à exactement deux triangles."""
    if weld:
        verts, faces = weld_vertices(verts, faces, decimals)
    if len(faces) == 0:
        return False
    _, counts = _unique_edges(faces)
    return bool(np.all(counts == 2))


# ------------------------------------------------------------------
# Métrique
# ------------------------------------------------------------------

def triangle_areas(verts, faces):
    """Aire de chaque triangle : ½‖(v2−v1) × (v3−v1)‖."""
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def mesh_area(verts, faces):
    """Aire totale de la surface."""
    return float(triangle_areas(verts, faces).sum())


def mesh_volume(verts, faces):
    """
    Volume enclos, par le théorème de la divergence :
        V = (1/6) Σ v1 · (v2 × v3)
    Renvoie la valeur absolue, car l'orientation produite par le marching
    cubes n'est pas garantie.
    """
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0))


# ------------------------------------------------------------------
# Gauss-Bonnet
# ------------------------------------------------------------------

def gauss_bonnet(verts, faces, K_gauss_per_vertex, weld=True, decimals=9):
    """
    Vérifie ∫K_G dA = 2πχ sur le maillage.

    L'intégrale est approchée triangle par triangle : la courbure du triangle
    est la moyenne de celle de ses trois sommets, pondérée par son aire
    (méthode de Kuchel et al. 2021, éq. 19-20).

        ∫K_G dA ≈ Σ_T aire(T) × moyenne(K_G aux sommets de T)

    C'est le test le plus sévère du pipeline : il ne passe que si le comptage
    d'arêtes ET le calcul de courbure sont corrects tous les deux. Sur un tore
    la valeur attendue est ZÉRO, donc les contributions positives (extérieur du
    tore) doivent annuler exactement les négatives (intérieur) — aucune erreur
    systématique ne peut s'y cacher.

    Les aires ne dépendent pas de la soudure des sommets ; seul χ en dépend.
    `K_gauss_per_vertex` doit être indexé comme `verts`.

    Renvoie {"integral","expected","chi","absolute_error","relative_error"}.
    """
    K = np.asarray(K_gauss_per_vertex, dtype=np.float64).ravel()
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if len(K) != len(V):
        raise ValueError(f"K_gauss a {len(K)} valeurs pour {len(V)} sommets")

    areas = triangle_areas(V, F)
    K_tri = K[F].mean(axis=1)
    integral = float(np.sum(K_tri * areas))

    topo = euler_characteristic(V, F, weld=weld, decimals=decimals)
    expected = 2.0 * np.pi * topo["chi"]
    abs_err = abs(integral - expected)
    # L'erreur relative n'a pas de sens quand l'attendu est nul (tore) :
    # on la laisse à None et c'est l'erreur absolue qui fait foi.
    rel_err = (abs_err / abs(expected)) if abs(expected) > 1e-12 else None

    return {"integral": integral, "expected": float(expected),
            "chi": topo["chi"], "absolute_error": abs_err,
            "relative_error": rel_err}


def report(verts, faces, K_gauss_per_vertex=None, name=""):
    """Bloc de contrôle lisible pour la console."""
    topo = euler_characteristic(verts, faces)
    lines = [
        f"{name}",
        f"  V={topo['V']}  E={topo['E']}  F={topo['F']}  "
        f"chi={topo['chi']}  genre={topo['genus']:.1f}",
        f"  étanche={is_watertight(verts, faces)}  "
        f"aire={mesh_area(verts, faces):.6f}  volume={mesh_volume(verts, faces):.6f}",
    ]
    if K_gauss_per_vertex is not None:
        gb = gauss_bonnet(verts, faces, K_gauss_per_vertex)
        rel = "n/a" if gb["relative_error"] is None else f"{100*gb['relative_error']:.3f}%"
        lines.append(f"  Gauss-Bonnet : ∫K dA={gb['integral']:+.6f}  "
                     f"attendu 2πχ={gb['expected']:+.6f}  "
                     f"écart abs={gb['absolute_error']:.6f}  rel={rel}")
    return "\n".join(lines)
