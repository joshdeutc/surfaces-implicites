"""
Les vraies données du BMBI : capsule déformable en écoulement dans un canal.

Ce module est la porte d'entrée du dataset fourni par F. De Vuyst (26/08/2026).
Il ne fait qu'une chose : transformer une simulation Éléments Finis en fichiers
au format d'échange du projet (`src/cloud_io.py`), pour que TOUTE la chaîne déjà
écrite — signe dedans/dehors, entraînement eikonal, reconstruction, métriques,
champ espace-temps — s'applique sans être modifiée.

CE QUE CONTIENT LE DATASET
--------------------------
    connect.npy               (5120, 3)          triangles, indices de noeuds
    off_centered_capsule.npy  (20001, 2562, 3)   x,y,z de chaque noeud, par pas

Le maillage est une icosphère de subdivision 4 : V=2562, F=5120, donc
E = 3F/2 = 7680 et chi = V - E + F = 2. Genre 0, fermée, manifold, et
l'orientation des triangles est déjà cohérente ET sortante (le volume signé du
maillage est positif au premier pas). Les 2562 noeuds sont des points MATÉRIELS
suivis dans le temps : la connectivité ne change jamais.

DEUX PIÈGES, TRAITÉS ICI
------------------------
1. **La capsule n'est pas centrée, et elle migre.** Le nom du fichier le dit
   (`off_centered`) : elle part à y = 0.4 de l'axe du canal et migre vers l'axe
   (|c| passe de 0.400 à 0.046). C'est de la physique, pas un défaut de données —
   mais un champ f(x,y,z,t) qui doit encaisser à la fois la translation et la
   déformation apprend deux choses d'un coup. `recentre=True` sépare les deux :
   la forme d'un côté, la trajectoire du centre de l'autre (elle est rendue à
   part, et c'est une fonction du temps à trois composantes, triviale à ajuster).

2. **Le mouvement est à 92 % TANGENTIEL** (tank-treading : la membrane tourne
   autour d'une forme qui, elle, bouge peu). Conséquence directe pour nous : on
   ne peut PAS valider un suivi lagrangien en advectant un point par la vitesse
   normale et en le comparant à la trajectoire d'un noeud du maillage. Une
   surface implicite ne connaît que le mouvement NORMAL, et c'est correct : le
   glissement tangentiel ne déplace pas la surface. Le contrôle à faire est donc
   sur v_n, jamais sur la trajectoire matérielle.

CONFIDENTIALITÉ
---------------
Le dataset est propriété du laboratoire BMBI. Il vit dans `Données/`, qui est
exclu de git. Ce module lit le dossier, il n'en recopie jamais le contenu
ailleurs que dans `Code/output/`, lui aussi hors git.
"""
import os

import numpy as np

from src.cloud_io import (mesh_signed_volume, orient_outward, save_cloud,
                          vertex_normals)

# Chemin par défaut : Données/Data_Capsule/ à la racine du dépôt
RACINE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "Données", "Data_Capsule")

N_PAS = 20001          # pas de temps dans le fichier
N_NOEUDS = 2562
N_TRIANGLES = 5120


def charger(racine=RACINE):
    """
    Ouvre le dataset. Le tableau des coordonnées fait 1.2 Go : il est ouvert en
    `mmap_mode='r'`, donc jamais chargé en entier — on ne paie que les instants
    réellement lus.

    Renvoie (faces, caps) avec caps de forme (20001, 2562, 3).
    """
    fc = os.path.join(racine, "connect.npy")
    fx = os.path.join(racine, "off_centered_capsule.npy")
    if not os.path.exists(fx):
        raise FileNotFoundError(
            f"dataset BMBI absent de {racine}.\n"
            "Il n'est pas versionné (confidentiel, 1.2 Go) : le récupérer via le "
            "lien du mail du 26/08/2026, cf. Encadrement/.")
    faces = np.load(fc)
    caps = np.load(fx, mmap_mode="r")
    if faces.shape != (N_TRIANGLES, 3) or caps.shape != (N_PAS, N_NOEUDS, 3):
        raise ValueError(f"formes inattendues : {faces.shape}, {caps.shape}")
    # L'orientation est déjà sortante dans le fichier, mais on ne le suppose pas.
    faces = orient_outward(np.asarray(caps[0], dtype=np.float64), faces)[0]
    return faces, caps


def instant(caps, t, recentre=True):
    """
    Les noeuds à l'instant `t`. Si `recentre`, le centroïde est ramené en 0 et
    renvoyé à part : (noeuds, centre).
    """
    P = np.asarray(caps[t], dtype=np.float64)
    c = P.mean(axis=0)
    return (P - c, c) if recentre else (P, np.zeros(3))


def instants_reguliers(n_snap=101, t_max=N_PAS - 1):
    """
    Les indices de `n_snap` instants équirépartis de 0 à `t_max`.

    F. De Vuyst recommande « par exemple 100 snapshots » sur les 20001 pas. À ce
    pas, le déplacement matériel entre deux snapshots vaut environ une demi-arête
    de maillage au début de la séquence — c'est fin. Mais attention à ce que
    cette régularité coûte : la déformation culmine vers t = 800 et tout le reste
    est une relaxation lente, si bien qu'un échantillonnage uniforme place 96 %
    de ses instants dans la partie molle. Voir `instants_transitoire`.
    """
    return np.unique(np.linspace(0, t_max, n_snap).astype(int))


def instants_transitoire(n_snap=101, t_max=N_PAS - 1, puissance=2.0):
    """
    Instants resserrés au DÉBUT, là où la capsule se déforme vraiment.

    La sphère initiale s'aplatit jusqu'à un rapport d'axes de 1.354 atteint vers
    t = 800, puis relaxe lentement vers 1.045 en 20000 pas. Un échantillonnage en
    t = t_max·(k/n)^puissance suit ce rythme au lieu de le lisser.
    """
    u = (np.arange(n_snap) / (n_snap - 1.0)) ** puissance
    return np.unique((u * t_max).astype(int))


def nuage(faces, caps, t, recentre=True, n_points=None, seed=0):
    """
    Un instant, au format « nuage orienté » du projet : points de surface et
    normales sortantes, calculées PAR TRIANGLE puis moyennées aux sommets — la
    méthode du mail de F. De Vuyst, celle qui ne suppose aucune formule.

    `n_points=None` garde les 2562 noeuds du maillage. Une valeur plus petite
    tire un sous-ensemble (pour éprouver la robustesse au nombre de points).
    """
    verts, centre = instant(caps, t, recentre=recentre)
    normals = vertex_normals(verts, faces)
    if n_points is not None and n_points < len(verts):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(verts), size=n_points, replace=False)
        verts, normals = verts[sel], normals[sel]
    return verts, normals, centre


def geometrie(faces, verts):
    """Volume et aire du maillage fermé, sans aucune formule implicite."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    aire = 0.5 * np.linalg.norm(cr, axis=1).sum()
    return mesh_signed_volume(verts, faces), aire


def demi_axes(verts):
    """
    Demi-axes équivalents (racines des valeurs propres du tenseur des noeuds) et
    rapport d'aplatissement grand axe / petit axe. Sert à décrire la déformation
    en un seul nombre, indépendant de toute reconstruction.
    """
    P = verts - verts.mean(axis=0)
    w = np.sqrt(np.linalg.eigvalsh(P.T @ P / len(P)))[::-1]
    return w, float(w[0] / w[2])


def part_non_affine(verts_a, verts_b):
    """
    Quelle FRACTION du mouvement de `verts_a` vers `verts_b` échappe à la
    meilleure application affine ?

    C'est LE chiffre qui décide si notre étalon « morphing entre deux instants
    voisins » est un adversaire sérieux. Sur la séquence synthétique de
    `src/temporal.py` il vaut 0 % par construction — la déformation y était
    affine, ce qui rendait le test d'interpolation non discriminant. Sur les
    vraies capsules il vaut 14 à 17 % entre snapshots voisins.

    Renvoie (résidu moyen, déplacement moyen, fraction).
    """
    a = verts_a - verts_a.mean(axis=0)
    b = verts_b - verts_b.mean(axis=0)
    A, *_ = np.linalg.lstsq(a, b, rcond=None)
    residu = float(np.linalg.norm(b - a @ A, axis=1).mean())
    depl = float(np.linalg.norm(b - a, axis=1).mean())
    return residu, depl, residu / max(depl, 1e-30)


def boite(faces, caps, indices, recentre=True, marge=1.25):
    """
    Demi-taille de boîte commune à tous les instants demandés.

    Une boîte par instant serait une erreur : le champ f(x,y,z,t) est appris sur
    UN domaine, et le terme eikonal y est tiré uniformément. La boîte doit donc
    contenir toute la séquence, avec de la marge pour que l'eikonal ait de la
    place autour de la surface.
    """
    r = max(np.abs(instant(caps, int(t), recentre=recentre)[0]).max()
            for t in indices)
    return float(marge * r)


def exporter_sequence(dossier, faces, caps, indices, recentre=True,
                      n_points=None, box=None, seed=0):
    """
    Écrit un `.npz` par instant, au format d'échange du projet, plus un
    `sequence.npz` qui garde les temps et la trajectoire du centre.

    C'est le point de bascule du projet : après cet appel, plus rien en aval ne
    sait que les données viennent d'un calcul Éléments Finis plutôt que d'une
    formule. La suite de la chaîne est déjà écrite.
    """
    os.makedirs(dossier, exist_ok=True)
    indices = [int(t) for t in indices]
    if box is None:
        box = boite(faces, caps, indices, recentre=recentre)

    centres, fichiers = [], []
    for k, t in enumerate(indices):
        P, Nn, c = nuage(faces, caps, t, recentre=recentre,
                         n_points=n_points, seed=seed + k)
        chemin = os.path.join(dossier, f"capsule_{k:04d}.npz")
        save_cloud(chemin, P, Nn, box, source=f"BMBI capsule, pas EF t={t}")
        centres.append(c)
        fichiers.append(chemin)

    np.savez(os.path.join(dossier, "sequence.npz"),
             pas=np.array(indices, dtype=np.int64),
             temps=np.linspace(0.0, 1.0, len(indices)),
             centres=np.asarray(centres, dtype=np.float64),
             box=np.float64(box))
    return fichiers, box


# ------------------------------------------------------------------
# Champ de référence : la « vérité » dont on dispose, tirée du maillage seul
# ------------------------------------------------------------------

def echantillon_surface(verts, faces, n=200000, seed=0):
    """
    Nuage dense tiré uniformément sur les triangles, avec la normale du triangle
    d'origine. Sert de référence pour le Chamfer et de support au champ signé.

    Le tirage est pondéré par l'aire des triangles, sans quoi les petits
    triangles seraient sur-représentés.
    """
    rng = np.random.default_rng(seed)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    aires = 0.5 * np.linalg.norm(cr, axis=1)
    k = rng.choice(len(faces), size=n, p=aires / aires.sum())
    u, v = rng.random(n), rng.random(n)
    hors = u + v > 1.0
    u[hors], v[hors] = 1.0 - u[hors], 1.0 - v[hors]
    P = v0[k] + u[:, None] * (v1[k] - v0[k]) + v[:, None] * (v2[k] - v0[k])
    Nn = cr[k] / np.maximum(np.linalg.norm(cr[k], axis=1, keepdims=True), 1e-300)
    return P, Nn


def champ_reference(verts, faces, box, grille=96, n_echantillons=60000, seed=0):
    """
    Distance signée approchée sur une grille, construite à partir du MAILLAGE
    SEUL — aucune formule, donc utilisable sur des données réelles.

    Distance : plus proche voisin dans un nuage dense tiré sur les triangles.
    Signe : produit scalaire avec la normale du plus proche point, exactement la
    règle de `src/inside_outside.py` et du mail de F. De Vuyst.

    Sert à trois choses, toutes indispensables pour juger un champ appris sans
    disposer de la vérité analytique :
      - le PLANCHER du Chamfer (marching cubes sur ce champ : aucune méthode ne
        peut faire mieux à cette grille) ;
      - l'occupation de référence, donc l'IoU et le volume ;
      - les étalons (image gelée, morphing), qui s'interpolent sur la grille.

    Renvoie (axe, valeurs (grille,grille,grille)).
    """
    from scipy.spatial import cKDTree

    S, Ns = echantillon_surface(verts, faces, n=n_echantillons, seed=seed)
    ax = np.linspace(-box, box, grille)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    d, i = cKDTree(S).query(P, k=1, workers=-1)
    signe = np.sign(np.einsum("ij,ij->i", P - S[i], Ns[i]))
    signe[signe == 0] = 1.0
    return ax, (signe * d).reshape(grille, grille, grille)


def vitesse_normale_maillage(caps, faces, pas, dpas, dt_par_pas, recentre=True):
    """
    Vitesse NORMALE de la membrane, lue sur le maillage par différence centrée.

    C'est le seul contrôle de vitesse valable sur ces données. Le mouvement des
    nœuds est à 93 % TANGENTIEL (tank-treading) : comparer la trajectoire d'un
    nœud à un point advecté par le champ appris n'a pas de sens, les deux ne
    décrivent pas la même chose. En revanche la projection de la vitesse d'un
    nœud sur la normale, elle, est bien la vitesse de la SURFACE — et c'est
    exactement ce que porte un champ implicite.

    `dt_par_pas` convertit un pas EF en unité du temps d'entraînement.

    Renvoie (points, v_n) à l'instant `pas`.
    """
    if pas - dpas < 0 or pas + dpas > len(caps) - 1:
        raise ValueError(
            f"différence centrée impossible : pas={pas}, dpas={dpas} sort du "
            f"tableau [0, {len(caps) - 1}]. Un indice négatif boucle "
            "silencieusement en fin de séquence, d'où ce garde-fou.")
    a = instant(caps, pas - dpas, recentre=recentre)[0]
    b = instant(caps, pas + dpas, recentre=recentre)[0]
    P = instant(caps, pas, recentre=recentre)[0]
    vitesse = (b - a) / (2.0 * dpas * dt_par_pas)
    n = vertex_normals(P, faces)
    return P, np.einsum("ij,ij->i", vitesse, n)
