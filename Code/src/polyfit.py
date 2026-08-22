"""
Représentation POLYNOMIALE d'une surface implicite.

Piste recommandée par F. De Vuyst dans son mail : « la distance signée n'est pas
forcément la meilleure fonction […] une recherche d'une fonction implicite sous
une forme particulière, par exemple un polynôme, aurait beaucoup d'intérêts : à
la fois en terme de compression d'information, mais aussi en simplicité
d'évaluation ou en explicabilité. »

Méthode : ajustement algébrique classique, sans réseau et sans la formule.
On cherche P(x,y,z) = Σ c_α · x^i y^j z^k (i+j+k ≤ degré) tel que P ≈ 0 sur le
nuage de surface. En notant Φ la matrice des monômes évalués aux points du
nuage, cela revient à

    minimiser ‖Φ·c‖  sous  ‖c‖ = 1

dont la solution est le vecteur singulier droit associé à la plus petite valeur
singulière de Φ. Le signe global (indéterminé par la SVD) est fixé ensuite à
l'aide de quelques points dont on sait s'ils sont dedans ou dehors.

Si la surface EST algébrique de ce degré, le vrai polynôme appartient au noyau
de Φ : l'ajustement le retrouve exactement, et l'écart entre les deux plus
petites valeurs singulières le prouve.
"""
import numpy as np


def monomial_exponents(degree):
    """
    Exposants (i, j, k) de tous les monômes de degré total ≤ `degree`,
    ordonnés par degré croissant puis lexicographiquement.
    """
    exps = []
    for total in range(degree + 1):
        for i in range(total, -1, -1):
            for j in range(total - i, -1, -1):
                exps.append((i, j, total - i - j))
    return exps


def n_coeffs(degree):
    """Nombre de coefficients d'un polynôme de degré `degree` en 3 variables."""
    return (degree + 1) * (degree + 2) * (degree + 3) // 6


def poly_features(points, degree):
    """Matrice Φ des monômes : une ligne par point, une colonne par coefficient."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    cols = [x ** i * y ** j * z ** k for (i, j, k) in monomial_exponents(degree)]
    return np.column_stack(cols)


def poly_eval(points, coeffs, degree):
    """Évalue le polynôme sur un ensemble de points."""
    return poly_features(points, degree) @ np.asarray(coeffs, dtype=np.float64)


def fit_implicit_poly(surface_points, degree=4, sign_points=None,
                      sign_values=None, normalize=True):
    """
    Ajuste P(x,y,z) ≈ 0 sur le nuage de surface.

    surface_points : (N,3), les seuls points réellement nécessaires
    sign_points / sign_values : quelques points dedans/dehors (+1 dehors,
        −1 dedans) servant UNIQUEMENT à orienter le signe global
    normalize : ramène le nuage dans la boule unité avant l'ajustement, puis
        reconvertit les coefficients dans le repère d'origine. Sans cela, une
        forme à l'échelle physique (le globule fait 8 µm) donne des monômes de
        degré 4 de l'ordre de 4⁴ = 256 et une matrice mal conditionnée.

    Renvoie (coeffs, exponents, info) où `info` contient les deux plus petites
    valeurs singulières et leur rapport — le diagnostic qui dit si le noyau est
    bien de dimension 1, donc si la forme est réellement algébrique à ce degré.
    """
    pts = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    exps = monomial_exponents(degree)

    L = float(np.abs(pts).max()) if normalize else 1.0
    L = L if L > 0 else 1.0
    Phi = poly_features(pts / L, degree)

    # plus petit vecteur singulier droit : min ‖Φc‖ sous ‖c‖ = 1
    _, sv, Vt = np.linalg.svd(Phi, full_matrices=False)
    c = Vt[-1].copy()

    info = {
        "sigma_min": float(sv[-1]),
        "sigma_avant_dernier": float(sv[-2]) if len(sv) > 1 else float("nan"),
        "rapport": float(sv[-2] / sv[-1]) if len(sv) > 1 and sv[-1] > 0 else float("inf"),
        "scale": L,
    }

    # retour au repère d'origine : x = L·x' donne c_α = c'_α / L^(i+j+k)
    if normalize:
        powers = np.array([i + j + k for (i, j, k) in exps], dtype=np.float64)
        c = c / (L ** powers)
        c = c / np.linalg.norm(c)

    # orientation : on veut P < 0 à l'intérieur
    if sign_points is not None and sign_values is not None:
        vals = poly_eval(sign_points, c, degree)
        sv_ref = np.asarray(sign_values, dtype=np.float64).ravel()
        if float(np.mean(np.sign(vals) * sv_ref)) < 0:
            c = -c

    return c, exps, info


def sparsify(coeffs, tol=1e-6):
    """
    Met à zéro les coefficients négligeables devant le plus grand.

    Renvoie (coeffs_nettoyés, nombre de coefficients réellement non nuls) :
    c'est ce nombre qui mesure la compression, pas la taille du vecteur.
    """
    c = np.asarray(coeffs, dtype=np.float64).copy()
    seuil = tol * np.abs(c).max()
    c[np.abs(c) < seuil] = 0.0
    return c, int(np.count_nonzero(c))


def coefficient_of(coeffs, exponents, monomial):
    """Coefficient d'un monôme donné, ex. coefficient_of(c, exps, (2, 0, 0))."""
    return float(coeffs[exponents.index(tuple(monomial))])


def normalize_to_monomial(coeffs, exponents, monomial=(4, 0, 0)):
    """
    Renormalise le vecteur pour que le monôme de référence vaille 1.

    La SVD ne détermine les coefficients qu'à un facteur près : pour comparer
    l'ajustement aux vrais coefficients d'une quartique, il faut d'abord fixer
    ce facteur sur un monôme connu (par exemple x⁴, dont le vrai coefficient
    vaut 1 dans (x²+y²+z²)²).
    """
    ref = coefficient_of(coeffs, exponents, monomial)
    if abs(ref) < 1e-300:
        raise ValueError(f"coefficient de référence {monomial} nul : "
                         "impossible de fixer l'échelle")
    return np.asarray(coeffs, dtype=np.float64) / ref
