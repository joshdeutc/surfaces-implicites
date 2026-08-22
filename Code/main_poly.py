"""
Représentation polynomiale d'une surface implicite — la piste « compression et
explicabilité » du mail de F. De Vuyst.

On ajuste P(x,y,z) = 0 sur le SEUL nuage de points de surface (SVD sur la
matrice des monômes), sans réseau et sans utiliser la valeur de f. La formule
exacte n'intervient qu'à deux endroits, tous deux d'évaluation :
  - le SIGNE dedans/dehors de quelques points, pour orienter le polynôme
    (c'est l'information qu'un vrai nuage fournit via ses normales) ;
  - la mesure finale de qualité (IoU, Chamfer).

Lancer :  python main_poly.py
"""
import numpy as np

from src.shapes import (get_surface, rbc_coeffs_scaled, sample_ellipsoid_surface,
                        sample_rbc_unit_surface, sample_biconcave_surface)
from src.polyfit import (fit_implicit_poly, poly_eval, n_coeffs, sparsify,
                         monomial_exponents, normalize_to_monomial)
from src.metrics import chamfer_distance, iou_occupancy
from src.model import IGRNet


def dense_reference(shape):
    """Nuage dense sur la vraie surface, pour le Chamfer (évaluation seule)."""
    if shape == "ellipsoid":
        return sample_ellipsoid_surface(n=50000, a=1.2, b=0.8, c=0.6)
    if shape == "rbc_unit":
        return sample_rbc_unit_surface(n=50000, seed=1)
    if shape == "biconcave":
        return sample_biconcave_surface(n=50000)
    raise ValueError(shape)


def fit_one(shape, degree, n_surf=6000, seed=0):
    """Ajuste un polynôme de degré donné et mesure la reconstruction."""
    rng = np.random.default_rng(seed)
    surface_points, implicit_exact, box = get_surface(shape)

    # points de signe : uniquement le SIGNE, jamais la valeur
    X_sign = rng.uniform(-box, box, size=(8000, 3))
    s_sign = np.sign(implicit_exact(X_sign))

    coeffs, exps, info = fit_implicit_poly(
        surface_points, degree=degree,
        sign_points=X_sign, sign_values=s_sign, normalize=True,
    )

    inside_poly = lambda p: poly_eval(p, coeffs, degree) < 0.0
    inside_ref = lambda p: implicit_exact(np.asarray(p, dtype=np.float64)) < 0.0
    iou = iou_occupancy(inside_poly, inside_ref, box, n_samples=100000, seed=7)

    ref = dense_reference(shape)
    resid = np.abs(poly_eval(ref, coeffs, degree)).mean()

    _, nnz = sparsify(coeffs, tol=1e-6)
    return {
        "shape": shape, "degree": degree,
        "n_coeffs": n_coeffs(degree), "n_nonzero": nnz,
        "iou": iou, "residu_surface": resid,
        "sigma_ratio": info["rapport"],
        "coeffs": coeffs, "exps": exps,
    }


def nom_monome(mono):
    """Écriture lisible d'un monôme : (2,0,2) -> 'x^2 z^2', (0,0,0) -> '1'."""
    i, j, k = mono
    if i == j == k == 0:
        return "1"
    parts = []
    for lettre, e in zip("xyz", (i, j, k)):
        if e == 1:
            parts.append(lettre)
        elif e > 1:
            parts.append(f"{lettre}^{e}")
    return " ".join(parts)


def compare_to_true_quartic(res):
    """
    Les coefficients ajustés retrouvent-ils la vraie quartique de Kuchel ?

    Le vrai polynôme normalisé s'écrit
        (x²+y²+z²)² + P'(x²+y²) + Q'z² + R'
      = x⁴+y⁴+z⁴ + 2x²y² + 2x²z² + 2y²z² + P'x² + P'y² + Q'z² + R'
    donc tous les autres monômes de degré ≤ 4 doivent être nuls.
    La SVD ne donne les coefficients qu'à un facteur près : on fixe l'échelle
    sur x⁴, dont le vrai coefficient vaut 1.
    """
    Pp, Qp, Rp = rbc_coeffs_scaled()
    c = normalize_to_monomial(res["coeffs"], res["exps"], (4, 0, 0))
    exps = res["exps"]
    attendu = {(4, 0, 0): 1.0, (0, 4, 0): 1.0, (0, 0, 4): 1.0,
               (2, 2, 0): 2.0, (2, 0, 2): 2.0, (0, 2, 2): 2.0,
               (2, 0, 0): Pp, (0, 2, 0): Pp, (0, 0, 2): Qp,
               (0, 0, 0): Rp}

    print("\n  Coefficients ajustes vs vrais (echelle fixee sur x^4) :")
    print(f"    {'monome':>10}   {'ajuste':>12}   {'vrai':>12}   {'ecart':>10}")
    ecart_max = 0.0
    for mono, vrai in attendu.items():
        obtenu = float(c[exps.index(mono)])
        ecart = abs(obtenu - vrai)
        ecart_max = max(ecart_max, ecart)
        print(f"    {nom_monome(mono):>10}   {obtenu:12.6f}   {vrai:12.6f}   {ecart:10.2e}")

    autres = [abs(float(c[i])) for i, e in enumerate(exps) if e not in attendu]
    print(f"    ecart max sur les 10 coefficients non nuls : {ecart_max:.2e}")
    print(f"    plus grand des {len(autres)} coefficients cense etre nul : {max(autres):.2e}")
    return ecart_max, max(autres)


def main():
    print("=" * 74)
    print("REPRESENTATION POLYNOMIALE — ajustement sur le seul nuage de surface")
    print("=" * 74)

    igr_params = sum(p.numel() for p in IGRNet(radius_init=1.0).parameters())
    print(f"\nReference reseau : IGRNet = {igr_params} poids\n")

    essais = [
        ("ellipsoid", 2),   # quadrique exacte : doit tomber juste
        ("rbc_unit", 4),    # quartique exacte : le test central
        ("rbc_unit", 2),    # degre insuffisant : doit echouer
        ("biconcave", 4),   # profil d'Evans-Fung : NON algebrique
        ("biconcave", 6),   #   -> il faut monter en degre pour l'approcher
    ]
    res = [fit_one(s, d) for s, d in essais]

    print(f"{'forme':<12} {'deg':>4} {'coeffs':>7} {'non nuls':>9} {'IoU':>8} "
          f"{'residu':>10} {'sigma_-2/sigma_-1':>18} {'compression':>12}")
    print("-" * 92)
    for r in res:
        comp = igr_params / r["n_nonzero"] if r["n_nonzero"] else float("nan")
        print(f"{r['shape']:<12} {r['degree']:>4} {r['n_coeffs']:>7} {r['n_nonzero']:>9} "
              f"{r['iou']:>8.4f} {r['residu_surface']:>10.2e} {r['sigma_ratio']:>18.1f} "
              f"{comp:>11.0f}x")

    print("\n" + "=" * 74)
    print("LE GLOBULE ROUGE EST-IL RETROUVE EXACTEMENT ?")
    print("=" * 74)
    ecart_max, residu_nul = compare_to_true_quartic(res[1])

    ell, rbc4 = res[0], res[1]
    print("\n" + "=" * 74)
    print("CRITERES")
    print("=" * 74)
    c1 = ell["n_coeffs"] == 10 and ell["iou"] >= 0.99
    c2 = rbc4["iou"] >= 0.99
    print(f"  ellipsoide degre 2 -> {ell['n_coeffs']} coefficients et IoU = {ell['iou']:.4f}"
          f"   [{'OK' if c1 else 'ECHEC'}]  (attendu 10 coefficients, IoU >= 0.99)")
    print(f"  globule quartique degre 4 -> IoU = {rbc4['iou']:.4f}"
          f"   [{'OK' if c2 else 'ECHEC'}]  (attendu IoU >= 0.99)")
    print(f"\n  Compression : le globule rouge tient en {rbc4['n_nonzero']} coefficients "
          f"non nuls contre {igr_params} poids pour IGRNet")
    print(f"  soit un facteur {igr_params / rbc4['n_nonzero']:.0f}. Les trois coefficients "
          f"P, Q, R suffisent a decrire la forme,")
    print(f"  le reste de la quartique etant fixe par sa structure (x^2+y^2+z^2)^2.")
    print("\n" + ("TOUS LES CRITERES SONT RESPECTES" if (c1 and c2)
                  else "AU MOINS UN CRITERE N'EST PAS RESPECTE"))
    print("=" * 74)


if __name__ == "__main__":
    main()
