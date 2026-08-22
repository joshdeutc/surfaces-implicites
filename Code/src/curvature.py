"""
Courbures d'une surface implicite f(x,y,z) = 0.

Formules de Goldman (2005), reprises par Kuchel et al., Sci. Rep. 11:13712 (2021),
équations 14 à 17. Elles ne font intervenir que le gradient et la hessienne de f,
donc elles s'appliquent AUSSI BIEN à une formule analytique qu'à un réseau de
neurones — dans ce dernier cas la hessienne vient de la différentiation
automatique d'ordre 2.

    K_gauss   = (∇f · H* · ∇fᵀ) / ‖∇f‖⁴        H* = comatrice de la hessienne
    K_moyenne = (∇f · H · ∇fᵀ − ‖∇f‖²·Tr H) / (2‖∇f‖³)
    k1, k2    = K_M ± √(K_M² − K_G)

CONVENTION DE SIGNE
-------------------
Avec la convention du projet (f < 0 dedans, f > 0 dehors, donc ∇f pointant vers
l'extérieur), ces formules donnent pour une SPHÈRE de rayon R :

    K_gauss   = +1/R²        (toujours positive, indépendante de l'orientation)
    K_moyenne = −1/R         (NÉGATIVE)
    k1 = k2   = −1/R

La courbure moyenne est donc négative pour une surface convexe fermée orientée
vers l'extérieur. C'est une conséquence directe de la formule : le terme
dominant est −‖∇f‖²·Tr H, et la trace de la hessienne est positive pour une
sphère. Inverser le signe de f inverse celui de K_moyenne et des courbures
principales, mais laisse K_gauss inchangée.

On compare donc toujours |K_moyenne| aux valeurs de référence de la littérature,
qui adoptent souvent la convention inverse.

La démonstration sur la sphère f = ‖x‖ − R :
    ∇f = x/r          → ‖∇f‖ = 1
    H   = (I − x̂x̂ᵀ)/r → valeurs propres 0, 1/r, 1/r  et  Tr H = 2/r
    ∇f·H·∇fᵀ = 0                     → K_M = (0 − 1·2/r)/2 = −1/r
    ∇f·H*·∇fᵀ = 1/r²                 → K_G = 1/r²
"""
import numpy as np
import torch


# ------------------------------------------------------------------
# Cœur du calcul : purement arithmétique, donc valable à l'identique
# sur des tableaux numpy et sur des tenseurs torch.
# ------------------------------------------------------------------

def curvatures_from_derivatives(g0, g1, g2,
                                h00, h01, h02, h11, h12, h22,
                                eps=1e-12):
    """
    Courbures à partir des composantes du gradient et de la hessienne (supposée
    symétrique, ce qui est le cas pour toute fonction deux fois différentiable).

    Chaque argument est un tableau (N,) — numpy ou torch, les deux marchent :
    seules des opérations élémentaires sont utilisées.

    Renvoie (K_gauss, K_moyenne, k1, k2), où k1 >= k2.
    """
    norm2 = g0 * g0 + g1 * g1 + g2 * g2
    norm = norm2 ** 0.5

    # Comatrice de la hessienne. Pour une matrice symétrique elle est elle-même
    # symétrique, et égale à l'adjointe (transposée de la comatrice).
    a00 = h11 * h22 - h12 * h12
    a01 = h02 * h12 - h01 * h22
    a02 = h01 * h12 - h11 * h02
    a11 = h00 * h22 - h02 * h02
    a12 = h01 * h02 - h00 * h12
    a22 = h00 * h11 - h01 * h01

    # Formes quadratiques ∇f · M · ∇fᵀ, écrites terme à terme.
    q_adj = (a00 * g0 * g0 + a11 * g1 * g1 + a22 * g2 * g2
             + 2.0 * (a01 * g0 * g1 + a02 * g0 * g2 + a12 * g1 * g2))
    q_hess = (h00 * g0 * g0 + h11 * g1 * g1 + h22 * g2 * g2
              + 2.0 * (h01 * g0 * g1 + h02 * g0 * g2 + h12 * g1 * g2))
    trace = h00 + h11 + h22

    K_gauss = q_adj / (norm2 * norm2 + eps)
    K_mean = (q_hess - norm2 * trace) / (2.0 * norm2 * norm + eps)

    # Discriminant : mathématiquement K_M² − K_G >= 0 (c'est ((k1−k2)/2)²),
    # mais il peut devenir légèrement négatif par erreur numérique près des
    # points ombilicaux — la sphère en est un cas limite exact. On le ramène
    # à zéro avec max(d, 0) écrit sans fonction de bibliothèque, pour rester
    # compatible numpy et torch.
    disc = K_mean * K_mean - K_gauss
    disc = (disc + abs(disc)) * 0.5
    root = disc ** 0.5

    k1 = K_mean + root
    k2 = K_mean - root
    return K_gauss, K_mean, k1, k2


def principal_from_KG_KM(K_gauss, K_mean):
    """Courbures principales à partir de K_gauss et K_moyenne (mêmes conventions)."""
    disc = K_mean * K_mean - K_gauss
    disc = (disc + abs(disc)) * 0.5
    root = disc ** 0.5
    return K_mean + root, K_mean - root


# ------------------------------------------------------------------
# Version numpy : différences finies centrées
# ------------------------------------------------------------------

def curvature_numpy(f, points, h=1e-4):
    """
    Courbures d'une fonction implicite numpy quelconque, par différences finies.

    f      : callable (N,3) -> (N,)
    points : (N,3) — en pratique des points de la surface
    h      : pas des différences finies. 1e-4 équilibre l'erreur de troncature
             (~h²) et l'erreur d'arrondi (~eps/h²) en double précision.

    Renvoie un dict {"K_gauss", "K_mean", "k1", "k2"}, chaque entrée en (N,).
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points doit être de forme (N,3), reçu {P.shape}")

    def ev(offset):
        return np.asarray(f(P + offset), dtype=np.float64).ravel()

    e = np.eye(3) * h
    f0 = ev(0.0)
    fp = [ev(e[i]) for i in range(3)]
    fm = [ev(-e[i]) for i in range(3)]

    g = [(fp[i] - fm[i]) / (2.0 * h) for i in range(3)]
    hdiag = [(fp[i] - 2.0 * f0 + fm[i]) / (h * h) for i in range(3)]

    def mixed(i, j):
        return (ev(e[i] + e[j]) - ev(e[i] - e[j])
                - ev(-e[i] + e[j]) + ev(-e[i] - e[j])) / (4.0 * h * h)

    h01, h02, h12 = mixed(0, 1), mixed(0, 2), mixed(1, 2)

    K_gauss, K_mean, k1, k2 = curvatures_from_derivatives(
        g[0], g[1], g[2], hdiag[0], h01, h02, hdiag[1], h12, hdiag[2]
    )
    return {"K_gauss": K_gauss, "K_mean": K_mean, "k1": k1, "k2": k2}


# ------------------------------------------------------------------
# Version torch : différentiation automatique d'ordre 2
# ------------------------------------------------------------------

def _hessian_autograd(model, x):
    """
    Gradient et hessienne de `model` en x, par autograd d'ordre 2.

    x : (B,3) avec requires_grad. Trois passes arrière suffisent : une par
    ligne de la hessienne. Les points étant indépendants (le réseau est évalué
    point par point), sommer avant de dériver donne bien le gradient par point.
    """
    f = model(x)
    if f.dim() > 1:
        f = f.reshape(f.shape[0], -1)[:, 0]

    g = torch.autograd.grad(f.sum(), x, create_graph=True)[0]        # (B,3)

    rows = []
    for i in range(3):
        last = (i == 2)
        row = torch.autograd.grad(g[:, i].sum(), x,
                                  retain_graph=not last,
                                  create_graph=False)[0]            # (B,3)
        rows.append(row)
    H = torch.stack(rows, dim=1)                                    # (B,3,3)
    return g.detach(), H.detach()


def curvature_autograd(model, points, device=None, chunk=4096, verbose=True):
    """
    Courbures du champ appris par un RÉSEAU, par différentiation automatique.

    C'est le cœur de l'analyse : le réseau n'a jamais vu de courbure pendant
    l'entraînement, on lui demande ici ses dérivées SECONDES. Le Chamfer teste
    l'ordre 0, l'eikonal l'ordre 1 ; ceci teste l'ordre 2.

    model  : nn.Module (N,3) -> (N,1)
    points : (N,3)
    device : forcé si fourni. Sinon on tente le device du modèle et on retombe
             sur CPU si l'autograd d'ordre 2 n'y est pas supporté (le cas se
             présente sur MPS selon les versions de torch).

    Renvoie un dict {"K_gauss","K_mean","k1","k2","device"} en numpy (N,).
    """
    P = np.asarray(points, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points doit être de forme (N,3), reçu {P.shape}")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    def compute_on(dev):
        m = model.to(dev)
        outs = []
        for start in range(0, len(P), chunk):
            block = torch.tensor(P[start:start + chunk], dtype=torch.float32,
                                 device=dev, requires_grad=True)
            g, H = _hessian_autograd(m, block)
            # Rapatrier AVANT de passer en double : MPS ne gère pas le float64.
            outs.append((g.cpu().double().numpy(), H.cpu().double().numpy()))
        g = np.concatenate([o[0] for o in outs], axis=0)
        H = np.concatenate([o[1] for o in outs], axis=0)
        return g, H

    used = device
    try:
        g, H = compute_on(device)
    except (RuntimeError, NotImplementedError, TypeError) as err:
        if device.type == "cpu":
            raise
        if verbose:
            print(f"  [courbure] autograd d'ordre 2 indisponible sur "
                  f"{device.type} ({type(err).__name__}) -> repli sur CPU")
        used = torch.device("cpu")
        g, H = compute_on(used)

    # La hessienne autograd peut être très légèrement asymétrique
    # (accumulation flottante) : on la symétrise, c'est exact au sens du
    # théorème de Schwarz.
    H = 0.5 * (H + np.transpose(H, (0, 2, 1)))

    K_gauss, K_mean, k1, k2 = curvatures_from_derivatives(
        g[:, 0], g[:, 1], g[:, 2],
        H[:, 0, 0], H[:, 0, 1], H[:, 0, 2],
        H[:, 1, 1], H[:, 1, 2], H[:, 2, 2],
    )
    return {"K_gauss": K_gauss, "K_mean": K_mean, "k1": k1, "k2": k2,
            "device": str(used)}


def summarize(curv, name="", quantiles=(0.05, 0.5, 0.95)):
    """Ligne de résumé lisible d'un champ de courbures."""
    kg, km = np.asarray(curv["K_gauss"]), np.asarray(curv["K_mean"])
    q = [f"{np.quantile(kg, t):+.4f}" for t in quantiles]
    return (f"{name:<22s} K_gauss moy={kg.mean():+.4f} "
            f"[q05,q50,q95]={','.join(q)}  |  K_moy moy={km.mean():+.4f} "
            f"min={km.min():+.4f} max={km.max():+.4f}")
