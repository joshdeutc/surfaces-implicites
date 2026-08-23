"""
Entraînement du champ espace-temps f(x,y,z,t), à partir des seuls fichiers.

La perte reprend celle d'IGR (Gropp et al. 2020), avec une modification qui
n'est pas cosmétique et un choix d'échantillonnage qui décide de tout.

LE PIÈGE — l'eikonal ne porte que sur le gradient d'ESPACE
-----------------------------------------------------------
Le réseau a quatre entrées, donc son gradient a quatre composantes :

    ∇f = ( ∂f/∂x , ∂f/∂y , ∂f/∂z , ∂f/∂t ).

La condition eikonal ‖∇f‖ = 1 dit que f est une distance signée. Or f est une
distance signée DANS L'ESPACE, à chaque instant figé. Le temps n'est pas une
quatrième dimension d'espace : ∂f/∂t n'est pas une pente géométrique, c'est la
vitesse à laquelle le champ change en un point fixe — une grandeur qui a sa
propre unité et sa propre échelle.

Écrire ‖∇f‖ = 1 sur les quatre composantes reviendrait donc à exiger

    (∂f/∂x)² + (∂f/∂y)² + (∂f/∂z)² = 1 − (∂f/∂t)²,

c'est-à-dire à demander au champ d'être d'autant MOINS une distance signée qu'il
bouge vite. Le terme censé régulariser la géométrie punirait le mouvement. Sur
une séquence où la forme se déforme franchement, cela se paie en surface épaissie
et en volume faux.

D'où, partout dans ce module : `grad[..., :3]`.

LE CHOIX QUI DÉCIDE DE TOUT — où l'on tire t
---------------------------------------------
Les termes de surface et de signe sont attachés aux six instants observés : ils
ne peuvent rien dire d'autre. Si le terme eikonal était lui aussi tiré sur ces
six valeurs, RIEN dans la perte ne contraindrait le champ entre deux instants :
le réseau serait libre d'y mettre n'importe quoi, et le test d'interpolation ne
mesurerait que le hasard de l'initialisation.

Le terme eikonal est donc tiré à t CONTINU dans [0,1]. C'est la seule pièce de
la perte qui parle des instants non observés, et elle n'y impose qu'une chose,
purement géométrique : que le champ y reste une distance signée. Elle ne dit
jamais quelle forme il doit y avoir.

C'est ce qui rend le test d'interpolation honnête. Si la reconstruction tient
aux instants cachés, ce n'est pas parce qu'on l'y a mise : c'est parce que
« rester une distance signée à tout instant » plus « coller aux six photos »
suffit à reconstituer le mouvement entre elles.

Le signe dedans/dehors vient du NUAGE, jamais d'une formule : produit scalaire
avec la normale du plus proche voisin, vote sur k = 5 (src/inside_outside.py).
"""
import numpy as np
import torch
import torch.optim as optim

from src.device import get_device
from src.inside_outside import OrientedCloud
from src.cloud_io import load_cloud


# ------------------------------------------------------------------
# Constitution du jeu d'entraînement — uniquement depuis les fichiers
# ------------------------------------------------------------------

def jeu_depuis_fichiers(instants, k=5, n_vol_par_instant=20000, seed=0,
                        vote="majorite"):
    """
    Lit la séquence de fichiers et en tire tout ce dont l'entraînement a besoin.

    `instants` : liste de (t, chemin). Aucune formule n'est appelée ici, ni
    ensuite : `load_cloud` est la seule porte d'entrée.

    Renvoie un dict de tableaux numpy :
        X_surf (Ns,4) : points de surface, quatrième colonne = t
        N_surf (Ns,3) : normales sortantes
        X_vol  (Nv,4) : points de volume, quatrième colonne = t
        s_vol  (Nv,)  : −1 dedans / +1 dehors, décidé par produit scalaire
        box, t_min, t_max, detail (par instant)
    """
    rng = np.random.default_rng(seed)
    Xs, Ns_, Xv, Sv, detail = [], [], [], [], []
    box = None

    for i, (t, chemin) in enumerate(instants):
        d = load_cloud(chemin)
        if box is None:
            box = d["box"]
        elif abs(d["box"] - box) > 1e-12:
            raise ValueError("les instants n'ont pas la même boîte d'étude")

        nuage = OrientedCloud(d["points"], d["normals"])
        P = rng.uniform(-box, box, size=(n_vol_par_instant, 3))
        s = nuage.sign(P, k=k, vote=vote)              # <- le signe vient du nuage

        col = np.full((len(nuage.points), 1), float(t))
        Xs.append(np.hstack([nuage.points, col]))
        Ns_.append(nuage.normals)
        Xv.append(np.hstack([P, np.full((len(P), 1), float(t))]))
        Sv.append(s)
        detail.append({"t": float(t), "fichier": chemin, "n_surface": len(nuage),
                       "source": d["source"],
                       "dedans": int((s < 0).sum()), "dehors": int((s > 0).sum())})

    ts = [t for t, _ in instants]
    return {"X_surf": np.vstack(Xs), "N_surf": np.vstack(Ns_),
            "X_vol": np.vstack(Xv), "s_vol": np.concatenate(Sv),
            "box": float(box), "t_min": float(min(ts)), "t_max": float(max(ts)),
            "detail": detail}


def en_tenseurs(jeu):
    """Conversion float32 pour PyTorch."""
    return {
        "X_surf": torch.tensor(jeu["X_surf"], dtype=torch.float32),
        "N_surf": torch.tensor(jeu["N_surf"], dtype=torch.float32),
        "X_vol": torch.tensor(jeu["X_vol"], dtype=torch.float32),
        "s_vol": torch.tensor(jeu["s_vol"][:, None], dtype=torch.float32),
    }


# ------------------------------------------------------------------
# La boucle
# ------------------------------------------------------------------

def gradient(y, x):
    """∇y par rapport à x — les QUATRE composantes ; le tri se fait après."""
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]


def train_temporel(model, X_surf, N_surf, X_vol, s_vol, box, t_min=0.0, t_max=1.0,
                   n_epochs=8000, batch_surf=2048, batch_vol=2048, lr=1e-3,
                   lam_surf=1.0, lam_normal=1.0, lam_eik=0.1, lam_sign=1.0,
                   decay_every=3000, decay=0.5, log_every=200, device=None,
                   seed=None):
    """
    Perte = surface + normales + eikonal SPATIAL + signe.

    - surface  : |f(x_s, t_s)| -> 0 aux points de surface de chaque instant ;
    - normales : ∇_x f = n aux mêmes points (gradient d'espace seul) ;
    - eikonal  : (‖∇_x f‖ − 1)² sur (x, t) tirés au hasard, t CONTINU ;
    - signe    : relu(−s·f) aux points de volume, s issu du nuage.

    Renvoie (model, historique).
    """
    device = device or get_device()
    model = model.to(device)
    X_surf, N_surf = X_surf.to(device), N_surf.to(device)
    X_vol, s_vol = X_vol.to(device), s_vol.to(device)

    if seed is not None:
        torch.manual_seed(int(seed))

    optimiseur = optim.Adam(model.parameters(), lr=lr)
    planning = optim.lr_scheduler.StepLR(optimiseur, step_size=decay_every,
                                         gamma=decay)
    Ns, Nv = X_surf.shape[0], X_vol.shape[0]
    etendue_t = t_max - t_min
    historique = []

    for epoch in range(n_epochs):
        si = torch.randint(0, Ns, (batch_surf,), device=device)
        xs = X_surf[si].clone().requires_grad_(True)

        # points eikonal : espace uniforme dans la boîte, TEMPS CONTINU
        xe_espace = (torch.rand(batch_vol, 3, device=device) * 2 - 1) * box
        te = torch.rand(batch_vol, 1, device=device) * etendue_t + t_min
        xe = torch.cat([xe_espace, te], dim=-1).requires_grad_(True)

        vi = torch.randint(0, Nv, (batch_vol,), device=device)
        xv, sv = X_vol[vi], s_vol[vi]

        f_s = model(xs)
        g_s = gradient(f_s, xs)[..., :3]        # <- gradient d'ESPACE
        f_e = model(xe)
        g_e = gradient(f_e, xe)[..., :3]        # <- gradient d'ESPACE
        f_v = model(xv)

        perte_surf = f_s.abs().mean()
        perte_norm = (g_s - N_surf[si]).norm(dim=-1).mean()
        perte_eik = ((g_e.norm(dim=-1) - 1.0) ** 2).mean()
        perte_signe = torch.relu(-sv * f_v).mean()

        perte = (lam_surf * perte_surf + lam_normal * perte_norm
                 + lam_eik * perte_eik + lam_sign * perte_signe)

        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        planning.step()

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            ligne = {"epoch": epoch, "perte": float(perte.item()),
                     "surf": float(perte_surf.item()),
                     "norm": float(perte_norm.item()),
                     "eik": float(perte_eik.item()),
                     "signe": float(perte_signe.item()),
                     "lr": float(optimiseur.param_groups[0]["lr"])}
            historique.append(ligne)
            print(f"  epoch {epoch:6d} | perte={ligne['perte']:.4e} "
                  f"| surf={ligne['surf']:.3e} norm={ligne['norm']:.3e} "
                  f"eik={ligne['eik']:.3e} signe={ligne['signe']:.3e}", flush=True)

    return model, historique


# ------------------------------------------------------------------
# Dérivées du champ appris — servent au suivi lagrangien
# ------------------------------------------------------------------

def derivees(model, points, t, device=None, chunk=8192):
    """
    ∇_x f et ∂f/∂t du champ APPRIS, en des points de l'espace à l'instant t.

    Renvoie (f, grad_x, dfdt) en numpy. C'est tout ce dont on a besoin pour la
    vitesse normale de la surface de niveau :

        v_n = − (∂f/∂t) / ‖∇_x f‖

    qui se lit directement sur la dérivée de f(x(t), t) = 0 le long du mouvement.
    """
    device = device or get_device()
    model = model.to(device).eval()
    P = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    f_out = np.empty(len(P), dtype=np.float64)
    g_out = np.empty((len(P), 3), dtype=np.float64)
    dt_out = np.empty(len(P), dtype=np.float64)

    for i in range(0, len(P), chunk):
        bloc = torch.as_tensor(P[i:i + chunk], device=device)
        col = torch.full((len(bloc), 1), float(t), device=device)
        x = torch.cat([bloc, col], dim=-1).requires_grad_(True)
        f = model(x)
        g = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=False)[0]
        f_out[i:i + chunk] = f.detach().cpu().numpy().ravel()
        g_out[i:i + chunk] = g[:, :3].detach().cpu().numpy()
        dt_out[i:i + chunk] = g[:, 3].detach().cpu().numpy()

    return f_out, g_out, dt_out


def vitesse_normale(model, points, t, device=None, eps=1e-9):
    """
    Vitesse normale de la surface de niveau au point donné, et la normale.

    Renvoie (v_n, n) avec n = ∇_x f / ‖∇_x f‖ (sortante) et le déplacement
    correspondant ẋ = v_n · n.
    """
    _, g, dfdt = derivees(model, points, t, device=device)
    norme = np.maximum(np.linalg.norm(g, axis=1), eps)
    return -dfdt / norme, g / norme[:, None]
