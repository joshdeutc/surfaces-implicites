"""
Section 4 du notebook : la boucle d'entraînement.

Le rituel PyTorch (identique au notebook) :
    pred -> loss -> zero_grad -> backward -> step
"""
import torch
import torch.nn as nn
import torch.optim as optim

from src.device import get_device


def train(model, X_train_t, y_train_t, n_epochs=2000, batch_size=2048, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    N = X_train_t.shape[0]

    for epoch in range(n_epochs):
        idx = torch.randint(0, N, (batch_size,))
        xb = X_train_t[idx]
        yb = y_train_t[idx]

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"epoch {epoch:5d} | loss = {loss.item():.6e}")

    return model


# ============================================================
# Étape 1 — entraînement RÉALISTE (IGR : surface + normales + eikonal + signe)
# ============================================================

def gradient(y, x):
    """∇y par rapport à x (autograd). x doit avoir requires_grad=True."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def train_igr(model, X_surf, normals, X_vol, sign_vol, box,
              n_epochs=3000, batch_surf=2048, batch_vol=2048, lr=1e-3,
              lam_surf=1.0, lam_normal=1.0, lam_eik=0.1, lam_sign=1.0):
    """
    Perte = surface (|f|=0) + normales (∇f=n) + eikonal (‖∇f‖=1)
            + signe (f du bon côté dedans/dehors).
    Aucune valeur exacte de f n'est utilisée.
    """
    device = get_device()
    model = model.to(device)
    X_surf = X_surf.to(device)
    X_vol = X_vol.to(device)
    sign_vol = sign_vol.to(device)
    if normals is not None:
        normals = normals.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    Ns = X_surf.shape[0]
    Nv = X_vol.shape[0]

    for epoch in range(n_epochs):
        # --- échantillonnage des trois jeux de points ---
        si = torch.randint(0, Ns, (batch_surf,), device=device)
        xs = X_surf[si].clone().requires_grad_(True)          # surface

        xe = ((torch.rand(batch_vol, 3, device=device) * 2 - 1) * box).requires_grad_(True)  # eikonal

        vi = torch.randint(0, Nv, (batch_vol,), device=device)
        xv = X_vol[vi]                                         # dedans/dehors
        sv = sign_vol[vi]

        # --- passes avant + gradients ---
        f_s = model(xs)
        g_s = gradient(f_s, xs)
        f_e = model(xe)
        g_e = gradient(f_e, xe)
        f_v = model(xv)

        # --- termes de la perte ---
        loss_surf = f_s.abs().mean()                          # f = 0 en surface
        loss_eik = ((g_e.norm(dim=-1) - 1.0) ** 2).mean()     # ‖∇f‖ = 1
        loss_sign = torch.relu(-sv * f_v).mean()              # bon signe dedans/dehors

        loss = lam_surf * loss_surf + lam_eik * loss_eik + lam_sign * loss_sign

        if normals is not None:
            n_s = normals[si]
            loss_normal = (g_s - n_s).norm(dim=-1).mean()     # ∇f aligné sur la normale
            loss = loss + lam_normal * loss_normal
        else:
            loss_normal = torch.tensor(0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            ln = loss_normal.detach().item() if torch.is_tensor(loss_normal) else 0.0
            print(f"epoch {epoch:5d} | loss={loss.item():.4e} "
                  f"| surf={loss_surf.item():.3e} eik={loss_eik.item():.3e} "
                  f"sign={loss_sign.item():.3e} norm={ln:.3e}", flush=True)

    return model
