"""
Section 2 du notebook : construction du jeu d'entraînement.

On mélange :
  - des points volumiques (tirés dans la boîte) étiquetés par la valeur exacte de f
  - des points de surface étiquetés f = 0
puis on convertit en tenseurs PyTorch.
"""
import numpy as np
import torch


def build_training_data(surface_points, implicit_exact, box, n_vol=20000):
    # Points aléatoires dans la boîte 3D, étiquetés par f exact
    X_vol = np.random.uniform(-box, box, size=(n_vol, 3))
    y_vol = implicit_exact(X_vol)

    # Points exactement sur la surface (f = 0)
    X_surf = surface_points
    y_surf = np.zeros((len(X_surf),), dtype=np.float32)

    # On empile les deux paquets
    X_train = np.vstack([X_vol, X_surf]).astype(np.float32)
    y_train = np.concatenate([y_vol, y_surf]).astype(np.float32)

    # On borne f pour éviter les grandes valeurs loin de la surface
    y_train = np.clip(y_train, -1.0, 1.0)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train[:, None], dtype=torch.float32)
    return X_train_t, y_train_t


def build_training_data_realistic(surface_points, implicit_exact, box,
                                  normals=None, n_vol=20000):
    """
    Étape 1 — données RÉALISTES : on n'utilise JAMAIS la valeur exacte de f.

    Renvoie :
      X_surf   : points de surface (f = 0)
      n_surf   : normales en surface (ou None)
      X_vol    : points de volume
      sign_vol : signe dedans/dehors de ces points (+1 dehors, -1 dedans)

    NB : implicit_exact n'est utilisé ici QUE pour fabriquer le SIGNE
    (ce qu'une vraie mesure dedans/dehors fournirait). Sa valeur précise
    est jetée : on ne garde que np.sign(...).
    """
    X_surf = surface_points.astype(np.float32)

    X_vol = np.random.uniform(-box, box, size=(n_vol, 3)).astype(np.float32)
    sign_vol = np.sign(implicit_exact(X_vol)).astype(np.float32)  # +1 / -1

    X_surf_t = torch.tensor(X_surf, dtype=torch.float32)
    X_vol_t = torch.tensor(X_vol, dtype=torch.float32)
    sign_vol_t = torch.tensor(sign_vol[:, None], dtype=torch.float32)
    normals_t = (torch.tensor(normals, dtype=torch.float32)
                 if normals is not None else None)
    return X_surf_t, normals_t, X_vol_t, sign_vol_t
