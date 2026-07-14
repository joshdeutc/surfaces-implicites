"""
Section 3 du notebook : le réseau de neurones implicite.

MLP : (x,y,z) -> 128 -> 128 -> 128 -> 1, activations Tanh.
La sortie unique est la valeur estimée de f(x,y,z).
"""
import numpy as np
import torch
import torch.nn as nn


class ImplicitMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class IGRNet(nn.Module):
    """
    Réseau adapté à l'apprentissage par eikonal (IGR, Gropp et al. 2020).

    Trois ingrédients qui font la différence pour apprendre une SDF :
      - initialisation GÉOMÉTRIQUE : à l'init, le réseau approxime déjà la SDF
        d'une sphère de rayon `radius_init` -> l'entraînement eikonal part
        d'un champ raisonnable au lieu du bruit ;
      - activation Softplus : lisse, donc gradient ∇f bien défini (requis par
        le terme eikonal) ;
      - skip-connection : on réinjecte l'entrée (x,y,z) au milieu du réseau.
    """

    def __init__(self, d_in=3, hidden_dims=(256, 256, 256, 256),
                 skip_in=(2,), radius_init=1.0, beta=100):
        super().__init__()
        dims = [d_in] + list(hidden_dims) + [1]
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(self.num_layers - 1):
            # si la couche suivante reçoit la skip, on réduit la sortie de d_in
            if (layer + 1) in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # initialisation géométrique
            if layer == self.num_layers - 2:  # dernière couche
                nn.init.normal_(lin.weight,
                                mean=np.sqrt(np.pi) / np.sqrt(dims[layer]),
                                std=1e-5)
                nn.init.constant_(lin.bias, -radius_init)
            else:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, f"lin{layer}", lin)

        self.activation = nn.Softplus(beta=beta)

    def forward(self, x):
        inp = x
        for layer in range(self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            if layer in self.skip_in:
                x = torch.cat([x, inp], dim=-1) / np.sqrt(2)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        return x


# ============================================================
# Étape 2 — encodage de Fourier (contre le biais spectral)
# ============================================================

class FourierFeatures(nn.Module):
    """
    Encodage positionnel : (x,y,z) -> [x, sin(2^k·π·x), cos(2^k·π·x), ...].
    Donne au MLP un accès direct aux hautes fréquences qu'un MLP lisse
    capte mal (biais spectral).
    """

    def __init__(self, d_in=3, n_freqs=6, include_input=True):
        super().__init__()
        self.include_input = include_input
        freqs = (2.0 ** torch.arange(n_freqs)) * np.pi   # [π, 2π, 4π, ...]
        self.register_buffer("freqs", freqs)
        self.d_out = d_in * 2 * n_freqs + (d_in if include_input else 0)

    def forward(self, x):
        xb = x[..., None] * self.freqs                   # (..., d_in, n_freqs)
        enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
        enc = enc.flatten(-2)                            # (..., d_in*2*n_freqs)
        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)
        return enc


class FourierNet(nn.Module):
    """MLP précédé d'un encodage de Fourier. Softplus pour garder ∇f lisse
    (requis par l'eikonal). Pas d'init géométrique : on s'appuie sur les
    termes surface + signe + eikonal."""

    def __init__(self, d_in=3, n_freqs=6, hidden=256, n_layers=4, beta=100):
        super().__init__()
        self.ff = FourierFeatures(d_in, n_freqs)
        dims = [self.ff.d_out] + [hidden] * n_layers + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Softplus(beta=beta))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.ff(x))
