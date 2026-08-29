"""
SIREN — réseau à activations sinusoïdales (Sitzmann et al., NeurIPS 2020).

Troisième voie de la comparaison d'architectures du projet, aux côtés de
Softplus (`IGRNet`) et de l'encodage de Fourier (`FourierNet`).

L'IDÉE
------
Un MLP à activations lisses classiques (Tanh, Softplus, ReLU) souffre du BIAIS
SPECTRAL : il apprend d'abord les basses fréquences et peine sur les détails
fins. Sur le globule rouge, cela se voit à l'œil nu — la fossette centrale,
qui est le détail à haute courbure de la forme, est arrondie.

Deux réponses possibles :
  - `FourierNet` attaque le problème EN ENTRÉE, en donnant au réseau des
    harmoniques sin(2^k π x) toutes prêtes ;
  - SIREN l'attaque DANS le réseau, en remplaçant l'activation elle-même par
    un sinus : phi(x) = sin(omega0 · Wx + b).

L'avantage de SIREN pour ce projet précis : la dérivée d'un sinus est un
sinus, donc toutes les dérivées d'un SIREN sont elles-mêmes des SIREN. Le
champ reste lisse à TOUS les ordres. Or le projet a besoin de l'ordre 1 (terme
eikonal ‖∇f‖ = 1) et de l'ordre 2 (les courbures de `src/curvature.py`). Une
activation ReLU serait ici inutilisable, une Softplus est acceptable mais ses
dérivées secondes sont molles.

L'INITIALISATION EST LE POINT CRITIQUE
--------------------------------------
Un SIREN mal initialisé ne converge pas du tout. Sitzmann et al. démontrent
(annexe 1.5) qu'il faut tirer les poids dans U(−sqrt(6/n_in)/omega0,
+sqrt(6/n_in)/omega0) pour que l'entrée de chaque sinus reste de variance
unitaire d'une couche à l'autre : sans cela le signal sature ou s'éteint en
profondeur. La première couche est un cas à part, en U(−1/n_in, +1/n_in),
parce que c'est elle qui porte le facteur omega0 appliqué aux COORDONNÉES.

`omega0` est donc le seul vrai réglage : il fixe la fréquence spatiale que le
réseau manipule naturellement. Trop petit, on retombe sur un réseau lisse qui
lisse la fossette ; trop grand, le champ oscille entre les points de données et
le terme eikonal se dégrade. Le balayage est fait dans `main_siren.py`.
"""
import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """Une couche y = sin(omega0 · (Wx + b)), avec l'initialisation SIREN."""

    def __init__(self, d_in, d_out, premiere=False, omega0=30.0):
        super().__init__()
        self.omega0 = float(omega0)
        self.premiere = bool(premiere)
        self.lin = nn.Linear(d_in, d_out)
        self.d_in = d_in
        self._init()

    def _init(self):
        with torch.no_grad():
            if self.premiere:
                # la couche d'entrée voit les coordonnées brutes : c'est elle
                # qui porte omega0, donc ses poids restent en 1/n_in
                borne = 1.0 / self.d_in
            else:
                borne = np.sqrt(6.0 / self.d_in) / self.omega0
            self.lin.weight.uniform_(-borne, borne)
            nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return torch.sin(self.omega0 * self.lin(x))


class SirenNet(nn.Module):
    """
    SIREN complet : sinus sur les couches cachées, couche finale LINÉAIRE.

    La dernière couche est linéaire et non sinusoïdale : la sortie est une
    valeur de fonction implicite, qui doit pouvoir croître en s'éloignant de la
    surface (idéalement comme une distance). Un sinus final la bornerait à
    [-1, 1] et écraserait le champ loin de la surface, ce qui rendrait le terme
    eikonal impossible à satisfaire.

    `omega0_cachees` peut différer de `omega0` : Sitzmann garde 30 pour la
    première couche et 30 pour les suivantes, mais dissocier les deux permet de
    régler séparément la fréquence VUE (entrée) et la fréquence PROPAGÉE.
    """

    def __init__(self, d_in=3, hidden=256, n_layers=4, omega0=30.0,
                 omega0_cachees=None):
        super().__init__()
        w_cache = float(omega0 if omega0_cachees is None else omega0_cachees)
        couches = [SineLayer(d_in, hidden, premiere=True, omega0=omega0)]
        for _ in range(n_layers - 1):
            couches.append(SineLayer(hidden, hidden, premiere=False,
                                     omega0=w_cache))
        finale = nn.Linear(hidden, 1)
        with torch.no_grad():
            borne = np.sqrt(6.0 / hidden) / w_cache
            finale.weight.uniform_(-borne, borne)
            nn.init.zeros_(finale.bias)
        couches.append(finale)
        self.net = nn.Sequential(*couches)
        self.omega0 = float(omega0)

    def forward(self, x):
        return self.net(x)


def nb_poids(model):
    return sum(p.numel() for p in model.parameters())
