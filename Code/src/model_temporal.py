"""
Le réseau espace-temps : f(x, y, z, t) -> valeur du champ implicite.

Une seule fonction pour toute la séquence
-----------------------------------------
Il y a deux façons d'apprendre une forme mobile. La première est d'entraîner un
réseau par instant : c'est facile, mais ce n'est qu'un album de photos — on n'a
rien condensé, et on ne sait rien dire entre deux clichés. La seconde, celle
d'ici, est d'ajouter le temps comme QUATRIÈME COORDONNÉE d'entrée et de
n'apprendre qu'un seul champ. C'est la formulation de la fiche
d'enregistrement : « condenser la forme mobile en une surface implicite
espace-temps ».

La conséquence pratique est qu'on peut alors demander au réseau la forme à un
instant qu'il n'a jamais vu. C'est le test central de main_temporal.py.

L'initialisation, et pourquoi on annule la colonne du temps
-----------------------------------------------------------
IGRNet doit sa robustesse à son initialisation GÉOMÉTRIQUE : avant tout
apprentissage, le réseau approxime déjà la distance signée à une sphère, si bien
que la descente de gradient part d'un champ sensé au lieu de partir du bruit.

Transposée telle quelle à quatre entrées, cette initialisation produirait la
sphère de R⁴ — un champ qui dépend de t dès le départ, et dans une direction
arbitraire. On préfère partir d'un champ RIGOUREUSEMENT INDÉPENDANT DU TEMPS :
au départ, le réseau dit « une sphère, la même à tout instant », et c'est
l'apprentissage seul qui introduit le mouvement.

Il suffit pour cela d'annuler, à l'initialisation, les deux colonnes de poids
par lesquelles t entre dans le réseau : celle de la première couche, et celle de
la skip-connection qui réinjecte l'entrée au milieu. Les poids ne sont pas
gelés — ils sont libres dès le premier pas de gradient. C'est un point de
départ, pas une contrainte.

Le temps est recentré en interne (t ∈ [0,1] -> [-1,1]) pour qu'il ait la même
amplitude que les coordonnées d'espace ; l'appelant manipule toujours le t
physique.
"""
import numpy as np
import torch
import torch.nn as nn


class TemporalIGRNet(nn.Module):
    """
    IGRNet à quatre entrées : Softplus, initialisation géométrique, skip.

    forward(x) accepte un tenseur (N,4) dont les colonnes sont (x, y, z, t) et
    renvoie (N,1).
    """

    D_IN = 4

    def __init__(self, hidden_dims=(256, 256, 256, 256), skip_in=(2,),
                 radius_init=1.0, beta=100, t_min=0.0, t_max=1.0):
        super().__init__()
        d_in = self.D_IN
        dims = [d_in] + list(hidden_dims) + [1]
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        for layer in range(self.num_layers - 1):
            out_dim = (dims[layer + 1] - d_in if (layer + 1) in skip_in
                       else dims[layer + 1])
            lin = nn.Linear(dims[layer], out_dim)

            if layer == self.num_layers - 2:               # dernière couche
                nn.init.normal_(lin.weight,
                                mean=np.sqrt(np.pi) / np.sqrt(dims[layer]),
                                std=1e-5)
                nn.init.constant_(lin.bias, -radius_init)
            else:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, f"lin{layer}", lin)

        self.activation = nn.Softplus(beta=beta)
        self._neutraliser_temps_a_l_init()

    def _neutraliser_temps_a_l_init(self):
        """
        Met à zéro les poids par lesquels t entre, pour que f(·, t) soit la même
        fonction à tout instant AU DÉPART.

        Deux portes d'entrée seulement :
          - la première couche, dernière colonne de sa matrice ;
          - la couche qui suit la skip, dont l'entrée est [caché | x, y, z, t] :
            la colonne de t y est la dernière.
        """
        with torch.no_grad():
            self.lin0.weight[:, self.D_IN - 1].zero_()
            for layer in self.skip_in:
                lin = getattr(self, f"lin{layer}")
                lin.weight[:, -1].zero_()

    def _recentrer(self, x):
        """(x,y,z,t) -> (x,y,z, 2(t−t_min)/(t_max−t_min) − 1)."""
        etendue = max(self.t_max - self.t_min, 1e-12)
        t = 2.0 * (x[..., 3:4] - self.t_min) / etendue - 1.0
        return torch.cat([x[..., :3], t], dim=-1)

    def forward(self, x):
        inp = self._recentrer(x)
        h = inp
        for layer in range(self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            if layer in self.skip_in:
                h = torch.cat([h, inp], dim=-1) / np.sqrt(2)
            h = lin(h)
            if layer < self.num_layers - 2:
                h = self.activation(h)
        return h


def figer_instant(model, t, device=None):
    """
    Vue statique du champ appris : un callable (N,3) -> (N,1) à t fixé.

    Indispensable pour réutiliser tel quel tout le code statique du projet
    (reconstruct.evaluate_grid, metrics.model_inside, metrics.eikonal_error),
    qui attend un modèle à trois entrées. Les gradients passent à travers, donc
    l'objet reste dérivable — c'est ce qui permet à eikonal_error de fonctionner.
    """

    class _Instant(nn.Module):
        def __init__(self):
            super().__init__()
            self.modele = model
            self.t = float(t)

        def forward(self, pts):
            colonne = torch.full((pts.shape[0], 1), self.t,
                                 dtype=pts.dtype, device=pts.device)
            return self.modele(torch.cat([pts, colonne], dim=-1))

    vue = _Instant()
    return vue.to(device) if device is not None else vue


def n_parametres(model):
    return sum(p.numel() for p in model.parameters())
