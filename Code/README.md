# Surfaces implicites apprises par réseau de neurones

Représentation d'une forme fermée 3D par une fonction implicite `f(x,y,z) = 0`
apprise par un réseau de neurones (PyTorch), puis reconstruite par
*marching cubes*. Application visée : microcapsules synthétiques (labo BMBI).

Ce dépôt part de la démo `Learn_SurfaceCapsule.ipynb` (sphère / ellipsoïde),
découpée en modules pour itérer proprement.

## Structure

```
src/
  shapes.py       # formes test + échantillonnage de surface   (notebook §1)
  data.py         # construction du jeu d'entraînement          (notebook §2)
  model.py        # réseau ImplicitMLP                          (notebook §3)
  train.py        # boucle d'entraînement                       (notebook §4)
  reconstruct.py  # grille 3D + marching cubes                  (notebook §5-6)
  visualize.py    # figure 3D (sauvegardée en PNG)              (notebook §7)
main.py           # orchestre tout le pipeline
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancer

```bash
python main.py
```

La figure reconstruite est écrite dans `output/surface.png`.

## Notes

- Une graine aléatoire fixe (`set_seed`) est ajoutée par rapport au notebook
  pour rendre les exécutions reproductibles.
- `main.py` enregistre une image PNG au lieu de l'affichage interactif
  `%matplotlib ipympl` du notebook (pas besoin d'écran).
