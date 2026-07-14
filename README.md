# TZ2 – Méthodes neuronales pour les surfaces implicites

TZ encadrée par Florian De Vuyst (labo BMBI, UTC), faite par Nathan Jacas et Joshua Nancey.

L'idée : représenter une surface fermée 3D par une fonction implicite f(x,y,z)=0
apprise par un réseau de neurones, puis la reconstruire avec du marching cubes.
À terme on veut appliquer ça aux capsules déformables du BMBI, et si possible
à des formes qui bougent dans le temps.

## Organisation du dossier

- `Encadrement/` : la fiche d'enregistrement de la TZ et les mails de Florian
  (dont celui avec toutes ses recommandations de travail).
- `Articles/` : la biblio en PDF (DeepSDF, Occupancy Networks, l'article Nature
  Comm 2021 sur le globule rouge).
- `Notebooks/` : les notebooks de départ (gyroïde, globule rouge analytique,
  premier apprentissage sur sphère/ellipsoïde) + un rendu.
- `Code/` : là où on code pour de vrai. Version modulaire (dossier `src/`) avec
  la loss eikonal, le globule rouge biconcave, les features de Fourier, etc.
  Voir `Code/README.md` et `Code/PLAN.md` pour le détail et l'avancement.
- `Données/` : pour les données de capsules du BMBI quand on les aura.

## Pour lancer le code

Tout est dans `Code/`. Installer les dépendances puis lancer un des scripts :

```
cd Code
pip install -r requirements.txt
python main_eikonal.py     # apprentissage réaliste (surface + dedans/dehors + eikonal)
```

Le détail des scripts (`main.py`, `main_biconcave.py`, `main_step2.py`) est dans
`Code/README.md`.

## Où on en est

Le pipeline marching cubes + rendu marche bien. Côté apprentissage, on est passés
de la version "triche" (MSE sur la vraie valeur de f) à une vraie loss façon
DeepSDF/IGR (surface + signe dedans/dehors + eikonal), validée sur ellipsoïde et
globule rouge. Restent : les courbures du globule rouge, la dimension temporelle
(planifiée dans `Code/.claude/`, pas encore codée) et les vraies données BMBI.
