# TZ2 – Méthodes neuronales pour les surfaces implicites

TZ encadrée par Florian De Vuyst (labo BMBI, UTC), faite par Nathan Jacas et Joshua Nancey.

Ce fichier sert de point d'entrée : si tu débarques sur le projet ou que tu veux
juste te repérer, lis-le en premier.

## Le sujet en deux lignes

On veut représenter une surface fermée 3D par une équation implicite f(x,y,z)=0,
mais au lieu de connaître f à l'avance, on l'apprend avec un réseau de neurones à
partir de points (dedans / dehors / sur la surface). Ensuite on reconstruit la
surface avec du marching cubes. Le but final, c'est d'appliquer ça aux capsules
déformables du BMBI, et si on a le temps, à des formes qui bougent dans le temps.

Tout le "pourquoi" et les directions de travail sont dans les mails de Florian
(dossier `Encadrement/`), en particulier celui avec ses recommandations.

## Comment le dossier est rangé

- `Encadrement/` — la fiche d'enregistrement de la TZ et les deux mails de Florian.
  Commence par là si tu veux comprendre ce qu'on doit faire et dans quel esprit.
  Le mail "recommandations" est le plus important (données volumiques, SDF,
  l'idée du polynôme, le morphing, les surfaces qui évoluent dans le temps).

- `Articles/` — la biblio en PDF. Les deux articles de fond sont DeepSDF (2019) et
  Occupancy Networks (2019) ; il y a aussi l'article Nature Comm 2021 sur le
  globule rouge qui sert pour la forme biconcave.

- `Notebooks/` — les notebooks du début, simples à lancer pour visualiser :
  la gyroïde, le globule rouge analytique, et le premier apprentissage sur une
  sphère / un ellipsoïde. C'est le bon endroit pour se faire la main.

- `Code/` — là où on code sérieusement. C'est une version propre et découpée du
  notebook d'apprentissage (dossier `src/`), avec la vraie loss (surface + signe
  dedans/dehors + eikonal), le globule rouge, etc. Le détail et l'avancement sont
  dans `Code/README.md` et `Code/PLAN.md`.

- `Données/` — vide pour l'instant, c'est là qu'on mettra les données de capsules
  quand Florian nous les enverra.

## Par où commencer (si tu prends le train en route)

1. Lis le mail de recommandations dans `Encadrement/`.
2. Ouvre un notebook dans `Notebooks/` pour voir à quoi ça ressemble.
3. Regarde `Code/README.md` puis lance `Code/main_eikonal.py` pour voir
   l'apprentissage "réaliste" tourner.

## Où on en est

Le pipeline marching cubes + rendu marche bien. Pour l'apprentissage, on est
passés d'une version "triche" (on donnait la vraie valeur de f) à une vraie loss
façon DeepSDF (surface + dedans/dehors + eikonal), testée sur l'ellipsoïde et le
globule rouge et ça reconstruit bien.

Ce qui reste : les courbures du globule rouge, la dimension temporelle (le plan
est déjà réfléchi mais pas encore codé) et surtout brancher les vraies données
du BMBI.
