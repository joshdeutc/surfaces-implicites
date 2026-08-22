# TZ2 – Méthodes neuronales pour les surfaces implicites

TZ encadrée par Florian De Vuyst (labo BMBI, UTC), faite par Nathan Jacas et Joshua Nancey.

Ce fichier sert de point d'entrée : si tu débarques sur le projet ou que tu veux
juste te repérer, lis-le en premier.

## Le sujet en deux lignes

On veut représenter une surface fermée 3D par une équation implicite f(x,y,z)=0,
mais au lieu de connaître f à l'avance, on l'apprend avec un réseau de neurones à
partir de points (dedans / dehors / sur la surface). Ensuite on reconstruit la
surface avec du marching cubes. Le but final, c'est d'appliquer ça aux capsules
déformables du BMBI, et de « condenser la forme mobile en une surface implicite
espace-temps » — c'est la formulation de la fiche d'enregistrement.

Tout le « pourquoi » et les directions de travail sont dans les mails de Florian
(dossier `Encadrement/`), en particulier celui avec ses recommandations.

## Comment le dossier est rangé

```
TZ2_Surfaces_Implicites_Neuronales/
├── README.md            ce fichier — le point d'entrée
├── Encadrement/         la fiche d'enregistrement de la TZ + les 2 mails de Florian
├── Articles/            la biblio en PDF (DeepSDF, Occupancy Networks, RBC Kuchel)
├── Notebooks/           les notebooks du début, simples à lancer pour visualiser
├── Données/             vide — en attente des capsules du BMBI
└── Code/                le code sérieux
    ├── README.md          structure des modules et liste des scripts
    ├── PLAN.md            feuille de route et journal d'avancement
    ├── RESULTATS_courbures.md   courbures, invariants, Gauss-Bonnet
    ├── RESULTATS_nuage.md       apprendre depuis un nuage de points seul
    ├── RESULTATS_morphing.md    morphing, booléens, contrôle de la courbure
    ├── src/               les modules (formes, données, réseaux, géométrie, métriques)
    ├── main_*.py          un script par sujet d'étude
    ├── validate_geometry.py     les vérifications des outils géométriques
    └── output/            figures et modèles produits (ignoré par git)
```

- `Encadrement/` — **commence par là** pour comprendre ce qu'on doit faire et dans
  quel esprit. Le mail « recommandations » est le plus important (données
  volumiques, SDF, l'idée du polynôme, le morphing, les surfaces qui évoluent
  dans le temps). La fiche d'enregistrement donne le programme officiel en heures.

- `Articles/` — les deux articles de fond sont DeepSDF (2019) et Occupancy
  Networks (2019) ; l'article Nature Comm 2021 sur le globule rouge donne à la
  fois l'équation quartique de la forme et les formules de courbure qu'on utilise.

- `Notebooks/` — la gyroïde, le globule rouge analytique, et le premier
  apprentissage sur une sphère / un ellipsoïde. Le bon endroit pour se faire la main.

- `Code/` — la version propre et découpée, avec la vraie loss (surface +
  dedans/dehors + eikonal), les métriques, la géométrie différentielle et
  l'ajustement polynomial. Le détail est dans `Code/README.md` et `Code/PLAN.md`.

- `Données/` — vide pour l'instant, c'est là qu'iront les données de capsules
  quand Florian nous les enverra.

## Par où commencer (si tu prends le train en route)

1. Lis le mail de recommandations dans `Encadrement/`.
2. Ouvre un notebook dans `Notebooks/` pour voir à quoi ça ressemble.
3. Regarde `Code/README.md`, puis lance `python Code/main_eikonal.py` pour voir
   l'apprentissage « réaliste » tourner (environ 40 secondes).
4. Lance `python Code/validate_geometry.py` : 28 vérifications sur des cas dont
   la réponse est connue d'avance. C'est le meilleur résumé de ce que le code
   sait faire.

## Où on en est

**Ce qui marche.** Le pipeline complet points → réseau → marching cubes → figure,
avec une loss réaliste : on n'utilise jamais la valeur exacte de f pour
entraîner, seulement des points de surface, le signe dedans/dehors et la loi
eikonal. Testé sur l'ellipsoïde et le globule rouge, reconstruction sous-voxel.

**Ce qu'on sait mesurer.** Des métriques indépendantes de toute formule (Chamfer,
IoU, Hausdorff, erreur eikonal) — donc transposables aux vraies capsules — et des
invariants géométriques (Euler, aire, volume, Gauss-Bonnet) qui disent si l'on
peut croire une surface reconstruite.

**Le résultat le plus intéressant.** Le Chamfer teste la position de la surface,
l'eikonal teste le gradient ; les courbures testent les dérivées secondes, et
personne ne le fait d'habitude. Verdict : le réseau est juste à l'ordre 2
globalement (Gauss-Bonnet à 0.02 % de 4π) mais il lisse les extrêmes de courbure
d'environ 16 % — au centre du creux du globule il annonce 12.1 au lieu de 14.4.

**Le globule rouge tient en 10 coefficients.** Son équation exacte est une
quartique ; l'ajustement polynomial la retrouve à 5e-7 près, contre 197 886 poids
pour le réseau. C'est l'argument de compression et d'explicabilité que Florian
recommandait.

**La chaîne « données réelles » est prête.** On sait apprendre une surface à
partir d'un simple fichier de nuage de points orienté, sans jamais aucune
formule — c'est ce qui s'appliquera aux capsules du BMBI. Le signe dedans/dehors
déduit du nuage est exact à 100 % dès qu'on s'éloigne un peu de la surface, et
l'entraînement qui s'en sert vaut celui qui triche avec la formule : l'écart
entre les deux est plus petit que la dispersion entre deux graines.

**Le morphing et les opérations booléennes marchent**, sur formules comme sur
champs appris, avec une vérification quantitative (l'identité
`V(A∪B) + V(A∩B) = V(A) + V(B)` tombe à 1.1e-06 % près).

**Ce qui reste.** Le spatio-temporel — le cœur du sujet — la robustesse au bruit
et aux vues partielles, les familles de formes à code latent, et surtout
brancher les vraies données du BMBI, qui n'ont pas encore été fournies. Le
détail et l'ordre sont dans `Code/PLAN.md`.
