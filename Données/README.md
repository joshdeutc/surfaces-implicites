# Données du BMBI — capsule déformable en écoulement

**Propriété du laboratoire BMBI de l'UTC, transmises sous clause de
confidentialité** (mail du 26/08/2026, voir `Encadrement/`). Rien de ce dossier
ne part sur un dépôt distant : `.gitignore` n'y laisse passer que ce README.

Le dataset fait 1.2 Go. Il n'est pas versionné : pour le récupérer, reprendre le
lien FileSender du mail, ou demander à Florian.

## Contenu

```
Data_Capsule/
├── connect.npy                (5120, 3)          triangles, indices de nœuds
├── off_centered_capsule.npy   (20001, 2562, 3)   x,y,z de chaque nœud, par pas
└── read_capsuleDataset.ipynb  le notebook de lecture fourni par Florian
```

Simulation Éléments Finis d'une **capsule unique** dans un canal. Le maillage est
une icosphère de subdivision 4 : `V = 2562`, `F = 5120`, `E = 7680`, donc
`χ = 2` — fermée, manifold, genre 0, et déjà orientée vers l'extérieur. La
connectivité ne change jamais : les 2562 nœuds sont des **points matériels**
suivis dans le temps.

## Par où l'utiliser

Ne pas relire les `.npy` à la main : passer par le module du projet, qui gère la
mémoire (1.2 Go en `mmap`), le recentrage et les normales.

```python
from src import capsules
faces, caps = capsules.charger()
verts, centre = capsules.instant(caps, 800)
```

`Code/main_capsules.py` produit la carte d'identité complète du dataset et
l'exporte au format d'échange du projet. Les résultats et les pièges sont dans
`Code/RESULTATS_capsules.md`.
