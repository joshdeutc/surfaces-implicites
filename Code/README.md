# Surfaces implicites apprises par réseau de neurones

Représentation d'une forme fermée 3D par une fonction implicite `f(x,y,z) = 0`
apprise par un réseau de neurones (PyTorch), puis reconstruite par
*marching cubes*. Application visée : microcapsules déformables (labo BMBI).

Ce dossier part de la démo `Learn_SurfaceCapsule.ipynb` (sphère / ellipsoïde),
découpée en modules pour itérer proprement.

## Deux régimes d'apprentissage

- **« Triche » (référence)** — `main.py` régresse la valeur exacte de `f` par MSE.
  Ne sert que de point de comparaison : sur de vraies données, aucune formule
  n'existe.
- **Réaliste (le vrai travail)** — on n'utilise jamais la valeur de `f`, seulement
  des points de surface, le signe dedans/dehors et la loi eikonal `‖∇f‖ = 1`
  (méthode IGR, Gropp et al. 2020).

Convention partout : `f < 0` intérieur · `f = 0` surface · `f > 0` extérieur.

## Structure

```
src/
  shapes.py       formes analytiques : sphère, ellipsoïde, globule biconcave,
                  globule rouge QUARTIQUE (Kuchel 2021), tore — et leurs normales
  shapes_complexes.py  gyroïde, Chmutov, Barth, metaballs — le stress-test
                  topologique, avec projection de Newton sur la surface
  capsules.py     LES VRAIES DONNÉES DU BMBI : lecture mmap du dataset,
                  recentrage, normales par triangle, export au format d'échange,
                  et les mesures qui décrivent la déformation
  data.py         construction des jeux d'entraînement (triche / réaliste)
  model.py        réseaux : ImplicitMLP (Tanh), IGRNet (Softplus + init
                  géométrique + skip), FourierNet
  model_siren.py  SIREN (Sitzmann 2020) — activations sinusoïdales, dérivées
                  lisses à tous les ordres, initialisation critique
  polyfit.py      ajustement algébrique de f par SVD (piste polynomiale)
  train.py        boucles d'entraînement : MSE et IGR
  reconstruct.py  évaluation sur grille 3D + marching cubes
  metrics.py      Chamfer, Hausdorff, IoU, eikonal — SANS la formule exacte
  inside_outside.py  dedans/dehors depuis le seul nuage orienté (KD-tree +
                  produit scalaire avec la normale, vote sur k voisins)
  cloud_io.py     format d'échange .npz, maillage triangulaire « façon EF »,
                  normales par triangle orientées vers l'extérieur
  csg.py          morphing θf₁+(1−θ)f₂, union, intersection, différence
  curvature.py    courbures gaussienne / moyenne / principales, en numpy
                  (différences finies) et en torch (autograd d'ordre 2)
  invariants.py   caractéristique d'Euler, aire, volume, Gauss-Bonnet
  temporal.py     séquence d'instants discrets : déformation à volume constant,
                  écriture des fichiers, et champ exact réservé à la MESURE
  model_temporal.py  réseau f(x,y,z,t) : IGRNet à quatre entrées, initialisé
                  indépendant du temps
  train_temporal.py  perte espace-temps (eikonal SPATIAL, t continu) et
                  dérivées ∇_x f, ∂f/∂t du champ appris
  visualize.py    figure 3D matplotlib (PNG)
  viz_pv.py       rendu PyVista hors écran, caméra déterministe, champ scalaire
                  aux sommets, exports OBJ / VTK / VTP / PVD
  device.py       CUDA / MPS (Apple) / CPU
```

## Scripts

| Script | Ce qu'il fait |
|--------|---------------|
| `main.py` | pipeline « triche » d'origine (référence) |
| `main_eikonal.py` | apprentissage réaliste sur l'ellipsoïde |
| `main_biconcave.py` | globule rouge biconcave + coupe de la fossette |
| `main_step2.py` | comparaison Softplus vs Fourier, sans normales |
| `main_poly.py` | représentation polynomiale ajustée sur le seul nuage |
| `main_cloud.py` | chaîne complète depuis un fichier de nuage, sans aucune formule |
| `main_morph.py` | morphing et booléens, sur formules et sur champs appris |
| `main_curvature.py` | courbures du champ appris + contrôle Gauss-Bonnet |
| `main_temporal.py` | **surface espace-temps** : un champ `f(x,y,z,t)` appris depuis six instants, testé sur cinq instants jamais vus |
| `main_capsules.py` | **les vraies données du BMBI** : carte d'identité du dataset, et ce qu'il change |
| `main_shapes.py` | surfaces complexes : topologie mesurée, puis apprentissage |
| `main_siren.py` | comparaison Softplus / Fourier / SIREN, sans normales |
| `main_render.py` | rendu PyVista et exports OBJ / VTK / VTP / PVD |
| `validate_geometry.py` | 28 vérifications des outils géométriques sur des cas connus |
| `reconstruct_from_cache.py` | refait figure et métrique sans réentraîner |

## Les données du BMBI

Le dataset de capsules (`Données/Data_Capsule/`) est **propriété du laboratoire
BMBI et confidentiel** : il n'est pas versionné, et ne doit pas l'être. Les
scripts qui en dépendent (`main_capsules.py`) échouent proprement s'il est
absent, avec le rappel de la marche à suivre.

## Installation

```bash
pip install -r requirements.txt
```

## Lancer

```bash
python main_eikonal.py        # ou n'importe quel script du tableau ci-dessus
python validate_geometry.py   # à lancer après toute modification de la géométrie
```

Les sorties (figures PNG, modèles `.pt`, grilles `.npz`) vont dans `output/`,
ignoré par git.

## Journal des résultats

- `PLAN.md` — la feuille de route et l'avancement.
- `RESULTATS_courbures.md` — courbures et invariants : démarche, validations,
  résultat et limites.
- `RESULTATS_nuage.md` — apprendre depuis un nuage de points seul : fidélité du
  signe, dimensionnement, réglage de `k`, et ce qu'on perd (rien) à se passer
  de la formule.
- `RESULTATS_morphing.md` — morphing et booléens, avec la vérification
  quantitative, et le contrôle de la figure de courbure contre l'article.
- `RESULTATS_temporel.md` — la surface qui évolue dans le temps : protocole,
  test sur instants cachés, conservation du volume, suivi lagrangien, et les
  contrôles qui disent ce que ces chiffres ne prouvent pas.
- `RESULTATS_rendu_formes.md` — rendu PyVista et formats d'échange, surfaces
  complexes (le stress-test topologique), et la comparaison d'architectures
  avec SIREN.
- `RESULTATS_capsules.md` — **les vraies données du BMBI** : le maillage passe
  sans retouche, la déformation est prouvée non affine, 93 % du mouvement est
  tangentiel, et un contrôle du projet doit être réécrit.

## Notes

- **Reproductibilité.** Une graine fixe (`set_seed`) est posée, mais elle ne
  suffit PAS ici : sur GPU Apple (MPS), deux entraînements à graine identique
  divergent — vérifié en comparant l'empreinte des poids après 300 époques. Sur
  CPU, ils sont identiques au bit près. Conséquence pratique : `main_eikonal.py`
  donne un `mean|f_exact|` qui varie entre 0.0040 et 0.0079 d'un run à l'autre.
  Pour comparer deux variantes de code, forcer le CPU ou moyenner sur plusieurs
  runs — un écart isolé de cet ordre ne prouve rien.
- **Deux modèles du globule rouge coexistent**, volontairement :
  `implicit_biconcave` (profil d'Evans-Fung, défini par morceaux — c'est
  l'historique) et `implicit_rbc_quartic` (l'équation algébrique exacte de
  Kuchel et al. 2021, lisse partout). Le second est préférable dès qu'on touche
  aux dérivées : les dérivées secondes du premier divergent au bord
  (voir `RESULTATS_courbures.md`, section 6).
- **Comparer deux variantes se fait sur CPU, pas sur MPS**, et sur plusieurs
  graines. C'est ce que fait `main_cloud.py` : trois graines, l'écart-type entre
  graines servant d'étalon pour juger si un écart mesuré veut dire quelque chose.
- **Le morphing n'est pas invariant par changement d'échelle des champs.**
  `csg.normalized` divise chaque champ par la médiane de `‖∇f‖` sur sa surface
  avant de l'interpoler — sans quoi le chemin est tiré vers le champ de plus
  forte pente. La surface `f = 0` n'en est pas déplacée.
- **L'eikonal d'un champ espace-temps ne porte QUE sur le gradient d'espace.**
  Le réseau `f(x,y,z,t)` a quatre dérivées partielles, mais `‖∇f‖ = 1` décrit une
  distance signée DANS L'ESPACE. Appliquer la condition aux quatre composantes
  reviendrait à exiger `‖∇_x f‖² = 1 − (∂f/∂t)²`, c'est-à-dire à demander au champ
  d'être d'autant moins une distance signée qu'il bouge vite. D'où `grad[..., :3]`
  partout dans `src/train_temporal.py`.
- **Le terme eikonal se tire à `t` continu**, pas seulement aux instants observés.
  C'est la seule pièce de la perte qui contraint le champ entre deux instants ;
  sans elle, rien n'y impose quoi que ce soit et le test d'interpolation ne
  mesurerait que le hasard de l'initialisation.
- **Le Chamfer sature.** À la grille 96³, le marching cubes appliqué au champ
  EXACT donne déjà 0.0078 : en dessous, aucun réseau ne peut descendre. Un
  Chamfer de 0.008 signifie « aussi bon que la grille le permet » et ne peut pas
  départager deux méthodes proches. Pour comparer deux méthodes, prendre une
  mesure qui ne sature pas — le volume ou la vitesse normale.
- `main.py` enregistre un PNG au lieu de l'affichage interactif `%matplotlib
  ipympl` du notebook (pas besoin d'écran).
- Les figures de `output/` ne sont pas versionnées. Celles qui servent à la
  documentation sont recopiées dans `figures/`, qui l'est.
