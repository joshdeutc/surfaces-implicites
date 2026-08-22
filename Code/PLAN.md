# Plan — Surfaces implicites apprises par réseau de neurones (TZ)

Représenter une forme fermée 3D par une fonction implicite `f(x,y,z) = 0`
apprise par réseau de neurones, reconstruite par *marching cubes*.
Application : microcapsules synthétiques (labo BMBI).

## Données (cadre annoncé par F. De Vuyst)

> « échantillonnage de la surface + points intérieurs/extérieurs pour discriminer l'intérieur »

Donc, en réalité, on dispose de :
- **points de surface** (`f = 0`),
- **points intérieurs/extérieurs** (le *signe* de `f` : `< 0` dedans, `> 0` dehors).

On ne dispose **PAS** de la valeur exacte de `f` (c'est la « triche » de la démo
initiale, qui sert seulement à valider la machinerie).

## Convention

`f < 0` intérieur · `f = 0` surface · `f > 0` extérieur.

---

## Feuille de route

| Étape | Objectif | Touche | État |
|------|----------|--------|------|
| **1** | **Apprendre SANS la formule** : surface + dedans/dehors + eikonal | `data.py`, `train.py`, `model.py` | fait |
| 2 | Meilleures activations (Fourier ; SIREN reste à faire) | `model.py` | partiel |
| 3 | Formes complexes — globule rouge biconcave (test des limites) | `shapes.py` | fait |
| 4 | Métriques quantitatives SANS la formule (Chamfer, IoU, eikonal) | `metrics.py` | fait |
| **5** | **Courbures et invariants géométriques** (article Kuchel 2021) | `curvature.py`, `invariants.py` | fait |
| **6** | **Représentation polynomiale** de `f` (compression, explicabilité) | `polyfit.py` | fait |
| **7** | **Dedans/dehors depuis un vrai nuage de points** (normales + produit scalaire) | `inside_outside.py`, `cloud_io.py` | fait |
| **8** | **Morphing `θf₁+(1−θ)f₂` + union / intersection / différence** | `csg.py` | fait |
| 9 | Robustesse : bruit, points épars, vue partielle | `data.py` | à faire |
| 10 | Famille de formes + code latent (lien réduction de modèle) | `model.py`, `train.py` | à faire |
| 11 | Brancher les vraies données de capsules (sorties EF du BMBI) | nouveau module | bloqué |
| 12 | Spatio-temporel `f(x,y,z,t)` + suivi d'un point (objectif final) | tout | à faire |

Les étapes 5 à 8 viennent des recommandations de F. De Vuyst (`Encadrement/`) et
du programme de la fiche d'enregistrement : les courbures (10 h prévues), la
piste polynomiale, le morphing, et la détermination réelle du dedans/dehors.

---

## Étape 1 — détail (en cours)

### Idée
Remplacer la régression de la valeur exacte de `f` (MSE sur `y_vol`) par un
apprentissage à partir des seules informations réalistes :

Perte = somme de termes (méthode IGR, Gropp et al. 2020) :
- **surface** : `|f|` ≈ 0 sur les points de peau ;
- **normales** (option, souvent estimables) : `∇f` aligné avec la normale en surface ;
- **eikonal** : `(‖∇f‖ − 1)²` sur des points tirés au hasard → force une vraie distance ;
- **signe** : pénalise `f` du mauvais signe sur les points dedans/dehors.

### Réseau adapté
- **Initialisation géométrique** (le réseau ≈ une sphère à l'init) → indispensable
  pour que l'entraînement eikonal converge bien.
- **Activation Softplus** (gradients lisses, requis par le terme eikonal).
- **Skip-connection** de l'entrée vers le milieu du réseau.

### Critère de réussite
- Le programme tourne de bout en bout.
- L'ellipsoïde est reconstruit **sans jamais utiliser la valeur exacte de `f`**.
- Erreur moyenne de reconstruction faible : `mean |f_exact(sommets reconstruits)|`
  petite (objectif < 0.03, l'ordre de grandeur de la maille de la grille).

### Fichiers
- `main.py` : version « triche » d'origine (référence / comparaison).
- `main_eikonal.py` : version réaliste étape 1.

---

## Statut

- [x] Découpage du notebook en modules `.py` (résultats identiques au notebook).
- [x] Étape 1 — version réaliste (surface + dedans/dehors + eikonal).
      → ellipsoïde reconstruit SANS la formule, erreur mean|f_exact| = 0.0041
      (≪ maille grille 0.05) → reconstruction sous-voxel. Réseau : IGRNet
      (init géométrique + Softplus). Cache modèle/grille ajouté.
- [x] Étape 3 — biconcave (globule rouge). Softplus + normales : mean|f|=0.0132,
      fossette bien capturée (la prévision "Softplus arrondit" était FAUSSE :
      les normales suffisent).
- [x] Étape 2 — comparaison Softplus vs Fourier, SANS normales (données brutes
      du mail). RÉSULTAT CONTRE-INTUITIF :
        * Softplus sans normales : mean|f|=0.0314 (propre)
        * Fourier sans normales  : mean|f|=0.1534, 108k sommets (BRUIT)
      → Fourier naïf DÉGRADE (rides/zéros parasites ; eikonal mal dompté).
      Leçons : (1) les normales comptent plus que l'activation ;
      (2) recette robuste = Softplus + eikonal (+ normales) ;
      (3) Fourier/SIREN exigent une régularisation soignée (moins de fréquences,
      poids eikonal plus fort, init adaptée) — pas plug-and-play.

- [x] Étape 4 — **métriques sans la formule** (`src/metrics.py`). Chamfer
      symétrique + Hausdorff, IoU d'occupation par Monte-Carlo (protocole
      Occupancy Networks), cohérence des normales, erreur eikonal. Aucune ne
      demande la valeur exacte de `f` : toutes sont transposables aux vraies
      capsules. Validées sur des cas dont la réponse est connue d'avance
      (IoU sphère R=1 contre R=1.1 = 0.751 = rapport des volumes).

- [x] Étape 5 — **courbures et invariants** (`src/curvature.py`,
      `src/invariants.py`, `main_curvature.py`, `validate_geometry.py`).
      Formules de Goldman/Kuchel appliquées au champ appris par autograd
      d'ordre 2. Outils validés par 28 vérifications, dont le tore
      (`χ = 0` et `∫K dA = 0` alors que `∫|K| dA = 25.1`).
      RÉSULTAT : le réseau est juste à l'ordre 2 **globalement** —
      `∫K dA = 12.5633` contre `4π = 12.5664`, soit 0.02 % d'écart — mais il
      **lisse les extrêmes** : localement l'erreur sur K_gauss est de 7.5 % de
      l'amplitude, et au centre du creux il annonce 12.1 au lieu de 14.4.
      Corrélation avec la courbure exacte : 0.996. Détail dans
      `RESULTATS_courbures.md`.

- [x] Étape 6 — **représentation polynomiale** (`src/polyfit.py`, `main_poly.py`).
      Ajustement algébrique par SVD sur le seul nuage de surface, sans réseau et
      sans la formule. Résultats :
        * ellipsoïde, degré 2 → 10 coefficients dont 4 non nuls, IoU = 1.0000
        * globule rouge quartique, degré 4 → 10 coefficients non nuls,
          IoU = 1.0000, coefficients retrouvés à 5e-7 près des vrais P, Q, R
        * globule d'Evans-Fung, degré 4 → IoU = 0.666 ; il faut le degré 6
      La conclusion antérieure « le globule demande le degré 6 » était un
      artefact du profil d'Evans-Fung, qui n'est pas algébrique. Le vrai globule
      rouge **est** une quartique : 10 coefficients contre 197 886 poids pour
      IGRNet, soit un facteur 19 789.

- [x] Étape 7 — **apprendre depuis un nuage de points seul** (`src/inside_outside.py`,
      `src/cloud_io.py`, `main_cloud.py`). C'est le mode qui s'appliquera aux
      vraies capsules : un maillage Éléments Finis ne vient pas avec une formule.
      Le signe dedans/dehors est déduit du point de surface le plus proche et du
      produit scalaire avec sa normale (méthode du mail de F. De Vuyst).
      Le faux jeu de données passe par un vrai maillage triangulaire dont les
      normales sont calculées par triangle puis orientées vers l'extérieur par
      le volume signé.
        * accord du signe hors de la coquille `|f| > 0.05` : **100 %** sur
          l'ellipsoïde comme sur le globule quartique
        * dimensionnement : 500 points → 99.72 % d'accord global, 6000 → 99.98 %,
          20000 → 99.99 %. À l'échelle du globule rouge (8 µm de diamètre), un
          espacement de nuage de 0.078 µm suffit.
        * effet du vote sur `k` voisins — **l'hypothèse de départ était fausse** :
          sur nuage propre `k = 1` est le meilleur partout (99.88 % dans le creux
          contre 99.41 % à `k = 15`), les voisins lointains venant de la cuvette
          opposée votent à contresens. L'ordre ne s'inverse que sur nuage bruité
          (96.57 % à `k = 1` contre 98.70 % à `k = 15`). `k` règle la robustesse
          au bruit, pas la géométrie. Retenu : `k = 5` pour des données mesurées.
        * **perd-on quelque chose à se passer de la formule ? Non.** Trois graines,
          sur CPU pour être reproductible : Chamfer 0.01593 ± 0.00113 (nuage)
          contre 0.01582 ± 0.00056 (formule), IoU 0.96876 contre 0.96898 — tous
          les écarts sont plus petits que la dispersion entre graines. Le
          mécanisme : les deux étiquetages ne diffèrent que sur 4 points sur
          20 000. Ce n'est donc pas une preuve de robustesse à un étiquetage
          dégradé, question non posée ici.
      Zéro appel à la formule pendant l'entraînement, vérifié par un compteur.
      Détail dans `RESULTATS_nuage.md`.

- [x] Étape 8 — **morphing et opérations booléennes** (`src/csg.py`, `main_morph.py`).
      Recommandation du mail : `θ·f₁ + (1−θ)·f₂` pour interpoler, `min` / `max` /
      `max(f₁,−f₂)` pour union, intersection et différence. Le module marche à
      l'identique sur des formules et sur des champs appris.
        * piège traité : le morphing n'est pas invariant par changement d'échelle
          des champs. La quartique du globule a une pente médiane de 2.12 sur sa
          surface contre 1.00 pour la sphère ; sans correction le chemin serait
          tiré d'un facteur deux. Chaque champ est donc divisé par la médiane de
          `‖∇f‖` sur sa surface, ce qui ne déplace pas `f = 0`.
        * booléens vérifiés quantitativement : volumes confrontés aux valeurs
          analytiques de la lentille sphérique (écart < 0.13 %), et identité
          d'inclusion-exclusion `V(A∪B) + V(A∩B) = V(A) + V(B)` à **1.1e-06 %**.
          Cette identité étant symétrique, elle ne prouve pas à elle seule que
          `min` est l'union : c'est l'encadrement `V(A∩B) < V(A) < V(A∪B)` qui
          tranche.
        * morphing entre deux réseaux : `χ = 2` conservé, sphéricité monotone.
          Réserve de protocole — au bloc analytique la sphère a le même volume
          que le globule pour que la variation de volume mesure quelque chose ;
          ce n'est pas le cas au bloc « champs appris », où la décroissance du
          volume est en partie imposée par les extrémités.
      Détail dans `RESULTATS_morphing.md`.

- [x] Contrôle de la figure de courbure. La distribution de `K_gauss` semblait
      contredire l'article de Kuchel. Vérification faite aux dimensions de
      l'article : aire 128.004 µm² contre 128 annoncés, volume 85.947 µm³ contre
      86 fL — la forme est la bonne. Et pour une surface de révolution,
      `K_G = z'z''/(r(1+z'²)²)` : au creux `z' = 0` et `z'' > 0` donc `K_G` y est
      MAXIMALE, à la crête d'épaisseur `z' = 0` donc `K_G = 0`, et entre les deux
      `K_G < 0`. Un disque biconcave doit avoir un anneau de courbure négative.
      Pas d'erreur de calcul : l'écart apparent venait des unités (coordonnées
      réduites contre micromètres, facteur 16). Figure comparable à l'article :
      `output/courbure_globule_micrometres.png`.

### Défaut trouvé et corrigé
La formule `implicit_biconcave` (profil d'Evans-Fung, `np.where(r < 1, ...)`) a
des dérivées secondes qui divergent au bord : Gauss-Bonnet y donne −2.6e8 au lieu
de 4π. Ce n'est pas un échec de la méthode mais de la formule. La quartique de
Kuchel, lisse partout, ramène l'intégrale à 12.5586. C'est l'argument le plus
direct pour lui substituer une forme algébrique.

### Reproductibilité — attention
La graine fixe ne suffit pas sur GPU Apple : deux entraînements identiques
divergent sur MPS (vérifié sur l'empreinte des poids), alors qu'ils sont
identiques au bit près sur CPU. `mean|f_exact|` varie entre 0.0040 et 0.0079
d'un run à l'autre sur `main_eikonal.py`. Un écart isolé de cet ordre ne prouve
rien : forcer le CPU ou moyenner sur plusieurs runs pour comparer deux variantes.

## Matériel
- Code device-aware (`src/device.py`) : CUDA, sinon MPS (Apple), sinon CPU.
- Machine actuelle : MacBook M2, MPS actif. Un entraînement complet prend 20 à
  80 s ; l'ensemble des scripts tourne en quelques minutes. Le notebook
  `colab.ipynb` reste disponible pour un GPU distant.
