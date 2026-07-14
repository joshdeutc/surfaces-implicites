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

| Étape | Objectif | Touche |
|------|----------|--------|
| **1** | **Apprendre SANS la formule** : surface + dedans/dehors + eikonal | `data.py`, `train.py`, `model.py` |
| 2 | Meilleures activations (SIREN / encodage de Fourier) | `model.py` |
| 3 | Formes complexes — globule rouge biconcave (test des limites) | `shapes.py` |
| 4 | Métriques quantitatives (Chamfer, IoU, erreur eikonal) | nouveau module |
| 5 | Robustesse : bruit, points épars, vue partielle | `data.py` |
| 6 | Famille de formes + code latent (lien réduction de modèle) | `model.py`, `train.py` |
| 7 | Spatio-temporel `f(x,y,z,t)` + suivi d'un point (objectif final) | tout |

Premier jalon visé : **1 + 2 + 3 + 4**.

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

## GPU / Colab
- Code rendu device-aware (src/device.py) ; notebook colab.ipynb prêt.
- Ce PC : pas de GPU CUDA (Intel UHD + torch CPU-only) → ~0,5-1 s/epoch,
  throttling thermique. Un T4 (Colab) ramènerait les runs de ~1 h à quelques min.
