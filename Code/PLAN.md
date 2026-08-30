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
| 2 | Meilleures activations (Fourier, SIREN) | `model.py`, `model_siren.py` | fait |
| 3 | Formes complexes — globule rouge biconcave (test des limites) | `shapes.py` | fait |
| 4 | Métriques quantitatives SANS la formule (Chamfer, IoU, eikonal) | `metrics.py` | fait |
| **5** | **Courbures et invariants géométriques** (article Kuchel 2021) | `curvature.py`, `invariants.py` | fait |
| **6** | **Représentation polynomiale** de `f` (compression, explicabilité) | `polyfit.py` | fait |
| **7** | **Dedans/dehors depuis un vrai nuage de points** (normales + produit scalaire) | `inside_outside.py`, `cloud_io.py` | fait |
| **8** | **Morphing `θf₁+(1−θ)f₂` + union / intersection / différence** | `csg.py` | fait |
| 9 | Robustesse : bruit, points épars, vue partielle | `data.py` | à faire |
| 10 | Famille de formes + code latent (lien réduction de modèle) | `model.py`, `train.py` | à faire |
| **11** | **Brancher les vraies données de capsules (sorties EF du BMBI)** | `capsules.py` | **données reçues et analysées** |
| **12** | **Spatio-temporel `f(x,y,z,t)` + suivi d'un point (objectif final)** | `temporal.py`, `model_temporal.py`, `train_temporal.py` | fait |
| 13 | Affichage PyVista au format VTK (nommé deux fois dans la fiche) | `viz_pv.py` | fait |
| **14** | **Apprendre `f(x,y,z,t)` SUR LA CAPSULE RÉELLE** | `capsules.py`, `main_capsules_temporal.py` | **fait** |
| **15** | **Combien d'instants faut-il ?** — les deux régimes de la séquence | `campagne.py`, `main_capsules_event.py`, `main_capsules_tanktread.py` | **fait** |

Les étapes 5 à 8 viennent des recommandations de F. De Vuyst (`Encadrement/`) et
du programme de la fiche d'enregistrement : les courbures (10 h prévues), la
piste polynomiale, le morphing, et la détermination réelle du dedans/dehors.

**26/08/2026 — l'étape 11 est débloquée.** F. De Vuyst a transmis le dataset de
capsules (1.2 Go, confidentiel, `Données/` exclu de git). L'analyse est faite et
documentée dans `RESULTATS_capsules.md` ; elle change l'ordre des priorités, voir
la section « Étape 14 » plus bas. Le reste du travail se fait désormais sur les
vraies données : c'est l'objet de la fiche d'enregistrement, et Florian évoque une
publication en cas de succès.

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

- [x] Étape 12 — **surface implicite espace-temps** (`src/temporal.py`,
      `src/model_temporal.py`, `src/train_temporal.py`, `main_temporal.py`).
      C'est le « Graal » du mail et la formulation de la fiche : condenser une
      forme mobile en un seul champ `f(x,y,z,t) = 0`. Protocole calqué sur ce que
      rendra le code EF du BMBI — six fichiers `.npz` écrits sur disque, aux
      instants t = 0, 0.2, 0.4, 0.6, 0.8, 1.0, et rien d'autre. Le globule
      quartique est étiré à volume constant, `A(t) = diag(ξ^-1/2, ξ^-1/2, ξ)`
      avec `det A = 1`, ξ de 1.0 à 1.5.
        * **test central — les instants jamais vus** (t = 0.1, 0.3, 0.5, 0.7, 0.9,
          aucun fichier fourni) : Chamfer **0.00815** contre 0.00814 aux instants
          observés, IoU **0.99103** contre 0.99058, `χ = 2` aux onze instants.
          Le réseau a condensé la séquence, il ne l'a pas mémorisée.
        * **volume**, invariant que l'entraînement ignore : dispersion de
          **0.041 %** sur toute la séquence (biais de −1.04 %, mais ce biais varie
          d'un run à l'autre sur MPS — il est passé de +0.79 % à −1.04 % entre
          deux entraînements de graine identique ; seule la dispersion est
          interprétable).
        * **suivi lagrangien** par la vitesse normale `v_n = −(∂f/∂t)/‖∇_x f‖`,
          lue directement sur le réseau : après 200 pas de RK4 le point reste à
          **0.0003** de la vraie membrane, et s'écarte de 0.0008 de la trajectoire
          obtenue avec le champ exact. Le champ exact, lui, dérive de 0.00000 —
          les dérivées analytiques et le schéma d'intégration sont donc hors de
          cause.
        * deux points de méthode. L'eikonal ne doit porter que sur le gradient
          d'ESPACE : appliqué aux quatre composantes il exigerait
          `‖∇_x f‖² = 1 − (∂f/∂t)²`, c'est-à-dire punirait le mouvement. Et le
          terme eikonal doit être tiré à `t` CONTINU : c'est la seule pièce de la
          perte qui contraint les instants intermédiaires, sans elle le test
          d'interpolation ne mesurerait que le hasard de l'initialisation.
        * **les contrôles nuancent le résultat, et c'est le plus instructif.**
          Le Chamfer est SATURÉ par la grille (marching cubes sur le champ exact
          donne déjà 0.00783 à 96³, le réseau 0.00815, soit 1.04×). Conséquence :
          sur la POSITION de la surface, le réseau 4D fait **jeu égal** avec un
          simple morphing entre les deux instants encadrants (0.00815 contre
          0.00806). Notre déformation est affine, donc trop favorable à cet
          étalon. Le gain réel est sur les DÉRIVÉES : erreur de **0.00134** sur la
          vitesse normale contre **0.02928** pour le morphing, soit un facteur
          **22**. Ce qu'apporte un champ `f(x,y,z,t)`, ce n'est pas de mieux
          placer les points, c'est de posséder un champ de vitesse dérivable
          partout.
        * étalon « image gelée » (rendre l'instant observé le plus proche) :
          0.01252, soit 1.53× l'erreur du réseau. L'amplitude totale du mouvement
          vaut 0.10655 : le réseau se trompe de 7.7 % de ce qu'il doit restituer.
      Zéro appel à la formule pendant la préparation des données ET pendant
      l'entraînement, vérifié par compteur. Détail dans `RESULTATS_temporel.md`.

      **Ce qu'il faudrait durcir.** La déformation est affine, monotone,
      axisymétrique, sans bruit, à topologie constante — le contrôle 5.3 montre
      que c'est trop facile. → **Réglé le 26/08/2026 sans avoir à le fabriquer :
      les vraies capsules sont non affines (15 % entre snapshots voisins) et
      non monotones. Voir l'étape 11.** Par ordre d'utilité : (1) déformation NON AFFINE
      (flexion, onde sur la membrane), qui casse l'étalon morphing et rendrait le
      test d'interpolation discriminant ; (2) mouvement non monotone
      (basculement, retour à une forme antérieure) ; (3) moins d'instants — la
      question pratique pour le BMBI : à quelle fréquence le code EF doit-il
      sauvegarder ? ; (4) bruit sur les points ; (5) changement de topologie.

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

- [x] Étape 13 — **rendu PyVista et formats d'échange** (`src/viz_pv.py`,
      `main_render.py`). Rendu hors écran avec caméra déterministe et caméras
      liées entre panneaux, champ scalaire aux sommets, export OBJ / VTK / VTP et
      série temporelle `.pvd` lisible par ParaView. Chaque export est **relu et
      recompté** : OBJ perd les champs scalaires, VTK et VTP les conservent.
      Le lissage est à zéro par défaut — il déplace les sommets, donc fausse
      toute aire, volume ou courbure mesurés après. Détail dans
      `RESULTATS_rendu_formes.md`, section 1.

- [x] Étape 3bis — **surfaces complexes**, le seul stress-test topologique du
      projet (`src/shapes_complexes.py`, `main_shapes.py`). Chmutov, gyroïde,
      Barth, metaballs, comme demandé par la fiche. Résultat inattendu :
      **deux des quatre surfaces ne sont pas fermées**, et tout le pipeline
      suppose un intérieur borné. Les metaballs donnent la démonstration la plus
      directe de l'argument du sujet — mêmes boules, seuil différent, `χ` passe
      de 0 à 2. Détail dans `RESULTATS_rendu_formes.md`, section 2.

- [x] Étape 2 — **SIREN**, la troisième architecture (`src/model_siren.py`,
      `main_siren.py`). Comparaison Softplus / Fourier / SIREN à ω₀ ∈ {10,15,30}
      sur le globule biconcave, sans normales. Détail dans
      `RESULTATS_rendu_formes.md`, section 3.

- [x] Étape 11 — **LES VRAIES DONNÉES DU BMBI** (`src/capsules.py`,
      `main_capsules.py`, `RESULTATS_capsules.md`). Dataset reçu le 26/08/2026 :
      une capsule dans un canal, maillage triangulaire de 2562 nœuds matériels,
      20001 pas de temps, 1.2 Go. **Confidentiel** : `Données/` est exclu de git.
        * **le maillage entre dans la chaîne sans retouche** : χ = 2, étanche,
          genre 0, volume signé positif donc normales déjà sortantes. Deux mois
          de travail « à l'aveugle » sur le bon format.
        * **la déformation est NON AFFINE, et prouvée telle.** 14.9 % de résidu
          en moyenne entre snapshots voisins (contre 0 % sur notre séquence
          synthétique). Et mieux qu'un pourcentage : la capsule part d'une sphère
          et **cesse d'être convexe** dès le pas 1500 (défaut de convexité 1.06 à
          1.61 %), or l'image affine d'un convexe est convexe. La limite n°1 du
          projet tombe, et l'étalon morphing part avec 15 % d'erreur.
        * **93 % du mouvement est TANGENTIEL** (tank-treading : la membrane
          tourne autour d'une forme qui bouge peu). **Conséquence de protocole :
          le bloc 4 de `RESULTATS_temporel.md` — advecter un point par `v_n` et
          le comparer à la trajectoire d'un nœud — NE SE TRANSPOSE PAS.** Une
          surface implicite ne porte que le mouvement normal, et c'est correct.
          Il faut réécrire ce contrôle : `v_n` appris contre `v_n` du maillage.
        * volume conservé à **0.0256 %**, aire étirée de 4.26 % : l'invariant
          gratuit est là aussi.
        * la capsule **migre** (centre de 0.400 à 0.046) : on sépare translation
          et déformation (`recentre=True`), la trajectoire du centre étant une
          fonction du temps à trois composantes, triviale à ajuster à part.
        * échantillonnage : 101 instants uniformes = 0.51 arête entre voisins,
          assez fin ; mais la déformation culmine au pas 800 et le reste est de
          la relaxation, donc `instants_transitoire` resserre le début.

---

## Étape 14 — `f(x,y,z,t)` sur la capsule réelle : fait

Campagne dans `main_capsules_temporal.py`, journal dans
`RESULTATS_capsules_temporel.md`. Fenêtre sur le transitoire (pas 0 à 4000),
21 instants, un sur deux caché, deux échantillonnages comparés.

- [x] **Le contrôle lagrangien a été réécrit**, comme le tank-treading l'exigeait :
      `v_n` appris contre `v_n` du MAILLAGE (vitesse des nœuds projetée sur la
      normale), jamais contre une trajectoire de nœud.
- [x] **Et il a fallu le réécrire une deuxième fois.** `v_n` du maillage est une
      différence centrée : à un écart de 200 pas elle est fausse de 3 à 12 %.
      Ramenée à 25 pas, où elle est stationnaire. Mesurer un champ appris contre
      une référence fausse de 10 % n'aurait rien dit.
- [x] **Vérité de référence sans formule** : `capsules.champ_reference`, distance
      au plus proche point d'un nuage dense tiré sur les triangles, signe par
      produit scalaire. Validée contre le test d'appartenance exact de trimesh :
      **100.000 % d'accord** sur 20000 points. Elle donne aussi le PLANCHER du
      Chamfer (marching cubes dessus : 0.00463 à la grille 96³).

**Le résultat central.** Échantillonnage uniforme (un instant tous les 400 pas) :
aux dix instants cachés, Chamfer 0.00500 contre 0.00533 au morphing et 0.00799 à
l'image gelée. **Le réseau bat enfin le morphing** — ce que la séquence
synthétique, affine, ne pouvait pas montrer. En excédent sur le plancher, la
seule lecture honnête d'une métrique saturée : 1.9× mieux que le morphing.

**Le résultat le plus utile.** À nombre d'instants ÉGAL, resserrer
l'échantillonnage au début (là où la sphère se déforme) change tout :

| aux instants cachés | uniforme | transitoire |
|---|---:|---:|
| Chamfer | 0.00500 | **0.00464** |
| excédent sur le plancher | 0.00036 (7.8 % du plancher) | **0.0000074 (0.16 %)** |
| IoU voxel | 99.251 % | **99.650 %** |
| erreur sur `v_n` | 18.0 % de l'amplitude | **4.6 %** |

Avec le bon échantillonnage, **le réseau est au plancher de la grille aux
instants qu'il n'a jamais vus**, et observés / cachés sont indiscernables
(0.004634 contre 0.004639). Le morphing, lui, reste à 5.2 % du plancher.

**Ce qui est vraiment appris ici** : le placement des instants compte plus que
leur nombre. C'est la réponse chiffrée à la question pratique du BMBI —
sauvegarder proportionnellement à la vitesse de déformation, pas à intervalle
constant. En uniforme, le réseau PERD contre le morphing au premier instant caché
(pas 200, où l'amplitude de `v_n` vaut 8× le reste), parce qu'il n'a que deux
photos pour couvrir la partie violente du mouvement.

**Réserve.** La dispersion du volume n'est pas reproductible sur MPS : trois
exécutions identiques du mode uniforme ont donné 0.055 %, 0.221 % et 0.462 %,
alors que le Chamfer restait à 0.00500 aux trois. Seul son ordre de grandeur est
interprétable.

---

## Étape 15 — combien d'instants faut-il ? : fait

Deux campagnes à protocole strictement identique (`src/campagne.py`), une par
régime. Journal dans `RESULTATS_compression.md`.

| aux instants cachés | 11 obs. | 6 obs. | 5 obs. | 3 obs. |
|---|---:|---:|---:|---:|
| **événement** (pas 0-2000) — erreur sur `v_n` | 6.3 % | 15.8 % | 24.3 % | **73.8 %** |
| **événement** — excédent, × morphing | 1.3× | **2.4×** | 2.1× | 1.1× |
| **tank-treading** (pas 4000-20000) — erreur sur `v_n` | 26.0 % | 68.5 % | 78.1 % | 87.5 % |
| **tank-treading** — excédent, × morphing | 2.0× | **0.9×** | 0.9× | 1.0× |

Quatre choses en sortent, dont deux qui contredisent ce qu'on attendait.

**Il existe un seuil net.** Sous 5 instants sur l'événement, l'erreur de vitesse
passe à 74 % : le champ ne dit plus rien d'exploitable sur le mouvement. La
topologie, elle, reste correcte (`χ = 2`) même là — un invariant topologique juste
ne garantit donc rien sur la géométrie.

**L'avantage du réseau est maximal au MILIEU.** À 11 instants tout le monde est
au plancher, à 3 personne n'y arrive ; c'est à 6 que l'écart se voit (2.4×).

**La prédiction sur le tank-treading était fausse.** On attendait que l'implicite
y brille, puisque 93 % du mouvement y est tangentiel. En fait c'est le régime où
la forme dérive le plus lentement et le plus linéairement, donc celui où
l'interpolation linéaire est la meilleure : en dessous de 11 instants le réseau
fait jeu égal, voire un peu moins bien.

**Le réseau n'est pas un compresseur.** 773 Ko de poids contre 661 Ko pour les 11
maillages qu'on lui donne : il coûte 1.2 à 4.3 fois son entrée. Ce qu'il achète
est de la continuité — valeur, dérivée, normale, courbure en tout point de
l'espace-temps — pas de la place. Ne pas le présenter comme un argument de
compression.

**Défaut de cadrage assumé.** L'étalon morphing interpole entre deux formes, donc
lui non plus ne stocke pas de trajectoires. Ce protocole n'oppose PAS l'implicite
au maillage lagrangien : il oppose un champ neuronal à une interpolation linéaire
de formes. L'argument sur le tank-treading reste vrai par construction, mais il
est structurel et non mesuré ici.

**La réponse au BMBI** : sauvegarder quand la membrane a bougé d'environ une arête
de maillage — soit 5 à 6 instants sur la déformation rapide, 11 et plus sur la
relaxation. Donc un échantillonnage non uniforme.

---

## Ce qui reste, par ordre d'utilité

1. **Mesurer vraiment l'argument implicite contre maillage** : il faudrait un
   troisième étalon, une séquence de maillages lagrangiens jugée sur sa capacité à
   restituer la forme. C'est le seul moyen de sortir l'argument du registre
   structurel. C'est désormais la piste la plus utile.
2. **L'extrapolation** : tous les instants cachés testés sont ENCADRÉS par des
   instants observés. Rien n'a été demandé au-delà du dernier.
3. **Étape 9 — robustesse** : bruit sur les points, vues partielles. Les capsules
   sont une simulation, donc sans bruit de mesure.
4. **Étape 10 — code latent** : une famille de capsules plutôt qu'une seule.
5. **Moyenner sur plusieurs graines**, sur CPU, pour que la dispersion du volume
   devienne interprétable.
6. **Le rapport** (40 h prévues à la fiche) — pas commencé. Les six
   `RESULTATS_*.md` en sont la matière.

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
