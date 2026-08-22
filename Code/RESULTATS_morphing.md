# Morphing, opérations booléennes, et contrôle de la figure de courbure

Journal de travail. Tout ce qui est écrit ici est reproductible par les deux
commandes de la dernière section.

---

## 1. La question

Le mail de recommandations de F. De Vuyst
(`Encadrement/mail_recommandations_surfaces-implicites.md`) se termine par un
paragraphe qui n'avait pas encore été traité :

> « les surfaces implicites sont très pratiques pour faire du morphing de
> formes : pour une forme f1 = 0 et une autre f2 = 0, une forme interpolée sera
> donnée par theta·f1 + (1−theta)·f2 = 0. On peut aussi définir des opérations
> d'union, d'intersection ou de différences de formes, très facilement.
> **D'où toute la puissance des surfaces implicites.** »

C'est l'argument central en faveur de la représentation implicite. Avec un
maillage, fusionner deux formes demande un algorithme de découpe de triangles ;
avec une fonction implicite, une union est un `min()`. Encore faut-il le
montrer, et le montrer **chiffré** : une image de deux sphères fusionnées est
jolie mais ne prouve rien.

Deux exigences ont donc guidé le travail :

1. les opérations doivent être **vérifiées par des nombres**, pas par des
   images ;
2. elles doivent marcher **à l'identique sur un champ appris par réseau**,
   puisque c'est le seul cas qui compte pour la suite du projet — sur des
   capsules mesurées au labo, aucune formule n'existera.

---

## 2. Ce qui a été ajouté

| Fichier | Rôle |
|---|---|
| `src/csg.py` | le module : `morph`, `union`, `intersection`, `difference`, `complement`, plus `from_model` (la passerelle réseau → champ) et `normalized` |
| `main_morph.py` | les trois expériences et leurs contrôles |
| `main_curvature.py`, section 4 | contrôle d'échelle du globule en micromètres, face à l'article de Kuchel |

Aucun fichier existant n'a été modifié, hormis l'ajout de la section 4 à
`main_curvature.py`.

Le point de conception : **tout le module manipule des champs**, c'est-à-dire
des fonctions `(N,3) -> (N,)`. Il ne sait jamais d'où vient le champ. Passer
d'une formule à un réseau, c'est remplacer

```python
f_globule = implicit_rbc_unit          # une formule
```
par
```python
f_globule = from_model(reseau)         # un réseau
```

et rien d'autre ne change. C'est ce qui rend la démonstration du bloc 3
convaincante : ce n'est pas un second programme écrit pour l'occasion, c'est
le même appelé sur autre chose.

### Un piège traité : la normalisation des champs

`theta·f1 + (1−theta)·f2` n'est pas invariant par changement d'échelle des
champs. Remplacer `f1` par `10·f1` décrit exactement la même surface, mais
déplace toutes les formes intermédiaires — `theta = 0.5` donnerait alors ce que
`theta ≈ 0.09` donnait avant. Seuls `theta = 0` et `theta = 1` sont invariants.

Or la quartique du globule a une pente médiane de **2.1186** sur sa propre
surface, contre **1.0000** pour la distance signée de la sphère. Sans
correction, le morphing serait tiré vers le globule d'un facteur deux.
`normalized(f, points)` divise donc chaque champ par la médiane de `‖∇f‖`
mesurée sur sa surface : les deux champs varient ensuite d'environ 1 par unité
de longueur, comme le ferait une distance signée. La surface `f = 0` est
évidemment inchangée — diviser par une constante positive ne déplace aucun
point.

Médiane et non moyenne : sur une quartique, `‖∇f‖` varie d'un facteur dix entre
le creux et le bord, et la moyenne serait tirée par le bord.

---

## 3. Résultat 1 — morphing sphère → globule (champs analytiques)

Sphère de **même volume** que le globule (rayon 0.6844 en coordonnées
réduites), de sorte que la variation de volume le long du chemin mesure
quelque chose au lieu d'être imposée par les extrémités.

| theta | forme | chi | volume | aire | sphéricité |
|---|---|---|---|---|---|
| 0.00 | sphère (départ) | 2 | 1.34226 | 5.88489 | 0.99993 |
| 0.25 | intermédiaire | 2 | 1.18882 | 5.51764 | 0.98358 |
| 0.50 | intermédiaire | 2 | 1.15369 | 5.72844 | 0.92863 |
| 0.75 | intermédiaire | 2 | 1.23106 | 6.64395 | 0.83607 |
| 1.00 | globule (arrivée) | 2 | 1.34189 | 7.99761 | 0.73565 |

Trois lectures :

- **chi = 2 tout du long.** Le morphing linéaire ne déchire ni ne perce la
  surface. Ce n'est pas garanti en général — deux formes éloignées peuvent
  donner un intermédiaire à deux composantes, ou percé.
- **La sphéricité décroît de façon monotone** (0.99993 → 0.73565). C'est la
  signature d'un passage progressif du plus compact au plus aplati. Le 0.99993
  au départ est au passage un contrôle du maillage : une vraie sphère vaut
  exactement 1.
- **Le volume n'est PAS monotone** : il descend à 1.154 à mi-chemin avant de
  remonter à sa valeur de départ. Les deux extrémités ont été choisies de même
  volume, mais rien dans la formule n'impose que le chemin le conserve.
  À retenir si l'on veut un jour un morphing à volume constant — une capsule
  qui se déforme conserve son volume, elle : c'est une contrainte à ajouter,
  pas une propriété acquise.

Figure : `output/morph_sphere_globule.png`.

---

## 4. Résultat 2 — opérations booléennes, avec vérification quantitative

Deux sphères de rayon 1, centres distants de 0.4, même grille 160³ pour les
cinq maillages.

| opération | chi | volume | aire |
|---|---|---|---|
| A | 2 | 4.187962 | 12.565060 |
| B | 2 | 4.187962 | 12.565060 |
| A ∪ B = min(fA, fB) | 2 | 5.427791 | 15.076076 |
| A ∩ B = max(fA, fB) | 2 | 2.948133 | 10.049414 |
| A \ B = max(fA, −fB) | 2 | 1.238279 | 12.332171 |

Contrôle préalable : le volume de A seul vaut 4.187962 contre 4πR³/3 =
4.188790, soit **−0.0198 %**. C'est le plancher d'erreur de la grille ; aucun
des contrôles suivants ne peut faire mieux.

### A. Confrontation aux volumes exacts

La lentille d'intersection de deux sphères égales a un volume analytique connu,
`V = π(4R + d)(2R − d)² / 12`. Les trois opérations ont donc chacune une
vérité de référence.

| opération | mesuré | exact | écart |
|---|---|---|---|
| A ∩ B | 2.948133 | 2.948908 | −0.0263 % |
| A ∪ B | 5.427791 | 5.428672 | −0.0162 % |
| A \ B | 1.238279 | 1.239882 | −0.1293 % |

### B. Identité d'inclusion-exclusion

    V(A∪B) + V(A∩B) = 5.427791 + 2.948133 = 8.37592453
    V(A)   + V(B)   = 4.187962 + 4.187962 = 8.37592444
    écart relatif = 1.12e-06 %

C'est une identité **ensembliste**, vraie pour n'importe quels ensembles :
l'écart mesuré ne peut venir que de la grille, jamais des formules. Il est ici
mille fois plus petit que l'erreur de discrétisation d'un volume isolé
(0.02 %), et ce n'est pas un hasard : dans chaque cellule de la grille, `min`
et `max` se partagent exactement les mêmes morceaux d'interface que A et B, si
bien que les erreurs se compensent case par case.

### C. Contrôle secondaire, et pourquoi il est nécessaire

    V(A\B) = 1.238279   contre V(A) − V(A∩B) = 1.239829   soit 0.1250 %

Ce contrôle n'est pas décoratif. **L'identité B est symétrique** : échanger
union et intersection la laisse vraie. Elle ne peut donc pas, à elle seule,
prouver que `min` est bien l'union et non l'intersection. Ce sont le tableau A,
ce contrôle C et l'encadrement ci-dessous qui tranchent :

    V(A∩B) = 2.948133  <  V(A) = 4.187962 = V(B)  <  V(A∪B) = 5.427791     VRAI

Recouvrement : l'intersection représente 70.40 % de chaque sphère.

Figure : `output/csg_booleens.png`.

---

## 5. Résultat 3 — le même morphing entre deux champs APPRIS

Deux réseaux IGR, l'un entraîné sur une sphère, l'autre sur le globule. Aucune
formule n'intervient. Le code appelé est celui du bloc 1, à l'identique.

Contrôle au passage : la médiane de `‖∇f‖` sur la surface vaut **0.9957** pour
le réseau du globule et **0.9992** pour celui de la sphère. C'est une mesure
directe de la qualité du terme eikonal — les réseaux ont bien appris une
fonction de type distance signée, sans qu'on le leur ait imposé point par
point.

| theta | chi | volume | aire | sphéricité |
|---|---|---|---|---|
| 0.00 | 2 | 4.19567 | 12.58063 | 0.99996 |
| 0.25 | 2 | 3.97448 | 12.21097 | 0.99370 |
| 0.50 | 2 | 3.73438 | 12.03245 | 0.96741 |
| 0.75 | 2 | 3.46179 | 12.21518 | 0.90598 |
| 1.00 | 2 | 3.17336 | 12.80749 | 0.81539 |

Même comportement qu'au bloc 1 : chi conservé, sphéricité monotone. C'est
l'argument de fond du lot — **le morphing ne demande pas de connaître
l'équation des formes.** Sur des capsules mesurées au BMBI, il n'y aura pas
d'équation ; il y aura deux réseaux entraînés sur deux nuages de points, et ce
même code s'appliquera.

Figure : `output/morph_reseaux.png`.

---

## 6. Verdict sur la figure de courbure

La question posée était : `output/courbure_globule.png` semble montrer
l'inverse de la figure 3a de Kuchel et al. 2021, qui placerait les valeurs de
courbure gaussienne les plus faibles au creux central. L'écart vient-il des
proportions de la forme, de l'échelle, ou d'une erreur de notre côté ?

L'analyse a été refaite sur `implicit_rbc_quartic` **aux dimensions de
l'article** (d = 8 µm, b = 1 µm, h = 2.12 µm), avec une hessienne analytique
exacte au lieu de différences finies (`main_curvature.py`, section 4).

**a) Les proportions ne sont pas en cause.** L'article donne deux valeurs
indépendantes pour son modèle, et nous les retrouvons :

| grandeur | mesuré | article | écart |
|---|---|---|---|
| aire | 128.004 µm² | 128 µm² | +0.003 % |
| volume | 85.947 µm³ | 86 fL | −0.062 % |

**b) Le calcul n'est pas en cause.** Gauss-Bonnet donne ∫K dA = 12.5637 contre
4π = 12.5664, soit **0.0215 %**. Et la hessienne analytique exacte s'accorde
avec les différences finies à **8.4e−08** près, pour une amplitude de 0.383 :
deux voies indépendantes, même résultat.

**c) L'échelle explique l'écart apparent de valeurs.** Nos sections
précédentes travaillent en coordonnées réduites (rayon 1), où les courbures
sont L² = 16 fois plus grandes qu'en micromètres. Ramenées aux bonnes unités,
nos valeurs vont de **−0.028 à +0.383 µm⁻²**, contre un axe allant de ~0 à
0.355 µm⁻² sur la figure 3a. Les deux bornes coïncident à mieux qu'une largeur
de classe : l'article découpe sa plage en dix classes et n'affiche que la
moyenne de chacune, si bien que sa première classe tombe vers 0 même quand le
minimum est légèrement négatif.

La répartition de l'aire par classe se superpose d'ailleurs à son histogramme :
première classe 0.333 de l'aire totale (l'article en montre ~0.31), décroissance
jusqu'à ~0.07, remontée à 0.136 vers K_G ≈ 0.28, puis chute à ~0.

**d) Sur le fond, c'est la lecture de la figure de l'article qui était
inversée.** Le profil radial mesuré est sans ambiguïté :

| zone | r (µm) | K_gauss médian |
|---|---|---|
| centre du creux | 0.0 – 0.3 | **+0.3356** |
| flanc du creux | 0.9 – 1.1 | +0.0013 |
| face plate (anneau) | 1.9 – 2.1 | **−0.0240** |
| crête d'épaisseur | 2.7 – 2.9 | +0.0207 |
| approche du bord | 3.4 – 3.6 | +0.1158 |
| bord (équateur) | 3.9 – 4.0 | **+0.2797** |

Le creux central et le bord portent les valeurs **les plus fortes** ; les plus
faibles, seules négatives, sont sur l'anneau plat entre les deux. C'est
exactement ce que montre la figure 3a : sa légende « Range Key: Low to High »
va du rouge au violet, et sur son image le creux central est violet tandis que
la face plate est rouge. La phrase de l'article sur « l'augmentation des zones
rouges, donc des valeurs basses » (figure 6) porte sur la face plate, pas sur
le creux.

C'est aussi ce que dit la géométrie, sans aucun calcul : pour une surface de
révolution de profil z(r), `K_G = z'z'' / (r(1+z'²)²)`. Au centre du creux
z' = 0 et z'' > 0, donc K_G = z''(0)² > 0 et maximal ; à la crête d'épaisseur
z' = 0 à nouveau, donc K_G = 0 exactement ; entre les deux z' > 0 et z'' < 0,
donc K_G < 0. Un disque biconcave **doit** avoir un anneau de courbure
gaussienne négative, et la valeur maximale **doit** être au creux ou au bord.

> **Verdict : pas d'erreur de notre côté, et pas de désaccord avec l'article.**
> Le seul écart réel tenait aux unités. `output/courbure_globule.png` reste
> juste ; elle est simplement tracée en coordonnées réduites. La figure
> `output/courbure_globule_micrometres.png` en donne la version comparable à
> l'article.

---

## 7. Ce que ça apporte au projet

1. **La recommandation (d) du mail est traitée**, et pas seulement illustrée :
   chaque opération est confrontée à un volume exact.
2. **Le lien avec l'article de Kuchel est chiffré** sur trois grandeurs
   indépendantes (aire, volume, plage de courbure), ce qui valide d'un coup la
   forme, l'échelle et la chaîne de calcul différentiel.
3. **Le morphing marche sur des champs appris**, donc il marchera sur les
   capsules du BMBI. C'est la première brique concrète vers les surfaces qui
   évoluent dans le temps — le « Graal » du mail : une déformation temporelle
   n'est rien d'autre qu'un morphing dont theta est le temps.
4. **Un contrôle secondaire de la qualité de l'eikonal** est tombé au passage :
   `‖∇f‖` médian de 0.9957 et 0.9992 sur la surface des deux réseaux.
5. **Une limite est documentée plutôt que cachée** : le morphing linéaire ne
   conserve pas le volume (1.342 → 1.154 → 1.342), alors qu'une capsule
   réelle, si.

## 8. Pistes ouvertes

- **Morphing à volume constant** : reparamétrer theta pour suivre une
  trajectoire à volume conservé, ce que fait une membrane réelle.
- **theta = t** : rejouer le morphing comme une animation temporelle et
  mesurer la courbure à chaque instant. C'est précisément l'expérience de
  Kuchel (figures 6 et 7 : courbure d'un globule progressivement étiré), mais
  sur un champ appris.
- **Congés d'arêtes** : `min`/`max` créent une arête vive à l'intersection, où
  les dérivées secondes n'existent pas — même défaut que la formule
  `implicit_biconcave` relevé dans `RESULTATS_courbures.md`. Les unions
  lissées (`smooth min`) le corrigent et rendraient les booléens compatibles
  avec l'analyse de courbure.
- **Morphing entre plus de deux formes** : la formule se généralise en
  barycentre, `Σ theta_i f_i` avec `Σ theta_i = 1`, ce qui ouvre un espace de
  formes plutôt qu'un chemin.

## 9. Relancer

```bash
cd Code
/usr/local/bin/python3 main_morph.py        # ~20 s (réseaux en cache)
/usr/local/bin/python3 main_curvature.py    # section 4 = contrôle micromètres
/usr/local/bin/python3 validate_geometry.py # doit rester 28/28
```

Sorties : `output/morph_sphere_globule.png`, `output/csg_booleens.png`,
`output/morph_reseaux.png`, `output/courbure_globule_micrometres.png`.
