# Une surface implicite qui évolue dans le temps

Journal de la campagne « spatio-temporel » : ce qu'on a cherché, comment,
ce qu'on a trouvé, et ce que ça ne prouve pas.

Reproduire : `python main_temporal.py` (environ 6 minutes sur un M2).

---

## 1. La question

C'est le point que Florian appelle « le Graal » dans son mail :

> « Le Graal serait d'étudier des surfaces qui évoluent dans le temps. C'est
> beaucoup plus compliqué, vous pouvez vous le réserver pour la fin si vous avez
> bien avancé avant. »

et que la fiche d'enregistrement formule ainsi :

> « Le groupe TZ essaiera alors de "condenser" la forme mobile en une surface
> implicite espace-temps. »

Le mot qui compte est **condenser**. Entraîner un réseau par instant est facile,
mais ce n'est qu'un album de photos : on n'a rien condensé, et on ne sait rien
dire entre deux clichés. La question est donc : peut-on apprendre **un seul**
champ `f(x,y,z,t) = 0` qui tienne toute la séquence — et, s'il la tient, que
sait-il faire qu'une suite de réseaux statiques ne saurait pas ?

## 2. Le protocole

Un code Éléments Finis ne rend pas une fonction du temps : il rend une **suite de
fichiers**, un par pas de temps sauvegardé. Le protocole imite exactement ça.

**La déformation.** Un étirement selon `z` à volume constant appliqué à la
quartique de Kuchel (le globule rouge) :

    A(t) = diag( xi^(-1/2), xi^(-1/2), xi ),   xi(t) = 1 + t/2,   det A = 1

`xi` va de 1.0 (globule au repos) à 1.5 (allongé de 50 %). C'est le geste
élémentaire d'une capsule dans un écoulement élongationnel : elle s'étire dans
une direction et se contracte dans les deux autres, à volume inchangé.

Le volume constant n'est pas un choix esthétique : il donne un **invariant que
l'entraînement ignore complètement** et qu'on peut donc lui opposer à la fin.

**La coupure.** Six instants sont écrits sur disque au format `.npz` du projet
(`src/cloud_io.py`), avec points de surface et normales **issues du maillage
triangulaire**, pas du gradient analytique :

| | instants | ce que reçoit le réseau |
|---|---|---|
| observés | t = 0, 0.2, 0.4, 0.6, 0.8, 1.0 | 6000 points + normales chacun |
| **cachés** | **t = 0.1, 0.3, 0.5, 0.7, 0.9** | **rien du tout** |

La formule exacte reste disponible de notre côté, mais **uniquement pour noter**.
Le script le prouve plutôt que de le déclarer : la fonction est placée sous
compteur avant même la préparation des données, et il affiche

    appels à la formule exacte pendant la préparation des données : 0
    appels à la formule exacte pendant l'entraînement            : 0

Les six nuages sont tirés avec des graines différentes : **aucune correspondance
point à point** d'un instant à l'autre. Un calcul lagrangien en offrirait une ;
s'en passer est le cas défavorable, et c'est celui qu'on éprouve.

## 3. Ce qui a été construit

| Fichier | Rôle |
|---|---|
| `src/temporal.py` | la déformation, l'écriture des six instants, et le champ exact réservé à la mesure |
| `src/model_temporal.py` | `TemporalIGRNet` : entrées `(x,y,z,t)`, Softplus, initialisation géométrique, 197 885 poids |
| `src/train_temporal.py` | la perte et la boucle, plus les dérivées `∇_x f` et `∂f/∂t` du champ appris |
| `main_temporal.py` | les quatre mesures, les contrôles, les figures |

### Le piège : l'eikonal ne porte que sur le gradient d'espace

Le réseau a quatre entrées, donc son gradient a quatre composantes. Or la
condition eikonal `‖∇f‖ = 1` dit que `f` est une distance signée **dans
l'espace**, à chaque instant figé. Le temps n'est pas une quatrième dimension
d'espace : `∂f/∂t` n'est pas une pente géométrique, c'est la vitesse à laquelle
le champ change en un point fixe.

L'écrire sur les quatre composantes reviendrait à exiger

    (∂f/∂x)² + (∂f/∂y)² + (∂f/∂z)² = 1 − (∂f/∂t)²

c'est-à-dire à demander au champ d'être **d'autant moins une distance signée
qu'il bouge vite**. Le terme censé régulariser la géométrie punirait le
mouvement. D'où, partout : `grad[..., :3]`.

### Le choix qui décide de tout : où l'on tire `t`

Les termes de surface, de normales et de signe sont attachés aux six instants
observés — ils ne peuvent rien dire d'autre. Si le terme eikonal était lui aussi
tiré sur ces six valeurs, **rien dans la perte ne contraindrait le champ entre
deux instants**, et le test d'interpolation ne mesurerait que le hasard de
l'initialisation.

Le terme eikonal est donc tiré à `t` **continu** dans [0, 1]. C'est la seule
pièce de la perte qui parle des instants non observés, et elle n'y impose qu'une
chose, purement géométrique : rester une distance signée. Elle ne dit jamais
quelle forme il doit y avoir.

### L'initialisation

L'initialisation géométrique d'IGR fait qu'avant tout apprentissage le réseau
approxime déjà la distance signée à une sphère. Transposée telle quelle à quatre
entrées, elle produirait la sphère de R⁴ — un champ qui dépend de `t` dès le
départ, dans une direction arbitraire. On annule donc les deux colonnes de poids
par lesquelles `t` entre (première couche, et skip-connection). Vérifié
numériquement : `max |f(x,t) − f(x,0)| = 0.0e+00` à l'initialisation. Les poids
ne sont pas gelés — ils sont libres dès le premier pas de gradient.

## 4. Résultats

Entraînement : 8000 époques, 149 s sur GPU Apple. Grille de reconstruction 96³
(pas de 0.0274).

### Bloc 1 et 2 — le test central

| t | statut | Chamfer | IoU | χ |
|---|---|---|---|---|
| 0.00 | vu | 0.00821 | 0.99023 | 2 |
| **0.10** | **jamais vu** | **0.00823** | **0.99140** | 2 |
| 0.20 | vu | 0.00808 | 0.99109 | 2 |
| **0.30** | **jamais vu** | **0.00822** | **0.99053** | 2 |
| 0.40 | vu | 0.00813 | 0.99104 | 2 |
| **0.50** | **jamais vu** | **0.00811** | **0.99025** | 2 |
| 0.60 | vu | 0.00807 | 0.99076 | 2 |
| **0.70** | **jamais vu** | **0.00812** | **0.99089** | 2 |
| 0.80 | vu | 0.00814 | 0.99235 | 2 |
| **0.90** | **jamais vu** | **0.00810** | **0.99208** | 2 |
| 1.00 | vu | 0.00823 | 0.98798 | 2 |

Moyennes : **0.00814 / 0.99058** aux instants vus, **0.00815 / 0.99103** aux
instants jamais vus. Rapport 1.001. La caractéristique d'Euler vaut 2 partout :
la topologie est correcte aux onze instants.

**Le réseau restitue des instants qu'on ne lui a jamais montrés, aussi bien que
ceux qu'il a vus.** Il a condensé la séquence en un champ continu.

### Bloc 3 — conservation du volume

Le volume exact vaut 1.343568 et ne dépend pas de `t` (det A = 1). Le marching
cubes commet lui-même une erreur : on la mesure en appliquant la même grille au
champ **exact**, ce qui donne le plancher (−0.19 % à −0.25 %).

| | valeur |
|---|---|
| volume appris, moyenne sur les 11 instants | 1.329617 |
| **écart-type au fil du temps** | **0.000539, soit 0.041 %** |
| biais par rapport au volume exact | −1.038 % |

**C'est la dispersion qui compte, pas le biais.** Un biais constant n'est qu'un
décalage de l'isosurface — pour une forme de rayon ~1, `dV/V = 3·dr/r` situe ce
décalage autour de 0.003, soit un tiers du Chamfer. Une dérive, elle, signifierait
que le réseau perd la forme en avançant dans le temps. Elle vaut 0.041 %.

Contrôle de robustesse : entre deux entraînements de graine identique sur GPU
Apple, le biais est passé de **+0.79 % à −1.04 %** — la non-reproductibilité MPS
déjà documentée dans `README.md`. La dispersion au fil du temps, elle, est restée
entre 0.04 % et 0.08 %. Le biais n'est donc pas une propriété de la méthode ; la
dispersion, si.

### Bloc 4 — suivi lagrangien

Sur la surface, `f(x(t), t) = 0` à tout instant. En dérivant le long du
mouvement, `∇_x f · ẋ + ∂f/∂t = 0`, donc un point qui suit la normale avance à

    v_n = − (∂f/∂t) / ‖∇_x f‖

et cette quantité **se lit directement sur le réseau** : `∂f/∂t` est sa quatrième
dérivée partielle. Trois points de membrane sont advectés par RK4, 200 pas, sur le
champ appris et sur le champ exact — même équation, même schéma.

| point de départ | écart moyen | écart final | dérive hors membrane |
|---|---|---|---|
| fossette (r = 0) | 0.00032 | 0.00081 | 0.00031 |
| flanc (r = 0.55) | 0.00058 | 0.00104 | 0.00048 |
| bord (r ≈ 1) | 0.00034 | 0.00056 | 0.00020 |
| **ensemble** | **0.00041** | **0.00080** | **0.00033** |

La dérive du champ **exact** vaut 0.00000 : les dérivées analytiques `∇_x f` et
`∂f/∂t` sont cohérentes entre elles à la précision machine, et le schéma
d'intégration ne dérive pas. Tout ce qu'on mesure est donc bien imputable au
réseau.

**Une remarque qui n'est pas un défaut.** La trajectoire *matérielle* (le point de
matière transporté par A(t)) s'écarte de la trajectoire normale exacte de 0.03766
à t = 1. Les deux sont justes : un point de matière glisse aussi
**tangentiellement**, et ce glissement est invisible pour une surface de niveau.
Une surface implicite donne le mouvement de la **surface**, pas celui de la
matière. C'est une limite de la représentation, à connaître avant de vouloir
suivre un marqueur sur une vraie membrane.

## 5. Les contrôles — ce que ces chiffres ne prouvent pas

Un résultat aussi net appelle des objections. On les a mesurées plutôt que de les
écarter.

### 5.1 Le Chamfer est saturé

Marching cubes appliqué au champ **exact**, mêmes grilles, même nuage de
référence :

| grille | pas | Chamfer du champ exact |
|---|---|---|
| 64³ | 0.0413 | 0.01049 |
| **96³** | **0.0274** | **0.00783** |
| 128³ | 0.0205 | 0.00655 |
| 160³ | 0.0164 | 0.00579 |

Le réseau donne 0.00815, soit **1.04 fois le plancher**. Le Chamfer dit donc
« aussi bon que la grille le permet », et rien de plus fin. Il ne peut pas
départager deux méthodes proches. Ce plancher vient de la densité du nuage de
référence et de la taille des triangles, pas d'une erreur de position.

### 5.2 Étalon « image gelée »

Un réseau paresseux qui rendrait l'instant observé le plus proche :

| | Chamfer |
|---|---|
| image gelée | 0.01252 |
| réseau | 0.00815 |
| rapport | **1.53×** |

Amplitude totale du mouvement (Chamfer entre t = 0 et t = 1) : 0.10655 — le
réseau se trompe de 7.7 % de cette amplitude. Le test d'interpolation mesure donc
bien quelque chose, mais la marge est modeste.

### 5.3 Étalon « morphing entre les deux instants encadrants » — le résultat gênant

La vraie objection : pourquoi un réseau à quatre entrées, plutôt qu'interpoler
linéairement entre les deux instants observés voisins ? On construit la distance
signée **exacte** aux deux bouts et on les morphe (`src/csg.py`, theta = 1/2).
Étalon très favorable au morphing : il part des surfaces exactes, le réseau n'a eu
que 6000 points bruts.

| | Chamfer moyen |
|---|---|
| morphing par paires | 0.00806 |
| réseau espace-temps | 0.00815 |
| rapport | **0.99×** |

**Sur la position de la surface, le réseau 4D n'apporte rien de plus qu'un
morphing par paires.** Les deux sont au plancher de la grille (5.1), et notre
déformation est **affine** — or interpoler linéairement entre deux instants
reproduit presque exactement une application linéaire. Le cas de test est trop
facile pour cette mesure-là. Il faut le dire.

### 5.4 Ce qui départage : la vitesse normale

On mesure alors la grandeur qui ne sature pas. Le morphing par paires a bien une
dérivée en temps, mais **constante par morceaux** : le champ est affine en `t`
entre deux instants observés, donc `∂f/∂t` y est figée et saute à chaque instant
observé. Le réseau, lui, apprend une fonction lisse des quatre variables.

Erreur moyenne sur `v_n` aux instants cachés, contre la référence analytique
(`|v_n|` vaut environ 0.10) :

| | erreur sur v_n |
|---|---|
| réseau espace-temps | **0.00134** |
| morphing par paires | 0.02928 |
| rapport | **21.9×** |

**C'est là que la représentation espace-temps gagne** — pas sur la position de la
surface, mais sur les dérivées. Ce qu'apporte un champ `f(x,y,z,t)`, ce n'est pas
de mieux placer les points : c'est de posséder un **champ de vitesse dérivable
partout**, qu'on peut interroger à n'importe quel instant et intégrer (bloc 4).

## 6. Ce qu'on retient

1. **Un seul réseau tient toute la séquence** et restitue des instants qu'il n'a
   jamais vus, à la précision de la grille, avec la bonne topologie.
2. **Il conserve un invariant qu'on ne lui a jamais donné** : le volume ne dérive
   que de 0.041 % sur toute la séquence.
3. **Il donne un champ de vitesse utilisable** : un point de membrane advecté par
   `v_n` reste à 0.0003 de la vraie surface après 200 pas d'intégration.
4. **Et il ne gagne pas partout** : sur la position seule, un morphing entre
   instants voisins fait aussi bien. Le gain est sur les dérivées, d'un facteur 22.
5. **La chaîne est celle des vraies données** : fichiers sur disque, normales de
   maillage, signe par produit scalaire, zéro appel à la formule.

## 7. Limites et suites

**Notre déformation est trop facile**, et le bloc 5.3 le prouve. Elle est affine,
monotone, axisymétrique, sans bruit, à topologie constante. Une vraie capsule dans
un écoulement n'est rien de tout cela. Par ordre de rapport valeur/coût :

1. **Déformation non affine** (flexion, onde qui court sur la membrane) — c'est
   l'escalade qui rendrait le bloc 2 discriminant, puisqu'elle casse l'étalon
   morphing. La plus utile, et la moins chère.
2. **Mouvement non monotone** (basculement, oscillation, retour à une forme
   antérieure) — teste si le réseau sait donner la même forme à deux instants
   différents.
3. **Moins d'instants** (4, puis 3, puis 2) — la question pratique pour le BMBI :
   à quelle fréquence le code EF doit-il sauvegarder ?
4. **Bruit sur les points**, pour le réalisme d'acquisition.
5. Changement de topologie (deux capsules qui fusionnent) — le cas extrême.

**Deux points de la fiche restent ouverts** : le dataset BMBI, qui n'a pas encore
été fourni (`Données/` est vide), et l'affichage PyVista au format VTK, nommé deux
fois dans le programme de travail.

## 8. Figures

| Fichier | Contenu |
|---|---|
| `output/temporel_donnees_vs_appris.png` | ce qu'on donne au réseau et ce qu'il produit, aux 11 instants |
| `output/temporel_3d.png` | les surfaces reconstruites en 3D (≈ 27 000 triangles, χ = 2) |
| `output/temporel_3d_nuage.png` | le nuage 3D orienté fourni, et la surface apprise par-dessus |
| `output/temporel_profils.png` | coupes méridiennes, appris contre vrai |
| `output/temporel_trajectoires.png` | suivi lagrangien et écart au fil du temps |
