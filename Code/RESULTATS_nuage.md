# Apprendre une surface depuis un nuage de points seul

Journal de la campagne `main_cloud.py`. Sortie brute complète : `output/sortie_nuage.txt`.

## Ce qui est en jeu

Tout le code du projet part d'une forme analytique : `get_surface(...)` renvoie une
fonction `implicit_exact`, et le signe dedans/dehors s'obtient en évaluant cette
fonction. Sur les capsules du BMBI, cette fonction n'existera pas. Il y aura un
fichier : des points sur la surface, un vecteur normal en chacun d'eux, et rien
d'autre.

La question posée ici est donc : **est-ce que la chaîne tient encore quand on lui
retire la formule ?**

## La chaîne, coupée en deux

    GÉNÉRATION                          EXPLOITATION
    la formule fabrique un maillage     on relit le fichier et on ne voit
    triangulaire, puis un fichier       plus que des points et des normales
    .npz (src/cloud_io.py)              (src/inside_outside.py, main_cloud.py)

La coupure n'est pas seulement déclarative. Pendant l'entraînement du bloc 4, la
fonction implicite est placée sous un compteur d'appels : **il affiche 0**. La
formule ne ressert qu'après, pour noter le résultat.

Deux points méritent d'être soulignés dans la génération :

- **Les normales ne sont pas le gradient analytique.** On suit le paragraphe
  « Plus difficile » du mail de F. De Vuyst : on maille la surface par des
  triangles, on numérote les sommets dans le sens trigonométrique, et la normale
  du triangle est le produit vectoriel de deux arêtes. Les normales aux sommets
  sont la moyenne des normales des triangles adjacents, pondérée par leur aire.
- **L'orientation se règle sans formule**, par le volume signé du maillage
  fermé, `V = (1/6) Σ v₀·(v₁×v₂)`, positif si et seulement si les normales
  sortent. Contrôle : l'ellipsoïde donne 2.41179 contre 4πabc/3 = 2.41274
  attendu, et les deux maillages sont trouvés déjà sortants.

## Bloc 1 — le signe déduit du nuage tient-il ?

Nuage de 6000 points, vote sur k = 5, 200 000 points de contrôle.

| forme | accord global | accord hors coquille `\|f\|>0.05` | tous les désaccords à une distance ≤ |
|---|---|---|---|
| ellipsoïde | 99.984 % | **100.000 %** | 0.0128 |
| globule quartique | 99.960 % | **100.000 %** | 0.0156 |

Le critère demandé (≥ 99 % hors coquille) est non seulement atteint mais saturé :
**pas un seul désaccord sur 195 000 points hors coquille, sur les deux formes.**

Le résultat important n'est pas le taux mais sa localisation : les 33 et 79 points
mal classés sont tous à moins de 0.016 de la surface, avec une médiane à 0.0064.
La méthode ne se trompe jamais « quelque part dans le volume » — elle hésite
uniquement dans une pellicule collée à la surface. C'est le comportement qu'on
espère, puisque c'est précisément là que la valeur de `f` est proche de zéro et
que le terme de signe de la perte ne pèse presque rien.

Note de lecture : le seuil `|f| > 0.05` n'a pas la même épaisseur pour les deux
formes, parce que `f` est une pseudo-distance pour l'ellipsoïde et une quartique
pour le globule. En distance réelle, la coquille exclue vaut 0.060 pour
l'ellipsoïde et 0.121 pour le globule. C'est pour cela que la colonne « distance
des désaccords » a été ajoutée : elle, se compare d'une forme à l'autre.

## Bloc 2 — combien de points faut-il ?

Le même maillage sert à chaque fois, seul le nombre de points tirés change.

**Ellipsoïde**

| points | espacement | accord global | hors coquille | épaisseur d'erreur |
|---|---|---|---|---|
| 500 | 0.0681 | 99.721 % | 99.998 % | 0.0611 |
| 2000 | 0.0348 | 99.937 % | 100.000 % | 0.0184 |
| 6000 | 0.0211 | 99.984 % | 100.000 % | 0.0184 |
| 20000 | 0.0121 | 99.995 % | 100.000 % | 0.0091 |
| 6000 bruité (σ=0.01) | — | 99.882 % | 100.000 % | 0.0182 |

**Globule quartique**

| points | espacement | accord global | hors coquille | épaisseur d'erreur |
|---|---|---|---|---|
| 500 | 0.0643 | 99.537 % | 99.953 % | 0.0797 |
| 2000 | 0.0328 | 99.882 % | 100.000 % | 0.0246 |
| 6000 | 0.0196 | 99.960 % | 100.000 % | 0.0156 |
| 20000 | 0.0119 | 99.989 % | 100.000 % | 0.0112 |
| 6000 bruité (σ=0.01) | — | 99.784 % | 100.000 % | 0.0210 |

**La règle qui s'en dégage : l'épaisseur de la couche d'erreur est de l'ordre de
l'espacement du nuage** (entre 0.5 et 1.2 fois, sur les huit cas mesurés). C'est
une règle utilisable telle quelle sur les vraies données : si l'on veut un signe
fiable partout sauf à moins de *d* de la surface, il faut un nuage d'espacement
inférieur à *d*.

Deux conséquences pratiques :

- **500 points suffisent déjà** à obtenir plus de 99.95 % hors coquille. Le
  procédé n'est pas gourmand ; c'est une bonne nouvelle pour des données
  expérimentales, toujours plus rares qu'on ne voudrait.
- Passer de 6000 à 20000 points ne gagne quasiment rien en accord global
  (99.96 → 99.99 %) mais divise l'épaisseur d'erreur par 1.4. **Densifier le nuage
  ne rend pas la méthode « plus juste », cela rapproche seulement de la surface la
  zone où elle hésite.**

## Bloc 3 — effet de k, et une hypothèse démentie

L'hypothèse de départ était que le creux du globule — là où les deux cuvettes sont
presque à égale distance de l'axe, et où le plus proche voisin peut basculer de
l'une à l'autre — exigerait un vote sur plusieurs voisins. **C'est faux.**

Sur nuage propre (3379 points de contrôle dans le creux) :

| k | accord global | hors coquille | dans le creux |
|---|---|---|---|
| 1 | **99.986 %** | 100.000 % | **99.882 %** |
| 3 | 99.973 % | 100.000 % | 99.763 % |
| 5 | 99.960 % | 100.000 % | 99.704 % |
| 9 | 99.937 % | 100.000 % | 99.615 % |
| 15 | 99.903 % | 100.000 % | 99.408 % |

Sur nuage bruité (σ = 0.01) :

| k | accord global | hors coquille | dans le creux |
|---|---|---|---|
| 1 | 99.564 % | 99.996 % | **96.567 %** |
| 3 | 99.724 % | 100.000 % | 97.810 % |
| 5 | 99.784 % | 100.000 % | 98.224 % |
| 9 | 99.831 % | 100.000 % | 98.520 % |
| 15 | **99.837 %** | 100.000 % | **98.698 %** |

Le classement s'inverse complètement. Sur données propres, `k = 1` est le meilleur
partout et le vote **dégrade** : interroger des voisins plus lointains fait entrer
dans l'urne des points de la cuvette opposée, qui votent à contresens. Moyenner sur
un voisinage plus large revient à lisser une particularité géométrique bien réelle.

Sur données bruitées, l'ordre s'inverse : `k = 1` perd 3.3 points dans le creux
(96.57 %), et le vote les récupère presque entièrement (98.70 % à k = 15).

**Conclusion : k est un réglage de robustesse au bruit, pas un réglage de
géométrie.** Sur un nuage exact, il ne faut pas voter. Sur un nuage mesuré, il
faut. Comme les données du BMBI seront mesurées, `k = 5` a été retenu pour la
suite : il capte l'essentiel du gain en bruité (98.22 % contre 96.57 %) sans
coûter grand-chose en propre (99.70 % contre 99.88 %).

## Blocs 4 et 5 — perd-on quelque chose à se passer de la formule ?

Entraînement IGR sur l'ellipsoïde, 2000 époques, points de surface et normales
lus dans le fichier, signe des 20 000 points de volume déduit du nuage (k = 5).
Appels à la formule pendant l'entraînement : **0**.

Les deux entraînements comparés ne diffèrent **que** par la source du signe :
mêmes points de surface, mêmes normales, mêmes positions de volume, mêmes graines.
Le calcul tourne sur **CPU et non sur MPS** : sur GPU Apple, deux entraînements à
graine identique divergent (constaté sur ce projet), et l'écart mesuré serait alors
un mélange de l'effet cherché et du bruit matériel. Trois graines, l'écart-type
entre graines servant d'étalon pour juger si un écart veut dire quelque chose.

| métrique | signe du NUAGE | signe de la FORMULE | écart | verdict |
|---|---|---|---|---|
| Chamfer | 0.01593 ± 0.00113 | 0.01582 ± 0.00056 | +0.00010 | dans le bruit |
| Hausdorff | 0.04208 ± 0.00141 | 0.04082 ± 0.00141 | +0.00126 | dans le bruit |
| IoU | 0.96876 ± 0.00797 | 0.96898 ± 0.00404 | −0.00021 | dans le bruit |
| eikonal | 0.01335 ± 0.00158 | 0.01361 ± 0.00104 | −0.00026 | dans le bruit |

**Réponse : non, on ne perd rien.** Les quatre écarts sont plus petits que la
dispersion d'un entraînement à l'autre. Sur l'IoU, l'écart vaut 0.0002 quand la
dispersion entre graines vaut 0.008 : quarante fois plus petit que le bruit. Le
Chamfer moyen (0.0159) reste par ailleurs bien en dessous de la maille de la
grille de reconstruction (3.2/64 = 0.05), donc l'erreur de reconstruction est
dominée par la discrétisation du marching cubes, pas par la provenance du signe.

### Pourquoi, et jusqu'où cette conclusion porte

Il faut dire par quel mécanisme elle est vraie, sinon on la surinterprète : sur les
20 000 points de volume, **le signe déduit du nuage et le signe exact ne diffèrent
que sur 4 points, soit 0.020 %.** Les deux entraînements voient donc des données
quasiment identiques, et il serait étonnant qu'ils divergent.

Le résultat démontré est donc précisément celui-ci : *le test dedans/dehors par
produit scalaire est assez fidèle pour que le réseau ne voie pas la différence*.
Ce n'est **pas** une démonstration que le réseau serait robuste à un étiquetage
franchement dégradé — cette question-là n'a pas été posée ici. Le bloc 2 indique
qu'un nuage de 500 points donnerait environ 0.3 % d'étiquettes fausses, soit
une soixantaine de points sur 20 000 ; savoir si l'entraînement encaisse ce
régime demanderait une campagne dédiée.

## Ce que ça change pour la suite

1. **Le mode « données réelles » est opérationnel et éprouvé.** Le jour où le BMBI
   livre un maillage de capsule, il suffit d'écrire un `.npz` au format de
   `src/cloud_io.py` ; le reste de la chaîne tourne sans modification.
2. **Le format d'échange est fixé** : `points`, `normals`, `box`, plus une
   étiquette de provenance. `load_cloud` refuse un fichier incohérent (normales
   manquantes, points hors boîte) plutôt que de produire un résultat silencieusement
   faux, et lit avec `allow_pickle=False`.
3. **Un dimensionnement chiffré à donner au labo** : espacement du nuage inférieur
   à la distance à laquelle on veut un signe fiable. À l'échelle du globule rouge
   (diamètre 8 µm), l'espacement de 0.0196 en unités réduites vaut 0.078 µm.
4. **Un réglage justifié** : `k = 5` sur données mesurées, `k = 1` sur données
   exactes — et la raison, qui n'est pas celle qu'on croyait.

## Limites assumées

- Les normales du nuage bruité restent celles du maillage propre : seules les
  positions sont perturbées. Sur de vraies données, les normales seraient estimées
  à partir de points bruités et seraient donc bruitées elles aussi. Le cas traité
  ici est le cas favorable.
- Les blocs 4 et 5 ne portent que sur l'ellipsoïde. Le globule, dont le creux est
  autrement plus difficile, n'a été passé au crible que pour le signe (blocs 1 à 3).
- La forme sous-jacente reste analytique, donc parfaitement lisse et fermée. Un
  maillage Éléments Finis réel aura des trous, des triangles dégénérés et une
  densité inhomogène ; aucun de ces défauts n'est simulé ici.
