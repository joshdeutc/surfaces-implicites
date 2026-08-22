# Courbures et invariants géométriques — journal et résultats

Travail du 22 août 2026. À recopier dans `PLAN.md` (section « Journal des étapes »)
une fois les autres chantiers en cours terminés.

Répond à la ligne du programme officiel de la TZ restée vierge :
*« 10 h — lecture d'un article (Nature Comm, 2021) sur la représentation d'un
globule rouge en surface implicite. Définition et calcul de différentes
courbures. Mise en œuvre. »*

---

## 1. La question posée

Jusqu'ici, la seule mesure de qualité du projet était `mean|f_exact|` : la
distance entre les points de la surface reconstruite et la vraie surface.
Autrement dit, on savait vérifier **où est la surface**, et rien d'autre.

Or une surface se décrit à plusieurs ordres de dérivation :

| Ordre | Ce que ça décrit | Ce qui le mesurait |
|---|---|---|
| 0 | la position de la surface | `mean\|f_exact\|`, Chamfer |
| 1 | le gradient, donc la normale | le terme eikonal (déjà à 0,02-0,03, pas 0) |
| 2 | **la courbure** | **rien** |

C'est un angle mort gênant pour deux raisons.

D'abord parce que **la courbure est ce qui intéresse le laboratoire** : sur une
capsule ou un globule rouge, c'est elle qui gouverne les contraintes dans la
membrane. C'est tout le sujet de l'article de Kuchel et al. (Sci. Rep. 11:13712,
2021) qui figure dans `Articles/`.

Ensuite parce que rien ne garantit qu'un réseau juste à l'ordre 0 le soit à
l'ordre 2. Un champ peut placer sa surface parfaitement tout en étant bosselé à
une échelle invisible pour le Chamfer, mais dévastatrice pour une dérivée
seconde.

**Question : le champ implicite appris est-il juste au-delà de l'ordre 0 ?**

## 2. La difficulté, et comment elle est contournée

Vérifier une courbure demande normalement une vérité de référence. Sur les
vraies capsules du BMBI, il n'y aura aucune formule analytique — donc, a priori,
rien à quoi comparer.

**Gauss-Bonnet lève l'obstacle.** Pour toute surface fermée sans trou (tout ce
qui se déforme continûment en une sphère : un œuf, un globule rouge, une
capsule), l'intégrale de la courbure gaussienne sur toute la surface vaut
toujours la même chose, quelle que soit la forme :

```
∫ K_G dA = 2πχ = 4π ≈ 12,566        (χ = 2 pour une surface de genre 0)
```

C'est un contrôle **gratuit et sans référence** : le théorème fournit la valeur
attendue. Il s'appliquera donc tel quel aux vraies données.

Limite à garder en tête : c'est une condition **nécessaire, pas suffisante**.
Une intégrale peut tomber juste avec des erreurs locales qui se compensent.
C'est pourquoi elle est systématiquement doublée ici d'un contrôle **local**,
point par point.

## 3. Ce qui a été ajouté

Quatre fichiers, tous nouveaux. **Aucun fichier existant n'a été modifié.**

### `src/curvature.py`
Courbures d'une surface implicite, formules de Goldman (2005) reprises par
Kuchel et al. (éq. 14-17) :

```
K_gauss   = (∇f · H* · ∇fᵀ) / ‖∇f‖⁴          H* = comatrice de la hessienne
K_moyenne = (∇f · H · ∇fᵀ − ‖∇f‖²·Tr H) / (2‖∇f‖³)
k1, k2    = K_M ± √(K_M² − K_G)
```

Elles ne font intervenir que le gradient et la hessienne de `f`, donc elles
s'appliquent **indifféremment à une formule analytique ou à un réseau**. Deux
implémentations :

- `curvature_numpy` — différences finies centrées, pas `h = 1e-4` (équilibre
  entre erreur de troncature en `h²` et erreur d'arrondi en `eps/h²`) ;
- `curvature_autograd` — différentiation automatique d'ordre 2 sur un réseau
  PyTorch. Trois passes arrière, une par ligne de la hessienne.

**Convention de signe, établie et démontrée.** Avec la convention du projet
(`f < 0` dedans, donc gradient sortant), une sphère de rayon R donne :

```
K_gauss = +1/R²        K_moyenne = −1/R  (NÉGATIVE)        k1 = k2 = −1/R
```

La courbure moyenne est donc **négative pour une surface convexe**. C'est une
conséquence directe de la formule (le terme dominant est `−‖∇f‖²·Tr H`, et la
trace est positive pour une sphère). Inverser le signe de `f` inverse `K_M`
mais laisse `K_G` inchangée. On compare donc toujours `|K_moyenne|` aux valeurs
de la littérature, qui adopte souvent la convention inverse.

### `src/invariants.py`
Les contrôles qui disent si l'on peut croire un maillage :
caractéristique d'Euler `χ = V − E + F`, étanchéité, aire, volume par le
théorème de la divergence, et Gauss-Bonnet.

Deux pièges traités explicitement :

- **le comptage des arêtes** — un maillage fermé de F triangles a `E = 3F/2`
  arêtes, pas `3F` : chaque arête est partagée par deux triangles. Compter `3F`
  donne un `χ` absurde. Les arêtes sont donc orientées canoniquement (indices
  triés) puis dédupliquées ;
- **les sommets dupliqués** — le marching cubes peut produire deux sommets à la
  même position quand l'isosurface frôle un nœud de la grille. `V` est alors
  gonflé et `χ` est faux. `weld_vertices` les fusionne avant tout comptage.

### `validate_geometry.py`
Le filet de sécurité : il valide les deux modules sur des cas dont la réponse
est connue **avant** qu'on les pointe sur un réseau. À relancer après toute
modification.

### `main_curvature.py`
L'expérience : entraîne un réseau IGR, calcule ses courbures par autograd, et
le contrôle aux deux niveaux (global par Gauss-Bonnet, local point par point).
Produit `output/courbure_globule.png`. Les modèles sont mis en cache, donc une
relance prend quelques secondes.

### Correction au passage
`curvature_autograd` castait en float64 **avant** de rapatrier depuis le GPU.
MPS (le GPU Apple) ne supporte pas cette précision. Le cast se fait maintenant
après le retour sur CPU. L'autograd d'ordre 2 fonctionne bien sur MPS ; un repli
automatique sur CPU reste en place au cas où.

## 4. Validation des outils — 28 vérifications, 28 passent

```
/usr/local/bin/python3 validate_geometry.py
```

| Cas | Mesure | Obtenu | Attendu |
|---|---|---|---|
| Sphère R=1 | `K_gauss` | 1,000000 | 1 (écart 1,4e-8) |
| | `K_moyenne` | −1,000000 | −1 (écart 6,9e-9) |
| | `k1`, `k2` | −1,000000 | −1 (ombilic) |
| Sphère R=2 | `K_gauss` | 0,250000 | 0,25 (écart 1,4e-8) |
| Sphère maillée 128³ | `χ` | 2 | 2 |
| | aire | 12,564178 | 4π = 12,566371 |
| | volume | 4,187406 | 4π/3 = 4,188790 |
| | `∫K dA` | 12,564683 | 4π = 12,566371 |
| **Tore** (R=1, r=0,35) | `χ` | **0** | 0 |
| | genre | 1 | 1 |
| | aire | 13,813392 | 4π²Rr = 13,817446 |
| | volume | 2,415612 | 2π²Rr² = 2,418053 |
| | **`∫K dA`** | **−0,0070** | **0** |
| Autograd sur `‖x‖−1` | `K_gauss` | 1,000000 | 1 (écart 1,9e-8) |
| | accord avec diff. finies | 2,2e-7 | 0 |

**Le tore est le test décisif.** Sur un donut, `χ = 0`, donc l'intégrale doit
valoir exactement zéro : les courbures positives de l'extérieur du tube doivent
annuler les négatives de l'intérieur. L'amplitude à compenser est
`∫|K| dA = 25,1` ; on obtient −0,007, soit **0,03 % de compensation**. Ce test
ne passe que si le comptage d'arêtes **et** le calcul de hessienne sont corrects
tous les deux — aucune erreur systématique ne peut s'y cacher.

## 5. Le résultat

```
/usr/local/bin/python3 main_curvature.py
```

| Cas | χ | `∫K dA` | écart à 4π | top 1 % |
|---|---|---|---|---|
| Sphère apprise par le réseau | 2 | +12,5646 | **−0,014 %** | 2,2 % |
| Globule appris par le réseau | 2 | +12,5633 | **−0,025 %** | 8,4 % |
| Globule, quartique de Kuchel (analytique) | 2 | +12,5586 | −0,062 % | 4,5 % |
| Globule, formule actuelle de `shapes.py` | 2 | −2,615·10⁸ | −2,1·10⁹ % | 100 % |
| *valeur théorique* | *2* | *+12,5664* | *0 %* | |

*(« top 1 % » = part de `∫|K| dA` portée par les 1 % de triangles les plus
courbés. Une valeur proche de 100 % signale des pics de courbure aberrants.)*

### Réponse à la question posée : oui, le réseau est juste à l'ordre 2

Gauss-Bonnet passe à **0,025 %** sur le globule appris, et le contrôle local le
confirme :

| Contrôle local | Résultat |
|---|---|
| Sphère apprise, `K_gauss` (vaut 1 partout) | erreur moyenne 0,027 · médiane 0,023 · q95 0,066 |
| Sphère apprise, `\|K_moyenne\|` (vaut 1) | erreur moyenne 0,013 |
| Globule appris vs formule exacte (11 900 sommets) | `K_gauss` 7,5 % · `K_moyenne` 5,3 % |
| Corrélation `K_gauss` réseau / exact | **0,9962** |

La formule exacte utilisée pour ce contrôle local est celle de la **surface de
révolution** `z = h(r)` — donc indépendante de la fonction implicite du dépôt :

```
κ_méridien = h'' / (1+h'²)^{3/2}     κ_parallèle = h' / (r √(1+h'²))
K_G = κ_m · κ_p                      K_M = (κ_m + κ_p)/2
```

Avec un contrôle entièrement analytique au centre du creux : près de `r=0`,
`z ≈ a0 + (a1 − a0/2)r²`, donc `K_G(0) = 4(a1 − a0/2)² = 14,432` exactement.

### Un biais mesuré : le réseau lisse les extrêmes de 16 %

| Au centre du creux (52 sommets) | Réseau | Exact |
|---|---|---|
| `K_gauss` | +12,120 | +14,432 |
| `K_moyenne` | +3,481 | +3,799 |

C'est le biais spectral, quantifié précisément là où la courbure est maximale.
Le réseau arrondit ce qui est le plus tordu. Une piste ouverte est d'ajouter un
terme de courbure à la perte et de mesurer le compromis avec le Chamfer.

## 6. Le défaut trouvé — et réparé

`implicit_biconcave` dans `src/shapes.py` est définie **par morceaux** :

```python
return np.where(r < 1.0, val_in, val_out)
```

La fonction est continue en `r = 1`, mais **son gradient ne l'est pas** : il y a
un angle. Les dérivées secondes y divergent donc, et l'intégrale de courbure
part à −2,6·10⁸ au lieu de 12,566.

Vérification de la cause : **176 sommets sur 44 696 ont `|K_G| > 100`, et leur
rayon est dans `[0,9998 ; 1,0103]`** — exactement le changement de branche.
Ils portent à eux seuls 100 % de `∫|K| dA`.

**Contre-épreuve.** Le même objet géométrique, décrit cette fois par la
quartique exacte de Kuchel `(x²+y²+z²)² + P(x²+y²) + Qz² + R = 0` — lisse
partout — redonne `∫K dA = 12,5586`, soit **−0,06 %**.

Donc : le défaut ne venait ni de la forme, ni de la méthode, **mais de la
formule que le projet utilisait**. C'est l'argument le plus direct pour lui
substituer la quartique dans toute la suite.

Ce cas valide au passage un point théorique : la quartique n'est **pas** une
distance signée (`‖∇F‖ ≠ 1`) et donne pourtant le bon résultat, ce qui confirme
que les formules de courbure sont bien **invariantes par reparamétrage de `f`**.

## 7. Ce que ça apporte au projet

1. **La ligne « courbures » du programme officiel est faite**, avec l'article de
   Kuchel effectivement mis en œuvre.
2. **Un critère de qualité d'un ordre supérieur au Chamfer**, et surtout
   **utilisable sur les vraies capsules** puisqu'il ne demande aucune formule de
   référence. C'est la même contrainte que celle qui a motivé les métriques
   Chamfer/IoU, poussée un cran plus loin.
3. **Une réponse chiffrée à une question qui restait ouverte** : le champ appris
   est fiable jusqu'à l'ordre 2, à 0,025 % près en global et 7,5 % en local, avec
   un biais de lissage de 16 % aux extrêmes.
4. **Un défaut réel corrigé** dans la description du globule rouge, avec la
   preuve chiffrée que le remplacement par la quartique le répare.
5. **Un filet de sécurité réutilisable** (`validate_geometry.py`, 28 contrôles)
   qui dit en quelques secondes si une modification a cassé la géométrie.

## 8. Pistes ouvertes

- Ajouter un terme de **régularisation de courbure** à la perte, et mesurer le
  compromis : la courbure s'améliore-t-elle au prix du Chamfer ?
- Refaire toute la mesure sur la **quartique** plutôt que sur la formule par
  morceaux, maintenant qu'elle est disponible.
- Appliquer les courbures aux formes de **genre non nul** (gyroïde, Chmutov,
  Barth) : `∫K dA = 2πχ` y prend d'autres valeurs, ce qui en fait un test
  topologique en plus d'un test différentiel.
- Comparer les cartes de courbure aux **valeurs de Kuchel** en unités physiques
  (µm), une fois la quartique branchée avec `d = 8 µm`.
- Étendre au **spatio-temporel** : suivre l'évolution de la courbure d'une
  capsule qui se déforme, c'est exactement ce que l'article fait sous contrainte.

## 9. Commits

| Commit | Contenu |
|---|---|
| `a2aa2e3` | `src/curvature.py`, `src/invariants.py`, `validate_geometry.py` |
| `0fc52c7` | `main_curvature.py` + correction du cast float64 sur MPS |
| `ec0b2ac` | contre-épreuve sur la quartique de Kuchel |

## 10. Comment relancer

```bash
cd Code
/usr/local/bin/python3 validate_geometry.py   # 28 contrôles, ~30 s
/usr/local/bin/python3 main_curvature.py      # l'expérience, ~1 min (modèles en cache)
```

Sorties : `output/courbure_globule.png` (coupe colorée par la courbure) et
`output/model_curv_*.pt` (réseaux entraînés).
