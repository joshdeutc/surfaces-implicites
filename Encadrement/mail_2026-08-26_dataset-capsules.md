Mail de Florian De Vuyst — mercredi 26/08/2026, 18:55
Objet : Big Dataset BMBI
Pour : Joshua Nancey, Nathan Jacas

C'est le mail qui débloque l'étape 11 : les vraies données de capsules.

---

Bonsoir Joshua, Nathan,

J'ai retrouvé le dataset, il fait 1.2 Go.

Il s'agit de l'enregistrement de capsules issus de simulation numerique de dynamique de
capsule dans un canal. Il y a 20000 pas de temps, les pas de temps sont tres petits, donc
vous pouvez largement sous-echantillonner le temps pour vous limiter, par exemple à 100
snapshots (càd frames).

Les capsules sont maillées par maillage triangulaire. La Topologie du maillage est données
dans connect.npy,

L'autre tableau est celui des coordonnées x, y, z des nœuds pour les différents frames.

Je vous ai fabriqué un mini-notebook pour lire et visualizer les capsules.

Les capsules ont déjà été recentrées en 0, vous n'avez pas à vous inquiéter de cela.

[lien FileSender RENATER — expire, ne pas rediffuser]

Comme convenu, j'ai besoin que vous m'écriviez, par retour de mail, la clause de
confidentialité suivante :

« Je soussigné, ..., m'engage à ne pas diffuser le fichier de données qui m'a été transmis
par M. Florian De Vuyst. Ce dataset est propriété du laboratoire BMBI de l'UTC »

Merci par avance,

J'espère que vous allez pouvoir appliquer vos méthodes de surfaces implicites, ce serait
vraiment super,

Cela pourrait conduire à la publication d'un article scientifique. Dans ce cas, vous seriez
naturellement associés en tant que co-auteurs,

On n'en n'est pas encore là mais nous pourrons en reparler en cas de succès

Bon travail,

Bien cordialement,
Florian

--
Florian De Vuyst
Professeur des Universités
Laboratoire BMBI UMR 7338, UTC, CNRS
Responsable du Master Mention « Ingénierie des systèmes complexes »
Co-animateur du Centre de Recherche Interdisciplinaire « IA et Systèmes technologiques »

---

## À faire de notre côté

- [ ] **Renvoyer la clause de confidentialité par mail**, nominativement, chacun de
      notre côté. C'est une condition posée, pas une formalité.
- Le dataset ne quitte pas la machine : `Données/` est exclu de git (voir
  `Données/.gitignore`). Le dépôt est privé, mais « ne pas diffuser » vaut aussi
  pour un hébergeur tiers.

## Un point où le mail et les données divergent

Le mail dit « les capsules ont déjà été recentrées en 0 ». Le fichier s'appelle
`off_centered_capsule.npy`, et le centroïde des nœuds part de `y = 0.400` pour
rejoindre `0.046`. Ce n'est pas une contradiction : la capsule est décentrée
**dans le canal** et migre vers l'axe — c'est de la physique, pas un défaut de
préparation. Ce qui est recentré, c'est le repère du canal, pas la capsule.
Détail traité dans `Code/src/capsules.py` (`recentre=True`).
