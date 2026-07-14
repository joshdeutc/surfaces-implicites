Mail de Florian De Vuyst — lundi (juin 2026, avant ses congés d'été)
Objet : TX Surfaces implicites
Pour : Joshua Nancey, Nathan Jacas

C'est le mail avec les recommandations de travail pour l'été.

---

Bonjour Joshua, Nathan,

Je ne vais malheureusement pas avoir le temps de préparer une présentation correcte avant mon départ en vacances,
Je pars en fin de semaine et j'ai des dossiers prioritaires avant de partir (doctorants, rapports …),

Au moins je peux vous faire part de recommandations pour vos travaux personnels :

Aidez-vous de Chatgpt ou Claude qui deviennent des outils très puissant en Sciences : ils vous ferons gagner du temps, notamment sur la bibliographie et l'état de l'art.

Génération de données : la surface d'intérêt est donnée par l'équation implicite f(x,y,z) = 0. Néanmoins les données d'entrée peuvent être volumiques : vous avez des points (xi, yi, zi) et des valeurs f_i . Si f_i = 0, vous êtes sur la surface, si fi<0 vous êtes à l'intérieur du volume délimité par la surface, si fi>0 , vous êtes à l'exterieur. Dans plusieurs articles, les auteurs utilisent la distance signée (Signed distance function, SDF). Elle et très pratique mais peut être coûteuse car le calcul de la distance d'un point à une surface est complexe. Si la surface est discrétisée avec N points, alors il faut calculer le minimum de distance sur les N points. Si vous avez M points au total, ça va demander (MN) évaluations. Une fois que vous avez les points (xi,yi,zi) et les valeurs fi, vous pouvez apprendre un Neural Network qui détermine f. Vous pouvez représenter ensuite la surface par marching cubes en utilisant f issu du réseau de neurones. Vous pouvez commencer par des géométries simples, puis par quelques fonctions analytiquement f(x,y,z) comme je vous avais donné dans quelques notebooks.

Plus difficile : dans beaucoup de cas en ingénierie, on possède seulement un nuage de points qui échantillonnent la surface. A ce nuage de points, on peut réaliser un maillage surfacique triangulaire. Pour chaque triangle K, on numérote les sommets du triangle K dans le sens trigonométrique. Il est facile alors de calculer un vecteur normal au triangle, qui pointe vers l'exterieur de la surface. C'est une information supplémentaire qui vous permet partiellement de discriminer l'intérieur de l'exterieur du volume délimité par la surface. Vous refaites un maillage volumique , et, par des considérations de produit scalaire, en cherchant le point de la surface de minimum de distance, on peut déterminer si un point est à l'intérieur ou à l'exterieur.

La distance signée n'est pas forcément la meilleure fonction. D'un côté elle est robuste car ||grad(f)||=1 partout, mais elle peut être irrégulière (avec des pointes), ce qui rend difficile l'apprentissage. Une recherche d'une fonction implicite sous une forme particulière, par exemple un polynôme, aurait beaucoup d'interets : à la fois en terme de compression d'information, mais aussi en simplicité d'évaluation ou en explicabilité.

Le Graal serait d'étudier des surfaces qui évoluent dans le temps. C'est beaucoup plus compliqué, vous pouvez vous le réserver pour la fi si vous avez bien avancé avant.

J'ai oublie de mentionner : les surfaces implicites sont très pratiques pour faire du morphing de formes : par exemple, pour une forme f1(x,y,z) = 0 et une autre f2(x,y,z) = 0, un forme interpolée sera donnée par theta f1(x,y,z) + (1-theta)f2(x,y,z) = 0, avec theta\in [0,1]. On peut aussi définir des opérations d'union, d'intersection ou de différences de formes, très facilement. D'où toute la puissance des surfaces implicites.

Je reviendrai devant un PC vers le 18-18 août, ensuite je reprends le travail le 24 août,
N'hésitez pas à m'envoyer par mail néanmoins vos questions, sinon vos points d'avancement et de blocage,

Très bonne période estivale,

Bien cordialement,
Florian De Vuyst

--
Florian De Vuyst
Professeur des Universités
Laboratoire BMBI UMR 7338, UTC, CNRS
