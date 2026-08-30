"""
Combien de photos faut-il quand la forme est figée mais que la MATIÈRE tourne ?

Second régime de la séquence du BMBI, et le plus intéressant pour la méthode.

    pas 4000 -> 20000   la forme relaxe très lentement (rapport d'axes 1.20 ->
                        1.045), mais 93 % du mouvement des nœuds est TANGENTIEL :
                        la membrane tourne autour d'une forme qui bouge à peine.
                        C'est le régime de tank-treading.

Pourquoi c'est le test qui compte. Une séquence de maillages lagrangiens doit
stocker la position de 2562 nœuds à chaque pas sauvegardé — donc l'essentiel de
son information décrit du glissement qui ne déplace PAS la surface. Un champ
implicite ne porte que la forme : par construction, il jette ce glissement.

Si le champ tient ces 16000 pas avec une poignée d'instants, l'argument pour la
représentation implicite n'est plus « ça marche aussi bien » mais « ça représente
la bonne chose ». C'est ce que ce script mesure.

Protocole strictement identique à `main_capsules_event.py` : même module
`src/campagne.py`, même réseau, même perte, mêmes époques, même graine, mêmes
étalons, même plancher de grille. Seule la fenêtre change.

Lancer :  /usr/local/bin/python3 main_capsules_tanktread.py
"""
from src.campagne import campagne

if __name__ == "__main__":
    campagne("tanktread", pas_min=4000, pas_max=20000)
