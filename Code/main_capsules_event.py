"""
Combien de photos faut-il pour tenir l'ÉVÉNEMENT de déformation ?

Premier des deux régimes de la séquence du BMBI. Le balayage des 20001 pas
(RESULTATS_capsules_temporel.md, §11) montre que la séquence contient un seul
événement, et qu'il est au début :

    pas 0 -> 2000     la sphère s'écrase (rapport d'axes 1.00 -> 1.35), s'écarte
                      d'un ellipsoïde jusqu'à 6.1 %, développe une concavité ;
                      c'est là que le mouvement est maximal.

Il n'y a PAS d'événement de rotation : le grand axe ne tourne que de 17° sur
toute la séquence, et la rotation spectaculaire qu'on croyait voir au pas 0
était un artefact — sur une sphère parfaite, l'axe principal n'existe pas.

La question n'est donc plus « est-ce que ça marche » (c'est établi dans
RESULTATS_capsules_temporel.md) mais celle qui a encore de la marge, et qui est
la question pratique du BMBI :

    COMBIEN d'instants faut-il fournir pour tenir cet événement,
    et à partir de quand ça casse ?

Tout le protocole est dans `src/campagne.py`, partagé avec la campagne
tank-treading — c'est le seul moyen de garantir que les deux régimes sont
comparés à protocole strictement identique.

Lancer :  /usr/local/bin/python3 main_capsules_event.py
"""
from src.campagne import campagne

if __name__ == "__main__":
    campagne("evenement", pas_min=0, pas_max=2000)
