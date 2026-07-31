# Rapports publiables

Ce dossier contient uniquement des résultats agrégés nécessaires à la
reproductibilité du MVP : métriques, courbes de loss, historique numérique,
contrôles d'intégrité et estimations de ressources.

Les fichiers publiés ici ne doivent contenir aucun audio, chemin local, nom
réel, transcription, référence, prédiction individuelle ou identifiant
d'échantillon. Les rapports de sélection sont réduits à des statistiques de
groupe.

Les sorties détaillées restent sous `IVOIREVOICE_ARTIFACTS_DIR`, hors Git :

- prédictions avant/après ;
- références textuelles ;
- rapports d'écoute manuelle ;
- identifiants des sélections train/validation/test ;
- checkpoints et caches de modèles.

Les deux expériences de benchmark restent séparées :

- validation pilote de 600 audios : Tiny baseline contre Tiny adapté ;
- pilote historique de 150 audios : Tiny baseline contre Small baseline.

Les métriques du pilote adapté ne constituent pas une évaluation finale.
