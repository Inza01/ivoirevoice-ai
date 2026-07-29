# Protocole expérimental

## Règles

1. Figer la version des données, de la configuration et du code.
2. Séparer entraînement, validation et test strictement par locuteur.
3. Ne sélectionner les hyperparamètres qu'avec l'ensemble de validation.
4. Conserver les seeds, journaux, artefacts et versions de dépendances.
5. Évaluer séparément le français et le dioula, puis documenter les sous-groupes
   pertinents lorsque les métadonnées et le consentement le permettent.
6. Publier uniquement des métriques calculées sur des données traçables.

Avant entraînement, le candidat dioula doit contenir un audio unique par ligne,
aucun conflit de transcription et un split validé humainement. Les stratégies
17/2/2, 15/3/3 et une recherche visant 75 % / 12,5 % / 12,5 % de durée doivent
être comparées. La recommandation privilégie zéro fuite, au moins trois
locuteurs en validation et test, la représentation des dossiers `men` et
`women`, puis l'équilibre des durées.

Les métriques ASR et critères d'acceptation seront définis avant les premières
expériences. Aucun résultat n'est disponible pendant la Phase 2.
