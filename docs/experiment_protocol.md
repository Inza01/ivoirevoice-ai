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

## Dataset dioula v0.1

La stratégie B validée humainement est figée avec 15/3/3 locuteurs et la seed
42. Avant toute baseline, `make validate-dioula-v01` doit confirmer les
19 199 audios uniques, les 21 locuteurs, l'absence de fuite, les chemins
relatifs et les hashes de provenance.

La cible d'apprentissage v0.1 est `text_without_tones_nfc` sans `↘`. Le texte
brut et les deux variantes NFC restent conservés, avec `intonation_falling`
pour rendre la décision réversible. Aucune expérience ne doit publier le
corpus, le manifeste ou un modèle dérivé tant que la licence et le consentement
restent inconnus.
