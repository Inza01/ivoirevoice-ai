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

Les métriques ASR sont définies avant inférence : WER et CER micro, moyennes
macro par locuteur, latences moyenne/p50/p95, RTF, taux d'échec, durée et temps
total. Une baseline peut être techniquement valide même avec un WER supérieur
à 100 %, notamment lorsque les insertions sont nombreuses.

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

## Phase 4A — baseline sans entraînement

Les baselines utilisent exclusivement le split test gelé. Le smoke sélectionne
les deux extraits les plus courts de chacun des trois locuteurs. Le pilote
sélectionne, avec la seed 42, 50 extraits par locuteur couvrant la distribution
des durées. Tiny et Small partagent le même hash de sélection.

Les modèles sont chargés localement en FP16 sur CUDA, sans forcer `dyu` comme
token de langue et sans envoyer d'audio à une API. Les poids sont épinglés :

- `openai/whisper-tiny` :
  `be0ba7c2f24f0127b27863a23a08002af4c2c279` ;
- `openai/whisper-small` :
  `973afd24965f72e36ca33b3055d56a652f456b4d`.

Résultats réels du pilote de 150 audios :

| Modèle | WER micro | CER micro | WER macro | CER macro | RTF | Échecs |
|---|---:|---:|---:|---:|---:|---:|
| Whisper Tiny | 114,36 % | 73,74 % | 114,18 % | 73,59 % | 0,0296 | 0 |
| Whisper Small | 152,35 % | 86,11 % | 151,19 % | 85,64 % | 0,0900 | 0 |

Sur le matériel diagnostiqué, les estimations issues du pilote sont d'environ
5 min 55 s pour Tiny et 17 min 59 s pour Small sur les 2 774 audios test.
L'évaluation complète n'a pas été lancée.

Tiny domine ici Small en WER, CER et vitesse. Il est donc retenu pour une
première validation d'adaptation contrôlée. Cette décision est limitée à ce
corpus privé et à cette sélection ; aucun fine-tuning n'est réalisé en
Phase 4A.

## Phase 5B — démonstrateur avec adaptation pilote

Les deux baselines et Tiny adapté pilote sont accessibles dans une interface
locale de comparaison. Le tableau de bord ne recalcule pas les benchmarks :
il charge leurs agrégats vérifiés et maintient deux expériences distinctes.

- Expérience A : Tiny baseline contre Tiny adapté sur les mêmes 600 audios de
  validation ;
- Expérience B : Tiny baseline contre Small baseline sur les 150 audios du
  pilote test historique.

Une comparaison personnalisée suit ce protocole :

1. valider l'extension, la taille, la durée et le décodage de l'audio ;
2. dériver un identifiant anonymisé depuis le SHA-256 ;
3. charger un seul backend depuis le registre ;
4. transcrire puis libérer immédiatement le backend ;
5. poursuivre avec le modèle suivant même si le premier échoue ;
6. calculer WER/CER uniquement lorsqu'une référence est fournie ;
7. exporter sans chemin audio local.

L'analyse des erreurs peut afficher localement les références et prédictions
privées de validation depuis `IVOIREVOICE_ARTIFACTS_DIR`. Elle n'utilise
jamais leur chemin comme libellé et ces fichiers ne sont pas versionnés.
L'interface reste liée à `127.0.0.1`, ne crée aucun lien public et n'ajoute
aucune nouvelle métrique expérimentale.

Le checkpoint `checkpoint-000140` reste hors Git. L'interface le résout via
`IVOIREVOICE_DIOULA_PILOT_MODEL_PATH`, le charge séquentiellement et ne
remplace jamais les deux baselines. Le modèle reste un pilote : le
`final_holdout` n'a pas été évalué.

## Évaluation finale one-time du refit gelé

L'évaluation finale n'est ni un benchmark multi-modèle ni une nouvelle phase
de sélection. Seul le checkpoint refit final, déjà gelé avant toute ouverture,
peut traiter les 2 624 éléments restants du test. La baseline, le pilote et les
checkpoints development n'accèdent pas à ce jeu.

Un préflight sans lecture des références verrouille les empreintes et crée
l'état `SEALED`. L'ouverture confirmée consomme immédiatement l'unique compteur
et passe à `EVALUATION_IN_PROGRESS`; un succès devient `EVALUATED`, tandis
qu'un échec après accès devient définitivement
`EVALUATION_FAILED_AFTER_ACCESS`. Aucun retry automatique n'est permis.

Le calcul conserve seulement des compteurs cumulés. Le résultat final contient
WER, CER, RTF, loss, erreurs d'édition, dénominateurs, nombres d'audios et de
locuteurs et provenance du runtime. Aucune référence, prédiction, identité ou
chemin individuel n'est écrit. Les métriques ne peuvent déclencher aucun
réentraînement, réglage ou second passage.
