# Carte des modèles

## Statut

`DummyBackend` reste utilisé par l'API et la CI. Il ne réalise
aucune inférence et son texte de sortie indique explicitement qu'il s'agit
d'une transcription fictive.

`WhisperBackend` est disponible pour les évaluations locales. Il charge une
révision épinglée de Whisper avec Transformers, lit les WAV localement,
normalise la fréquence d'échantillonnage à 16 kHz et libère le modèle après le
run. Il ne force pas le token `dyu`, absent des langues déclarées par Whisper.
Le tableau de bord Gradio accède à ce backend uniquement à travers les
services et le `ModelRegistry`. Wav2Vec2 XLSR n'est pas encore implémenté.

## Contrat

Tout futur backend doit implémenter `ASRBackend` et retourner le schéma
`TranscriptionResult`. Il doit déclarer ses langues prises en charge et libérer
ses ressources avec `unload()`.

## Modèles évalués

- `openai/whisper-tiny`, révision
  `be0ba7c2f24f0127b27863a23a08002af4c2c279` ;
- `openai/whisper-small`, révision
  `973afd24965f72e36ca33b3055d56a652f456b4d`.

La révision Small choisie contient les poids `safetensors`. Les deux modèles
sont utilisés en transcription multilingue avec détection automatique de la
langue, FP16, batch unitaire et sans chunking expérimental, car tous les
extraits test sont inférieurs à dix secondes.

## Résultats des baselines sur le pilote test historique

Les deux modèles sont comparés sur les mêmes 150 audios privés, soit 50 par
locuteur test :

| Modèle | WER micro | CER micro | RTF | p50 (s) | p95 (s) |
|---|---:|---:|---:|---:|---:|
| Whisper Tiny | 1,1436 | 0,7374 | 0,0296 | 0,1225 | 0,3178 |
| Whisper Small | 1,5235 | 0,8611 | 0,0900 | 0,2533 | 0,9155 |

Aucun des 300 traitements n'a échoué. Ces résultats mesurent une baseline
zéro-shot, pas un modèle adapté au dioula. Tiny est provisoirement recommandé
pour la première expérience d'adaptation parce qu'il obtient de meilleurs
WER/CER tout en étant environ trois fois plus rapide sur ce pilote.

## Adaptation pilote Phase 4C

Whisper Tiny a été adapté pendant une époque sur un sous-ensemble déterministe
de 2 250 audios du train et évalué sur 600 audios du split validation. Le
pilote test historique et le holdout final n'ont été ni chargés ni transcrits.

| Modèle sur validation identique | WER micro | CER micro | RTF |
|---|---:|---:|---:|
| Whisper Tiny non adapté | 1,1545 | 0,7172 | 0,0123 |
| Checkpoint pilote adapté | 0,7822 | 0,3483 | 0,0187 |

La réduction relative atteint 32,25 % pour le WER et 51,44 % pour le CER.
Sur les 600 audios, 461 prédictions ont moins d'erreurs de mots, 89 sont
inchangées et 50 se dégradent. Le checkpoint retenu est
`checkpoint-000140`; il est stocké hors Git et a été rechargé avec succès.

Ces résultats justifient seulement la faisabilité d'une expérimentation plus
large. Ils ne constituent pas une performance finale et ne doivent pas être
comparés directement au pilote test de 150 audios, dont la composition est
différente.

## Full development et refit final

Le développement complet a utilisé les 13 764 audios train et les 2 661
audios validation avec des locuteurs disjoints. Sur les mêmes 2 661 audios de
validation, le checkpoint pilote obtient `WER=0,774152` et `CER=0,361319`, puis
`checkpoint-001720` obtient `WER=0,441286` et `CER=0,183654`.

Cette sélection a fixé un budget de 2 052 updates. Le refit final est reparti
des poids de `checkpoint-000140`, avec optimizer, scheduler et GradScaler
neufs, sur les 16 425 audios train+validation et 18 locuteurs. Le modèle gelé
est `checkpoint-002052`. `checkpoint-001720` n'a servi qu'à déterminer le
budget et n'est pas le modèle final.

## Évaluation finale indépendante

Le modèle final a ouvert une seule fois le final holdout de 2 624 audios et 3
locuteurs. Ce jeu n'a servi ni à l'entraînement, ni à la validation, ni à la
sélection ou au réglage des hyperparamètres.

| Métrique final holdout | Valeur |
|---|---:|
| WER | 0,332625 |
| CER | 0,123804 |
| RTF | 0,007853 |
| Loss finale | 0,346437 |
| Substitutions | 5 690 |
| Insertions | 1 363 |
| Suppressions | 1 864 |
| Correspondances exactes | 334 |

Ces valeurs sont des agrégats de l'unique évaluation. Aucune prédiction,
transcription, identité de locuteur ou référence individuelle n'est publiée.
Elles ne doivent pas être comparées directement aux résultats pilote ou
validation, car les jeux diffèrent. Le modèle et le protocole sont restés
inchangés après lecture du holdout.

## Limites et usage

- les transcriptions, prédictions et checkpoints restent locaux ;
- la licence et le consentement de redistribution du corpus ne sont pas
  confirmés ; ni le dataset ni le modèle dérivé ne sont publiés ;
- le modèle est spécialisé au dioula et au contexte local disponible ;
- les performances peuvent varier selon l'accent, le bruit, le microphone et
  le domaine ;
- le corpus disponible limite la couverture acoustique et linguistique ;
- le final holdout contient seulement 3 groupes de locuteurs ;
- aucune métrique ne garantit une performance identique en production ;
- aucune évaluation française n'a été réalisée ;
- l'API n'utilise pas encore le backend réel ;
- Gradio utilise le backend réel, mais reste un démonstrateur local sans
  authentification ni exposition publique ;
- le baoulé et les autres langues ivoiriennes restent des travaux futurs.
