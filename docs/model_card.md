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

## Résultats du pilote local

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

## Limites et usage

- les transcriptions et prédictions restent privées et hors Git ;
- la licence du corpus ne permet ni publication du dataset, ni modèle dérivé ;
- les résultats ne doivent pas être généralisés au-delà du split test local ;
- aucun fine-tuning et aucune évaluation française n'ont été réalisés ;
- l'API n'utilise pas encore le backend réel ;
- Gradio utilise le backend réel, mais reste un démonstrateur local sans
  authentification ni exposition publique.
