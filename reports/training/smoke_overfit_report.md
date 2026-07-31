# Smoke-overfit Whisper Tiny dioula — Phase 4B

> Rapport public agrégé, sans identifiant, transcription ni prédiction.
> Aucune conclusion de généralisation.

- Statut : **réussi**
- Audios validés du train : 16
- Test officiel utilisé : False
- Validation utilisée : False
- Pilote utilisé : False
- Modèle : `openai/whisper-tiny`
- Révision : `be0ba7c2f24f0127b27863a23a08002af4c2c279`
- Tâche : `transcribe`
- Langue : `multilingual_without_forced_dyu_token`
- Device : `cuda`
- Seed : 42
- Steps : 80
- Learning rate : 0.0001
- Temps total : 27.235 s
- Mémoire GPU maximale : 2752.85 MiB

## Loss de mémorisation

- Première loss : 4.958414
- Dernière loss : 0.092581
- Moyenne initiale : 2.385014
- Moyenne finale : 0.021900
- Réduction des moyennes : 99.08%

| Step | Loss |
|---:|---:|
| 1 | 4.958414 |
| 10 | 0.867767 |
| 20 | 0.158369 |
| 30 | 0.229863 |
| 40 | 0.133642 |
| 50 | 0.013327 |
| 60 | 0.028415 |
| 70 | 0.028873 |
| 80 | 0.092581 |

## Métriques sur le micro-train uniquement

| Mesure de mémorisation | Avant | Après |
|---|---:|---:|
| WER micro-train | 0.974026 | 0.006494 |
| CER micro-train | 0.500000 | 0.003125 |

Ces valeurs sont calculées sur les mêmes audios que ceux de l'entraînement. Elles mesurent la mémorisation, pas la généralisation.

## Anomalies d'exécution

- NaN/Inf : False
- Crash : False
- Fuite de split : False
