# Protocole d’entraînement local — Phases 4B et 4C

## Portée

La Phase 4B est un diagnostic de mémorisation de **Whisper Tiny** sur 10 à
20 audios du split `train`. Elle ne constitue ni un fine-tuning complet, ni
une mesure de généralisation. Le split `test` officiel et le pilote de
150 audios sont interdits pour l’entraînement, le choix des hyperparamètres et
les décisions de normalisation.

Le dataset et les modèles dérivés restent limités à
`local_research_only`. Aucune publication n’est autorisée.

## Ordre obligatoire

1. Exécuter l’audit automatisé :

   ```bash
   make audit-dioula-training \
     DIOULA_DATA_DIR="/chemin/absolu/vers/voices_data" \
     ARTIFACTS_DIR="/chemin/absolu/vers/artifacts" \
     MODEL_CACHE_DIR="/chemin/absolu/vers/cache/models"
   ```

2. Démarrer l’outil d’écoute sur l’interface loopback :

   ```bash
   make review-dioula-training \
     DIOULA_DATA_DIR="/chemin/absolu/vers/voices_data" \
     ARTIFACTS_DIR="/chemin/absolu/vers/artifacts" \
     MODEL_CACHE_DIR="/chemin/absolu/vers/cache/models"
   ```

   Ouvrir `http://127.0.0.1:7861`, écouter chaque extrait, puis choisir
   `correct`, `texte partiellement incorrect`, `mauvais alignement`,
   `audio inutilisable` ou `à vérifier`. Au moins 10 audios doivent être
   réellement marqués `correct`.

3. Lancer le smoke-overfit seulement après cette validation :

   ```bash
   make smoke-overfit-dy \
     DIOULA_DATA_DIR="/chemin/absolu/vers/voices_data" \
     ARTIFACTS_DIR="/chemin/absolu/vers/artifacts" \
     MODEL_CACHE_DIR="/chemin/absolu/vers/cache/models"
   ```

## Garde-fous

- La configuration exige `mode: smoke_overfit`, `split: train` et
  `openai/whisper-tiny`.
- Toute configuration utilisant `validation` ou `test` est rejetée.
- Les `speaker_id` et `audio_sha256` doivent être disjoints entre les splits.
- Les identifiants du pilote doivent tous appartenir au test.
- La sélection smoke est recroisée avec les identifiants et hashes du test
  juste avant le chargement du modèle.
- Le runner refuse de démarrer sans au moins 10 validations auditives
  `correct`.
- Une loss non finie provoque un arrêt immédiat.
- Les poids ne sont jamais sauvegardés pendant ce diagnostic.

## Normalisation

`target_text_mvp` est la colonne canonique du smoke-overfit. Elle représente
la variante NFC sans tons et sans marque d’intonation `↘`. Ce choix limite la
sparsité orthographique sur le très petit échantillon. Les variantes
`text_raw`, `text_with_tones_nfc`, `text_without_tones_nfc` et
`target_text_mvp` restent toutes conservées dans le manifeste gelé.

## Reproductibilité et sorties

- seed fixe : `42` ;
- modèle et révision Hugging Face épinglés dans
  `configs/models/whisper_tiny.yaml` ;
- 16 audios représentatifs au maximum, puis uniquement ceux confirmés
  `correct` ;
- 80 steps, batch 4, AdamW, clipping de gradient ;
- CPU supporté en float32 ; CUDA utilise la précision mixte disponible ;
- masque d'attention audio transmis explicitement ;
- attention `eager` et algorithmes PyTorch déterministes pour le smoke CUDA ;
- historique complet dans
  `artifacts/training/smoke_overfit_whisper_tiny_dy/loss_history.csv` ;
- métriques agrégées dans `metrics.json` ;
- prédictions privées avant/après dans `predictions_private.csv`.

Les livrables locaux consolidés sont écrits dans `reports/training/` :

- `smoke_overfit_metrics.json` ;
- `smoke_overfit_report.md` ;
- `smoke_overfit_loss.csv` ;
- `smoke_overfit_loss.png`.

Le rapport Markdown versionné est strictement agrégé. Les prédictions,
références et variantes textuelles restent uniquement dans
`artifacts/training/smoke_overfit_whisper_tiny_dy/predictions_private.csv`.

Le rapport Markdown de validation et les annotations sont enregistrés dans
`reports/data/`. Ces deux fichiers privés sont ignorés par Git car le premier
contient des transcriptions. Les rapports agrégés de normalisation et
d’intégrité sont également générés dans ce dossier à partir du manifeste gelé,
sans chemin, identifiant ni transcription privés.

## Critères du diagnostic

Le smoke test est réussi si la loss reste finie, baisse d’au moins 20 % entre
les fenêtres initiale et finale, et si le WER ou le CER calculé **sur le même
micro-train** s’améliore. Ces métriques mesurent seulement la capacité de
mémorisation et ne doivent jamais être présentées comme des performances de
généralisation.

Si CUDA est indisponible, le runner fonctionne sur CPU. Le notebook Colab
n’est donc nécessaire que lorsque le temps CPU local empêche réellement
l’exécution ; dans ce cas il devra installer et importer le package du dépôt,
sans recopier le pipeline.

## Résultat du smoke-overfit Phase 4B

Le run local seed 42 sur les 16 audios du train validés auditivement a réussi :

- modèle `openai/whisper-tiny`, révision
  `be0ba7c2f24f0127b27863a23a08002af4c2c279` ;
- 80 steps, batch 4, learning rate `1e-4` ;
- première/dernière loss : `4.958414` / `0.092581` ;
- loss moyenne des 10 premiers/derniers steps :
  `2.385014` / `0.021900`, soit une réduction de `99.08 %` ;
- WER micro-train : `0.974026` avant, `0.006494` après ;
- CER micro-train : `0.500000` avant, `0.003125` après ;
- aucune loss non finie, aucun crash et aucune fuite de split ;
- durée d'entraînement : `11.616 s` ; durée totale : `27.235 s` ;
- pic mémoire CUDA : `2752.85 MiB`.

Ces chiffres démontrent uniquement la capacité du pipeline à mémoriser le
micro-train. Ils ne mesurent pas la généralisation et n'autorisent pas le
lancement d'une Phase 4C sans décision explicite.

## Phase 4C — pilote train/validation

La Phase 4C utilise exclusivement :

- 2 250 audios du train, couvrant les 15 locuteurs ;
- 600 audios de validation, soit 200 pour chacun des 3 locuteurs ;
- `target_text_mvp` comme référence canonique ;
- le même sous-ensemble validation pour la baseline et le modèle adapté.

Le test est partitionné uniquement par ses métadonnées anonymisées :
`pilot_test` contient les 150 identifiants historiques et `final_holdout` les
2 624 autres. Aucun audio de ces deux partitions n'est décodé ou transcrit
pendant la Phase 4C.

La commande locale est :

```bash
make pilot-finetune-dy \
  DIOULA_DATA_DIR="/chemin/absolu/vers/voices_data" \
  ARTIFACTS_DIR="/chemin/absolu/vers/artifacts" \
  MODEL_CACHE_DIR="/chemin/absolu/vers/cache/models" \
  CHECKPOINT_DIR="/chemin/absolu/vers/checkpoints"
```

Le pilote utilise une époque, un batch CUDA de 4, une accumulation de 4,
FP16, gradient checkpointing non-reentrant, warmup 5 %, weight decay 0,01,
évaluation tous les 35 steps et early stopping de patience 2. Les checkpoints
reprenables sont stockés hors Git et seuls le meilleur et le plus récent sont
conservés.

### Résultat Phase 4C

Le run seed 42 a exécuté 141 steps sans NaN, Inf, erreur CUDA ni step optimizer
fp16 ignoré. Le meilleur checkpoint `checkpoint-000140` a été rechargé avec
succès.

| Métrique validation | Whisper Tiny baseline | Meilleur checkpoint |
|---|---:|---:|
| WER micro | 1,154527 | 0,782165 |
| CER micro | 0,717216 | 0,348294 |
| Validation loss | 5,267802 | 1,229892 |
| RTF | 0,012340 | 0,018742 |
| Latence moyenne (s) | 0,054136 | 0,082223 |

- réduction WER absolue : `0,372362` ;
- réduction WER relative : `32,25 %` ;
- réduction CER absolue : `0,368922` ;
- réduction CER relative : `51,44 %` ;
- prédictions avec moins/autant/plus d'erreurs de mots :
  `461 / 89 / 50` ;
- durée du train incluant les validations régulières : `782,980 s` ;
- pic VRAM : `1 957,83 MiB` ;
- taille du meilleur checkpoint : `454 359 808` octets.

Ces résultats mesurent une amélioration sur validation, pas sur le test final.
Le modèle adapté reste local et non publiable. La Phase 4C ne déclenche pas
l'entraînement complet : celui-ci exige une nouvelle décision explicite.
