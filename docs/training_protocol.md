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

## Fine-tuning complet — développement, refit et holdout scellé

Le workflow complet est volontairement découpé en commandes indépendantes.
Aucune commande n’enchaîne automatiquement l’étape suivante.

1. `make full-finetune-preflight` valide le dépôt propre, les 13 764 audios
   train, les 2 661 audios validation, leurs empreintes, le checkpoint pilote,
   l’espace disque et CUDA/FP16. Il recharge `checkpoint-000140` et effectue
   une inférence de contrôle.
2. `make full-finetune-fp16-diagnostic` est un diagnostic numérique borné et
   non reprenable. Il repart de `checkpoint-000140` avec des états neufs,
   exécute au plus 32 tentatives d’optimizer sur `train` uniquement et ne crée
   ni checkpoint ni modèle. Il n’ouvre pas le pilote historique et ne construit
   aucune ligne validation, test ou final-holdout. Un skip AMP est journalisé,
   n’avance ni le scheduler ni le compteur de steps réussis, puis le diagnostic
   continue. Cette politique est strictement locale au diagnostic.
3. `make full-finetune-dev` repart des poids pilotes avec un optimiseur et un
   scheduler neufs. Le développement utilise uniquement train, avec validation
   aux quarts d’époque, jusqu’à 1 722 steps et early stopping de patience 3.
4. Le meilleur step est gelé selon WER micro, CER micro, loss validation puis
   step le plus précoce. Le budget refit vaut
   `floor((best_step / 861) × 1027 + 0,5)`.
5. `make full-finetune-refit` repart à nouveau des poids pilotes avec des états
   d’optimisation neufs, puis utilise exactement les 16 425 audios
   train+validation. Aucune validation n’est décodée pendant le refit.
6. L’unique évaluation finale exige explicitement :

   ```bash
   make evaluate-final-holdout-dy \
     CONFIRM_FINAL_HOLDOUT=EVALUATE_FROZEN_MODEL_ONCE
   ```

Toutes les commandes exigent les mêmes variables locales que la Phase 4C :
`DIOULA_DATA_DIR`, `ARTIFACTS_DIR`, `MODEL_CACHE_DIR`, `CHECKPOINT_DIR` et
`DIOULA_PILOT_MODEL_PATH`.

### Diagnostic FP16 train-only

Le rapport unique est écrit hors Git sous
`artifacts/training/full_finetune_whisper_tiny_dy/fp16_diagnostic/` et un
répertoire existant provoque un arrêt afin de ne jamais écraser une observation
antérieure. `fp16_diagnostic_summary.json` contient les statistiques numériques
par tentative et uniquement des hashes HMAC irréversibles produits avec une clé
éphémère non persistée. Il ne contient ni transcription, texte, speaker ID,
audio ou chemin local.

La finitude des gradients est observée après `GradScaler.unscale_` et avant le
clipping. Le clipping conserve `max_grad_norm=1.0`. La baisse de scale après
`GradScaler.update()` reste le signal public d’un optimizer step ignoré. Le
diagnostic classe ensuite l’observation comme calibration initiale, skip
occasionnel, overflows répétés, signal lié aux caractéristiques techniques des
batches ou autre instabilité numérique. Une seule baisse de scale n’est pas
considérée comme un échec.

La prise en charge BF16 n’est pas activée. Sa capacité matérielle doit seulement
être vérifiée sur l’hôte GPU, sans modifier le protocole :

```bash
.venv/bin/python -c \
  'import torch; print(torch.cuda.is_bf16_supported())'
```

Après versionnement de l’implémentation et nouveau préflight correspondant au
même commit/configuration, le diagnostic se lance explicitement avec :

```bash
make full-finetune-fp16-diagnostic \
  IVOIREVOICE_DIOULA_PILOT_MODEL_PATH="/chemin/vers/checkpoint-000140" \
  IVOIREVOICE_DIOULA_DATA_DIR="/chemin/vers/voices_data"
```

Le vrai `full-finetune-dev` applique désormais la politique validée : tout skip
AMP est compté séparément et ne fait avancer ni scheduler, ni step réussi,
ni validation ou checkpoint périodique. Le batch ignoré n’est pas rejoué et le
parcours continue avec le groupe suivant. Quatre skips consécutifs sont tolérés ;
le cinquième arrête le run avec un diagnostic numérique. Ce seuil est une
politique de sécurité IvoireVoice, pas une règle générale de PyTorch.

Le diagnostic hôte validé a observé une calibration initiale : une tentative
ignorée à la scale 65536, puis 31 updates réussies et stables à 32768. Le GPU
prend en charge BF16, mais la précision retenue pour le full training reste FP16
car cette observation ne démontre aucune instabilité persistante. BF16 n’est
donc ni activé ni injecté automatiquement.

Les checkpoints récupérables exigent `scaler.pt` en plus du modèle, de
l’optimiseur, du scheduler et de l’état d’entraînement. Une reprise restaure la
scale et refuse les compteurs AMP incohérents ; elle ne recrée jamais
arbitrairement une scale 65536 lorsqu’une scale 32768 a été sauvegardée. Les
rapports agrégés conservent `precision`, les scales initiale/finale, les nombres
de tentatives, updates réussies et skips, ainsi que le maximum de skips
consécutifs observé, sans données individuelles.

### Invariants complets

- géométrie : 861 optimizer steps/époque en développement, 1 027 en refit ;
- seed 42, batch CUDA 4, accumulation 4, LR `1e-5`, warmup 5 %, FP16 ;
- `global_step` désigne exclusivement une update optimizer réellement exécutée ;
  `optimizer_attempts`, `successful_optimizer_steps`, `amp_skipped_steps` et
  `consecutive_amp_skips` restent distincts et persistés ;
- le dernier groupe d’accumulation est divisé par son nombre réel de
  micro-batches ;
- checkpoint pilote : poids seulement au début de chaque phase ; son
  optimiseur, scheduler, scaler et état `completed` sont ignorés ;
- reprise autorisée uniquement si code, configuration, manifeste, sélection
  et checkpoint initial ont les mêmes empreintes ;
- checkpoints et rapports privés restent sous les racines externes fournies ;
- les rapports partageables ne contiennent que des agrégats et restent eux
  aussi hors Git jusqu’à une revue humaine ;
- les 150 audios du pilote historique ne sont jamais mélangés au
  `final_holdout` de 2 624 audios ;
- l’évaluation finale compare Tiny original, le pilote et le refit sur la même
  sélection, séquentiellement et depuis le cache local ;
- dès que le reçu final est créé, tout nouveau développement ou refit est
  refusé, y compris après un résultat négatif ;
- licence inconnue : corpus, modèle, checkpoints et prédictions restent
  `local_research_only`.

L’absence de CUDA est un arrêt attendu, pas un motif de fallback CPU ou Colab.
L’implémentation seule ne constitue ni un entraînement exécuté ni une
évaluation du holdout.
