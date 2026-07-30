# Protocole d’entraînement local — Phase 4B

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
- historique complet dans
  `artifacts/training/smoke_overfit_whisper_tiny_dy/loss_history.csv` ;
- métriques agrégées dans `metrics.json` ;
- prédictions privées avant/après dans `predictions_private.csv`.

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
