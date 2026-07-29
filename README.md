# IvoireVoice AI

IvoireVoice AI est une plateforme **en cours de développement** pour la
transcription vocale adaptée au contexte ivoirien. Le MVP cible le français
(`fr`) et le dioula (`dyu`). Le baoulé est prévu comme extension future, sans
modifier le cœur des interfaces.

Cette version comprend le squelette logiciel et le pipeline d'audit local du
corpus dioula. Elle utilise uniquement un backend ASR fictif et ne fournit
encore **aucun résultat final, benchmark ou modèle ASR entraîné**.

## Périmètre actuel

- package Python sous `src/ivoirevoice/` ;
- configuration YAML surchargeable par variables d'environnement ;
- contrats pour les futurs backends ASR ;
- API FastAPI et interface Gradio utilisant `DummyBackend` ;
- pipeline reproductible de découverte, manifeste et audit dioula ;
- tests hors ligne, sans GPU et sans téléchargement de modèle.

Les datasets bruts, données préparées et checkpoints ne sont pas versionnés.
Placez-les dans les espaces de stockage prévus par votre environnement ; les
répertoires locaux `data/`, `datasets/`, `artifacts/` et `checkpoints/` sont
ignorés par Git.

## Installation (Python 3.11)

```bash
make setup
make install-dev
```

L'installation de développement n'installe pas les dépendances ML lourdes.
Pour préparer ultérieurement un environnement ML :

```bash
.venv/bin/python -m pip install -e ".[core,ml]"
```

Copiez `.env.example` vers `.env` si nécessaire, puis exportez les variables
avec l'outil de votre choix. Le chargeur lit les variables préfixées par
`IVOIREVOICE_`; le séparateur `__` représente un niveau YAML, par exemple
`IVOIREVOICE_API__MAX_UPLOAD_SIZE_MB=10`.

## Qualité et tests

```bash
make lint
make typecheck
make test
make verify
```

Les tests ASR utilisent exclusivement `DummyBackend` et les tests de données
créent de petits WAV et JSON artificiels. Ils ne lisent pas le corpus réel et
ne nécessitent ni connexion internet, ni GPU, ni modèle téléchargé.

## Auditer le corpus dioula

Les données et artefacts doivent rester hors du dépôt. Le pipeline ne modifie
jamais les fichiers bruts, ignore les MP4, ne conserve aucune URL `audioSrc` et
pseudonymise les identifiants de locuteurs.

Tant que la licence et le consentement ne sont pas confirmés, le corpus dioula
est limité à `local_research_only` : aucune donnée, manifeste complet ou modèle
dérivé ne doit être envoyé vers un service externe ou publié.

```bash
export IVOIREVOICE_DIOULA_DATA_DIR="/chemin/vers/voices_data"
export IVOIREVOICE_ARTIFACTS_DIR="/chemin/vers/artifacts"

make manifest-dioula
make audit-dioula
```

Une surcharge ponctuelle est aussi possible :

```bash
make audit-dioula \
  DIOULA_DATA_DIR="/chemin/vers/voices_data" \
  ARTIFACTS_DIR="/chemin/vers/artifacts"
```

Les principales sorties, relatives à `IVOIREVOICE_ARTIFACTS_DIR`, sont :

- `manifests/dioula_manifest_draft.csv` ;
- `reports/data_audit/dioula_inventory.json` ;
- `reports/data_audit/dioula_summary.json` ;
- `reports/data_audit/dioula_audit.md` ;
- `reports/data_audit/dioula_split_proposal.json` ;
- `reports/data_audit/unmatched_audio.csv` et `ambiguous_audio.csv`.

Le split reste vide dans le manifeste tant que sa proposition n'a pas été
validée humainement. Le hash SHA-256 peut être désactivé avec
`IVOIREVOICE_HASH_AUDIO=false`.

## Curater le candidat d'entraînement dioula

Après génération du manifeste d'audit :

```bash
make curate-dioula
make compare-dioula-splits
```

La curation conserve une ligne par chemin audio, puis une ligne par SHA-256.
Elle met en quarantaine tout audio associé à des transcriptions normalisées
différentes. Un `sentence_id` répété avec des audios distincts est conservé.

Le candidat local est écrit dans
`manifests/dioula_manifest_curated_candidate.csv`. Il conserve :

- `text_raw` inchangé ;
- `text_with_tones_nfc` ;
- `text_without_tones_nfc` ;
- `target_text_mvp`, égal pour le premier MVP à la variante sans tons.

Ce choix sans tons est une simplification technique réversible, pas une
décision linguistique. Les rapports de curation et de comparaison des splits
sont écrits sous `reports/data_curation/`, toujours relativement à
`IVOIREVOICE_ARTIFACTS_DIR`.

La récupération des audios manquants reste désactivée par défaut. La curation
produit uniquement `missing_audio_recovery_plan.json`. Toute conversion exige
à la fois `recover_missing_audio: true`, une destination externe configurée et
l'option explicite `--execute` du module `ivoirevoice.data.recovery`.

## Lancer les interfaces fictives

API :

```bash
make api
```

Les routes sont `GET /health`, `GET /models` et `POST /transcribe`. La route de
transcription attend un formulaire multipart avec `file` et `language` (`fr`
ou `dyu`).

Interface Gradio :

```bash
make ui
```

Le texte affiché est explicitement fictif. Les implémentations Whisper et
Wav2Vec2 ne font pas partie de cette phase.

## Documentation

- [Architecture](docs/architecture.md)
- [Carte des données](docs/data_card.md)
- [Carte des modèles](docs/model_card.md)
- [Protocole expérimental](docs/experiment_protocol.md)
