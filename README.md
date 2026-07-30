# IvoireVoice AI

IvoireVoice AI est une plateforme **en cours de développement** pour la
transcription vocale adaptée au contexte ivoirien. Le MVP cible le français
(`fr`) et le dioula (`dyu`). Le baoulé est prévu comme extension future, sans
modifier le cœur des interfaces.

Cette version comprend le pipeline de données local, une première baseline
ASR dioula reproductible et un tableau de bord Gradio local. Whisper Tiny et
Whisper Small ont été évalués sans entraînement sur un pilote privé de
150 audios. L'API FastAPI reste branchée sur le backend fictif ; l'interface
Gradio utilise les modèles réels à la demande.

## Périmètre actuel

- package Python sous `src/ivoirevoice/` ;
- configuration YAML surchargeable par variables d'environnement ;
- contrats pour les futurs backends ASR ;
- API FastAPI utilisant `DummyBackend` ;
- interface Gradio de comparaison Tiny/Small à cinq onglets ;
- pipeline reproductible de découverte, audit, curation et gel local dioula ;
- backend Whisper local et modèles épinglés par révision ;
- smoke tests et pilotes déterministes sur le seul split test ;
- WER, CER, métriques par locuteur, latence, RTF et reprise progressive ;
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
Pour préparer un environnement ML :

```bash
.venv/bin/python -m pip install -e ".[core,data,ml,ui]"
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

Les tests ASR utilisent `DummyBackend` ou une pipeline Whisper injectée et
factice. Les tests de données créent de petits WAV et JSON artificiels. Ils ne
lisent pas le corpus réel et ne nécessitent ni connexion internet, ni GPU, ni
modèle téléchargé.

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

La récupération des audios manquants est verrouillée à `false` pendant cette
phase. La curation produit uniquement `missing_audio_recovery_plan.json` et le
gel refuse une configuration qui demanderait une récupération ou conversion.
Une phase future explicitement autorisée devra d'abord lever ce verrou avant
de pouvoir utiliser le module `ivoirevoice.data.recovery`.

## Geler et valider le dataset dioula v0.1

La décision humaine de Phase 3A.2 retient la stratégie B : 15 locuteurs en
entraînement, 3 en validation et 3 en test. Le gel s'appuie sur l'affectation
pseudonymisée déjà enregistrée dans le rapport de comparaison :

```bash
make freeze-dioula-v01
make validate-dioula-v01
```

Le gel écrit, relativement à `IVOIREVOICE_ARTIFACTS_DIR` :

- `manifests/dioula_manifest_v0.1.csv` ;
- `manifests/dioula_dataset_v0.1_metadata.json` ;
- `reports/data_curation/dioula_v0.1_report.md` ;
- `reports/data_curation/dioula_v0.1_split_report.json`.

La cible MVP est la variante NFC sans tons, privée du seul marqueur `↘`. Le
marqueur reste dans `text_raw` et alimente le booléen `intonation_falling`.
Les textes brut, avec tons NFC et sans tons NFC restent inchangés.

La version `0.1.0-local` est immuable : relancer la même configuration est
idempotent, tandis qu'une tentative d'écraser un artefact v0.1 par un contenu
différent échoue. Une modification future exige un nouveau numéro de version.
La validation bloque notamment les fuites de locuteur, doublons audio, chemins
non relatifs, URLs, splits vides et toute autorisation de publication.

## Exécuter les baselines dioula

Définissez également un cache modèle hors du dépôt :

```bash
export IVOIREVOICE_MODEL_CACHE_DIR="/chemin/vers/cache/models"

make check-ml-environment
make inspect-baseline-models
make baseline-dy-smoke MODEL=whisper_tiny
make baseline-dy-pilot MODEL=whisper_tiny
make baseline-dy-smoke MODEL=whisper_small
make baseline-dy-pilot MODEL=whisper_small
make compare-dy-baselines
```

Le niveau `smoke` utilise deux audios courts par locuteur test ; le niveau
`pilot` en utilise 50 par locuteur avec une sélection déterministe couvrant
plusieurs durées. Le niveau complet est verrouillé :

```bash
make baseline-dy-full MODEL=whisper_tiny CONFIRM_FULL=1
```

Il ne doit être lancé qu'après examen du pilote. Les prédictions, références
et analyses textuelles restent privées sous `artifacts/baselines/`. Seules les
synthèses agrégées de `artifacts/reports/baselines/` sont partageables.

Sur le pilote local de 150 audios, Whisper Tiny obtient un WER micro de
114,36 %, un CER de 73,74 % et un RTF de 0,0296. Whisper Small obtient
respectivement 152,35 %, 86,11 % et 0,0900. Les deux modèles sont donc très
insuffisants sans adaptation ; Tiny est retenu pour la première validation de
fine-tuning, qui ne fait pas partie de cette phase.

## Lancer le tableau de bord local

L'interface exige les variables de données, d'artefacts et de cache déjà
utilisées par les baselines :

```bash
make ui \
  DIOULA_DATA_DIR="/chemin/vers/voices_data" \
  ARTIFACTS_DIR="/chemin/vers/artifacts" \
  MODEL_CACHE_DIR="/chemin/vers/cache/models"
```

Elle écoute uniquement sur `127.0.0.1:7860`, sans lien public Gradio. Ses cinq
onglets permettent :

1. de transcrire un enregistrement ou un fichier avec Tiny et Small ;
2. de consulter le benchmark structuré et trois graphiques ;
3. d'analyser localement les erreurs des 150 échantillons privés ;
4. de calculer des métriques personnalisées et exporter JSON/CSV/TXT ;
5. de présenter le projet, son architecture et ses limites.

Le WER et le CER sont calculés uniquement lorsqu'une référence est fournie.
Les modèles sont chargés puis libérés séquentiellement. L'échec de Small ne
supprime pas le résultat de Tiny. Les formats acceptés sont WAV, MP3, M4A et
OGG, avec une limite de 25 Mio et 180 secondes.

Un futur modèle adapté peut être ajouté sans changer le code : ajoutez sa
configuration, renseignez `IVOIREVOICE_FINETUNED_MODEL_CONFIG`, puis activez
son entrée dans `configs/ui/models.yaml`.

Le guide de présentation est disponible dans
[docs/demo_guide.md](docs/demo_guide.md).

## Lancer l'API fictive

API :

```bash
make api
```

Les routes sont `GET /health`, `GET /models` et `POST /transcribe`. La route de
transcription attend un formulaire multipart avec `file` et `language` (`fr`
ou `dyu`).

L'API conserve un texte explicitement fictif. Le branchement du service de
comparaison dans l'API pourra être réalisé après stabilisation de l'interface.

## Documentation

- [Architecture](docs/architecture.md)
- [Carte des données](docs/data_card.md)
- [Carte des modèles](docs/model_card.md)
- [Protocole expérimental](docs/experiment_protocol.md)
