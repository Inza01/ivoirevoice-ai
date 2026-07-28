# IvoireVoice AI

IvoireVoice AI est une plateforme **en cours de développement** pour la
transcription vocale adaptée au contexte ivoirien. Le MVP cible le français
(`fr`) et le dioula (`dyu`). Le baoulé est prévu comme extension future, sans
modifier le cœur des interfaces.

Cette version est un squelette logiciel : elle utilise uniquement un backend
fictif et ne fournit encore **aucun résultat final, benchmark ou modèle ASR
entraîné**.

## Périmètre actuel

- package Python sous `src/ivoirevoice/` ;
- configuration YAML surchargeable par variables d'environnement ;
- contrats pour les futurs backends ASR ;
- API FastAPI et interface Gradio utilisant `DummyBackend` ;
- tests hors ligne, sans GPU et sans téléchargement de modèle.

Les datasets bruts, données préparées et checkpoints ne sont pas versionnés.
Placez-les dans les espaces de stockage prévus par votre environnement ; les
répertoires locaux `data/` et `checkpoints/` sont ignorés par Git.

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

Les tests utilisent exclusivement `DummyBackend`. Ils ne nécessitent ni
connexion internet, ni GPU, ni modèle téléchargé.

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

