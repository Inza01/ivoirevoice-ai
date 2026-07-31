# IvoireVoice AI

Local, reproducible automatic speech recognition experiments for French and
Dioula in the Ivorian context.

IvoireVoice AI provides a Gradio comparison dashboard, a speaker-disjoint data
pipeline, Whisper baselines, a bounded Dioula adaptation pilot, ASR metrics
and privacy-aware exports. The project is a research MVP, not an
industrial-ready transcription service.

> **Résumé en français —** IvoireVoice AI est un démonstrateur local de
> transcription français–dioula. Il compare Whisper Tiny, Whisper Small et un
> Whisper Tiny adapté sur des données dioula locales. Le corpus réel, les
> transcriptions individuelles et le checkpoint ne sont pas publiés.

## Problem

General-purpose ASR models have limited coverage of low-resource West African
languages and local acoustic conditions. This project explores whether a
small, reproducible Whisper adaptation can improve Dioula transcription while
keeping:

- speaker identities and audio private;
- train, validation and test speakers strictly separated;
- experimental claims limited to the evaluated data;
- model loading and execution possible on a local machine.

## Objectives

- inventory and validate a local Dioula speech corpus;
- build deterministic speaker-disjoint splits;
- compare public Whisper Tiny and Whisper Small baselines;
- adapt Whisper Tiny on a bounded train/validation pilot;
- compare baseline and adapted outputs in a local interface;
- report WER, CER, latency, RTF and edit counts;
- keep the architecture extensible to Baoulé and other Ivorian languages.

## MVP features

- upload or record WAV, MP3, M4A or OGG audio;
- sequential comparison of three ASR configurations;
- optional reference-aware WER and CER;
- substitutions, insertions and deletions;
- latency and real-time factor;
- JSON, CSV and TXT exports without local audio paths;
- separate benchmark views for non-comparable experiments;
- local error analysis backed by private artifacts;
- FastAPI health and demonstration endpoints;
- offline unit tests using `DummyBackend` or injected model doubles.

## Models actually used

| Model | Role | Status |
|---|---|---|
| [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | Primary public baseline and adaptation base | Used |
| [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | Public comparison baseline | Used |
| Whisper Tiny Dioula pilot, `checkpoint-000140` | Locally adapted pilot | Used locally, not distributed |

Only Whisper Tiny was fine-tuned. Wav2Vec2, XLS-R and MMS are possible future
research directions; they were **not implemented or evaluated in this MVP**.
Model usage remains subject to each upstream model and library license.

## Architecture

```text
Gradio
  |
ComparisonService  -- isolates failures and preserves partial results
  |
TranscriptionService -- validates audio and runs one model at a time
  |
ModelRegistry -- lazy factories configured in YAML
  |
WhisperBackend -- load / transcribe / unload
  |
EvaluationService ---- WER, CER and edit counts
ExportService -------- JSON, CSV, TXT and temporary previews
```

The adapted checkpoint path is resolved only at runtime. No private model path
is embedded in the code or rendered UI. See
[`docs/architecture.md`](docs/architecture.md) for the full data and service
flows.

## Data pipeline

The local pipeline performs:

1. read-only discovery of audio and transcription metadata;
2. decoding checks and streamed SHA-256 hashing;
3. text normalization with raw and tone-preserving variants retained;
4. path and audio-hash deduplication;
5. quarantine of conflicting or invalid rows;
6. deterministic splitting by speaker;
7. immutable local manifest validation.

The inventoried corpus contains 19,199 unique audio files across 21
speaker-derived groups. The frozen local strategy uses 15 train, 3 validation
and 3 test groups. A repeated `speaker_id` or `audio_sha256` across splits is a
blocking error.

`target_text_mvp` is the current training target: NFC, without tone marks and
without the falling-intonation marker. Raw and alternative text variants
remain available locally so this choice is reversible.

The real corpus is excluded. A synthetic schema example is available at
[`examples/sample_manifest.csv`](examples/sample_manifest.csv), and the local
data contract is documented in [`data/README.md`](data/README.md).

## Fine-tuning methodology

The Phase 4C pilot used:

- base model: `openai/whisper-tiny`, pinned revision
  `be0ba7c2f24f0127b27863a23a08002af4c2c279`;
- 2,250 train audios from 15 train speakers;
- 600 validation audios from 3 validation speakers;
- one epoch and 141 optimizer steps;
- learning rate `1e-5`;
- batch size 4, gradient accumulation 4, effective batch 16;
- FP16 and non-reentrant gradient checkpointing;
- fixed seed 42;
- task `transcribe`, without forcing an unsupported `dyu` language token;
- best validation checkpoint: `checkpoint-000140`.

The historical 150-audio test pilot and the final holdout were not used for
training or hyperparameter selection. The full training phase has not been
launched.

## Metrics

- **WER**: word-level substitutions, insertions and deletions divided by the
  reference word count.
- **CER**: the corresponding character-level error rate.
- **RTF**: processing time divided by audio duration.

Lower WER, CER and RTF are better. WER can exceed 100% when a model produces
many insertions.

## Pilot results

Both models below were evaluated on the **same 600 validation audios and
references**:

| Model | Dataset | WER | CER | RTF |
|---|---|---:|---:|---:|
| Whisper Tiny baseline | 600 validation audios | 1.154527 | 0.717216 | 0.012340 |
| Whisper Tiny adapted pilot | 600 validation audios | 0.782165 | 0.348294 | 0.018742 |

- WER relative reduction: **32.25%**
- CER relative reduction: **51.44%**

These are pilot validation results, not final test or industrial performance.
The adapted model is still a pilot.

### Separate historical baseline experiment

A different experiment used 150 historical test-pilot audios:

| Model | Dataset | WER | CER | RTF |
|---|---|---:|---:|---:|
| Whisper Tiny baseline | 150 historical pilot audios | 1.1436 | 0.7374 | 0.0296 |
| Whisper Small baseline | 150 historical pilot audios | 1.5235 | 0.8611 | 0.0900 |

These values must not be mixed with the 600-audio validation benchmark.

## Repository structure

```text
.
├── configs/                 # Data, model, experiment and UI YAML
├── data/README.md           # Public local-data contract
├── docs/                    # Architecture, cards and protocols
├── examples/                # Synthetic manifest only
├── reports/                 # Aggregate, privacy-reviewed results
├── scripts/                 # Lightweight environment validation
├── src/ivoirevoice/
│   ├── api/                 # FastAPI adapter
│   ├── data/                # Audit, curation, manifest and split logic
│   ├── evaluation/          # Metrics and baseline evaluation
│   ├── models/              # Backend contract, registry and Whisper
│   ├── services/            # Comparison, evaluation and export services
│   ├── training/            # Smoke and bounded pilot workflows
│   └── ui/                  # Gradio application
├── tests/                   # Offline unit and integration tests
├── .env.example
├── Makefile
└── pyproject.toml
```

## Requirements

- Python 3.11;
- enough disk space for public Whisper caches and local checkpoints;
- CUDA is required only for reproducing the Phase 4C pilot configuration;
- CPU is supported for lightweight checks, tests and the smoke runner.

Dependency ranges are maintained in `pyproject.toml`. The convenience files
`requirements.txt` and `requirements-dev.txt` reference those extras instead
of duplicating a full `pip freeze`.

The publication checks were executed in this local environment:

| Component | Tested version |
|---|---:|
| Python | 3.11.15 |
| PyTorch | 2.10.0+cu130 |
| Transformers | 4.57.6 |
| Gradio | 5.50.0 |
| FastAPI | 0.140.13 |
| NumPy | 2.4.6 |
| pandas | 2.3.3 |
| PyYAML | 6.0.3 |
| SoundFile | 0.14.0 |
| Pillow | 11.3.0 |

No incompatibility was observed with the declared ranges in this environment.
The project uses a direct PyTorch training loop and internal edit-distance
metrics, so `datasets`, `accelerate`, `torchaudio` and `jiwer` are not required
runtime dependencies.

## Installation

Lightweight development environment:

```bash
make setup
make install-dev
```

Full local MVP environment:

```bash
.venv/bin/python -m pip install -e ".[core,data,ml,api,ui,dev]"
```

Equivalent requirements-file installation:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
```

No test downloads a model or dataset.

## Configuration

Copy `.env.example` to a local `.env` if desired, but never commit `.env`.
Important variables:

| Variable | Purpose |
|---|---|
| `IVOIREVOICE_DIOULA_DATA_DIR` | Authorized local corpus root |
| `IVOIREVOICE_ARTIFACTS_DIR` | Private manifests, predictions and reports |
| `IVOIREVOICE_MODEL_CACHE_DIR` | Hugging Face model cache |
| `IVOIREVOICE_CHECKPOINT_DIR` | Training checkpoint root |
| `IVOIREVOICE_DIOULA_PILOT_MODEL_PATH` | Local `checkpoint-000140` directory |
| `IVOIREVOICE_UI_HOST` | Local host, default `127.0.0.1` |
| `IVOIREVOICE_UI_PORT` | Gradio port, default `7860` |

The repository contains placeholders only. If
`IVOIREVOICE_DIOULA_PILOT_MODEL_PATH` is missing, the adapted model returns a
clear isolated configuration error while the two public baselines remain
usable. The application never downloads the private checkpoint.

## Run the Gradio dashboard

```bash
make ui \
  DIOULA_DATA_DIR="/path/to/voices_data" \
  ARTIFACTS_DIR="/path/to/artifacts" \
  MODEL_CACHE_DIR="/path/to/cache/models" \
  DIOULA_PILOT_MODEL_PATH="/path/to/checkpoint-000140"
```

Open `http://127.0.0.1:7860`. Public Gradio sharing is disabled.

The dashboard provides:

1. transcription and sequential model comparison;
2. the separate 600-audio and 150-audio benchmarks;
3. local baseline/adapted error analysis;
4. reference-aware custom evaluation and exports;
5. project scope, metrics and limitations.

The three-minute presentation flow and recorded-results fallback are in
[`docs/demo_guide.md`](docs/demo_guide.md).

## Run data and training workflows

```bash
make manifest-dioula
make audit-dioula
make curate-dioula
make compare-dioula-splits
make validate-dioula-v01
make audit-dioula-training
make review-dioula-training
make smoke-overfit-dy
```

`make pilot-finetune-dy` reproduces only the bounded pilot and requires the
authorized local data, cache, artifact and checkpoint roots. Do not launch a
full training or final-holdout evaluation without a separate experimental
decision.

## API

```bash
make api
```

The FastAPI adapter exposes `GET /health`, `GET /models` and
`POST /transcribe`. Its transcription route currently uses the explicit
`DummyBackend`; real Whisper comparison is implemented in the Gradio service
path.

## Tests

```bash
make verify
```

Or run individual checks:

```bash
make lint
make typecheck
make test
python3.11 -m compileall -q src scripts tests
```

Tests use synthetic WAV files, synthetic manifests, `DummyBackend` or injected
Whisper pipelines. Real model loading is optional and is not part of the unit
test suite.

## Example usage

Programmatic metric calculation:

```python
from ivoirevoice.services.evaluation_service import EvaluationService

result = EvaluationService().evaluate(
    reference="synthetic reference",
    prediction="synthetic hypothesis",
)
print(result.wer, result.cer)
```

The example contains no corpus-derived text.

## Data and privacy policy

The following must remain outside Git:

- real audio and complete manifests;
- participant or speaker metadata;
- individual references and predictions;
- manual listening reports;
- signed source URLs;
- caches, model weights and checkpoints.

Only synthetic fixtures and privacy-reviewed aggregates are publishable. The
corpus license and redistribution consent remain unconfirmed, so the local
scope is `local_research_only`.

## Checkpoint policy

`checkpoint-000140` is not included and must not be copied into the
repository. Authorized local users set
`IVOIREVOICE_DIOULA_PILOT_MODEL_PATH`. The checkpoint and derived model remain
subject to the source-data governance decision.

## Limitations

- no final-holdout evaluation;
- no complete fine-tuning run;
- limited speakers and one pilot epoch;
- no industrial robustness claim for noise, accents or recording devices;
- French evaluation has not yet been completed;
- the real ASR backend is not connected to the FastAPI route;
- the Gradio interface is local and has no public authentication layer;
- checkpoint and corpus redistribution are not authorized.

## Future work

- linguist-reviewed tone and normalization policy;
- broader authorized Dioula training and final evaluation;
- French benchmarking;
- Baoulé data governance and ASR experiments;
- optional exploration of Wav2Vec2, XLS-R or MMS;
- authenticated deployment only after privacy and licensing review.

## Security and reproducibility

- model revisions and experiment seeds are pinned;
- the repository rejects unsafe relative paths in critical configuration;
- model failures are isolated;
- exports reject personal local paths;
- checkpoints, secrets, audio and caches are ignored;
- aggregate reports preserve the original experimental values.

## Agent-first engineering harness

The repository is designed so a coding agent can discover context, implement a
bounded change and obtain mechanical feedback without relying on chat history.

- [`AGENTS.md`](AGENTS.md) is the concise navigation map.
- [`docs/index.md`](docs/index.md) is the versioned knowledge-system entry
  point.
- [`docs/knowledge-map.yaml`](docs/knowledge-map.yaml) maps each domain to its
  source, tests and documentation.
- execution plans preserve progress and decisions for multi-domain work;
- architecture, documentation links and agent contracts are checked by
  `make harness-check`;
- local development and GitHub Actions share the same `make verify` gate.

Use the short loop while iterating:

```bash
make verify-fast
```

Before handoff or merge, run the complete loop:

```bash
make verify
```

## Author

Project repository: [github.com/Inza01/ivoirevoice-ai](https://github.com/Inza01/ivoirevoice-ai)

Maintained as an academic capstone project by the IvoireVoice AI contributor.

## License and attribution

No standalone redistribution license has been selected for the project source
at this time. All rights remain reserved pending the author's explicit license
choice. The Dioula corpus and adapted checkpoint are not distributed.

OpenAI Whisper models and third-party libraries remain governed by their own
licenses. Review those licenses before reuse or distribution.
