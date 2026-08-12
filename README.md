# IvoireVoice AI

Local, reproducible automatic speech recognition experiments for French and
Dioula in the Ivorian context.

IvoireVoice AI provides a Gradio comparison dashboard, a speaker-disjoint data
pipeline, Whisper baselines, a full Dioula Whisper Tiny adaptation, ASR metrics
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
- adapt Whisper Tiny through a bounded pilot, full development and frozen
  refit;
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
| Whisper Tiny Dioula Final, `checkpoint-002052` | Frozen final refit | Used locally, not distributed |

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

The Phase 4C pilot established the training pipeline with:

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

Full development then used all 13,764 train audios and selected its budget on
the 2,661 validation audios. `checkpoint-001720` fixed a 2,052-step refit
budget. The final refit restarted from the pilot weights with fresh optimizer,
scheduler and GradScaler states, then trained on all 16,425 train+validation
audios from 18 speakers. The 150-audio historical pilot and the 2,624-audio
final holdout were excluded from training and model selection.

## Metrics

- **WER**: word-level substitutions, insertions and deletions divided by the
  reference word count.
- **CER**: the corresponding character-level error rate.
- **RTF**: processing time divided by audio duration.

Lower WER, CER and RTF are better. WER can exceed 100% when a model produces
many insertions.

## Final Dioula ASR Results

| Metric | Final Holdout |
|---|---:|
| WER | 33.26% |
| CER | 12.38% |
| RTF | 0.00785 |
| Final loss | 0.3464 |
| Audios | 2,624 |
| Speakers | 3 |

- substitutions: 5,690
- insertions: 1,363
- deletions: 1,864
- exact matches: 334

These aggregate results come from the independent **final holdout**. Its 2,624
audios were never used for training, validation, model selection or
hyperparameter tuning. The frozen refit was evaluated exactly once, and the
model, decoding and normalization were not changed afterward. The reviewed
machine-readable report is
[`reports/final_holdout_metrics.json`](reports/final_holdout_metrics.json).

## Experimental progression

The following stages use different protocols and datasets. Their metrics
document project progression but **must not be interpreted as a direct
cross-dataset comparison**.

### Pilot — historical validation protocol

Both models below were evaluated on the **same 600 validation audios and
references**:

| Model | Dataset | WER | CER | RTF |
|---|---|---:|---:|---:|
| Whisper Tiny baseline | 600 validation audios | 1.154527 | 0.717216 | 0.012340 |
| Whisper Tiny adapted pilot | 600 validation audios | 0.782165 | 0.348294 | 0.018742 |

- WER relative reduction: **32.25%**
- CER relative reduction: **51.44%**

These are pilot validation results, not final-holdout or industrial
performance.

### Full development — 2,661 validation audios

| Model | Dataset | WER | CER |
|---|---|---:|---:|
| `checkpoint-000140` | 2,661 validation audios | 0.774152 | 0.361319 |
| `checkpoint-001720` | 2,661 validation audios | 0.441286 | 0.183654 |

`checkpoint-001720` was selected only to determine the frozen refit budget. It
is not the final model and never accessed the final holdout.

### Final refit — independent final holdout

| Model | Dataset | WER | CER | RTF |
|---|---|---:|---:|---:|
| `checkpoint-002052` | 2,624 final-holdout audios | 0.332625 | 0.123804 | 0.007853 |

The final model was accepted as evaluated. No gain is calculated between this
row and validation rows because the datasets differ.

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
│   ├── training/            # Smoke, pilot, full refit and sealed evaluation
│   └── ui/                  # Gradio application
├── tests/                   # Offline unit and integration tests
├── .env.example
├── Makefile
└── pyproject.toml
```

## Requirements

- Python 3.11;
- enough disk space for public Whisper caches and local checkpoints;
- CUDA is required for reproducing the pilot and full-training workflows;
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
| `IVOIREVOICE_DIOULA_PILOT_MODEL_PATH` | Historical training source `checkpoint-000140` |
| `IVOIREVOICE_DIOULA_FINAL_MODEL_PATH` | Live UI model `checkpoint-002052` |
| `IVOIREVOICE_UI_HOST` | Local host, default `127.0.0.1` |
| `IVOIREVOICE_UI_PORT` | Gradio port, default `7860` |

The repository contains placeholders only. If
`IVOIREVOICE_DIOULA_FINAL_MODEL_PATH` is missing or does not point to
`checkpoint-002052`, the final model returns a clear isolated configuration
error while the two public baselines remain usable. The application never
downloads the private checkpoint.

## Run the Gradio dashboard

```bash
make ui \
  DIOULA_DATA_DIR="/path/to/voices_data" \
  ARTIFACTS_DIR="/path/to/artifacts" \
  MODEL_CACHE_DIR="/path/to/cache/models" \
  DIOULA_FINAL_MODEL_PATH="/path/to/checkpoint-002052"
```

Open `http://127.0.0.1:7860`. Public Gradio sharing is disabled.

For a corpus-free demonstration on a clean `main`, use the guarded entry point:

```bash
export IVOIREVOICE_DIOULA_FINAL_MODEL_PATH="/path/to/checkpoint-002052"
export IVOIREVOICE_MODEL_CACHE_DIR="/path/to/cache/models"
make demo-preflight
make demo
```

Demo mode never exposes the corpus or private-artifact roots to Gradio. It
accepts only a new recording or an explicitly confirmed external demo audio;
see [`docs/demo_guide.md`](docs/demo_guide.md).

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

`make pilot-finetune-dy` describes the bounded historical pilot and requires
the authorized local data, cache, artifact and checkpoint roots. The full
development, refit and one-time holdout workflows are retained for provenance
and regression tests. Their official cycle is complete: do not relaunch
training, refit or final-holdout evaluation.

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

Neither `checkpoint-000140` nor the final `checkpoint-002052` is included or
authorized for publication. Historical training uses
`IVOIREVOICE_DIOULA_PILOT_MODEL_PATH`; the local UI uses
`IVOIREVOICE_DIOULA_FINAL_MODEL_PATH`. Both checkpoints remain subject to the
source-data governance decision.

## Limitations

- the final model is specialized to the available Dioula/local context;
- performance may vary with accent, noise, microphone and application domain;
- the corpus is limited to the available collection context;
- the final holdout contains only 3 speaker groups;
- no result guarantees equivalent production performance;
- French evaluation has not yet been completed;
- the real ASR backend is not connected to the FastAPI route;
- the Gradio interface is local and has no public authentication layer;
- checkpoint and corpus redistribution are not authorized.

## Future work

- linguist-reviewed tone and normalization policy;
- broader authorized Dioula data and external evaluation;
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
