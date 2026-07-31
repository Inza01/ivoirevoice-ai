# IvoireVoice AI agent guide

This file is a map, not the full manual. Read only the linked material needed
for the task, and keep durable decisions in the repository.

## Start here

1. Read `docs/index.md`.
2. Read `docs/product-specs/mvp.md` for product scope.
3. Read `docs/architecture.md` before changing dependencies or boundaries.
4. Read `docs/SECURITY.md` before touching data, reports, audio or models.
5. For multi-file work, create or update an execution plan under
   `docs/exec-plans/active/`.

The machine-readable map is `docs/knowledge-map.yaml`.

## Working loop

1. Inspect `git status --short --branch`; preserve unrelated user changes.
2. State acceptance criteria before implementation.
3. Make the smallest coherent change.
4. Add or update tests with the implementation.
5. Run `make verify-fast` during iteration.
6. Run `make verify` before handoff.
7. Update documentation or technical debt when behavior or constraints change.
8. Report exact commands, failures, warnings and remaining risks.

Never claim a command passed unless it was executed in the current worktree.

## Non-negotiable boundaries

- Real corpus audio, full manifests, participant metadata, individual
  transcriptions, predictions and checkpoints stay outside Git.
- Never train on validation, historical test-pilot or final-holdout data.
- Never present the 600-audio validation benchmark and the historical
  150-audio test-pilot as one experiment.
- Whisper Tiny and Whisper Small are baselines; only Whisper Tiny has a local
  pilot adaptation.
- Wav2Vec2, XLS-R and MMS are future directions, not implemented MVP models.
- Unit tests must not download models or datasets.
- Configuration and external data shapes are validated at their boundaries.
- UI and API adapters do not contain model-training logic.
- Model failures remain isolated so other comparison results survive.
- Public reports contain aggregates only.

## Architecture

Permitted package dependencies are enforced by
`scripts/check_harness.py`. Do not bypass the checker; update
`docs/architecture.md` and the checker together when an intentional boundary
change is approved.

Domain entry points:

- data: `src/ivoirevoice/data/`
- model adapters: `src/ivoirevoice/models/`
- evaluation: `src/ivoirevoice/evaluation/`
- training: `src/ivoirevoice/training/`
- orchestration: `src/ivoirevoice/services/`
- adapters: `src/ivoirevoice/api/`, `src/ivoirevoice/ui/`

## Commands

```bash
make verify-fast     # environment, compile, publication and harness checks
make test            # full offline test suite
make verify          # complete required gate
make ui              # local Gradio app; private paths supplied at runtime
make api             # local FastAPI adapter
```

Use Python 3.11 and the repository `.venv`. Do not install dependencies or
download models without explicit authorization.

## Definition of done

- acceptance criteria are satisfied;
- relevant tests cover success and failure behavior;
- `make verify` passes;
- privacy and split invariants still hold;
- docs and execution plans reflect the final behavior;
- no secrets, private artifacts, checkpoints or personal paths are staged;
- unresolved work is recorded in `docs/technical-debt.md`.
