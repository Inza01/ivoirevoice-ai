# Full Dioula fine-tuning implementation

## Status

Completed — implementation and offline quality gates passed.

## Objective

Implement a guarded local Whisper Tiny workflow with three explicit stages:
development on train with validation-based budget selection, refit on
train+validation, and one sealed final-holdout evaluation.

The implementation must not launch training or inspect holdout audio while CUDA
is unavailable.

## Scope

Included: strict settings, frozen exhaustive selections, shared step mechanics,
development, refit, sealed evaluation, Make targets, documentation and offline
tests.

Excluded: executing training, decoding the holdout, publishing a model or
pushing the implementation branch.

## Acceptance criteria

- strict frozen counts, hashes, split isolation and local checkpoint identity;
- fresh optimizer/scheduler when starting development and refit;
- exact 861/1,722/1,027 step geometry and deterministic refit budget;
- correct normalization of incomplete gradient-accumulation groups;
- resumable atomic checkpoints with bounded retention;
- no validation decoding during refit and no test decoding during training;
- a one-model sealed receipt for final-holdout evaluation;
- aggregate public outputs and private identifiers/predictions outside Git;
- four explicit Make entry points and offline unit coverage;
- `make verify` passes without running ML stages.

## Progress

- The harness was committed separately as `f4fc4f5`.
- A dedicated `feat/full-dioula-finetune` branch was created.
- Configuration, frozen selection checks and shared training mechanics were
  scaffolded.
- Development, refit and sealed-evaluation orchestration are implemented.
- The pilot and full runner share accumulation, atomic checkpoint, retention
  and resume-discovery mechanics.
- Four explicit Make targets and 19 focused full-training tests are present.

## Decisions

- Runtime artifacts, including aggregate shareable reports, stay outside Git so
  a successful preflight does not make the next clean-worktree gate fail.
- Every train/validation audio is content-hashed before ML; holdout bytes are
  not touched until the explicit sealed evaluation.
- Development and refit each initialize optimizer, scheduler and scaler from
  scratch while loading only the pilot model weights.
- A started final receipt blocks any later training and only the same frozen
  model/selection can resume an interrupted evaluation.

## Validation

- `make verify` — passed on the final implementation: environment valid,
  compile passed, publication audit passed, harness passed, Ruff passed, mypy
  passed, 118 tests passed.
- Targeted training tests — 30 development/pilot tests passed during
  iteration.
- `nvidia-smi` — failed to communicate with the NVIDIA driver.
- PyTorch `2.10.0+cu130` reported `torch.cuda.is_available() == False`.
- No ML stage or holdout access was executed; full preflight remains correctly
  blocked until CUDA is repaired.

### Non-goals

- no full training, refit or holdout evaluation in this implementation change;
- no model publication, commit push, corpus copy or checkpoint copy into Git;
- no CPU or Colab fallback.
