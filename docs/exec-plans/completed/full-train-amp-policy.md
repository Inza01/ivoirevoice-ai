# Full-training AMP skip policy

## Status

Completed — implementation and offline quality gates passed on 2026-08-11.

## Objective

Allow the full development and refit runners to continue after bounded, normal
FP16 GradScaler calibration skips while keeping every update-based schedule tied
only to optimizer steps that actually executed.

## Scope

Included: shared AMP outcome handling for development/refit, strict consecutive
skip guard, distinct persistent counters, mandatory GradScaler checkpoint state,
resume validation, aggregate numerical reporting, protocol documentation and
offline tests.

Excluded: running development or refit, final-holdout access, BF16 activation,
hyperparameter/text/split changes, checkpoint publication, commit and push.

## Acceptance criteria

- an AMP scale reduction records one attempted/skipped group and continues;
- scheduler, global/successful step, validation and scheduled checkpointing do
  not advance on a skipped update;
- data progress advances normally and skipped batches are not replayed in the
  live run;
- the fifth consecutive skip fails closed when the configured limit is four;
- loss, divergence, CUDA, checkpoint and post-update parameter-finiteness guards
  remain strict;
- model, optimizer, scheduler, GradScaler and all AMP counters survive an atomic
  checkpoint/resume without resetting scale 32768 to 65536;
- aggregate development/refit reports expose only numerical AMP provenance;
- FP16, LR, batch, accumulation, clipping, normalization, splits and model stay
  unchanged;
- targeted tests and `make verify` pass without executing an ML stage.

## Progress

- Host diagnostic classified one 65536→32768 skip followed by 31 stable updates
  as initial-scale calibration.
- Diagnostic implementation was committed separately as `769260f`.
- Development and refit now share one tested AMP outcome policy.
- GradScaler state is mandatory for checkpoint discovery and validated against
  the persisted AMP counters on resume.
- Aggregate development/refit outputs contain only numerical AMP provenance.

## Decisions

- Keep `global_step` as the successful optimizer-step clock and persist an
  explicit equal `successful_optimizer_steps` counter for auditability.
- Advance `groups_completed_in_epoch` for every consumed optimizer attempt,
  including skips, while never replaying the failed batch automatically.
- Treat four consecutive skips as tolerated calibration and stop on the fifth;
  this is an IvoireVoice safety policy, not a PyTorch rule.
- Require `scaler.pt` for a checkpoint to be considered resumable and reject
  inconsistent counter/scale state.
- Keep FP16 because the host evidence stabilized at scale 32768; BF16 support is
  documented but remains an unused fallback.

## Validation

- `.venv/bin/pytest -q tests/unit/training/test_full_amp_policy.py tests/unit/training/test_full_finetune.py tests/unit/training/test_fp16_diagnostic.py`
  — 35 passed.
- `.venv/bin/ruff check .` — passed.
- `.venv/bin/mypy src scripts` — passed for 58 source files.
- `make verify` — passed: environment, compile, publication audit, harness,
  Ruff, mypy and 134 tests; 19 pre-existing Gradio deprecation warnings.
- CUDA development/refit/final evaluation — explicitly excluded.

### Non-goals

- no training, refit or final-holdout evaluation;
- no BF16 switch or training-hyperparameter change;
- no checkpoint/report publication, commit or push.
