# FP16 train-only diagnostic

## Status

Completed — implementation and offline quality gates passed on 2026-08-11.

## Objective

Add a bounded, train-only FP16 diagnostic that observes 32 optimizer attempts,
records non-sensitive numerical evidence and never creates a checkpoint or
changes the full-development AMP policy.

## Scope

Included: one explicit CLI stage and Make target, fresh pilot-based runtime,
pre-clipping gradient inspection, AMP skip accounting, privacy-safe batch
metadata, one external JSON report, host BF16 capability check instructions,
tests and protocol documentation.

Excluded: full development, validation decoding, historical test access,
final-holdout access, refit, checkpoint/model output, precision or
hyperparameter changes, commit and push.

## Acceptance criteria

- exactly 32 optimizer attempts by default with unchanged batch, accumulation,
  LR, clipping, FP16, checkpoint, collator and seed settings;
- an AMP skip does not advance the scheduler or successful-step counter and
  does not stop the diagnostic;
- explicit gradient statistics are captured after unscale and before clipping;
- only train rows can reach the collator and no validation/test/holdout loader
  is reachable from the diagnostic stage;
- no checkpoint or final model can be written;
- the external report contains no text, speaker, audio or local path and an
  existing diagnostic output is never overwritten;
- the existing `full-finetune-dev` skip policy remains unchanged;
- targeted tests and `make verify` pass without running a CUDA stage.

## Progress

- Root cause confirmed as a real GradScaler scale reduction from 65536 to
  32768, not a `scaler.step()` return-value false positive.
- The diagnostic has an isolated CLI dispatch and Make target.
- Train-only loading, 32-attempt bounding, numerical instrumentation,
  classification, private HMAC identifiers and one-shot reporting are covered.
- The production development runner remains unchanged.

## Decisions

- Keep the production development loop untouched until diagnostic evidence is
  reviewed.
- Use public tensor finiteness operations plus the public GradScaler scale
  comparison; do not inspect private GradScaler state.
- Hash utterance identifiers with an ephemeral HMAC key that is never written,
  preventing linkage or dictionary reversal from the persisted report.
- Refuse an existing diagnostic directory instead of overwriting it.
- Require a clean committed implementation and a matching fresh preflight
  before the host diagnostic can run.

## Validation

- `.venv/bin/ruff check .` — passed.
- `.venv/bin/mypy src scripts` — passed for 58 source files.
- `.venv/bin/pytest -q tests/unit/training/test_fp16_diagnostic.py tests/unit/training/test_full_finetune.py`
  — 27 passed.
- `make verify` — passed: environment, compile, publication audit, harness,
  Ruff, mypy and 126 tests; 19 pre-existing Gradio deprecation warnings.
- `make -n full-finetune-fp16-diagnostic ...` — confirmed explicit
  `--stage fp16-diagnostic` dispatch and Make-variable propagation.
- CUDA diagnostic execution — explicitly excluded from the Codex sandbox.

### Non-goals

- no full training, refit or final-holdout evaluation;
- no change from FP16 to BF16;
- no model/checkpoint publication, commit or push.
