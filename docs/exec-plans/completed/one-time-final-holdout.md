# One-time frozen refit holdout evaluation

## Status

Completed — Codex, 2026-08-11. No real holdout access was performed or
authorized during implementation.

## Objective

Replace the ambiguous historical three-model final-evaluation route with a
guarded, final-refit-only, aggregate-only and irreversible one-time holdout
evaluation workflow.

## Scope

Included: metadata-only preflight, one-time state machine, final checkpoint
identity guards, streaming aggregate evaluation, hard-disabled legacy target,
synthetic tests, Make targets, documentation and publication checks.

Excluded: reading the real holdout, running either new command, loading the
real final model, evaluation, training, refit, committing or pushing.

## Acceptance criteria

- only the frozen refit checkpoint from `final_model_manifest.json` is allowed;
- every identity guard runs before the holdout access boundary;
- exact confirmation `EVALUATE_FROZEN_REFIT_ONCE` is mandatory;
- state transitions are `SEALED -> EVALUATION_IN_PROGRESS -> EVALUATED`;
- post-access failure becomes terminal `EVALUATION_FAILED_AFTER_ACCESS`;
- a completed, failed or interrupted access can never run automatically again;
- no individual reference, prediction, identifier, audio or path is persisted;
- aggregate report includes speaker count, WER, CER, RTF and edit totals;
- the legacy target fails before any data/model access;
- tests use synthetic rows and dummy runtime components only;
- `make verify`, publication audit and `git diff --check` pass.

## Progress

- Current refit metadata and historical evaluator contract reviewed.
- Blocking three-model and private-cache behavior documented.
- Final-refit-only approval receipt, metadata-only preflight and irreversible
  evaluation state machine implemented.
- Streaming aggregate evaluator and aggregate-only report contract implemented.
- Historical three-model route hard-disabled before context construction.
- Make targets, security/architecture/training documentation and synthetic
  regression coverage completed.

## Decisions

- The state file is created outside Git under the experiment artifact root.
- `evaluation_count` becomes one at the irreversible access boundary, not only
  after successful completion.
- Preflight may hash the frozen checkpoint and inspect CUDA, but cannot build a
  dataset, manifest selection or collator.
- Evaluation aggregates in memory and atomically writes one aggregate report;
  it never invokes the historical private evaluation writer.
- The historical stage remains as an explicit hard failure for traceability.

## Validation

- targeted one-time and full-runner regression tests — 45 passed.
- `make verify` — passed; 201 tests passed with 19 known Gradio deprecation
  warnings.
- publication audit — passed; 162 candidate files, largest 52,584 bytes.
- harness check — passed; 32 documents and 57 Python modules checked.
- Ruff — passed.
- MyPy — passed for 61 source files.
- `git diff --check` — passed at final handoff.
