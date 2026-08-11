# Terminal development checkpoint validation

## Status

Completed — implementation and offline gates passed on 2026-08-11.

## Objective

Add one guarded, auditable and idempotent validation-only stage for the
terminal development checkpoint. The stage proposes, but does not finalize, a
best-checkpoint and refit-budget update.

## Scope

Included: a new CLI/Make entry point, validation-only loading, immutable run
identity checks, reuse of the existing evaluator/ranking/budget functions,
aggregate and private external reports, tests and protocol documentation.

Excluded: executing the GPU evaluation, changing the frozen development
decision, launching refit, accessing final-holdout content, changing training
hyperparameters, committing or pushing.

## Acceptance criteria

- the stage requires a frozen development run and the exact terminal checkpoint;
- refit or final-holdout activity blocks the stage before CUDA or evaluation;
- only validation rows are materialized and only validation audio is decoded;
- `_evaluate_validation`, `metric_rank` and `refit_step_budget` are reused;
- no optimizer, scheduler, scaler, backward pass or checkpoint write exists;
- current and candidate identities include run, code, config, pilot,
  checkpoint, validation and decoding provenance;
- a candidate report never overwrites `development_decision.json`;
- identical reruns reuse the same report and mismatched reruns fail closed;
- public output contains aggregates only and private output remains outside Git;
- targeted tests, Ruff, mypy, publication audit, `git diff --check` and
  `make verify` pass without running ML.

## Progress

- Requirements and repository contracts reviewed.
- Validation-only loader, guarded stage, Make target and refit-start marker
  implemented.
- Candidate selection, privacy and idempotence tests implemented.
- Protocol and knowledge map updated.
- All offline validation gates passed; no ML stage was executed.

## Decisions

- The amendment is recorded before any final-holdout access.
- The evaluation commit is distinct from the historical development commit;
  both are recorded, while checkpoint membership is checked against the
  historical development run identity.
- The global frozen manifest may be hash-checked and streamed, but only
  validation rows may be materialized; test rows and their audio are never
  selected or decoded by this stage.
- A refit-start marker is written before the first possible optimizer update so
  terminal validation cannot be launched after refit activity.

## Validation

- `pytest -q tests/unit/training/test_development_final_validation.py` — 22
  stage-specific tests collected and passed as part of the targeted run.
- targeted terminal-validation and full-training tests — 43 passed.
- `make verify-fast` — passed: compile, publication audit, harness, Ruff and
  mypy all passed.
- `make verify` — passed: 156 tests passed with 19 existing Gradio
  deprecation warnings.
- `make audit-repository` — passed; 154 candidates, largest candidate 51 964
  bytes.
- `git diff --check` — passed.
- Candidate checkpoint hash was recomputed as
  `ac99b6abdbb6fb692561e38b6d6db546e0a7a320ba2b5db2a1148e8c55094847`.
- No GPU evaluation, refit, final-holdout evaluation, commit or push occurred.
