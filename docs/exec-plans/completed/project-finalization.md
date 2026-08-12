# Project finalization after one-time holdout

## Status

Completed — Codex, 2026-08-12.

## Objective

Publish the reviewed aggregate Dioula result, expose the frozen final model in
the local UI through an external path, and align the public project narrative
with the completed training protocol.

## Scope

Included: README and technical documentation, one sanitized aggregate report,
the UI model catalog/runtime guard, environment and Make wiring, synthetic UI
tests, publication and repository audits.

Excluded: any training, refit, holdout access, model inference on private data,
weight changes, checkpoint/corpus publication, commit or push.

## Acceptance criteria

- the live UI offers Tiny baseline, Small baseline and Tiny Dioula Final;
- the final checkpoint is resolved only from
  `IVOIREVOICE_DIOULA_FINAL_MODEL_PATH` and absence fails clearly in isolation;
- historical pilot evidence remains labelled by its own dataset;
- final metrics are published only as reviewed aggregates without paths,
  identifiers, transcripts, predictions or audio;
- README and model/training/architecture documentation describe the completed
  19,199-audio workflow and honest limitations;
- synthetic tests cover comparison, metrics, failure isolation and JSON/CSV/TXT
  exports without accessing the final holdout;
- `make verify`, publication audit and `git diff --check` pass.

## Outcome

- The UI catalog now exposes the frozen `checkpoint-002052` through an
  environment-only path and keeps baseline failures isolated.
- The README, model card, protocols and demo guide separate pilot, development
  and independent final-holdout evidence.
- `reports/final_holdout_metrics.json` contains only reviewed aggregate facts.
- No real audio, checkpoint, private artifact or holdout row was accessed.

## Progress

- Repository contracts and the completed experimental evidence were reviewed.
- Documentation and the sanitized aggregate report were published locally.
- Final-model UI configuration and its failure guards were implemented.
- Synthetic targeted and complete verification gates passed.
- Privacy, ignore rules and Git candidates were audited; commit and push remain
  pending explicit authorization.

## Decisions

- `docs/model_card.md` is the repository's actual model document; no duplicate
  `docs/model.md` was created.
- Pilot reports remain historical evidence, while the live third model is the
  frozen final refit.
- Real UI inference is outside automated acceptance; tests use synthetic WAV
  data and `DummyBackend` only.
- No commit or push is included in this plan.

## Validation

- `.venv/bin/python -m compileall -q src scripts tests` — passed.
- final-model UI tests — 9 passed.
- targeted UI/service/evaluation/split tests — 20 passed, 11 known Gradio 6
  deprecation warnings before the final identity-guard test was added.
- `make verify` — passed: 205 tests, 19 known Gradio 6 deprecation warnings.
- Ruff — passed.
- MyPy — passed for 61 source files.
- publication audit — passed, 164 candidate files, largest 52,584 bytes.
- harness check — passed, 33 documents and 57 Python modules checked.
