# Demo readiness

## Status

Completed — Codex, 2026-08-12.

## Objective

Provide a guarded, offline-capable startup path for demonstrating the frozen
Dioula model without accessing any corpus split, private prediction or final
holdout artifact.

## Scope

Included: a local demo preflight, a safe `make demo` entry point, cache and
checkpoint checks, CUDA load-only verification, synthetic UI tests, and an
updated three-minute demo/fallback guide.

Excluded: training, refit, holdout evaluation, private dataset discovery,
model/normalization/decoding changes, optimization, checkpoint publication,
commit, merge or push.

## Acceptance criteria

- preflight requires clean `main`, Python 3.11, the repository `.venv`, CUDA,
  the frozen checkpoint identity/hash, valid UI configuration, available port
  and sufficient disk;
- the preflight loads the final model on CUDA but performs no transcription;
- pinned Tiny and Small revisions are confirmed in the local cache;
- `make demo` deliberately withholds corpus and private-artifact roots;
- a demo audio is either explicitly confirmed as external or reported as
  `DEMO AUDIO REQUIRED` without selecting corpus data;
- tests are synthetic/offline and no holdout or training entry point is called;
- `make verify`, publication audit and `git diff --check` pass.

## Progress

- Git, UI configuration, final checkpoint and baseline caches audited.
- Official checkpoint hash matches the frozen public hash.
- Versioned preflight and synthetic smoke implemented.
- Host CUDA load-only probe, real synthetic comparison and Gradio HTTP check
  passed.
- All automated gates passed.

## Decisions

- The demo target does not expose `IVOIREVOICE_DIOULA_DATA_DIR` or
  `IVOIREVOICE_ARTIFACTS_DIR`; historical error examples remain unavailable in
  demo mode rather than reading private predictions.
- Missing pre-recorded demo audio is a visible warning, not an automatic corpus
  fallback; the status is `READY WITH DEMO AUDIO REQUIRED` and a new microphone
  recording remains allowed.
- The existing model backend is reused unchanged; no optimization or inference
  policy is altered.

## Validation

- official `directory_sha256(checkpoint-002052)` — passed,
  `d9dd6469cd102e98b17d1e0750e51fa9107f3eb0847f130984cf993f033151c1`;
- host CUDA load-only probe — passed, RTX 5070 Ti Laptop, 1.882 s load,
  72.97 MiB PyTorch peak allocation;
- targeted UI/demo tests — 20 passed, 11 known Gradio 6 deprecation warnings;
- versioned real synthetic model smoke — Tiny, Small and Dioula Final passed;
  comparison 7.419 s, WER/CER path and JSON/CSV/TXT exports passed, no private
  path exposed;
- earlier repeated final-model call — 0.853 s, peak comparison allocation
  786.15 MiB, no OOM;
- Gradio loopback launch — HTTP 200 on `127.0.0.1:7860`, clean shutdown;
- `make verify` — passed: 215 tests, Ruff and MyPy passed;
- publication audit — passed: 168 candidate files checked;
- harness — passed: 34 documents and 57 Python modules checked;
- `git diff --check` — passed.
