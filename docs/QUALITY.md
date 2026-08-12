# Quality score

This scorecard is evidence, not marketing. Update it when a gate, contract or
known gap changes.

| Domain | Grade | Evidence | Main gap |
|---|---:|---|---|
| Data pipeline | A- | deterministic split, hashes, quarantine, unit tests | corpus governance remains local |
| Model adapters | B+ | typed backend, lazy load/unload, pinned revisions | real-model CI is optional |
| Evaluation | A- | internal WER/CER, comparable-run checks, one-time aggregate holdout evidence | final holdout has only 3 speaker groups |
| Training | A- | smoke/pilot/full gates, deterministic seed, split checks, atomic recovery | training orchestration remains complex |
| Services | A- | isolated failures, typed results, privacy-safe exports | limited runtime telemetry |
| Gradio UI | B+ | offline construction tests, benchmark separation | Gradio 6 deprecations |
| FastAPI | B | health/integration tests | real Whisper route not connected |
| Repository harness | A- | agent map, knowledge map, CI, structural checks | freshness remains partly review-based |

## Required evidence

- `make verify-fast` for iterative changes;
- `make verify` before handoff or merge;
- an execution plan for cross-domain work;
- updated tests and sources of truth in the same change;
- explicit recording of unresolved work in `technical-debt.md`.

Grades must decrease when evidence is removed or a known invariant becomes
manual-only.
