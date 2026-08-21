# Technical debt tracker

| ID | Priority | Area | Evidence | Next action |
|---|---:|---|---|---|
| TD-001 | Medium | UI | Gradio 6 deprecation warnings in tests | migrate launch and event parameters in a dedicated change |
| TD-003 | Low | Observability | no local logs/metrics query stack | define minimum structured events before deployment |
| TD-004 | Medium | Training | pilot workflow is a large module | split only after characterization tests cover orchestration |
| TD-005 | High | Governance | source-code license not selected | obtain owner decision before broader reuse |
| TD-007 | High | Web security | the local upload path now sniffs and deletes audio, but authentication, authorization, rate limits and abuse controls remain absent | complete public-runtime threat model and controls before Internet exposure |
| TD-008 | Medium | Accessibility | jsdom checks semantics but cannot prove browser contrast, reflow or full keyboard behavior | add Playwright plus browser axe, zoom, reduced-motion and keyboard journeys |

Debt is not fixed by hiding it. Close an item only when the validating evidence
lands in the same change, and update `QUALITY.md` when the score changes.

## Closed

- **TD-006 — final-holdout evidence:** closed by the one-time evaluation of the
  frozen refit. Only the reviewed aggregate result is published in
  `reports/final_holdout_metrics.json`; the terminal receipt prevents a second
  access.
- **TD-002 — real ASR API boundary:** closed by the versioned model/language
  discovery and secure synchronous transcription adapter over
  `TranscriptionService`. The legacy Dummy routes remain only for compatibility.
