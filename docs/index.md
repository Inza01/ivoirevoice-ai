# Repository knowledge index

This directory is the system of record for IvoireVoice AI. `AGENTS.md` points
here so humans and coding agents can progressively load only the context
needed for a task.

## Product and architecture

- [MVP product contract](product-specs/mvp.md): supported user journeys,
  non-goals and acceptance criteria.
- [Architecture](architecture.md): packages, runtime flows and boundaries.
- [Core engineering beliefs](design-docs/core-beliefs.md): durable design
  choices that guide implementation.
- [Experiment protocol](experiment_protocol.md): comparable datasets,
  metrics and evaluation constraints.

## Data and ML

- [Data card](data_card.md): corpus structure and governance.
- [Model card](model_card.md): model scope, evidence and limitations.
- [Training protocol](training_protocol.md): smoke and bounded pilot rules.
- [Demo guide](demo_guide.md): three-minute demonstration and fallback.

## Operations and governance

- [Security](SECURITY.md): privacy boundaries and publication policy.
- [Reliability](RELIABILITY.md): failure modes, recovery and observability.
- [Quality score](QUALITY.md): evidence-based domain scorecard.
- [Technical debt](technical-debt.md): explicit, prioritized follow-up work.
- [Execution plans](exec-plans/README.md): format for non-trivial changes.

## Machine-readable navigation

[`knowledge-map.yaml`](knowledge-map.yaml) connects domains to source paths,
tests, documentation and validation commands. `make harness-check` validates
the map, Markdown links, plans and package dependency rules.

When code changes a documented contract, update the corresponding source of
truth in the same change.
