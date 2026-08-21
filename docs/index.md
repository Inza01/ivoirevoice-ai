# Repository knowledge index

This directory is the system of record for IvoireVoice AI. `AGENTS.md` points
here so humans and coding agents can progressively load only the context
needed for a task.

## Product and architecture

- [MVP product contract](product-specs/mvp.md): supported user journeys,
  non-goals and acceptance criteria.
- [Language-learning platform product contract](product-specs/language-learning-platform.md):
  Phase 2 personas, journeys, honest capability states and textual wireframes.
- [Architecture](architecture.md): packages, runtime flows and boundaries.
- [Phase 2 architecture](phase2_architecture.md): target web, API, service,
  persistence and migration boundaries.
- [ASR web integration](asr-web-integration.md): versioned HTTP contracts,
  temporary upload policy, model lifecycle and local runtime.
- [Design system](design-system.md): accessible visual and interaction rules
  for the new web platform.
- [Core engineering beliefs](design-docs/core-beliefs.md): durable design
  choices that guide implementation.
- [Experiment protocol](experiment_protocol.md): comparable datasets,
  metrics and evaluation constraints.

## Data and ML

- [Data card](data_card.md): corpus structure and governance.
- [Model card](model_card.md): model scope, evidence and limitations.
- [Training protocol](training_protocol.md): smoke, pilot, full refit and sealed
  evaluation rules.
- [Demo guide](demo_guide.md): three-minute demonstration and fallback.

## Operations and governance

- [Security](SECURITY.md): privacy boundaries and publication policy.
- [Reliability](RELIABILITY.md): failure modes, recovery and observability.
- [Quality score](QUALITY.md): evidence-based domain scorecard.
- [Technical debt](technical-debt.md): explicit, prioritized follow-up work.
- [Execution plans](exec-plans/README.md): format for non-trivial changes.
- [Phase 2 Foundation execution plan](exec-plans/completed/language-learning-platform-foundation.md):
  implementation scope, decisions and validation evidence for the new web shell.
- [Phase 2A ASR execution plan](exec-plans/completed/web-asr-integration.md):
  completed integration scope, decisions and validation evidence.

## Machine-readable navigation

[`knowledge-map.yaml`](knowledge-map.yaml) connects domains to source paths,
tests, documentation and validation commands. `make harness-check` validates
the map, Markdown links, plans and package dependency rules.

When code changes a documented contract, update the corresponding source of
truth in the same change.
