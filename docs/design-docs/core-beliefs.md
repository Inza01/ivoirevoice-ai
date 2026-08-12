# Core engineering beliefs

These principles are stable defaults. A change may override one only with an
explicit execution-plan decision and a corresponding mechanical guardrail.

## Humans specify intent; the repository carries context

Prompts are temporary. Product contracts, architecture decisions, experiment
constraints and validation evidence belong in versioned repository files.
`AGENTS.md` is a map to that knowledge, not a monolithic instruction manual.

## Make correctness observable

Every important invariant should have an executable signal: a test, schema,
static check, report or health endpoint. A prose-only rule is considered
incomplete when it can be checked mechanically.

## Validate boundaries, allow local autonomy

External configuration, manifests, audio metadata and model outputs are
validated when they enter the system. Within those contracts, implementations
may evolve without central micromanagement.

## Reproducibility outranks convenience

Experiments pin model revisions, seeds, data selections and comparable
datasets. Expensive or destructive actions require explicit confirmation.
Reported results are never silently recomputed or improved.

## Privacy is an architecture boundary

Corpus data, per-sample text and model checkpoints are external dependencies,
not repository assets. Public artifacts are aggregates. A missing private
resource causes a clear isolated failure rather than an implicit download.

## Prefer repairable, boring systems

Use standard Python, explicit YAML, typed dataclasses and local commands that
agents can inspect. New abstractions must reduce ambiguity or enforce an
invariant; novelty alone is not a reason to add a dependency.

## Convert repeated feedback into leverage

When the same mistake recurs, improve the harness: documentation, a checker,
a test or a safer interface. Do not rely on increasingly forceful prompts.

These beliefs adapt the agent-first practices described in OpenAI's
[Harness engineering](https://openai.com/index/harness-engineering/) report
to a privacy-sensitive, low-resource ASR research repository.
