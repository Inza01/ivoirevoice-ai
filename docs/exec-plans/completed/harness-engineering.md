# Plan: establish an agent-first engineering harness

## Status

Completed — 2026-07-31.

## Objective

Make repository knowledge, architectural constraints and validation loops
directly discoverable and executable by coding agents.

## Scope

Included: agent map, structured knowledge base, execution plans, architecture
checks, documentation checks, CI and quality governance.

Excluded: ASR model behavior, new training, data changes, deployment and
checkpoint publication.

## Acceptance criteria

- a concise `AGENTS.md` routes agents to versioned sources of truth;
- domain source, tests and documentation are machine-mapped;
- package dependency directions and documentation links are checked;
- local and CI validation use the same `make verify` entry point;
- the complete pre-existing test suite remains green;
- no private artifact enters Git.

## Progress

- Repository and current package dependencies audited.
- Concise agent map and structured repository knowledge added.
- Machine-readable domain map and execution-plan workflow added.
- Documentation, package boundaries and plan structure made executable.
- Local and GitHub Actions feedback loops aligned on `make verify`.

## Decisions

- Adapt harness engineering incrementally instead of restructuring the ASR
  codebase.
- Enforce existing dependency directions before attempting any refactor.
- Keep expensive model and corpus checks optional and external.

## Validation

- `make harness-check` — passed; 26 documents and 49 Python modules checked.
- `make verify-fast` — passed; environment, compilation, publication,
  harness, lint and type checks succeeded.
- `make verify` — passed; 99 tests, 0 failures, 19 Gradio deprecation
  warnings, 62% coverage.
