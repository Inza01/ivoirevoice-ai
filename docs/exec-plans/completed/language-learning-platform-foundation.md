# Plan: Phase 2 language-learning platform foundation

## Status

Completed — Codex, 2026-08-13.

## Objective

Deliver a reviewable, accessible and responsive web foundation for IvoireVoice
on `feat/language-learning-platform`, while preserving the validated ASR,
training history, sealed final holdout and legacy Gradio application unchanged.

## Scope

In scope:

- document the Phase 2 product, architecture, wireframes and design system;
- create an isolated Next.js/React/TypeScript/Tailwind application under
  `web/`;
- implement the shared shell, homepage and the eleven approved routes as
  honest Foundation states;
- add reusable accessible components, French/English UI messages, the
  `fr`/`en`/`dyu` language registry and a typed API-client boundary;
- add deterministic frontend lint, format, type-check, test and build gates;
- connect those gates to the repository harness and CI;
- document coexistence of the new web platform and legacy Gradio demo.

Out of scope:

- model training, refit, evaluation or checkpoint changes;
- any access to corpus, validation, historical pilot or final-holdout data;
- migration of the real ASR runtime into FastAPI;
- a translation engine, validated Dioula lessons, pronunciation scoring,
  authentication, database, object storage or public deployment;
- deletion or replacement of the Gradio application;
- pushing the branch.

## Acceptance criteria

- All approved routes render through one coherent responsive shell.
- Foundation components meet documented WCAG 2.2 AA interaction and contrast
  rules, with automated semantic tests and a documented manual-test backlog.
- UI messages are externalized for French and English; `dyu` remains the
  canonical Dioula language code and UI locale support is not misrepresented.
- Capability states distinguish `available`, `experimental` and
  `coming_soon`; unsupported translations never return simulated output.
- The web client depends only on typed HTTP contracts and never imports the
  Python package, models, checkpoints, manifests or private paths.
- The legacy `make ui` and Python package remain structurally unchanged.
- `make verify`, frontend lint/typecheck/tests/build, publication audit and
  `git diff --check` pass in the current worktree.
- No checkpoint, corpus, audio, private transcript, individual prediction,
  secret or personal path is tracked or staged.
- Changes are staged explicitly into a small number of logical local commits;
  nothing is pushed.

## Progress

- 2026-08-13 — verified clean `main` at
  `e3ecb0a1610e621d1104ec43f58527e9b0bfe7f1`, synchronized with
  `origin/main`, and created `feat/language-learning-platform`.
- 2026-08-13 — audited Python services, FastAPI adapter, Gradio, tests,
  security boundary, CI and absence of a prior frontend.
- 2026-08-13 — drafted architecture, product specification/wireframes and
  accessible design system.
- 2026-08-13 — implemented the isolated `web/` Foundation, eleven routes,
  typed contracts, honest capability registry, FR/EN UI messages, local upload
  validation, same-origin proxy boundary and security headers.
- 2026-08-13 — generated the npm lock, remediated initial PostCSS/Vitest
  advisories to zero known vulnerabilities and completed all quality gates.

## Decisions

- Use Next.js App Router, React, strict TypeScript, Tailwind CSS and npm in
  `web/`; this is an isolated presentation layer, not a Python dependency.
- Use Node 24 LTS, recorded in `.nvmrc` and CI. Local tooling is unpacked under
  `/tmp` and never installed into the repository or system directories.
- Keep the browser same-origin boundary behind an allowlisted Next route
  handler; never expose a private FastAPI URL through `NEXT_PUBLIC_*`.
- Keep API integration contract-only in this lot because the current FastAPI
  transcription route uses `DummyBackend` and is not the production model
  endpoint.
- Treat French ASR and Dioula final ASR as experimental until their specific
  user-facing integrations are connected; English ASR and every translation
  pair are `coming_soon` in the Foundation.
- Use `dyu`, the ISO 639-3 code already used by the ASR package, rather than
  introducing the ambiguous `dy` example from the prompt.
- Use system fonts and code-native geometric accents so builds require no font
  or image download and the visual identity remains original and restrained.
- Do not add PostgreSQL, SQLAlchemy, storage or authentication dependencies in
  the Foundation; their contracts and future sequence are documented.

## Validation

- `npm run format:check --prefix web` — passed.
- `npm run lint --prefix web` — passed with zero warnings.
- `npm run typecheck --prefix web` — passed in strict mode.
- `npm test --prefix web` — 53 tests passed across 8 files.
- `npm run audit --prefix web` — zero known vulnerabilities.
- `npm run build --prefix web` — passed; 11 Foundation routes generated.
- `make verify-fast` — passed before the final full gate.
- `make verify` — passed with 215 Python tests and 53 frontend tests.
- `.venv/bin/python scripts/audit_repository.py` — passed.
- `git diff --check` — passed.
