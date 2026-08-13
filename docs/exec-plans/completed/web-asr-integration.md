# Plan: Phase 2A production ASR web integration

## Status

Completed — Codex, 2026-08-13.

## Objective

Connect the existing Next.js transcription experience to a versioned FastAPI
adapter and the existing ASR service boundary, while keeping the frozen Dioula
model, private datasets and one-time final holdout completely untouched.

## Scope

Implemented:

- privacy-safe discovery and synchronous transcription contracts under
  `/api/v1`;
- reuse of configured `ASRBackend`, `ModelRegistry` and `TranscriptionService`
  lifecycle behavior rather than duplicated inference logic;
- strict model/language routing and bounded temporary audio processing with
  guaranteed cleanup;
- an allowlisted same-origin Next.js bridge and an accessible `/transcribe`
  experience with loading, success, error, copy and public TXT/JSON exports;
- metadata-only observability, boundary tests, local run commands, guarded
  real-model smoke and publication documentation;
- continued availability of the legacy Gradio interface.

Explicitly excluded:

- training, refit, evaluation or final-holdout access;
- model-weight, checkpoint, metric, normalization or split changes;
- translation providers, authentication, PostgreSQL, learning persistence or
  community implementation;
- permanent audio retention, public deployment or Internet exposure;
- experimental inference optimization, microphone capture and automatic
  language detection.

## Acceptance criteria

- `GET /api/health`, `GET /api/v1/languages` and `GET /api/v1/models` expose
  validated public fields and never load a model.
- `POST /api/v1/transcriptions` accepts only an allowlisted extension/MIME,
  bounded payload, known language, known model and compatible pairing.
- Each upload has a random internal temporary name, is never logged or
  persisted, and is removed on success and every tested failure path.
- Inference uses lazy, sequential load/transcribe/unload behavior; an
  unavailable checkpoint fails without exposing its path.
- Unit/integration tests use synthetic fixtures and fake backends only; no
  test downloads or reads a model or dataset.
- `/transcribe` implements accessible idle, selected, processing, success and
  error states plus copy, safe TXT/JSON download, clear and disabled
  translation/microphone actions.
- Full repository and web gates, npm audit, runtime smoke, publication audit
  and `git diff --check` pass with no private artifact or personal path.

## Progress

- 2026-08-13 — verified clean, synchronized `main` at `59090b9` and created
  `feat/web-asr-integration`.
- 2026-08-13 — audited existing model, service, API, web, privacy and harness
  boundaries; assigned backend, frontend and security reviews in parallel.
- 2026-08-13 — implemented versioned discovery, secure temporary upload,
  serialized ASR inference and privacy-safe errors/logs.
- 2026-08-13 — connected the accessible Next.js transcription experience,
  runtime discovery, compatibility routing and public exports.
- 2026-08-13 — completed offline test gates and independent security review.
- 2026-08-13 — ran authorized synthetic FR/EN model smokes, Dioula final and
  Whisper Small load checks, the same-origin Next proxy, and Gradio legacy.
- 2026-08-13 — reproduced and fixed the Next.js `localhost` versus
  `127.0.0.1` same-origin mismatch, then validated its regression tests and
  the real proxy path.

## Decisions

- Keep the MVP request synchronous while retaining status and random request
  identifiers compatible with a later job API.
- Treat `dyu` as the canonical Dioula code; UI locale and audio language stay
  separate.
- Expose connected ASR capabilities as `experimental`: French/English are not
  benchmarked and Dioula evidence remains bounded to its model card.
- Accept WAV, MP3, FLAC and OGG only. The installed SoundFile/libsndfile stack
  does not guarantee M4A/MP4/WebM.
- Limit the web path to 25 MiB and 30 seconds. The existing non-chunked Whisper
  decoding is not declared safe for longer inputs.
- Force Whisper language tokens for `fr` and `en` only; never force a `dyu`
  token, and always keep the task `transcribe`.
- Serialize inference with one active backend per process and document one
  Uvicorn worker per GPU.
- Keep legacy Dummy routes for compatibility while using only `/api/v1` from
  the new web platform.
- Compare mutation `Origin` with the real HTTP `Host` as well as the canonical
  request URL. Next.js dev canonicalizes the internal URL to `localhost` even
  when the browser uses `127.0.0.1`; protocol matching remains mandatory.
- Disable Next.js auto-generated nested agent files because repository-level
  project instructions are already authoritative.

## Validation evidence

- Backend/service/Whisper/smoke targeted tests: 45 passed.
- Next.js proxy regression after the runtime same-origin fix: 17 passed.
- The complete gate before the final TypeScript-only proxy correction passed
  with 241 Python tests and 74 frontend tests. After that correction,
  `make verify-fast` passed again: publication audit 243 candidates, harness
  41 documents/59 Python modules, Ruff and MyPy 66 source files.
- Final frontend gates passed sequentially: Prettier, ESLint, strict
  TypeScript, 76 tests across 11 files and the 11-page Next.js build.
- `npm audit --audit-level=high` reported zero vulnerabilities; dependencies
  and lockfile were unchanged by the final proxy correction.
- Real-model smoke on CUDA with newly generated temporary speech:
  French and English through Whisper Tiny passed; transcript content was not
  printed or persisted.
- Dioula final `checkpoint-002052`: load/unload and public `dyu` routing
  contract passed without using an audio.
- Whisper Small: local cache load/unload passed.
- Next.js `/transcribe`, same-origin proxy health and proxied Whisper Tiny
  transcription passed on loopback.
- Legacy Gradio: loopback launch and HTTP content smoke passed without corpus
  or artifact roots.
- All generated smoke audio and HTML files were deleted from `/tmp`.
- The final full-gate counts are recorded in the Phase 2A handoff report.

## Remaining reservations

- This remains a trusted-operator local MVP. Authentication, rate limiting,
  quotas, a true ASGI body limiter and reviewed public-network controls are
  required before Internet exposure.
- The inference lock is process-local, and cancellation can wait for a running
  worker before cleanup; use one worker per GPU.
- The frozen model hash remains a demo-preflight responsibility rather than a
  liveness check.
- Browser E2E, manual zoom/reader review and a licensed external Dioula speech
  smoke remain follow-up evidence, not hidden claims.
