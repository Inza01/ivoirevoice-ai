# Security and privacy contract

## Protected assets

- speaker audio and metadata;
- full manifests and individual transcriptions;
- per-sample references and predictions;
- manual listening annotations;
- checkpoints, optimizer state and model caches;
- credentials, tokens, cookies and personal local paths.

These assets remain outside Git and are supplied through validated environment
variables. The repository contains only synthetic fixtures and reviewed
aggregates.

## Trust boundaries

1. Environment variables provide external roots.
2. Configuration loaders reject missing, malformed or unsafe values.
3. Data pipelines derive relative paths and pseudonymous identifiers.
4. UI and export services remove private paths.
5. Publication checks reject forbidden paths, suffixes and credential
   patterns.
6. The Phase 2 `web/` Foundation calls only a same-origin proxy contract; its
   FastAPI origin is server-only and never exposed through `NEXT_PUBLIC_*`.
7. The Phase 2A transcription flow sends audio only through the allowlisted
   same-origin proxy to the versioned FastAPI adapter. The browser never sees
   the private backend origin or a checkpoint path.
8. Uploaded audio is written under a random internal name in a private
   temporary directory, validated, processed once and deleted immediately.

## Required controls

- Run `make audit-repository` before staging publication changes.
- Run `make verify` before handoff.
- Never use the validation or test split as training data.
- Never enable Gradio public sharing in repository defaults.
- Never download or publish the private pilot or final checkpoint automatically.
- Keep detailed predictions under `IVOIREVOICE_ARTIFACTS_DIR`.
- Never persist per-sample references, predictions, speaker identifiers or
  audio paths from the one-time final holdout, even under the private artifact
  root; only aggregate counters are allowed.
- Treat `EVALUATION_IN_PROGRESS`, `EVALUATED` and
  `EVALUATION_FAILED_AFTER_ACCESS` as irreversible holdout-consumption states.
- Rotate a credential immediately if a real secret enters Git history; do not
  rewrite history without explicit authorization.
- Accept only WAV, MP3, FLAC or OGG whose extension, declared MIME, binary
  signature and decoded container agree; reject empty, non-finite, oversized
  or longer-than-30-second audio.
- Keep the web-ASR runtime at one Uvicorn worker and one active inference per
  process so concurrent requests cannot retain several models in VRAM.
- Log only generated request IDs, public model IDs, language, status, safe
  error code and duration; never log audio, filename, path or transcription.
- Keep `IVOIREVOICE_API__AUDIO_RETENTION=delete_immediately`; any other
  retention policy requires a new privacy and consent review.

## Review triggers

Require human review for changes to:

- data licensing, consent or retention;
- split assignments or final-holdout access;
- public networking, authentication or deployment;
- checkpoint distribution;
- secret-handling and publication rules.

The threat model assumes a trusted local operator. The MVP is not hardened for
untrusted public uploads or internet-facing deployment.

The Phase 2A upload path adds content sniffing and immediate deletion but does
not change that threat model. Public deployment remains blocked until
authentication, authorization, rate limiting, abuse controls, explicit CORS
where needed and production privacy-safe observability have been implemented
and reviewed. The current same-origin proxy buffers a bounded multipart body
and is acceptable only for the local MVP.
