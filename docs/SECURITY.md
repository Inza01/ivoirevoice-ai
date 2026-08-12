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

## Required controls

- Run `make audit-repository` before staging publication changes.
- Run `make verify` before handoff.
- Never use the validation or test split as training data.
- Never enable Gradio public sharing in repository defaults.
- Never download or publish the private pilot checkpoint automatically.
- Keep detailed predictions under `IVOIREVOICE_ARTIFACTS_DIR`.
- Never persist per-sample references, predictions, speaker identifiers or
  audio paths from the one-time final holdout, even under the private artifact
  root; only aggregate counters are allowed.
- Treat `EVALUATION_IN_PROGRESS`, `EVALUATED` and
  `EVALUATION_FAILED_AFTER_ACCESS` as irreversible holdout-consumption states.
- Rotate a credential immediately if a real secret enters Git history; do not
  rewrite history without explicit authorization.

## Review triggers

Require human review for changes to:

- data licensing, consent or retention;
- split assignments or final-holdout access;
- public networking, authentication or deployment;
- checkpoint distribution;
- secret-handling and publication rules.

The threat model assumes a trusted local operator. The MVP is not hardened for
untrusted public uploads or internet-facing deployment.
