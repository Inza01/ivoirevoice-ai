# Reliability contract

## Expected behavior

- public baselines continue when the private final checkpoint is unavailable;
- one model failure does not discard successful comparison results;
- models load sequentially and release memory after use;
- audio validation rejects unsupported, oversized or invalid inputs early;
- training stops on non-finite loss;
- experiment selection and split violations fail closed;
- full training refuses CPU fallback and any run without working CUDA/FP16;
- final-holdout access is consumed and sealed to one frozen model identity;
- unit tests remain offline and deterministic.
- web uploads are removed after success, validation failure, model failure or
  cancellation;
- liveness and discovery never create a backend or initialize CUDA;
- one inference lock per process prevents concurrent model residency;
- public API errors remain stable and omit backend messages and private paths.

## Feedback loops

| Loop | Command | Purpose |
|---|---|---|
| Environment | `scripts/verify_environment.py` | Python and lightweight registry |
| Publication | `scripts/audit_repository.py` | secrets, private files and sizes |
| Harness | `scripts/check_harness.py` | knowledge, links and architecture |
| Static | `make lint typecheck` | syntax, style and type contracts |
| Behavioral | `make test` | offline application behavior |
| Complete | `make verify` | required handoff and CI gate |

## Observability

Current observability consists of explicit errors, experiment metrics,
processing time, RTF, hardware metadata and persisted aggregate reports.
Logs and metrics are not yet exposed through a dedicated local observability
stack; this is tracked as technical debt rather than implied as implemented.

The web-ASR adapter emits metadata-only request events containing a generated
request ID, public model ID, language, elapsed time, status and safe error code.
It deliberately omits audio, original filename, temporary path, transcription,
backend exception and model path.

## Recovery

- model and UI failures are retriable without modifying data;
- a web transcription failure leaves no retained upload and the next request
  starts from a fresh lazy model lifecycle;
- pilot and final checkpoints are external to Git;
- full development/refit checkpoints are atomic, bounded and content-addressed;
- the terminal `EVALUATED` holdout receipt permanently blocks further
  training and holdout access;
- frozen dataset artifacts are immutable;
- generated private artifacts can be regenerated from authorized sources;
- repository changes are recoverable through normal Git branches and commits.

Real-model smoke tests are optional because they require authorized local
assets. Their absence must be reported, never silently treated as success.
