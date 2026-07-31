# Reliability contract

## Expected behavior

- public models continue when the private pilot checkpoint is unavailable;
- one model failure does not discard successful comparison results;
- models load sequentially and release memory after use;
- audio validation rejects unsupported, oversized or invalid inputs early;
- training stops on non-finite loss;
- experiment selection and split violations fail closed;
- unit tests remain offline and deterministic.

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

## Recovery

- model and UI failures are retriable without modifying data;
- pilot training checkpoints are external and resumable;
- frozen dataset artifacts are immutable;
- generated private artifacts can be regenerated from authorized sources;
- repository changes are recoverable through normal Git branches and commits.

Real-model smoke tests are optional because they require authorized local
assets. Their absence must be reported, never silently treated as success.
