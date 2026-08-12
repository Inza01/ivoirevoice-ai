# Execution plans

Execution plans make non-trivial work resumable by a human or agent without
depending on chat history.

Use a plan when work spans multiple domains, changes an architectural
boundary, modifies data governance or needs more than one validation loop.
Copy `template.md` into `active/`, update it during work, then move it to
`completed/` with exact validation evidence.

A plan must contain:

- status and objective;
- in-scope and out-of-scope work;
- measurable acceptance criteria;
- progress with timestamps or commit references when useful;
- decisions and their rationale;
- exact validation commands and outcomes.

Small single-file fixes may use an ephemeral plan, but durable decisions still
belong in the relevant product, architecture or governance document.
