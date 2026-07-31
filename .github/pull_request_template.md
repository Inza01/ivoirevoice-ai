## Intent

Describe the observable outcome and link the active execution plan when one is
required.

## Acceptance criteria

- [ ] Product behavior and non-goals are explicit.
- [ ] Relevant tests cover success and failure paths.
- [ ] Architecture, security and experiment boundaries remain valid.

## Validation

List exact commands and results. Do not write “tests pass” without evidence.

```text
make verify
```

## Agent self-review

- [ ] I inspected the complete diff.
- [ ] I preserved unrelated work.
- [ ] I updated repository knowledge when behavior changed.
- [ ] I recorded unresolved work in `docs/technical-debt.md`.
- [ ] I did not add corpus data, per-sample text, secrets or checkpoints.
- [ ] I did not mix the 600-audio and 150-audio experiments.

## Human decisions required

List licensing, data, deployment, final-holdout or publication decisions that
cannot be inferred safely.
