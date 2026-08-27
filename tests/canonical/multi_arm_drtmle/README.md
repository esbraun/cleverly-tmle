# Multi-arm DR-TMLE versus R `drtmle`

This registered study gives Cleverly and pinned R `drtmle` 1.1.2 the same samples,
five-fold assignment, and initial out-of-fold nuisance predictions. Each implementation
runs its own armwise reduced regressions and correction cycle. The row uses the reporting
policy because the source theorem is binary; any multi-arm divergence remains visible in
the committed results instead of being converted into a theorem claim.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.multi_arm_drtmle.regenerate
```

The reader-facing results are in
[`multi-arm-dr-tmle.md`](../../../docs/technical-reference/method-evidence/multi-arm-dr-tmle.md).
