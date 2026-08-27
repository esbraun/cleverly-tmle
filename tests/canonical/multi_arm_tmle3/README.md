# Ordinary multi-arm TMLE versus R `tmle3`

This registered study gives Cleverly and pinned R `tmle3` 0.2.0 the same labelled
three-arm samples. Both fit an intercept-only outcome regression, the same correctly
specified multinomial-logistic treatment model, and pointwise 95% intervals. The
working outcome regression makes targeting load-bearing while the treatment model
retains the estimator's consistency. The
published parameters are all three arm means plus the two reference-arm ATEs, risk
ratios, and odds ratios.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.multi_arm_tmle3.regenerate
```

The reader-facing results are in
[`ordinary-multi-arm-tmle.md`](../../../docs/technical-reference/method-evidence/ordinary-multi-arm-tmle.md).
