# Multi-arm outcome-adaptive C-TMLE versus R `ctmle3`

This registered study compares Cleverly's three-arm outcome-adaptive construction with
archived `ctmle3` 0.1.0. The R snapshot's categorical counterfactual GLM adapter drops a
factor column, so the shared evidence law is expressed through numeric arm codes on the R
side and labelled arms on the Cleverly side. Its outcome mean is linear in those codes,
making both nuisance fits exactly specified.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.multi_arm_ctmle3_oat.regenerate
```

The reader-facing results are in
[`outcome-adaptive-multi-arm-c-tmle.md`](../../../docs/technical-reference/method-evidence/outcome-adaptive-multi-arm-c-tmle.md).
