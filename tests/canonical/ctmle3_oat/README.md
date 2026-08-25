# Outcome-adaptive C-TMLE versus archived tlverse `ctmle3`

This directory freezes a paired repeated-sampling comparison of
`CTMLE(strategy="oat")` with archived tlverse
[`ctmle3`](https://github.com/tlverse/ctmle3) at commit `a4ea77b`, using the
contemporaneous [`tmle3`](https://github.com/tlverse/tmle3/tree/3a610058cd89c17bb417c15fc891254388787f33) commit
`3a61005` and [`sl3`](https://github.com/tlverse/sl3/tree/821ca890cb8701fdb59f823e28c6356e50d092bc) commit `821ca89`. The base R image is pinned by
digest. `manifest.json` records full package commits, source hashes, configuration, result
hashes, and observed package versions.

Python generates every binary-outcome sample and exact truth. Both implementations fit the
same three-covariate logistic outcome regression, form the outcome-adaptive treatment design
from the two arm-specific outcome predictions, and use a non-cross-fitted pointwise 95%
Wald report. The study checks both treatment-specific means and their ATE, marginal risk
ratio, and marginal odds-ratio transformations, including the log-scale influence curves
for the ratio estimands.

The independent property study exercises Cleverly's stricter public cross-fitted behavior.
It checks the outcome-correct robustness contract against an outcome-wrong control, two
root-n rates, efficiency, calibration, type-I error, power, and a flexible-tree comparison
whose in-sample control must understate uncertainty.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.ctmle3_oat.regenerate
```

For a deliberately underpowered primary-only probe:

```powershell
uv run --extra dev python -m tests.canonical.ctmle3_oat.regenerate --replicates 10 --n 300 --primary-only --output $env:TEMP\ctmle3-oat-probe
```

The parity claim is limited to a binary, two-arm, complete-outcome GLM law and the archived
non-cross-fitted implementation. The archived stack fails on the analogous continuous law
because its continuous-outcome bounds are used as a length-two value in a scalar condition;
that source failure is recorded as a limitation, not treated as a dropped replication.
Outcome-adaptive C-TMLE also does not claim the treatment-correct-only leg of ordinary
double robustness: its treatment mechanism sees the generated outcome-regression design,
not the original covariates. Missing outcomes, weights, clusters, strata, multi-valued
treatment parity, simultaneous or bootstrap intervals, broad learner-library selection,
and severe practical-positivity behavior are outside scope.

The reader-facing measurements and limitations are in
[`docs/technical-reference/method-evidence/outcome-adaptive-point-treatment-c-tmle.md`](../../../docs/technical-reference/method-evidence/outcome-adaptive-point-treatment-c-tmle.md).
