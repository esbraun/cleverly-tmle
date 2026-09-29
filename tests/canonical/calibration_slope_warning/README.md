# Calibration-slope warning

This directory holds a repeated-sampling study of the calibration-slope rule in
`cleverly.validation.nuisance`. The rule reads the propensity report's calibration slope, a
logistic recalibration of the treatment on `logit(g_hat)` with one intercept per validation fold,
and its sandwich standard error. It warns when the Bonferroni interval over the tested models lies
above 0 and excludes 1.

Each fit is a three-fold `TMLE` at `n = 2000` with a binary treatment and the binary outcome
`Y ~ Bernoulli(expit(0.5 A + 0.5 W1 - 0.25))`. The outcome learner returns that known regression,
so each fit tests two models.

The primary scenarios give the treatment learner the known propensity, at a weak signal
(`logit g0 = 0.15 W1`) and a strong one (`logit g0 = W1 - 0.5 W2`). Their estimand is the
propensity's calibration slope, whose truth is 1.

The property study holds two families. `warning_rate` bounds the rule's false-warning rate on five
laws where a warning is false, and four `fixed_band` controls read the band the package applied
before RM15 on the same fits. `power` measures detection on two tempered learners with limit slopes
of 1/2 and 2.

No canonical implementation is compared. No maintained implementation computes this rule.
`equivalence.csv` is empty and schema-valid.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.calibration_slope_warning.regenerate
```

The declarations are in `tests/studies/calibration_slope_warning.py` and
`tests/studies/calibration_slope_warning_properties.py`. The RM15 plan in `docs/roadmap.md` fixed
them before the run. The reader-facing scope, measurements and limitations are in
[`docs/technical-reference/method-evidence/calibration-slope-warning.md`](../../../docs/technical-reference/method-evidence/calibration-slope-warning.md).
