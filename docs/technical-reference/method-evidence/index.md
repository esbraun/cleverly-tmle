# Implementation validation studies

Each page here reports one study of a method that `cleverly` implements. A page exists only for
a study registered in `tests/studies/evidence/registry.py`. The shared machinery in
`tests/studies/evidence/` computes every verdict, and `tests/unit/test_method_evidence.py` checks
these pages against the committed results. Every table is generated from those results, so a stale
number is a test failure and not a reading error.

Read the pages in this order.

| page | what it gives you |
| --- | --- |
| [Implementation validation grid](validation-grid.md) | the eighty-five validation studies in one table, with the counts and the declared limits |
| [How to read these studies](how-to-read.md) | the three questions, the verdict rules, and the terms every study below applies |
| [The red-cell ledger](red-cells.md) | every red verdict the studies publish, and the roadmap ask that owns each one |
| the eighty-seven study pages | one row per committed test, with what it checked and the verdict its own endpoints produced |

A validation study declares property cells or names a reference. A study that names a reference
is comparator-backed, and its agreement block names the comparison kind of each row. The
[learned-rule boundary study](learned-rule-cvtmle-boundary.md) declares neither, so it has a page
and no grid row.

To register a new study, follow
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row).

```{toctree}
:maxdepth: 2

validation-grid
how-to-read
red-cells
canonical-point-treatment-tmle
weighted-point-treatment-tmle
learned-weighted-point-treatment-tmle
ordinary-multi-arm-tmle
stratified-point-treatment-tmle
stratified-incremental-msm
stratified-msm-identity
stratified-dr-tmle
stacked-point-treatment-cv-tmle
clustered-point-treatment-cv-tmle
clustered-unequal-cv-tmle
clustered-few-cluster-tmle
clustered-cross-fitted-longitudinal-tmle
few-cluster-cross-fitted-longitudinal-tmle
fold-evaluated-point-treatment-cv-tmle
fold-targeted-point-treatment-cv-tmle
repeated-cross-fitting
repeated-cross-fitted-longitudinal
selector-based-point-treatment-c-tmle
selector-based-multi-arm-c-tmle
c-tmle-candidate-sequences
ctmle-derivative
outcome-adaptive-point-treatment-c-tmle
outcome-adaptive-multi-arm-c-tmle
outcome-adaptive-per-arm-c-tmle
outcome-adaptive-per-arm-c-tmle-strata-missing
outcome-adaptive-per-arm-c-tmle-parametric
outcome-adaptive-per-arm-c-tmle-parametric-crossfit
outcome-adaptive-per-arm-c-tmle-parametric-extensions
canonical-dr-tmle
multi-arm-dr-tmle
ordinary-missing-outcome-tmle
ordinary-missing-outcome-natural-course-tmle
stacked-missing-outcome-natural-course-cvtmle
stacked-arm-indexed-missing-outcome-cvtmle
missing-outcome-attributable-effects-tmle
controlled-direct-effect-tmle
randomized-missing-outcome-dr-tmle
observational-missing-data-dr-tmle
randomized-multi-arm-missing-outcome-dr-tmle
point-treatment-msm-projection
deterministic-point-treatment-regimes
learned-rule-cvtmle
learned-rule-cvtmle-boundary
stochastic-point-treatment-regimes
continuous-modified-treatment-policies
point-modified-treatment-policies
incremental-propensity-interventions
ordinary-end-of-study-longitudinal-tmle
ordinary-weighted-end-of-study-longitudinal-tmle
ordinary-categorical-longitudinal-tmle
stochastic-categorical-longitudinal-tmle
longitudinal-modified-treatment-policies
longitudinal-incremental-interventions
longitudinal-incremental-interventions-dropout
categorical-incremental
vector-modified-treatment-policies
mtp-boundaries
mtp-pricing
mtp-tuning
sequentially-doubly-robust-estimation
ordinary-longitudinal-msm-projection
cross-fitted-longitudinal-msm-projection
cross-fitted-end-of-study-longitudinal-tmle
cross-fitted-weighted-end-of-study-longitudinal-tmle
cross-fitted-categorical-longitudinal-tmle
ordinary-survival-curve-longitudinal-tmle
cross-fitted-survival-curve-longitudinal-tmle
ordinary-competing-risk-longitudinal-tmle
cross-fitted-competing-risk-longitudinal-tmle
point-treatment-survival
cross-fitted-point-treatment-survival
point-survival-hazard
point-survival-curve
point-survival-curve-crossfit
point-treatment-survival-policies
survival-node-bounds
known-treatment-mechanism
known-mechanism-dr-tmle
dr-tmle-one-inconsistent-nuisance
adaptive-iptw
known-node-mechanisms-ltmle
sequential-baselines
structural-nested-mean
parametric-gformula
repeated-outcome-longitudinal-tmle
repeated-outcome-ipsi
repeated-outcome-sdr
omitted-variable-bound-standard-error
omitted-variable-bound-plugin-standard-error
ovb-limit-coverage
evalue-limit-coverage
calibration-slope-warning
refuter-calibration-longitudinal
default-simultaneous-bands
full-refit-bootstrap-and-derived-contrasts
```
