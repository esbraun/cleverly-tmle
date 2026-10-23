# Band-rank placebo verdict rule

This study calibrates the placebo `t` rule that `cleverly.refute` prints on a continuous dose
whose declared support is a function. On such a dose the placebo draws each dose's position inside
its band, then maps the position into the receiving unit's band. The
[band-rank placebo](../validation-methods.md#refutation-operations) paragraph of the validation
reference defines the draw.

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 band-rank placebo refits against the exact null, `t` with 19 degrees of freedom | the two-sided p-value is below 0.05 |

The study asks two questions for each configuration. Is the rule grossly miscalibrated under its
null? Does the rule detect a declared defect? A rule that passes both checks prints a verdict on a
fit inside the calibrated scope. On any other fit, the rule prints its numbers and no verdict.

**No canonical implementation is compared.** No maintained implementation computes this rule.
The zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| sample | $n = 2000$ units, one fit per trial, five-fold cross-fitting, 20 placebo refits per reading |
| law P | $W_1, W_2 \sim U(-1, 1)$. The dose band is $[L, U]$ with $L = W_1 + 0.5 W_2$ and $U - L = 2 + 0.75 W_1$. The position $u = (A - L) / (U - L)$ is a normal with mean $0.5 + 0.2 W_2$ and SD 0.25, truncated to $[0, 1]$. $Y = A + 0.8 W_1 + 0.6 W_2 + e$, with $e \sim N(0, 1)$ |
| law L | the shape of the `make_sequential_pricing` law: three nodes, a binary engagement $L_t$ and a price $A_t$. The band at node $t$ has the lower end $2 + L_t + 0.5 A_{t-1}$ and the width $2 + L_t$. The position follows a truncated normal with SD 0.4 that reads $W$ and the earlier position. The outcome is binary |
| cell PJ1 | `TMLE` of the policy `up` against the natural course on law P. `up` adds 0.5 to the dose where the result stays in the band. An OLS outcome and a classifier density ratio with a quadratic logistic. The draw is `band_permute` |
| cell LJ1 | `LTMLE` of the regimen `up` against the natural course on law L. `up` adds 0.25 to each price where the result stays in the band. Logistic GLM outcome and pseudo-outcome regressions, and the same classifier ratio. The draw is `band_markov` |
| size cells | 200 trials per cell. A cell passes at no more than 20 rejections |
| power cells | trials 0 to 49 of the size cell under a stale-row placebo. A cell passes at no fewer than 38 detections |
| defect | a declared share of units keeps its own dose (PJ1, 0.10) or its whole observed trajectory (LJ1, 0.15). The other units keep their band-rank draws |
| primary phase | 200 replications of the PJ1 contrast against its exact truth, as a sanity check of the fit that the rule reads |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

`X82-J-plan.md` rev 2 fixed the laws, the cells, the seeds and the red-cell route before the run.
The design review set the size cutoff to 20, because the 20 refits of one PJ1 trial share one
remainder of the OLS outcome. The module docstring of `tests/studies/refuter_calibration_band.py`
states each law and the defect. `tests/unit/test_refuter_calibration_band_design.py` reads the
sizing probe in `tests/diagnostics/x82_sj_sizing/`.

| quantity | PJ1 | LJ1 |
| --- | --- | --- |
| placebo bias, in placebo SD | -0.031 | 0.000 |
| stale-row shift, in placebo SD | 1.51 | 1.02 |
| probe trials | 100 | 60 |

A size cell with 200 trials checks gross miscalibration only. A rule with true size 0.05 passes with
probability 0.999, and a rule with true size 0.15 passes with probability 0.025. A power cell with
50 trials passes with probability 1.000 at true power 0.95. It passes with probability 0.814 at true
power 0.80.

## Accuracy against known truth

The primary phase checks the fit that the rule reads. It does not calibrate the rule.

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law P: a continuous dose in a band [L(W), U(W)] whose location and width read W; TMLE of a capped +0.5 shift at five folds | `ate` | average treatment effect | `cleverly` fit that the band-rank placebo rule reads | -0.0024 to 0.0078 | 0.9500 | 1.0145 | **fail** |
<!-- /generated -->

## Theory properties

A `refuter_size` cell counts the rejections of the rule under its null. A `refuter_power` cell
counts the detections of the stale-row defect on the first 50 trials of the same configuration.
The cell name gives the configuration, then the reading.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `refuter_power` | `LJ1__placebo_stale_rows` | control | LTMLE of a dose shift at three nodes on the pricing-shaped law L: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `PJ1__placebo_stale_rows` | control | TMLE of a capped dose shift on law P: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_size` | `LJ1__placebo` | positive | LTMLE of a dose shift at three nodes on the pricing-shaped law L: the placebo t rule against the exact band-rank null | no more than 20 rejections in 200 trials | rejections 12 of 200 | pass |
| `refuter_size` | `PJ1__placebo` | positive | TMLE of a capped dose shift on law P: the placebo t rule against the exact band-rank null | no more than 20 rejections in 200 trials | rejections 10 of 200 | pass |
<!-- /generated -->

## Result

All four property cells pass. The size cells reject in 10 (PJ1) and 12 (LJ1) of 200 trials, below
the limit of 20. The power cells detect the defect in 50 (PJ1) and 49 (LJ1) of 50 trials, above the
floor of 38. The registry `VERDICT_EVIDENCE` therefore holds one placebo row for each cell. Each
row's scope is the configuration that its cells recorded in `cells.csv`.

`tests/unit/test_refuter_calibration_band_registry.py` checks each scope against the passing cells
of the committed run.

The primary truth row is red. The primary phase has 200 replications, which cannot resolve its
coverage floor or bias margin. The coverage limit is 0.8960, below the floor of 0.90. The bias
interval, -0.0024 to 0.0078 in outcome units, crosses the margin of 0.0070 in the same units. The SE-ratio interval contains 1. The [red-cell ledger](red-cells.md)
names the owner.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 200 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 1 | primary tests against truth |
| `independent_tests_passed` | 0 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 4 | repeated-sampling property cells |
| `property_cells_passed` | 4 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0978 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9500 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9052 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1631 | highest bootstrap primary SE-ratio endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the level of the rule |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:type_i_ceiling` | 0.1000 | the rate a type-I cell must bound. No cell of this study reads it; a size cell reads its count |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power family must clear. No cell of this study reads it; a power cell reads its count |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 2 | the same bound, in primary replications |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | The fast tier recomputes each verdict and does not fail on a red one |
| There is no cross-implementation evidence | No maintained implementation computes this rule |
| Gross miscalibration only | A rule with true size 0.10 passes a size cell with probability 0.559. The check does not certify the size to two decimals |
| One law and one size per cell | Each row covers $n \ge 2000$ at the cell's covariates per row. A smaller fit prints no verdict |
| The study estimand and node count only | The PJ1 row covers the recorded parameter `ate_policy[up vs natural course]`, as the SG policy rows do. A policy with another name prints no verdict. The LJ1 row covers the recorded stem `ate_regimen`, because a longitudinal fit records the stem. It therefore covers any regimen contrast at three nodes with the study learners. A fit with another fold count or node count prints no verdict |
| The study learners only | Both rows name the fingerprint of `QuadraticLogistic`, a learner class of the study module in `tests/studies`. A user fit gets a verdict only through a learner with that exact fingerprint. In practice a banded user fit stays descriptive |
| The classifier ratio only | `ratio="density"` ran in no cell, so that reading stays descriptive |
| The stale-row power is approximate | The probe sizes the population shift of the placebo. The shift of one trial varies about it. LJ1's stale-row cell detected 49 of 50 |
| The primary phase is under-resolved | 200 replications cannot put the 99% coverage limit above 0.90. The truth row is red with an owner, and no rule row reads it |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_band/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_band/manifest.json)
records the seeds, the margins, the laws, the source hashes and the result hashes. The
[cell configurations](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_band/cells.csv)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_band/properties.csv)
carry every published row.
