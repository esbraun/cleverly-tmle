# Point-treatment refutation verdict rules

This study calibrates three decision rules that `cleverly.refute` prints on a point-treatment fit.
Each rule reads a refutation statistic and rejects at the 0.05 level. The
[refutation operations](../validation-methods.md#refutation-operations) section of the validation
reference defines each statistic.

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 permutation placebo refits against the exact null, `t` with 19 degrees of freedom | the two-sided p-value is below 0.05 |
| subset `rho` | 20 subset refits against the sampling reference of the fit's standard error | the two-sided $\chi^2_{19}$ p-value is below 0.05 |
| negative-control `z` | one refit on a declared negative-control outcome | the two-sided p-value is below 0.05 |

The study asks two questions for each rule and configuration. Is the rule grossly miscalibrated
under its null? Does the rule detect a declared defect? A rule that passes both checks prints a
verdict on a fit inside the calibrated scope. On any other fit, the rule prints its numbers and
no verdict.

**No canonical implementation is compared.** No maintained implementation computes these rules.
The zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| sample | $n = 2000$ units, 20 standard normal covariates, one fit per trial, five-fold cross-fitting |
| law | a binary treatment with $\operatorname{logit} g = 0.4 W_1 - 0.3 W_2 + 0.05 (W_3 + \dots + W_{20}) + 0.5 U$. The unmeasured $U$ drives the treatment and the confounded control only |
| Gaussian outcome | $Y = A + 0.5 W_1 + 0.4 W_2 + 0.05 (W_3 + \dots + W_{20}) + e$, with $e \sim N(0, 1)$. The ATE is 1 |
| binary outcome | $\operatorname{logit} P(Y = 1) = -2.65 + 0.5 A + 0.5 W_1 + 0.4 W_2 + 0.05 (W_3 + \dots + W_{20})$. The prevalence is 0.100 |
| negative controls | a valid control with the outcome's covariate part and no treatment term, and a confounded control that also reads $U$. Each control has the outcome's family |
| cell T1 | `TMLE` with linear and logistic GLM nuisances, on the Gaussian law. It reads all three rules |
| cell T2 | `TMLE` with logistic GLM nuisances, on the binary law. It reads all three rules |
| cell T3 | `TMLE` with depth-2 `HistGradientBoostingClassifier` nuisances (100 iterations, early stopping off), on the binary law. It reads the placebo rule |
| cell C1 | `CTMLE` with the greedy strategy and logistic GLM nuisances, on the binary law. It reads the placebo rule |
| cell D1 | `DRTMLE` with the treatment mechanism declared known and a logistic GLM outcome, on the binary law. It reads all three rules |
| cell D2 | `DRTMLE` with an estimated mechanism and logistic GLM nuisances, on the binary law. It reads the placebo rule |
| size cells | 200 trials per rule and cell. A cell passes at no more than 17 rejections |
| power cells | trials 0 to 49 of the size cell under a declared defect. A cell passes at no fewer than 38 detections |
| defects | a stale-row placebo for the placebo rule, a halved and a doubled reported standard error for the subset rule, and a confounded control for the negative-control rule |
| primary phase | 200 replications of the T1 and T2 ATE, as a sanity check of the fits that the rules read |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

The X82 plan fixed the law, the cells, the thresholds, the seeds and the red-cell route before the
run. The module docstring of `tests/studies/refuter_calibration_point.py` states the law and each
defect. `tests/unit/test_refuter_calibration_point_design.py` sizes each defect from the law.

A size cell with 200 trials checks gross miscalibration only. A rule with true size 0.05 passes with
probability 0.988, and a rule with true size 0.15 passes with probability 0.004. A power cell with
50 trials passes with probability 1.000 at true power 0.95. It passes with probability 0.814 at true
power 0.80.

## Accuracy against known truth

The primary phase checks the fits that the rules read. It does not calibrate a rule.

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the point law with 20 covariates and a binary outcome of prevalence 0.100; TMLE at five folds | `ate` | average treatment effect | `cleverly` fit that the point-treatment refutation rules read | -0.0018 to 0.0031 | 0.9650 | 1.0462 | **fail** |
| the point law with 20 covariates and a Gaussian outcome; TMLE at five folds | `ate` | average treatment effect | `cleverly` fit that the point-treatment refutation rules read | -0.0035 to 0.0140 | 0.9400 | 1.0087 | **fail** |
<!-- /generated -->

## Theory properties

A `refuter_size` cell counts the rejections of one rule under its null. A `refuter_power` cell
counts the detections of one declared defect on the first 50 trials of the same configuration.
The cell name gives the configuration, then the reading.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `refuter_power` | `C1__placebo_stale_rows` | control | greedy C-TMLE, binary outcome: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 45 of 50 | pass |
| `refuter_power` | `D1__negative_control_confounded` | control | DR-TMLE with a declared mechanism, binary outcome: the negative-control z rule, on a control that reads an unmeasured cause of A | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `D1__placebo_stale_rows` | control | DR-TMLE with a declared mechanism, binary outcome: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 48 of 50 | pass |
| `refuter_power` | `D1__subset_double_se` | control | DR-TMLE with a declared mechanism, binary outcome: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `D1__subset_half_se` | control | DR-TMLE with a declared mechanism, binary outcome: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `D2__placebo_stale_rows` | control | DR-TMLE with an estimated mechanism, binary outcome: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 47 of 50 | pass |
| `refuter_power` | `T1__negative_control_confounded` | control | TMLE with GLM nuisances, Gaussian outcome: the negative-control z rule, on a control that reads an unmeasured cause of A | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `T1__placebo_stale_rows` | control | TMLE with GLM nuisances, Gaussian outcome: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `T1__subset_double_se` | control | TMLE with GLM nuisances, Gaussian outcome: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `T1__subset_half_se` | control | TMLE with GLM nuisances, Gaussian outcome: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `T2__negative_control_confounded` | control | TMLE with GLM nuisances, binary outcome: the negative-control z rule, on a control that reads an unmeasured cause of A | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `T2__placebo_stale_rows` | control | TMLE with GLM nuisances, binary outcome: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 44 of 50 | pass |
| `refuter_power` | `T2__subset_double_se` | control | TMLE with GLM nuisances, binary outcome: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `T2__subset_half_se` | control | TMLE with GLM nuisances, binary outcome: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `T3__placebo_stale_rows` | control | TMLE with a depth-2 booster, binary outcome: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_size` | `C1__placebo` | positive | greedy C-TMLE, binary outcome: the placebo t rule against the exact permutation null | no more than 17 rejections in 200 trials | rejections 11 of 200 | pass |
| `refuter_size` | `D1__negative_control` | positive | DR-TMLE with a declared mechanism, binary outcome: the negative-control z rule on a valid control | no more than 17 rejections in 200 trials | rejections 8 of 200 | pass |
| `refuter_size` | `D1__placebo` | positive | DR-TMLE with a declared mechanism, binary outcome: the placebo t rule against the exact permutation null | no more than 17 rejections in 200 trials | rejections 14 of 200 | pass |
| `refuter_size` | `D1__subset` | positive | DR-TMLE with a declared mechanism, binary outcome: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 5 of 200 | pass |
| `refuter_size` | `D2__placebo` | positive | DR-TMLE with an estimated mechanism, binary outcome: the placebo t rule against the exact permutation null | no more than 17 rejections in 200 trials | rejections 5 of 200 | pass |
| `refuter_size` | `T1__negative_control` | positive | TMLE with GLM nuisances, Gaussian outcome: the negative-control z rule on a valid control | no more than 17 rejections in 200 trials | rejections 10 of 200 | pass |
| `refuter_size` | `T1__placebo` | positive | TMLE with GLM nuisances, Gaussian outcome: the placebo t rule against the exact permutation null | no more than 17 rejections in 200 trials | rejections 8 of 200 | pass |
| `refuter_size` | `T1__subset` | positive | TMLE with GLM nuisances, Gaussian outcome: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 9 of 200 | pass |
| `refuter_size` | `T2__negative_control` | positive | TMLE with GLM nuisances, binary outcome: the negative-control z rule on a valid control | no more than 17 rejections in 200 trials | rejections 11 of 200 | pass |
| `refuter_size` | `T2__placebo` | positive | TMLE with GLM nuisances, binary outcome: the placebo t rule against the exact permutation null | no more than 17 rejections in 200 trials | rejections 14 of 200 | pass |
| `refuter_size` | `T2__subset` | positive | TMLE with GLM nuisances, binary outcome: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 7 of 200 | pass |
| `refuter_size` | `T3__placebo` | positive | TMLE with a depth-2 booster, binary outcome: the placebo t rule against the exact permutation null | no more than 17 rejections in 200 trials | rejections 9 of 200 | pass |
<!-- /generated -->

## Result

All 27 property cells pass. The size cells reject in 5 to 14 of 200 trials, below the limit of 17.
The power cells detect their defect in 44 to 50 of 50 trials, above the floor of 38. The registry
`VERDICT_EVIDENCE` therefore holds one row for each of the 12 size readings. Each row's scope is
the configuration that its cells recorded in `cells.csv`. A negative-control row reads the event
count of the binary control, which the registry test recomputes from the declared draws.

`tests/unit/test_refuter_calibration_point_registry.py` checks each scope against the passing cells
of the committed run.

The two primary truth rows are red. The primary phase has 200 replications, which cannot resolve
its coverage floor, bias margin or SE-ratio screen. The Gaussian coverage limit is 0.8829,
below the floor of 0.90, and its bias interval crosses the margin. The binary SE-ratio interval,
0.9118 to 1.2189, crosses the screen limit of 1.20. Both SE-ratio intervals contain 1. The
[red-cell ledger](red-cells.md) names the owner.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 200 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | primary tests against truth |
| `independent_tests_passed` | 0 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 27 | repeated-sampling property cells |
| `property_cells_passed` | 27 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.1098 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9400 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.8712 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.2189 | highest bootstrap primary SE-ratio endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the level of each rule |
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
| There is no cross-implementation evidence | No maintained implementation computes these rules |
| Gross miscalibration only | A rule with true size 0.08 passes a size cell with probability 0.663. The check does not certify the size to two decimals |
| One law and one size per cell | Each row covers $n \ge 2000$ at no more than 0.01 covariates per row. A smaller fit or more covariates per row prints no verdict |
| Two arms, the ATE and the study learners only | A fit with another learner, estimand, arm count or fold count prints no verdict. The C1 row covers the greedy strategy only |
| Three rules on three cells | T3, C1 and D2 read only the placebo rule. Their subset and negative-control readings print no verdict |
| The stale-row power is approximate | The design sizes the population shift of the placebo. The shift of one trial varies about it. T2's stale-row cell detected 44 of 50 |
| D1's stale rows also break the declared mechanism | The placebo column keeps the declared arm shares on the stale rows. The cell still measures the rule's response to a wrong placebo |
| No future-treatment row | The point study ran no future-treatment exposure, so that reading stays descriptive |
| The primary phase is under-resolved | 200 replications cannot put the 99% coverage limit above 0.90. The two truth rows are red with an owner, and no rule row reads them |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_point/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_point/manifest.json)
records the seeds, the margins, the law, the source hashes and the result hashes. The
[cell configurations](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_point/cells.csv)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_point/properties.csv)
carry every published row.
