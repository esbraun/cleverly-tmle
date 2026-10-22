# Longitudinal refutation verdict rules

This study calibrates three decision rules that `cleverly.refute` prints on a longitudinal fit.
Each rule reads a refutation statistic and rejects at the 0.05 level. The
[verdicts and evidence](../validation-methods.md#verdicts-and-evidence) section of the validation
reference defines each statistic.

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 Markov placebo refits against the exact null, `t` with 19 degrees of freedom | the two-sided p-value is below 0.05 |
| subset `rho` | 20 subset refits against the sampling reference of the fit's standard error | the two-sided $\chi^2_{19}$ p-value is below 0.05 |
| negative-control `z` | one refit on a declared negative-control outcome | the two-sided p-value is below 0.05 |

The study asks two questions for each rule and configuration. Is the rule grossly miscalibrated
under its null? Does the rule detect a declared defect? A rule that passes both checks prints a
verdict on a fit inside the calibrated scope. On any other fit, the rule prints its numbers and
no verdict.

**No canonical implementation is compared.** No maintained implementation computes these rules
on a longitudinal fit. The zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| sample | $n = 1000$ units, two treatment nodes, one fit per trial |
| law E | an end-of-study law with a Gaussian outcome and a time-varying confounder $L_2$. An unmeasured $U$ drives the first treatment and the confounded control only. The contrast `always` against `never` is 1.2 |
| law S | the registered censored survival law of `cleverly.datasets.make_longitudinal_survival`, read as the risk contrast at $t = 2$ |
| cell L1 | `LTMLE` with static regimens, GLM nuisances and five folds, on law S. It reads the placebo and the subset rule |
| cell S1 | `SDR` with static regimens, GLM nuisances and five folds, on law E |
| cell I1 | `IPSI` at odds `x2` against the natural course, GLM nuisances and five folds, on law E |
| cell B1 | `SequentialGComputation` with a Gaussian GLM, in sample, on law E |
| cell B2 | `SequentialIPW` (Hajek) with a logistic GLM, in sample, on law E |
| cell B3 | `ParametricGFormula` with GLM models and 10 simulation draws per unit, in sample, on law E |
| size cells | 200 trials per rule and cell. A cell passes at no more than 17 rejections |
| power cells | trials 0 to 49 of the size cell under a declared defect. A cell passes at no fewer than 38 detections |
| defects | a stale-row placebo for the placebo rule, a halved and a doubled reported standard error for the subset rule, and a confounded control for the negative-control rule |
| primary phase | 200 replications of the S1 contrast on law E and of the L1 risk contrast on law S, as a sanity check of the fits that the rules read |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

The X82 plan fixed the laws, the cells, the thresholds, the seeds and the red-cell route before
the run. The module docstring of `tests/studies/refuter_calibration_longitudinal.py` states both
laws and each defect.

A size cell with 200 trials checks gross miscalibration only. A rule with true size 0.05 passes with
probability 0.988, and a rule with true size 0.15 passes with probability 0.004. A power cell with
50 trials passes with probability 1.000 at true power 0.95. It passes with probability 0.814 at true
power 0.80.

## Accuracy against known truth

The primary phase checks the fits that the rules read. It does not calibrate a rule.

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law E, two nodes, a Gaussian outcome and a time-varying confounder; `SDR` at five folds | `ate` | average treatment effect | `cleverly` fit that the longitudinal refutation rules read | -0.0303 to 0.0065 | 0.9150 | 0.9271 | **fail** |
| law S, the registered censored survival law, risk at t = 2; `LTMLE` at five folds | `ate` | average treatment effect | `cleverly` fit that the longitudinal refutation rules read | -0.0068 to 0.0085 | 0.9450 | 0.9993 | **fail** |
<!-- /generated -->

## Theory properties

A `refuter_size` cell counts the rejections of one rule under its null. A `refuter_power` cell
counts the detections of one declared defect on the first 50 trials of the same configuration.
The cell name gives the configuration, then the reading.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `refuter_power` | `B1__negative_control_confounded` | control | SequentialGComputation, law E: the negative-control z rule, on a control that reads an unmeasured cause of A1 | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B1__placebo_stale_rows` | control | SequentialGComputation, law E: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B1__subset_double_se` | control | SequentialGComputation, law E: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 48 of 50 | pass |
| `refuter_power` | `B1__subset_half_se` | control | SequentialGComputation, law E: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B2__negative_control_confounded` | control | SequentialIPW, law E: the negative-control z rule, on a control that reads an unmeasured cause of A1 | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `B2__placebo_stale_rows` | control | SequentialIPW, law E: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B2__subset_double_se` | control | SequentialIPW, law E: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B2__subset_half_se` | control | SequentialIPW, law E: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B3__negative_control_confounded` | control | ParametricGFormula, law E: the negative-control z rule, on a control that reads an unmeasured cause of A1 | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B3__placebo_stale_rows` | control | ParametricGFormula, law E: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B3__subset_double_se` | control | ParametricGFormula, law E: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `B3__subset_half_se` | control | ParametricGFormula, law E: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I1__negative_control_confounded` | control | IPSI at odds x2, law E: the negative-control z rule, on a control that reads an unmeasured cause of A1 | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I1__placebo_stale_rows` | control | IPSI at odds x2, law E: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I1__subset_double_se` | control | IPSI at odds x2, law E: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `I1__subset_half_se` | control | IPSI at odds x2, law E: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `L1__placebo_stale_rows` | control | LTMLE, law S: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `L1__subset_double_se` | control | LTMLE, law S: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `L1__subset_half_se` | control | LTMLE, law S: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `S1__negative_control_confounded` | control | SDR, law E: the negative-control z rule, on a control that reads an unmeasured cause of A1 | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `S1__placebo_stale_rows` | control | SDR, law E: the placebo t rule, with a declared share of stale rows | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `S1__subset_double_se` | control | SDR, law E: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `S1__subset_half_se` | control | SDR, law E: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_size` | `B1__negative_control` | positive | SequentialGComputation, law E: the negative-control z rule on a valid control | no more than 17 rejections in 200 trials | rejections 7 of 200 | pass |
| `refuter_size` | `B1__placebo` | positive | SequentialGComputation, law E: the placebo t rule against the exact Markov null | no more than 17 rejections in 200 trials | rejections 10 of 200 | pass |
| `refuter_size` | `B1__subset` | positive | SequentialGComputation, law E: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 12 of 200 | pass |
| `refuter_size` | `B2__negative_control` | positive | SequentialIPW, law E: the negative-control z rule on a valid control | no more than 17 rejections in 200 trials | rejections 10 of 200 | pass |
| `refuter_size` | `B2__placebo` | positive | SequentialIPW, law E: the placebo t rule against the exact Markov null | no more than 17 rejections in 200 trials | rejections 10 of 200 | pass |
| `refuter_size` | `B2__subset` | positive | SequentialIPW, law E: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 11 of 200 | pass |
| `refuter_size` | `B3__negative_control` | positive | ParametricGFormula, law E: the negative-control z rule on a valid control | no more than 17 rejections in 200 trials | rejections 10 of 200 | pass |
| `refuter_size` | `B3__placebo` | positive | ParametricGFormula, law E: the placebo t rule against the exact Markov null | no more than 17 rejections in 200 trials | rejections 14 of 200 | pass |
| `refuter_size` | `B3__subset` | positive | ParametricGFormula, law E: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 10 of 200 | pass |
| `refuter_size` | `I1__negative_control` | positive | IPSI at odds x2, law E: the negative-control z rule on a valid control | no more than 17 rejections in 200 trials | rejections 6 of 200 | pass |
| `refuter_size` | `I1__placebo` | positive | IPSI at odds x2, law E: the placebo t rule against the exact Markov null | no more than 17 rejections in 200 trials | rejections 11 of 200 | pass |
| `refuter_size` | `I1__subset` | positive | IPSI at odds x2, law E: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 11 of 200 | pass |
| `refuter_size` | `L1__placebo` | positive | LTMLE, law S: the placebo t rule against the exact Markov null | no more than 17 rejections in 200 trials | rejections 10 of 200 | pass |
| `refuter_size` | `L1__subset` | positive | LTMLE, law S: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 13 of 200 | pass |
| `refuter_size` | `S1__negative_control` | positive | SDR, law E: the negative-control z rule on a valid control | no more than 17 rejections in 200 trials | rejections 6 of 200 | pass |
| `refuter_size` | `S1__placebo` | positive | SDR, law E: the placebo t rule against the exact Markov null | no more than 17 rejections in 200 trials | rejections 12 of 200 | pass |
| `refuter_size` | `S1__subset` | positive | SDR, law E: the subset rho rule, both tails | no more than 17 rejections in 200 trials | rejections 9 of 200 | pass |
<!-- /generated -->

## Result

All 40 property cells pass. The size cells reject in 6 to 14 of 200 trials, below the limit of
17. The power cells detect their defect in 48 to 50 of 50 trials, above the floor of 38. The
registry `VERDICT_EVIDENCE` therefore holds one row for each of the 17 size readings. Each row's
scope is the configuration that its cells recorded in `cells.csv`.

`tests/unit/test_refuter_calibration_longitudinal_registry.py` checks each scope against the
passing cells of the committed run.

The two primary truth rows are red. 200 replications cannot resolve the coverage floor or the
bias margin of the shared primary phase. The lower 99% coverage limits are 0.8513 and 0.8894,
below the floor of 0.90. The end-of-study bias interval, -0.0303 to 0.0065, contains 0 and
crosses the margin of -0.0250. The SE-ratio intervals contain 1. The
[red-cell ledger](red-cells.md) names the owner.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 200 | primary replications |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 2 | primary tests against truth |
| `independent_tests_passed` | 0 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 40 | repeated-sampling property cells |
| `property_cells_passed` | 40 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.1186 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9150 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.8235 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1508 | highest bootstrap primary SE-ratio endpoint |
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
| There is no cross-implementation evidence | No maintained implementation computes these rules on a longitudinal fit |
| Gross miscalibration only | A rule with true size 0.08 passes a size cell with probability 0.663. The check does not certify the size to two decimals |
| One law and one size per cell | Each row covers $n \ge 1000$ at no more than 0.003 covariates per row. A smaller fit or a wider history prints no verdict |
| Two nodes, two arms and GLM nuisances only | A fit with another learner, another node count or another regimen kind prints no verdict |
| No negative-control row on a survival fit | `refute` refuses a negative-control outcome on a survival fit, because no replacement keeps the risk sets |
| I1's iterated regression is misspecified | Under the odds tilt the node-1 regression is not linear. The mechanisms are correct, so the nulls stay exact. The I1 subset cell passed with 11 rejections |
| B3's simulation error is outside its standard error | The subset scatter includes the g-formula's Monte Carlo error. Only the shipped 10 draws per unit are in scope |
| The primary phase is under-resolved | 200 replications cannot put the 99% coverage limit above 0.90. The two truth rows are red with an owner, and no rule row reads them |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_longitudinal/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_longitudinal/manifest.json)
records the seeds, the margins, the laws, the source hashes and the result hashes. The
[cell configurations](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_longitudinal/cells.csv)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_longitudinal/properties.csv)
carry every published row.
