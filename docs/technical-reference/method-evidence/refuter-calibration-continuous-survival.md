# Continuous-time survival refutation verdict rules

This study calibrates two rules that `cleverly.refute` prints on a risk difference of a
`ContinuousSurvivalTMLE` fit. The
[continuous-time survival refutation](../validation-methods.md#continuous-time-survival-refutation)
paragraphs of the validation reference define the draws and their references.

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 refits on the treatment permuted among the units, against the exact null 0 of a risk difference, `t` with 19 degrees of freedom | the two-sided p-value is below 0.05 |
| subset `rho` | 20 refits on 700 of the 1,000 units, each keeping its outer fold, against the sampling reference of the fit's standard error | the two-sided $\chi^2_{19}$ p-value is below 0.05 |

The study asks two questions for each risk difference. Is the rule grossly miscalibrated under its
null? Does the rule detect a declared defect? A rule that passes both checks prints a verdict on a
fit inside the calibrated scope. On any other fit, the rule prints its numbers and no verdict.

**No canonical implementation is compared.** No maintained implementation computes these rules on
a continuous-time survival fit. The zero-row equivalence artifact records the absence of a
comparator.

## What was tested

| setting | declaration |
| --- | --- |
| sample | $n = 1000$ units, one fit per trial: the default `ContinuousSurvivalTMLE` with ten outer and five inner folds, the default Cox hazard libraries and treatment learner, and the switched variance. 20 refits per reading |
| size law | the registered base law of `continuous-time-survival`: two causes, independent censoring, about half treated, and an arm-specific term in the cause-1 hazard |
| estimands | the risk difference of each cause at the median (1.12) and upper quartile (2.32) of the observed event times. The target times hold 43% to 56% and 70% to 79% of the observed events over the size trials |
| size cells | 600 trials. Each trial's eight readings come from the same 41 fits. A cell passes at no more than 68 rejections |
| power cells | 50 trials on a power law that raises the arm-specific term to 2.0 and the cause-2 arm coefficient to -1.0, under a placebo that keeps 35% of the recorded treatments; and size trials 0 to 49 read with the reported SE halved and doubled. A cell passes at no fewer than 38 detections |
| reporting cells | 100 trials each, placebo only, with censoring that depends on the arm and the covariates, and with about 25% treated. No verdict |
| primary phase | 200 replications of the four differences against their exact truths, as a sanity check of the fit that the rules read |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

The X101 plan (rev 2, section 5) fixed the laws, the cells, the seeds and the red-cell route before
the run. The plan declared 200 size trials and the X82 rule of at most 17 rejections, and required
a predictive family pass of 0.90 before the declaration. The 200-trial pilot read 0.39 at that
size. The owner amended the size cells on 2026-11-19 to 600 trials at no more than 68 rejections,
the largest count whose 99% Clopper-Pearson upper endpoint stays under that of 17 of 200. The
module docstring of `tests/studies/refuter_calibration_continuous_survival.py` states each law and
defect, and `tests/unit/test_refuter_calibration_continuous_survival_design.py` reads the pilot.

The amended size cell is looser than the X82 rule. A rule with true size 0.05 passes with
probability 1.000, 0.08 with 0.998, 0.10 with 0.875, and 0.15 with 0.006. Under 17 of 200 a rule
at 0.10 passes with probability 0.285. A power cell with 50 trials passes with probability 1.000 at
true power 0.95 and 0.814 at true power 0.80.

## Accuracy against known truth

The primary phase checks the fit that the rules read. It does not calibrate the rules.

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the registered two-cause continuous-time survival law at n = 1,000; the default ContinuousSurvivalTMLE at ten folds | `ate_regimen[treated vs control, cause1 @ t=1.12]` | difference in the risk of cause 1 by t = 1.12 between treatment and no treatment | `cleverly` fit that the continuous-time survival refutation rules read | -0.0043 to 0.0075 | 0.9550 | 1.0003 | pass |
| the registered two-cause continuous-time survival law at n = 1,000; the default ContinuousSurvivalTMLE at ten folds | `ate_regimen[treated vs control, cause1 @ t=2.32]` | difference in the risk of cause 1 by t = 2.32 between treatment and no treatment | `cleverly` fit that the continuous-time survival refutation rules read | -0.0068 to 0.0051 | 0.9500 | 1.0506 | **fail** |
| the registered two-cause continuous-time survival law at n = 1,000; the default ContinuousSurvivalTMLE at ten folds | `ate_regimen[treated vs control, cause2 @ t=1.12]` | difference in the risk of cause 2 by t = 1.12 between treatment and no treatment | `cleverly` fit that the continuous-time survival refutation rules read | -0.0018 to 0.0083 | 0.9400 | 1.0131 | **fail** |
| the registered two-cause continuous-time survival law at n = 1,000; the default ContinuousSurvivalTMLE at ten folds | `ate_regimen[treated vs control, cause2 @ t=2.32]` | difference in the risk of cause 2 by t = 2.32 between treatment and no treatment | `cleverly` fit that the continuous-time survival refutation rules read | 0.0014 to 0.0129 | 0.9600 | 1.0350 | **fail** |
<!-- /generated -->

## Theory properties

A `refuter_size` cell counts the rejections of a rule under its null. A `refuter_power` cell counts
the detections of a defect. The cell name gives the estimand, then the reading.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `refuter_power` | `c1_t1.12__placebo_stale_rows` | control | the cause-1 risk difference at t = 1.12: the placebo t rule, with 35% of rows keeping their treatment | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `c1_t1.12__subset_double_se` | control | the cause-1 risk difference at t = 1.12: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 48 of 50 | pass |
| `refuter_power` | `c1_t1.12__subset_half_se` | control | the cause-1 risk difference at t = 1.12: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `c1_t2.32__placebo_stale_rows` | control | the cause-1 risk difference at t = 2.32: the placebo t rule, with 35% of rows keeping their treatment | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `c1_t2.32__subset_double_se` | control | the cause-1 risk difference at t = 2.32: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 48 of 50 | pass |
| `refuter_power` | `c1_t2.32__subset_half_se` | control | the cause-1 risk difference at t = 2.32: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 49 of 50 | pass |
| `refuter_power` | `c2_t1.12__placebo_stale_rows` | control | the cause-2 risk difference at t = 1.12: the placebo t rule, with 35% of rows keeping their treatment | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `c2_t1.12__subset_double_se` | control | the cause-2 risk difference at t = 1.12: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 46 of 50 | pass |
| `refuter_power` | `c2_t1.12__subset_half_se` | control | the cause-2 risk difference at t = 1.12: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `c2_t2.32__placebo_stale_rows` | control | the cause-2 risk difference at t = 2.32: the placebo t rule, with 35% of rows keeping their treatment | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_power` | `c2_t2.32__subset_double_se` | control | the cause-2 risk difference at t = 2.32: the subset rho rule, at the reported SE doubled | at least 38 detections in 50 trials | detections 48 of 50 | pass |
| `refuter_power` | `c2_t2.32__subset_half_se` | control | the cause-2 risk difference at t = 2.32: the subset rho rule, at the reported SE halved | at least 38 detections in 50 trials | detections 50 of 50 | pass |
| `refuter_size` | `c1_t1.12__placebo` | positive | the cause-1 risk difference at t = 1.12: the placebo t rule against the exact null 0 | no more than 68 rejections in 600 trials | rejections 28 of 599 | pass |
| `refuter_size` | `c1_t1.12__subset` | positive | the cause-1 risk difference at t = 1.12: the subset rho rule against the fit's standard error | no more than 68 rejections in 600 trials | rejections 57 of 599 | pass |
| `refuter_size` | `c1_t2.32__placebo` | positive | the cause-1 risk difference at t = 2.32: the placebo t rule against the exact null 0 | no more than 68 rejections in 600 trials | rejections 27 of 599 | pass |
| `refuter_size` | `c1_t2.32__subset` | positive | the cause-1 risk difference at t = 2.32: the subset rho rule against the fit's standard error | no more than 68 rejections in 600 trials | rejections 48 of 599 | pass |
| `refuter_size` | `c2_t1.12__placebo` | positive | the cause-2 risk difference at t = 1.12: the placebo t rule against the exact null 0 | no more than 68 rejections in 600 trials | rejections 33 of 599 | pass |
| `refuter_size` | `c2_t1.12__subset` | positive | the cause-2 risk difference at t = 1.12: the subset rho rule against the fit's standard error | no more than 68 rejections in 600 trials | rejections 55 of 599 | pass |
| `refuter_size` | `c2_t2.32__placebo` | positive | the cause-2 risk difference at t = 2.32: the placebo t rule against the exact null 0 | no more than 68 rejections in 600 trials | rejections 33 of 599 | pass |
| `refuter_size` | `c2_t2.32__subset` | positive | the cause-2 risk difference at t = 2.32: the subset rho rule against the fit's standard error | no more than 68 rejections in 600 trials | rejections 52 of 599 | pass |
<!-- /generated -->

## Result

All 20 property cells pass. One of the 600 size trials was lost to a recorded fitting failure (a
subset refit whose initial hazard fit put a combined jump of one or more at one time), under the
failure bound of 0.01, so each size cell reads 599 trials.

The placebo cells reject in 28, 27, 33 and 33 trials, rates of 0.045 to 0.055. The subset cells
reject in 57, 48, 55 and 52 trials (cause 1 at 1.12 and 2.32, then cause 2), rates of 0.080 to
0.095. Every subset cell's 99% interval lies above 0.05, so the subset rule rejects 1.6 to 1.9
times as often as the 0.05 level it targets. It passes because the amended limit tolerates rates up
to about 0.11.

The power cells detect their defects in 46 to 50 of 50 trials. The reporting cells reject in 2% to 9% of their trials.

The registry `VERDICT_EVIDENCE` holds one placebo row and one subset row. Each covers the recorded
configuration of the size cells: `ContinuousSurvivalTMLE` with two causes, the stem
`ate_regimen`, the default learners, the switched variance, ten folds, 20 refits, $n \ge 1000$, at
least 17 events in every arm and cause by the target time, and an event share of 0.4336 to 0.7946.
A fit with `variance="influence_curve"` prints no verdict. The subset row's sentence states its
measured false-alarm rate. `tests/unit/test_refuter_calibration_continuous_survival_registry.py`
checks each scope against the committed run.

Three primary truth rows are red. The `cause1 @ t=2.32` row cannot resolve its coverage floor or
SE-ratio screen at 200 replications. The `cause2 @ t=1.12` row cannot resolve its coverage floor or
bias margin. The `cause2 @ t=2.32` row reads a bias interval of 0.0014 to 0.0129, which excludes 0
and crosses the margin of 0.0078. Every SE-ratio interval contains 1, and no rule row reads these
rows. The [red-cell ledger](red-cells.md) names the owner.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 200 | primary replications |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 4 | primary tests against truth |
| `independent_tests_passed` | 1 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 20 | repeated-sampling property cells |
| `property_cells_passed` | 20 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.2293 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9400 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.8918 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.2106 | highest bootstrap primary SE-ratio endpoint |
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
| There is no cross-implementation evidence | No maintained implementation computes these rules on this fit |
| A looser size tolerance than the X82 rule | A rule with true size 0.10 passes a size cell with probability 0.875. A pass does not certify the size to two decimals, and the subset rule measured 0.080 to 0.095 |
| The subset rule over-rejects | About one fit in eleven fails the subset check with nothing wrong. Read a subset failure with that rate in mind |
| One law and one size | Each row covers $n \ge 1000$ at 0.004 covariates per row, two causes and the measured event shares. A smaller fit, a single-cause fit or a target time outside the shares prints no verdict |
| The default fit only | The rows name the default Cox libraries and treatment learner, ten folds and the switched variance. Any other learner, fold count or variance prints no verdict |
| Risk differences only | A ratio or an arm's risk prints no verdict |
| The primary phase is under-resolved | 200 replications cannot put the 99% coverage limit above its floor. Three truth rows are red with an owner, and no rule row reads them |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_continuous_survival/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_continuous_survival/manifest.json)
records the seeds, the margins, the laws, the source hashes and the result hashes. The
[cell configurations](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_continuous_survival/cells.csv)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/refuter_calibration_continuous_survival/properties.csv)
carry every published row.
