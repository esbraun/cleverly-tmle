# Adaptive TMLE with external data

This study measures `AdaptiveTMLEMethod` on a randomized trial pooled with external rows
(`TrialWithExternalData`, `TrialATE`). [Adaptive TMLE](../adaptive-tmle.md) gives the estimand, the
estimator and its conditions. The interval of a fit covers the working-model projection
$\Psi_w$ of the dictionary the fit selected. It covers the trial effect $\Psi$ only under the
oracle-approximation condition of the source, which no data check confirms. The study therefore
reads each scenario against the target its law supports and reports the coverage of all three
targets ($\Psi$, $\Psi_2$ and $\Psi_w$) beside it.

The study declares no property cell and no reference, so it is not a validation row, as
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row) states, and
the [validation grid](validation-grid.md) does not list it. The method therefore has no grid row.
The study keeps the `gated` policy of its declaration, as a named exception: its own hooks read the
primary rows, the controls and the failure accounting, and the run publishes only if every gate
passes. The declared run passed all of them.

**No canonical implementation is compared.** The R `atmle` comparator is declared and pending, as
a separate pairing record on constant truths, because a record whose truth varies by replicate
cannot declare a pending reference. It never blocks this study.

## What was tested

| setting | declaration |
| --- | --- |
| laws | the App B laws of the source (a) to (d), a positivity stress law, a separating law `e1`, a heterogeneous-effect law `h2`, a binary-outcome law `b1y`, controls-only external data, and missing outcomes. The trial share is 0.25, independent of $W$, which the source omits |
| fit | `study.identify(TrialATE(population)).estimate(method=AdaptiveTMLEMethod(...))` with spline learners, five outer and five inner folds, the default linear-plus-hinge dictionary, and only each working model's constant unpenalized |
| size | n = 4,000 pooled rows: 1,000 trial and 3,000 external. The ladder scenarios run at n = 1,000 and 2,000 |
| primary budget | 2,800 replications of each of 17 scenarios, set by the repeated-sampling sizing rule at an 800-replicate pilot |
| truth | quadrature of $\Psi$ and $\Psi_2$ from the law, and of $\Psi_w$ on each fit's stored fold transformers, so the truth varies by replicate |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20290311` and `resampling_seed=20290312` |
| policy | `gated` |

Each scenario gates one target. `a0`, `a1a`, `a1b`, `a3`, `a3_pi`, `a4`, `a5`, `h2_pooled`, `f1a`
and `f1b` gate $\Psi$, on laws where the dictionary spans the external bias. `h2_trial` gates
$\Psi_2$. `a2c`, `a2d`, `e1` and the two ladder scenarios gate the projection, because the bias
is outside the dictionary's span. `b1y` is a reported stress row and gates nothing. The laws,
scenarios, controls, budgets and seeds were fixed before any run, in the T3 plan and
`tests/studies/adaptive_tmle_trial.py`, and
`tests/unit/test_adaptive_tmle_trial_design.py` recomputes the truths, the separations, the
control powers and the sizing without a run.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the paper's law (a) with no external bias, cross-validated form, n = 4,000 | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0019 to 0.0026 | 0.9464 | 0.9639 | pass |
| the paper's law (a): external bias 0.2 + 1.1 W1 (1 - A), cross-validated form | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0081 to -0.0012 | 0.9532 | 0.9988 | pass |
| the paper's law (b): external bias 0.5 + 3.1 W1 (1 - A) + 0.8 W3, cross-validated form | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0097 to -0.0025 | 0.9479 | 0.9918 | pass |
| the paper's law (c): uniform covariates, external bias with an indicator outside the dictionary's span | `projection` | the working-model projection of the fit's own selected dictionary, which the interval covers whether or not the dictionary spans the truth | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0043 to 0.0025 | 0.9468 | 0.9886 | pass |
| the paper's law (d): uniform covariates, external bias with a squared term outside the dictionary's span | `projection` | the working-model projection of the fit's own selected dictionary, which the interval covers whether or not the dictionary spans the truth | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0036 to 0.0031 | 0.9543 | 1.0110 | pass |
| positivity stress: the enrollment probability depends on W and is small for some W | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0022 to -0.000763 | 0.9457 | 0.9851 | pass |
| the positivity-stress law with a plain logistic regression for the enrollment probability, which is misspecified | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0026 to -0.0011 | 0.9354 | 0.9540 | pass |
| the paper's law (b) with controls-only external data | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0055 to 0.0015 | 0.9536 | 1.0230 | pass |
| the paper's law (b) with missing outcomes, the external missingness depending on the outcome | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0076 to -0.000312 | 0.9600 | 1.0431 | pass |
| reported stress row, passing by declaration: a binary outcome, the external control risk a shrunken trial risk, so the external bias passes through expit | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0103 to -0.0082 | 0.8900 | 0.8851 | pass |
| the separating law: the selected projection differs from the trial effect by at least three standard errors | `projection` | the working-model projection of the fit's own selected dictionary, which the interval covers whether or not the dictionary spans the truth | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0034 to 0.0034 | 0.9554 | 1.0273 | pass |
| the separating law, unpenalized main-terms working model, n = 1,000 (rung of the bias ladder) | `projection` | the working-model projection of the fit's own selected dictionary, which the interval covers whether or not the dictionary spans the truth | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0367 to -0.0200 | 0.9489 | 1.0238 | pass |
| the separating law, unpenalized main-terms working model, n = 2,000 (rung of the bias ladder) | `projection` | the working-model projection of the fit's own selected dictionary, which the interval covers whether or not the dictionary spans the truth | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0182 to -0.0071 | 0.9475 | 1.0006 | pass |
| the paper's law (a), form='pooled' (no cross-fitting of the working model) | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0040 to 0.0028 | 0.9429 | 0.9753 | pass |
| the paper's law (b), form='pooled' (no cross-fitting of the working model) | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0062 to 0.000691 | 0.9489 | 0.9857 | pass |
| a heterogeneous effect, so the effect among all rows differs from the trial's; the pooled population | `psi` | the trial effect over the pooled covariate distribution, which the interval covers when the working model approximates the external bias | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | -0.0053 to 0.000355 | 0.9554 | 1.0063 | pass |
| the heterogeneous-effect law with the target population set to the trial rows | `psi_2` | the trial effect over the trial rows' own covariate distribution | `cleverly` adaptive TMLE of a trial pooled with external data: a lasso-selected working model for the effect and one for the external bias, a refit, and a fluctuation of the enrollment mechanism, with an influence-curve Wald interval | 0.0038 to 0.0089 | 0.9336 | 0.9472 | pass |
<!-- /generated -->

All 16 gated tests pass, and so does the reported row by declaration. No fit failed.
Coverage of the gated target runs from 0.9336 to 0.9600 and the SE ratio from 0.947 to 1.043. The
lowest coverage is `h2_trial` (0.9336) and `a3_pi` (0.9354). The highest SE ratio is `a5` (1.043),
the scenario with missing outcomes.

## Controls

Each control rides on the draws of a host scenario and must fail where it should. The efficiency
contrast is the one comparison against a different estimator, the trial-only TMLE.

| control | host | what it reads | expected | measured |
| --- | --- | --- | --- | --- |
| efficiency | `a0` | RMSE of adaptive TMLE over RMSE of the trial-only TMLE, upper 99% bound below 1 | gates | 0.682, 99% interval 0.655 to 0.708 |
| efficiency | `a1a`, `a1b` | the same ratio | reported, no gain guaranteed | 1.018 and 1.048 |
| K1, naive pooled TMLE | `a1b` | coverage of $\Psi$ when the external rows are treated as exchangeable | below the 0.90 floor | 0.846 |
| K2, main-terms working model | `e1` | coverage of the projection, and of $\Psi$ | projection passes, $\Psi$ fails | 0.956 and 0.0146 |
| K3, with and without the $\Pi$ step | `a3_pi` | coverage with the targeted fit, and with the $\Pi$ fluctuation removed | targeted passes, removed fails | 0.935 and 0.537 |
| K5, pooled fit read against $\Psi_2$ | `h2_pooled` | coverage of $\Psi_2$ | below the 0.90 floor | 0.0246 |

The efficiency contrast gates on `a0` only. There the external rows carry no bias and the
selection that the oracle would make leaves the constant alone, which gives the influence function
0.48 of the trial-only standard deviation. On `a1a` and `a1b` a gain needs the lasso to leave out
every column that gives the external arms a free mean shift. The design review read ratios of
about 1.05 and 1.14 on scratch draws, so those two rows report the ratio. The committed run reads
1.018 and 1.048, with upper bounds 1.034 and 1.068: no gain there, and a small loss.

## The declared stress row

`b1y` has a binary outcome, and the external control risk is a shrunken trial risk, so the
external bias passes through `expit`. The guarantee of the method is pointwise (Theorem 1 and
Appendix C Theorem 2 of arXiv:2405.07186v2, and Theorems 5 to 7 of arXiv:2307.12544v2): the
interval covers $\Psi$ only when the selected working model approximates the bias, and the
linear-plus-hinge dictionary does not span an `expit` bias. The two external arms' biases differ
by $\Psi^\# = -0.0148$, about 1.3 standard errors of that difference. The declaration expects the lasso at
n = 4,000 to keep only the constant in most replications, so that the projection misses $\Psi$ by
about $\Psi^\#$. The committed projection offset, 0.44 error SDs, agrees with that reading.

The committed run reads that. The interval covers $\Psi$ in 0.8900 of replications, with a
standardized bias of -0.4405 and an SE ratio of 0.8851, and covers the projection in 0.9400. No
uniformly valid interval can be much shorter than the robust one at a zero difference (Hodges and
Le Cam super-efficiency; Leeb and Pötscher on post-selection inference; Armstrong and Kolesar
(2018) on adaptive intervals), so the row measures the cost of the narrower interval and gates
neither the coverage of $\Psi$, the bias nor the standard error. It is a stress row, and a defect
would show elsewhere in the table. The failure bound still gates it, and no replicate was lost.

## The bias ladder

Control K4 refits the unpenalized main-terms working model on `e1` at n = 1,000, 2,000 and 4,000.
On `e1` that projection differs from $\Psi$ by a fixed 0.3207, so the ratio of the bias of $\Psi$
to its standard error is expected to grow with n while the interval for the projection keeps its
coverage. The committed run shows both. The ratio for $\Psi$ is 1.67, 2.70 and 4.01 at the three
sizes, the same ratio for the projection stays at 0.161, 0.111 and 0.122, and the projection
coverage stays at 0.949, 0.948 and 0.956. The coverage of $\Psi$ falls from 0.609 to 0.225 to 0.0146.
K2 is the n = 4,000 rung, and the ladder is a reporting row that fails no verdict.

## Result

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 17 of 17 | `performance-tests.csv` |
| lost replicates | 0 | `manifest.json` and the run log |
| controls passed | 12 of 12 | `controls.csv` |

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2800 | primary replications per scenario |
| `n` | 4000 | observations per primary replication |
| `independent_tests_total` | 17 | scenario tests against the replication truth |
| `independent_tests_passed` | 17 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 0 | repeated-sampling property cells, none declared |
| `property_cells_passed` | 0 | of those, passing |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.4405 | largest absolute primary bias in error SDs, over all 17 rows |
| `min_coverage` | 0.8900 | lowest measured primary coverage, over all 17 rows |
| `min_se_ratio_ci_lower` | 0.8509 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0799 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-adaptive-tmle/h2_trial/psi_2]:coverage` | 0.9336 | lowest gated coverage |
| `performance[cleverly-adaptive-tmle/h2_trial/psi_2]:coverage_ci_lower` | 0.9205 | its 99% lower endpoint, against 0.90 |
| `performance[cleverly-adaptive-tmle/a3_pi/psi]:coverage` | 0.9354 | second-lowest gated coverage |
| `performance[cleverly-adaptive-tmle/a5/psi]:coverage` | 0.9600 | highest gated coverage |
| `performance[cleverly-adaptive-tmle/a5/psi]:se_ratio` | 1.0431 | highest gated SE ratio |
| `performance[cleverly-adaptive-tmle/h2_trial/psi_2]:se_ratio` | 0.9472 | lowest gated SE ratio |
| `performance[cleverly-adaptive-tmle/b1y/psi]:coverage` | 0.8900 | reported row, coverage of $\Psi$ |
| `performance[cleverly-adaptive-tmle/b1y/psi]:standardized_bias` | -0.4405 | reported row, bias in error SDs |
| `performance[cleverly-adaptive-tmle/b1y/psi]:se_ratio` | 0.8851 | reported row, SE ratio |
| `reported[b1y]:projection_coverage` | 0.9400 | reported row, coverage of the projection |
| `controls[efficiency/a0]:value` | 0.6819 | RMSE ratio on `a0` |
| `controls[efficiency/a0]:lower` | 0.6553 | its 99% lower bound |
| `controls[efficiency/a0]:upper` | 0.7083 | its 99% upper bound, against 1 |
| `controls[efficiency/a1a]:value` | 1.0183 | RMSE ratio on `a1a`, reported |
| `controls[efficiency/a1b]:value` | 1.0476 | RMSE ratio on `a1b`, reported |
| `controls[k1_naive_pooled/a1b]:value` | 0.8461 | K1, coverage of $\Psi$ |
| `controls[k2_main_terms_projection/e1]:value` | 0.9561 | K2, coverage of the projection |
| `controls[k2_main_terms_psi/e1]:value` | 0.0146 | K2, coverage of $\Psi$ |
| `controls[k3_targeted/a3_pi]:value` | 0.9354 | K3, coverage with the $\Pi$ step |
| `controls[k3_no_pi_step/a3_pi]:value` | 0.5371 | K3, coverage without it |
| `controls[k5_pooled_for_psi_2/h2_pooled]:value` | 0.0246 | K5, coverage of $\Psi_2$ |
| `controls[k4_ladder/e1_k2_n1000]:value` | 1.6656 | K4, bias over SE of $\Psi$ at n = 1,000 |
| `controls[k4_ladder/e1_k2_n2000]:value` | 2.6984 | K4, at n = 2,000 |
| `controls[k4_ladder/e1_k2_n4000]:value` | 4.0102 | K4, at n = 4,000 |
| `controls[k4_ladder/e1_k2_n1000]:lower` | 0.1612 | K4, bias over SE of the projection at n = 1,000 |
| `controls[k4_ladder/e1_k2_n2000]:lower` | 0.1107 | K4, at n = 2,000 |
| `controls[k4_ladder/e1_k2_n4000]:lower` | 0.1216 | K4, at n = 4,000 |
| `reported[e1_k2_n1000]:projection_coverage` | 0.9489 | K4, projection coverage at n = 1,000 |
| `reported[e1_k2_n2000]:projection_coverage` | 0.9475 | K4, at n = 2,000 |
| `reported[e1]:projection_coverage` | 0.9554 | `e1`, the penalized fit, at n = 4,000 |
| `reported[e1_k2_n1000]:psi_coverage` | 0.6093 | K4, coverage of $\Psi$ at n = 1,000 |
| `reported[e1_k2_n2000]:psi_coverage` | 0.2246 | K4, at n = 2,000 |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 28 | failed primary fits that bound allows |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. The study declares no calibration cell |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |

## Limitations

| limitation | what it means for use |
| --- | --- |
| No grid row | The study declares no property cell and no reference, so the validation grid does not list it and the method has no validation row. The fast tier recomputes every verdict and every control from the committed rows |
| There is no cross-implementation evidence | The R `atmle` pairing is declared, pending, and not run. The efficiency contrast compares against the trial-only TMLE, a different estimator |
| The interval covers the projection | The coverage of $\Psi$ is read only on laws where the dictionary spans the external bias. `b1y` shows the cost where it does not, and no data check tells the two cases apart |
| One size and one learner class | Every primary scenario draws n = 4,000 with additive spline learners that are close to correct on each law. The ladder adds n = 1,000 and 2,000 for one law. A flexible learner library is not measured here |
| Selective enrollment is not measured | The enrollment probability depends on $W$ in `a3` and `h2` only. The selective-enrollment study (S6) is separate and has no result yet |
| Binary treatment, difference scale | The study measures the difference of means for a binary treatment. Other treatments and scales refuse before any learner |
| The efficiency gain is measured on `a0` | The ratio of 0.682 holds where the external rows carry no bias. On `a1a` and `a1b` the ratios of 1.02 and 1.05 report no gain |
| Compositions outside the study | Weights, clusters, a second outcome and repeated fits are not measured |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_tmle_trial/README.md)
gives the run command and the sizing. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_tmle_trial/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_tmle_trial/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_tmle_trial/replicates.parquet),
the [controls](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_tmle_trial/controls.csv)
and the [reported readings](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_tmle_trial/reported.csv)
carry every published row.
