# Longitudinal incremental propensity-score interventions under dropout

This study validates `IPSI` in `cleverly.longitudinal` with a censoring mechanism. `IPSI` is the
one-step estimator of Kim, Kennedy and Naimi (2021), Algorithm 1 and Theorem 5.1. The
[incremental interventions over time](../longitudinal-tmle.md#incremental-interventions-over-time)
section states the estimator, its conditions and its exact-law evidence.

No maintained package ships this construction with dropout.
So no canonical implementation is compared. The study rests on exact truths and on
repeated-sampling properties. The [complete-data study](longitudinal-incremental-interventions.md)
pairs the same estimator with R `npcausal`, where that package applies. Its comparator runs are
pending.

## What was fitted

| scenario | law and estimands | folds |
| --- | --- | --- |
| `dropout_folds_1` | law `C`: three nodes with a continuous end-of-study outcome and dropout after each arm. The three plan means and their two contrasts with the natural course | one |
| `dropout_folds_5` | law `C`, the same estimands | five |
| `survival_folds_1` | law `D`: three nodes with a binary absorbing event and the same dropout. The risk of each plan by horizons 1, 2 and 3, and the two contrasts at each horizon | one |
| `survival_folds_5` | law `D`, the same estimands | five |

Three plans tilt every node by 0.5, 1 (the natural course) and 2. The laws are in
`tests/discrete_law_longitudinal_incremental.py`. Law `C` is law `A` of the
[complete-data study](longitudinal-incremental-interventions.md) with dropout at random given the
history. The dropout probabilities $\omega_t$ lie in [0.85, 0.97]. Law `D` has event hazards in
[0.05, 0.30]. Every learner is a saturated cell mean.

The one-fold scenarios are primary rows. A one-fold fit with saturated cell means is the plug-in
g-formula on the empirical cell frequencies. Its tilt is nonlinear in $\hat\pi$, so it carries a
bias of order (cells)/n. The conditions for the one-fold interval are in the table below.

| source | what it states |
| --- | --- |
| Kennedy (2019), Algorithm 2 (Appendix 8.3) | the one-fold interval, under Donsker conditions |
| Kim et al. (2021), Theorem 5.1 | the dropout result, with sample splitting only |
| the argument of this study | saturated cell means on a finite history set form a finite-dimensional class. The remainder is a product of $O_P(n^{-1/2})$ errors. The Donsker argument therefore extends the dropout result to one fold |

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

TODO(results): after the run, add one sentence on any red cell and the roadmap row that owns it.
Add the count of failed fits per cell, if any.

Every property fit uses five folds, saturated cell means at every slot and 3,000 draws. The
default size is n = 48,000. `x2` is the plan that doubles the odds of treatment at every node.
`contrast_x2` is its contrast with the natural course.
`tests/studies/longitudinal_incremental_dropout_properties.py` declares every cell.

| family | cells and fits | what must hold |
| --- | --- | --- |
| `interval_calibration` | seven labels with three cells each: the correctly specified fit and two derived controls | SE ratio and coverage intervals inside the calibration bands, and both efficiency-ratio intervals inside their band. The shrunken-SE control falls below the band. The noise control rises above the efficiency-ratio band |
| `simultaneous_coverage` | the band over the risks of three plans at horizons 1 to 3 and their six contrasts, on law `D` at n = 48,000, and its pointwise control | the band's joint coverage interval lies inside the calibration coverage band. The pointwise control's joint coverage interval lies below the nominal rate |
| `clustered_inference` | `ey_regimen[odds x2]` on law `Ck` in 800 clusters of 10 (n = 8,000): the cluster-robust standard error, and the standard error that treats each unit as independent (the control) | the cluster-robust arm is calibrated. The control's SE-ratio interval lies below 0.80. The coverage gain of the pair is at least 0.03 |
| `censoring_complete_case` | `ey_regimen[odds x2]` on law `C`, fitted to the rows that never drop out, with no censoring model | no verdict. The cell prints its bias beside the exact complete-case limit (`complete_case_limit`) |

The calibration labels:

| label | law and estimand | n |
| --- | --- | --- |
| `dropout_contrast_x2` | `C`, `contrast_x2` at the end of study | 48,000 |
| `survival_contrast_x2_h3` | `D`, `contrast_x2` of the risk by horizon 3, read off the band fit | 48,000 |
| `dropout_contrast_x2_in_sample` | `C`, `contrast_x2`, one fold (Kennedy 2019, Algorithm 2) | 48,000 |
| `survival_contrast_x2_h3_in_sample` | `D`, `contrast_x2` of the risk by horizon 3, one fold, on the band draws | 48,000 |
| `competing_contrast_x2_h3` | `Dc`, `contrast_x2` of the cause-1 incidence by horizon 3 | 96,000 |
| `msm_log_delta` | `C`, the slope of an identity-link working model linear in $\log\delta$ over the three plans | 48,000 |
| `weighted_contrast_x0_5` | `C` with the informative weight $1 + A_1 + W/2$, `odds x0.5 vs natural course`. The truth is the weighted law's | 48,000 |

Law `Dc` adds a second, competing cause at each node. Law `Ck` is law `C` with a shared binary
cluster effect. The one-fold cells are gated like the cross-fitted ones. The
`censoring_complete_case` cell shows why the weights matter: a fit that ignores dropout sits away
from the target, and the cell reports how far.

## Sizing

The five-fold contrasts carry a second-order bias of order 1/n, so their standardized bias falls
as $1/\sqrt n$. The sizing probe in `tests/diagnostics/x19_sizing/` measured the bias at
n = 8,000. The size rule requires that every primary row passes the bias rule with probability at
least 0.95 at the projected bias. It requires 0.90 for each calibration cell.

| quantity | value | source |
| --- | --- | --- |
| standardized bias at n = 8,000, dropout contrasts | 0.31 | `tests/studies/canonical_longitudinal_incremental_dropout.py` |
| one-fold standardized bias at n = 8,000 | about 0.2 | the same |
| primary n | 80,000 | the same |
| calibration replications | 3,000. At 2,000 an ideal cell passes with probability 0.935, and at 3,000 with 0.996 | `tests/studies/longitudinal_incremental_dropout_properties.py` |
| calibration n | 48,000. The calibration cells pass jointly with probability 0.93 | the same, with `tests/unit/test_longitudinal_incremental_design.py` |
| competing-risk n | 96,000. At n = 8,000 a training fold can hold one or two treated units of a rare history, and the censoring fit reaches its lower bound. The SE ratio reads 0.948 at 32,000 and 0.980 at 64,000 | `tests/studies/longitudinal_incremental_dropout_properties.py` |
| clustered cell | n = 8,000 and 3,000 draws. The calibration rule cannot pass at 1,000 | the same |
| weighted truth | the informative weight moves the truth by 9.5 standard errors at n = 48,000 | `tests/unit/test_longitudinal_incremental_design.py` |
| independent-unit control | the exact SE ratio is 0.60 and the exact coverage is 0.76 | the same |
| complete-case limit | 6.2 standard errors from the truth | the same |
| band control | the pointwise intervals cover jointly with exact probability 0.695. The control clears the control-power rule at that value plus 0.005 | the same |

The fallback-read bias bound holds far below these sizes. The study fixed every size,
replication count, margin, law, learner, fold count and seed before any run.

## Red-cell route

| event | route, declared before any run |
| --- | --- |
| a red primary truth row | blocks the merge and gets a diagnosis |
| a red property cell that reads as a finite-sample limit | is re-registered `reporting` with an owner row before any re-run |
| a replication that raises | is never redrawn. The failing cell drops to its owner with its failure count stated as a page limit, and the run repeats without it |
| any verdict | moves no budget, margin, law, learner, fold count or seed. Controls carry the verdict their family states |

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
exact standard errors of the efficient influence curve at the primary n. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 80000 | observations per primary replication |
| `independent_tests_passed` | TODO(results) | truth tests passing |
| `independent_tests_total` | TODO(results) | truth tests reported |
| `property_cells_passed` | TODO(results) | property cells passing |
| `property_cells_total` | TODO(results) | property cells reported |
| `max_standardized_bias` | TODO(results) | largest primary standardized bias |
| `min_coverage` | TODO(results) | lowest primary coverage |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:iid_control_se_ceiling` | 0.8000 | independent-unit control SE-ratio ceiling |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `bound:ate_regimen[odds x0.5 vs natural course @ t=1]_standard_error` | 0.000359 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x0.5 vs natural course @ t=2]_standard_error` | 0.000675 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x0.5 vs natural course @ t=3]_standard_error` | 0.000901 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x0.5 vs natural course]_standard_error` | 0.0034 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x2 vs natural course @ t=1]_standard_error` | 0.000451 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x2 vs natural course @ t=2]_standard_error` | 0.000762 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x2 vs natural course @ t=3]_standard_error` | 0.000962 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x2 vs natural course]_standard_error` | 0.0035 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ey_regimen[natural course]_standard_error` | 0.0089 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ey_regimen[odds x0.5]_standard_error` | 0.0088 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ey_regimen[odds x2]_standard_error` | 0.0088 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[natural course @ t=1]_standard_error` | 0.0014 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[natural course @ t=2]_standard_error` | 0.0018 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[natural course @ t=3]_standard_error` | 0.0019 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x0.5 @ t=1]_standard_error` | 0.0015 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x0.5 @ t=2]_standard_error` | 0.0020 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x0.5 @ t=3]_standard_error` | 0.0021 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x2 @ t=1]_standard_error` | 0.0014 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x2 @ t=2]_standard_error` | 0.0019 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x2 @ t=3]_standard_error` | 0.0021 | exact influence-curve standard error of this estimand at the primary n = 80,000 |

## Limits

| limit | what it means for use |
| --- | --- |
| dropout at random given the history, after each arm | the laws set $\omega_t$ in [0.85, 0.97]. Heavier dropout and informative dropout are outside this evidence |
| saturated cell means at every slot | the study does not test a flexible learner. The one-fold claim rests on the finite-dimensional class |
| one-fold primary rows rest on an argument | Kim et al. (2021) state the dropout result with sample splitting only. The Donsker argument above is the extension, and no source states it |
| finite-sample bias is sized, not removed | at n = 8,000 the standardized bias reads 0.31 on the dropout contrasts. The rows pass at the sizes above, and smaller sizes are not claimed |
| three nodes and horizons 1 to 3 | longer panels and later horizons are not claimed |
| no comparator | no pinned implementation fits this design, so the study has no paired verdict |
| the clustered cell reads one mean | `ey_regimen[odds x2]` on law `Ck` in 800 clusters of 10. Fewer clusters are unmeasured |
| the complete-case cell states no verdict | it reports a bias beside the exact limit |

## Reproduction

`tests/canonical/longitudinal_incremental_dropout/regenerate.py` regenerates the study. It runs no
reference process, because the study has no comparator.
