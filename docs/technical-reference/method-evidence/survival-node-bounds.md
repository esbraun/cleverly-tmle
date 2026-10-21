# Survival node bounds

This study validates `LTMLE(node_bounds=..., hazard_pooling="pooled")` on a held point-treatment
survival design with five visits. It fits two models on every sample. The `mean_model` is the
iterated-mean recursion with every node bounded. The `hazard_model` is the hazard fit with
per-horizon targeting and every node's hazard bounded. The sections
[node bounds](../longitudinal-tmle.md#node-bounds),
[hazard-based survival targeting](../longitudinal-tmle.md#hazard-based-survival-targeting) and
[pooled censoring](../longitudinal-tmle.md#pooled-censoring) state the estimators and their exact-law
evidence.

The declared comparator is R [`survtmle`](https://cran.r-project.org/package=survtmle) 1.1.1 with
`method = "mean"` for the `mean_model` and `method = "hazard"` for the `hazard_model`
(Benkeser, Carone and Gilbert 2018). Its runs are pending as issue #302. This page holds Python
property evidence, with comparator runs pending.

## What was compared

The R column states the declared design of the pending comparator runs.

| setting | `cleverly` | R `survtmle` |
| --- | --- | --- |
| datasets | 7,200 samples of 2,000 rows for each model, from the law of `tests/studies/survival_node_bounds_law.py` | the identical rows |
| law | binary `W1` and `W2` with $P(W_1 = 1) = 0.3$ and $P(W_2 = 1) = 0.5$, a binary arm with $g(1 \mid W) = \operatorname{expit}(-0.5 + 0.5 W_1 - 0.4 W_2)$, an event hazard $\operatorname{expit}(-4.5 + 0.5 A + 0.6 W_1 - 0.4 W_2 + 0.1 t)$ and a censoring hazard $\operatorname{expit}(-4.0 + 0.2 A + 0.3 W_1 + 0.15 t)$, for $t = 1, \ldots, 5$ | the same |
| reported parameters | the risk of each arm at visits 3 and 5, and the difference of the arms at visits 3 and 5 | the same, one `t0` per call |
| bounds | $l_t = 0$ and $u_t$ equal to 1.1 times the largest at-risk iterated mean at node $t$ (`mean_model`), or 1.1 times the largest true hazard at node $t$ (`hazard_model`). Both hold the truth | `bounds=` with the same values |
| outcome model | a quasibinomial regression on `W1` and `W2` for each arm and node (`mean_model`). One unpenalized logistic hazard on `trt*(W1 + W2) + t`, pooled over the nodes (`hazard_model`) | `glm.ftime = "trt*(W1 + W2)"` and `"trt*(W1 + W2) + t"` |
| treatment model | an unpenalized logistic regression on `W1 + W2` | `glm.trt = "W1 + W2"` |
| censoring model | one unpenalized logistic regression on `trt + W1 + W2 + t`, pooled over nodes 2 to 5 (`hazard_pooling="pooled"`). Node 1 retains every unit, and the fit declares that retention known | `glm.ctime = "trt + W1 + W2 + t"`, with $g_{dC} = 1$ at the first visit |
| targeting | `targeting="iterative"`, stopped at a relative score of 1e-10 or after 100 iterations, in sample | one cause at a time, `maxIter = 100` |
| intervals | pointwise 95% Wald from the influence curve | the same |

Each truth and each efficiency bound is a finite sum over the four covariate cells, the two arms
and the ten observed histories. The design test
`tests/unit/test_survival_node_bounds_design.py` checks each one on a second route. The quasibinomial
iterated mean is additive on the logit scale and is not saturated. Its exact-law standardized bias
is below 1e-3 standard deviations at $n$ = 2,000.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the same law; the pooled-hazard fit with per-horizon targeting, every node's hazard bounded | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` | -0.000198 to 0.000463 | 0.9492 | 0.9976 | pass |
| the same law; the pooled-hazard fit with per-horizon targeting, every node's hazard bounded | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | `cleverly` | -0.000346 to 0.000549 | 0.9432 | 0.9832 | pass |
| the same law; the pooled-hazard fit with per-horizon targeting, every node's hazard bounded | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | `cleverly` | -0.000319 to 0.000037 | 0.9425 | 0.9884 | pass |
| the same law; the pooled-hazard fit with per-horizon targeting, every node's hazard bounded | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | `cleverly` | -0.000369 to 0.000105 | 0.9483 | 1.0016 | pass |
| the same law; the pooled-hazard fit with per-horizon targeting, every node's hazard bounded | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | `cleverly` | -0.000285 to 0.000270 | 0.9486 | 1.0049 | pass |
| the same law; the pooled-hazard fit with per-horizon targeting, every node's hazard bounded | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | `cleverly` | -0.000406 to 0.000345 | 0.9475 | 0.9898 | pass |
| point-treatment survival law with five visits, binary arm and two binary baseline covariates; the iterated-mean fit, every node bounded | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` | -0.000194 to 0.000457 | 0.9506 | 1.0137 | pass |
| point-treatment survival law with five visits, binary arm and two binary baseline covariates; the iterated-mean fit, every node bounded | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | `cleverly` | -0.000427 to 0.000449 | 0.9510 | 1.0049 | pass |
| point-treatment survival law with five visits, binary arm and two binary baseline covariates; the iterated-mean fit, every node bounded | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | `cleverly` | -0.000234 to 0.000119 | 0.9435 | 0.9958 | pass |
| point-treatment survival law with five visits, binary arm and two binary baseline covariates; the iterated-mean fit, every node bounded | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | `cleverly` | -0.000208 to 0.000263 | 0.9501 | 1.0094 | pass |
| point-treatment survival law with five visits, binary arm and two binary baseline covariates; the iterated-mean fit, every node bounded | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | `cleverly` | -0.000204 to 0.000351 | 0.9493 | 1.0069 | pass |
| point-treatment survival law with five visits, binary arm and two binary baseline covariates; the iterated-mean fit, every node bounded | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | `cleverly` | -0.000331 to 0.000408 | 0.9474 | 1.0064 | pass |
<!-- /generated -->

No replication raised an error, so no cell lost a replicate.

## Agreement with the canonical implementation

Python property evidence, with comparator runs pending. The R `survtmle` runner
`tests/canonical/survival_node_bounds/run_study.R` is written, and it has not run. No paired verdict
is published, and `equivalence.csv` has no rows. The change that runs it pins the `survtmle` image
digest and adds the paired verdicts.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `binding_witness` | `hazard_model__binding` | diagnostic | hazard fit with valid node bounds, five visits: the share of draws on which the unbounded fit leaves the valid bounds on a node follower, and the share on which the bounded and unbounded estimates of the risk by visit 5 under arm 1 or of the risk difference differ by more than 1e-4 | both shares at least one half for the iterated-mean fit; the hazard fit reports its shares without a verdict | exceeds-bound share 0.4649, moved share 0.1011 | reported |
| `binding_witness` | `mean_model__binding` | positive | iterated-mean fit with valid node bounds, five visits: the share of draws on which the unbounded fit leaves the valid bounds on a node follower, and the share on which the bounded and unbounded estimates of the risk by visit 5 under arm 1 or of the risk difference differ by more than 1e-4 | both shares at least one half for the iterated-mean fit; the hazard fit reports its shares without a verdict | exceeds-bound share 0.5983, moved share 0.6547 | pass |
| `bound_necessity` | `f5_arm1__excluding_bound` | control | risk by visit 5 under arm 1 of the iterated-mean fit at n = 16,000, with a saturated outcome regression and an intercept-only treatment mechanism: the upper bound at every node lowered to half the way from the arm-1 follower mean of the true iterated mean to its largest value, so it excludes the truth in the high-risk covariate cells | the displacement from the valid-bounds fit must reach one quarter of the valid-bounds standard deviation, and the bias interval must fall beyond the margin | bias 0.0027 to 0.0030, margin 0.0012 | pass |
| `bound_necessity` | `f5_arm1__valid_bounds` | positive | risk by visit 5 under arm 1 of the iterated-mean fit at n = 16,000, with a saturated outcome regression and an intercept-only treatment mechanism: the valid bounds, a saturated outcome regression and an intercept-only treatment mechanism | bias interval must fall inside the margin | bias 0.000037 to 0.000301, margin 0.0011 | pass |
| `interval_calibration` | `hazard_model_f5_arm1__correctly_specified` | positive | hazard fit with valid node bounds, risk by visit 5 under arm 1: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9370 to 0.9510, SE ratio 0.9700 to 1.0127, empirical efficiency ratio 0.9871 to 1.0303, reported efficiency ratio 0.9977 to 1.0008 | pass |
| `interval_calibration` | `hazard_model_f5_arm1__noise_control` | control | hazard fit with valid node bounds, risk by visit 5 under arm 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8202 to 0.8430, SE ratio 0.6899 to 0.7194, empirical efficiency ratio 1.3889 to 1.4484, reported efficiency ratio 0.9977 to 1.0009 | pass |
| `interval_calibration` | `hazard_model_f5_arm1__shrunken_se_control` | control | hazard fit with valid node bounds, risk by visit 5 under arm 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8085 to 0.8319, SE ratio 0.6790 to 0.7081, empirical efficiency ratio 0.9877 to 1.0305, reported efficiency ratio 0.6984 to 0.7006 | pass |
| `interval_calibration` | `hazard_model_rd5__correctly_specified` | positive | hazard fit with valid node bounds, risk difference by visit 5: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9365 to 0.9506, SE ratio 0.9691 to 1.0129, empirical efficiency ratio 0.9868 to 1.0317, reported efficiency ratio 0.9985 to 1.0009 | pass |
| `interval_calibration` | `hazard_model_rd5__noise_control` | control | hazard fit with valid node bounds, risk difference by visit 5: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8172 to 0.8402, SE ratio 0.6895 to 0.7184, empirical efficiency ratio 1.3915 to 1.4499, reported efficiency ratio 0.9986 to 1.0008 | pass |
| `interval_calibration` | `hazard_model_rd5__shrunken_se_control` | control | hazard fit with valid node bounds, risk difference by visit 5: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8172 to 0.8402, SE ratio 0.6792 to 0.7086, empirical efficiency ratio 0.9875 to 1.0309, reported efficiency ratio 0.6990 to 0.7006 | pass |
| `interval_calibration` | `mean_model_f5_arm1__correctly_specified` | positive | iterated-mean fit with valid node bounds, risk by visit 5 under arm 1: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9359 to 0.9501, SE ratio 0.9700 to 1.0126, empirical efficiency ratio 0.9872 to 1.0308, reported efficiency ratio 0.9981 to 1.0012 | pass |
| `interval_calibration` | `mean_model_f5_arm1__noise_control` | control | iterated-mean fit with valid node bounds, risk by visit 5 under arm 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8196 to 0.8425, SE ratio 0.6912 to 0.7217, empirical efficiency ratio 1.3851 to 1.4464, reported efficiency ratio 0.9981 to 1.0012 | pass |
| `interval_calibration` | `mean_model_f5_arm1__shrunken_se_control` | control | iterated-mean fit with valid node bounds, risk by visit 5 under arm 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8078 to 0.8312, SE ratio 0.6791 to 0.7084, empirical efficiency ratio 0.9878 to 1.0304, reported efficiency ratio 0.6987 to 0.7009 | pass |
| `interval_calibration` | `mean_model_rd5__correctly_specified` | positive | iterated-mean fit with valid node bounds, risk difference by visit 5: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9367 to 0.9508, SE ratio 0.9693 to 1.0120, empirical efficiency ratio 0.9879 to 1.0317, reported efficiency ratio 0.9988 to 1.0012 | pass |
| `interval_calibration` | `mean_model_rd5__noise_control` | control | iterated-mean fit with valid node bounds, risk difference by visit 5: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8256 to 0.8481, SE ratio 0.6951 to 0.7255, empirical efficiency ratio 1.3786 to 1.4387, reported efficiency ratio 0.9988 to 1.0012 | pass |
| `interval_calibration` | `mean_model_rd5__shrunken_se_control` | control | iterated-mean fit with valid node bounds, risk difference by visit 5: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8157 to 0.8388, SE ratio 0.6783 to 0.7087, empirical efficiency ratio 0.9878 to 1.0320, reported efficiency ratio 0.6992 to 0.7008 | pass |
| `reported_calibration` | `hazard_model_f3_arm0__reported` | diagnostic | hazard fit with valid node bounds, risk by visit 3 under arm 0: the bounded fit on the draws of the calibration cells, for a parameter the bounds do not gate | reported with no verdict | coverage 0.9380 to 0.9520, SE ratio 1.0141 | reported |
| `reported_calibration` | `hazard_model_f3_arm1__reported` | diagnostic | hazard fit with valid node bounds, risk by visit 3 under arm 1: the bounded fit on the draws of the calibration cells, for a parameter the bounds do not gate | reported with no verdict | coverage 0.9374 to 0.9514, SE ratio 1.0014 | reported |
| `reported_calibration` | `hazard_model_f5_arm0__reported` | diagnostic | hazard fit with valid node bounds, risk by visit 5 under arm 0: the bounded fit on the draws of the calibration cells, for a parameter the bounds do not gate | reported with no verdict | coverage 0.9415 to 0.9551, SE ratio 1.0039 | reported |
| `reported_calibration` | `mean_model_f3_arm0__reported` | diagnostic | iterated-mean fit with valid node bounds, risk by visit 3 under arm 0: the bounded fit on the draws of the calibration cells, for a parameter the bounds do not gate | reported with no verdict | coverage 0.9383 to 0.9522, SE ratio 1.0138 | reported |
| `reported_calibration` | `mean_model_f3_arm1__reported` | diagnostic | iterated-mean fit with valid node bounds, risk by visit 3 under arm 1: the bounded fit on the draws of the calibration cells, for a parameter the bounds do not gate | reported with no verdict | coverage 0.9374 to 0.9514, SE ratio 1.0009 | reported |
| `reported_calibration` | `mean_model_f5_arm0__reported` | diagnostic | iterated-mean fit with valid node bounds, risk by visit 5 under arm 0: the bounded fit on the draws of the calibration cells, for a parameter the bounds do not gate | reported with no verdict | coverage 0.9415 to 0.9551, SE ratio 1.0037 | reported |
| `unbounded_fit` | `hazard_model_f5_arm1__unbounded` | diagnostic | hazard fit with valid node bounds, risk by visit 5 under arm 1: the identical fit with no node bounds, on the draws of the calibration cells | reported with no verdict | bias -0.000256 to 0.000495, coverage 0.9370 to 0.9510, RMSE 0.0124, bounded RMSE 0.0124 | reported |
| `unbounded_fit` | `hazard_model_rd5__unbounded` | diagnostic | hazard fit with valid node bounds, risk difference by visit 5: the identical fit with no node bounds, on the draws of the calibration cells | reported with no verdict | bias -0.000331 to 0.000558, coverage 0.9365 to 0.9506, RMSE 0.0146, bounded RMSE 0.0146 | reported |
| `unbounded_fit` | `mean_model_f5_arm1__unbounded` | diagnostic | iterated-mean fit with valid node bounds, risk by visit 5 under arm 1: the identical fit with no node bounds, on the draws of the calibration cells | reported with no verdict | bias -0.000257 to 0.000494, coverage 0.9364 to 0.9505, RMSE 0.0124, bounded RMSE 0.0124 | reported |
| `unbounded_fit` | `mean_model_rd5__unbounded` | diagnostic | iterated-mean fit with valid node bounds, risk difference by visit 5: the identical fit with no node bounds, on the draws of the calibration cells | reported with no verdict | bias -0.000333 to 0.000556, coverage 0.9365 to 0.9506, RMSE 0.0146, bounded RMSE 0.0146 | reported |
<!-- /generated -->

One draw of the calibration stream is fitted four times: the bounded and the unbounded fit of
each model. So the calibration, witness and unbounded cells read the same draws.

| family | what it tests | size |
| --- | --- | --- |
| `interval_calibration` | coverage, SE ratio and efficiency ratio of the risk of arm 1 at visit 5 and of the difference at visit 5, for each bounded model. A shrunken-SE control and a noise control read each cell and must fail the same bands | $n$ = 2,000, 7,200 replications |
| `reported_calibration` | the same intervals for the other three risks of each bounded model. The role is `diagnostic`, and no rule gates them | $n$ = 2,000, 7,200 replications |
| `binding_witness` | the share of draws on which the unbounded fit leaves the bounds on a node follower, and the share on which the bounded and unbounded estimates differ by more than 1e-4. Each share must reach one half for the `mean_model`. The `hazard_model` cell reports its shares | $n$ = 2,000, 7,200 replications |
| `unbounded_fit` | the unbounded fit of each model on the same draws, with the RMSE of both fits. The role is `diagnostic` | $n$ = 2,000, 7,200 replications |
| `bound_necessity` | the bounded `mean_model` with a saturated outcome regression and an intercept-only treatment mechanism. The valid bounds keep the bias inside the margin. The control lowers every upper bound to half the way from the arm-1 follower mean of the true iterated mean to its largest value, and its bias must leave the margin | $n$ = 16,000, 7,200 replications |

The bounds bind on the `mean_model`. The unbounded fit left them on 59.8% of the draws, and the
two estimates differ on 65.5%. The `hazard_model` bounds clip the initial hazards on 46.5% of the
draws and move the estimate on 10.1%. The hazard witness is reported for that reason. The RMSE
of the bounded and unbounded fits agree to 1e-5 on every cell. Benkeser, Carone and Gilbert (2018,
Section 5) report a gain from bounds at smaller samples and stronger positivity limits than this
law has.

Every gated cell also reads the failure rule of the declaration. A cell with more than 1% of its
fits unconverged fails. The largest unconverged share of a gated cell is 0.125%, on the
`excluding_bound` control. Three of 7,200 bounded `mean_model` fits did not converge.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
exact standard errors of the efficient influence curve at the primary n. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 7200 | primary replications per model |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 12 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 12 | of those, passing |
| `paired_tests_total` | 0 | paired comparisons with R `survtmle` |
| `paired_tests_passed` | 0 | of those, passing |
| `property_cells_total` | 15 | repeated-sampling property cells, reported cells excluded |
| `property_cells_passed` | 15 | cells whose own and family verdicts pass |
| `property_cells_reported` | 11 | cells reported and not gated |
| `max_standardized_bias` | 0.0240 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9425 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9624 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0353 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control multiplies the standard error by |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |
| `margin:necessity_displacement` | 0.2500 | smallest paired displacement of the necessity control, in empirical standard deviations |
| `margin:witness_share` | 0.5000 | floor of each binding-witness share |
| `margin:unconverged_share` | 0.0100 | largest share of unconverged fits a gated cell may carry |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 72 | the same bound, in primary replications |
| `bound:ate_regimen[arm1 vs arm0 @ t=3]_standard_error` | 0.0109 | exact influence-curve standard error of this estimand at the primary n = 2,000 |
| `bound:ate_regimen[arm1 vs arm0 @ t=5]_standard_error` | 0.0145 | exact influence-curve standard error of this estimand at the primary n = 2,000 |
| `bound:risk_regimen[arm0 @ t=3]_standard_error` | 0.0058 | exact influence-curve standard error of this estimand at the primary n = 2,000 |
| `bound:risk_regimen[arm0 @ t=5]_standard_error` | 0.0078 | exact influence-curve standard error of this estimand at the primary n = 2,000 |
| `bound:risk_regimen[arm1 @ t=3]_standard_error` | 0.0092 | exact influence-curve standard error of this estimand at the primary n = 2,000 |
| `bound:risk_regimen[arm1 @ t=5]_standard_error` | 0.0123 | exact influence-curve standard error of this estimand at the primary n = 2,000 |

## Limits

| limit | what it means for use |
| --- | --- |
| the comparator runs are pending | the claim rests on exact truths, the calibration cells and the controls. The pending R `survtmle` runs add a paired check on the same draws |
| one law with five visits, binary covariates and in-sample fits | the evidence covers correctly specified GLM learners. A flexible learner, cross-fitting, weights and clusters are outside it |
| node 1 retention is declared known | the support rows `G45.mean_model` and `G45.hazard_model` cover a fit that declares it. A fit that estimates node 1 retention is outside them |
| the bounds hold the truth at 1.1 times the largest true value | the study does not measure bounds closer to the truth, or bounds that a user sets from the data |
| the hazard bounds rarely move the estimate | the `hazard_model` witness reports a moved share of 10.1%, so the hazard calibration cells measure a fit that is mostly the unbounded fit |
| every node is bounded, and the censoring model is pooled | a fit that bounds some nodes, or that fits a censoring model per node with bounds, has no row |

The causal interpretation requires consistency, conditional exchangeability given `W1` and `W2`,
positivity, and censoring that is independent of the event given the arm and `W1` and `W2`.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/survival_node_bounds/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/survival_node_bounds/manifest.json)
records the pending comparator, the declared R runner, the seeds and the artifact hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/survival_node_bounds/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/survival_node_bounds/properties.csv)
carry every published row.
