# Incremental propensity interventions

This study validates three incremental treatment-odds multipliers, including the natural course
at one. It compares `cleverly` with R `npcausal` at commit `56a5ac1` on identical samples.
Kennedy (2019) publishes `npcausal`, and its influence values carry the derivative through the
treatment mechanism. The comparison therefore gates inference as well as the point curve.

R `imtp` 0.1.0 held this role in an earlier revision and no longer does. Its reported influence
curve omits that derivative, so it can witness the point curve and cannot gate a standard error.
[Choosing a comparator](../../development/method-benchmarking.md#choose-the-comparator-before-you-write-the-runner)
records the full survey.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x0.5 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | `cleverly` | -0.000039 to 0.000378 | 0.9431 | 0.9932 | pass |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x0.5 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | R `npcausal` | -0.000187 to 0.000234 | 0.9444 | 0.9869 | pass |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x2 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | `cleverly` | -0.000412 to 0.000034 | 0.9463 | 0.9986 | pass |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x2 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | R `npcausal` | -0.000244 to 0.000209 | 0.9475 | 0.9889 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[natural course]` | mean under the incremental intervention leave the observed treatment mechanism unchanged | `cleverly` | -0.000628 to 0.000839 | 0.9381 | 0.9824 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[natural course]` | mean under the incremental intervention leave the observed treatment mechanism unchanged | R `npcausal` | -0.000628 to 0.000839 | 0.9381 | 0.9824 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x0.5]` | mean under the incremental intervention multiply the treatment odds by 0.5 | `cleverly` | -0.000477 to 0.0010 | 0.9444 | 0.9885 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x0.5]` | mean under the incremental intervention multiply the treatment odds by 0.5 | R `npcausal` | -0.000622 to 0.000881 | 0.9425 | 0.9895 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x2]` | mean under the incremental intervention multiply the treatment odds by two | `cleverly` | -0.000850 to 0.000683 | 0.9381 | 0.9762 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x2]` | mean under the incremental intervention multiply the treatment odds by two | R `npcausal` | -0.000681 to 0.000858 | 0.9363 | 0.9741 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x0.5 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | 0.000146 | 0.2993 | 0.9984 | -0.0012 | 0.0180 vs 0.0500 | equivalent |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x2 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | -0.000172 | 0.3287 | 0.9944 | -0.0012 | 0.0254 vs 0.0500 | equivalent |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[natural course]` | mean under the incremental intervention leave the observed treatment mechanism unchanged | 6.730e-13 | 3.942e-10 | 1.0000 | 0 | 4.226e-11 vs 0.0500 | equivalent |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x0.5]` | mean under the incremental intervention multiply the treatment odds by 0.5 | 0.000146 | 0.0834 | 1.0026 | 0.0019 | 0.0019 vs 0.0500 | equivalent |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x2]` | mean under the incremental intervention multiply the treatment odds by two | -0.000172 | 0.0962 | 0.9993 | 0.0019 | 0.0052 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `contrast__correctly_specified` | positive | odds-x2 contrast: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9421 to 0.9598, SE ratio 0.9778 to 1.0365, empirical efficiency ratio 0.9641 to 1.0221, reported efficiency ratio 0.9987 to 0.9997 | pass |
| `interval_calibration` | `contrast__noise_control` | control | odds-x2 contrast: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8181 to 0.8486, SE ratio 0.6895 to 0.7295, empirical efficiency ratio 1.3696 to 1.4490, reported efficiency ratio 0.9987 to 0.9997 | pass |
| `interval_calibration` | `contrast__shrunken_se_control` | control | odds-x2 contrast: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8214 to 0.8518, SE ratio 0.6840 to 0.7256, empirical efficiency ratio 0.9642 to 1.0226, reported efficiency ratio 0.6991 to 0.6998 | pass |
| `mechanism_requirement` | `both_correct` | positive | both the outcome regression and treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.000288 to 0.000233, margin 0.000875 | pass |
| `mechanism_requirement` | `mechanism_wrong` | control | the wrong treatment mechanism is held fixed when the incremental target is evaluated | bias interval must fall entirely outside the margin | bias -0.0116 to -0.0112, margin 0.000627 | pass |
| `mechanism_requirement` | `outcome_wrong` | positive | the treatment mechanism is correct and the outcome regression is misspecified | bias interval inside the equivalence margin | bias -0.000205 to 0.000306, margin 0.000857 | pass |
| `natural_course_identity` | `natural__ipsi` | positive | natural course: the odds multiplier one is evaluated through the incremental axis | the paired estimate must equal the observed sample mean exactly | maximum paired difference 8.408e-11 | pass |
| `natural_course_identity` | `natural__mean` | positive | natural course: the observed sample mean is retained as the identity control | the paired intervention estimate must equal it exactly | maximum paired difference 8.408e-11 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `contrast__n_2000` | positive | odds-x2 contrast: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000136, coverage 0.9416 to 0.9776, SE ratio 1.0402 | pass |
| `root_n_and_efficiency` | `contrast__n_500` | control | odds-x2 contrast: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.000560, coverage 0.9282 to 0.9688, SE ratio 1.0041 | pass |
| `root_n_and_efficiency` | `contrast__n_8000` | positive | odds-x2 contrast: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000050, coverage 0.9297 to 0.9698, SE ratio 0.9987 | pass |
| `root_n_rate` | `contrast__empirical_sd` | positive | odds-x2 contrast: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5302 to -0.4656 | pass |
| `root_n_rate` | `contrast__reported_se` | positive | odds-x2 contrast: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5003 to -0.4987 | pass |
| `targeting_necessity` | `mechanism__targeted` | positive | treatment-mechanism targeting: the treatment mechanism is fluctuated after outcome targeting | bias interval inside the equivalence margin | bias -0.000313 to 0.000198, margin 0.000857 | pass |
| `targeting_necessity` | `mechanism__untargeted` | control | treatment-mechanism targeting: the identical targeted outcome regression is evaluated at the unfluctuated mechanism | bias interval must fall entirely outside the margin | bias -0.0135 to -0.0130, margin 0.000870 | pass |
| `targeting_necessity` | `outcome__targeted` | positive | outcome-regression targeting: the outcome regression is fluctuated before the incremental target is evaluated | bias interval inside the equivalence margin | bias -0.000421 to 0.000100, margin 0.000875 | pass |
| `targeting_necessity` | `outcome__untargeted` | control | outcome-regression targeting: the identical targeted mechanism is evaluated with the initial outcome regression | bias interval must fall entirely outside the margin | bias -0.0904 to -0.0903, margin 0.000123 | pass |
| `treatment_score_necessity` | `odds_x2__full_eif` | positive | odds-x2 incremental mean: the incremental mean uses the complete efficient influence curve | the reported-to-empirical SE-ratio interval must stay inside the declared band | SE ratio 0.9641 to 1.0547 | pass |
| `treatment_score_necessity` | `odds_x2__regime_curve_control` | control | odds-x2 incremental mean: the same point estimates use an influence curve with the treatment-score term removed | the SE-ratio interval must fall below the calibration band | SE ratio 0.6077 to 0.6630 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0512, 0.0333 to 0.0748 | pass |
<!-- /generated -->

The property study tests the one-sided mechanism requirement, exact-EIF calibration, null and
power controls, both targeting equations, the treatment-score term, and the natural-course identity.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | paired replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 10 | truth tests passing |
| `independent_tests_total` | 10 | truth tests reported |
| `paired_tests_passed` | 5 | paired comparisons passing |
| `paired_tests_total` | 5 | paired comparisons reported |
| `property_cells_passed` | 21 | property cells passing |
| `property_cells_total` | 21 | property cells reported |
| `max_standardized_bias` | 0.0546 | largest primary standardized bias |
| `min_coverage` | 0.9363 | lowest primary coverage |
| `max_margin_utilization` | 0.3287 | largest paired similarity-margin share |
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
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |

## Limits

- The laws use binary treatment, binary outcome, and one three-level baseline covariate.
- Both implementations saturate that covariate. `cleverly` fits one-hot indicators and their
  interaction with treatment. `npcausal` fits `SL.glm.interaction` over the same design. The
  treatment mechanism is supplied exactly to `cleverly` and is estimated by `npcausal`.
- An earlier revision fitted the covariate as one numeric column. That model is misspecified, and
  it inflated the standard error of each contrast against the natural course by five to six
  percent above the exact-law efficient influence curve. Bias and coverage did not show it,
  because a correct mechanism keeps a targeted estimator consistent whatever the outcome model
  says. Consistency under a wrong outcome model is claimed by the `mechanism_requirement`
  property cells, which use exact nuisances and carry their own control.
- `npcausal` cross-fits its nuisances over two folds and `cleverly` does not. The single-fold path
  that `?ipsi` documents selects an empty training set and is not usable.
- Only pointwise Wald inference is validated for either implementation.
- The study covers ordinary, non-cross-fitted `cleverly` targeting and three fixed odds multipliers.
- It excludes missing outcomes, weights, clusters, simultaneous bands, flexible learners, multinomial treatment, and longitudinal interventions.
