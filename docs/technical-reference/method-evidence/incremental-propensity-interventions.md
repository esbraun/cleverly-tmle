# Incremental propensity interventions

This study validates three incremental treatment-odds multipliers, including the natural course
at one. Pinned R `imtp` at commit `d4b5204` supplies a point-curve witness on identical samples.
Its influence curve omits the derivative through the treatment mechanism, so it is not treated
as a canonical inference comparator; `cleverly` inference is gated independently.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x0.5 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | `cleverly` | -0.000039 to 0.000378 | 0.9569 | 1.0489 | pass |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x0.5 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | R `imtp` point-curve witness | 0.000059 to 0.000721 | 0.9631 | 1.0966 | pass |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x2 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | `cleverly` | -0.000412 to 0.000034 | 0.9594 | 1.0581 | pass |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x2 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | R `imtp` point-curve witness | -0.000378 to 0.000341 | 0.9731 | 1.1180 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[natural course]` | mean under the incremental intervention leave the observed treatment mechanism unchanged | `cleverly` | -0.000628 to 0.000839 | 0.9381 | 0.9824 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[natural course]` | mean under the incremental intervention leave the observed treatment mechanism unchanged | R `imtp` point-curve witness | -0.000612 to 0.000910 | 0.9350 | 0.9472 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x0.5]` | mean under the incremental intervention multiply the treatment odds by 0.5 | `cleverly` | -0.000477 to 0.0010 | 0.9463 | 0.9964 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x0.5]` | mean under the incremental intervention multiply the treatment odds by 0.5 | R `imtp` point-curve witness | -0.000228 to 0.0013 | 0.7894 | 0.6448 | **fail** |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x2]` | mean under the incremental intervention multiply the treatment odds by two | `cleverly` | -0.000850 to 0.000683 | 0.9400 | 0.9808 | pass |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x2]` | mean under the incremental intervention multiply the treatment odds by two | R `imtp` point-curve witness | -0.000662 to 0.000924 | 0.9875 | 1.2503 | **fail** |
<!-- /generated -->

## Agreement with the point-curve witness

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x0.5 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | -0.000221 | 0.3432 | 0.6593 | -0.0062 | n/a | equivalent |
| binary-outcome law with three incremental odds multipliers | `ate_ipsi[odds x2 vs natural course]` | difference in means under the incremental interventions "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | -0.000171 | 0.2455 | 0.6501 | -0.0138 | n/a | equivalent |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[natural course]` | mean under the incremental intervention leave the observed treatment mechanism unchanged | -0.000043 | 0.0249 | 0.9791 | 0.0031 | n/a | equivalent |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x0.5]` | mean under the incremental intervention multiply the treatment odds by 0.5 | -0.000264 | 0.1495 | 0.9925 | 0.1569 | n/a | equivalent |
| binary-outcome law with three incremental odds multipliers | `ey_ipsi[odds x2]` | mean under the incremental intervention multiply the treatment odds by two | -0.000214 | 0.1180 | 0.9809 | -0.0475 | n/a | equivalent |
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
| `independent_tests_passed` | 8 | truth tests passing |
| `independent_tests_total` | 10 | truth tests reported |
| `paired_tests_passed` | 5 | paired point-curve comparisons passing |
| `paired_tests_total` | 5 | paired point-curve comparisons reported |
| `property_cells_passed` | 21 | property cells passing |
| `property_cells_total` | 21 | property cells reported |
| `max_standardized_bias` | 0.0760 | largest primary standardized bias |
| `min_coverage` | 0.7894 | lowest primary coverage |
| `max_margin_utilization` | 0.3432 | largest paired similarity-margin share |
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
- The primary outcome model is logistic and the treatment mechanism is supplied exactly to `cleverly`.
- R `imtp` uses saturated baseline indicators and one-fold main-effects Super Learner fits.
- Only pointwise `cleverly` Wald inference is validated; `imtp` inference is explicitly noncanonical.
- The study covers ordinary, non-cross-fitted targeting and three fixed odds multipliers.
- It excludes missing outcomes, weights, clusters, simultaneous bands, flexible learners, multinomial treatment, and longitudinal interventions.
