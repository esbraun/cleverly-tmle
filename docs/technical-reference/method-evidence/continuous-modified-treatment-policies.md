# Continuous modified treatment policies

This study validates the natural course, an uncapped dose shift, and an actively capped dose
shift. It compares `cleverly` with pinned R `lmtp` 1.5.4 on identical continuous-outcome samples
from the same conditional-normal dose law. R receives the analytic ratio; `cleverly` evaluates
the law through its pooled-hazard density representation.

This is a **reporting** study: independently valid rows and failed or inconclusive paired claims
are published together. The policy prevents a post-hoc relaxation of the shared comparison
margins while keeping the unresolved comparison visible.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-dose law with uncapped and capped shifts | `ate_shift[+0.25 vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.25" against "leave the observed treatment mechanism unchanged" | `cleverly` | -0.000377 to 0.000573 | 0.9525 | 1.0317 | pass |
| continuous-dose law with uncapped and capped shifts | `ate_shift[+0.25 vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.25" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000575 to 0.000348 | 0.9437 | 0.9892 | pass |
| continuous-dose law with uncapped and capped shifts | `ate_shift[+0.5 capped vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.5 subject to the declared cap" against "leave the observed treatment mechanism unchanged" | `cleverly` | -0.000619 to 0.0013 | 0.9625 | 1.0465 | pass |
| continuous-dose law with uncapped and capped shifts | `ate_shift[+0.5 capped vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.5 subject to the declared cap" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000590 to 0.0014 | 0.9637 | 1.0547 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_shift[+0.25]` | mean under the modified treatment policy shift dose by 0.25 | `cleverly` | -0.0024 to 0.0077 | 0.9500 | 1.0161 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_shift[+0.25]` | mean under the modified treatment policy shift dose by 0.25 | R `lmtp` | -0.0026 to 0.0075 | 0.9487 | 1.0143 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_shift[+0.5 capped]` | mean under the modified treatment policy shift dose by 0.5 subject to the declared cap | `cleverly` | -0.0018 to 0.0076 | 0.9500 | 1.0141 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_shift[+0.5 capped]` | mean under the modified treatment policy shift dose by 0.5 subject to the declared cap | R `lmtp` | -0.0018 to 0.0076 | 0.9500 | 1.0135 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_shift[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | `cleverly` | -0.0023 to 0.0074 | 0.9525 | 1.0191 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_shift[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | R `lmtp` | -0.0023 to 0.0074 | 0.9525 | 1.0191 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-dose law with uncapped and capped shifts | `ate_shift[+0.25 vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.25" against "leave the observed treatment mechanism unchanged" | 0.000211 | 0.2745 | 1.0583 | 0.0088 | 0.0450 vs 0.0500 | **inconclusive** |
| continuous-dose law with uncapped and capped shifts | `ate_shift[+0.5 capped vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.5 subject to the declared cap" against "leave the observed treatment mechanism unchanged" | -0.000020 | 0.0123 | 1.0167 | -0.0012 | 0.0147 vs 0.0500 | equivalent |
| continuous-dose law with uncapped and capped shifts | `ey_shift[+0.25]` | mean under the modified treatment policy shift dose by 0.25 | 0.000211 | 0.0255 | 1.0020 | 0.0012 | 0.0024 vs 0.0500 | equivalent |
| continuous-dose law with uncapped and capped shifts | `ey_shift[+0.5 capped]` | mean under the modified treatment policy shift dose by 0.5 subject to the declared cap | -0.000020 | 0.0025 | 1.0007 | 0 | 0.0013 vs 0.0500 | equivalent |
| continuous-dose law with uncapped and capped shifts | `ey_shift[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | -1.162e-11 | 1.462e-09 | 1.0000 | 0 | 6.944e-11 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `cap_necessity` | `shift__declared_cap` | positive | capped modified treatment policy: the 0.5 shift leaves doses unchanged when the declared cap would be crossed | bias interval inside the equivalence margin | bias -0.000961 to 0.000688, margin 0.0028 | pass |
| `cap_necessity` | `shift__uncapped_control` | control | capped modified treatment policy: the same named policy removes the cap and shifts every dose | bias interval must fall entirely outside the margin | bias 0.2719 to 0.2731, margin 0.0021 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.000628 to 0.0010, margin 0.0028 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0098 to 0.0115, margin 0.0028 | pass |
| `double_robustness` | `density_correct` | positive | only the continuous-dose density ratio is correctly specified | bias interval inside the equivalence margin | bias -0.000172 to 0.0015, margin 0.0028 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.000619 to 0.000889, margin 0.0025 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9448 to 0.9621, SE ratio 0.9854 to 1.0431 | pass |
| `interval_calibration` | `shrunken_se_control` | control | the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8142 to 0.8450, SE ratio 0.6888 to 0.7305 | pass |
| `natural_course_identity` | `natural__mean` | positive | natural course: the observed sample mean is retained as the identity control | the paired intervention estimate must equal it exactly | maximum paired difference 1.184e-09 | pass |
| `natural_course_identity` | `natural__shift` | positive | natural course: the zero-shift policy is evaluated through the continuous-policy axis | the paired estimate must equal the observed sample mean exactly | maximum paired difference 1.184e-09 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `ratio_necessity` | `shift__declared` | positive | capped modified treatment policy: the shifted-to-natural density ratio is used in the declared direction | bias interval inside the equivalence margin | bias -0.000053 to 0.0016, margin 0.0028 | pass |
| `ratio_necessity` | `shift__reversed_control` | control | capped modified treatment policy: the density probabilities are deliberately inverted before the pooled-hazard ratio is formed | bias interval must fall entirely outside the margin | bias -0.0123 to -0.0108, margin 0.0026 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000024, coverage 0.9282 to 0.9688, SE ratio 0.9932 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.000102, coverage 0.9078 to 0.9544, SE ratio 0.9705 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000021, coverage 0.9341 to 0.9727, SE ratio 1.0083 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5444 to -0.4775 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5004 to -0.4955 | pass |
| `targeting_necessity` | `shift__targeted` | positive | capped modified treatment policy: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias 0.000210 to 0.0019, margin 0.0029 | pass |
| `targeting_necessity` | `shift__untargeted` | control | capped modified treatment policy: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0441 to 0.0454, margin 0.0022 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0488, 0.0312 to 0.0718 | pass |
<!-- /generated -->

The property study tests both robustness routes, repeated-sampling calibration, null and power controls,
targeting, density-ratio direction, the active cap, and the exact natural-course identity.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 10 | truth tests passing |
| `independent_tests_total` | 10 | truth tests reported |
| `paired_tests_passed` | 4 | paired comparisons passing |
| `paired_tests_total` | 5 | paired comparisons reported |
| `property_cells_passed` | 21 | property cells passing |
| `property_cells_total` | 21 | property cells reported |
| `max_standardized_bias` | 0.0564 | largest primary standardized bias |
| `min_coverage` | 0.9437 | lowest primary coverage |
| `max_margin_utilization` | 0.2745 | largest paired similarity-margin share |
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
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:necessity_displacement` | 0.2500 | minimum policy-control displacement |

## Limits

- The primary law has one conditional-normal dose, a Gaussian-noise continuous outcome, and three baseline covariates.
- The primary and property laws use quadratic curvature 0.15, so the density remains load-bearing. Cleverly's declared quadratic regression and R's quadratic GLM both represent the primary outcome regression exactly.
- The primary pooled-hazard density uses 320 bins against R's analytic normal-density ratio. Property controls use 20 bins except for the density-correct, outcome-misspecified arm, where 320 bins keep density approximation error from masking the robustness claim.
- The study covers ordinary, non-cross-fitted targeting and pointwise Wald intervals.
- The uncapped shift requires extrapolation at rare sample-edge doses; the support warning is an explicit limitation.
- One of five paired comparisons is inconclusive at the shared 99% non-inferiority margin. The reporting policy publishes it without changing the observed margin.
- That row is `ate_shift[+0.25 vs natural course]`, and it fails calibration non-inferiority alone.
  Its cause is the density asymmetry this page declares above, not a defect in either targeting
  step. Three measurements locate it. The two natural-course rows are identical to six figures,
  because the ratio is one there and the discretization does nothing. The two `ey_shift[+0.25]`
  SE ratios agree to within 0.2 percent. The discretized ratio differs from the analytic ratio by
  a median of 0.3 percent and a mean of 1.8 percent on the primary law, and a contrast subtracts
  two influence curves that nearly cancel, which turns that small relative deviation into a four
  percent difference in the contrast standard error.
- Bin resolution is not the cause. A shift of 0.25 moves 99.6 percent of rows across a bin edge on
  the primary law, against a median bin width of 0.013, so the policy is not invisible to the
  estimator.
- It excludes missing outcomes, weights, clusters, simultaneous bands, and flexible learning.
- It does not validate categorical, longitudinal, or multiple-time modified treatment policies.
