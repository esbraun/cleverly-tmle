# Ordinary continuous modified treatment policies

This study validates the natural course, an uncapped dose shift, and an actively capped dose
shift. It compares `cleverly` with pinned R `lmtp` 1.5.4 on identical continuous-outcome samples
from the same conditional-normal dose law. R receives the analytic ratio; `cleverly` evaluates
the law through its pooled-hazard density representation.

This is a **reporting** study: independently valid rows and failed or inconclusive paired claims
are published together. The policy prevents a post-hoc relaxation of the shared comparison
margins. On the current rows every paired claim concludes `equivalent`.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-dose law with uncapped and capped shifts | `ate_policy[+0.25 vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.25" against "leave the observed treatment mechanism unchanged" | `cleverly` | -0.000431 to 0.000496 | 0.9463 | 0.9866 | pass |
| continuous-dose law with uncapped and capped shifts | `ate_policy[+0.25 vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.25" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000575 to 0.000348 | 0.9437 | 0.9892 | pass |
| continuous-dose law with uncapped and capped shifts | `ate_policy[+0.5 capped vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.5 subject to the declared cap" against "leave the observed treatment mechanism unchanged" | `cleverly` | -0.000607 to 0.0013 | 0.9663 | 1.0512 | pass |
| continuous-dose law with uncapped and capped shifts | `ate_policy[+0.5 capped vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.5 subject to the declared cap" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000590 to 0.0014 | 0.9637 | 1.0547 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_policy[+0.25]` | mean under the modified treatment policy shift dose by 0.25 | `cleverly` | -0.0025 to 0.0076 | 0.9513 | 1.0148 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_policy[+0.25]` | mean under the modified treatment policy shift dose by 0.25 | R `lmtp` | -0.0026 to 0.0075 | 0.9487 | 1.0143 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_policy[+0.5 capped]` | mean under the modified treatment policy shift dose by 0.5 subject to the declared cap | `cleverly` | -0.0018 to 0.0076 | 0.9500 | 1.0139 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_policy[+0.5 capped]` | mean under the modified treatment policy shift dose by 0.5 subject to the declared cap | R `lmtp` | -0.0018 to 0.0076 | 0.9500 | 1.0135 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | `cleverly` | -0.0023 to 0.0074 | 0.9537 | 1.0191 | pass |
| continuous-dose law with uncapped and capped shifts | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | R `lmtp` | -0.0023 to 0.0074 | 0.9525 | 1.0191 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
Comparator-backed: a paired comparison with `lmtp` on the same samples. Each row reports estimate equivalence within the declared margin, the RMSE ratio and the coverage difference, and names its comparison kind.

| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | comparison kind | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-dose law with uncapped and capped shifts | `ate_policy[+0.25 vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.25" against "leave the observed treatment mechanism unchanged" | 0.000146 | 0.1918 | 1.0156 | 0.0025 | different estimator | 0.0099 vs 0.0500 | equivalent |
| continuous-dose law with uncapped and capped shifts | `ate_policy[+0.5 capped vs natural course]` | difference in means under the modified treatment policies "shift dose by 0.5 subject to the declared cap" against "leave the observed treatment mechanism unchanged" | -0.000013 | 0.0078 | 1.0108 | 0.0025 | different estimator | 0.0074 vs 0.0500 | equivalent |
| continuous-dose law with uncapped and capped shifts | `ey_policy[+0.25]` | mean under the modified treatment policy shift dose by 0.25 | 0.000146 | 0.0176 | 1.0007 | 0.0025 | different estimator | 0.000939 vs 0.0500 | equivalent |
| continuous-dose law with uncapped and capped shifts | `ey_policy[+0.5 capped]` | mean under the modified treatment policy shift dose by 0.5 subject to the declared cap | -0.000013 | 0.0016 | 1.0004 | 0 | different estimator | 0.000918 vs 0.0500 | equivalent |
| continuous-dose law with uncapped and capped shifts | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | -1.016e-11 | 1.278e-09 | 1.0000 | 0.0012 | different estimator | 5.619e-11 vs 0.0500 | equivalent |
<!-- /generated -->

Before Fix-9's conditional density, `ate_policy[+0.25 vs natural course]` was inconclusive on
its calibration leg alone: the excess bound was 0.064762 against the margin of 0.05. RM18 design
CD, declared before its run, targeted the same `cleverly` fit again with the analytic normal
density in place of the 320 bins and read the excess as `cleverly density representation`. On
the regenerated rows the excess bound is 0.012462 and the row passes. Rerun on those rows, design
CD reads `unresolved`, because the binned and analytic fits no longer differ.
[`tests/diagnostics/rm18_comparator_density/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/rm18_comparator_density)
holds the rows and the current reading, and the
[roadmap at commit `985849c6`](https://github.com/esbraun/cleverly-tmle/blob/985849c668a18cf800094714ddae3fea0675cb91/docs/roadmap.md#what-design-cd-found)
gives the first.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `cap_necessity` | `shift__declared_cap` | positive | capped modified treatment policy: the 0.5 shift leaves doses unchanged when the declared cap would be crossed | bias interval inside the equivalence margin | bias -0.000967 to 0.000704, margin 0.0028 | pass |
| `cap_necessity` | `shift__uncapped_control` | control | capped modified treatment policy: the same named policy removes the cap and shifts every dose | bias interval must fall entirely outside the margin | bias 0.2716 to 0.2731, margin 0.0026 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000551 to 0.0011, margin 0.0028, SE ratio 1.0003 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0064 to -0.0048, margin 0.0028, SE ratio 1.0628 | pass |
| `double_robustness` | `density_correct` | positive | only the continuous-dose density ratio is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0016 to 0.000077, margin 0.0028, SE ratio 1.0111 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000775 to 0.000738, margin 0.0025, SE ratio 1.1793 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9394 to 0.9575, SE ratio 0.9587 to 1.0153 | pass |
| `interval_calibration` | `shrunken_se_control` | control | the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8041 to 0.8356, SE ratio 0.6707 to 0.7109 | pass |
| `natural_course_identity` | `natural__mean` | positive | natural course: the observed sample mean is retained as the identity control | the paired intervention estimate must equal it exactly | maximum paired difference 2.275e-11 | pass |
| `natural_course_identity` | `natural__shift` | positive | natural course: the zero-shift policy is evaluated through the continuous-policy axis | the paired estimate must equal the observed sample mean exactly | maximum paired difference 2.275e-11 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `ratio_necessity` | `shift__declared` | positive | capped modified treatment policy: the shifted-to-natural density ratio is used in the declared direction | bias interval inside the equivalence margin | bias -0.0015 to 0.000173, margin 0.0028 | pass |
| `ratio_necessity` | `shift__reversed_control` | control | capped modified treatment policy: the density probabilities are deliberately inverted before the pooled-hazard ratio is formed | bias interval must fall entirely outside the margin | bias -0.0192 to -0.0176, margin 0.0027 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000030, coverage 0.9223 to 0.9647, SE ratio 0.9728 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.000052, coverage 0.9150 to 0.9596, SE ratio 0.9544 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000070, coverage 0.9223 to 0.9647, SE ratio 0.9726 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5388 to -0.4721 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5016 to -0.4969 | pass |
| `targeting_necessity` | `shift__targeted` | positive | capped modified treatment policy: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0012 to 0.000506, margin 0.0028 | pass |
| `targeting_necessity` | `shift__untargeted` | control | capped modified treatment policy: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0441 to 0.0454, margin 0.0022 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0587, 0.0394 to 0.0835 | pass |
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
| `paired_tests_passed` | 5 | paired comparisons passing |
| `paired_tests_total` | 5 | paired comparisons reported |
| `property_cells_passed` | 21 | property cells passing |
| `property_cells_total` | 21 | property cells reported |
| `max_standardized_bias` | 0.0564 | largest primary standardized bias |
| `min_coverage` | 0.9437 | lowest primary coverage |
| `max_margin_utilization` | 0.1918 | largest paired similarity-margin share |
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
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:necessity_displacement` | 0.2500 | minimum policy-control displacement |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 8 | the same bound, in primary replications |
| `margin:reference_failure_bound` | 0.0100 | declared share of replications a cell of R `lmtp` may lose to a recordable fitting failure |
| `margin:reference_failures_allowed` | 8 | the same bound, in primary replications |

## Limits

- The primary law has one conditional-normal dose, a Gaussian-noise continuous outcome, and three baseline covariates.
- The primary and property laws use quadratic curvature 0.15, so the density remains load-bearing. Cleverly's declared quadratic regression and R's quadratic GLM both represent the primary outcome regression exactly.
- The primary pooled-hazard density uses 320 bins against R's analytic normal-density ratio. Property controls use 20 bins except for the density-correct, outcome-misspecified arm, where 320 bins keep density approximation error from masking the robustness claim.
- The study covers ordinary, non-cross-fitted targeting and pointwise Wald intervals.
- A point fit without a data-tuned cap keeps the sample-range outcome scale when `q_bounds=None`. A revenue outcome is unbounded and positive, so declare `Targeting(q_bounds=(lower, upper))` with its known support.
- The uncapped shift requires extrapolation at rare sample-edge doses; the support warning is an explicit limitation.
- It excludes missing outcomes, weights, clusters, simultaneous bands, and flexible learning. The [default-band study](default-simultaneous-bands.md) measures the default band of this study's subject fit.
- It does not validate categorical, longitudinal, or multiple-time modified treatment policies.
