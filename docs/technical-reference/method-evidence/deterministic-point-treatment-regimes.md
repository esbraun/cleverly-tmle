# Deterministic point-treatment regimes

This study validates ordinary static and covariate-dependent deterministic regimes. It compares
`cleverly` with pinned R `lmtp` 1.5.4 on identical binary-outcome samples and exact treatment
probabilities. The dynamic rule assigns both arms and differs from either static endpoint.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with a covariate-dependent deterministic rule | `ate_regime[rule vs never]` | difference in means under the regimes "follow the covariate-dependent rule" against "assign no treatment" | `cleverly` | -0.0027 to 0.000738 | 0.9525 | 0.9990 | pass |
| binary-outcome law with a covariate-dependent deterministic rule | `ate_regime[rule vs never]` | difference in means under the regimes "follow the covariate-dependent rule" against "assign no treatment" | R `lmtp` | -0.0026 to 0.000768 | 0.9513 | 1.0014 | pass |
| binary-outcome law with a covariate-dependent deterministic rule | `ey_regime[never]` | mean under the regime assign no treatment | `cleverly` | -0.000873 to 0.0018 | 0.9500 | 1.0070 | pass |
| binary-outcome law with a covariate-dependent deterministic rule | `ey_regime[never]` | mean under the regime assign no treatment | R `lmtp` | -0.000902 to 0.0017 | 0.9500 | 1.0071 | pass |
| binary-outcome law with a covariate-dependent deterministic rule | `ey_regime[rule]` | mean under the regime follow the covariate-dependent rule | `cleverly` | -0.0020 to 0.0010 | 0.9563 | 1.0322 | pass |
| binary-outcome law with a covariate-dependent deterministic rule | `ey_regime[rule]` | mean under the regime follow the covariate-dependent rule | R `lmtp` | -0.0021 to 0.0010 | 0.9550 | 1.0335 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with a covariate-dependent deterministic rule | `ate_regime[rule vs never]` | difference in means under the regimes "follow the covariate-dependent rule" against "assign no treatment" | -0.000025 | 0.0091 | 1.0009 | 0.0012 | 0.0054 vs 0.0500 | equivalent |
| binary-outcome law with a covariate-dependent deterministic rule | `ey_regime[never]` | mean under the regime assign no treatment | 0.000029 | 0.0132 | 1.0011 | 0 | 0.000900 vs 0.0500 | equivalent |
| binary-outcome law with a covariate-dependent deterministic rule | `ey_regime[rule]` | mean under the regime follow the covariate-dependent rule | 0.000003 | 0.0013 | 0.9957 | 0.0013 | 0.0044 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0013 to 0.0014, margin 0.0046 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.2079 to -0.2047, margin 0.0054 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0020 to 0.000562, margin 0.0043 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias -0.0020 to 0.0010, margin 0.0050 | pass |
| `interval_calibration` | `rule__correctly_specified` | positive | covariate-dependent rule: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9451 to 0.9623, SE ratio 0.9829 to 1.0392, empirical efficiency ratio 0.9615 to 1.0169, reported efficiency ratio 0.9986 to 1.0001 | pass |
| `interval_calibration` | `rule__noise_control` | control | covariate-dependent rule: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8302 to 0.8599, SE ratio 0.6993 to 0.7411, empirical efficiency ratio 1.3484 to 1.4288, reported efficiency ratio 0.9985 to 1.0001 | pass |
| `interval_calibration` | `rule__shrunken_se_control` | control | covariate-dependent rule: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8183 to 0.8489, SE ratio 0.6875 to 0.7277, empirical efficiency ratio 0.9612 to 1.0177, reported efficiency ratio 0.6990 to 0.7001 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000937, coverage 0.9447 to 0.9796, SE ratio 1.0152 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000019, coverage 0.9267 to 0.9677, SE ratio 0.9942 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000003, coverage 0.9223 to 0.9647, SE ratio 0.9797 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5263 to -0.4613 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5009 to -0.4983 | pass |
| `rule_necessity` | `rule__declared` | positive | covariate-dependent rule: the declared covariate-dependent rule assigns both treatment arms | bias interval inside the equivalence margin | bias -0.0015 to 0.0012, margin 0.0046 | pass |
| `rule_necessity` | `rule__static_control` | control | covariate-dependent rule: the same fit replaces the rule with an always-treated static plan | bias interval must fall entirely outside the margin | bias 0.0885 to 0.0916, margin 0.0052 | pass |
| `static_reduction` | `never__arm` | positive | never-treated plan: the same target is requested as the ordinary untreated-arm mean | the paired estimate must equal the regime estimate exactly | maximum paired difference 0 | pass |
| `static_reduction` | `never__regime` | positive | never-treated plan: the never-treated target is requested through the regime axis | the paired estimate must equal the treatment-arm estimate exactly | maximum paired difference 0 | pass |
| `targeting_necessity` | `rule__targeted` | positive | covariate-dependent rule: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000676 to 0.0022, margin 0.0048 | pass |
| `targeting_necessity` | `rule__untargeted` | control | covariate-dependent rule: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.3803 to -0.3798, margin 0.000718 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0375, 0.0224 to 0.0584 | pass |
<!-- /generated -->

The property study separately tests both sides of double robustness, exact-EIF calibration, a
confounded sharp null, targeting, the declared rule, and exact static reduction.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `paired_tests_total` | 3 | paired comparisons reported |
| `property_cells_passed` | 20 | property cells passing |
| `property_cells_total` | 20 | property cells reported |
| `max_standardized_bias` | 0.0517 | largest primary standardized bias |
| `min_coverage` | 0.9500 | lowest primary coverage |
| `max_margin_utilization` | 0.0132 | largest paired similarity-margin share |
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
| `margin:necessity_displacement` | 0.2500 | minimum rule-control displacement |

## Limits

- The laws use binary treatment, binary outcome, and one three-level baseline covariate.
- The primary working outcome model is logistic and the treatment mechanism is supplied exactly.
- The study covers ordinary, non-cross-fitted targeting and pointwise Wald intervals.
- It excludes missing outcomes, observation weights, clusters, simultaneous bands, and flexible learners.
- It does not validate stochastic, continuous-dose, incremental, or longitudinal interventions.
