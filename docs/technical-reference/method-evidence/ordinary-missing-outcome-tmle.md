# Ordinary missing-outcome TMLE

This study validates ordinary, non-cross-fitted point-treatment TMLE when the outcome is missing
at random. It compares `cleverly` with R [`tmle`](https://cran.r-project.org/package=tmle) 2.1.1
on identical samples, using the same realized nuisance predictions and bounds. The observational
law has a nonconstant treatment mechanism and a response mechanism that depends on treatment and
baseline covariates.

The comparison isolates the observed-data targeting construction. It reports the two
treatment-specific means and their average treatment-effect contrast with pointwise 95% Wald
intervals.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome observational law with MAR outcomes | `ate` | average treatment effect | `cleverly` missing-outcome TMLE | -0.000900 to 0.0050 | 0.9587 | 1.0324 | pass |
| binary-outcome observational law with MAR outcomes | `ate` | average treatment effect | R `tmle` | -0.000900 to 0.0050 | 0.9587 | 1.0324 | pass |
| binary-outcome observational law with MAR outcomes | `ey0` | counterfactual mean under no treatment | `cleverly` missing-outcome TMLE | -0.0022 to 0.0021 | 0.9550 | 1.0081 | pass |
| binary-outcome observational law with MAR outcomes | `ey0` | counterfactual mean under no treatment | R `tmle` | -0.0022 to 0.0021 | 0.9550 | 1.0081 | pass |
| binary-outcome observational law with MAR outcomes | `ey1` | counterfactual mean under treatment | `cleverly` missing-outcome TMLE | -0.000084 to 0.0041 | 0.9425 | 1.0097 | pass |
| binary-outcome observational law with MAR outcomes | `ey1` | counterfactual mean under treatment | R `tmle` | -0.000084 to 0.0041 | 0.9425 | 1.0097 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome observational law with MAR outcomes | `ate` | average treatment effect | 1.244e-09 | 2.566e-07 | 1.0000 | 0 | 5.149e-09 vs 0.0500 | equivalent |
| binary-outcome observational law with MAR outcomes | `ey0` | counterfactual mean under no treatment | -8.769e-10 | 2.483e-07 | 1.0000 | 0 | 4.212e-09 vs 0.0500 | equivalent |
| binary-outcome observational law with MAR outcomes | `ey1` | counterfactual mean under treatment | 3.675e-10 | 1.065e-07 | 1.0000 | 0 | 4.198e-09 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `ate__correctly_specified` | positive | average treatment effect: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9334 to 0.9525, SE ratio 0.9574 to 1.0146, empirical efficiency ratio 0.9826 to 1.0413, reported efficiency ratio 0.9960 to 0.9982 | pass |
| `interval_calibration` | `ate__noise_control` | control | average treatment effect: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8147 to 0.8455, SE ratio 0.6824 to 0.7241, empirical efficiency ratio 1.3770 to 1.4608, reported efficiency ratio 0.9960 to 0.9982 | pass |
| `interval_calibration` | `ate__shrunken_se_control` | control | average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8113 to 0.8424, SE ratio 0.6700 to 0.7098, empirical efficiency ratio 0.9834 to 1.0416, reported efficiency ratio 0.6972 to 0.6988 | pass |
| `mar_robustness` | `both_correct` | positive | the outcome regression, treatment mechanism and observation mechanism are correct | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0026 to 0.0024, margin 0.0083, SE ratio 1.0011 | pass |
| `mar_robustness` | `mechanisms_correct` | positive | the treatment and observation mechanisms are correct and the outcome regression is not | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0039 to 0.0023, margin 0.0103, SE ratio 0.9785 | pass |
| `mar_robustness` | `observation_wrong` | control | only the treatment mechanism is correct | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -0.4824 to -0.4783, margin 0.0069, SE ratio 1.6716 | pass |
| `mar_robustness` | `outcome_correct` | positive | only the outcome regression is correct | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.000857 to 0.0023, margin 0.0053, SE ratio 2.6184 | pass |
| `mar_robustness` | `treatment_wrong` | control | only the observation mechanism is correct | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -0.2110 to -0.2059, margin 0.0084, SE ratio 1.5397 | pass |
| `missingness_necessity` | `ate__complete_case_control` | control | average treatment effect: the identical estimator silently discards unobserved outcomes and ignores selection | bias interval must fall entirely outside the margin | bias -0.1229 to -0.1172, margin 0.0096 | pass |
| `missingness_necessity` | `ate__declared` | positive | average treatment effect: the observation indicator is declared, so correct mechanisms carry a wrong outcome model | bias interval inside the equivalence margin | bias -0.0055 to 0.000397, margin 0.0099 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000585, coverage 0.9252 to 0.9667, SE ratio 0.9909 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0012, coverage 0.9165 to 0.9606, SE ratio 0.9635 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000099, coverage 0.9165 to 0.9606, SE ratio 1.0001 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5441 to -0.4770 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4993 to -0.4954 | pass |
| `targeting_necessity` | `ate__targeted` | positive | average treatment effect: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0043 to 0.0018, margin 0.0101 | pass |
| `targeting_necessity` | `ate__untargeted` | control | average treatment effect: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.4208 to -0.4198, margin 0.0017 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0450, 0.0283 to 0.0674 | pass |
<!-- /generated -->

The property grid separates the three-nuisance robustness contract. A correct outcome regression
must rescue wrong treatment and observation mechanisms; when the outcome regression is wrong,
both mechanisms must be correct. Separate controls leave only one mechanism correct. Further cells
test root-n contraction, exact-law efficiency, interval calibration, type-I error and power,
targeting, and the consequence of silently analyzing complete cases.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `paired_tests_total` | 3 | paired comparisons reported |
| `property_cells_passed` | 19 | property cells passing |
| `property_cells_total` | 19 | property cells reported |
| `max_standardized_bias` | 0.0877 | largest primary standardized bias |
| `min_coverage` | 0.9425 | lowest primary coverage |
| `max_margin_utilization` | 2.566e-07 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
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
| `margin:missingness_displacement` | 0.2500 | minimum complete-case displacement |

## Limits

- The study covers binary treatment, binary outcome, one three-level baseline covariate, and an
  observational MAR response mechanism.
- The nuisance functions are supplied as finite-support oracles. This isolates targeting and
  inference; it does not validate flexible learner wrappers or cross-fitting.
- The comparison uses pointwise Wald intervals and does not cover simultaneous bands.
- The study reports the two arm means and their difference. The `att`, `atc`, `rr` and `or`
  estimands keep only their exact-law Gateaux and remainder evidence under `delta=`.
- The study excludes weights, clusters, missing treatment, multinomial treatment, MNAR outcomes,
  longitudinal data, and DR-TMLE correction cycles.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar/properties.csv)
carry the protocol, provenance, and every published row.
