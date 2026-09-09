# Randomized missing-outcome DR-TMLE

This study validates randomized missing-outcome DR-TMLE on a binary trial with a response
mechanism that depends on treatment and baseline covariates. It supplies the known treatment
probability and compares the both-correct limit with R `drtmle` 1.1.2 at pinned commit
[`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c).

The comparator boundary matters. R `drtmle` represents treatment and response through one joint
mechanism. `cleverly` implements separate treatment and observation mechanisms and the five
reduced regressions from the randomized missing-outcome construction. The paired comparison can
witness the shared both-correct limit; it cannot establish parity for that internal correction
cycle. Independent property cells test the two one-correct drift directions and mutate the cycle
directly.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome randomized law with MAR outcomes | `ate` | average treatment effect | `cleverly` randomized missing-outcome DR-TMLE | -0.000928 to 0.0050 | 0.9537 | 1.0224 | pass |
| binary-outcome randomized law with MAR outcomes | `ate` | average treatment effect | R `drtmle` with a joint treatment-response mechanism | -0.000917 to 0.0050 | 0.9550 | 1.0238 | pass |
| binary-outcome randomized law with MAR outcomes | `ey0` | counterfactual mean under no treatment | `cleverly` randomized missing-outcome DR-TMLE | -0.0032 to 0.0013 | 0.9575 | 1.0294 | pass |
| binary-outcome randomized law with MAR outcomes | `ey0` | counterfactual mean under no treatment | R `drtmle` with a joint treatment-response mechanism | -0.0032 to 0.0013 | 0.9613 | 1.0307 | pass |
| binary-outcome randomized law with MAR outcomes | `ey1` | counterfactual mean under treatment | `cleverly` randomized missing-outcome DR-TMLE | -0.000839 to 0.0030 | 0.9513 | 1.0065 | pass |
| binary-outcome randomized law with MAR outcomes | `ey1` | counterfactual mean under treatment | R `drtmle` with a joint treatment-response mechanism | -0.000814 to 0.0031 | 0.9500 | 1.0075 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome randomized law with MAR outcomes | `ate` | average treatment effect | -0.000009 | 0.0018 | 1.0022 | -0.0012 | 0.0038 vs 0.0500 | equivalent |
| binary-outcome randomized law with MAR outcomes | `ey0` | counterfactual mean under no treatment | -0.000015 | 0.0040 | 1.0016 | -0.0037 | 0.0035 vs 0.0500 | equivalent |
| binary-outcome randomized law with MAR outcomes | `ey1` | counterfactual mean under treatment | -0.000023 | 0.0073 | 1.0021 | 0.0013 | 0.0030 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `corrected_mar_inference` | `both_correct` | positive | the outcome regression and observation mechanism are correctly specified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0051 to 0.0012, coverage 0.9121 to 0.9575, SE ratio 0.9701 | pass |
| `corrected_mar_inference` | `both_wrong` | control | the outcome regression and observation mechanism are both misspecified | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -0.1327 to -0.1268, coverage 0.0070 to 0.0316, SE ratio 0.9983 | pass |
| `corrected_mar_inference` | `observation_drift` | positive | the outcome regression is correct and the observation mechanism is misspecified | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias 0.000650 to 0.0063, coverage 0.9194 to 0.9627, SE ratio 1.0103 | pass |
| `corrected_mar_inference` | `outcome_drift` | positive | the outcome regression is misspecified and the observation mechanism is correct | bias interval inside the margin, coverage clears the floor, SE ratio inside the band, SE ratio must remain between 0.1 and 10.0 | bias -0.0017 to 0.0043, coverage 0.9341 to 0.9727, SE ratio 1.0282 | pass |
| `correction_necessity` | `five_reduction_cycle__closed_score` | positive | five-reduction correction cycle: the correction scores after the complete five-reduction cycle | the upper confidence endpoint is below the declared fraction of the initial-score lower endpoint | score -3.342e-08 to 2.323e-07 | pass |
| `correction_necessity` | `five_reduction_cycle__initial_score_control` | control | five-reduction correction cycle: the same correction scores before the cycle is run | the lower confidence endpoint clears the declared unresolved-score floor | score 0.0033 to 0.0038 | pass |
| `interval_calibration` | `ate__correctly_specified` | positive | average treatment effect: all three required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9319 to 0.9563, SE ratio 0.9581 to 1.0289, empirical efficiency ratio 0.9717 to 1.0435, reported efficiency ratio 0.9985 to 1.0012 | pass |
| `interval_calibration` | `ate__noise_control` | control | average treatment effect: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8146 to 0.8540, SE ratio 0.6781 to 0.7297, empirical efficiency ratio 1.3703 to 1.4743, reported efficiency ratio 0.9985 to 1.0013 | pass |
| `interval_calibration` | `ate__shrunken_se_control` | control | average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8059 to 0.8461, SE ratio 0.6694 to 0.7213, empirical efficiency ratio 0.9701 to 1.0456, reported efficiency ratio 0.6989 to 0.7009 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000264, coverage 0.9392 to 0.9704, SE ratio 1.0032 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000770, coverage 0.9248 to 0.9599, SE ratio 0.9935 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000680, coverage 0.9267 to 0.9613, SE ratio 0.9861 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5237 to -0.4723 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5015 to -0.4987 | pass |
<!-- /generated -->

The property grid reports the both-correct limit, outcome drift, observation drift, and a
both-wrong control. It also tests root-n contraction, exact-law efficiency, interval calibration,
and the five-reduction cycle's empirical score reduction against the same scores before correction.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `paired_tests_total` | 3 | paired comparisons reported |
| `property_cells_passed` | 14 | property cells passing |
| `property_cells_total` | 14 | property cells reported |
| `max_standardized_bias` | 0.0631 | largest primary standardized bias |
| `min_coverage` | 0.9500 | lowest primary coverage |
| `max_margin_utilization` | 0.0073 | largest paired similarity-margin share |
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
| `margin:correction_score_ratio` | 0.0100 | maximum closed-to-initial score-endpoint ratio |
| `margin:uncorrected_score_floor` | 0.0010 | unresolved initial-score floor |

## Limits

- The study covers binary treatment, binary outcome, one three-level baseline covariate, known
  1:1 randomization, and an MAR response mechanism.
- Initial outcome and observation regressions are finite-support oracles; the reduced regressions
  use linear and logistic models without cross-fitting.
- The comparison uses pointwise Wald intervals. Its joint-mechanism R comparator establishes only
  the shared both-correct limit, not internal five-reduction parity.
- The property grid declares no type-I error, power, or targeting cell. The ordinary
  missing-outcome study carries those three instruments for the `delta=` path.
- The study excludes observational treatment assignment, weights, clusters, simultaneous bands,
  missing treatment, multinomial treatment, MNAR outcomes, and other DR-TMLE compositions.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_mar/properties.csv)
carry the protocol, provenance, and every published row.
