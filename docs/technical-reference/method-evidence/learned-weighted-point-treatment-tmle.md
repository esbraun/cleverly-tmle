# Learned weighted point-treatment TMLE

This study validates ordinary point-treatment TMLE when fixed observation weights reach both
nuisance learners. It draws 2,000 observations from a selected population. The weighted fit
targets a uniform population and learns both nuisance regressions from each sample.

The selected density of `W1` is `(1 + 0.75 * W1) / 2` on `[-1, 1]`. `W2` is uniform on the
same interval. The fixed weight `1 / (1 + 0.75 * W1)` recovers the uniform target law.
Treatment is randomized with probability 0.5. The continuous outcome has unit-variance Gaussian
noise and conditional mean
`0.5 + 0.5 * W1 + 0.25 * W2 + A * (1 + 2 * W1)`.

The target arm means are 0.5 and 1.5, so the target ATE is 1.0. The selected ATE is 1.5.
That difference supplies a nonvacuous witness that learner weights change the fitted outcome
regression.

## What was compared

| setting | `cleverly` | R `tmle` |
| --- | --- | --- |
| estimator | ordinary TMLE, cross-fitting disabled | ordinary `tmle` |
| samples | exact-size draws from the selected continuous law | the identical rows |
| outcome nuisance | weighted main-effects linear regression | weighted Gaussian main-effects GLM |
| treatment nuisance | weighted unpenalized logistic regression | weighted binomial main-effects GLM |
| probability weights | exact target-to-selected density ratio | the same values through `obsWeights=` |
| fluctuation | linear | linear |
| estimands | both arm means and ATE | the same three parameters |
| intervals | pointwise 95% Wald | pointwise 95% Wald |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-outcome law selected by a covariate-dependent density | `ate` | average treatment effect | `cleverly` weighted point-treatment TMLE with learned nuisances | -0.0036 to 0.0075 | 0.9413 | 0.9894 | pass |
| continuous-outcome law selected by a covariate-dependent density | `ate` | average treatment effect | R `tmle` with learned weighted nuisances | -0.0036 to 0.0075 | 0.9413 | 0.9894 | pass |
| continuous-outcome law selected by a covariate-dependent density | `ey0` | counterfactual mean under no treatment | `cleverly` weighted point-treatment TMLE with learned nuisances | -0.0029 to 0.0037 | 0.9712 | 1.1152 | pass |
| continuous-outcome law selected by a covariate-dependent density | `ey0` | counterfactual mean under no treatment | R `tmle` with learned weighted nuisances | -0.0029 to 0.0037 | 0.9712 | 1.1153 | pass |
| continuous-outcome law selected by a covariate-dependent density | `ey1` | counterfactual mean under treatment | `cleverly` weighted point-treatment TMLE with learned nuisances | -0.0026 to 0.0072 | 0.9563 | 1.0606 | pass |
| continuous-outcome law selected by a covariate-dependent density | `ey1` | counterfactual mean under treatment | R `tmle` with learned weighted nuisances | -0.0026 to 0.0072 | 0.9563 | 1.0607 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-outcome law selected by a covariate-dependent density | `ate` | average treatment effect | 2.830e-07 | 0.000031 | 1.0000 | 0 | 0.000008 vs 0.0500 | equivalent |
| continuous-outcome law selected by a covariate-dependent density | `ey0` | counterfactual mean under no treatment | 0.000002 | 0.000325 | 1.0002 | 0 | 0.000146 vs 0.0500 | equivalent |
| continuous-outcome law selected by a covariate-dependent density | `ey1` | counterfactual mean under treatment | 0.000002 | 0.000257 | 1.0002 | 0 | 0.000111 vs 0.0500 | equivalent |
<!-- /generated -->

The implementations agree on all three parameters. This comparison is independent because R
`tmle` fits its own weighted Gaussian and binomial nuisance regressions on the shared samples.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `ate__noise_control` | control | average treatment effect: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8238 to 0.8623, SE ratio 0.6964 to 0.7493 | pass |
| `interval_calibration` | `ate__shrunken_se_control` | control | average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8203 to 0.8592, SE ratio 0.6804 to 0.7328 | pass |
| `interval_calibration` | `ate__treatment_correct` | positive | average treatment effect: the randomized treatment mechanism is correct while the outcome regression omits effect modification | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9420 to 0.9645, SE ratio 0.9715 to 1.0471 | pass |
| `learner_weight_necessity` | `ate__unweighted_plugin_control` | control | average treatment effect: the untargeted plug-in from nuisance fits that omit weights | target bias outside its margin, selected-population bias inside its margin, and paired displacement above its threshold | bias 0.4962 to 0.5038, margin 0.0128, selected-target bias -0.0038 to 0.0038, margin 0.0128 | pass |
| `learner_weight_necessity` | `ate__unweighted_targeted` | positive | average treatment effect: nuisance fits omit weights, while targeting and averaging retain them | target-population bias interval inside the equivalence margin | bias -0.0047 to 0.0047, margin 0.0157 | pass |
| `learner_weight_necessity` | `ate__weighted_plugin` | positive | average treatment effect: the untargeted plug-in from both weighted nuisance fits | target-population bias interval inside the equivalence margin | bias -0.0038 to 0.0056, margin 0.0157 | pass |
| `learner_weight_necessity` | `ate__weighted_targeted` | positive | average treatment effect: the weighted nuisance fits followed by weighted targeting and averaging | target-population bias interval inside the equivalence margin | bias -0.0045 to 0.0048, margin 0.0157 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0032, coverage 0.9356 to 0.9737, SE ratio 0.9941 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.0031, coverage 0.9282 to 0.9688, SE ratio 0.9982 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000978, coverage 0.9179 to 0.9616, SE ratio 0.9805 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5278 to -0.4580 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5009 to -0.4969 | pass |
| `type_i_error` | `target_null` | positive | a selected law whose weighted target contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0400, 0.0243 to 0.0614 | pass |
<!-- /generated -->

The learner-weight control changes only nuisance fitting. Both arms retain weights in targeting,
plug-in averaging, covariance, and inference. The weighted untargeted plug-in recovers the target
ATE, while the unweighted plug-in recovers the selected ATE. Their paired separation exceeds the
declared threshold.

Both targeted arms recover the target ATE. This is expected because the correctly specified
randomized treatment mechanism repairs the unweighted outcome learner through double robustness.
The two targeted passes therefore do not replace the untargeted learner-weight witness.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `paired_tests_total` | 3 | paired comparisons reported |
| `property_cells_passed` | 14 | property cells passing |
| `property_cells_total` | 14 | property cells reported |
| `max_standardized_bias` | 0.0432 | largest primary standardized bias |
| `min_coverage` | 0.9413 | lowest primary coverage |
| `max_margin_utilization` | 0.000325 | largest paired similarity-margin share |
| `properties[learner_weight_necessity/ate__unweighted_plugin_control]:necessity_displacement` | 7.9655 | paired learner-weight displacement |
| `properties[learner_weight_necessity/ate__unweighted_plugin_control]:alternative_truth` | 1.5000 | exact selected-population ATE |
| `properties[learner_weight_necessity/ate__unweighted_plugin_control]:alternative_bias_ci_lower` | -0.0038 | lower bias endpoint against the selected target |
| `properties[learner_weight_necessity/ate__unweighted_plugin_control]:alternative_bias_ci_upper` | 0.0038 | upper bias endpoint against the selected target |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
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
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:learner_weight_displacement` | 0.5000 | minimum learner-weight displacement |

## Limits

- The law has binary treatment, a continuous outcome, two continuous baseline covariates, and
  known fixed weights.
- The nuisance learners are ordinary main-effects regressions, and the treatment assignment is
  randomized. Flexible learners and observational treatment remain outside this row.
- The weights are exact and moderately variable. Estimated, trimmed, calibrated, normalized, and
  replicate weights remain outside this row.
- The study covers ordinary pointwise inference for two arm means and their difference. It excludes
  ratios, ATT, ATC, missing outcomes, clusters, strata, and simultaneous bands.
- Cross-fitted weighted nuisances and weighted longitudinal targeting remain outside the registered
  studies.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_learned_weighted/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_learned_weighted/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_learned_weighted/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_learned_weighted/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_learned_weighted/properties.csv)
carry the protocol, provenance, and every published row.
