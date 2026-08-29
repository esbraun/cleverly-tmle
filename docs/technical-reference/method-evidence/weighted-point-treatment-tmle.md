# Weighted point-treatment TMLE

This study validates ordinary point-treatment TMLE with fixed probability weights. It draws
exactly 2,000 observations from a biased selected population and asks whether inverse-selection
weights recover the population target. The comparison uses R
[`tmle`](https://cran.r-project.org/package=tmle) 2.1.1 on the identical rows, exact nuisance
predictions, and observation weights.

The finite law has three baseline levels. Their population probabilities are 0.50, 0.30, and
0.20, while inclusion probabilities are 0.15, 0.50, and 0.90. The resulting selected law
overrepresents the level with the largest treatment effect. Consequently, the population ATE is
0.33 and the selected-population ATE is 0.5222. The supplied weights are the inverse inclusion
probabilities, so their tilted law is exactly the declared population law.

## What was compared

| setting | `cleverly` | R `tmle` |
| --- | --- | --- |
| estimator | ordinary TMLE, cross-fitting disabled | ordinary `tmle` |
| samples | exact-size draws from the selected finite law | the identical rows |
| nuisances | exact outcome and treatment probabilities | the same predictions through `Q=` and `g1W=` |
| probability weights | inverse inclusion probability | the same values through `obsWeights=` |
| estimands | both arm means, ATE, RR, and OR | the same five parameters |
| intervals | pointwise 95% Wald | pointwise 95% Wald |
| RR and OR scale | log, exponentiated | log, exponentiated |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law sampled with unequal selection probabilities | `ate` | average treatment effect | `cleverly` weighted point-treatment TMLE | -0.0034 to 0.0017 | 0.9450 | 1.0013 | pass |
| binary-outcome law sampled with unequal selection probabilities | `ate` | average treatment effect | R `tmle` with observation weights | -0.0034 to 0.0017 | 0.9450 | 1.0013 | pass |
| binary-outcome law sampled with unequal selection probabilities | `ey0` | counterfactual mean under no treatment | `cleverly` weighted point-treatment TMLE | -0.0014 to 0.0013 | 0.9525 | 1.0278 | pass |
| binary-outcome law sampled with unequal selection probabilities | `ey0` | counterfactual mean under no treatment | R `tmle` with observation weights | -0.0014 to 0.0013 | 0.9525 | 1.0278 | pass |
| binary-outcome law sampled with unequal selection probabilities | `ey1` | counterfactual mean under treatment | `cleverly` weighted point-treatment TMLE | -0.0031 to 0.0013 | 0.9400 | 0.9862 | pass |
| binary-outcome law sampled with unequal selection probabilities | `ey1` | counterfactual mean under treatment | R `tmle` with observation weights | -0.0031 to 0.0013 | 0.9400 | 0.9862 | pass |
| binary-outcome law sampled with unequal selection probabilities | `or` | marginal odds ratio, reported on the log scale | `cleverly` weighted point-treatment TMLE | -0.0127 to 0.0112 | 0.9537 | 1.0089 | pass |
| binary-outcome law sampled with unequal selection probabilities | `or` | marginal odds ratio, reported on the log scale | R `tmle` with observation weights | -0.0127 to 0.0112 | 0.9500 | 0.9906 | pass |
| binary-outcome law sampled with unequal selection probabilities | `rr` | marginal risk ratio, reported on the log scale | `cleverly` weighted point-treatment TMLE | -0.0076 to 0.0075 | 0.9575 | 1.0182 | pass |
| binary-outcome law sampled with unequal selection probabilities | `rr` | marginal risk ratio, reported on the log scale | R `tmle` with observation weights | -0.0076 to 0.0075 | 0.9575 | 1.0182 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law sampled with unequal selection probabilities | `ate` | average treatment effect | -1.264e-12 | 3.000e-10 | 1.0000 | 0 | 3.590e-11 vs 0.0500 | equivalent |
| binary-outcome law sampled with unequal selection probabilities | `ey0` | counterfactual mean under no treatment | 4.439e-14 | 2.025e-11 | 1.0000 | 0 | 1.208e-12 vs 0.0500 | equivalent |
| binary-outcome law sampled with unequal selection probabilities | `ey1` | counterfactual mean under treatment | -1.220e-12 | 3.363e-10 | 1.0000 | 0 | 3.295e-12 vs 0.0500 | equivalent |
| binary-outcome law sampled with unequal selection probabilities | `or` | marginal odds ratio, reported on the log scale | -2.211e-11 | 2.507e-10 | 1.0000 | 0.0037 | 0.0200 vs 0.0500 | equivalent |
| binary-outcome law sampled with unequal selection probabilities | `rr` | marginal risk ratio, reported on the log scale | -6.396e-12 | 1.986e-10 | 1.0000 | 0 | 1.501e-11 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0010 to 0.0031, margin 0.0069, SE ratio 1.0210 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.1029 to 0.1060, margin 0.0052, SE ratio 1.6568 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0013 to 0.0018, margin 0.0052, SE ratio 1.6633 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000518 to 0.0056, margin 0.0085, SE ratio 1.0021 | pass |
| `interval_calibration` | `ate__correctly_specified` | positive | average treatment effect: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9324 to 0.9567, SE ratio 0.9650 to 1.0379, empirical efficiency ratio 0.9619 to 1.0344, reported efficiency ratio 0.9965 to 0.9997 | pass |
| `interval_calibration` | `ate__noise_control` | control | average treatment effect: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8020 to 0.8425, SE ratio 0.6759 to 0.7244, empirical efficiency ratio 1.3776 to 1.4766, reported efficiency ratio 0.9965 to 0.9997 | pass |
| `interval_calibration` | `ate__shrunken_se_control` | control | average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8111 to 0.8509, SE ratio 0.6753 to 0.7268, empirical efficiency ratio 0.9613 to 1.0345, reported efficiency ratio 0.6975 to 0.6998 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000015, coverage 0.9165 to 0.9606, SE ratio 0.9899 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0030, coverage 0.9006 to 0.9491, SE ratio 0.9295 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000645, coverage 0.9386 to 0.9757, SE ratio 1.0132 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5601 to -0.4961 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4992 to -0.4949 | pass |
| `targeting_necessity` | `ate__targeted` | positive | average treatment effect: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0014 to 0.0037, margin 0.0086 | pass |
| `targeting_necessity` | `ate__untargeted` | control | average treatment effect: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.3702 to -0.3696, margin 0.0010 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0587, 0.0394 to 0.0835 | pass |
| `weight_necessity` | `ate__omitted_control` | control | average treatment effect: the identical selected rows analyzed without their inverse-selection weights | population-target bias outside its margin and selected-target bias inside its margin | bias 0.1909 to 0.1937, margin 0.0048, selected-target bias -0.0014 to 0.0015, margin 0.0048 | pass |
| `weight_necessity` | `ate__weighted` | positive | average treatment effect: the selected sample analyzed with its fixed inverse-selection weights | population-target bias interval inside the equivalence margin | bias -0.0019 to 0.0023, margin 0.0070 | pass |
<!-- /generated -->

The weight-necessity pair uses the same selected samples twice. The positive arm supplies the
inverse-selection weights and must recover the population ATE. The control omits only those
weights: it must miss the population ATE, recover the exact selected-population ATE, and separate
from the weighted estimate by at least the declared standardized displacement. This distinguishes
a valid estimator of another population from an arbitrary failure.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 10 | truth tests passing |
| `independent_tests_total` | 10 | truth tests reported |
| `paired_tests_passed` | 5 | paired comparisons passing |
| `paired_tests_total` | 5 | paired comparisons reported |
| `property_cells_passed` | 18 | property cells passing |
| `property_cells_total` | 18 | property cells reported |
| `max_standardized_bias` | 0.0375 | largest primary standardized bias |
| `min_coverage` | 0.9400 | lowest primary coverage |
| `max_margin_utilization` | 3.363e-10 | largest paired similarity-margin share |
| `properties[weight_necessity/ate__weighted]:necessity_displacement` | 6.8141 | paired weight-necessity displacement |
| `properties[weight_necessity/ate__omitted_control]:alternative_truth` | 0.5222 | exact selected-population ATE |
| `properties[weight_necessity/ate__omitted_control]:alternative_bias_ci_lower` | -0.0014 | lower bias endpoint against the selected target |
| `properties[weight_necessity/ate__omitted_control]:alternative_bias_ci_upper` | 0.0015 | upper bias endpoint against the selected target |
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
| `margin:union_model_se_lower` | 0.1000 | robustness-cell SE-ratio lower screen |
| `margin:union_model_se_upper` | 10 | robustness-cell SE-ratio upper screen |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:weight_displacement` | 0.5000 | minimum weight-necessity displacement |

## Limits

- The law has binary treatment and outcome, one three-level baseline covariate, and known fixed
  inclusion probabilities.
- Exact nuisance functions isolate weighted targeting and inference. The study does not validate
  learned sampling weights, flexible learners, or cross-fitting.
- The study covers one set of strictly positive, moderately unequal weights; it does not cover
  near-zero inclusion, trimming, normalization choices, calibration weights, or replicate weights.
- The comparison is ordinary, pointwise inference for five marginal parameters. It excludes ATT,
  ATC, missing outcomes, multiple treatment levels, clusters, strata, and simultaneous bands.
- This point-treatment row does not establish weighted longitudinal targeting or covariance.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_weighted/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_weighted/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_weighted/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_weighted/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_weighted/properties.csv)
carry the protocol, provenance, and every published row.
