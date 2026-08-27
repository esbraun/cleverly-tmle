# Ordinary multi-arm TMLE

This study validates ordinary, non-cross-fitted TMLE for a labelled three-arm treatment.
The canonical comparison uses R [`tmle3`](https://github.com/tlverse/tmle3) at pinned commit
`ed72f8a`. Both implementations receive identical draws, fit the same intercept-only outcome
regression, and use correctly specified multinomial-logistic treatment mechanisms. This nuisance
pair makes the fluctuation load-bearing while retaining consistency through the treatment model.

The binary-outcome law deliberately orders source labels differently from the estimator's
sorted internal codes. It has practical overlap under the declared propensity bounds and
reports every arm mean plus two ATEs, risk ratios, and odds ratios against the `high` arm.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` ordinary multi-arm TMLE | -0.0019 to 0.0035 | 0.9700 | 1.0938 | pass |
| three-arm binary-outcome law | `ate[low vs high]` | difference in counterfactual means, low versus high | R `tmle3` multi-arm TMLE | -0.0020 to 0.0035 | 0.9700 | 1.0938 | pass |
| three-arm binary-outcome law | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` ordinary multi-arm TMLE | -0.0033 to 0.0023 | 0.9663 | 1.0978 | pass |
| three-arm binary-outcome law | `ate[medium vs high]` | difference in counterfactual means, medium versus high | R `tmle3` multi-arm TMLE | -0.0033 to 0.0023 | 0.9663 | 1.0979 | pass |
| three-arm binary-outcome law | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` ordinary multi-arm TMLE | -0.0023 to 0.0017 | 0.9537 | 1.0608 | pass |
| three-arm binary-outcome law | `ey[high]` | counterfactual mean under treatment arm 'high' | R `tmle3` multi-arm TMLE | -0.0023 to 0.0018 | 0.9537 | 1.0609 | pass |
| three-arm binary-outcome law | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` ordinary multi-arm TMLE | -0.0015 to 0.0025 | 0.9550 | 1.0207 | pass |
| three-arm binary-outcome law | `ey[low]` | counterfactual mean under treatment arm 'low' | R `tmle3` multi-arm TMLE | -0.0015 to 0.0025 | 0.9550 | 1.0206 | pass |
| three-arm binary-outcome law | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` ordinary multi-arm TMLE | -0.0028 to 0.0013 | 0.9637 | 1.0496 | pass |
| three-arm binary-outcome law | `ey[medium]` | counterfactual mean under treatment arm 'medium' | R `tmle3` multi-arm TMLE | -0.0028 to 0.0012 | 0.9637 | 1.0497 | pass |
| three-arm binary-outcome law | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` ordinary multi-arm TMLE | -0.0069 to 0.0157 | 0.9712 | 1.0947 | pass |
| three-arm binary-outcome law | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | R `tmle3` multi-arm TMLE | -0.0070 to 0.0156 | 0.9712 | 1.0947 | pass |
| three-arm binary-outcome law | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` ordinary multi-arm TMLE | -0.0117 to 0.0123 | 0.9675 | 1.0961 | pass |
| three-arm binary-outcome law | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | R `tmle3` multi-arm TMLE | -0.0119 to 0.0121 | 0.9675 | 1.0962 | pass |
| three-arm binary-outcome law | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` ordinary multi-arm TMLE | -0.0038 to 0.0087 | 0.9650 | 1.0975 | pass |
| three-arm binary-outcome law | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | R `tmle3` multi-arm TMLE | -0.0038 to 0.0086 | 0.9650 | 1.0976 | pass |
| three-arm binary-outcome law | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` ordinary multi-arm TMLE | -0.0053 to 0.0066 | 0.9663 | 1.0928 | pass |
| three-arm binary-outcome law | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | R `tmle3` multi-arm TMLE | -0.0054 to 0.0065 | 0.9663 | 1.0929 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law | `ate[low vs high]` | difference in counterfactual means, low versus high | 0.000018 | 0.0039 | 1.0001 | 0 | 0.000096 vs 0.0500 | equivalent |
| three-arm binary-outcome law | `ate[medium vs high]` | difference in counterfactual means, medium versus high | 0.000034 | 0.0075 | 1.0001 | 0 | 0.000122 vs 0.0500 | equivalent |
| three-arm binary-outcome law | `ey[high]` | counterfactual mean under treatment arm 'high' | -0.000017 | 0.0052 | 1.0001 | 0 | 0.000139 vs 0.0500 | equivalent |
| three-arm binary-outcome law | `ey[low]` | counterfactual mean under treatment arm 'low' | 2.195e-07 | 0.000066 | 1.0000 | 0 | 0.000062 vs 0.0500 | equivalent |
| three-arm binary-outcome law | `ey[medium]` | counterfactual mean under treatment arm 'medium' | 0.000017 | 0.0051 | 1.0001 | 0 | 0.000224 vs 0.0500 | equivalent |
| three-arm binary-outcome law | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | 0.000127 | 0.0039 | 1.0002 | 0 | 0.000101 vs 0.0500 | equivalent |
| three-arm binary-outcome law | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | 0.000441 | 0.0075 | 1.0004 | 0 | 0.000120 vs 0.0500 | equivalent |
| three-arm binary-outcome law | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | 0.000061 | 0.0044 | 1.0002 | 0 | 0.000107 vs 0.0500 | equivalent |
| three-arm binary-outcome law | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | 0.000119 | 0.0072 | 1.0003 | 0 | 0.000121 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0013 to 0.0061, margin 0.0088 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0686 to 0.0764, margin 0.0093 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0026 to 0.0049, margin 0.0089 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias -0.0026 to 0.0029, margin 0.0066 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9343 to 0.9630, SE ratio 0.9509 to 1.0404 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000823, coverage 0.9118 to 0.9720, SE ratio 0.9643 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.0021, coverage 0.9180 to 0.9756, SE ratio 1.0142 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000034, coverage 0.9212 to 0.9774, SE ratio 1.0184 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5520 to -0.4533 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5031 to -0.4996 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0425, 0.0208 to 0.0757 | pass |
<!-- /generated -->

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 1500 | observations per primary replication |
| `independent_tests_total` | 18 | implementation-estimand truth tests |
| `independent_tests_passed` | 18 | truth tests passing |
| `paired_tests_total` | 9 | paired comparison tests |
| `paired_tests_passed` | 9 | paired tests passing |
| `property_cells_total` | 12 | repeated-sampling property cells |
| `property_cells_passed` | 12 | property cells passing |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples for bootstrap intervals |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical SDs |
| `margin:coverage_floor` | 0.9000 | lower coverage gate |
| `margin:over_coverage_ceiling` | 0.9900 | reported conservative-coverage ceiling |
| `margin:se_ratio_sanity_lower` | 0.8000 | lower SE-ratio screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | upper SE-ratio screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | lower calibration band |
| `margin:calibration_se_ratio_upper` | 1.0700 | upper calibration band |
| `margin:calibration_coverage_lower` | 0.9200 | lower calibration-coverage band |
| `margin:calibration_coverage_upper` | 0.9800 | upper calibration-coverage band |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate |
| `margin:paired_difference` | 0.1500 | paired similarity margin in empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio non-inferiority margin |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference non-inferiority margin |
| `margin:calibration_noninferiority` | 0.0500 | SE-calibration non-inferiority margin |
| `margin:minimum_power` | 0.8000 | lower power gate |
| `margin:root_n_slope` | -0.5000 | root-n contraction target |
| `margin:root_n_slope_lower` | -0.6250 | lower contraction band |
| `margin:root_n_slope_upper` | -0.3750 | upper contraction band |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |

## Limitations

The comparison covers one binary-outcome law, one reference arm, pointwise intervals, and
ordinary GLM nuisance fitting. It does not validate simultaneous intervals, conditional effects,
continuous outcomes, cross-fitting, flexible learners, or behavior under serious
practical-positivity failure. It excludes missing outcomes, weights, clusters, fold repeats, and
longitudinal treatment. Each of those is a separate row, not an implied one.

R `tmle3` receives numeric arm codes because the pinned adapter mishandles a factor column in
counterfactual prediction. The law is linear in that code. `cleverly` still receives the original
labels and a saturated arm design, so the label-to-column mapping stays under test.
