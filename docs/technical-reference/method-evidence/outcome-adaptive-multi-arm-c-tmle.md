# Outcome-adaptive multi-arm C-TMLE

This study validates outcome-adaptive C-TMLE for a labelled three-arm treatment. The canonical
comparison uses archived R [`ctmle3`](https://github.com/tlverse/ctmle3) at pinned commit
`a4ea77b`. Both implementations receive identical draws and start from corresponding empirical
treatment mechanisms and correctly specified outcome regressions.

The study uses the family-wide three-arm binary law and reports arm means, ATEs, risk ratios,
and odds ratios against the `high` arm. Its property design separately checks the method's
outcome-regression robustness contract and the precision cost of estimating the generated
outcome-adaptive design.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law, outcome-adaptive selector | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0020 to 0.0036 | 0.9525 | 0.9675 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ate[low vs high]` | difference in counterfactual means, low versus high | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0019 to 0.0036 | 0.9525 | 0.9700 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0034 to 0.0020 | 0.9350 | 0.9527 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0033 to 0.0021 | 0.9350 | 0.9527 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0020 to 0.0018 | 0.9450 | 0.9780 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[high]` | counterfactual mean under treatment arm 'high' | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0021 to 0.0018 | 0.9463 | 0.9786 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0013 to 0.0027 | 0.9413 | 0.9822 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[low]` | counterfactual mean under treatment arm 'low' | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0013 to 0.0027 | 0.9413 | 0.9844 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0027 to 0.0011 | 0.9325 | 0.9386 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0027 to 0.0012 | 0.9363 | 0.9391 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0071 to 0.0158 | 0.9487 | 0.9678 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0070 to 0.0159 | 0.9487 | 0.9703 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0125 to 0.0108 | 0.9350 | 0.9529 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0123 to 0.0111 | 0.9350 | 0.9529 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0041 to 0.0085 | 0.9525 | 0.9687 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0040 to 0.0085 | 0.9525 | 0.9710 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm outcome-adaptive C-TMLE | -0.0057 to 0.0057 | 0.9413 | 0.9645 | pass |
| three-arm binary-outcome law, outcome-adaptive selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | R `ctmle3` multi-arm outcome-adaptive TMLE | -0.0056 to 0.0058 | 0.9413 | 0.9646 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law, outcome-adaptive selector | `ate[low vs high]` | difference in counterfactual means, low versus high | -0.000020 | 0.0043 | 1.0031 | 0 | 0.000452 vs 0.0500 | equivalent |
| three-arm binary-outcome law, outcome-adaptive selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | -0.000056 | 0.0125 | 1.0005 | 0 | 0.000446 vs 0.0500 | equivalent |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[high]` | counterfactual mean under treatment arm 'high' | 0.000025 | 0.0080 | 1.0010 | -0.0013 | 0.000390 vs 0.0500 | equivalent |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[low]` | counterfactual mean under treatment arm 'low' | 0.000006 | 0.0017 | 1.0025 | 0 | 0.000227 vs 0.0500 | equivalent |
| three-arm binary-outcome law, outcome-adaptive selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | -0.000030 | 0.0094 | 1.0010 | -0.0037 | 0.000366 vs 0.0500 | equivalent |
| three-arm binary-outcome law, outcome-adaptive selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | -0.000074 | 0.0023 | 1.0030 | 0 | 0.000454 vs 0.0500 | equivalent |
| three-arm binary-outcome law, outcome-adaptive selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | -0.000709 | 0.0124 | 1.0002 | 0 | 0.000438 vs 0.0500 | equivalent |
| three-arm binary-outcome law, outcome-adaptive selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | -0.000063 | 0.0045 | 1.0027 | 0 | 0.000477 vs 0.0500 | equivalent |
| three-arm binary-outcome law, outcome-adaptive selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | -0.000186 | 0.0117 | 1.0004 | 0 | 0.000456 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `generated_design` | `estimated` | control | the same design is estimated from the data, as a real fit does | the SE-ratio deficit must reach the declared shortfall | SE ratio 0.9635 to 1.0923 | **fail** |
| `generated_design` | `oracle_design` | positive | the outcome-adaptive design is supplied rather than estimated | SE ratio interval inside the calibration band | SE ratio 0.9719 to 1.1046 | **fail** |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9343 to 0.9630, SE ratio 0.9650 to 1.0553 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `robustness_contract` | `outcome_correct` | positive | the outcome regression is correct and the mechanism is not | bias interval inside the equivalence margin | bias -0.0022 to 0.0022, margin 0.0052 | pass |
| `robustness_contract` | `outcome_wrong` | control | the outcome regression is misspecified | bias interval must fall entirely outside the margin | bias 0.0664 to 0.0738, margin 0.0088 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0021, coverage 0.9212 to 0.9774, SE ratio 0.9798 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.0039, coverage 0.8996 to 0.9646, SE ratio 1.0022 | **fail** |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000044, coverage 0.9212 to 0.9774, SE ratio 1.0291 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5667 to -0.4726 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5127 to -0.5074 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0550, 0.0298 to 0.0913 | pass |
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
| `property_cells_passed` | 9 | property cells passing |
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
| `margin:generated_design_deficit` | 0.0100 | required estimated-design SE-ratio deficit |

## Limitations

This row has reporting policy, so its red cells publish. The generated-design pair did not
resolve: the oracle design's SE-ratio interval leaves the calibration band, and the estimated
design showed no measurable deficit against it. The n = 500 coverage bound also misses the floor.
The row therefore prices no cost for estimating the treatment design.

The archived R comparison requires numeric arm codes because of its counterfactual-prediction
adapter. The study covers one binary-outcome law, ordinary GLM nuisance fits, pointwise intervals,
and one outcome-adaptive strategy. It does not transfer the result to selector-based C-TMLE,
cross-fitted primary fitting, flexible learners, simultaneous inference, or severe
practical-positivity violations. It excludes missing outcomes, weights, clusters, fold repeats,
and longitudinal treatment.
