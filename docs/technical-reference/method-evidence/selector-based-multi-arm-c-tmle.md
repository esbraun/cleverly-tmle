# Selector-based multi-arm C-TMLE

This reporting study exercises the greedy, ordered, and discrete selector strategies for a
labelled three-arm treatment. No canonical implementation is compared: R `ctmle` 0.1.2 is
binary-treatment only, and separate binary fits would target different parameters rather than
form a legitimate multi-arm comparator.

Each strategy is tested against the known truth for three arm means and two ATEs, risk ratios,
and odds ratios. The property record also pairs the collaborative selector with an empty-path
control on identical draws, so a selector that stops immediately is exposed rather than inferred
from otherwise plausible point estimates.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law, discrete selector | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm selector C-TMLE | -0.0020 to 0.0036 | 0.9375 | 0.9923 | pass |
| three-arm binary-outcome law, discrete selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm selector C-TMLE | -0.0015 to 0.0039 | 0.9400 | 0.9794 | pass |
| three-arm binary-outcome law, discrete selector | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm selector C-TMLE | -0.0027 to 0.0012 | 0.9437 | 0.9939 | pass |
| three-arm binary-outcome law, discrete selector | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm selector C-TMLE | -0.0020 to 0.0020 | 0.9475 | 0.9863 | pass |
| three-arm binary-outcome law, discrete selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm selector C-TMLE | -0.0016 to 0.0024 | 0.9387 | 0.9661 | pass |
| three-arm binary-outcome law, discrete selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0069 to 0.0158 | 0.9400 | 0.9921 | pass |
| three-arm binary-outcome law, discrete selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0044 to 0.0193 | 0.9375 | 0.9776 | pass |
| three-arm binary-outcome law, discrete selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0036 to 0.0089 | 0.9400 | 0.9922 | pass |
| three-arm binary-outcome law, discrete selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0022 to 0.0095 | 0.9313 | 0.9842 | pass |
| three-arm binary-outcome law, greedy selector | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm selector C-TMLE | -0.0043 to 0.0014 | 0.9325 | 0.9587 | pass |
| three-arm binary-outcome law, greedy selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm selector C-TMLE | -0.0040 to 0.0014 | 0.9450 | 0.9903 | pass |
| three-arm binary-outcome law, greedy selector | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm selector C-TMLE | -0.0012 to 0.0027 | 0.9487 | 0.9835 | pass |
| three-arm binary-outcome law, greedy selector | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm selector C-TMLE | -0.0027 to 0.0013 | 0.9413 | 0.9884 | pass |
| three-arm binary-outcome law, greedy selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm selector C-TMLE | -0.0025 to 0.0014 | 0.9313 | 0.9728 | pass |
| three-arm binary-outcome law, greedy selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0166 to 0.0068 | 0.9350 | 0.9586 | pass |
| three-arm binary-outcome law, greedy selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0148 to 0.0083 | 0.9437 | 0.9902 | pass |
| three-arm binary-outcome law, greedy selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0091 to 0.0038 | 0.9363 | 0.9577 | pass |
| three-arm binary-outcome law, greedy selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0075 to 0.0040 | 0.9463 | 0.9898 | pass |
| three-arm binary-outcome law, ordered selector | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm selector C-TMLE | -0.0027 to 0.0029 | 0.9437 | 0.9847 | pass |
| three-arm binary-outcome law, ordered selector | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm selector C-TMLE | -0.0029 to 0.0024 | 0.9500 | 0.9992 | pass |
| three-arm binary-outcome law, ordered selector | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm selector C-TMLE | -0.000962 to 0.0031 | 0.9387 | 0.9665 | pass |
| three-arm binary-outcome law, ordered selector | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm selector C-TMLE | -0.000877 to 0.0031 | 0.9487 | 0.9984 | pass |
| three-arm binary-outcome law, ordered selector | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm selector C-TMLE | -0.0011 to 0.0027 | 0.9437 | 1.0014 | pass |
| three-arm binary-outcome law, ordered selector | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0102 to 0.0126 | 0.9463 | 0.9851 | pass |
| three-arm binary-outcome law, ordered selector | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0101 to 0.0130 | 0.9513 | 1.0005 | pass |
| three-arm binary-outcome law, ordered selector | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0061 to 0.0065 | 0.9463 | 0.9834 | pass |
| three-arm binary-outcome law, ordered selector | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm selector C-TMLE | -0.0061 to 0.0054 | 0.9500 | 0.9888 | pass |
<!-- /generated -->

## Canonical comparison

No canonical implementation is compared for this row. The committed `equivalence.csv` is
therefore intentionally empty.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9085 to 0.9427, SE ratio 0.8949 to 0.9828 | **fail** |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000115, coverage 0.9087 to 0.9702, SE ratio 0.9545 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.000897, coverage 0.9057 to 0.9683, SE ratio 0.9881 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000707, coverage 0.9087 to 0.9702, SE ratio 0.9834 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5492 to -0.4578 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5092 to -0.5030 | pass |
| `selector_necessity` | `collaborative` | positive | the selector chooses its own mechanism path | bias interval inside the equivalence margin | bias 0.1591 to 0.1662, margin 0.0069 | **fail** |
| `selector_necessity` | `empty_control` | control | the selector is forced to stop at an empty path | bias interval must fall entirely outside the margin | bias 0.1591 to 0.1662, margin 0.0069 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0700, 0.0412 to 0.1095 | **fail** |
<!-- /generated -->

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications per selector strategy |
| `n` | 1500 | observations per primary replication |
| `independent_tests_total` | 27 | implementation-estimand truth tests |
| `independent_tests_passed` | 27 | truth tests passing |
| `subject_tests_total` | 27 | selector truth tests |
| `subject_tests_passed` | 27 | selector truth tests passing |
| `property_cells_total` | 10 | repeated-sampling property cells |
| `property_cells_passed` | 6 | property cells passing |
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
| `margin:selector_rmse_ratio` | 0.8000 | required collaborative-to-control RMSE ratio |

## Limitations

This row has reporting policy because multi-arm repeated-sampling evidence was absent when the
study was declared. Its negative control is intentionally demanding: if the selected path and
empty path coincide, both cells and their shared RMSE ratio remain visible as failures. The row
does not establish equivalence to an external package, simultaneous inference, conditional
effects, cross-fitted primary performance, or behavior under severe positivity loss.
