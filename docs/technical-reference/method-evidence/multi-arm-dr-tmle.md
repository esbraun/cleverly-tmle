# Multi-arm DR-TMLE

This study exercises the armwise multi-treatment extension implemented by `cleverly` and R
[`drtmle`](https://github.com/benkeser/drtmle) 1.1.2 at pinned commit `538a3a2`. The two
implementations receive identical samples, exact five-fold assignments, and the same Python
out-of-fold initial outcome and treatment predictions. That isolates the reduced regressions,
corrections, targeting, and inference from nuisance-fit differences.

The binary-outcome law reports three arm means and ATE, risk-ratio, and odds-ratio contrasts
against the `high` reference arm. A separate fit-diagnostics artifact audits empirical scores
and solver status for every replication.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` multi-arm DR-TMLE | -0.0026 to 0.0020 | 0.9537 | 1.0577 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[low vs high]` | difference in counterfactual means, low versus high | R `drtmle` multi-arm extension | -0.0026 to 0.0020 | 0.9537 | 1.0577 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` multi-arm DR-TMLE | -0.0031 to 0.0017 | 0.9525 | 1.0114 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[medium vs high]` | difference in counterfactual means, medium versus high | R `drtmle` multi-arm extension | -0.0031 to 0.0017 | 0.9525 | 1.0114 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` multi-arm DR-TMLE | -0.000915 to 0.0026 | 0.9600 | 0.9971 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[high]` | counterfactual mean under treatment arm 'high' | R `drtmle` multi-arm extension | -0.000910 to 0.0026 | 0.9600 | 0.9971 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` multi-arm DR-TMLE | -0.0011 to 0.0023 | 0.9563 | 1.0299 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[low]` | counterfactual mean under treatment arm 'low' | R `drtmle` multi-arm extension | -0.0011 to 0.0023 | 0.9563 | 1.0296 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` multi-arm DR-TMLE | -0.0015 to 0.0018 | 0.9637 | 1.0423 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[medium]` | counterfactual mean under treatment arm 'medium' | R `drtmle` multi-arm extension | -0.0015 to 0.0018 | 0.9637 | 1.0424 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` multi-arm DR-TMLE | -0.0098 to 0.0092 | 0.9563 | 1.0561 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | R `drtmle` multi-arm extension | -0.0098 to 0.0091 | 0.9563 | 1.0561 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` multi-arm DR-TMLE | -0.0115 to 0.0092 | 0.9537 | 1.0114 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | R `drtmle` multi-arm extension | -0.0116 to 0.0092 | 0.9537 | 1.0114 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` multi-arm DR-TMLE | -0.0057 to 0.0049 | 0.9550 | 1.0476 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | R `drtmle` multi-arm extension | -0.0057 to 0.0048 | 0.9550 | 1.0476 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` multi-arm DR-TMLE | -0.0063 to 0.0041 | 0.9563 | 1.0016 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | R `drtmle` multi-arm extension | -0.0063 to 0.0041 | 0.9563 | 1.0016 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[low vs high]` | difference in counterfactual means, low versus high | 0.000003 | 0.000669 | 1.0004 | 0 | 0.000491 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[medium vs high]` | difference in counterfactual means, medium versus high | 0.000010 | 0.0026 | 1.0004 | 0 | 0.000351 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[high]` | counterfactual mean under treatment arm 'high' | -0.000005 | 0.0017 | 1.0004 | 0 | 0.000288 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[low]` | counterfactual mean under treatment arm 'low' | -0.000003 | 0.000910 | 1.0001 | 0 | 0.000740 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[medium]` | counterfactual mean under treatment arm 'medium' | 0.000005 | 0.0019 | 1.0005 | 0 | 0.000389 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | 0.000022 | 0.000794 | 1.0005 | 0 | 0.000497 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | 0.000132 | 0.0026 | 1.0005 | 0 | 0.000340 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | 0.000012 | 0.0010 | 1.0005 | 0 | 0.000430 vs 0.0500 | equivalent |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | 0.000035 | 0.0025 | 1.0004 | 0 | 0.000312 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robust_contraction` | `both_wrong_n2000` | control | both nuisances are misspecified, at n = 2,000 | the exact coverage interval must fall below the floor | coverage 0.4650 to 0.5714, bias 0.0505 | pass |
| `double_robust_contraction` | `both_wrong_n4000` | control | both nuisances are misspecified, at n = 4,000 | the exact coverage interval must fall below the floor | coverage 0.2058 to 0.2983, bias 0.0488 | pass |
| `double_robust_contraction` | `both_wrong_n8000` | control | both nuisances are misspecified, at n = 8,000 | the exact coverage interval must fall below the floor | coverage 0.0223 to 0.0654, bias 0.0489 | pass |
| `double_robust_contraction` | `outcome_correct_n2000` | positive | only the outcome regression is correctly specified, at n = 2,000 | the exact coverage interval clears the declared floor | coverage 0.9265 to 0.9727, bias 0.0011 | pass |
| `double_robust_contraction` | `outcome_correct_n4000` | positive | only the outcome regression is correctly specified, at n = 4,000 | the exact coverage interval clears the declared floor | coverage 0.8988 to 0.9542, bias 0.000926 | **fail** |
| `double_robust_contraction` | `outcome_correct_n8000` | positive | only the outcome regression is correctly specified, at n = 8,000 | the exact coverage interval clears the declared floor | coverage 0.9145 to 0.9649, bias 0.0017 | pass |
| `double_robust_contraction` | `rate_both_wrong` | control | the same regression with both nuisances misspecified | slope interval must not establish contraction | slope -0.0657 to 0.0237 | pass |
| `double_robust_contraction` | `rate_outcome_correct` | positive | log absolute bias regressed on log n across three sizes, outcome regression correct | slope interval entirely below zero, so the bias contracts | slope -1.1804 to 3.6785 | **fail** |
| `double_robust_contraction` | `rate_treatment_correct` | positive | the same regression with only the treatment mechanism correct | slope interval entirely below zero, so the bias contracts | slope -3.3803 to 3.2787 | **fail** |
| `double_robust_contraction` | `treatment_correct_n2000` | positive | only the treatment mechanism is correctly specified, at n = 2,000 | the exact coverage interval clears the declared floor | coverage 0.9145 to 0.9649, bias 0.000233 | pass |
| `double_robust_contraction` | `treatment_correct_n4000` | positive | only the treatment mechanism is correctly specified, at n = 4,000 | the exact coverage interval clears the declared floor | coverage 0.9204 to 0.9688, bias -0.000672 | pass |
| `double_robust_contraction` | `treatment_correct_n8000` | positive | only the treatment mechanism is correctly specified, at n = 8,000 | the exact coverage interval clears the declared floor | coverage 0.9285 to 0.9740, bias 0.000686 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0050 to 0.0026, margin 0.0090, SE ratio 1.0624 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0463 to 0.0540, margin 0.0092, SE ratio 1.0337 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000186 to 0.0079, margin 0.0095, SE ratio 0.9992 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0011 to 0.0068, margin 0.0068, SE ratio 0.9925 | **fail** |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9272 to 0.9576, SE ratio 0.9246 to 1.0107 | **fail** |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000510, coverage 0.9373 to 0.9859, SE ratio 1.0061 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.000840, coverage 0.9087 to 0.9702, SE ratio 0.9853 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000517, coverage 0.9243 to 0.9792, SE ratio 1.0458 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5837 to -0.4827 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5158 to -0.5079 | pass |
<!-- /generated -->

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 18 | implementation-estimand truth tests |
| `independent_tests_passed` | 18 | truth tests passing |
| `paired_tests_total` | 9 | paired comparison tests |
| `paired_tests_passed` | 9 | paired tests passing |
| `property_cells_total` | 22 | repeated-sampling property cells |
| `property_cells_passed` | 17 | property cells passing |
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
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## Limitations

This is a reporting study. A red scientific verdict publishes rather than prevents the record
from existing. The source theorem and the original R study are binary-treatment results. Both
packages expose a vector intervention API, and this row measures their armwise extension. It does
not claim a new multi-arm theorem.

The contraction ladder explains most of the red cells above. The `treatment_correct` level cell
misses its equivalence margin at n = 2,000 by less than the interval's own last printed digit.
The ladder redraws that regime on independent streams at three sizes and puts the bias inside the
margin at every one, so the level cell records Monte Carlo error rather than a remainder.

Both fitted slopes then have nothing to regress. A one-correct bias that sits at the noise floor
and changes sign across the ladder gives a wide slope interval, and neither interval establishes
contraction. Read the ladder through its coverage rungs instead. Those rungs hold near the
nominal rate in both one-correct regimes, while the both-wrong control's coverage collapses as
the sample grows and its bias does not move.

Two cells are red on their own terms. The `interval_calibration` SE-ratio interval reaches just
below the lower calibration band. The `outcome_correct` rung at n = 4,000 has a coverage lower
bound just under the floor.

One R replication out of 800 exceeds the shared empirical-score bar. Every `cleverly` replication
clears it. The propensity bound stays inactive throughout, and the subject fit is refused if it
does not.

The row covers one binary-outcome law, one fold count, pooled reduced cross-fitting, univariate
reduction, ordinary GLM nuisance fits, and pointwise intervals. It excludes flexible learners,
practical-positivity stress, missing outcomes, weights, clusters, fold repeats, simultaneous
bands, and longitudinal treatment.
