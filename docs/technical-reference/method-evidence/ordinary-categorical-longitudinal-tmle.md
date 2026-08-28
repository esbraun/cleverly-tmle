# Ordinary categorical longitudinal TMLE

This study validates ordinary longitudinal TMLE with three labelled treatment levels at two
nodes. It covers static, mixed, and history-dependent deterministic plans. The comparison uses R
[`lmtp`](https://github.com/nt-williams/lmtp) 1.5.4 at commit `f04a2b4`.

The finite-support law scrambles raw arm codes and semantic arm order. Its direct g-formula and
Gateaux derivative provide independent parameter and influence-curve oracles.

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | complete binary-outcome samples generated in Python | the identical rows |
| treatment | three string-labelled levels at each of two nodes | the identical labelled columns |
| plans | three constant plans, one mixed plan, and one dynamic plan | the same shifted columns |
| mechanism | the exact categorical probability for the assigned arm | the same exact per-node density ratios |
| sequential regressions | follower-stratified quasibinomial regressions | `SL.glm` quasibinomial regressions |
| fitting | one ordinary fit over all rows | one fold with all rows in training and validation |
| intervals | pointwise 95% identity-scale Wald intervals | the same influence-curve intervals |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | `cleverly` ordinary categorical LTMLE | -0.0024 to 0.0038 | 0.9370 | 0.9637 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | R `lmtp` | -0.0021 to 0.0044 | 0.9405 | 0.9724 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | `cleverly` ordinary categorical LTMLE | -0.0043 to 0.000901 | 0.9480 | 1.0090 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | R `lmtp` | -0.0036 to 0.0017 | 0.9480 | 1.0182 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | `cleverly` ordinary categorical LTMLE | -0.0027 to 0.0026 | 0.9430 | 0.9879 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | R `lmtp` | -0.0021 to 0.0035 | 0.9475 | 1.0010 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | `cleverly` ordinary categorical LTMLE | -0.0040 to 0.0019 | 0.9400 | 0.9797 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | R `lmtp` | -0.0042 to 0.0020 | 0.9435 | 0.9946 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | `cleverly` ordinary categorical LTMLE | -0.0013 to 0.0035 | 0.9415 | 0.9732 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | R `lmtp` | -0.0011 to 0.0038 | 0.9425 | 0.9779 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | `cleverly` ordinary categorical LTMLE | -0.0015 to 0.0024 | 0.9475 | 0.9970 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | R `lmtp` | -0.0018 to 0.0022 | 0.9515 | 1.0022 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | `cleverly` ordinary categorical LTMLE | -0.0030 to 0.000552 | 0.9430 | 0.9913 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | R `lmtp` | -0.0026 to 0.0010 | 0.9440 | 0.9956 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | `cleverly` ordinary categorical LTMLE | -0.0014 to 0.0023 | 0.9400 | 0.9772 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | R `lmtp` | -0.0011 to 0.0029 | 0.9415 | 0.9895 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | `cleverly` ordinary categorical LTMLE | -0.0029 to 0.0016 | 0.9300 | 0.9692 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | R `lmtp` | -0.0032 to 0.0014 | 0.9360 | 0.9860 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | -0.000484 | 0.0584 | 0.9817 | -0.0035 | 0.0162 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | -0.000714 | 0.1042 | 0.9867 | 0 | 0.0252 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | -0.000684 | 0.0960 | 0.9574 | -0.0045 | 0.0166 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | 0.000017 | 0.0021 | 0.9794 | -0.0035 | 0.0128 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | -0.000215 | 0.0338 | 0.9872 | -0.0010 | 0.0163 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | 0.000270 | 0.0527 | 0.9701 | -0.0040 | 0.0161 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | -0.000444 | 0.0943 | 0.9912 | -0.0010 | 0.0101 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | -0.000414 | 0.0829 | 0.9429 | -0.0015 | 0.0192 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | 0.000287 | 0.0482 | 0.9761 | -0.0060 | 0.0152 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `categorical_probability_necessity` | `third_arm__assigned_probability` | positive | third-arm static plan: the clever covariate selects the assigned third arm's own probability | bias interval inside the equivalence margin | bias -0.0051 to 0.0019, margin 0.0118 | pass |
| `categorical_probability_necessity` | `third_arm__binary_complement` | control | third-arm static plan: the same fit replaces the third arm's probability with a binary complement | bias interval must fall entirely outside the margin | bias 0.1075 to 0.1140, margin 0.0109 | pass |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0042 to 0.0025, margin 0.0112, SE ratio 1.0080 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0260 to 0.0332, margin 0.0120, SE ratio 0.9022 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0053 to 0.0015, margin 0.0114, SE ratio 1.0309 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0036 to 0.0034, margin 0.0118, SE ratio 0.8896 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0041 to 0.0026, margin 0.0112, SE ratio 1.0103 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.1459 to 0.1524, margin 0.0111, SE ratio 1.0216 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0032 to 0.0042, margin 0.0125, SE ratio 0.9838 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0042 to 0.0025, margin 0.0113, SE ratio 0.9506 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9388 to 0.9571, SE ratio 0.9584 to 1.0169, empirical efficiency ratio 0.9744 to 1.0336, reported efficiency ratio 0.9893 to 0.9920 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8196 to 0.8501, SE ratio 0.6871 to 0.7280, empirical efficiency ratio 1.3606 to 1.4419, reported efficiency ratio 0.9893 to 0.9920 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8080 to 0.8392, SE ratio 0.6711 to 0.7112, empirical efficiency ratio 0.9750 to 1.0336, reported efficiency ratio 0.6925 to 0.6944 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9289 to 0.9486, SE ratio 0.9479 to 1.0045, empirical efficiency ratio 0.9857 to 1.0438, reported efficiency ratio 0.9882 to 0.9914 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8181 to 0.8486, SE ratio 0.6865 to 0.7271, empirical efficiency ratio 1.3616 to 1.4415, reported efficiency ratio 0.9882 to 0.9914 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7959 to 0.8279, SE ratio 0.6637 to 0.7033, empirical efficiency ratio 0.9858 to 1.0437, reported efficiency ratio 0.6917 to 0.6940 | pass |
| `power` | `dynamic__alternative` | positive | dynamic plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9738, 0.9555 to 0.9861 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000638, coverage 0.9121 to 0.9575, SE ratio 0.9601 | pass |
| `root_n_and_efficiency` | `dynamic__n_500` | control | dynamic plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000644, coverage 0.9035 to 0.9512, SE ratio 0.9357 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000860, coverage 0.9092 to 0.9554, SE ratio 0.9683 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0022, coverage 0.9194 to 0.9627, SE ratio 0.9786 | pass |
| `root_n_and_efficiency` | `static__n_500` | control | static plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000915, coverage 0.8864 to 0.9385, SE ratio 0.9096 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000661, coverage 0.9107 to 0.9565, SE ratio 0.9581 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5310 to -0.4655 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4885 to -0.4841 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5394 to -0.4711 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4893 to -0.4838 | pass |
| `rule_necessity` | `dynamic__declared_rule` | positive | dynamic plan: the declared categorical rule selects its second-node arm from the history | bias interval inside the equivalence margin | bias -0.0038 to 0.0030, margin 0.0114 | pass |
| `rule_necessity` | `dynamic__reversed_rule` | control | dynamic plan: the same fit reverses the rule's two history-specific arm assignments | bias interval must fall entirely outside the margin | bias -0.3159 to -0.3084, margin 0.0125 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0021 to 0.0049, margin 0.0117 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0233 to 0.0301, margin 0.0113 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0019 to 0.0056, margin 0.0125 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.1466 to 0.1533, margin 0.0113 | pass |
| `type_i_error` | `dynamic__sharp_null` | positive | dynamic plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0400, 0.0243 to 0.0614 | pass |
<!-- /generated -->

The property study uses saturated nuisance learners against the exact law. It tests sequential
double robustness, root-n behavior, efficiency, interval calibration, null size, and power.
Paired mutations remove targeting, replace a third-arm probability, and reverse the dynamic rule.

## Measured values

Names that begin with `margin:` are declared thresholds. The other values come from committed
artifacts.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | paired replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 18 | implementation-estimand truth tests |
| `independent_tests_passed` | 18 | truth tests passing |
| `paired_tests_total` | 9 | paired comparisons |
| `paired_tests_passed` | 9 | paired comparisons passing |
| `property_cells_total` | 34 | independent property cells |
| `property_cells_passed` | 34 | property cells passing |
| `max_standardized_bias` | 0.0399 | largest primary standardized bias |
| `min_coverage` | 0.9300 | lowest primary coverage |
| `max_margin_utilization` | 0.1042 | largest paired similarity margin share |
| `max_rmse_ratio_upper` | 0.9912 | largest paired RMSE-ratio bound |
| `properties[categorical_probability_necessity/third_arm__binary_complement]:standardized_bias` | 2.5356 | binary-complement control |
| `properties[rule_necessity/dynamic__reversed_rule]:standardized_bias` | -6.2599 | reversed-rule control |
| `properties[targeting_necessity/dynamic__untargeted]:standardized_bias` | 0.5887 | untargeted dynamic control |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | interval and test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage ceiling |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin |
| `margin:rmse_noninferiority` | 1.1000 | RMSE non-inferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage non-inferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration non-inferiority bound |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval excludes |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio lower screen |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio upper screen |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.1000 | targeting displacement floor |
| `margin:necessity_displacement` | 0.1000 | dynamic-rule displacement floor |
| `margin:categorical_probability_displacement` | 0.1000 | arm-probability displacement floor |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The law has complete outcomes | The row excludes censoring and missingness |
| The law has two nodes and three treatment levels | More nodes, continuous doses, and stochastic categorical policies need separate evidence |
| The primary comparison uses one quasibinomial GLM family | Flexible primary learner parity is not established |
| The property fits use saturated discrete-cell learners | Their efficiency claim applies to this finite-support law |
| The fitting scheme is ordinary | Cross-fitting has a separate registered row |
| Inference is pointwise 95% Wald | Simultaneous bands and bootstrap intervals are excluded |
| Plans are deterministic | Survival, competing risks, and stochastic policies are different parameters |
| The design is independent and unweighted | Weights, clusters, fold repeats, and time-respecting splits are excluded |

The causal interpretation requires consistency, sequential exchangeability, and longitudinal
positivity. Single-correct-nuisance cells establish consistency, not calibrated inference.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/categorical_ltmle/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/categorical_ltmle/manifest.json)
records the seeds, pins, source hashes, and artifact hashes.
