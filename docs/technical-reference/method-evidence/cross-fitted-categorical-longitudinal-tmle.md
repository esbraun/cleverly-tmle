# Cross-fitted categorical longitudinal TMLE

This study validates five-fold longitudinal TMLE with three labelled treatment levels at two
nodes. It covers the same deterministic plans as the ordinary study. The comparison uses R
[`lmtp`](https://github.com/nt-williams/lmtp) 1.5.4 at commit `f04a2b4`.

Both implementations receive the exact rowwise folds and the exact assigned-arm probabilities.
Each training recursion predicts only its matching held-out rows.

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | complete binary-outcome samples generated in Python | the identical rows |
| treatment | three string-labelled levels at each of two nodes | the identical labelled columns |
| plans | three constant plans, one mixed plan, and one dynamic plan | the same shifted columns |
| mechanism | the exact categorical probability for the assigned arm | the same exact per-node density ratios |
| folds | one declared balanced five-fold assignment | the identical rowwise assignment |
| sequential regressions | fold-specific quasibinomial regressions | fold-specific `SL.glm` regressions |
| intervals | pointwise 95% identity-scale Wald intervals | the same influence-curve intervals |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | `cleverly` cross-fitted categorical LTMLE | -0.0015 to 0.0029 | 0.9477 | 1.0016 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | R `lmtp` | -0.0010 to 0.0035 | 0.9443 | 0.9987 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | `cleverly` cross-fitted categorical LTMLE | -0.0019 to 0.0018 | 0.9485 | 1.0053 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | R `lmtp` | -0.000941 to 0.0029 | 0.9467 | 0.9986 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | `cleverly` cross-fitted categorical LTMLE | -0.000774 to 0.0029 | 0.9510 | 1.0175 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | R `lmtp` | 0.000567 to 0.0045 | 0.9527 | 1.0248 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | `cleverly` cross-fitted categorical LTMLE | -0.0022 to 0.0020 | 0.9470 | 1.0024 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | R `lmtp` | -0.0025 to 0.0019 | 0.9493 | 1.0054 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | `cleverly` cross-fitted categorical LTMLE | -0.0013 to 0.0020 | 0.9437 | 1.0127 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | R `lmtp` | -0.0014 to 0.0020 | 0.9477 | 1.0063 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | `cleverly` cross-fitted categorical LTMLE | -0.0017 to 0.000999 | 0.9510 | 1.0131 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | R `lmtp` | -0.0023 to 0.000499 | 0.9493 | 1.0123 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | `cleverly` cross-fitted categorical LTMLE | -0.0017 to 0.000887 | 0.9433 | 0.9878 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | R `lmtp` | -0.0012 to 0.0014 | 0.9453 | 0.9911 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | `cleverly` cross-fitted categorical LTMLE | -0.000583 to 0.0020 | 0.9395 | 0.9931 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | R `lmtp` | 0.000212 to 0.0030 | 0.9510 | 1.0045 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | `cleverly` cross-fitted categorical LTMLE | -0.0020 to 0.0011 | 0.9475 | 1.0073 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | R `lmtp` | -0.0028 to 0.000406 | 0.9480 | 1.0142 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | -0.000544 | 0.0668 | 0.9759 | 0.0035 | 0.0127 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | -0.0010 | 0.1485 | 0.9741 | 0.0018 | 0.0110 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | -0.0015 | 0.2080 | 0.9526 | -0.0018 | 0.0154 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | 0.000200 | 0.0256 | 0.9740 | -0.0022 | 0.0144 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | 0.000019 | 0.0030 | 0.9831 | -0.0040 | 0.0114 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | 0.000562 | 0.1102 | 0.9665 | 0.0017 | 0.0109 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | -0.000476 | 0.0999 | 0.9953 | -0.0020 | 0.0078 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | -0.000896 | 0.1802 | 0.9414 | -0.0115 | 0.0192 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | 0.000762 | 0.1307 | 0.9757 | -0.000500 | 0.0195 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `categorical_probability_necessity` | `third_arm__assigned_probability` | positive | third-arm static plan: the clever covariate selects the assigned third arm's own probability | bias interval inside the equivalence margin | bias -0.000484 to 0.0068, margin 0.0122 | pass |
| `categorical_probability_necessity` | `third_arm__binary_complement` | control | third-arm static plan: the same fit replaces the third arm's probability with a binary complement | bias interval must fall entirely outside the margin | bias 0.1113 to 0.1178, margin 0.0109 | pass |
| `crossfit_overfitting` | `cross_fitted_categorical_ltmle` | positive | five-fold categorical LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.1694 to 1.1903 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.2684 to 0.2734 | pass |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0033 to 0.0036, margin 0.0115, SE ratio 0.9995 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0275 to 0.0342, margin 0.0112, SE ratio 0.9672 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000836 to 0.0059, margin 0.0112, SE ratio 1.0525 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000970 to 0.0057, margin 0.0112, SE ratio 0.9591 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0034 to 0.0036, margin 0.0117, SE ratio 0.9844 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.1451 to 0.1517, margin 0.0111, SE ratio 1.0299 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0012 to 0.0086, margin 0.0124, SE ratio 0.9947 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000070 to 0.0067, margin 0.0114, SE ratio 0.9570 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9372 to 0.9557, SE ratio 0.9661 to 1.0213, empirical efficiency ratio 0.9905 to 1.0474, reported efficiency ratio 1.0104 to 1.0132 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8302 to 0.8599, SE ratio 0.6977 to 0.7388, empirical efficiency ratio 1.3692 to 1.4502, reported efficiency ratio 1.0104 to 1.0133 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8080 to 0.8392, SE ratio 0.6757 to 0.7166, empirical efficiency ratio 0.9886 to 1.0480, reported efficiency ratio 0.7073 to 0.7093 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9391 to 0.9573, SE ratio 0.9719 to 1.0295, empirical efficiency ratio 0.9823 to 1.0399, reported efficiency ratio 1.0095 to 1.0127 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8168 to 0.8474, SE ratio 0.6894 to 0.7298, empirical efficiency ratio 1.3853 to 1.4660, reported efficiency ratio 1.0095 to 1.0127 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8113 to 0.8424, SE ratio 0.6797 to 0.7205, empirical efficiency ratio 0.9827 to 1.0411, reported efficiency ratio 0.7066 to 0.7089 | pass |
| `power` | `dynamic__alternative` | positive | dynamic plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9700, 0.9508 to 0.9833 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000744, coverage 0.9252 to 0.9667, SE ratio 1.0013 | pass |
| `root_n_and_efficiency` | `dynamic__n_500` | control | dynamic plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0030, coverage 0.9078 to 0.9544, SE ratio 0.9514 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000695, coverage 0.9282 to 0.9688, SE ratio 0.9883 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0017, coverage 0.9371 to 0.9747, SE ratio 1.0137 | pass |
| `root_n_and_efficiency` | `static__n_500` | control | static plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0029, coverage 0.9020 to 0.9502, SE ratio 0.9496 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000160, coverage 0.9282 to 0.9688, SE ratio 1.0139 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5620 to -0.4968 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5186 to -0.5135 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5732 to -0.5022 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5173 to -0.5118 | pass |
| `rule_necessity` | `dynamic__declared_rule` | positive | dynamic plan: the declared categorical rule selects its second-node arm from the history | bias interval inside the equivalence margin | bias -0.0031 to 0.0037, margin 0.0115 | pass |
| `rule_necessity` | `dynamic__reversed_rule` | control | dynamic plan: the same fit reverses the rule's two history-specific arm assignments | bias interval must fall entirely outside the margin | bias -0.3150 to -0.3075, margin 0.0127 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0041 to 0.0029, margin 0.0117 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0211 to 0.0279, margin 0.0113 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0042 to 0.0029, margin 0.0119 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.1450 to 0.1513, margin 0.0106 | pass |
| `type_i_error` | `dynamic__sharp_null` | positive | dynamic plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0575, 0.0384 to 0.0821 | pass |
<!-- /generated -->

The property study repeats every ordinary instrument with five outer folds. A paired tree control
compares held-out predictions with the same learner fitted and evaluated in sample.

## Measured values

Names that begin with `margin:` are declared thresholds. The other values come from committed
artifacts.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4000 | paired replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 18 | implementation-estimand truth tests |
| `independent_tests_passed` | 18 | truth tests passing |
| `paired_tests_total` | 9 | paired comparisons |
| `paired_tests_passed` | 9 | paired comparisons passing |
| `property_cells_total` | 36 | independent property cells |
| `property_cells_passed` | 36 | property cells passing |
| `max_standardized_bias` | 0.0525 | largest primary standardized bias |
| `min_coverage` | 0.9395 | lowest primary coverage |
| `max_margin_utilization` | 0.2080 | largest paired similarity margin share |
| `max_rmse_ratio_upper` | 0.9953 | largest paired RMSE-ratio bound |
| `properties[crossfit_overfitting/cross_fitted_categorical_ltmle]:coverage` | 0.9768 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.4037 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_categorical_ltmle]:coverage_gain_ci_lower` | 0.5667 | paired coverage-gain lower bound |
| `properties[categorical_probability_necessity/third_arm__binary_complement]:standardized_bias` | 2.6175 | binary-complement control |
| `properties[rule_necessity/dynamic__reversed_rule]:standardized_bias` | -6.1323 | reversed-rule control |
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
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio floor |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio ceiling |
| `margin:overfit_coverage_gain` | 0.1500 | paired coverage-gain floor |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The law has complete outcomes | The row excludes censoring and missingness |
| The law has two nodes and three treatment levels | More nodes, continuous doses, and stochastic categorical policies need separate evidence |
| The primary comparison uses one quasibinomial GLM family | Flexible primary learner parity is not established |
| The property fits use saturated discrete-cell learners | Their efficiency claim applies to this finite-support law |
| One fixed five-fold assignment is studied | Fold repeats and time-respecting splits are excluded |
| Inference is pointwise 95% Wald | Simultaneous bands and bootstrap intervals are excluded |
| Plans are deterministic | Survival and competing risks are different parameters |
| The design is independent and unweighted | Weights and clusters are excluded |

The causal interpretation requires consistency, sequential exchangeability, and longitudinal
positivity. Single-correct-nuisance cells establish consistency, not calibrated inference.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/categorical_ltmle_crossfit/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/categorical_ltmle_crossfit/manifest.json)
records the seeds, pins, source hashes, and artifact hashes.
