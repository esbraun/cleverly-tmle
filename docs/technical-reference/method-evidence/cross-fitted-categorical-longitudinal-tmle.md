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
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | `cleverly` cross-fitted categorical LTMLE | -0.0015 to 0.0029 | 0.9480 | 1.0021 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | R `lmtp` | -0.001000 to 0.0035 | 0.9447 | 0.9986 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | `cleverly` cross-fitted categorical LTMLE | -0.0019 to 0.0018 | 0.9495 | 1.0046 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | R `lmtp` | -0.000955 to 0.0029 | 0.9463 | 0.9986 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | `cleverly` cross-fitted categorical LTMLE | -0.000770 to 0.0029 | 0.9517 | 1.0169 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | R `lmtp` | 0.000560 to 0.0045 | 0.9530 | 1.0245 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | `cleverly` cross-fitted categorical LTMLE | -0.0021 to 0.0020 | 0.9465 | 1.0019 | pass |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | R `lmtp` | -0.0025 to 0.0019 | 0.9480 | 1.0056 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | `cleverly` cross-fitted categorical LTMLE | -0.0013 to 0.0020 | 0.9437 | 1.0137 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | R `lmtp` | -0.0014 to 0.0021 | 0.9467 | 1.0062 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | `cleverly` cross-fitted categorical LTMLE | -0.0017 to 0.000999 | 0.9503 | 1.0126 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | R `lmtp` | -0.0023 to 0.000506 | 0.9490 | 1.0121 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | `cleverly` cross-fitted categorical LTMLE | -0.0017 to 0.000866 | 0.9427 | 0.9877 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | R `lmtp` | -0.0012 to 0.0014 | 0.9453 | 0.9912 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | `cleverly` cross-fitted categorical LTMLE | -0.000579 to 0.0020 | 0.9395 | 0.9930 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | R `lmtp` | 0.000212 to 0.0030 | 0.9515 | 1.0043 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | `cleverly` cross-fitted categorical LTMLE | -0.0020 to 0.0011 | 0.9483 | 1.0065 | pass |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | R `lmtp` | -0.0028 to 0.000411 | 0.9473 | 1.0143 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[high vs low]` | difference in mean outcome between the plans "assign the high arm at both times" against "assign the low arm at both times" | -0.000545 | 0.0670 | 0.9753 | 0.0032 | 0.0129 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[respond vs low]` | difference in mean outcome between the plans "assign standard first, then high if L2 equals one and low otherwise" against "assign the low arm at both times" | -0.0010 | 0.1494 | 0.9749 | 0.0032 | 0.0110 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[standard vs low]` | difference in mean outcome between the plans "assign the standard arm at both times" against "assign the low arm at both times" | -0.0014 | 0.2063 | 0.9526 | -0.0012 | 0.0157 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ate_regimen[step_down vs low]` | difference in mean outcome between the plans "assign high first, then standard" against "assign the low arm at both times" | 0.000229 | 0.0293 | 0.9745 | -0.0015 | 0.0153 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[high]` | mean outcome under the plan assign the high arm at both times | 0.000010 | 0.0016 | 0.9822 | -0.0030 | 0.0116 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | 0.000554 | 0.1087 | 0.9668 | 0.0013 | 0.0109 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[respond]` | mean outcome under the plan assign standard first, then high if L2 equals one and low otherwise | -0.000491 | 0.1029 | 0.9957 | -0.0025 | 0.0079 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[standard]` | mean outcome under the plan assign the standard arm at both times | -0.000892 | 0.1795 | 0.9411 | -0.0120 | 0.0189 vs 0.0500 | equivalent |
| two-time-point law with three treatment levels at both nodes | `ey_regimen[step_down]` | mean outcome under the plan assign high first, then standard | 0.000784 | 0.1343 | 0.9765 | 0.0010 | 0.0213 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `categorical_probability_necessity` | `third_arm__assigned_probability` | positive | third-arm static plan: the clever covariate selects the assigned third arm's own probability | bias interval inside the equivalence margin | bias -0.000451 to 0.0068, margin 0.0122 | pass |
| `categorical_probability_necessity` | `third_arm__binary_complement` | control | third-arm static plan: the same fit replaces the third arm's probability with a binary complement | bias interval must fall entirely outside the margin | bias 0.1113 to 0.1179, margin 0.0109 | pass |
| `crossfit_overfitting` | `cross_fitted_categorical_ltmle` | positive | five-fold categorical LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.1739 to 1.1953 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.2684 to 0.2734 | pass |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0032 to 0.0036, margin 0.0115, SE ratio 1.0019 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0275 to 0.0342, margin 0.0112, SE ratio 0.9662 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000829 to 0.0059, margin 0.0112, SE ratio 1.0526 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0010 to 0.0056, margin 0.0112, SE ratio 0.9597 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0034 to 0.0036, margin 0.0117, SE ratio 0.9852 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.1451 to 0.1517, margin 0.0111, SE ratio 1.0286 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0012 to 0.0086, margin 0.0124, SE ratio 0.9948 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000157 to 0.0066, margin 0.0114, SE ratio 0.9549 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9359 to 0.9546, SE ratio 0.9657 to 1.0213, empirical efficiency ratio 0.9909 to 1.0481, reported efficiency ratio 1.0104 to 1.0132 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8300 to 0.8597, SE ratio 0.6973 to 0.7385, empirical efficiency ratio 1.3705 to 1.4511, reported efficiency ratio 1.0104 to 1.0133 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8085 to 0.8397, SE ratio 0.6753 to 0.7164, empirical efficiency ratio 0.9887 to 1.0485, reported efficiency ratio 0.7073 to 0.7093 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9383 to 0.9566, SE ratio 0.9712 to 1.0284, empirical efficiency ratio 0.9832 to 1.0409, reported efficiency ratio 1.0094 to 1.0126 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8163 to 0.8469, SE ratio 0.6890 to 0.7295, empirical efficiency ratio 1.3861 to 1.4667, reported efficiency ratio 1.0094 to 1.0126 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8134 to 0.8443, SE ratio 0.6790 to 0.7196, empirical efficiency ratio 0.9837 to 1.0422, reported efficiency ratio 0.7066 to 0.7088 | pass |
| `power` | `dynamic__alternative` | positive | dynamic plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9688, 0.9493 to 0.9824 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000773, coverage 0.9238 to 0.9657, SE ratio 1.0011 | pass |
| `root_n_and_efficiency` | `dynamic__n_500` | control | dynamic plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0024, coverage 0.9150 to 0.9596, SE ratio 0.9508 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000676, coverage 0.9282 to 0.9688, SE ratio 0.9880 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0017, coverage 0.9356 to 0.9737, SE ratio 1.0148 | pass |
| `root_n_and_efficiency` | `static__n_500` | control | static plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0030, coverage 0.8963 to 0.9460, SE ratio 0.9509 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000159, coverage 0.9311 to 0.9708, SE ratio 1.0145 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5620 to -0.4965 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5181 to -0.5130 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5737 to -0.5014 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5173 to -0.5118 | pass |
| `rule_necessity` | `dynamic__declared_rule` | positive | dynamic plan: the declared categorical rule selects its second-node arm from the history | bias interval inside the equivalence margin | bias -0.0031 to 0.0037, margin 0.0115 | pass |
| `rule_necessity` | `dynamic__reversed_rule` | control | dynamic plan: the same fit reverses the rule's two history-specific arm assignments | bias interval must fall entirely outside the margin | bias -0.3151 to -0.3075, margin 0.0127 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0041 to 0.0029, margin 0.0117 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0211 to 0.0279, margin 0.0113 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0042 to 0.0029, margin 0.0119 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.1450 to 0.1513, margin 0.0106 | pass |
| `type_i_error` | `dynamic__sharp_null` | positive | dynamic plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0525, 0.0343 to 0.0762 | pass |
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
| `max_standardized_bias` | 0.0524 | largest primary standardized bias |
| `min_coverage` | 0.9395 | lowest primary coverage |
| `max_margin_utilization` | 0.2063 | largest paired similarity margin share |
| `max_rmse_ratio_upper` | 0.9957 | largest paired RMSE-ratio bound |
| `properties[crossfit_overfitting/cross_fitted_categorical_ltmle]:coverage` | 0.9774 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.4037 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_categorical_ltmle]:coverage_gain_ci_lower` | 0.5672 | paired coverage-gain lower bound |
| `properties[categorical_probability_necessity/third_arm__binary_complement]:standardized_bias` | 2.6191 | binary-complement control |
| `properties[rule_necessity/dynamic__reversed_rule]:standardized_bias` | -6.1396 | reversed-rule control |
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
