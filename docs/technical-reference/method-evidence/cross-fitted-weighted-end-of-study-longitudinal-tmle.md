# Cross-fitted weighted end-of-study longitudinal TMLE

This reporting study evaluates `cleverly`'s cross-fitted weighted LTMLE on a selected sample from
a known two-time-point law. The canonical comparison uses R `lmtp` 1.5.4. Each replication
contains exactly 2,000 selected rows, and both implementations use the identical rowwise
five-fold assignment.

The row is reporting evidence rather than a gated claim that every predeclared Monte Carlo cell
passed. The complete tables publish the independent validity checks beside the distributional R
comparison.

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| samples | exact-size rejection samples selected by baseline `W1` | the identical selected rows |
| plans | never, always, and the dynamic rule | the same plans |
| outcome regressions | fold-specific weighted quasibinomial GLMs | a shared weighted GLM adapter |
| mechanisms | generating treatment and censoring probabilities | the same density ratios |
| observation weights | fixed inverse-selection probabilities | `weights=` on every task |
| folds | one exact five-fold assignment | the identical assignment |
| intervals | pointwise 95% identity-scale Wald intervals | the corresponding influence-curve intervals |

The R learner adapter reads an auxiliary observation-weight column and removes it from the
predictor design. `lmtp` keeps the task weights in targeting, averaging, and covariance. The
independent property law tests target weighting and learner weighting directly, in addition to
longitudinal double robustness, targeting, root-n behavior, interval calibration, type-I error,
and power.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` cross-fitted weighted LTMLE | -0.0019 to 0.0048 | 0.9325 | 0.9766 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` with observation weights | -0.0018 to 0.0049 | 0.8938 | 0.8506 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` cross-fitted weighted LTMLE | -0.0036 to 0.0032 | 0.9450 | 0.9843 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` with observation weights | -0.0038 to 0.0030 | 0.9062 | 0.8618 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` cross-fitted weighted LTMLE | -0.0012 to 0.0026 | 0.9313 | 0.9214 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` with observation weights | -0.0011 to 0.0027 | 0.9450 | 1.0062 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` cross-fitted weighted LTMLE | -0.0035 to 0.0021 | 0.9387 | 0.9944 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` with observation weights | -0.0036 to 0.0020 | 0.8675 | 0.7695 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` cross-fitted weighted LTMLE | -0.0029 to 0.0011 | 0.9450 | 0.9634 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` with observation weights | -0.0032 to 0.000735 | 0.9587 | 1.0329 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -0.000119 | 0.0215 | 1.0095 | 0.0387 | 0.0687 vs 0.0500 **>** | superior |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 0.000212 | 0.0378 | 1.0130 | 0.0387 | 0.0838 vs 0.0500 **>** | superior |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | -0.000014 | 0.0046 | 1.0110 | -0.0137 | 0.0169 vs 0.0500 | **inconclusive** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 0.000105 | 0.0225 | 1.0147 | 0.0712 | 0.0976 vs 0.0500 **>** | superior |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | 0.000317 | 0.0981 | 1.0167 | -0.0138 | 0.0672 vs 0.0500 **>** | *underpowered* |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0023 to 0.0021, margin 0.0074, SE ratio 0.9887 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0167 to 0.0218, margin 0.0085, SE ratio 1.0428 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0020 to 0.0028, margin 0.0082, SE ratio 0.9641 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0018 to 0.0025, margin 0.0072, SE ratio 1.1459 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000949 to 0.0099, margin 0.0182, SE ratio 0.9643 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0249 to -0.0149, margin 0.0167, SE ratio 0.6656 | **fail** |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0036 to 0.0069, margin 0.0177, SE ratio 1.0039 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0060 to 0.0047, margin 0.0178, SE ratio 0.6058 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9269 to 0.9522, SE ratio 0.9484 to 1.0231, empirical efficiency ratio 0.9881 to 1.0658, reported efficiency ratio 1.0079 to 1.0138 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8068 to 0.8469, SE ratio 0.6789 to 0.7344, empirical efficiency ratio 1.3773 to 1.4887, reported efficiency ratio 1.0079 to 1.0138 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7985 to 0.8394, SE ratio 0.6640 to 0.7167, empirical efficiency ratio 0.9875 to 1.0652, reported efficiency ratio 0.7056 to 0.7097 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9255 to 0.9511, SE ratio 0.9509 to 1.0287, empirical efficiency ratio 1.0122 to 1.0922, reported efficiency ratio 1.0343 to 1.0452 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8037 to 0.8441, SE ratio 0.6851 to 0.7381, empirical efficiency ratio 1.4092 to 1.5172, reported efficiency ratio 1.0342 to 1.0452 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7912 to 0.8326, SE ratio 0.6657 to 0.7177, empirical efficiency ratio 1.0146 to 1.0916, reported efficiency ratio 0.7238 to 0.7316 | pass |
| `learner_weight_necessity` | `static__discarded_learner_weight_control` | control | static plan: nuisance learners discard sampling weights while later estimator stages retain them | the paired displacement must exceed the declared necessity margin | bias 0.0379 to 0.0481, margin 0.0172 | pass |
| `learner_weight_necessity` | `static__weighted_learners` | positive | static plan: sampling weights enter nuisance learning, targeting, averaging, and covariance | population-target bias interval inside the equivalence margin | bias -0.0067 to 0.0033, margin 0.0167 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.8788, 0.8461 to 0.9068 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000002, coverage 0.9326 to 0.9717, SE ratio 1.0036 | pass |
| `root_n_and_efficiency` | `dynamic__n_500` | control | dynamic plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0027, coverage 0.9238 to 0.9657, SE ratio 1.0064 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000020, coverage 0.9416 to 0.9776, SE ratio 1.0132 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000944, coverage 0.9208 to 0.9637, SE ratio 1.0074 | pass |
| `root_n_and_efficiency` | `static__n_500` | control | static plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0029, coverage 0.8780 to 0.9320, SE ratio 1.0432 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000571, coverage 0.9401 to 0.9767, SE ratio 1.0639 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5518 to -0.4862 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5206 to -0.5126 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5756 to -0.5091 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5437 to -0.5263 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0029 to 0.0018, margin 0.0080 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0220 to 0.0275, margin 0.0093 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0053 to 0.0048, margin 0.0170 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0264 to -0.0171, margin 0.0158 | pass |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0612, 0.0415 to 0.0864 | pass |
| `weight_necessity` | `dynamic__omitted_weight_control` | control | dynamic plan: the identical selected rows analyzed without any observation weights | the paired displacement must exceed the declared necessity margin | bias -0.0318 to -0.0276, margin 0.0071 | pass |
| `weight_necessity` | `dynamic__weighted` | positive | dynamic plan: the selected sample analyzed with its fixed inverse-selection weights | population-target bias interval inside the equivalence margin | bias -0.0016 to 0.0038, margin 0.0090 | pass |
| `weight_necessity` | `static__omitted_weight_control` | control | static plan: the identical selected rows analyzed without any observation weights | the paired displacement must exceed the declared necessity margin | bias -0.0344 to -0.0271, margin 0.0122 | pass |
| `weight_necessity` | `static__weighted` | positive | static plan: the selected sample analyzed with its fixed inverse-selection weights | population-target bias interval inside the equivalence margin | bias -0.0052 to 0.0040, margin 0.0154 | pass |
<!-- /generated -->

Agreement with `lmtp` is distributional because its sequential regression implementation is not
the same solver path as `cleverly`'s. All five `cleverly` truth tests pass. Three R truth tests
fail their coverage or SE screens. The paired results therefore include three superiority
conclusions, one inconclusive coverage comparison, and one comparison whose SE-calibration margin
the run could not resolve. The static both-wrong control is the only failed property cell. Its
bias interval overlaps the discrimination margin instead of clearing it.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is generated
from the committed artifacts and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | selected observations per paired replication |
| `independent_tests_total` | 10 | implementation-estimand truth tests |
| `independent_tests_passed` | 7 | truth tests passing |
| `paired_tests_total` | 5 | paired estimand comparisons |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `property_cells_total` | 36 | independent property cells |
| `property_cells_passed` | 35 | property cells passing |
| `max_standardized_bias` | 0.0571 | largest primary standardized bias |
| `min_coverage` | 0.8675 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.8338 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.7245 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1043 | highest primary SE-ratio endpoint |
| `max_margin_utilization` | 0.0981 | largest share of paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0167 | largest paired RMSE-ratio bound |
| `min_coverage_difference_lower` | -0.0262 | smallest paired coverage-difference bound |
| `max_calibration_excess_upper` | 0.0893 | largest paired calibration-excess bound |
| `properties[weight_necessity/static__weighted]:weight_displacement` | 0.4879 | target-weight positive control displacement |
| `properties[learner_weight_necessity/static__weighted_learners]:learner_weight_displacement` | 0.6665 | learner-weight positive control displacement |
| `properties[double_robustness/static__both_wrong]:bias_ci_upper` | -0.0149 | both-wrong control endpoint |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
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
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope upper bound |
| `margin:excluded_slope` | -0.2500 | rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio lower screen |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio upper screen |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least targeting must move the estimate |
| `margin:weight_displacement` | 0.2500 | least target weighting must move the estimate |
| `margin:learner_weight_displacement` | 0.2500 | least learner weighting must move the estimate |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The study is reporting evidence | Failed cells stay red. The row does not support a blanket claim that every predeclared finite-sample property passed |
| Agreement with `lmtp` is distributional | Different sequential regression implementations can differ at statistical rather than solver scale |
| The observation weights are known and fixed | The study does not cover estimated weights, weight-model uncertainty, replicate weights, or calibration weights |
| One fixed five-fold assignment is studied | The row does not validate repeated folds or time-respecting splits |
| One selection law is studied | The selected sample uses one baseline-dependent probability with moderate weight variation, not severe practical positivity |
| The row covers one terminal binary mean per plan | Survival curves, competing risks, and longitudinal MSM projections have different parameters |
| The primary learner is a weighted GLM | The learner-weight property is a separate finite-support control, not parity for arbitrary learner libraries |
| Inference is pointwise | The row does not validate simultaneous bands or clustered covariance |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, conditionally independent censoring, and a selection model that identifies the
target population.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/weighted_lmtp_ltmle/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/weighted_lmtp_ltmle/manifest.json)
records the seeds, configuration, pinned R source, adapter digests, study-module digests, and
artifact hashes.
