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
| intervals | pointwise 95% identity-scale Wald intervals | native Horvitz-Thompson influence-curve intervals |

The R learner adapter reads an auxiliary observation-weight column and removes it from the
predictor design. `lmtp` keeps the task weights in targeting, averaging, and covariance. The
diagnostic artifact records the native standard error beside direct Horvitz-Thompson and Hájek
calculations. It confirms that the native result uses the uncentered Horvitz-Thompson form.

The target-weight control uses regimen means because baseline selection shifts both means in the
same direction. That common shift can cancel in a contrast. The double-robustness and targeting
controls use contrasts because their deliberate mutations move those contrasts. The learner-weight
control uses a separate history-dependent selection law that moves every sequential nuisance.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` cross-fitted weighted LTMLE | -0.0020 to 0.0048 | 0.9313 | 0.9787 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` with observation weights | -0.0018 to 0.0049 | 0.9375 | 0.9810 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` cross-fitted weighted LTMLE | -0.0037 to 0.0032 | 0.9450 | 0.9860 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` with observation weights | -0.0039 to 0.0030 | 0.9475 | 0.9888 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` cross-fitted weighted LTMLE | -0.0012 to 0.0026 | 0.9287 | 0.9216 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` with observation weights | -0.0012 to 0.0026 | 0.9625 | 1.0897 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` cross-fitted weighted LTMLE | -0.0035 to 0.0022 | 0.9400 | 0.9980 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` with observation weights | -0.0036 to 0.0020 | 0.9487 | 1.0316 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` cross-fitted weighted LTMLE | -0.0029 to 0.0010 | 0.9450 | 0.9641 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` with observation weights | -0.0032 to 0.000733 | 0.9712 | 1.1109 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -0.000148 | 0.0267 | 1.0082 | -0.0062 | 0.0068 vs 0.0500 | equivalent |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 0.000180 | 0.0319 | 1.0122 | -0.0025 | 0.0078 vs 0.0500 | equivalent |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | -0.000029 | 0.0091 | 1.0109 | -0.0338 | 0.1121 vs 0.0500 **>** | *underpowered* |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 0.000120 | 0.0258 | 1.0131 | -0.0088 | 0.0659 vs 0.0500 **>** | equivalent |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | 0.000299 | 0.0926 | 1.0157 | -0.0262 | 0.1113 vs 0.0500 **>** | *underpowered* |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0021, margin 0.0074, SE ratio 0.9902 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0167 to 0.0218, margin 0.0085, SE ratio 1.0429 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0021 to 0.0028, margin 0.0082, SE ratio 0.9637 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0018 to 0.0025, margin 0.0072, SE ratio 1.1484 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000882 to 0.0099, margin 0.0181, SE ratio 0.9679 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0249 to -0.0150, margin 0.0167, SE ratio 0.6653 | **fail** |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0037 to 0.0068, margin 0.0177, SE ratio 1.0040 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0058 to 0.0048, margin 0.0178, SE ratio 0.6075 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9278 to 0.9529, SE ratio 0.9493 to 1.0242, empirical efficiency ratio 0.9875 to 1.0651, reported efficiency ratio 1.0084 to 1.0142 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8072 to 0.8473, SE ratio 0.6785 to 0.7341, empirical efficiency ratio 1.3779 to 1.4900, reported efficiency ratio 1.0083 to 1.0141 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7998 to 0.8406, SE ratio 0.6646 to 0.7171, empirical efficiency ratio 0.9871 to 1.0645, reported efficiency ratio 0.7059 to 0.7099 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9260 to 0.9514, SE ratio 0.9501 to 1.0269, empirical efficiency ratio 1.0130 to 1.0923, reported efficiency ratio 1.0334 to 1.0443 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8016 to 0.8421, SE ratio 0.6845 to 0.7373, empirical efficiency ratio 1.4098 to 1.5172, reported efficiency ratio 1.0334 to 1.0444 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7907 to 0.8322, SE ratio 0.6652 to 0.7175, empirical efficiency ratio 1.0146 to 1.0917, reported efficiency ratio 0.7233 to 0.7310 | pass |
| `learner_weight_necessity` | `static__discarded_learner_weight_control` | control | static plan: nuisance learners discard sampling weights while later estimator stages retain them | population-target bias outside its margin, learner-selected-target bias inside its margin, and paired displacement above its threshold | bias 0.0378 to 0.0480, margin 0.0171, selected-target bias -0.000246 to 0.0100, margin 0.0171 | pass |
| `learner_weight_necessity` | `static__weighted_learners` | positive | static plan: sampling weights enter nuisance learning, targeting, averaging, and covariance | population-target bias interval inside the equivalence margin | bias -0.0067 to 0.0033, margin 0.0167 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.8812, 0.8489 to 0.9090 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000026, coverage 0.9311 to 0.9708, SE ratio 1.0038 | pass |
| `root_n_and_efficiency` | `dynamic__n_500` | control | dynamic plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0023, coverage 0.9267 to 0.9677, SE ratio 1.0050 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000009, coverage 0.9416 to 0.9776, SE ratio 1.0124 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000848, coverage 0.9194 to 0.9627, SE ratio 1.0057 | pass |
| `root_n_and_efficiency` | `static__n_500` | control | static plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0025, coverage 0.8641 to 0.9211, SE ratio 1.0429 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000573, coverage 0.9416 to 0.9776, SE ratio 1.0653 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5520 to -0.4870 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5206 to -0.5125 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5771 to -0.5104 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5451 to -0.5275 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0029 to 0.0018, margin 0.0080 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0220 to 0.0275, margin 0.0093 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0053 to 0.0048, margin 0.0170 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0265 to -0.0171, margin 0.0157 | pass |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0625, 0.0425 to 0.0879 | pass |
| `weight_necessity` | `dynamic__omitted_weight_control` | control | dynamic plan: the identical selected rows analyzed without any observation weights | population-target bias outside its margin, selected-target bias inside its margin, and paired displacement above its threshold | bias -0.0318 to -0.0276, margin 0.0071, selected-target bias -0.000578 to 0.0036, margin 0.0071 | pass |
| `weight_necessity` | `dynamic__weighted` | positive | dynamic plan: the selected sample analyzed with its fixed inverse-selection weights | population-target bias interval inside the equivalence margin | bias -0.0016 to 0.0037, margin 0.0090 | pass |
| `weight_necessity` | `static__omitted_weight_control` | control | static plan: the identical selected rows analyzed without any observation weights | population-target bias outside its margin, selected-target bias inside its margin, and paired displacement above its threshold | bias -0.0344 to -0.0271, margin 0.0122, selected-target bias -0.0031 to 0.0041, margin 0.0122 | pass |
| `weight_necessity` | `static__weighted` | positive | static plan: the selected sample analyzed with its fixed inverse-selection weights | population-target bias interval inside the equivalence margin | bias -0.0052 to 0.0040, margin 0.0154 | pass |
<!-- /generated -->

Agreement with `lmtp` is distributional because its sequential regression implementation is not
the same solver path as `cleverly`'s. All ten implementation-estimand truth tests pass. Three
paired comparisons conclude equivalence.

The two remaining regimen-mean comparisons use `lmtp`'s native Horvitz-Thompson standard errors.
The `reference-inference.csv.gz` artifact confirms that identity and records the centered Hájek
result separately. The failed paired gates therefore name an inference-convention difference
instead of an estimator disagreement. The static both-wrong control remains marginal because its
bias interval overlaps the discrimination boundary.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is generated
from the committed artifacts and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | selected observations per paired replication |
| `independent_tests_total` | 10 | implementation-estimand truth tests |
| `independent_tests_passed` | 10 | truth tests passing |
| `paired_tests_total` | 5 | paired estimand comparisons |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `property_cells_total` | 36 | independent property cells |
| `property_cells_passed` | 35 | property cells passing |
| `max_standardized_bias` | 0.0572 | largest primary standardized bias |
| `min_coverage` | 0.9287 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9020 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.8624 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1876 | highest primary SE-ratio endpoint |
| `max_margin_utilization` | 0.0926 | largest share of paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0157 | largest paired RMSE-ratio bound |
| `min_coverage_difference_lower` | -0.0500 | smallest paired coverage-difference bound |
| `max_calibration_excess_upper` | 0.1009 | largest paired calibration-excess bound |
| `properties[weight_necessity/static__weighted]:weight_displacement` | 0.4896 | target-weight positive control displacement |
| `properties[learner_weight_necessity/static__weighted_learners]:learner_weight_displacement` | 0.6679 | learner-weight positive control displacement |
| `properties[double_robustness/static__both_wrong]:bias_ci_upper` | -0.0150 | both-wrong control endpoint |
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
