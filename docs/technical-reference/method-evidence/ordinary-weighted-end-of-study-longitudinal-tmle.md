# Ordinary weighted end-of-study longitudinal TMLE

This reporting study evaluates `cleverly`'s ordinary weighted LTMLE on a selected sample from a
known two-time-point law. The canonical comparison uses R `ltmle` 1.3-0 with
`observation.weights=`. Each replication contains exactly 2,000 selected rows, and the fixed
inverse-selection weights recover the population law.

The row is reporting evidence rather than a gated claim that every predeclared Monte Carlo cell
passed. The tables below preserve both the strong numerical agreement with R and the failed
finite-sample checks.

## What was compared

| setting | `cleverly` | R `ltmle` |
| --- | --- | --- |
| samples | exact-size rejection samples selected by baseline `W1` | the identical selected rows |
| plans | never, always, and the dynamic rule | the same plans |
| outcome regressions | weighted quasibinomial GLMs | the same weighted GLMs |
| mechanisms | generating treatment and censoring probabilities | the same probabilities |
| observation weights | fixed inverse-selection probabilities | `observation.weights=` |
| folds | one all-row fold | ordinary fitting |
| intervals | pointwise 95% identity-scale Wald intervals | the same, with influence-curve variance |

The selection probability is 0.3 when `W1` is positive and 0.9 otherwise. The independent
property law separately tests whether weighting changes the target, whether nuisance learners
consume the weights, longitudinal double robustness, targeting, root-n behavior, interval
calibration, type-I error, and power.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` ordinary weighted LTMLE | -0.0046 to 0.0024 | 0.9187 | 0.9200 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `ltmle` with observation weights | -0.0046 to 0.0024 | 0.9187 | 0.9200 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` ordinary weighted LTMLE | -0.0040 to 0.0030 | 0.9237 | 0.9472 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `ltmle` with observation weights | -0.0040 to 0.0030 | 0.9237 | 0.9472 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` ordinary weighted LTMLE | -0.0020 to 0.0016 | 0.9487 | 0.9832 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `ltmle` with observation weights | -0.0020 to 0.0016 | 0.9487 | 0.9832 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` ordinary weighted LTMLE | -0.0021 to 0.0039 | 0.9237 | 0.9286 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `ltmle` with observation weights | -0.0021 to 0.0039 | 0.9237 | 0.9286 | **fail** |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` ordinary weighted LTMLE | -0.0015 to 0.0022 | 0.9563 | 1.0226 | pass |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `ltmle` with observation weights | -0.0015 to 0.0022 | 0.9563 | 1.0226 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | 8.957e-12 | 1.544e-09 | 1.0000 | 0 | 1.042e-08 vs 0.0500 | equivalent |
| selected two-time-point law with monotone censoring and fixed observation weights | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 1.145e-10 | 1.987e-08 | 1.0000 | 0 | 2.907e-08 vs 0.0500 | equivalent |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[always]` | mean outcome under the plan treat at both times | -3.696e-11 | 1.270e-08 | 1.0000 | 0 | 1.928e-09 vs 0.0500 | equivalent |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[never]` | mean outcome under the plan treat at neither time | -4.592e-11 | 9.375e-09 | 1.0000 | 0 | 3.057e-10 vs 0.0500 | equivalent |
| selected two-time-point law with monotone censoring and fixed observation weights | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | 6.856e-11 | 2.264e-08 | 1.0000 | 0 | 1.319e-07 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0012 to 0.0031, margin 0.0072, SE ratio 1.0111 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0167 to 0.0219, margin 0.0088, SE ratio 1.0026 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0013 to 0.0031, margin 0.0074, SE ratio 1.0734 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0029 to 0.0015, margin 0.0073, SE ratio 1.1091 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0090 to 0.0015, margin 0.0175, SE ratio 0.9602 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0281 to -0.0186, margin 0.0160, SE ratio 0.6887 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0044 to 0.0059, margin 0.0174, SE ratio 1.0373 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0070 to 0.0031, margin 0.0170, SE ratio 0.6069 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9301 to 0.9548, SE ratio 0.9683 to 1.0445, empirical efficiency ratio 0.9558 to 1.0303, reported efficiency ratio 0.9948 to 1.0007 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8037 to 0.8441, SE ratio 0.6770 to 0.7283, empirical efficiency ratio 1.3704 to 1.4736, reported efficiency ratio 0.9948 to 1.0006 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8181 to 0.8572, SE ratio 0.6772 to 0.7315, empirical efficiency ratio 0.9549 to 1.0310, reported efficiency ratio 0.6964 to 0.7004 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9187 to 0.9454, SE ratio 0.9263 to 0.9985, empirical efficiency ratio 0.9855 to 1.0615, reported efficiency ratio 0.9783 to 0.9899 | **fail** |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7985 to 0.8394, SE ratio 0.6625 to 0.7153, empirical efficiency ratio 1.3759 to 1.4850, reported efficiency ratio 0.9784 to 0.9899 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7860 to 0.8278, SE ratio 0.6481 to 0.6988, empirical efficiency ratio 0.9863 to 1.0622, reported efficiency ratio 0.6849 to 0.6929 | pass |
| `learner_weight_necessity` | `static__discarded_learner_weight_control` | control | static plan: nuisance learners discard sampling weights while later estimator stages retain them | the paired displacement must exceed the declared necessity margin | bias 0.0347 to 0.0449, margin 0.0171 | pass |
| `learner_weight_necessity` | `static__weighted_learners` | positive | static plan: sampling weights enter nuisance learning, targeting, averaging, and covariance | population-target bias interval inside the equivalence margin | bias -0.0094 to 0.000558, margin 0.0166 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.8888, 0.8571 to 0.9156 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000218, coverage 0.9238 to 0.9657, SE ratio 0.9862 | pass |
| `root_n_and_efficiency` | `dynamic__n_500` | control | dynamic plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0013, coverage 0.8921 to 0.9428, SE ratio 0.9049 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000309, coverage 0.9223 to 0.9647, SE ratio 0.9965 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000089, coverage 0.9006 to 0.9491, SE ratio 0.9373 | pass |
| `root_n_and_efficiency` | `static__n_500` | control | static plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0114, coverage 0.7852 to 0.8559, SE ratio 0.7482 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0015, coverage 0.9194 to 0.9627, SE ratio 1.0283 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5573 to -0.4938 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4956 to -0.4866 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5767 to -0.5066 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4344 to -0.4200 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0025 to 0.0019, margin 0.0075 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0229 to 0.0284, margin 0.0093 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0027 to 0.0075, margin 0.0172 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0242 to -0.0148, margin 0.0158 | **fail** |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0750, 0.0530 to 0.1022 | **fail** |
| `weight_necessity` | `dynamic__omitted_weight_control` | control | dynamic plan: the identical selected rows analyzed without any observation weights | the paired displacement must exceed the declared necessity margin | bias -0.0333 to -0.0293, margin 0.0068 | pass |
| `weight_necessity` | `dynamic__weighted` | positive | dynamic plan: the selected sample analyzed with its fixed inverse-selection weights | population-target bias interval inside the equivalence margin | bias -0.0022 to 0.0029, margin 0.0085 | pass |
| `weight_necessity` | `static__omitted_weight_control` | control | static plan: the identical selected rows analyzed without any observation weights | the paired displacement must exceed the declared necessity margin | bias -0.0328 to -0.0258, margin 0.0118 | pass |
| `weight_necessity` | `static__weighted` | positive | static plan: the selected sample analyzed with its fixed inverse-selection weights | population-target bias interval inside the equivalence margin | bias -0.0016 to 0.0072, margin 0.0148 | pass |
<!-- /generated -->

The ordinary estimator and R reference agree to numerical-solver precision. That agreement does
not turn the failed independent checks into passes. In this realized run, the static calibration
coverage bound, the static untargeted negative control, and the type-I upper bound miss their
predeclared gates. The remaining property cells include direct positive and negative controls for
both target weighting and learner weighting.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is generated
from the committed artifacts and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | selected observations per paired replication |
| `independent_tests_total` | 10 | implementation-estimand truth tests |
| `independent_tests_passed` | 4 | truth tests passing |
| `paired_tests_total` | 5 | paired estimand comparisons |
| `paired_tests_passed` | 5 | paired comparisons passing |
| `property_cells_total` | 36 | independent property cells |
| `property_cells_passed` | 30 | property cells passing |
| `max_standardized_bias` | 0.0280 | largest primary standardized bias |
| `min_coverage` | 0.9187 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.8907 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.8595 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0957 | highest primary SE-ratio endpoint |
| `max_margin_utilization` | 2.264e-08 | largest share of paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0000 | largest paired RMSE-ratio bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference bound |
| `max_calibration_excess_upper` | 1.703e-07 | largest paired calibration-excess bound |
| `properties[weight_necessity/static__weighted]:weight_displacement` | 0.5414 | target-weight positive control displacement |
| `properties[learner_weight_necessity/static__weighted_learners]:learner_weight_displacement` | 0.6648 | learner-weight positive control displacement |
| `properties[interval_calibration/static__correctly_specified]:coverage_ci_lower` | 0.9187 | failed static calibration endpoint |
| `properties[type_i_error/static__sharp_null]:rejection_ci_upper` | 0.1022 | failed type-I endpoint |
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
| The observation weights are known and fixed | The study does not cover estimated weights, weight-model uncertainty, replicate weights, or calibration weights |
| One selection law is studied | The selected sample uses one baseline-dependent probability with moderate weight variation, not severe practical positivity |
| The row covers one terminal binary mean per plan | Survival curves, competing risks, and longitudinal MSM projections have different parameters |
| The primary learner is a weighted GLM | The learner-weight property is a separate finite-support control, not parity for arbitrary learner libraries |
| Inference is pointwise | The row does not validate simultaneous bands or clustered covariance |
| Fitting is ordinary | Cross-fitted weighted LTMLE is evaluated in its own registered row |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, conditionally independent censoring, and a selection model that identifies the
target population.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/weighted_ltmle/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/weighted_ltmle/manifest.json)
records the seeds, configuration, pinned R source, study-module digests, and artifact hashes.
