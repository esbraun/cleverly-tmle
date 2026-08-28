# Repeated point-treatment cross-fitted TMLE

This study validates `cleverly`'s repeated stacked CV-TMLE report: five-fold nuisance fitting,
one pooled targeting update per draw, whole-sample plug-in evaluation, arithmetic-mean point
aggregation over three independent fold draws, and variance from the row-aligned averaged
influence curve. Risk ratios and odds ratios are aggregated on their log inference scale.

**No canonical implementation is compared.** [Chernozhukov et al. (2018), Definition
3.3](https://academic.oup.com/ectj/article/21/1/C1/5056401), explicitly define mean and median
aggregation over repeated partitions, and Corollary 3.3 gives fixed-repeat first-order validity.
Their mean variance rule adds the within-partition variance and the squared between-partition point
deviation. `cleverly` instead takes the variance of the row-aligned averaged influence curve. Mean
aggregation is therefore not intrinsically invalid; the exact finite-sample variance rule lacks
parity evidence and is tested here against calibration and a study-only implementation of the
published adjustment. A zero-row equivalence artifact records the absence of an
implementation-level comparator.

## What was tested

| setting | declaration |
| --- | --- |
| primary construction | five folds, three complete fold draws, pooled targeting and whole-sample evaluation |
| primary variance | variance of the row-aligned averaged influence curve |
| primary estimands | arm means, ATE, ATT, ATC, observed mean, PAR, PAF, RR, and OR where defined by the law |
| primary laws | binary and bounded-continuous point-treatment laws with exact truth |
| fixed-sample stability | 200 fold seeds on one nonlinear sample of 600 rows; three-draw averaging against one split and equal-fold evaluation |
| aggregation decision | 400 samples of 1,000 rows and five identical fold draws; mean against median under oracle and rare-tail stress nuisance regimes |
| variance decision | current averaged-curve interval against the Chernozhukov/DML mean split-dispersion adjustment on the same fits |
| Monte Carlo inference | 99% intervals around all declared endpoints |

The stress law is generated rather than imposed on fitted estimates. About 4% of observations are
in a rare tail stratum, treatment probability is 0.12 there and 0.50 elsewhere, and treatment
effect modification is strong in that stratum. The outcome nuisance is a fully grown tree and the
treatment mechanism is exact. The specificity control uses the exact outcome and treatment
nuisances on the same law, where a robust repeat rule should have no material advantage.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` repeated stacked CV-TMLE | -0.0018 to 0.0038 | 0.9500 | 1.0111 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` repeated stacked CV-TMLE | -0.0015 to 0.0040 | 0.9537 | 1.0080 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` repeated stacked CV-TMLE | -0.0014 to 0.0043 | 0.9513 | 1.0107 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` repeated stacked CV-TMLE | -0.0027 to 0.0014 | 0.9513 | 0.9913 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` repeated stacked CV-TMLE | -0.0014 to 0.0026 | 0.9537 | 1.0241 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | `cleverly` repeated stacked CV-TMLE | -0.0014 to 0.0016 | 0.9387 | 0.9705 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | `cleverly` repeated stacked CV-TMLE | -0.0047 to 0.0181 | 0.9525 | 1.0081 | pass |
| binary-outcome law | `paf` | population attributable fraction | `cleverly` repeated stacked CV-TMLE | -0.0014 to 0.0048 | 0.9537 | 1.0070 | pass |
| binary-outcome law | `par` | population attributable risk | `cleverly` repeated stacked CV-TMLE | -0.000730 to 0.0022 | 0.9525 | 1.0088 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | `cleverly` repeated stacked CV-TMLE | -0.0025 to 0.0100 | 0.9487 | 1.0008 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` repeated stacked CV-TMLE | -0.000332 to 0.000934 | 0.9550 | 1.0160 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` repeated stacked CV-TMLE | -0.000337 to 0.000847 | 0.9600 | 1.0296 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` repeated stacked CV-TMLE | -0.000326 to 0.000940 | 0.9500 | 1.0047 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` repeated stacked CV-TMLE | -0.000860 to 0.000476 | 0.9363 | 0.9593 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` repeated stacked CV-TMLE | -0.000680 to 0.000805 | 0.9563 | 1.0151 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | `cleverly` repeated stacked CV-TMLE | -0.000874 to 0.000607 | 0.9337 | 0.9607 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | `cleverly` repeated stacked CV-TMLE | -0.000352 to 0.000469 | 0.9525 | 0.9903 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.5300 to 0.6412 | pass |
| `crossfit_overfitting` | `repeated_cvtmle` | positive | three-draw stacked CV-TMLE with a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9304 to 1.1312 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0099 to 0.0038, margin 0.0231, SE ratio 0.9716 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.3443 to -0.3265, margin 0.0298, SE ratio 1.0364 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0061 to 0.0069, margin 0.0218, SE ratio 0.9744 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0303 to -0.0085, margin 0.0366, SE ratio 0.9344 | pass |
| `fold_repeat_stability` | `equal_fold_average` | control | one equal-fold evaluation on the same sample and fold seed | the paired spread ratio against whole-sample evaluation stays above the ceiling | to one split 0.9854 to 1.0031 | pass |
| `fold_repeat_stability` | `one_fixed_split` | control | one pooled split on the same sample and fold seed | the paired spread ratio against equal-fold evaluation stays above the ceiling | to equal-fold 0.9969 to 1.0144 | pass |
| `fold_repeat_stability` | `rowwise_three_draw_average` | positive | three row-aligned fold draws averaged on one fixed nonlinear sample | both paired spread-ratio upper bounds stay below the declared ceiling | to one split 0.4847 to 0.7514, to equal-fold 0.4825 to 0.7509 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9648, SE ratio 0.9743 to 1.0451 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `repeat_aggregation` | `oracle_mean` | positive | the five-draw mean with exact outcome and treatment nuisances | RMSE is non-inferior to the median under the declared ratio | RMSE ratio 0.9986 to 1.0014, 90th-percentile error ratio 0.9920 to 1.0079 | pass |
| `repeat_aggregation` | `oracle_median` | control | the five-draw median on the identical oracle fits | the paired specificity comparison meets the declared ratio | RMSE ratio 0.9986 to 1.0014, 90th-percentile error ratio 0.9920 to 1.0079 | pass |
| `repeat_aggregation` | `stress_mean` | control | the five-draw mean with a fully grown outcome tree on the rare-stratum law | the paired stress comparison detects the median's declared RMSE improvement | RMSE ratio 0.9919 to 1.0393, 90th-percentile error ratio 0.9246 to 1.0768 | **fail** |
| `repeat_aggregation` | `stress_median` | positive | the five-draw median on the identical rare-stratum fits | RMSE improves over the mean by the declared ratio | RMSE ratio 0.9919 to 1.0393, 90th-percentile error ratio 0.9246 to 1.0768 | **fail** |
| `repeat_variance` | `oracle_averaged_ic` | positive | the averaged-curve interval with exact nuisances | coverage clears the floor and the SE-ratio interval stays inside the sanity band | coverage 0.9275 to 0.9809, SE ratio 0.9294 to 1.1566 | pass |
| `repeat_variance` | `oracle_dml_mean` | positive | the split-adjusted mean interval on the identical exact-nuisance fits | coverage clears the floor and the SE-ratio interval stays inside the sanity band | coverage 0.9275 to 0.9809, SE ratio 0.9326 to 1.1596 | pass |
| `repeat_variance` | `stress_averaged_ic` | control | the averaged-curve interval on the rare-stratum tree fits | coverage or the SE-ratio interval must resolve below its validity threshold | coverage 0.8875 to 0.9569, SE ratio 0.8688 to 1.0576 | **fail** |
| `repeat_variance` | `stress_dml_mean` | positive | the split-adjusted mean interval on the identical rare-stratum tree fits | coverage clears the floor and the SE-ratio interval stays inside the sanity band | coverage 0.9439 to 0.9891, SE ratio 1.0935 to 1.3290 | **fail** |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000546, coverage 0.9208 to 0.9637, SE ratio 0.9815 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000887, coverage 0.9282 to 0.9688, SE ratio 1.0156 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000278, coverage 0.9297 to 0.9698, SE ratio 1.0094 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5374 to -0.4727 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5084 to -0.5058 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0300, 0.0125 to 0.0594 | pass |
<!-- /generated -->

## Decision result

The ordinary repeated estimator and every shared CV-TMLE property passed. Three-draw rowwise
averaging also reduced fold-seed spread against both no-averaging controls.

The robust-alternative decision was red. In the oracle specificity arm, mean and median errors
were practically indistinguishable, as intended. In the stress arm, the 99% paired RMSE-ratio
interval for median over mean crossed one and did not clear the predeclared 0.95 upper bound. The
90th-percentile absolute-error ratio was likewise unresolved. This study therefore supplies no
evidence to replace the mean with the median for the tested repeat count and stress law.

The current stress interval's point coverage sat below nominal, but its coverage interval crossed
both the declared floor and nominal coverage. The study therefore established neither validity
nor material undercoverage. The split-dispersion adjustment raised point coverage above nominal,
but its SE-ratio interval exceeded the declared sanity ceiling. It was conservative rather than
calibrated. Neither variance rule passed the declared stress decision, so the public implementation
remains unchanged and the choice is flagged for reconsideration instead of being tuned to this run.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications per law |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 17 | estimand-law tests against truth |
| `independent_tests_passed` | 17 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 25 | repeated-sampling property cells |
| `property_cells_passed` | 17 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0549 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9337 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9078 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.8960 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0998 | highest bootstrap primary SE-ratio endpoint |
| `properties[fold_repeat_stability/rowwise_three_draw_average]:spread_ratio_one_split_ci_upper` | 0.7514 | upper spread-ratio bound against one split |
| `properties[fold_repeat_stability/rowwise_three_draw_average]:spread_ratio_equal_fold_ci_upper` | 0.7509 | upper spread-ratio bound against equal-fold evaluation |
| `properties[repeat_aggregation/oracle_mean]:rmse_ratio_ci_lower` | 0.9986 | oracle mean-to-median RMSE-ratio lower endpoint |
| `properties[repeat_aggregation/oracle_mean]:rmse_ratio_ci_upper` | 1.0014 | oracle mean-to-median RMSE-ratio upper endpoint |
| `properties[repeat_aggregation/stress_median]:rmse_ratio_ci_lower` | 0.9919 | stress median-to-mean RMSE-ratio lower endpoint |
| `properties[repeat_aggregation/stress_median]:rmse_ratio_ci_upper` | 1.0393 | stress median-to-mean RMSE-ratio upper endpoint |
| `properties[repeat_aggregation/stress_median]:p90_error_ratio_ci_lower` | 0.9246 | stress 90th-percentile error-ratio lower endpoint |
| `properties[repeat_aggregation/stress_median]:p90_error_ratio_ci_upper` | 1.0768 | stress 90th-percentile error-ratio upper endpoint |
| `properties[repeat_variance/stress_averaged_ic]:coverage` | 0.9275 | current stress interval coverage |
| `properties[repeat_variance/stress_averaged_ic]:coverage_ci_lower` | 0.8875 | current stress coverage lower endpoint |
| `properties[repeat_variance/stress_averaged_ic]:coverage_ci_upper` | 0.9569 | current stress coverage upper endpoint |
| `properties[repeat_variance/stress_dml_mean]:coverage` | 0.9725 | adjusted stress interval coverage |
| `properties[repeat_variance/stress_dml_mean]:se_ratio_ci_lower` | 1.0935 | adjusted stress SE-ratio lower endpoint |
| `properties[repeat_variance/stress_dml_mean]:se_ratio_ci_upper` | 1.3290 | adjusted stress SE-ratio upper endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size the one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the root-n interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling the in-sample control's upper bound must stay below |
| `margin:overfit_coverage_gain` | 0.1500 | coverage cross-fitting must buy over the in-sample control |
| `margin:fold_repeat_spread_ratio` | 0.8000 | largest accepted repeated-to-control spread ratio |
| `margin:repeat_mean_rmse_ratio` | 1.1000 | largest accepted oracle mean-to-median RMSE ratio |
| `margin:repeat_median_rmse_ratio` | 0.9500 | largest accepted stress median-to-mean RMSE ratio |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | This is exact-truth and statistical-property evidence, not numerical parity with another package |
| The robust aggregation comparison is red | The natural rare-tail/tree regime did not show a median advantage; it does not establish that mean is robust to every split-instability regime or that median can never help |
| Neither stress variance rule passed | The current interval was unresolved and the split-dispersion adjustment was conservative; no variance-rule change is justified by this study |
| The repeat budgets are fixed | Primary evidence covers three fold draws and the decision study covers five; it does not establish behavior for arbitrary repeat counts |
| The scientific scope is point treatment | The row does not validate clustering, observation weights, missing outcomes, longitudinal data, simultaneous intervals, bootstrap inference, or severe positivity violations outside the declared law |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/repeated_crossfit/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/repeated_crossfit/manifest.json)
records the seeds, margins, exact estimator configuration, source hashes, and result hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/repeated_crossfit/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/repeated_crossfit/properties.csv)
carry every published row.
