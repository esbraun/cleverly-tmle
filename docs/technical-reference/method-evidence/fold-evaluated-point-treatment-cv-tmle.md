# Fold-evaluated point-treatment CV-TMLE

This study validates `cleverly`'s fold-evaluated CV-TMLE report: treatment-stratified ten-fold
nuisance fitting, one pooled targeting update, equal-fold plug-in evaluation, and the
cross-validated influence-curve variance. It is separate from the stacked row above because
averaging fold reports, rather than evaluating the updated regression over the whole sample, is a
genuine finite-sample method choice. The construction and its source boundary are mapped in
[CV-TMLE and cross-fitting](../cv-tmle.md#the-algorithm-as-implemented).

**No canonical implementation is compared.** No maintained package ships this construction, so the
study rests on the accuracy and theory-property questions alone. A zero-row equivalence artifact
records that absence rather than borrowing the stacked R comparison.

## What was compared

| setting | `cleverly` |
| --- | --- |
| construction | ten-fold nuisance fitting, pooled update, equal-fold plug-in evaluation |
| variance | cross-validated influence curve |
| estimands | `ey1`, `ey0`, `ate`, `att`, `atc` |
| laws | the same binary and bounded-continuous laws the ordinary TMLE study uses |
| nuisance learners | corresponding logistic and linear GLM |
| propensity bounds | 0.025 to 0.975 |
| intervals | pointwise 95% identity-scale Wald |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` fold-evaluated CV-TMLE | -0.000879 to 0.0031 | 0.9456 | 0.9987 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` fold-evaluated CV-TMLE | -0.0011 to 0.0028 | 0.9506 | 0.9997 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` fold-evaluated CV-TMLE | -0.0015 to 0.0026 | 0.9550 | 1.0048 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` fold-evaluated CV-TMLE | -0.0019 to 0.000936 | 0.9525 | 1.0093 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` fold-evaluated CV-TMLE | -0.0011 to 0.0018 | 0.9575 | 1.0155 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` fold-evaluated CV-TMLE | -0.000360 to 0.000559 | 0.9513 | 0.9866 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` fold-evaluated CV-TMLE | -0.000264 to 0.000593 | 0.9500 | 1.0034 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` fold-evaluated CV-TMLE | -0.000186 to 0.000735 | 0.9375 | 0.9756 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` fold-evaluated CV-TMLE | -0.000702 to 0.000203 | 0.9519 | 0.9967 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` fold-evaluated CV-TMLE | -0.000603 to 0.000433 | 0.9537 | 1.0243 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `fold_evaluated_cvtmle` | positive | fold-evaluated CV-TMLE with a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9091 to 1.0908 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.5308 to 0.6408 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0100 to 0.0038, margin 0.0231, SE ratio 0.9693 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.3437 to -0.3259, margin 0.0299, SE ratio 1.0330 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0064 to 0.0067, margin 0.0219, SE ratio 0.9709 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0305 to -0.0086, margin 0.0366, SE ratio 0.9331 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9652, SE ratio 0.9730 to 1.0456 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000506, coverage 0.9238 to 0.9657, SE ratio 0.9825 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000654, coverage 0.9252 to 0.9667, SE ratio 1.0139 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000293, coverage 0.9326 to 0.9717, SE ratio 1.0088 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5374 to -0.4732 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5084 to -0.5058 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0325, 0.0141 to 0.0627 | pass |
<!-- /generated -->

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 10 | estimand-law tests against truth |
| `independent_tests_passed` | 10 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 14 | repeated-sampling property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0384 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9375 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9203 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9303 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0737 | highest bootstrap SE-ratio endpoint |
| `properties[double_robustness/outcome_correct]:bias` | 0.000149 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0196 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3348 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5053 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5374 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4732 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5071 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0080 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9730 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0456 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9550 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage` | 0.8950 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:se_ratio` | 0.9910 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6500 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5792 | SE calibration of that control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage_gain_ci_lower` | 0.1875 | paired 99% lower bound for coverage gained over the control |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval below |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the exact coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size the one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest RMSE ratio the one-sided upper bound may reach |
| `margin:coverage_noninferiority` | -0.0250 | smallest coverage difference the one-sided lower bound may reach |
| `margin:calibration_noninferiority` | 0.0500 | largest excess SE-calibration error the upper bound may reach |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power control must clear |
| `margin:root_n_slope` | -0.5000 | the contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | the slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling the in-sample control's upper bound must stay below |
| `margin:overfit_coverage_gain` | 0.1500 | coverage cross-fitting must buy over the in-sample control |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | The row rests on accuracy against known truth and on the theory properties. It is not parity evidence for stacked R CV-TMLE, and it does not inherit the stacked row's comparison |
| The cross-fit overfitting cells are relative evidence | A fully grown regression tree is fitted on the nonlinear law twice, once with held-out predictions and once in sample, on the identical 400 samples of size 500. The cross-fitted cell's evidence is restored SE calibration and a load-bearing improvement over the control. The primary GLM study carries the absolute coverage gate |
| The row is bounded to one fixed ten-fold assignment per sample | It does not establish repeated or nested cross-fitting, a fold-specific targeting epsilon, simultaneous or bootstrap intervals, missing outcomes, weights, clusters, strata, multi-valued treatment, ratio estimands, observed-risk functionals, or behaviour under severe practical-positivity violations |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/manifest.json)
records the primary and control samples, every margin and seed, the exact estimator configuration,
and the source and result hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/properties.csv)
carry every published row.
