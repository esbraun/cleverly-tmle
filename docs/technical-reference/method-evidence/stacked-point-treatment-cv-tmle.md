# Stacked point-treatment CV-TMLE

This study validates `cleverly`'s default cross-validated point-treatment construction, described
by Levy (2018). Nuisance predictions are out of fold, one targeting regression is fitted over the
stacked validation rows, and the updated regression is evaluated over the whole sample. Zheng and
van der Laan (2011) supply the wider CV-TMLE framework. The source boundary is mapped in
[CV-TMLE and cross-fitting](../cv-tmle.md#the-algorithm-as-implemented).

## What was compared

| setting | `cleverly` | R `tmle3` CV-TMLE |
| --- | --- | --- |
| construction | stacked update, whole-sample plug-in evaluation | `tmle3_Update(cvtmle = TRUE)` |
| folds | treatment-stratified ten-fold, generated in Python | the identical validation indices, rebuilt with `origami` and asserted on the task |
| nuisance learners | GLM | corresponding GLM wrapped in `Lrnr_cv` |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| intervals | pointwise 95% Wald | pointwise 95% Wald |
| PAF scale | identity, from the PAF influence curve | negative-log-complement, transformed |

Supplying the exact row-to-fold assignment is what makes this a fold-matched comparison. A common
seed or fold count is not enough when splitters or dependency versions differ. The R runner aborts
the whole study on any failed fit, changed fold, missing estimand, or dropped replication.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` stacked CV-TMLE | -0.0043 to -0.000378 | 0.9456 | 1.0139 | pass |
| binary-outcome law | `atc` | average effect on the untreated | R `tmle3` CV-TMLE | -0.0042 to -0.000352 | 0.9469 | 1.0195 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` stacked CV-TMLE | -0.0040 to -0.000199 | 0.9494 | 1.0163 | pass |
| binary-outcome law | `ate` | average treatment effect | R `tmle3` CV-TMLE | -0.0040 to -0.000201 | 0.9506 | 1.0164 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` stacked CV-TMLE | -0.0039 to 0.000098 | 0.9519 | 1.0216 | pass |
| binary-outcome law | `att` | average effect on the treated | R `tmle3` CV-TMLE | -0.0039 to 0.000081 | 0.9525 | 1.0257 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` stacked CV-TMLE | -0.000696 to 0.0021 | 0.9556 | 1.0076 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | R `tmle3` CV-TMLE | -0.000696 to 0.0021 | 0.9556 | 1.0076 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` stacked CV-TMLE | -0.0028 to 0.000051 | 0.9375 | 0.9906 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | R `tmle3` CV-TMLE | -0.0028 to 0.000051 | 0.9375 | 0.9906 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | `cleverly` stacked CV-TMLE | -0.0013 to 0.000802 | 0.9456 | 0.9815 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | R `tmle3` CV-TMLE | -0.0013 to 0.000802 | 0.9456 | 0.9815 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | `cleverly` stacked CV-TMLE | -0.0151 to 0.000844 | 0.9494 | 1.0167 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | R `tmle3` CV-TMLE | -0.0151 to 0.000844 | 0.9494 | 1.0166 | pass |
| binary-outcome law | `paf` | population attributable fraction | `cleverly` stacked CV-TMLE | -0.0039 to 0.000426 | 0.9500 | 1.0234 | pass |
| binary-outcome law | `paf` | population attributable fraction | R `tmle3` CV-TMLE | -0.0040 to 0.0014 | 0.9525 | 1.0233 | pass |
| binary-outcome law | `par` | population attributable risk | `cleverly` stacked CV-TMLE | -0.0020 to 0.000061 | 0.9500 | 1.0209 | pass |
| binary-outcome law | `par` | population attributable risk | R `tmle3` CV-TMLE | -0.0020 to 0.000061 | 0.9506 | 1.0212 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | `cleverly` stacked CV-TMLE | -0.0078 to 0.000826 | 0.9475 | 1.0206 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | R `tmle3` CV-TMLE | -0.0078 to 0.000826 | 0.9475 | 1.0205 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` stacked CV-TMLE | -0.000556 to 0.000353 | 0.9494 | 0.9995 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | R `tmle3` CV-TMLE | 0.000163 to 0.0011 | 0.9500 | 1.0061 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` stacked CV-TMLE | -0.000502 to 0.000337 | 0.9494 | 1.0260 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | R `tmle3` CV-TMLE | -0.000500 to 0.000335 | 0.9506 | 1.0391 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` stacked CV-TMLE | -0.000483 to 0.000426 | 0.9444 | 0.9887 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | R `tmle3` CV-TMLE | -0.0012 to -0.000308 | 0.9406 | 0.9904 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` stacked CV-TMLE | -0.000446 to 0.000473 | 0.9469 | 0.9851 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | R `tmle3` CV-TMLE | -0.000438 to 0.000480 | 0.9463 | 0.9853 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` stacked CV-TMLE | -0.000593 to 0.000456 | 0.9544 | 1.0142 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | R `tmle3` CV-TMLE | -0.000592 to 0.000456 | 0.9556 | 1.0159 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | `cleverly` stacked CV-TMLE | -0.000462 to 0.000556 | 0.9431 | 0.9862 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | R `tmle3` CV-TMLE | -0.000462 to 0.000556 | 0.9431 | 0.9862 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | `cleverly` stacked CV-TMLE | -0.000261 to 0.000327 | 0.9387 | 0.9762 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | R `tmle3` CV-TMLE | -0.000289 to 0.000297 | 0.9394 | 0.9852 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | -0.000037 | 0.0081 | 1.0070 | -0.0012 | 0.0118 vs 0.0500 | equivalent |
| binary-outcome law | `ate` | average treatment effect | 0.000002 | 0.000412 | 1.0001 | -0.0013 | 0.000252 vs 0.0500 | equivalent |
| binary-outcome law | `att` | average effect on the treated | 0.000010 | 0.0021 | 1.0054 | -0.000625 | 0.0086 vs 0.0500 | equivalent |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | 7.188e-08 | 0.000022 | 1.0000 | 0 | 0.000012 vs 0.0500 | equivalent |
| binary-outcome law | `ey1` | counterfactual mean under treatment | 8.047e-08 | 0.000024 | 1.0000 | 0 | 0.000088 vs 0.0500 | equivalent |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | -7.174e-12 | 2.974e-09 | 1.0000 | 0 | 2.163e-10 vs 0.0500 | equivalent |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | 6.970e-09 | 1.750e-07 | 1.0000 | 0 | 0.000013 vs 0.0500 | equivalent |
| binary-outcome law | `paf` | population attributable fraction | -3.430e-07 | 0.000068 | 1.0007 | -0.0025 | n/a | equivalent |
| binary-outcome law | `par` | population attributable risk | -1.467e-07 | 0.000062 | 1.0007 | -0.000625 | 0.000664 vs 0.0500 | equivalent |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | -2.158e-08 | 0.000001 | 1.0000 | 0 | 0.000013 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | -0.000715 | 0.6790 | 1.0128 | -0.000625 | 0.0153 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | 2.801e-07 | 0.000288 | 1.0082 | -0.0013 | 0.0247 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | 0.000733 | 0.6941 | 1.0041 | 0.0037 | 0.0039 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | -0.000007 | 0.0069 | 1.0032 | 0.000625 | 0.0015 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | -6.210e-07 | 0.000509 | 1.0039 | -0.0012 | 0.0047 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | -1.684e-13 | 1.421e-10 | 1.0000 | 0 | 3.920e-11 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | 0.000029 | 0.0422 | 1.0072 | -0.000625 | 0.0024 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.5298 to 0.6400 | pass |
| `crossfit_overfitting` | `stacked_cvtmle` | positive | stacked CV-TMLE with a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9094 to 1.0843 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0100 to 0.0038, margin 0.0231 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.3437 to -0.3259, margin 0.0299 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0064 to 0.0067, margin 0.0219 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias -0.0305 to -0.0086, margin 0.0366 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9652, SE ratio 0.9738 to 1.0450 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000506, coverage 0.9238 to 0.9657, SE ratio 0.9823 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000654, coverage 0.9252 to 0.9667, SE ratio 1.0128 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000293, coverage 0.9326 to 0.9717, SE ratio 1.0087 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5384 to -0.4737 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5080 to -0.5055 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0325, 0.0141 to 0.0627 | pass |
<!-- /generated -->

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 34 | implementation-estimand tests against truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | scenario-estimand paired tests |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 14 | repeated-sampling property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.1082 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9375 | lowest measured coverage of a nominal 95% interval |
| `min_coverage_ci_lower` | 0.9203 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9329 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0884 | highest bootstrap SE-ratio endpoint |
| `max_se_ratio_resolution` | 0.0884 | farthest the widest SE-ratio interval reaches from 1.0 |
| `max_margin_utilization` | 0.6941 | largest share of a paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0128 | largest one-sided RMSE-ratio bound, against 1.10 |
| `min_coverage_difference_lower` | -0.0081 | smallest one-sided coverage-difference bound, against -0.025 |
| `max_calibration_excess_upper` | 0.0116 | largest SE-calibration-excess bound, against 0.05 |
| `properties[double_robustness/outcome_correct]:bias` | 0.000149 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0196 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3348 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5053 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5384 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4737 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5067 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0077 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9738 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0450 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9550 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/stacked_cvtmle]:coverage` | 0.8950 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/stacked_cvtmle]:se_ratio` | 0.9880 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6500 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5792 | SE calibration of that control |
| `properties[crossfit_overfitting/stacked_cvtmle]:coverage_gain_ci_lower` | 0.1875 | paired 99% lower bound for coverage gained over the control |
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
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling the in-sample control's upper bound must stay below |
| `margin:overfit_coverage_gain` | 0.1500 | coverage cross-fitting must buy over the in-sample control |

## Limitations

| limitation | what it means for use |
| --- | --- |
| PAF is compared on different native scales | `cleverly` reports its fraction-scale influence curve and `tmle3` transforms a negative-log-complement interval. Point performance and coverage are compared. Raw standard errors and finite-sample endpoints on those scales are not declared equivalent |
| The cross-fit overfitting cells are relative evidence | Held-out predictions must restore the SE ratio to its band, the in-sample control's upper bound must stay below 0.75, and the paired coverage gain must clear its floor. The measured cross-fitted coverage is evidence of relative recovery and calibrated influence-curve scale. It is not a separate absolute coverage claim; the primary GLM study carries that gate |
| Continuous-law ATT and ATC use the most similarity margin | Both stay inside the predeclared bound, and the RMSE, coverage, and calibration bounds stay well inside their non-inferiority margins. These are the finite-sample cells that use the most evidence budget, not tuned exceptions |
| The row is bounded to one ten-fold split | It does not establish repeated or nested cross-fitting, fold-evaluated or fold-specific-epsilon CV-TMLE, simultaneous or bootstrap intervals, missing outcomes, weights, clusters, strata, multi-valued treatment, broad learner-library selection, or severe practical-positivity behaviour |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/README.md)
gives the full and smoke regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/manifest.json)
records the configuration and the provenance. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/properties.csv)
carry every published row.
