# Outcome-adaptive point-treatment C-TMLE

This study validates `CTMLE(strategy="oat")`, whose treatment mechanism is fitted on the vector of
arm-specific outcome-regression predictions rather than on a selected subset of the original
covariates. The canonical comparison uses archived tlverse
[`ctmle3`](https://github.com/tlverse/ctmle3) 0.1.0 at commit
[`a4ea77b`](https://github.com/tlverse/ctmle3/tree/a4ea77b07747dfee9b2eecb9cbca88262e0559ea), with
contemporaneous `tmle3` at
[`3a61005`](https://github.com/tlverse/tmle3/tree/3a610058cd89c17bb417c15fc891254388787f33) and
`sl3` at
[`821ca89`](https://github.com/tlverse/sl3/tree/821ca890cb8701fdb59f823e28c6356e50d092bc). The
theory is Ju et al. (2019).

## What was compared

| setting | `cleverly` | R `ctmle3` |
| --- | --- | --- |
| construction | outcome-adaptive, non-cross-fitted | the archived non-cross-fitted OAT construction |
| datasets | binary samples generated in Python with exact truths | the identical rows |
| outcome regression | three-covariate logistic GLM | the same three-covariate logistic GLM |
| estimands | both treatment-specific means, ATE, marginal RR, marginal OR | the same |
| ratio scale | log, delta-method curves | log, delta-method curves |
| propensity bounds | 0.025 to 0.975 | no truncation of its own |

The bound difference is declared rather than hidden. On this law the true propensity spans 0.085 to
0.904, so the bound is never active and no paired comparison is affected. The manifest's `g_bounds`
entry therefore describes the subject's setting and not a shared one.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `ate` | average treatment effect | `cleverly` outcome-adaptive C-TMLE | -0.0031 to 0.0024 | 0.9425 | 0.9787 | pass |
| binary-outcome law | `ate` | average treatment effect | R `ctmle3` | -0.0032 to 0.0024 | 0.9425 | 0.9785 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` outcome-adaptive C-TMLE | -0.000870 to 0.0031 | 0.9625 | 1.0049 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | R `ctmle3` | -0.000869 to 0.0031 | 0.9625 | 1.0049 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` outcome-adaptive C-TMLE | -0.0013 to 0.0028 | 0.9475 | 0.9814 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | R `ctmle3` | -0.0013 to 0.0028 | 0.9475 | 0.9813 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | `cleverly` outcome-adaptive C-TMLE | -0.0117 to 0.0115 | 0.9437 | 0.9808 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | R `ctmle3` | -0.0117 to 0.0114 | 0.9437 | 0.9806 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | `cleverly` outcome-adaptive C-TMLE | -0.0070 to 0.0055 | 0.9475 | 0.9865 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | R `ctmle3` | -0.0070 to 0.0055 | 0.9463 | 0.9864 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `ate` | average treatment effect | 0.000005 | 0.0011 | 1.0003 | 0 | 0.000826 vs 0.0500 | equivalent |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | 3.839e-08 | 0.000012 | 1.0005 | 0 | 0.000388 vs 0.0500 | equivalent |
| binary-outcome law | `ey1` | counterfactual mean under treatment | 0.000005 | 0.0016 | 1.0002 | 0 | 0.000605 vs 0.0500 | equivalent |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | 0.000043 | 0.0011 | 1.0004 | 0 | 0.000804 vs 0.0500 | equivalent |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | 0.000014 | 0.000940 | 1.0004 | 0.0012 | 0.000755 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_oat` | positive | outcome-adaptive C-TMLE with cross-fitted nuisances and a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9826 to 1.1711 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.5116 to 0.6100 | pass |
| `generated_design` | `estimated` | control | the same design is estimated from the data, as a real fit does | the SE-ratio deficit must reach the declared shortfall | SE ratio 0.9042 to 1.0047 | pass |
| `generated_design` | `oracle_design` | positive | the outcome-adaptive design is supplied rather than estimated | SE ratio interval inside the calibration band | SE ratio 0.9396 to 1.0430 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9337 to 0.9578, SE ratio 0.9506 to 1.0250 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `robustness_contract` | `outcome_correct` | positive | the outcome regression is correct and the mechanism is not | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.0048 to 0.0075, margin 0.0206, SE ratio 0.9776 | pass |
| `robustness_contract` | `outcome_wrong` | control | the outcome regression is misspecified | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias -0.3084 to -0.2904, margin 0.0302, SE ratio 0.9363 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0027, coverage 0.9136 to 0.9585, SE ratio 0.9667 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0019, coverage 0.9194 to 0.9627, SE ratio 0.9687 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0011, coverage 0.9238 to 0.9657, SE ratio 0.9542 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5287 to -0.4651 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5033 to -0.5011 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0725, 0.0509 to 0.0994 | pass |
<!-- /generated -->

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications of the binary law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 10 | implementation-estimand tests against truth |
| `independent_tests_passed` | 10 | of those, passing |
| `paired_tests_total` | 5 | paired estimand tests |
| `paired_tests_passed` | 5 | of those, passing |
| `property_cells_total` | 14 | independent property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0513 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9425 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9179 | lowest exact 99% coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9211 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0769 | highest bootstrap SE-ratio endpoint |
| `max_margin_utilization` | 0.0016 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0005 | largest paired RMSE-ratio upper bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference lower bound |
| `max_calibration_excess_upper` | 0.000592 | largest paired SE-calibration-excess upper bound |
| `properties[robustness_contract/outcome_correct]:standardized_bias` | 0.0166 | outcome-correct OAT bias |
| `properties[robustness_contract/outcome_wrong]:standardized_bias` | -2.4768 | outcome-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5287 | 99% lower endpoint of the empirical-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4651 | its upper endpoint, which must also exclude -0.25 |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4967 | fitted empirical-spread rate |
| `properties[root_n_rate/reported_se]:slope` | -0.5022 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9467 | calibration-cell coverage |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 0.9870 | calibration-cell SE ratio |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9506 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0250 | its 99% upper endpoint |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0725 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0994 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[type_i_error/sharp_null]:coverage_ci_lower` | 0.9006 | its exact 99% coverage endpoint, against a floor of 0.90 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the power control |
| `properties[crossfit_overfitting/cross_fitted_oat]:coverage` | 0.9325 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/cross_fitted_oat]:standardized_bias` | -0.4361 | that arm's bias, in empirical standard deviations |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6225 | coverage with the in-sample tree control |
| `properties[crossfit_overfitting/cross_fitted_oat]:se_ratio` | 1.0681 | SE ratio with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.5565 | SE ratio without cross-fitting |
| `properties[crossfit_overfitting/cross_fitted_oat]:coverage_gain_ci_lower` | 0.2525 | paired 99% lower bound for coverage gained by cross-fitting |
| `properties[generated_design/oracle_design]:se_ratio` | 0.9892 | SE ratio with the design pinned at the truth |
| `properties[generated_design/oracle_design]:se_ratio_ci_lower` | 0.9396 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[generated_design/oracle_design]:se_ratio_ci_upper` | 1.0430 | its 99% upper endpoint |
| `properties[generated_design/estimated]:se_ratio` | 0.9525 | the same ratio with the design estimated |
| `properties[generated_design/estimated]:se_ratio_ci_lower` | 0.9042 | its 99% lower endpoint |
| `properties[generated_design/estimated]:se_ratio_ci_upper` | 1.0047 | its 99% upper endpoint |
| `properties[generated_design/estimated]:se_ratio_deficit_lower` | -0.0546 | paired 99% lower endpoint for estimated minus pinned |
| `properties[generated_design/estimated]:se_ratio_deficit_upper` | -0.0191 | its upper endpoint, which must clear the floor below |
| `margin:confidence_level` | 0.9900 | confidence level of Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples per bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical standard deviations |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:coverage_floor` | 0.9000 | validity floor for the exact coverage lower endpoint |
| `margin:over_coverage_ceiling` | 0.9900 | coverage above this is labeled conservative |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | one-sided type-I error ceiling |
| `margin:paired_difference` | 0.1500 | paired mean-difference margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | paired RMSE-ratio upper limit |
| `margin:coverage_noninferiority` | -0.0250 | paired coverage-difference lower limit |
| `margin:calibration_noninferiority` | 0.0500 | paired calibration-excess upper limit |
| `margin:minimum_power` | 0.8000 | power-control rejection lower bound |
| `margin:root_n_slope` | -0.5000 | predicted root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:overfit_se_floor` | 0.8500 | cross-fit SE-ratio lower limit |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample SE-ratio upper limit |
| `margin:overfit_coverage_gain` | 0.1500 | cross-fit coverage-gain lower limit |
| `margin:generated_design_deficit` | 0.0100 | smallest paired SE-ratio deficit the control must establish |

## Limitations

| limitation | what it means for use |
| --- | --- |
| OAT has a narrower robustness contract than selector C-TMLE | With the outcome regression correct, the bias interval must fit inside the equivalence margin. With it wrong, the control must be discriminated outside it. No treatment-correct-only claim is made, because OAT's mechanism is a projection on the generated outcome-regression design rather than a fit of treatment on the original covariates |
| The reported interval omits the adaptive-`g` term | OAT fits the treatment mechanism on `Qbar`, so when `Qbar` is estimated the model class `g` is chosen from is random too, and the influence curve does not see that. The `generated_design` cells measure the consequence: with the design pinned the SE ratio is `properties[generated_design/oracle_design]:se_ratio`, and with it estimated `properties[generated_design/estimated]:se_ratio` |
| Neither design cell's interval on its own excludes 1 | An SE ratio's Monte Carlo error is dominated by the empirical spread in its denominator, worth about two percent at these replication counts. The *paired* difference resolves, because the two cells share their draws and that common error cancels. It runs from `properties[generated_design/estimated]:se_ratio_deficit_lower` to `properties[generated_design/estimated]:se_ratio_deficit_upper`, entirely below zero. The omission is worth a few percent of a reported standard error and does not show up as invalid coverage |
| The design control's margin is a floor on a defect, not a tolerance | If the reported covariance is ever made to carry this term, the control stops being discriminated and this row goes red. That is the correct signal that the limitation has gone stale, rather than a regression |
| The cross-fit overfitting cells are relative evidence | A fully grown tree on this law carries `properties[crossfit_overfitting/cross_fitted_oat]:standardized_bias` empirical standard deviations of nuisance bias. The cell is gated on its SE ratio and on the paired gain rather than on the coverage floor. The primary GLM study carries the absolute gate |
| The parity claim is narrow | It is binary, two-arm, complete-outcome, GLM, and non-cross-fitted. The archived stack fails the analogous continuous law because its length-two outcome bounds enter a scalar `if` condition; the runner treats that as a reference limitation and not a dropped replication. The row does not establish continuous or multi-arm parity, missing outcomes, weights, clusters, strata, simultaneous or bootstrap intervals, broad learner libraries, or severe practical-positivity behaviour. Cross-fitted public behaviour rests on the property study |

A nonparametric bootstrap reruns the whole construction and so carries the omitted terms. It was
measured on this law and did not improve calibration over the reported interval, so it is not
presented here as a remedy.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/README.md)
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/manifest.json)
record the provenance and the regeneration commands. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle3_oat/properties.csv)
carry every published row.
