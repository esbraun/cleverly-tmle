# Canonical point-treatment TMLE

This study validates ordinary, non-cross-fitted point-treatment TMLE in `cleverly`. The canonical
comparison uses R [`tmle3`](https://github.com/tlverse/tmle3) 0.2.0 at pinned commit
[`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27). The
independent claims come from the parameter oracles and from the properties in van der Laan and
Rubin's [original TMLE paper](https://doi.org/10.2202/1557-4679.1043).

The row is deliberately named **ordinary** TMLE. The public default cross-fits its nuisances. The
comparison disables cross-fitting in `cleverly` because `tmle3`'s ordinary specs are not CV-TMLE.
Cross-fitted and CV-TMLE constructions are separate methods and do not inherit this result.

## What was compared

| setting | `cleverly` | R `tmle3` |
| --- | --- | --- |
| estimator | ordinary TMLE, cross-fitting disabled | ordinary `tmle3` spec |
| datasets | generated in Python | the identical rows and all DGP covariates |
| nuisance regressions | GLM | corresponding GLM |
| treatment contrast | 1 versus 0 | 1 versus 0 |
| overlap | comfortable, no active bound | the same law |
| intervals | pointwise 95% Wald | pointwise 95% Wald |
| RR and OR scale | log, exponentiated | log, exponentiated |
| PAF scale | identity, from the PAF influence curve | negative-log-complement, transformed by `1 - exp(-x)` |
| ATT and ATC updater | default | the constrained one-dimensional updater the package's own tests exercise |

Both laws are point-treatment laws with complete outcomes. The bounded continuous-outcome law with
effect modification covers `ey1`, `ey0`, `ate`, `att`, `atc`, `ey_obs`, and `par`. The
binary-outcome law covers those plus `paf`, `rr`, and `or`.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` | -0.0026 to 0.0013 | 0.9506 | 0.9997 | pass |
| binary-outcome law | `atc` | average effect on the untreated | R `tmle3` | -0.0026 to 0.0013 | 0.9544 | 1.0049 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` | -0.0026 to 0.0012 | 0.9494 | 1.0067 | pass |
| binary-outcome law | `ate` | average treatment effect | R `tmle3` | -0.0026 to 0.0012 | 0.9487 | 1.0066 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` | -0.0028 to 0.0012 | 0.9506 | 1.0088 | pass |
| binary-outcome law | `att` | average effect on the treated | R `tmle3` | -0.0028 to 0.0012 | 0.9500 | 1.0141 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.000802 to 0.0020 | 0.9531 | 1.0102 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | R `tmle3` | -0.000802 to 0.0020 | 0.9531 | 1.0101 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` | -0.0016 to 0.0013 | 0.9425 | 0.9861 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | R `tmle3` | -0.0016 to 0.0013 | 0.9425 | 0.9861 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | `cleverly` | -0.000858 to 0.0012 | 0.9450 | 0.9919 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | R `tmle3` | -0.000858 to 0.0012 | 0.9450 | 0.9919 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | `cleverly` | -0.0095 to 0.0064 | 0.9500 | 1.0074 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | R `tmle3` | -0.0095 to 0.0064 | 0.9500 | 1.0074 | pass |
| binary-outcome law | `paf` | population attributable fraction | `cleverly` | -0.0030 to 0.0013 | 0.9494 | 1.0118 | pass |
| binary-outcome law | `paf` | population attributable fraction | R `tmle3` | -0.0029 to 0.0026 | 0.9487 | 1.0114 | pass |
| binary-outcome law | `par` | population attributable risk | `cleverly` | -0.0014 to 0.000586 | 0.9469 | 1.0082 | pass |
| binary-outcome law | `par` | population attributable risk | R `tmle3` | -0.0014 to 0.000587 | 0.9469 | 1.0081 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | `cleverly` | -0.0053 to 0.0033 | 0.9556 | 1.0117 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | R `tmle3` | -0.0053 to 0.0033 | 0.9556 | 1.0116 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` | -0.000400 to 0.000508 | 0.9463 | 0.9893 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | R `tmle3` | 0.000387 to 0.0013 | 0.9475 | 0.9957 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` | -0.000459 to 0.000392 | 0.9481 | 1.0015 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | R `tmle3` | -0.000442 to 0.000406 | 0.9500 | 1.0125 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` | -0.000602 to 0.000314 | 0.9394 | 0.9706 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | R `tmle3` | -0.0013 to -0.000431 | 0.9400 | 0.9726 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.000147 to 0.000743 | 0.9531 | 1.0146 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | R `tmle3` | -0.000147 to 0.000742 | 0.9537 | 1.0149 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` | -0.000260 to 0.000788 | 0.9519 | 1.0107 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | R `tmle3` | -0.000244 to 0.000802 | 0.9519 | 1.0122 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | `cleverly` | -0.000234 to 0.000779 | 0.9506 | 0.9908 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | R `tmle3` | -0.000234 to 0.000779 | 0.9506 | 0.9908 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | `cleverly` | -0.000327 to 0.000276 | 0.9344 | 0.9451 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | R `tmle3` | -0.000325 to 0.000277 | 0.9375 | 0.9526 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | 0.000023 | 0.0050 | 1.0066 | -0.0037 | 0.0107 vs 0.0500 | equivalent |
| binary-outcome law | `ate` | average treatment effect | -6.992e-07 | 0.000157 | 1.0000 | 0.000625 | 0.000096 vs 0.0500 | equivalent |
| binary-outcome law | `att` | average effect on the treated | -0.000022 | 0.0049 | 1.0067 | 0.000625 | 0.0115 vs 0.0500 | equivalent |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | 9.290e-08 | 0.000029 | 1.0000 | 0 | 0.000010 vs 0.0500 | equivalent |
| binary-outcome law | `ey1` | counterfactual mean under treatment | -4.085e-10 | 1.216e-07 | 1.0000 | 0 | 0.000071 vs 0.0500 | equivalent |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | -8.722e-12 | 3.654e-09 | 1.0000 | 0 | 5.378e-10 vs 0.0500 | equivalent |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | -5.723e-07 | 0.000014 | 1.0000 | 0 | 0.000006 vs 0.0500 | equivalent |
| binary-outcome law | `paf` | population attributable fraction | -6.674e-07 | 0.000132 | 1.0002 | 0.000625 | n/a | equivalent |
| binary-outcome law | `par` | population attributable risk | -2.965e-07 | 0.000126 | 1.0002 | 0 | 0.000105 vs 0.0500 | equivalent |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | -2.861e-07 | 0.000019 | 1.0000 | 0 | 0.000007 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | -0.000783 | 0.7445 | 1.0088 | -0.0012 | 0.0033 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | -0.000016 | 0.0161 | 1.0062 | -0.0019 | 0.0239 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | 0.000744 | 0.6995 | 1.0019 | -0.000625 | 0.0033 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | -6.584e-08 | 0.000064 | 1.0030 | -0.000625 | 0.0019 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | -0.000015 | 0.0122 | 1.0032 | 0 | 0.0040 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | -5.673e-13 | 4.814e-10 | 1.0000 | 0 | 1.179e-10 vs 0.0500 | equivalent |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | -0.000001 | 0.0020 | 1.0045 | -0.0031 | 0.0020 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0104 to 0.0037, margin 0.0236 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.3420 to -0.3244, margin 0.0295 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0062 to 0.0069, margin 0.0220 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias -0.0292 to -0.0072, margin 0.0369 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9420 to 0.9645, SE ratio 0.9679 to 1.0385 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000513, coverage 0.9194 to 0.9627, SE ratio 0.9777 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000770, coverage 0.9223 to 0.9647, SE ratio 0.9913 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000285, coverage 0.9297 to 0.9698, SE ratio 1.0077 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5374 to -0.4725 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4998 to -0.4974 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0325, 0.0141 to 0.0627 | pass |
<!-- /generated -->

## Measured values

Every figure this section quotes is resolved by name and checked at the precision it is printed to.
Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results. Both resolve the same way, so moving a threshold in the code changes this
table rather than leaving it asserting a rule the study never applied.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 34 | implementation-estimand tests against truth |
| `independent_tests_passed` | 34 | of those, passing |
| `paired_tests_total` | 17 | scenario-estimand cells compared with `tmle3` |
| `paired_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 12 | repeated-sampling property cells |
| `property_cells_passed` | 12 | of those, passing |
| `max_standardized_bias` | 0.1254 | largest bias, in empirical standard deviations |
| `min_coverage` | 0.9344 | lowest measured coverage of a nominal 95% interval |
| `min_coverage_ci_lower` | 0.9168 | lowest exact 99% coverage endpoint, against a floor of 0.90 |
| `min_se_ratio_ci_lower` | 0.9040 | lowest bootstrap SE-ratio endpoint, against a band of 0.80--1.20 |
| `max_se_ratio_ci_upper` | 1.0649 | highest bootstrap SE-ratio endpoint |
| `max_se_ratio_resolution` | 0.0960 | how far from 1.0 the widest SE-ratio interval still reaches |
| `max_margin_utilization` | 0.7445 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0088 | largest one-sided RMSE-ratio bound, against a margin of 1.10 |
| `min_coverage_difference_lower` | -0.0081 | smallest one-sided coverage-difference bound, against -0.025 |
| `max_calibration_excess_upper` | 0.0129 | largest comparable SE-calibration-excess bound, against 0.05 |
| `summary_cells` | 34 | summary cells over both laws and both implementations |
| `cells_with_se_ratio_below_one` | 13 | of those, reporting a standard error below the empirical spread |
| `cells_with_coverage_below_nominal` | 17 | of those, covering below 95% |
| `summary[cleverly/continuous/par]:coverage` | 0.9344 | the study's lowest coverage, cleverly |
| `summary[tmle3/continuous/par]:coverage` | 0.9375 | the same cell in `tmle3` |
| `summary[cleverly/continuous/att]:bias` | -0.000144 | continuous ATT bias against known truth, cleverly |
| `summary[tmle3/continuous/att]:bias` | -0.000888 | the same, `tmle3` |
| `summary[cleverly/continuous/atc]:bias` | 0.000054 | continuous ATC bias against known truth, cleverly |
| `summary[tmle3/continuous/atc]:bias` | 0.000837 | the same, `tmle3` |
| `continuous_summary_cells` | 14 | continuous-law summary cells, both implementations |
| `continuous_cells_with_se_ratio_below_one` | 8 | of those, reporting a standard error below the empirical spread |
| `continuous_cells_with_coverage_below_nominal` | 7 | of those, covering below 95% |
| `properties[double_robustness/both_correct]:bias` | -0.0034 | bias with both nuisances correct |
| `properties[double_robustness/outcome_correct]:bias` | 0.000306 | bias with only the outcome regression correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.0182 | bias with only the treatment mechanism correct |
| `properties[double_robustness/both_wrong]:bias` | -0.3332 | bias with both nuisances wrong; this is the negative control |
| `properties[double_robustness/both_wrong]:coverage` | 0.1967 | coverage of the same control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5045 | fitted log-log contraction rate of the sampling spread |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5374 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4725 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.4986 | the same rate for the mean reported standard error |
| `properties[root_n_and_efficiency/n_500]:se_ratio` | 0.9913 | SE calibration at n = 500 |
| `properties[root_n_and_efficiency/n_2000]:se_ratio` | 0.9777 | SE calibration at n = 2,000 |
| `properties[root_n_and_efficiency/n_8000]:se_ratio` | 1.0077 | SE calibration at n = 8,000 |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0027 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9679 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0385 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9542 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0325 | rejection under a confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0627 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under a real effect; this is the positive control |
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

## Limitations

| limitation | what it means for use |
| --- | --- |
| The SE sanity band is a screen, not a calibration proof | The widest bootstrap SE-ratio interval still reaches `max_se_ratio_resolution` from 1.0. These 34 cells cannot rule out a systematic misstatement smaller than about a tenth. The `interval_calibration` cell carries that claim instead, and it excludes exactly that misstatement |
| PAF is compared on different native scales | Point performance and coverage are compared. Raw standard-error parity is not claimed. The two intervals share a first-order delta-method limit but need not share finite-sample endpoints |
| Continuous ATT and ATC use the most paired margin | `max_margin_utilization` of it at the widest, and both stay inside. Measured against known truth, `cleverly`'s absolute bias on those two estimands is an order of magnitude smaller than the constrained R path's, with essentially unchanged RMSE and coverage. The gate stays asymmetric so a future change reversing this fails the suite |
| The lowest coverage is continuous-law PAR | Both implementations sit low together and both clear the declared floor. What remains is a property of ordinary non-cross-fitted TMLE on this law, not of either implementation. The `root_n_and_efficiency` cells are the discriminator: a gap that shrinks with `n` is finite-sample noise, and one that persisted would be a wrong variance formula |
| Interval calibration shows no systematic direction | The reported standard error falls below the empirical spread in `cells_with_se_ratio_below_one` of `summary_cells` cells and above it in the rest. An earlier run at a quarter of the replication count had every continuous-law cell below one and was read as systematic understatement. Raising the replication count dissolved it |
| The row is bounded to ordinary TMLE on two laws | It does not establish cross-fitting, repeated cross-fitting, CV-TMLE evaluation, simultaneous intervals, bootstrap inference, missing outcomes, weights, clustering, strata, multi-valued treatment, or flexible learners. Each is a separate method or composition and needs its own study |

## Reproduction

The container pins R 4.5.2 by digest, `tmle3` at `ed72f8a`, and `sl3` at
[`0e8f236`](https://github.com/tlverse/sl3/tree/0e8f2365bcbe54010b8120c04a7a2dcfc8119227). The
[fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/README.md)
gives the regeneration commands.
[`manifest.json`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/manifest.json)
records the configuration and the provenance. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/replicates.csv.gz),
[performance tests](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/performance-tests.csv),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/properties.csv)
carry every row the tables above publish.

R uses its public specifications except for ATT and ATC. The public ATT convenience path fails to
converge on a small fraction of bounded-continuous samples, so both use the constrained path the
pinned package's own tests exercise. That path is fixed for all replications. The R side aborts on
any failed replication rather than dropping it, and the property study refuses to summarize a run
that lost one.
