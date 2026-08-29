# Clustered point-treatment CV-TMLE

This study validates cluster-robust inference for cross-fitted point-treatment TMLE. It uses the
existing continuous-outcome law with ten observations per cluster. A shared hidden effect modifies
the treatment effect without changing treatment assignment. The population ATE is therefore one.

## What was compared

| setting | `cleverly` | R `lmtp` 1.5.4 |
| --- | --- | --- |
| construction | stacked point-treatment CV-TMLE | `lmtp_tmle` internals with the supplied folds |
| folds | five treatment-stratified grouped folds | the identical rowwise assignment |
| treatment mechanism | exact propensity from the law | the identical exact density ratio |
| outcome regression | linear regression | `SL.glm` for a continuous outcome |
| independent unit | cluster identifier | the same identifier passed to `LmtpTask` and `ife` |
| ATE inference | joint difference influence curve | subtraction of the two `ife` arm objects |
| intervals | pointwise 95% Wald | pointwise 95% Wald |

Both runners reject a fold assignment that splits a cluster. The Python runner writes its realized
assignment beside each sampled row. The R adapter retains that assignment without rebuilding it.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ate` | average treatment effect | `cleverly` clustered point-treatment CV-TMLE | -0.0249 to 0.0109 | 0.9413 | 0.9720 | pass |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ate` | average treatment effect | R `lmtp` | -0.0251 to 0.0107 | 0.9413 | 0.9725 | pass |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` clustered point-treatment CV-TMLE | -0.0154 to 0.0063 | 0.9463 | 1.0014 | pass |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ey0` | counterfactual mean under no treatment | R `lmtp` | -0.0153 to 0.0063 | 0.9437 | 1.0021 | pass |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ey1` | counterfactual mean under treatment | `cleverly` clustered point-treatment CV-TMLE | -0.0362 to 0.0132 | 0.9437 | 0.9789 | pass |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ey1` | counterfactual mean under treatment | R `lmtp` | -0.0363 to 0.0130 | 0.9450 | 0.9795 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ate` | average treatment effect | 0.000170 | 0.0058 | 1.0017 | 0 | 0.0014 vs 0.0500 | equivalent |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ey0` | counterfactual mean under no treatment | -0.000029 | 0.0016 | 1.0020 | 0.0025 | 0.0026 vs 0.0500 | equivalent |
| continuous-outcome law with ten rows per cluster and shared effect modification | `ey1` | counterfactual mean under treatment | 0.000141 | 0.0035 | 1.0012 | -0.0012 | 0.000807 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `clustered_inference` | `cluster_robust` | positive | five-fold point-treatment TMLE with cluster-robust ATE inference | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9291 to 0.9541, SE ratio 0.9570 to 1.0310, paired coverage gain 0.0883 to 0.1200 | pass |
| `clustered_inference` | `iid_control` | control | the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.8185 to 0.8576, SE ratio 0.6850 to 0.7384, paired coverage gain 0.0883 to 0.1200 | pass |
<!-- /generated -->

The property study fits each sample once. Its IID control reuses the same rows, ATE estimate, and
rowwise influence curve. Only the variance aggregation changes. This pairing isolates the cluster
covariance calculation from nuisance fitting and targeting.

## Measured values

Names beginning `margin:` are thresholds declared before the run. The remaining values come from
the committed artifacts. The documentation gate checks every printed value.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 6 | implementation-estimand truth tests |
| `independent_tests_passed` | 6 | truth tests passing |
| `paired_tests_total` | 3 | paired implementation tests |
| `paired_tests_passed` | 3 | paired tests passing |
| `property_cells_total` | 2 | clustered-inference cells |
| `property_cells_passed` | 2 | property cells passing |
| `max_standardized_bias` | 0.0431 | largest absolute standardized bias |
| `min_coverage` | 0.9413 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9165 | lowest primary 99% coverage endpoint |
| `max_margin_utilization` | 0.0058 | largest share of the paired margin used |
| `properties[clustered_inference/cluster_robust]:coverage` | 0.9425 | cluster-robust ATE coverage |
| `properties[clustered_inference/cluster_robust]:se_ratio` | 0.9924 | cluster-robust ATE SE ratio |
| `properties[clustered_inference/cluster_robust]:se_ratio_ci_lower` | 0.9570 | cluster-robust SE-ratio lower endpoint |
| `properties[clustered_inference/cluster_robust]:se_ratio_ci_upper` | 1.0310 | cluster-robust SE-ratio upper endpoint |
| `properties[clustered_inference/iid_control]:coverage` | 0.8387 | IID-control ATE coverage |
| `properties[clustered_inference/iid_control]:se_ratio` | 0.7107 | IID-control ATE SE ratio |
| `properties[clustered_inference/iid_control]:se_ratio_ci_upper` | 0.7384 | IID-control SE-ratio upper endpoint |
| `properties[clustered_inference/cluster_robust]:coverage_gain_ci_lower` | 0.0883 | paired coverage-gain lower endpoint |
| `margin:confidence_level` | 0.9900 | confidence level for Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the exact coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | cluster-robust SE-ratio lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | cluster-robust SE-ratio upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | cluster-robust coverage lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | cluster-robust coverage upper limit |
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
| `margin:iid_control_se_ceiling` | 0.8000 | IID-control SE-ratio ceiling |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |

## Limitations

| limitation | what it means for use |
| --- | --- |
| One cluster size and one dependence law | The row validates clusters of ten under shared effect modification. It does not cover informative cluster size |
| Continuous outcome and binary treatment | The row does not establish binary outcomes, multi-valued treatments, missing outcomes, or longitudinal treatment |
| One five-fold split | The row does not establish repeated, fold-evaluated, or fold-specific targeting |
| Exact treatment mechanism | The comparison isolates targeting and inference. It does not compare learned propensity models |
| Pointwise identity-scale intervals | The row does not cover simultaneous bands, bootstrap intervals, ratios, weights, or strata |
| Linear outcome learners | The row does not establish flexible learner-library parity under clustering |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_clustered_tmle/README.md)
gives smoke and full commands. The manifest records the container, runner, adapter, harness,
configuration, and result-determining Python modules.
