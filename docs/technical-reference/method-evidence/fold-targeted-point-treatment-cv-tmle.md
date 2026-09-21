# Fold-targeted point-treatment CV-TMLE

This study compares `cleverly` with Python `zEpid` 0.9.1 at commit `16a0f96`. Both
implementations target and evaluate inside each validation fold. `cleverly` averages fold plug-ins
at equal weight $1/V$, while zEpid takes the mean over the stacked targeted rows and therefore
size-weights folds. The two point estimates use the same weights here only because the registered
folds have the same size. Both use cross-validated influence-curve variance.

The two-fold boundary is load-bearing. Each zEpid nuisance model trains on the other split, which
is the complete validation-fold complement. More zEpid splits would train each nuisance on only
one split, while `cleverly` would train on the complete complement.

## What was compared

| setting | `cleverly` | Python `zEpid` |
| --- | --- | --- |
| construction | fixed two-fold nuisances, fold targeting, fold evaluation | `SingleCrossfitTMLE`, two splits, one partition |
| folds | equal assignment stored with every sample | native split with the identical row identities |
| outcome scale | identity, and no declaration | the same binary scale |
| nuisance learners | main-effects logistic regressions | corresponding scikit-learn logistic regressions |
| treatment bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| estimand | ATE | risk difference, which is the ATE for the binary outcome |
| intervals | pointwise 95% identity-scale Wald | pointwise 95% identity-scale Wald |
| fold variance | raw within-fold influence-curve second moment | within-fold sample variance with `ddof=1` |

The reference runner checks every native split before it fits the nuisances. A changed row
identity aborts the complete run. The study reports no zEpid RR or OR because those are different
fold functionals and `cleverly` refuses them under fold evaluation.

The supplied split reads neither the treatment nor the outcome. zEpid draws it from the row
positions alone, and the `cleverly` fit records the same unstratified plan. The law is binary, so
the outcome scaler is the identity on both sides and neither implementation declares a bound.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `ate` | average treatment effect | `cleverly` fold-targeted CV-TMLE | -0.0012 to 0.0027 | 0.9569 | 1.0030 | pass |
| binary-outcome law | `ate` | average treatment effect | Python `zEpid` single-crossfit TMLE | -0.0012 to 0.0027 | 0.9575 | 1.0039 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `ate` | average treatment effect | 3.689e-07 | 0.000081 | 1.0000 | -0.000625 | 0.0019 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

The property cells draw their own two folds rather than taking the primary row's supplied
partition. Each one samples an outcome in the open interval 0 to 1, drawn as a beta variate
around the law's conditional mean. A proportion has a known support, so each cell declares
`q_bounds` of 0 to 1 and the outcome scaler is the identity. Each law is a bounded twin of the
Gaussian law the ordinary point-treatment row samples, and it keeps that law's treatment
mechanism unchanged. `tests/studies/bounded_cv_laws.py` records the pilot measurement behind
every declared constant.

The double-robustness cells use a bounded nonlinear confounded law with exact ATE 0.2368749. Its
treatment mechanism stays between 0.076 and 0.924, so the configured bounds do not clip it. Its
conditional mean stays between 0.088 and 0.924 on the quadrature grid, so both beta shapes stay
away from zero.

The wrong main-effects outcome regression imposes a constant contrast, while the true contrast
varies with `W1` and `I(W2 > 0)`. The treatment-correct cell uses n = 2,000. The other three
cells use n = 700. Each cell uses 1,200 replications and the existing predeclared margin.

The root-n ladder and the calibration cell sample a bounded linear law with exact ATE 0.1072047.
A quasibinomial solver recovers that law's six coefficients to 1.4e-15 on a quadrature grid, so
both nuisances are correctly specified there.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `fold_targeted_cvtmle` | positive | fold-targeted CV-TMLE with a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9149 to 1.1134 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.4453 to 0.5399 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000576 to 0.0011, margin 0.0029, SE ratio 0.9647 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0361 to -0.0337, margin 0.0041, SE ratio 0.9891 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000865 to 0.000707, margin 0.0026, SE ratio 1.0572 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0011 to 0.000388, margin 0.0025, SE ratio 0.9687 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9365 to 0.9600, SE ratio 0.9633 to 1.0371 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000085, coverage 0.9179 to 0.9616, SE ratio 0.9720 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000140, coverage 0.9297 to 0.9698, SE ratio 1.0009 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000054, coverage 0.9223 to 0.9647, SE ratio 1.0018 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5446 to -0.4823 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5146 to -0.5116 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0400, 0.0191 to 0.0725 | pass |
<!-- /generated -->

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else comes from the
committed results and is checked at the precision shown.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 2 | implementation-estimand tests against truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 1 | paired implementation-estimand tests |
| `paired_tests_passed` | 1 | of those, passing |
| `property_cells_total` | 14 | repeated-sampling property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0256 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9569 | lowest measured primary-study coverage |
| `max_margin_utilization` | 0.000081 | largest share of a paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0000 | largest one-sided RMSE-ratio bound |
| `min_coverage_difference_lower` | -0.0025 | smallest one-sided coverage-difference bound |
| `max_calibration_excess_upper` | 0.000966 | largest SE-calibration-excess bound |
| `properties[double_robustness/outcome_correct]:bias` | -0.000079 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.000355 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.0349 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5134 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/reported_se]:slope` | -0.5131 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 0.9984 | SE calibration where both nuisances are correct |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0400 | rejection under the confounded sharp null |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/fold_targeted_cvtmle]:coverage` | 0.9175 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/fold_targeted_cvtmle]:se_ratio` | 1.0037 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.4875 | coverage with the in-sample tree control |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.4874 | SE calibration of that control |
| `properties[crossfit_overfitting/fold_targeted_cvtmle]:coverage_gain_ci_lower` | 0.3575 | paired lower bound for coverage gained over the control |
| `margin:confidence_level` | 0.9900 | confidence level for Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal size of the estimator intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage of those intervals |
| `margin:bootstrap_replicates` | 10000 | resamples for each bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias margin in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | lower-endpoint coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | conservative-coverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper limit |
| `margin:type_i_ceiling` | 0.1000 | one-sided type-I ceiling |
| `margin:paired_difference` | 0.1500 | paired similarity margin |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio non-inferiority margin |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference non-inferiority margin |
| `margin:calibration_noninferiority` | 0.0500 | SE-calibration non-inferiority margin |
| `margin:minimum_power` | 0.8000 | rejection lower bound for power |
| `margin:root_n_slope` | -0.5000 | predicted root-n contraction rate |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio lower screen |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio upper screen |
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling for the in-sample control |
| `margin:overfit_coverage_gain` | 0.1500 | required coverage gain over the control |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The variance formulas differ at finite fold size | `cleverly` uses the raw within-fold influence-curve second moment. zEpid uses the within-fold sample variance with `ddof=1`, on rows centred on the overall estimate rather than on the fold mean. The `ddof=1` divisor dominates the gap, and it raises the zEpid variance by one part in the fold size. Fold targeting drives each fold's own score to zero, so recentring shifts only the plug-in contrast and contributes an order of magnitude less. The two terms do not cancel, and `equivalence.csv` reports the measured `se_ratio_difference`. The row does not claim the formulas are identical |
| The row reports one estimand on one binary law | zEpid does not report treatment-specific means, ATT, or ATC. The row does not claim continuous-outcome parity |
| RR and OR remain refused | zEpid pools targeted risks for those ratios. Fold evaluation changes their gradients by fold and needs another targeting score in `cleverly` |
| The comparison uses one fixed unstratified two-fold partition | It does not establish repeat aggregation, more than two zEpid splits, nested cross-fitting, or fold stability |
| The property evidence needs a known outcome support | Every property cell draws a proportion and declares `q_bounds` of 0 to 1. Those cells say nothing about a continuous outcome whose support the analyst does not know, because the package refuses that composition under cross-fitting |
| The primary comparison uses main-effects logistic learners | The property study adds a flexible-tree control. Neither study covers broad learner libraries or practical-positivity stress |
| The row excludes additional compositions | It does not cover missing outcomes, weights, clusters, strata, multi-valued treatment, or simultaneous and bootstrap intervals |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/zepid_cvtmle/README.md)
gives the smoke and full commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/zepid_cvtmle/manifest.json)
records the configuration, dependency pins, seeds, and hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/zepid_cvtmle/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/zepid_cvtmle/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/zepid_cvtmle/properties.csv)
carry every published row.
