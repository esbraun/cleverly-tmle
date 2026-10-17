# MTP tuning

This study validates a modified treatment policy whose cap is tuned on the data. The policy is
`Shift(1.0, cap=c(h))` at both nodes of one law. The cap is a dose quantile (`DoseQuantile`) or a
linear quantile regression of the dose (`SupportBoundary(LinearQuantileRegression(...))`). The
[modified treatment policies at a node](../longitudinal-tmle.md#modified-treatment-policies-at-a-node)
section states the estimator and its conditions.

**Status of this page.** The study is registered, and its Python phases have run. No other
implementation fits the tuning term of a tuned cap, so the study has no comparator and no pending
reference. The publication policy is `reporting`. A red cell stays red on this page and in the
[red-cell ledger](red-cells.md), with the owner row that names it.

| part | what it is | status |
| --- | --- | --- |
| Python property evidence | exact law by quadrature, interval calibration with three controls for each positive | the tables below give the verdicts. It decides whether the method ships |
| comparator runs | none | no implementation fits a tuned cap, so the row rests on the other two columns |

## What was compared

The primary phase draws 1,000 samples of 2,000 rows for each of two scenarios.

| scenario | subject |
| --- | --- |
| `population_quantile_ltmle` | `LTMLE` at one fold, population target, `DoseQuantile` caps, the oracle density on the density route |
| `population_quantile_sdr` | the same draw, `SDR` |

Law Q has two nodes. The baseline $W$ is a truncated normal, the first dose $A_1$ is normal given
$W$, the covariate $L_2$ is Bernoulli given $A_1$ and $W$, and the second dose $A_2$ is normal given
the history. A Beta law draws the outcome. The policy adds 1 to the dose, and holds the dose when
the sum passes the cap. The truth and the influence curve come from quadrature
(`tests/law_q_tuned.py`), and the implementation never enters that module.

| target | definition | what the curve adds |
| --- | --- | --- |
| population | $\Psi_0 = \Phi(\theta_0)$ at the population value of the tuner | the tuning term $\sum_t \partial_t\Phi' \mathrm{IF}_t$ |
| fold average | the mean of $\Phi(\theta^{(-k)})$ over the fold maps | none. The truth varies by replication |

The first declaration used 200 primary replications. They could not pass the bias-equivalence and
coverage-floor gates at any plausible bias, and both rows read coverage 0.935 and an SE ratio of
0.92. The second declaration sizes the primary at 1,000. The claims are the property cells.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-node law Q with a dose-quantile cap, LTMLE | `ey_regimen[tuned_cap]` | mean outcome under the plan add 1 to the dose at both times where the result stays below the tuned cap | `cleverly` LTMLE and SDR under a shift with a tuned cap | -0.0012 to 0.000719 | 0.9470 | 1.0024 | pass |
| two-node law Q with a dose-quantile cap, SDR | `ey_regimen[tuned_cap]` | mean outcome under the plan add 1 to the dose at both times where the result stays below the tuned cap | `cleverly` LTMLE and SDR under a shift with a tuned cap | -0.0012 to 0.000718 | 0.9470 | 1.0024 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

None claimed. No canonical implementation is compared, because no other implementation fits the tuning term of a tuned cap. [0/0 external comparisons; the row rests on the other two columns]

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `fold_average_dose_quantile_tau2__correctly_specified` | positive | the fold-average target of the dose-quantile cap: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9398 to 0.9535, SE ratio 0.9726 to 1.0160 | pass |
| `interval_calibration` | `fold_average_dose_quantile_tau2__noise_control` | control | the fold-average target of the dose-quantile cap: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8264 to 0.8489, SE ratio 0.6891 to 0.7193 | pass |
| `interval_calibration` | `fold_average_dose_quantile_tau2__population_truth_control` | control | the fold-average target of the dose-quantile cap: the fold-average rows are scored on the population target | the SE-ratio interval must fall below the calibration band | coverage 0.7628 to 0.7882, SE ratio 0.6024 to 0.6285 | pass |
| `interval_calibration` | `fold_average_dose_quantile_tau2__shrunken_se_control` | control | the fold-average target of the dose-quantile cap: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8187 to 0.8417, SE ratio 0.6812 to 0.7116 | pass |
| `interval_calibration` | `fold_average_quantile_regression_tau2__correctly_specified` | positive | the fold-average target of the quantile-regression cap: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9390 to 0.9529, SE ratio 0.9615 to 1.0043 | pass |
| `interval_calibration` | `fold_average_quantile_regression_tau2__noise_control` | control | the fold-average target of the quantile-regression cap: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8147 to 0.8379, SE ratio 0.6815 to 0.7122 | pass |
| `interval_calibration` | `fold_average_quantile_regression_tau2__population_truth_control` | control | the fold-average target of the quantile-regression cap: the fold-average rows are scored on the population target | the SE-ratio interval must fall below the calibration band | coverage 0.8771 to 0.8965, SE ratio 0.7968 to 0.8325 | pass |
| `interval_calibration` | `fold_average_quantile_regression_tau2__shrunken_se_control` | control | the fold-average target of the quantile-regression cap: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8093 to 0.8327, SE ratio 0.6731 to 0.7030 | pass |
| `interval_calibration` | `population_dose_quantile_crossfit_classifier_tau2__correctly_specified` | positive | the population target of the dose-quantile cap, five folds, classifier route: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9440 to 0.9573, SE ratio 0.9887 to 1.0338 | pass |
| `interval_calibration` | `population_dose_quantile_crossfit_classifier_tau2__noise_control` | control | the population target of the dose-quantile cap, five folds, classifier route: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8213 to 0.8441, SE ratio 0.6951 to 0.7252 | pass |
| `interval_calibration` | `population_dose_quantile_crossfit_classifier_tau2__shrunken_se_control` | control | the population target of the dose-quantile cap, five folds, classifier route: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8236 to 0.8462, SE ratio 0.6927 to 0.7237 | pass |
| `interval_calibration` | `population_dose_quantile_crossfit_classifier_tau2__term_dropped_control` | control | the population target of the dose-quantile cap, five folds, classifier route: the reported standard error omits the tuning term of the curve | the SE-ratio interval must fall below the calibration band | coverage 0.7622 to 0.7877, SE ratio 0.6085 to 0.6356 | pass |
| `interval_calibration` | `population_dose_quantile_crossfit_tau2__correctly_specified` | positive | the population target of the dose-quantile cap, five folds, density route: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9582, SE ratio 1.0017 to 1.0446 | pass |
| `interval_calibration` | `population_dose_quantile_crossfit_tau2__noise_control` | control | the population target of the dose-quantile cap, five folds, density route: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8293 to 0.8516, SE ratio 0.7036 to 0.7348 | pass |
| `interval_calibration` | `population_dose_quantile_crossfit_tau2__shrunken_se_control` | control | the population target of the dose-quantile cap, five folds, density route: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8269 to 0.8494, SE ratio 0.7012 to 0.7323 | pass |
| `interval_calibration` | `population_dose_quantile_crossfit_tau2__term_dropped_control` | control | the population target of the dose-quantile cap, five folds, density route: the reported standard error omits the tuning term of the curve | the SE-ratio interval must fall below the calibration band | coverage 0.7675 to 0.7927, SE ratio 0.6163 to 0.6432 | pass |
| `interval_calibration` | `population_dose_quantile_tau2__correctly_specified` | positive | the population target of the dose-quantile cap, one fold: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9564, SE ratio 0.9954 to 1.0382 | pass |
| `interval_calibration` | `population_dose_quantile_tau2__noise_control` | control | the population target of the dose-quantile cap, one fold: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8259 to 0.8484, SE ratio 0.6981 to 0.7286 | pass |
| `interval_calibration` | `population_dose_quantile_tau2__shrunken_se_control` | control | the population target of the dose-quantile cap, one fold: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8194 to 0.8423, SE ratio 0.6964 to 0.7263 | pass |
| `interval_calibration` | `population_dose_quantile_tau2__term_dropped_control` | control | the population target of the dose-quantile cap, one fold: the reported standard error omits the tuning term of the curve | the SE-ratio interval must fall below the calibration band | coverage 0.7625 to 0.7880, SE ratio 0.6123 to 0.6387 | pass |
| `interval_calibration` | `population_quantile_regression_tau2__correctly_specified` | positive | the population target of the quantile-regression cap, one fold: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9399 to 0.9537, SE ratio 0.9749 to 1.0183 | pass |
| `interval_calibration` | `population_quantile_regression_tau2__noise_control` | control | the population target of the quantile-regression cap, one fold: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8243 to 0.8469, SE ratio 0.6951 to 0.7257 | pass |
| `interval_calibration` | `population_quantile_regression_tau2__shrunken_se_control` | control | the population target of the quantile-regression cap, one fold: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8206 to 0.8434, SE ratio 0.6823 to 0.7130 | pass |
| `interval_calibration` | `population_quantile_regression_tau2__term_dropped_control` | control | the population target of the quantile-regression cap, one fold: the reported standard error omits the tuning term of the curve | the SE-ratio interval must fall below the calibration band | coverage 0.8865 to 0.9052, SE ratio 0.8098 to 0.8457 | pass |
<!-- /generated -->

| calibration positive | construction and truth |
| --- | --- |
| `population_dose_quantile_tau2` | one fold, `DoseQuantile`, oracle density. $\Psi_0$ |
| `population_quantile_regression_tau2` | one fold, `LinearQuantileRegression`. $\Psi_0$ |
| `population_dose_quantile_crossfit_tau2` | five folds, oracle density, ratio at $\theta_n$. $\Psi_0$ |
| `population_dose_quantile_crossfit_classifier_tau2` | five folds, the declared classifier at $\theta^{(-k)}$. $\Psi_0$ |
| `fold_average_dose_quantile_tau2` | five folds, fold-average target |
| `fold_average_quantile_regression_tau2` | the same with `LinearQuantileRegression` |

Each positive has a shrunken-SE control and a noise control. It also has one derived control that
removes a term which can vanish at the truth.

| control | cells | what it removes | expected SE ratio |
| --- | --- | --- | --- |
| `term_dropped_control` | population cells | the standard error of the curve without the tuning term | 0.612 (dose quantile) and 0.832 (regression) |
| `population_truth_control` | fold-average cells | the target: the same rows scored on $\Psi_0$ | the same reading |

Both controls pass when the SE ratio leaves the band below. The sizes and the pilot are those of
[`mtp-pricing`](mtp-pricing.md), because the six arms enter its 18-arm pooled rule.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_passed` | 2 | truth tests passing |
| `independent_tests_total` | 2 | truth tests reported |
| `paired_tests_passed` | 0 | paired comparisons passing |
| `paired_tests_total` | 0 | paired comparisons reported |
| `property_cells_passed` | 24 | property cells passing |
| `property_cells_total` | 24 | property cells reported |
| `max_standardized_bias` | 0.0208 | largest primary standardized bias |
| `min_coverage` | 0.9470 | lowest primary coverage |
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
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| one law with two nodes and a Beta outcome | the evidence covers one tuned-cap law. The study has no size ladder, so it claims no rate |
| the density is an oracle on the one-fold cells | the density route reads the generating density. The cross-fitted classifier cell reads a learned ratio |
| the caps are two tuners | a dose quantile and a linear quantile regression. Other tuners are outside this evidence |
| the kernel bandwidth follows a declared rule | undersmoothed for the dose quantile and Silverman for the regression, set by the relative bias of the kernel terms at n = 2,000 |
| there is no external comparator | no other implementation fits a tuned cap |
