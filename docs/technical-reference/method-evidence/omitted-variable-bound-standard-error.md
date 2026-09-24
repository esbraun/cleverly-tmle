# Omitted-variable bound standard error

This study validates the standard error of the one-sided limits of the omitted-variable bound.
[Theorem 4 in Section 4 of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis
(2026)](../validation-methods.md#standard-error-of-the-omitted-variable-bound) gives those limits.
For the ATT and the ATC, the curve of $\nu^2$ carries the influence term of the estimated share of
the conditioning arm, which RM22 added. The study reads each bound as an estimate of a known
population value. It reads the bound's standard error back off the reported limit.

**No canonical implementation is compared.** DoubleML omits the share term. The R package
`dml.sensemakr` carries it, but it estimates the share out of fold and divides the point estimate
of $\nu^2$ as well, so its bounds are a different estimate. The zero-row equivalence artifact
records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law | `make_linear_ate(n=1000)`. Four standard normal covariates, the treatment mechanism $\operatorname{expit}(0.3 W_1 - 0.2 W_2 + 0.1 W_3)$, standard normal outcome noise, and a constant effect of 1.5 |
| fit | in-sample TMLE with a linear outcome regression and an unpenalized main-effects logistic treatment model. Both are correctly specified. The fit reports the ATE, the ATT and the ATC |
| estimator of $\nu^2$ | the default, which resolves to the doubly robust estimator |
| strength | $c_Y = 0.5$, $c_D = 0.3$, $\rho = 1$ |
| primary estimands | the lower and the upper bound of the ATT, the ATC and the ATE |
| primary standard error | `(lower - ci_lower) / z_0.95` for a lower bound, and `(ci_upper - upper) / z_0.95` for an upper bound |
| primary interval | the bound plus or minus 1.959964 standard errors |
| truth | closed form. $\nu^2$ is $4 e^{0.07}$ for the ATT and the ATC and $2 + 2 e^{0.07}$ for the ATE, and $\sigma^2 = 1$ |
| control | the same fits, with the standard error from the curve of $\nu^2$ without the share term. That is the curve before RM22 |
| Monte Carlo inference | 99% intervals around every declared endpoint |

The RM22 plan in the [roadmap](../../roadmap.md#rm22-standard-error-of-the-omitted-variable-bound)
fixed the law, the budget, the seeds, the margins and the red-cell route before any run.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear Gaussian-outcome law with a constant effect, `make_linear_ate` | `atc_lower` | lower omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.3, rho 1 | `cleverly` omitted-variable bound | -0.0025 to 0.0011 | 0.9511 | 1.0026 | pass |
| linear Gaussian-outcome law with a constant effect, `make_linear_ate` | `atc_upper` | upper omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.3, rho 1 | `cleverly` omitted-variable bound | 0.0015 to 0.0052 | 0.9482 | 0.9965 | pass |
| linear Gaussian-outcome law with a constant effect, `make_linear_ate` | `ate_lower` | lower omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.3, rho 1 | `cleverly` omitted-variable bound | -0.000286 to 0.0032 | 0.9507 | 1.0018 | pass |
| linear Gaussian-outcome law with a constant effect, `make_linear_ate` | `ate_upper` | upper omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.3, rho 1 | `cleverly` omitted-variable bound | -0.000631 to 0.0029 | 0.9487 | 0.9946 | pass |
| linear Gaussian-outcome law with a constant effect, `make_linear_ate` | `att_lower` | lower omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.3, rho 1 | `cleverly` omitted-variable bound | -0.0024 to 0.0012 | 0.9530 | 1.0042 | pass |
| linear Gaussian-outcome law with a constant effect, `make_linear_ate` | `att_upper` | upper omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.3, rho 1 | `cleverly` omitted-variable bound | 0.0013 to 0.0050 | 0.9478 | 0.9916 | pass |
<!-- /generated -->

## Theory properties

Each cell reads the same 10,000 draws, and one fit per draw gives every row. A positive cell must
put its SE-ratio interval inside the calibration band and its coverage interval inside the
coverage band. Each end of the ATT bound also has an `inflated_se_control`. The control reads the
curve without the share term, which is too wide on this law, so its SE-ratio interval must rise
above the band. The ATC has no control. The plan's probe put its ratios without the term at 1.098
and 1.111, too close to 1.07 to predict a failure.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `atc_lower__correctly_specified` | positive | lower omitted-variable bound on the ATC: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9462 to 0.9573, SE ratio 0.9909 to 1.0278 | pass |
| `interval_calibration` | `atc_upper__correctly_specified` | positive | upper omitted-variable bound on the ATC: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9454 to 0.9566, SE ratio 0.9831 to 1.0198 | pass |
| `interval_calibration` | `ate_lower__correctly_specified` | positive | lower omitted-variable bound on the ATE: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9463 to 0.9574, SE ratio 0.9883 to 1.0240 | pass |
| `interval_calibration` | `ate_upper__correctly_specified` | positive | upper omitted-variable bound on the ATE: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9554, SE ratio 0.9808 to 1.0173 | pass |
| `interval_calibration` | `att_lower__correctly_specified` | positive | lower omitted-variable bound on the ATT: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9442 to 0.9555, SE ratio 0.9852 to 1.0217 | pass |
| `interval_calibration` | `att_lower__inflated_se_control` | control | lower omitted-variable bound on the ATT: the standard errors come from the curve of nu^2 without the conditioning-share term, as before RM22 | the SE-ratio interval must fall above the calibration band | coverage 0.9632 to 0.9724, SE ratio 1.0791 to 1.1189 | pass |
| `interval_calibration` | `att_upper__correctly_specified` | positive | upper omitted-variable bound on the ATT: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9453 to 0.9565, SE ratio 0.9794 to 1.0157 | pass |
| `interval_calibration` | `att_upper__inflated_se_control` | control | upper omitted-variable bound on the ATT: the standard errors come from the curve of nu^2 without the conditioning-share term, as before RM22 | the SE-ratio interval must fall above the calibration band | coverage 0.9635 to 0.9726, SE ratio 1.0732 to 1.1129 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The controls show that the calibration band
detects the curve without the share term on this law.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 10000 | primary replications |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 6 | bound-end tests against truth |
| `independent_tests_passed` | 6 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 8 | repeated-sampling property cells |
| `property_cells_passed` | 8 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0475 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9478 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9418 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9743 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0228 | highest bootstrap primary SE-ratio endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. A control must rise above it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | Every declared cell is green. The `reporting` policy does not assert that, so the fast tier recomputes each verdict and does not fail on a red one |
| There is no cross-implementation evidence | No maintained implementation computes the same bounds with the same curve |
| The fit is in sample, with correctly specified GLMs | The row does not cover cross-fitting, a flexible learner, or a misspecified nuisance |
| One law, one sample size and one strength | The row covers `make_linear_ate` at $n = 1000$ and $c_Y = 0.5$, $c_D = 0.3$, $\rho = 1$. At the default strength of 0.03 the share term moves no ratio by 0.001 |
| Two arms, no weights and no clusters | The exact-law witness covers three arms, weights and clusters. This row does not |
| Only the doubly robust estimator | The plug-in estimator reports no limits, so the row says nothing about it |
| The ATC has no control | The ATC's positive cells pass, and no control shows that the band would detect its omission |
| Point treatment only | The bound refuses longitudinal fits, so the row covers none |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/omitted_variable_bound/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/omitted_variable_bound/manifest.json)
records the seeds, margins, exact estimator configuration, source hashes, and result hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/omitted_variable_bound/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/omitted_variable_bound/properties.csv)
carry every published row.
