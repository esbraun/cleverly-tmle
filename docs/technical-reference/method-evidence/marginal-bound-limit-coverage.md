# Marginal sensitivity bound limit coverage

This study measures the coverage of the confidence limits on each end of the marginal sensitivity
model bounds of X82 part H. The function `marginal_sensitivity_bounds` resolves each limit through
the evidence registry with the instrument `msm_bound_limit` and one rule. A passing cell adds one
registry row for the configuration it measured. The study reads each end of the ATE bound at
$\Gamma = 1.5$ as an estimate of the population end of the law. It also reads the reported
standard error of that end.

**No canonical implementation is compared.** The authors' code (Dorn, Guo and Kallus) is R and
fits `grf` forests, so it computes another estimator. The zero-row equivalence artifact records
the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| cells | SH1 `continuous`, SH2 `binary`. One cell for each `msm_bound_limit` rule |
| laws | `tests/studies/marginal_bound_law.py`: 20 independent standard normal covariates and a logistic propensity. SH1 has a Gaussian linear outcome. SH2 has a logistic outcome with prevalence near 0.33 |
| fit | `TMLE` with five-fold cross-fitting. The treatment learner is `LogisticRegression(C=1e6)`. The outcome learner is `LinearRegression` with `q_bounds=(-12, 12)` (SH1) or `LogisticRegression(C=1e6)` (SH2). Every nuisance is correctly specified |
| bounds | `marginal_sensitivity_bounds(result, (1.25, 1.5))` with the default quantile and tail learners |
| primary estimands | the lower and the upper end of the ATE bound at $\Gamma = 1.5$, 4 rows |
| primary interval | the end plus or minus 1.959964 reported standard errors |
| truth | the population ATE bound of the law at $\Gamma = 1.5$: the difference of the sharp arm-mean bounds, by tensor Gauss-Legendre quadrature. SH1 $[0.678, 1.322]$, SH2 $[0.0341, 0.2508]$ |
| size | 2,000 replications at $n = 8000$ per cell, and 6,800 property draws per cell |
| Monte Carlo inference | 99% intervals around every declared endpoint |

`X82-H-plan.md` rev 2.5, section 8, in the plans repository fixed the cells, the size, the seeds,
the controls and the red-cell route before any run. Pilots on seeds outside the declared streams
sized the run.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian linear outcome, 20 covariates, logistic propensity; GLMs | `ate_lower` | lower end of the ATE bound under the marginal sensitivity model at Gamma 1.5 | `cleverly` marginal sensitivity bound limits (X82 part H) | -0.0050 to -0.0022 | 0.9510 | 1.0211 | pass |
| Gaussian linear outcome, 20 covariates, logistic propensity; GLMs | `ate_upper` | upper end of the ATE bound under the marginal sensitivity model at Gamma 1.5 | `cleverly` marginal sensitivity bound limits (X82 part H) | 0.0021 to 0.0049 | 0.9495 | 1.0259 | pass |
| logistic outcome, prevalence near 0.33, 20 covariates; GLMs | `ate_lower` | lower end of the ATE bound under the marginal sensitivity model at Gamma 1.5 | `cleverly` marginal sensitivity bound limits (X82 part H) | -0.0012 to 0.000059 | 0.9510 | 1.0039 | pass |
| logistic outcome, prevalence near 0.33, 20 covariates; GLMs | `ate_upper` | upper end of the ATE bound under the marginal sensitivity model at Gamma 1.5 | `cleverly` marginal sensitivity bound limits (X82 part H) | -0.000098 to 0.0011 | 0.9495 | 0.9925 | pass |
<!-- /generated -->

An estimated quantile moves each end outward at second order. The bias is about 0.15 empirical
standard deviations in SH1 and 0.05 in SH2. Both lie inside the 0.25 margin.

## Theory properties

Each scenario's cells read the same 6,800 draws, and one fit per draw gives every row. A positive
cell must put its SE-ratio interval inside the calibration band and its coverage interval inside
the coverage band. Each row has two controls. The `inflated_se_control` multiplies the reported
standard error by 1.5, and its SE-ratio interval must rise above the band. The
`wrong_gamma_control` reads the end at $\Gamma = 1.25$ against the $\Gamma = 1.5$ truth, and its
coverage interval must fall below the band.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `SH1_ate_lower__correctly_specified` | positive | lower end of the ATE bound under the marginal sensitivity model at Gamma 1.5: every nuisance is correctly specified, and the row reads its reported standard error | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9417 to 0.9556, SE ratio 0.9834 to 1.0248 | pass |
| `interval_calibration` | `SH1_ate_lower__inflated_se_control` | control | lower end of the ATE bound under the marginal sensitivity model at Gamma 1.5: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9951 to 0.9986, SE ratio 1.4723 to 1.5387 | pass |
| `interval_calibration` | `SH1_ate_lower__wrong_gamma_control` | control | lower end of the ATE bound under the marginal sensitivity model at Gamma 1.5: the end at Gamma 1.25, read against the Gamma 1.5 truth | the coverage interval must fall below the calibration band | coverage 0 to 0.000779, SE ratio 0.9834 to 1.0275 | pass |
| `interval_calibration` | `SH1_ate_upper__correctly_specified` | positive | upper end of the ATE bound under the marginal sensitivity model at Gamma 1.5: every nuisance is correctly specified, and the row reads its reported standard error | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9466 to 0.9599, SE ratio 0.9904 to 1.0355 | pass |
| `interval_calibration` | `SH1_ate_upper__inflated_se_control` | control | upper end of the ATE bound under the marginal sensitivity model at Gamma 1.5: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9942 to 0.9980, SE ratio 1.4861 to 1.5514 | pass |
| `interval_calibration` | `SH1_ate_upper__wrong_gamma_control` | control | upper end of the ATE bound under the marginal sensitivity model at Gamma 1.5: the end at Gamma 1.25, read against the Gamma 1.5 truth | the coverage interval must fall below the calibration band | coverage 7.371e-07 to 0.0011, SE ratio 0.9881 to 1.0321 | pass |
| `interval_calibration` | `SH2_ate_lower__correctly_specified` | positive | lower end of the ATE bound under the marginal sensitivity model at Gamma 1.5: every nuisance is correctly specified, and the row reads its reported standard error | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9488 to 0.9618, SE ratio 0.9907 to 1.0346 | pass |
| `interval_calibration` | `SH2_ate_lower__inflated_se_control` | control | lower end of the ATE bound under the marginal sensitivity model at Gamma 1.5: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9959 to 0.9990, SE ratio 1.4859 to 1.5525 | pass |
| `interval_calibration` | `SH2_ate_lower__wrong_gamma_control` | control | lower end of the ATE bound under the marginal sensitivity model at Gamma 1.5: the end at Gamma 1.25, read against the Gamma 1.5 truth | the coverage interval must fall below the calibration band | coverage 0.0013 to 0.0047, SE ratio 0.9902 to 1.0334 | pass |
| `interval_calibration` | `SH2_ate_upper__correctly_specified` | positive | upper end of the ATE bound under the marginal sensitivity model at Gamma 1.5: every nuisance is correctly specified, and the row reads its reported standard error | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9461 to 0.9594, SE ratio 0.9925 to 1.0370 | pass |
| `interval_calibration` | `SH2_ate_upper__inflated_se_control` | control | upper end of the ATE bound under the marginal sensitivity model at Gamma 1.5: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9953 to 0.9987, SE ratio 1.4874 to 1.5563 | pass |
| `interval_calibration` | `SH2_ate_upper__wrong_gamma_control` | control | upper end of the ATE bound under the marginal sensitivity model at Gamma 1.5: the end at Gamma 1.25, read against the Gamma 1.5 truth | the coverage interval must fall below the calibration band | coverage 0.0039 to 0.0089, SE ratio 0.9918 to 1.0371 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The registry gains one `msm_bound_limit` row for
each rule.

| rule | covered configuration |
| --- | --- |
| `continuous` | the `ate` bound of a Gaussian fit with `LinearRegression` and `LogisticRegression(C=1e6)` nuisances, and the default quantile and tail learners |
| `binary` | the `ate` bound of a binary fit with `LogisticRegression(C=1e6)` nuisances, and at least 875 events and non-events in each arm |

Both rows also require `TMLE`, five-fold cross-fitting, influence-curve inference, no declared
mechanism, two arms, one cross-fitting draw, $n \ge 8000$, and at most 0.0025 adjustment columns
per row. The 875 floor lies between the minimum (827) and the 5th percentile (884) of the SH2
primary draws.

The set interval `[lower - z se, upper + z se]` covered the population set in 96.8% of the SH1 primary
draws and in 95.7% of the SH2 draws. This joint reading is descriptive.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications |
| `n` | 8000 | observations per primary replication |
| `independent_tests_total` | 4 | bound-end tests against truth |
| `independent_tests_passed` | 4 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 12 | repeated-sampling property cells |
| `property_cells_passed` | 12 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.1462 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9495 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9355 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9544 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0715 | highest bootstrap primary SE-ratio endpoint |
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
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 20 | the same bound, in primary replications |


## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained Python implementation computes these bounds. The R code of the authors fits other learners |
| One law, size and learner per rule | A fit outside the covered configuration prints the descriptive sentence and names the covered one |
| The `ate` only | The arm means, `att`, `atc`, the ratios and every composition rule stay descriptive |
| The default quantile and tail learners only | A caller learner reads `continuous+custom_learners`, which no row covers |
| One cross-fitting draw | A fit with `repeats` above 1 reads `+repeats`, which no row covers |
| No sharpness claim for the ATE set | The arm-mean ends are sharp. The ATE set is their difference, and the page claims coverage of that set only |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/marginal_bound_limit_coverage/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/marginal_bound_limit_coverage/manifest.json)
records the seeds, margins, exact estimator configuration, source hashes, and result hashes.
`cells.csv` records the fit configuration of each row.
