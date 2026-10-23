# E-value limit coverage

This study measures the coverage of the risk-ratio intervals that two part G E-value limits map.
An E-value limit is a monotone map of its risk-ratio limit, so the coverage of that interval is the
claim. Each part G configuration resolves its limit through the evidence registry with the
instrument `evalue_limit` and one rule. A passing cell adds one registry row for the configuration
it measured.

**No canonical implementation is compared.** The zero-row equivalence artifact records the
absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| cells | E3 `stratum` and E4 `time_to_event` |
| E3 law and fit | `tests/studies/stratified_law.py`, three strata of `V`, a binary outcome. In-sample TMLE with logistic outcome and treatment models, estimands `rr`, as `canonical-stratified-tmle` fits |
| E4 law and fit | the `survival` law of `point-treatment-survival`. In-sample hazard LTMLE on a `TimeToEvent` container with unpenalized logistic hazards, as `point-treatment-survival-hazard` fits |
| primary estimands | E3 `rr[V=0]`, `rr[V=1]` and `rr[V=2]`. E4 the ratio of `risk_regimen[arm1 @ t=5]` to `risk_regimen[arm0 @ t=5]` from `result.ratio(..., kind="rr")`, the interval that `evalue(result, (numerator, denominator))` reads |
| primary interval | the reported interval, on the log scale |
| size | 1,600 replications at $n = 2000$ per cell, and 4,000 property draws per cell |
| Monte Carlo inference | 99% intervals around every declared endpoint |

`X82-SG-plan.md` rev 2 in the plans repository fixed the cells, the budget, the seeds and the
red-cell route before any run.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stratified_law`, three strata, as `canonical-stratified-tmle` fits | `rr[V=0]` | marginal risk ratio, reported on the log scale, in stratum V=0 | `cleverly` risk-ratio interval behind the E-value limit | -0.0036 to 0.0043 | 0.9575 | 1.0381 | pass |
| `stratified_law`, three strata, as `canonical-stratified-tmle` fits | `rr[V=1]` | marginal risk ratio, reported on the log scale, in stratum V=1 | `cleverly` risk-ratio interval behind the E-value limit | -0.0028 to 0.0035 | 0.9487 | 0.9863 | pass |
| `stratified_law`, three strata, as `canonical-stratified-tmle` fits | `rr[V=2]` | marginal risk ratio, reported on the log scale, in stratum V=2 | `cleverly` risk-ratio interval behind the E-value limit | -0.000099 to 0.0044 | 0.9469 | 1.0074 | pass |
| the `survival` law of `point-treatment-survival`, hazard LTMLE | `rr_t5` | ratio of the arm-1 and arm-0 risks at t = 5, on the log scale | `cleverly` risk-ratio interval behind the E-value limit | -0.000826 to 0.0057 | 0.9619 | 1.0247 | pass |
<!-- /generated -->

## Theory properties

Each scenario's cells read the same 4,000 draws. A positive cell must put its SE-ratio interval
inside the calibration band and its coverage interval inside the coverage band. Each row also has
an `inflated_se_control`, which multiplies the reported standard error by 1.5. Its SE-ratio
interval must rise above the band.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `E3_rr[V=0]__correctly_specified` | positive | risk ratio in stratum V=0, on the log scale: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9442 to 0.9617, SE ratio 0.9762 to 1.0349 | pass |
| `interval_calibration` | `E3_rr[V=0]__inflated_se_control` | control | risk ratio in stratum V=0, on the log scale: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9923 to 0.9979, SE ratio 1.4624 to 1.5538 | pass |
| `interval_calibration` | `E3_rr[V=1]__correctly_specified` | positive | risk ratio in stratum V=1, on the log scale: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9628, SE ratio 0.9871 to 1.0428 | pass |
| `interval_calibration` | `E3_rr[V=1]__inflated_se_control` | control | risk ratio in stratum V=1, on the log scale: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9950 to 0.9992, SE ratio 1.4796 to 1.5658 | pass |
| `interval_calibration` | `E3_rr[V=2]__correctly_specified` | positive | risk ratio in stratum V=2, on the log scale: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9511 to 0.9673, SE ratio 0.9872 to 1.0451 | pass |
| `interval_calibration` | `E3_rr[V=2]__inflated_se_control` | control | risk ratio in stratum V=2, on the log scale: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9943 to 0.9989, SE ratio 1.4795 to 1.5670 | pass |
| `interval_calibration` | `E4_rr_t5__correctly_specified` | positive | ratio of the arm-1 and arm-0 risks at t = 5, on the log scale: every nuisance is correctly specified, and the row reads its reported limit | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9367 to 0.9553, SE ratio 0.9703 to 1.0286 | pass |
| `interval_calibration` | `E4_rr_t5__inflated_se_control` | control | ratio of the arm-1 and arm-0 risks at t = 5, on the log scale: the reported standard error is multiplied by 1.5 | the SE-ratio interval must fall above the calibration band | coverage 0.9933 to 0.9984, SE ratio 1.4551 to 1.5435 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The registry gains two `evalue_limit` rows.

| rule | covered configuration |
| --- | --- |
| `stratum` | `rr[V=0]`, `rr[V=1]` or `rr[V=2]`, in-sample TMLE, logistic learners with `C = 1e6`, binary outcome, two arms, a smallest event count of 200 |
| `time_to_event` | the ratio of two static regimen risks of an in-sample hazard LTMLE with five nodes, unpenalized logistic hazards, two arms, a smallest event count of 100 |

Both rows also require influence-curve inference, no declared mechanism, $n \ge 2000$, and at most
0.001 adjustment columns per row.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 4 | bound-end tests against truth |
| `independent_tests_passed` | 4 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 8 | repeated-sampling property cells |
| `property_cells_passed` | 8 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0616 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9469 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9307 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9443 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0899 | highest bootstrap primary SE-ratio endpoint |
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
| `margin:subject_failures_allowed` | 16 | the same bound, in primary replications |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained implementation reports these limits |
| One law, size and learner per rule | A fit outside the covered configuration prints the descriptive sentence and names the covered one |
| The `cde` limit has no row | The `cde-tmle` study measures the `rr` coverage at 0.943 and 0.946, but its fits use oracle nuisances. No user fit has that configuration, so a pointer row would cover nothing |
| `response_sd` and the derived `CTMLE` and `DRTMLE` limits have no cell | Those limits stay descriptive (`X82-SG-plan.md` rev 2, section 1) |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/evalue_limit_coverage/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/evalue_limit_coverage/manifest.json)
records the seeds, margins, exact estimator configuration, source hashes, and result hashes.
`cells.csv` records the fit configuration of each row.
