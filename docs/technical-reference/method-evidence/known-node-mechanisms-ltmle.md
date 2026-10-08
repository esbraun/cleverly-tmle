# LTMLE on declared known node mechanisms

This study validates longitudinal TMLE on data that declares its treatment and censoring
factors at each node. The fit reads the declared factors and runs no mechanism learner. With
every factor known, the remainder of the sequential TMLE is zero. The curve is then
$D^*(\bar Q_\infty, g_0)$ (van der Laan and Gruber 2012, Theorem 2 and Section 4). The section
[known node mechanisms](../longitudinal-tmle.md#known-node-mechanisms) states the estimator and
its exact-law evidence.

## What was compared

| setting | `cleverly` | R `ltmle` |
| --- | --- | --- |
| datasets | 1,000 samples of 2,000 rows. The two scenarios read the same samples | the identical rows |
| law | a two-node SMART. $A_1 \sim \operatorname{Bernoulli}(0.5)$. Retention after node 1 is 0.9 at $L_0 = 0$ and 0.8 at $L_0 = 1$. $A_2$ is $\operatorname{Bernoulli}(0.3)$ at $L_1 = 0$ and $\operatorname{Bernoulli}(0.6)$ at $L_1 = 1$. The outcome $Y$ is binary | the same |
| scenarios | `smart_q_correct`: $Y$ on $L_0 + L_1$ and $L_1$ on $L_0$, saturated within each plan. `smart_q_wrong`: $Y$ on $L_0$ and $L_1$ on an intercept | the same `Qform` |
| reported parameters | the means of the plans `always` and `never`, and their contrast | the same |
| mechanism | the declared treatment and censoring factors, read from probability columns | a numeric `gform` with the same factors, and `gbounds = c(1e-8, 1)` |
| sequential regressions | follower-stratified quasibinomial regressions | the same |
| intervals | pointwise 95% Wald from the influence curve | `variance.method = "ic"` |

The R runner refuses a fit where a cumulative bound binds. Each truth is an exact enumeration of
the law.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-node SMART with known factors, correct outcome regressions | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` LTMLE on declared known node mechanisms | -0.0032 to 0.0018 | 0.9480 | 0.9804 | pass |
| two-node SMART with known factors, correct outcome regressions | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `ltmle` with the known factors as a numeric `gform` | -0.0032 to 0.0018 | 0.9480 | 0.9804 | pass |
| two-node SMART with known factors, correct outcome regressions | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` LTMLE on declared known node mechanisms | -0.0021 to 0.0012 | 0.9350 | 0.9589 | pass |
| two-node SMART with known factors, correct outcome regressions | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `ltmle` with the known factors as a numeric `gform` | -0.0021 to 0.0012 | 0.9350 | 0.9589 | pass |
| two-node SMART with known factors, correct outcome regressions | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` LTMLE on declared known node mechanisms | -0.0017 to 0.0022 | 0.9550 | 0.9896 | pass |
| two-node SMART with known factors, correct outcome regressions | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `ltmle` with the known factors as a numeric `gform` | -0.0017 to 0.0022 | 0.9550 | 0.9896 | pass |
| two-node SMART with known factors, outcome regressions without L1 | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` LTMLE on declared known node mechanisms | -0.0033 to 0.0018 | 0.9530 | 0.9801 | pass |
| two-node SMART with known factors, outcome regressions without L1 | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `ltmle` with the known factors as a numeric `gform` | -0.0033 to 0.0018 | 0.9530 | 0.9801 | pass |
| two-node SMART with known factors, outcome regressions without L1 | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` LTMLE on declared known node mechanisms | -0.0021 to 0.0011 | 0.9400 | 0.9617 | pass |
| two-node SMART with known factors, outcome regressions without L1 | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `ltmle` with the known factors as a numeric `gform` | -0.0021 to 0.0011 | 0.9400 | 0.9617 | pass |
| two-node SMART with known factors, outcome regressions without L1 | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` LTMLE on declared known node mechanisms | -0.0017 to 0.0022 | 0.9450 | 0.9866 | pass |
| two-node SMART with known factors, outcome regressions without L1 | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `ltmle` with the known factors as a numeric `gform` | -0.0017 to 0.0022 | 0.9450 | 0.9866 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-node SMART with known factors, correct outcome regressions | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | 1.932e-10 | 4.172e-08 | 1.0000 | 0 | 1.890e-09 vs 0.0500 | equivalent |
| two-node SMART with known factors, correct outcome regressions | `ey_regimen[always]` | mean outcome under the plan treat at both times | 1.899e-10 | 6.390e-08 | 1.0000 | 0 | 8.035e-09 vs 0.0500 | equivalent |
| two-node SMART with known factors, correct outcome regressions | `ey_regimen[never]` | mean outcome under the plan treat at neither time | -3.305e-12 | 9.239e-10 | 1.0000 | 0 | 2.146e-10 vs 0.0500 | equivalent |
| two-node SMART with known factors, outcome regressions without L1 | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | 2.441e-10 | 5.203e-08 | 1.0000 | 0 | 9.728e-10 vs 0.0500 | equivalent |
| two-node SMART with known factors, outcome regressions without L1 | `ey_regimen[always]` | mean outcome under the plan treat at both times | 1.859e-10 | 6.253e-08 | 1.0000 | 0 | 5.748e-09 vs 0.0500 | equivalent |
| two-node SMART with known factors, outcome regressions without L1 | `ey_regimen[never]` | mean outcome under the plan treat at neither time | -5.819e-11 | 1.596e-08 | 1.0000 | 0 | 4.328e-10 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `known_mechanism_accuracy` | `ate_regimen__ltmle_known__q_wrong` | positive | contrast of always against never, two-node SMART: LTMLE with every factor declared, outcome regressions without L1 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000880 to 0.0041, margin 0.0077, coverage 0.9249 to 0.9627, SE ratio 0.9988 | pass |
| `known_mechanism_accuracy` | `ate_regimen__ltmle_known_treatment__censoring_estimated` | positive | contrast of always against never, two-node SMART: LTMLE with the treatment declared and the censoring estimated, outcome regressions without L1 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000880 to 0.0041, margin 0.0077, coverage 0.9272 to 0.9644, SE ratio 0.9974 | pass |
<!-- /generated -->

Both property cells fit the contrast with the wrong outcome regressions. They read one draw for
each replication, and these draws differ from the primary draws. The first cell declares every
factor. The second declares the treatment and estimates the censoring
factor with a logistic regression on $(L_0, A_1)$. That is the usual SMART analysis, where the
double-robust condition applies to the censoring factor only. Each cell must be unbiased, cover
at the floor and report a standard error inside the SE-ratio screen.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 12 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 12 | of those, passing |
| `paired_tests_total` | 6 | paired comparisons with R `ltmle` |
| `paired_tests_passed` | 6 | of those, passing |
| `property_cells_total` | 2 | repeated-sampling property cells |
| `property_cells_passed` | 2 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0242 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9350 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9063 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0475 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |

## Limitations

| limit | what it means for use |
| --- | --- |
| one two-node SMART with static plans | a modified treatment policy, a held survival design, competing risks and time-to-event input on declared factors have exact-law evidence only |
| in sample, with parametric regressions | cross-fitting and flexible learners on declared factors have fast-tier evidence only |
| no estimated-mechanism control | the study does not show a bias when the declaration is removed. The exact-law tests show that a perturbed declaration moves the estimate |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_node_mechanisms/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.known_node_mechanisms.regenerate` to regenerate the
artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_node_mechanisms/replicates.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_node_mechanisms/properties.csv)
carry every published row.
