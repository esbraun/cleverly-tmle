# DR-TMLE with baseline strata and fold-wise reduced regressions

This study measures the cross-fitted `DRTMLE` with baseline strata and
`reduced_crossfit="foldwise"`, the shipped default. The
[stratified DR-TMLE study](stratified-dr-tmle.md) fits `reduced_crossfit="pooled"`, the
`cvFolds` construction of R `drtmle`, so a default stratified fit had no registered truth,
coverage or SE-ratio rows of its own. This study refits that study's primary configuration with
the one setting changed, and adds a size ladder for the `treatment_correct` regime, whose pooled
cells are red. Support row `D13` rests on it.

No canonical implementation is compared. R `drtmle` has no foldwise construction, so the study
declares no reference and runs no R. A zero-row equivalence artifact records the absence. The
study uses the `gated` policy. The study modules are `tests/studies/stratified_drtmle_foldwise.py`
and `tests/studies/stratified_drtmle_foldwise_ladder.py`.

## What was fitted

| setting | value |
| --- | --- |
| law | the stratified paper law of `canonical-stratified-drtmle`, with a baseline stratum `V` in {0, 1, 2}; stratum 2 holds about a fifth of the rows |
| primary estimands | the two arm means and the ATE in each stratum, and the marginal ATE `ate`, whose exact truth is the law's mixture |
| learners | the `canonical-stratified-drtmle` GLMs |
| construction | `cross_fit=True`, `n_folds=10`, `reduced_crossfit="foldwise"`, `reduction="univariate"`, `guard=("Q", "g")`, `update_order="drtmle"`; every reduced regression is fitted inside each stratum on `(g_n, V)` |
| folds | the unstratified ten-fold vector of `canonical-stratified-drtmle`, drawn by `random_partition` from the sample seed |
| primary draws | 4,000 replications of `stratified_both_correct` at n = 2,000 from that study's seed and scenario. Replication `r < 800` is its registered replication `r` |
| ladder | `treatment_correct` at n = 2,000, 4,000 and 8,000, with a `both_wrong` control at each size; the marginal ATE and the three stratum ATEs of each fit. The n = 2,000 rung draws from the stream of the pooled `treatment_correct` cells |
| intervals | pointwise 95% Wald from the influence curve |
| failure bound | 0.01 of the replications of a cell |

The arm-pooled joint predictive rule sized the primary budget from the Fix-6 pilots over every
row except `ey[1][V=2]`, with a target of 0.95. The design test,
`tests/unit/test_drtmle_foldwise_fix6_design.py`, pins the paired draws and the budgets, and
`tests/unit/test_drtmle_foldwise_fix6_ladder_reading.py` holds the mutation controls of the
ladder rule.

### The readings, declared before the run

Every primary row answers to the shared margins of `performance-tests.csv`. `ey[1][V=2]` was
declared as an expected finite-sample row, with a predictive pass of 0.73 at the 8,000 ceiling.
Its pooled SE ratio already sat below 1, so a red row was expected to fail the finite-sample
rule. In that case it is a defect suspect, held by `X8-drtmle-small-stratum` until that owner's
small-stratum correction re-reads it.

The run predates the logit interval of a bounded mean, so its rows hold the identity interval.
On the same committed estimates and standard errors, the logit interval covers `ey[1][V=2]` at
0.919, with a 99% interval from 0.9075, above the floor
(`tests/canonical/stratified_drtmle_foldwise/interval-audit.parquet`). The SE ratio is unchanged,
so the row stays published red with its owner as the record of the identity interval.

The ladder reads each rung's SE ratio only; its bias and coverage are published. A rung's
under-coverage is mostly the bias the slope already reads, so each rung's coverage is published
beside the coverage its measured standardized bias `b` and SE ratio `r` imply,
`Phi(1.96 r - b) - Phi(-1.96 r - b)`. The rate rows of the marginal ATE and of `v1` are gated:
the 99% interval of the slope of log |bias| on log n must end below -1/2, with a half-width of at
most 0.25. The `both_wrong` control must fail to contract. The `X8-drtmle-one-sided-bias`
acceptance in the [red-cell owners](../../roadmap.md#red-cell-owners) table reads these rows.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate` | average treatment effect | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | 0.000140 to 0.0021 | 0.9460 | 0.9792 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=0]` | average treatment effect, in stratum V=0 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000885 to 0.0018 | 0.9430 | 0.9896 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=1]` | average treatment effect, in stratum V=1 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0012 to 0.0025 | 0.9383 | 0.9568 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=2]` | average treatment effect, in stratum V=2 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | 0.0011 to 0.0060 | 0.9250 | 0.9200 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=0]` | counterfactual mean under treatment arm '0', in stratum V=0 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0016 to 0.000711 | 0.9405 | 0.9748 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=1]` | counterfactual mean under treatment arm '0', in stratum V=1 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0020 to 0.000612 | 0.9443 | 0.9781 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=2]` | counterfactual mean under treatment arm '0', in stratum V=2 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0014 to 0.0012 | 0.9475 | 1.0002 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=0]` | counterfactual mean under treatment arm '1', in stratum V=0 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000828 to 0.000904 | 0.9507 | 1.0015 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=1]` | counterfactual mean under treatment arm '1', in stratum V=1 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0014 to 0.0014 | 0.9360 | 0.9550 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=2]` | counterfactual mean under treatment arm '1', in stratum V=2 | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | 0.0013 to 0.0056 | 0.9073 | 0.8820 | **fail** |
<!-- /generated -->

The accuracy table reads the primary rows at n = 2,000.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `stratified_contraction` | `marginal_ate__both_wrong_n2000` | control | average treatment effect over every stratum, from the stratified fit: both nuisances are misspecified, on the stratified paper law, at n = 2,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.9114 to 1.0823, bias 10.1370 SD, coverage 0 | pass |
| `stratified_contraction` | `marginal_ate__both_wrong_n4000` | control | average treatment effect over every stratum, from the stratified fit: both nuisances are misspecified, on the stratified paper law, at n = 4,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.8492 to 1.0344, bias 13.7082 SD, coverage 0 | pass |
| `stratified_contraction` | `marginal_ate__both_wrong_n8000` | control | average treatment effect over every stratum, from the stratified fit: both nuisances are misspecified, on the stratified paper law, at n = 8,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.8167 to 0.9927, bias 18.7917 SD, coverage 0 | pass |
| `stratified_contraction` | `marginal_ate__rate_both_wrong` | control | average treatment effect over every stratum, from the stratified fit: the same regression with both nuisances misspecified | for the marginal ATE and stratum 1, the slope interval must end at or above -1/2, so the control fails to contract; for strata 0 and 2, published and not read | slope -0.0041 to 0.0169 | pass |
| `stratified_contraction` | `marginal_ate__rate_treatment_correct` | positive | average treatment effect over every stratum, from the stratified fit: log absolute bias regressed on log n across three sizes, only the treatment mechanism correct | for the marginal ATE and stratum 1, the 99% slope interval ends below -1/2 with a half-width of at most 0.25; for strata 0 and 2, published and not read | slope -1.2198 to -0.9314 | pass |
| `stratified_contraction` | `marginal_ate__treatment_correct_n2000` | positive | average treatment effect over every stratum, from the stratified fit: only the treatment mechanism is correct, on the stratified paper law, at n = 2,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9340 to 1.0632, bias 0.3967 SD, coverage 0.9187 against 0.9301 implied | pass |
| `stratified_contraction` | `marginal_ate__treatment_correct_n4000` | positive | average treatment effect over every stratum, from the stratified fit: only the treatment mechanism is correct, on the stratified paper law, at n = 4,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9805 to 1.1190, bias 0.2713 SD, coverage 0.9425 against 0.9516 implied | pass |
| `stratified_contraction` | `marginal_ate__treatment_correct_n8000` | positive | average treatment effect over every stratum, from the stratified fit: only the treatment mechanism is correct, on the stratified paper law, at n = 8,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9619 to 1.0849, bias 0.2003 SD, coverage 0.9513 against 0.9498 implied | pass |
| `stratified_contraction` | `v0_ate__both_wrong_n2000` | control | average treatment effect in stratum V = 0: both nuisances are misspecified, on the stratified paper law, at n = 2,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.8908 to 1.0850, bias 7.1096 SD, coverage 0 | pass |
| `stratified_contraction` | `v0_ate__both_wrong_n4000` | control | average treatment effect in stratum V = 0: both nuisances are misspecified, on the stratified paper law, at n = 4,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.8302 to 0.9918, bias 9.4772 SD, coverage 0 | pass |
| `stratified_contraction` | `v0_ate__both_wrong_n8000` | control | average treatment effect in stratum V = 0: both nuisances are misspecified, on the stratified paper law, at n = 8,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.7909 to 0.9630, bias 13.0209 SD, coverage 0 | pass |
| `stratified_contraction` | `v0_ate__rate_both_wrong` | control | average treatment effect in stratum V = 0: the same regression with both nuisances misspecified | for the marginal ATE and stratum 1, the slope interval must end at or above -1/2, so the control fails to contract; for strata 0 and 2, published and not read | slope -0.0046 to 0.0257 | pass |
| `stratified_contraction` | `v0_ate__rate_treatment_correct` | positive | average treatment effect in stratum V = 0: log absolute bias regressed on log n across three sizes, only the treatment mechanism correct | for the marginal ATE and stratum 1, the 99% slope interval ends below -1/2 with a half-width of at most 0.25; for strata 0 and 2, published and not read | slope -1.3997 to -0.7754 | pass |
| `stratified_contraction` | `v0_ate__treatment_correct_n2000` | positive | average treatment effect in stratum V = 0: only the treatment mechanism is correct, on the stratified paper law, at n = 2,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9144 to 1.0376, bias 0.1850 SD, coverage 0.9375 against 0.9388 implied | pass |
| `stratified_contraction` | `v0_ate__treatment_correct_n4000` | positive | average treatment effect in stratum V = 0: only the treatment mechanism is correct, on the stratified paper law, at n = 4,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9617 to 1.0753, bias 0.1258 SD, coverage 0.9513 against 0.9518 implied | pass |
| `stratified_contraction` | `v0_ate__treatment_correct_n8000` | positive | average treatment effect in stratum V = 0: only the treatment mechanism is correct, on the stratified paper law, at n = 8,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9490 to 1.0739, bias 0.0754 SD, coverage 0.9475 against 0.9510 implied | pass |
| `stratified_contraction` | `v1_ate__both_wrong_n2000` | control | average treatment effect in stratum V = 1: both nuisances are misspecified, on the stratified paper law, at n = 2,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.9395 to 1.1269, bias 5.9300 SD, coverage 0 | pass |
| `stratified_contraction` | `v1_ate__both_wrong_n4000` | control | average treatment effect in stratum V = 1: both nuisances are misspecified, on the stratified paper law, at n = 4,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.8119 to 0.9786, bias 7.3121 SD, coverage 0 | pass |
| `stratified_contraction` | `v1_ate__both_wrong_n8000` | control | average treatment effect in stratum V = 1: both nuisances are misspecified, on the stratified paper law, at n = 8,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.8821 to 1.0596, bias 11.2560 SD, coverage 0 | pass |
| `stratified_contraction` | `v1_ate__rate_both_wrong` | control | average treatment effect in stratum V = 1: the same regression with both nuisances misspecified | for the marginal ATE and stratum 1, the slope interval must end at or above -1/2, so the control fails to contract; for strata 0 and 2, published and not read | slope -0.0158 to 0.0196 | pass |
| `stratified_contraction` | `v1_ate__rate_treatment_correct` | positive | average treatment effect in stratum V = 1: log absolute bias regressed on log n across three sizes, only the treatment mechanism correct | for the marginal ATE and stratum 1, the 99% slope interval ends below -1/2 with a half-width of at most 0.25; for strata 0 and 2, published and not read | slope -1.2873 to -0.8833 | pass |
| `stratified_contraction` | `v1_ate__treatment_correct_n2000` | positive | average treatment effect in stratum V = 1: only the treatment mechanism is correct, on the stratified paper law, at n = 2,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9173 to 1.0405, bias 0.2885 SD, coverage 0.9463 against 0.9341 implied | pass |
| `stratified_contraction` | `v1_ate__treatment_correct_n4000` | positive | average treatment effect in stratum V = 1: only the treatment mechanism is correct, on the stratified paper law, at n = 4,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9483 to 1.0809, bias 0.1896 SD, coverage 0.9463 against 0.9482 implied | pass |
| `stratified_contraction` | `v1_ate__treatment_correct_n8000` | positive | average treatment effect in stratum V = 1: only the treatment mechanism is correct, on the stratified paper law, at n = 8,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9461 to 1.0783, bias 0.1408 SD, coverage 0.9437 against 0.9496 implied | pass |
| `stratified_contraction` | `v2_ate__both_wrong_n2000` | control | average treatment effect in stratum V = 2: both nuisances are misspecified, on the stratified paper law, at n = 2,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.8689 to 1.0410, bias 4.1923 SD, coverage 0.0250 | pass |
| `stratified_contraction` | `v2_ate__both_wrong_n4000` | control | average treatment effect in stratum V = 2: both nuisances are misspecified, on the stratified paper law, at n = 4,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.9000 to 1.0722, bias 6.0783 SD, coverage 0 | pass |
| `stratified_contraction` | `v2_ate__both_wrong_n8000` | control | average treatment effect in stratum V = 2: both nuisances are misspecified, on the stratified paper law, at n = 8,000 | published and not read; the control's slope row carries its verdict | SE ratio 0.8949 to 1.0813, bias 8.7532 SD, coverage 0 | pass |
| `stratified_contraction` | `v2_ate__rate_both_wrong` | control | average treatment effect in stratum V = 2: the same regression with both nuisances misspecified | for the marginal ATE and stratum 1, the slope interval must end at or above -1/2, so the control fails to contract; for strata 0 and 2, published and not read | slope -0.0218 to 0.0269 | pass |
| `stratified_contraction` | `v2_ate__rate_treatment_correct` | positive | average treatment effect in stratum V = 2: log absolute bias regressed on log n across three sizes, only the treatment mechanism correct | for the marginal ATE and stratum 1, the 99% slope interval ends below -1/2 with a half-width of at most 0.25; for strata 0 and 2, published and not read | slope -1.3532 to -0.8732 | pass |
| `stratified_contraction` | `v2_ate__treatment_correct_n2000` | positive | average treatment effect in stratum V = 2: only the treatment mechanism is correct, on the stratified paper law, at n = 2,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9334 to 1.0743, bias 0.2383 SD, coverage 0.9350 against 0.9435 implied | pass |
| `stratified_contraction` | `v2_ate__treatment_correct_n4000` | positive | average treatment effect in stratum V = 2: only the treatment mechanism is correct, on the stratified paper law, at n = 4,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9354 to 1.0764, bias 0.1565 SD, coverage 0.9275 against 0.9476 implied | pass |
| `stratified_contraction` | `v2_ate__treatment_correct_n8000` | positive | average treatment effect in stratum V = 2: only the treatment mechanism is correct, on the stratified paper law, at n = 8,000 | SE-ratio interval inside the sanity band; bias and coverage published | SE ratio 0.9768 to 1.1115, bias 0.1561 SD, coverage 0.9563 against 0.9559 implied | pass |
<!-- /generated -->

## Result

Nine of the ten primary rows pass. All 32 ladder rows pass, and the `X8-drtmle-one-sided-bias`
acceptance holds.

| claim | measured | verdict |
| --- | --- | --- |
| the default recovers the truth for every row except `ey[1][V=2]` | largest absolute standardized bias 0.0665; every row but one covers with a lower endpoint above 0.90 | 9/10 pass |
| the smallest stratum's treated mean covers | `ey[1][V=2]` coverage 0.9073 (99% lower endpoint 0.8948), SE ratio 0.8820 (99% interval 0.8584 to 0.9079), bias inside the margin | **fail** |
| the marginal `treatment_correct` bias contracts faster than n^-1/2, consistent with second order | slope -1.0666, 99% interval -1.2198 to -0.9314 | pass |
| the `v1` `treatment_correct` bias contracts faster than n^-1/2, consistent with second order | slope -1.0693, 99% interval -1.2873 to -0.8833 | pass |
| the `both_wrong` controls fail to contract | marginal slope 0.0062 (99% interval -0.0041 to 0.0169) | pass |

The red row's SE-ratio interval excludes 1, so it does not meet the finite-sample rule. It is a
defect suspect under the declared reading, and the [red-cell ledger](red-cells.md) lists it
under `X8-drtmle-small-stratum`. Stratum 2 holds about 400 rows, and the pooled fit reads the
same row at coverage 0.900 with SE ratio 0.939.

### The ladder

The theory-properties table gives each rung's standardized bias, SE-ratio interval and coverage,
beside the coverage the bias and SE ratio imply. Each `treatment_correct` rung's coverage lies
within 0.021 of its implied coverage, so the shortfall from 0.95 is mostly the bias the slope
reads. The marginal bias falls from 0.40 SD at n = 2,000 to 0.27 and 0.20 SD at 4,000 and 8,000.

The `v0` and `v2` slopes are published at the width they reach and are not read: -1.05 (99%
interval -1.40 to -0.78) and -1.09 (-1.35 to -0.87). The pooled cells read the marginal bias at
0.56 of its spread at n = 2,000; the default reads 0.40.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4000 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 10 | primary tests against truth |
| `independent_tests_passed` | 9 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 32 | repeated-sampling property cells |
| `property_cells_passed` | 32 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | primary fits that failed |
| `max_standardized_bias` | 0.0665 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9073 | lowest primary coverage, `ey[1][V=2]` |
| `min_coverage_ci_lower` | 0.8948 | its 99% lower endpoint |
| `min_se_ratio_ci_lower` | 0.8584 | lowest primary SE-ratio 99% lower endpoint, `ey[1][V=2]` |
| `performance[cleverly-drtmle-foldwise/stratified_both_correct/ate]:standardized_bias` | 0.0465 | primary marginal ATE bias in SD |
| `performance[cleverly-drtmle-foldwise/stratified_both_correct/ey[1][V=2]]:se_ratio` | 0.8820 | SE ratio of the red row |
| `performance[cleverly-drtmle-foldwise/stratified_both_correct/ey[1][V=2]]:se_ratio_ci_upper` | 0.9079 | its 99% upper endpoint |
| `properties[stratified_contraction/marginal_ate__rate_treatment_correct]:slope` | -1.0666 | marginal `treatment_correct` contraction slope |
| `properties[stratified_contraction/marginal_ate__rate_treatment_correct]:slope_ci_lower` | -1.2198 | its 99% lower endpoint |
| `properties[stratified_contraction/marginal_ate__rate_treatment_correct]:slope_ci_upper` | -0.9314 | its 99% upper endpoint |
| `properties[stratified_contraction/v1_ate__rate_treatment_correct]:slope` | -1.0693 | `v1` `treatment_correct` contraction slope |
| `properties[stratified_contraction/v1_ate__rate_treatment_correct]:slope_ci_lower` | -1.2873 | its 99% lower endpoint |
| `properties[stratified_contraction/v1_ate__rate_treatment_correct]:slope_ci_upper` | -0.8833 | its 99% upper endpoint |
| `properties[stratified_contraction/marginal_ate__rate_both_wrong]:slope` | 0.0062 | marginal `both_wrong` control slope |
| `properties[stratified_contraction/marginal_ate__treatment_correct_n2000]:standardized_bias` | 0.3967 | marginal bias in SD at n = 2,000 |
| `properties[stratified_contraction/marginal_ate__treatment_correct_n8000]:standardized_bias` | 0.2003 | marginal bias in SD at n = 8,000 |
| `properties[stratified_contraction/marginal_ate__treatment_correct_n2000]:coverage` | 0.9187 | marginal coverage at n = 2,000 |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary and rung SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary and rung SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations. This study has no comparator |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound. This study has no comparator |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound. This study has no comparator |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound. This study has no comparator |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict; the ladder's slope gate ends below it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 40 | the same bound, in primary replications |

## Limitations

| limit | what it means for use |
| --- | --- |
| the smallest stratum's treated-mean interval is a defect suspect | in a stratum of about 400 rows, `ey[1][V=2]` under-reports its SE by about 12% (SE ratio 0.882, 99% interval 0.858 to 0.908) and covers 0.907, pending the `X8-drtmle-small-stratum` correction (Fix-3). Support row `D13` marks every interval of a default stratified fit experimental until then |
| the `treatment_correct` bias is a finite-sample term | the marginal ATE carries 0.40 SD of bias at n = 2,000 and 0.20 SD at 8,000; it contracts faster than n^-1/2, consistent with second order, but is not negligible at these sizes |
| one stratified binary law with GLM nuisance learners | other stratum counts, outcome types, covariate laws and flexible learners have no witness here |
| complete, unweighted, independent data at ten folds | the study measures no missing data, weights, clusters, fold repeats or bands |
| no comparator | no maintained implementation fits the foldwise construction |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/stratified_drtmle_foldwise/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.stratified_drtmle_foldwise.regenerate --output <scratch>`
to regenerate the artifacts under the declared run form. The
[run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/stratified_drtmle_foldwise/run.log)
records the declared run. The
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/stratified_drtmle_foldwise/properties.csv)
carry every published property row.
