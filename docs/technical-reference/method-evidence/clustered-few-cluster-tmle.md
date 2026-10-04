# Clustered TMLE intervals on a t reference at few clusters

This study measures the Student $t$ reference that a clustered fit takes below 40 clusters with
positive weight mass. An estimate that reads $J$ such clusters uses $J-2$ degrees of freedom, as
Nugent et al. (2024), Section 2.2, last paragraph, recommend. A fold-evaluated estimate over $V$
folds uses $\min(J-2, J-V)$, because its variance centers the cluster totals inside each fold.
Below 10 clusters a fit reports no interval, and an `LTMLE` fit below 20, because no registered
study measures one there.
[Clusters](../inference.md#clusters) states the rules.

The publication policy is `reporting`. Every cell may read red, and
[F28](../../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) owns any red cell. The
policy, the laws, the learners and every budget were declared before the run.

## What was compared

The primary rows pair in-sample `LTMLE` on one treatment node with R `ltmle` 1.3-0 at 20
clusters. Each draw reads the informative law of `tests/studies/clustered_unequal_laws.py`, with
sizes uniform on 2 to 18.

| setting | `cleverly` | R `ltmle` 1.3-0 |
| --- | --- | --- |
| construction | in-sample LTMLE, plans `always` and `never` | `ltmle` with `abar = 1` and `abar = 0` |
| treatment mechanism | exact propensity | the identical probabilities as a numeric `gform` |
| outcome regression | quasibinomial GLM in W1 and W2 among each plan's followers | `Q.kplus1 ~ W1 + W2`, `stratify = TRUE` |
| independent unit | cluster sums | `id=`, the household curve |
| reference distribution | $t$ with $J-2$ degrees of freedom | $t$ with $J-1$ below 100 clusters |

The point estimates and the household standard errors are the same quantity. The intervals differ
by a declared convention, so this package's interval is never narrower than `ltmle`'s. The R runner
checks that its household standard error equals `ltmle`'s own to $10^{-10}$.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
<!-- /generated -->

The `few_cluster_reference` family shares each draw among five fits, four at 10 clusters: stacked CV-TMLE,
fold-evaluated CV-TMLE, in-sample TMLE, cross-fitted DR-TMLE and in-sample LTMLE. It crosses 10, 20
and 30 clusters with two size laws: clusters of 10 rows from the law of
`clustered_dgp(10, "binomial")`, and the informative law. Each fit publishes three arms. The
`t_reference` arm is the reported interval. The `iid_t_control` arm keeps the point and the
quantile and treats the rows as independent. The `normal_reference` arm keeps the cluster-robust
standard error with the normal quantile and is reported only.

The fold-evaluated fit adds a
reported `t_j_minus_2_reference` arm, which keeps $J-2$ on the same fits. Every learner is
parametric. Wang, Park, Small and Li (2024), Remark 3, caution against complex learners at about
20 clusters.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 0 | primary replications |
| `n` | 0 | nominal observations per primary replication |
| `independent_tests_passed` | 0 | truth tests passing |
| `independent_tests_total` | 0 | truth tests reported |
| `paired_tests_passed` | 0 | paired comparisons passing |
| `paired_tests_total` | 0 | paired comparisons reported |
| `property_cells_passed` | 0 | property cells passing |
| `property_cells_total` | 0 | property cells reported |
| `max_standardized_bias` | 0 | largest primary standardized bias |
| `min_coverage` | 0 | lowest primary coverage |
| `max_margin_utilization` | 0 | largest paired similarity-margin share |
| `margin:confidence_level` | 0 | Monte Carlo confidence level |
| `margin:alpha` | 0 | nominal test size |
| `margin:nominal_coverage` | 0 | nominal interval coverage |
| `margin:bootstrap_replicates` | 0 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0 | standardized-bias margin |
| `margin:coverage_floor` | 0 | coverage floor of a `t_reference` cell |
| `margin:over_coverage_ceiling` | 0 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 0 | primary SE-ratio upper screen |
| `margin:paired_difference` | 0 | paired similarity margin, in pooled empirical standard deviations |
| `margin:coverage_noninferiority` | 0 | smallest external-comparison coverage difference bound |
| `margin:rmse_noninferiority` | 0 | largest external-comparison RMSE ratio bound |
| `margin:calibration_noninferiority` | 0 | largest external-comparison calibration excess bound |

## Limits

| limit | what it means for use |
| --- | --- |
| 10 to 30 clusters | 4 to 9 clusters are unmeasured, and a fit there reports no interval. At 4 clusters a pre-run probe found 0.25% to 9% of draws on which a fit cannot run. In-sample `LTMLE` starts at 20 clusters: the first full run lost 1 `LTMLE` fit in 4,000 at 10 clusters in each size law, so those cells left the grid before any verdict was read, and the package floor for `LTMLE` is 20 |
| parametric learners | the study does not establish flexible learners at few clusters |
| binary outcome and treatment, exact propensity | the study isolates targeting and inference |
| one primary cluster count | the paired comparison runs at 20 clusters only |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_few_cluster/README.md)
gives the smoke and full commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_few_cluster/manifest.json)
records the container, runner, harness, configuration and every result-determining module.
