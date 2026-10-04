# Clustered cross-fitted end-of-study longitudinal TMLE

This study measures cross-fitted `LTMLE` with `id=`. The fit draws five whole-cluster folds,
groups every inner Super Learner split on the same labels, pools the targeting over every
follower, and sums the curve within each cluster.
[Longitudinal clusters](../longitudinal-tmle.md#clusters) states the construction and its
step from Díaz, Williams, Hoffman and Schenck (2023), Theorem 3.

The publication policy was declared `gated`, with a red-cell route declared before any run. The
first full run read one red cell, the simultaneous band. The diagnosis found no defect, so the
record moved to `reporting` before one re-run, as the route declared.

The re-run reused the R
`lmtp` rows of the first run (`python -m tests.canonical.lmtp_clustered_ltmle.regenerate --skip-reference`) and refit the Python and property phases. The red cell
reads no R input. The route named F28 as the owner. The diagnosis matched the reading of the
[`band-finite-sample`](../../roadmap.md#red-cell-owners) owner, which now holds the cell.

## What was compared

Each draw holds 100 clusters of 40 rows from the end-of-study law of
`tests/studies/clustered_longitudinal_laws.py`. The law adds a mean-zero cluster component to the
final outcome, so the regimen truths are the `make_longitudinal` quadrature truths exactly.

| setting | `cleverly` | R `lmtp` 1.5.4 with `ife` 0.2.3 |
| --- | --- | --- |
| construction | cross-fitted `LTMLE`, pooled fluctuation per node | `lmtp_tmle`, fluctuation on the training rows of each fold |
| folds | five whole-cluster folds from the cluster labels and seed 0 | the realized assignment of the same draw |
| mechanism | the law's own treatment and censoring probabilities | the identical per-node density ratios |
| node regressions | quasibinomial GLM in `W1`, `W2` and `L2` | `SL.glm` on the same designs |
| independent unit | cluster sums, $J\,\widehat{\mathrm{var}}(S_c)/n^2$ | `ife` cluster means, `id = "id"` |

At equal cluster sizes the `ife` cluster-mean rule equals the cluster-sum rule. The R runner
checks that the `ife` standard error equals the cluster-sum standard error of `lmtp`'s row curve
to $10^{-10}$ on every fit. The two constructions differ by the fold of the fluctuation, as in the
[cross-fitted end-of-study study](cross-fitted-end-of-study-longitudinal-tmle.md).

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` clustered cross-fitted LTMLE | -0.0025 to 0.0028 | 0.9463 | 0.9864 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` | -0.0025 to 0.0029 | 0.9456 | 0.9887 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` clustered cross-fitted LTMLE | -0.0021 to 0.0034 | 0.9381 | 0.9757 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` | -0.0021 to 0.0033 | 0.9387 | 0.9781 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` clustered cross-fitted LTMLE | -0.0018 to 0.0012 | 0.9437 | 0.9910 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` | -0.0018 to 0.0013 | 0.9450 | 0.9935 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` clustered cross-fitted LTMLE | -0.0027 to 0.0018 | 0.9425 | 0.9852 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` | -0.0027 to 0.0018 | 0.9450 | 0.9871 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` clustered cross-fitted LTMLE | -0.0013 to 0.0017 | 0.9425 | 0.9965 | pass |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` | -0.0013 to 0.0017 | 0.9437 | 1.0004 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -0.000052 | 0.0084 | 1.0052 | 0.000625 | 0.0022 vs 0.0500 | equivalent |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 0.000005 | 0.000812 | 1.0067 | -0.000625 | 0.0030 vs 0.0500 | equivalent |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[always]` | mean outcome under the plan treat at both times | -0.000071 | 0.0202 | 1.0053 | -0.0012 | 0.0023 vs 0.0500 | equivalent |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[never]` | mean outcome under the plan treat at neither time | -0.000019 | 0.0037 | 1.0048 | -0.0025 | 0.0023 vs 0.0500 | equivalent |
| 100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored two-node panel | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | -0.000014 | 0.0041 | 1.0111 | -0.0012 | 0.0046 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `clustered_inference` | `cluster_robust_dynamic` | positive | cross-fitted clustered LTMLE, the dynamic rule against never, on the primary fits | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9425 to 0.9571, SE ratio 0.9753 to 1.0221, paired coverage gain 0.2133 to 0.2410 | pass |
| `clustered_inference` | `cluster_robust_static` | positive | cross-fitted clustered LTMLE, ate_regimen[always vs never], 100 clusters of 40 rows | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9386 to 0.9537, SE ratio 0.9690 to 1.0177, paired coverage gain 0.2235 to 0.2513 | pass |
| `clustered_inference` | `cluster_robust_survival` | positive | cross-fitted clustered survival LTMLE, the risk contrast at t=2 | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9407 to 0.9556, SE ratio 0.9676 to 1.0154, paired coverage gain 0.1353 to 0.1590 | pass |
| `clustered_inference` | `cluster_robust_unequal` | positive | cross-fitted clustered LTMLE at cluster sizes uniform on 10 to 70 | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9402 to 0.9551, SE ratio 0.9753 to 1.0217, paired coverage gain 0.2553 to 0.2847 | pass |
| `clustered_inference` | `iid_control_dynamic` | control | the dynamic contrast's curve treated as independent rows | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7079 to 0.7378, SE ratio 0.5424 to 0.5683, paired coverage gain 0.2133 to 0.2410 | pass |
| `clustered_inference` | `iid_control_static` | control | the static LTMLE contrast's curve treated as independent rows | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.6940 to 0.7243, SE ratio 0.5252 to 0.5508, paired coverage gain 0.2235 to 0.2513 | pass |
| `clustered_inference` | `iid_control_survival` | control | the survival contrast's curve treated as independent rows | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7876 to 0.8143, SE ratio 0.6415 to 0.6720, paired coverage gain 0.1353 to 0.1590 | pass |
| `clustered_inference` | `iid_control_unequal` | control | the unequal-size contrast's curve treated as independent rows | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.6622 to 0.6935, SE ratio 0.4952 to 0.5191, paired coverage gain 0.2553 to 0.2847 | pass |
| `simultaneous_coverage` | `clustered_regimens__pointwise_joint_control` | control | the three regimen means and two contrasts of the cross-fitted clustered LTMLE, with cluster multipliers: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8111 to 0.8509 | pass |
| `simultaneous_coverage` | `clustered_regimens__simultaneous_band` | positive | the three regimen means and two contrasts of the cross-fitted clustered LTMLE, with cluster multipliers: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9191 to 0.9458 | **fail** |
<!-- /generated -->

The `clustered_inference` family has four pairs, at 6,000 replications each. Each pair reads one
fit per draw, and both arms read the same influence curve. The positive arm uses the cluster-sum
variance, and the control treats the rows as independent. The pairs are the static contrast and
the dynamic contrast on shared fits, the static contrast at cluster sizes uniform on 10 to 70, and
the risk contrast at `t=2` on the survival law. The `simultaneous_coverage` cell reads the band
over the five reported names with cluster multipliers, at 2,400 replications.

## Readings of the red cells

| cell | reading |
| --- | --- |
| `clustered_regimens__simultaneous_band` | the band covered 0.9333, with a 99% interval from 0.919 to 0.946, against the band 0.92 to 0.98. Its mean critical value is 2.387 against the design oracle 2.409. The max-t statistic has a 95% quantile of 2.50, because the five pointwise SE ratios read 0.976 to 0.997 at 100 clusters. R `lmtp` reads the same pointwise shortfall on the same draws, with SE ratios 0.978 to 1.000. With the oracle critical value the band covers 0.939, so the multiplier explains a small part of the shortfall. The point-treatment clustered band of the [default-band study](default-simultaneous-bands.md) also covers below 0.95 at 200 clusters, 0.938, and passes. The diagnosis found no defect in the band construction. `tests/unit/test_band_shortfall_reading.py` rebuilds the reading |

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications |
| `n` | 4000 | observations per primary replication |
| `independent_tests_passed` | 10 | truth tests passing |
| `independent_tests_total` | 10 | truth tests reported |
| `paired_tests_passed` | 5 | paired comparisons passing |
| `paired_tests_total` | 5 | paired comparisons reported |
| `property_cells_passed` | 9 | property cells passing |
| `property_cells_total` | 10 | property cells reported |
| `max_standardized_bias` | 0.0155 | largest primary standardized bias |
| `min_coverage` | 0.9381 | lowest primary coverage |
| `max_margin_utilization` | 0.0202 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | cluster-robust SE-ratio lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | cluster-robust SE-ratio upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | cluster-robust and band coverage lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | cluster-robust and band coverage upper limit |
| `margin:iid_control_se_ceiling` | 0.8000 | IID-control SE-ratio ceiling |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |

## Limits

| limit | what it means for use |
| --- | --- |
| parametric learners | the study does not establish flexible learners. Wang, Park, Small and Li (2024), Remark 3, caution against complex learners at few clusters |
| 100 clusters | the paired comparison and the band run at 100 clusters. [The few-cluster study](few-cluster-cross-fitted-longitudinal-tmle.md) measures 20 and 30 |
| equal sizes in the pair | `ife` aggregates cluster means, which equal cluster sums at equal sizes only. The unequal-size pair has no comparator |
| no competing-risk cell | `make_longitudinal_competing` has no cluster option. The fast tier pins the exact clustered identity |
| no informative size | the cluster size enters no outcome of the law |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_clustered_ltmle/README.md)
gives the smoke and full commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_clustered_ltmle/manifest.json)
records the container, runner, harness, configuration and every result-determining module.
