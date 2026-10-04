# Cross-fitted clustered longitudinal TMLE on a t reference at few clusters

This study measures the Student $t$ reference of the cross-fitted clustered `LTMLE` at 20 and 30
clusters. An estimate that reads $J$ clusters with positive weight mass uses $J-2$ degrees of
freedom. The targeting pools every follower, so the fold count does not enter the degrees of
freedom. Below 20 clusters an `LTMLE` fit reports no interval, because no registered study
measures one there. [Longitudinal clusters](../longitudinal-tmle.md#clusters) states the rules.

The publication policy is `reporting`. Every cell may read red, and
[F28](../../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) owns any red cell. The
policy, the law, the learners and every budget were declared before the run.

## What was fitted

The subject is the subject of the
[clustered cross-fitted study](clustered-cross-fitted-longitudinal-tmle.md): five whole-cluster
folds, quasibinomial node regressions and the law's own mechanism. The law is the end-of-study law
of `tests/studies/clustered_longitudinal_laws.py` with a scaled latent $u = s v$, where $s$ is a
sign and $v$ is uniform on $[0.5, 1]$. No pinned implementation cross-fits a clustered
longitudinal fit with a $t$ reference, so no canonical implementation is compared.

The grid was fixed by a failure-only probe before the run. The probe read every declared stream
and recorded no estimate. At 10 clusters it found 3 and 8 draws in 2,000 that admit no
cross-fitted fit, and the harness refuses a cell that loses a replication. So the 10-cluster cells
left the grid before any run. At 20 and 30 clusters it found no failure on all 4,000 property
streams of each cell and all 1,000 primary streams.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 clusters of 40 rows, a scaled mean-zero cluster component on the final outcome | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` clustered cross-fitted LTMLE on a t reference at few clusters | -0.0034 to 0.0094 | 0.9450 | 0.9829 | pass |
| 20 clusters of 40 rows, a scaled mean-zero cluster component on the final outcome | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` clustered cross-fitted LTMLE on a t reference at few clusters | -0.0045 to 0.0084 | 0.9400 | 0.9778 | pass |
| 20 clusters of 40 rows, a scaled mean-zero cluster component on the final outcome | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` clustered cross-fitted LTMLE on a t reference at few clusters | -0.0023 to 0.0049 | 0.9470 | 1.0059 | pass |
| 20 clusters of 40 rows, a scaled mean-zero cluster component on the final outcome | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` clustered cross-fitted LTMLE on a t reference at few clusters | -0.0070 to 0.0036 | 0.9380 | 0.9730 | pass |
| 20 clusters of 40 rows, a scaled mean-zero cluster component on the final outcome | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` clustered cross-fitted LTMLE on a t reference at few clusters | -0.0034 to 0.0040 | 0.9380 | 0.9824 | pass |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `few_cluster_reference` | `ltmle_crossfit__equal40__j20__iid_t_control` | control | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 20 clusters of 40 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0046 to 0.0017, coverage 0.8064 to 0.8378, SE ratio 0.6256 to 0.6625 | pass |
| `few_cluster_reference` | `ltmle_crossfit__equal40__j20__normal_reference` | diagnostic | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 20 clusters of 40 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0046 to 0.0017, coverage 0.9182 to 0.9393, SE ratio 0.9542 to 1.0141 | reported |
| `few_cluster_reference` | `ltmle_crossfit__equal40__j20__t_reference` | positive | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 20 clusters of 40 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0046 to 0.0017, coverage 0.9372 to 0.9557, SE ratio 0.9540 to 1.0141 | pass |
| `few_cluster_reference` | `ltmle_crossfit__equal40__j30__iid_t_control` | control | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 30 clusters of 40 rows: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0031 to 0.0020, coverage 0.8018 to 0.8335, SE ratio 0.6364 to 0.6758 | pass |
| `few_cluster_reference` | `ltmle_crossfit__equal40__j30__normal_reference` | diagnostic | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 30 clusters of 40 rows: the cluster-robust standard error with the normal quantile | reported only | bias -0.0031 to 0.0020, coverage 0.9275 to 0.9474, SE ratio 0.9762 to 1.0373 | reported |
| `few_cluster_reference` | `ltmle_crossfit__equal40__j30__t_reference` | positive | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 30 clusters of 40 rows: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0031 to 0.0020, coverage 0.9402 to 0.9582, SE ratio 0.9781 to 1.0384 | pass |
| `few_cluster_reference` | `ltmle_crossfit__unequal40__j20__iid_t_control` | control | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 20 clusters of size uniform on 10 to 70: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias 0.000246 to 0.0070, coverage 0.7889 to 0.8214, SE ratio 0.5966 to 0.6323 | pass |
| `few_cluster_reference` | `ltmle_crossfit__unequal40__j20__normal_reference` | diagnostic | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 20 clusters of size uniform on 10 to 70: the cluster-robust standard error with the normal quantile | reported only | bias 0.000246 to 0.0070, coverage 0.9081 to 0.9304, SE ratio 0.9421 to 1.0012 | reported |
| `few_cluster_reference` | `ltmle_crossfit__unequal40__j20__t_reference` | positive | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 20 clusters of size uniform on 10 to 70: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias 0.000246 to 0.0070, coverage 0.9286 to 0.9483, SE ratio 0.9418 to 1.0014 | pass |
| `few_cluster_reference` | `ltmle_crossfit__unequal40__j30__iid_t_control` | control | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 30 clusters of size uniform on 10 to 70: the same point and t quantile with the IID row standard error | the SE-ratio upper endpoint falls below 0.80 | bias -0.0045 to 0.000958, coverage 0.7678 to 0.8015, SE ratio 0.5953 to 0.6312 | pass |
| `few_cluster_reference` | `ltmle_crossfit__unequal40__j30__normal_reference` | diagnostic | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 30 clusters of size uniform on 10 to 70: the cluster-robust standard error with the normal quantile | reported only | bias -0.0045 to 0.000958, coverage 0.9246 to 0.9449, SE ratio 0.9557 to 1.0149 | reported |
| `few_cluster_reference` | `ltmle_crossfit__unequal40__j30__t_reference` | positive | cross-fitted end-of-study regimen means and contrasts: with id= and five whole-cluster folds, 30 clusters of size uniform on 10 to 70: the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the fold-evaluated fit | exact coverage lower bound clears the floor, bias inside the margin | bias -0.0045 to 0.000958, coverage 0.9324 to 0.9516, SE ratio 0.9565 to 1.0137 | pass |
<!-- /generated -->

Each cell reads one fit per draw and publishes three arms on `ate_regimen[always vs never]`. The
`t_reference` arm is the reported interval. The `iid_t_control` arm keeps the point and the
quantile and treats the rows as independent. The `normal_reference` arm keeps the cluster-robust
standard error with the normal quantile and is reported only. Every learner is parametric.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 800 | observations per primary replication |
| `independent_tests_passed` | 5 | truth tests passing |
| `independent_tests_total` | 5 | truth tests reported |
| `property_cells_passed` | 8 | property cells passing |
| `property_cells_total` | 8 | property cells reported |
| `max_standardized_bias` | 0.0383 | largest primary standardized bias |
| `min_coverage` | 0.9380 | lowest primary coverage |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | coverage floor of a `t_reference` cell |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound. No cell of this study reads it |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |

## Limits

| limit | what it means for use |
| --- | --- |
| 20 and 30 clusters | 4 to 19 clusters are unmeasured for a cross-fitted fit, and a fit there reports no interval |
| parametric learners | the study does not establish flexible learners at few clusters. Wang, Park, Small and Li (2024), Remark 3, caution against complex learners at about 20 clusters |
| the scaled latent | the latent of this law is weaker than the one of the 100-cluster study, which a few-cluster fit needs to run on every draw |
| no comparator | no pinned implementation fits this design |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/few_cluster_crossfit_ltmle/README.md)
gives the smoke and full commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/few_cluster_crossfit_ltmle/manifest.json)
records the configuration and every result-determining module.
