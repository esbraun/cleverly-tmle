# Clustered point-treatment CV-TMLE at unequal cluster sizes

This study validates cluster-robust inference for cross-fitted point-treatment TMLE when the
clusters differ in size and the size enters the outcome. Each replication draws 200 clusters. The
size of each cluster is uniform on 2 to 18 rows, so a draw holds about 2,000 rows. The law is
declared in `tests/studies/clustered_unequal_laws.py`. It keeps the propensity and the main terms
of `clustered_dgp(family="binomial")`, removes the arm-by-latent term, and adds
$\delta\,s(N)+\gamma\,a\,s(N)$ to the outcome logit with $s(N)=(N-10)/\sqrt{24}$, $\delta=1$ and
$\gamma=-3$. The size does not enter the propensity.

The fit estimates the row-weighted mean $\mu_I$ of Wang, Park, Small and Li (2024), Section 2.
Weights equal to one over the cluster size give the cluster-average mean $\mu_C$. On this law the
two ATEs are -0.1213 and 0.1267. A pilot declared before the run measured their gap at 6.42
standard deviations of the stacked ATE. The law module records that pilot and the decision that
replaced the plan's first pilot rule.

| scenario | sizes | truth |
| --- | --- | --- |
| `unequal_noninformative` | uniform on 2 to 18, absent from the outcome | $\mu_I=\mu_C$ |
| `unequal_informative` | uniform on 2 to 18, in the outcome | $\mu_I$ |
| `unequal_informative_cluster_average` | the samples of `unequal_informative`, with weights one over the size | $\mu_C$ |

No canonical implementation is compared. Of the pinned implementations only R `ltmle`
aggregates a clustered curve by cluster sums, and it does not cross-fit. The
[cluster survey](../../development/method-benchmarking.md) records each comparator.

## What was fitted

| setting | value |
| --- | --- |
| construction | stacked point-treatment CV-TMLE |
| folds | five grouped folds from `random_partition`, `random_state=0` |
| treatment mechanism | exact propensity from the law |
| outcome regression | unpenalized logistic regression, `C=1e6` |
| independent unit | the cluster, aggregated as cluster sums |
| intervals | pointwise 95% Wald, normal reference at 200 clusters for the stacked and DR-TMLE fits; the fold-evaluated covariate pair takes $t_{30}$, with $J-V$ degrees of freedom at 40 clusters in 10 folds |

Every row publishes the nominal `n = 2000`, because the schema requires each row to carry the
record's size. The realized row count varies by draw.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 clusters of size uniform on 2 to 18, the size in the outcome; truth the row-weighted mean | `ate` | average treatment effect | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.0059 to 0.0011 | 0.9513 | 1.0140 | pass |
| 200 clusters of size uniform on 2 to 18, the size in the outcome; truth the row-weighted mean | `ey0` | counterfactual mean under no treatment | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.0011 to 0.0028 | 0.9550 | 1.0094 | pass |
| 200 clusters of size uniform on 2 to 18, the size in the outcome; truth the row-weighted mean | `ey1` | counterfactual mean under treatment | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.0039 to 0.000764 | 0.9500 | 1.0252 | pass |
| the informative samples weighted by one over the cluster size; truth the cluster-average mean | `ate` | average treatment effect | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.0065 to 0.0013 | 0.9587 | 1.0081 | pass |
| the informative samples weighted by one over the cluster size; truth the cluster-average mean | `ey0` | counterfactual mean under no treatment | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.0012 to 0.0031 | 0.9625 | 1.0293 | pass |
| the informative samples weighted by one over the cluster size; truth the cluster-average mean | `ey1` | counterfactual mean under treatment | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.0043 to 0.000928 | 0.9525 | 1.0067 | pass |
| 200 clusters of size uniform on 2 to 18, the size absent from the outcome | `ate` | average treatment effect | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.0026 to 0.0018 | 0.9287 | 0.9644 | pass |
| 200 clusters of size uniform on 2 to 18, the size absent from the outcome | `ey0` | counterfactual mean under no treatment | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.000548 to 0.0030 | 0.9437 | 0.9633 | pass |
| 200 clusters of size uniform on 2 to 18, the size absent from the outcome | `ey1` | counterfactual mean under treatment | `cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes | -0.000889 to 0.0026 | 0.9487 | 0.9921 | pass |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `cluster_aggregation_rule` | `cluster_mean` | diagnostic | the cluster mean of the curve, as R tmle 2.1.1 and tmle3 aggregate it | reported only | coverage 0.9570 to 0.9761, SE ratio 1.1337 | reported |
| `cluster_aggregation_rule` | `cluster_sum` | diagnostic | the cluster sum of the curve, as this package and R ltmle aggregate it | reported only | coverage 0.9356 to 0.9593, SE ratio 1.0161 | reported |
| `clustered_inference` | `cluster_robust` | positive | five-fold point-treatment TMLE with cluster-robust ATE inference | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9420 to 0.9645, SE ratio 0.9616 to 1.0360, paired coverage gain 0.1858 to 0.2279 | pass |
| `clustered_inference` | `cluster_robust_drtmle` | positive | cross-fitted DR-TMLE at unequal cluster sizes with cluster-robust inference | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9337 to 0.9578, SE ratio 0.9579 to 1.0334, paired coverage gain 0.1667 to 0.2083 | pass |
| `clustered_inference` | `cluster_robust_fold_evaluated_cluster_covariate` | positive | fold-evaluated CV-TMLE of the treated mean, 40 clusters in 10 folds, a covariate shared within a cluster, on t with J - V = 30 degrees of freedom | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9478 to 0.9646, SE ratio 0.9908 to 1.0493, paired coverage gain 0.4183 to 0.4587 | pass |
| `clustered_inference` | `iid_control` | control | the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7244 to 0.7704, SE ratio 0.5655 to 0.6080, paired coverage gain 0.1858 to 0.2279 | pass |
| `clustered_inference` | `iid_control_drtmle` | control | the DR-TMLE curve treated as independent rows | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7355 to 0.7809, SE ratio 0.5650 to 0.6098, paired coverage gain 0.1667 to 0.2083 | pass |
| `clustered_inference` | `iid_control_fold_evaluated_cluster_covariate` | control | the same fold-evaluated curve treated as independent rows, with the normal quantile | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.4978 to 0.5387, SE ratio 0.3487 to 0.3691, paired coverage gain 0.4183 to 0.4587 | pass |
| `clustered_inference` | `normal_reference_fold_evaluated_cluster_covariate` | diagnostic | the same fold-evaluated point and cluster-robust standard error with the normal quantile, the reference before Fix-4 | reported only | coverage 0.9375 to 0.9559, SE ratio 1.0192 | reported |
| `estimand_weighting` | `cluster_average_truth` | control | the unweighted interval against the cluster-average mean | exact coverage upper bound falls below 0.50 | coverage 0 to 0.0022 | pass |
| `estimand_weighting` | `cluster_average_weights` | positive | the interval with weights of one over the cluster size against the cluster-average mean | exact coverage lower bound clears the floor | coverage 0.9406 to 0.9634 | pass |
| `estimand_weighting` | `individual_average` | positive | the unweighted interval against the row-weighted mean | exact coverage lower bound clears the floor | coverage 0.9379 to 0.9611 | pass |
| `estimand_weighting` | `individual_average_truth` | control | the weighted interval against the row-weighted mean | exact coverage upper bound falls below 0.50 | coverage 0 to 0.0022 | pass |
<!-- /generated -->

The `clustered_inference` family fits three pairs. Each pair reads one fit and changes only the
variance aggregation. The stacked and DR-TMLE pairs read the informative law. The third pair
reads a law with a covariate shared within each cluster, at 40 equal clusters of 30 rows in 10
folds, for the treated mean. That pair is the configuration of the fold-evaluated variance defect
that commit 6c237420 fixed.

The covariate pair first read coverage 0.9325, lower endpoint 0.918 against the 0.92 floor, on the
normal reference. Its variance has $J-V = 30$ degrees of freedom, and the fit now reports $t_{30}$
at 40 clusters, as at every cluster count. The pair is re-measured on a fresh stream at 4,000
replications. `tests/unit/test_clustered_unequal_study.py` sizes that count without the first
reading: it keeps the pass probability at 0.95 or more down to a planning ratio of 0.98, below
the 0.9917 that normal-theory Jensen gives for a $t_{30}$ standard error, to cover the kurtosis of
binary cluster totals. The same fits publish a reported arm,
`normal_reference_fold_evaluated_cluster_covariate`, with the same point and standard error and the
normal quantile, so the run shows on the same draws what $t_{30}$ adds over the earlier reference.
That arm states no verdict.

On the fresh stream the covariate pair passes. The $t_{30}$ interval's coverage has the 99%
interval 0.948 to 0.965, and its SE ratio 0.991 to 1.049. The normal arm on the same fits covers
0.938 to 0.956, so the $t_{30}$ quantile adds about one percentage point of coverage on these
draws.

The first run also fitted a fold-evaluated pair on the informative law at 200 clusters. It passed,
with a point biased by 0.13 empirical SDs against 0.02 for the stacked fit. That pair left the
study, because `cv_evaluation=True` now refuses clusters of unequal size and redirects to the
stacked report, which the `cluster_robust` pair measures on the same law
([clusters of unequal mass](../cv-tmle.md#clusters-of-unequal-mass)). Every retained pair other
than the covariate pair keeps its seeds.

The `estimand_weighting` family fits each draw twice, unweighted and
with weights one over the size. Each interval must cover its own estimand and must miss the other.
The `cluster_aggregation_rule` family reports the cluster-mean rule of R `tmle` 2.1.1 and `tmle3`
beside the cluster sum.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 2000 | nominal observations per replication |
| `independent_tests_passed` | 9 | truth tests passing |
| `independent_tests_total` | 9 | truth tests reported |
| `property_cells_passed` | 10 | property cells passing |
| `property_cells_total` | 10 | property cells reported |
| `max_standardized_bias` | 0.0634 | largest primary standardized bias |
| `min_coverage` | 0.9287 | lowest primary coverage |
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
| `margin:calibration_coverage_lower` | 0.9200 | cluster-robust coverage lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | cluster-robust coverage upper limit |
| `margin:iid_control_se_ceiling` | 0.8000 | IID-control SE-ratio ceiling |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | largest supported type-I rate. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 8 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| one size law | sizes uniform on 2 to 18. The study does not cover heavy-tailed sizes or a size that confounds |
| 200 clusters | the stacked and DR-TMLE intervals use the normal reference. The [few-cluster study](clustered-few-cluster-tmle.md) measures the $t$ reference below 40 clusters |
| fold evaluation at equal sizes only | `cv_evaluation=True` refuses clusters of unequal size, so the only fold-evaluated pair is the covariate pair at 40 equal clusters |
| binary outcome and treatment, exact propensity | the study isolates targeting and inference. It does not compare learned propensity models |
| main-terms outcome learner | the study does not establish flexible learners under clustering |
| no comparator | no pinned implementation cross-fits with cluster sums at unequal sizes |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/clustered_unequal_cvtmle/README.md)
gives the smoke and full commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/clustered_unequal_cvtmle/manifest.json)
records the configuration and every result-determining module.
