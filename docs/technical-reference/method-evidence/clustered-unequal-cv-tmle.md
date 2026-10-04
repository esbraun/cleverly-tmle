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

The study has no comparator. Of the pinned implementations only R `ltmle` aggregates a clustered
curve by cluster sums, and it does not cross-fit. The
[cluster survey](../../development/method-benchmarking.md) records each comparator.

## What was fitted

| setting | value |
| --- | --- |
| construction | stacked point-treatment CV-TMLE |
| folds | five grouped folds from `random_partition`, `random_state=0` |
| treatment mechanism | exact propensity from the law |
| outcome regression | unpenalized logistic regression, `C=1e6` |
| independent unit | the cluster, aggregated as cluster sums |
| intervals | pointwise 95% Wald, normal reference at 200 clusters |

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
| `clustered_inference` | `cluster_robust_drtmle` | positive | cross-fitted DR-TMLE at unequal cluster sizes with cluster-robust inference | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9351 to 0.9589, SE ratio 0.9584 to 1.0343, paired coverage gain 0.1696 to 0.2100 | pass |
| `clustered_inference` | `cluster_robust_fold_evaluated` | positive | fold-evaluated CV-TMLE at unequal cluster sizes, centered cluster variance per fold | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9402 to 0.9630, SE ratio 0.9817 to 1.0573, paired coverage gain 0.1725 to 0.2146 | pass |
| `clustered_inference` | `cluster_robust_fold_evaluated_cluster_covariate` | positive | fold-evaluated CV-TMLE of the treated mean, 40 clusters in 10 folds, a covariate shared within a cluster | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9182 to 0.9450, SE ratio 0.9453 to 1.0187, paired coverage gain 0.4021 to 0.4542 | **fail** |
| `clustered_inference` | `iid_control` | control | the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7244 to 0.7704, SE ratio 0.5655 to 0.6080, paired coverage gain 0.1858 to 0.2279 | pass |
| `clustered_inference` | `iid_control_drtmle` | control | the DR-TMLE curve treated as independent rows | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7347 to 0.7801, SE ratio 0.5664 to 0.6110, paired coverage gain 0.1696 to 0.2100 | pass |
| `clustered_inference` | `iid_control_fold_evaluated` | control | the fold-evaluated fit's curve treated as independent rows | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7360 to 0.7813, SE ratio 0.5782 to 0.6222, paired coverage gain 0.1725 to 0.2146 | pass |
| `clustered_inference` | `iid_control_fold_evaluated_cluster_covariate` | control | the same fold-evaluated curve treated as independent rows | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.4781 to 0.5310, SE ratio 0.3326 to 0.3580, paired coverage gain 0.4021 to 0.4542 | pass |
| `estimand_weighting` | `cluster_average_truth` | control | the unweighted interval against the cluster-average mean | exact coverage upper bound falls below 0.50 | coverage 0 to 0.0022 | pass |
| `estimand_weighting` | `cluster_average_weights` | positive | the interval with weights of one over the cluster size against the cluster-average mean | exact coverage lower bound clears the floor | coverage 0.9406 to 0.9634 | pass |
| `estimand_weighting` | `individual_average` | positive | the unweighted interval against the row-weighted mean | exact coverage lower bound clears the floor | coverage 0.9379 to 0.9611 | pass |
| `estimand_weighting` | `individual_average_truth` | control | the weighted interval against the row-weighted mean | exact coverage upper bound falls below 0.50 | coverage 0 to 0.0022 | pass |
<!-- /generated -->

The `clustered_inference` family fits four pairs. Each pair reads one fit and changes only the
variance aggregation. The stacked, fold-evaluated and DR-TMLE pairs read the informative law. The
fourth pair reads a law with a covariate shared within each cluster, at 40 clusters of 30 rows in
10 folds, for the treated mean. That pair is the configuration of the fold-evaluated variance
defect this release fixes.

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
| `property_cells_total` | 12 | property cells reported |
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

## Limits

| limit | what it means for use |
| --- | --- |
| one size law | sizes uniform on 2 to 18. The study does not cover heavy-tailed sizes or a size that confounds |
| 200 clusters | the intervals use the normal reference. The [few-cluster study](clustered-few-cluster-tmle.md) measures the $t$ reference below 40 clusters |
| binary outcome and treatment, exact propensity | the study isolates targeting and inference. It does not compare learned propensity models |
| main-terms outcome learner | the study does not establish flexible learners under clustering |
| no comparator | no pinned implementation cross-fits with cluster sums at unequal sizes |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/clustered_unequal_cvtmle/README.md)
gives the smoke and full commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/clustered_unequal_cvtmle/manifest.json)
records the configuration and every result-determining module.
