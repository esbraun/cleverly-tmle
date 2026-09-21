# Clustered point-treatment CV-TMLE

This study validates cluster-robust inference for cross-fitted point-treatment TMLE. It uses the
binary-outcome clustered law with ten observations per cluster. Each replication draws 200
clusters of that size. A shared hidden effect modifies the treatment effect without changing
treatment assignment. The population ATE is 0.10405 and the population risk ratio is 1.2513.

The law's arm-by-latent coefficient is declared against a measurement, not chosen by hand. The
rule matches the influence-curve design effect of the Gaussian clustered law this row replaces.
A pilot measures 1.954 for the binary law and 1.949 for the Gaussian one. `clustered_dgp` records
the rule, the pilot, and the seeds. The outcome intraclass correlation is a different quantity,
and it is larger. It is a sample statistic rather than a population one, so it needs its
condition: 0.319 at n = 200,000 and 0.313 at n = 20,000, both at seed 11, against a design
effect near two. `tests/unit/test_datasets.py` reads the smaller figure.

## What was compared

| setting | `cleverly` | R `lmtp` 1.5.4 |
| --- | --- | --- |
| construction | stacked point-treatment CV-TMLE | `lmtp_tmle` internals with the supplied folds |
| folds | five grouped folds, balanced on nothing | the identical rowwise assignment |
| treatment mechanism | exact propensity from the law | the identical exact density ratio |
| outcome regression | unpenalized logistic regression, `C=1e6` | `SL.glm` for a binomial outcome |
| independent unit | cluster identifier, aggregated as cluster sums | the same identifier, aggregated by `ife` as cluster means |
| ATE inference | joint difference influence curve | subtraction of the two `ife` arm objects |
| intervals | pointwise 95% Wald | pointwise 95% Wald |

Both runners reject a fold assignment that splits a cluster. The Python runner writes its realized
assignment beside each sampled row. The R adapter retains that assignment without rebuilding it.

`cleverly` fits an unpenalized logistic regression because `SL.glm` is unpenalized. A default
penalty would make the two working models differ before targeting ran.

## The split this row is conditional on

Every replication of this row uses one grouped partition. The estimator's `random_state` is the
fixed integer 0. Every draw emits the same cluster labels in the same order. The partition
therefore depends on nothing a draw varies, so all 800 primary and 2,400 property replications
share one byte-identical assignment. `canonical_clustered_tmle.assert_the_declared_fixed_partition`
refuses a fit that ran under any other assignment, and
`tests/unit/test_clustered_tmle_method_study.py` states the claim on two different draws.

The partition is external in the sense the split law requires: it reads nothing from the sample.
The published coverage is conditional on it. The coverage is not an average over the split
distribution. A reader who repartitions gets a different split, and this row does not measure the
extra variability that adds.

## What supports a grouped split

One reviewed source covers the **partition** and nothing else. Wang, Park, Small and Li (2024),
*Journal of the American Statistical Association* 119(548):2959-2971, partition the clusters at
random into parts of roughly equal size, and prove a cross-fitted result under that partition.
Section 4.2 and Theorem 4(b) carry it. Both locators are a section number and a theorem number,
read in the NIHMS author manuscript (PMC11795269), which carries the published volume, issue, pages
and DOI. [References](../../references.md#grouped-folds-and-clustered-cross-fitting) gives the
entry in full.

Their estimator is an AIPW-type estimator with cluster-level treatment, not this package's TMLE
with row-level treatment. The source supports the split law. It does not prove the estimator here.

The rest is the package's own estimating-equation argument. It treats clusters as the independent
units, and it needs four conditions: clusters independent of one another, equal cluster sizes, no
interference between clusters, and the usual remainder rates on the nuisances. At equal cluster
sizes the argument reduces to Zheng and van der Laan (2011), Theorem 2, with clusters in place of
rows. The design in this row satisfies all four conditions by construction.

This study is the empirical witness for that argument, and it is the only one. No source read here
proves the estimator is valid under clustering.

The two implementations aggregate the influence curve differently. `cleverly` sums the rowwise
values inside each cluster. Pinned `ife` takes the variance of the cluster means. The two
formulas agree only when every cluster holds the same number of rows. This design fixes that
number at ten. The
[clustered inference audit](../../references.md#longitudinal-survival-and-marginal-structural-models)
records the pinned `ife` behavior and the archive it pins.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with ten rows per cluster and shared effect modification | `ate` | average treatment effect | `cleverly` clustered point-treatment CV-TMLE | -0.0056 to 0.000556 | 0.9525 | 0.9721 | pass |
| binary-outcome law with ten rows per cluster and shared effect modification | `ate` | average treatment effect | R `lmtp` | -0.0056 to 0.000558 | 0.9537 | 0.9737 | pass |
| binary-outcome law with ten rows per cluster and shared effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` clustered point-treatment CV-TMLE | -0.000789 to 0.0025 | 0.9500 | 1.0306 | pass |
| binary-outcome law with ten rows per cluster and shared effect modification | `ey0` | counterfactual mean under no treatment | R `lmtp` | -0.000841 to 0.0024 | 0.9513 | 1.0305 | pass |
| binary-outcome law with ten rows per cluster and shared effect modification | `ey1` | counterfactual mean under treatment | `cleverly` clustered point-treatment CV-TMLE | -0.0049 to 0.0015 | 0.9437 | 0.9841 | pass |
| binary-outcome law with ten rows per cluster and shared effect modification | `ey1` | counterfactual mean under treatment | R `lmtp` | -0.0049 to 0.0015 | 0.9425 | 0.9837 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with ten rows per cluster and shared effect modification | `ate` | average treatment effect | -0.000005 | 0.000986 | 1.0031 | -0.0012 | 0.0020 vs 0.0500 | equivalent |
| binary-outcome law with ten rows per cluster and shared effect modification | `ey0` | counterfactual mean under no treatment | 0.000049 | 0.0183 | 1.0016 | -0.0013 | 0.0026 vs 0.0500 | equivalent |
| binary-outcome law with ten rows per cluster and shared effect modification | `ey1` | counterfactual mean under treatment | 0.000044 | 0.0084 | 1.0006 | 0.0012 | 0.0021 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `clustered_inference` | `cluster_robust` | positive | five-fold point-treatment TMLE with cluster-robust ATE inference | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9346 to 0.9585, SE ratio 0.9698 to 1.0455, paired coverage gain 0.0908 to 0.1229 | pass |
| `clustered_inference` | `iid_control` | control | the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.8212 to 0.8600, SE ratio 0.6909 to 0.7454, paired coverage gain 0.0908 to 0.1229 | pass |
<!-- /generated -->

The property study fits each sample once. Its IID control reuses the same rows, ATE estimate, and
rowwise influence curve. Only the variance aggregation changes. This pairing isolates the cluster
covariance calculation from nuisance fitting and targeting.

## Measured values

Names beginning `margin:` are thresholds declared before the run. The remaining values come from
the committed artifacts. The documentation gate checks every printed value.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 6 | implementation-estimand truth tests |
| `independent_tests_passed` | 6 | truth tests passing |
| `paired_tests_total` | 3 | paired implementation tests |
| `paired_tests_passed` | 3 | paired tests passing |
| `property_cells_total` | 2 | clustered-inference cells |
| `property_cells_passed` | 2 | property cells passing |
| `max_standardized_bias` | 0.0748 | largest absolute standardized bias |
| `min_coverage` | 0.9425 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9179 | lowest primary 99% coverage endpoint |
| `max_margin_utilization` | 0.0183 | largest share of the paired margin used |
| `properties[clustered_inference/cluster_robust]:coverage` | 0.9475 | cluster-robust ATE coverage |
| `properties[clustered_inference/cluster_robust]:se_ratio` | 1.0055 | cluster-robust ATE SE ratio |
| `properties[clustered_inference/cluster_robust]:se_ratio_ci_lower` | 0.9698 | cluster-robust SE-ratio lower endpoint |
| `properties[clustered_inference/cluster_robust]:se_ratio_ci_upper` | 1.0455 | cluster-robust SE-ratio upper endpoint |
| `properties[clustered_inference/iid_control]:coverage` | 0.8413 | IID-control ATE coverage |
| `properties[clustered_inference/iid_control]:se_ratio` | 0.7176 | IID-control ATE SE ratio |
| `properties[clustered_inference/iid_control]:se_ratio_ci_upper` | 0.7454 | IID-control SE-ratio upper endpoint |
| `properties[clustered_inference/cluster_robust]:coverage_gain_ci_lower` | 0.0908 | paired coverage-gain lower endpoint |
| `margin:confidence_level` | 0.9900 | confidence level for Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the exact coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | cluster-robust SE-ratio lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | cluster-robust SE-ratio upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | cluster-robust coverage lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | cluster-robust coverage upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size the one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest RMSE ratio the one-sided upper bound may reach |
| `margin:coverage_noninferiority` | -0.0250 | smallest coverage difference the one-sided lower bound may reach |
| `margin:calibration_noninferiority` | 0.0500 | largest excess SE-calibration error the upper bound may reach |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power control must clear |
| `margin:root_n_slope` | -0.5000 | the contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | the slower rate the interval must exclude |
| `margin:iid_control_se_ceiling` | 0.8000 | IID-control SE-ratio ceiling |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |

## Limitations

| limitation | what it means for use |
| --- | --- |
| One cluster size and one dependence law | The row validates clusters of ten under shared effect modification. It does not cover informative cluster size |
| Equal cluster sizes | `cleverly` aggregates cluster sums and pinned `ife` aggregates cluster means, as the clustered inference audit records. The two formulas agree only at equal sizes, so the row does not establish parity for unbalanced clusters |
| 200 clusters and a normal reference | The intervals use a normal reference with no small-sample cluster correction. The row does not establish coverage at a small cluster count |
| Binary outcome and binary treatment | The row does not establish continuous outcomes, multi-valued treatments, missing outcomes, or longitudinal treatment |
| One fixed partition | Every replication uses the one grouped assignment described above. The coverage is conditional on it, and the row does not measure the variability a repartition adds |
| One five-fold split | The row does not establish repeated, fold-evaluated, or fold-specific targeting |
| Exact treatment mechanism | The comparison isolates targeting and inference. It does not compare learned propensity models |
| Pointwise identity-scale intervals | The row does not cover simultaneous bands, bootstrap intervals, ratios, weights, or strata |
| Main-effects outcome learners | The row does not establish flexible learner-library parity under clustering |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_clustered_tmle/README.md)
gives smoke and full commands. The manifest records the container, runner, adapter, harness,
configuration, and result-determining Python modules.
