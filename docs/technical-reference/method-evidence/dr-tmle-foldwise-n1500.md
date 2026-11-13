# DR-TMLE fold-wise default on the RM18 binary draws

This study measures the cross-fitted `DRTMLE` with `reduced_crossfit="foldwise"`, the shipped
default, at n = 1,500 on the binary paper law. Two cells of
[`canonical-drtmle`](canonical-dr-tmle.md), `double_robustness/outcome_correct` and
`double_robustness/treatment_correct`, are red at that size. Both fit
`reduced_crossfit="pooled"`, the `cvFolds` construction of R `drtmle`. The
[`drtmle-foldwise` study](dr-tmle-foldwise.md) measured the default at n = 1,000 and 3,000 only.
This study fits the default on the red cells' own draws, so the two constructions are compared
draw by draw at the size where the pooled cells fail.

No canonical implementation is compared. R `drtmle` has no foldwise construction, so the study
declares no reference and runs no R. A zero-row equivalence artifact records the absence. The
study uses the `gated` policy. The study module is `tests/studies/drtmle_foldwise_n1500.py`.

## What was fitted

| setting | value |
| --- | --- |
| law | the binary complete-data law of Benkeser et al. (2017), Section 5.1, as `canonical-drtmle` draws it |
| nuisance regimes | `both_correct` in the primary phase; `outcome_correct` and `treatment_correct` in the two property cells. A misspecified model omits the `W1 W2` term |
| learners | the `canonical-drtmle` GLMs: logistic GLMs for the outcome regression and the mechanism, a Gaussian GLM for `Q_r` and a binomial GLM for `g_r` |
| construction | `cross_fit=True`, `n_folds=10`, `reduced_crossfit="foldwise"`, `reduction="univariate"`, `guard=("Q", "g")`, `update_order="drtmle"`, `max_outer=100` |
| folds | the unstratified ten-fold vector of `canonical-drtmle`, drawn by `random_partition` from the sample seed |
| primary draws | 2,000 replications of `both_correct` at n = 1,500 from this record's own fresh seed |
| cell draws | 5,600 replications per cell from the stream of the matching `canonical-drtmle` cell. Replication `r < 800` is that cell's registered replication `r`, with the same rows and fold vector |
| intervals | pointwise 95% Wald from the influence curve |
| failure bound | 0.01 of the replications of a cell |

The arm-pooled joint predictive rule sized both budgets from the Fix-6 pilots, with a
family-wise target of 0.95. The design test, `tests/unit/test_drtmle_foldwise_fix6_design.py`,
pins the paired draws, the budgets and a nonzero witness: on one registered draw the foldwise
fit differs from the pooled refit, and the pooled refit reproduces the committed `canonical-drtmle`
estimate to 1e-10.

### The red-row reading, declared before the run

The `RM18-one-sided-bias` owner closes when both paired cells pass, with the primary rows
passing. A red cell is a defect suspect, diagnosed and fixed under a fresh declaration. No
replication is added after a verdict. The roadmap's
[red-cell owners](../../roadmap.md#red-cell-owners) table carries the acceptance text verbatim.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper binary law, both nuisances correct | `ate` | average treatment effect | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0018 to 0.0011 | 0.9510 | 1.0129 | pass |
| paper binary law, both nuisances correct | `ey0` | counterfactual mean under no treatment | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0012 to 0.0011 | 0.9515 | 0.9922 | pass |
| paper binary law, both nuisances correct | `ey1` | counterfactual mean under treatment | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0015 to 0.000722 | 0.9550 | 1.0175 | pass |
<!-- /generated -->

The accuracy table reads the `both_correct` primary rows at n = 1,500.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000311 to 0.0021, margin 0.0066, SE ratio 1.0002 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0021 to 0.0039, margin 0.0069, SE ratio 0.9866 | pass |
<!-- /generated -->

Each cell fits the ATE under the shared `double_robustness` rule: the 99% interval of the bias
lies inside a quarter of an empirical standard deviation, and the reported standard error lies
on the scale of the empirical spread.

## Result

All 3 primary rows pass, and both property cells pass. The table gives the endpoints that each
claim reads.

| claim | measured | verdict |
| --- | --- | --- |
| the default recovers the truth when both nuisances are correct | largest absolute standardized bias 0.0204; lowest coverage 0.9510 | 3/3 pass |
| the default stays inside the margin when only the outcome regression is correct | standardized bias 0.0463, coverage 0.9500 | pass |
| the default stays inside the margin when only the mechanism is correct | standardized bias 0.1095, coverage 0.9441 | pass |

So the `RM18-one-sided-bias` acceptance holds. The pooled cells stay red and keep their verdicts,
as the standing record of the pooled option.

### The fold leak on the paired draws

Replication `r < 800` of each cell uses the same rows and folds as the `canonical-drtmle` cell.
The table pairs the committed pooled estimates with the foldwise ones on `(cell, replicate)` over
those 800 pairs, and gives the bias and the paired difference in units of the foldwise SD. It reads
`pooled-displacement.csv`, and no test gates these numbers.

| regime | pooled bias | foldwise bias | pooled − foldwise, 99% interval |
| --- | --- | --- | --- |
| `outcome_correct` | 0.0051 | 0.0021 | 0.0025 to 0.0033 (+0.111 SD) |
| `treatment_correct` | 0.0094 | 0.0031 | 0.0053 to 0.0073 (+0.232 SD) |

The pooled construction adds a positive bias in both regimes, and the foldwise construction
removes most of it, as at n = 1,000 and 3,000 in `drtmle-foldwise`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications |
| `n` | 1500 | observations per primary replication |
| `independent_tests_total` | 3 | primary tests against truth |
| `independent_tests_passed` | 3 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 2 | repeated-sampling property cells |
| `property_cells_passed` | 2 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | primary fits that failed |
| `max_standardized_bias` | 0.0204 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9510 | lowest primary coverage |
| `performance[cleverly-drtmle-foldwise/both_correct/ate]:standardized_bias` | -0.0132 | primary ATE bias in SD |
| `properties[double_robustness/outcome_correct]:standardized_bias` | 0.0463 | `outcome_correct` cell bias in SD |
| `properties[double_robustness/outcome_correct]:coverage` | 0.9500 | its coverage |
| `properties[double_robustness/outcome_correct]:replicates` | 5600 | its replications |
| `properties[double_robustness/treatment_correct]:standardized_bias` | 0.1095 | `treatment_correct` cell bias in SD |
| `properties[double_robustness/treatment_correct]:bias_ci_upper` | 0.0039 | its 99% upper endpoint |
| `properties[double_robustness/treatment_correct]:bias_margin` | 0.0069 | its margin, 0.25 empirical SD |
| `properties[double_robustness/treatment_correct]:coverage` | 0.9441 | its coverage |
| `properties[double_robustness/treatment_correct]:se_ratio` | 0.9866 | its SE ratio |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio screen, upper limit |
| `margin:union_model_se_lower` | 0.1000 | property-cell SE-ratio band, lower limit |
| `margin:union_model_se_upper` | 10 | property-cell SE-ratio band, upper limit |
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
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 20 | the same bound, in primary replications |

## Limitations

| limit | what it means for use |
| --- | --- |
| one binary law with GLM nuisance learners | the paper law only. Other outcome types, covariate laws and flexible learners have no witness here |
| one size | the cells answer at n = 1,500. `drtmle-foldwise` covers n = 1,000 and 3,000 |
| the treatment-correct cell uses most of its margin | its 99% bias interval reaches 0.0039 against a margin of 0.0069, so the default still carries a positive bias of about 0.11 SD at this size |
| no comparator | no maintained implementation fits the foldwise construction |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_foldwise_n1500/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.drtmle_foldwise_n1500.regenerate --output <scratch>` to
regenerate the artifacts under the declared run form. The
[run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_foldwise_n1500/run.log)
records the declared run. The
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_foldwise_n1500/properties.csv)
carry every published property row.
