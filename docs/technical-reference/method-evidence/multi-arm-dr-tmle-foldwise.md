# Multi-arm DR-TMLE with fold-wise reduced regressions

This study measures the cross-fitted three-arm `DRTMLE` with `reduced_crossfit="foldwise"`, the
shipped default. The [multi-arm DR-TMLE study](multi-arm-dr-tmle.md) fits
`reduced_crossfit="pooled"`, the `cvFolds` construction of R `drtmle`, so a default multi-arm fit
had no registered truth, coverage or SE-ratio rows of its own. This study refits that study's
primary configuration with the one setting changed. Support row `D12` rests on it.

No canonical implementation is compared. R `drtmle` has no foldwise construction, so the study
declares no reference and runs no R. A zero-row equivalence artifact records the absence. The
study uses the `gated` policy. The multi-arm law is not H1, so no theory claim is made for it,
and the study is read on its rows alone. The study module is
`tests/studies/multi_arm_drtmle_foldwise.py`.

## What was fitted

| setting | value |
| --- | --- |
| law | the three-arm binary-outcome law of `canonical-multi-arm-drtmle`, arms `low`, `medium` and `high` |
| estimands | the three arm means, and the ATE, risk-ratio and odds-ratio contrasts against `high` |
| learners | the `canonical-multi-arm-drtmle` primary learners: unpenalized logistic regressions for the outcome regression and the multinomial mechanism, least squares for `Q_r` and `g_{r,2}`, and a logistic regression for `g_{r,1}` |
| construction | `cross_fit=True`, `n_folds=5`, `reduced_crossfit="foldwise"`, `reduction="univariate"`, `guard=("Q", "g")`, `update_order="drtmle"`, `max_outer=100`, `g_bounds=(0.025, 0.975)` |
| folds | the unstratified five-fold vector of `canonical-multi-arm-drtmle`, drawn by `random_partition` from the sample seed |
| primary draws | 1,200 replications at n = 2,000 from the seed and scenario of `canonical-multi-arm-drtmle`. Replication `r < 800` is that study's registered replication `r` |
| property cell | `double_robustness/outcome_correct`: the registered `canonical-multi-arm-drtmle` cell's law, nuisances and stream at n = 1,000, refitted foldwise, 800 replications. Replication `r < 600` is that cell's registered replication `r` |
| intervals | pointwise 95% Wald from the influence curve: on the identity scale for the arm means and ATEs, on the log scale for the ratios |
| failure bound | 0.01 of the replications of a cell |

The arm-pooled joint predictive rule sized both budgets from the Fix-6 pilots, with a
family-wise target of 0.95. The design test, `tests/unit/test_drtmle_foldwise_fix6_design.py`,
pins the paired draws and the budgets.

The other one-correct branch is measured elsewhere. The registered `drtmle-foldwise` cell
`double_robustness/multi_arm_drtmle__treatment_correct` fits the default on the draws of the
`canonical-multi-arm-drtmle` `treatment_correct` cell at n = 2,000 and passes
([DR-TMLE with fold-wise reduced regressions](dr-tmle-foldwise.md#theory-properties)).

### The red-row reading, declared before the run

Every primary row answers to the shared margins of `performance-tests.csv`. A red bias row is a
defect suspect. A red calibration row reads `finite-sample` only when its 99% SE-ratio interval
contains 1, its coverage interval contains 0.95 and its bias passes; it then goes to
`Fix6-finite-sample`. Every other red row is a defect suspect. No replication is added after a
verdict.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[low vs high]` | difference in counterfactual means, low versus high | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0021 to 0.0017 | 0.9550 | 1.0397 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ate[medium vs high]` | difference in counterfactual means, medium versus high | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0024 to 0.0015 | 0.9583 | 1.0184 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[high]` | counterfactual mean under treatment arm 'high' | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000802 to 0.0021 | 0.9475 | 0.9880 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[low]` | counterfactual mean under treatment arm 'low' | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000978 to 0.0018 | 0.9583 | 1.0235 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `ey[medium]` | counterfactual mean under treatment arm 'medium' | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0011 to 0.0015 | 0.9642 | 1.0484 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[low vs high]` | marginal odds ratio, low versus high, reported on the log scale | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0081 to 0.0076 | 0.9542 | 1.0384 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `or[medium vs high]` | marginal odds ratio, medium versus high, reported on the log scale | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0085 to 0.0082 | 0.9567 | 1.0189 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[low vs high]` | marginal risk ratio, low versus high, reported on the log scale | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0046 to 0.0041 | 0.9517 | 1.0310 | pass |
| three-arm binary-outcome law with shared cross-fitted nuisances | `rr[medium vs high]` | marginal risk ratio, medium versus high, reported on the log scale | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0047 to 0.0037 | 0.9550 | 1.0042 | pass |
<!-- /generated -->

The accuracy table reads the primary rows at n = 2,000.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0021 to 0.0049, margin 0.0096, SE ratio 0.9786 | pass |
<!-- /generated -->

The cell fits the medium-versus-high ATE under the shared `double_robustness` rule.

## Result

All 9 primary rows pass, and the property cell passes. The table gives the endpoints that each
claim reads.

| claim | measured | verdict |
| --- | --- | --- |
| the default recovers the truth for every arm mean and contrast | largest absolute standardized bias 0.0331; lowest coverage 0.9475 | 9/9 pass |
| the reported standard errors match the spread | SE-ratio 99% intervals from 0.9334 to 1.1028 | inside 0.8 to 1.2 |
| the default stays inside the margin when only the outcome regression is correct | standardized bias 0.0368, coverage 0.9400 at n = 1,000 | pass |

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1200 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 9 | primary tests against truth |
| `independent_tests_passed` | 9 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 1 | repeated-sampling property cells |
| `property_cells_passed` | 1 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | primary fits that failed |
| `max_standardized_bias` | 0.0331 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9475 | lowest primary coverage |
| `min_se_ratio_ci_lower` | 0.9334 | lowest primary SE-ratio 99% lower endpoint |
| `max_se_ratio_ci_upper` | 1.1028 | highest primary SE-ratio 99% upper endpoint |
| `performance[cleverly-drtmle-foldwise/multi_arm_binary_drtmle/ate[medium vs high]]:standardized_bias` | -0.0170 | primary medium-versus-high ATE bias in SD |
| `properties[double_robustness/outcome_correct]:standardized_bias` | 0.0368 | `outcome_correct` cell bias in SD |
| `properties[double_robustness/outcome_correct]:coverage` | 0.9400 | its coverage |
| `properties[double_robustness/outcome_correct]:se_ratio` | 0.9786 | its SE ratio |
| `properties[double_robustness/outcome_correct]:n` | 1000 | its sample size |
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
| `margin:subject_failures_allowed` | 12 | the same bound, in primary replications |

## Limitations

| limit | what it means for use |
| --- | --- |
| one three-arm binary law with GLM nuisance learners | other arm counts, outcome types, covariate laws and flexible learners have no witness here |
| the outcome-correct cell runs at n = 1,000 only | no foldwise multi-arm `outcome_correct` cell runs at n = 2,000, the size of the primary rows and of the `treatment_correct` cell in `drtmle-foldwise` |
| no theory claim | the law is not H1, so the one-correct cells are read empirically, and a passing cell does not establish a second-order remainder |
| complete, unweighted, independent data at five folds | the study measures no missing data, weights, clusters, fold repeats or bands |
| no comparator | no maintained implementation fits the foldwise construction |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/multi_arm_drtmle_foldwise/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.multi_arm_drtmle_foldwise.regenerate --output <scratch>`
to regenerate the artifacts under the declared run form. The
[run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/multi_arm_drtmle_foldwise/run.log)
records the declared run. The
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/multi_arm_drtmle_foldwise/properties.csv)
carry every published property row.
