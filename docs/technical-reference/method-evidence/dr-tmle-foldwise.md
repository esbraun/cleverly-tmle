# DR-TMLE with fold-wise reduced regressions

This study measures the cross-fitted `DRTMLE` with `reduced_crossfit="foldwise"`, the shipped
default. Under this setting, fold `k`'s reduced regressions train on slab `k`. A slab holds the
production arrays at fold-`k` rows and leave-two-out arrays at the rows of each other fold.
[Reduced-regression cross-fitting](../dr-tmle/targeting.md#reduced-regression-cross-fitting)
defines the two settings and gives the argument for the foldwise construction.

The registered DR-TMLE studies with an R comparator fit `reduced_crossfit="pooled"`. That is the
`cvFolds` construction of R `drtmle`. In it, fold `k`'s data reach fold `k`'s reduced
regressions through the designs of the other folds. This study gives the default its own
evidence. It also measures that fold leak against the foldwise construction on the same draws.

No canonical implementation is compared. R `drtmle` has no foldwise construction, so the study
declares no reference and runs no R. A zero-row equivalence artifact records the absence. The
study uses the `gated` policy. The study modules are `tests/studies/drtmle_foldwise.py` and
`tests/studies/drtmle_foldwise_properties.py`.

## What was fitted

| setting | value |
| --- | --- |
| law | the binary complete-data law of Benkeser et al. (2017), Section 5.1, as `canonical-drtmle` draws it |
| nuisance regimes | `both_correct`, `outcome_correct` and `treatment_correct`. A misspecified model omits the `W1 W2` term |
| learners | the `canonical-drtmle` GLMs: logistic GLMs for the outcome regression and the mechanism, a Gaussian GLM for `Q_r` and a binomial GLM for `g_r` |
| construction | `cross_fit=True`, `n_folds=10`, `reduced_crossfit="foldwise"`, `reduction="univariate"`, `guard=("Q", "g")`, `update_order="drtmle"`, `max_outer=100` |
| folds | the unstratified ten-fold vector of `canonical-drtmle`, drawn by `random_partition` from the sample seed |
| draws | the seed and scenario order of `canonical-drtmle`. Replication `r < 800` of each scenario is that study's registered replication `r` |
| intervals | pointwise 95% Wald from the influence curve |
| primary budget | 6,000 replications at n = 3,000 for each regime |
| failure bound | 0.01 of the replications of a cell. A fit that raises, a non-finite score audit and an active propensity bound each count as a failure |

The design test, `tests/unit/test_drtmle_foldwise_study_design.py`, pins the shared draws, the
cells and the budgets.

## The property cells

Each cell fits the ATE. Each cell except `ate__pooled` uses the foldwise construction. Each
`treatment_correct` variant changes one setting of the primary fit.

| cell | role | n | R | what the cell changes |
| --- | --- | --- | --- | --- |
| `fold_locality/ate__foldwise` | positive | 1,000 | 4,400 | `treatment_correct` |
| `fold_locality/ate__pooled` | control | 1,000 | 4,400 | the same draws, with `reduced_crossfit="pooled"` |
| `double_robustness/ate__benkeser_order` | positive | 3,000 | 1,600 | `update_order="benkeser"` |
| `double_robustness/ate__bivariate` | positive | 3,000 | 4,000 | `reduction="bivariate"` |
| `double_robustness/ate__bivariate_oc` | positive | 3,000 | 1,600 | `reduction="bivariate"` under `outcome_correct` |
| `double_robustness/ate__k5` | positive | 3,000 | 2,000 | `n_folds=5` |
| `double_robustness/ate__spline_qr` | positive | 3,000 | 2,800 | cubic B-spline `Q_r` learner with 5 knots, canonical GLMs for `g_{r,1}` and `g_{r,2}` |
| `double_robustness/ate__both_wrong` | control | 3,000 | 400 | both nuisances misspecified |
| `double_robustness/multi_arm_drtmle__treatment_correct` | positive | 2,000 | 3,600 | the three-arm law of `canonical-multi-arm-drtmle`, five folds, on that study's registered draws |
| `double_robustness/<label>__stratified` | positive | 2,000 | 3,200 | the stratified paper law under `outcome_correct`: the marginal ATE and the three stratum ATEs of one fit |

The arm-pooled joint predictive rule of the property module sized each budget from the X83
pilots. At the declared budgets the rule predicts a family-wise pass of about 0.963.

### How to read the leak control

The `fold_locality` family fits each draw twice at n = 1,000, once foldwise and once pooled. The
family has three clauses.

| clause | what must hold |
| --- | --- |
| foldwise | the 99% bias interval of `ate__foldwise` lies inside the equivalence margin |
| pooled | the 99% bias interval of `ate__pooled` lies entirely outside the margin |
| displacement | the paired displacement, $\lvert \bar{\psi}_{\text{pooled}} - \bar{\psi}_{\text{foldwise}} \rvert / \operatorname{sd}(\psi_{\text{foldwise}})$, reaches `margin:fold_locality_displacement` |

A foldwise path that ran the pooled construction would give a displacement of exactly zero. So
the displacement clause fails it. The control does not separate the two channels of the leak:
the tilts and the leave-two-out designs. The unit tests in `tests/unit/test_drtmle_crossfit.py`
pin each channel.

The leak is a second-order term. It falls like `1/n` in the estimate, and the error SD falls
like `n^(-1/2)`. So the standardized leak grows as n falls. The pilots read the pooled row at
+0.317 SD at n = 1,500, too close to the margin to size. At n = 1,000 they read +0.396 SD.
The property module records both pilots.

### The red-row reading, declared before the run

The policy is `gated`. A red positive row is a defect suspect. The route is diagnosis, then a fix
and a re-run under a fresh declaration. A failing control is a design failure. No replication is
added after a verdict.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper binary law, both nuisances correct | `ate` | average treatment effect | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000337 to 0.000924 | 0.9428 | 0.9755 | pass |
| paper binary law, both nuisances correct | `ey0` | counterfactual mean under no treatment | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000779 to 0.000138 | 0.9485 | 0.9925 | pass |
| paper binary law, both nuisances correct | `ey1` | counterfactual mean under treatment | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000500 to 0.000446 | 0.9427 | 0.9802 | pass |
| paper binary law, outcome regression correct | `ate` | average treatment effect | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | 0.000063 to 0.0013 | 0.9468 | 0.9879 | pass |
| paper binary law, outcome regression correct | `ey0` | counterfactual mean under no treatment | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000739 to 0.000175 | 0.9468 | 0.9897 | pass |
| paper binary law, outcome regression correct | `ey1` | counterfactual mean under treatment | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000065 to 0.000870 | 0.9473 | 0.9913 | pass |
| paper binary law, treatment mechanism correct | `ate` | average treatment effect | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | 0.000749 to 0.0020 | 0.9477 | 1.0003 | pass |
| paper binary law, treatment mechanism correct | `ey0` | counterfactual mean under no treatment | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.0014 to -0.000455 | 0.9477 | 0.9955 | pass |
| paper binary law, treatment mechanism correct | `ey1` | counterfactual mean under treatment | `cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions | -0.000025 to 0.000940 | 0.9452 | 0.9915 | pass |
<!-- /generated -->

The accuracy table reads the three primary regimes at n = 3,000.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `ate__benkeser_order` | positive | average treatment effect: only the treatment mechanism is correct, n = 3,000, the published six-step round order | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000351 to 0.0021, margin 0.0047, SE ratio 1.0206 | pass |
| `double_robustness` | `ate__bivariate` | positive | average treatment effect: only the treatment mechanism is correct, n = 3,000, the single bivariate reduced mechanism | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0016 to 0.0031, margin 0.0047, SE ratio 1.0000 | pass |
| `double_robustness` | `ate__bivariate_oc` | positive | average treatment effect: only the outcome regression is correct, n = 3,000, the single bivariate reduced mechanism | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0013 to 0.0011, margin 0.0046, SE ratio 1.0095 | pass |
| `double_robustness` | `ate__both_wrong` | control | average treatment effect: both nuisances are misspecified, n = 3,000, ten folds | bias interval must fall entirely outside the margin, with the reported standard error on the scale of the empirical spread | bias 0.2209 to 0.2279, margin 0.0068, SE ratio 0.6504 | pass |
| `double_robustness` | `ate__k5` | positive | average treatment effect: only the treatment mechanism is correct, n = 3,000, five folds | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000172 to 0.0023, margin 0.0047, SE ratio 1.0150 | pass |
| `double_robustness` | `ate__spline_qr` | positive | average treatment effect: only the treatment mechanism is correct, n = 3,000, a cubic-spline reduced outcome regression Q_r with the canonical GLMs for g_r1 and g_r2; piecewise-constant reduced learners (capped boosting, k-NN) reached max_outer = 100 under both constructions on the pilot, so the study covers smooth reduced learners only | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0036 to 0.0055, margin 0.0048, SE ratio 0.9865 | **fail** |
| `double_robustness` | `marginal_ate__stratified` | positive | average treatment effect over every stratum, from the stratified fit: the stratified paper law; only the outcome regression is correct, n = 2,000, ten folds | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000356 to 0.0025, margin 0.0059, SE ratio 1.0613 | pass |
| `double_robustness` | `multi_arm_drtmle__treatment_correct` | positive | medium-versus-high ATE of the three-arm law: only the treatment mechanism is correct, n = 2,000, five folds, on the draws of the registered multi-arm cell | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000456 to 0.0028, margin 0.0067, SE ratio 1.0068 | pass |
| `double_robustness` | `v0_ate__stratified` | positive | average treatment effect in stratum V = 0: the stratified paper law; only the outcome regression is correct, n = 2,000, ten folds | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000239 to 0.0033, margin 0.0084, SE ratio 1.0104 | pass |
| `double_robustness` | `v1_ate__stratified` | positive | average treatment effect in stratum V = 1: the stratified paper law; only the outcome regression is correct, n = 2,000, ten folds | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000682 to 0.0032, margin 0.0106, SE ratio 1.0063 | pass |
| `double_robustness` | `v2_ate__stratified` | positive | average treatment effect in stratum V = 2: the stratified paper law; only the outcome regression is correct, n = 2,000, ten folds | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0015 to 0.0034, margin 0.0136, SE ratio 1.2093 | pass |
| `fold_locality` | `ate__foldwise` | positive | average treatment effect: only the treatment mechanism is correct, n = 1,000, ten folds; each fold's reduced regressions train on leave-two-out designs | bias interval inside the equivalence margin | bias 0.0040 to 0.0067, margin 0.0086 | pass |
| `fold_locality` | `ate__pooled` | control | average treatment effect: the same draw with pooled reduced regressions, R drtmle's cvFolds construction, which lets each fold's data reach its own reduced regressions | bias interval must fall entirely outside the margin | bias 0.0131 to 0.0159, margin 0.0091 | pass |
<!-- /generated -->

## Result

All 9 primary rows pass. 12 of the 13 property rows pass. The table gives the endpoints that
each claim reads.

| claim | measured | verdict |
| --- | --- | --- |
| the default construction recovers the truth in all three regimes | largest absolute standardized bias 0.0723 (`treatment_correct` ATE); lowest coverage 0.9427 | 9/9 pass |
| the foldwise fit removes most of the fold leak | at n = 1,000: foldwise +0.156 SD, pooled +0.400 SD, displacement 0.266 | pass |
| a pooled fit leaks | the pooled bias interval lies entirely outside the margin, with coverage 0.912 | pass |
| one-wrong robustness holds across the variants | the benkeser order, both bivariate cells, five folds, the multi-arm law and the four stratified cells pass | pass |
| the both-wrong control fails | bias +8.22 SD | pass |
| the spline reduced learners keep the bias inside the margin | +0.239 SD; the 99% interval reaches above 0.25 SD | **fail** |

The measured-values table below checks each of these numbers against the committed rows.

### The fold leak on the registered draws

Replication `r < 800` of each primary regime uses the same rows and folds as `canonical-drtmle`.
That study fits the pooled construction. The two row files therefore pair on `(scenario,
replicate)`. The initial estimates of the pairs agree to 3e-16. The table gives the ATE bias and
the paired difference in units of the foldwise SD. It reads `replicates.parquet` of both studies,
and no test gates these numbers.

| regime | pooled (`canonical-drtmle`) | foldwise | pooled − foldwise, 99% interval |
| --- | --- | --- | --- |
| `both_correct` | +0.066 | +0.063 | −0.011 to +0.016 |
| `outcome_correct` | +0.097 | +0.021 | +0.059 to +0.093 |
| `treatment_correct` | +0.156 | +0.010 | +0.111 to +0.199 |

The leak is zero within resolution when both nuisances are correct. It is positive in each
one-correct regime, and it is largest under `treatment_correct`.

### The red spline cell

The declared run's spline cell, `double_robustness/ate__spline_reduced`, is red on its bias clause. The cell fits the primary
`treatment_correct` configuration at n = 3,000 with cubic B-spline reduced learners. Its
standardized bias is +0.239 SD. The 99% interval of the bias is 0.0036 to 0.0055 against a margin
of 0.0048. Its coverage is 0.936 and its SE ratio 0.987. No fit failed.

The planning value of the
cell was +0.097 SD. The primary `treatment_correct` ATE, with the GLM reduced learners, reads
+0.072 SD.

A diagnosis read the cell after the run. It is recorded in the X83 plans record
(`cleverly-beta-plans`, `reviews/x83/spline-diagnosis.md`), not in this repository. It fitted
every arm on each of 59 fresh draws at n = 3,000 under `treatment_correct`. The table gives the
paired shift of each arm, in SD of the GLM foldwise fit.

| arm | shift |
| --- | --- |
| spline on `Q_r` and `g_{r,2}`, foldwise | +0.215 ± 0.056 |
| spline on `Q_r` and `g_{r,2}`, pooled | +0.423 ± 0.058 |
| 12-knot spline on `Q_r` and `g_{r,2}`, foldwise | +0.185 ± 0.052 |
| spline on `Q_r` only, foldwise | +0.004 ± 0.042 |
| spline on `g_{r,2}` only, foldwise | +0.319 ± 0.057 |

The pooled arm reads +0.209 ± 0.049 above the foldwise arm, which is the fold leak. At
n = 1,000 the foldwise shift is +0.218 ± 0.033 on 153 draws, so it does not fall in SD units.

The red row is a finding about a flexible learner on the `g_{r,2}` regression. The cell fits
the spline on `Q_r` and on `g_{r,2}`. The arm that fits the spline on `Q_r` only shows no shift
(+0.004 ± 0.042). Under `treatment_correct` the
main-effects outcome GLM gives a near-constant `Q̄_n(a, W)`. The design of `g_{r,2}` is then near
degenerate, and a flexible learner fits noise to a target near 0. The spline error for `Q_r` is
about 3e-3 in L2, against an SD of 0.21.

The same shift appears under the pooled construction, so the diagnosis finds no defect of the
foldwise construction. The red row stands as a published finding. No replication was added.
`DRTMLE` gives the `g_{r,2}` regression its own learner, `reduced_mechanism_residual_learner`. Its
default keeps the default fits unchanged. The cell `ate__spline_qr` replaces the red cell. It fits
the spline on `Q_r` only and keeps the canonical GLMs for `g_{r,1}` and `g_{r,2}`.

"0 failed fits" does not mean that every fit exited on tolerance. A fit that reaches `max_outer`
and whose closing pass meets the tolerance reports `converged=True` and no failure. The spline
arms reached the cap on 7% to 10% of the diagnosis draws.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 6000 | primary replications per regime |
| `n` | 3000 | observations per primary replication |
| `independent_tests_total` | 9 | primary tests against truth |
| `independent_tests_passed` | 9 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 13 | repeated-sampling property cells |
| `property_cells_passed` | 12 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | primary fits that failed |
| `max_standardized_bias` | 0.0723 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9427 | lowest primary coverage |
| `performance[cleverly-drtmle-foldwise/treatment_correct/ate]:standardized_bias` | 0.0723 | primary `treatment_correct` ATE bias in SD |
| `performance[cleverly-drtmle-foldwise/outcome_correct/ate]:standardized_bias` | 0.0367 | primary `outcome_correct` ATE bias in SD |
| `performance[cleverly-drtmle-foldwise/both_correct/ate]:standardized_bias` | 0.0155 | primary `both_correct` ATE bias in SD |
| `properties[fold_locality/ate__foldwise]:standardized_bias` | 0.1555 | foldwise leak-pair bias in SD, n = 1,000 |
| `properties[fold_locality/ate__pooled]:standardized_bias` | 0.4000 | pooled leak-pair bias in SD, n = 1,000 |
| `properties[fold_locality/ate__pooled]:coverage` | 0.9118 | pooled leak-pair coverage |
| `properties[fold_locality/ate__foldwise]:fold_locality_displacement` | 0.2664 | paired displacement of the leak pair |
| `properties[double_robustness/ate__both_wrong]:standardized_bias` | 8.2191 | both-wrong control bias in SD |
| `properties[double_robustness/ate__spline_qr]:standardized_bias` | 0.2393 | spline `Q_r` bias in SD |
| `properties[double_robustness/ate__spline_qr]:bias_ci_lower` | 0.0036 | its 99% lower endpoint |
| `properties[double_robustness/ate__spline_qr]:bias_ci_upper` | 0.0055 | its 99% upper endpoint |
| `properties[double_robustness/ate__spline_qr]:bias_margin` | 0.0048 | its margin, 0.25 empirical SD |
| `properties[double_robustness/ate__spline_qr]:coverage` | 0.9361 | its coverage |
| `properties[double_robustness/ate__spline_qr]:se_ratio` | 0.9865 | its SE ratio |
| `properties[double_robustness/ate__spline_qr]:failed_replicates` | 0 | its failed fits |
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
| `margin:fold_locality_displacement` | 0.1000 | least paired displacement of the leak pair, in foldwise SD |
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
| `margin:subject_failures_allowed` | 60 | the same bound, in primary replications |

## Limitations

| limit | what it means for use |
| --- | --- |
| the spline reduced cell is red | the cell fits the spline on `Q_r` and on `g_{r,2}`. A flexible learner on `g_{r,2}` biases the one-wrong ATE. The evidence covers the GLM reduced learners on both regressions |
| a fit at the outer cap can count as converged | a fit that reaches `max_outer` and meets the tolerance on its closing pass reports no failure. The failure count does not show how many fits reached the cap |
| smooth reduced learners only | piecewise-constant reduced learners (capped histogram boosting, a 100-neighbour k-NN) reached `max_outer = 100` under both constructions on two pilot draws at n = 3,000. The study does not measure them |
| one binary law with GLM nuisance learners | the paper law, plus the three-arm and the stratified laws in single property cells. Other outcome types, covariate laws and flexible primary learners have no witness here |
| the stratified `treatment_correct` regime is not gated | the sizing pilot read foldwise +0.29 SD (marginal) and +0.30 SD (stratum 1) on 200 draws at n = 2,000, outside the margin. Support row `D03` states this limit |
| the leak control does not separate its two channels | the control measures the tilts and the leave-two-out designs together. The unit tests in `tests/unit/test_drtmle_crossfit.py` pin each channel |
| complete, unweighted, independent data at 5 and 10 folds | the study measures no missing data, weights, clusters, fold repeats or bands |
| no comparator | no maintained implementation fits the foldwise construction |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_foldwise/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.drtmle_foldwise.regenerate --output <scratch>` to
regenerate the artifacts under the declared run form. The
[run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_foldwise/run.log)
records the declared run. The
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_foldwise/properties.csv)
carry every published property row.
