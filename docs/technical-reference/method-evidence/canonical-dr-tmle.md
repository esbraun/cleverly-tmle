# Canonical DR-TMLE

This reporting study evaluates the binary complete-data DR-TMLE against the primary simulation
law in Benkeser et al. (2017), Section 4, and against R
[`drtmle`](https://github.com/benkeser/drtmle) 1.1.2 at pinned commit
[`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c).
The truth and acceptance rules do not come from the R implementation: treatment-specific means
are computed by independent quadrature, and the robustness regimes and corrected influence curve
come from the paper's theorem and simulation design.

This row uses a **reporting** publication policy. Scientific failures remain committed and render
red; incomplete replications, invalid schemas, non-finite fits, active bounds, and broken
provenance still abort publication. Thus the tables can show whether Cleverly established
equivalence, coverage superiority, or an inconclusive result without selecting only favorable runs.

## What was compared

| setting | `cleverly` | R `drtmle` |
| --- | --- | --- |
| data-generating law | binary complete-data law from Benkeser et al. (2017), Section 4 | identical realized rows |
| nuisance regimes | outcome correct, treatment correct, and both correct | identical nuisance predictions supplied as `Qn` and `gn` |
| cross-fitting | one deterministic treatment-stratified ten-fold assignment | the identical fold vector through the package's documented `cvFolds` route |
| construction | pooled cross-fitted, univariate reductions, guards `Q` and `g`, R-package update order | native `drtmle()` with the corresponding univariate guards and two Q steps |
| estimands | `ey0`, `ey1`, and `ate` | the same three quantities, with ATE covariance formed from the native joint covariance |
| intervals | pointwise 95% Wald | pointwise 95% Wald |
| alternation rounds | `max_outer = 100` | `maxIter = 100` |
| fit audit | `score_check`'s own bar, `1e-3 x se / sqrt(n)` | the same bar, from the reference's own reported standard errors |

The primary paired route is deliberately difficult. “Superior” requires a positive 99% lower
confidence bound for Cleverly minus R coverage, plus truth validity and non-inferiority in RMSE
and SE calibration. If that route fails, ordinary bounded similarity and non-inferiority can
still establish “equivalent.”

Two rows of that table used to say something else, and both mattered.

The audit bar was `1e-4` on the Cleverly side and `1 / n` on the R side. Each was defended
against R `drtmle`'s default, which is the wrong reference for a claim about Cleverly: the
library ships `1e-3 x se / sqrt(n)`, about `3.4e-7` here. The rule is now imported from
`score_check` rather than restated, so the study cannot drift from the library.

The alternation row said `max_iter: 100`, matching the R runner. That was the cap on the Newton
steps inside one fluctuation. The alternation itself ran at a hard-coded 50 that no caller could
reach. `max_outer` is now a keyword and the value that applied is recorded on the fit.

## What the solver diagnostics do and do not compare

R `drtmle` exposes no convergence flag. The runner had no honest value to write for one, and
wrote `TRUE`. The published study then read “24 Cleverly solver failures against 0” off a column
the reference could not fail.

`solver_reported` now marks which side reports a flag at all, and the reference's `solver_passed`
is left empty. What **is** comparable is the score audit, and at the shared bar it runs the other
way: Cleverly fails 13 of 2,400 fits and R `drtmle` fails 33. Cleverly's median score is `2e-11`
to `5e-11` against the reference's `7e-9` to `9e-9`. Every Cleverly failure sits in one regime,
the one where the treatment mechanism is misspecified. The reference fails in all three.

## The two implementations reach different roots when the outcome regression is wrong

The two sides receive identical rows and identical initial nuisance predictions, agreeing to
`6e-16`. Every difference below is targeting.

| regime | per-replication disagreement in `ate`, as a share of one sampling standard deviation |
| --- | ---: |
| both nuisances correct | 3.4% |
| outcome regression correct | 10.8% |
| treatment mechanism correct | 47.6% |

The last figure is not a convergence failure on either side. It correlates with neither
implementation's score quality, at Spearman `0.045` and `-0.020`. **Cleverly's own two documented
update orders disagree by about the same amount in the same cell**, with both routes solving all
three equations to about `1e-11`. The three equations do not pin down one answer on this law once
the outcome regression is misspecified, and the route decides which answer is reached.
[Targeting and cross-fitting](../dr-tmle/targeting.md#the-update-order) carries that measurement.

This is also why the paired calibration test cannot conclude in that regime. The published
resolution of the calibration statistic tracks the disagreement exactly: `0.0024` to `0.0031`
where both nuisances are correct, `0.0072` to `0.0179` where the outcome regression is, and
`0.0306` to `0.0521` where only the mechanism is. The last exceeds the declared `0.05` margin, so
the cell reports `underpowered` rather than `inconclusive`. The measured calibration excess there
is zero: Cleverly is the better-calibrated side.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper binary law, both nuisances correct | `ate` | average treatment effect | `cleverly` | -0.000572 to 0.0029 | 0.9437 | 0.9798 | pass |
| paper binary law, both nuisances correct | `ate` | average treatment effect | R `drtmle` | -0.000636 to 0.0028 | 0.9437 | 0.9801 | pass |
| paper binary law, both nuisances correct | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.0025 to -0.000011 | 0.9600 | 1.0278 | pass |
| paper binary law, both nuisances correct | `ey0` | counterfactual mean under no treatment | R `drtmle` | -0.0024 to 0.000018 | 0.9613 | 1.0266 | pass |
| paper binary law, both nuisances correct | `ey1` | counterfactual mean under treatment | `cleverly` | -0.0014 to 0.0012 | 0.9387 | 0.9779 | pass |
| paper binary law, both nuisances correct | `ey1` | counterfactual mean under treatment | R `drtmle` | -0.0014 to 0.0012 | 0.9375 | 0.9774 | pass |
| paper binary law, outcome regression correct | `ate` | average treatment effect | `cleverly` | 0.000168 to 0.0036 | 0.9437 | 1.0017 | pass |
| paper binary law, outcome regression correct | `ate` | average treatment effect | R `drtmle` | 0.000212 to 0.0036 | 0.9475 | 1.0070 | pass |
| paper binary law, outcome regression correct | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.0015 to 0.000894 | 0.9550 | 1.0287 | pass |
| paper binary law, outcome regression correct | `ey0` | counterfactual mean under no treatment | R `drtmle` | -0.0016 to 0.000871 | 0.9550 | 1.0296 | pass |
| paper binary law, outcome regression correct | `ey1` | counterfactual mean under treatment | `cleverly` | 0.000238 to 0.0028 | 0.9300 | 0.9829 | pass |
| paper binary law, outcome regression correct | `ey1` | counterfactual mean under treatment | R `drtmle` | 0.000255 to 0.0029 | 0.9387 | 0.9861 | pass |
| paper binary law, treatment mechanism correct | `ate` | average treatment effect | `cleverly` | 0.0014 to 0.0050 | 0.9375 | 1.0085 | **fail** |
| paper binary law, treatment mechanism correct | `ate` | average treatment effect | R `drtmle` | 0.000729 to 0.0043 | 0.9463 | 1.0153 | pass |
| paper binary law, treatment mechanism correct | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.0036 to -0.000913 | 0.9513 | 0.9880 | pass |
| paper binary law, treatment mechanism correct | `ey0` | counterfactual mean under no treatment | R `drtmle` | -0.0034 to -0.000765 | 0.9525 | 1.0141 | pass |
| paper binary law, treatment mechanism correct | `ey1` | counterfactual mean under treatment | `cleverly` | -0.000429 to 0.0023 | 0.9475 | 0.9851 | pass |
| paper binary law, treatment mechanism correct | `ey1` | counterfactual mean under treatment | R `drtmle` | -0.000937 to 0.0018 | 0.9375 | 0.9854 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper binary law, both nuisances correct | `ate` | average treatment effect | 0.000066 | 0.0232 | 1.0037 | 0 | 0.0024 vs 0.0500 | equivalent |
| paper binary law, both nuisances correct | `ey0` | counterfactual mean under no treatment | -0.000028 | 0.0140 | 1.0025 | -0.0013 | 0.0031 vs 0.0500 | equivalent |
| paper binary law, both nuisances correct | `ey1` | counterfactual mean under treatment | 0.000038 | 0.0176 | 1.0026 | 0.0012 | 0.0029 vs 0.0500 | equivalent |
| paper binary law, outcome regression correct | `ate` | average treatment effect | -0.000037 | 0.0133 | 1.0127 | -0.0037 | 0.0179 vs 0.0500 | equivalent |
| paper binary law, outcome regression correct | `ey0` | counterfactual mean under no treatment | 0.000025 | 0.0124 | 1.0077 | 0 | 0.0122 vs 0.0500 | equivalent |
| paper binary law, outcome regression correct | `ey1` | counterfactual mean under treatment | -0.000012 | 0.0057 | 1.0113 | -0.0087 | 0.0072 vs 0.0500 | equivalent |
| paper binary law, treatment mechanism correct | `ate` | average treatment effect | 0.000694 | 0.2379 | 1.0537 | -0.0088 | 0.0402 vs 0.0500 | equivalent |
| paper binary law, treatment mechanism correct | `ey0` | counterfactual mean under no treatment | -0.000183 | 0.0843 | 1.0702 | -0.0012 | 0.0521 vs 0.0500 **>** | *underpowered* |
| paper binary law, treatment mechanism correct | `ey1` | counterfactual mean under treatment | 0.000511 | 0.2280 | 1.0397 | 0.0100 | 0.0306 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robust_contraction` | `both_wrong_n1500` | control | both nuisances are misspecified, at n = 1,500 | the exact coverage interval must fall below the floor | coverage 0.0233 to 0.0599, bias 0.2295 | pass |
| `double_robust_contraction` | `both_wrong_n3000` | control | both nuisances are misspecified, at n = 3,000 | the exact coverage interval must fall below the floor | coverage 0.0185 to 0.0523, bias 0.2303 | pass |
| `double_robust_contraction` | `both_wrong_n6000` | control | both nuisances are misspecified, at n = 6,000 | the exact coverage interval must fall below the floor | coverage 0.0233 to 0.0599, bias 0.2306 | pass |
| `double_robust_contraction` | `outcome_correct_n1500` | positive | only the outcome regression is correctly specified, at n = 1,500 | the exact coverage interval clears the declared floor | coverage 0.9165 to 0.9606, bias 0.0039 | pass |
| `double_robust_contraction` | `outcome_correct_n3000` | positive | only the outcome regression is correctly specified, at n = 3,000 | the exact coverage interval clears the declared floor | coverage 0.9238 to 0.9657, bias 0.0020 | pass |
| `double_robust_contraction` | `outcome_correct_n6000` | positive | only the outcome regression is correctly specified, at n = 6,000 | the exact coverage interval clears the declared floor | coverage 0.9179 to 0.9616, bias 0.0011 | pass |
| `double_robust_contraction` | `rate_both_wrong` | control | the same regression with both nuisances misspecified | slope interval must not establish contraction | slope -0.0054 to 0.0125 | pass |
| `double_robust_contraction` | `rate_outcome_correct` | positive | log absolute bias regressed on log n across three sizes, outcome regression correct | slope interval entirely below zero, so the bias contracts | slope -3.5206 to -0.0604 | pass |
| `double_robust_contraction` | `rate_treatment_correct` | positive | the same regression with only the treatment mechanism correct | slope interval entirely below zero, so the bias contracts | slope -1.9723 to -0.7546 | pass |
| `double_robust_contraction` | `treatment_correct_n1500` | positive | only the treatment mechanism is correctly specified, at n = 1,500 | the exact coverage interval clears the declared floor | coverage 0.8921 to 0.9428, bias 0.0102 | **fail** |
| `double_robust_contraction` | `treatment_correct_n3000` | positive | only the treatment mechanism is correctly specified, at n = 3,000 | the exact coverage interval clears the declared floor | coverage 0.9238 to 0.9657, bias 0.0034 | pass |
| `double_robust_contraction` | `treatment_correct_n6000` | positive | only the treatment mechanism is correctly specified, at n = 6,000 | the exact coverage interval clears the declared floor | coverage 0.9150 to 0.9596, bias 0.0020 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000280 to 0.0045, margin 0.0065, SE ratio 1.0192 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.2263 to 0.2313, margin 0.0069, SE ratio 1.8952 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0025 to 0.0075, margin 0.0068, SE ratio 0.9786 | **fail** |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0050 to 0.0104, margin 0.0073, SE ratio 0.9567 | **fail** |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9439 to 0.9659, SE ratio 0.9832 to 1.0557 | pass |
| `root_n_and_efficiency` | `n_1500` | positive | bias, coverage and SE calibration at n = 1,500 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0016, coverage 0.9165 to 0.9606, SE ratio 0.9523 | pass |
| `root_n_and_efficiency` | `n_4500` | positive | bias, coverage and SE calibration at n = 4,500 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000463, coverage 0.9208 to 0.9637, SE ratio 0.9811 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0016, coverage 0.9252 to 0.9667, SE ratio 1.0330 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5253 to -0.4440 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5112 to -0.5047 | pass |
<!-- /generated -->

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications per primary nuisance regime |
| `n` | 3000 | observations per primary replication |
| `subject_tests_passed` | 8 | Cleverly truth tests passing |
| `subject_tests_total` | 9 | Cleverly truth tests reported |
| `paired_tests_passed` | 8 | paired cells concluding equivalent or superior |
| `paired_tests_total` | 9 | paired comparison cells reported |
| `property_cells_passed` | 19 | repeated-sampling property cells passing their own and family verdicts |
| `property_cells_total` | 22 | repeated-sampling property cells reported |
| `min_coverage` | 0.9300 | lowest implementation-estimand coverage |
| `min_coverage_ci_lower` | 0.9035 | lowest exact 99% coverage lower endpoint |
| `max_standardized_bias` | 0.1633 | largest absolute standardized bias |
| `max_rmse_ratio_upper` | 1.0702 | largest paired 99% RMSE-ratio upper endpoint |
| `min_coverage_difference_lower` | -0.0250 | smallest paired 99% coverage-difference lower endpoint |
| `max_calibration_excess_upper` | 0.0500 | largest paired 99% excess-calibration upper endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples per bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor for the exact coverage lower endpoint |
| `margin:over_coverage_ceiling` | 0.9900 | coverage above this is labeled conservative |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | one-sided type-I error ceiling |
| `margin:paired_difference` | 0.1500 | paired mean-difference margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | paired RMSE-ratio upper limit |
| `margin:coverage_noninferiority` | -0.0250 | paired coverage-difference lower limit |
| `margin:calibration_noninferiority` | 0.0500 | paired calibration-excess upper limit |
| `margin:minimum_power` | 0.8000 | power-control rejection lower bound |
| `margin:root_n_slope` | -0.5000 | predicted root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## What a red double-robustness cell means here

The two one-correct `double_robustness` cells exceed the equivalence margin at `n = 1,500`. That
margin is a quarter of an empirical standard deviation, and the test is resolvable at that size:
its 99% half-width is about `0.0025` against margins of `0.0068` and `0.0073`. The bias really is
larger than the margin.

One size cannot say what kind of failure that is. A second-order remainder that has not yet
decayed and an inconsistent estimator look identical at one `n` and mean opposite things. The
`double_robust_contraction` family fits log absolute bias on log `n` over three sizes to separate
them. A slope near `-1` is a second-order remainder, near `-1/2` a first-order one, and near `0`
an estimator that is not consistent. The `both_wrong` arm is the control that must fail to
contract.

The measured slopes are `-0.94` with the outcome regression correct and `-1.19` with the mechanism
correct, both with intervals entirely below zero. The control sits at `+0.003` with an interval
that straddles zero, and its standardized bias *grows* with `n`, from `8.5` to `12.2` to `16.1`.
So both red level cells are a second-order remainder that has not yet decayed, and the control is
what an estimator that never decays looks like beside them.

Raising the level margin was considered and rejected. The standardized bias under a correct
mechanism runs `0.357`, `0.171` and `0.135` across the ladder, so no affordable size brings the
99% interval inside `0.25`. The level cell stays red and the contraction family says which red it
is.

The ladder also surfaced a result the single-size study could not reach. At `n = 1,500` with the
outcome regression misspecified, the interval does not clear the declared coverage floor: exact
coverage is `0.9200` with a 99% lower endpoint of `0.8921` against a floor of `0.90`. It clears at
`n = 3,000` and at `n = 6,000`. That cell is red, and it is a statement about small samples in one
regime rather than about the construction.

## Limitations

The comparison is binary, two-arm, complete-outcome, pointwise, and confined to the paper law and
declared GLMs. It does not establish missing-outcome, multi-arm, weighted, clustered,
simultaneous-inference, broad machine-learning, or practical-positivity parity. Supplying the
same initial nuisance predictions isolates the corrected construction; it does not compare the
two projects' learner wrappers. The study measures finite-sample behavior at its declared sizes
and cannot verify the theorem's unobservable second-order remainder condition for a future fit.

The contraction family fits three points, so its slope interval is wide and it establishes a
direction rather than an exponent. It does not identify the remainder's order.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/equivalence.csv),
[fit diagnostics](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/fit-diagnostics.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/properties.csv)
carry the protocol, provenance, and every published row.
