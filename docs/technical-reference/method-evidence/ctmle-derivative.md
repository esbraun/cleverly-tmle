# CTMLE derivative

This study validates `CTMLE(strategy="derivative")`, which is C-TMLE0. One TMLE update adds, for
each arm, the derivative of the clever covariate along the lasso penalty. The canonical comparison
uses R [`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at pinned commit
[`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c),
through `ctmleGlmnet(ctmletype = 3)`. The theory is Ju, Wyss et al. (2019), Section 4.2. The
[C-TMLE0 section](../collaborative-tmle.md#c-tmle0-the-derivative-covariate) states the
construction and the status of the interval.

The study publishes under the `reporting` policy, which was declared before the run. Both
comparison scenarios pass. The ordinary TMLE on the same initial fit sits next to every result
as a descriptive control. One property cell is red: the both-wrong control of `double_robustness`.
The diagnosis finds no defect. The [red cell](#the-red-cell) section gives the reading.

## What was compared

| setting | `cleverly` | R `ctmle` |
| --- | --- | --- |
| scenario `binary_declared_pair` | `lambdas=` the declared pair of `penalties.csv` | `glmnet(lambda = PAIR, thresh = 1e-14)`, with both predictions passed to `ctmleGlmnet` |
| scenario `binary_pipeline` | `lambdas=None` with `learner_folds=10`. The partner is the nearest grid neighbour, which is the next smaller penalty. The grid includes the penalty 0 | `cv.glmnet(nfolds = 10)`. The mechanism is `lambda.min` and the partner is `1.05 lambda.min`, as the `ctmle` README states |
| datasets | binary-outcome samples generated in Python with their exact ATE | the identical rows and all four covariates |
| outcome regression | an unpenalized logistic GLM on `A`, `W1`, `W2`, `W3` and `W4` | the same GLM |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| targeting shrink | `alpha = 0.995` | `alpha = 0.995` |
| cross-fitting | disabled for the comparison | not applicable |
| variance | `std_error`, the standard error of the TMLE status | `var.psi`, which divides by $n - 1$ |

The declared pair is fixed before any run by `penalty_design` in
`tests/studies/canonical_ctmle_derivative.py`. The function fits the package pipeline on 50 design
draws of the primary law and takes the lower median cross-validated penalty. `penalty-design.csv`
keeps every draw. In `binary_declared_pair` both sides fit the same estimator, so the paired row
is parity at scale. In `binary_pipeline` each side runs its own penalty cross-validation, so the
row compares two pipelines.

The two pipelines differ in the cross-validation folds, in the partner, and in the grid. The
`chosen-penalties.csv` file records the penalty pair of every fit on both sides. The pairs
differ across draws, and each side's mean mechanism penalty is of order $10^{-3}$.

The property study is independent of R. It fits the internal penalty cross-validation in sample,
unless the cell says otherwise. It covers interval calibration in five cells: in sample,
cross-fitted over five folds, three arms, missing outcomes, and 100 clusters of 20 rows. It also
covers type-I error, power, a root-n ladder, three double-robustness cells, and targeting
necessity.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, both sides at one declared pair of penalties | `ate` | average treatment effect | `cleverly` C-TMLE0 derivative covariate | -0.0030 to 0.0013 | 0.9275 | 0.9318 | pass |
| binary-outcome law, both sides at one declared pair of penalties | `ate` | average treatment effect | R `ctmle` | -0.0030 to 0.0013 | 0.9275 | 0.9318 | pass |
| binary-outcome law, each side's own penalty cross-validation | `ate` | average treatment effect | `cleverly` C-TMLE0 derivative covariate | -0.0024 to 0.0017 | 0.9525 | 0.9813 | pass |
| binary-outcome law, each side's own penalty cross-validation | `ate` | average treatment effect | R `ctmle` | -0.0024 to 0.0017 | 0.9537 | 0.9808 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, both sides at one declared pair of penalties | `ate` | average treatment effect | 4.366e-07 | 0.000123 | 1.0000 | 0 | 0.000020 vs 0.0500 | equivalent |
| binary-outcome law, each side's own penalty cross-validation | `ate` | average treatment effect | -0.000002 | 0.000614 | 1.0003 | -0.0012 | 0.0017 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0031 to 0.0065, margin 0.0057, SE ratio 1.0590 | **fail** |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0021 to 0.0013, margin 0.0056, SE ratio 0.9665 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000689 to 0.0027, margin 0.0056, SE ratio 1.0898 | pass |
| `interval_calibration` | `clustered` | positive | 100 clusters of 20 rows with within-cluster outcome correlation, cross-fitted, with the cluster identifier declared | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9319 to 0.9563, SE ratio 0.9392 to 1.0127 | pass |
| `interval_calibration` | `cross_fitted` | positive | the same law and nuisances, cross-fitted over five folds | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9360 to 0.9597, SE ratio 0.9548 to 1.0255 | pass |
| `interval_calibration` | `in_sample` | positive | the internal lasso is consistent for the main-terms treatment law, and the outcome regression is a correct GLM | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9379 to 0.9611, SE ratio 0.9509 to 1.0251 | pass |
| `interval_calibration` | `missing_outcome` | positive | outcomes are missing at random, with a correct response learner | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9406 to 0.9634, SE ratio 0.9607 to 1.0330 | pass |
| `interval_calibration` | `multi_arm` | positive | three arms with a multinomial lasso, on the contrast of arm 1.0 against arm 0.0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9328 to 0.9570, SE ratio 0.9547 to 1.0289 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9208, 0.8986 to 0.9396 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000100, coverage 0.9267 to 0.9613, SE ratio 0.9667 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000431, coverage 0.9192 to 0.9556, SE ratio 0.9553 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000051, coverage 0.9163 to 0.9534, SE ratio 0.9270 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5023 to -0.4510 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4883 to -0.4859 | pass |
| `targeting_necessity` | `ate__targeted` | positive | average treatment effect: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0020 to 0.0014, margin 0.0057 | pass |
| `targeting_necessity` | `ate__untargeted` | control | average treatment effect: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.1014 to -0.1014, margin 0 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0458, 0.0355 to 0.0580 | pass |
<!-- /generated -->

## The ordinary TMLE on the same fit

Each primary row and each calibration, null and power row carries the ordinary `TMLE`. It uses
the same initial outcome fit, the same mechanism and the same bounds. Only the derivative columns
are removed. The numbers in `paired-tmle.csv` are descriptive, and no verdict reads them.

| scenario | mean C-TMLE0 minus TMLE | SD of the difference | coverage, C-TMLE0 and TMLE | mean standard error, C-TMLE0 and TMLE |
| --- | --- | --- | --- | --- |
| `binary_declared_pair` | 0.000009 | 0.0018 | 0.9275 and 0.9338 | 0.02207 and 0.02219 |
| `binary_pipeline` | 0.000015 | 0.0020 | 0.9525 and 0.9550 | 0.02201 and 0.02214 |

The two estimators agree on average to $10^{-5}$. The standard deviation of their difference is
under one tenth of the empirical standard deviation of the estimates, 0.0224 to 0.0237. When the outcome
regression is correct, the derivative columns leave the estimate and its interval close to the
ordinary TMLE. The study does not test the claim that C-TMLE0 repairs an inconsistent outcome
regression.

## The red cell

The `double_robustness/both_wrong` cell is a control. Both nuisances are misspecified on the
`outcome_correct` law, so the bias interval must fall entirely outside the equivalence margin.
The bias is 0.0048, and its 99% interval is 0.0031 to 0.0065. The margin is 0.0057. The interval
is neither inside the margin nor outside it, so the control does not discriminate. The
[measured values](#measured-values) table quotes each number.

The declared reading of a red interval, null or power cell does not cover a control. A separate
diagnosis fitted draws of 1,000,000 rows from the `outcome_correct` law.

The fit used a constant outcome regression and a fixed lasso pair. The limit bias of C-TMLE0 was about +0.006,
and the limit bias of the ordinary TMLE was about -0.004. The Hajek limit of the main-terms
projection is -0.0041. Both limits lie inside the margin at $n$ = 2,000. The `W1 W2` term in
the treatment logit therefore misspecifies the mechanism too little for this control to
separate.

The diagnosis finds no defect in the derivative fit. The plan forbids moving a law, a margin or a
size after a verdict, so the cell stays red under the `reporting` policy. The diagnostic is not a
committed test. The owner `X30-derivative-reading` in the
[red-cell owners](../../roadmap.md#red-cell-owners) names the condition that closes the cell.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications per scenario |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 4 | implementation-scenario tests against truth |
| `independent_tests_passed` | 4 | of those, passing |
| `paired_tests_total` | 2 | paired scenario tests |
| `paired_tests_passed` | 2 | of those, passing |
| `property_cells_total` | 17 | independent property cells |
| `property_cells_passed` | 16 | of those, passing |
| `max_standardized_bias` | 0.0369 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9275 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9006 | lowest exact 99% coverage endpoint |
| `min_se_ratio_ci_lower` | 0.8783 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0430 | highest bootstrap SE-ratio endpoint |
| `max_margin_utilization` | 0.000614 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0003 | largest paired RMSE-ratio upper bound |
| `min_coverage_difference_lower` | -0.0063 | smallest paired coverage-difference lower bound |
| `max_calibration_excess_upper` | 0.0011 | largest paired SE-calibration-excess upper bound |
| `equivalence[binary_declared_pair/ate]:mean_difference` | 4.366e-07 | paired mean difference at the declared pair |
| `equivalence[binary_pipeline/ate]:mean_difference` | -0.000002 | paired mean difference of the two pipelines |
| `properties[double_robustness/both_wrong]:bias` | 0.0048 | both-wrong control bias |
| `properties[double_robustness/both_wrong]:bias_ci_lower` | 0.0031 | its 99% lower endpoint |
| `properties[double_robustness/both_wrong]:bias_ci_upper` | 0.0065 | its 99% upper endpoint, against the margin below |
| `properties[double_robustness/both_wrong]:bias_margin` | 0.0057 | the equivalence margin that cell answers to |
| `properties[double_robustness/both_wrong]:standardized_bias` | 0.2114 | its bias in empirical standard deviations |
| `properties[double_robustness/both_wrong]:se_ratio` | 1.0590 | reported SE over empirical spread in that cell |
| `properties[double_robustness/outcome_correct]:bias` | -0.000380 | bias with only the outcome regression correct |
| `properties[double_robustness/treatment_correct]:bias` | 0.000990 | bias with only the treatment mechanism correct |
| `properties[interval_calibration/in_sample]:coverage` | 0.9504 | in-sample calibration coverage |
| `properties[interval_calibration/in_sample]:se_ratio` | 0.9865 | in-sample calibration SE ratio |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0458 | rejection under the confounded sharp null |
| `properties[power/alternative]:rejection_rate` | 0.9208 | rejection under the alternative |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4763 | fitted empirical-spread rate |
| `properties[root_n_rate/reported_se]:slope` | -0.4871 | fitted reported-SE rate |
| `properties[targeting_necessity/ate__untargeted]:bias` | -0.1014 | bias of the initial plug-in estimate |
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
| `margin:targeting_displacement` | 2 | least the fluctuation must move the estimate |

## Limitations

The [red-cell ledger](red-cells.md) lists each red cell of this study with the roadmap ask that
owns it.

| limitation | what it means for use |
| --- | --- |
| The both-wrong control is red | The 99% upper bias endpoint, 0.0065, is above the margin of 0.0057, so the control does not separate from the margin. The law's mechanism misspecification is too mild at $n$ = 2,000. The study does not show that the double-robustness cells would fail under a wrong fit, and it shows no defect. No margin, law or size moved after the run |
| The two pipelines differ in three ways | `binary_pipeline` differs from R in the cross-validation folds, the partner and the grid. Agreement there is agreement of two pipelines. `binary_declared_pair` fixes both sides at one pair, so only that scenario is parity of one estimator |
| The declared pair comes from package draws | The pair is the lower median of 50 package pipeline fits on the primary law. R does not choose it. The pair is a single point in the penalty path, and the study does not show agreement elsewhere |
| The parity claim is narrow | It is binary, two-arm, complete-outcome, GLM-outcome, non-cross-fitted ATE with a lasso mechanism pair. The cross-fitted, three-arm, missing-outcome and clustered fits have calibration cells and no R comparator |
| The interval is measured under a correct outcome regression | The primary law has a correct GLM outcome and a consistent lasso mechanism. The study does not test the collaborative claim, an interval under an inconsistent outcome regression. [F18](../../roadmap.md#f18-selector-path-c-tmle-inference) holds that claim. The sources report low coverage there |
| The comparison with the ordinary TMLE is descriptive | The paired TMLE rows read no verdict. They show that the two estimators agree when the outcome regression is correct. They do not show an advantage of the derivative covariate |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/README.md)
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/manifest.json)
record the provenance and the regeneration commands. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/properties.csv)
carry every published row.
