# C-TMLE0 derivative covariate

This study validates C-TMLE0, `CTMLE(strategy="derivative")`. One TMLE update adds, for
each arm, the derivative of the clever covariate along the lasso penalty. The declared comparator
is R [`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at pinned commit
[`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c),
through `ctmleGlmnet(ctmletype = 3)`. The theory is Ju, Wyss et al. (2019), Section 4.2. The
[C-TMLE0 section](../collaborative-tmle.md#c-tmle0-the-derivative-covariate) states the
construction and the status of the interval.

The evidence on this page is Python property evidence, with comparator runs pending. The Python
primary phase and the property study ran. The R `ctmle` runs wait for beta-plans TRACKER item 18,
so this page has no paired row and no parity result yet.

The study publishes under the `reporting` policy, which was declared before the run. All three
primary scenarios pass their accuracy tests. All 21 property rows pass, and no cell is red. The
ordinary TMLE on the same initial fit sits next to every result as a descriptive control.

## What was compared

| setting | `cleverly` | R `ctmle` (declared, runs pending) |
| --- | --- | --- |
| scenario `binary_declared_pair` | `lambdas=` the declared pair of `penalties.csv` | `glmnet(lambda = PAIR, thresh = 1e-14)`, with both predictions passed to `ctmleGlmnet` |
| scenario `binary_pipeline` | `lambdas=None` with `learner_folds=10`. The partner is the nearest grid neighbour, which is the next smaller penalty. The grid includes the penalty 0 | `cv.glmnet(nfolds = 10)`. The mechanism is `lambda.min` and the partner is `1.05 lambda.min`, as the `ctmle` README states |
| scenario `binary_constant_q` | the declared pair, with the constant outcome regression `mean(Y)` for both arms | the same pair and the same constant outcome regression |
| datasets | binary-outcome samples generated in Python with their exact ATE | the identical rows and all four covariates |
| outcome regression | an unpenalized logistic GLM on `A`, `W1`, `W2`, `W3` and `W4`, except in `binary_constant_q` | the same GLM, or the same constant |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| targeting shrink | `alpha = 0.995` | `alpha = 0.995` |
| cross-fitting | disabled for the comparison | not applicable |
| variance | `std_error`, the standard error of the TMLE status | `var.psi`, which divides by $n - 1$ |

The R side is declared and its runner is committed, but no R run exists. The manifest records the
comparator as pending, and `equivalence.csv` has no rows. The comparator never blocks the method.

The declared pair is fixed before any run by `penalty_design` in
`tests/studies/canonical_ctmle_derivative.py`. The function fits the package pipeline on 50 design
draws of the primary law and takes the lower median cross-validated penalty. `penalty-design.csv`
keeps every draw. In `binary_declared_pair` and `binary_constant_q` both sides would fit the same
estimator, so a paired row there is parity at scale. In `binary_pipeline` each side runs its own
penalty cross-validation, so a paired row compares two pipelines.

The scenario `binary_constant_q` is the parity-witness scenario. With the correct GLM, the
derivative columns barely move the estimate, so a side that dropped them would still match R.
With a constant outcome regression, the update carries the estimate. The witness columns of
`paired-tmle.csv` are empty until the R runs exist.

The two pipelines differ in the cross-validation folds, in the partner, and in the grid. The
`chosen-penalties.csv` file records the penalty pair of every `cleverly` fit. The mean mechanism
penalty is 0.0010 for the declared pair and 0.0019 for the package pipeline.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, one declared pair of penalties, a constant outcome regression | `ate` | average treatment effect | `cleverly` C-TMLE0 derivative covariate | -0.0012 to 0.0030 | 0.9688 | 1.0783 | pass |
| binary-outcome law, both sides at one declared pair of penalties | `ate` | average treatment effect | `cleverly` C-TMLE0 derivative covariate | -0.0030 to 0.0013 | 0.9275 | 0.9318 | pass |
| binary-outcome law, each side's own penalty cross-validation | `ate` | average treatment effect | `cleverly` C-TMLE0 derivative covariate | -0.0024 to 0.0017 | 0.9525 | 0.9813 | pass |
<!-- /generated -->

## Theory properties

The property study is independent of R. It fits the internal penalty cross-validation in sample,
unless the cell says otherwise. The table lists each family.

| family | cells | what the family tests |
| --- | --- | --- |
| `interval_calibration` | `in_sample`, `cross_fitted`, `multi_arm`, `missing_outcome` | coverage and SE ratio inside the calibration bands, with a correct outcome regression |
| `clustered_inference` | `cluster_robust`, `iid_control` | 100 clusters of 20 rows, with treatment and outcome correlated inside a cluster. The `iid_control` row uses the IID variance of the same influence curve on the same fits, and it must miss the bands |
| `type_i_error` | `sharp_null` | one-sided rejection under a confounded law with a zero effect |
| `power` | `alternative` | rejection under a law with a real effect |
| `root_n_and_efficiency` | `n_500`, `n_2000`, `n_8000` | bias, coverage and SE ratio at three sizes |
| `root_n_rate` | `empirical_sd`, `reported_se` | the log-log slope of the spread and of the reported error against $n$ |
| `double_robustness` | `treatment_correct`, `outcome_correct`, `both_wrong`, `multi_arm_treatment_correct`, `missing_outcome_treatment_correct`, `cross_fitted_treatment_correct` | bias inside the margin when one nuisance is correct. The `both_wrong` control must fall outside the margin |
| `targeting_necessity` | `ate__targeted`, `ate__untargeted` | the update removes the bias of a misspecified outcome GLM that omits `W2`. The untargeted plug-in must be biased |

The cells `multi_arm_treatment_correct`, `missing_outcome_treatment_correct` and
`cross_fitted_treatment_correct` fit a wrong outcome regression in the three-arm, missing-outcome
and cross-fitted designs. The update then carries the estimate, so each cell answers to the union
model with a correct treatment law.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `clustered_inference` | `cluster_robust` | positive | five-fold point-treatment TMLE with cluster-robust ATE inference | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9250 to 0.9507, SE ratio 0.9390 to 1.0117, paired coverage gain 0.1517 to 0.1913 | pass |
| `clustered_inference` | `iid_control` | control | the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7441 to 0.7890, SE ratio 0.5890 to 0.6341, paired coverage gain 0.1517 to 0.1913 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0142 to 0.0190, margin 0.0081, SE ratio 0.8636 | pass |
| `double_robustness` | `cross_fitted_treatment_correct` | positive | five cross-fitting folds, a consistent lasso mechanism, and an outcome GLM that omits a confounder | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0013 to 0.0021, margin 0.0057, SE ratio 1.0004 | pass |
| `double_robustness` | `missing_outcome_treatment_correct` | positive | outcomes missing at random with a correct response learner, and an outcome GLM that omits a confounder | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0029 to 0.0011, margin 0.0067, SE ratio 0.9963 | pass |
| `double_robustness` | `multi_arm_treatment_correct` | positive | three arms with a multinomial lasso and a constant outcome regression, on the contrast of arm 1.0 against arm 0.0 | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0017 to 0.0017, margin 0.0069, SE ratio 1.0783 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0021 to 0.0021, margin 0.0070, SE ratio 0.8746 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000689 to 0.0027, margin 0.0056, SE ratio 1.0898 | pass |
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
| `targeting_necessity` | `ate__targeted` | positive | average treatment effect: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0029 to 0.000440, margin 0.0057 | pass |
| `targeting_necessity` | `ate__untargeted` | control | average treatment effect: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0549 to 0.0581, margin 0.0053 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0458, 0.0355 to 0.0580 | pass |
<!-- /generated -->

## The ordinary TMLE on the same fit

Each primary row and each calibration, clustered, null, power and robustness row carries the
ordinary `TMLE`. It uses the same initial outcome fit, the same mechanism and the same bounds.
Only the derivative columns are removed. The numbers in `paired-tmle.csv` are descriptive, and no
verdict reads them.

| scenario | mean C-TMLE0 minus TMLE | SD of the difference | coverage, C-TMLE0 and TMLE | mean standard error, C-TMLE0 and TMLE |
| --- | --- | --- | --- | --- |
| `binary_declared_pair` | 0.000009 | 0.0018 | 0.9275 and 0.9338 | 0.02207 and 0.02219 |
| `binary_pipeline` | 0.000015 | 0.0020 | 0.9525 and 0.9550 | 0.02201 and 0.02214 |
| `binary_constant_q` | -0.000555 | 0.0052 | 0.9688 and 0.9713 | 0.02464 and 0.02484 |

With the correct GLM, the two estimators agree on average to $10^{-5}$. The standard deviation of
their difference is under one tenth of the empirical standard deviation of the estimates, 0.0224
to 0.0237. With the constant outcome regression, the difference has a mean of $-5.6 \times
10^{-4}$ and a standard deviation of 0.0052. That value is about 0.23 times the empirical standard
deviation of 0.0228. The derivative columns move the estimate there, and the interval stays close
to the ordinary TMLE interval. The study does not test the claim that C-TMLE0 repairs an
inconsistent outcome regression.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications per scenario |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 3 | implementation-scenario tests against truth |
| `independent_tests_passed` | 3 | of those, passing |
| `property_cells_total` | 21 | independent property cells |
| `property_cells_passed` | 21 | of those, passing |
| `max_standardized_bias` | 0.0401 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9275 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9006 | lowest exact 99% coverage endpoint |
| `min_se_ratio_ci_lower` | 0.8783 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1511 | highest bootstrap SE-ratio endpoint |
| `properties[clustered_inference/cluster_robust]:coverage` | 0.9387 | cluster-robust coverage |
| `properties[clustered_inference/cluster_robust]:se_ratio` | 0.9742 | cluster-robust SE ratio |
| `properties[clustered_inference/iid_control]:coverage` | 0.7671 | coverage of the IID variance on the same fits |
| `properties[clustered_inference/iid_control]:se_ratio` | 0.6107 | SE ratio of the IID variance on the same fits |
| `properties[double_robustness/both_wrong]:bias` | 0.0166 | both-wrong control bias |
| `properties[double_robustness/both_wrong]:bias_ci_lower` | 0.0142 | its 99% lower endpoint, against the margin below |
| `properties[double_robustness/both_wrong]:bias_margin` | 0.0081 | the equivalence margin that cell answers to |
| `properties[double_robustness/both_wrong]:standardized_bias` | 0.5093 | its bias in empirical standard deviations |
| `properties[double_robustness/outcome_correct]:bias` | -0.000006 | bias with only the outcome regression correct |
| `properties[double_robustness/treatment_correct]:bias` | 0.000990 | bias with only the treatment mechanism correct |
| `properties[interval_calibration/in_sample]:coverage` | 0.9504 | in-sample calibration coverage |
| `properties[interval_calibration/in_sample]:se_ratio` | 0.9865 | in-sample calibration SE ratio |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0458 | rejection under the confounded sharp null |
| `properties[power/alternative]:rejection_rate` | 0.9208 | rejection under the alternative |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4763 | fitted empirical-spread rate |
| `properties[root_n_rate/reported_se]:slope` | -0.4871 | fitted reported-SE rate |
| `properties[targeting_necessity/ate__untargeted]:bias` | 0.0565 | bias of the initial plug-in estimate |
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
| `margin:paired_difference` | 0.1500 | paired similarity margin in empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio non-inferiority margin |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference non-inferiority margin |
| `margin:calibration_noninferiority` | 0.0500 | SE-calibration non-inferiority margin |
| `margin:minimum_power` | 0.8000 | power-control rejection lower bound |
| `margin:root_n_slope` | -0.5000 | predicted root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:targeting_displacement` | 1.5000 | least the fluctuation must move the estimate |
| `margin:iid_control_se_ceiling` | 0.8000 | IID-control SE-ratio ceiling |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |

## Limitations

The [red-cell ledger](red-cells.md) lists no red cell of this study.

| limitation | what it means for use |
| --- | --- |
| The R comparison is pending | The page holds Python property evidence, with comparator runs pending. No paired row, no parity result and no equivalence verdict exists. The R runs wait for beta-plans TRACKER item 18. The comparator does not block the method |
| The parity witness waits for R | The witness compares the Python minus R difference with the C-TMLE0 minus TMLE difference on `binary_declared_pair` and `binary_constant_q`. Its columns in `paired-tmle.csv` are empty until the R runs exist. `tests/unit/test_ctmle_derivative_study_design.py` skips the witness mutation test while the comparator is pending |
| The two pipelines will differ in three ways | `binary_pipeline` differs from R in the cross-validation folds, the partner and the grid. When R lands, agreement there is agreement of two pipelines. `binary_declared_pair` fixes both sides at one pair, so only that scenario and `binary_constant_q` are parity of one estimator |
| The declared pair comes from package draws | The pair is the lower median of 50 package pipeline fits on the primary law. R does not choose it. The pair is a single point in the penalty path |
| The parity claim will be narrow | It is binary, two-arm, complete-outcome, non-cross-fitted ATE with a lasso mechanism pair. The cross-fitted, three-arm, missing-outcome and clustered fits have property cells and no R comparator |
| The interval is measured under a correct outcome regression | The primary law has a correct GLM outcome and a consistent lasso mechanism. The wrong-outcome cells have a correct treatment law. The study does not test the collaborative claim, an interval under an inconsistent outcome regression. [F18](../../roadmap.md#f18-selector-path-c-tmle-inference) holds that claim. The sources report low coverage there |
| The comparison with the ordinary TMLE is descriptive | The paired TMLE rows read no verdict. They show how far the derivative columns move the estimate and the interval. They do not show an advantage of the derivative covariate |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/README.md)
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/manifest.json)
record the provenance and the regeneration commands. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_derivative/properties.csv)
carry every published row.
