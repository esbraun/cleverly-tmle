# CV-TMLE of the fold-local learned-rule value

This study validates the interval of `LearnedRuleValue`, which the engine spells
`TMLE(learned_rule=LearnedRule())`. The fit learns the plug-in rule of each fold's outcome
regression on that fold's training rows. It targets the rules with one pooled fluctuation, averages
the fold plug-ins with the weight $1/V$, and reports the cross-validated variance.
[Learned rules](../point-treatment-tmle.md#learned-rules) gives the contract and the conditions
C1 to C6.

The target is a data-adaptive parameter. It depends on the realized split and on the fitted
rules, so each replication carries its own truth. The record sets `truth_varies_by_replicate`, and
every statistic reads the error, the estimate minus the replication's truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** The table gives each candidate and why this study
rejects it. A zero-row equivalence artifact records the absence of a comparator.

| candidate | why it is not a comparator |
| --- | --- |
| R `tmle3mopttx` at `3c0e843` | it weights the folds by $n_v / n$ and centres the curve at the pooled estimate. Its `sl3` metalearner reads every fold's predictions, so a fold's rule is not a function of that fold's training rows alone |
| R `polle` | its Algorithm 4 pools doubly robust scores, a different construction that [X11](../../roadmap.md#x11-learned-policy-follow-ups) part (e) holds |
| R `DynTxRegime` | it estimates a rule and its value by IPW or AIPW, with no fold-average CV-TMLE target |

## What was tested

| setting | declaration |
| --- | --- |
| law | $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and $Y \sim \operatorname{Bernoulli}(\bar Q_0(A, W))$ with $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W)$ |
| fit | `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")` at n = 2,000, so each fold holds 200 rows |
| outcome learner | `Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), LogisticRegression(C=1e6, max_iter=5000, tol=1e-10))` on $(A, W_1, W_2)$ |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, a correct parametric model |
| primary budget | 6,000 replications for each law |
| truth | each replication refits each fold's outcome learner on the fit's own training rows. The refit rule must equal the fit's rule on every validation row. The truth is $(1/V) \sum_v \Psi_{d_{nv}}(P_0)$, by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$, with the known $\bar Q_0$ |
| oracle SE | $\sqrt{V^{-2} \sum_v \operatorname{Var}_{P_0} D^*(d_{nv}, \bar Q_0, g_0) / n_v}$ for each replication. It is a descriptive column of `harness.csv.gz`, and no rule reads it |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20263001` and `resampling_seed=20263002` |
| policy | `gated` |

The table gives the two laws. $d_0$ is the optimal rule. $d_1$ is the rule of the learner's limit
blip, which a quadrature of the population logistic fit computes.

| law | $b(W)$ | $d_0$ treats | $d_1$ treats | $\Psi_{d_0}(P_0)$ | $\Psi_{d_1}(P_0)$ | condition C3 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `non_exceptional` | $0.1 + W_1$ | $W_1 > -0.1$ | $d_0$ | 0.577115 | 0.577115 | holds, with $d_1 = d_0$ |
| `misspecified_limit` | $0.8 W_1 + W_1^2 - 0.3$ | $W_1 > 0.278233$ | $W_1 > 0.005025$ at $W_2 = 0$, and $W_1 > 0.001387$ at $W_2 = 1$ | 0.560251 | 0.554747 | holds, with $d_1 \ne d_0$ |

The outcome learner is wrong at `misspecified_limit`, and the treatment learner is correct. That
cell therefore rests on Corollary 3 of van der Laan and Luedtke (2015), which gives a conservative
interval. The Corollary 3 projection term predicts an SE ratio of 1.0002 there. The cell tests
coverage, and it does not test conservativeness.

The [RM30 record](../../roadmap.md#rm30-learned-policy-value-evaluation) names the commit that
holds the declaration. That declaration fixed the laws, the learners, the budgets, the seeds, the
truth rule, the failed-fit rule and the run form before any run.
`tests/unit/test_learned_rule_study_constants.py` computes every constant of the law tables again
by quadrature, to 1e-5.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, a misspecified outcome learner | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated CV-TMLE of the learned-rule value | -0.000765 to 0.000251 | 0.9413 | 0.9810 | pass |
| learned-rule law with the blip 0.1 + W1, correctly specified learners | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated CV-TMLE of the learned-rule value | -0.000458 to 0.000537 | 0.9465 | 0.9972 | pass |
<!-- /generated -->

## Theory properties

Every property cell draws the `non_exceptional` law with the declared learners, unless the table
names another learner.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of the positive cell lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, at 17,000 replications | a shrunken-SE control multiplies each SE by 0.70. A noise control adds noise with SD 0.664444 / $\sqrt{2000}$ to each estimate. Each control's SE-ratio interval must lie below 0.93 |
| `root_n_and_efficiency` | do bias, coverage and the SE ratio pass at n = 2,000 and 8,000, with n = 500 as the control side, at 6,000 replications each | the `root_n_rate` slopes of the error SD and of the mean SE must lie inside -0.625 to -0.375 and exclude -0.25 |
| `targeting_necessity` | is the targeted fit unbiased when the outcome learner omits $W_1$ | the same fit without its fluctuation must miss the bias margin, and the paired displacement must reach 0.5 SD. 2,655 replications |
| `fold_locality` | is the fit unbiased with a random forest outcome learner, `RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0)` | a study-only subclass learns each fold's rule on that fold's validation rows. Its bias must miss the margin, and the paired displacement must reach 0.5 SD. 2,655 replications. Each arm's truth is the value of the rules that it used |

The `targeting_necessity` and `fold_locality` families read bias only. Their positive arms publish
a coverage and an SE ratio, and no verdict reads either one.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `fold_locality` | `non_exceptional__fold_local` | positive | the learned-rule law with a blip that is nowhere zero: a random forest learns each fold's rule on the fold's training rows | bias interval inside the equivalence margin | bias -0.0011 to 0.000743, margin 0.0046 | pass |
| `fold_locality` | `non_exceptional__validation_rows` | control | the learned-rule law with a blip that is nowhere zero: the same fit, with each fold's rule learned on the fold's own validation rows | bias interval must fall entirely outside the margin | bias 0.1188 to 0.1202, margin 0.0035 | pass |
| `interval_calibration` | `non_exceptional__correctly_specified` | positive | the learned-rule law with a blip that is nowhere zero: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9438 to 0.9526, SE ratio 0.9849 to 1.0134 | pass |
| `interval_calibration` | `non_exceptional__noise_control` | control | the learned-rule law with a blip that is nowhere zero: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8310 to 0.8456, SE ratio 0.6992 to 0.7196 | pass |
| `interval_calibration` | `non_exceptional__shrunken_se_control` | control | the learned-rule law with a blip that is nowhere zero: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8253 to 0.8401, SE ratio 0.6895 to 0.7093 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000025, coverage 0.9450 to 0.9593, SE ratio 0.9956 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.000034, coverage 0.9306 to 0.9467, SE ratio 0.9573 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000117, coverage 0.9377 to 0.9529, SE ratio 0.9842 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5283 to -0.5039 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5064 to -0.5057 | pass |
| `targeting_necessity` | `non_exceptional__targeted` | positive | the learned-rule law with a blip that is nowhere zero: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0011 to 0.000741, margin 0.0045 | pass |
| `targeting_necessity` | `non_exceptional__untargeted` | control | the learned-rule law with a blip that is nowhere zero: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0203 to 0.0217, margin 0.0036 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The interval covers the fold-average target at
the two laws that meet the published conditions, with the declared learners at n = 2,000.

| reading | value | source |
| --- | --- | --- |
| primary coverage, `non_exceptional` | 0.9465 | `performance-tests.csv` |
| primary coverage, `misspecified_limit` | 0.9413 | `performance-tests.csv` |
| rule rows that the truth harness checked | 2,000 in each of 12,000 primary replications | `harness.csv.gz`, `rule_rows_checked` |
| solver warnings, primary | 0 | `harness.csv.gz`, `solver_warnings` |
| solver warnings, property | 18, on 9 rows of `interval_calibration` and `targeting_necessity` | `property-replicates.csv.gz`, `solver_warnings` |

The 9 property rows come from four draws. Calibration replicate 12314 warned, and its two controls
reuse its rows. `targeting_necessity` replicates 100, 509 and 2018 warned, and each pair of arms
shares one fit. Rule L4 of the declaration counts a solver warning and does not treat it as a
failed fit. No rule reads the count. A harness refit whose rule differed from the fit's rule would have stopped the
run, and the run completed.

The readings do not show the items in this list.

- Coverage with a flexible outcome learner. The forest arm covers 0.9115 with an SE ratio of
  0.8542, and the `W1`-dropping arm covers 0.9175 with an SE ratio of 0.9054. Neither number is a
  verdict, and both come from 2,655 replications.
- Coverage at an exceptional law or at a weak blip. The
  [boundary study](learned-rule-cvtmle-boundary.md) reads both laws, and it under-covers at each.
- The value of the optimal rule, or the value of one rule fitted on all rows.

## Measured values

Names beginning `margin:` are thresholds declared before the run. The name beginning `bound:` is
the SD of the efficient influence curve of the `non_exceptional` law over $\sqrt{n}$. The record
keys that bound by the law `non_exceptional`, and not by an estimand, so the name carries the law.
It sizes the noise control only. Everything else is measured from the committed results and
checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 6000 | primary replications per law |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | law tests against the replication truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 12 | repeated-sampling property cells |
| `property_cells_passed` | 12 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0168 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9413 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9331 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9590 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0214 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-cvtmle/non_exceptional/ey_learned_rule[learned rule]]:coverage` | 0.9465 | primary coverage at `non_exceptional` |
| `performance[cleverly-learned-rule-cvtmle/misspecified_limit/ey_learned_rule[learned rule]]:coverage` | 0.9413 | primary coverage at `misspecified_limit` |
| `performance[cleverly-learned-rule-cvtmle/misspecified_limit/ey_learned_rule[learned rule]]:se_ratio` | 0.9810 | primary SE ratio at `misspecified_limit`, against the predicted 1.0002 |
| `properties[interval_calibration/non_exceptional__correctly_specified]:se_ratio_ci_lower` | 0.9849 | calibration SE-ratio lower endpoint, against 0.93 |
| `properties[interval_calibration/non_exceptional__correctly_specified]:se_ratio_ci_upper` | 1.0134 | calibration SE-ratio upper endpoint, against 1.07 |
| `properties[interval_calibration/non_exceptional__correctly_specified]:coverage_ci_lower` | 0.9438 | calibration coverage lower endpoint, against 0.92 |
| `properties[interval_calibration/non_exceptional__correctly_specified]:coverage_ci_upper` | 0.9526 | calibration coverage upper endpoint, against 0.98 |
| `properties[interval_calibration/non_exceptional__shrunken_se_control]:se_ratio_ci_upper` | 0.7093 | shrunken-SE control upper endpoint, which must lie below 0.93 |
| `properties[interval_calibration/non_exceptional__noise_control]:se_ratio_ci_upper` | 0.7196 | noise control upper endpoint, which must lie below 0.93 |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5160 | fitted slope of the log error SD on log n |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5283 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.5039 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5060 | fitted slope of the log mean SE on log n |
| `properties[targeting_necessity/non_exceptional__untargeted]:standardized_bias` | 1.4716 | bias of the fit without its fluctuation, in error SDs |
| `properties[targeting_necessity/non_exceptional__targeted]:targeting_displacement` | 1.1762 | paired displacement, against 0.5 |
| `properties[targeting_necessity/non_exceptional__targeted]:coverage` | 0.9175 | coverage of the targeted arm, which no verdict reads |
| `properties[targeting_necessity/non_exceptional__targeted]:se_ratio` | 0.9054 | SE ratio of the targeted arm, which no verdict reads |
| `properties[fold_locality/non_exceptional__validation_rows]:standardized_bias` | 8.4371 | bias of the validation-row rules, in error SDs |
| `properties[fold_locality/non_exceptional__fold_local]:fold_locality_displacement` | 6.5179 | paired displacement, against 0.5 |
| `properties[fold_locality/non_exceptional__fold_local]:coverage` | 0.9115 | coverage of the forest arm, which no verdict reads |
| `properties[fold_locality/non_exceptional__fold_local]:se_ratio` | 0.8542 | SE ratio of the forest arm, which no verdict reads |
| `bound:non_exceptional_standard_error` | 0.0149 | the efficiency bound of the law `non_exceptional` at n = 2,000 |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control applies to each SE |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:targeting_displacement` | 0.5000 | smallest paired displacement of `targeting_necessity`, in positive-arm error SDs |
| `margin:fold_locality_displacement` | 0.5000 | smallest paired displacement of `fold_locality`, in positive-arm error SDs |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits this construction. The row rests on accuracy against the replication truth and on the theory properties |
| Two regular laws only | Both primary laws meet condition C3. At an exceptional law and at a weak blip the interval under-covers, as the [boundary study](learned-rule-cvtmle-boundary.md) reads |
| One learner class for each cell | The primary and calibration cells use a logistic outcome model with treatment interactions. The forest and the `W1`-dropping learners enter bias verdicts only, so the row does not claim coverage with them |
| A binary treatment and a binary outcome | A continuous outcome with declared `q_bounds` and every other design are not tested. The package refuses more than two arms, weights, clusters, missing outcomes, strata, repeats and the full-refit bootstrap |
| Equal folds | Each fold holds 200 rows, so the weights $1/V$ and $n_v / n$ agree and the two variance forms coincide. `tests/unit/test_learned_rule_variance.py` covers unequal folds |
| One primary size | The primary laws run at n = 2,000. The root-n ladder runs at 500, 2,000 and 8,000 on `non_exceptional` only |
| Other targets are not tested | The row does not test the value of the optimal rule, or the value of one rule fitted on all rows |
| The truth harness needs deterministic learners | The truth refits each fold's learner. A learner that draws its own randomness cannot give the same rule again |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle/replicates.csv.gz),
the [harness columns](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_cvtmle/properties.csv)
carry every published row.
