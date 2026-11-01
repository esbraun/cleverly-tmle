# Learned treatment rules over three arms

This study validates the interval of the cross-fitted learned-rule value on a treatment with three
arms. The estimator is `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True,
targeting_scheme="pooled", g_bounds="auto")`. Each fold learns the plug-in rule
$d_v(w) = \arg\max_a \bar Q_v(a, w)$ of its outcome regression on its training rows, over the arms
0, 1 and 2, and a tie goes to the lowest arm code. The clever covariate is
$\mathbb 1\{A = d_v(W)\} / g_v(A \mid W)$ with the multinomial mechanism, and one pooled
fluctuation targets every fold. The fit reports `ey_learned_rule[learned rule]` under the
fold-evaluated report.
[More than two arms](../point-treatment-tmle.md#learned-rules) gives the construction and the
no-tie form of the limiting-rule condition it rests on.

The target is data-adaptive: $\tilde\psi_{0n} = V^{-1}\sum_v \Psi_{d_v}(P_0)$, the value of the
rules the folds learned. Each replication carries its own truth, and every statistic reads the
error, the estimate minus that truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** No maintained package fits a cross-validated TMLE of
a learned rule over more than two arms. A zero-row equivalence artifact records the absence of a
comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law | `three_arms`: $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$; $g_0(a \mid W) \propto \exp(\eta_a)$ with $\eta_0 = 0$, $\eta_1 = 0.7 W_1 + 0.2 W_2$ and $\eta_2 = -0.4 W_1 + 0.2 W_2$, so every arm probability is at least 0.166; $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + b_a(W)$ with $b_0 = 0$, $b_1 = -1 + 6 W_1$ and $b_2 = -1.75 - 6 W_1 - W_2 - 4 (W_1^2 - 1/3)$ |
| fit | `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")` at n = 2,000 with `repeats=1` and pointwise intervals |
| outcome learner | `ArmLogistic(terms)`: an unpenalized logistic regression (`C=inf`, `tol=1e-10`) on fixed products of the design $(A_1, A_2, W_1, W_2)$, marked `cleverly_learner_family = "parametric"`. `declared` holds every arm's intercept and $W_1$ and $W_2$ slopes and $A_2 W_1^2$, a correct model; `missing_quadratic` drops $A_2 W_1^2$; `drop_w1` keeps the arm indicators, $W_2$ and their products |
| treatment learner | `LogisticRegression(C=inf, max_iter=5000)` on $(W_1, W_2)$, the multinomial maximum-likelihood fit of a correct model |
| support key | family `P11`, `treatment` `("arms3",)`, `splitting` `("fold_evaluated",)`, pinned by `test_the_fit_keys_three_arms_fold_evaluated` in `tests/unit/test_learned_rule_arms_harness.py` |
| truth | per replication: refit each fold's outcome learner on the fit's training rows with the package's `fit_on_rows`, take the argmax at the three arms, require it to equal the fit's rule on every validation row and the stored fold rule on every row, and integrate $\bar Q_0(d_v(W), W)$ by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$; weight $1/V$ |
| stored variance | $V^{-2}\sum_v n_v^{-2}\sum_{i\in v} D_i^2$ of the stored curve. The interval is $\hat\psi \pm 1.96\,\mathrm{SE}$ |
| primary budget | 1,200 replications of the `declared` fit |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20268111` and `resampling_seed=20268112` |
| policy | `gated` |

A law cell gates only when its outcome learner's population limit meets a margin condition
declared before any draw: every arm's share of the limit rule is at least 0.10, and the top two
arms of the limit lie within 0.01 of each other on at most 2% of the population. Quadrature on the
law gives each limit.

| cell | outcome learner | arm shares of its limit rule | top-two gap below 0.01 | differs from the optimal rule | role |
| --- | --- | --- | --- | --- | --- |
| `three_arms` | `declared` | 0.175, 0.417, 0.408 | 1.59% | 0 | primary and `interval_calibration` |
| `missing_quadratic` | `missing_quadratic` | 0.2455, 0.417, 0.338 | 1.83% | 7.05% | `conservative_coverage` |
| `drop_w1` | `drop_w1` | 0, 1, 0 | 0 | not gated | `targeting_necessity` |

The `missing_quadratic` model is misspecified in arm 2's regression only. Every term interacts
fully with the arm, so the population score equations decouple by arm, and the limits of arms 0
and 1 are exact. The 7.05% disagreement is where the wrong arm-2 curve moves the argmax. The
`drop_w1` limit assigns arm 1 everywhere, so the gate refuses it on its shares, and it serves as
the necessity control. The law, the learners, the budgets, the seeds, the truth rule, the margin
condition and the support-row condition were fixed before any run, in the X11 plan, its review,
the implementation review and the study design record.

`tests/unit/test_learned_rule_arms_design.py` pins each limit, the efficiency bound, the predicted
SE ratio, the sizing and the seeds, and `tests/unit/test_learned_rule_arms_harness.py` checks the
fits, the truth, the controls and the property path. `tests/unit/test_learned_rule_arms.py` holds
the unit witnesses of the construction.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-arm law with a multinomial logistic treatment mechanism, a correct arm-interacted logistic outcome learner, fold-evaluated report | `ey_learned_rule[learned rule]` | fold average of the values of the three-arm rules learned on each training fold | `cleverly` cross-validated TMLE of the learned-rule value over three arms: each fold's rule the argmax of its outcome model over the arms, a tie to the lowest arm code, one pooled fluctuation and the fold-evaluated report | -0.0013 to 0.000915 | 0.9467 | 0.9974 | pass |
<!-- /generated -->

## Theory properties

Every property cell fits n = 2,000 with the declared treatment learner on the `three_arms` law.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of the `declared` fit lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, with 3,200 replications | a shrunken-SE control multiplies each SE by 0.80, and a noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. Each control's SE-ratio interval must fall below the band |
| `conservative_coverage` | with the `missing_quadratic` learner, does the positive keep its bias inside the margin and the lower end of its exact coverage interval at or above 0.92, with 1,200 replications. The SE ratio is published and no verdict reads it | the control multiplies each SE by 0.60, and its coverage interval must lie wholly below 0.92 |
| `targeting_necessity` | does the fluctuation remove the plug-in's bias with the `drop_w1` learner, with 1,200 replications | the control is the same fit's untargeted fold average of the initial plug-ins. The positive's bias interval must lie inside the margin, the control's outside it, and the two means must sit 0.5 positive-arm SDs or more apart |
| `rule_margin_diagnostic` | for each law cell, how far the fitted rules sit from the outcome learner's limit rule: the mean share of rows whose rule differs from it, the mean share whose own fold's top two arms lie within 0.01, the mean smallest arm share, and the share of draws that leave an arm with no rows | nothing fails. The cells have the role `diagnostic`, and no margin is read |

The noise scale is the SD of $\sqrt{n}$ times the estimate at the optimal rule, 0.670220 by
quadrature, and no verdict reads it as an efficiency bound. The sizes were set before the run from
a 4,000-draw pilot of every cell, and a second 4,000-draw pilot of the two law cells in the design
review gave the same counts. The coverage floor 0.92 is the calibration band's lower end. The
control's shrink of 0.60 was declared before the pilot: at a true SE ratio of 1.30 the shrunken
interval covers about 0.87, still below the floor.

The `drop_w1` learner fits the cell means of $(A, W_2)$, so its untargeted plug-in carries the
confounding through $W_1$. At its limit rule the plug-in sits 0.129 above that rule's value,
9.4 SEs at n = 2,000. Its fold rules have not settled at this size: at $W_2 = 0$ the limit gap
between arms 1 and 0 is 0.0245, about 0.6 SDs of a training fold's difference of cell means. Each
fold's target is the value of its own learned rule, so a fold that learns another rule does not
bias the targeted value. The pair therefore reads bias only, and it tests that the fluctuation
removes the confounded plug-in's bias, not the multi-arm argmax at its limit, which the two law
cells carry.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `conservative_coverage` | `missing_quadratic__declared` | positive | the three-arm law with the arm-interacted logistic outcome learner without arm 2's quadratic term, so arm 2's regression is misspecified and arms 0 and 1 are exact: the declared fit with the correct multinomial logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.0019 to 0.000403, margin 0.0039, coverage 0.9201 to 0.9563, SE ratio 0.9960 | pass |
| `conservative_coverage` | `missing_quadratic__shrunken_coverage_control` | control | the three-arm law with the arm-interacted logistic outcome learner without arm 2's quadratic term, so arm 2's regression is misspecified and arms 0 and 1 are exact: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7294 to 0.7935, SE ratio 0.5976 | pass |
| `interval_calibration` | `three_arms__declared` | positive | the three-arm law with the arm-interacted logistic outcome learner, a correct model: the declared fit with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9311 to 0.9526, SE ratio 0.9559 to 1.0179 | pass |
| `interval_calibration` | `three_arms__noise_control` | control | the three-arm law with the arm-interacted logistic outcome learner, a correct model: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8213 to 0.8551, SE ratio 0.6969 to 0.7423 | pass |
| `interval_calibration` | `three_arms__shrunken_se_control` | control | the three-arm law with the arm-interacted logistic outcome learner, a correct model: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8632 to 0.8932, SE ratio 0.7644 to 0.8154 | pass |
| `rule_margin_diagnostic` | `missing_quadratic__rule_margin` | diagnostic | the three-arm law with the arm-interacted logistic outcome learner without arm 2's quadratic term, so arm 2's regression is misspecified and arms 0 and 1 are exact: the declared fit's rules on every draw of the law cell, read against the outcome learner's limit rule | none. The cell reports the mean share of rows whose rule differs from the limit rule, the mean share whose own fold's top two arms lie within 0.01, the mean smallest arm share and the share of draws that leave an arm with no rows | limit disagreement 0.0383, near-tie share 0.0183, min arm share 0.2452, lost-arm share 0 | reported |
| `rule_margin_diagnostic` | `three_arms__rule_margin` | diagnostic | the three-arm law with the arm-interacted logistic outcome learner, a correct model: the declared fit's rules on every draw of the law cell, read against the outcome learner's limit rule | none. The cell reports the mean share of rows whose rule differs from the limit rule, the mean share whose own fold's top two arms lie within 0.01, the mean smallest arm share and the share of draws that leave an arm with no rows | limit disagreement 0.0364, near-tie share 0.0165, min arm share 0.1772, lost-arm share 0 | reported |
| `targeting_necessity` | `drop_w1__targeted` | positive | the three-arm law with an outcome learner that has no W1 term, the cell means of (A, W2): the declared fit with that learner, so targeting does the adjusting | bias interval inside the equivalence margin | bias -0.0030 to -0.000093, margin 0.0049 | pass |
| `targeting_necessity` | `drop_w1__untargeted` | control | the three-arm law with an outcome learner that has no W1 term, the cell means of (A, W2): the same fit's untargeted fold average of the initial plug-ins at its fold rules, with no fluctuation | bias interval must fall entirely outside the margin | bias 0.1170 to 0.1212, margin 0.0070 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed. The primary test reads a standardized
bias of -0.014, coverage of 0.9467 and an SE ratio of 0.997 over 1,200 draws. The calibration
positive reads an SE ratio of 0.986 and coverage of 0.9425 over 3,200 draws, both intervals inside
their bands. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 1 of 1 | `performance-tests.csv` |
| gated property cells passed | 7 of 7 | `properties.csv` |
| diagnostic cells, reported with no verdict | 2 | `properties.csv`, `rule_margin_diagnostic` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |
| calibration SE-ratio interval | 0.9559 to 1.0179 | `properties.csv`, `three_arms__declared` |
| calibration coverage interval | 0.9311 to 0.9526 | `properties.csv`, `three_arms__declared` |
| coverage-floor lower endpoint | 0.9201 | `properties.csv`, `missing_quadratic__declared` |

The support row was declared before the run: one P11 row with `treatment=("arms3",)` under the
fold-evaluated report. It needed the primary bias leg, coverage floor and SE sanity, every
`interval_calibration` verdict, both `conservative_coverage` verdicts and the `targeting_necessity`
verdicts to pass. They did, so the registered row `T45` covers the plug-in rule over three arms
under the fold-evaluated report on rows. Four or more arms, weights, clusters, strata, fold
targeting, several splits, the stacked report, the in-sample fit and known regimes beside the
rule over three arms stay experimental.

### The margin of the fitted rules

The margin condition holds at both learners' limits, and the fitted rules stay close to those
limits. With the correct model the fitted rules differ from the optimal rule on 3.6% of rows on
average, 1.6% of rows have their fold's top two arms within 0.01, and the smallest arm share
averages 0.177. With `missing_quadratic` the same readings are 3.8%, 1.8% and 0.245 against its
own limit rule. No draw of either cell left an arm without rows.

### Coverage under a misspecified arm

With `missing_quadratic` the positive covers 0.940, with a lower endpoint of 0.9201, at a realized
SE ratio of 0.996. The cells measure coverage when arm 2's outcome regression omits its quadratic
term and arms 0 and 1 are exact at the limit. They do not show that the interval is conservative.
The curve omits the treatment model's projection term, and Corollary 2 of van der Laan and Luedtke
(2015, eq. 11) makes that variance conservative for a maximum-likelihood $g$ in a correct model,
which the paper extends to the cross-validated TMLE by analogy only. On this law the predicted SE
ratio of that term is 1.0004, so the study cannot tell that route from exact calibration.

### The necessity pair

The targeted `drop_w1` value reads a standardized bias of -0.079, and its untargeted plug-in
+4.23, so the two means sit 6.18 positive-arm SDs apart. The targeted arm's SE ratio of 0.948 and
coverage of 0.932 are a weak-margin reading of a rule that has not settled at n = 2,000. No
verdict reads them, and they are neither calibration evidence nor a claim of valid inference under
a misspecified outcome model.

### Solver warnings

The harness counts the scikit-learn and `cleverly` convergence warnings of each draw's fits and
truth refits. No primary, calibration or coverage draw warned. One necessity draw of 1,200
(replicate 1046) raised two warnings and was kept. The truth check found the fit's rule on every
validation row and the stored fold rule on every row of every draw. No rule reads the
warning counts.

## Measured values

Names beginning `margin:` are thresholds declared before the run. The name beginning `bound:` is
the SD of $\sqrt{n}$ times the estimate at the optimal rule over $\sqrt{2000}$. It sizes the noise
control only, and no verdict reads it as an efficiency bound. Everything else is measured from the
committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1200 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 1 | parameter tests against the truth |
| `independent_tests_passed` | 1 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 7 | gated repeated-sampling property cells |
| `property_cells_passed` | 7 | cells whose own and family verdicts pass |
| `property_cells_reported` | 2 | diagnostic cells, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0139 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9467 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9277 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9489 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0523 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-arms/three_arms/ey_learned_rule[learned rule]]:se_ratio` | 0.9974 | primary SE ratio |
| `summary[cleverly-learned-rule-arms/three_arms/ey_learned_rule[learned rule]]:coverage_se` | 0.0065 | primary Monte Carlo coverage SE |
| `summary[cleverly-learned-rule-arms/three_arms/ey_learned_rule[learned rule]]:truth_min` | 0.7230 | the smallest primary truth |
| `summary[cleverly-learned-rule-arms/three_arms/ey_learned_rule[learned rule]]:truth_max` | 0.7325 | the largest primary truth |
| `properties[interval_calibration/three_arms__declared]:se_ratio` | 0.9860 | calibration SE ratio |
| `properties[interval_calibration/three_arms__declared]:coverage` | 0.9425 | calibration coverage |
| `properties[interval_calibration/three_arms__shrunken_se_control]:se_ratio_ci_upper` | 0.8154 | shrunken-SE control upper endpoint, against 0.93 |
| `properties[interval_calibration/three_arms__noise_control]:se_ratio_ci_upper` | 0.7423 | noise-control upper endpoint, against 0.93 |
| `properties[conservative_coverage/missing_quadratic__declared]:coverage` | 0.9400 | coverage with `missing_quadratic` |
| `properties[conservative_coverage/missing_quadratic__declared]:se_ratio` | 0.9960 | its realized SE ratio, which no verdict reads |
| `properties[conservative_coverage/missing_quadratic__shrunken_coverage_control]:coverage_ci_upper` | 0.7935 | coverage-control upper endpoint, against 0.92 |
| `properties[targeting_necessity/drop_w1__targeted]:standardized_bias` | -0.0792 | necessity positive standardized bias |
| `properties[targeting_necessity/drop_w1__targeted]:se_ratio` | 0.9475 | its SE ratio, which no verdict reads |
| `properties[targeting_necessity/drop_w1__targeted]:coverage` | 0.9317 | its coverage, which no verdict reads |
| `properties[targeting_necessity/drop_w1__untargeted]:standardized_bias` | 4.2347 | necessity control standardized bias |
| `properties[targeting_necessity/drop_w1__targeted]:targeting_displacement` | 6.1835 | displacement between the arms, in positive-arm SDs, against 0.5 |
| `properties[rule_margin_diagnostic/three_arms__rule_margin]:mean_limit_disagreement` | 0.0364 | mean share of rows whose rule differs from the limit rule, correct model |
| `properties[rule_margin_diagnostic/three_arms__rule_margin]:mean_near_tie_share` | 0.0165 | mean share of rows whose fold's top two arms lie within 0.01, correct model |
| `properties[rule_margin_diagnostic/three_arms__rule_margin]:mean_min_arm_share` | 0.1772 | mean smallest arm share, correct model |
| `properties[rule_margin_diagnostic/three_arms__rule_margin]:lost_arm_share` | 0 | share of draws that leave an arm with no rows, correct model |
| `properties[rule_margin_diagnostic/missing_quadratic__rule_margin]:mean_limit_disagreement` | 0.0383 | the same, `missing_quadratic` |
| `properties[rule_margin_diagnostic/missing_quadratic__rule_margin]:mean_near_tie_share` | 0.0183 | the same, `missing_quadratic` |
| `properties[rule_margin_diagnostic/missing_quadratic__rule_margin]:mean_min_arm_share` | 0.2452 | the same, `missing_quadratic` |
| `properties[rule_margin_diagnostic/missing_quadratic__rule_margin]:lost_arm_share` | 0 | the same, `missing_quadratic` |
| `bound:three_arms_standard_error` | 0.0150 | the optimal-rule SD over the root of n |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit, and the `conservative_coverage` floor |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.8000 | factor the shrunken-SE control applies to each SE |
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
| `margin:targeting_displacement` | 0.5000 | least displacement the necessity pair must show |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 12 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a cross-validated TMLE of a learned rule over more than two arms. The row rests on accuracy against the truth and on the theory properties |
| One law, one size, one fold count and one learner per role | The study fits n = 2,000 on one three-arm law with ten equal folds, pooled targeting, `repeats=1`, an arm-interacted logistic outcome model, a correct multinomial logistic treatment model fitted by maximum likelihood and the plug-in rule, on a binary outcome, iid rows and no weights. The key holds neither n, the fold count nor the learner, so another n, V or learner reads supported and was not measured |
| Compositions outside the row | Four or more arms (`arms4` and above), weights, clusters, strata, fold targeting, several splits, the stacked report, the in-sample fit and known regimes beside the rule over three arms match no row and read experimental. The rule learners and the omitted-variable bound refuse over more than two arms |
| The no-tie condition is the package's own step | C3 for the argmax asks that the top two arms of the learner's limit tie only on a set of probability zero. The sources are silent on multi-arm exceptional laws, and the study measures two laws that meet the margin condition, not one near a tie |
| Coverage, not conservativeness, under a misspecified arm | With `missing_quadratic` the cells claim coverage at or above 0.92, at a realized SE ratio of 0.996, and no SE calibration. Arms 0 and 1 are exact at that limit, and the omitted projection term is negligible there (predicted SE ratio 1.0004), so the study does not test the conservative-variance route of Corollary 2, which the paper extends to the cross-validated TMLE by analogy only |
| One treatment learner | The coverage was measured only with a correct parametric $g$ fitted by maximum likelihood. A known $g$ (`treatment_probabilities=`, which keys `known_mechanism` and matches no row) and a super-learner $g$ were not measured |
| A necessity pair that reads bias only | The `drop_w1` fold rules have not settled at n = 2,000, so its targeted SE ratio of 0.948 and coverage of 0.932 are a weak-margin reading. They are not calibration evidence, and the pair does not show valid inference under a misspecified outcome model |
| Pointwise intervals | The study reads the pointwise interval of the one learned-rule value |
| One necessity draw with solver warnings | One `drop_w1` draw of 1,200 raised two convergence warnings and was kept. No other draw warned |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_arms/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_arms/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_arms/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_arms/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_arms/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_arms/properties.csv)
carry every published row.
