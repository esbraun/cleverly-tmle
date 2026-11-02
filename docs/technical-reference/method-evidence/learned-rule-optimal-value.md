# The value of the optimal rule

This study validates the interval of `ey_optimal_rule[learned rule]`, the value of the optimal
rule, at one law that meets the conditions of its licence. The estimator is
`TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"], n_folds=10, cv_evaluation=True,
targeting_scheme="pooled", g_bounds="auto")` with the known treatment mechanism passed as
`treatment_probabilities=`. Each fold learns the plug-in rule of its outcome regression on its
training rows, and one pooled fluctuation targets every fold. On the same data and seeds the point
and the interval are those of the fold-evaluated learned-rule fit; only the target differs.
[The value of the optimal rule](../point-treatment-tmle.md#learned-rules) gives the construction
and the conditions C3f, C7 and C8.

The target is the fixed parameter $\Psi_{d_0}(P_0) = E_0\,\bar Q_0(d_0(W), W)$, with
$d_0(w) = \mathbb 1\{\bar Q_0(1, w) > \bar Q_0(0, w)\}$ and a tie going to arm 0. Every verdict
reads the error against $\Psi_{d_0}$. The fit's own fold rules have the data-adaptive value
$\tilde\psi_{0n} = V^{-1}\sum_v \Psi_{d_v}(P_0)$, and the regret $\Psi_{d_0} - \tilde\psi_{0n}$ is
never negative. Each cell publishes the mean regret and its readings against $\tilde\psi_{0n}$
beside the verdicts, and no verdict reads them. Van der Laan and Luedtke (2015), Section 7.2,
last paragraph, license the interval for $\Psi_{d_0}$ when the regret is $o_P(n^{-1/2})$.

**No canonical implementation is compared.** No maintained package fits a cross-validated TMLE of
the value of the optimal rule. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$ and $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\,b(W)$, with three blips: `non_exceptional`, $b = 0.1 + W_1$; `wrong_limit`, $b = 3 (W_1^2 - 0.5)$; `partially_exceptional`, $b = (W_1 - 0.2)_+$ |
| fit | `TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"], n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")` with `repeats=1` and pointwise intervals |
| known $g$ | every cell but one passes `treatment_probabilities=[1 - g0(1 \| W), g0(1 \| W)]` |
| outcome learner | a logistic regression (`C=1e6`, `tol=1e-10`) on every pairwise product of $(A, W_1, W_2)$, a correct model of `non_exceptional`; on `partially_exceptional` the same six products and $A (W_1 - 0.2)_+$, a correct model of that law |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, fitted only in the estimated-$g$ cell |
| support key | family `P12`, `splitting` `("fold_evaluated",)`, qualifier `known_mechanism`, pinned by `tests/unit/test_learned_rule_optimal_value_harness.py` |
| truth | $\Psi_{d_0}$ by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$, one number per law: 0.577115, 0.557421 and 0.547362 |
| realized rule value | per replication: refit each fold's outcome learner on the fit's training rows with the package's `fit_on_rows`, require the fit's rule on every validation row, and integrate each fold rule's value by the same rule; $\tilde\psi_{0n}$ is the mean over the folds |
| stored variance | $V^{-2}\sum_v n_v^{-2}\sum_{i\in v} D_i^2$ of the raw fold curves, uncentred within each fold. The interval is $\hat\psi \pm 1.96\,\mathrm{SE}$ |
| primary budget | 1,600 replications of the known-$g$ `non_exceptional` fit at n = 2,000 |
| primary verdicts | the framework defaults on the error against $\Psi_{d_0}$: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20269151` and `resampling_seed=20269152` |
| policy | `gated` |

A law cell gates only when its learner's population limit meets three readings, declared before
any draw and computed by quadrature. Its limit has no tie mass (C7). Its limit rule differs from
$d_0$ off the tie set on at most 0.001 of the population (C3f). And
$\max_t P_0(0 < |b_0(W)| \le t)/t$ over $t$ in 0.005, 0.01, 0.02 and 0.05 is at most 10, a bounded
density of the blip at 0 (C8 with margin exponent 1). Each blip is smooth with a nonzero
derivative at its roots and $W_1$ is uniform, so the exponent itself holds analytically.

| law | cell | n | $d_0$ treats | tie mass | limit differs from $d_0$ | margin reading | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `non_exceptional` | `non_exceptional_known_g` | 2,000 | 0.55 | 0 | 0 | 4.07 | primary and `interval_calibration` |
| `wrong_limit` | `wrong_limit_known_g` | 2,500 | 0.293 | 0 | 0.293 | 1.965 | `optimal_target_control` |
| `non_exceptional` | `non_exceptional_estimated_g` | 2,000 | 0.55 | 0 | 0 | 4.07 | `estimated_g_report`, no verdict |
| `partially_exceptional` | `partially_exceptional_known_g` | 2,000 | 0.40 | 0.60 | 0 | 2.08 | `exceptional_law_report`, no verdict |

The `wrong_limit` learner's limit rule treats nobody, so its target converges to the value of that
rule, 0.512179, which is 0.0452 below $\Psi_{d_0}$. Its n of 2,500 is the smallest multiple of 500
at which that gap is at least three SEs of the limit rule's curve: 2,085 rows are needed, and
2,000 gives 2.94. Every other cell keeps n = 2,000. The hinge term of the `partially_exceptional`
learner makes its model correct, so C3f holds off the tie set and only C7 fails there. The laws,
the learners, the budgets, the seeds, the gate and the support-row condition were fixed before
any run, in the X11 plan, its review, the implementation review and the study design record.

`tests/unit/test_learned_rule_optimal_value_design.py` pins each limit, the gate and its
mutations, the efficiency bound, the control's n, the sizing and the seeds.
`tests/unit/test_learned_rule_optimal_value_harness.py` checks the fits, the known-$g$ key, the
truth, the regret, the rule refit and the property path. `tests/unit/test_learned_rule_optimal.py`
holds the unit witnesses of the construction.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, fold-evaluated report | `ey_optimal_rule[learned rule]` | value of the optimal rule, E Qbar0(d0(W), W) with d0(w) = 1{Qbar0(1, w) > Qbar0(0, w)}, a fixed parameter | `cleverly` cross-validated TMLE of the value of the optimal rule: the fold-evaluated learned-rule fit with the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value | -0.0026 to -0.000703 | 0.9444 | 0.9881 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | on `non_exceptional` with the known $g$ at n = 2,000, does the SE ratio of the declared fit lie inside 0.93 to 1.07, and its coverage of $\Psi_{d_0}$ inside 0.92 to 0.98, with 13,200 replications | a shrunken-SE control multiplies each SE by 0.80, and a noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. Each control's SE-ratio interval must fall below the band |
| `optimal_target_control` | on `wrong_limit` with the known $g$ at n = 2,500, does the interval miss $\Psi_{d_0}$ when the learner's limit rule is not $d_0$, with 1,200 replications | the control passes when its exact 99% coverage interval lies wholly below 0.92 |
| `estimated_g_report` | on `non_exceptional` with the treatment model fitted at n = 2,000, the bias, SE ratio and coverage of $\Psi_{d_0}$, with 1,200 replications | nothing fails. The cell has the role `diagnostic`, and no margin is read |
| `exceptional_law_report` | on `partially_exceptional` with the known $g$ at n = 2,000, the bias, SE ratio and coverage of $\Psi_{d_0}$, with 1,200 replications | nothing fails. The cell has the role `diagnostic`, and no margin is read |

The noise scale is the SD of $\sqrt{n}$ times the estimate at $d_0$ on `non_exceptional` with the
known $g$, 0.664438 by quadrature, and no verdict reads it as an efficiency bound. The gated counts
are the smallest multiples of 400, at least 1,000, whose predictive pass probability under the
declared rule reaches 0.95 on a 4,000-draw pilot and on 8,000 draws pooled with a second probe,
taking the larger count. The two reports take the study's floor of 1,200 draws, which no reading
moves. At that count a coverage reads to about 0.014 (95%), enough to show a gross failure and
not enough for a claim.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `estimated_g_report` | `non_exceptional_estimated_g__estimated_g` | diagnostic | the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and a fitted logistic treatment model, n = 2,000: the declared fit with the logistic treatment model fitted in place of the known g | none. The cell reports its bias, SE ratio and coverage against the value of the optimal rule, with the mean regret and the readings against the realized rule value | bias -0.0026 to -0.000369, coverage 0.9296 to 0.9634, SE ratio 0.9749, mean regret 0.0013, against the realized rule value coverage 0.9542 and SE ratio 0.9855, rule disagreement 0.0275 to 0.2135 | reported |
| `exceptional_law_report` | `partially_exceptional_known_g__exceptional` | diagnostic | the law with the blip (W1 - 0.2)_+, zero on 60% of the population, a correct outcome learner with the hinge term and the known treatment mechanism, n = 2,000: the declared fit with the known treatment mechanism | none. The cell reports its bias, SE ratio and coverage against the value of the optimal rule, with the mean regret, the readings against the realized rule value and the range of the fold-rule disagreement | bias -0.0021 to 0.000515, coverage 0.8885 to 0.9315, SE ratio 0.8878, mean regret 0.000865, against the realized rule value coverage 0.9192 and SE ratio 0.8931, rule disagreement 0 to 0.7350 | reported |
| `interval_calibration` | `non_exceptional_known_g__declared` | positive | the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, n = 2,000: the declared fit with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9439 to 0.9538, SE ratio 0.9818 to 1.0130 | pass |
| `interval_calibration` | `non_exceptional_known_g__noise_control` | control | the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8281 to 0.8447, SE ratio 0.7011 to 0.7239 | pass |
| `interval_calibration` | `non_exceptional_known_g__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8746 to 0.8891, SE ratio 0.7854 to 0.8102 | pass |
| `optimal_target_control` | `wrong_limit_known_g__target_gap_control` | control | the law with the even blip 3 (W1^2 - 0.5), whose interaction logistic outcome learner's limit rule treats nobody while the optimal rule treats 29.3%, with the known treatment mechanism, n = 2,500: the declared fit's own estimate and SE, read against the value of the optimal rule, 0.0452 above the value of the learner's limit rule | the coverage interval must lie wholly below 0.92 | coverage 0.0700 to 0.1134, SE ratio 1.0221, mean regret 0.0450, coverage of the realized rule value 0.9583 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. The primary test reads
a standardized bias of -0.111, coverage of 0.944 and an SE ratio of 0.988 over 1,600 draws. The
calibration positive reads an SE ratio of 0.997 and coverage of 0.949 over 13,200 draws, both
intervals inside their bands. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 1 of 1 | `performance-tests.csv` |
| gated property cells passed | 4 of 4 | `properties.csv` |
| diagnostic cells, reported with no verdict | 2 | `properties.csv`, `estimated_g_report` and `exceptional_law_report` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |
| primary bias interval | -0.0026 to -0.0007 | `performance-tests.csv` |
| calibration SE-ratio interval | 0.9818 to 1.0130 | `properties.csv`, `non_exceptional_known_g__declared` |
| calibration coverage interval | 0.9439 to 0.9538 | `properties.csv`, `non_exceptional_known_g__declared` |
| target-control coverage interval | 0.0700 to 0.1134 | `properties.csv`, `wrong_limit_known_g__target_gap_control` |

The support row was declared before the run: one P12 row under the fold-evaluated report with the
qualifier `known_mechanism`. It needed the primary bias leg, coverage floor and SE sanity, every
`interval_calibration` verdict and the `optimal_target_control` verdict to pass. They did, so the
registered row `T46` covers the value of the optimal rule under the fold-evaluated report on rows
with the known treatment mechanism. An estimated $g$, a rule learner, clusters, several splits,
fold targeting, the stacked report and more than two arms stay experimental.

### The regret

The interval covers $\Psi_{d_0}$ only up to the regret, which Lemma 1 makes $O_P(1/n)$ under these
conditions. In the calibration cell the mean regret is 0.00138, about 0.09 estimate SDs, and it is
the bias the verdicts read: standardized bias -0.088 against $\Psi_{d_0}$, +0.004 against
$\tilde\psi_{0n}$. The SE ratio is 0.997 against $\Psi_{d_0}$ and 1.006 against
$\tilde\psi_{0n}$, and the coverage 0.949 and 0.952. The primary bias interval excludes zero for
the same reason, and the regret's own spread widens the error's SD by about 0.9% at n = 2,000, so
it lowers the SE ratio against $\Psi_{d_0}$ by about that much. The study measures one n, so it
does not show the rate.

### The target control

At n = 2,500 the `wrong_limit` fit misses $\Psi_{d_0}$: coverage 0.09, standardized bias -3.32,
with an SE ratio of 1.022. The miss is the limit gap, a mean regret of 0.0450 against the
quadrature gap of 0.0452. Against its own $\tilde\psi_{0n}$ the same fit covers at 0.958 with an
SE ratio of 1.014, so the control fails because its learner targets another rule's value, not
because the interval is wrong for the value of the rules it learned.

### The estimated-$g$ reading

With the treatment model fitted, the `non_exceptional` fit reads a bias interval of -0.0026 to
-0.0004, an SE ratio of 0.975 and coverage of 0.948, with a 99% interval of 0.930 to 0.963, over
1,200 draws. Against $\tilde\psi_{0n}$ it reads coverage of 0.954 and an SE ratio of 0.986, with a
mean regret of 0.00135. The source states the result with $g$ known, and an estimated $g$ is the
package's step. This reading is published for decision E2 of the X11 plan, not as a validated
claim, and such a fit reads experimental.

### The partially exceptional law

Where the blip is zero on 60% of the population (C7 fails), the interval under-covers: coverage
0.912 with an SE ratio of 0.888 against $\Psi_{d_0}$, and 0.919 with 0.893 against
$\tilde\psi_{0n}$. The regret is small there, a mean of 0.00086, so it is not the cause. The fold
rules do not settle on the tie set: they disagree on 0.280 of rows on average, from 0 to 0.735.
That breaks the fixed-limit condition of Theorem 6, so even the data-adaptive target loses its
linearisation. No claim is made at this law.

No primary or property draw raised a solver warning, and the rule refit found the fit's rule on
every validation row of every draw.

## Measured values

Names beginning `margin:` are thresholds declared before the run. The name beginning `bound:` is
the SD of $\sqrt{n}$ times the estimate at $d_0$ over $\sqrt{2000}$. It sizes the noise control
only, and no verdict reads it as an efficiency bound. Everything else is measured from the
committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications |
| `n` | 2000 | observations per primary replication; the target control runs at 2,500 |
| `independent_tests_total` | 1 | parameter tests against the truth |
| `independent_tests_passed` | 1 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 4 | gated repeated-sampling property cells |
| `property_cells_passed` | 4 | cells whose own and family verdicts pass |
| `property_cells_reported` | 2 | diagnostic cells, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.1111 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9444 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9279 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9457 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0354 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-optimal-value/non_exceptional/ey_optimal_rule[learned rule]]:bias` | -0.0017 | primary bias against the optimal value |
| `performance[cleverly-learned-rule-optimal-value/non_exceptional/ey_optimal_rule[learned rule]]:bias_ci_lower` | -0.0026 | primary bias interval, lower end |
| `performance[cleverly-learned-rule-optimal-value/non_exceptional/ey_optimal_rule[learned rule]]:bias_ci_upper` | -0.000703 | primary bias interval, upper end |
| `performance[cleverly-learned-rule-optimal-value/non_exceptional/ey_optimal_rule[learned rule]]:se_ratio` | 0.9881 | primary SE ratio |
| `mean_oracle_se_non_exceptional` | 0.0149 | mean oracle SE at the primary fits' fold rules |
| `properties[interval_calibration/non_exceptional_known_g__declared]:se_ratio` | 0.9970 | calibration SE ratio |
| `properties[interval_calibration/non_exceptional_known_g__declared]:coverage` | 0.9490 | calibration coverage |
| `properties[interval_calibration/non_exceptional_known_g__declared]:standardized_bias` | -0.0882 | calibration standardized bias against the optimal value |
| `properties[interval_calibration/non_exceptional_known_g__declared]:mean_regret` | 0.0014 | calibration mean regret |
| `properties[interval_calibration/non_exceptional_known_g__declared]:rule_value_standardized_bias` | 0.0041 | calibration standardized bias against the realized rule value |
| `properties[interval_calibration/non_exceptional_known_g__declared]:rule_value_se_ratio` | 1.0061 | calibration SE ratio against the realized rule value |
| `properties[interval_calibration/non_exceptional_known_g__declared]:rule_value_coverage` | 0.9516 | calibration coverage of the realized rule value |
| `properties[interval_calibration/non_exceptional_known_g__declared]:mean_rule_disagreement` | 0.0768 | calibration mean fold-rule disagreement |
| `properties[interval_calibration/non_exceptional_known_g__shrunken_se_control]:se_ratio_ci_upper` | 0.8102 | shrunken-SE control upper endpoint, against 0.93 |
| `properties[interval_calibration/non_exceptional_known_g__noise_control]:se_ratio_ci_upper` | 0.7239 | noise-control upper endpoint, against 0.93 |
| `properties[optimal_target_control/wrong_limit_known_g__target_gap_control]:coverage` | 0.0900 | target-control coverage of the optimal value, at n = 2,500 |
| `properties[optimal_target_control/wrong_limit_known_g__target_gap_control]:coverage_ci_upper` | 0.1134 | its upper endpoint, against 0.92 |
| `properties[optimal_target_control/wrong_limit_known_g__target_gap_control]:standardized_bias` | -3.3165 | its standardized bias against the optimal value |
| `properties[optimal_target_control/wrong_limit_known_g__target_gap_control]:mean_regret` | 0.0450 | its mean regret |
| `properties[optimal_target_control/wrong_limit_known_g__target_gap_control]:rule_value_coverage` | 0.9583 | its coverage of the realized rule value |
| `properties[optimal_target_control/wrong_limit_known_g__target_gap_control]:rule_value_se_ratio` | 1.0140 | its SE ratio against the realized rule value |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:bias_ci_lower` | -0.0026 | estimated-$g$ bias interval, lower end |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:bias_ci_upper` | -0.000369 | estimated-$g$ bias interval, upper end |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:se_ratio` | 0.9749 | estimated-$g$ SE ratio |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:coverage` | 0.9483 | estimated-$g$ coverage |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:coverage_ci_lower` | 0.9296 | its coverage interval, lower end |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:coverage_ci_upper` | 0.9634 | its coverage interval, upper end |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:mean_regret` | 0.0013 | its mean regret |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:rule_value_coverage` | 0.9542 | its coverage of the realized rule value |
| `properties[estimated_g_report/non_exceptional_estimated_g__estimated_g]:rule_value_se_ratio` | 0.9855 | its SE ratio against the realized rule value |
| `properties[exceptional_law_report/partially_exceptional_known_g__exceptional]:coverage` | 0.9117 | partially exceptional coverage |
| `properties[exceptional_law_report/partially_exceptional_known_g__exceptional]:se_ratio` | 0.8878 | its SE ratio |
| `properties[exceptional_law_report/partially_exceptional_known_g__exceptional]:rule_value_coverage` | 0.9192 | its coverage of the realized rule value |
| `properties[exceptional_law_report/partially_exceptional_known_g__exceptional]:rule_value_se_ratio` | 0.8931 | its SE ratio against the realized rule value |
| `properties[exceptional_law_report/partially_exceptional_known_g__exceptional]:mean_regret` | 0.000865 | its mean regret |
| `properties[exceptional_law_report/partially_exceptional_known_g__exceptional]:mean_rule_disagreement` | 0.2804 | its mean fold-rule disagreement |
| `properties[exceptional_law_report/partially_exceptional_known_g__exceptional]:max_rule_disagreement` | 0.7350 | its largest fold-rule disagreement |
| `bound:non_exceptional_known_g_standard_error` | 0.0149 | the SD at $d_0$ over the root of n |
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
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit, and the target control's ceiling |
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
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 16 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a cross-validated TMLE of the value of the optimal rule. The row rests on accuracy against the truth and on the theory properties |
| One law, one size and one learner per role | The study fits n = 2,000 on one law with a binary treatment and outcome, ten equal folds, pooled targeting, the plug-in rule, `repeats=1`, iid rows and no weights, strata or known regimes. The outcome model is a correctly specified parametric logistic model, so its limit rule is the optimal rule, and the blip $0.1 + W_1$ has no tie set and a bounded density at 0. The key holds neither the learner, the law nor n, so a misspecified or flexible learner, a law with ties, a steeper margin or another n reads supported and was not measured |
| The interval covers the optimal value up to the regret | The regret $\Psi_{d_0} - \tilde\psi_{0n}$ is $O_P(1/n)$ under these conditions, and at n = 2,000 its mean is 0.00138, about 0.09 estimate SDs. The primary bias interval excludes zero because of it, and its spread lowers the SE ratio against $\Psi_{d_0}$ by about 0.9%. Against $\tilde\psi_{0n}$ the calibration cell reads a standardized bias of +0.004, an SE ratio of 1.006 and coverage of 0.952. One n is measured, so the rate is not shown |
| The target control runs at n = 2,500 | With a learner whose limit rule treats nobody while $d_0$ treats 29.3%, the interval covers the value of the learner's rules (0.958) and misses $\Psi_{d_0}$ (0.09). The study shows that C3f matters, not how far a learner may stray from $d_0$ |
| Known treatment mechanism only | The row needs `treatment_probabilities=`, which keys `known_mechanism`. With an estimated $g$ the fit reads a bias interval of -0.0026 to -0.0004, an SE ratio of 0.975 and coverage of 0.948 (99% interval 0.930 to 0.963) over 1,200 draws. That is a reading for decision E2, not a claim, and such a fit reads experimental |
| No claim at an exceptional law | Where the blip is zero on 60% of the population, the interval covers 0.912 at an SE ratio of 0.888 against $\Psi_{d_0}$, and 0.919 at 0.893 against $\tilde\psi_{0n}$. The fold rules do not settle on the tie set (disagreement 0 to 0.735), which breaks the fixed-limit condition; the regret, 0.00086 on average, is not the cause |
| Compositions outside the row | A rule learner, clusters, `repeats` above 1, fold targeting, the stacked report, more than two arms and an estimated $g$ match no row and read experimental. Weights, strata, the in-sample fit and known regimes beside the optimal value refuse |
| Pointwise intervals | The study reads the pointwise interval of the one optimal value |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_value/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_value/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_value/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_value/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_value/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_value/properties.csv)
carry every published row.
