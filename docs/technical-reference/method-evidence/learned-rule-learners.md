# Learned-rule value, rule learners

This study validates the interval of `LearnedRuleValue` when a rule learner sets each fold's rule.
The estimator is `TMLE(learned_rule=LearnedRule(loss=..., learner=...), n_folds=10,
cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")` with `loss="blip"` or
`loss="weighted_classification"`. Each fold fits its rule learner on the fold's training rows to the
doubly robust blip pseudo-outcome, whose outcome and treatment nuisances are cross-fitted in five
inner folds of those rows. The fluctuation, the estimate and the variance are those of the plug-in
rule's fit of the [gated learned-rule study](learned-rule-cvtmle.md).
[Rule learners](../point-treatment-tmle.md#learned-rules) gives the pseudo-outcome, the two losses
and the conditions.

The target is data-adaptive, so each replication carries its own truth: the fold average of the
value of the rules that the fit learned. The record sets `truth_varies_by_replicate`, and every
statistic reads the error, the estimate minus the replication's truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** The gated study names the candidates and why each
fails for the base estimator. None of them fits a pseudo-outcome rule learner inside each training
fold either. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law | the RM30 `misspecified_limit` law of the [gated study](learned-rule-cvtmle.md#what-was-tested): $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W)$ with $b(W) = 0.8 W_1 + W_1^2 - 0.3$ |
| fit | the estimator above on rows only: no weights, no `id=`, pooled targeting and `repeats=1` |
| rule learners | `blip`: `Pipeline(PolynomialFeatures(2), LinearRegression())` on $(W_1, W_2)$. `weighted_classification`: `Pipeline(PolynomialFeatures(2), LogisticRegression(C=1e6, max_iter=5000))` on $(W_1, W_2)$, with sample weights $\lvert\tilde D\rvert$ |
| outcome learner | `Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), LogisticRegression(C=1e6, max_iter=5000, tol=1e-10))` on $(A, W_1, W_2)$, which has no $W_1^2$ term |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, a correct parametric model |
| sizes and budgets | primary: n = 2,000 and 2,000 replications for each loss. `interval_calibration`: n = 4,000 and 7,200 replications for each cell. `small_sample_diagnostic`: n = 1,000 and 2,000 replications for each loss, with no verdict |
| truth | each replication refits each fold's rule through the package's `fit_fold_rule` with the fold's recorded inner seed and the fit's resolved `g_bounds`. The refit rule must equal the fit's rule on every validation row. The truth is the fold average of $E_0[\bar Q_0(d_{nv}(W), W)]$, by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$ |
| oracle SE | the SD of the row-level curve at the fit's fold rules, by the same quadrature. It is a descriptive column of `harness.csv.gz`, and no rule reads it |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20263031` and `resampling_seed=20263032` |
| policy | `gated` |

The truth refits the rule through the code that the fit ran, so the harness checks that the record
reproduces. It does not check that the construction is right. That rests on the unit witnesses W-g2
and W-g4 in `tests/unit/test_learned_rule_learners.py`, which rebuild a fold's pseudo-outcome and
rule from scikit-learn and check deliberate mutations. The limit rules and the efficiency bounds
that size the noise controls come from a quadrature of each learner's population projection, in
`tests/studies/_learned_rule_learners_law.py`. `tests/unit/test_learned_rule_learners_design.py`
computes the design constants again, and `tests/unit/test_learned_rule_learners_harness.py` checks
the harness and the controls.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, a misspecified outcome learner, and a blip rule learner | `ey_learned_rule[learned rule]` | fold average of the value of the rule that a rule learner fits on each fold's training rows | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, with a rule learner fitted to the blip pseudo-outcome on each fold's training rows | -0.0011 to 0.000683 | 0.9375 | 0.9644 | pass |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, a misspecified outcome learner, and a weighted-classification rule learner | `ey_learned_rule[learned rule]` | fold average of the value of the rule that a rule learner fits on each fold's training rows | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, with a rule learner fitted to the blip pseudo-outcome on each fold's training rows | -0.0011 to 0.000739 | 0.9380 | 0.9525 | pass |
<!-- /generated -->

## Theory properties

Every property cell uses the declared learners at the one law.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each positive cell lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, at n = 4,000 and 7,200 replications. The positives are the two losses | a shrunken-SE control multiplies each SE by 0.70. A noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. Each must put its SE-ratio interval below 0.93. A fold-locality control learns each fold's rule on that fold's validation rows and must put its coverage interval below 0.92 |
| `small_sample_diagnostic` | what the SE ratio, coverage and bias are at n = 1,000, with 2,000 replications | nothing fails. The cell has the role `diagnostic`, and no margin is read |

The calibration cells fit n = 4,000 and not the primary n = 2,000. At n = 2,000 the SE ratio is
inside the framework's sanity band and below the calibration band, for a reason that
[the size of the interval](#the-size-of-the-interval) gives. The size was fixed before any
registered run, from a throwaway pilot and the per-arm sizing rule of the studies, and a red cell
at 4,000 would have been published as a finite-sample reading and not met by a larger n. The
noise scale is the asymptotic SD of $\sqrt{n}$ times the estimate at each loss's limit rule, 0.6632
for the blip and 0.6629 for the classification, and no verdict reads it as an efficiency bound.

The fold-locality control is a deliberate mutation, `ValidationRowRule`. It fits fold $v$'s rule
learner on fold $v$'s validation rows, to the pseudo-outcome of the fit's outer cross-fitted
nuisances, so the rows that evaluate each rule also chose it. A validation fold whose weighted rows
carry one sign gets the constant rule of that sign, the package's convention. Its estimate is
optimistic for the value of its own rules, so its coverage must fall below the band.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `blip__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='blip')` on a quadratic basis and linear regression: the declared fit with the study's rule learner and logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9432 to 0.9565, SE ratio 0.9875 to 1.0300 | pass |
| `interval_calibration` | `blip__noise_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='blip')` on a quadratic basis and linear regression: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8240 to 0.8467, SE ratio 0.6984 to 0.7275 | pass |
| `interval_calibration` | `blip__shrunken_se_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='blip')` on a quadratic basis and linear regression: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8204 to 0.8433, SE ratio 0.6908 to 0.7210 | pass |
| `interval_calibration` | `blip__validation_rule_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='blip')` on a quadratic basis and linear regression: the same sample and folds with each fold's rule learner fitted on that fold's own validation rows, so the rule has seen the rows that evaluate it | the coverage interval must fall below the calibration band | coverage 0.2340 to 0.2603, SE ratio 1.0297 to 1.0738 | pass |
| `interval_calibration` | `weighted_classification__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='weighted_classification')` on a quadratic basis and logistic regression: the declared fit with the study's rule learner and logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9383 to 0.9522, SE ratio 0.9675 to 1.0102 | pass |
| `interval_calibration` | `weighted_classification__noise_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='weighted_classification')` on a quadratic basis and logistic regression: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8242 to 0.8468, SE ratio 0.6872 to 0.7181 | pass |
| `interval_calibration` | `weighted_classification__shrunken_se_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='weighted_classification')` on a quadratic basis and logistic regression: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8145 to 0.8376, SE ratio 0.6775 to 0.7073 | pass |
| `interval_calibration` | `weighted_classification__validation_rule_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='weighted_classification')` on a quadratic basis and logistic regression: the same sample and folds with each fold's rule learner fitted on that fold's own validation rows, so the rule has seen the rows that evaluate it | the coverage interval must fall below the calibration band | coverage 0.2542 to 0.2812, SE ratio 1.0033 to 1.0468 | pass |
| `small_sample_diagnostic` | `blip__n1000` | diagnostic | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='blip')` on a quadratic basis and linear regression: the declared fit at n = 1,000, reported without a verdict | none. The cell reports the SE ratio, coverage and bias at the smaller size | SE ratio 0.9244 to 1.0019 | reported |
| `small_sample_diagnostic` | `weighted_classification__n1000` | diagnostic | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term, with `LearnedRule(loss='weighted_classification')` on a quadratic basis and logistic regression: the declared fit at n = 1,000, reported without a verdict | none. The cell reports the SE ratio, coverage and bias at the smaller size | SE ratio 0.9159 to 1.0007 | reported |
<!-- /generated -->

## Result

Every primary test and every property cell passed. At n = 4,000 the interval is calibrated for both
losses. At n = 1,000 and 2,000 the reported SE is about 3.5% to 5% too small and the coverage is
0.93 to 0.94. The next section gives that size, and the one after it the mechanism.

| reading | value | source |
| --- | --- | --- |
| primary coverage, `blip` | 0.9375 | `performance-tests.csv` |
| primary coverage, `weighted_classification` | 0.9380 | `performance-tests.csv` |
| primary bias, `blip` | -0.000213 | `performance-tests.csv` |
| primary bias, `weighted_classification` | -0.000168 | `performance-tests.csv` |
| calibration coverage, `blip` | 0.9432 to 0.9565 | `properties.csv`, `blip__declared` |
| calibration coverage, `weighted_classification` | 0.9383 to 0.9522 | `properties.csv`, `weighted_classification__declared` |
| rule rows that the truth harness checked | 2,000 in each of 4,000 primary replications | `harness.csv.gz`, `rule_rows_checked` |
| solver warnings, primary | 0 | `harness.csv.gz`, `solver_warnings` |
| solver warnings, property | 4, on 4 of 61,600 rows | `property-replicates.parquet`, `solver_warnings` |

The RM30 rule L4 counts a solver warning and does not treat it as a failed fit, and no rule reads
the count. A harness refit whose rule differed from the fit's rule would have stopped the run, and
the run completed without a failed fit.

### The size of the interval

The SE ratio is the mean reported SE over the SD of the error. It falls from the calibration size
to the two smaller sizes, and the coverage follows it.

| n | cell | replications | `blip` SE ratio (99% interval) | `blip` coverage | `weighted_classification` SE ratio (99% interval) | `weighted_classification` coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | `small_sample_diagnostic`, no verdict | 2,000 | 0.9615 (0.9244 to 1.0019) | 0.9410 | 0.9557 (0.9159 to 1.0007) | 0.9325 |
| 2,000 | primary, gated | 2,000 | 0.9644 (0.9250 to 1.0074) | 0.9375 | 0.9525 (0.9153 to 0.9916) | 0.9380 |
| 4,000 | `interval_calibration`, gated | 7,200 | 1.0080 (0.9875 to 1.0300) | 0.9501 | 0.9886 (0.9675 to 1.0102) | 0.9456 |

The primary cells pass because their bands are wide, 0.80 to 1.20 for the SE ratio and a coverage
floor of 0.90. They are not evidence that the interval is calibrated at n = 2,000. Both primary
SE-ratio intervals reach below the calibration band's lower end of 0.93, and the point estimates
are 0.9644 and 0.9525. The diagnostic cell reads n = 1,000 and forms no verdict, because the study
declared its calibration claim at n = 4,000 only. A user of the interval at n = 1,000 or 2,000
should expect it to be about 3.5% to 5% too narrow, with coverage of about 0.93 to 0.94.

The calibration size was moved once, from 2,000 to 4,000, and the move happened before any
registered run. That makes the band easier to meet, which is why this page states the smaller
sizes beside it. The route after a red cell was declared in the same record: the size does not
move again, and a finite-sample diagnosis publishes the study as `reporting` and not as `gated`.

### Why the interval is small at the smaller sizes

The oracle SE, the SD of the row curve at the fit's own fold rules, shows the same deficit as the
reported SE. It is 0.961 and 0.949 of the error SD in the primary phase, against SE ratios of
0.964 and 0.952. The variance estimator is therefore not at fault. The error spreads more than the
fixed-rule variance of the rules that the fit realized.

Within a fold, the variance of the fold's term given its training rows is exactly that fixed-rule
variance. The excess is a covariance between folds, because fold $u$'s rule is fitted on fold $v$'s
validation rows and fold $v$'s rule on fold $u$'s. The first-order projection of one fold's term on
another is zero, so the excess is a second-order remainder, of relative order about $1/n$ in a
heuristic expansion. Its constant depends on how strongly the rule's parameter noise correlates
with the curve near the threshold. A rule learner fitted to the pseudo-outcome carries the inverse
probability residual $(2A - 1)(Y - \bar Q)/g$ in that noise, so its constant is larger than the
plug-in rule's, whose noise is a logistic score. This is the $o_P(n^{-1/2})$ cross-fold remainder
of Theorem 6 of JCI, and the study does not measure its rate.

Two readings support the mechanism and neither is a gated verdict. The paired plug-in fit on the
same draws and outer folds has an SE ratio of 0.986 and 0.991 in the primary phase, and the error
variance of the rule-learner fit is 1.045 and 1.081 times the plug-in's. Both are recomputed from
`replicates.parquet` and `harness.csv.gz`. A pilot of 3,000 to 4,000 draws at three sizes gave an
error-variance ratio of about 1.05 at n = 1,000, 1.03 to 1.06 at n = 2,000 and 0.97 to 0.99 at
n = 4,000, and SE ratios of 0.946, 0.955 and 1.001 for the blip at those sizes. The pilot supports
"a finite-sample excess that declines with $n$ and is within about 3% of zero at n = 4,000". It
does not pin a rate.

### What the fold-locality control shows

The control's coverage is 0.2469 for the blip and 0.2675 for the classification, with a bias of
+0.0279 and +0.0275, about 2.7 and 2.6 error SDs. Its SE ratio is 1.0510 and 1.0243, so the SE is
honest and the optimism is in the point. A coverage rule is the right instrument for it, and it
passes by a wide margin. Had the mutation not been wired, the rule would come from the training
rows, the bias would vanish, and the control would turn red.

Passing shows that the instrument resolves a gross fold-locality violation. It is no evidence
about the subtle hazard, in which the pseudo-outcome reads outer cross-fitted nuisances at the rows
of a fold's own validation set. That leak is of order $1/n$ and is invisible to any calibration
cell. It rests on the unit witnesses W-g3 and W-g4 in `tests/unit/test_learned_rule_learners.py`,
and the declared fit passing its band at n = 4,000 is the repeated-sampling evidence that the fit
is fold-local.

### What the learner buys

The harness and the property rows carry the plug-in rule's fit of the same sample and folds. The
value gain is the truth of the fit's rules minus the plug-in rule's truth. It is descriptive, no
verdict reads it, and the study makes no claim about it.

| reading | value | source |
| --- | --- | --- |
| mean gain at n = 2,000, `blip` | 0.0036 | `harness.csv.gz`, `value_gain_mean_blip` |
| share of replications with a positive gain at n = 2,000, `blip` | 0.8490 | `harness.csv.gz`, `value_gain_positive_share_blip` |
| mean gain at n = 2,000, `weighted_classification` | 0.0038 | `harness.csv.gz`, `value_gain_mean_weighted_classification` |
| share of replications with a positive gain at n = 2,000, `weighted_classification` | 0.8700 | `harness.csv.gz`, `value_gain_positive_share_weighted_classification` |
| mean gain at n = 4,000, `blip` | 0.0046 | `properties.csv`, `blip__declared` |
| share of replications with a positive gain at n = 4,000, `blip` | 0.9383 | `properties.csv`, `blip__declared` |
| mean gain at n = 4,000, `weighted_classification` | 0.0048 | `properties.csv`, `weighted_classification__declared` |
| share of replications with a positive gain at n = 4,000, `weighted_classification` | 0.9589 | `properties.csv`, `weighted_classification__declared` |

The population gap between the two limit rules at this law is 0.5602 against 0.5547, about 0.0054,
and the gain rises toward it with $n$. At n = 4,000 it is near half an SE, so the gain is small
against the interval's width and usually positive. The law is built so that the plug-in rule's
misspecified outcome learner loses value that a rule learner with a $W_1^2$ term recovers.

The readings do not show the items in this list.

- A rule learner other than a linear regression or a logistic regression on a quadratic basis of
  two covariates. The deficit grows with the learner's variability, so a flexible learner may need
  a larger $n$ for the same calibration.
- Weights, clusters, fold targeting, or `repeats` above 1 with a rule learner. Each such fit reads
  experimental.
- A law other than `misspecified_limit`, and the exceptional and weak-blip laws that the
  [boundary study](learned-rule-cvtmle-boundary.md) reads for the plug-in rule.
- A rate for the finite-sample excess.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the asymptotic SD at the limit rule over $\sqrt{2000}$, keyed by loss. They size the noise
controls only, and no verdict reads them as an efficiency bound. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications per loss |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | loss tests against the replication truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 8 | repeated-sampling property cells |
| `property_cells_passed` | 8 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0137 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9375 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9222 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9153 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0074 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-learners/blip/ey_learned_rule[learned rule]]:coverage` | 0.9375 | primary coverage of `blip` |
| `performance[cleverly-learned-rule-learners/weighted_classification/ey_learned_rule[learned rule]]:coverage` | 0.9380 | primary coverage of `weighted_classification` |
| `performance[cleverly-learned-rule-learners/blip/ey_learned_rule[learned rule]]:se_ratio` | 0.9644 | primary SE ratio of `blip` |
| `performance[cleverly-learned-rule-learners/weighted_classification/ey_learned_rule[learned rule]]:se_ratio` | 0.9525 | primary SE ratio of `weighted_classification` |
| `performance[cleverly-learned-rule-learners/blip/ey_learned_rule[learned rule]]:se_ratio_ci_lower` | 0.9250 | its lower endpoint |
| `performance[cleverly-learned-rule-learners/blip/ey_learned_rule[learned rule]]:se_ratio_ci_upper` | 1.0074 | its upper endpoint |
| `performance[cleverly-learned-rule-learners/weighted_classification/ey_learned_rule[learned rule]]:se_ratio_ci_lower` | 0.9153 | its lower endpoint |
| `performance[cleverly-learned-rule-learners/weighted_classification/ey_learned_rule[learned rule]]:se_ratio_ci_upper` | 0.9916 | its upper endpoint |
| `performance[cleverly-learned-rule-learners/blip/ey_learned_rule[learned rule]]:bias` | -0.000213 | primary bias of `blip` |
| `performance[cleverly-learned-rule-learners/weighted_classification/ey_learned_rule[learned rule]]:bias` | -0.000168 | primary bias of `weighted_classification` |
| `properties[interval_calibration/blip__declared]:se_ratio` | 1.0080 | `blip` calibration SE ratio at n = 4,000 |
| `properties[interval_calibration/blip__declared]:se_ratio_ci_lower` | 0.9875 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/blip__declared]:se_ratio_ci_upper` | 1.0300 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/blip__declared]:coverage` | 0.9501 | `blip` calibration coverage |
| `properties[interval_calibration/blip__declared]:coverage_ci_lower` | 0.9432 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/blip__declared]:coverage_ci_upper` | 0.9565 | its upper endpoint, against 0.98 |
| `properties[interval_calibration/weighted_classification__declared]:se_ratio` | 0.9886 | `weighted_classification` calibration SE ratio at n = 4,000 |
| `properties[interval_calibration/weighted_classification__declared]:se_ratio_ci_lower` | 0.9675 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/weighted_classification__declared]:se_ratio_ci_upper` | 1.0102 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/weighted_classification__declared]:coverage` | 0.9456 | `weighted_classification` calibration coverage |
| `properties[interval_calibration/weighted_classification__declared]:coverage_ci_lower` | 0.9383 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/weighted_classification__declared]:coverage_ci_upper` | 0.9522 | its upper endpoint, against 0.98 |
| `properties[small_sample_diagnostic/blip__n1000]:se_ratio` | 0.9615 | `blip` SE ratio at n = 1,000 |
| `properties[small_sample_diagnostic/blip__n1000]:se_ratio_ci_lower` | 0.9244 | its lower endpoint |
| `properties[small_sample_diagnostic/blip__n1000]:se_ratio_ci_upper` | 1.0019 | its upper endpoint |
| `properties[small_sample_diagnostic/blip__n1000]:coverage` | 0.9410 | `blip` coverage at n = 1,000 |
| `properties[small_sample_diagnostic/weighted_classification__n1000]:se_ratio` | 0.9557 | `weighted_classification` SE ratio at n = 1,000 |
| `properties[small_sample_diagnostic/weighted_classification__n1000]:se_ratio_ci_lower` | 0.9159 | its lower endpoint |
| `properties[small_sample_diagnostic/weighted_classification__n1000]:se_ratio_ci_upper` | 1.0007 | its upper endpoint |
| `properties[small_sample_diagnostic/weighted_classification__n1000]:coverage` | 0.9325 | `weighted_classification` coverage at n = 1,000 |
| `properties[interval_calibration/blip__validation_rule_control]:coverage` | 0.2469 | fold-locality control coverage, `blip` |
| `properties[interval_calibration/weighted_classification__validation_rule_control]:coverage` | 0.2675 | fold-locality control coverage, `weighted_classification` |
| `properties[interval_calibration/blip__validation_rule_control]:bias` | 0.0279 | fold-locality control bias, `blip` |
| `properties[interval_calibration/weighted_classification__validation_rule_control]:bias` | 0.0275 | fold-locality control bias, `weighted_classification` |
| `properties[interval_calibration/blip__validation_rule_control]:se_ratio` | 1.0510 | fold-locality control SE ratio, `blip` |
| `properties[interval_calibration/weighted_classification__validation_rule_control]:se_ratio` | 1.0243 | fold-locality control SE ratio, `weighted_classification` |
| `properties[interval_calibration/blip__noise_control]:se_ratio_ci_upper` | 0.7275 | noise control upper endpoint, `blip` |
| `properties[interval_calibration/weighted_classification__noise_control]:se_ratio_ci_upper` | 0.7181 | noise control upper endpoint, `weighted_classification` |
| `properties[interval_calibration/blip__shrunken_se_control]:se_ratio_ci_upper` | 0.7210 | shrunken-SE control upper endpoint, `blip` |
| `properties[interval_calibration/weighted_classification__shrunken_se_control]:se_ratio_ci_upper` | 0.7073 | shrunken-SE control upper endpoint, `weighted_classification` |
| `properties[interval_calibration/blip__declared]:value_gain_mean` | 0.0046 | mean value gain at n = 4,000, `blip` |
| `properties[interval_calibration/blip__declared]:value_gain_positive_share` | 0.9383 | its positive share |
| `properties[interval_calibration/weighted_classification__declared]:value_gain_mean` | 0.0048 | mean value gain at n = 4,000, `weighted_classification` |
| `properties[interval_calibration/weighted_classification__declared]:value_gain_positive_share` | 0.9589 | its positive share |
| `value_gain_mean_blip` | 0.0036 | mean value gain at n = 2,000, `blip`, from `harness.csv.gz` |
| `value_gain_positive_share_blip` | 0.8490 | its positive share |
| `value_gain_mean_weighted_classification` | 0.0038 | mean value gain at n = 2,000, `weighted_classification` |
| `value_gain_positive_share_weighted_classification` | 0.8700 | its positive share |
| `mean_oracle_se_blip` | 0.0149 | mean oracle SE of the fit's fold rules, `blip`, from `harness.csv.gz`. No rule reads it |
| `mean_oracle_se_weighted_classification` | 0.0149 | mean oracle SE, `weighted_classification` |
| `bound:blip_standard_error` | 0.0148 | the asymptotic SD at the limit rule of `blip`, at n = 2,000 |
| `bound:weighted_classification_standard_error` | 0.0148 | the asymptotic SD at the limit rule of `weighted_classification`, at n = 2,000 |
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
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 20 | failed primary fits that bound allows per setting |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a pseudo-outcome rule learner inside each training fold of a cross-validated learned-rule target. The row rests on accuracy against the replication truth and on the theory properties |
| The interval is calibrated at n = 4,000 only | At n = 1,000 and 2,000 the reported SE is about 3.5% to 5% too small and the coverage about 0.93 to 0.94. The support key holds no sample size, so a smaller fit also reads supported, and the row's limits state the deficit |
| The deficit has an unmeasured rate | The mechanism is the cross-fold covariance of rules fitted on each other's validation rows, a second-order term of relative order about $1/n$. The pilot supports a decline with $n$ and does not pin the rate |
| The calibration size was moved once | The cells moved from 2,000 to 4,000 before any registered run, by the declared per-arm rule. The size is fixed, and the page states the smaller sizes beside it |
| Two rule learners on one basis | The study fits a linear regression and a logistic regression on a quadratic basis of two covariates. A more flexible learner, such as a forest or a `SuperLearner`, rests on the theory alone, and the deficit grows with the learner's variability |
| One law | The study fits `misspecified_limit` with the main-terms outcome learner. The exceptional and weak-blip laws are not tested with a rule learner |
| The fold-locality control resolves a gross violation only | It shows that a rule learned on the validation rows reads far below the band. The subtle hazard of outer nuisances reused for the pseudo-outcome rests on the unit witnesses W-g3 and W-g4 |
| The value gain is descriptive | No verdict reads it. The study does not claim that a rule learner improves the value, only that its interval is calibrated at n = 4,000 |
| Rows only, pooled targeting, one split | A fit with weights, fold targeting, clusters or `repeats` above 1 reads experimental |
| The truth harness needs deterministic learners | The truth refits each fold's rule learner and its inner cross-fit from the recorded seeds. A learner that draws its own randomness cannot give the same rule again |
| A binary treatment and a binary outcome | A continuous outcome with declared `q_bounds` is not tested |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_learners/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_learners/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_learners/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_learners/replicates.parquet),
the [harness columns](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_learners/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_learners/properties.csv)
carry every published row.
