# Learned-rule value, EP rule learner

This study validates the interval of `LearnedRuleValue` when the EP curve sets each fold's rule.
The estimator is `TMLE(learned_rule=LearnedRule(loss="ep", modifiers=Modifiers(["W1", "W2"],
domains={"W1": (-1, 1)}, levels={"W2": (0, 1)})), n_folds=10, cv_evaluation=True,
targeting_scheme="pooled", g_bounds="auto")`. Each fold's rule is the sign of the EP curve of the
effect given $(W_1, W_2)$, fitted on the fold's training rows from the shipped 5-fold inner
cross-fit. The default sieve's size is chosen by cross-validation over the inner folds, with the
nuisances refitted inside each. The fluctuation, the estimate and the variance are those of the
plug-in rule's fit of the [gated learned-rule study](learned-rule-cvtmle.md).
The EP rule learner subsection of [learned rules](../point-treatment-tmle.md#learned-rules)
gives the curve, its limit and the conditions.

The target is data-adaptive, so each replication carries its own truth: the fold average of the
value of the rules that the fit learned. The record sets `truth_varies_by_replicate`, and every
statistic reads the error, the estimate minus the replication's truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule. One cell, `optimal_known_g__declared`, reads the fixed value of the optimal rule
instead.

**No canonical implementation is compared.** No maintained package fits a cross-validated TMLE
of the value of a rule learned from the EP curve inside each training fold. A zero-row
equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | the RM30 laws of the [gated study](learned-rule-cvtmle.md#what-was-tested): $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W)$, with $b(W) = 0.1 + W_1$ at `non_exceptional` and $b(W) = 0.8 W_1 + W_1^2 - 0.3$ at `misspecified_limit`. The reporting cell reads the RM30 `exceptional` law, which has no effect anywhere |
| fit | the estimator above on rows only: no weights, no `id=`, pooled targeting and `repeats=1` |
| rule learner | `LearnedRule(loss="ep")` with `sieve=None`, which reads `Sieve.default(modifiers)`: the family `Additive(Cosine(["W1"], 6), Indicators("W2"))`, its size 1 to 6 chosen by the one-step risk over the 5 inner folds, and the least-squares final stage |
| outcome learner | `Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), LogisticRegression(C=1e6, max_iter=5000, tol=1e-10))` on $(A, W_1, W_2)$, correct at `non_exceptional` and without a $W_1^2$ term at `misspecified_limit` |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, a correct parametric model |
| optimal value | the same fit with `estimands=["ey_optimal_rule"]` and `treatment_probabilities=` $g_0$ at `non_exceptional`, read against $\Psi_{d_0}(P_0) = 0.577115$ |
| sizes and budgets | primary: n = 2,000 and 2,000 replications for each law. `interval_calibration`: n = 16,000 and 7,200 replications for each positive, and the fold-locality control at n = 2,000 with 500. `small_sample_diagnostic`: n = 1,000 and 2,000 replications for each law. `exceptional_law_report`: n = 2,000 and 2,400 replications. The last two state no verdict |
| truth | each replication refits each fold's rule through the package's `fit_fold_rule` with the fold's recorded inner seed and the fit's resolved `g_bounds` and `alpha`. The refit rule must equal the fit's rule on every validation row. The truth is the fold average of $E_0[\bar Q_0(d_{nv}(W), W)]$, by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$ |
| paired fits | every primary and diagnostic row also carries the plug-in rule's fit of the same sample and folds, and every calibration row the fit of the same sample and folds whose rule is fixed at the EP limit rule. Both are descriptive, and no verdict reads them |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20261481` and `resampling_seed=20261482` |
| policy | `gated` |

The truth refits the rule through the code that the fit ran, so the harness checks that the record
reproduces. It does not check that the construction is right. That rests on the unit witnesses
W-c1 to W-c3 in `tests/unit/test_learned_rule_ep.py`. W-c1 rebuilds a fold's curve on a finite
population and checks five deliberate mutations, and W-c2 and W-c3 check that the curve reads
neither the fold's own validation rows nor, in the size step, its inner fold. The limit rules and
the efficiency bounds that size the noise controls come from a quadrature of the projection of
the effect on the largest default member, in `tests/studies/_learned_rule_ep_law.py`.
`tests/unit/test_learned_rule_ep_design.py` recomputes the design constants and the sizing
readings from the committed pilot summaries.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the default sieve | `ey_learned_rule[learned rule]` | fold average of the value of the rule that the EP curve fitted on each fold's training rows sets | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, each fold's rule the sign of the EP curve of the effect given (W1, W2) fitted on the fold's training rows, with the default sieve's size chosen by cross-validation over five inner folds | -0.000864 to 0.000896 | 0.9440 | 0.9827 | pass |
| learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve | `ey_learned_rule[learned rule]` | fold average of the value of the rule that the EP curve fitted on each fold's training rows sets | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, each fold's rule the sign of the EP curve of the effect given (W1, W2) fitted on the fold's training rows, with the default sieve's size chosen by cross-validation over five inner folds | -0.0011 to 0.000673 | 0.9505 | 0.9938 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each positive cell lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, at n = 16,000 and 7,200 replications. The positives are the two laws and the optimal value with $g_0$ known | a shrunken-SE control multiplies each SE by 0.70. A noise control adds noise of one efficiency-bound unit to each estimate. Each must put its SE-ratio interval below 0.93. A fold-locality control fits each fold's EP curve on that fold's validation rows and must put its coverage interval below 0.92 |
| `small_sample_diagnostic` | what the SE ratio, coverage and bias are at n = 1,000, with 2,000 replications | nothing fails. The cell has the role `diagnostic`, and no margin is read |
| `exceptional_law_report` | what the coverage is at the `exceptional` law at n = 2,000, with 2,400 replications | nothing fails. The cell has the role `diagnostic` and publishes its declared reading |

The calibration cells fit n = 16,000, eight times the primary size. At n = 4,000 the declared
sizing pilot read an SE ratio of 0.970 and coverage 0.9365 at `non_exceptional` (2,000 draws).
The plan's sizing rule then put the support rows out of reach, and a second pilot read 0.997 at
n = 16,000. The size moved after both pilots and before any declared draw, and the
[run README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep/README.md)
gives both pilots and the assurance of each cell. Because the size moved, this page and the
support rows state the smaller sizes beside the calibrated one.

The fold-locality control is a deliberate mutation, `ValidationRowRule`. It fits fold $v$'s EP
curve on fold $v$'s validation rows from the outer cross-fitted nuisances, so the rows that
evaluate each rule also chose it. Its estimate is optimistic for the value of its own rules, so
its coverage must fall below the band.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `exceptional_law_report` | `exceptional__n2000` | diagnostic | the learned-rule law with no effect anywhere (RM30 `exceptional`), with `LearnedRule(loss='ep')` on the default sieve: the declared fit at n = 2,000, where condition C3 fails because the fold curves have no fixed sign | none. The cell reports its coverage with the declared reading: an upper 99% coverage endpoint below 0.95 reads under-coverage | coverage 0.8845 to 0.9163, SE ratio 0.8335, reading: under-covers at the exceptional law | reported |
| `interval_calibration` | `misspecified_limit__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the default sieve: the declared fit at n = 16,000 with the study's logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9471 to 0.9600, SE ratio 0.9879 to 1.0297 | pass |
| `interval_calibration` | `misspecified_limit__noise_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the default sieve: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8210 to 0.8438, SE ratio 0.6951 to 0.7257 | pass |
| `interval_calibration` | `misspecified_limit__shrunken_se_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the default sieve: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8170 to 0.8400, SE ratio 0.6911 to 0.7208 | pass |
| `interval_calibration` | `non_exceptional__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve: the declared fit at n = 16,000 with the study's logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9436 to 0.9569, SE ratio 0.9783 to 1.0198 | pass |
| `interval_calibration` | `non_exceptional__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8249 to 0.8475, SE ratio 0.6934 to 0.7235 | pass |
| `interval_calibration` | `non_exceptional__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8145 to 0.8376, SE ratio 0.6844 to 0.7141 | pass |
| `interval_calibration` | `non_exceptional__validation_rule_control` | control | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve: the same fit at n = 2,000 with each fold's EP curve fitted on that fold's own validation rows, so the rule has seen the rows that evaluate it | the coverage interval must fall below the calibration band | coverage 0.0013 to 0.0250, SE ratio 0.9943 to 1.1760 | pass |
| `interval_calibration` | `optimal_known_g__declared` | positive | the value of the optimal rule at the learned-rule law with the blip 0.1 + W1, with `LearnedRule(loss='ep')` on both adjustment columns and the treatment mechanism known, read against the fixed optimal value 0.577115: the declared fit at n = 16,000 with the study's logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9396 to 0.9534, SE ratio 0.9648 to 1.0073 | pass |
| `interval_calibration` | `optimal_known_g__noise_control` | control | the value of the optimal rule at the learned-rule law with the blip 0.1 + W1, with `LearnedRule(loss='ep')` on both adjustment columns and the treatment mechanism known, read against the fixed optimal value 0.577115: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8173 to 0.8403, SE ratio 0.6921 to 0.7222 | pass |
| `interval_calibration` | `optimal_known_g__shrunken_se_control` | control | the value of the optimal rule at the learned-rule law with the blip 0.1 + W1, with `LearnedRule(loss='ep')` on both adjustment columns and the treatment mechanism known, read against the fixed optimal value 0.577115: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8073 to 0.8308, SE ratio 0.6753 to 0.7049 | pass |
| `small_sample_diagnostic` | `misspecified_limit__n1000` | diagnostic | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the default sieve: the declared fit at n = 1,000, reported without a verdict | none. The cell reports the SE ratio, coverage and bias at the smaller size | SE ratio 0.8769 to 0.9496 | reported |
| `small_sample_diagnostic` | `non_exceptional__n1000` | diagnostic | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve: the declared fit at n = 1,000, reported without a verdict | none. The cell reports the SE ratio, coverage and bias at the smaller size | SE ratio 0.9494 to 1.0296 | reported |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and every control failed as designed.
At n = 16,000 the interval is calibrated at both laws and for the optimal value with $g_0$ known.
At n = 2,000 both laws pass the primary screens with SE ratios of 0.994 and 0.983. At the
`exceptional` law the interval under-covers, as its declared reading states.

| reading | value | source |
| --- | --- | --- |
| primary coverage, `non_exceptional` | 0.9505 | `performance-tests.csv` |
| primary coverage, `misspecified_limit` | 0.9440 | `performance-tests.csv` |
| calibration coverage, `non_exceptional` | 0.9436 to 0.9569 | `properties.csv`, `non_exceptional__declared` |
| calibration coverage, `misspecified_limit` | 0.9471 to 0.9600 | `properties.csv`, `misspecified_limit__declared` |
| calibration coverage, optimal value | 0.9396 to 0.9534 | `properties.csv`, `optimal_known_g__declared` |
| coverage at the `exceptional` law | 0.8845 to 0.9163 | `properties.csv`, `exceptional__n2000` |
| rule rows that the truth harness checked | every validation row of every replication | `harness.csv.gz` and `property-replicates.parquet`, `rule_rows_checked` |

No fit failed. A harness refit whose rule differed from the fit's rule would have stopped the
run. The RM30 rule L4 counts a solver warning and does not treat it as a failed fit: 4 primary
warnings fell on `non_exceptional`, and each gated calibration cell carries 4 on 2 of its 7,200
draws.

### The size of the interval

The SE ratio is the mean reported SE over the SD of the error.

| n | cell | replications | `non_exceptional` SE ratio (99% interval) | `non_exceptional` coverage | `misspecified_limit` SE ratio (99% interval) | `misspecified_limit` coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | `small_sample_diagnostic`, no verdict | 2,000 | 0.9872 (0.9494 to 1.0296) | 0.9480 | 0.9111 (0.8769 to 0.9496) | 0.9180 |
| 2,000 | primary, gated | 2,000 | 0.9938 (0.9540 to 1.0354) | 0.9505 | 0.9827 (0.9426 to 1.0278) | 0.9440 |
| 16,000 | `interval_calibration`, gated | 7,200 | 0.9987 (0.9783 to 1.0198) | 0.9506 | 1.0079 (0.9879 to 1.0297) | 0.9539 |

The primary cells pass the framework's wide screens, 0.80 to 1.20 for the SE ratio and a coverage
floor of 0.90, and their SE-ratio intervals also lie inside the calibration band. The study
declared its calibration claim at n = 16,000 only. At n = 1,000 at `misspecified_limit` the SE is
about 9% small and the coverage 0.918, so a user of the interval at that size should expect it to
be too narrow at a law whose outcome regression is wrong. The pilot at n = 4,000 read 0.970 at
`non_exceptional`.

The oracle SE, the SD of the row-level curve at the fit's own fold rules, tracks the reported SE.
Over the error SD it is 0.991 and 0.978 in the primary phase, and 0.998 and 1.007 in the
calibration cells, so the variance estimator is right for the realized rules. What rule learning
adds is a covariance between folds whose rules are fitted on each other's validation rows, a
second-order remainder. The pilots read it as about 13% (SD 9%) of the error variance at
n = 2,000 and 6.5% (SD 3%) at 4,000 at `non_exceptional`. This run's primary SE ratios, 0.994 and
0.983, sit above the first pilot's 0.943 at n = 2,000, which rested on 300 draws.

### The fixed-rule fit on the same draws

Each calibration draw also fits the fit whose rule is fixed at the EP limit rule, on the same
sample and folds. Its error correlates with the learned fit's at about 0.96 to 0.97. The ratio of
the two error variances reads the cost of learning the rule more tightly than the SE ratio alone.
These numbers are recomputed from `property-replicates.parquet`, and no verdict reads them.

| cell | fixed-rule SE ratio | fixed-rule coverage | learned over fixed-rule error variance |
| --- | --- | --- | --- |
| `non_exceptional__declared` | 1.004 | 0.950 | 1.011 |
| `misspecified_limit__declared` | 1.011 | 0.954 | 1.009 |
| `optimal_known_g__declared` | 0.996 | 0.950 | 1.022 |

At n = 16,000 learning the rule adds about 1% to the error variance at the two learned-value laws
and about 2% for the optimal value. Every SE ratio is inside its band, so no red-cell reading of
the run applied.

### The value of the optimal rule

The optimal cell is the only one that sees whether the EP rule learner learns a good rule. The
learned-value target moves with the learned rule, so a learner that treats nobody still covers
its own value in the other cells. With $g_0$ known the cell reads an SE ratio of 0.9852 and
coverage 0.9468 against the fixed $\Psi_{d_0}$. Its bias is $-0.00037$, standardized bias $-0.070$,
and the 99% interval $-0.00053$ to $-0.00021$ excludes zero and lies inside the margin of 0.25 SD.

The EP limit rule is the sign of the projection of the effect onto the largest default member, six
cosines of $W_1$ plus the indicator of $W_2$. It differs from $d_0$ on 0.98% of the population near
the threshold and gives up $2.4 \times 10^{-5}$ of value, 0.005 SE at n = 16,000. On the same
draws the learned fit sits 0.00031 (SE 0.000016) below the fixed-rule fit, which is unbiased for
the limit rule's value, so most of the bias is the regret of the rules learned at n = 16,000, not
of the limit rule. At `non_exceptional` the main-terms outcome
regression is correct, so the debiasing's $1/g$ weight multiplies a zero residual there, and the
unit witness W-c1, not this cell, sees that term.

### The exceptional law

At the `exceptional` law the effect is zero everywhere, so the EP curve has no fixed sign and
condition C3 fails. The cell reads coverage 0.9013 (99% interval 0.8845 to 0.9163) and an SE ratio
of 0.833. The upper end of the coverage interval lies below 0.95, so the declared reading is
"under-covers at the exceptional law". The oracle SE ratio is also 0.83: the reported SE is right
for each realized rule, and the error spreads more because the rules do not settle. The plug-in
rule's [boundary study](learned-rule-cvtmle-boundary.md) measured coverage 0.894 at the same law.
No check in the package detects such a law, so the support rows make no claim there.

### What the fold-locality control shows

The control's coverage is 0.008, with a bias of +0.063, about 4.6 error SDs. Its SE ratio is
1.076, so the SE is honest and the optimism is in the point. A coverage rule is the right
instrument for it, and its interval, 0.0013 to 0.0250, lies far below 0.92. Had the mutation not
been wired, the curve would come from the training rows, the bias would vanish, and the control
would turn red.

Passing shows that the instrument resolves a gross fold-locality violation. The subtle leaks, the
outer nuisances at the rows of $T_v$ and a size step that sees its own inner fold, are invisible to
any calibration cell. They rest on the unit witnesses W-c2 and W-c3.

### What the learner buys

The primary and diagnostic rows carry the plug-in rule's fit of the same sample and folds. The
value gain is the truth of the fit's rules minus the plug-in rule's truth. It is descriptive, and
the study makes no claim about it. At `non_exceptional` the plug-in rule's outcome regression is
correct, and the gain is slightly negative. At `misspecified_limit` the EP curve corrects the
outcome regression with the treatment model, and the gain is positive in most draws.

| reading | value | source |
| --- | --- | --- |
| mean gain at n = 2,000, `non_exceptional` | -0.0010 | `harness.csv.gz`, `value_gain_mean_non_exceptional` |
| share of replications with a positive gain at n = 2,000, `non_exceptional` | 0.3825 | `harness.csv.gz`, `value_gain_positive_share_non_exceptional` |
| mean gain at n = 2,000, `misspecified_limit` | 0.0030 | `harness.csv.gz`, `value_gain_mean_misspecified_limit` |
| share of replications with a positive gain at n = 2,000, `misspecified_limit` | 0.8330 | `harness.csv.gz`, `value_gain_positive_share_misspecified_limit` |
| mean gain at n = 16,000, `misspecified_limit` | 0.0053 | `properties.csv`, `misspecified_limit__declared` |
| share of replications with a positive gain at n = 16,000, `misspecified_limit` | 0.9993 | `properties.csv`, `misspecified_limit__declared` |

The readings do not show the items in this list.

- A declared sieve, a fixed size, a schedule or a declared final learner. Each such fit reads
  experimental.
- Another modifier set, or more than two modifiers.
- Weights, clusters, fold targeting, or `repeats` above 1 with the EP rule. Each such fit reads
  experimental.
- The optimal value with an estimated treatment mechanism, or at the `misspecified_limit` law.
- A rate for the finite-sample excess.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the asymptotic SD at the limit rule over $\sqrt{2000}$, keyed by cell label. They size the noise
controls only, and no verdict reads them as an efficiency bound. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications per law |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | law tests against the replication truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 10 | gated repeated-sampling property cells |
| `property_cells_passed` | 10 | cells whose own and family verdicts pass |
| `property_cells_reported` | 3 | diagnostic cells, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0129 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9440 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9294 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9426 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0354 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-ep/non_exceptional/ey_learned_rule[learned rule]]:coverage` | 0.9505 | primary coverage of `non_exceptional` |
| `performance[cleverly-learned-rule-ep/misspecified_limit/ey_learned_rule[learned rule]]:coverage` | 0.9440 | primary coverage of `misspecified_limit` |
| `performance[cleverly-learned-rule-ep/non_exceptional/ey_learned_rule[learned rule]]:se_ratio` | 0.9938 | primary SE ratio of `non_exceptional` |
| `performance[cleverly-learned-rule-ep/misspecified_limit/ey_learned_rule[learned rule]]:se_ratio` | 0.9827 | primary SE ratio of `misspecified_limit` |
| `performance[cleverly-learned-rule-ep/non_exceptional/ey_learned_rule[learned rule]]:se_ratio_ci_lower` | 0.9540 | its lower endpoint |
| `performance[cleverly-learned-rule-ep/non_exceptional/ey_learned_rule[learned rule]]:se_ratio_ci_upper` | 1.0354 | its upper endpoint |
| `performance[cleverly-learned-rule-ep/misspecified_limit/ey_learned_rule[learned rule]]:se_ratio_ci_lower` | 0.9426 | its lower endpoint |
| `performance[cleverly-learned-rule-ep/misspecified_limit/ey_learned_rule[learned rule]]:se_ratio_ci_upper` | 1.0278 | its upper endpoint |
| `performance[cleverly-learned-rule-ep/non_exceptional/ey_learned_rule[learned rule]]:bias` | -0.000194 | primary bias of `non_exceptional` |
| `performance[cleverly-learned-rule-ep/misspecified_limit/ey_learned_rule[learned rule]]:bias` | 0.000016 | primary bias of `misspecified_limit` |
| `properties[interval_calibration/non_exceptional__declared]:se_ratio` | 0.9987 | `non_exceptional` calibration SE ratio at n = 16,000 |
| `properties[interval_calibration/non_exceptional__declared]:se_ratio_ci_lower` | 0.9783 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/non_exceptional__declared]:se_ratio_ci_upper` | 1.0198 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/non_exceptional__declared]:coverage` | 0.9506 | `non_exceptional` calibration coverage |
| `properties[interval_calibration/non_exceptional__declared]:coverage_ci_lower` | 0.9436 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/non_exceptional__declared]:coverage_ci_upper` | 0.9569 | its upper endpoint, against 0.98 |
| `properties[interval_calibration/misspecified_limit__declared]:se_ratio` | 1.0079 | `misspecified_limit` calibration SE ratio at n = 16,000 |
| `properties[interval_calibration/misspecified_limit__declared]:se_ratio_ci_lower` | 0.9879 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/misspecified_limit__declared]:se_ratio_ci_upper` | 1.0297 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/misspecified_limit__declared]:coverage` | 0.9539 | `misspecified_limit` calibration coverage |
| `properties[interval_calibration/misspecified_limit__declared]:coverage_ci_lower` | 0.9471 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/misspecified_limit__declared]:coverage_ci_upper` | 0.9600 | its upper endpoint, against 0.98 |
| `properties[interval_calibration/optimal_known_g__declared]:se_ratio` | 0.9852 | optimal-value calibration SE ratio at n = 16,000 |
| `properties[interval_calibration/optimal_known_g__declared]:se_ratio_ci_lower` | 0.9648 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/optimal_known_g__declared]:se_ratio_ci_upper` | 1.0073 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/optimal_known_g__declared]:coverage` | 0.9468 | optimal-value calibration coverage |
| `properties[interval_calibration/optimal_known_g__declared]:coverage_ci_lower` | 0.9396 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/optimal_known_g__declared]:coverage_ci_upper` | 0.9534 | its upper endpoint, against 0.98 |
| `properties[interval_calibration/optimal_known_g__declared]:bias` | -0.000372 | optimal-value bias against the fixed optimal value |
| `properties[interval_calibration/optimal_known_g__declared]:bias_ci_lower` | -0.000534 | its lower endpoint |
| `properties[interval_calibration/optimal_known_g__declared]:bias_ci_upper` | -0.000210 | its upper endpoint |
| `properties[interval_calibration/optimal_known_g__declared]:standardized_bias` | -0.0696 | the same bias in error SDs, against 0.25 |
| `properties[small_sample_diagnostic/non_exceptional__n1000]:se_ratio` | 0.9872 | `non_exceptional` SE ratio at n = 1,000 |
| `properties[small_sample_diagnostic/non_exceptional__n1000]:se_ratio_ci_lower` | 0.9494 | its lower endpoint |
| `properties[small_sample_diagnostic/non_exceptional__n1000]:se_ratio_ci_upper` | 1.0296 | its upper endpoint |
| `properties[small_sample_diagnostic/non_exceptional__n1000]:coverage` | 0.9480 | `non_exceptional` coverage at n = 1,000 |
| `properties[small_sample_diagnostic/misspecified_limit__n1000]:se_ratio` | 0.9111 | `misspecified_limit` SE ratio at n = 1,000 |
| `properties[small_sample_diagnostic/misspecified_limit__n1000]:se_ratio_ci_lower` | 0.8769 | its lower endpoint |
| `properties[small_sample_diagnostic/misspecified_limit__n1000]:se_ratio_ci_upper` | 0.9496 | its upper endpoint |
| `properties[small_sample_diagnostic/misspecified_limit__n1000]:coverage` | 0.9180 | `misspecified_limit` coverage at n = 1,000 |
| `properties[exceptional_law_report/exceptional__n2000]:coverage` | 0.9012 | coverage at the `exceptional` law |
| `properties[exceptional_law_report/exceptional__n2000]:coverage_ci_lower` | 0.8845 | its lower endpoint |
| `properties[exceptional_law_report/exceptional__n2000]:coverage_ci_upper` | 0.9163 | its upper endpoint, against 0.95 for the declared reading |
| `properties[exceptional_law_report/exceptional__n2000]:se_ratio` | 0.8335 | SE ratio at the `exceptional` law |
| `properties[interval_calibration/non_exceptional__validation_rule_control]:coverage` | 0.0080 | fold-locality control coverage |
| `properties[interval_calibration/non_exceptional__validation_rule_control]:coverage_ci_upper` | 0.0250 | its upper endpoint, against 0.92 |
| `properties[interval_calibration/non_exceptional__validation_rule_control]:bias` | 0.0633 | fold-locality control bias |
| `properties[interval_calibration/non_exceptional__validation_rule_control]:se_ratio` | 1.0761 | fold-locality control SE ratio |
| `properties[interval_calibration/non_exceptional__noise_control]:se_ratio_ci_upper` | 0.7235 | noise control upper endpoint, `non_exceptional` |
| `properties[interval_calibration/misspecified_limit__noise_control]:se_ratio_ci_upper` | 0.7257 | noise control upper endpoint, `misspecified_limit` |
| `properties[interval_calibration/optimal_known_g__noise_control]:se_ratio_ci_upper` | 0.7222 | noise control upper endpoint, optimal value |
| `properties[interval_calibration/non_exceptional__shrunken_se_control]:se_ratio_ci_upper` | 0.7141 | shrunken-SE control upper endpoint, `non_exceptional` |
| `properties[interval_calibration/misspecified_limit__shrunken_se_control]:se_ratio_ci_upper` | 0.7208 | shrunken-SE control upper endpoint, `misspecified_limit` |
| `properties[interval_calibration/optimal_known_g__shrunken_se_control]:se_ratio_ci_upper` | 0.7049 | shrunken-SE control upper endpoint, optimal value |
| `properties[interval_calibration/misspecified_limit__declared]:value_gain_mean` | 0.0053 | mean value gain at n = 16,000, `misspecified_limit` |
| `properties[interval_calibration/misspecified_limit__declared]:value_gain_positive_share` | 0.9993 | its positive share |
| `properties[interval_calibration/non_exceptional__declared]:value_gain_mean` | -0.000193 | mean value gain at n = 16,000, `non_exceptional` |
| `properties[interval_calibration/non_exceptional__declared]:value_gain_positive_share` | 0.2183 | its positive share |
| `value_gain_mean_non_exceptional` | -0.0010 | mean value gain at n = 2,000, `non_exceptional`, from `harness.csv.gz` |
| `value_gain_positive_share_non_exceptional` | 0.3825 | its positive share |
| `value_gain_mean_misspecified_limit` | 0.0030 | mean value gain at n = 2,000, `misspecified_limit` |
| `value_gain_positive_share_misspecified_limit` | 0.8330 | its positive share |
| `mean_oracle_se_non_exceptional` | 0.0149 | mean oracle SE of the fit's fold rules, `non_exceptional`, from `harness.csv.gz`. No rule reads it |
| `mean_oracle_se_misspecified_limit` | 0.0149 | mean oracle SE, `misspecified_limit` |
| `bound:non_exceptional_standard_error` | 0.0149 | the asymptotic SD at the limit rule of `non_exceptional`, at n = 2,000 |
| `bound:misspecified_limit_standard_error` | 0.0148 | the asymptotic SD at the limit rule of `misspecified_limit`, at n = 2,000 |
| `bound:optimal_known_g_standard_error` | 0.0149 | the asymptotic SD at the limit rule for the optimal value, at n = 2,000 |
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
| There is no cross-implementation evidence | No maintained package fits a cross-validated TMLE of the value of an EP rule. The rows rest on accuracy against the replication truth and on the theory properties |
| The interval is calibrated at n = 16,000 only | At n = 2,000 both laws pass the primary screens. At n = 1,000 at `misspecified_limit` the reported SE is about 9% too small and the coverage 0.918. The support key holds no sample size, so a smaller fit also reads supported, and the rows' limits state the deficit |
| The calibration size was moved once | The cells moved from 4,000 to 16,000 after the sizing pilot read an SE ratio of 0.970 at 4,000, and before any declared draw. The rows' limits name the move |
| The exceptional law under-covers | At a law with no effect, or with an effect zero on a set of positive probability, the fold curves have no fixed sign and condition C3 fails. At the RM30 `exceptional` law the coverage is 0.901 and the SE ratio 0.833 at n = 2,000. No check detects such a law, and the rows make no claim there |
| One sieve and one final stage | The study fits the default sieve with its size chosen by cross-validation and least squares. A declared sieve, a fixed size, a schedule or a declared final learner reads experimental |
| Two laws and two modifiers | The study fits RM30 `non_exceptional` and `misspecified_limit` with the main-terms outcome learner and both adjustment columns as modifiers. Another law or modifier set rests on the theory alone |
| The optimal value is measured with $g_0$ known at one law | The interval covers $\Psi_{d_0}$ up to the regret of the EP limit rule, which is 0.005 SE at `non_exceptional`. An estimated treatment mechanism reads experimental |
| The fold-locality control resolves a gross violation only | It shows that a curve fitted on the validation rows reads far below the band. The subtle leaks rest on the unit witnesses W-c2 and W-c3 |
| The value gain is descriptive | No verdict reads it. The study does not claim that the EP rule improves the value, only that its interval is calibrated at n = 16,000 |
| Rows only, pooled targeting, one split | A fit with weights, fold targeting, clusters or `repeats` above 1 reads experimental |
| A binary treatment and a binary outcome | A continuous outcome with declared `q_bounds` is not tested |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep/README.md)
gives the run command, the run form and both sizing pilots. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep/run.log)
records the commit, the clean tree, the runtime and the wall time. The red-cell readings, declared
before any run, are in the docstring of
[`regenerate.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep/regenerate.py).
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep/replicates.parquet),
the [harness columns](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep/properties.csv)
carry every published row.
