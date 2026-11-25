# Learned-rule value, EP rule with a declared sieve or learner

This study validates the interval of `LearnedRuleValue` when the EP curve that sets each fold's
rule comes from a declared sieve or a declared final learner. The
[EP rule-learner study](learned-rule-ep.md) measured the default sieve with least squares, and
the support key holds neither choice. Two declared curves are fitted. `S` is
`LearnedRule(loss="ep", modifiers=..., sieve=Sieve(Additive(BSpline(["W1"], 4),
Indicators("W2")), size=4))`, another family at a fixed size, so no size is chosen. `L` is the
default sieve with its size chosen by cross-validation and the final learner
`HistGradientBoostingRegressor(max_depth=2, learning_rate=0.1, max_iter=100,
early_stopping=False)`, a greedy learner inside the size step. The EP rule learner subsection of
[learned rules](../point-treatment-tmle.md#learned-rules) gives the curve, its limit and the
conditions.

The target is data-adaptive, so each replication carries its own truth: the fold average of the
value of the rules that the fit learned. The record sets `truth_varies_by_replicate`, and every
statistic reads the error, the estimate minus the replication's truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule. The two optimal-value cells, `optimal__S` and `optimal__L`, read the fixed value
of the optimal rule instead.

**No canonical implementation is compared.** No maintained package fits a cross-validated TMLE
of the value of a rule learned from the EP curve inside each training fold. A zero-row
equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | the RM30 laws of the [EP rule-learner study](learned-rule-ep.md#what-was-tested), `non_exceptional` (blip $0.1 + W_1$) and `misspecified_limit` (blip $0.8 W_1 + W_1^2 - 0.3$), and the value of the optimal rule at `non_exceptional` with $g_0$ supplied (`optimal_known_g`) |
| fit | `TMLE(learned_rule=..., n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")` on rows only: no weights, no `id=`, pooled targeting and `repeats=1` |
| curves | `S`: the declared sieve above, whose EP limit is the projection of the effect onto that member. `L`: the default sieve, its size chosen by cross-validation over the 5 inner folds, with the declared booster as the final learner. No limit is claimed for `L` |
| outcome and treatment learners | those of the EP rule-learner study: the main-terms logistic outcome learner with every pairwise interaction on $(A, W_1, W_2)$, correct at `non_exceptional` and without a $W_1^2$ term at `misspecified_limit`, and a correct logistic treatment learner |
| optimal value | the same fit with `estimands=["ey_optimal_rule"]` and `treatment_probabilities=` $g_0$ at `non_exceptional`, read against $\Psi_{d_0}(P_0) = 0.577115$ |
| sizes and budgets | primary: n = 2,000 and 2,000 replications for each curve and law. `interval_calibration`: n = 16,000 and 12,000 replications for each gated positive, 2,400 for `optimal__L`, and the fold-locality control at n = 2,000 with 500 |
| truth | each replication integrates the fit's own fold rules, recorded as the fit learns them. Every 25th replication refits each fold's rule through the package's `fit_fold_rule` and compares rows and grid exactly. The truth is the fold average of $E_0[\bar Q_0(d_{nv}(W), W)]$, by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$ |
| paired fits | every `S` calibration row also carries the fit of the same sample and folds whose rule is fixed at the `S` limit rule. It is descriptive, and no verdict reads it |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20261491` and `resampling_seed=20261492` |
| policy | `gated` |

The calibration size of 12,000 per cell is the point reading of two pooled pilots of 1,400 draws
per cell on their own streams (`tests/studies/learned_rule_ep_declared_pilot.csv`). X32a's sizing
rule read the gap between the `S` cells' pilot ratios as a spread of true ratios and gave no
budget up to 50,000. At each cell's pooled pilot ratio, 12,000 passes both arms together with
probability 0.989. `tests/unit/test_learned_rule_ep_declared_design.py` recomputes the sieve's
limit, the pass probability and the harness checks.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the default sieve with its size chosen by cross-validation and the declared final learner HistGradientBoostingRegressor(max_depth=2) | `ey_learned_rule[learned rule]` | fold average of the value of the rule that the EP curve, from the declared sieve or final learner, fitted on each fold's training rows sets | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, each fold's rule the sign of the EP curve of the effect given (W1, W2) fitted on the fold's training rows, from the sieve and final stage the law names | -0.000310 to 0.0015 | 0.9375 | 0.9544 | pass |
| learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve with its size chosen by cross-validation and the declared final learner HistGradientBoostingRegressor(max_depth=2) | `ey_learned_rule[learned rule]` | fold average of the value of the rule that the EP curve, from the declared sieve or final learner, fitted on each fold's training rows sets | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, each fold's rule the sign of the EP curve of the effect given (W1, W2) fitted on the fold's training rows, from the sieve and final stage the law names | -0.000359 to 0.0013 | 0.9540 | 1.0096 | pass |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the declared sieve Additive(BSpline(['W1'], 4), Indicators('W2')) at its fixed size 4 | `ey_learned_rule[learned rule]` | fold average of the value of the rule that the EP curve, from the declared sieve or final learner, fitted on each fold's training rows sets | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, each fold's rule the sign of the EP curve of the effect given (W1, W2) fitted on the fold's training rows, from the sieve and final stage the law names | -0.0012 to 0.000587 | 0.9415 | 0.9522 | pass |
| learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the declared sieve Additive(BSpline(['W1'], 4), Indicators('W2')) at its fixed size 4 | `ey_learned_rule[learned rule]` | fold average of the value of the rule that the EP curve, from the declared sieve or final learner, fitted on each fold's training rows sets | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, each fold's rule the sign of the EP curve of the effect given (W1, W2) fitted on the fold's training rows, from the sieve and final stage the law names | -0.000855 to 0.0010 | 0.9305 | 0.9289 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each gated positive lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, at n = 16,000. The positives are `S` and `L` at both laws and the optimal value with `S`; `optimal__L` states no verdict | a shrunken-SE control multiplies each `nonexc__S` SE by 0.70 and must put its SE-ratio interval below 0.93. A fold-locality control fits each fold's `L` curve on that fold's validation rows and must put its coverage interval below 0.92 |

`optimal__L` has the role `diagnostic`. Its truth is the value of $d_0$, and no limit is claimed
for a final learner, so it measures the greedy learner's regret, which no claim covers.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `misspec__L` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the default sieve with its size chosen by cross-validation and the declared final learner HistGradientBoostingRegressor(max_depth=2): the declared fit at n = 16,000 with the study's logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9447 to 0.9551, SE ratio 0.9792 to 1.0118 | pass |
| `interval_calibration` | `misspec__S` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose main-terms interaction logistic outcome learner has no W1^2 term, with `LearnedRule(loss='ep')` on the declared sieve Additive(BSpline(['W1'], 4), Indicators('W2')) at its fixed size 4: the declared fit at n = 16,000 with the study's logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9424 to 0.9529, SE ratio 0.9751 to 1.0081 | pass |
| `interval_calibration` | `nonexc_L__validation_rule_control` | control | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve with its size chosen by cross-validation and the declared final learner HistGradientBoostingRegressor(max_depth=2): the same fit at n = 2,000 with each fold's EP curve fitted on that fold's own validation rows, so the rule has seen the rows that evaluate it | the coverage interval must fall below the calibration band | coverage 0.0075 to 0.0423, SE ratio 0.9277 to 1.0892 | pass |
| `interval_calibration` | `nonexc_S__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the declared sieve Additive(BSpline(['W1'], 4), Indicators('W2')) at its fixed size 4: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8207 to 0.8384, SE ratio 0.6862 to 0.7096 | pass |
| `interval_calibration` | `nonexc__L` | positive | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the default sieve with its size chosen by cross-validation and the declared final learner HistGradientBoostingRegressor(max_depth=2): the declared fit at n = 16,000 with the study's logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9541, SE ratio 0.9816 to 1.0146 | pass |
| `interval_calibration` | `nonexc__S` | positive | the learned-rule law with the blip 0.1 + W1, whose main-terms interaction logistic outcome learner is correct, with `LearnedRule(loss='ep')` on the declared sieve Additive(BSpline(['W1'], 4), Indicators('W2')) at its fixed size 4: the declared fit at n = 16,000 with the study's logistic nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9426 to 0.9532, SE ratio 0.9804 to 1.0132 | pass |
| `interval_calibration` | `optimal__L` | diagnostic | the value of the optimal rule at the learned-rule law with the blip 0.1 + W1, with `LearnedRule(loss='ep')` on the default sieve with its size chosen by cross-validation and the declared final learner HistGradientBoostingRegressor(max_depth=2) on both adjustment columns and the treatment mechanism known, read against the fixed optimal value 0.577115: the declared fit at n = 16,000, which measures the learner's regret | none. The cell reports its coverage, its SE ratio with its interval and the mean regret, with no verdict | coverage 0.9365 to 0.9600, SE ratio 0.9489 to 1.0211 | reported |
| `interval_calibration` | `optimal__S` | positive | the value of the optimal rule at the learned-rule law with the blip 0.1 + W1, with `LearnedRule(loss='ep')` on the declared sieve Additive(BSpline(['W1'], 4), Indicators('W2')) at its fixed size 4 on both adjustment columns and the treatment mechanism known, read against the fixed optimal value 0.577115: the declared fit at n = 16,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9414 to 0.9521, SE ratio 0.9810 to 1.0140 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and both controls failed as designed.
No cell was red, so none of the declared red-cell readings applied. At n = 16,000 the interval is
calibrated for both declared curves at both laws, and for the optimal value with the declared
sieve. At n = 2,000 all four primary cells pass the framework screens, but three SE-ratio
intervals lie wholly below 1.

| reading | value | source |
| --- | --- | --- |
| calibration SE ratio, `nonexc__S` | 0.9965 | `properties.csv` |
| calibration SE ratio, `misspec__S` | 0.9919 | `properties.csv` |
| calibration SE ratio, `nonexc__L` | 0.9980 | `properties.csv` |
| calibration SE ratio, `misspec__L` | 0.9956 | `properties.csv` |
| calibration SE ratio, `optimal__S` | 0.9967 | `properties.csv` |
| calibration coverage, every gated positive | 0.9469 to 0.9501 | `properties.csv` |
| primary SE ratio | 0.9289 to 1.0096 | `performance-tests.csv` |
| primary coverage | 0.9305 to 0.9540 | `performance-tests.csv` |

No fit failed. A harness refit whose rule differed from the fit's rule would have stopped the
run, and every 25th calibration replication was refitted. The RM30 rule L4 counts a solver warning
and does not treat it as a failed fit: the property cells carry 13 warnings over 74,900 fits.

### The size of the interval

The SE ratio is the mean reported SE over the SD of the error.

| curve, law | n = 2,000 SE ratio (99% interval) | n = 2,000 coverage | n = 16,000 SE ratio (99% interval) | n = 16,000 coverage |
| --- | --- | --- | --- | --- |
| `S`, `non_exceptional` | 0.9289 (0.8926 to 0.9701) | 0.9305 | 0.9965 (0.9804 to 1.0132) | 0.9481 |
| `S`, `misspecified_limit` | 0.9522 (0.9161 to 0.9907) | 0.9415 | 0.9919 (0.9751 to 1.0081) | 0.9478 |
| `L`, `non_exceptional` | 1.0096 (0.9696 to 1.0498) | 0.9540 | 0.9980 (0.9816 to 1.0146) | 0.9491 |
| `L`, `misspecified_limit` | 0.9544 (0.9162 to 0.9949) | 0.9375 | 0.9956 (0.9792 to 1.0118) | 0.9501 |

The study declared its calibration claim at n = 16,000. At n = 2,000 the primary cells pass the
wide screens, 0.80 to 1.20 for the SE ratio and a coverage floor of 0.90. Three of the four
SE-ratio intervals lie wholly below 1, and at `S` on `non_exceptional` the SE is about 7% small
with coverage 0.9305. The default sieve read 0.994 and 0.983 at the same size, so a user of a
declared curve at that size should expect the interval to be somewhat narrow.

The oracle SE is the SD of the row-level curve at the fit's own fold rules. Over the error SD it
is 0.996 and 0.990 for `S` and 0.998 and 0.994 for `L` in the calibration cells, so the variance
estimator is right for the realized rules. What rule learning adds is a covariance between folds
whose rules are fitted on each other's validation rows, a second-order remainder.

### The fixed-rule fit on the same draws

Each `S` calibration draw also fits the fit whose rule is fixed at the `S` limit rule, on the same
sample and folds. Its error correlates with the learned fit's at about 0.95 to 0.96. The ratio of
the two error variances reads the cost of learning the rule. These ratios are recomputed from
`property-replicates.parquet`, and no verdict reads them.

| cell | fixed-rule SE ratio | learned over fixed-rule error variance |
| --- | --- | --- |
| `nonexc__S` | 1.0073 | 1.023 |
| `misspec__S` | 1.0074 | 1.036 |
| `optimal__S` | 1.0152 | 1.038 |

At n = 16,000 learning the rule from the declared sieve adds about 2% to 4% to the error variance.
The fixed-rule SE ratios sit just above 1, the fixed-rule CV-TMLE's own slightly conservative
variance. Every learned SE ratio is inside its band.

### The value of the optimal rule

With the declared sieve and $g_0$ known, the optimal cell reads an SE ratio of 0.9967 and coverage
0.9469 against the fixed $\Psi_{d_0}$. Its bias is $-0.00046$, standardized bias $-0.088$, and its
99% interval excludes zero and lies inside the margin of 0.25 SD. The mean regret of the fold
rules learned at n = 16,000 is 0.00047, so the bias is that regret. The `S` limit rule's own regret
is below 0.01 SE at this size (`tests/unit/test_learned_rule_ep_declared_design.py`).

With the declared learner the cell has no verdict. It reads an SE ratio of 0.9844 (99% interval
0.9489 to 1.0211) and coverage 0.9492 on 2,400 draws, with a mean regret of 0.00030. The reading
is good at this law, but no limit is claimed for a final learner, so the optimal value with a
declared final learner reads experimental.

### What the controls show

The shrunken-SE control reads an SE ratio of 0.6975, its interval ending at 0.7096, below 0.93.
The fold-locality control fits each fold's `L` curve on that fold's own validation rows. Its
coverage is 0.0200, with a bias of 4.08 error SDs, and its SE ratio is 1.0010, so the SE is
honest and the optimism is in the point. Passing shows that the instrument resolves a gross
fold-locality violation with the declared learner. The subtle leaks, the outer nuisances at the
training rows and a size step that sees its inner fold, rest on the unit witnesses EP2, W-c2 and
W-c3 in `tests/unit/test_learned_rule_ep.py`.

The readings do not show the items in this list.

- A declared final learner for the value of the optimal rule. It reads experimental.
- Another sieve family or size, or another final learner, beyond the two measured. The key holds
  neither, so such a fit reads supported and rests on condition C3, which the user judges.
- Weights, clusters, fold targeting, or `repeats` above 1 with the EP rule.
- A calibration below n = 16,000.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications per cell |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 4 | primary tests against the replication truth |
| `independent_tests_passed` | 4 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 7 | gated repeated-sampling property cells |
| `property_cells_passed` | 7 | cells whose own and family verdicts pass |
| `property_cells_reported` | 1 | diagnostic cells, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0379 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9305 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9146 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.8926 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0498 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-ep/S_nonexc_2k/ey_learned_rule[learned rule]]:coverage` | 0.9305 | primary coverage, `S` at `non_exceptional` |
| `performance[cleverly-learned-rule-ep/S_misspec_2k/ey_learned_rule[learned rule]]:coverage` | 0.9415 | primary coverage, `S` at `misspecified_limit` |
| `performance[cleverly-learned-rule-ep/L_nonexc_2k/ey_learned_rule[learned rule]]:coverage` | 0.9540 | primary coverage, `L` at `non_exceptional` |
| `performance[cleverly-learned-rule-ep/L_misspec_2k/ey_learned_rule[learned rule]]:coverage` | 0.9375 | primary coverage, `L` at `misspecified_limit` |
| `performance[cleverly-learned-rule-ep/S_nonexc_2k/ey_learned_rule[learned rule]]:se_ratio` | 0.9289 | primary SE ratio, `S` at `non_exceptional` |
| `performance[cleverly-learned-rule-ep/S_nonexc_2k/ey_learned_rule[learned rule]]:se_ratio_ci_lower` | 0.8926 | its lower endpoint |
| `performance[cleverly-learned-rule-ep/S_nonexc_2k/ey_learned_rule[learned rule]]:se_ratio_ci_upper` | 0.9701 | its upper endpoint |
| `performance[cleverly-learned-rule-ep/S_misspec_2k/ey_learned_rule[learned rule]]:se_ratio` | 0.9522 | primary SE ratio, `S` at `misspecified_limit` |
| `performance[cleverly-learned-rule-ep/S_misspec_2k/ey_learned_rule[learned rule]]:se_ratio_ci_upper` | 0.9907 | its upper endpoint |
| `performance[cleverly-learned-rule-ep/L_nonexc_2k/ey_learned_rule[learned rule]]:se_ratio` | 1.0096 | primary SE ratio, `L` at `non_exceptional` |
| `performance[cleverly-learned-rule-ep/L_misspec_2k/ey_learned_rule[learned rule]]:se_ratio` | 0.9544 | primary SE ratio, `L` at `misspecified_limit` |
| `performance[cleverly-learned-rule-ep/L_misspec_2k/ey_learned_rule[learned rule]]:se_ratio_ci_upper` | 0.9949 | its upper endpoint |
| `properties[interval_calibration/nonexc__S]:se_ratio` | 0.9965 | `S` calibration SE ratio at `non_exceptional` |
| `properties[interval_calibration/nonexc__S]:se_ratio_ci_lower` | 0.9804 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/nonexc__S]:se_ratio_ci_upper` | 1.0132 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/nonexc__S]:coverage` | 0.9481 | its coverage |
| `properties[interval_calibration/misspec__S]:se_ratio` | 0.9919 | `S` calibration SE ratio at `misspecified_limit` |
| `properties[interval_calibration/misspec__S]:se_ratio_ci_lower` | 0.9751 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/misspec__S]:se_ratio_ci_upper` | 1.0081 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/misspec__S]:coverage` | 0.9478 | its coverage |
| `properties[interval_calibration/nonexc__L]:se_ratio` | 0.9980 | `L` calibration SE ratio at `non_exceptional` |
| `properties[interval_calibration/nonexc__L]:se_ratio_ci_lower` | 0.9816 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/nonexc__L]:se_ratio_ci_upper` | 1.0146 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/nonexc__L]:coverage` | 0.9491 | its coverage |
| `properties[interval_calibration/misspec__L]:se_ratio` | 0.9956 | `L` calibration SE ratio at `misspecified_limit` |
| `properties[interval_calibration/misspec__L]:se_ratio_ci_lower` | 0.9792 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/misspec__L]:se_ratio_ci_upper` | 1.0118 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/misspec__L]:coverage` | 0.9501 | its coverage |
| `properties[interval_calibration/optimal__S]:se_ratio` | 0.9967 | optimal-value calibration SE ratio with `S` |
| `properties[interval_calibration/optimal__S]:se_ratio_ci_lower` | 0.9810 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/optimal__S]:se_ratio_ci_upper` | 1.0140 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/optimal__S]:coverage` | 0.9469 | its coverage |
| `properties[interval_calibration/optimal__S]:coverage_ci_lower` | 0.9414 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/optimal__S]:bias` | -0.000464 | its bias against the fixed optimal value |
| `properties[interval_calibration/optimal__S]:standardized_bias` | -0.0880 | the same bias in error SDs, against 0.25 |
| `properties[interval_calibration/optimal__S]:regret_mean` | 0.000468 | mean regret of the learned fold rules |
| `properties[interval_calibration/optimal__L]:se_ratio` | 0.9844 | optimal-value SE ratio with `L`, no verdict |
| `properties[interval_calibration/optimal__L]:se_ratio_ci_lower` | 0.9489 | its lower endpoint |
| `properties[interval_calibration/optimal__L]:se_ratio_ci_upper` | 1.0211 | its upper endpoint |
| `properties[interval_calibration/optimal__L]:coverage` | 0.9492 | its coverage |
| `properties[interval_calibration/optimal__L]:regret_mean` | 0.000304 | mean regret of the learned fold rules |
| `properties[interval_calibration/nonexc__S]:limit_se_ratio` | 1.0073 | fixed-rule SE ratio, `nonexc__S` |
| `properties[interval_calibration/misspec__S]:limit_se_ratio` | 1.0074 | fixed-rule SE ratio, `misspec__S` |
| `properties[interval_calibration/optimal__S]:limit_se_ratio` | 1.0152 | fixed-rule SE ratio, `optimal__S` |
| `properties[interval_calibration/nonexc__S]:oracle_se_ratio` | 0.9961 | oracle SE ratio, `nonexc__S` |
| `properties[interval_calibration/misspec__S]:oracle_se_ratio` | 0.9904 | oracle SE ratio, `misspec__S` |
| `properties[interval_calibration/nonexc__L]:oracle_se_ratio` | 0.9976 | oracle SE ratio, `nonexc__L` |
| `properties[interval_calibration/misspec__L]:oracle_se_ratio` | 0.9943 | oracle SE ratio, `misspec__L` |
| `properties[interval_calibration/nonexc_S__shrunken_se_control]:se_ratio` | 0.6975 | shrunken-SE control SE ratio |
| `properties[interval_calibration/nonexc_S__shrunken_se_control]:se_ratio_ci_upper` | 0.7096 | its upper endpoint, against 0.93 |
| `properties[interval_calibration/nonexc_L__validation_rule_control]:coverage` | 0.0200 | fold-locality control coverage |
| `properties[interval_calibration/nonexc_L__validation_rule_control]:coverage_ci_upper` | 0.0423 | its upper endpoint, against 0.92 |
| `properties[interval_calibration/nonexc_L__validation_rule_control]:standardized_bias` | 4.0832 | fold-locality control bias in error SDs |
| `properties[interval_calibration/nonexc_L__validation_rule_control]:se_ratio` | 1.0010 | fold-locality control SE ratio |
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
| The interval is calibrated at n = 16,000 only | At n = 2,000 all four primary cells pass the screens, but three SE-ratio intervals lie wholly below 1, and with the declared sieve at `non_exceptional` the SE is about 7% small. The support key holds no sample size, so a smaller fit also reads supported |
| One declared sieve and one declared learner | The study fits a fixed-size B-spline sieve and a shallow gradient-boosting final learner. The key holds neither, so another sieve or learner reads supported and rests on condition C3, that the fold rules converge to a fixed rule, which the package cannot check. A more variable learner may need a larger n |
| The optimal value with a declared learner is not claimed | No limit is claimed for a final learner, so `optimal__L` states no verdict and such a fit reads experimental |
| Two laws and two modifiers | The study fits RM30 `non_exceptional` and `misspecified_limit` with both adjustment columns as modifiers. The exceptional law is not refitted here; the EP rule-learner study reads it under-covering with the default sieve |
| The fold-locality control resolves a gross violation only | The subtle leaks rest on the unit witnesses EP2, W-c2 and W-c3 |
| Rows only, pooled targeting, one split | A fit with weights, fold targeting, clusters or `repeats` above 1 reads experimental |
| The calibration size is a point reading | X32a's rule gave no budget, so the declaration read each cell at its pooled pilot ratio, a stated plan deviation |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep_declared/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep_declared/run.log)
records the commit, the clean tree and the publish step's runtime. The declared run fitted every
replication and then stopped at the manifest step because the study module defined no
`CONFIGURATION`. A wrapper set that dictionary in memory at the unchanged commit and reused every
checkpoint, and `tests/canonical/provenance-revisions.md` records the step. The red-cell
readings, declared before any run, are in the docstring of
[`regenerate.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep_declared/regenerate.py),
whose command `python -m tests.canonical.learned_rule_ep_declared.regenerate` reruns the study.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep_declared/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_ep_declared/properties.csv)
carry every published row.
