# In-sample learned-rule value

This study validates the interval of the in-sample learned-rule value. The estimator is
`TMLE(learned_rule=LearnedRule(), cross_fit=False, cv_evaluation=False, g_bounds="auto")` with a
parametric outcome learner and a parametric treatment learner. It learns one plug-in rule $d_n$
from the outcome regression fitted on every row, and runs one regime TMLE at $d = d_n$ on the
same rows: one fluctuation solving $P_n D^*(d_n, \bar Q^*_n, g_n) = 0$, the plug-in estimate, and
the curve $\phi_i = Z_i - \hat\psi$ with $Z_i = H_i(Y_i - \bar Q^*_i) + \bar Q^*_i(d_n)$.
[One rule fitted on all rows](../point-treatment-tmle.md#learned-rules) gives the
construction and its conditions.

The target is data-adaptive: $\Psi_{d_n}(P_0)$, the value of the one rule learned on every row
(van der Laan and Luedtke 2015, Section 6 and Theorem 5, conditions C3a, C4a and C5a). Each
replication carries its own truth, and every statistic reads the error, the estimate minus that
truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** No maintained package fits the in-sample
learned-rule TMLE. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | the RM30 laws of the [gated study](learned-rule-cvtmle.md#what-was-tested): $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W)$ |
| fit | `TMLE(learned_rule=LearnedRule(), cross_fit=False, cv_evaluation=False, g_bounds="auto")` at n = 2,000 with pointwise intervals |
| outcome learner | `InteractionLogistic`: `PolynomialFeatures(2, interaction_only=True, include_bias=False)` then `LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)` on $(A, W_1, W_2)$, in a class marked `cleverly_learner_family = "parametric"`, because a bare `Pipeline` reads adaptive |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, a correct parametric model |
| support key | every fit is checked: `splitting` `("in_sample",)`, `outcome_learner_family` and `treatment_learner_family` both `parametric`, one pseudo-fold with weight 1 |
| truth | per replication: refit the outcome learner on every row with the package's `fit_on_rows`, require the fit's rule on every row, and integrate the rule's value by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$ with the known $\bar Q_0$ |
| stored variance | $\{n(n-1)\}^{-1}\sum_i (\phi_i - \bar\phi)^2$, with $\bar\phi = 0$ to solver tolerance, a factor $n/(n-1) = 1.0005$ over $n^{-2}\sum_i \phi_i^2$. The interval is $\hat\psi \pm 1.96\,\mathrm{SE}$ |
| primary budget | 14,400 replications for each law |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20263081` and `resampling_seed=20263082` |
| policy | `gated` |

| law | $b(W)$ | what the outcome learner can represent | n |
| --- | --- | --- | ---: |
| `non_exceptional` | $0.1 + W_1$ | the blip: the model is correct | 2,000 |
| `misspecified_limit` | $0.8 W_1 + W_1^2 - 0.3$ | no $W_1^2$ term, so the rule converges to the rule of the model's projection, whose value is 0.554749 against the optimal 0.560251; the correct treatment model carries the remainder | 2,000 |

The primary count is set by the bias leg, not by the SE rule. The in-sample point leans
optimistic, and the pre-run readings put that lean at +0.12 error SDs (`non_exceptional`, 4,000
pilot draws) and +0.21 (`misspecified_limit`, 8,000 pilot and design-review probe draws pooled).
14,400 is the smallest multiple of 400 at which both 99% bias intervals fall inside 0.25 SD with
predictive probability 0.95. The X11 plan, its review and the study design record fixed the laws,
the learners, the budgets, the seeds, the truth rule, the variance rule and the support-row
condition before any run. `tests/unit/test_learned_rule_in_sample_design.py` pins the bounds, the
limit rules, the sizing and the seeds, and `tests/unit/test_learned_rule_in_sample_harness.py`
checks the learners, the key, the stored variance, the truth and the property path.
`tests/unit/test_learned_rule_in_sample.py` holds the unit witnesses of the construction.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, an interaction logistic outcome learner with no W1^2 term | `ey_learned_rule[learned rule]` | value of the one rule learned on every row | `cleverly` in-sample TMLE of the learned-rule value: one plug-in rule learned from the outcome regression fitted on every row, and one regime fluctuation at that rule on the same rows | 0.0028 to 0.0034 | 0.9469 | 1.0095 | pass |
| learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner | `ey_learned_rule[learned rule]` | value of the one rule learned on every row | `cleverly` in-sample TMLE of the learned-rule value: one plug-in rule learned from the outcome regression fitted on every row, and one regime fluctuation at that rule on the same rows | 0.0020 to 0.0026 | 0.9501 | 1.0099 | pass |
<!-- /generated -->

## Theory properties

Every property cell fits n = 2,000 with the declared treatment learner.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each positive lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, with 7,200 replications of each law | a shrunken-SE control multiplies each SE by 0.80, and a noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. On `non_exceptional`, `interpolating_control` fits an interpolating tree (`DecisionTreeClassifier(min_samples_leaf=1)`, probabilities clipped to [0.01, 0.99]) in sample on the same draw and reads it against its own rule's value: C5a fails, and its curve understates the spread of its error. Each control must put its SE-ratio interval below 0.93 |
| `targeting_necessity` | does the fluctuation remove the plug-in's bias, with 1,200 replications of `non_exceptional` | the positive arm fits `WithoutW1MainLogistic`, a logistic outcome model on $(A, W_2, A W_1, A W_2)$ marked parametric, which misses the control arm's $W_1$ slope. The control is the same fit's initial plug-in at its own rule. The positive's bias interval must lie inside the margin, the control's outside it, and the two means must sit 0.5 positive-arm SDs or more apart |
| `weak_limit_necessity` | the same pair with RM30's `drop_w1` learner, whose limit benefit is 0.037 and 0.049 at the two values of $W_2$, too weak at n = 2,000 for one rule to settle | nothing fails. The cells have the role `diagnostic`, and no margin is read. The summary publishes the share of draws whose rule differs from the limit rule, and the targeted arm's bias on settled and on flipped draws |
| `target_distinction` | how far the in-sample target sits from the fold average $(1/10)\sum_v \Psi_{d_v}(P_0)$ of the rules the declared learner learns on ten training folds of the same draw, the target of the cross-fitted fit | nothing fails. The cells have the role `diagnostic`, and no margin is read |

The noise scale is the SD of $\sqrt{n}$ times the estimate at each law's limit rule, and no
verdict reads it as an efficiency bound. The calibration size was set before the run by the
per-arm sizing rule over the two positives on a 4,000-draw pilot. The plan named `drop_w1` as the
necessity learner. The design replaced it before the run, because its rule does not settle at
this n: the learner is saturated in $(A, W_2)$, so its benefit at each level of $W_2$ is a
difference of two cell means, with z-values of 1.18 and 1.53 at n = 2,000. From the exact law its
rule flips in about 17.5% of samples. `WithoutW1MainLogistic` was chosen on its limit benefit,
which crosses zero once at an interior $W_1$, so sampling error moves its boundary a little and
flips no whole level.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `misspecified_limit__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term: the declared in-sample fit with the study's parametric logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9393 to 0.9531, SE ratio 0.9900 to 1.0343 | pass |
| `interval_calibration` | `misspecified_limit__noise_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8240 to 0.8467, SE ratio 0.6991 to 0.7286 | pass |
| `interval_calibration` | `misspecified_limit__shrunken_se_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8664 to 0.8865, SE ratio 0.7920 to 0.8266 | pass |
| `interval_calibration` | `non_exceptional__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct: the declared in-sample fit with the study's parametric logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9440 to 0.9573, SE ratio 0.9912 to 1.0336 | pass |
| `interval_calibration` | `non_exceptional__interpolating_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct: an interpolating tree (min_samples_leaf=1, probabilities clipped to [0.01, 0.99]), an adaptive outcome learner fitted in sample on the same draw and read against the value of its own rule | the SE-ratio interval must fall below the calibration band | coverage 0 to 0.000736, SE ratio 0.3830 to 0.3992 | pass |
| `interval_calibration` | `non_exceptional__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8214 to 0.8442, SE ratio 0.6959 to 0.7258 | pass |
| `interval_calibration` | `non_exceptional__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8713 to 0.8910, SE ratio 0.7929 to 0.8272 | pass |
| `target_distinction` | `misspecified_limit__fold_average` | diagnostic | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term: every declared calibration fit read against the fold average (1/10) sum_v Psi_{d_v}(P0) of the rules the declared learner learns on ten training folds of the same draw | none. The cell reports the mean gap between the two targets, its SD over the error SD, and the share of the grid on which the fold rules differ from the full-sample rule | bias 0.0031 to 0.0040, coverage 0.9402 to 0.9539, truth-gap SD / error SD 0.0150, grid disagreement 0.0241 | reported |
| `target_distinction` | `non_exceptional__fold_average` | diagnostic | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct: every declared calibration fit read against the fold average (1/10) sum_v Psi_{d_v}(P0) of the rules the declared learner learns on ten training folds of the same draw | none. The cell reports the mean gap between the two targets, its SD over the error SD, and the share of the grid on which the fold rules differ from the full-sample rule | bias 0.0020 to 0.0029, coverage 0.9446 to 0.9578, truth-gap SD / error SD 0.0081, grid disagreement 0.0189 | reported |
| `targeting_necessity` | `non_exceptional__targeted` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct: the in-sample fit with WithoutW1MainLogistic, a parametric outcome learner with no W1 main term, so targeting does the adjusting | bias interval inside the equivalence margin | bias 0.000577 to 0.0028, margin 0.0038 | pass |
| `targeting_necessity` | `non_exceptional__untargeted` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct: the same fit's initial plug-in mean at its own rule, with no fluctuation | bias interval must fall entirely outside the margin | bias 0.0286 to 0.0305, margin 0.0032 | pass |
| `weak_limit_necessity` | `non_exceptional_drop_w1__targeted` | diagnostic | the learned-rule law with the blip 0.1 + W1: the in-sample fit with RM30's drop_w1 outcome learner, whose limit benefit (0.037 and 0.049) is too weak at n = 2,000 for its rule to settle | none. The cell reports the bias, the share of draws whose rule flipped, and the bias on settled and on flipped draws | bias 0.0038 to 0.0062, coverage 0.9182 to 0.9549, flip share 0.1942, bias on settled draws 0.1323 SD, on flipped draws 1.0712 SD | reported |
| `weak_limit_necessity` | `non_exceptional_drop_w1__untargeted` | diagnostic | the learned-rule law with the blip 0.1 + W1: the same drop_w1 fit's initial plug-in mean at its own rule, with no fluctuation | none. The cell reports the bias and the displacement from the targeted arm | bias 0.0194 to 0.0216, coverage 0.7451 to 0.8077 | reported |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed. The 2 primary tests read standardized
biases of 0.156 (`non_exceptional`) and 0.211 (`misspecified_limit`), coverage of 0.9501 and
0.9469, and SE ratios of 1.010 and 1.009. The 2 calibration positives read SE ratios of 1.012
and coverage of 0.9510 and 0.9465, and every interval lies inside its band. Each control fails
as declared, the interpolating tree by far: an SE ratio of 0.391, coverage of 0, and a
standardized bias of 7.93 against its own rule's value.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 2 of 2 | `performance-tests.csv` |
| gated property cells passed | 9 of 9 | `properties.csv` |
| diagnostic cells, reported with no verdict | 4 | `properties.csv`, `target_distinction` and `weak_limit_necessity` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |
| smallest calibration SE-ratio lower endpoint | 0.9900 | `properties.csv`, `misspecified_limit__declared` |
| smallest calibration coverage lower endpoint | 0.9393 | `properties.csv`, `misspecified_limit__declared` |

The support row was declared before the run: a P11 row with `splitting` `in_sample` and both
learner families `parametric`, if and only if the primary tests of both laws, every
`interval_calibration` verdict and the `targeting_necessity` verdicts passed. They did, so the
registered row `T43` covers the plug-in rule fitted in sample with a parametric outcome learner
and a parametric treatment learner.

### The in-sample point leans optimistic

The fit chooses the rule and estimates its value on the same rows, so the estimate leans above
the value of the rule it chose. At n = 2,000 the lean is +0.156 error SDs on `non_exceptional`
and +0.211 on `misspecified_limit`, close to the pre-run readings. A passed bias leg means
"within 0.25 SD at n = 2,000", not "unbiased". The `misspecified_limit` calibration positive,
with half the draws, reads +0.224 SD, and its 99% bias interval reaches just past 0.25; the
calibration verdict reads its SE ratio and coverage, which pass. The bias was measured at
n = 2,000 only. Theory makes it vanish beside the SE as n grows once the rule settles (C3a), and
smaller samples lean further.

### A rule that does not settle

The `weak_limit_necessity` cells show the condition the interval needs. With RM30's `drop_w1`
learner the rule differed from its limit rule in 19.4% of draws, near the 17.5% the exact law
predicts. The targeted arm's error was +0.13 SD on settled draws and +1.07 SD on flipped ones,
+0.31 SD overall. Its coverage was 0.938 over 1,200 draws, and its 99% coverage interval
reaches below 0.92, so this cell, which reads no verdict, would not pass the calibration band.
Cross-fitted, the same
learner on the same law reads -0.009 SD in the
[gated study](learned-rule-cvtmle.md), because a flipping rule does not bias a fold target. The
row covers fixed-form outcome models whose limit benefit is near zero, and no check in the
package detects them.

### What the verdicts cannot see

The calibration verdicts cannot tell the in-sample target from the fold average of the
cross-fitted fit at this size. The SD of the gap between the two targets is 0.008 error SDs on
`non_exceptional` and 0.015 on `misspecified_limit`, and the fold rules disagree with the
full-sample rule on about 2% of the grid. Read against the fold average, the declared fit covers
0.9515 and 0.9474. The unit witnesses of `tests/unit/test_learned_rule_in_sample.py` establish
which rule the truth reads: each row's rule is the sign of the blip refitted on every row, and
the rule of cross-fitted predictions fails that check. The witness that supplied treatment
probabilities enter the clever covariate is also a unit witness that no verdict here sees.

### Solver warnings

The harness counts the scikit-learn and `cleverly` convergence warnings of each fit and its truth
refit. No primary fit and no necessity fit warned. One `misspecified_limit` calibration draw of
7,200 (replicate 6,472) raised two lbfgs warnings, one in the declared fit and one in its truth
refit of the same model. The truth check found the fit's rule on all 2,000 rows, and the draw
was kept. The three `misspecified_limit` calibration cells and the `misspecified_limit`
`target_distinction` cell read that draw.

The eleven refits that build the `target_distinction` target run outside the counter, so the
artifacts do not record their warnings. No rule reads the warning counts.

The readings do not show the items in this list.

- Weights, clusters, strata, a rule learner or supplied treatment probabilities in sample. Each
  reads experimental.
- Continuous outcomes, which the in-sample fit scales by their observed range, and more than two
  arms.
- Another sample size or learner. The key holds the learner family, not the learner.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the SD of $\sqrt{n}$ times the estimate at the limit rule over $\sqrt{2000}$. They size the noise
controls only, and no verdict reads them as an efficiency bound. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 14400 | primary replications per law |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | parameter tests against the truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 9 | gated repeated-sampling property cells |
| `property_cells_passed` | 9 | cells whose own and family verdicts pass |
| `property_cells_reported` | 4 | diagnostic cells, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.2114 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9469 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9419 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9949 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0258 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-in-sample/non_exceptional/ey_learned_rule[learned rule]]:standardized_bias` | 0.1557 | primary standardized bias, `non_exceptional` |
| `performance[cleverly-learned-rule-in-sample/misspecified_limit/ey_learned_rule[learned rule]]:standardized_bias` | 0.2114 | primary standardized bias, `misspecified_limit` |
| `performance[cleverly-learned-rule-in-sample/non_exceptional/ey_learned_rule[learned rule]]:coverage` | 0.9501 | primary coverage, `non_exceptional` |
| `performance[cleverly-learned-rule-in-sample/misspecified_limit/ey_learned_rule[learned rule]]:coverage` | 0.9469 | primary coverage, `misspecified_limit` |
| `performance[cleverly-learned-rule-in-sample/non_exceptional/ey_learned_rule[learned rule]]:bias_ci_upper` | 0.0026 | upper 99% bias endpoint, `non_exceptional` |
| `performance[cleverly-learned-rule-in-sample/non_exceptional/ey_learned_rule[learned rule]]:bias_margin` | 0.0037 | its margin, 0.25 error SDs |
| `performance[cleverly-learned-rule-in-sample/misspecified_limit/ey_learned_rule[learned rule]]:bias_ci_upper` | 0.0034 | upper 99% bias endpoint, `misspecified_limit` |
| `performance[cleverly-learned-rule-in-sample/misspecified_limit/ey_learned_rule[learned rule]]:bias_margin` | 0.0037 | its margin |
| `summary[cleverly-learned-rule-in-sample/non_exceptional/ey_learned_rule[learned rule]]:truth_min` | 0.5506 | the smallest `non_exceptional` truth |
| `summary[cleverly-learned-rule-in-sample/misspecified_limit/ey_learned_rule[learned rule]]:truth_min` | 0.5140 | the smallest `misspecified_limit` truth |
| `mean_oracle_se_non_exceptional` | 0.0149 | mean oracle SE at the fit's rule, `non_exceptional` |
| `mean_oracle_se_misspecified_limit` | 0.0149 | the same, `misspecified_limit` |
| `properties[interval_calibration/non_exceptional__declared]:se_ratio` | 1.0123 | calibration SE ratio, `non_exceptional` |
| `properties[interval_calibration/non_exceptional__declared]:coverage` | 0.9510 | calibration coverage, `non_exceptional` |
| `properties[interval_calibration/misspecified_limit__declared]:se_ratio` | 1.0116 | calibration SE ratio, `misspecified_limit` |
| `properties[interval_calibration/misspecified_limit__declared]:se_ratio_ci_lower` | 0.9900 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/misspecified_limit__declared]:coverage` | 0.9465 | calibration coverage, `misspecified_limit` |
| `properties[interval_calibration/misspecified_limit__declared]:coverage_ci_lower` | 0.9393 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/misspecified_limit__declared]:standardized_bias` | 0.2238 | calibration standardized bias, `misspecified_limit` |
| `properties[interval_calibration/misspecified_limit__declared]:bias_ci_upper` | 0.0038 | its upper 99% endpoint |
| `properties[interval_calibration/misspecified_limit__declared]:bias_margin` | 0.0037 | against this margin, which the calibration verdict does not read |
| `properties[interval_calibration/non_exceptional__interpolating_control]:se_ratio` | 0.3908 | interpolating control SE ratio |
| `properties[interval_calibration/non_exceptional__interpolating_control]:se_ratio_ci_upper` | 0.3992 | its upper endpoint, against 0.93 |
| `properties[interval_calibration/non_exceptional__interpolating_control]:coverage` | 0 | interpolating control coverage |
| `properties[interval_calibration/non_exceptional__interpolating_control]:standardized_bias` | 7.9305 | its standardized bias against its own rule's value |
| `properties[interval_calibration/non_exceptional__shrunken_se_control]:se_ratio_ci_upper` | 0.8272 | the highest shrunken-SE control upper endpoint |
| `properties[interval_calibration/misspecified_limit__noise_control]:se_ratio_ci_upper` | 0.7286 | the highest noise-control upper endpoint |
| `properties[targeting_necessity/non_exceptional__targeted]:standardized_bias` | 0.1129 | necessity positive standardized bias |
| `properties[targeting_necessity/non_exceptional__targeted]:coverage` | 0.9425 | its coverage |
| `properties[targeting_necessity/non_exceptional__untargeted]:standardized_bias` | 2.3420 | necessity control standardized bias |
| `properties[targeting_necessity/non_exceptional__targeted]:targeting_displacement` | 1.8536 | displacement between the arms, in positive-arm SDs, against 0.5 |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__targeted]:weak_limit_flip_share` | 0.1942 | share of `drop_w1` draws whose rule flipped |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__targeted]:weak_limit_settled_bias` | 0.1323 | targeted mean error on settled draws, in its error SD |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__targeted]:weak_limit_flipped_bias` | 1.0712 | the same on flipped draws |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__targeted]:standardized_bias` | 0.3146 | targeted standardized bias on every draw |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__targeted]:coverage` | 0.9383 | targeted coverage |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__targeted]:coverage_ci_lower` | 0.9182 | its lower endpoint |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__targeted]:se_ratio` | 1.0187 | targeted SE ratio |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__untargeted]:standardized_bias` | 1.3734 | untargeted standardized bias |
| `properties[weak_limit_necessity/non_exceptional_drop_w1__targeted]:weak_limit_displacement` | 0.9742 | displacement between the arms, in targeted-arm SDs |
| `properties[target_distinction/non_exceptional__fold_average]:mean_truth_gap` | 0.000154 | mean in-sample truth minus fold-average truth, `non_exceptional` |
| `properties[target_distinction/non_exceptional__fold_average]:truth_gap_sd_ratio` | 0.0081 | SD of that gap over the error SD |
| `properties[target_distinction/non_exceptional__fold_average]:mean_grid_disagreement` | 0.0189 | mean share of the grid where a fold rule differs from the full-sample rule |
| `properties[target_distinction/non_exceptional__fold_average]:coverage` | 0.9515 | coverage against the fold-average target |
| `properties[target_distinction/misspecified_limit__fold_average]:mean_truth_gap` | 0.000208 | mean truth gap, `misspecified_limit` |
| `properties[target_distinction/misspecified_limit__fold_average]:truth_gap_sd_ratio` | 0.0150 | SD of that gap over the error SD |
| `properties[target_distinction/misspecified_limit__fold_average]:mean_grid_disagreement` | 0.0241 | mean grid disagreement |
| `properties[target_distinction/misspecified_limit__fold_average]:coverage` | 0.9474 | coverage against the fold-average target |
| `bound:non_exceptional_standard_error` | 0.0149 | the limit-rule SD over the root of n, `non_exceptional` |
| `bound:misspecified_limit_standard_error` | 0.0149 | the same, `misspecified_limit` |
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
| `margin:subject_failures_allowed` | 144 | failed primary fits that bound allows per law |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits the in-sample learned-rule TMLE. The row rests on accuracy against the truth and on the theory properties |
| One size and one learner per role | The study fits n = 2,000 with the RM30 interaction logistic outcome model, marked parametric, a correct logistic treatment model and the plug-in rule, on binary outcomes and treatments, iid rows and no weights. The key holds the learner family, not the learner or n, so another parametric learner or n reads supported and was not measured |
| Compositions that were not measured | Weights, clusters, strata, a rule learner, supplied treatment probabilities, continuous outcomes and more than two arms in sample match no row and read experimental |
| The point leans optimistic | +0.16 and +0.21 error SDs at n = 2,000, measured at that size only. Theory makes the lean vanish beside the SE as n grows once the rule settles (C3a); smaller samples lean further |
| The interval needs the rule to settle | With an outcome model whose limit benefit is near zero, the rule flipped in 19.4% of draws and the point leaned +1.07 SD on those draws. The row covers such fits, and no check in the package detects them |
| A data-adaptive target | The target is the value of the one rule learned on every row. At this size the verdicts cannot tell it from the fold-average target of the cross-fitted fit. Which rule the truth reads rests on the unit witnesses |
| Adaptive learners are outside the row | An adaptive learner in sample keys `adaptive` and warns with `InSampleLearnerWarning`. The interpolating control shows why: SE ratio 0.391 and coverage 0 |
| The necessity learner was replaced before the run | `drop_w1`'s rule does not settle at this n (C3a), so `WithoutW1MainLogistic` carries the verdict and `drop_w1` is published as a diagnostic |
| One calibration draw with solver warnings | One `misspecified_limit` positive draw of 7,200 raised two lbfgs warnings and was kept. The fold refits of the `target_distinction` target are not counted |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_in_sample/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_in_sample/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_in_sample/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_in_sample/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_in_sample/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_in_sample/properties.csv)
carry every published row.
