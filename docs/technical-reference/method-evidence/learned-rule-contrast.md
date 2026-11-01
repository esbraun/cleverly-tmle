# Learned-rule contrasts with known regimes

This study validates the intervals of a learned-rule fit that reports known regimes beside the
rule. The estimator is `TMLE(learned_rule=LearnedRule(), interventions=[Static(1), Static(0)],
n_folds=10, targeting_scheme="pooled", g_bounds="auto")`. Each fold learns the plug-in rule
$d_v$ of its outcome regression on its training rows. The rule is column 0 of the fit's regime
set and each known regime one more column, on the same folds and the same fitted $g_v$, and one
pooled fluctuation targets every column. One fit reports five parameters: the learned-rule value,
the values of `Static(1)` and `Static(0)`, and the two contrasts `ate_learned_rule`, the
learned-rule value minus each known value.

A contrast's curve is $D_L - D_K$, the difference of the two curves on the same rows, and the
report's variance rule is applied to it.
[Contrasts with known regimes](../point-treatment-tmle.md#learned-rules) gives the construction
and the joint-normality step it rests on.

The learned-rule target is data-adaptive: $\tilde\psi_{0n} = \sum_v \omega_v \Psi_{d_v}(P_0)$,
the value of the rules the folds learned, with $\omega_v = 1/V$ under the fold-evaluated report
and $n_v/n$ under the stacked one. A known regime's target is $E_0 \bar Q_0(a, W)$, and a
contrast's is the difference. Each replication carries its own truth, and every statistic reads
the error, the estimate minus that truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** No maintained package fits a cross-validated TMLE
of a learned rule beside known regimes. A zero-row equivalence artifact records the absence of a
comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | the RM30 laws of the [gated study](learned-rule-cvtmle.md#what-was-tested): $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W)$ |
| fit | `TMLE(learned_rule=LearnedRule(), interventions=[Static(1), Static(0)], n_folds=10, targeting_scheme="pooled", g_bounds="auto")` at n = 2,000 with `repeats=1` and pointwise intervals, under both reports on one fold seed: `cv_evaluation=True` (fold-evaluated, the primary report) and `cv_evaluation=False` (stacked) |
| outcome learner | `Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), LogisticRegression(C=1e6, max_iter=5000, tol=1e-10))` on $(A, W_1, W_2)$ |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, a correct parametric model fitted by maximum likelihood |
| support key | every fit is checked: family `P11`, `splitting` `("fold_evaluated",)` or `("stacked",)`, qualifier `known_regimes` |
| truth | per replication: refit each fold's outcome learner on the fit's training rows with the package's `fit_on_rows`, require the fit's rule on every validation row, integrate each fold rule's value by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$ with the known $\bar Q_0$, and weight fold $v$ by the report's $\omega_v$, checked against the record's fold weights. A known regime's truth is $E\bar Q_0(a, W)$ by the same rule |
| stored variance | fold-evaluated: $V^{-2}\sum_v n_v^{-2}\sum_{i\in v} D_i^2$ of each curve. Stacked: $\{n(n-1)\}^{-1}\sum_i(\phi_i - \bar\phi)^2$ of the stored curve. A contrast reads the difference of the two curves. The interval is $\hat\psi \pm 1.96\,\mathrm{SE}$ |
| primary budget | 1,200 replications of `non_exceptional`, fold-evaluated report, all five parameters |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20263091` and `resampling_seed=20263092` |
| policy | `gated` |

| law | $b(W)$ | what the outcome learner can represent | role | n |
| --- | --- | --- | --- | ---: |
| `non_exceptional` | $0.1 + W_1$ | the blip: the model is correct | primary and `interval_calibration` | 2,000 |
| `misspecified_limit` | $0.8 W_1 + W_1^2 - 0.3$ | no $W_1^2$ term, so the treated arm is misspecified. The untreated arm $\bar Q_0(0, \cdot)$ is the same in both laws, and the model represents it to within 0.005 | `conservative_coverage` | 2,000 |

The true contrasts are nonzero and differ from each other on both laws. On `non_exceptional` the
optimal value is 0.57711, against 0.53162 for `Static(1)` and 0.51218 for `Static(0)`; on
`misspecified_limit` the same values are 0.56025, 0.51083 and 0.51218. The primary law is
`non_exceptional` only, because the primary gate's SE band is an SE claim, which a misspecified
outcome model does not license. The laws, the learners, the budgets, the seeds, the truth rule,
the variance rules and the support-row condition were fixed before any run, in the X11 plan, its
review, the Corollary 3 reading and the study design record. `tests/unit/test_learned_rule_contrast_design.py`
pins the bounds, the true contrasts, the necessity learner's limit, the sizing and the seeds, and
`tests/unit/test_learned_rule_contrast_harness.py` checks the fits, the truth, the controls and
the property path. `tests/unit/test_learned_rule_contrast.py` holds the unit witnesses of the
construction.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner, fold-evaluated report | `ate_learned_rule[learned rule vs always 0]` | the learned-rule value minus the mean outcome if no one is treated | `cleverly` cross-validated TMLE of the learned-rule value beside two known regimes: each fold's plug-in rule and Static(1) and Static(0) as columns of one regime set on the same folds, one pooled fluctuation over every column, and each contrast's curve the difference of the two curves | -0.0015 to 0.000933 | 0.9367 | 0.9696 | pass |
| learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner, fold-evaluated report | `ate_learned_rule[learned rule vs always 1]` | the learned-rule value minus the mean outcome if everyone is treated | `cleverly` cross-validated TMLE of the learned-rule value beside two known regimes: each fold's plug-in rule and Static(1) and Static(0) as columns of one regime set on the same folds, one pooled fluctuation over every column, and each contrast's curve the difference of the two curves | -0.0015 to 0.000644 | 0.9450 | 1.0093 | pass |
| learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner, fold-evaluated report | `ey_known_regime[always 0]` | mean outcome if no one is treated | `cleverly` cross-validated TMLE of the learned-rule value beside two known regimes: each fold's plug-in rule and Static(1) and Static(0) as columns of one regime set on the same folds, one pooled fluctuation over every column, and each contrast's curve the difference of the two curves | -0.0010 to 0.0013 | 0.9475 | 1.0133 | pass |
| learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner, fold-evaluated report | `ey_known_regime[always 1]` | mean outcome if everyone is treated | `cleverly` cross-validated TMLE of the learned-rule value beside two known regimes: each fold's plug-in rule and Static(1) and Static(0) as columns of one regime set on the same folds, one pooled fluctuation over every column, and each contrast's curve the difference of the two curves | -0.000924 to 0.0015 | 0.9442 | 0.9747 | pass |
| learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner, fold-evaluated report | `ey_learned_rule[learned rule]` | fold average of the values of the rules learned on each training fold | `cleverly` cross-validated TMLE of the learned-rule value beside two known regimes: each fold's plug-in rule and Static(1) and Static(0) as columns of one regime set on the same folds, one pooled fluctuation over every column, and each contrast's curve the difference of the two curves | -0.0013 to 0.000964 | 0.9442 | 0.9914 | pass |
<!-- /generated -->

## Theory properties

Every property cell fits n = 2,000 with the declared treatment learner, and a law draw fits both
reports on one fold seed.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each of the ten positives (five parameters under two reports) on `non_exceptional` lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, with 5,600 replications | a shrunken-SE control multiplies each SE by 0.80, and a noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. Each control's SE-ratio interval must fall below the band |
| `conservative_coverage` | on `misspecified_limit`, does each positive keep its bias inside the margin and the lower end of its exact coverage interval at or above 0.92, with 3,200 replications. The SE ratio is published and no verdict reads it | the control multiplies each SE by 0.60, and its coverage interval must lie wholly below 0.92 |
| `contrast_covariance` | does each contrast's SE keep the covariance of the learned and the known value | each contrast of both laws and reports is read with the SE $\lvert se_L - se_K\rvert$ it would have at correlation one, and its coverage interval must lie wholly below 0.80 |
| `independent_se_diagnostic` | how each contrast covers with the SE $\sqrt{se_L^2 + se_K^2}$ it would have if the two values were independent | nothing fails. The cells have the role `diagnostic`, and no margin is read |
| `targeting_necessity` | does the fluctuation remove the plug-in's bias of the contrast against `Static(0)`, with 1,200 replications of `non_exceptional` under the fold-evaluated report | the positive arm fits RM30's `drop_w1` learner, saturated in $(A, W_2)$ with no $W_1$ term. The control is the same fit's untargeted contrast, the difference of the fold-weighted initial plug-ins. The positive's bias interval must lie inside the margin, the control's outside it, and the two means must sit 0.5 positive-arm SDs or more apart |

The noise scale is the SD of $\sqrt{n}$ times each estimate at the limit rule of the declared
learner, and no verdict reads it as an efficiency bound. The calibration and coverage sizes were
set before the run from a 4,000-draw pilot, over the five fold-evaluated parameters: at ten equal
folds the stacked point is the fold-evaluated point, so its cells are not separate arms. The
coverage floor 0.92 is the calibration band's lower end, and the control's shrink of 0.60 makes
it fail at any SE ratio up to about 1.45. The correlation-one ceiling of 0.80 was recorded after
the pilot and before any declared draw. It lies below the claim's own floor of 0.92, so it is the
stricter choice.

The necessity pair reads the contrast against `Static(0)` only. The `drop_w1` learner's limit
benefit is 0.037 and 0.049 at the two values of $W_2$, so its limit rule treats every row, which
is `Static(1)`. On every fold whose rule settles there, the contrast against `Static(1)` and its
curve are exactly zero: the pilot found exact zeros in 62% of draws. The untargeted contrast
against `Static(0)` converges to the confounded difference of cell means, 0.0234 above the truth.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_contrast_0__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000948 to 0.000446, margin 0.0038, coverage 0.9307 to 0.9523, SE ratio 0.9913 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_contrast_0__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7259 to 0.7658, SE ratio 0.5948 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_contrast_1__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(1) under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000876 to 0.000557, margin 0.0039, coverage 0.9362 to 0.9569, SE ratio 0.9956 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_contrast_1__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(1) under the fold-evaluated report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7326 to 0.7722, SE ratio 0.5974 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_known_0__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the value of Static(0) under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000346 to 0.0011, margin 0.0039, coverage 0.9403 to 0.9603, SE ratio 1.0028 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_known_0__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the value of Static(0) under the fold-evaluated report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7342 to 0.7737, SE ratio 0.6017 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_known_1__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the value of Static(1) under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000458 to 0.000986, margin 0.0040, coverage 0.9399 to 0.9600, SE ratio 1.0040 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_known_1__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the value of Static(1) under the fold-evaluated report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7464 to 0.7853, SE ratio 0.6024 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_value__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000581 to 0.000790, margin 0.0038, coverage 0.9392 to 0.9594, SE ratio 0.9968 | pass |
| `conservative_coverage` | `misspecified_limit_fold_evaluated_value__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value under the fold-evaluated report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7375 to 0.7768, SE ratio 0.5981 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_contrast_0__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(0) under the stacked report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000948 to 0.000446, margin 0.0038, coverage 0.9307 to 0.9523, SE ratio 0.9911 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_contrast_0__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(0) under the stacked report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7259 to 0.7658, SE ratio 0.5947 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_contrast_1__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(1) under the stacked report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000876 to 0.000557, margin 0.0039, coverage 0.9362 to 0.9569, SE ratio 0.9954 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_contrast_1__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(1) under the stacked report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7326 to 0.7722, SE ratio 0.5973 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_known_0__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the value of Static(0) under the stacked report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000346 to 0.0011, margin 0.0039, coverage 0.9403 to 0.9603, SE ratio 1.0026 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_known_0__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the value of Static(0) under the stacked report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7342 to 0.7737, SE ratio 0.6016 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_known_1__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the value of Static(1) under the stacked report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000458 to 0.000986, margin 0.0040, coverage 0.9403 to 0.9603, SE ratio 1.0039 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_known_1__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the value of Static(1) under the stacked report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7461 to 0.7850, SE ratio 0.6023 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_value__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value under the stacked report: the declared fit beside Static(1) and Static(0), with the correct logistic treatment model fitted by maximum likelihood | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.000581 to 0.000790, margin 0.0038, coverage 0.9392 to 0.9594, SE ratio 0.9967 | pass |
| `conservative_coverage` | `misspecified_limit_stacked_value__shrunken_coverage_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value under the stacked report: the declared fit's standard errors multiplied by 0.60 | the coverage interval must lie wholly below 0.92 | coverage 0.7375 to 0.7768, SE ratio 0.5980 | pass |
| `contrast_covariance` | `misspecified_limit_fold_evaluated_contrast_0__correlation_one_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the declared contrast with the standard error abs(se_L - se_K), which it would have if the learned and the known value had correlation one | the coverage interval must lie wholly below 0.80 | coverage 0.0352 to 0.0543, SE ratio 0.0306 | pass |
| `contrast_covariance` | `misspecified_limit_fold_evaluated_contrast_1__correlation_one_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(1) under the fold-evaluated report: the declared contrast with the standard error abs(se_L - se_K), which it would have if the learned and the known value had correlation one | the coverage interval must lie wholly below 0.80 | coverage 0.0774 to 0.1038, SE ratio 0.0581 | pass |
| `contrast_covariance` | `misspecified_limit_stacked_contrast_0__correlation_one_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(0) under the stacked report: the declared contrast with the standard error abs(se_L - se_K), which it would have if the learned and the known value had correlation one | the coverage interval must lie wholly below 0.80 | coverage 0.0352 to 0.0543, SE ratio 0.0305 | pass |
| `contrast_covariance` | `misspecified_limit_stacked_contrast_1__correlation_one_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(1) under the stacked report: the declared contrast with the standard error abs(se_L - se_K), which it would have if the learned and the known value had correlation one | the coverage interval must lie wholly below 0.80 | coverage 0.0771 to 0.1035, SE ratio 0.0581 | pass |
| `contrast_covariance` | `non_exceptional_fold_evaluated_contrast_0__correlation_one_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the declared contrast with the standard error abs(se_L - se_K), which it would have if the learned and the known value had correlation one | the coverage interval must lie wholly below 0.80 | coverage 0.0448 to 0.0603, SE ratio 0.0336 | pass |
| `contrast_covariance` | `non_exceptional_fold_evaluated_contrast_1__correlation_one_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the fold-evaluated report: the declared contrast with the standard error abs(se_L - se_K), which it would have if the learned and the known value had correlation one | the coverage interval must lie wholly below 0.80 | coverage 0.0731 to 0.0922, SE ratio 0.0515 | pass |
| `contrast_covariance` | `non_exceptional_stacked_contrast_0__correlation_one_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the stacked report: the declared contrast with the standard error abs(se_L - se_K), which it would have if the learned and the known value had correlation one | the coverage interval must lie wholly below 0.80 | coverage 0.0446 to 0.0601, SE ratio 0.0334 | pass |
| `contrast_covariance` | `non_exceptional_stacked_contrast_1__correlation_one_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the stacked report: the declared contrast with the standard error abs(se_L - se_K), which it would have if the learned and the known value had correlation one | the coverage interval must lie wholly below 0.80 | coverage 0.0733 to 0.0924, SE ratio 0.0516 | pass |
| `independent_se_diagnostic` | `misspecified_limit_fold_evaluated_contrast_0__independent_se` | diagnostic | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the declared contrast with the standard error sqrt(se_L^2 + se_K^2), which it would have if the learned and the known value were independent | none. The cell reports the SE ratio and the coverage | coverage 0.9876 to 0.9959, SE ratio 1.4080 | reported |
| `independent_se_diagnostic` | `misspecified_limit_fold_evaluated_contrast_1__independent_se` | diagnostic | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(1) under the fold-evaluated report: the declared contrast with the standard error sqrt(se_L^2 + se_K^2), which it would have if the learned and the known value were independent | none. The cell reports the SE ratio and the coverage | coverage 0.9876 to 0.9959, SE ratio 1.3900 | reported |
| `independent_se_diagnostic` | `misspecified_limit_stacked_contrast_0__independent_se` | diagnostic | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(0) under the stacked report: the declared contrast with the standard error sqrt(se_L^2 + se_K^2), which it would have if the learned and the known value were independent | none. The cell reports the SE ratio and the coverage | coverage 0.9876 to 0.9959, SE ratio 1.4078 | reported |
| `independent_se_diagnostic` | `misspecified_limit_stacked_contrast_1__independent_se` | diagnostic | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction logistic outcome learner has no W1^2 term, the learned-rule value minus the value of Static(1) under the stacked report: the declared contrast with the standard error sqrt(se_L^2 + se_K^2), which it would have if the learned and the known value were independent | none. The cell reports the SE ratio and the coverage | coverage 0.9876 to 0.9959, SE ratio 1.3898 | reported |
| `independent_se_diagnostic` | `non_exceptional_fold_evaluated_contrast_0__independent_se` | diagnostic | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the declared contrast with the standard error sqrt(se_L^2 + se_K^2), which it would have if the learned and the known value were independent | none. The cell reports the SE ratio and the coverage | coverage 0.9881 to 0.9946, SE ratio 1.3405 | reported |
| `independent_se_diagnostic` | `non_exceptional_fold_evaluated_contrast_1__independent_se` | diagnostic | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the fold-evaluated report: the declared contrast with the standard error sqrt(se_L^2 + se_K^2), which it would have if the learned and the known value were independent | none. The cell reports the SE ratio and the coverage | coverage 0.9934 to 0.9979, SE ratio 1.4796 | reported |
| `independent_se_diagnostic` | `non_exceptional_stacked_contrast_0__independent_se` | diagnostic | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the stacked report: the declared contrast with the standard error sqrt(se_L^2 + se_K^2), which it would have if the learned and the known value were independent | none. The cell reports the SE ratio and the coverage | coverage 0.9881 to 0.9946, SE ratio 1.3403 | reported |
| `independent_se_diagnostic` | `non_exceptional_stacked_contrast_1__independent_se` | diagnostic | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the stacked report: the declared contrast with the standard error sqrt(se_L^2 + se_K^2), which it would have if the learned and the known value were independent | none. The cell reports the SE ratio and the coverage | coverage 0.9934 to 0.9979, SE ratio 1.4796 | reported |
| `interval_calibration` | `non_exceptional_fold_evaluated_contrast_0__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9599, SE ratio 0.9705 to 1.0172 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_contrast_0__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the fold-evaluated report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8178 to 0.8438, SE ratio 0.6903 to 0.7247 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_contrast_0__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8658 to 0.8885, SE ratio 0.7763 to 0.8140 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_contrast_1__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9403 to 0.9557, SE ratio 0.9727 to 1.0204 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_contrast_1__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the fold-evaluated report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8044 to 0.8311, SE ratio 0.6741 to 0.7091 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_contrast_1__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the fold-evaluated report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8643 to 0.8872, SE ratio 0.7784 to 0.8162 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_known_0__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(0) under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9483 to 0.9627, SE ratio 0.9808 to 1.0292 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_known_0__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(0) under the fold-evaluated report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8342 to 0.8591, SE ratio 0.6957 to 0.7316 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_known_0__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(0) under the fold-evaluated report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8714 to 0.8937, SE ratio 0.7849 to 0.8234 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_known_1__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(1) under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9424 to 0.9575, SE ratio 0.9756 to 1.0235 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_known_1__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(1) under the fold-evaluated report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8182 to 0.8441, SE ratio 0.6891 to 0.7233 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_known_1__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(1) under the fold-evaluated report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8652 to 0.8880, SE ratio 0.7805 to 0.8183 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_value__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value under the fold-evaluated report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9374 to 0.9532, SE ratio 0.9523 to 1.0004 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_value__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value under the fold-evaluated report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8195 to 0.8453, SE ratio 0.6853 to 0.7206 | pass |
| `interval_calibration` | `non_exceptional_fold_evaluated_value__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value under the fold-evaluated report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8625 to 0.8855, SE ratio 0.7616 to 0.7992 | pass |
| `interval_calibration` | `non_exceptional_stacked_contrast_0__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the stacked report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9599, SE ratio 0.9703 to 1.0175 | pass |
| `interval_calibration` | `non_exceptional_stacked_contrast_0__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the stacked report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8209 to 0.8467, SE ratio 0.6921 to 0.7261 | pass |
| `interval_calibration` | `non_exceptional_stacked_contrast_0__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(0) under the stacked report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8658 to 0.8885, SE ratio 0.7770 to 0.8138 | pass |
| `interval_calibration` | `non_exceptional_stacked_contrast_1__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the stacked report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9403 to 0.9557, SE ratio 0.9719 to 1.0217 | pass |
| `interval_calibration` | `non_exceptional_stacked_contrast_1__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the stacked report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8187 to 0.8446, SE ratio 0.6926 to 0.7269 | pass |
| `interval_calibration` | `non_exceptional_stacked_contrast_1__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value minus the value of Static(1) under the stacked report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8641 to 0.8870, SE ratio 0.7771 to 0.8159 | pass |
| `interval_calibration` | `non_exceptional_stacked_known_0__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(0) under the stacked report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9483 to 0.9627, SE ratio 0.9812 to 1.0287 | pass |
| `interval_calibration` | `non_exceptional_stacked_known_0__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(0) under the stacked report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8266 to 0.8521, SE ratio 0.6961 to 0.7313 | pass |
| `interval_calibration` | `non_exceptional_stacked_known_0__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(0) under the stacked report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8714 to 0.8937, SE ratio 0.7846 to 0.8224 | pass |
| `interval_calibration` | `non_exceptional_stacked_known_1__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(1) under the stacked report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9424 to 0.9575, SE ratio 0.9747 to 1.0239 | pass |
| `interval_calibration` | `non_exceptional_stacked_known_1__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(1) under the stacked report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8141 to 0.8403, SE ratio 0.6892 to 0.7238 | pass |
| `interval_calibration` | `non_exceptional_stacked_known_1__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the value of Static(1) under the stacked report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8652 to 0.8880, SE ratio 0.7806 to 0.8189 | pass |
| `interval_calibration` | `non_exceptional_stacked_value__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value under the stacked report: the declared fit beside Static(1) and Static(0), with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9376 to 0.9534, SE ratio 0.9524 to 0.9995 | pass |
| `interval_calibration` | `non_exceptional_stacked_value__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value under the stacked report: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8132 to 0.8394, SE ratio 0.6713 to 0.7062 | pass |
| `interval_calibration` | `non_exceptional_stacked_value__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome learner is correct, the learned-rule value under the stacked report: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8625 to 0.8855, SE ratio 0.7621 to 0.7995 | pass |
| `targeting_necessity` | `drop_w1_contrast_0__targeted` | positive | the learned-rule law with the blip 0.1 + W1 and RM30's drop_w1 outcome learner, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the declared fit beside Static(1) and Static(0) with drop_w1, a saturated outcome learner in (A, W2) with no W1 term, so targeting does the adjusting | bias interval inside the equivalence margin | bias -0.0023 to 0.0010, margin 0.0055 | pass |
| `targeting_necessity` | `drop_w1_contrast_0__untargeted` | control | the learned-rule law with the blip 0.1 + W1 and RM30's drop_w1 outcome learner, the learned-rule value minus the value of Static(0) under the fold-evaluated report: the same fit's initial plug-in contrast, the difference of the fold-weighted untargeted values, with no fluctuation | bias interval must fall entirely outside the margin | bias 0.0247 to 0.0275, margin 0.0047 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed. The 5 primary tests read standardized
biases of -0.030 to +0.017, coverage of 0.9367 to 0.9475, and SE ratios of 0.970 to 1.013. The 10
calibration positives read SE ratios of 0.975 to 1.004 and coverage of 0.9457 to 0.9559, and
every interval lies inside its band. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 5 of 5 | `performance-tests.csv` |
| gated property cells passed | 60 of 60 | `properties.csv` |
| diagnostic cells, reported with no verdict | 8 | `properties.csv`, `independent_se_diagnostic` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |
| smallest calibration SE-ratio lower endpoint | 0.9523 | `properties.csv`, `non_exceptional_fold_evaluated_value__declared` |
| smallest calibration coverage lower endpoint | 0.9374 | `properties.csv`, `non_exceptional_fold_evaluated_value__declared` |
| smallest coverage-floor lower endpoint | 0.9307 | `properties.csv`, `misspecified_limit_fold_evaluated_contrast_0__declared` |

The support rows were declared before the run: two P11 rows with the qualifier `known_regimes`,
one for each report. The fold-evaluated row needed the five primary tests, every
`interval_calibration`, `conservative_coverage` and `contrast_covariance` verdict of the
fold-evaluated labels, and the `targeting_necessity` verdicts to pass. The stacked row needed the
same verdicts with the stacked labels, and the primary and necessity verdicts too. They did, so
the registered rows `T44.fold_evaluated` and `T44.stacked` cover the plug-in rule beside known
regimes under each report. A row labels every estimate of a fit, so each covers
`ey_learned_rule`, `ey_known_regime` and `ate_learned_rule`.

### The lowest primary cell

The contrast against `Static(0)` reads the lowest primary coverage, 0.9367, and the lowest SE
ratio, 0.970, over 1,200 draws. Both pass the declared primary rules: the lower coverage endpoint
0.9163 clears 0.90, and the SE-ratio interval 0.922 to 1.024 lies inside 0.80 to 1.20. At an SE
ratio of 0.970 a normal interval covers about 0.943, and the reading sits within one Monte Carlo
SE (0.007) of that. The calibration cell of the same parameter, on 5,600 independent draws, reads
an SE ratio of 0.993 and coverage of 0.9529, both inside the calibration bands, and on
`misspecified_limit` its coverage is 0.9422. The primary cell is read as Monte Carlo noise at
1,200 draws, not as a narrow interval.

### Coverage under a misspecified treated arm

On `misspecified_limit` every positive's coverage interval clears 0.92, and the realized SE
ratios lie close to 1. Under the fold-evaluated report they are 0.997 (value), 1.004 (`Static(1)`),
1.003 (`Static(0)`), 0.996 (the contrast against `Static(1)`) and 0.991 (the contrast against
`Static(0)`); the stacked report reads the same to within 0.0002. These cells measure coverage
when the outcome model cannot represent the treated arm. They do not show that the interval is
conservative. The curves omit the treatment model's projection term, and Corollary 2 of van der
Laan and Luedtke (2015, eq. 11) makes that variance conservative for a maximum-likelihood $g$ in
a correct model, which the paper extends to the cross-validated TMLE by analogy only. On this law
the omitted term is negligible: the predicted asymptotic SE ratios are at most 1.0005, so the
study cannot tell that route from exact calibration.

The `Static(0)` cells read an outcome arm the model represents correctly, because the two laws
share $\bar Q_0(0, \cdot)$, so they are not misspecified-outcome evidence. The coverage was
measured with the correct logistic treatment model fitted by maximum likelihood only. The version
of the conservative-variance argument for a contrast is the research session's own derivation,
not a statement in the paper.

### The covariance of the two values

The contrast's SE keeps the covariance of the learned and the known value. Read at correlation
one, the contrasts cover 0.044 to 0.090. Read as if the two values were independent, their SE
ratios are 1.34 to 1.48 and they cover 0.992 to 0.996. The calibration positives of the
contrasts, at SE ratios of 0.993 to 0.996, sit between the two.

### Solver warnings

The harness counts the scikit-learn and `cleverly` convergence warnings of each draw's fits and
truth refits. No primary, calibration or coverage draw warned. One necessity draw of 1,200
(replicate 404) raised two warnings, and its truth check found the fit's rule on all 2,000 rows,
so the draw was kept. The truth check found the fit's rule on every validation row of every draw.
No rule reads the warning counts.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the SD of $\sqrt{n}$ times the estimate at the limit rule over $\sqrt{2000}$. They size the noise
controls only, and no verdict reads them as an efficiency bound. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1200 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 5 | parameter tests against the truth |
| `independent_tests_passed` | 5 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 60 | gated repeated-sampling property cells |
| `property_cells_passed` | 60 | cells whose own and family verdicts pass |
| `property_cells_reported` | 8 | diagnostic cells, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0298 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9367 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9163 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9222 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0695 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-contrast/non_exceptional/ate_learned_rule[learned rule vs always 0]]:coverage` | 0.9367 | primary coverage of the contrast against `Static(0)` |
| `performance[cleverly-learned-rule-contrast/non_exceptional/ate_learned_rule[learned rule vs always 0]]:se_ratio` | 0.9696 | its SE ratio |
| `performance[cleverly-learned-rule-contrast/non_exceptional/ate_learned_rule[learned rule vs always 0]]:coverage_ci_lower` | 0.9163 | its lower coverage endpoint, against 0.90 |
| `performance[cleverly-learned-rule-contrast/non_exceptional/ate_learned_rule[learned rule vs always 0]]:se_ratio_ci_lower` | 0.9222 | its lower SE-ratio endpoint, against 0.80 |
| `performance[cleverly-learned-rule-contrast/non_exceptional/ate_learned_rule[learned rule vs always 0]]:se_ratio_ci_upper` | 1.0238 | its upper SE-ratio endpoint, against 1.20 |
| `summary[cleverly-learned-rule-contrast/non_exceptional/ate_learned_rule[learned rule vs always 0]]:coverage_se` | 0.0070 | its Monte Carlo coverage SE |
| `summary[cleverly-learned-rule-contrast/non_exceptional/ate_learned_rule[learned rule vs always 1]]:truth_min` | 0.0330 | the smallest true contrast against `Static(1)` |
| `summary[cleverly-learned-rule-contrast/non_exceptional/ate_learned_rule[learned rule vs always 0]]:truth_min` | 0.0524 | the smallest true contrast against `Static(0)` |
| `properties[interval_calibration/non_exceptional_fold_evaluated_contrast_0__declared]:se_ratio` | 0.9934 | calibration SE ratio of the same contrast |
| `properties[interval_calibration/non_exceptional_fold_evaluated_contrast_0__declared]:coverage` | 0.9529 | its calibration coverage |
| `properties[interval_calibration/non_exceptional_fold_evaluated_value__declared]:se_ratio` | 0.9754 | the lowest calibration SE ratio |
| `properties[interval_calibration/non_exceptional_fold_evaluated_value__declared]:se_ratio_ci_lower` | 0.9523 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/non_exceptional_fold_evaluated_value__declared]:coverage_ci_lower` | 0.9374 | its lower coverage endpoint, against 0.92 |
| `properties[interval_calibration/non_exceptional_fold_evaluated_value__shrunken_se_control]:se_ratio_ci_upper` | 0.7992 | a shrunken-SE control upper endpoint |
| `properties[interval_calibration/non_exceptional_fold_evaluated_known_0__shrunken_se_control]:se_ratio_ci_upper` | 0.8234 | the highest shrunken-SE control upper endpoint |
| `properties[interval_calibration/non_exceptional_fold_evaluated_known_0__noise_control]:se_ratio_ci_upper` | 0.7316 | the highest noise-control upper endpoint |
| `properties[conservative_coverage/misspecified_limit_fold_evaluated_value__declared]:se_ratio` | 0.9968 | realized SE ratio, value, `misspecified_limit` |
| `properties[conservative_coverage/misspecified_limit_fold_evaluated_known_1__declared]:se_ratio` | 1.0040 | realized SE ratio, `Static(1)` |
| `properties[conservative_coverage/misspecified_limit_fold_evaluated_known_0__declared]:se_ratio` | 1.0028 | realized SE ratio, `Static(0)` |
| `properties[conservative_coverage/misspecified_limit_fold_evaluated_contrast_1__declared]:se_ratio` | 0.9956 | realized SE ratio, the contrast against `Static(1)` |
| `properties[conservative_coverage/misspecified_limit_fold_evaluated_contrast_0__declared]:se_ratio` | 0.9913 | realized SE ratio, the contrast against `Static(0)` |
| `properties[conservative_coverage/misspecified_limit_fold_evaluated_contrast_0__declared]:coverage` | 0.9422 | the lowest coverage on `misspecified_limit` |
| `properties[conservative_coverage/misspecified_limit_fold_evaluated_contrast_0__declared]:coverage_ci_lower` | 0.9307 | its lower endpoint, against 0.92 |
| `properties[conservative_coverage/misspecified_limit_fold_evaluated_known_1__shrunken_coverage_control]:coverage_ci_upper` | 0.7853 | the highest coverage-control upper endpoint, against 0.92 |
| `properties[contrast_covariance/misspecified_limit_fold_evaluated_contrast_0__correlation_one_control]:coverage` | 0.0441 | the lowest correlation-one coverage |
| `properties[contrast_covariance/misspecified_limit_fold_evaluated_contrast_1__correlation_one_control]:coverage` | 0.0900 | the highest correlation-one coverage |
| `properties[contrast_covariance/misspecified_limit_fold_evaluated_contrast_1__correlation_one_control]:coverage_ci_upper` | 0.1038 | its upper endpoint, against 0.80 |
| `properties[independent_se_diagnostic/non_exceptional_fold_evaluated_contrast_0__independent_se]:se_ratio` | 1.3405 | the lowest independent-SE ratio |
| `properties[independent_se_diagnostic/non_exceptional_fold_evaluated_contrast_1__independent_se]:se_ratio` | 1.4796 | the highest independent-SE ratio |
| `properties[independent_se_diagnostic/non_exceptional_fold_evaluated_contrast_0__independent_se]:coverage` | 0.9918 | the lowest independent-SE coverage |
| `properties[targeting_necessity/drop_w1_contrast_0__targeted]:standardized_bias` | -0.0292 | necessity positive standardized bias |
| `properties[targeting_necessity/drop_w1_contrast_0__targeted]:coverage` | 0.9258 | its coverage, which no verdict reads |
| `properties[targeting_necessity/drop_w1_contrast_0__untargeted]:standardized_bias` | 1.4029 | necessity control standardized bias |
| `properties[targeting_necessity/drop_w1_contrast_0__targeted]:targeting_displacement` | 1.2103 | displacement between the arms, in positive-arm SDs, against 0.5 |
| `bound:non_exceptional_fold_evaluated_value_standard_error` | 0.0149 | the limit-rule SD over the root of n, the learned-rule value, fold-evaluated report |
| `bound:non_exceptional_fold_evaluated_known_1_standard_error` | 0.0156 | the limit-rule SD over the root of n, `Static(1)`, fold-evaluated report |
| `bound:non_exceptional_fold_evaluated_known_0_standard_error` | 0.0154 | the limit-rule SD over the root of n, `Static(0)`, fold-evaluated report |
| `bound:non_exceptional_fold_evaluated_contrast_1_standard_error` | 0.0146 | the limit-rule SD over the root of n, the contrast against `Static(1)`, fold-evaluated report |
| `bound:non_exceptional_fold_evaluated_contrast_0_standard_error` | 0.0159 | the limit-rule SD over the root of n, the contrast against `Static(0)`, fold-evaluated report |
| `bound:non_exceptional_stacked_value_standard_error` | 0.0149 | the limit-rule SD over the root of n, the learned-rule value, stacked report |
| `bound:non_exceptional_stacked_known_1_standard_error` | 0.0156 | the limit-rule SD over the root of n, `Static(1)`, stacked report |
| `bound:non_exceptional_stacked_known_0_standard_error` | 0.0154 | the limit-rule SD over the root of n, `Static(0)`, stacked report |
| `bound:non_exceptional_stacked_contrast_1_standard_error` | 0.0146 | the limit-rule SD over the root of n, the contrast against `Static(1)`, stacked report |
| `bound:non_exceptional_stacked_contrast_0_standard_error` | 0.0159 | the limit-rule SD over the root of n, the contrast against `Static(0)`, stacked report |
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
| There is no cross-implementation evidence | No maintained package fits a cross-validated TMLE of a learned rule beside known regimes. The rows rest on accuracy against the truth and on the theory properties |
| One size, one fold count and one learner per role | The study fits n = 2,000 with ten equal folds, pooled targeting, `repeats=1`, the RM30 interaction logistic outcome model, a correct logistic treatment model fitted by maximum likelihood and the plug-in rule, beside `Static(1)` and `Static(0)`, on binary outcomes and treatments, iid rows and no weights. The key holds neither n, the fold count nor the learner, so another n, V or learner reads supported and was not measured |
| Compositions the key covers that were not measured | Fold counts other than 10, `reference=` naming the learned rule (sign-flipped contrasts), known dynamic `Rule` regimes and more than two known regimes read supported. Each rests on the unit witnesses W-c1 and W-c2 and the linearity argument, as the repeat count of `T38.split_mean` does |
| Compositions outside the rows | Strata, weights, clusters, fold targeting, a rule learner, several splits, the in-sample fit and supplied treatment probabilities (`treatment_probabilities=`, which keys `known_mechanism`) beside known regimes match no row and read experimental |
| Coverage, not conservativeness, under a misspecified treated arm | On `misspecified_limit` the cells claim coverage at or above 0.92, with realized SE ratios of 0.991 to 1.004 under either report, and no SE calibration. The omitted projection term is negligible there, so the study does not test the conservative-variance route of Corollary 2, which the paper extends to the cross-validated TMLE by analogy only. The `Static(0)` cells read a correctly specified arm |
| One treatment learner under misspecification | The misspecified-outcome coverage was measured only with a correct parametric $g$ fitted by maximum likelihood. A known $g$ and a super-learner $g$ were not measured; a cross-fitted key holds no learner family, so a fit with a flexible $g$ reads supported |
| The stacked report at equal folds only | The stacked point equals the fold-evaluated point at equal folds without weights, as in every draw here. Without weights or clusters the folds of a covered fit differ by a few rows at most, and only the variance rule differs |
| A narrow necessity scope | The necessity pair reads the contrast against `Static(0)`. Against `Static(1)` the `drop_w1` learner's limit rule is `Static(1)`, so that contrast was exactly zero in 62% of pilot draws and carries no verdict |
| Pointwise intervals | A contrast fit skips its default simultaneous band, and `simultaneous_bands()` refuses on it |
| One necessity draw with solver warnings | One `drop_w1` draw of 1,200 raised two convergence warnings and was kept. No other draw warned |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_contrast/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_contrast/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_contrast/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_contrast/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_contrast/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_contrast/properties.csv)
carry every published row.
