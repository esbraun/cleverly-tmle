# Compositions of the value of the optimal rule

This study validates the interval of the value of the optimal rule under the five compositions
that X11 (f3) admits: fixed observation weights, baseline strata, known regimes beside it, the
in-sample fit and an intermediate held at each level. The estimator is
`TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"], n_folds=10, cv_evaluation=True,
targeting_scheme="pooled", g_bounds="auto")` with the known treatment mechanism passed as
`treatment_probabilities=`, and each composition adds its own setting. Under every composition
the point, variance and interval are the learned-rule fit's with the same setting, bit for bit;
only the target differs. [The value of the optimal rule](../point-treatment-tmle.md#learned-rules)
gives the construction, the conditions C3f, C7 and C8 and where each composition reads them.

Each label's target is a fixed parameter at the setting's optimal rule. With
$d_0(w) = \mathbb 1\{\bar Q_0(1, w) > \bar Q_0(0, w)\}$, a tie going to arm 0:

| label | estimand | target |
| --- | --- | --- |
| `weighted` | `ey_optimal_rule[learned rule]` under `weights="w"`, $w = 1 + AY$ | $\Psi^w_{d^w_0}(P_w) = E_{P_w}\bar Q_w(d^w_0(W), W)$ on the tilt $dP_w = w\,dP_0/E_0 w$, with $\bar Q_w(a, W) = E_0[wY \mid a, W]/E_0[w \mid a, W]$ and $d^w_0 = \arg\max_a \bar Q_w(a, \cdot) = \mathbb 1\{W_1 > -0.1 - \log 2\}$ |
| `strata_all` | `ey_optimal_rule[learned rule]` under `strata=["W2"]` | $\Psi_{d_0} = E_0\bar Q_0(d_0(W), W)$, the share mixture of the stratum values |
| `strata_s0`, `strata_s1` | `ey_optimal_rule[learned rule][W2=0.0]`, `[W2=1.0]` | $E_0[\bar Q_0(d_0(W), W) \mid W_2 = s]$ |
| `contrast_1`, `contrast_0` | `ate_optimal_rule[learned rule vs always 1]`, `[... vs always 0]` | $\Psi_{d_0} - \Psi_a$, with $\Psi_a = E_0\bar Q_0(a, W)$ |
| `in_sample` | `ey_optimal_rule[learned rule]` under `cross_fit=False` | $\Psi_{d_0}$ |
| `intermediate_z0`, `intermediate_z1` | `ey_optimal_rule[learned rule][Z=0]`, `[Z=1]` | $\Psi^z_{d^z_0} = E_0\bar Q_0(d^z_0(W), z, W)$ with $d^z_0(w) = \arg\max_a \bar Q_0(a, z, w)$ |

Every verdict reads the error against these fixed values. The fit's own fold rules have a
data-adaptive value $\tilde\psi_{0n}$ under the same functional, and the regret
$\Psi_{\rm opt} - \tilde\psi_{0n}$ is never negative. The error splits into the learned-rule
claim with that setting, $\hat\psi - \tilde\psi_{0n}$, and minus the regret, which Lemma 1 of van
der Laan and Luedtke (2015) makes $o_P(n^{-1/2})$ under the margin and rate conditions. Each cell
publishes the mean regret and its readings against $\tilde\psi_{0n}$ beside the verdicts, and no
verdict reads them. The in-sample label rests on Theorem 5 of that paper, under the Donsker
condition C5a.

**No canonical implementation is compared.** No maintained package fits a cross-validated TMLE
of the value of the optimal rule under these compositions. A zero-row equivalence artifact
records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | RM30's `non_exceptional` law for four settings: $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a(0.1 + W_1)$. The `intermediate` setting uses the law of the [intermediate study](learned-rule-intermediate.md): a binary $Z$ after $A$, with level blips $0.1 + W_1 - 0.6z$ |
| fit | `TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"], n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")` with `repeats=1` and pointwise intervals |
| settings | `weighted`: `weights="w"` with $w = 1 + AY$. `strata`: `strata=["W2"]`. `known_regimes`: `interventions=[Static(1), Static(0)]`. `in_sample`: `cross_fit=False, cv_evaluation=False`, the outcome learner marked parametric. `intermediate`: `intermediate="Z"`, both levels |
| known $g$ | every cell passes `treatment_probabilities=`: RM30's $g_0$ in four settings, and in `weighted` the weighted population's own mechanism $g_w(1 \mid W) = g_0(1 \mid W)(1 + \bar Q_0(1, W))/(1 + g_0(1 \mid W)\bar Q_0(1, W))$, which is $g_0 E[w \mid A, W]/E[w \mid W]$ |
| outcome learner | a logistic regression (`C=1e6`, `tol=1e-10`) on every pairwise product of $(A, W_1, W_2)$, a correct model of the law; on the intermediate law the same on $(A, W_1, W_2, Z)$, with a logistic intermediate learner on $(A, W_1, W_2)$ |
| support keys | family `P12` with the qualifier `known_mechanism` and the axis that names the composition; `tests/unit/test_learned_rule_optimal_compositions_design.py` builds each declared row and requires it to cover every fitted key of its setting with equal options and qualifiers |
| truths | adaptive quadrature over $W_1$ for each $W_2$, split at the rule's threshold, one number per label; the design test recomputes each with a second quadrature written independently |
| realized rule value | per replication: refit each fold's outcome learner on the fit's training rows with the package's `fit_on_rows`, require the fit's rule on every validation row (the harness stops a draw that differs), and integrate each label's value at each fold rule by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$; $\tilde\psi_{0n}$ is the mean over the folds |
| primary budget | 4,400 replications of each setting at n = 2,000, nine parameters |
| primary verdicts | the framework defaults on the error against the fixed truth: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20277921` and `resampling_seed=20283472` |
| policy | `gated` |

| label | truth | SD of $\sqrt n\,\hat\psi$ at $d_0$ | role |
| --- | --- | --- | --- |
| `weighted` | 0.690108 | 0.604640 | primary, `interval_calibration` and `weighted_target_control` |
| `strata_all` | 0.577115 | 0.664444 | primary and `interval_calibration` |
| `strata_s0` | 0.610609 | 0.925806 | primary and `interval_calibration`, with the unnormalised-curve control |
| `strata_s1` | 0.543621 | 0.950968 | primary and `interval_calibration`, with the unnormalised-curve control |
| `contrast_1` | 0.045499 | 0.650760 | primary and `interval_calibration` |
| `contrast_0` | 0.064936 | 0.708934 | primary and `interval_calibration` |
| `in_sample` | 0.577115 | 0.664444 | primary and `interval_calibration` |
| `intermediate_z0` | 0.577115 | 1.120875 | primary and `interval_calibration` |
| `intermediate_z1` | 0.597071 | 0.882349 | primary and `interval_calibration` |

The weight $1 + AY$ moves the optimal rule from $\mathbb 1\{W_1 > -0.1\}$ to
$\mathbb 1\{W_1 > -0.1 - \log 2\}$, so the weighted target lies 0.0277 above the weighted value of
the unweighted rule, $\Psi^w_{d_0}(P_w) = 0.662385$, and 0.1130 above $\Psi_{d_0}(P_0)$. The SD
column sizes the noise controls only, at one unit, and no verdict reads it as an efficiency bound.
The laws, the learners, the budgets, the seeds, the gates, the controls and the support rows were
fixed before any run, in the X11 plan, its review and the PR 2 implementation notes.

`tests/unit/test_learned_rule_optimal_compositions_design.py` pins each truth, the sizing, the
controls' gaps and power, and the declared support rows.
`tests/unit/test_learned_rule_optimal_compositions_harness.py` checks the draws, the fits, the
known-$g$ key, the regret, the rule refit and the property path.
`tests/unit/test_learned_rule_optimal_compositions.py` holds the unit witnesses of the
construction, among them W-f3-id, the bit-for-bit identity with the learned-rule fit.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, the outcome learner marked parametric, in-sample fit | `ey_optimal_rule[learned rule]` | value of the optimal rule of the setting, a fixed parameter: under weights the optimal value of the weighted population, otherwise E Qbar0(d0(W), W) | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | 0.000682 to 0.0018 | 0.9470 | 0.9996 | pass |
| the learned-rule law with a binary intermediate Z after A and the level blips 0.1 + W1 - 0.6 z, correct interaction logistic outcome and intermediate learners and the known treatment mechanism, intermediate held at each level, fold-evaluated report | `ey_optimal_rule[learned rule][Z=0]` | value of the optimal rule with the intermediate held at 0, E Qbar0(d0^0(W), 0, W) with d0^0(w) = argmax_a Qbar0(a, 0, w) | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | -0.0040 to -0.0020 | 0.9441 | 0.9758 | pass |
| the learned-rule law with a binary intermediate Z after A and the level blips 0.1 + W1 - 0.6 z, correct interaction logistic outcome and intermediate learners and the known treatment mechanism, intermediate held at each level, fold-evaluated report | `ey_optimal_rule[learned rule][Z=1]` | value of the optimal rule with the intermediate held at 1, E Qbar0(d0^1(W), 1, W) with d0^1(w) = argmax_a Qbar0(a, 1, w) | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | -0.0032 to -0.0016 | 0.9511 | 1.0068 | pass |
| the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, beside Static(1) and Static(0), fold-evaluated report | `ate_optimal_rule[learned rule vs always 0]` | value of the optimal rule minus the mean outcome when no one is treated | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | -0.0019 to -0.000616 | 0.9430 | 0.9817 | pass |
| the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, beside Static(1) and Static(0), fold-evaluated report | `ate_optimal_rule[learned rule vs always 1]` | value of the optimal rule minus the mean outcome when everyone is treated | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | -0.0018 to -0.000669 | 0.9370 | 0.9804 | pass |
| the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, strata of W2, fold-evaluated report | `ey_optimal_rule[learned rule]` | value of the optimal rule of the setting, a fixed parameter: under weights the optimal value of the weighted population, otherwise E Qbar0(d0(W), W) | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | -0.0022 to -0.0010 | 0.9414 | 0.9799 | pass |
| the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, strata of W2, fold-evaluated report | `ey_optimal_rule[learned rule][W2=0.0]` | value of the optimal rule within the stratum W2 = 0, E[Qbar0(d0(W), W) given W2 = 0] | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | -0.0025 to -0.000834 | 0.9423 | 0.9868 | pass |
| the learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome learner and the known treatment mechanism, strata of W2, fold-evaluated report | `ey_optimal_rule[learned rule][W2=1.0]` | value of the optimal rule within the stratum W2 = 1, E[Qbar0(d0(W), W) given W2 = 1] | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | -0.0024 to -0.000672 | 0.9384 | 0.9725 | pass |
| the learned-rule law with the blip 0.1 + W1 and the observation weight 1 + A Y, a correct interaction logistic outcome learner and the weighted population's own treatment mechanism, fold-evaluated report | `ey_optimal_rule[learned rule]` | value of the optimal rule of the setting, a fixed parameter: under weights the optimal value of the weighted population, otherwise E Qbar0(d0(W), W) | `cleverly` cross-validated TMLE of the value of the optimal rule under a composition: the learned-rule fit with the same setting, the plug-in rule, one pooled fluctuation and the known treatment mechanism, read against the fixed optimal value of the setting | -0.0024 to -0.0014 | 0.9439 | 0.9839 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | at each of the nine labels at n = 2,000, does the SE ratio of the declared fit lie inside 0.93 to 1.07, and its coverage of the fixed truth inside 0.92 to 0.98, with 10,000 draws per setting, each draw serving all of its setting's labels | a shrunken-SE control multiplies each SE by 0.80, and a noise control adds noise of one SD unit over $\sqrt n$ to each estimate. At each stratum label, the unnormalised-curve control reads the stratum curve without its fold share $1/P_{n,v}(W_2 = s)$, so its SE is about half the reported one. Each control's SE-ratio interval must fall below the band |
| `weighted_target_control` | on 1,200 draws of the weighted setting, does the interval miss $\Psi_{d_0}(P_0) = 0.577115$ (`target_gap_control`, 8.36 SEs below the target), and $\Psi^w_{d_0}(P_w) = 0.662385$ (`unweighted_rule_control`, 2.05 SEs below) | each control passes when its exact 99% coverage interval lies wholly below 0.92. The first would cover if the weights were dropped, the second if the rule were learned without them |

The calibration count is sized on the joint family, the arm-pooled SE-ratio family times the
nine coverage legs. On a 2,000-draw pilot per setting it first reaches a predictive pass
probability of 0.95 at 9,600, and a stress like the optimal-value study's leaves it at 0.949, so
the count is 10,000 (0.957, stressed 0.954). The primary count, 4,400, is the smallest multiple
of 400 at which the nine parameters pass together with probability at least 0.95; the bias legs
bind, because the regret biases each estimate by up to 0.160 SDs on the pilot.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `contrast_0__declared` | positive | the optimal value minus the value of Static(0), 0.064936, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9342 to 0.9465, SE ratio 0.9698 to 1.0059 | pass |
| `interval_calibration` | `contrast_0__noise_control` | control | the optimal value minus the value of Static(0), 0.064936, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8185 to 0.8380, SE ratio 0.6876 to 0.7136 | pass |
| `interval_calibration` | `contrast_0__shrunken_se_control` | control | the optimal value minus the value of Static(0), 0.064936, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8660 to 0.8832, SE ratio 0.7758 to 0.8049 | pass |
| `interval_calibration` | `contrast_1__declared` | positive | the optimal value minus the value of Static(1), 0.045499, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9308 to 0.9434, SE ratio 0.9600 to 0.9954 | pass |
| `interval_calibration` | `contrast_1__noise_control` | control | the optimal value minus the value of Static(1), 0.045499, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8182 to 0.8377, SE ratio 0.6857 to 0.7116 | pass |
| `interval_calibration` | `contrast_1__shrunken_se_control` | control | the optimal value minus the value of Static(1), 0.045499, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8599 to 0.8774, SE ratio 0.7681 to 0.7965 | pass |
| `interval_calibration` | `in_sample__declared` | positive | the optimal value from the in-sample fit, 0.577115, at the learned-rule law with the blip 0.1 + W1, a parametric outcome learner and the known treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9551, SE ratio 0.9821 to 1.0184 | pass |
| `interval_calibration` | `in_sample__noise_control` | control | the optimal value from the in-sample fit, 0.577115, at the learned-rule law with the blip 0.1 + W1, a parametric outcome learner and the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8236 to 0.8429, SE ratio 0.6948 to 0.7212 | pass |
| `interval_calibration` | `in_sample__shrunken_se_control` | control | the optimal value from the in-sample fit, 0.577115, at the learned-rule law with the blip 0.1 + W1, a parametric outcome learner and the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8727 to 0.8895, SE ratio 0.7857 to 0.8150 | pass |
| `interval_calibration` | `intermediate_z0__declared` | positive | the optimal value with the intermediate held at 0, 0.577115, at the learned-rule intermediate law with the known treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9361 to 0.9482, SE ratio 0.9664 to 1.0031 | pass |
| `interval_calibration` | `intermediate_z0__noise_control` | control | the optimal value with the intermediate held at 0, 0.577115, at the learned-rule intermediate law with the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8260 to 0.8452, SE ratio 0.6991 to 0.7252 | pass |
| `interval_calibration` | `intermediate_z0__shrunken_se_control` | control | the optimal value with the intermediate held at 0, 0.577115, at the learned-rule intermediate law with the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8644 to 0.8816, SE ratio 0.7733 to 0.8019 | pass |
| `interval_calibration` | `intermediate_z1__declared` | positive | the optimal value with the intermediate held at 1, 0.597071, at the learned-rule intermediate law with the known treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9397 to 0.9515, SE ratio 0.9831 to 1.0205 | pass |
| `interval_calibration` | `intermediate_z1__noise_control` | control | the optimal value with the intermediate held at 1, 0.597071, at the learned-rule intermediate law with the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8252 to 0.8444, SE ratio 0.7014 to 0.7276 | pass |
| `interval_calibration` | `intermediate_z1__shrunken_se_control` | control | the optimal value with the intermediate held at 1, 0.597071, at the learned-rule intermediate law with the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8727 to 0.8895, SE ratio 0.7870 to 0.8162 | pass |
| `interval_calibration` | `strata_all__declared` | positive | the marginal optimal value of the fit with strata of W2, 0.577115, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9396 to 0.9514, SE ratio 0.9709 to 1.0077 | pass |
| `interval_calibration` | `strata_all__noise_control` | control | the marginal optimal value of the fit with strata of W2, 0.577115, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8164 to 0.8360, SE ratio 0.6777 to 0.7027 | pass |
| `interval_calibration` | `strata_all__shrunken_se_control` | control | the marginal optimal value of the fit with strata of W2, 0.577115, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8675 to 0.8846, SE ratio 0.7767 to 0.8059 | pass |
| `interval_calibration` | `strata_s0__declared` | positive | the optimal value within W2 = 0, 0.610609, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9416 to 0.9532, SE ratio 0.9780 to 1.0151 | pass |
| `interval_calibration` | `strata_s0__noise_control` | control | the optimal value within W2 = 0, 0.610609, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8239 to 0.8432, SE ratio 0.6941 to 0.7204 | pass |
| `interval_calibration` | `strata_s0__shrunken_se_control` | control | the optimal value within W2 = 0, 0.610609, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8729 to 0.8897, SE ratio 0.7827 to 0.8120 | pass |
| `interval_calibration` | `strata_s0__unnormalised_curve_control` | control | the optimal value within W2 = 0, 0.610609, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.6604 to 0.6847, SE ratio 0.4878 to 0.5059 | pass |
| `interval_calibration` | `strata_s1__declared` | positive | the optimal value within W2 = 1, 0.543621, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9423 to 0.9538, SE ratio 0.9753 to 1.0116 | pass |
| `interval_calibration` | `strata_s1__noise_control` | control | the optimal value within W2 = 1, 0.543621, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8195 to 0.8390, SE ratio 0.6930 to 0.7186 | pass |
| `interval_calibration` | `strata_s1__shrunken_se_control` | control | the optimal value within W2 = 1, 0.543621, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8693 to 0.8863, SE ratio 0.7803 to 0.8086 | pass |
| `interval_calibration` | `strata_s1__unnormalised_curve_control` | control | the optimal value within W2 = 1, 0.543621, at the learned-rule law with the blip 0.1 + W1 and the known treatment mechanism, n = 2,000: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.6569 to 0.6812, SE ratio 0.4863 to 0.5044 | pass |
| `interval_calibration` | `weighted__declared` | positive | the optimal value of the population the weight 1 + A Y describes, 0.690108, at the learned-rule law with the blip 0.1 + W1, with the weighted population's own treatment mechanism, n = 2,000: the declared fit of the setting with the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9333 to 0.9457, SE ratio 0.9547 to 0.9903 | pass |
| `interval_calibration` | `weighted__noise_control` | control | the optimal value of the population the weight 1 + A Y describes, 0.690108, at the learned-rule law with the blip 0.1 + W1, with the weighted population's own treatment mechanism, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8171 to 0.8366, SE ratio 0.6868 to 0.7118 | pass |
| `interval_calibration` | `weighted__shrunken_se_control` | control | the optimal value of the population the weight 1 + A Y describes, 0.690108, at the learned-rule law with the blip 0.1 + W1, with the weighted population's own treatment mechanism, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8586 to 0.8762, SE ratio 0.7636 to 0.7920 | pass |
| `weighted_target_control` | `weighted__target_gap_control` | control | the optimal value of the population the weight 1 + A Y describes, 0.690108, at the learned-rule law with the blip 0.1 + W1, with the weighted population's own treatment mechanism, n = 2,000: the weighted fit's own estimate and SE, read against the unweighted optimal value 0.577115, 0.1130 below the target, which a fit that dropped the weights would cover | the coverage interval must lie wholly below 0.92 | coverage 0 to 0.0044, SE ratio 0.9797, mean regret 0.0014, coverage of the realized rule value 0.9425 | pass |
| `weighted_target_control` | `weighted__unweighted_rule_control` | control | the optimal value of the population the weight 1 + A Y describes, 0.690108, at the learned-rule law with the blip 0.1 + W1, with the weighted population's own treatment mechanism, n = 2,000: the weighted fit's own estimate and SE, read against the weighted value of the unweighted optimal rule 0.662385, 0.0277 below the target, which a fit that learned its rule without the weights would cover | the coverage interval must lie wholly below 0.92 | coverage 0.4658 to 0.5409, SE ratio 0.9797, mean regret 0.0014, coverage of the realized rule value 0.9425 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. The nine primary
parameters read standardized biases of -0.139 to +0.085, coverage of 0.937 to 0.951 and SE
ratios of 0.972 to 1.007 over 4,400 draws per setting. The nine calibration positives read SE
ratios of 0.972 to 1.002 and coverage of 0.937 to 0.950 over 10,000 draws per setting, every
interval inside its band. Every control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 9 of 9 | `performance-tests.csv` |
| gated property cells passed | 31 of 31 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |
| lowest primary coverage | 0.9370 | `performance-tests.csv`, `contrast_1` |
| largest primary standardized bias | 0.1386 | `performance-tests.csv`, `weighted` |
| lowest calibration SE ratio | 0.9723 | `properties.csv`, `weighted__declared` |
| weighted target-gap coverage interval | 0 to 0.0044 | `properties.csv`, `weighted__target_gap_control` |
| weighted unweighted-rule coverage interval | 0.4658 to 0.5409 | `properties.csv`, `weighted__unweighted_rule_control` |

The support rows were declared before the run: one P12 row per composition with the qualifier
`known_mechanism`. Each needed its setting's primary tests, every calibration verdict of its
labels and, for the weighted row, both target controls to pass. They did, so the registered rows
`T55.weighted`, `T55.strata`, `T55.known_regimes`, `T55.in_sample` and `T55.intermediate`
cover these compositions on rows with the known treatment mechanism. An estimated $g$, estimated
weights, a rule learner, clusters, several splits, fold targeting, the stacked report and more
than two arms stay experimental.

### The regret

The mean regret is 0.0012 to 0.0028 across the labels, largest at the two levels of the
intermediate. It is the main part of the bias the verdicts read: in the calibration cells the
standardized bias against the fixed truth runs from -0.153 (`weighted`) to -0.063 (`strata_s0`)
outside the in-sample label, and against $\tilde\psi_{0n}$ from -0.052 to +0.013. The primary
bias intervals exclude zero for the same reason. The study measures one n, so it does not show
the rate.

### The weighted target controls

Read against $\Psi_{d_0}(P_0)$ the weighted interval covers 0.000 of 1,200 draws, and against
$\Psi^w_{d_0}(P_w)$ 0.503. The second control's gap, 2.05 SEs, is the one that shows the fit
learns its rule under the weights, not only that it weights the value.

The weighted calibration positive reads an SE ratio of 0.972 with a 99% interval of 0.955 to
0.990, which excludes 1, and coverage of 0.940. Against $\tilde\psi_{0n}$ the SE ratio is 0.973,
so the shortfall is not the regret's spread: the reported SE is about 3% small at n = 2,000 on
this law. The contrast against `Static(1)` shows the same pattern at about 2% (0.977, interval
0.960 to 0.995). Both lie inside the declared band, and no other calibration positive's interval excludes 1.

### The in-sample lean

The in-sample label is the only one with a positive bias: standardized bias +0.096 against
$\Psi_{d_0}$ in the calibration cell, with an SE ratio of 1.000 and coverage of 0.950. Against
the value of the one rule it learned, $\Psi_{d_n}$, the standardized bias is +0.179. That is the
in-sample fit's optimism, which the regret, 0.0012 on average, partly offsets. The declaration
publishes this reading with no verdict.

### The stratum cells

Each stratum positive passes with an SE ratio of 0.996 and 0.993, and each unnormalised-curve
control reads 0.50, as its construction predicts. That control resolves a gross normalisation
error only. The fold-local share itself rests on W-f3-id and on the witnesses of
`tests/unit/test_learned_rule_strata.py`.

The calibration count pools the SE-ratio legs of labels whose SE constructions differ. That is
optimistic for the label furthest from 1 on the pilot, `strata_s1` at 1.028: with a true ratio
one pilot SD higher it would pass its SE-ratio leg with probability about 0.79 at 10,000 draws.
The design did not size for that case. The declared run read 0.993 there, so the caveat did not
bind.

### What no cell sees

The weighted row covers fixed weights with the weighted population's mechanism supplied.
Supplying the design mechanism $g_0$ beside weights that depend on the treatment or the outcome
is the Fix-16 defect: the support key cannot see it and no cell measures it.

The intermediate cells cannot tell a fit that ignores $Z$. The total-effect optimum, 0.588, is
0.44 and 0.46 SEs from the two level truths, and the level-blind rule 0.27 and 0.22 SEs, so
neither would move a verdict. That case rests on W-f3-id, which the
[intermediate study](learned-rule-intermediate.md) measures level by level, on witness W-f3z,
in which the level optimum beats the total-effect rule by a value gap, and on the harness, which
stops a draw whose level rules differ from the refit's.

### Stack drift

The run recorded python 3.13.16, numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, scikit-learn 1.9.1
and joblib 1.6.0, against the declared 3.13.7, 2.4.6, 1.18.0, 3.0.5, 1.9.0 and 1.5.3. Each is a
minor or patch release of the declared library, and the run log records every one. No primary
or property draw raised a solver warning, and the rule refit found the fit's rule on every
validation row of every draw.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the SD column above over $\sqrt{2000}$. They size the noise controls only, and no verdict reads
them as efficiency bounds. Everything else is measured from the committed results and checked at
the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4400 | primary replications per setting |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 9 | parameter tests against the truth |
| `independent_tests_passed` | 9 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 31 | gated repeated-sampling property cells |
| `property_cells_passed` | 31 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.1386 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9370 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9270 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9474 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0346 | highest bootstrap primary SE-ratio endpoint |
| `properties[interval_calibration/weighted__declared]:se_ratio` | 0.9723 | weighted calibration SE ratio |
| `properties[interval_calibration/weighted__declared]:se_ratio_ci_lower` | 0.9547 | its interval, lower end |
| `properties[interval_calibration/weighted__declared]:se_ratio_ci_upper` | 0.9903 | its interval, upper end |
| `properties[interval_calibration/weighted__declared]:coverage` | 0.9397 | weighted calibration coverage |
| `properties[interval_calibration/weighted__declared]:standardized_bias` | -0.1534 | its standardized bias against the fixed truth |
| `properties[interval_calibration/weighted__declared]:rule_value_standardized_bias` | -0.0517 | its standardized bias against the realized rule value |
| `properties[interval_calibration/weighted__declared]:rule_value_se_ratio` | 0.9729 | its SE ratio against the realized rule value |
| `properties[interval_calibration/strata_s0__declared]:se_ratio` | 0.9963 | stratum W2 = 0 calibration SE ratio |
| `properties[interval_calibration/strata_s0__declared]:standardized_bias` | -0.0629 | its standardized bias |
| `properties[interval_calibration/strata_s1__declared]:se_ratio` | 0.9928 | stratum W2 = 1 calibration SE ratio |
| `properties[interval_calibration/strata_s0__unnormalised_curve_control]:se_ratio` | 0.4966 | unnormalised-curve control, W2 = 0 |
| `properties[interval_calibration/strata_s1__unnormalised_curve_control]:se_ratio` | 0.4954 | unnormalised-curve control, W2 = 1 |
| `properties[interval_calibration/contrast_1__declared]:se_ratio` | 0.9773 | contrast against Static(1), calibration SE ratio |
| `properties[interval_calibration/contrast_1__declared]:se_ratio_ci_lower` | 0.9600 | its interval, lower end |
| `properties[interval_calibration/contrast_1__declared]:se_ratio_ci_upper` | 0.9954 | its interval, upper end |
| `properties[interval_calibration/contrast_1__declared]:coverage` | 0.9373 | its coverage |
| `properties[interval_calibration/contrast_1__declared]:rule_value_standardized_bias` | 0.0128 | its standardized bias against the realized rule value |
| `properties[interval_calibration/in_sample__declared]:se_ratio` | 1.0001 | in-sample calibration SE ratio |
| `properties[interval_calibration/in_sample__declared]:coverage` | 0.9496 | its coverage |
| `properties[interval_calibration/in_sample__declared]:standardized_bias` | 0.0959 | its standardized bias against the fixed truth |
| `properties[interval_calibration/in_sample__declared]:rule_value_standardized_bias` | 0.1787 | the in-sample lean, against the value of its own rule |
| `properties[interval_calibration/in_sample__declared]:mean_regret` | 0.0012 | its mean regret |
| `properties[interval_calibration/intermediate_z0__declared]:mean_regret` | 0.0028 | mean regret at z = 0 |
| `properties[weighted_target_control/weighted__target_gap_control]:coverage` | 0 | target-gap control coverage of the unweighted optimal value |
| `properties[weighted_target_control/weighted__target_gap_control]:coverage_ci_upper` | 0.0044 | its upper endpoint, against 0.92 |
| `properties[weighted_target_control/weighted__unweighted_rule_control]:coverage` | 0.5033 | unweighted-rule control coverage |
| `properties[weighted_target_control/weighted__unweighted_rule_control]:coverage_ci_lower` | 0.4658 | its interval, lower end |
| `properties[weighted_target_control/weighted__unweighted_rule_control]:coverage_ci_upper` | 0.5409 | its upper endpoint, against 0.92 |
| `bound:weighted_standard_error` | 0.0135 | the weighted label's SD over the root of n |
| `bound:strata_all_standard_error` | 0.0149 | the marginal stratum label's SD over the root of n |
| `bound:strata_s0_standard_error` | 0.0207 | the stratum W2 = 0 label's SD over the root of n |
| `bound:strata_s1_standard_error` | 0.0213 | the stratum W2 = 1 label's SD over the root of n |
| `bound:contrast_1_standard_error` | 0.0146 | the SD of the contrast against Static(1) over the root of n |
| `bound:contrast_0_standard_error` | 0.0159 | the SD of the contrast against Static(0) over the root of n |
| `bound:in_sample_standard_error` | 0.0149 | the in-sample label's SD over the root of n |
| `bound:intermediate_z0_standard_error` | 0.0251 | the z = 0 label's SD over the root of n |
| `bound:intermediate_z1_standard_error` | 0.0197 | the z = 1 label's SD over the root of n |
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
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit, and the target controls' ceiling |
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
| `margin:subject_failures_allowed` | 44 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a cross-validated TMLE of the value of the optimal rule under these compositions. The rows rest on accuracy against the truth and on the theory properties |
| One law per setting, one size and one learner per role | The study fits n = 2,000 with a binary treatment and outcome, ten equal folds, pooled targeting, the plug-in rule, `repeats=1` and iid rows. The outcome model is a correctly specified parametric logistic model, so its limit rule is the setting's optimal rule, and each blip has no tie set. The key holds neither the learner, the law nor n, so a misspecified or flexible learner, a law with ties, a steeper margin or another n reads supported and was not measured |
| One composition at a time, each in one form | The weighted row measures one weight, $1 + AY$; the stratum row strata of one binary column; the known-regimes row `Static(1)` and `Static(0)`, not `reference=`; the intermediate row one binary intermediate. Two compositions together, such as known regimes with strata, have no row |
| The interval covers the optimal value up to the regret | The mean regret is 0.0012 to 0.0028 across the labels at n = 2,000, and the primary bias intervals exclude zero because of it. One n is measured, so the rate is not shown |
| Two SEs read slightly small | The weighted label's SE ratio is 0.972 (99% interval 0.955 to 0.990) and the contrast against `Static(1)` 0.977 (0.960 to 0.995), inside the band but excluding 1 |
| The in-sample fit leans | Against the value of the one rule it learned the in-sample label reads a standardized bias of +0.179, and +0.096 against the optimal value. The row needs a parametric outcome learner, which keys condition C5a |
| Known treatment mechanism only, and the weighted population's for weights | Every row needs `treatment_probabilities=`, which keys `known_mechanism`; an estimated $g$ reads experimental. Under weights the probabilities must be $g_w$; the design $g_0$ beside a weight in $A$ or $Y$ is the Fix-16 defect, which the key cannot see and no cell measures |
| Blind spots of the intermediate and stratum cells | The intermediate cells cannot tell a fit that ignores $Z$, and the stratum control resolves a gross normalisation error only; both rest on the unit witnesses and on W-f3-id |
| Pointwise intervals | The study reads the pointwise interval of each label. A fit with known regimes or strata builds no band over the optimal value |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_compositions/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_compositions/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_compositions/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_compositions/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_compositions/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_optimal_compositions/properties.csv)
carry every published row.
