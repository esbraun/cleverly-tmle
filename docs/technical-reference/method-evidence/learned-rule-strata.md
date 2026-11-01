# Learned-rule value inside baseline strata

This study validates the interval of the fold-evaluated learned-rule value when the fit declares
baseline strata. The estimator is `TMLE(learned_rule=LearnedRule(), cross_fit=True, n_folds=5,
cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto", simultaneous=False)` fitted with
`strata=["S"]` on complete data, rows as the unit and `repeats=1`. Fold $v$'s plug-in rule $d_v$
comes from its training rows. Each stratum estimate is the $1/V$ average of the stratum plug-ins
at $d_v$ on the validation folds, each fold's stratum curve divides by the fold's own stratum share
$P_{n,v}(S=s)$, and the marginal estimate is the share mixture of the stratum estimates. That is
the construction of [fold-evaluated CV-TMLE with baseline strata](fold-evaluated-strata.md) with
the fold rule as the regime. Each fit reports four parameters: `ey_learned_rule[learned rule]` in
each of three strata and marginally.

**No canonical implementation is compared.** No pinned comparator reports a fold-averaged
stratified learned-rule value. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law | the law of the fold-evaluated strata study: $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and mean $m = \bar Q_0(A, W)$ with $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W_1)$. $S$ cuts $W_1$ at $-0.6$ and $0.1$, so the stratum shares are 0.2, 0.35 and 0.45 |
| bounded outcome | $Y \sim \operatorname{Beta}(12 m, 12 (1 - m))$ with $b = 0.1 + 1.5 W_1$, fitted with `q_bounds=(0, 1)`. The true rule's boundary $W_1 = -1/15$ lies in stratum 1 (share 0.35). Stratum 0 is asymptotically treat-no-one and stratum 2 treat-everyone |
| binary outcome | $Y \sim \operatorname{Bernoulli}(m)$ with $b = -1.2 + 3.0 W_1$, fitted with no `q_bounds`. The boundary $W_1 = 0.4$ lies in stratum 2 (share 0.45), and strata 0 and 1 are asymptotically treat-no-one |
| settings | `unweighted` and `weighted` on the bounded outcome, `binary_steep` and `binary_steep_weighted` on the binary one. The weight $w = 1 + 1.5 W_2$ varies inside every stratum |
| outcome learner | the pairwise products of $(A, W_1, W_2, S)$ with a quasibinomial GLM on the bounded outcome, and with a logistic regression on the binary one. Both are correctly specified on the logit scale, and each fold's plug-in rule is the sign of the fitted blip |
| treatment learner | a logistic regression on $(W_1, W_2, S)$, a correct parametric model |
| truth | per replication: each fold's rule is refitted from its training rows with the fit's own learner and checked against the fit's rule on every row. Each stratum value $\Psi_{s,d_v}(P_0)$ is integrated by the trapezoid rule on 4,001 points of $W_1$ per stratum for each $W_2$, weighted by $w(W_2)$, and averaged over the folds. The marginal truth is the share mixture |
| sizes and budgets | primary: n = 1,000 and 2,000 replications for each setting. `interval_calibration`: n = 1,000 and 14,800 replications for each setting. `targeting_necessity`: n = 1,000 and 3,600 replications. `weak_margin_diagnostic`: n = 1,000 and 4,000 replications for each of two laws, with no verdict |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| policy | `gated` |

The target is data-adaptive. The fit estimates $\psi_s = V^{-1} \sum_v \Psi_{s,d_v}(P_0)$, the
value of the rules the folds learned, so the boundary stratum's target varies with the learned
rules. A stratum that the rules treat alike has a fixed target: stratum 0 reads the same truth in
every replication, while the binary boundary stratum's truth ranges from 0.6548 to 0.6932. Each
row is read against the truth of its own replication. The rule boundary was never placed in the
smallest stratum.

Each blip has a nonzero slope at its root, 1.5 on the bounded outcome and 3.0 on the logit scale
on the binary one, so the boundary has probability zero and the law meets the margin condition
of the learned-rule value (van der Laan and Luedtke 2015, Theorem 6). The binary slope was
chosen before any registered run. The fold-evaluated strata study's binary blip, $0.1 - 0.3 W_1$,
is weak across the whole support ($|b| \le 0.4$), and the learned-rule value under-covers there
in every stratum. That is the finite-sample weak-margin limit of
[F27](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions), which the
[learned-rule boundary study](learned-rule-cvtmle-boundary.md) measures at its `weak_blip` law.
This study publishes that regime as the descriptive `weak_margin_diagnostic` family.

`tests/unit/test_learned_rule_strata.py` checks the construction on an exact discrete law: the
fold average at each fold's own rule, the fold-local share and the share term of the marginal
curve. A deliberate mutation, the full-sample rule in every fold, fails it, and three
perturbations of the reported curve show that the checks have power: the whole-sample share,
another fold's rule and a curve without the stratum share. `tests/unit/test_learned_rule_strata_design.py` recomputes the
limit-rule values, the efficiency bounds that size the noise controls and the asymptotic
share-term ratios, and `tests/unit/test_learned_rule_strata_harness.py` checks the fits, the
truth harness and the controls.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a binary outcome with the blip -1.2 + 3.0 W1, rows only | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000799 to 0.0016 | 0.9490 | 1.0036 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a binary outcome with the blip -1.2 + 3.0 W1, rows only | `ey_learned_rule[learned rule][S=0]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=0 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.0022 to 0.0031 | 0.9490 | 1.0153 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a binary outcome with the blip -1.2 + 3.0 W1, rows only | `ey_learned_rule[learned rule][S=1]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=1 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.0025 to 0.0018 | 0.9430 | 0.9969 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a binary outcome with the blip -1.2 + 3.0 W1, rows only | `ey_learned_rule[learned rule][S=2]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=2 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000482 to 0.0031 | 0.9485 | 0.9831 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a binary outcome with the blip -1.2 + 3.0 W1 and the known weight 1 + 1.5 W2 | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.0013 to 0.0014 | 0.9445 | 0.9948 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a binary outcome with the blip -1.2 + 3.0 W1 and the known weight 1 + 1.5 W2 | `ey_learned_rule[learned rule][S=0]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=0 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.0037 to 0.0020 | 0.9390 | 1.0004 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a binary outcome with the blip -1.2 + 3.0 W1 and the known weight 1 + 1.5 W2 | `ey_learned_rule[learned rule][S=1]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=1 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.0016 to 0.0029 | 0.9475 | 1.0129 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a binary outcome with the blip -1.2 + 3.0 W1 and the known weight 1 + 1.5 W2 | `ey_learned_rule[learned rule][S=2]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=2 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.0019 to 0.0021 | 0.9365 | 0.9635 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a beta outcome with the blip 0.1 + 1.5 W1, rows only | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000571 to 0.000325 | 0.9470 | 0.9983 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a beta outcome with the blip 0.1 + 1.5 W1, rows only | `ey_learned_rule[learned rule][S=0]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=0 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000972 to 0.000530 | 0.9495 | 1.0125 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a beta outcome with the blip 0.1 + 1.5 W1, rows only | `ey_learned_rule[learned rule][S=1]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=1 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000282 to 0.000935 | 0.9525 | 1.0052 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a beta outcome with the blip 0.1 + 1.5 W1, rows only | `ey_learned_rule[learned rule][S=2]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=2 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000624 to 0.000377 | 0.9505 | 1.0240 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a beta outcome with the blip 0.1 + 1.5 W1 and the known weight 1 + 1.5 W2 | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000358 to 0.000646 | 0.9450 | 0.9826 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a beta outcome with the blip 0.1 + 1.5 W1 and the known weight 1 + 1.5 W2 | `ey_learned_rule[learned rule][S=0]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=0 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000800 to 0.000818 | 0.9435 | 0.9867 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a beta outcome with the blip 0.1 + 1.5 W1 and the known weight 1 + 1.5 W2 | `ey_learned_rule[learned rule][S=1]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=1 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000531 to 0.000824 | 0.9385 | 0.9677 | pass |
| W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut from W1 at shares 0.2, 0.35 and 0.45; a beta outcome with the blip 0.1 + 1.5 W1 and the known weight 1 + 1.5 W2 | `ey_learned_rule[learned rule][S=2]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression, in stratum S=2 | `cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal is their share mixture | -0.000220 to 0.000946 | 0.9460 | 0.9928 | pass |
<!-- /generated -->

## Theory properties

Every property cell fits n = 1,000 with the declared learners, unless the cell says otherwise.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each positive cell lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, with 14,800 replications. The positives are the four parameters of each of the four settings | a shrunken-SE control multiplies each SE by 0.70, and a noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. Two construction controls read the same fit: `unnormalised_curve_control` on each stratum parameter and `share_dropped_control` on the marginal value of the bounded settings. Each control must put its SE-ratio interval below 0.93 |
| `targeting_necessity` | is the targeted stratum estimate unbiased when the outcome learner sees neither $W_1$ nor the stratum, with 3,600 replications of the bounded rows-only setting | the same fit without its fluctuation, the fold average of the initial stratum plug-ins at the fit's rule, must be biased outside the margin, and the least displaced pair must move by at least 0.5 positive-arm SDs |
| `weak_margin_diagnostic` | what the coverage and the SE ratio of the declared fit are on two binary laws with a weaker margin, and of the same estimator without `strata=` on the same samples and folds | nothing fails. The cells have the role `diagnostic`, and no margin is read |

The noise scale is the efficient influence curve SD of each parameter at the limit rule, by the
same quadrature as the truth, and no verdict reads it as an efficiency bound. The sizes were set
before any registered run, by the per-arm sizing rule of the studies over all 16 positives on
pilots at the declared size.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `binary_steep_all__declared` | positive | the binary learned-rule law with baseline strata, rows only, the learned-rule value marginally: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9419 to 0.9515, SE ratio 0.9790 to 1.0091 | pass |
| `interval_calibration` | `binary_steep_all__noise_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value marginally: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8280 to 0.8437, SE ratio 0.6995 to 0.7210 | pass |
| `interval_calibration` | `binary_steep_all__shrunken_se_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value marginally: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8191 to 0.8352, SE ratio 0.6853 to 0.7066 | pass |
| `interval_calibration` | `binary_steep_s0__declared` | positive | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9434 to 0.9528, SE ratio 0.9889 to 1.0198 | pass |
| `interval_calibration` | `binary_steep_s0__noise_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8262 to 0.8420, SE ratio 0.7003 to 0.7217 | pass |
| `interval_calibration` | `binary_steep_s0__shrunken_se_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8238 to 0.8397, SE ratio 0.6924 to 0.7140 | pass |
| `interval_calibration` | `binary_steep_s0__unnormalised_curve_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.2930 to 0.3125, SE ratio 0.1957 to 0.2019 | pass |
| `interval_calibration` | `binary_steep_s1__declared` | positive | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9535, SE ratio 0.9870 to 1.0158 | pass |
| `interval_calibration` | `binary_steep_s1__noise_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8246 to 0.8405, SE ratio 0.6978 to 0.7193 | pass |
| `interval_calibration` | `binary_steep_s1__shrunken_se_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8183 to 0.8344, SE ratio 0.6908 to 0.7116 | pass |
| `interval_calibration` | `binary_steep_s1__unnormalised_curve_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.4942 to 0.5154, SE ratio 0.3436 to 0.3541 | pass |
| `interval_calibration` | `binary_steep_s2__declared` | positive | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9388 to 0.9486, SE ratio 0.9727 to 1.0025 | pass |
| `interval_calibration` | `binary_steep_s2__noise_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8270 to 0.8428, SE ratio 0.6987 to 0.7200 | pass |
| `interval_calibration` | `binary_steep_s2__shrunken_se_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8171 to 0.8332, SE ratio 0.6807 to 0.7015 | pass |
| `interval_calibration` | `binary_steep_s2__unnormalised_curve_control` | control | the binary learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.6069 to 0.6275, SE ratio 0.4364 to 0.4499 | pass |
| `interval_calibration` | `binary_steep_weighted_all__declared` | positive | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value marginally: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9548, SE ratio 0.9843 to 1.0145 | pass |
| `interval_calibration` | `binary_steep_weighted_all__noise_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value marginally: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8302 to 0.8459, SE ratio 0.7013 to 0.7226 | pass |
| `interval_calibration` | `binary_steep_weighted_all__shrunken_se_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value marginally: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8177 to 0.8338, SE ratio 0.6895 to 0.7099 | pass |
| `interval_calibration` | `binary_steep_weighted_s0__declared` | positive | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 0: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9531, SE ratio 0.9946 to 1.0250 | pass |
| `interval_calibration` | `binary_steep_weighted_s0__noise_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 0: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8344 to 0.8499, SE ratio 0.7132 to 0.7353 | pass |
| `interval_calibration` | `binary_steep_weighted_s0__shrunken_se_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8229 to 0.8389, SE ratio 0.6957 to 0.7174 | pass |
| `interval_calibration` | `binary_steep_weighted_s0__unnormalised_curve_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 0: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.2964 to 0.3159, SE ratio 0.1969 to 0.2030 | pass |
| `interval_calibration` | `binary_steep_weighted_s1__declared` | positive | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 1: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9428 to 0.9523, SE ratio 0.9894 to 1.0194 | pass |
| `interval_calibration` | `binary_steep_weighted_s1__noise_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 1: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8306 to 0.8462, SE ratio 0.7026 to 0.7237 | pass |
| `interval_calibration` | `binary_steep_weighted_s1__shrunken_se_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8223 to 0.8382, SE ratio 0.6922 to 0.7138 | pass |
| `interval_calibration` | `binary_steep_weighted_s1__unnormalised_curve_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 1: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.4970 to 0.5183, SE ratio 0.3444 to 0.3553 | pass |
| `interval_calibration` | `binary_steep_weighted_s2__declared` | positive | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 2: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9358 to 0.9459, SE ratio 0.9678 to 0.9978 | pass |
| `interval_calibration` | `binary_steep_weighted_s2__noise_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 2: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8221 to 0.8381, SE ratio 0.6956 to 0.7169 | pass |
| `interval_calibration` | `binary_steep_weighted_s2__shrunken_se_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8127 to 0.8290, SE ratio 0.6781 to 0.6982 | pass |
| `interval_calibration` | `binary_steep_weighted_s2__unnormalised_curve_control` | control | the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 2: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.5996 to 0.6203, SE ratio 0.4347 to 0.4479 | pass |
| `interval_calibration` | `unweighted_all__declared` | positive | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value marginally: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9478 to 0.9569, SE ratio 0.9894 to 1.0187 | pass |
| `interval_calibration` | `unweighted_all__noise_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value marginally: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8320 to 0.8476, SE ratio 0.7053 to 0.7262 | pass |
| `interval_calibration` | `unweighted_all__share_dropped_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value marginally: the same fit with the marginal curve missing its share term, the spread of the stratum estimates over the strata | the SE-ratio interval must fall below the calibration band | coverage 0.8674 to 0.8815, SE ratio 0.7727 to 0.7961 | pass |
| `interval_calibration` | `unweighted_all__shrunken_se_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value marginally: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8217 to 0.8377, SE ratio 0.6926 to 0.7134 | pass |
| `interval_calibration` | `unweighted_s0__declared` | positive | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9457 to 0.9550, SE ratio 0.9919 to 1.0222 | pass |
| `interval_calibration` | `unweighted_s0__noise_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8309 to 0.8466, SE ratio 0.7038 to 0.7251 | pass |
| `interval_calibration` | `unweighted_s0__shrunken_se_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8234 to 0.8393, SE ratio 0.6944 to 0.7154 | pass |
| `interval_calibration` | `unweighted_s0__unnormalised_curve_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.2920 to 0.3115, SE ratio 0.1965 to 0.2024 | pass |
| `interval_calibration` | `unweighted_s1__declared` | positive | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9544, SE ratio 0.9900 to 1.0203 | pass |
| `interval_calibration` | `unweighted_s1__noise_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8322 to 0.8478, SE ratio 0.7044 to 0.7272 | pass |
| `interval_calibration` | `unweighted_s1__shrunken_se_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8272 to 0.8430, SE ratio 0.6931 to 0.7146 | pass |
| `interval_calibration` | `unweighted_s1__unnormalised_curve_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.5000 to 0.5212, SE ratio 0.3449 to 0.3558 | pass |
| `interval_calibration` | `unweighted_s2__declared` | positive | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9448 to 0.9541, SE ratio 0.9888 to 1.0189 | pass |
| `interval_calibration` | `unweighted_s2__noise_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8264 to 0.8423, SE ratio 0.6988 to 0.7198 | pass |
| `interval_calibration` | `unweighted_s2__shrunken_se_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8228 to 0.8387, SE ratio 0.6925 to 0.7131 | pass |
| `interval_calibration` | `unweighted_s2__unnormalised_curve_control` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.6162 to 0.6368, SE ratio 0.4436 to 0.4572 | pass |
| `interval_calibration` | `weighted_all__declared` | positive | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value marginally: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9452 to 0.9545, SE ratio 0.9869 to 1.0174 | pass |
| `interval_calibration` | `weighted_all__noise_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value marginally: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8293 to 0.8450, SE ratio 0.7024 to 0.7241 | pass |
| `interval_calibration` | `weighted_all__share_dropped_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value marginally: the same fit with the marginal curve missing its share term, the spread of the stratum estimates over the strata | the SE-ratio interval must fall below the calibration band | coverage 0.8640 to 0.8783, SE ratio 0.7658 to 0.7889 | pass |
| `interval_calibration` | `weighted_all__shrunken_se_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value marginally: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8231 to 0.8390, SE ratio 0.6911 to 0.7117 | pass |
| `interval_calibration` | `weighted_s0__declared` | positive | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 0: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9446 to 0.9540, SE ratio 0.9907 to 1.0205 | pass |
| `interval_calibration` | `weighted_s0__noise_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 0: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8263 to 0.8421, SE ratio 0.7027 to 0.7239 | pass |
| `interval_calibration` | `weighted_s0__shrunken_se_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8207 to 0.8368, SE ratio 0.6930 to 0.7141 | pass |
| `interval_calibration` | `weighted_s0__unnormalised_curve_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 0: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.2927 to 0.3123, SE ratio 0.1961 to 0.2021 | pass |
| `interval_calibration` | `weighted_s1__declared` | positive | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 1: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9399 to 0.9496, SE ratio 0.9763 to 1.0048 | pass |
| `interval_calibration` | `weighted_s1__noise_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 1: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8230 to 0.8389, SE ratio 0.6959 to 0.7175 | pass |
| `interval_calibration` | `weighted_s1__shrunken_se_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8187 to 0.8348, SE ratio 0.6833 to 0.7035 | pass |
| `interval_calibration` | `weighted_s1__unnormalised_curve_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 1: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.4855 to 0.5067, SE ratio 0.3400 to 0.3502 | pass |
| `interval_calibration` | `weighted_s2__declared` | positive | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 2: the declared stratified learned-rule fit with correctly specified learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9475 to 0.9566, SE ratio 1.0026 to 1.0328 | pass |
| `interval_calibration` | `weighted_s2__noise_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 2: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8300 to 0.8456, SE ratio 0.7019 to 0.7227 | pass |
| `interval_calibration` | `weighted_s2__shrunken_se_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8282 to 0.8440, SE ratio 0.7020 to 0.7229 | pass |
| `interval_calibration` | `weighted_s2__unnormalised_curve_control` | control | the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2, the learned-rule value in stratum S = 2: the same fit with each fold's stratum curve not divided by the fold's own stratum share, so its SE is about the stratum share times the reported one | the SE-ratio interval must fall below the calibration band | coverage 0.6138 to 0.6344, SE ratio 0.4503 to 0.4635 | pass |
| `targeting_necessity` | `unweighted_s0__targeted` | positive | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: the fit with an outcome learner that sees neither W1 nor the stratum, so its initial plug-in is nearly the same in every stratum and its rule reads W2 alone; the truth is the harness refit with that learner | bias interval inside the equivalence margin | bias 0.0012 to 0.0023, margin 0.0032 | pass |
| `targeting_necessity` | `unweighted_s0__untargeted` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 0: the same fit without its fluctuation: the fold average of the stratum-blind initial stratum plug-ins at the fit's rule | bias interval must fall entirely outside the margin | bias 0.3480 to 0.3504, margin 0.0068 | pass |
| `targeting_necessity` | `unweighted_s1__targeted` | positive | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: the fit with an outcome learner that sees neither W1 nor the stratum, so its initial plug-in is nearly the same in every stratum and its rule reads W2 alone; the truth is the harness refit with that learner | bias interval inside the equivalence margin | bias -0.000771 to 0.000320, margin 0.0032 | pass |
| `targeting_necessity` | `unweighted_s1__untargeted` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 1: the same fit without its fluctuation: the fold average of the stratum-blind initial stratum plug-ins at the fit's rule | bias interval must fall entirely outside the margin | bias 0.1305 to 0.1318, margin 0.0038 | pass |
| `targeting_necessity` | `unweighted_s2__targeted` | positive | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: the fit with an outcome learner that sees neither W1 nor the stratum, so its initial plug-in is nearly the same in every stratum and its rule reads W2 alone; the truth is the harness refit with that learner | bias interval inside the equivalence margin | bias -0.0011 to -0.000248, margin 0.0023 | pass |
| `targeting_necessity` | `unweighted_s2__untargeted` | control | the bounded (beta) learned-rule law with baseline strata, rows only, the learned-rule value in stratum S = 2: the same fit without its fluctuation: the fold average of the stratum-blind initial stratum plug-ins at the fit's rule | bias interval must fall entirely outside the margin | bias -0.2083 to -0.2069, margin 0.0040 | pass |
| `weak_margin_diagnostic` | `binary_moderate_all__diagnostic` | diagnostic | the binary learned-rule law with baseline strata and the blip -0.6 + 1.5 W1, rows only, the learned-rule value marginally: the declared fit and the same fit without strata on the same samples and folds at n = 1,000, reported without a verdict | none. The cell reports both fits' coverage and SE ratio and the boundary stratum's fold-rule disagreement | SE ratio 0.9706 to 1.0269 | reported |
| `weak_margin_diagnostic` | `binary_moderate_s0__diagnostic` | diagnostic | the binary learned-rule law with baseline strata and the blip -0.6 + 1.5 W1, rows only, the learned-rule value in stratum S = 0: the declared fit and the same fit without strata on the same samples and folds at n = 1,000, reported without a verdict | none. The cell reports both fits' coverage and SE ratio and the boundary stratum's fold-rule disagreement | SE ratio 0.9777 to 1.0360 | reported |
| `weak_margin_diagnostic` | `binary_moderate_s1__diagnostic` | diagnostic | the binary learned-rule law with baseline strata and the blip -0.6 + 1.5 W1, rows only, the learned-rule value in stratum S = 1: the declared fit and the same fit without strata on the same samples and folds at n = 1,000, reported without a verdict | none. The cell reports both fits' coverage and SE ratio and the boundary stratum's fold-rule disagreement | SE ratio 0.9776 to 1.0371 | reported |
| `weak_margin_diagnostic` | `binary_moderate_s2__diagnostic` | diagnostic | the binary learned-rule law with baseline strata and the blip -0.6 + 1.5 W1, rows only, the learned-rule value in stratum S = 2: the declared fit and the same fit without strata on the same samples and folds at n = 1,000, reported without a verdict | none. The cell reports both fits' coverage and SE ratio and the boundary stratum's fold-rule disagreement | SE ratio 0.9400 to 0.9979 | reported |
| `weak_margin_diagnostic` | `binary_weak_all__diagnostic` | diagnostic | the binary learned-rule law with baseline strata and the weak blip 0.1 - 0.3 W1, rows only, the learned-rule value marginally: the declared fit and the same fit without strata on the same samples and folds at n = 1,000, reported without a verdict | none. The cell reports both fits' coverage and SE ratio and the boundary stratum's fold-rule disagreement | SE ratio 0.8767 to 0.9292 | reported |
| `weak_margin_diagnostic` | `binary_weak_s0__diagnostic` | diagnostic | the binary learned-rule law with baseline strata and the weak blip 0.1 - 0.3 W1, rows only, the learned-rule value in stratum S = 0: the declared fit and the same fit without strata on the same samples and folds at n = 1,000, reported without a verdict | none. The cell reports both fits' coverage and SE ratio and the boundary stratum's fold-rule disagreement | SE ratio 0.9378 to 0.9954 | reported |
| `weak_margin_diagnostic` | `binary_weak_s1__diagnostic` | diagnostic | the binary learned-rule law with baseline strata and the weak blip 0.1 - 0.3 W1, rows only, the learned-rule value in stratum S = 1: the declared fit and the same fit without strata on the same samples and folds at n = 1,000, reported without a verdict | none. The cell reports both fits' coverage and SE ratio and the boundary stratum's fold-rule disagreement | SE ratio 0.9300 to 0.9882 | reported |
| `weak_margin_diagnostic` | `binary_weak_s2__diagnostic` | diagnostic | the binary learned-rule law with baseline strata and the weak blip 0.1 - 0.3 W1, rows only, the learned-rule value in stratum S = 2: the declared fit and the same fit without strata on the same samples and folds at n = 1,000, reported without a verdict | none. The cell reports both fits' coverage and SE ratio and the boundary stratum's fold-rule disagreement | SE ratio 0.8648 to 0.9168 | reported |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed. The 16 primary tests at n = 1,000 have a
worst standardized bias of 0.0419, coverage of 0.9365 to 0.9525, and SE ratios of 0.963 to 1.024.
The 16 calibration positives have SE ratios of 0.9827 to 1.0177 and coverage of 0.9410 to 0.9525,
and every interval lies inside its band. The tightest positive was the one declared before the
run, `binary_steep_weighted_s2`, the cell with the most fold-rule disagreement.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 16 of 16 | `performance-tests.csv` |
| gated property cells passed | 68 of 68 | `properties.csv` |
| diagnostic cells, reported with no verdict | 8 | `properties.csv`, `weak_margin_diagnostic` |
| failed primary fits | 0 | `manifest.json` and the run log |
| smallest calibration SE-ratio lower endpoint | 0.9678 | `properties.csv`, `binary_steep_weighted_s2__declared` |
| smallest calibration coverage lower endpoint | 0.9358 | `properties.csv`, `binary_steep_weighted_s2__declared` |
| fold-rule disagreement in the boundary stratum | 7.6% and 7.8% bounded, 14.0% and 14.3% binary | `strata-harness.csv.gz` |

The fold-rule disagreement is the share of the boundary stratum's rows on which the five stored
fold rules disagree, averaged over the primary replications. It measures how far the rules are
from settled, and no verdict reads it.

### What each control shows

The two construction controls show different things, so the page counts them apart: 12 of 12
`unnormalised_curve_control` cells and 2 of 2 `share_dropped_control` cells fall below the band.

`unnormalised_curve_control` fails by construction. Its SE is about the stratum share times the
reported one, and its SE ratios are 0.199 to 0.457 against shares of 0.2 to 0.45. It shows that
the instrument resolves a gross normalisation error. It does not show that the fold-local share
is the right one, which rests on the unit witnesses of `tests/unit/test_learned_rule_strata.py`
and of the fold-evaluated strata study.

`share_dropped_control` tests the marginal derivation for the learned rule. Without its share term
the marginal value reads SE ratios of 0.7843 rows only and 0.7770 weighted, against the asymptotic
0.778 and 0.773 at the limit rule, and its upper endpoints lie below 0.80. On the binary outcome
the asymptotic ratio is 0.98, which the band cannot resolve, so no binary cell carries this
control.

### What the targeting control shows

The control learner keeps $A$ and $W_2$ of the design $(A, W_1, W_2, S)$, so it is stratum-blind:
its plug-in is nearly the same in every stratum, and its plug-in rule is a function of $W_2$ alone.
Each of its truths is the harness refitted with that learner, the value of the rules it learned.
Its untargeted standardized stratum biases run from 8.55 to 12.8 in size, and the targeted fit's
largest is 0.138. The cell shows that targeting repairs a stratum-blind initial fit at a learned
rule. It does not show that targeting repairs a subtle misspecification within a stratum.

### Weak margins at the rule boundary

The `weak_margin_diagnostic` family fits two binary rows-only laws with a weaker margin and the
same estimator without `strata=` on the same samples and folds. At the blip $0.1 - 0.3 W_1$ the
five fold rules disagree on 44.4% of the boundary stratum's rows, and the stratified fit covers
0.9153 to 0.9385 in every stratum and 0.9243 marginally, with SE ratios of 0.890 to 0.966. The
unstratified fit on the same draws covers 0.9215 with an SE ratio of 0.900. So the shortfall is
the learned-rule value's own, as the boundary study's `weak_blip` row reads at n = 2,000, and not
the stratified construction's.

At the blip $-0.6 + 1.5 W_1$ the fold rules disagree on 25.6% of the boundary stratum's rows. The
boundary stratum reads an SE ratio of 0.968 and coverage of 0.9395, while the other two strata read
SE ratios of 1.007 and the marginal value 0.998. The unstratified marginal fit reads 0.996. With a
weaker margin the target moves with the learned rules, and the variance of learning the rule is
not yet negligible at n = 1,000, first in the boundary stratum. No verdict reads the family, and
F27 owns the regime.

The readings do not show the items in this list.

- A rule-learner loss, fold targeting, clusters or `repeats` above 1 with strata. Each reads
  experimental or refuses.
- More than five folds, smaller strata, another sample size, or a rule boundary in the smallest
  stratum.
- A flexible learner or a simultaneous band.
- The fold-local share against the whole-sample share, which the exact-law tests carry.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the efficient influence curve SD of a parameter at the limit rule over $\sqrt{1000}$. They size
the noise controls only, and no verdict reads them as an efficiency bound. Everything else is
measured from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications per setting |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 16 | parameter tests against the truth |
| `independent_tests_passed` | 16 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 68 | gated repeated-sampling property cells |
| `property_cells_passed` | 68 | cells whose own and family verdicts pass |
| `property_cells_reported` | 8 | diagnostic cells, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0419 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9365 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9211 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9275 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0696 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-strata/unweighted/ey_learned_rule[learned rule]]:coverage` | 0.9470 | primary coverage of the marginal value, bounded rows only |
| `performance[cleverly-learned-rule-strata/unweighted/ey_learned_rule[learned rule][S=1]]:coverage` | 0.9525 | primary coverage in the bounded boundary stratum |
| `performance[cleverly-learned-rule-strata/binary_steep_weighted/ey_learned_rule[learned rule][S=2]]:coverage` | 0.9365 | primary coverage in the binary boundary stratum, weighted |
| `performance[cleverly-learned-rule-strata/binary_steep_weighted/ey_learned_rule[learned rule][S=2]]:se_ratio` | 0.9635 | its primary SE ratio |
| `summary[cleverly-learned-rule-strata/binary_steep/ey_learned_rule[learned rule][S=0]]:truth_min` | 0.4139 | the smallest truth of stratum 0, binary rows only |
| `summary[cleverly-learned-rule-strata/binary_steep/ey_learned_rule[learned rule][S=0]]:truth_max` | 0.4139 | its largest truth: the stratum's target is fixed |
| `summary[cleverly-learned-rule-strata/binary_steep/ey_learned_rule[learned rule][S=2]]:truth_min` | 0.6548 | the smallest truth of the binary boundary stratum |
| `summary[cleverly-learned-rule-strata/binary_steep/ey_learned_rule[learned rule][S=2]]:truth_max` | 0.6932 | its largest truth |
| `boundary_disagreement_mean_unweighted` | 0.0757 | mean fold-rule disagreement in the boundary stratum, bounded rows only |
| `boundary_disagreement_mean_weighted` | 0.0785 | the same, bounded weighted |
| `boundary_disagreement_mean_binary_steep` | 0.1398 | the same, binary rows only |
| `boundary_disagreement_mean_binary_steep_weighted` | 0.1433 | the same, binary weighted |
| `properties[interval_calibration/unweighted_all__declared]:se_ratio` | 1.0038 | calibration SE ratio of the marginal value, bounded rows only |
| `properties[interval_calibration/unweighted_s1__declared]:se_ratio` | 1.0053 | calibration SE ratio in the bounded boundary stratum |
| `properties[interval_calibration/weighted_s2__declared]:se_ratio` | 1.0177 | the largest calibration SE ratio |
| `properties[interval_calibration/weighted_s2__declared]:se_ratio_ci_upper` | 1.0328 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/binary_steep_weighted_s2__declared]:se_ratio` | 0.9827 | calibration SE ratio in the binary boundary stratum, weighted |
| `properties[interval_calibration/binary_steep_weighted_s2__declared]:se_ratio_ci_lower` | 0.9678 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/binary_steep_weighted_s2__declared]:coverage` | 0.9410 | its calibration coverage |
| `properties[interval_calibration/binary_steep_weighted_s2__declared]:coverage_ci_lower` | 0.9358 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/unweighted_s0__unnormalised_curve_control]:se_ratio` | 0.1994 | unnormalised-curve SE ratio in the 0.2 stratum |
| `properties[interval_calibration/weighted_s2__unnormalised_curve_control]:se_ratio` | 0.4569 | unnormalised-curve SE ratio in the 0.45 stratum, weighted |
| `properties[interval_calibration/unweighted_all__share_dropped_control]:se_ratio` | 0.7843 | share-dropped SE ratio of the marginal value, rows only |
| `properties[interval_calibration/unweighted_all__share_dropped_control]:se_ratio_ci_upper` | 0.7961 | its upper endpoint, against 0.93 |
| `properties[interval_calibration/weighted_all__share_dropped_control]:se_ratio` | 0.7770 | share-dropped SE ratio of the marginal value, weighted |
| `properties[interval_calibration/weighted_all__share_dropped_control]:se_ratio_ci_upper` | 0.7889 | its upper endpoint, against 0.93 |
| `properties[interval_calibration/binary_steep_weighted_s0__noise_control]:se_ratio_ci_upper` | 0.7353 | the highest noise-control upper endpoint |
| `properties[interval_calibration/weighted_s2__shrunken_se_control]:se_ratio_ci_upper` | 0.7229 | the highest shrunken-SE control upper endpoint |
| `properties[targeting_necessity/unweighted_s0__targeted]:standardized_bias` | 0.1378 | the largest targeted standardized bias |
| `properties[targeting_necessity/unweighted_s2__untargeted]:standardized_bias` | -12.8322 | the largest untargeted standardized bias in size |
| `properties[targeting_necessity/unweighted_s1__untargeted]:standardized_bias` | 8.5513 | the smallest untargeted standardized bias |
| `properties[targeting_necessity/unweighted_s0__targeted]:targeting_displacement` | 10.3526 | the least displacement over the labels, against 0.5 |
| `properties[weak_margin_diagnostic/binary_weak_s2__diagnostic]:boundary_disagreement_mean` | 0.4438 | fold-rule disagreement in the boundary stratum at the blip $0.1 - 0.3 W_1$ |
| `properties[weak_margin_diagnostic/binary_weak_s2__diagnostic]:coverage` | 0.9153 | the stratified fit's coverage in that stratum |
| `properties[weak_margin_diagnostic/binary_weak_s1__diagnostic]:coverage` | 0.9385 | the highest stratified coverage at that blip |
| `properties[weak_margin_diagnostic/binary_weak_all__diagnostic]:coverage` | 0.9243 | the stratified fit's marginal coverage |
| `properties[weak_margin_diagnostic/binary_weak_all__diagnostic]:coverage_ci_upper` | 0.9346 | its upper endpoint |
| `properties[weak_margin_diagnostic/binary_weak_all__diagnostic]:se_ratio` | 0.9017 | the stratified fit's marginal SE ratio |
| `properties[weak_margin_diagnostic/binary_weak_s2__diagnostic]:se_ratio` | 0.8899 | its SE ratio in the boundary stratum |
| `properties[weak_margin_diagnostic/binary_weak_s0__diagnostic]:se_ratio` | 0.9660 | its SE ratio in stratum 0 |
| `properties[weak_margin_diagnostic/binary_weak_all__diagnostic]:unstratified_coverage` | 0.9215 | the unstratified fit's coverage on the same draws |
| `properties[weak_margin_diagnostic/binary_weak_all__diagnostic]:unstratified_coverage_ci_upper` | 0.9321 | its upper endpoint |
| `properties[weak_margin_diagnostic/binary_weak_all__diagnostic]:unstratified_se_ratio` | 0.9000 | the unstratified fit's SE ratio on the same draws |
| `properties[weak_margin_diagnostic/binary_moderate_s2__diagnostic]:boundary_disagreement_mean` | 0.2564 | fold-rule disagreement in the boundary stratum at the blip $-0.6 + 1.5 W_1$ |
| `properties[weak_margin_diagnostic/binary_moderate_s2__diagnostic]:se_ratio` | 0.9675 | the stratified fit's SE ratio in that stratum |
| `properties[weak_margin_diagnostic/binary_moderate_s2__diagnostic]:coverage` | 0.9395 | its coverage there |
| `properties[weak_margin_diagnostic/binary_moderate_s0__diagnostic]:se_ratio` | 1.0068 | its SE ratio in stratum 0 |
| `properties[weak_margin_diagnostic/binary_moderate_s1__diagnostic]:se_ratio` | 1.0065 | its SE ratio in stratum 1 |
| `properties[weak_margin_diagnostic/binary_moderate_all__diagnostic]:se_ratio` | 0.9982 | its marginal SE ratio |
| `properties[weak_margin_diagnostic/binary_moderate_all__diagnostic]:unstratified_se_ratio` | 0.9957 | the unstratified fit's SE ratio on the same draws |
| `bound:unweighted_s0_standard_error` | 0.0129 | the efficient SD in stratum 0, bounded rows only, at n = 1,000 |
| `bound:unweighted_s1_standard_error` | 0.0105 | the efficient SD in stratum 1, bounded rows only, at n = 1,000 |
| `bound:unweighted_s2_standard_error` | 0.0088 | the efficient SD in stratum 2, bounded rows only, at n = 1,000 |
| `bound:unweighted_all_standard_error` | 0.0077 | the efficient SD of the marginal value, bounded rows only, at n = 1,000 |
| `bound:weighted_s0_standard_error` | 0.0136 | the efficient SD in stratum 0, bounded weighted, at n = 1,000 |
| `bound:weighted_s1_standard_error` | 0.0113 | the efficient SD in stratum 1, bounded weighted, at n = 1,000 |
| `bound:weighted_s2_standard_error` | 0.0100 | the efficient SD in stratum 2, bounded weighted, at n = 1,000 |
| `bound:weighted_all_standard_error` | 0.0085 | the efficient SD of the marginal value, bounded weighted, at n = 1,000 |
| `bound:binary_steep_s0_standard_error` | 0.0456 | the efficient SD in stratum 0, binary rows only, at n = 1,000 |
| `bound:binary_steep_s1_standard_error` | 0.0362 | the efficient SD in stratum 1, binary rows only, at n = 1,000 |
| `bound:binary_steep_s2_standard_error` | 0.0297 | the efficient SD in stratum 2, binary rows only, at n = 1,000 |
| `bound:binary_steep_all_standard_error` | 0.0209 | the efficient SD of the marginal value, binary rows only, at n = 1,000 |
| `bound:binary_steep_weighted_s0_standard_error` | 0.0483 | the efficient SD in stratum 0, binary weighted, at n = 1,000 |
| `bound:binary_steep_weighted_s1_standard_error` | 0.0386 | the efficient SD in stratum 1, binary weighted, at n = 1,000 |
| `bound:binary_steep_weighted_s2_standard_error` | 0.0329 | the efficient SD in stratum 2, binary weighted, at n = 1,000 |
| `bound:binary_steep_weighted_all_standard_error` | 0.0227 | the efficient SD of the marginal value, binary weighted, at n = 1,000 |
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
| `margin:targeting_displacement` | 0.5000 | least displacement the targeting pair must show, in positive-arm SDs |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 20 | failed primary fits that bound allows per setting |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No pinned comparator reports a fold-averaged stratified learned-rule value. The row rests on accuracy against the truth and on the theory properties |
| One size, five folds and one partition | The study fits n = 1,000 with 5 folds and three strata whose smallest share is 0.2. The support key holds neither the fold count nor the shares. More folds, smaller strata or another size were not measured |
| A margin at the rule boundary | The blip has slope 1.5 on the bounded outcome and 3.0 on the logit scale on the binary one, and the fold rules disagreed on about 8% and 14% of the boundary stratum's rows. With a weaker margin the interval under-covers: at the blip $0.1 - 0.3 W_1$ the diagnostic cell covers 0.915 to 0.939 in every stratum and marginally, and the unstratified fit under-covers alike on the same draws. That is the weak-margin limit of the learned-rule value ([F27](../../roadmap.md#f27-learned-policy-value-outside-the-published-conditions), the `weak_blip` row of the [boundary study](learned-rule-cvtmle-boundary.md)), not of the stratified construction |
| The boundary stratum leaves calibration first | At the blip $-0.6 + 1.5 W_1$ the boundary stratum reads an SE ratio of 0.968 while the other strata are calibrated. No verdict reads it |
| The boundary never lies in the smallest stratum | It lies in a stratum of share 0.35 (bounded) or 0.45 (binary). A boundary in a stratum of share 0.2 was not measured |
| A data-adaptive target | The boundary stratum's value varies with the learned rules, and a stratum the rules treat alike has a fixed target. The interval covers the value of the rules the folds learned, not the value of the true rule |
| The plug-in rule only | The study fits `LearnedRule()` with its plug-in rule. A rule-learner loss with strata matches no row and reads experimental |
| Pointwise intervals and `repeats=1` only | The study fits `simultaneous=False`. The band and `repeats` above 1 are not measured and read experimental |
| Rows, complete data and pooled targeting | Fold targeting and clusters with strata read experimental or refuse. Estimated weights are not measured; the weighted settings use the known $w = 1 + 1.5 W_2$ |
| Correctly specified parametric learners | The outcome and treatment learners contain the truth on the logit scale. A flexible learner rests on the theory alone |
| The share term is tested on the bounded outcome only | Its asymptotic SE ratio without it is 0.98 on the binary outcome, which the band cannot resolve |
| The construction controls resolve gross errors only | The unnormalised-curve control fails by construction, and the fold-local share against the whole-sample share rests on the exact-law tests. The targeting control is stratum-blind, so it shows repair of an initial fit that ignores the strata, not of a subtle misspecification within a stratum |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_strata/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_strata/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_strata/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_strata/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_strata/strata-harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_strata/properties.csv)
carry every published row.
