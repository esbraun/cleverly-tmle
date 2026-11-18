# A treatment at several nodes with covariates after baseline on the hazard path

This study validates longitudinal TMLE with `survival_model="hazard"` and
`targeting="one_step"` when a binary treatment is decided at each of three nodes and a binary
covariate $L_k$ is measured before the treatment at every node after the first. Each regimen is
targeted by one backward pass. At each node the pass fluctuates the node's event hazard together
with a regression of the later events on the node's history, and it reads the hazard at the
observed history with only the current treatment set. The section on
[a treatment at several nodes with covariates after baseline](../longitudinal-tmle.md#a-treatment-at-several-nodes-with-covariates-after-baseline)
states the estimand, the slots and the exact-law evidence.

The declared run at commit `10d19ab`, regenerated at `6fe1bac` under the shipped intervals, passes every gated row: 21 of 21 primary tests against
known truth and 46 of 46 property rows with a verdict. The nine agreement rows with the
iterated-mean fit are reported with no verdict. One cross-fitted calibration fit stopped short
of `tol` and was recorded as a lost replicate, inside the failure bound of 0.01
([the lost replicate](#the-lost-replicate)). No other fit failed or held a cell. The support
rows `G63.multinode.in_sample` and `G63.multinode.cross_fitted` license the fits it covers.

## The estimand and the law

For a regimen $r$ the target is $\psi_r(t) = E\,Q^r_1(H_1)$, the risk of an event by node $t$
when each node's treatment follows $r$. The regimens are always treating, never treating, a rule
(treat when $W_1 = 1$ at node 1 and when $L_k = +1$ after) and a known policy,
$q_1(1 \mid W_1) = 0.3 + 0.4\,[W_1 = 1]$ and
$q_k(1 \mid L_k, A_{k-1}) = 0.05 + 0.3\,[L_k = +1] + 0.6\,A_{k-1}$ after. A fit reports 21
estimates: each regimen's risk at each node and each of always, the rule and the policy against
never at each node. Identification needs sequential randomization of each $A_k$ given the
history, censoring at random given the history, and positivity of each regimen (Díaz, Williams,
Hoffman and Schenck 2023, Theorem 1).

$W_1 \in \{0, 1\}$ and $W_2 \in \{0, \dots, 3\}$ are independent and uniform. At node $k \ge 2$
the covariate $L_k$ is drawn first, logistic in $W_1$, $A_{k-1}$ and $L_{k-1}$. The treatment is
logistic in $W$ at node 1 and in $(W_1, A_{k-1}, L_k)$ after, dropout is logistic in
$(W, A_k, L_k)$, and the event hazard of a unit still followed is logistic in
$(k, W, A_{k-1}, A_k, L_k)$. The hazard reads the earlier treatment, so a slot that set it to
the current level, or read the treatment held from node 1, would miss. The law is Markov, so a
logistic regression on $(W_1, W_2, L_k, A_{k-1}, A_k)$ is a correctly specified hazard without
being saturated. Its coefficients are in `tests/studies/hazard_time_varying_multinode_law.py`.

| reading | value | requirement |
| --- | --- | --- |
| treatment probability, every node and history | 0.198 to 0.786 | within $[0.15, 0.85]$ |
| smallest cumulative mechanism over every treatment and covariate path | 0.0136 | above the default bound of 0.01 |
| event hazard, every node and history | 0.0033 to 0.940 | inside the fit's clip $[5 \times 10^{-4}, 0.9995]$ |
| smallest follower mass at node 3 | 0.072 (never) | at least 0.05 |

Every truth is an exact sum over the law's 928 support points, and every efficiency bound is
the exact standard deviation of the efficient influence function over the same support. Neither
calls the estimator. `tests/unit/test_hazard_time_varying_multinode_study_design.py` checks the
truths against an independent forward sum over every path to $10^{-14}$. It also checks that
the exact-sample fit with the true learners returns every truth within $10^{-8}$, and that its
influence curve is the efficient influence function.

The pilot's `ic_gap` label reads each fit's influence curve against the exact efficient influence
function. Over the two fit labels, the median gap of five fits at n = 2,000 runs from 0.087
(`crossfit` risk) to 0.133 (`crossfit` policy), against a bound of 0.3. At n = 20,000 the gap lies
between 0.022 and 0.041, against a bound of 0.1, and between 0.20 and 0.35 times its median at
n = 2,000, against the halving rule's 0.5. At n = 40,000 each fit's influence-curve SD is between
0.932 (`crossfit` policy) and 1.019 (`multinode` policy) times the efficiency bound. One fit's
ratio has a sampling SD of about $\sqrt{(\kappa - 1)/(4n)}$, with $\kappa$ the kurtosis of the
efficient influence function, and each reading lies within three of those SDs: 0.18 on the policy
arms, 0.09 to 0.12 on the rest.

| cell | estimand, by node 3 | truth | efficiency bound | kurtosis $\kappa$ of the influence function |
| --- | --- | --- | --- | --- |
| `multinode_contrast` | always minus never | -0.7276 | 1.040 | 195 |
| `multinode_rule` | the rule minus never | -0.3192 | 1.019 | 142 |
| `multinode_policy` | the policy minus never | -0.3370 | 0.597 | 599 |
| `multinode_risk` | always | 0.1290 | 0.748 | 242 |

The influence functions are heavy-tailed: a unit that follows a rarely followed path carries a
large inverse-probability weight. The policy's $\kappa$ of 599 compares with 7 to 17 on the arms of
[the study with one baseline treatment](hazard-time-varying.md).

## The fits

`multinode` fits in sample and is the primary scenario. `crossfit` fits the same law with five
outer folds and carries calibration cells only, since a five-fold fit predicts each support
point out of fold and has no exact-sample limit. The correct hazard learner is the logistic
regression above. The correct continuation learner is the weighted cell mean over every column
of the continuation design but the earlier covariates. The correct treatment and dropout
learners are logistic in their parents. Each wrong learner drops $L_k$. The configurations
combine them.

| configuration | hazard | continuation | mechanism |
| --- | --- | --- | --- |
| `both_correct` | correct | correct | correct |
| `outcome_correct` | correct | correct | without $L_k$ |
| `mechanism_correct` | without $L$ | without $L$ | correct |
| `continuation_wrong` | correct | without $L$ | correct |
| `both_wrong` | without $L$ | without $L$ | without $L_k$ |

No maintained package targets discrete-time hazards with covariates after baseline by a
backward pass. R `ltmle` and `lmtp` estimate the same estimand by the iterated-mean algorithm,
which the in-package iterated-mean fit already ties to `ltmle` in `canonical-ltmle-survival`.
The study names no comparator, so no canonical implementation is compared.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[always vs never @ t=1]` | the risk by node 1 under treatment at every node minus the risk under no treatment | `cleverly` | -0.000948 to 0.000657 | 0.9517 | 1.0063 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[always vs never @ t=2]` | the risk by node 2 under treatment at every node minus the risk under no treatment | `cleverly` | -0.0010 to 0.0012 | 0.9513 | 1.0065 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[always vs never @ t=3]` | the risk by node 3 under treatment at every node minus the risk under no treatment | `cleverly` | -0.0014 to 0.0011 | 0.9421 | 0.9862 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[policy vs never @ t=1]` | the risk by node 1 under the known policy that treats at node 1 with probability 0.3 + 0.4 W1 and later with probability 0.05 + 0.3 [high covariate] + 0.6 A(k-1) minus the risk under no treatment | `cleverly` | -0.000463 to 0.000462 | 0.9513 | 1.0092 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[policy vs never @ t=2]` | the risk by node 2 under the known policy that treats at node 1 with probability 0.3 + 0.4 W1 and later with probability 0.05 + 0.3 [high covariate] + 0.6 A(k-1) minus the risk under no treatment | `cleverly` | -0.000577 to 0.000869 | 0.9504 | 1.0003 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[policy vs never @ t=3]` | the risk by node 3 under the known policy that treats at node 1 with probability 0.3 + 0.4 W1 and later with probability 0.05 + 0.3 [high covariate] + 0.6 A(k-1) minus the risk under no treatment | `cleverly` | -0.000697 to 0.000707 | 0.9450 | 0.9745 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[rule vs never @ t=1]` | the risk by node 1 under the known rule that treats at node 1 when W1 = 1 and later when the covariate is high minus the risk under no treatment | `cleverly` | -0.000526 to 0.000738 | 0.9463 | 1.0076 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[rule vs never @ t=2]` | the risk by node 2 under the known rule that treats at node 1 when W1 = 1 and later when the covariate is high minus the risk under no treatment | `cleverly` | -0.000845 to 0.0014 | 0.9454 | 0.9812 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `ate_regimen[rule vs never @ t=3]` | the risk by node 3 under the known rule that treats at node 1 when W1 = 1 and later when the covariate is high minus the risk under no treatment | `cleverly` | -0.0012 to 0.0011 | 0.9571 | 1.0051 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[always @ t=1]` | the risk of the event by node 1 under treatment at every node | `cleverly` | -0.000496 to 0.000130 | 0.9513 | 0.9984 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[always @ t=2]` | the risk of the event by node 2 under treatment at every node | `cleverly` | -0.000755 to 0.000230 | 0.9463 | 0.9768 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[always @ t=3]` | the risk of the event by node 3 under treatment at every node | `cleverly` | -0.0010 to 0.000761 | 0.9458 | 0.9862 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[never @ t=1]` | the risk of the event by node 1 under no treatment at any node | `cleverly` | -0.000769 to 0.000694 | 0.9542 | 1.0180 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[never @ t=2]` | the risk of the event by node 2 under no treatment at any node | `cleverly` | -0.0014 to 0.000676 | 0.9500 | 1.0035 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[never @ t=3]` | the risk of the event by node 3 under no treatment at any node | `cleverly` | -0.000827 to 0.000907 | 0.9358 | 0.9601 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[policy @ t=1]` | the risk of the event by node 1 under the known policy that treats at node 1 with probability 0.3 + 0.4 W1 and later with probability 0.05 + 0.3 [high covariate] + 0.6 A(k-1) | `cleverly` | -0.000440 to 0.000362 | 0.9529 | 1.0092 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[policy @ t=2]` | the risk of the event by node 2 under the known policy that treats at node 1 with probability 0.3 + 0.4 W1 and later with probability 0.05 + 0.3 [high covariate] + 0.6 A(k-1) | `cleverly` | -0.000734 to 0.000335 | 0.9467 | 1.0004 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[policy @ t=3]` | the risk of the event by node 3 under the known policy that treats at node 1 with probability 0.3 + 0.4 W1 and later with probability 0.05 + 0.3 [high covariate] + 0.6 A(k-1) | `cleverly` | -0.000531 to 0.000621 | 0.9454 | 0.9795 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[rule @ t=1]` | the risk of the event by node 1 under the known rule that treats at node 1 when W1 = 1 and later when the covariate is high | `cleverly` | -0.000456 to 0.000593 | 0.9487 | 0.9906 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[rule @ t=2]` | the risk of the event by node 2 under the known rule that treats at node 1 when W1 = 1 and later when the covariate is high | `cleverly` | -0.000877 to 0.000711 | 0.9521 | 1.0009 | pass |
| three nodes, one cause, two baseline covariates, a binary covariate before the treatment at nodes 2 and 3 that reads the previous covariate and treatment, a binary treatment at every node logistic in W1, the previous arm and the covariate, logistic dropout and event hazard that read the covariate and the last two arms | `risk_regimen[rule @ t=3]` | the risk of the event by node 3 under the known rule that treats at node 1 when W1 = 1 and later when the covariate is high | `cleverly` | -0.0010 to 0.0010 | 0.9575 | 1.0129 | pass |
<!-- /generated -->

The primary rows draw 2,400 samples of 2,000 rows from law (M) and report all 21 risks and
contrasts of the `multinode` fit. The shared primary rule (`repeated_sizing.primary_replicates`)
sets that count from the committed `multinode` calibration pilot, and at that pilot it gives
2,400.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `multinode_contrast__both_correct` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: correct logistic hazard, treatment and censoring learners and the correct cell-mean continuation | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0016 to 0.0018, margin 0.0059, SE ratio 0.9865 | pass |
| `double_robustness` | `multinode_contrast__both_wrong` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: hazard, continuation, treatment and censoring learners that all omit the covariate | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.1612 to 0.1662, margin 0.0084, SE ratio 0.9987 | pass |
| `double_robustness` | `multinode_contrast__continuation_wrong` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: a correct hazard, treatment and censoring learners with a continuation that omits the earlier covariates; it witnesses the covariate term of the panel | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0015 to 0.0018, margin 0.0057, SE ratio 1.0184 | pass |
| `double_robustness` | `multinode_contrast__mechanism_correct` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: a hazard and continuation that omit the covariate, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000230 to 0.0035, margin 0.0063, SE ratio 1.0448 | pass |
| `double_robustness` | `multinode_contrast__outcome_correct` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: a correct hazard and continuation with treatment and censoring learners that omit the covariate | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0019 to 0.0041, margin 0.0056, SE ratio 1.4075 | pass |
| `double_robustness` | `multinode_policy__both_correct` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: correct logistic hazard, treatment and censoring learners and the correct cell-mean continuation | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000928 to 0.0011, margin 0.0034, SE ratio 0.9582 | pass |
| `double_robustness` | `multinode_policy__continuation_wrong` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: a correct hazard, treatment and censoring learners with a continuation that omits the earlier covariates; it witnesses the covariate term of the panel | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000960 to 0.0010, margin 0.0033, SE ratio 0.9923 | pass |
| `double_robustness` | `multinode_policy__mechanism_correct` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: a hazard and continuation that omit the covariate, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000034 to 0.0022, margin 0.0038, SE ratio 1.0244 | pass |
| `double_robustness` | `multinode_policy__outcome_correct` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: a correct hazard and continuation with treatment and censoring learners that omit the covariate | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0012 to 0.0024, margin 0.0031, SE ratio 1.4775 | pass |
| `double_robustness` | `multinode_policy__outcome_correct_untargeted` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: the backward plug-in, with every targeting step removed, of a correct hazard and continuation with treatment and censoring learners that omit the covariate, on draws of 8,000 rows in place of 2,000; it carries the regimen's slots, which targeting would repair | bias interval inside the equivalence margin | bias -0.000198 to 0.000512, margin 0.0012, SE ratio 1.9072 | pass |
| `double_robustness` | `multinode_rule__both_correct` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known rule against never by node 3: correct logistic hazard, treatment and censoring learners and the correct cell-mean continuation | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0014 to 0.0021, margin 0.0059, SE ratio 0.9671 | pass |
| `double_robustness` | `multinode_rule__outcome_correct` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known rule against never by node 3: a correct hazard and continuation with treatment and censoring learners that omit the covariate | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0017 to 0.0038, margin 0.0055, SE ratio 1.7116 | pass |
| `double_robustness` | `multinode_rule__outcome_correct_untargeted` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known rule against never by node 3: the backward plug-in, with every targeting step removed, of a correct hazard and continuation with treatment and censoring learners that omit the covariate, on draws of 8,000 rows in place of 2,000; it carries the regimen's slots, which targeting would repair | bias interval inside the equivalence margin | bias -0.000542 to 0.000608, margin 0.0019, SE ratio 2.4660 | pass |
| `interval_calibration` | `crossfit_contrast__correctly_specified` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, five outer folds, n = 2,000, the risk difference of treating at every node against never by node 3: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9435 to 0.9527, SE ratio 0.9733 to 1.0022, empirical efficiency ratio 1.0035 to 1.0337, reported efficiency ratio 1.0029 to 1.0091 | pass |
| `interval_calibration` | `crossfit_contrast__noise_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, five outer folds, n = 2,000, the risk difference of treating at every node against never by node 3: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8191 to 0.8348, SE ratio 0.6949 to 0.7156, empirical efficiency ratio 1.4055 to 1.4479, reported efficiency ratio 1.0027 to 1.0090 | pass |
| `interval_calibration` | `crossfit_contrast__shrunken_se_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, five outer folds, n = 2,000, the risk difference of treating at every node against never by node 3: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8146 to 0.8304, SE ratio 0.6814 to 0.7019, empirical efficiency ratio 1.0027 to 1.0340, reported efficiency ratio 0.7020 to 0.7064 | pass |
| `interval_calibration` | `crossfit_policy__correctly_specified` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, five outer folds, n = 2,000, the risk difference of the known policy against never by node 3: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9433 to 0.9525, SE ratio 0.9691 to 0.9985, empirical efficiency ratio 0.9952 to 1.0271, reported efficiency ratio 0.9895 to 0.9990 | pass |
| `interval_calibration` | `crossfit_policy__noise_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, five outer folds, n = 2,000, the risk difference of the known policy against never by node 3: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8216 to 0.8372, SE ratio 0.6947 to 0.7154, empirical efficiency ratio 1.3897 to 1.4315, reported efficiency ratio 0.9899 to 0.9991 | pass |
| `interval_calibration` | `crossfit_policy__shrunken_se_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, five outer folds, n = 2,000, the risk difference of the known policy against never by node 3: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8178 to 0.8335, SE ratio 0.6784 to 0.6985, empirical efficiency ratio 0.9956 to 1.0266, reported efficiency ratio 0.6930 to 0.6993 | pass |
| `interval_calibration` | `multinode_contrast__correctly_specified` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9387 to 0.9483, SE ratio 0.9715 to 1.0003, empirical efficiency ratio 0.9888 to 1.0184, reported efficiency ratio 0.9863 to 0.9921 | pass |
| `interval_calibration` | `multinode_contrast__noise_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8138 to 0.8296, SE ratio 0.6871 to 0.7072, empirical efficiency ratio 1.3987 to 1.4396, reported efficiency ratio 0.9864 to 0.9922 | pass |
| `interval_calibration` | `multinode_contrast__shrunken_se_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8153 to 0.8311, SE ratio 0.6801 to 0.7004, empirical efficiency ratio 0.9887 to 1.0183, reported efficiency ratio 0.6904 to 0.6945 | pass |
| `interval_calibration` | `multinode_policy__correctly_specified` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9417 to 0.9510, SE ratio 0.9615 to 0.9905, empirical efficiency ratio 0.9846 to 1.0157, reported efficiency ratio 0.9717 to 0.9801 | pass |
| `interval_calibration` | `multinode_policy__noise_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8112 to 0.8271, SE ratio 0.6778 to 0.6978, empirical efficiency ratio 1.3980 to 1.4392, reported efficiency ratio 0.9715 to 0.9801 | pass |
| `interval_calibration` | `multinode_policy__shrunken_se_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8142 to 0.8300, SE ratio 0.6730 to 0.6932, empirical efficiency ratio 0.9848 to 1.0157, reported efficiency ratio 0.6801 to 0.6860 | pass |
| `interval_calibration` | `multinode_risk__correctly_specified` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk under treatment at every node by node 3: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9372 to 0.9469, SE ratio 0.9668 to 0.9959, empirical efficiency ratio 0.9910 to 1.0216, reported efficiency ratio 0.9838 to 0.9906 | pass |
| `interval_calibration` | `multinode_risk__noise_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk under treatment at every node by node 3: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8148 to 0.8306, SE ratio 0.6838 to 0.7042, empirical efficiency ratio 1.4018 to 1.4432, reported efficiency ratio 0.9837 to 0.9907 | pass |
| `interval_calibration` | `multinode_risk__shrunken_se_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk under treatment at every node by node 3: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8114 to 0.8274, SE ratio 0.6763 to 0.6974, empirical efficiency ratio 0.9906 to 1.0219, reported efficiency ratio 0.6887 to 0.6934 | pass |
| `interval_calibration` | `multinode_rule__correctly_specified` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known rule against never by node 3: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9447 to 0.9538, SE ratio 0.9792 to 1.0083, empirical efficiency ratio 0.9840 to 1.0139, reported efficiency ratio 0.9901 to 0.9948 | pass |
| `interval_calibration` | `multinode_rule__noise_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known rule against never by node 3: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8194 to 0.8350, SE ratio 0.6925 to 0.7124, empirical efficiency ratio 1.3934 to 1.4334, reported efficiency ratio 0.9901 to 0.9948 | pass |
| `interval_calibration` | `multinode_rule__shrunken_se_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known rule against never by node 3: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8180 to 0.8337, SE ratio 0.6855 to 0.7059, empirical efficiency ratio 0.9841 to 1.0138, reported efficiency ratio 0.6931 to 0.6964 | pass |
| `iterated_mean_difference` | `multinode__iterated_curve_decrease` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, beside the iterated-mean fit of the same regimens on the same draw: the largest decrease of the iterated-mean risk curve of either arm between consecutive visits | no verdict; the share of draws whose curve decreases somewhere is reported | decreasing share 0, 99% interval 0 to 0.0066 | reported |
| `iterated_mean_difference` | `multinode_contrast__hazard_fit` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9092 to 0.9554, SE ratio 0.9423 | reported |
| `iterated_mean_difference` | `multinode_contrast__isotonic_difference` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.4092, mean difference -0.0050 to -0.0033 | reported |
| `iterated_mean_difference` | `multinode_contrast__iterated_fit` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.8752 to 0.9298, SE ratio 0.8710 | reported |
| `iterated_mean_difference` | `multinode_contrast__paired_difference` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.4092, mean difference -0.0050 to -0.0033 | reported |
| `iterated_mean_difference` | `multinode_policy__hazard_fit` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9020 to 0.9502, SE ratio 0.9348 | reported |
| `iterated_mean_difference` | `multinode_policy__isotonic_difference` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.4928, mean difference -0.0038 to -0.0026 | reported |
| `iterated_mean_difference` | `multinode_policy__iterated_fit` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.8530 to 0.9123, SE ratio 0.8297 | reported |
| `iterated_mean_difference` | `multinode_policy__paired_difference` | diagnostic | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.4928, mean difference -0.0038 to -0.0026 | reported |
| `power` | `multinode_contrast__alternative` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9997 to 1 | pass |
| `root_n_and_efficiency` | `multinode_contrast__n_1000` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: bias, coverage and SE calibration at n = 1,000, a rate rung that carries no claim | coverage interval lies below nominal or clears the declared floor | bias -0.000344, coverage 0.9297 to 0.9698, SE ratio 0.9977 | pass |
| `root_n_and_efficiency` | `multinode_contrast__n_2000` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000315, coverage 0.9371 to 0.9747, SE ratio 1.0006 | pass |
| `root_n_and_efficiency` | `multinode_contrast__n_8000` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000221, coverage 0.9179 to 0.9616, SE ratio 1.0143 | pass |
| `root_n_and_efficiency` | `multinode_policy__n_1000` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: bias, coverage and SE calibration at n = 1,000, a rate rung that carries no claim | coverage interval lies below nominal or clears the declared floor | bias -0.000308, coverage 0.9311 to 0.9708, SE ratio 0.9952 | pass |
| `root_n_and_efficiency` | `multinode_policy__n_2000` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000051, coverage 0.9297 to 0.9698, SE ratio 0.9797 | pass |
| `root_n_and_efficiency` | `multinode_policy__n_8000` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000163, coverage 0.9238 to 0.9657, SE ratio 0.9947 | pass |
| `root_n_rate` | `multinode_contrast__empirical_sd` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5427 to -0.4559 | pass |
| `root_n_rate` | `multinode_contrast__reported_se` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4985 to -0.4828 | pass |
| `root_n_rate` | `multinode_policy__empirical_sd` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5266 to -0.4380 | pass |
| `root_n_rate` | `multinode_policy__reported_se` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of the known policy against never by node 3: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4916 to -0.4699 | pass |
| `simultaneous_coverage` | `multinode_curve__pointwise_joint_control` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, every risk and risk difference of every regimen read as one joint set: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5488 to 0.5694 | pass |
| `simultaneous_coverage` | `multinode_curve__simultaneous_band` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, every risk and risk difference of every regimen read as one joint set: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9365 to 0.9463 | pass |
| `targeting_necessity` | `multinode_contrast__targeted` | positive | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the estimator with a hazard and continuation that omit the covariate and correct treatment and censoring learners, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000230 to 0.0035, margin 0.0063 | pass |
| `targeting_necessity` | `multinode_contrast__untargeted` | control | the three-node law with a binary covariate before the treatment at nodes 2 and 3, n = 2,000, the risk difference of treating at every node against never by node 3: the same fit's backward plug-in with every targeting step removed | bias interval must fall entirely outside the margin | bias 0.1776 to 0.1812, margin 0.0060 | pass |
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | six arms (`multinode` contrast, rule, policy and risk; `crossfit` contrast and policy), each with the shrunken-SE and noise controls, which must fail | 15,600 per fit label at n = 2,000 | an SE ratio or coverage interval outside its band, or an efficiency ratio outside its band |
| `double_robustness` | the contrast under all five configurations, `both_wrong` as the control; the rule under `both_correct` and `outcome_correct`; the policy under every configuration but `both_wrong` | at n = 2,000: 2,800 for `outcome_correct`, 1,200 for each other configuration | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `double_robustness`, untargeted | the untargeted rule and policy contrasts of `outcome_correct` draws of their own | 1,200 at n = 8,000 | a bias interval outside the margin |
| `targeting_necessity` | the targeted and the untargeted contrast of the `mechanism_correct` draws | the robustness fits | the targeted bias outside the margin, or the plug-in inside it |
| `root_n_and_efficiency`, `root_n_rate` | the contrast and the policy at n = 1,000, 2,000 and 8,000; the n = 1,000 rung carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the contrast against zero | the `multinode` calibration fits | a rejection lower bound below the minimum power |
| `simultaneous_coverage` | the band over the 21 estimates and its pointwise control | the `multinode` calibration fits | joint coverage outside the calibration band |
| every family | `nonconverged_fits`: the fits that stopped short of `tol` with a final relative score at or below $10^{-6}$, beside each cell's count of unexpected warnings | as the cell | a non-converged fit above $10^{-6}$ is a lost replicate under the 0.01 failure bound; more than 1% non-converged fits makes the cell a defect suspect |
| `iterated_mean_difference` | the in-package iterated-mean TMLE without and with `isotonic=True`, each fit against the truth, their paired difference from the hazard fit, and the largest decrease of the iterated-mean curve | 800 at n = 2,000 | nothing: the family is reported with no verdict |

Every gated cell reads the one-step hazard TMLE. A draw whose fit holds a cell is a recorded
lost replicate, charged to the failure bound of 0.01, and no draw of the run held one. Every kept
fit converged to `tol` = $10^{-10}$ with no unexpected warning. The necessity cells read the
`mechanism_correct` robustness fits, and the power and band cells the `multinode` calibration
fits, so none adds a fit set. The targeted necessity cell therefore restates the
`mechanism_correct` robustness contrast on the same draws, and the untargeted plug-in control is
the cell that adds evidence.

The controls fail as declared. The shrunken-SE controls read SE ratios of 0.68 to 0.70 and the
noise controls efficiency ratios of 1.41 to 1.43, against bands of $[0.93, 1.07]$ and
$[0.9, 1.1]$. The `both_wrong` contrast sits 4.90 standard deviations from the truth and the
untargeted `mechanism_correct` plug-in 7.43, against a margin of 0.25. The pointwise intervals
covered all 21 estimates at once in 55.7% of the calibration draws, where the band covered them
in 94.2%.

The targeted `outcome_correct` arms read standardized biases of 0.12 to 0.15 at n = 2,000,
inside the margin of 0.25 at their 2,800 replicates. Their SE ratios of 1.41 to 1.71 sit inside
the union band: with the mechanism wrong the influence curve overstates the variance, and these
arms cover 99.3% to 99.9%. The untargeted `outcome_correct` arms at n = 8,000 read 0.004 (rule)
and 0.033 (policy).

### The lost replicate

Replicate 14,254 of the `crossfit` calibration fit set ran a hazard walk that stopped short of
`tol` with a final relative score of 0.00887. That is above the $10^{-6}$ a fit may keep, so the
row check recorded the fit as a failure (`property-failures.csv`), not a converged estimate. The
fit writes six cells, the `crossfit` contrast and policy and their shrunken-SE and noise
controls, and each lost that one replicate: 15,599 of 15,600 kept, against an allowance of 156
at the bound of 0.01.

`property-failure-accounting.csv` refills the lost value with every candidate each clause of
the six cells declares: the range of the surviving draws for an SE or efficiency ratio, and both
values of the coverage indicator. Every clause passes in the worst case, so the lost draw could
not have turned a verdict. The draw's witness, the largest absolute standardized value in the
sample, is 9.07 and sits at the 0.50 quantile of the surviving draws, so nothing in the data
marks it as extreme. One stalled walk in 15,600 cross-fitted fits is 0.006%, far below the 1%
at which non-converged fits would make a cell a defect suspect.

### Agreement with the iterated-mean fit

Over 800 draws, the SD of the paired difference between the hazard fit and the iterated-mean fit,
over the efficiency-bound standard error, is 0.409 for the contrast and 0.493 for the policy. The
hazard fit reads standardized biases of 0.002 and -0.014 on these arms, and the iterated-mean fit
0.165 and 0.199. No iterated-mean curve decreased (99% interval for the share 0 to 0.007), so
`isotonic=True` changed no estimate. The study reports these readings with no floor, as plan
X70 asks.

## The pilot

The pilot (`tests/studies/hazard_time_varying_multinode_pilot.py`) draws from its own seed
streams and writes `tests/canonical/hazard_time_varying_multinode/pilot.csv.gz`. Every fit of
every label converged to `tol` = $10^{-10}$, no draw was refused or held, and no fit raised a
warning outside the expected `SparseInfluenceWarning`. The rows record each fit's accepted steps
over every node walk of every regimen, but not its time. The seconds below are each label's wall
time on the runner's 4 cores, times 4, over its draws; the n = 8,000 label's figure is one
thread's time for one fit.

| pilot label | draws, n | walk steps: mean (most) | core-seconds a draw |
| --- | --- | --- | --- |
| `calibration_multinode` (with the band) | 8,000, 2,000 | 1,407 (12,746) | 1.15 |
| `calibration_crossfit` | 8,000, 2,000 | 1,727 (8,930) | 1.53 |
| `wrong_learner_outcome_correct` | 800, 2,000 | 1,660 (3,571) | 1.22 |
| `wrong_learner_mechanism_correct` | 800, 2,000 | 5,230 (8,718) | 2.42 |
| `wrong_learner_continuation_wrong` | 800, 2,000 | 1,532 (4,408) | 1.26 |
| `wrong_learner_both_wrong` | 800, 2,000 | 2,182 (4,380) | 1.49 |
| `untargeted_outcome_correct_n8000` | 800, 8,000 | none: no walk | 0.41 |
| `fits_differ_multinode` | 200, 2,000 | 1,402 (3,312) | 1.54 |

A `wrong_learner` draw's time includes its untargeted pass, and a `fits_differ` draw's time its
two iterated-mean fits. The 14 `ic_gap` fits took 160 core-seconds. Node 1's walk moved its
estimates by a mean of 0.19 (`multinode`) and 0.21 (`crossfit`) of their standard errors, at most
2.1. The smallest Kish count of a calibration fit's influence curves had a median of 14 to 15, and its
largest predicted Wald coverage loss had median 0.009.

### Calibration

The calibration screen asks of each arm that its pilot SE ratio, mean SE over the SD of the
estimates, sit near the ratio a correct variance estimator would read. That ratio is not one
here. The reported SE is $\hat\sigma/\sqrt n$ with $\hat\sigma$ the SD of the influence curve, and
for a heavy-tailed curve $E\hat\sigma$ falls below $\sigma$ while $E\hat\sigma^2 = \sigma^2$. An
exact simulation of the law (40,000 multinomial samples of n = 2,000 from the support, scored
with the exact efficient influence function, no fit; `expected_se_ratio`) gives $E\hat\sigma/\sigma$
of 0.990, 0.994, 0.978 and 0.987 on the contrast, rule, policy and risk. The first-order
$1 - (\kappa - 1)/(8n)$ overstates the policy's deficit, at 0.963. The screen statistic also
divides by the estimates' SD, which the pilot reads at 1.003 to 1.026 times the bound, a
second-order excess the law alone does not give.

| arm | mean SE / SD | mean SE / bound | SD / bound | expected $E\hat\sigma/\sigma$ | RMS SE / SD | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| `multinode_contrast` | 0.980 | 0.989 | 1.010 | 0.990 | 0.990 | 0.9458 |
| `multinode_rule` | 0.987 | 0.992 | 1.006 | 0.994 | 0.993 | 0.9473 |
| `multinode_policy` | 0.964 | 0.976 | 1.012 | 0.978 | 0.985 | 0.9449 |
| `multinode_risk` | 0.984 | 0.987 | 1.003 | 0.987 | 0.998 | 0.9465 |
| `crossfit_contrast` | 0.997 | 1.007 | 1.011 | 0.990 | 1.008 | 0.9438 |
| `crossfit_policy` | 0.970 | 0.996 | 1.026 | 0.978 | 0.994 | 0.9415 |

Of the `multinode` policy arm's deficit of 0.036, about 0.022 is the heavy-tail term and about
0.012 the SD excess. The screen is centred on each arm's expected ratio and counts the excess as
departure. Every arm lies within the screen's half-width of 0.032 of its centre, the largest
departure being -0.014 on the `multinode` policy, and the mean departure, -0.006, is inside the
off-mean bound of 0.015. Each arm's root-mean-square SE over the SD, which reads
$E[SE^2] = \mathrm{Var}\,\hat\psi$ with no approximation, lies within the same half-width of one.

The arms' expected ratios differ, so they are not pooled: the calibration count is the per-arm
rule of `mtp_study_tools.per_arm_replicates` at each arm's pilot ratio. It gives 15,600
replicates, with a smallest per-cell pass probability of 0.961 (the `multinode` policy) and a
family-wise probability of 0.951. At that count the shrunken-SE control leaves the SE-ratio band,
and the noise control the efficiency band, with probability above 0.999. A red calibration row
takes the finite-sample route only when the 99% interval of its SE ratio contains the arm's
expected ratio in the table, not 1 (`regenerate.py`), together with the coverage and diagnosis
conditions of the route.

The band covered all 21 estimates in 94.54% of the `multinode` pilot draws, with a standard
error of 0.25 points, and the pointwise intervals covered all at once in 56.2%. At the pilot's
joint coverage minus one standard error, the 99% interval of the declared cell at 15,600
replicates lies inside the band with probability above 0.9999, against the 0.95 the rule
requires. The shared primary rule gives 2,400 replicates at this pilot: the lowest pointwise
coverage is 0.9431 (never's risk by node 3), and the largest standardized bias 0.029 (the rule
against never by node 3).

### Robustness arms

Each wrong-learner configuration's count is the larger of 1,200 and the smallest multiple of 400
at which every targeted positive arm of the configuration passes with probability 0.99, read at
the arm's pilot standardized bias plus one standard error. The rule was fixed before the pilot was
read and stops the study above 4,000.

| configuration | arm readings: contrast, rule, policy | replicates |
| --- | --- | --- |
| `outcome_correct` | 0.105, 0.151, 0.134 | 2,800 |
| `mechanism_correct` | 0.108, none, 0.079 | 1,200 |
| `continuation_wrong` | 0.047, none, 0.036 | 1,200 |
| `both_wrong` | the control: 4.99 standard deviations, 7.02 efficiency SEs | 1,200 |

The `both_correct` arms read at most 0.040 in the calibration pilot. The untargeted
`mechanism_correct` plug-in, the necessity control, sits 7.35 standard deviations from the truth,
and the untargeted `mechanism_correct` and `continuation_wrong` plug-ins centre on their
exact-sample limits within 1.6 of their standard errors.

At n = 2,000 the untargeted `outcome_correct` rule and policy contrasts read standardized biases
of 0.267 and 0.254, or 0.302 and 0.289 with one standard error, above the margin of 0.25, so no
count would pass them there. Their limit is the truth, and the bias is finite-sample. On fresh
scratch draws it falls from 0.150 efficiency SEs at n = 2,000 to 0.023 at 8,000 and 0.009 at
32,000 on the rule contrast.

The fitted hazard does not carry the bias, since the law's own hazard in its
place leaves it unchanged. It comes from the continuation learner, a cell mean that answers a
history cell it has not seen with the mean over all its cells. The fits of never and of the
policy predict at cells that their followers rarely reach, such as an untreated unit with
$W_1 = 1$, so that fallback moves never's estimate in 19 of 20 draws at n = 2,000 and in 6 of 20
at n = 8,000. The always and rule risks do not move. The contrasts against never inherit the
bias, and targeting reduces it, to about 0.1 of the SD on the targeted `outcome_correct` arms. The
untargeted arms therefore run at n = 8,000, where the pilot reads 0.073 and 0.077 with one
standard error, so they take 1,200 replicates.

### The pilot's agreement reading

Over 200 draws, the SD of the paired difference between the hazard fit and the iterated-mean fit,
over the efficiency-bound standard error, is 0.399 for the contrast, 0.438 for the rule, 0.481 for
the policy and 0.230 for always's risk. No iterated-mean curve decreased (99% interval for the
share 0 to 0.026), so `isotonic=True` changed no estimate and its ratios equal these. The study
reports these readings with no floor.

## What each component is checked by

Exact-law checks cannot see a term that vanishes at the truth. The design test computes, before
any run, the exact limit that each such component moves. It also computes the probability that
the arm carrying the witness tells the mutation from the truth at the arm's 1,200 replicates,
by `sizing.bias_discriminated_probability` at a Monte Carlo spread of 1 to 1.5 times the bound.
Each gap is in efficiency SEs at the arm's n: 8,000 for the untargeted `outcome_correct` arms and
2,000 for the rest. An exact limit does not depend on n, so a gap at 8,000 is twice its value at
2,000.

| component | witness | gap | floor | discriminated |
| --- | --- | --- | --- | --- |
| the current-level slot | always minus never, from zero | -31.3 | 5 | 1.00 |
| the policy mixture at nodes 2 and 3 | the observed treatment in place of the mixture, `outcome_correct` policy contrast (policy risk 4.36) | 3.50 | 3 | 1.00 |
| the level plan | every node set to the slot's level, untargeted `outcome_correct` rule contrast | -8.37 | 3 | 1.00 |
| the held slot | the hazard at the treatment of node 1, untargeted `outcome_correct` policy contrast (untargeted rule 2.99, targeted policy 1.33) | 3.90 | 3 | 1.00 |
| the policy's prefix | the policy read at the level path, untargeted `outcome_correct` policy contrast (targeted 1.14) | 2.29 | 3, or 1 recorded | 1.00 |
| the hazard term and the walk | the untargeted `mechanism_correct` contrast | 7.68 | 3 | 1.00 |
| the covariate term of the panel | the untargeted `continuation_wrong` contrast | 0.80 | 1 | 1.00 |
| $L_k$ in the mechanism | the targeted `both_wrong` contrast | 7.03 | discriminated | 1.00 |
| $L_k$ as a confounder of later treatment | always and never fitted with $L$ dropped | 7.03 | 3 | 1.00 |

At a static or rule node the backward pass fits the slot hazard to the followers' events, so it
repairs a wrong slot under any mechanism. The targeted rule contrast therefore moves only 0.11
SEs under the level plan and -0.19 under the held slot, and is discriminated with probability 0.
The untargeted pass has no such repair, so the untargeted `outcome_correct` arms carry the rule's
slots. A defect of a policy slot changes the mixture the target averages, which no fit to the
events repairs, so the targeted policy contrast shows it too: 0.86 SEs under the level plan,
discriminated with probability 1.00.

The policy's prefix reaches its 1 SE fallback, the largest gap found inside the positivity,
hazard-clip and held-cell limits; the held slot meets 3. The covariate term is the one declared
shortfall, 0.80 SEs against a floor of 1. The current-level slot has no witness that reads the
observed treatment on a static arm, where the regimen's mixture is its level, so the contrast's
distance from zero stands in for it. Each component also keeps its deliberate-mutation control
in `tests/unit/test_hazard_time_varying_multinode.py`.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 2000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 21 | truth tests passing |
| `independent_tests_total` | 21 | truth tests reported |
| `property_cells_passed` | 46 | property cells passing |
| `property_cells_total` | 46 | property cells reported |
| `max_standardized_bias` | 0.0308 | largest primary standardized bias |
| `min_coverage` | 0.9358 | lowest primary coverage |
| `properties[interval_calibration/multinode_contrast__correctly_specified]:se_ratio` | 0.9859 | SE ratio of the `multinode` contrast's calibration cell |
| `properties[interval_calibration/multinode_policy__correctly_specified]:se_ratio` | 0.9756 | SE ratio of the `multinode` policy's calibration cell |
| `properties[interval_calibration/crossfit_contrast__correctly_specified]:se_ratio` | 0.9877 | SE ratio of the cross-fitted contrast calibration cell |
| `properties[interval_calibration/crossfit_contrast__correctly_specified]:failed_replicates` | 1 | lost replicates of the cross-fitted contrast calibration cell |
| `properties[double_robustness/multinode_contrast__both_wrong]:standardized_bias` | 4.9019 | standardized bias of the both-wrong control |
| `properties[double_robustness/multinode_contrast__outcome_correct]:standardized_bias` | 0.1333 | standardized bias of the `outcome_correct` contrast |
| `properties[targeting_necessity/multinode_contrast__untargeted]:standardized_bias` | 7.4250 | standardized bias of the plug-in control |
| `properties[simultaneous_coverage/multinode_curve__simultaneous_band]:coverage` | 0.9415 | joint coverage of the band |
| `properties[iterated_mean_difference/multinode_contrast__paired_difference]:paired_sd_over_bound` | 0.4092 | paired ratio of the contrast |
| `properties[iterated_mean_difference/multinode_policy__paired_difference]:paired_sd_over_bound` | 0.4928 | paired ratio of the policy |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 24 | the same bound, in primary replications |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |

## Limitations

| limit | what it means for use |
| --- | --- |
| one cause | the competing-risk path rests on the two-cause exact-law tests of `tests/unit/test_hazard_time_varying_multinode.py` |
| one binary Markov $L_k$ and a binary treatment at every node | a continuous, multivariate or non-Markov $L_k$, and unequal level counts across nodes, rest on the unit tests |
| no observation weights, clusters or `node_bounds=` | the unit tests fit weighted data, since every exact-law fit weights the support by its mass; clusters and `node_bounds=` are not measured on this design |
| declared factors and `markov_order=` | covered by the unit tests only |
| `targeting="one_step"` only | the per-horizon algorithm is refused on this design (H28) |
| static arms, a known rule and a known policy | a modified treatment policy or an odds tilt of a declared mechanism at any node is refused on this design (H27) |
| the witnesses below their floor | the policy's prefix sits between 1 and 3 SEs, and the covariate term at 0.80 SEs; each also rests on its unit-test mutation control |
| the untargeted arms at n = 8,000 | at n = 2,000 the untargeted `outcome_correct` plug-in carries a finite-sample bias of about a quarter of its SD, from the study's continuation learner (see [the pilot](#the-pilot)); the arms that read it run at n = 8,000 |
| support rows | the support key's `later_covariates` option on the unheld treatment axis keeps these fits off `G62` and `G63.held`. `G63.multinode.in_sample` licenses the in-sample fit and its band, and `G63.multinode.cross_fitted` the five-fold fit without a band |
| no comparator | R `ltmle` and `lmtp` reach this estimand by the iterated-mean algorithm only, which `canonical-ltmle-survival` already ties to the in-package fit, so they would add transitive evidence only |

## Reproduction

The smoke gate is `python -m tests.studies.canonical_hazard_time_varying_multinode`, which runs
`SMOKE_GATE()` on the pilot stream and writes
`tests/canonical/hazard_time_varying_multinode/smoke.json`. It fits two draws of every
configuration at n = 2,000, the five-fold fit, the n = 1,000 and n = 8,000 rungs, and the two
iterated-mean fits. It also fits the untargeted `mechanism_correct` pass at n = 2,000 and the
untargeted `outcome_correct` pass at n = 8,000. Every smoke fit must converge with every node score at most
$10^{-10}$, hold no cell and take at most twice its entry in `COST_SECONDS`: 1.3 s for an in-sample
configuration, 2.5 s for `mechanism_correct`, 1.5 s for `both_wrong`, 1.8 s for five folds, 1.0
and 4.5 s for the rungs, 0.4 s for an untargeted fit and 0.5 s for an iterated-mean fit. The
design test reads the committed record.

The pilot runs one label per run, as
`python -m tests.studies.hazard_time_varying_multinode_pilot --jobs N --labels LABEL`, and writes
`tests/canonical/hazard_time_varying_multinode/pilot.csv.gz`. The declared run is
`python tests/canonical/hazard_time_varying_multinode/regenerate.py --output DIR`, with an empty
scratch directory outside the repository.
