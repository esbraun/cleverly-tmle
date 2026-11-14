# A treatment at several nodes with covariates after baseline on the hazard path

This study is written to validate longitudinal TMLE with `survival_model="hazard"` and
`targeting="one_step"` when a binary treatment is decided at each of three nodes and a binary
covariate $L_k$ is measured before the treatment at every node after the first. Each regimen is
targeted by one backward pass. At each node the pass fluctuates the node's event hazard together
with a regression of the later events on the node's history, and it reads the hazard at the
observed history with only the current treatment set. The section on
[a treatment at several nodes with covariates after baseline](../longitudinal-tmle.md#a-treatment-at-several-nodes-with-covariates-after-baseline)
states the estimand, the slots and the exact-law evidence.

The study has not run. Its design, its cells and the readings of a red cell are declared before
any run, in `tests/studies/canonical_hazard_time_varying_multinode.py` and
`tests/canonical/hazard_time_varying_multinode/regenerate.py`. The record is not registered, so a
fit of this design reads `experimental` until the declared run passes and adds the support rows
`G63.multinode.in_sample` and `G63.multinode.cross_fitted`.

## The estimand and the law

For a regimen $r$ the target is $\psi_r(t) = E\,Q^r_1(H_1)$, the risk of an event by node $t$
when each node's treatment follows $r$. The regimens are always treating, never treating, a rule
(treat when $W_1 = 1$ at node 1 and when $L_k = +1$ after) and a known policy,
$q_1(1 \mid W_1) = 0.3 + 0.4\,[W_1 = 1]$ and
$q_k(1 \mid L_k, A_{k-1}) = 0.2 + 0.4\,[L_k = +1] + 0.2\,A_{k-1}$ after. A fit reports 21
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

| cell | estimand, by node 3 | truth | efficiency bound |
| --- | --- | --- | --- |
| `multinode_contrast` | always minus never | -0.7276 | 1.040 |
| `multinode_rule` | the rule minus never | -0.3192 | 1.019 |
| `multinode_policy` | the policy minus never | -0.3225 | 0.668 |
| `multinode_risk` | always | 0.1290 | 0.748 |

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

## Declared cells

| family | cells | fits |
| --- | --- | --- |
| `interval_calibration` | six arms (`multinode` contrast, rule, policy and risk; `crossfit` contrast and policy), each with the shrunken-SE and noise controls, which must fail | 4,000 per fit label at n = 2,000, provisional until the calibration pilot |
| `double_robustness` | the contrast under all five configurations, `both_wrong` as the control; the rule under `both_correct` and `outcome_correct`; the policy under every configuration but `both_wrong` | 1,200 per configuration at n = 2,000, the wrong-learner configurations provisional until their pilot |
| `targeting_necessity` | the targeted and the untargeted contrast of the `mechanism_correct` draws | the robustness fits |
| `root_n_and_efficiency`, `root_n_rate` | the contrast and the policy at n = 1,000, 2,000 and 8,000 | 800 each; the n = 1,000 rung carries no claim |
| `power` | the contrast against zero | the `multinode` calibration fits |
| `simultaneous_coverage` | the band over the 21 estimates and its pointwise control | the `multinode` calibration fits |
| `iterated_mean_difference` | reported, no verdict: the in-package iterated-mean TMLE without and with `isotonic=True` | 800 at n = 2,000 |

The primary run is 2,400 replicates at n = 2,000, from the shared primary rule, provisional
until the calibration pilot. Every gated cell reads the one-step hazard TMLE. A draw whose fit
holds a cell is a recorded lost replicate, charged to the failure bound of 0.01. The design test
bounds the expected held fits of the study and its pilot inside each set's allowance.

## What each component is checked by

Exact-law checks cannot see a term that vanishes at the truth. The design test computes, before
any run, the exact limit that each such component moves. Each gap is in efficiency SEs at
n = 2,000.

| component | witness | gap | plan floor |
| --- | --- | --- | --- |
| the current-level slot | always minus never, from zero | -31.3 | 5 |
| the policy mixture at nodes 2 and 3 | the observed treatment in place of the mixture, `outcome_correct`, policy risk | 3.07 | 3 |
| the level plan | every node set to the slot's level, `outcome_correct`, policy contrast | 1.63 | 3, or 1 recorded |
| the held slot | the hazard at the treatment of node 1, `outcome_correct`, policy contrast | -0.50 | 3, or 1 recorded |
| the policy's prefix | the policy read at the level path, `outcome_correct`, policy contrast | 0.15 | 3, or 1 recorded |
| the rule's slots | the level plan and the held slot, `outcome_correct`, rule contrast | 0.11 and -0.19 | 3, or 1 recorded |
| the hazard term and the walk | the untargeted `mechanism_correct` contrast | 7.68 | 3 |
| the covariate term of the panel | the untargeted `continuation_wrong` contrast | 0.80 | 1 |
| $L_k$ in the mechanism | the targeted `both_wrong` contrast | 7.03 | discriminated |
| $L_k$ as a confounder of later treatment | always and never fitted with $L$ dropped | 7.03 | 3 |

Five witnesses fall below the plan's floor: the held slot, the policy's prefix and the rule's
slots below the 1 SE fallback, and the covariate term below 1 SE. The coefficients were searched
inside the positivity, hazard-clip and held-cell limits before any pilot, and these are the gaps
at the declared ones. The rule's followers hold different treatments at consecutive nodes only
when $L$ changes, which the Markov covariate makes rare. The policy reads the earlier treatment
only through the plan's fixed 0.2 term. The policy's held slot reached 1.3 SEs only at a
baseline hazard that held a cell in 7% of the n = 1,000 fits. Each of these components keeps its
deliberate-mutation control in `tests/unit/test_hazard_time_varying_multinode.py`.

## Limitations

| limit | what it means for use |
| --- | --- |
| one cause | the competing-risk path rests on the two-cause exact-law tests of `tests/unit/test_hazard_time_varying_multinode.py` |
| one binary Markov $L_k$ and a binary treatment at every node | a continuous, multivariate or non-Markov $L_k$, and unequal level counts across nodes, rest on the unit tests |
| no observation weights, clusters or `node_bounds=` | the unit tests fit weighted data, since every exact-law fit weights the support by its mass; clusters and `node_bounds=` are not measured on this design |
| declared factors and `markov_order=` | covered by the unit tests only |
| `targeting="one_step"` only | the per-horizon algorithm is refused on this design (H28) |
| the slot witnesses above their floor | the rule's slots, the policy's prefix and held slot, and the covariate term rest on the unit tests' mutation controls |
| no comparator | R `ltmle` and `lmtp` reach this estimand by the iterated-mean algorithm only, which `canonical-ltmle-survival` already ties to the in-package fit, so they would add transitive evidence only |

## Reproduction

The smoke gate fits two draws of each fit label and configuration on the pilot stream, as
`SMOKE_GATE()` in `tests/studies/canonical_hazard_time_varying_multinode.py`. Every smoke fit
must converge with every node score at most $10^{-10}$, hold no cell and take at most twice the
cost table's 1.3 s (in sample) or 1.8 s (five folds). The pilot runs one label per run, as
`python -m tests.studies.hazard_time_varying_multinode_pilot --jobs N --labels LABEL`, and writes
`tests/canonical/hazard_time_varying_multinode/pilot.csv.gz`. The declared run is
`python tests/canonical/hazard_time_varying_multinode/regenerate.py --output DIR`, with an empty
scratch directory outside the repository.
