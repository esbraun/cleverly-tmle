# A treatment at several nodes with covariates after baseline on the hazard path

This study is written to validate longitudinal TMLE with `survival_model="hazard"` and
`targeting="one_step"` when a binary treatment is decided at each of three nodes and a binary
covariate $L_k$ is measured before the treatment at every node after the first. Each regimen is
targeted by one backward pass. At each node the pass fluctuates the node's event hazard together
with a regression of the later events on the node's history, and it reads the hazard at the
observed history with only the current treatment set. The section on
[a treatment at several nodes with covariates after baseline](../longitudinal-tmle.md#a-treatment-at-several-nodes-with-covariates-after-baseline)
states the estimand, the slots and the exact-law evidence.

The study has not run. Its design, its cells, its replicate counts and the readings of a red cell
are declared before any run, in `tests/studies/canonical_hazard_time_varying_multinode.py` and
`tests/canonical/hazard_time_varying_multinode/regenerate.py`. The counts come from a committed
pilot on seed streams that no declared cell reads, and the [pilot section](#the-pilot) states
its readings. The record is not registered, so a
fit of this design reads `experimental` until the declared run passes and adds the support rows
`G63.multinode.in_sample` and `G63.multinode.cross_fitted`.

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

## Declared cells

| family | cells | fits |
| --- | --- | --- |
| `interval_calibration` | six arms (`multinode` contrast, rule, policy and risk; `crossfit` contrast and policy), each with the shrunken-SE and noise controls, which must fail | 15,600 per fit label at n = 2,000 |
| `double_robustness` | the contrast under all five configurations, `both_wrong` as the control; the rule under `both_correct` and `outcome_correct`; the policy under every configuration but `both_wrong` | at n = 2,000: 2,800 for `outcome_correct`, 1,200 for each other configuration |
| `double_robustness`, untargeted | the untargeted rule and policy contrasts of `outcome_correct` draws of their own | 1,200 at n = 8,000 |
| `targeting_necessity` | the targeted and the untargeted contrast of the `mechanism_correct` draws | the robustness fits |
| `root_n_and_efficiency`, `root_n_rate` | the contrast and the policy at n = 1,000, 2,000 and 8,000 | 800 each; the n = 1,000 rung carries no claim |
| `power` | the contrast against zero | the `multinode` calibration fits |
| `simultaneous_coverage` | the band over the 21 estimates and its pointwise control | the `multinode` calibration fits |
| `iterated_mean_difference` | reported, no verdict: the in-package iterated-mean TMLE without and with `isotonic=True` | 800 at n = 2,000 |

The primary run is 2,400 replicates at n = 2,000, which the shared primary rule
(`repeated_sizing.primary_replicates`) gives at the committed `multinode` calibration pilot. Every gated cell reads the one-step hazard TMLE. A draw whose fit
holds a cell is a recorded lost replicate, charged to the failure bound of 0.01. The design test
bounds the expected held fits of the study and its pilot inside each set's allowance.

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

### Agreement with the iterated-mean fit

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

## Limitations

| limit | what it means for use |
| --- | --- |
| one cause | the competing-risk path rests on the two-cause exact-law tests of `tests/unit/test_hazard_time_varying_multinode.py` |
| one binary Markov $L_k$ and a binary treatment at every node | a continuous, multivariate or non-Markov $L_k$, and unequal level counts across nodes, rest on the unit tests |
| no observation weights, clusters or `node_bounds=` | the unit tests fit weighted data, since every exact-law fit weights the support by its mass; clusters and `node_bounds=` are not measured on this design |
| declared factors and `markov_order=` | covered by the unit tests only |
| `targeting="one_step"` only | the per-horizon algorithm is refused on this design (H28) |
| the witnesses below their floor | the policy's prefix sits between 1 and 3 SEs, and the covariate term at 0.80 SEs; each also rests on its unit-test mutation control |
| the untargeted arms at n = 8,000 | at n = 2,000 the untargeted `outcome_correct` plug-in carries a finite-sample bias of about a quarter of its SD, from the study's continuation learner (see [the pilot](#the-pilot)); the arms that read it run at n = 8,000 |
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
