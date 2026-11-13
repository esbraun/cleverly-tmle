---
orphan: true
---

# Covariates after baseline on the hazard path

This study validates longitudinal TMLE with `survival_model="hazard"` and
`targeting="one_step"` when a baseline treatment is held over the visits and a binary covariate
$L_k$ is measured at every later visit. Each regimen is targeted by one backward pass that
fluctuates each visit's event hazards together with a regression of the later events on the
visit's history. The section on
[covariates after baseline on the hazard path](../longitudinal-tmle.md#covariates-after-baseline-on-the-hazard-path)
states the estimand, the influence curve and the exact-law evidence.

The study is declared and has not run. The page is not linked from the index until the run
commits its artifacts and the record is registered, so the three result blocks below are empty.

## The estimand and the laws

For a regimen $d$ and cause $j$ the target is $\psi_d(t, j) = E\,Q_1(W, d(A); t, j)$, the risk
of a cause-$j$ event by visit $t$ when each unit's baseline treatment is set by $d$ and held. The
binary laws compare always treating with never treating, and the primary contrast is
$\psi_1(T, 1) - \psi_0(T, 1)$ at the last visit $T$. The three-level law compares "one level up,
the top stays", a modified treatment policy at the baseline decision, with the natural course.
Identification needs no unmeasured confounding of $A$ given $W$, censoring at random given the
history $(W, A, \bar L_k)$, and positivity of the regimen (Díaz, Williams, Hoffman and Schenck
2023, Theorem 1, with the identity policy after baseline).

$W_1 \in \{0, 1\}$ and $W_2 \in \{0, \dots, 3\}$ are independent and uniform. At each visit
$k \ge 2$ the covariate $L_k$ is drawn first, logistic in $W_1$, $A$ and $L_{k-1}$, and treatment
lowers it. The unit then drops out with a probability logistic in the visit, $A$, $W$ and $L_k$,
and a unit still followed has each event with a logistic pseudo-hazard in the same terms. The
coefficient of $L_k$ is 3.4 in the dropout and 3.3 in the first cause's hazard, so a fit that
ignores $L_k$ treats an informative dropout as ignorable.

| law | visits | treatment | regimens | smallest treatment probability | smallest retention through the last visit | smallest product |
| --- | --- | --- | --- | --- | --- | --- |
| `one_cause` | 5 | binary, logistic in $W$ | always, never | 0.289 | 0.136 | 0.039 |
| `competing` | 4 | binary, as `one_cause`; relapse and death compete | always, never | 0.289 | 0.223 | 0.064 |
| `three_arm` | 5 | three levels, multinomial logit in $W$ | one level up, the natural course | 0.224 | 0.136 | 0.031 |

The smallest product of a treatment probability and the retention is above twice the default
cumulative bound of 0.01, so no bound truncates a weight. Each law is Markov: $L_k$ reads
$L_{k-1}$ only, and every factor at visit $k$ reads $L_k$ only. A logistic regression on
$(W_1, W_2, A, L_k)$ is then a correctly specified hazard and censoring model without being
saturated, and the cell mean over $(W, A, L_k, s)$ is the true regression of the later events.

Plan X70 named the law of `make_point_survival` for `one_cause` and expected this study to need
its own law. That law's dropout is linear on the probability scale and at most 0.12, so the
`both_wrong` control could not clear its margin. The study's laws keep its structure and draw a
logistic dropout that reads $L_k$; they are defined in `tests/studies/hazard_time_varying_law.py`.

Every truth is an exact sum over the law's support: the eight strata of $W$, the treatment
levels and the $2^{T-1}$ covariate paths. Every efficiency bound is the exact standard deviation
of the efficient influence function over the same support. Neither calls the estimator.
`tests/unit/test_hazard_time_varying_study_design.py` checks the truths against a backward
recursion to $10^{-14}$ and against a fit on the exact sample with the true learners, and checks
that the influence curve of that fit is the efficient influence function. The `ic_gap` pilot
label adds one fit at n = 40,000 per fit label, whose influence-curve SD is read against the
bound, and the design test reads it from the committed pilot.

| cell | estimand | truth | efficiency bound |
| --- | --- | --- | --- |
| `one_cause_contrast` | always minus never, by visit 5 | -0.4582 | 1.063 |
| `one_cause_risk` | always, by visit 5 | 0.1930 | 0.551 |
| `competing_contrast` | always minus never, relapse by visit 4 | -0.3228 | 1.028 |
| `competing_risk` | always, relapse by visit 4 | 0.1246 | 0.457 |
| `three_arm_contrast` | one level up minus the natural course, by visit 5 | -0.1625 | 0.432 |
| `three_arm_risk` | one level up, by visit 5 | 0.1375 | 0.440 |

## The fits

| fit label | law | regimens | folds |
| --- | --- | --- | --- |
| `one_cause` | `one_cause` | static | 1 |
| `competing` | `competing` | static | 1 |
| `three_arm` | `three_arm` | a modified treatment policy and the natural course | 1 |
| `crossfit` | `one_cause` | static | 5 |

| nuisance | correct learner | wrong learner |
| --- | --- | --- |
| node hazard | unpenalized logistic regression on $W_1$, $W_2$, $A$ and $L_k$ at each visit (`LastL`) | the same without any $L$ column (`NoL`) |
| continuation (`pseudo_learner`) | the cell mean over $(W, A, L_k, s)$, with the later visit $s$ as indicators | the cell mean over $(W, A, s)$ |
| treatment | logistic or multinomial logistic regression on $W$ | none: the treatment reads no $L$, so it is correct in every configuration |
| censoring | logistic regression on $W_1$, $W_2$, $A$ and $L_k$ at each visit | the same without $L_k$ |

In the `double_robustness` family the wrong outcome side is the hazard and the continuation
without $L$, and the wrong mechanism is the censoring model without $L_k$. A fifth
configuration, `continuation_wrong`, keeps the hazard and the censoring model correct and drops
$L$ from the continuation alone. Plan X70 lists no cross-fitted cell for this study. The
`crossfit` fit label is a calibration cell added so that the cross-fitted held row has evidence
of its own.

A cell whose events are absent from a draw is held by the shipped fit: when no unit of one
treatment level at a visit has an event of a cause, every estimate that reads that node hazard
reports the plug-in without an interval. The study turns such a fit into a lost replicate on
every path, charged to the cell's failure bound of 0.01. The rarest cell is level 2 at visit 1
on `three_arm`, with mass 0.0065 per unit, and the design test puts the expected count of held
fits over the whole study and its pilot at 0.05.

No maintained package targets discrete-time hazards with covariates after baseline. R `ltmle`
and `lmtp` estimate the same estimand by the iterated-mean algorithm, which the in-package
iterated-mean fit already ties to `ltmle` in `canonical-ltmle-survival`. The study names no
comparator, and its agreement block stays empty.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

The primary rows draw 2,400 samples of 2,000 rows for each of `one_cause`, `competing` and
`three_arm` and report every risk and contrast of each fit. The shared primary rule
(`repeated_sizing.primary_replicates`) sets that count from the committed calibration pilots of
these fit labels, so 2,400 is provisional until they are committed.

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | the contrast and the regimen's risk of each fit label, with the shrunken-SE and noise controls on the `one_cause` contrast | 4,000 at n = 2,000 (provisional) | an SE ratio or coverage interval outside its band, or an efficiency ratio outside its band |
| `simultaneous_coverage` | the band over every estimate of the `one_cause` and `competing` fits, node-1 risks included, and its pointwise control | the calibration fits | joint coverage outside the calibration band |
| `double_robustness` | the `one_cause` contrast with both sides correct, the outcome side correct, the mechanism correct, the continuation alone wrong, and both wrong; the `competing` and `three_arm` contrasts with the mechanism correct | 1,200 at n = 2,000; each arm with a wrong learner as its own pilot sets | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `targeting_necessity` | the `one_cause` contrast of the `mechanism_correct` robustness fits, targeted and as the untargeted backward plug-in of the same draws | those fits' count | the targeted bias outside the margin, or the plug-in inside it |
| `root_n_and_efficiency`, `root_n_rate` | the `one_cause` contrast and risk at n = 1,000, 2,000 and 8,000; the n = 1,000 rung is a rate rung that carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the `one_cause` contrast against zero, on the `one_cause` calibration fits | the calibration count | a rejection lower bound below the minimum power |
| every family | `nonconverged_fits`: the fits that stopped short of `tol` with a final relative score at or below $10^{-6}$, beside each cell's count of unexpected warnings | as the cell | a non-converged fit above $10^{-6}$ is a lost replicate under the 0.01 failure bound; more than 1% non-converged fits makes the cell a defect suspect |
| every family | `smallest_kish`, `largest_relative_sd` and `largest_edgeworth_loss`: the influence-curve measures of each row's fit over its estimates | as the cell | nothing: recorded beside the verdicts |
| `iterated_mean_difference` | on `one_cause` and `competing`: the hazard estimate minus the iterated-mean estimate of the same draw, without and with `isotonic=True`, each fit against the truth, and the largest decrease of the iterated-mean curve. On `competing`, each fit's largest sum of the two cumulative incidences minus one. On the `one_cause` `mechanism_correct` draws, the same paired difference with every regression omitting $L$ | 800 at n = 2,000, and the robustness draws | nothing: the family is reported with no verdict |

The calibration count is the arm-pooled SE-ratio rule of `mtp_study_tools.smallest_replicates`
over the eight arms, read at the committed calibration pilot of 8,000 draws per fit label. At a
calibrated pilot it gives 4,000. The design test checks that the shrunken-SE and noise controls
leave their bands at the declared count, and that the power cell has power on the calibration
fits. The necessity cells read the `mechanism_correct` robustness fits and the power cell the
calibration fits, so neither adds a fit set.

The bands read the node-1 risks, whose influence curves have Kish counts of 20 to 24 and a
predicted Wald coverage loss of about 0.009 each. Their losses do not average out of a maximum,
so the `one_cause` and `competing` calibration pilots fit the band, as the declared fits do. A
rule fixed before the pilot is read then requires the declared band cell to pass with
probability at least 0.90 at the pilot's joint coverage plus and minus one standard error. If it
does not, the declaration stops for a diagnosis, and the band is not widened.

| arm | untargeted plug-in, efficiency SEs at n = 2,000 | what it witnesses |
| --- | --- | --- |
| `one_cause` `mechanism_correct` | +4.23 | the hazard fluctuation, whose weights read $L_k$ |
| `one_cause` `continuation_wrong` | +1.37 | the covariate term of the walk alone |
| `competing` `mechanism_correct` | +3.20 | the two-cause soft-label panel |
| `three_arm` `mechanism_correct` | +3.63 | the policy ratio times the retention that reads $L_k$ |
| `one_cause` `both_wrong` (control) | +4.23; targeted +4.22 | must miss |

Each figure is the exact-sample limit, and every positive arm's targeted fit returns the truth
on the exact sample to $10^{-7}$. With a hazard that omits $L$, every fitted continuation is a
function of $(W, A)$, so the covariate term is zero before targeting and the `mechanism_correct`
arms cannot see it. With the hazard correct, the hazard term is zero, so the
`continuation_wrong` arm reaches the truth only through the covariate term.

The `one_cause` plug-in limit is also computed from the law by an independent logistic fit on the
at-risk cells, and with dropout that does not read $L_k$ it lies within one standard error of
the truth, so the gap comes from the informative dropout. The `both_wrong` control and the
untargeted necessity arm read one bias: with every factor a function of $(W, A)$, which the
initial fit nearly saturates, the walk barely moves. They are valid controls, but not two
independent witnesses. If either does not clear the margin, the control has failed and the
study is a defect suspect.

The arms with a wrong learner carry a finite-sample bias of their own, so each arm's size comes
from its own pilot of 800 draws at n = 2,000; the `one_cause` `mechanism_correct` reading pools
the 200 `mechanism_correct` paired draws, which fit the same estimator. Each arm takes 1,200
replicates, or the smallest multiple of 400 that passes with probability 0.99 at the pilot's
standardized bias plus one standard error, whichever is larger. The rule was fixed before any
pilot was read, has no cap, and stops the study above 4,000. The `outcome_correct` arm walks
with censoring weights that omit $L_k$, and its own bias is not piloted: the `both_correct`
reading is its proxy, and at 1,200 replicates it tolerates a standardized bias of 0.108.

With correctly specified learners the hazard fit and the iterated-mean fit are both efficient,
so they differ by $o_P(n^{-1/2})$. The study reports their agreement, the share of decreasing
iterated-mean curves with its interval, and on `competing` how far each fit's incidences sum
above one. The decrease and coherence rows record zero as a reference, not a truth: their true
values are negative, and the hazard fit's sum is at most one by construction. So only each
reading's share above zero and its largest value are reported, with no floor: X70 reports the agreement of the two fits instead of requiring a gap
(the owner's decision of 2026-11-13).

## Limitations

| limit | what it means for use |
| --- | --- |
| one baseline decision | treatment at several visits beside $L_k$ is part (b2) of plan X70 |
| one scalar binary $L_k$ on Markov laws and finite $W$ | the evidence covers correct or misspecified logistic and cell-mean learners. A flexible learner, or a continuous, multivariate or non-Markov $L_k$, is outside it; the unit tests cover a continuous $L_k$ |
| `targeting="one_step"` only | the per-horizon algorithm is refused on this design (H28) |
| static regimens and one modified treatment policy at visit 1 | plan X70 puts a rule and a known policy on a held design in scope; they rest on `tests/unit/test_hazard_time_varying.py` |
| no observation weights, clusters or `node_bounds=` | each is supported by the fit and covered by the unit tests only |
| the cross-fitted stitch | with parametric learners an in-fold and an out-of-fold stitch have the same limit, so the `crossfit` cells cannot see the stitch; it rests on the unit tests |
| held cells | a held cell costs a replicate; the design test expects 0.05 held fits over the study and its pilot. The held-cell estimate itself rests on `tests/unit/test_hazard_time_varying.py` |
| support rows | the support key has no axis for covariates after baseline, so `G31` and `G32` cover these fits today and the declared `G63.held` rows cannot be registered until the key gains a `later_covariates` option, set only on the hazard path. The option and the removal of the X70 condition land in one change. The rows on `one_cause` and `competing` license the band |
| no comparator | no external implementation targets these hazards with covariates after baseline |

## Reproduction

Run `python tests/canonical/hazard_time_varying/regenerate.py --output DIR` with an empty
scratch directory outside the repository to regenerate the artifacts. The pilot runs separately,
one label per run, as `python -m tests.studies.hazard_time_varying_pilot --jobs N --labels LABEL`,
and writes `tests/canonical/hazard_time_varying/pilot.csv.gz`. The red-cell readings are
declared in `tests/canonical/hazard_time_varying/regenerate.py` before any run.
