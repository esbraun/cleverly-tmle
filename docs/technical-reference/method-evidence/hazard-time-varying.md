# Covariates after baseline on the hazard path

This study validates longitudinal TMLE with `survival_model="hazard"` and
`targeting="one_step"` when a baseline treatment is held over the visits and a binary covariate
$L_k$ is measured at every later visit. Each regimen is targeted by one backward pass that
fluctuates each visit's event hazards together with a regression of the later events on the
visit's history. The section on
[covariates after baseline on the hazard path](../longitudinal-tmle.md#covariates-after-baseline-on-the-hazard-path)
states the estimand, the influence curve and the exact-law evidence.

The declared run at commit `5aeef06` passed every gated row: 54 of 54 primary tests against
known truth and 34 of 34 property cells with a verdict. It ran 2 h 50 min on 16 cores, and no
fit failed or held a cell. The support rows `G63.held` license the fits it covers.

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

Over the four fit labels in that pilot, the median gap of the five fits at n = 2,000 runs from
0.080 (`three_arm` risk) to 0.150 (`one_cause` contrast), against a bound of 0.3. At n = 20,000
the gap lies between 0.024 and 0.066, against a bound of 0.1, and between 0.20 and 0.46 times
its median at n = 2,000, against the halving rule's 0.5. At n = 40,000 each fit's influence-curve SD is between 0.985 (`crossfit` contrast) and
1.003 times the efficiency bound.

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
fits over the whole study and its pilot at 0.05. The committed pilot held no draw of any label.

Every fit of the committed pilot converged to `tol` = $10^{-10}$, and no draw was refused or
held. The pilot rows record each fit's accepted steps over every node walk of every regimen and
the warnings outside the expected `SparseInfluenceWarning`, but not its time. The seconds below
are each label's wall time from its pilot commit on the runner's 4 cores, times 4, over its
draws.

| pilot label | draws | walk steps: mean (most) | draws with an unexpected warning | core-seconds a draw |
| --- | --- | --- | --- | --- |
| `calibration_one_cause` | 8,000 | 1,016 (1,874) | 2 | 1.22 |
| `calibration_competing` | 8,000 | 984 (1,936) | 1 | 2.40 |
| `calibration_three_arm` | 8,000 | 748 (928) | 2 | 1.70 |
| `calibration_crossfit` | 8,000 | 1,214 (3,698) | 8 | 1.62 |
| `wrong_learner_one_cause` | 800 | 3,847 (5,376) | 0 | 2.83 |
| `wrong_learner_continuation` | 800 | 1,713 (2,985) | 0 | 1.81 |
| `wrong_learner_competing` | 800 | 2,644 (3,611) | 1 | 4.14 |
| `wrong_learner_three_arm` | 800 | 1,154 (1,382) | 0 | 2.26 |
| `fits_differ_one_cause` | 200 | 1,014 (1,507) | 0 | 1.76 |
| `fits_differ_competing` | 200 | 987 (1,311) | 0 | 3.04 |
| `fits_differ_mechanism_correct` | 200 | 3,772 (5,273) | 0 | 3.16 |

The `one_cause` and `competing` calibration draws fit the band as well. A `wrong_learner` draw's
time includes its untargeted pass, and a `fits_differ` draw's time its iterated-mean fits. The
28 `ic_gap` fits took 344 core-seconds. Node 1's walk moved its estimates by a mean of 0.04
(`one_cause`), 0.07 (`competing`), 0.06 (`three_arm`) and 0.17 (`crossfit`) of their standard
errors, at most 1.5.

No maintained package targets discrete-time hazards with covariates after baseline. R `ltmle`
and `lmtp` estimate the same estimand by the iterated-mean algorithm, which the in-package
iterated-mean fit already ties to `ltmle` in `canonical-ltmle-survival`. The study names no
comparator, so no canonical implementation is compared.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never, death @ t=1]` | the cumulative incidence of death by visit 1 under treatment at every visit minus that under no treatment | `cleverly` | -0.000721 to 0.000535 | 0.9529 | 1.0051 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never, death @ t=2]` | the cumulative incidence of death by visit 2 under treatment at every visit minus that under no treatment | `cleverly` | -0.000939 to 0.000832 | 0.9471 | 0.9820 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never, death @ t=3]` | the cumulative incidence of death by visit 3 under treatment at every visit minus that under no treatment | `cleverly` | -0.0014 to 0.000705 | 0.9400 | 0.9804 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never, death @ t=4]` | the cumulative incidence of death by visit 4 under treatment at every visit minus that under no treatment | `cleverly` | -0.0017 to 0.000631 | 0.9454 | 0.9830 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never, relapse @ t=1]` | the cumulative incidence of relapse by visit 1 under treatment at every visit minus that under no treatment | `cleverly` | -0.000456 to 0.000340 | 0.9463 | 0.9860 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never, relapse @ t=2]` | the cumulative incidence of relapse by visit 2 under treatment at every visit minus that under no treatment | `cleverly` | -0.0010 to 0.000748 | 0.9471 | 0.9783 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never, relapse @ t=3]` | the cumulative incidence of relapse by visit 3 under treatment at every visit minus that under no treatment | `cleverly` | -0.0016 to 0.000684 | 0.9425 | 0.9809 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never, relapse @ t=4]` | the cumulative incidence of relapse by visit 4 under treatment at every visit minus that under no treatment | `cleverly` | -0.0017 to 0.000800 | 0.9417 | 0.9771 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[always, death @ t=1]` | the cumulative incidence of death by visit 1 under treatment at every visit | `cleverly` | -0.000548 to 0.000342 | 0.9446 | 1.0012 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[always, death @ t=2]` | the cumulative incidence of death by visit 2 under treatment at every visit | `cleverly` | -0.000782 to 0.000391 | 0.9533 | 1.0165 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[always, death @ t=3]` | the cumulative incidence of death by visit 3 under treatment at every visit | `cleverly` | -0.0011 to 0.000224 | 0.9487 | 1.0093 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[always, death @ t=4]` | the cumulative incidence of death by visit 4 under treatment at every visit | `cleverly` | -0.0012 to 0.000244 | 0.9508 | 0.9987 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[always, relapse @ t=1]` | the cumulative incidence of relapse by visit 1 under treatment at every visit | `cleverly` | -0.000171 to 0.000298 | 0.9450 | 0.9934 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[always, relapse @ t=2]` | the cumulative incidence of relapse by visit 2 under treatment at every visit | `cleverly` | -0.000438 to 0.000312 | 0.9421 | 0.9837 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[always, relapse @ t=3]` | the cumulative incidence of relapse by visit 3 under treatment at every visit | `cleverly` | -0.000556 to 0.000394 | 0.9383 | 0.9815 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[always, relapse @ t=4]` | the cumulative incidence of relapse by visit 4 under treatment at every visit | `cleverly` | -0.000784 to 0.000309 | 0.9425 | 0.9835 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[never, death @ t=1]` | the cumulative incidence of death by visit 1 under no treatment at any visit | `cleverly` | -0.000464 to 0.000445 | 0.9387 | 0.9837 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[never, death @ t=2]` | the cumulative incidence of death by visit 2 under no treatment at any visit | `cleverly` | -0.000787 to 0.000502 | 0.9437 | 0.9836 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[never, death @ t=3]` | the cumulative incidence of death by visit 3 under no treatment at any visit | `cleverly` | -0.000884 to 0.000644 | 0.9433 | 0.9917 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[never, death @ t=4]` | the cumulative incidence of death by visit 4 under no treatment at any visit | `cleverly` | -0.000847 to 0.000864 | 0.9458 | 0.9841 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[never, relapse @ t=1]` | the cumulative incidence of relapse by visit 1 under no treatment at any visit | `cleverly` | -0.000193 to 0.000436 | 0.9458 | 1.0001 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[never, relapse @ t=2]` | the cumulative incidence of relapse by visit 2 under no treatment at any visit | `cleverly` | -0.000716 to 0.000881 | 0.9479 | 0.9925 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[never, relapse @ t=3]` | the cumulative incidence of relapse by visit 3 under no treatment at any visit | `cleverly` | -0.000671 to 0.0014 | 0.9429 | 0.9729 | pass |
| four visits, relapse and death as competing causes, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `cif_regimen[never, relapse @ t=4]` | the cumulative incidence of relapse by visit 4 under no treatment at any visit | `cleverly` | -0.000923 to 0.0013 | 0.9379 | 0.9679 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never @ t=1]` | the risk by visit 1 under treatment at every visit minus the risk under no treatment | `cleverly` | -0.000668 to 0.000125 | 0.9467 | 0.9924 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never @ t=2]` | the risk by visit 2 under treatment at every visit minus the risk under no treatment | `cleverly` | -0.0013 to 0.000445 | 0.9521 | 1.0021 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never @ t=3]` | the risk by visit 3 under treatment at every visit minus the risk under no treatment | `cleverly` | -0.0017 to 0.000680 | 0.9450 | 0.9827 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never @ t=4]` | the risk by visit 4 under treatment at every visit minus the risk under no treatment | `cleverly` | -0.0018 to 0.000711 | 0.9463 | 0.9779 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `ate_regimen[always vs never @ t=5]` | the risk by visit 5 under treatment at every visit minus the risk under no treatment | `cleverly` | -0.0016 to 0.000928 | 0.9454 | 0.9838 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[always @ t=1]` | the risk of the event by visit 1 under treatment at every visit | `cleverly` | -0.000211 to 0.000264 | 0.9354 | 0.9797 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[always @ t=2]` | the risk of the event by visit 2 under treatment at every visit | `cleverly` | -0.000416 to 0.000332 | 0.9517 | 1.0141 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[always @ t=3]` | the risk of the event by visit 3 under treatment at every visit | `cleverly` | -0.000558 to 0.000407 | 0.9508 | 1.0166 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[always @ t=4]` | the risk of the event by visit 4 under treatment at every visit | `cleverly` | -0.000780 to 0.000365 | 0.9467 | 1.0087 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[always @ t=5]` | the risk of the event by visit 5 under treatment at every visit | `cleverly` | -0.000747 to 0.000537 | 0.9513 | 1.0088 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[never @ t=1]` | the risk of the event by visit 1 under no treatment at any visit | `cleverly` | -0.000014 to 0.000611 | 0.9413 | 1.0108 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[never @ t=2]` | the risk of the event by visit 2 under no treatment at any visit | `cleverly` | -0.000400 to 0.0012 | 0.9529 | 1.0066 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[never @ t=3]` | the risk of the event by visit 3 under no treatment at any visit | `cleverly` | -0.000643 to 0.0015 | 0.9404 | 0.9850 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[never @ t=4]` | the risk of the event by visit 4 under no treatment at any visit | `cleverly` | -0.000771 to 0.0015 | 0.9496 | 0.9830 | pass |
| five visits, one cause, a binary treatment at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate | `risk_regimen[never @ t=5]` | the risk of the event by visit 5 under no treatment at any visit | `cleverly` | -0.000847 to 0.0013 | 0.9442 | 0.9906 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `ate_regimen[up vs natural @ t=1]` | the risk by visit 1 under the policy minus the risk under the natural course | `cleverly` | -0.000150 to 0.000141 | 0.9504 | 0.9857 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `ate_regimen[up vs natural @ t=2]` | the risk by visit 2 under the policy minus the risk under the natural course | `cleverly` | -0.000486 to 0.000158 | 0.9487 | 0.9837 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `ate_regimen[up vs natural @ t=3]` | the risk by visit 3 under the policy minus the risk under the natural course | `cleverly` | -0.000636 to 0.000212 | 0.9496 | 0.9952 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `ate_regimen[up vs natural @ t=4]` | the risk by visit 4 under the policy minus the risk under the natural course | `cleverly` | -0.000679 to 0.000270 | 0.9575 | 1.0150 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `ate_regimen[up vs natural @ t=5]` | the risk by visit 5 under the policy minus the risk under the natural course | `cleverly` | -0.000473 to 0.000564 | 0.9454 | 0.9859 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[natural @ t=1]` | the risk of the event by visit 1 under the natural course | `cleverly` | -0.000190 to 0.000167 | 0.9387 | 0.9823 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[natural @ t=2]` | the risk of the event by visit 2 under the natural course | `cleverly` | -0.000074 to 0.000658 | 0.9408 | 0.9811 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[natural @ t=3]` | the risk of the event by visit 3 under the natural course | `cleverly` | -0.000198 to 0.000760 | 0.9483 | 0.9999 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[natural @ t=4]` | the risk of the event by visit 4 under the natural course | `cleverly` | -0.000590 to 0.000518 | 0.9433 | 0.9989 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[natural @ t=5]` | the risk of the event by visit 5 under the natural course | `cleverly` | -0.000953 to 0.000250 | 0.9437 | 0.9945 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[up @ t=1]` | the risk of the event by visit 1 under the policy | `cleverly` | -0.000223 to 0.000191 | 0.9363 | 0.9861 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[up @ t=2]` | the risk of the event by visit 2 under the policy | `cleverly` | -0.000187 to 0.000443 | 0.9429 | 0.9855 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[up @ t=3]` | the risk of the event by visit 3 under the policy | `cleverly` | -0.000332 to 0.000470 | 0.9417 | 0.9775 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[up @ t=4]` | the risk of the event by visit 4 under the policy | `cleverly` | -0.000706 to 0.000225 | 0.9425 | 0.9866 | pass |
| five visits, one cause, a treatment on three levels at every visit, two baseline covariates, a binary covariate at every later visit that the previous treatment lowers, logistic visit dropout that depends on the covariate; the policy raises each visit's level by one, capped at the top level | `risk_regimen[up @ t=5]` | the risk of the event by visit 5 under the policy | `cleverly` | -0.000823 to 0.000211 | 0.9492 | 1.0021 | pass |
<!-- /generated -->

The primary rows draw 2,400 samples of 2,000 rows for each of `one_cause`, `competing` and
`three_arm` and report every risk and contrast of each fit. The shared primary rule
(`repeated_sizing.primary_replicates`) sets that count from the committed calibration pilots of
these fit labels, and at the committed pilots it gives 2,400.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `competing_contrast__mechanism_correct` | positive | the two-cause law under always against never treating, n = 2,000, the relapse incidence difference at the last visit: node hazards and continuations that omit the time-varying covariate, with correct treatment and censoring models; it witnesses the hazard fluctuation, whose weights read the covariate | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0017 to 0.0018, margin 0.0059, SE ratio 1.0288 | pass |
| `double_robustness` | `one_cause_contrast__both_correct` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0025 to 0.0010, margin 0.0059, SE ratio 1.0010 | pass |
| `double_robustness` | `one_cause_contrast__both_wrong` | control | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: every outcome and censoring learner omits the time-varying covariate; it reads the same bias as the untargeted necessity control | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0979 to 0.1016, margin 0.0062, SE ratio 0.9851 | pass |
| `double_robustness` | `one_cause_contrast__continuation_wrong` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: correct node hazards and censoring model with continuations that omit the time-varying covariate; it witnesses the covariate term of the walk alone | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000997 to 0.0025, margin 0.0059, SE ratio 1.0205 | pass |
| `double_robustness` | `one_cause_contrast__mechanism_correct` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: node hazards and continuations that omit the time-varying covariate, with correct treatment and censoring models; it witnesses the hazard fluctuation, whose weights read the covariate | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000708 to 0.0029, margin 0.0060, SE ratio 1.0539 | pass |
| `double_robustness` | `one_cause_contrast__outcome_correct` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: correct node hazards and continuations with a censoring model that omits the time-varying covariate | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0013 to 0.0023, margin 0.0059, SE ratio 0.9693 | pass |
| `double_robustness` | `three_arm_contrast__mechanism_correct` | positive | the three-level treatment law under one level up, capped at the top level, against the natural course, n = 2,000, the risk difference against the natural course at the last visit: node hazards and continuations that omit the time-varying covariate, with correct treatment and censoring models; it witnesses the hazard fluctuation, whose weights read the covariate | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000862 to 0.000678, margin 0.0026, SE ratio 1.0030 | pass |
| `interval_calibration` | `competing_contrast__correctly_specified` | positive | the two-cause law under always against never treating, n = 2,000, the relapse incidence difference at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9331 to 0.9506, SE ratio 0.9639 to 1.0167, empirical efficiency ratio 0.9834 to 1.0373, reported efficiency ratio 0.9988 to 1.0008 | pass |
| `interval_calibration` | `competing_risk__correctly_specified` | positive | the two-cause law under always against never treating, n = 2,000, the relapse incidence under always treating by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9407 to 0.9572, SE ratio 0.9772 to 1.0287, empirical efficiency ratio 0.9714 to 1.0221, reported efficiency ratio 0.9977 to 1.0005 | pass |
| `interval_calibration` | `crossfit_contrast__correctly_specified` | positive | the one-cause law under always against never treating, five outer folds, n = 2,000, the risk difference at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9422 to 0.9585, SE ratio 0.9908 to 1.0446, empirical efficiency ratio 0.9797 to 1.0328, reported efficiency ratio 1.0217 to 1.0255 | pass |
| `interval_calibration` | `crossfit_risk__correctly_specified` | positive | the one-cause law under always against never treating, five outer folds, n = 2,000, the risk under always treating by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9434 to 0.9595, SE ratio 0.9932 to 1.0452, empirical efficiency ratio 0.9668 to 1.0172, reported efficiency ratio 1.0094 to 1.0116 | pass |
| `interval_calibration` | `one_cause_contrast__correctly_specified` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9409 to 0.9574, SE ratio 0.9632 to 1.0140, empirical efficiency ratio 0.9854 to 1.0373, reported efficiency ratio 0.9975 to 1.0009 | pass |
| `interval_calibration` | `one_cause_contrast__noise_control` | control | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8180 to 0.8460, SE ratio 0.6913 to 0.7277, empirical efficiency ratio 1.3733 to 1.4452, reported efficiency ratio 0.9975 to 1.0009 | pass |
| `interval_calibration` | `one_cause_contrast__shrunken_se_control` | control | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8100 to 0.8385, SE ratio 0.6734 to 0.7106, empirical efficiency ratio 0.9845 to 1.0385, reported efficiency ratio 0.6983 to 0.7006 | pass |
| `interval_calibration` | `one_cause_risk__correctly_specified` | positive | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9429 to 0.9591, SE ratio 0.9821 to 1.0344, empirical efficiency ratio 0.9678 to 1.0189, reported efficiency ratio 0.9997 to 1.0018 | pass |
| `interval_calibration` | `three_arm_contrast__correctly_specified` | positive | the three-level treatment law under one level up, capped at the top level, against the natural course, n = 2,000, the risk difference against the natural course at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9360 to 0.9531, SE ratio 0.9664 to 1.0183, empirical efficiency ratio 0.9854 to 1.0382, reported efficiency ratio 1.0018 to 1.0044 | pass |
| `interval_calibration` | `three_arm_risk__correctly_specified` | positive | the three-level treatment law under one level up, capped at the top level, against the natural course, n = 2,000, the policy's risk by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9382 to 0.9551, SE ratio 0.9786 to 1.0323, empirical efficiency ratio 0.9694 to 1.0225, reported efficiency ratio 0.9995 to 1.0024 | pass |
| `iterated_mean_difference` | `competing__hazard_excess` | diagnostic | the two-cause law under always against never treating, n = 2,000, beside the iterated-mean fit of the same plans on the same draw: the largest sum of the two cumulative incidences minus one under the hazard fit, over both plans and every visit | no verdict; reported beside the other fits' largest sums | coverage 0.9934 to 1, SE ratio 1.0183 | reported |
| `iterated_mean_difference` | `competing__isotonic_excess` | diagnostic | the two-cause law under always against never treating, n = 2,000, beside the iterated-mean fit of the same plans on the same draw: the largest sum of the two cumulative incidences minus one under the isotonic iterated-mean fit, over both plans and every visit | no verdict; reported beside the other fits' largest sums | coverage 0.9934 to 1, SE ratio 1.0035 | reported |
| `iterated_mean_difference` | `competing__iterated_curve_decrease` | diagnostic | the two-cause law under always against never treating, n = 2,000, beside the iterated-mean fit of the same plans on the same draw: the largest decrease of the iterated-mean risk curve of either arm between consecutive visits | no verdict; the share of draws whose curve decreases somewhere is reported | decreasing share 0, 99% interval 0 to 0.0066 | reported |
| `iterated_mean_difference` | `competing__iterated_excess` | diagnostic | the two-cause law under always against never treating, n = 2,000, beside the iterated-mean fit of the same plans on the same draw: the largest sum of the two cumulative incidences minus one under the iterated-mean fit, over both plans and every visit | no verdict; reported beside the other fits' largest sums | coverage 0.9934 to 1, SE ratio 1.0035 | reported |
| `iterated_mean_difference` | `competing_contrast__hazard_fit` | diagnostic | the two-cause law under always against never treating, n = 2,000, the relapse incidence difference at the last visit: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9386 to 0.9757, SE ratio 1.0474 | reported |
| `iterated_mean_difference` | `competing_contrast__isotonic_difference` | diagnostic | the two-cause law under always against never treating, n = 2,000, the relapse incidence difference at the last visit: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1382, mean difference 0.000050 to 0.000630 | reported |
| `iterated_mean_difference` | `competing_contrast__iterated_fit` | diagnostic | the two-cause law under always against never treating, n = 2,000, the relapse incidence difference at the last visit: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9326 to 0.9717, SE ratio 1.0265 | reported |
| `iterated_mean_difference` | `competing_contrast__paired_difference` | diagnostic | the two-cause law under always against never treating, n = 2,000, the relapse incidence difference at the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1382, mean difference 0.000050 to 0.000630 | reported |
| `iterated_mean_difference` | `competing_risk__hazard_fit` | diagnostic | the two-cause law under always against never treating, n = 2,000, the relapse incidence under always treating by the last visit: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9223 to 0.9647, SE ratio 0.9717 | reported |
| `iterated_mean_difference` | `competing_risk__isotonic_difference` | diagnostic | the two-cause law under always against never treating, n = 2,000, the relapse incidence under always treating by the last visit: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1433, mean difference 0.000200 to 0.000467 | reported |
| `iterated_mean_difference` | `competing_risk__iterated_fit` | diagnostic | the two-cause law under always against never treating, n = 2,000, the relapse incidence under always treating by the last visit: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9208 to 0.9637, SE ratio 0.9512 | reported |
| `iterated_mean_difference` | `competing_risk__paired_difference` | diagnostic | the two-cause law under always against never treating, n = 2,000, the relapse incidence under always treating by the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1433, mean difference 0.000200 to 0.000467 | reported |
| `iterated_mean_difference` | `one_cause__iterated_curve_decrease` | diagnostic | the one-cause law under always against never treating, n = 2,000, beside the iterated-mean fit of the same plans on the same draw: the largest decrease of the iterated-mean risk curve of either arm between consecutive visits | no verdict; the share of draws whose curve decreases somewhere is reported | decreasing share 0, 99% interval 0 to 0.0066 | reported |
| `iterated_mean_difference` | `one_cause_contrast__hazard_fit` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9165 to 0.9606, SE ratio 0.9743 | reported |
| `iterated_mean_difference` | `one_cause_contrast__isotonic_difference` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1611, mean difference 0.000258 to 0.000957 | reported |
| `iterated_mean_difference` | `one_cause_contrast__iterated_fit` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9107 to 0.9565, SE ratio 0.9577 | reported |
| `iterated_mean_difference` | `one_cause_contrast__mechanism_correct_difference` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw, with every outcome regression omitting the time-varying covariate and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1188, mean difference -0.0019 to -0.0015 | reported |
| `iterated_mean_difference` | `one_cause_contrast__paired_difference` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1611, mean difference 0.000258 to 0.000957 | reported |
| `iterated_mean_difference` | `one_cause_risk__hazard_fit` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9252 to 0.9667, SE ratio 0.9880 | reported |
| `iterated_mean_difference` | `one_cause_risk__isotonic_difference` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1322, mean difference 0.000245 to 0.000542 | reported |
| `iterated_mean_difference` | `one_cause_risk__iterated_fit` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9165 to 0.9606, SE ratio 0.9654 | reported |
| `iterated_mean_difference` | `one_cause_risk__mechanism_correct_difference` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw, with every outcome regression omitting the time-varying covariate and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0539, mean difference 0.000030 to 0.000129 | reported |
| `iterated_mean_difference` | `one_cause_risk__paired_difference` | diagnostic | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1322, mean difference 0.000245 to 0.000542 | reported |
| `power` | `one_cause_contrast__alternative` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9989 to 1 | pass |
| `root_n_and_efficiency` | `one_cause_contrast__n_1000` | control | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0013, coverage 0.9179 to 0.9616, SE ratio 0.9907 | pass |
| `root_n_and_efficiency` | `one_cause_contrast__n_2000` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0014, coverage 0.9238 to 0.9657, SE ratio 1.0354 | pass |
| `root_n_and_efficiency` | `one_cause_contrast__n_8000` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000398, coverage 0.9282 to 0.9688, SE ratio 0.9980 | pass |
| `root_n_and_efficiency` | `one_cause_risk__n_1000` | control | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.000764, coverage 0.9223 to 0.9647, SE ratio 0.9697 | pass |
| `root_n_and_efficiency` | `one_cause_risk__n_2000` | positive | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000464, coverage 0.9238 to 0.9657, SE ratio 1.0353 | pass |
| `root_n_and_efficiency` | `one_cause_risk__n_8000` | positive | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000511, coverage 0.9165 to 0.9606, SE ratio 0.9766 | pass |
| `root_n_rate` | `one_cause_contrast__empirical_sd` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5401 to -0.4554 | pass |
| `root_n_rate` | `one_cause_contrast__reported_se` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5006 to -0.4953 | pass |
| `root_n_rate` | `one_cause_risk__empirical_sd` | positive | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5399 to -0.4538 | pass |
| `root_n_rate` | `one_cause_risk__reported_se` | positive | the one-cause law under always against never treating, n = 2,000, the risk under always treating by the last visit: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5016 to -0.4981 | pass |
| `simultaneous_coverage` | `competing_curve__pointwise_joint_control` | control | the two-cause law under always against never treating, n = 2,000, every risk and difference of both plans read as one joint set: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.4632 to 0.5006 | pass |
| `simultaneous_coverage` | `competing_curve__simultaneous_band` | positive | the two-cause law under always against never treating, n = 2,000, every risk and difference of both plans read as one joint set: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9346 to 0.9520 | pass |
| `simultaneous_coverage` | `one_cause_curve__pointwise_joint_control` | control | the one-cause law under always against never treating, n = 2,000, every risk and difference of both plans read as one joint set: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5977 to 0.6341 | pass |
| `simultaneous_coverage` | `one_cause_curve__simultaneous_band` | positive | the one-cause law under always against never treating, n = 2,000, every risk and difference of both plans read as one joint set: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9351 to 0.9524 | pass |
| `targeting_necessity` | `one_cause_contrast__targeted` | positive | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the estimator with node hazards and continuations that omit the time-varying covariate and correct treatment and censoring models, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000708 to 0.0029, margin 0.0060 | pass |
| `targeting_necessity` | `one_cause_contrast__untargeted` | control | the one-cause law under always against never treating, n = 2,000, the risk difference at the last visit: the same fit's backward plug-in with every targeting step removed | bias interval must fall entirely outside the margin | bias 0.0988 to 0.1023, margin 0.0059 | pass |
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | the contrast and the regimen's risk of each fit label, with the shrunken-SE and noise controls on the `one_cause` contrast | 4,800 at n = 2,000 | an SE ratio or coverage interval outside its band, or an efficiency ratio outside its band |
| `simultaneous_coverage` | the band over every estimate of the `one_cause` and `competing` fits, node-1 risks included, and its pointwise control | the calibration fits | joint coverage outside the calibration band |
| `double_robustness` | the `one_cause` contrast with both sides correct, the outcome side correct, the mechanism correct, the continuation alone wrong, and both wrong; the `competing` and `three_arm` contrasts with the mechanism correct | 1,200 at n = 2,000; each arm with a wrong learner as its own pilot sets | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `targeting_necessity` | the `one_cause` contrast of the `mechanism_correct` robustness fits, targeted and as the untargeted backward plug-in of the same draws | those fits' count | the targeted bias outside the margin, or the plug-in inside it |
| `root_n_and_efficiency`, `root_n_rate` | the `one_cause` contrast and risk at n = 1,000, 2,000 and 8,000; the n = 1,000 rung is a rate rung that carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the `one_cause` contrast against zero, on the `one_cause` calibration fits | the calibration count | a rejection lower bound below the minimum power |
| every family | `nonconverged_fits`: the fits that stopped short of `tol` with a final relative score at or below $10^{-6}$, beside each cell's count of unexpected warnings | as the cell | a non-converged fit above $10^{-6}$ is a lost replicate under the 0.01 failure bound; more than 1% non-converged fits makes the cell a defect suspect |
| every family | `smallest_kish`, `largest_relative_sd` and `largest_edgeworth_loss`: the influence-curve measures of each row's fit over its estimates | as the cell | nothing: recorded beside the verdicts |
| `iterated_mean_difference` | on `one_cause` and `competing`: the hazard estimate minus the iterated-mean estimate of the same draw, without and with `isotonic=True`, each fit against the truth, and the largest decrease of the iterated-mean curve. On `competing`, each fit's largest sum of the two cumulative incidences minus one. On the `one_cause` `mechanism_correct` draws, the same paired difference with every regression omitting $L$ | 800 at n = 2,000, and the robustness draws | nothing: the family is reported with no verdict |

The calibration count is the arm-pooled SE-ratio rule of `mtp_study_tools.replicates_from_pilot`
over the eight arms, read at the committed calibration pilot of 8,000 draws per fit label, none
held. The pilot's SE ratios run from 0.995 (`one_cause` contrast) to 1.024 (`crossfit`
contrast), with mean 1.004. The rule gives 4,800 replicates, with per-cell and family-wise pass
probabilities 0.994 and 0.954. At 4,800 replicates the shrunken-SE control (standard errors
scaled by 0.7) leaves the SE-ratio band, and the noise control leaves the efficiency band, with
probability above 0.999999. The power cell's design power at n = 2,000 rounds to one. The
necessity cells read the `mechanism_correct` robustness fits and the power cell the
calibration fits, so neither adds a fit set.

The bands read the node-1 risks, whose influence curves had a median smallest Kish count of 19
in the calibration pilots (5% quantile 12 to 13) and a median predicted Wald coverage loss of
0.011. Their losses do not average out of a maximum,
so the `one_cause` and `competing` calibration pilots fit the band, as the declared fits do. A
rule fixed before the pilot is read then requires the declared band cell to pass with
probability at least 0.90 at the pilot's joint coverage plus and minus one standard error. If it
does not, the declaration stops for a diagnosis, and the band is not widened.

The band covered every estimate in 94.65% of the `one_cause` pilot draws (15 estimates) and
95.08% of the `competing` draws (24 estimates), each with a standard error of 0.25 points. The
pointwise intervals covered every estimate at once in 61.6% and 49.8%. At the pilot's joint
coverage minus one standard error, the 99% interval of the declared cell at 4,800 replicates
lies inside the band with probability above 0.9999 on both fit labels, against the 0.90 the
rule requires.

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

The pilot's standardized biases of the targeted contrast, with one standard error added, are
0.061 for the `one_cause` `mechanism_correct` arm (1,000 pooled draws), 0.077 for
`continuation_wrong`, 0.076 for the `competing` arm and 0.037 for the `three_arm` arm, so every
arm takes 1,200 replicates, and so does the targeted necessity arm. The `both_correct` reading
is 0.024. The untargeted plug-ins sit at +4.18, +1.41, +3.22 and +3.59 efficiency SEs from the
truth, against exact-sample limits of +4.23, +1.37, +3.20 and +3.63.

With correctly specified learners the hazard fit and the iterated-mean fit are both efficient,
so they differ by $o_P(n^{-1/2})$. The study reports their agreement, the share of decreasing
iterated-mean curves with its interval, and on `competing` how far each fit's incidences sum
above one. The decrease and coherence rows record zero as a reference, not a truth: their true
values are negative, and the hazard fit's sum is at most one by construction. So only each
reading's share above zero and its largest value are reported, with no floor: X70 reports the agreement of the two fits instead of requiring a gap
(the owner's decision of 2026-11-13).

In the committed pilot of 200 draws per fit label, the standard deviation of the paired
difference, over the efficiency-bound standard error, is 0.148 for the `one_cause` contrast and
0.143 for its risk, and 0.139 and 0.142 on `competing`. No iterated-mean curve decreased in any
of the 400 draws (99% interval for the share 0 to 0.026 per fit label), so `isotonic=True`
changed no estimate and its ratios equal these. On the `one_cause` `mechanism_correct` draws,
with every regression omitting $L$, the ratios are 0.126 for the contrast and 0.047 for the
risk. On `competing` no fit's incidences summed above one in any draw: each fit's largest sum
minus one was at most -0.289, with median -0.341.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 2000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 54 | truth tests passing |
| `independent_tests_total` | 54 | truth tests reported |
| `property_cells_passed` | 34 | property cells passing |
| `property_cells_total` | 34 | property cells reported |
| `max_standardized_bias` | 0.0503 | largest primary standardized bias |
| `min_coverage` | 0.9354 | lowest primary coverage |
| `properties[interval_calibration/one_cause_contrast__correctly_specified]:se_ratio` | 0.9879 | SE ratio of the `one_cause` contrast's calibration cell |
| `properties[interval_calibration/one_cause_contrast__correctly_specified]:coverage` | 0.9496 | coverage of the `one_cause` contrast's calibration cell |
| `properties[interval_calibration/crossfit_contrast__correctly_specified]:se_ratio` | 1.0168 | SE ratio of the cross-fitted contrast calibration cell |
| `properties[interval_calibration/three_arm_contrast__correctly_specified]:se_ratio` | 0.9912 | SE ratio of the policy contrast calibration cell |
| `properties[double_robustness/one_cause_contrast__both_wrong]:standardized_bias` | 4.0347 | standardized bias of the both-wrong control |
| `properties[double_robustness/one_cause_contrast__continuation_wrong]:standardized_bias` | 0.0321 | standardized bias of the wrong-continuation cell |
| `properties[targeting_necessity/one_cause_contrast__untargeted]:standardized_bias` | 4.2648 | standardized bias of the plug-in control |
| `properties[simultaneous_coverage/one_cause_curve__simultaneous_band]:coverage` | 0.9442 | joint coverage of the `one_cause` band |
| `properties[simultaneous_coverage/competing_curve__simultaneous_band]:coverage` | 0.9437 | joint coverage of the `competing` band |
| `properties[iterated_mean_difference/one_cause_contrast__paired_difference]:paired_sd_over_bound` | 0.1611 | paired ratio of the `one_cause` contrast |
| `properties[iterated_mean_difference/competing_contrast__paired_difference]:paired_sd_over_bound` | 0.1382 | paired ratio of the `competing` contrast |
| `properties[iterated_mean_difference/one_cause_risk__mechanism_correct_difference]:paired_sd_over_bound` | 0.0539 | paired ratio of the `one_cause` risk, `mechanism_correct` draws |
| `properties[iterated_mean_difference/one_cause__iterated_curve_decrease]:rejection_rate` | 0 | share of decreasing iterated-mean curves, `one_cause` |
| `properties[iterated_mean_difference/competing__iterated_curve_decrease]:rejection_rate` | 0 | share of decreasing iterated-mean curves, `competing` |
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
| one baseline decision | treatment at several visits beside $L_k$ is part (b2) of plan X70 |
| one scalar binary $L_k$ on Markov laws and finite $W$ | the evidence covers correct or misspecified logistic and cell-mean learners. A flexible learner, or a continuous, multivariate or non-Markov $L_k$, is outside it; the unit tests cover a continuous $L_k$ |
| `targeting="one_step"` only | the per-horizon algorithm is refused on this design (H28) |
| static regimens and one modified treatment policy at visit 1 | plan X70 puts a rule and a known policy on a held design in scope; they rest on `tests/unit/test_hazard_time_varying.py` |
| no observation weights, clusters or `node_bounds=` | each is supported by the fit and covered by the unit tests only |
| the cross-fitted stitch | with parametric learners an in-fold and an out-of-fold stitch have the same limit, so the `crossfit` cells cannot see the stitch; it rests on the unit tests |
| held cells | a held cell costs a replicate; the design test expects 0.05 held fits over the study and its pilot. The held-cell estimate itself rests on `tests/unit/test_hazard_time_varying.py` |
| support rows | the support key's `later_covariates` option, set only on a hazard fit with a block at a node after the first, keeps these fits off `G31` and `G32`. The `G63.held` rows license the in-sample, competing, policy and cross-fitted fits, and the rows on `one_cause` and `competing` license the band |
| no comparator | no external implementation targets these hazards with covariates after baseline |

## Reproduction

Run `python tests/canonical/hazard_time_varying/regenerate.py --output DIR` with an empty
scratch directory outside the repository to regenerate the artifacts. The pilot runs separately,
one label per run, as `python -m tests.studies.hazard_time_varying_pilot --jobs N --labels LABEL`,
and writes `tests/canonical/hazard_time_varying/pilot.csv.gz`. The red-cell readings are
declared in `tests/canonical/hazard_time_varying/regenerate.py` before any run.
