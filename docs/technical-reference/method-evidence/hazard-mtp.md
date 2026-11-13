---
orphan: true
---

# Modified treatment policy on the hazard path

This study validates longitudinal TMLE with `survival_model="hazard"` and
`targeting="one_step"` when a regimen is a modified treatment policy at the baseline decision,
held over every visit. Each policy is targeted by its own walk, whose loss is weighted by
$\omega_k = w\,r(A, W)/G_k(A, W)$ at the observed dose. The risk curve is the plug-in average
of the targeted hazards at each unit's shifted dose. The section on
[hazard-based survival targeting](../longitudinal-tmle.md#hazard-based-survival-targeting)
states the estimand, the influence curve and the exact-law evidence.

The study is declared and has not run. The page is not linked from the index until the run
commits its artifacts and the record is registered, so the three result blocks below are empty.

## The estimand and the laws

For a policy $d$ the target is $\psi_d(t) = E\,F(t \mid d(A, W), W)$, the risk of the event by
visit $t$ when each unit's baseline treatment is moved by $d$. Every law also fits the natural
course (the identity policy) as the reference, and the primary contrast is
$\psi_d(T) - \psi_{\mathrm{natural}}(T)$ at the last visit $T$. Identification needs no
unmeasured confounding of $A$ given $W$, censoring at random given $(A, W)$, and positivity of the
policy (Díaz, Williams, Hoffman and Schenck 2023, Theorem 1 at one decision).

$W_1 \in \{0, 1\}$ and $W_2 \in \{0, \dots, 3\}$ are independent and uniform. The event hazard at
each visit is logistic in the visit, the dose, $W_1$ and $W_2$ (and the arm on `vector`), and the
visit dropout is logistic in the dose and $W$.

| law | visits | treatment | policy | overlap over all eight strata |
| --- | --- | --- | --- | --- |
| `levels` | 5 | a dose on $0, \dots, 5$, $P(A = a \mid W) \propto \exp\{-(a - c(W))^2/8\}$ | one level up, the top stays | level probabilities 0.035 or more; policy ratio at most 3.12 |
| `gaussian` | 5 | $A \mid W \sim N(c(W), 1.5^2)$ | $A + 1$ | policy ratio at most 7.2 over the central 99.9% of each stratum |
| `truncated` | 5 | the `gaussian` dose truncated to $[0, 5.5]$ | $A + 1$ when $A + 1 \le 5.5$, otherwise $A$ | policy ratio at most 5.74 over the whole support |
| `vector` | 1 | a binary arm $F$, logistic in $W$, and $A \mid F, W \sim N(c(W) + 0.5 F, 1.5^2)$ | $A + 1$, the arm kept | arm probabilities 0.35 to 0.69; policy ratio as `gaussian` |

Here $c(W) = 1.5 + W_1 + 0.5 W_2$. Retention through the last visit is at least 0.64. On
`levels` the smallest product of a level probability and that retention is 0.028, above twice
the default cumulative bound of 0.01, so no bound truncates a weight. A vector treatment held
over several visits cannot be declared in the data container, so `vector` has one visit. On
`truncated` a unit whose shifted dose would pass the cap keeps its own dose, so its ratio is
$g(a - 1 \mid W)/g(a \mid W)\,1\{a \le 5.5\} + 1\{a > 4.5\}$.

Plan X70 named the shared `dose` scenario of
[point-treatment survival](point-treatment-survival.md) for `levels`. Its dropout is linear on
the probability scale, so no logistic learner is correct for it, and its hazard leaves the wrong
hazard learner's plug-in contrast about one efficiency standard error from the truth. The study's
laws keep its dose family and replace the dropout and the hazard; they are defined in
`tests/studies/hazard_mtp_law.py`.

Every truth is an exact sum over the dose levels, a Gauss-Hermite quadrature of 120 nodes over
a Gaussian dose, or a Gauss-Legendre quadrature over each smooth piece of the truncated dose.
None of them calls the estimator. Every efficiency bound is the exact standard deviation of the
efficient influence function over the law's support.
`tests/unit/test_hazard_mtp_study_design.py` checks the truths against a backward recursion,
an adaptive quadrature to $10^{-10}$ and a fit on the exact `levels` sample. It checks the bounds
against the cause-specific hazard form of the curve at every support point, against an adaptive
quadrature of its square for both the risk and the contrast. The `ic_gap` pilot label adds one
fit at n = 40,000 per fit label, whose influence-curve SD lies within 2% of the bound, and the
design test reads it from the committed pilot.

## The fits

| fit label | law | ratio | folds |
| --- | --- | --- | --- |
| `levels` | `levels` | the level probabilities | 1 |
| `gaussian_density` | `truncated` | binned density, a fitted truncated-normal location model read at `oracle_bins(n)` bins (320 at n = 2,000) | 1 |
| `gaussian_classifier` | `gaussian` | logistic classifier on `W1`, `W2` and the dose | 1 |
| `gaussian_crossfit` | `gaussian` | the same classifier | 5 |
| `vector` | `vector` | logistic classifier on `W1`, `W2`, the arm and the dose, and a logistic arm model | 1 |
| `untruncated_density` (reported) | `gaussian` | binned density, a fitted Gaussian location model at `oracle_bins(n)` bins | 1 |

| nuisance | correct learner | wrong learner |
| --- | --- | --- |
| hazard | unpenalized logistic regression on `W1`, `W2`, the dose (and the arm), at each visit | logistic regression on the dose (and the arm) alone (`NoWLogistic`) |
| treatment | multinomial logistic on `W1`, `W2`; a truncated-normal location density by maximum likelihood; a logistic classifier | the prior share; a density with no covariates; a classifier on the dose alone |
| censoring | logistic regression on `W1`, `W2` and the dose (and the arm), at each visit | the prior share |

The correct learners are correctly specified and are not saturated. With saturated level and
retention fits on a finite $W$, the targeted contrast of the right and the wrong hazard learner
coincide, so no cell could see the hazard fit. The design test keeps that witness. It also
fits a two-branch randomized policy on the exact `levels` sample with the wrong hazard learner,
which returns the policy's exact risk while the initial hazards miss it.

The density route is gated on `truncated` only. On `gaussian` the policy ratio grows
exponentially in the dose, and the top quantile bins of the shipped binned density are wider
than the shift. A dose and its pre-image then share a bin, and the binned ratio reads one where
the true ratio is 5 to 8. The `ic_gap` pilot label measures each fit's influence-curve gap, the
standard deviation of its influence curve minus the exact efficient influence function, in
units of the efficiency bound, and the design test checks the committed readings:

| fit label | median gap, 5 draws at n = 2,000 | gap at n = 20,000 |
| --- | --- | --- |
| `levels` | 0.21 | 0.05 |
| `gaussian_density` | 0.10 | 0.02 |
| `gaussian_classifier` | 0.08 | 0.03 |
| `gaussian_crossfit` | 0.10 | 0.04 |
| `vector` | 0.08 | 0.03 |
| `untruncated_density` | 0.31 | 0.26 |

The figures are for the contrast. The claim is that every gated fit's gap at least halves from
n = 2,000 to n = 20,000, close to the root-n factor of 0.32. Each also stays within 0.30 and
0.10, loose sanity bounds set after the scratch pilot. The untruncated density route's gap does
not halve, so
its rows are reported with no verdict. The cause is the shipped route's tail binning, not the
hazard walk, and the shipped route's limitation is tracked apart from this study.

The cross-fitted variant exercises the cross-fitting of the shifted-dose predictions, which
does not depend on the ratio route, so one route covers it. The study's density learner is
handed its sample through its constructor, so the classifier carries the cross-fitted cell. No
maintained package targets discrete-time hazards under a modified treatment policy, so the
study names no comparator and its agreement block stays empty.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

The primary rows draw 2,400 samples of 2,000 rows for each of `levels`, `gaussian_density`,
`gaussian_classifier` and `vector` and report every risk and contrast of each fit. The shared
primary rule (`repeated_sizing.primary_replicates`) sets that count from the committed
calibration pilots of these fit labels.

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | the contrast and the policy's risk of each gated fit label, with the shrunken-SE and noise controls on the `levels` and `gaussian_density` contrasts | 4,400 at n = 2,000 | an SE ratio or coverage interval outside its band, or an efficiency ratio outside 0.90 to 1.10 |
| `interval_calibration`, reported | the contrast and the policy's risk of `untruncated_density` | 800 at n = 2,000 | nothing: the rows are reported with no verdict |
| `simultaneous_coverage` | the band over every estimate of the `levels` and `gaussian_density` fits, and its pointwise control | 4,400 at n = 2,000 | joint coverage outside the calibration band |
| `double_robustness` | the `levels` contrast with both, one or neither nuisance correct; the contrast of `gaussian_density`, `gaussian_classifier`, `gaussian_crossfit` and `vector` with the wrong hazard learner and the correct mechanism | 1,200 at n = 2,000 | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `targeting_necessity` | the `levels` contrast from the wrong hazard learner, targeted and plug-in | 1,200 at n = 2,000 | the targeted bias outside the margin, or a displacement below 0.25 |
| `root_n_and_efficiency`, `root_n_rate` | the `levels` contrast and risk at n = 1,000, 2,000 and 8,000; the n = 1,000 rung is a rate rung that carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the `levels` contrast against zero | 800 at n = 2,000 | a rejection lower bound below the minimum power |
| every family | `nonconverged_fits`: the fits that stopped short of `tol` with a final relative score at or below $10^{-6}$, beside each cell's count of unexpected warnings | as the cell | a non-converged fit above $10^{-6}$ is a lost replicate under the 0.01 failure bound; more than 1% non-converged fits makes the cell a defect suspect |
| `iterated_mean_difference` | on `levels` and `gaussian_density`: the hazard estimate minus the iterated-mean estimate of the same draw, without and with `isotonic=True`, each fit against the truth, and the largest decrease of the iterated-mean curve. On the `levels` `mechanism_correct` draws: the same paired difference with both regressions omitting $W$ | 800 at n = 2,000, and the 1,200 robustness draws | nothing: the family is reported with no verdict |

The calibration count is the arm-pooled SE-ratio rule of `mtp_study_tools.smallest_replicates`
over the ten gated arms, read at the committed calibration pilot of 8,000 draws per fit label.
At 4,400 replicates the shrunken-SE
control leaves the SE-ratio band and the noise control leaves the efficiency band with
probability above 0.999.

The wrong hazard learner's plug-in contrast has an exact limit 3.5 efficiency standard errors
from the truth at n = 2,000 on `levels`, 4.8 on `truncated`, 5.5 on `gaussian` and 3.5 on
`vector`, so the plug-in arm of `targeting_necessity` can fail. With both nuisances wrong the
exact `levels` limit lies 3.4 efficiency standard errors from the truth, so the `both_wrong`
control can fail; if it does not clear the margin, the control has failed and the study is a
defect suspect. The power cell's exact-law power at n = 2,000 is above 0.99. At bias zero every
positive bias arm passes with probability above 0.99. That stays true while the true bias is
below 0.108 standard deviations at 1,200 replicates and below 0.077 at 800.

With correctly specified parametric learners the hazard fit and the iterated-mean fit are both
efficient, so on the gated laws they differ by $o_P(n^{-1/2})$. A 200-draw scratch run before the
declaration read the SD of their paired difference at most 4% of an efficiency standard error
for every estimate on `levels` and `gaussian`. With both regressions omitting $W$ and the
mechanism correct, it read at most 2%. No iterated-mean curve decreased in 200 draws per law.
The study reports the committed pilot's ratios and the decreasing share with its interval, with
no floor: X70 reports the agreement of the two fits instead of requiring a gap (the owner's
decision of 2026-11-13).

## Limitations

| limit | what it means for use |
| --- | --- |
| one baseline decision | a policy at several visits is part (a) of plan X70 |
| a vector treatment at one visit | the data container does not hold a vector over several visits |
| finite covariates and low-dimensional learners | the evidence covers correct or misspecified logistic and Gaussian learners. A flexible learner is outside it |
| the density route on a bounded dose | the gated density cells use a truncated dose with a capped shift. On an unbounded Gaussian dose the shipped route's tail bins are wider than the shift, and its reported rows show the miss |
| held cells | every policy here is deterministic and no level is empty at n = 2,000, so no fit holds a cell out. Held cells rest on `tests/unit/test_hazard_mtp.py` |
| no comparator | no external implementation targets these hazards under a policy |

## Reproduction

Run `python tests/canonical/hazard_mtp/regenerate.py --output DIR` with an empty scratch
directory outside the repository to regenerate the artifacts. The pilot runs separately, one
label per run, as `python -m tests.studies.hazard_mtp_pilot --jobs N --labels LABEL`, and writes
`tests/canonical/hazard_mtp/pilot.csv.gz`. The red-cell readings are declared in
`tests/canonical/hazard_mtp/regenerate.py` before any run.
