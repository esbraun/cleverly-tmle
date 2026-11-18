# Modified treatment policy on the hazard path

This study validates longitudinal TMLE with `survival_model="hazard"` and
`targeting="one_step"` when a regimen is a modified treatment policy at the baseline decision,
held over every visit. Each policy is targeted by its own walk, whose loss is weighted by
$\omega_k = w\,r(A, W)/G_k(A, W)$ at the observed dose. The risk curve is the plug-in average
of the targeted hazards at each unit's shifted dose. The section on
[hazard-based survival targeting](../longitudinal-tmle.md#hazard-based-survival-targeting)
states the estimand, the influence curve and the exact-law evidence.

The declared run passed every gated verdict, with no failed fit: 48 of 48 primary rows and 39
of 39 property cells with a verdict. The `iterated_mean_difference` rows and the reported
`untruncated_density` rows carry no verdict.

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
| treatment | multinomial logistic on `W1`, `W2`; a truncated-normal location density by maximum likelihood; a logistic classifier | the prior share, on the six-level law only: the fit sets of the other laws keep the treatment and censoring models correct |
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
maintained package targets discrete-time hazards under a modified treatment policy,
so no canonical implementation is compared.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `ate_regimen[shift vs natural @ t=1]` | the risk by visit 1 under the policy minus the risk under the natural course | `cleverly` | -0.000118 to 0.000358 | 0.9496 | 0.9970 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `ate_regimen[shift vs natural @ t=2]` | the risk by visit 2 under the policy minus the risk under the natural course | `cleverly` | -0.000163 to 0.000477 | 0.9450 | 0.9843 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `ate_regimen[shift vs natural @ t=3]` | the risk by visit 3 under the policy minus the risk under the natural course | `cleverly` | -0.000416 to 0.000325 | 0.9442 | 0.9803 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `ate_regimen[shift vs natural @ t=4]` | the risk by visit 4 under the policy minus the risk under the natural course | `cleverly` | -0.000600 to 0.000208 | 0.9433 | 0.9816 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `ate_regimen[shift vs natural @ t=5]` | the risk by visit 5 under the policy minus the risk under the natural course | `cleverly` | -0.000646 to 0.000210 | 0.9404 | 0.9838 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[natural @ t=1]` | the risk of the event by visit 1 under the natural course | `cleverly` | -0.000325 to 0.000426 | 0.9525 | 1.0153 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[natural @ t=2]` | the risk of the event by visit 2 under the natural course | `cleverly` | -0.000571 to 0.000430 | 0.9546 | 1.0002 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[natural @ t=3]` | the risk of the event by visit 3 under the natural course | `cleverly` | -0.000748 to 0.000379 | 0.9533 | 1.0088 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[natural @ t=4]` | the risk of the event by visit 4 under the natural course | `cleverly` | -0.000933 to 0.000275 | 0.9529 | 1.0031 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[natural @ t=5]` | the risk of the event by visit 5 under the natural course | `cleverly` | -0.0010 to 0.000231 | 0.9529 | 1.0001 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[shift @ t=1]` | the risk of the event by visit 1 under the policy | `cleverly` | -0.000220 to 0.000561 | 0.9500 | 1.0009 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[shift @ t=2]` | the risk of the event by visit 2 under the policy | `cleverly` | -0.000456 to 0.000629 | 0.9471 | 0.9846 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[shift @ t=3]` | the risk of the event by visit 3 under the policy | `cleverly` | -0.000861 to 0.000401 | 0.9454 | 0.9987 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[shift @ t=4]` | the risk of the event by visit 4 under the policy | `cleverly` | -0.0012 to 0.000187 | 0.9463 | 0.9823 | pass |
| the same Gaussian dose law, the policy's ratio from a logistic classifier | `risk_regimen[shift @ t=5]` | the risk of the event by visit 5 under the policy | `cleverly` | -0.0014 to 0.000155 | 0.9433 | 0.9781 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `ate_regimen[shift vs natural @ t=1]` | the risk by visit 1 under the policy minus the risk under the natural course | `cleverly` | -0.000127 to 0.000423 | 0.9496 | 0.9947 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `ate_regimen[shift vs natural @ t=2]` | the risk by visit 2 under the policy minus the risk under the natural course | `cleverly` | -0.000199 to 0.000535 | 0.9450 | 0.9811 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `ate_regimen[shift vs natural @ t=3]` | the risk by visit 3 under the policy minus the risk under the natural course | `cleverly` | -0.000248 to 0.000575 | 0.9450 | 1.0017 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `ate_regimen[shift vs natural @ t=4]` | the risk by visit 4 under the policy minus the risk under the natural course | `cleverly` | -0.000367 to 0.000525 | 0.9475 | 0.9972 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `ate_regimen[shift vs natural @ t=5]` | the risk by visit 5 under the policy minus the risk under the natural course | `cleverly` | -0.000478 to 0.000456 | 0.9471 | 0.9975 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[natural @ t=1]` | the risk of the event by visit 1 under the natural course | `cleverly` | -0.000229 to 0.000537 | 0.9521 | 0.9957 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[natural @ t=2]` | the risk of the event by visit 2 under the natural course | `cleverly` | -0.000470 to 0.000521 | 0.9504 | 1.0115 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[natural @ t=3]` | the risk of the event by visit 3 under the natural course | `cleverly` | -0.000582 to 0.000540 | 0.9554 | 1.0126 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[natural @ t=4]` | the risk of the event by visit 4 under the natural course | `cleverly` | -0.000649 to 0.000557 | 0.9521 | 1.0054 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[natural @ t=5]` | the risk of the event by visit 5 under the natural course | `cleverly` | -0.000677 to 0.000581 | 0.9450 | 0.9924 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[shift @ t=1]` | the risk of the event by visit 1 under the policy | `cleverly` | -0.000120 to 0.000723 | 0.9442 | 0.9954 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[shift @ t=2]` | the risk of the event by visit 2 under the policy | `cleverly` | -0.000378 to 0.000765 | 0.9467 | 0.9941 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[shift @ t=3]` | the risk of the event by visit 3 under the policy | `cleverly` | -0.000518 to 0.000803 | 0.9475 | 1.0062 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[shift @ t=4]` | the risk of the event by visit 4 under the policy | `cleverly` | -0.000691 to 0.000756 | 0.9479 | 1.0100 | pass |
| five visits, one cause, a Gaussian baseline dose truncated to [0, 5.5], logistic visit dropout; the policy adds one to the dose up to 5.5, its ratio from a binned truncated Gaussian density | `risk_regimen[shift @ t=5]` | the risk of the event by visit 5 under the policy | `cleverly` | -0.000830 to 0.000712 | 0.9558 | 1.0060 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `ate_regimen[shift vs natural @ t=1]` | the risk by visit 1 under the policy minus the risk under the natural course | `cleverly` | -0.000292 to 0.000189 | 0.9500 | 0.9955 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `ate_regimen[shift vs natural @ t=2]` | the risk by visit 2 under the policy minus the risk under the natural course | `cleverly` | -0.000348 to 0.000283 | 0.9513 | 0.9855 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `ate_regimen[shift vs natural @ t=3]` | the risk by visit 3 under the policy minus the risk under the natural course | `cleverly` | -0.000392 to 0.000306 | 0.9504 | 1.0055 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `ate_regimen[shift vs natural @ t=4]` | the risk by visit 4 under the policy minus the risk under the natural course | `cleverly` | -0.000339 to 0.000407 | 0.9496 | 1.0055 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `ate_regimen[shift vs natural @ t=5]` | the risk by visit 5 under the policy minus the risk under the natural course | `cleverly` | -0.000456 to 0.000336 | 0.9413 | 0.9829 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[natural @ t=1]` | the risk of the event by visit 1 under the natural course | `cleverly` | -0.000439 to 0.000325 | 0.9525 | 1.0173 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[natural @ t=2]` | the risk of the event by visit 2 under the natural course | `cleverly` | -0.000550 to 0.000454 | 0.9496 | 1.0103 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[natural @ t=3]` | the risk of the event by visit 3 under the natural course | `cleverly` | -0.000571 to 0.000558 | 0.9500 | 1.0132 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[natural @ t=4]` | the risk of the event by visit 4 under the natural course | `cleverly` | -0.000723 to 0.000464 | 0.9587 | 1.0238 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[natural @ t=5]` | the risk of the event by visit 5 under the natural course | `cleverly` | -0.000900 to 0.000323 | 0.9533 | 1.0190 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[shift @ t=1]` | the risk of the event by visit 1 under the policy | `cleverly` | -0.000510 to 0.000292 | 0.9496 | 1.0068 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[shift @ t=2]` | the risk of the event by visit 2 under the policy | `cleverly` | -0.000622 to 0.000462 | 0.9513 | 1.0063 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[shift @ t=3]` | the risk of the event by visit 3 under the policy | `cleverly` | -0.000683 to 0.000583 | 0.9487 | 1.0037 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[shift @ t=4]` | the risk of the event by visit 4 under the policy | `cleverly` | -0.000781 to 0.000589 | 0.9513 | 1.0154 | pass |
| five visits, one cause, a baseline dose on six levels with a multinomial-logit mechanism, logistic visit dropout; the policy raises the dose one level | `risk_regimen[shift @ t=5]` | the risk of the event by visit 5 under the policy | `cleverly` | -0.0011 to 0.000375 | 0.9496 | 1.0152 | pass |
| one visit, a binary arm beside a Gaussian dose, dropout at the visit; the joint policy adds one to the dose and keeps the arm, its ratio from a logistic classifier | `ate_regimen[shift vs natural @ t=1]` | the risk by visit 1 under the policy minus the risk under the natural course | `cleverly` | -0.000206 to 0.000508 | 0.9400 | 0.9727 | pass |
| one visit, a binary arm beside a Gaussian dose, dropout at the visit; the joint policy adds one to the dose and keeps the arm, its ratio from a logistic classifier | `risk_regimen[natural @ t=1]` | the risk of the event by visit 1 under the natural course | `cleverly` | -0.000553 to 0.000522 | 0.9529 | 1.0080 | pass |
| one visit, a binary arm beside a Gaussian dose, dropout at the visit; the joint policy adds one to the dose and keeps the arm, its ratio from a logistic classifier | `risk_regimen[shift @ t=1]` | the risk of the event by visit 1 under the policy | `cleverly` | -0.000451 to 0.000722 | 0.9542 | 1.0101 | pass |
<!-- /generated -->

The primary rows draw 2,400 samples of 2,000 rows for each of `levels`, `gaussian_density`,
`gaussian_classifier` and `vector` and report every risk and contrast of each fit. The shared
primary rule (`repeated_sizing.primary_replicates`) sets that count from the committed
calibration pilots of these fit labels.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `gaussian_classifier_contrast__mechanism_correct` | positive | the Gaussian dose law under a shift of one, classifier ratio, the risk difference against the natural course at the last visit: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000837 to 0.000373, margin 0.0020, SE ratio 1.0359 | pass |
| `double_robustness` | `gaussian_crossfit_contrast__mechanism_correct` | positive | the Gaussian dose law under a shift of one, classifier ratio, five outer folds, the risk difference against the natural course at the last visit: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0011 to -0.000039, margin 0.0021, SE ratio 1.0360 | pass |
| `double_robustness` | `gaussian_density_contrast__mechanism_correct` | positive | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the risk difference against the natural course at the last visit: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000848 to 0.000467, margin 0.0022, SE ratio 1.0572 | pass |
| `double_robustness` | `levels_contrast__both_correct` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000617 to 0.000523, margin 0.0019, SE ratio 0.9653 | pass |
| `double_robustness` | `levels_contrast__both_wrong` | control | the six-level dose law under one level up, the risk difference against the natural course at the last visit: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0245 to 0.0256, margin 0.0019, SE ratio 1.0040 | pass |
| `double_robustness` | `levels_contrast__mechanism_correct` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000686 to 0.000424, margin 0.0019, SE ratio 1.0443 | pass |
| `double_robustness` | `levels_contrast__outcome_correct` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000443 to 0.000617, margin 0.0018, SE ratio 0.9959 | pass |
| `double_robustness` | `vector_contrast__mechanism_correct` | positive | the one-visit arm-and-dose law under a joint shift of the dose by one, classifier ratio, the risk difference against the natural course at the last visit: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000436 to 0.000556, margin 0.0017, SE ratio 1.0152 | pass |
| `interval_calibration` | `gaussian_classifier_contrast__correctly_specified` | positive | the Gaussian dose law under a shift of one, classifier ratio, the risk difference against the natural course at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9399 to 0.9573, SE ratio 0.9881 to 1.0457, empirical efficiency ratio 0.9583 to 1.0135, reported efficiency ratio 0.9991 to 1.0051 | pass |
| `interval_calibration` | `gaussian_classifier_risk__correctly_specified` | positive | the Gaussian dose law under a shift of one, classifier ratio, the policy's risk by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9370 to 0.9547, SE ratio 0.9686 to 1.0237, empirical efficiency ratio 0.9780 to 1.0338, reported efficiency ratio 1.0002 to 1.0027 | pass |
| `interval_calibration` | `gaussian_crossfit_contrast__correctly_specified` | positive | the Gaussian dose law under a shift of one, classifier ratio, five outer folds, the risk difference against the natural course at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9414 to 0.9585, SE ratio 0.9730 to 1.0278, empirical efficiency ratio 0.9837 to 1.0391, reported efficiency ratio 1.0078 to 1.0139 | pass |
| `interval_calibration` | `gaussian_crossfit_risk__correctly_specified` | positive | the Gaussian dose law under a shift of one, classifier ratio, five outer folds, the policy's risk by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9390 to 0.9564, SE ratio 0.9789 to 1.0342, empirical efficiency ratio 0.9721 to 1.0272, reported efficiency ratio 1.0044 to 1.0069 | pass |
| `interval_calibration` | `gaussian_density_contrast__correctly_specified` | positive | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the risk difference against the natural course at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9399 to 0.9573, SE ratio 0.9707 to 1.0258, empirical efficiency ratio 0.9726 to 1.0277, reported efficiency ratio 0.9961 to 0.9990 | pass |
| `interval_calibration` | `gaussian_density_contrast__noise_control` | control | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the risk difference against the natural course at the last visit: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8164 to 0.8457, SE ratio 0.6872 to 0.7261, empirical efficiency ratio 1.3741 to 1.4518, reported efficiency ratio 0.9961 to 0.9990 | pass |
| `interval_calibration` | `gaussian_density_contrast__shrunken_se_control` | control | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the risk difference against the natural course at the last visit: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8110 to 0.8406, SE ratio 0.6797 to 0.7174, empirical efficiency ratio 0.9737 to 1.0272, reported efficiency ratio 0.6973 to 0.6993 | pass |
| `interval_calibration` | `gaussian_density_risk__correctly_specified` | positive | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the policy's risk by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9382 to 0.9558, SE ratio 0.9734 to 1.0284, empirical efficiency ratio 0.9712 to 1.0258, reported efficiency ratio 0.9978 to 0.9994 | pass |
| `interval_calibration` | `levels_contrast__correctly_specified` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9373 to 0.9549, SE ratio 0.9782 to 1.0331, empirical efficiency ratio 0.9842 to 1.0396, reported efficiency ratio 1.0151 to 1.0186 | pass |
| `interval_calibration` | `levels_contrast__noise_control` | control | the six-level dose law under one level up, the risk difference against the natural course at the last visit: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8175 to 0.8467, SE ratio 0.6940 to 0.7324, empirical efficiency ratio 1.3883 to 1.4654, reported efficiency ratio 1.0151 to 1.0186 | pass |
| `interval_calibration` | `levels_contrast__shrunken_se_control` | control | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8185 to 0.8476, SE ratio 0.6850 to 0.7232, empirical efficiency ratio 0.9843 to 1.0392, reported efficiency ratio 0.7106 to 0.7131 | pass |
| `interval_calibration` | `levels_risk__correctly_specified` | positive | the six-level dose law under one level up, the policy's risk by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9471 to 0.9633, SE ratio 0.9908 to 1.0471, empirical efficiency ratio 0.9596 to 1.0141, reported efficiency ratio 1.0041 to 1.0055 | pass |
| `interval_calibration` | `untruncated_density_contrast__reported` | diagnostic | the untruncated Gaussian dose law under a shift of one, binned density ratio, whose tail bins are wider than the shift, the risk difference against the natural course at the last visit: the density route's interval on the untruncated Gaussian law, whose quantile tail bins are wider than the shift | no verdict; coverage, SE ratio and efficiency ratio are reported | coverage 0.9267 to 0.9677, SE ratio 0.9717 to 1.0979, empirical efficiency ratio 0.9629 to 1.0870, reported efficiency ratio 1.0424 to 1.0710 | reported |
| `interval_calibration` | `untruncated_density_risk__reported` | diagnostic | the untruncated Gaussian dose law under a shift of one, binned density ratio, whose tail bins are wider than the shift, the policy's risk by the last visit: the density route's interval on the untruncated Gaussian law, whose quantile tail bins are wider than the shift | no verdict; coverage, SE ratio and efficiency ratio are reported | coverage 0.9267 to 0.9677, SE ratio 0.9502 to 1.0703, empirical efficiency ratio 0.9547 to 1.0748, reported efficiency ratio 1.0159 to 1.0274 | reported |
| `interval_calibration` | `vector_contrast__correctly_specified` | positive | the one-visit arm-and-dose law under a joint shift of the dose by one, classifier ratio, the risk difference against the natural course at the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9343 to 0.9524, SE ratio 0.9529 to 1.0071, empirical efficiency ratio 0.9877 to 1.0445, reported efficiency ratio 0.9916 to 0.9990 | pass |
| `interval_calibration` | `vector_risk__correctly_specified` | positive | the one-visit arm-and-dose law under a joint shift of the dose by one, classifier ratio, the policy's risk by the last visit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9377 to 0.9554, SE ratio 0.9690 to 1.0234, empirical efficiency ratio 0.9753 to 1.0306, reported efficiency ratio 0.9966 to 1.0006 | pass |
| `iterated_mean_difference` | `gaussian_density__iterated_curve_decrease` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the iterated-mean fit of the same plans, n = 2,000: the largest decrease of the iterated-mean risk curve of either arm between consecutive visits | no verdict; the share of draws whose curve decreases somewhere is reported | decreasing share 0, 99% interval 0 to 0.0066 | reported |
| `iterated_mean_difference` | `gaussian_density_contrast__hazard_fit` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the risk difference against the natural course at the last visit: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9297 to 0.9698, SE ratio 1.0111 | reported |
| `iterated_mean_difference` | `gaussian_density_contrast__isotonic_difference` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the risk difference against the natural course at the last visit: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0275, mean difference -0.000013 to 0.000031 | reported |
| `iterated_mean_difference` | `gaussian_density_contrast__iterated_fit` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the risk difference against the natural course at the last visit: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9311 to 0.9708, SE ratio 1.0107 | reported |
| `iterated_mean_difference` | `gaussian_density_contrast__paired_difference` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the risk difference against the natural course at the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0275, mean difference -0.000013 to 0.000031 | reported |
| `iterated_mean_difference` | `gaussian_density_risk__hazard_fit` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the policy's risk by the last visit: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9311 to 0.9708, SE ratio 1.0336 | reported |
| `iterated_mean_difference` | `gaussian_density_risk__isotonic_difference` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the policy's risk by the last visit: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0168, mean difference -0.000018 to 0.000027 | reported |
| `iterated_mean_difference` | `gaussian_density_risk__iterated_fit` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the policy's risk by the last visit: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9311 to 0.9708, SE ratio 1.0337 | reported |
| `iterated_mean_difference` | `gaussian_density_risk__paired_difference` | diagnostic | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, the policy's risk by the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0168, mean difference -0.000018 to 0.000027 | reported |
| `iterated_mean_difference` | `levels__iterated_curve_decrease` | diagnostic | the six-level dose law under one level up, the iterated-mean fit of the same plans, n = 2,000: the largest decrease of the iterated-mean risk curve of either arm between consecutive visits | no verdict; the share of draws whose curve decreases somewhere is reported | decreasing share 0, 99% interval 0 to 0.0066 | reported |
| `iterated_mean_difference` | `levels_contrast__hazard_fit` | diagnostic | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9136 to 0.9585, SE ratio 1.0077 | reported |
| `iterated_mean_difference` | `levels_contrast__isotonic_difference` | diagnostic | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0312, mean difference -0.000022 to 0.000020 | reported |
| `iterated_mean_difference` | `levels_contrast__iterated_fit` | diagnostic | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9194 to 0.9627, SE ratio 1.0130 | reported |
| `iterated_mean_difference` | `levels_contrast__mechanism_correct_difference` | diagnostic | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw, with both outcome regressions omitting W and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0167, mean difference -0.000017 to 9.284e-07 | reported |
| `iterated_mean_difference` | `levels_contrast__paired_difference` | diagnostic | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0312, mean difference -0.000022 to 0.000020 | reported |
| `iterated_mean_difference` | `levels_risk__hazard_fit` | diagnostic | the six-level dose law under one level up, the policy's risk by the last visit: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9311 to 0.9708, SE ratio 1.0266 | reported |
| `iterated_mean_difference` | `levels_risk__isotonic_difference` | diagnostic | the six-level dose law under one level up, the policy's risk by the last visit: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0245, mean difference -0.000041 to 0.000022 | reported |
| `iterated_mean_difference` | `levels_risk__iterated_fit` | diagnostic | the six-level dose law under one level up, the policy's risk by the last visit: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9311 to 0.9708, SE ratio 1.0256 | reported |
| `iterated_mean_difference` | `levels_risk__mechanism_correct_difference` | diagnostic | the six-level dose law under one level up, the policy's risk by the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw, with both outcome regressions omitting W and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0187, mean difference -0.000012 to 0.000027 | reported |
| `iterated_mean_difference` | `levels_risk__paired_difference` | diagnostic | the six-level dose law under one level up, the policy's risk by the last visit: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0245, mean difference -0.000041 to 0.000022 | reported |
| `power` | `levels_contrast__alternative` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `levels_contrast__n_1000` | control | the six-level dose law under one level up, the risk difference against the natural course at the last visit: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000625, coverage 0.9252 to 0.9667, SE ratio 1.0178 | pass |
| `root_n_and_efficiency` | `levels_contrast__n_2000` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000190, coverage 0.9252 to 0.9667, SE ratio 0.9895 | pass |
| `root_n_and_efficiency` | `levels_contrast__n_8000` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000035, coverage 0.9282 to 0.9688, SE ratio 0.9859 | pass |
| `root_n_and_efficiency` | `levels_risk__n_1000` | control | the six-level dose law under one level up, the policy's risk by the last visit: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000387, coverage 0.9238 to 0.9657, SE ratio 0.9979 | pass |
| `root_n_and_efficiency` | `levels_risk__n_2000` | positive | the six-level dose law under one level up, the policy's risk by the last visit: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000560, coverage 0.9297 to 0.9698, SE ratio 0.9720 | pass |
| `root_n_and_efficiency` | `levels_risk__n_8000` | positive | the six-level dose law under one level up, the policy's risk by the last visit: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000019, coverage 0.9107 to 0.9565, SE ratio 0.9790 | pass |
| `root_n_rate` | `levels_contrast__empirical_sd` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5458 to -0.4532 | pass |
| `root_n_rate` | `levels_contrast__reported_se` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5157 to -0.5106 | pass |
| `root_n_rate` | `levels_risk__empirical_sd` | positive | the six-level dose law under one level up, the policy's risk by the last visit: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5403 to -0.4526 | pass |
| `root_n_rate` | `levels_risk__reported_se` | positive | the six-level dose law under one level up, the policy's risk by the last visit: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5047 to -0.5026 | pass |
| `simultaneous_coverage` | `gaussian_density_curve__pointwise_joint_control` | control | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, every risk and risk difference of both curves read as one joint set: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6413 to 0.6783 | pass |
| `simultaneous_coverage` | `gaussian_density_curve__simultaneous_band` | positive | the Gaussian dose law truncated to [0, 5.5] under a shift of one capped at 5.5, binned density ratio, every risk and risk difference of both curves read as one joint set: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9412 to 0.9583 | pass |
| `simultaneous_coverage` | `levels_curve__pointwise_joint_control` | control | the six-level dose law under one level up, every risk and risk difference of both curves read as one joint set: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6542 to 0.6909 | pass |
| `simultaneous_coverage` | `levels_curve__simultaneous_band` | positive | the six-level dose law under one level up, every risk and risk difference of both curves read as one joint set: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9424 to 0.9593 | pass |
| `targeting_necessity` | `levels_contrast__targeted` | positive | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000586 to 0.000514, margin 0.0018 | pass |
| `targeting_necessity` | `levels_contrast__untargeted` | control | the six-level dose law under one level up, the risk difference against the natural course at the last visit: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0250 to 0.0260, margin 0.0016 | pass |
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | the contrast and the policy's risk of each gated fit label, with the shrunken-SE and noise controls on the `levels` and `gaussian_density` contrasts | 4,400 at n = 2,000 | an SE ratio or coverage interval outside its band, or an efficiency ratio outside 0.90 to 1.10 |
| `interval_calibration`, reported | the contrast and the policy's risk of `untruncated_density` | 800 at n = 2,000 | nothing: the rows are reported with no verdict |
| `simultaneous_coverage` | the band over every estimate of the `levels` and `gaussian_density` fits, and its pointwise control | 4,400 at n = 2,000 | joint coverage outside the calibration band |
| `double_robustness` | the `levels` contrast with both, one or neither nuisance correct; the contrast of `gaussian_density`, `gaussian_classifier`, `gaussian_crossfit` and `vector` with the wrong hazard learner and the correct mechanism | 1,200 at n = 2,000; 1,600 for `gaussian_crossfit` | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `targeting_necessity` | the `levels` contrast from the wrong hazard learner, targeted and plug-in | 1,200 at n = 2,000 | the targeted bias outside the margin, or a displacement below 0.25 |
| `root_n_and_efficiency`, `root_n_rate` | the `levels` contrast and risk at n = 1,000, 2,000 and 8,000; the n = 1,000 rung is a rate rung that carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the `levels` contrast against zero | 800 at n = 2,000 | a rejection lower bound below the minimum power |
| every family | `nonconverged_fits`: the fits that stopped short of `tol` with a final relative score at or below $10^{-6}$, beside each cell's count of unexpected warnings | as the cell | a non-converged fit above $10^{-6}$ is a lost replicate under the 0.01 failure bound; more than 1% non-converged fits makes the cell a defect suspect |
| `iterated_mean_difference` | on `levels` and `gaussian_density`: the hazard estimate minus the iterated-mean estimate of the same draw, without and with `isotonic=True`, each fit against the truth, and the largest decrease of the iterated-mean curve. On the `levels` `mechanism_correct` draws: the same paired difference with both regressions omitting $W$ | 800 at n = 2,000, and the 1,200 robustness draws | nothing: the family is reported with no verdict |

The calibration count is the arm-pooled SE-ratio rule of `mtp_study_tools.smallest_replicates`
over the ten gated arms, read at the committed calibration pilot of 8,000 draws per fit label.
At 4,400 replicates the shrunken-SE control leaves the SE-ratio band and the noise control leaves the efficiency band with
probability above 0.999.

The wrong hazard learner's plug-in contrast has an exact limit 3.5 efficiency standard errors
from the truth at n = 2,000 on `levels`, 4.8 on `truncated`, 5.5 on `gaussian` and 3.5 on
`vector`, so the plug-in arm of `targeting_necessity` can fail. With both nuisances wrong the
exact `levels` limit lies 3.4 efficiency standard errors from the truth, so the `both_wrong`
control can fail; if it does not clear the margin, the control has failed and the study is a
defect suspect. The power cell's exact-law power at n = 2,000 is above 0.99. At bias zero every
positive bias arm passes with probability above 0.99. That stays true while the true bias is
below 0.108 standard deviations at 1,200 replicates and below 0.077 at 800.

The arms with the wrong hazard learner and the correct mechanism carry a finite-sample bias of
their own, so their size comes from the committed wrong-learner pilot of 800 draws per fit label
at n = 2,000. Each arm takes 1,200 replicates, or the smallest multiple of 400 that passes with
probability 0.99 at the pilot's standardized bias plus one standard error, whichever is larger.
That rule was fixed before the pilot was read. On `levels` the reading pools the 200
`mechanism_correct` paired draws, which fit the same estimator: bias 0.036 standard deviations
over 1,000 draws. The density route reads 0.028, the classifier route -0.027, the cross-fitted
classifier route -0.076 and `vector` -0.004, so `gaussian_crossfit` takes 1,600 replicates and
the others 1,200. At n = 8,000 the density route reads -0.022 ± 0.050 over 400 draws, no larger
than at n = 2,000, as a finite-sample remainder and not a binning bias predicts.

With correctly specified parametric learners the hazard fit and the iterated-mean fit are both
efficient, so on the gated laws they differ by $o_P(n^{-1/2})$. A 200-draw scratch run before the
declaration read the SD of their paired difference at most 4% of an efficiency standard error
for every estimate on `levels` and `gaussian`. With both regressions omitting $W$ and the
mechanism correct, it read at most 2%. No iterated-mean curve decreased in 200 draws per law.
The study reports the committed pilot's ratios and the decreasing share with its interval, with
no floor: X70 reports the agreement of the two fits instead of requiring a gap (the owner's
decision of 2026-11-13).

The declared run reads the same picture on 800 draws per law and on the 1,200 `levels`
`mechanism_correct` draws. The paired ratio is 0.0312 for the `levels` contrast and 0.0275 for
the `gaussian_density` contrast, and at most 0.0187 on the `mechanism_correct` draws. The hazard
and iterated-mean fits read SE ratios within 0.006 of each other on every estimate, and no
iterated-mean curve decreased in 800 draws on either law, so the `isotonic=True` difference
equals the plain one.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 2000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 48 | truth tests passing |
| `independent_tests_total` | 48 | truth tests reported |
| `property_cells_passed` | 39 | property cells passing |
| `property_cells_total` | 39 | property cells reported |
| `max_standardized_bias` | 0.0420 | largest primary standardized bias |
| `min_coverage` | 0.9400 | lowest primary coverage |
| `properties[interval_calibration/levels_contrast__correctly_specified]:se_ratio` | 1.0046 | SE ratio of the `levels` contrast's calibration cell |
| `properties[interval_calibration/levels_contrast__correctly_specified]:coverage` | 0.9466 | coverage of the `levels` contrast's calibration cell |
| `properties[interval_calibration/gaussian_density_contrast__correctly_specified]:se_ratio` | 0.9973 | SE ratio of the density route's contrast calibration cell |
| `properties[interval_calibration/gaussian_crossfit_contrast__correctly_specified]:se_ratio` | 0.9999 | SE ratio of the cross-fitted contrast calibration cell |
| `properties[interval_calibration/untruncated_density_contrast__reported]:coverage` | 0.9500 | coverage of the reported untruncated density route's contrast |
| `properties[double_robustness/levels_contrast__both_wrong]:standardized_bias` | 3.3688 | standardized bias of the both-wrong control |
| `properties[double_robustness/gaussian_crossfit_contrast__mechanism_correct]:standardized_bias` | -0.0692 | standardized bias of the cross-fitted wrong-hazard cell |
| `properties[targeting_necessity/levels_contrast__untargeted]:standardized_bias` | 3.8846 | standardized bias of the plug-in control |
| `properties[simultaneous_coverage/levels_curve__simultaneous_band]:coverage` | 0.9514 | joint coverage of the `levels` band |
| `properties[simultaneous_coverage/gaussian_density_curve__simultaneous_band]:coverage` | 0.9502 | joint coverage of the `gaussian_density` band |
| `properties[iterated_mean_difference/levels_contrast__paired_difference]:paired_sd_over_bound` | 0.0312 | paired ratio of the `levels` contrast |
| `properties[iterated_mean_difference/gaussian_density_contrast__paired_difference]:paired_sd_over_bound` | 0.0275 | paired ratio of the `gaussian_density` contrast |
| `properties[iterated_mean_difference/levels_risk__mechanism_correct_difference]:paired_sd_over_bound` | 0.0187 | paired ratio of the `levels` risk, `mechanism_correct` draws |
| `properties[iterated_mean_difference/levels__iterated_curve_decrease]:rejection_rate` | 0 | share of decreasing iterated-mean curves, `levels` |
| `properties[iterated_mean_difference/gaussian_density__iterated_curve_decrease]:rejection_rate` | 0 | share of decreasing iterated-mean curves, `gaussian_density` |
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
