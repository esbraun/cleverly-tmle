# Continuous-time longitudinal TMLE

The study validates `ContinuousLongitudinalTMLE`, the TMLE of Rytgaard, Gerds and van der Laan
(2022) for a treatment and covariates that change at irregular visits, with no time grid.
[Continuous-time longitudinal TMLE](../continuous-time-longitudinal.md) states the estimator,
its visit cap and its variance. No maintained package implements this construction,
so no canonical implementation is compared. The study rests on Monte Carlo truths, on
repeated-sampling properties, and on a reported pairing with the discrete-time hazard LTMLE on
the same draws.

The declared run at commit 8cead461 passes every gated row: the 48 primary rows, the 12 rows of
`mechanism_correct`, the plug-in control and both calibration rows with their shrunk-SE controls.
One fit of 3,200 was lost to an engine defect that has since been fixed, and the
[red-cell ledger](red-cells.md) accepts the two accounting rows it turns red by a named exception
([failed fits](#failed-fits)). A risk or risk difference of a fit like the one measured reads
supported under the conditions of [limits](#limits).

## The estimand and the laws

For regime $r$ and cause $j$ the target is the risk $P^{G_r}(T \le \tau, J = j)$ at days
$\tau = 10, 20, 30$: the treatment set by the regime at baseline and at every visit, the visits
left as they run under the regime, and censoring removed. The differences are against never
treat. The regimes are always treat, never treat and the rule "treat when $L(s) = 1$", decided at
baseline and at every visit. Identification needs consistency, no unmeasured confounding of
each treatment decision given the observed history, censoring independent of the event process
given the history, and positivity of following the regime
([estimand and identification](../continuous-time-longitudinal.md#estimand-and-identification)).
Law CT reads its history through the last visit only, so the engine's history window holds.

Law CT runs on days $[0, 30]$ with baseline covariates $W_1$ (binary) and $W_2$ (uniform on
$\{1, \dots, 6\}$), a binary covariate $L$ and a binary treatment $A$ that change at visits,
and constant intensities between visits given the history. Its coefficients are in `tests/studies/continuous_time_longitudinal_law.py`.

| process | intensity or law |
| --- | --- |
| visit | $0.07 \exp(0.4 L - 0.3 A)$ per day |
| $L(s)$ at a visit | $\operatorname{expit}(-0.4 + 1.2 L(s^-) - 0.8 A(s^-) + 0.4 W_1)$ |
| $A(s)$ at a visit | $\operatorname{expit}(-1.5 + 1.0 L(s) + 2.0 A(s^-) + 0.2 W_2 / 6)$ |
| cause 1 | $0.010 \exp(-0.6 A + 0.8 L + 0.3 W_1)$ per day |
| cause 2 | $0.006 \exp(0.2 A + 0.5 L - 0.3 W_1 + 0.1 W_2)$ per day |
| censoring | $0.008 \exp(0.6 L - 0.4 A)$ per day |

The module gives the laws of $L(0)$ and $A(0)$ too. Law CT-day is the same law on whole days,
with each day's events in the order death, visit, censoring, so every day carries ties and the
engine's tie rule acts. The first design's visit rate was 0.10 and its visit-treatment intercept $-1.0$; a
truth-only check before the declaration lowered them because the 99th percentile of never's true
inverse weight was 41.5, above the bound of 20. The law now gives about 0.065 visits per day of
follow-up, about two over 30 days in place of the first design's three. Each truth is a Monte Carlo risk
on 10,000,000 units per law and regime.

## The visit cap

The fit models each regime through its first $K^*_r$ visits, so it targets $\psi^{(K^*_r)}$: the
regime at baseline and at those visits, and the observed treatment law after them. The truth
every gate reads is the full-regime risk $\psi_r$, which stays the target of inference, so the
measured bias includes the cap's gap. On this law at n = 1,000 the engine's cap is 4 for about 90%
of fits and 3 or 5 for the rest, and its mean gap is at most 0.0017 in absolute value, at
never's cause-1 day-30 risk and the cause-1 day-30 difference (about 0.03 of the true-weight IPW
SD). Each row records its regime's cap and the capped truth $\psi^{(K^*_r)}$, simulated for every
cap from 1 to 8, so a red row can be read against both. The estimated share of followers past the
cap reached 0.178 in the gated fits, above the 0.05 at which the fit warns, so the warning fires
on many default fits of this law, while the gap itself stays below 0.002. In the declared run the
cap was 3, 4 or 5 in every fit, and the mean gap was at most 0.0019 in absolute value.

## The fit

The fit is the shipped default: the default libraries, cross-fitted with 10 outer folds and
selection inside 5 inner folds, and the influence-curve variance, at n = 1,000.

| cell | law | regimes | rows | count | role |
| --- | --- | --- | --- | --- | --- |
| `day_cf` | CT-day | always, never, rule | 30 (12 rule rows) | 3,200 | gated |
| `cont_cf` | CT | always, never | 18 | 3,200 | gated |
| `mechanism_correct` | CT-day | always, never | 12 (days 20 and 30) | 2,400 | gated |
| plug-in control | CT-day | always, never | 4 day-30 risks | 2,400 | gated control |
| calibration | both | always, never | difference of cause 1 at day 20 in each cell | 3,200 | gated |
| `outcome_correct`, `both_wrong` | CT-day | always, never | 12 | 800 each | reported |
| `sparse_calibration` | CT-day | always, never | difference of cause 2 at day 10 | 3,200 | reported |
| X70 pairing | CT-day panel | always, never | 18 | 400 | reported |

The primary rule gates every row of `day_cf`, `cont_cf` and `mechanism_correct`. A row passes when
its coverage interval lies above 0.90, the 99% interval of its SE ratio lies inside (0.8, 1.2), and
its bias is within 0.2 SD of zero. `mechanism_correct` fits the cause hazards, the visit intensity
and the continuation without $L$ and the treatment and censoring models with it, so its rows test
double robustness. That needs the default models to hold the law's mechanism: the default treatment
model gives baseline coefficients of its own, so it holds the baseline and the visit treatment
models, and the Cox main-terms model holds the censoring hazard, which is constant in time. The
cell's untargeted plug-in is the control, which must miss by more than 0.2 SD at day 30 for always
or never.

The calibration rows take the tighter rule: the SE ratio inside (0.93, 1.07) and coverage inside
(0.92, 0.98). The shrunk-SE control multiplies their SEs by 0.85 and must fall below that band or
wholly below the row's own SE-ratio interval. The second reading carries a calibration row that
reads conservative, with an SE-ratio interval above 1.

## The pairing with the discrete-time LTMLE

The first 400 `day_cf` draws are refitted by `LTMLE(survival_model="hazard")` on a 9-node panel
whose nodes end at days 3, 6, 10, 13, 16, 20, 23, 26 and 30. The nodes end at every target day,
but the estimands differ: a visit inside a node changes $L$ and the treatment, which the panel
sees only at the next node start, where its regime resets the treatment. The panel therefore
identifies a coarsened risk, simulated on 10,000,000 units per regime, which differs from
$\psi_r$ by up to 0.0054 at a risk and 0.0085 at the cause-1 day-30 difference. The pairing
publishes the 18 static rows with no verdict: each engine's bias against its own truth, the paired
difference against its expected value (the panel's risk minus $\psi_r$), and the SE ratio. A
paired difference that departs from its expected value by more than 0.5 of the continuous
engine's SE starts a diagnosis of both engines on the same draws.

The first design paired 1,000 draws on a 30-node daily panel. It changed before any run, because a
30-node fit did not finish in 3,000 s, which by itself projects past the study's hold of 200
CPU-hours. The rule rows are not paired. When the discrete fit reads a constant
node, it withholds that row's interval (`constant_node_plugin`). The point estimate still pairs,
the discrete fit's SE and coverage read only the draws with an interval, and the share without
one is reported.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause1 @ t=10]` | difference in the risk of cause 1 by day 10 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0014 to 0.000872 | 0.9519 | 1.0096 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause1 @ t=20]` | difference in the risk of cause 1 by day 20 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0013 to 0.0018 | 0.9534 | 1.0273 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause1 @ t=30]` | difference in the risk of cause 1 by day 30 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000574 to 0.0045 | 0.9456 | 0.9996 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause2 @ t=10]` | difference in the risk of cause 2 by day 10 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000942 to 0.000992 | 0.9506 | 1.0084 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause2 @ t=20]` | difference in the risk of cause 2 by day 20 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0019 to 0.000964 | 0.9422 | 0.9764 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause2 @ t=30]` | difference in the risk of cause 2 by day 30 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0032 to 0.000318 | 0.9431 | 0.9805 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause1 @ t=10]` | risk of cause 1 by day 10 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000250 to 0.000913 | 0.9556 | 1.0173 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause1 @ t=20]` | risk of cause 1 by day 20 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000245 to 0.0018 | 0.9609 | 1.0503 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause1 @ t=30]` | risk of cause 1 by day 30 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000907 to 0.0029 | 0.9609 | 1.0411 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause2 @ t=10]` | risk of cause 2 by day 10 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000291 to 0.000996 | 0.9597 | 1.0198 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause2 @ t=20]` | risk of cause 2 by day 20 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000320 to 0.0015 | 0.9553 | 1.0069 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause2 @ t=30]` | risk of cause 2 by day 30 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000539 to 0.0016 | 0.9584 | 1.0328 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause1 @ t=10]` | risk of cause 1 by day 10 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000383 to 0.0016 | 0.9522 | 1.0057 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause1 @ t=20]` | risk of cause 1 by day 20 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000555 to 0.0022 | 0.9481 | 1.0140 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause1 @ t=30]` | risk of cause 1 by day 30 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0023 to 0.0011 | 0.9425 | 0.9827 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause2 @ t=10]` | risk of cause 2 by day 10 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000399 to 0.0011 | 0.9503 | 0.9885 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause2 @ t=20]` | risk of cause 2 by day 20 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000055 to 0.0022 | 0.9278 | 0.9396 | pass |
| law CT in continuous time: a binary covariate and treatment that change at visits, two causes and censoring that depend on both; always and never; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause2 @ t=30]` | risk of cause 2 by day 30 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000587 to 0.0034 | 0.9281 | 0.9304 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause1 @ t=10]` | difference in the risk of cause 1 by day 10 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000755 to 0.0015 | 0.9463 | 0.9920 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause1 @ t=20]` | difference in the risk of cause 1 by day 20 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0013 to 0.0018 | 0.9475 | 0.9974 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause1 @ t=30]` | difference in the risk of cause 1 by day 30 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000909 to 0.0048 | 0.9406 | 0.9814 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause2 @ t=10]` | difference in the risk of cause 2 by day 10 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0014 to 0.000484 | 0.9494 | 0.9992 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause2 @ t=20]` | difference in the risk of cause 2 by day 20 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0020 to 0.000688 | 0.9528 | 1.0219 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[always vs never, cause2 @ t=30]` | difference in the risk of cause 2 by day 30 between treatment at every visit and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0031 to 0.000310 | 0.9487 | 0.9967 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[rule vs never, cause1 @ t=10]` | difference in the risk of cause 1 by day 10 between the rule that treats when the last covariate is 1 and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000580 to 0.0015 | 0.9425 | 0.9726 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[rule vs never, cause1 @ t=20]` | difference in the risk of cause 1 by day 20 between the rule that treats when the last covariate is 1 and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000588 to 0.0026 | 0.9400 | 0.9609 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[rule vs never, cause1 @ t=30]` | difference in the risk of cause 1 by day 30 between the rule that treats when the last covariate is 1 and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.0021 to 0.0061 | 0.9287 | 0.9635 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[rule vs never, cause2 @ t=10]` | difference in the risk of cause 2 by day 10 between the rule that treats when the last covariate is 1 and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0010 to 0.000614 | 0.9428 | 0.9919 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[rule vs never, cause2 @ t=20]` | difference in the risk of cause 2 by day 20 between the rule that treats when the last covariate is 1 and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0011 to 0.0014 | 0.9431 | 0.9899 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `ate_regimen[rule vs never, cause2 @ t=30]` | difference in the risk of cause 2 by day 30 between the rule that treats when the last covariate is 1 and no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0011 to 0.0025 | 0.9391 | 0.9371 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause1 @ t=10]` | risk of cause 1 by day 10 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000207 to 0.000954 | 0.9559 | 1.0087 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause1 @ t=20]` | risk of cause 1 by day 20 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000329 to 0.0013 | 0.9534 | 1.0226 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause1 @ t=30]` | risk of cause 1 by day 30 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000244 to 0.0022 | 0.9531 | 1.0255 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause2 @ t=10]` | risk of cause 2 by day 10 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000243 to 0.0010 | 0.9522 | 1.0113 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause2 @ t=20]` | risk of cause 2 by day 20 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000208 to 0.0015 | 0.9613 | 1.0403 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[always, cause2 @ t=30]` | risk of cause 2 by day 30 under treatment at every visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000464 to 0.0017 | 0.9563 | 1.0326 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause1 @ t=10]` | risk of cause 1 by day 10 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000967 to 0.000961 | 0.9422 | 0.9935 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause1 @ t=20]` | risk of cause 1 by day 20 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0012 to 0.0016 | 0.9484 | 0.9787 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause1 @ t=30]` | risk of cause 1 by day 30 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.0033 to 0.000062 | 0.9334 | 0.9610 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause2 @ t=10]` | risk of cause 2 by day 10 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000159 to 0.0016 | 0.9450 | 0.9830 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause2 @ t=20]` | risk of cause 2 by day 20 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000286 to 0.0024 | 0.9491 | 0.9914 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[never, cause2 @ t=30]` | risk of cause 2 by day 30 under no treatment at any visit | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000682 to 0.0033 | 0.9356 | 0.9699 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[rule, cause1 @ t=10]` | risk of cause 1 by day 10 under the rule that treats when the last covariate is 1 | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | -0.000248 to 0.0012 | 0.9500 | 0.9833 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[rule, cause1 @ t=20]` | risk of cause 1 by day 20 under the rule that treats when the last covariate is 1 | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000130 to 0.0023 | 0.9419 | 0.9742 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[rule, cause1 @ t=30]` | risk of cause 1 by day 30 under the rule that treats when the last covariate is 1 | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.0010 to 0.0039 | 0.9303 | 0.9372 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[rule, cause2 @ t=10]` | risk of cause 2 by day 10 under the rule that treats when the last covariate is 1 | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000021 to 0.0013 | 0.9447 | 0.9740 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[rule, cause2 @ t=20]` | risk of cause 2 by day 20 under the rule that treats when the last covariate is 1 | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.000466 to 0.0025 | 0.9394 | 0.9651 | pass |
| law CT on whole days: a binary covariate and treatment that change at visits, two causes and censoring that depend on both, events recorded by day so that ties occur every day; always, never and the rule 'treat when L = 1'; the default libraries, ten outer folds and five inner folds, n = 1,000, target days 10, 20 and 30 | `cif_regimen[rule, cause2 @ t=30]` | risk of cause 2 by day 30 under the rule that treats when the last covariate is 1 | `cleverly` ContinuousLongitudinalTMLE: the TMLE of Rytgaard, Gerds and van der Laan (2022) for a binary treatment and covariates that change at irregular visits, one backward pass per regime, with the influence-curve standard error | 0.0014 to 0.0040 | 0.9313 | 0.9462 | pass |
<!-- /generated -->

Every primary row passes, the 12 rule rows among them. Coverage runs from 0.928 to 0.961, no row
reads conservative, the SE ratio runs from 0.930 to 1.050, and the largest standardized bias is
0.092 SD, at a rule row. `cont_cf` holds 3,199 fits, one fewer than declared
([failed fits](#failed-fits)).

| reading | result |
| --- | --- |
| `mechanism_correct`, 12 rows at 2,400 fits | every row passes: coverage 0.936 to 0.959, SE ratio 0.959 to 1.052, standardized bias at most 0.088 SD |
| plug-in control | fires on all four day-30 risks: the untargeted plug-in misses by -0.92 and -0.29 SD under never and 0.61 and 0.39 SD under always, against the 0.2 SD it must exceed |
| calibration, `day_cf` | coverage 0.9475 (0.937 to 0.957), SE ratio 0.997 (0.965 to 1.032): passes |
| calibration, `cont_cf` | coverage 0.9534 (0.943 to 0.962), SE ratio 1.027 (0.996 to 1.060): passes |
| shrunk-SE controls | SE ratios 0.848 and 0.873, upper ends 0.877 and 0.901, below the band's 0.93 in each cell |
| `sparse_calibration` (reported) | coverage 0.949, SE ratio 0.999 (0.968 to 1.032) |
| `outcome_correct` (reported) | standardized bias at most 0.071 SD, coverage 0.924 to 0.974, SE ratio 0.921 to 1.078 |
| `both_wrong` (reported) | standardized bias up to 0.84 SD, coverage down to 0.849 |
| X70 pairing (reported) | no row starts a diagnosis: the largest departure from the expected paired difference is 0.13 of the continuous engine's SE, against 0.5 |

All five rows the sizing exposed pass, so no declared route was taken (`exposed-rows.csv`). Each
interval in the table is a 99% interval.
`calibration.csv`, `mechanism-correct.csv`, `misspecified.csv`, `ltmle-pairing.csv` and
`visit-cap.csv` hold the readings.

## Failed fits

One fit was lost: `cont_cf` replicate 936 raised `ValueError` from scikit-learn, because one inner
cross-validation fold of the default continuation learner held no event although its whole
training set did. The learner now predicts the constant label in such a fold, as its outer guard
already did, so the fit succeeds; the change acts only where the old code raised, so every
completed fit is unchanged. A loss of 1 in 3,200 is inside the declared bound of 1%, and every
truth row passes with the lost fit imputed at its worst case. The study's own audits count a lost
fit as a failing row of the calibration rule and of the shrunk-SE control, so
`failure-accounting.csv` reads those two audits red. Imputing the lost fit at any combination of
the cell's smallest, mean or largest estimate with its smallest or largest SE keeps both
readings: calibration coverage 0.9531 to 0.9534, SE ratio 1.0255 to 1.0276 with its 99% interval
at most 0.9937 to 1.0618, and the shrunk control's SE-ratio upper end at most 0.9025. The
[red-cell ledger](red-cells.md) accepts the two accounting rows by a named exception keyed to
this run's replicates.

## Sizing

The counts are declared: 3,200 draws for `day_cf` and `cont_cf` and 2,400 for
`mechanism_correct`, raised from the design values of 2,400 and 1,600 so that the rule rows
could stay gated. All 60 primary rows are gated, the 12 rule rows among them, and no row moves to
reported for a low pilot coverage.

The pilot runs 400 draws of each of `day_cf`, `cont_cf`, `mechanism_correct` and
`outcome_correct`, and chooses no count. It reads each gated row at its group's pooled coverage
and SE ratio, the groups being `cont_cf` static, `day_cf` static, `day_cf` rule and
`mechanism_correct`, and at the row's own standardized bias. A product of 60 single-row readings
from 400 draws each would be pessimistic, since each row's coverage carries an SE near 0.011. A
group mean of signed biases would let opposite biases cancel.

Read this way, five rows pass below 0.96 at the declared counts: the rule regime's cause-1 risks
at days 10, 20 and 30, and in `mechanism_correct` the cause-1 day-30 risk of never and difference
of always minus never. They stay gated with a declared route, given
under [Red-cell route](#red-cell-route). A declared run is refused unless the product of the other
55 rows' pass probabilities reaches 0.95. On the second pilot it is 0.988; over all 60 rows it is
0.544, and with pooled biases 0.997.

At the pilot's seconds per fit the declared run costs 159.5 CPU-hours, below the declared hold of
200, and runs as three shards of 53.2 CPU-hours.
`tests/canonical/continuous_time_longitudinal/pilot-diagnosis.md` holds the measurements and
every count's pass as a report.

A first pilot failed the family rule: on static rows the estimated SE ran 7 to 8% above the
Monte Carlo spread at every n. The default treatment model was one logistic fit pooled over
baseline and later visits, which cannot hold law CT's baseline mechanism beside its visit
mechanism. The engine now gives baseline coefficients of its own, and on refits of the first 20
pilot draws the influence curve's spread is within 1% of the exact efficient one on static rows.
The first pilot's primary draws were dropped, and a second pilot ran on the fixed engine.

The plug-in control was checked before any replicate. On one draw of 200,000 units, the plug-in
limit of `mechanism_correct`'s outcome side misses the day-30 risk of cause 1 by 0.58 true-weight
IPW SDs under never and by 0.53 under always, above the required 0.5, so the control can
fire.

## Red-cell route

`tests/canonical/continuous_time_longitudinal/regenerate.py` declares the readings before any
run. A red calibration row is a finite-sample red only when its SE-ratio interval contains 1, its
coverage is consistent with 0.95 and no defect is found. One exception is declared: an SE-ratio
interval above 1.07 with coverage at or above 0.95 reads conservative, the direction the
influence-curve variance takes when the treatment model is right and the continuation is
approximated. A red bias at a day-30 row of never, or at a difference against never, is first read
against the capped truth at the fit's cap: a row green against it is a cap red, governed by the
share condition of the support labels, not an engine defect suspect. A red rule row keeps only
rules experimental. A plug-in control that does not fire, or a shrunk-SE control that fails its
reading, voids the run.

The five rows the sizing names stay gated, each with a foreseen failure mode. The three rule rows carry the rule levels' positive cause-1 bias at n = 1,000, about
0.12 to 0.13 SD in both pilots, which targeting does not remove and whose cause is not shown.
`mechanism_correct`'s day-30 risk of never carries the visit cap's gap (-0.112 SD against the full
risk, -0.076 against the capped truth). Its day-30 difference reads +0.134 SD, or +0.094 against
the capped truth, which is the always arm's +0.07 SD less the never arm's -0.08 SD. The two
mechanism rows can therefore turn red together.

If one of these rows is red on bias alone, with its coverage and SE ratio passing, the record
still registers and the row's claim stays experimental: rule regimes, or double robustness under
a correct treatment model. Any other red, including a coverage or SE-ratio red on one of them,
fails the record. `exposed-rows.csv` records each row's verdict and route. A routed row would
have gone to an owner row of the roadmap, and a routed rule row, red in the published table, to a
named exception of the [red-cell ledger](red-cells.md).

In the declared run no gated row is red and none of the five exposed rows took its route, so no
claim is held back and no owner row exists. The only red rows are the two accounting rows of the
lost fit ([failed fits](#failed-fits)).

The support constants of the labels are read after the run from the per-row table `floors.csv`,
by rules declared before it. Every gated row passed, the rule rows among them, so rules are
licensed beside static regimes. The largest median `variance_rel_sd` of a gated row is 0.224, the
gated rows' median event shares run from 0.092 (cause 2 at day 10) to 0.244 (cause 1 at day 30),
the fewest followers of a regime at risk at day 30 in any gated fit is 49, the largest share past
the cap is 0.178, and the largest weight-floor share is 0.0018. No hazard or visit jump was
bounded. [Labels](../continuous-time-longitudinal.md#labels) states how each constant enters the
support rule.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 3200 | primary replications |
| `n` | 1000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 48 | truth tests passing |
| `independent_tests_total` | 48 | truth tests reported |
| `property_cells_total` | 0 | repeated-sampling property cells, none declared |
| `subject_failures` | 1 | primary fits lost to a recordable fitting failure |
| `max_standardized_bias` | 0.0923 | largest primary standardized bias |
| `min_coverage` | 0.9278 | lowest primary coverage |
| `min_se_ratio_ci_lower` | 0.9010 | lowest lower end of a primary SE-ratio interval |
| `max_se_ratio_ci_upper` | 1.0853 | highest upper end of a primary SE-ratio interval |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:standardized_bias` | 0.2000 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations. No comparator reads it |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound. No comparator reads it |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound. No comparator reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound. No comparator reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 32 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| one law, in tied and untied form | the evidence covers law CT and its daily version, two causes and three target times fitted jointly. A fit with another number of causes reads experimental |
| the default libraries, cross-fitted, n = 1,000 | a fit with any other library or learner, in sample, with a setting away from its default, or with fewer than 1,000 units reads experimental |
| the measured ranges | an estimate reads supported only inside the event shares, influence-curve sparsity, followers, share past the cap and weight-floor share the gated fits reached, with no bounded jump ([labels](../continuous-time-longitudinal.md#labels)) |
| static regimes and one known rule | any declared rule reads supported under the same conditions, but the study measured only "treat when $L(s) = 1$" |
| risks and risk differences | ratios, smooth contrasts and the band are measured by no study and read experimental |
| no comparator | no maintained implementation of this construction was located; the discrete-time pairing gates nothing |

## Reproduction

Run the pilot first, as
`python -m tests.studies.continuous_time_longitudinal_pilot --jobs N --kinds primary --resume`.
Run `python -m tests.canonical.continuous_time_longitudinal.regenerate --output DIR` with an empty
scratch directory outside the repository to regenerate the artifacts. It runs no reference
process, because the study has no comparator, and it refuses until the committed pilot agrees
with the declaration.
