---
orphan: true
---

# Continuous-time longitudinal TMLE

**Pending.** This study is declared and has not run. Its pilot and its declared run are pending,
so it holds no verdict, and every estimate of `ContinuousLongitudinalTMLE` reads experimental.
The page states the design fixed before any run. When the declared run's artifacts are committed,
the change that registers the study fills its tables from the record and adds the page to the
evidence index.

The study validates `ContinuousLongitudinalTMLE`, the TMLE of Rytgaard, Gerds and van der Laan
(2022) for a treatment and covariates that change at irregular visits, with no time grid.
[Continuous-time longitudinal TMLE](../continuous-time-longitudinal.md) states the estimator,
its visit cap and its variance. No maintained package implements this construction, so no
canonical implementation is compared. The study rests on Monte Carlo truths, on repeated-sampling
properties, and on a reported pairing with the discrete-time hazard LTMLE on the same draws.

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
engine's tie rule acts. The plan's visit rate was 0.10 and its visit-treatment intercept $-1.0$; a
truth-only check before the declaration lowered them because the 99th percentile of never's true
inverse weight was 41.5, above the bound of 20. The law now gives about 0.065 visits per day of
follow-up, about two over 30 days in place of the plan's three. Each truth is a Monte Carlo risk
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
cap reached 0.093 in the smoke, above the 0.05 at which the fit warns, so the warning fires on
many default fits of this law, while the gap itself stays below 0.002.

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

This is a deviation from the plan, declared before any run. The plan paired 1,000 draws on a
30-node daily panel, but a 30-node fit did not finish in 3,000 s, which by itself projects past the
plan's hold of 200 CPU-hours. The rule rows are not paired. When the discrete fit reads a constant
node, it withholds that row's interval (`constant_node_plugin`). The point estimate still pairs,
the discrete fit's SE and coverage read only the draws with an interval, and the share without
one is reported.

## Accuracy against known truth

Pending. The declared run fills this section.

## Sizing

The counts are declared: 3,200 draws for `day_cf` and `cont_cf` and 2,400 for
`mechanism_correct`, fixed by Eric's choice on 2026-11-25 against plan 5.7's design values of
2,400 and 1,600. All 60 primary rows are gated, the 12 rule rows among them, and no row moves to
reported for a low pilot coverage.

The pilot runs 400 draws of each of `day_cf`, `cont_cf`, `mechanism_correct` and
`outcome_correct`, and chooses no count. A declared run is refused unless the family pass at the
declared counts, the product of the 60 rows' pass probabilities, reaches 0.95. Each row is read
at its group's pooled coverage, SE ratio and standardized bias, the groups being `cont_cf`
static, `day_cf` static, `day_cf` rule and `mechanism_correct`. A product of 60 single-row
readings from 400 draws each would be pessimistic, since each row's coverage carries an SE near
0.011.

The second pilot's pooled readings put that pass at 0.997. A group mean of signed biases lets
opposite biases cancel, so the pilot also reads each row at its own bias with the pooled coverage
and SE ratio. That pass is 0.544: five rows pass below 0.96 on their own bias, the rule regime's
cause-1 risks and the cause-1 day-30 risk of never and difference of always minus never in
`mechanism_correct`. How the declaration treats those rows is pending.

At the pilot's seconds per fit the declared run costs 159.5 CPU-hours, below the plan's hold of
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
IPW SDs under never and by 0.53 under always, above the 0.5 the plan requires, so the control can
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

The support constants of the labels (the largest `variance_rel_sd` rung, the event-share window,
the fewest followers and the largest share past the cap) are read after the run from the
per-row table `floors.csv`, by rules declared before it.

## Limits

| limit | what it means for use |
| --- | --- |
| one law, in tied and untied form | the evidence will cover law CT and its daily version at n = 1,000 |
| the default libraries | a fit with any other library or learner reads experimental |
| static regimes and one known rule | other rules are measured by no study |
| no comparator | no maintained implementation of this construction was located; the discrete-time pairing gates nothing |

## Reproduction

Run the pilot first, as
`python -m tests.studies.continuous_time_longitudinal_pilot --jobs N --kinds primary --resume`.
Run `python -m tests.canonical.continuous_time_longitudinal.regenerate --output DIR` with an empty
scratch directory outside the repository to regenerate the artifacts. It runs no reference
process, because the study has no comparator, and it refuses until the committed pilot agrees
with the declaration.
