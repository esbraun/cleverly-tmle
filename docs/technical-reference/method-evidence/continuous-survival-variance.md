---
orphan: true
---

# Continuous-time survival switched variance

This study validates the default standard error of `ContinuousSurvivalTMLE`, the switched
variance, and decides whether it stays the default. The switched variance is the influence-curve
variance, raised to the predictable variation of the curve's martingale part when the curve is
sparse. The section on [inference](../continuous-time-survival.md#inference) states the
estimator, its switch and the exact-law evidence.

The study is declared and has not run. The page is not linked from the index until the run
commits its artifacts and the record is registered, so the three result blocks below are empty.

## The estimand and the laws

The targets are each arm's risk of each cause, $F_j(t \mid a) = E\,F_j(t \mid a, X)$, and each
risk difference, at the three quartiles of the observed event times. Identification needs no
unmeasured confounding of the treatment given $X$, censoring independent of the event times given
$(A, X)$, and positivity of the treatment and of remaining uncensored through the last time.

Each cell changes one thing in the base law of the
[continuous-time survival study](continuous-time-survival.md), so that no cell repeats a law a
registered study measures:

| cell | law | n | fit | what it stresses |
| --- | --- | --- | --- | --- |
| `v1_rare_cause2` | cause-2 log-hazard intercept $-2.6$ in place of $-2$; times 0.52, 1.29, 2.69 | 1,000 | cross-fitted | a rarer cause: about 9.5 treated cause-2 events by the first quartile |
| `v2_heavy_weights` | treatment model $\omega = (2, -1, 1.5)$ | 1,000 | cross-fitted | heavier control weights; the risks do not move |
| `v3_half_n` | base | 500 | cross-fitted | half the events; published only, and the size floor stays 1,000 |
| `v4_in_sample` | base | 1,000 | in sample | where the influence-curve standard error runs lowest |
| `v7_quadratic` | $0.3\,(X_3^2 - 1)$ added to the cause-1 log hazard | 1,000 | cross-fitted | a hazard no default Cox model holds |

Every hazard stays constant given $(Z, X)$, so each truth is a quadrature over $X_3$, and the
efficient standard deviation has a closed form, in
`tests/studies/continuous_survival_variance_law.py`. The rare cause's target times are that law's
event-time quartiles on a fixed draw of 2,000,000; its median reads 1.29.

## The fit

Each cell fits the default libraries with 10 outer folds and selection inside 5 inner folds, as
the registered study does. Every interval is the one the library ships: the switched standard
error, range-logit on $(0, 1)$ for a risk and identity for a difference. Each fit also records,
for every target, the influence-curve and predictable-variation standard errors, the relative
standard deviation the switch reads (`variance_rel_sd`) and the oracle standard error, so every
candidate interval is read on the same fits. No comparator is named, so the agreement block stays
empty.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

The twelve targets at the median and upper quartile of `v1_rare_cause2`, `v4_in_sample` and
`v7_quadratic` are gated by the primary rule: coverage above 0.90 and below 0.99, the SE ratio
inside (0.8, 1.2) and the bias within 0.2 SD. `v2_heavy_weights` and `v3_half_n` are reported
cells: their rows publish the same readings with no verdict, enter the `variance_rel_sd` floor,
and stay in the ship rule. V3 does not move the size floor.
The six first-quartile targets of each fit are read under the calibration rule (SE ratio inside
(0.93, 1.07), coverage inside (0.92, 0.98), both at 99%) and are no gate.

The split was made from the pilot, before any primary run. In 1,000 pilot draws the switched
interval of `v2_heavy_weights` covered control cause 1 at time 1.12 at 0.894 with an SE ratio of
0.88 (the influence curve's covered 0.874), the cause-1 difference there at 0.907 and treated
cause 2 at 0.915, and with V2 the sizing rule found no count up to 50,000. `v3_half_n` lost 12 of
1,000 fits to `DataError`, against a bound of 10, and with it the rule asked for 10,800. Those
refusals came from a loss of digits in the clever covariate where the fitted survival is tiny;
the stable form now fits those draws, and the pilot reruns at that code before the declaration.

A reported cell's failure share is published beside its rows and fails nothing. Roadmap item
[X6 (a2)](../../roadmap.md#x6-continuous-time-survival-and-competing-risks) owns both reported
cells. A fit with fewer than 1,000 units reads experimental under the size floor
(`MEASURED_N`) until (a2) moves it. No label detects heavy weights: such a fit's estimates read
experimental only where their `variance_rel_sd` lies above the floor, and (a2) owns measuring
that configuration.

The pilot sets the count before the run: the larger of the shared primary rule on the gated
cells and 3,600, the count at which the 30 first-quartile
rows pass the calibration rule together with probability 0.977 at an SE ratio of 1. Each cell
runs 6,000 replicates, where a row whose true bias is 0.15 SD passes the bias gate with
probability 0.90. The cap is 7,200; a pilot that asks for more stops the declaration, and the run
is never cut to the cap. The pilot also decides the V7 coefficient (0.3, or 0.15 if the
influence-curve interval covers a cause-1 target at the median or upper quartile below 0.93) and
whether V7 is gated: if the untargeted plug-in of no cause-1 risk at the median is biased by more
than 0.2 SD, V7 leaves the record and feeds neither the ship rule nor the floors.

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

The record declares no property family. Its decision and its floors are computed from the primary
rows and committed as `ship-rule.csv`, `candidate-readings.csv` and `floors.csv`:

| reading | rule | what it decides |
| --- | --- | --- |
| harm | at no row does the switched SE ratio's interval lie wholly above 1.07, or its coverage interval wholly above 0.98, while the influence curve's does not | with benefit, whether the switched variance stays the default |
| benefit | at one first-quartile row at least, the paired 99% bootstrap interval of $\lvert\mathrm{cov}_{IC} - 0.95\rvert - \lvert\mathrm{cov}_S - 0.95\rvert$ lies above 0 | as above |
| controls | the predictable variation times 1.3 must fire the harm clause; a switch that never fires must fail the benefit clause; the standard error times 0.85 must fail the calibration rule at every gated row | the run is void if one does not behave so |
| reported controls | the gated rows the primary rule turns red at 0.85 times the standard error (about 0.96 per row), and the cells whose sparsest first-quartile influence-curve row passes the calibration rule | nothing; the first is the merge gate's own sensitivity, the second says whether a cell exercises the sparse-cell failure |
| paired readings | each candidate's coverage minus the influence curve's, and minus the influence-curve interval widened to the candidate's mean width in its row, each with its paired 99% bootstrap interval | nothing; the evidence of roadmap item X6 (a2) |
| floors | the largest `variance_rel_sd` rung at which every gated row passes; the event-share floor (0.40, or 0.25 if every first-quartile row passes); the size floor stays 1,000 (`v3_half_n` is published only) | which risks and risk differences read `supported` |

At 3,600 replicates the harm clause passes with probability 0.997 or more when every true switched
row is inside the band, and fires with probability 0.89 at a true SE ratio of 1.12. The benefit
clause passes with probability 1.000 (X6 plan, section 5.1). The declared count is at least 3,600,
so these powers are lower bounds for the harm clause's detection and the benefit clause.

If the rule fails, the switched variance does not ship: the default returns to the
influence-curve variance, the floors read the influence curve's rows, and the record's rows on the
switched interval are reported, not gated.

## Limitations

| limit | what it means for use |
| --- | --- |
| two causes and a binary treatment | one event type and three causes are measured by their own study |
| the base law's family | the cells change one feature each of one law |
| no comparator | no external implementation reports this variance |

## Reproduction

Run the pilot first, as `python -m tests.studies.continuous_survival_variance_pilot --jobs N`.
Run `python -m tests.canonical.continuous_survival_variance.regenerate --output DIR` with an
empty scratch directory outside the repository to regenerate the artifacts. The red-cell readings,
the pilot's checks and the budget are declared in
`tests/canonical/continuous_survival_variance/regenerate.py` before any run.
