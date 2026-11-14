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
| `v3_half_n` | base | 500 | cross-fitted | half the events; decides whether the size floor falls to 500 |
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

The twelve targets at the median and upper quartile of each cell are gated by the primary rule:
coverage above 0.90 and below 0.99, the SE ratio inside (0.8, 1.2) and the bias within 0.2 SD.
The six first-quartile targets of each fit are read under the calibration rule (SE ratio inside
(0.93, 1.07), coverage inside (0.92, 0.98), both at 99%) and are no gate. Each cell runs 3,600
replicates. At an SE ratio of 1 and coverage 0.95 a calibration row then passes with probability
0.9992; the pilot can extend a cell up to 7,200.

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
| controls | the predictable variation times 1.3 must fire the harm clause; a switch that never fires must fail the benefit clause; the standard error times 0.85 must fail every gated row | the run is void if one does not behave so |
| width-matched readings | each candidate's coverage against the influence-curve interval widened to the same mean width in its row | nothing; the evidence of roadmap item X6 (a2) |
| floors | the largest `variance_rel_sd` rung at which every gated row passes; the event-share floor (0.40, or 0.25 if every first-quartile row passes); the size floor (1,000, or 500 if every `v3_half_n` row passes) | which risks and risk differences read `supported` |

At 3,600 replicates the harm clause passes with probability 0.997 or more when every true switched
row is inside the band, and fires with probability 0.89 at a true SE ratio of 1.12. The benefit
clause passes with probability 1.000 (X6 plan, section 5.1).

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
