---
orphan: true
---

# Continuous-time survival restricted means and risk ratios

This study validates the restricted means and risk ratios of `ContinuousSurvivalTMLE`: the fit
with `restricted_horizons=(2.32,)`, whose walk targets each arm's restricted mean time lost
(RMTL) to each cause beside the risks, and the risk ratios every fit reports. The section on
[restricted means, ratios and the curve](../continuous-time-survival.md#restricted-means-ratios-and-the-curve)
states the estimands, the influence curves and the exact-law evidence.

The study is declared and has not run. The page is not linked from the index until the run
commits its artifacts and the record is registered, so the three result blocks below are empty.

## The estimand and the law

For arm $a$ and cause $j$ the targets are the risk $F_j(t \mid a) = E\,F_j(t \mid a, X)$, the
risk ratio $F_j(t \mid 1) / F_j(t \mid 0)$, the RMTL $E \int_0^\tau F_j(u \mid a, X)\,du$ and the
restricted mean survival time (RMST) $\tau - \sum_j \mathrm{RMTL}_j$, with the differences of the
two arms. Identification needs no unmeasured confounding of the treatment given $X$, censoring
independent of the event times given $(A, X)$, and positivity of the treatment and of remaining
uncensored through $\tau$.

The law is the base law of the [continuous-time survival study](continuous-time-survival.md):
two causes with constant cause-specific hazards given $(Z, X)$, an arm-specific term that only
the Cox interaction model holds, and exponential censoring at rate 0.15. The restriction time
$\tau = 2.32$ is the upper quartile of the observed event times. Each RMTL truth is
$E\,(h_j / h)\{\tau - (1 - e^{-h\tau}) / h\}$ over $X$, by the quadrature the risks use, in
`tests/studies/continuous_survival_functionals_law.py`.
`tests/unit/test_continuous_survival_functionals_design.py` checks each RMTL truth against the
integral of the risks over $(0, \tau)$ and each RMST against $\tau$ minus the RMTLs.

## The fit

The fit is the registered study's gated configuration: the default hazard and treatment
libraries, cross-fitted with 10 outer folds and selection inside 5 inner folds, at n = 1,000,
with target times at the three event-time quartiles and the restricted horizon added. Every
interval is the one the library ships: range-logit on $(0, 1)$ for a risk and on $(0, \tau)$ for
an RMTL or RMST, log for a ratio, identity for a difference. No maintained package targets these
functionals in continuous time, so the study names no comparator and its agreement block stays
empty.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

The primary rows report 25 targets of each fit: each arm's risk of each cause and each risk
difference at the median and upper quartile, each cause's risk ratio there on the log scale, and
at $\tau$ the RMTL of each arm and cause, the RMST of each arm and their differences. Each row
takes the primary rule: coverage above 0.90, the SE ratio inside (0.8, 1.2) and the bias within
0.2 SD. The pilot of 800 draws on its own stream sets the count by the shared primary rule
before the run, about 6,000 to 7,200 at the registered study's coverages.

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | the RMST difference at $\tau$, with the shrunken-SE control (factor 0.85) and the noise control | 3,200 at n = 1,000, by the at-target rule checked against the pilot's SE ratio | an SE ratio or coverage interval outside its band |
| targeting necessity (reported) | the RMTLs of a fit whose walk targets the risks only, on 1,000 of the primary draws | 1,000 at n = 1,000 | nothing: the share of fits whose RMTL score exceeds the cutoff, the bias and the coverage are reported |
| `simultaneous_coverage` (second phase) | the band over every risk and risk difference, and its pointwise control | under its own declaration | joint coverage outside the calibration band |

The band row waits for the default band of the band thread's PR 2 and runs as a second phase.

## Limitations

| limit | what it means for use |
| --- | --- |
| one restriction time | a fit with several restricted horizons stays experimental |
| one law and one size | the evidence covers the base law at n = 1,000 with default learners |
| no comparator | no external implementation targets these functionals in continuous time |

## Reproduction

Run the pilot first, as `python -m tests.studies.continuous_survival_functionals_pilot --jobs N`.
Run `python -m tests.canonical.continuous_survival_functionals.regenerate --output DIR` with an
empty scratch directory outside the repository to regenerate the artifacts. The red-cell
readings and the budget are declared in
`tests/canonical/continuous_survival_functionals/regenerate.py` before any run.
