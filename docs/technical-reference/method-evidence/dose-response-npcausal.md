# Kernel-smoothed dose-response CV-TMLE against npcausal ctseff

This study compares `DoseResponseTMLE` with R `npcausal` `ctseff` (Kennedy, Ma, McHugh and Small
2017) at commit `56a5ac1` on identical samples. The two estimate different smoothed curves in
general. `DoseResponse` smooths $\theta(a) = E\{\bar Q(a, W)\}$ with the local-linear equivalent
kernel on Lebesgue measure over $D = [0, 5]$, and `ctseff` regresses a doubly robust pseudo-outcome
on the dose, which smooths $\theta$ against the marginal dose density. On this study's law
$\theta$ is a line, and every local-linear smoother reproduces a line, so both targets equal
$\theta(a_0)$ at every declared dose and bandwidth. The comparison is of target recovery on that
shared value, not of the two smoothing measures, and its kind is "related target".

The study is declared and has not run. R cannot run in the environment that declared it, so
every count below waits for the local run, and the generated tables are empty until then.
[Smoothed dose-response curve](../dose-response.md) gives the estimand and the estimator. The
[dose-response study](dose-response.md) holds the estimator's property evidence.

## What was compared

| setting | declaration |
| --- | --- |
| law | $W \sim N(0, I_3)$; $A = 2.5 + 0.5 W_1 + 0.3 W_2 - 0.3 W_3 + 1.5\varepsilon$; $Y = 0.2 + 0.5 A + (0.8 + 0.1 A)(0.7 W_1 - 0.5 W_3) + \eta$, with standard normal $\varepsilon$ and $\eta$. $W_1$ confounds, and $\theta(a) = 0.2 + 0.5a$ |
| pairs | doses 1.5, 2.5 and 3.5 by triweight bandwidths 1.0 and 1.5: six pairs, each kernel's support inside $D$ |
| sample | 2,800 replications of n = 4,000, each fitted by both implementations |
| `cleverly` | `DoseResponseTMLE` with the triweight kernel, five folds, the linear fluctuation and `q_bounds=(-25, 25)` |
| `ctseff` | one call per bandwidth, `bw.seq` set to the matched Gaussian bandwidth, so its leave-one-out selection has one candidate |
| matched kernels | `ctseff` receives $h_G = h\,R(K_G)/R(K_T)$, with $R(K) = \int K^2$: 0.3458 for $h = 1$ and 0.5187 for $h = 1.5$. At an interior dose both estimators then have the leading variance $R(K)\,E_W\{\sigma^2 / g_0(a_0 \mid W)\}/(nh)$ |
| outcome regression | main terms and pairwise products of $(a, W)$ in both: `SL.glm.interaction` in R, `PolynomialFeatures(2, interaction_only=True)` with `LinearRegression` here. The design holds $\bar Q_0$ exactly |
| dose density | `ctseff` fits the dose mean and variance with `SL.glm.interaction` and smooths the standardized residuals with a kernel density estimate. `cleverly` uses the pooled-hazard binned density of the dose-response study |
| evaluation grid | `ctseff` reports its curve on `seq(a.rng[1], a.rng[2], length.out = n.pts)`. The runner widens the observed dose range to multiples of 1/8, so every declared dose is a grid point exactly and the grid spans the data |
| inference | Wald intervals: $z_{0.975}$ times the influence-curve SE for `cleverly`, and `ctseff`'s own `est` $\pm$ 1.96 `se`$/\sqrt n$ |

A single-learner `SuperLearner` gives that learner weight 1, so each R prediction is the learner's
full-sample fit and no R fit depends on the fold draw.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

None declared: 0/0 cells. The row rests on the comparison and the truth tests, and the
[dose-response study](dose-response.md#theory-properties) holds the estimator's property cells.

## Sizing

The count comes from the shared primary rule (`tests/studies/repeated_sizing.py`) at the X32b
clause shares, read on a committed pilot of 1,000 draws
(`tests/studies/dose_response_npcausal_pilot.py`). The pilot fitted each draw with `cleverly` and
with a Python transcription of `ctseff`, which stands in for the reference because R could not
run there. The transcription is not the comparator. It replaces the binned `locpoly` fit by an
exact local-linear fit and R's `smooth.spline` by a GCV smoothing spline, and it keeps `ctseff`'s
pairing of two density grids.

| reading | coverage the rule reads | replications |
| --- | --- | --- |
| A | `cleverly`'s pilot rows for both implementations, the assumption the rule makes for a paired cell | 2,400 |
| B | each implementation's own pilot rows | 2,800 |

The record declares the larger count, 2,800. Reading B is bound by the transcription's lowest
coverage, 0.935 at dose 2.5 and bandwidth 1.0. `tests/unit/test_dose_response_npcausal_design.py`
recomputes both readings. The local smoke run checks that `ctseff` returns finite rows, sits near
the transcription on the same draws, and costs seconds per call. On the pilot, `cleverly` took
1.7 seconds per fit on one core.

## Projected verdicts and red-cell route

The pilot projects four of the six paired cells green on every leg. At dose 3.5 `cleverly`'s
reported SE runs above its spread, with a pilot SE ratio of 1.07 at both bandwidths against 1.00
and 1.01 for the transcription. The [dose-response study](dose-response.md) reports the same
conservatism at interior pairs. The calibration leg there reads a calibration excess near 0.06
against the margin 0.05, so those two cells are projected `inconclusive` at any count. The record
is `gated`, and the declared route for that reading is in
[`regenerate.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/dose_response_npcausal/regenerate.py).

| diagnosis | route |
| --- | --- |
| a red `cleverly` truth row | a defect suspect: the truths are exact, the outcome design holds $\bar Q_0$ and the doses are interior |
| a red `npcausal` truth row | a finding about the reference at this law, published with `accepted_reference_failure`. It does not turn the subject's row red |
| a red calibration leg at dose 3.5 with `cleverly`'s SE ratio above 1 and its truth row green | the projected reading: a conservative influence-curve SE, moved to `reporting` with the measured ratios |
| a red calibration leg with `cleverly`'s SE ratio below 1 | a variance defect suspect |
| a red RMSE or similarity leg with both truth rows green | the two estimators' finite-sample difference on one target, moved to `reporting` with the measured values |

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. The other rows are declared
counts. The run adds the measured rows and the failure bounds it applied.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2800 | paired replications |
| `n` | 4000 | observations per replication |
| `independent_tests_total` | 12 | truth tests reported, six pairs for each implementation |
| `paired_tests_total` | 6 | paired comparisons reported |
| `property_cells_total` | 0 | property cells. The [dose-response study](dose-response.md#theory-properties) holds them |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |

## Limits

| limit | what it means for use |
| --- | --- |
| a linear curve only | the two targets coincide only where $\theta$ is linear on each kernel's support. On a curved $\theta$ they part: on $\sin(2a)$ by up to 0.75 efficient SEs at n = 4,000, which the design test checks. Agreement here says nothing about the smoothing measure |
| interior doses | every kernel's support lies inside $D$. Boundary pairs, where the Lebesgue equivalent kernel is asymmetric, are not compared |
| the triweight against a Gaussian kernel | no shared kernel exists, so the bandwidths are matched by roughness. The Epanechnikov and uniform kernels are not compared |
| in-sample R nuisances | `ctseff` fits in sample and `cleverly` cross-fits over five folds |
| two density estimators | the dose densities come from different estimators, so the comparison is not of one targeting step on shared nuisances |
| a Gaussian outcome | binary outcomes and the logistic fluctuation are not compared |
| no band, strata, weights or clusters | only pointwise Wald intervals are compared |

## Reproduction

The smoke run is `python -m tests.canonical.dose_response_npcausal.regenerate --replicates 20
--output <empty directory outside the repository> --allow-failures --jobs <workers>`. The declared
run is `python -m tests.canonical.dose_response_npcausal.regenerate --output <empty directory>
--jobs <workers>`. Both build the unchanged image of the incremental studies from
`tests/canonical/npcausal_incremental/Dockerfile` and mount `run_study.R`.
