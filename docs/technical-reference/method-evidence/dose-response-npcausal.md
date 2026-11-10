# Kernel-smoothed dose-response CV-TMLE against npcausal ctseff

This study compares `DoseResponseTMLE` with R `npcausal` `ctseff` (Kennedy, Ma, McHugh and Small
2017) at commit `56a5ac1` on identical samples. The two estimate different smoothed curves in
general. `DoseResponse` smooths $\theta(a) = E\{\bar Q(a, W)\}$ with the local-linear equivalent
kernel on Lebesgue measure over $D = [-1.5, 6.5]$, and `ctseff` regresses a doubly robust
pseudo-outcome on the dose, which smooths $\theta$ against the marginal dose density. On this
study's law $\theta$ is a line, and every local-linear smoother reproduces a line, so both targets
equal $\theta(a_0)$ at every declared dose and bandwidth. The comparison is of target recovery on
that shared value, not of the two smoothing measures, and its kind is "related target".

The study is declared and has not run, and it is not yet registered. Its bandwidths and its
replication count are provisional: a rule reads them off two pilots, and until the local R pilot
commits real `ctseff` rows, a Python transcription of `ctseff` stands in for it. The generated
tables stay empty until the declared run. [Smoothed dose-response curve](../dose-response.md)
gives the estimand and the estimator. The [dose-response study](dose-response.md) holds the
estimator's property evidence.

## What was compared

| setting | declaration |
| --- | --- |
| law | $W \sim N(0, I_3)$; $A = 2.5 + 0.5 W_1 + 0.3 W_2 - 0.3 W_3 + 1.5\varepsilon$; $Y = 0.2 + 0.5 A + (0.8 + 0.1 A)(0.7 W_1 - 0.5 W_3) + \eta$, with standard normal $\varepsilon$ and $\eta$. $W_1$ confounds, and $\theta(a) = 0.2 + 0.5a$ |
| pairs | doses 1.5, 2.5 and 3.5 at two consecutive triweight bandwidths of the candidates 1.0, 1.5, 2.0 and 2.5, provisionally 2.0 and 2.5: six pairs, each kernel's support inside $D$ |
| sample | provisionally 4,800 replications of n = 4,000, each fitted by both implementations |
| `cleverly` | `DoseResponseTMLE` with the triweight kernel, five folds, the linear fluctuation and `q_bounds=(-25, 25)` |
| `ctseff` | one call per bandwidth, `bw.seq` set to the matched Gaussian bandwidth, so its leave-one-out selection has one candidate. It still computes that candidate's risk |
| matched kernels | `ctseff` receives $h_G = h\,R(K_G)/R(K_T)$, with $R(K) = \int K^2$: 0.3458, 0.5187, 0.6915 and 0.8644 for $h$ = 1, 1.5, 2 and 2.5. At an interior dose both estimators then have the leading variance $R(K)\,E_W\{\sigma^2 / g_0(a_0 \mid W)\}/(nh)$, within 10% of the subject's exact term at $h \le 1.5$ and within 25% at $h = 2.5$ |
| outcome regression | main terms and pairwise products of $(a, W)$ in both: `SL.glm.interaction` in R, `PolynomialFeatures(2, interaction_only=True)` with `LinearRegression` here. The design holds $\bar Q_0$ exactly |
| dose density | `ctseff` fits the dose mean and variance with `SL.glm.interaction` and smooths the standardized residuals with a kernel density estimate. `cleverly` uses the pooled-hazard binned density of the dose-response study |
| evaluation grid | `ctseff` reports its curve on `seq(a.rng[1], a.rng[2], length.out = n.pts)`. The runner widens the observed dose range to multiples of 1/8, so every declared dose is a grid point exactly and the grid spans the data |
| inference | Wald intervals: $z_{0.975}$ times the influence-curve SE for `cleverly`, and `ctseff`'s own `est` $\pm$ 1.96 `se`$/\sqrt n$ |

A single-learner `SuperLearner` gives that learner weight 1, so each R prediction is the learner's
full-sample fit and no R fit depends on the fold draw. $D$ is symmetric about 2.5, so the
reflection $a \mapsto 5 - a$ maps it and the doses to themselves, and the chance that a declared
sample has no dose beyond either edge of $D$, which `DoseResponseTMLE` would refuse, is below
$10^{-6}$.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

None declared: 0/0 cells. The row rests on the comparison and the truth tests, and the
[dose-response study](dose-response.md#theory-properties) holds the estimator's property cells.

## Bandwidth selection

`ctseff` computes its leave-one-out bandwidth risk even when `bw.seq` holds one value. Its
binned local-linear fit (`KernSmooth::locpoly`) cuts the Gaussian kernel at four bandwidths, so
when the lowest or highest dose lies more than $4h_G$ from the next one, the end node's 2x2 system
has one bin or none and its solution is `NaN`. `approx` then returns `NA` at that dose, the risk
is `NA`, `which.min` selects nothing, and the call errors. With a normal dose at n = 4,000 the gap
between the two most extreme doses has a mean near 0.4, so the failure rate falls quickly as the
bandwidth grows.

The candidates 1.0, 1.5, 2.0 and 2.5 were declared before any R output: the two bandwidths first
declared and two wider fallbacks, with $D$ widened from $[0, 5]$ so that every candidate's kernel
lies inside it. The rule takes consecutive pairs in that order and selects the first whose pilot
failures, on the draws where either `ctseff` call failed, keep the chance of exceeding the
declared 1% failure bound at or below 0.01 at the declared count, computed at the failure rate's
one-sided 95% Clopper-Pearson upper bound. The Python transcription reproduces the failure path
and fails 74, 7, 1 and 0 of 1,000 calls at the four candidates, so it selects 2.0 and 2.5. The
R pilot replaces those counts with `ctseff`'s own.

## Sizing

The count is the largest of three readings, each a multiple of 400:

| reading | what it reads | provisional replications |
| --- | --- | --- |
| A | the shared primary rule (`tests/studies/repeated_sizing.py`) at the X32b clause shares, with `cleverly`'s pilot rows for both implementations | 2,400 |
| B | the same rule with each implementation's own pilot rows | 2,400 |
| paired legs | the smallest count at which the similarity, RMSE, coverage and calibration legs of every cell pass together with probability 0.95, by a normal approximation from the pilot's bootstrap spread. The RMSE and calibration legs at dose 3.5 are left out, because the pre-registered red signature admits them | 4,800 |

The paired-leg rule is capped at 8,000, about 30 R CPU-hours. Past the cap a declared straddle
reading governs a calibration red at doses 1.5 and 2.5 instead: the subject's SE ratio in
$[1, 1.07]$, the reference's below 1, both truth rows green. On the pilot, `cleverly` took 2.3
seconds per fit on one core at twelve pairs.

The provisional readings come from the transcription in place of `ctseff`. Its binned `locpoly`
fit is transcribed from the Fortran, and it differs from `ctseff` in three declared places. Its
smoothing spline places a knot at every grid point, where R places about 60. The outcome spline
smooths a line, which both reproduce, so only the marginal-density spline can differ. Its
residual density is an exact Gaussian kernel sum, where R bins and convolves, and its
one-learner `SuperLearner` is the least-squares fit of the same design.

## Projected verdicts and red-cell route

At dose 3.5 `cleverly`'s mean reported SE is 1.10 and 1.11 times the exact efficient SD at
bandwidths 2.0 and 2.5, against 1.01 to 1.04 at doses 1.5 and 2.5. The reflected pilot fits move
the law with $A \mapsto 5 - A$, and the excess stays at dose 3.5, so it comes from the
pooled-hazard density, whose bin hazards compound upward through the survival product, and not
from the law or from `ctseff`. The [dose-response study](dose-response.md) sees the same effect at
its upper boundary dose 4.75; this law's wider dose spread brings it to an interior dose. At
bandwidth 2.5 the subject's spread rises to 1.11 times efficient as well, so its SE is calibrated
there, the RMSE ratio against the transcription is 1.10, and the RMSE leg projects red at any
count. At bandwidth 2.0 the RMSE ratio is 1.05. The four cells at doses 1.5 and 2.5 pass every
leg with projected probability at least 0.96 at 4,800 replications.

The record is `reporting`, declared before any run, and the design test admits only these
signatures in the run's `equivalence.csv`. Owner rows follow the run.

| red paired row | reading |
| --- | --- |
| dose 3.5, failing only the RMSE leg, the calibration leg, or both, with the subject's mean SE above 1.05 efficient SDs, its coverage at least 0.95 and its truth row green | the pre-registered upper-dose excess of the pooled-hazard density |
| dose 1.5 or 2.5, failing only the calibration leg, when the paired-leg rule hit its cap | the straddle reading above |
| anything else | not admitted. The design test fails, and the row is diagnosed before publication |

A red `cleverly` truth row is a defect suspect: the truths are exact, the outcome design holds
$\bar Q_0$ and the doses are interior. A red `npcausal` truth row is a finding about the
reference at this law. It does not turn the subject's row red.

## Local R pilot

The R pilot runs on a machine with Docker. It builds the image of the incremental studies from
`tests/canonical/npcausal_incremental/Dockerfile` on first use and fits through the declared
runner `run_study.R`. Each `ctseff` call is its own harness group, so a failed call is one
failure record with R's message. The probe comes first and the full pilot second:

```console
python -m tests.canonical.dose_response_npcausal.pilot probe --scratch <dir> --jobs <n>
python -m tests.canonical.dose_response_npcausal.pilot reference --scratch <dir> --jobs <n>
```

The probe checks the image, confirms or refutes the failure path on constructed draws with an
isolated extreme dose, compares R with the transcription on 20 pilot draws (failures, estimates
within 0.05 SE, standard errors within 1%), reruns two draws for identical rows, and times a
call. At 5 to 10 CPU-seconds per call it costs 0.15 to 0.3 CPU-hours, and the full pilot, 4,000
calls, costs 5.5 to 11. The full pilot works in resumable chunks of 50 draws and commits
`tests/studies/dose_response_npcausal_pilot_r.csv.gz`, `..._pilot_r_failures.csv` and
`..._pilot_r.json`. The design test then requires the declared bandwidths and count to equal the
rule's reading of those rows.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. The other rows are declared
counts. The run adds the measured rows and the failure bounds it applied.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4800 | paired replications, provisional |
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
| wide bandwidths | the selected bandwidths are the narrowest at which `ctseff`'s bandwidth risk does not fail on isolated extreme doses. Narrower bandwidths are not compared |
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
`tests/canonical/npcausal_incremental/Dockerfile` and mount `run_study.R`. Neither runs before
the R pilot is committed and the declaration is final. The smoke draws declared replicates 0
to 19, so no declared value is read from it.
