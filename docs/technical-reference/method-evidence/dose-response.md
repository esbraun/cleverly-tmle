# Kernel-smoothed dose-response CV-TMLE

This study validates `DoseResponseTMLE` with a `DoseResponse` declaration in
`cleverly.estimators`. The target at each declared bandwidth $h$ and dose $a_0$ is the smoothed
curve $\theta_h(a_0) = \int_D w_{h,a_0}(a)\,\theta(a)\,da$, with $w_{h,a_0}$ the local-linear
equivalent kernel on the declared interval $D$. Each pair's estimate is a cross-validated TMLE
with one pooled fluctuation, each interval is a Wald interval, and the band is the multiplier
max-$t$ band over every declared pair.
[Smoothed dose-response curve](../dose-response.md) gives the estimand, the estimator and its
conditions.

The evidence label is **Python property evidence, with comparator runs pending**. The study
declares npcausal `ctseff` as its comparator. Its runs have not registered, so this page
publishes no paired verdict.
[Comparator runs pending](../../development/method-benchmarking.md#comparator-runs-pending) states
the rule.

**The study is declared and has not run.** Every number below that is not a declaration or a
pilot reading is filled by `python -m tests.studies.evidence.document --slug dose-response` from
the committed artifacts of the declared run.

## What was tested

| setting | declaration |
| --- | --- |
| law | $W \sim N(0, I_3)$. The dose is $A = m(W) + 1.5\varepsilon$ with $m(W) = 2.5 + 0.7 \tanh W_1 + 0.4 \tanh(W_2 - W_3)$ and $\varepsilon \sim N(0, 1)$, so the dose density is positive on the whole line and bounded below on $D = [0, 5]$. The binary outcome is $\bar Q_0(a, W) = \operatorname{expit}\{f(a) + c(a) U\}$ with $U = 0.7 W_1 - 0.5 W_3$, $f(a) = -0.3 + 0.9 \sin\{1.1(a - 1.5)\}$ and $c(a) = 0.8 + 0.1a$. The Gaussian outcome has mean $f(a) + c(a) U$ and unit noise. $W_1$ moves both the dose and the outcome, so it confounds |
| scenario `binary` | n = 4,000. Doses 0.25, 1.5, 2.5, 3.5 and 4.75 by bandwidths 0.5, 1.0 and 1.5: 15 pairs, the default logistic fluctuation |
| scenario `strata` | n = 8,000. The binary law with $V = 1\{W_1 > 0\}$ as a fourth covariate and `strata=["V"]`, doses 0.25, 2.5 and 4.75 at bandwidth 1.0: six estimands, one per pair and stratum |
| scenario `gaussian` | n = 4,000. The Gaussian outcome with `fluctuation="linear"` and `q_bounds=(-15, 15)`, the five doses at bandwidth 1.0 |
| scenario `tail` | n = 1,000. One pair, bandwidth 0.5 at dose 0.4, whose expected Kish size under the true density is 62.3, just above the default support floor of 50. Its kernel reaches below $D$, so it reads the truncated boundary weight, which puts mass at the edge of $D$ |
| fit | five folds, the default density bins ($\max(20, \lceil 2 n^{1/3} \rceil)$: 32 at n = 4,000 and 20 at n = 1,000), the Epanechnikov kernel, eight quadrature nodes per piece, `random_state=0` |
| learners | outcome: a cubic B-spline of the dose with eight knots, tensored with $(1, W)$, then a standardized `LogisticRegression(C=100)`, or `LinearRegression` for the Gaussian outcome. The sieve holds the $W$ part of the law exactly and approximates $f$ by a spline. Density: the package's pooled-hazard binned density with a standardized `LogisticRegression(C=100)` on the bin indicators, $W$, $W^2$, and their products with the bin index |
| truth | composite Gauss-Legendre quadrature on the pieces between the ends of $D$ and $a_0 \pm h$, over a Gauss-Hermite integral in the outcome index; a stratified truth is a two-dimensional Gauss-Hermite integral. The design test recomputes every truth with adaptive quadrature to $10^{-10}$ |
| primary budget | 4,000 replications of each scenario |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| property budget | 7,600 replications of the band family, 6,000 of the calibration family and 800 of each robustness cell, all at n = 4,000 on the binary law |
| failed fits | at most 1% of the replications of any cell |
| seeds | `seed=2026119301` and `resampling_seed=2026119302`. Every property family and the pilot draw from their own labelled streams |
| policy | `gated`: any red verdict fails the run |

The laws, scenarios, learners, budgets, controls, seeds and red-cell readings were fixed before
the run, in `tests/studies/dose_response.py`, `tests/studies/dose_response_properties.py` and
`tests/canonical/dose_response/regenerate.py`. `tests/unit/test_dose_response_design.py` checks
the truths, the tail dose, the robustness limits, the band design numbers and the sizes.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Agreement with the canonical implementation

**Python property evidence, with comparator runs pending.** This section holds no paired table,
because the comparator study has not run.

| item | state |
| --- | --- |
| comparator | npcausal `ctseff` (Kennedy, Ma, McHugh and Small 2017) |
| what it estimates | a related target: a curve smoothed against the marginal dose density rather than the Lebesgue kernel weight of `DoseResponse`, so a paired comparison needs a matched target |
| reason pending | roadmap item X93, part S2, builds the R runner and the matched target |
| paired cell of the grid | 0/0 external comparisons, with the pending label |

The owner's rule applies: a method merges on its Python evidence, and the comparator study
follows ([Theory and comparators](../../architecture-invariants.md#theory-and-comparators)).

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `simultaneous_coverage` | does the band over the 15 binary pairs cover every truth at once, with its 99% joint-coverage interval inside 0.92 to 0.98 | the pointwise intervals read jointly must cover below 0.95. Their exact joint coverage under the efficient influence correlation is 0.676 |
| `interval_calibration` | at $h = 1.0$, $a_0 = 2.5$, does the SE-ratio interval lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98, with the empirical SD and the mean reported SE each within 0.90 to 1.10 of the exact efficient SD | the reported SE times 0.70, and one efficient SD of independent noise added to each estimate, must each leave the band |
| `double_robustness` | at $h = 1.0$, $a_0 = 3.5$, with a density that ignores $W$ and the declared outcome sieve, is the 99% bias interval inside $\pm 0.25$ SD | with the dose-only spline outcome as well, the bias interval must lie wholly outside $\pm 0.25$ SD |

The wrong density is the hazard classifier on the bin indicators alone: the binned marginal
density of the dose. In the population the positive cell's limit is 0.017 efficient SEs off the
truth at n = 4,000, and the control's is 3.03, the unadjusted kernel regression of $Y$ on $A$.
A main-terms logistic outcome on $[A, W]$ was the first control design. It was rejected before
any run, because its limit is 0.40 efficient SEs off the truth at this pair.

No cell measures an outcome regression that is wrong with a density that is right. The binned
density is consistent but at no known rate, so the remainder is first order in its error, and a
bias interval at one $n$ cannot separate a slowly vanishing bias from a defect. The exact
remainder tests in `tests/unit/test_dose_response.py` carry that direction.

<!-- generated: properties -->
<!-- /generated -->

## Claim boundary

The pointwise interval targets the smoothed curve, not $\theta(a_0)$. Read against the
unsmoothed curve at dose 2.5, where the curve bends, its coverage at n = 4,000 is expected to be
0.938 at $h = 0.5$, 0.597 at $h = 1.0$ and 0.010 at $h = 1.5$, from the smoothing bias and the
efficient SE. The MSE-optimal bandwidth at that dose is 0.60. No shared family reads a coverage
that must fall below nominal, so the run reports these readings from the binary rows and gates
nothing on them.

## Sizing

The sizes were fixed from a pilot on its own seed stream,
`tests/studies/dose_response_pilot.py`, whose rows are `tests/studies/dose_response_pilot.csv.gz`.

Each sized term draws on the study's failure allowance through the shared
`repeated_sizing` rules. The pilot drew 1,000 samples of each primary scenario and of the
calibration pair, and 300 of each robustness cell. No pilot fit failed.

| sizing input | value | source |
| --- | --- | --- |
| primary replications | 4,000, from the shared primary rule on the 1,000 pilot draws of each scenario, one bias group per scenario | `tests/studies/dose_response.py` |
| lowest pilot coverage | 0.931, dose 0.25 in stratum $V = 1$ at n = 8,000 (Kish size 141). At n = 4,000 the same cell read 0.908 (Kish size 69, SE ratio 0.94), whose one-sided 80% lower bound lies below the 0.90 floor, so no budget sized it and the scenario moved to n = 8,000 before any run | `tests/studies/dose_response_pilot.csv.gz` |
| boundary pairs | the upper boundary dose 4.75 reads SE ratios 1.10 to 1.12 on the binary outcome, a conservative interval; the lower boundary dose 0.25 reads 0.98 to 0.99 and coverage 0.933 to 0.935 | the same |
| tail pair | coverage 0.935, SE ratio 1.06, mean Kish size 58; 21% of the fits fall below the floor and carry the `low_dose_support_plugin` label | the same |
| band replications | 7,600. The pilot band covers jointly in 942 of 1,000 draws; the rule reads its one-sided 80% lower bound, 0.935 | `tests/studies/dose_response_properties.py` |
| band control | the pointwise intervals cover jointly with exact probability 0.676 under the efficient influence correlation (the pilot reads 0.698); the oracle critical value is 2.751 against the pilot's mean 2.733 | `tests/unit/test_dose_response_design.py` |
| calibration replications | 6,000, at the pooled pilot SE ratio 1.023 of the calibration pair (1.052 from the single-pair pilot, 0.996 from the same pair in the 15-pair pilot). Read with that ratio's pilot noise, the pass probability is 0.89 | the same |
| efficiency ratios | the single-pair pilot's empirical SD is 0.973 of the exact efficient SD and its mean reported SE 1.023 | the same |
| robustness replications | 800. The positive's pilot standardized bias is 0.010 (one-sided 80% upper bound 0.059); the control's is 2.87 | the same |

The reported standard error is conservative by a few percent at most interior pairs at
n = 4,000 (mean reported SE 1.02 to 1.06 times the exact efficient SD). The calibration cell
can therefore turn red on its upper SE-ratio end, and the red-cell route below declares the
reading for that event.

## Red-cell route

| event | route, declared before any run |
| --- | --- |
| a red primary row of `binary`, `strata` or `gaussian` | a defect suspect of the quadrature, the fluctuation or the influence curve. A red boundary row (dose 0.25 or 4.75) with an equivalent bias and an SE-ratio interval below 1 reads as the variance limit of a boundary weight at n = 4,000, and moves to `reporting` with an owner row if no defect is found |
| a red `tail` row | the measurement of the support floor. The route is a scope change: raise the default support floor or the smallest default bandwidth fraction, and re-run under that declaration |
| a red band cell below 0.92 while every `binary` row passes | the finite-sample limit of the band over 15 near-collinear curves, moved to `reporting` with the measured coverage. Above 0.98 it is a critical-value defect suspect |
| a control that does not leave its band | an instrument defect, and the run is void |
| a red calibration SE ratio with green coverage and efficiency ratios | a finite-sample variance finding, moved to `reporting`. Red on coverage or an efficiency ratio is a defect suspect |
| a red robustness positive | a defect suspect of the targeting step |
| a replication that raises | is never redrawn. A `DataError` naming a zero fitted density inside a kernel's support voids the run |
| any verdict | moves no budget, margin, law, learner, fold count or seed |

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results, and reads `pending` until the declared run registers.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | pending | primary replications |
| `n` | pending | observations per primary replication |
| `independent_tests_passed` | pending | truth tests passing |
| `independent_tests_total` | pending | truth tests reported |
| `property_cells_passed` | pending | property cells passing |
| `property_cells_total` | pending | property cells reported |
| `max_standardized_bias` | pending | largest primary standardized bias |
| `min_coverage` | pending | lowest primary coverage |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10.0000 | union-model SE-ratio screen, upper limit |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it until the comparator runs register |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it until the comparator runs register |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it until the comparator runs register |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it until the comparator runs register |

## Limits

| limit | what it means for use |
| --- | --- |
| the npcausal comparison is pending | the page publishes no paired verdict |
| one law family | a normal location model for the dose, with an outcome that reads $W$ through one linear index, on $D = [0, 5]$ |
| declared sieve learners | the outcome sieve holds the $W$ part of the law exactly. The default learner library is not measured |
| one sample size per scenario | n = 4,000, n = 8,000 for the stratified scenario and n = 1,000 for the tail pair. Smaller sizes are not claimed |
| the band covers 15 declared pairs | a band over other doses or bandwidths reads `experimental` |
| no cell measures a wrong outcome with a right density | the exact remainder tests carry that direction |
| no weights, clusters or missing outcomes | the study measures none of them |

## Reproduction

The declared run is `python tests/canonical/dose_response/regenerate.py --output <empty
directory>`, whose docstring gives the run form and the red-cell readings. It starts no
reference process while the comparator is pending.
