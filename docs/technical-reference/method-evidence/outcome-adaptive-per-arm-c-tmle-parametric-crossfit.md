# Outcome-adaptive per-arm C-TMLE (parametric crossfit)

This study measures the interval of
`CTMLE(strategy="oat", oat_design="per_arm", cross_fit=True, n_folds=5)` with `ParametricOutcome`
and `ParametricMechanism`. The fit is an exact V-fold stacked Z-estimator. Each fold fits one
outcome GLM and one mechanism per arm. A pooled fluctuation per arm and the whole-sample mean per
arm complete one system. With `repeats` above 1, the fit reports the median of the draws with the
split-adjusted variance of `median_estimates`.

The section
[Parametric per-arm intervals](../collaborative-tmle.md#parametric-per-arm-intervals) states the
construction, its conditions and what the interval covers. The study
[`ctmle-oat-per-arm-parametric`](outcome-adaptive-per-arm-c-tmle-parametric.md) measures the
in-sample form of the same curve.

**Python property evidence, no comparator.** No canonical implementation is compared. R
[`drtmle`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c)
with `adapt_g = TRUE` reports the shipped curve of Theorem 1 of Benkeser, Cai and van der Laan
(2020). The stack corrects that curve, so no R program measures the corrected uncertainty. The
empty equivalence artifact records the absence of a comparator. The evidence is the exact truth of
each law, the declared calibration bands, and two negative controls that must read red.

The study publishes under the `gated` policy. Every gated row passed, and both negative controls
read red as declared. The [red-cell ledger](red-cells.md) lists no cell of this study.

## What was tested

| setting | declaration |
| --- | --- |
| subject | `CTMLE(strategy="oat", oat_design="per_arm", cross_fit=True, n_folds=5, simultaneous=False)` with `ParametricOutcome()` and `ParametricMechanism(degree=3)`. The `repeats` value is 1, and 3 in the group `repeats`. The study sets `g_bounds=(0.01, 0.99)`, `max_iter=100` and `tol=1e-10`, and leaves `alpha` at its default |
| primary phase | `binary_active` cross-fitted at $n = 1500$, 1,200 replications, with `ey0`, `ey1`, `ate`, `rr` and `or`. Each replication must report the status `"influence_curve"` for every estimand. This phase is not the in-sample phase of the parametric study. The harness gates its rows |
| property phase | 7 fit groups. Each group fits one shared sample per replicate, and its cells read the same fits |
| truth | a quadrature of each law (`tests/studies/oat_parametric_quadrature.py`), so no truth comes from a simulation. A cell of the pseudo-true law reads the limit $\psi^*$ |
| verdicts | each calibration cell needs its SE-ratio and coverage intervals inside the calibration bands. The type-I cell needs its rejection bound under the ceiling. A control cell needs its SE-ratio interval outside the band on the declared side |
| failed replicates | a fit that raises, or a fit whose stack status is not `"influence_curve"`, is a failed replicate. The run records it and draws no replacement. A cell may lose `floor(0.01 R)` replicates, and one more reads `bound exceeded`. No replicate was lost |

The laws share $W = (W_1, W_2, W_3)$, three independent standard normals.

| law | role | construction |
| --- | --- | --- |
| `binary_active` | primary, calibration and repeats | a binary outcome with an arm-specific index in $W_1$ and $W_2$. The treatment law is a softmax in $W_1$ and $W_3$, so $W_3$ is an instrument. The outcome model is correct |
| `binary_null` | type-I error | the treatment law of `binary_active`, with arm 1 reading the index of arm 0, so the ATE is zero |
| `pseudo_true` | wrong outcome model | `binary_active` fitted with $W_1$ dropped from the outcome features. The truth of the cell is the limit $\psi^*$ |

The budgets in the table are the declared replications of each fit group.

| group | law | $n$ | replications | estimands | role |
| --- | --- | ---: | ---: | --- | --- |
| primary phase | `binary_active` | 1,500 | 1,200 | `ey0`, `ey1`, `ate`, `rr`, `or` | gated against truth |
| `crossfit_1500` | `binary_active` | 1,500 | 15,200 | `ey0`, `ey1`, `ate`, log `rr`, log `or` | calibration and the shipped-curve control |
| `crossfit_500` | `binary_active` | 500 | 64,800 | `ey0`, `ey1`, `ate` | calibration, and the $\sqrt{n}$ SD |
| `crossfit_8000` | `binary_active` | 8,000 | 10,000 | `ey0`, `ey1`, `ate` | calibration, and the $\sqrt{n}$ SD |
| `crossfit_pseudo_true` | `pseudo_true` | 1,500 | 10,000 | `ey0`, `ey1`, `ate` | calibration against $\psi^*$ |
| `crossfit_generated_design` | `binary_active`, the estimated outcome and a pinned true outcome | 1,500 | 10,800 | `ate` | the generated-design pair |
| `repeats` | `binary_active`, `repeats=3` | 1,500 | 10,800 | `ey0`, `ey1`, `ate` | calibration and the median shipped-curve control |
| `crossfit_null` | `binary_null` | 1,000 | 1,200 | `ate` | type-I error |

The study sizes each unit so that the whole study passes with probability at least 0.95 when
nothing is wrong. The count $G$ is 10: six calibration groups with the generated-design pair among
them, one type-I group, the primary phase and two negative controls. Each unit needs a pass
probability of $0.95^{1/10}$, which is 0.99488. A predictive model sets the size. It centres each
row on its own pilot SE ratio and adds a pooled spread $\tau_{80}$ over the gated rows.

The declared-size pilot used seed 9,158,000,058 at commit `0cb5fd5b`. It read every group at its
declared budget on pilot-only seeds. It reports a study-wise pass probability near 0.9997. The
pilot does not read the primary phase. A bootstrap of the 15,200 `crossfit_1500` pilot replicates
gives its joint pass probability. That probability is 0.99985 at 1,200 replications and 0.9915 at
800.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate` | average treatment effect | `cleverly` cross-fitted per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0019 to 0.0025 | 0.9433 | 0.9831 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey0` | counterfactual mean under no treatment | `cleverly` cross-fitted per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0019 to 0.0012 | 0.9408 | 0.9840 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey1` | counterfactual mean under treatment | `cleverly` cross-fitted per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0016 to 0.0016 | 0.9483 | 0.9678 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | `cleverly` cross-fitted per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0069 to 0.0110 | 0.9433 | 0.9832 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | `cleverly` cross-fitted per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0034 to 0.0056 | 0.9417 | 0.9844 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `generated_design` | `binary_active__estimated` | positive | two-arm law with an arm-specific outcome index and an instrument: the same design is estimated from the data, as a real fit does | SE ratio and coverage intervals both inside their calibration bands; the paired deficit is reported only | coverage 0.9476 to 0.9581, SE ratio 0.9941 to 1.0299 | pass |
| `generated_design` | `binary_active__oracle_design` | positive | two-arm law with an arm-specific outcome index and an instrument: the outcome-adaptive design is supplied rather than estimated | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9471 to 0.9577, SE ratio 0.9935 to 1.0279 | pass |
| `interval_calibration` | `crossfit_1500__ate__corrected` | positive | the cross-fitted stacked curve at n = 1,500, five folds: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9427 to 0.9521, SE ratio 0.9787 to 1.0090 | pass |
| `interval_calibration` | `crossfit_1500__ate__noise_control` | control | a declared scale of independent noise is added to each ATE estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8239 to 0.8396, SE ratio 0.6918 to 0.7126 | pass |
| `interval_calibration` | `crossfit_1500__ate__shipped_curve__shrunken_se_control` | control | the ATE standard errors come from the shipped adaptive-propensity curve of the same cross-fitted fits, which omits the generated-design term | the SE-ratio interval must fall below the calibration band | coverage 0.8941 to 0.9067, SE ratio 0.8304 to 0.8560 | pass |
| `interval_calibration` | `crossfit_1500__ate__shrunken_se_control` | control | the ATE standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8186 to 0.8345, SE ratio 0.6853 to 0.7063 | pass |
| `interval_calibration` | `crossfit_1500__ey0__corrected` | positive | the cross-fitted stacked curve at n = 1,500, five folds: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9399 to 0.9495, SE ratio 0.9776 to 1.0072 | pass |
| `interval_calibration` | `crossfit_1500__ey1__corrected` | positive | the cross-fitted stacked curve at n = 1,500, five folds: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9458 to 0.9549, SE ratio 0.9926 to 1.0232 | pass |
| `interval_calibration` | `crossfit_1500__log_or__corrected` | positive | the cross-fitted stacked curve at n = 1,500, five folds: log odds ratio | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9435 to 0.9528, SE ratio 0.9785 to 1.0086 | pass |
| `interval_calibration` | `crossfit_1500__log_rr__corrected` | positive | the cross-fitted stacked curve at n = 1,500, five folds: log risk ratio | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9438 to 0.9531, SE ratio 0.9780 to 1.0079 | pass |
| `interval_calibration` | `crossfit_500__ate__corrected` | positive | the cross-fitted stacked curve at n = 500, five folds: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9471 to 0.9516, SE ratio 1.0010 to 1.0155 | pass |
| `interval_calibration` | `crossfit_500__ate__noise_control` | control | a declared scale of independent noise is added to each ATE estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8337 to 0.8411, SE ratio 0.7086 to 0.7188 | pass |
| `interval_calibration` | `crossfit_500__ate__shrunken_se_control` | control | the ATE standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8285 to 0.8361, SE ratio 0.7006 to 0.7110 | pass |
| `interval_calibration` | `crossfit_500__ey0__corrected` | positive | the cross-fitted stacked curve at n = 500, five folds: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9467 to 0.9512, SE ratio 1.0011 to 1.0153 | pass |
| `interval_calibration` | `crossfit_500__ey1__corrected` | positive | the cross-fitted stacked curve at n = 500, five folds: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9455 to 0.9500, SE ratio 1.0013 to 1.0161 | pass |
| `interval_calibration` | `crossfit_8000__ate__corrected` | positive | the cross-fitted stacked curve at n = 8,000, five folds: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9477 to 0.9587, SE ratio 0.9869 to 1.0228 | pass |
| `interval_calibration` | `crossfit_8000__ate__noise_control` | control | a declared scale of independent noise is added to each ATE estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8213 to 0.8407, SE ratio 0.6920 to 0.7177 | pass |
| `interval_calibration` | `crossfit_8000__ate__shrunken_se_control` | control | the ATE standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8227 to 0.8420, SE ratio 0.6904 to 0.7153 | pass |
| `interval_calibration` | `crossfit_8000__ey0__corrected` | positive | the cross-fitted stacked curve at n = 8,000, five folds: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9423 to 0.9538, SE ratio 0.9891 to 1.0246 | pass |
| `interval_calibration` | `crossfit_8000__ey1__corrected` | positive | the cross-fitted stacked curve at n = 8,000, five folds: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9462 to 0.9573, SE ratio 0.9854 to 1.0211 | pass |
| `interval_calibration` | `crossfit_pseudo_true__ate__corrected` | positive | the cross-fitted stacked curve with W1 dropped from the outcome features, so the interval covers the limit psi*: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9452 to 0.9564, SE ratio 0.9955 to 1.0322 | pass |
| `interval_calibration` | `crossfit_pseudo_true__ate__noise_control` | control | a declared scale of independent noise is added to each ATE estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8328 to 0.8517, SE ratio 0.7076 to 0.7335 | pass |
| `interval_calibration` | `crossfit_pseudo_true__ate__shrunken_se_control` | control | the ATE standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8290 to 0.8481, SE ratio 0.6971 to 0.7229 | pass |
| `interval_calibration` | `crossfit_pseudo_true__ey0__corrected` | positive | the cross-fitted stacked curve with W1 dropped from the outcome features, so the interval covers the limit psi*: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9444 to 0.9557, SE ratio 1.0000 to 1.0379 | pass |
| `interval_calibration` | `crossfit_pseudo_true__ey1__corrected` | positive | the cross-fitted stacked curve with W1 dropped from the outcome features, so the interval covers the limit psi*: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9488 to 0.9596, SE ratio 0.9975 to 1.0346 | pass |
| `interval_calibration` | `repeats__ate__corrected` | positive | the median over three cross-fitting draws, with the split-adjusted variance of the stacked curve: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9468 to 0.9574, SE ratio 0.9929 to 1.0285 | pass |
| `interval_calibration` | `repeats__ate__noise_control` | control | a declared scale of independent noise is added to each ATE estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8262 to 0.8446, SE ratio 0.7032 to 0.7278 | pass |
| `interval_calibration` | `repeats__ate__shipped_curve__shrunken_se_control` | control | the ATE standard errors are the median over the three draws of the shipped adaptive-propensity standard error, with the split-adjusted variance | the SE-ratio interval must fall below the calibration band | coverage 0.8987 to 0.9132, SE ratio 0.8428 to 0.8729 | pass |
| `interval_calibration` | `repeats__ate__shrunken_se_control` | control | the ATE standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8249 to 0.8435, SE ratio 0.6952 to 0.7196 | pass |
| `interval_calibration` | `repeats__ey0__corrected` | positive | the median over three cross-fitting draws, with the split-adjusted variance of the stacked curve: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9487 to 0.9592, SE ratio 1.0016 to 1.0374 | pass |
| `interval_calibration` | `repeats__ey1__corrected` | positive | the median over three cross-fitting draws, with the split-adjusted variance of the stacked curve: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9550, SE ratio 0.9886 to 1.0242 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0567, 0.0408 to 0.0761 | pass |
<!-- /generated -->

Read the two `generated_design` cells with this study's setting. The shared descriptions come
from the parent study. In this study the `oracle_design` cell pins the true outcome regression
and fits none, so the shipped curve is its full curve. The `estimated` cell fits the outcome
model per fold and reads the stacked curve. The study gates both cells. The phrase "the paired
deficit is reported only" does not apply here.

## Declared reading

The study fixed its reading before the first run, in the module
`tests/studies/canonical_ctmle_oat_per_arm_parametric_crossfit.py`.

| rule | text |
| --- | --- |
| red calibration row | a finite-sample result only when its 99% SE-ratio interval contains 1, its coverage agrees with 0.95, and the diagnosis finds no defect. Any other red row is a defect suspect, diagnosed before publication |
| negative control | each control must read red. A control row publishes `passed = True` exactly when its SE-ratio interval leaves the calibration band on the declared side. A control that stays inside the band, or straddles it, cannot see its term. The study is then invalid and the design changes before any reading |
| the two declared controls | `crossfit_1500__ate__shipped_curve__shrunken_se_control` and `repeats__ate__shipped_curve__shrunken_se_control`. Both must read below the band |
| reported only | the quantities of [the next section](#quantities-reported-and-not-gated) gate nothing |
| gate count | $G$ is 10 and the per-unit target is $0.95^{1/10}$, as [What was tested](#what-was-tested) states |
| failures | a failed replicate is recorded and not redrawn. Dropping it would select on success. Substituting the shipped standard error would mix two estimators |

## Negative controls

Both declared controls read red. The table gives each control, the term it removes, and the
measured SE ratio.

| control | cell | term removed | expected SE ratio | side | measured |
| --- | --- | --- | --- | --- | --- |
| shipped curve | `crossfit_1500__ate__shipped_curve__shrunken_se_control` | the generated-design term. The standard error comes from the shipped adaptive-propensity curve of the same cross-fitted fits | 0.845 exact, 0.852 pilot | below the band | `properties[interval_calibration/crossfit_1500__ate__shipped_curve__shrunken_se_control]:se_ratio`, with an upper endpoint of `properties[interval_calibration/crossfit_1500__ate__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` |
| shipped curve, median | `repeats__ate__shipped_curve__shrunken_se_control` | the same term. The standard error is the median over the three draws of the stored shipped standard error, with the split-adjusted variance | 0.845 exact, 0.853 pilot | below the band | `properties[interval_calibration/repeats__ate__shipped_curve__shrunken_se_control]:se_ratio`, with an upper endpoint of `properties[interval_calibration/repeats__ate__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` |

The exact ratio 0.845 is the in-sample value of the quadrature. It is also the cross-fitted limit
(check CW4 in `tests/unit/test_oat_crossfit_stack.py`). The median control equals the single-draw value up to the draw-spread
term, which is about 0.2% of the variance.

The first control uses the same fits as
`crossfit_1500__ate__corrected`. Its coverage is
`properties[interval_calibration/crossfit_1500__ate__shipped_curve__shrunken_se_control]:coverage`,
so the shipped curve under-covers where the stacked curve calibrates. The second control uses the
fits of `repeats__ate__corrected`. Its coverage is
`properties[interval_calibration/repeats__ate__shipped_curve__shrunken_se_control]:coverage`.

The harness also derives a `shrunken_se_control` and a `noise_control` from the ATE row of each of
five groups: `crossfit_1500`, `crossfit_500`, `crossfit_8000`, `crossfit_pseudo_true` and
`repeats`. The shrunken control multiplies the standard error by the declared factor
`margin:shrunken_se_factor`. All ten derived controls read below the band.

The sharp-null cell can see the stacked term. On the sharp-null law the exact shipped ratio of the
ATE is 0.833. A shipped curve in that cell would reject at about 0.10 and fail the ceiling.

## Quantities reported and not gated

These quantities describe the fit, and no verdict reads them.

| quantity | cell | what it shows | measured |
| --- | --- | --- | --- |
| coverage of $\psi_0$ | `crossfit_pseudo_true__ey0__corrected`, `crossfit_pseudo_true__ey1__corrected`, `crossfit_pseudo_true__ate__corrected` | the interval covers $\psi^*$ and not $\psi_0$ when the outcome model is wrong. The gated coverage of $\psi^*$ in the same cells passes | `properties[interval_calibration/crossfit_pseudo_true__ey0__corrected]:psi0_coverage`, `properties[interval_calibration/crossfit_pseudo_true__ey1__corrected]:psi0_coverage` and `properties[interval_calibration/crossfit_pseudo_true__ate__corrected]:psi0_coverage` |
| $\sqrt{n}$ SD at $n = 500$ and $n = 8000$ | `crossfit_500__ate__corrected`, `crossfit_8000__ate__corrected` | the empirical and reported $\sqrt{n}$ SD of the ATE against the quadrature SD | the table of [the root-n SD](#the-root-n-sd) |
| per-draw spread | `repeats__ate__corrected` | the mean SD of the three draws' ATE estimates | `properties[interval_calibration/repeats__ate__corrected]:draw_spread_mean` |
| paired deficit | `binary_active__estimated` | the SE-ratio deficit of the estimated design against the pinned design | an interval from `properties[generated_design/binary_active__estimated]:se_ratio_deficit_lower` to `properties[generated_design/binary_active__estimated]:se_ratio_deficit_upper` |

### The coverage of the limit

The study declared the coverage of $\psi_0$ before the run. The quadrature gives the limit
$\psi^* = (0.404085, 0.595915)$ of the fit without $W_1$ and the truth $\psi_0 = (0.440704,
0.559296)$. At $n = 1500$ the standardized gaps are 1.797 for an arm mean and 2.495 for the ATE.
The predicted coverage is $\Phi(1.96 - b) - \Phi(-1.96 - b)$ for the gap $b$. The declared values
are `psi0_coverage_expected` in the study configuration.

| cell | declared | measured |
| --- | ---: | --- |
| `crossfit_pseudo_true__ey0__corrected` | 0.5646 | `properties[interval_calibration/crossfit_pseudo_true__ey0__corrected]:psi0_coverage` |
| `crossfit_pseudo_true__ey1__corrected` | 0.5646 | `properties[interval_calibration/crossfit_pseudo_true__ey1__corrected]:psi0_coverage` |
| `crossfit_pseudo_true__ate__corrected` | 0.2963 | `properties[interval_calibration/crossfit_pseudo_true__ate__corrected]:psi0_coverage` |

The two arm values lie within one Monte Carlo SD of the declared value. At 10,000 replications
that SD is $\sqrt{p(1 - p)/10000}$, which is 0.005. The ATE value lies 0.0110 above the declared
value, which is 2.4 Monte Carlo SDs. The pilot also read 0.303. The study gates nothing on this
gap and does not explain it. The gated coverage of $\psi^*$ in the same cells passes.

### The root-n SD

The quadrature gives the asymptotic $\sqrt{n}$ SD of the ATE curve as
`properties[interval_calibration/crossfit_500__ate__corrected]:quadrature_sd` and of an arm mean
as `properties[interval_calibration/crossfit_500__ey0__corrected]:quadrature_sd`. The table gives
the empirical SD and the mean reported standard error, each times $\sqrt{n}$, for the ATE.

| $n$ | empirical $\sqrt{n}$ SD | reported $\sqrt{n}$ SE |
| ---: | --- | --- |
| 500 | `properties[interval_calibration/crossfit_500__ate__corrected]:sqrt_n_empirical_sd` | `properties[interval_calibration/crossfit_500__ate__corrected]:sqrt_n_reported_se` |
| 8,000 | `properties[interval_calibration/crossfit_8000__ate__corrected]:sqrt_n_empirical_sd` | `properties[interval_calibration/crossfit_8000__ate__corrected]:sqrt_n_reported_se` |

## Deviations from the plan

The plan is the X58 plan, section 13.6. The table lists each deviation and its reason.

| plan | study | reason |
| --- | --- | --- |
| $G$ counts nine gated units, with the primary phase outside the count (rev 3.1) | $G$ is 10, with the primary phase as a unit, and the primary phase has 1,200 replications | the pre-queue design review found that the driver gates the cross-fitted primary rows. At 800 replications their joint pass probability was 0.9915, below the per-unit target 0.99432. At 1,200 it is 0.99985 against 0.99488 |
| the declared reading names `shipped_curve_control` and `shipped_curve_median_control` | the reading names the cells `crossfit_1500__ate__shipped_curve__shrunken_se_control` and `repeats__ate__shipped_curve__shrunken_se_control` | the names in the plan were not cell names. Each control declares the exact ratio 0.845 and its pilot ratio |
| the coverage of $\psi_0$ reported | the coverage of $\psi_0$ declared at 0.5646, 0.5646 and 0.2963 before the run, and reported | the quadrature predicts the values exactly, so the declaration costs nothing |
| the pilot steps 2 and 4 read independent seeds | both steps read the same pilot seed, so step 4 contains the 5,200 draws of step 2 | the step 2 model at the declared budgets gave a study-wise 0.913, with five groups below target. The budgets stayed at the rev 3.1 values. Step 4 reads 1.000 at those budgets, and its interval half-widths equal those of the declared run, so its figure governs. At three times the data, every positive row lies at least 3.9 sampling SDs inside the band |
| the clause binding of `coverage in calibration band` is `bias` (copied from the parametric study) | the binding comes from the binding search on the committed rows | the `bias` perturbation does not decide that clause alone on these rows, and the seeded `affine` map does, as [Reproduction](#reproduction) states |

No replicate was lost in either phase, so no deviation concerns the failure bound.

## Result

Every one of the `property_cells_total` cells passes, and so does each of the
`independent_tests_total` primary tests. The lowest primary coverage is `min_coverage`, with a lower
endpoint of `min_coverage_ci_lower`. The primary SE-ratio endpoints run from `min_se_ratio_ci_lower`
to `max_se_ratio_ci_upper`. The largest standardized bias is `max_standardized_bias`. The run lost
`subject_failures` primary fits, and every property cell reports `failed_replicates` of 0 against
its allowance of `floor(0.01 R)`.

The stacked curve calibrates at five folds from $n = 500$ to $n = 8000$. It calibrates for the
arm means, the ATE, the log risk ratio and the log odds ratio at $n = 1500$. It calibrates for the
median over three draws. The generated-design pair passes, and so does the sharp null. The shipped
curve fails the same band in both controls, so the correction is what the study measures.

The coverage of $\psi^*$ in the pseudo-true cells is calibrated, and the coverage of $\psi_0$ is
not. A user who reads the interval as an interval for $\psi_0$ under a wrong outcome model gets
the second number. [Parametric per-arm intervals](../collaborative-tmle.md#parametric-per-arm-intervals)
states the claim.

The sharp-null cell rejects at `properties[type_i_error/sharp_null]:rejection_rate`, with an upper
endpoint of `properties[type_i_error/sharp_null]:rejection_ci_upper` against the ceiling
`margin:type_i_ceiling`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1200 | primary replications |
| `n` | 1500 | observations per primary replication |
| `independent_tests_total` | 5 | primary tests against truth |
| `independent_tests_passed` | 5 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 32 | repeated-sampling property cells |
| `property_cells_passed` | 32 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0181 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9408 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9211 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9211 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0425 | highest bootstrap primary SE-ratio endpoint |
| `subject_failures` | 0 | primary fits lost |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. A control may rise above it |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:subject_failure_bound` | 0.0100 | share of replicates a cell may lose |
| `margin:subject_failures_allowed` | 12 | replicates one primary cell may lose |
| `margin:type_i_ceiling` | 0.1000 | the rate the null cell must bound |
| `properties[interval_calibration/crossfit_1500__ate__shipped_curve__shrunken_se_control]:se_ratio` | 0.8431 | SE ratio, shipped-curve control |
| `properties[interval_calibration/crossfit_1500__ate__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` | 0.8560 | its 99% upper endpoint |
| `properties[interval_calibration/crossfit_1500__ate__shipped_curve__shrunken_se_control]:coverage` | 0.9005 | coverage, shipped-curve control |
| `properties[interval_calibration/repeats__ate__shipped_curve__shrunken_se_control]:se_ratio` | 0.8572 | SE ratio, median shipped-curve control |
| `properties[interval_calibration/repeats__ate__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` | 0.8729 | its 99% upper endpoint |
| `properties[interval_calibration/repeats__ate__shipped_curve__shrunken_se_control]:coverage` | 0.9061 | coverage, median shipped-curve control |
| `properties[interval_calibration/crossfit_pseudo_true__ey0__corrected]:psi0_coverage` | 0.5676 | coverage of $\psi_0$, arm-0 mean, wrong outcome model |
| `properties[interval_calibration/crossfit_pseudo_true__ey1__corrected]:psi0_coverage` | 0.5687 | coverage of $\psi_0$, arm-1 mean, wrong outcome model |
| `properties[interval_calibration/crossfit_pseudo_true__ate__corrected]:psi0_coverage` | 0.3073 | coverage of $\psi_0$, ATE, wrong outcome model |
| `properties[interval_calibration/crossfit_500__ey0__corrected]:quadrature_sd` | 0.7943 | asymptotic $\sqrt{n}$ SD of an arm mean |
| `properties[interval_calibration/crossfit_500__ate__corrected]:quadrature_sd` | 1.1234 | asymptotic $\sqrt{n}$ SD of the ATE curve |
| `properties[interval_calibration/crossfit_500__ate__corrected]:sqrt_n_empirical_sd` | 1.1273 | empirical $\sqrt{n}$ SD, $n = 500$ |
| `properties[interval_calibration/crossfit_500__ate__corrected]:sqrt_n_reported_se` | 1.1365 | reported $\sqrt{n}$ SE, $n = 500$ |
| `properties[interval_calibration/crossfit_8000__ate__corrected]:sqrt_n_empirical_sd` | 1.1190 | empirical $\sqrt{n}$ SD, $n = 8000$ |
| `properties[interval_calibration/crossfit_8000__ate__corrected]:sqrt_n_reported_se` | 1.1236 | reported $\sqrt{n}$ SE, $n = 8000$ |
| `properties[interval_calibration/repeats__ate__corrected]:draw_spread_mean` | 0.0012 | mean SD of the three draws' ATE estimates |
| `properties[generated_design/binary_active__estimated]:se_ratio_deficit_lower` | -0.0127 | paired SE-ratio deficit, lower endpoint |
| `properties[generated_design/binary_active__estimated]:se_ratio_deficit_upper` | 0.0139 | paired SE-ratio deficit, upper endpoint |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0567 | rejection rate at the sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0761 | its 99% upper endpoint |

## Limits

| limit | what it means |
| --- | --- |
| Five folds only | The study fits `n_folds=5`. The support row `C11.stacked` matches any fold count inside the cap of 20 folds, and the preflight labels a fit at another count `"experimental"` with the measured count as the reason. The interval also needs 30 rows in each held-out fold |
| One draw and three draws | The study measures `repeats` of 1 and 3. A fit at another repeat count is `"experimental"`. A fit at `repeats=3` with `rr` or `or` matches no row and is `"experimental"`, because the `repeats` group measures no ratio |
| Two arms, binary outcome, complete unweighted data | The study measures a binary treatment and a binary outcome with no weight, stratum, missing value or cluster. A continuous outcome with declared `q_bounds` has no row and is `"experimental"`. X58 part (b) holds more arms and part (d) holds weights |
| Two parametric learners only | Only `ParametricOutcome` and `ParametricMechanism` are admitted, as exact classes. A scikit-learn learner keeps the `"generated_design_plugin"` status, and [F19](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) owns its interval |
| The interval covers $\psi^*$ | Under a wrong outcome model the interval covers the limit $\psi^*$ and not $\psi_0$. The pseudo-true cells report the coverage of $\psi_0$ and gate nothing on it. The claim is pointwise, not locally uniform over the nonparametric model |
| Default `alpha` only | Every cell runs at the default `alpha`. No cell sets `alpha` below it |
| Balanced folds | Every studied $n$ is a multiple of 5, so the folds are exactly balanced. The unbalanced fold map is checked by CW2 in `tests/unit/test_oat_crossfit_stack.py`, not by this study |
| The split-adjustment term | The mean SD of the draws' ATE estimates is about 0.001 against a standard error near 0.03, so the term is about 0.2% of the variance. The `repeats` cells measure the calibration of the median rule and cannot see that term. `test_the_repeated_shipped_control_reads_the_reported_rule_on_stored_standard_errors` pins the identity |
| No simultaneous band | The study sets `simultaneous=False`. The fit builds no band on an admitted fit, and X58 part (b) holds the band |
| No comparator | The evidence is the exact truth of each law, the calibration bands and the negative controls. No R program reports the corrected curve |

## Reproduction

Run `python -m tests.canonical.ctmle_oat_per_arm_parametric_crossfit.regenerate` to regenerate the
artifacts. The directory
[`tests/canonical/ctmle_oat_per_arm_parametric_crossfit`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/canonical/ctmle_oat_per_arm_parametric_crossfit)
holds `manifest.json`, `run.log` and the committed results. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_oat_per_arm_parametric_crossfit/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_oat_per_arm_parametric_crossfit/properties.csv)
carry every published row. The run used commit `a1a87015` on a clean tree and 4,566 wall seconds on
four cores.

The binding search of the `property-differential` job chose the perturbation of each clause in
`tests/canonical/clause-bindings.csv` for this study. The command is
`python -m tests.canonical.property_differential --bindings <dir> --subjects ctmle-oat-per-arm-parametric-crossfit`.
The search bound 8 of 8 clauses with the shared kinds and needed no targeted kind. It found the
seeded `affine` map for `coverage in calibration band`. The plan copied `bias` from the parametric
study, and `bias` does not decide that clause on the rows of this study.
