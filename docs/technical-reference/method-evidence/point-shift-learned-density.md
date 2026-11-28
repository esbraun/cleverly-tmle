# A point additive shift with a learned density

This study measures the interval of `TMLE` for a point additive shift of a continuous dose when the
dose's conditional density is learned on the binned density route. The registered shift studies,
[continuous modified treatment policies](continuous-modified-treatment-policies.md) and
[point modified treatment policies](point-modified-treatment-policies.md), give the method the
exact density, so they say nothing about a learned one. Three arms fit every draw with the same
random state: the interventions example's in-sample fit, the same learners cross-fitted on three
folds, and the law's exact density on the fit's own grid, a diagnostic arm that isolates the
learned density's error.

**No canonical implementation is compared.** No external package fits this estimator with this
density learner. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law | `shift_dgp()` at its defaults, the law of `make_shift_dose`: $W \sim N(0, I_3)$, $A \mid W \sim N(2 + 0.7 W_1 - 0.3 W_2, 1)$, $Y = 1 + 0.5 A + 0.25 A^2 + W_1 - 0.5 W_2 + 0.2 W_3 + 0.8 \varepsilon$, at n = 3,000 |
| parameters | three contrasts against the natural course, `+0.5` capped at 5, `+0.5` uncapped and `+1.0` uncapped, and the four policy means. The truths are closed forms of the law |
| outcome learner | `make_pipeline(PolynomialFeatures(2), LinearRegression())` in every arm, which represents $E[Y \mid A, W]$ exactly |
| `in_sample_learned` | `TMLE(cross_fit=False, ratio="density")` with `HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0)` on 40 core bins: the interventions example's fit |
| `crossfit_learned` | the same learners, `TMLE(cross_fit=True, n_folds=3, q_bounds=(-7.89, 35.91), ratio="density")`. The outcome range is the law's $10^{-10}$ and $1 - 10^{-10}$ quantiles, rounded outward |
| `in_sample_oracle` | the in-sample fit with the law's exact density on the fit's own grid. It licenses nothing |
| primary | 1,200 draws, all three arms on each draw, 21 labelled parameters. The framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| `interval_calibration` | the cross-fitted arm's three contrasts on 19,600 property draws each, with a shrunken-SE control (factor 0.70) and a noise control (independent normal noise at the law's influence-curve SD over $\sqrt{n}$). No efficiency band is read: the claim is calibration, not efficiency |
| seeds | `seed=20261221` and `resampling_seed=20261222` |
| policy | `reporting` |

The calibration count is the declared ceiling. The 3,200-draw sizing pilot
(`tests/studies/point_shift_learned_density_pilot.csv`) gave SE ratios of 1.010 to 1.046 and a
spread constant up to 1.180, at which the per-cell predictive rule cannot size the cells, so by the
rule declared before the pilot the run proceeded at the ceiling of 19,600.
`tests/unit/test_point_shift_learned_density_design.py` recomputes the truths, the efficiency
bounds, the outcome range and the sizing figures.

The reading was declared before any run, in the docstring of
[`tests/studies/point_shift_learned_density.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/studies/point_shift_learned_density.py).
If every primary verdict and every `interval_calibration` verdict of the cross-fitted arm passes,
the registering change adds the support row. The in-sample learned arm adds no row whatever it
reads, and a red cell of that arm is a finite-sample limit of the in-sample learned density,
published under `reporting` with the owner `X121-learned-density-finite-sample`.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `crossfit_learned: ate_policy[+0.5 capped at 5 vs current practice]` | mean outcome with each dose raised by 0.5 and capped at 5, minus the natural course; cross-fitted on three folds, the same learned density, and the outcome range declared from the law | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.000875 to 0.000663 | 0.9625 | 1.0515 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `crossfit_learned: ate_policy[+0.5 uncapped vs current practice]` | mean outcome with each dose raised by 0.5, minus the natural course; cross-fitted on three folds, the same learned density, and the outcome range declared from the law | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.000702 to 0.000815 | 0.9542 | 1.0251 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `crossfit_learned: ate_policy[+1.0 uncapped vs current practice]` | mean outcome with each dose raised by 1.0, minus the natural course; cross-fitted on three folds, the same learned density, and the outcome range declared from the law | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0011 to 0.0024 | 0.9583 | 1.0382 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `crossfit_learned: ey_policy[+0.5 capped at 5]` | mean outcome with each dose raised by 0.5 and capped at 5; cross-fitted on three folds, the same learned density, and the outcome range declared from the law | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0032 to 0.0053 | 0.9542 | 1.0064 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `crossfit_learned: ey_policy[+0.5 uncapped]` | mean outcome with each dose raised by 0.5; cross-fitted on three folds, the same learned density, and the outcome range declared from the law | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0031 to 0.0056 | 0.9533 | 1.0045 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `crossfit_learned: ey_policy[+1.0 uncapped]` | mean outcome with each dose raised by 1.0; cross-fitted on three folds, the same learned density, and the outcome range declared from the law | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0032 to 0.0068 | 0.9517 | 1.0092 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `crossfit_learned: ey_policy[current practice]` | mean outcome under the natural course; cross-fitted on three folds, the same learned density, and the outcome range declared from the law | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0027 to 0.0051 | 0.9450 | 1.0037 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_learned: ate_policy[+0.5 capped at 5 vs current practice]` | mean outcome with each dose raised by 0.5 and capped at 5, minus the natural course; in sample, the density a regularized HistGradientBoostingClassifier on 40 core bins | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.000983 to 0.000575 | 0.9325 | 0.9394 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_learned: ate_policy[+0.5 uncapped vs current practice]` | mean outcome with each dose raised by 0.5, minus the natural course; in sample, the density a regularized HistGradientBoostingClassifier on 40 core bins | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.000875 to 0.000658 | 0.9267 | 0.9136 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_learned: ate_policy[+1.0 uncapped vs current practice]` | mean outcome with each dose raised by 1.0, minus the natural course; in sample, the density a regularized HistGradientBoostingClassifier on 40 core bins | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0023 to 0.0013 | 0.8967 | 0.8025 | **fail** |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_learned: ey_policy[+0.5 capped at 5]` | mean outcome with each dose raised by 0.5 and capped at 5; in sample, the density a regularized HistGradientBoostingClassifier on 40 core bins | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0032 to 0.0052 | 0.9550 | 1.0013 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_learned: ey_policy[+0.5 uncapped]` | mean outcome with each dose raised by 0.5; in sample, the density a regularized HistGradientBoostingClassifier on 40 core bins | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0033 to 0.0054 | 0.9508 | 1.0002 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_learned: ey_policy[+1.0 uncapped]` | mean outcome with each dose raised by 1.0; in sample, the density a regularized HistGradientBoostingClassifier on 40 core bins | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0043 to 0.0056 | 0.9508 | 0.9773 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_learned: ey_policy[current practice]` | mean outcome under the natural course; in sample, the density a regularized HistGradientBoostingClassifier on 40 core bins | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0027 to 0.0051 | 0.9450 | 1.0037 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_oracle: ate_policy[+0.5 capped at 5 vs current practice]` | mean outcome with each dose raised by 0.5 and capped at 5, minus the natural course; in sample, the exact density of the law on the fit's own grid | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.000827 to 0.000632 | 0.9558 | 1.0033 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_oracle: ate_policy[+0.5 uncapped vs current practice]` | mean outcome with each dose raised by 0.5, minus the natural course; in sample, the exact density of the law on the fit's own grid | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.000729 to 0.000712 | 0.9500 | 0.9892 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_oracle: ate_policy[+1.0 uncapped vs current practice]` | mean outcome with each dose raised by 1.0, minus the natural course; in sample, the exact density of the law on the fit's own grid | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0023 to 0.0011 | 0.9442 | 0.9570 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_oracle: ey_policy[+0.5 capped at 5]` | mean outcome with each dose raised by 0.5 and capped at 5; in sample, the exact density of the law on the fit's own grid | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0031 to 0.0053 | 0.9517 | 1.0017 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_oracle: ey_policy[+0.5 uncapped]` | mean outcome with each dose raised by 0.5; in sample, the exact density of the law on the fit's own grid | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0032 to 0.0055 | 0.9500 | 1.0005 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_oracle: ey_policy[+1.0 uncapped]` | mean outcome with each dose raised by 1.0; in sample, the exact density of the law on the fit's own grid | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0044 to 0.0055 | 0.9500 | 0.9911 | pass |
| the shift law of `make_shift_dose` (`shift_dgp()` at its defaults): three normal covariates, a normal dose whose mean depends on them, and a quadratic outcome, at n = 3,000 | `in_sample_oracle: ey_policy[current practice]` | mean outcome under the natural course; in sample, the exact density of the law on the fit's own grid | `cleverly` TMLE of a point additive shift on the density route, with the outcome learner degree-2 least squares and the arm the estimand names fitting the dose density | -0.0027 to 0.0051 | 0.9450 | 1.0037 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each cross-fitted contrast lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, at n = 3,000 | a shrunken-SE control multiplies each SE by 0.70, and a noise control adds independent noise to each estimate. Each must put its SE-ratio interval below 0.93 |

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `crossfit_learned_capped__declared` | positive | the contrast of the +0.5 shift capped at 5 against the natural course, cross-fitted with the learned density, on the shift law at n = 3,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9557 to 0.9630, SE ratio 1.0285 to 1.0561 | pass |
| `interval_calibration` | `crossfit_learned_capped__noise_control` | control | the same cross-fitted fit of the +0.5 shift capped at 5, with independent normal noise added to each estimate at a declared scale, the law's influence-curve SD over the square root of n | the SE-ratio interval must fall below the calibration band | coverage 0.8506 to 0.8635, SE ratio 0.7444 to 0.7639 | pass |
| `interval_calibration` | `crossfit_learned_capped__shrunken_se_control` | control | the same cross-fitted fit of the +0.5 shift capped at 5, its reported standard errors multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8415 to 0.8548, SE ratio 0.7201 to 0.7393 | pass |
| `interval_calibration` | `crossfit_learned_one__declared` | positive | the contrast of the +1.0 shift against the natural course, cross-fitted with the learned density, on the shift law at n = 3,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9502 to 0.9579, SE ratio 1.0180 to 1.0471 | pass |
| `interval_calibration` | `crossfit_learned_one__noise_control` | control | the same cross-fitted fit of the +1.0 shift, with independent normal noise added to each estimate at a declared scale, the law's influence-curve SD over the square root of n | the SE-ratio interval must fall below the calibration band | coverage 0.8372 to 0.8506, SE ratio 0.7358 to 0.7569 | pass |
| `interval_calibration` | `crossfit_learned_one__shrunken_se_control` | control | the same cross-fitted fit of the +1.0 shift, its reported standard errors multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8224 to 0.8363, SE ratio 0.7126 to 0.7331 | pass |
| `interval_calibration` | `crossfit_learned_uncapped__declared` | positive | the contrast of the +0.5 shift against the natural course, cross-fitted with the learned density, on the shift law at n = 3,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9497 to 0.9575, SE ratio 0.9963 to 1.0226 | pass |
| `interval_calibration` | `crossfit_learned_uncapped__noise_control` | control | the same cross-fitted fit of the +0.5 shift, with independent normal noise added to each estimate at a declared scale, the law's influence-curve SD over the square root of n | the SE-ratio interval must fall below the calibration band | coverage 0.8442 to 0.8574, SE ratio 0.7268 to 0.7458 | pass |
| `interval_calibration` | `crossfit_learned_uncapped__shrunken_se_control` | control | the same cross-fitted fit of the +0.5 shift, its reported standard errors multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8267 to 0.8405, SE ratio 0.6971 to 0.7157 | pass |
<!-- /generated -->

## Result

Every primary verdict and every `interval_calibration` verdict of the cross-fitted arm passed, and
all six controls failed as designed, so the reading added the support row
`T-point-shift-learned-density`. A cross-fitted density-route fit on a bounded continuous outcome
whose density learner has exactly the measured booster's settings, on 40 bins, now reads
`supported`. The declaration pinned only the booster class; the row was narrowed before
registration to the settings and bin count measured, because a cross-fitted default booster
matched the class and its SE is several times its spread in the probe of the
[modified treatment policies](../point-treatment-tmle.md#modified-treatment-policies) reference.
Narrowing licenses less and moves no verdict. One primary cell of the in-sample learned arm is
red: the `+1.0` contrast covers 0.8967 with an SE ratio of 0.8025. No fit failed.

| reading | value | source |
| --- | --- | --- |
| calibration SE ratio, `+0.5` capped | 1.0419 | `properties.csv` |
| calibration SE ratio, `+0.5` uncapped | 1.0090 | `properties.csv` |
| calibration SE ratio, `+1.0` | 1.0322 | `properties.csv` |
| calibration coverage, every cross-fitted contrast | 0.9537 to 0.9595 | `properties.csv` |
| primary coverage, cross-fitted arm | 0.9450 to 0.9625 | `performance-tests.csv` |
| primary coverage, in-sample learned arm | 0.8967 to 0.9550 | `performance-tests.csv` |
| primary coverage, oracle arm | 0.9442 to 0.9558 | `performance-tests.csv` |

### The size of the interval

The SE ratio is the mean reported SE over the empirical SD of the estimates. The natural-course
mean needs no density ratio, and the three arms read 0.9450 on it.

| contrast | in-sample learned SE ratio (99% interval) | coverage | cross-fitted SE ratio (99% interval) | coverage | oracle SE ratio | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| `+0.5` capped at 5 | 0.9394 (0.8926 to 0.9913) | 0.9325 | 1.0515 (0.9999 to 1.1099) | 0.9625 | 1.0033 | 0.9558 |
| `+0.5` uncapped | 0.9136 (0.8669 to 0.9658) | 0.9267 | 1.0251 (0.9721 to 1.0824) | 0.9542 | 0.9892 | 0.9500 |
| `+1.0` uncapped | 0.8025 (0.7628 to 0.8484) | 0.8967 | 1.0382 (0.9853 to 1.1000) | 0.9583 | 0.9570 | 0.9442 |

The in-sample learned interval is narrow on every contrast: each of its three SE-ratio intervals
lies wholly below 1, and the `+0.5` cells pass only the wide screens. The oracle arm, fitted on the
same draws in sample, is close to calibrated, and so is the learned density cross-fitted. The
shortfall appears only when the density is learned and read on the same rows; the cross-fitted arm
also differs in its outcome fit and its declared outcome range, so the comparison does not isolate
one cause. The harness records the fitted density ratio at the observed doses. For the `+1.0`
shift its mean SD over the draws is 1.0925 in sample, against 1.3109 for the true ratio, so the
in-sample ratio is less variable than the true one, which fits a narrow interval but is not shown
to cause it. The cross-fitted ratio is noisier, with a mean SD of 1.9701, and its interval is
slightly wide: at 19,600 draws the capped contrast's SE-ratio interval lies wholly above 1, from
1.0285, while staying inside the calibration band.

### What the controls show

Each shrunken-SE control reads an SE ratio of 0.7063 to 0.7294, and each noise control 0.7360 to
0.7540. All six SE-ratio intervals end below 0.93, so the instrument resolves an interval about a
quarter too short on each contrast.

The readings do not show the items in this list.

- An in-sample learned density. It adds no row and reads `experimental`.
- Another law, sample size, outcome learner or policy kind. The row pins the density learner, its
  settings and the bin count, and reads none of these, so such a fit can read `supported` on the
  strength of this one law; the study's outcome model was correct for the law.
- A cross-fitted fit without a declared outcome range. The study declared it from the law, which
  real data do not supply.
- Other policy kinds than additive shifts, with or without a cap.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1200 | primary replications |
| `n` | 3000 | observations per replication |
| `independent_tests_total` | 21 | primary tests against the known truth |
| `independent_tests_passed` | 20 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 9 | gated repeated-sampling property cells |
| `property_cells_passed` | 9 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0272 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.8967 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.8720 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.7628 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1099 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+0.5 capped at 5 vs current practice]]:se_ratio` | 0.9394 | in-sample learned SE ratio, `+0.5` capped |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+0.5 capped at 5 vs current practice]]:se_ratio_ci_lower` | 0.8926 | its lower endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+0.5 capped at 5 vs current practice]]:se_ratio_ci_upper` | 0.9913 | its upper endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+0.5 capped at 5 vs current practice]]:coverage` | 0.9325 | its coverage |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+0.5 uncapped vs current practice]]:se_ratio` | 0.9136 | in-sample learned SE ratio, `+0.5` uncapped |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+0.5 uncapped vs current practice]]:se_ratio_ci_lower` | 0.8669 | its lower endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+0.5 uncapped vs current practice]]:se_ratio_ci_upper` | 0.9658 | its upper endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+0.5 uncapped vs current practice]]:coverage` | 0.9267 | its coverage |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+1.0 uncapped vs current practice]]:se_ratio` | 0.8025 | in-sample learned SE ratio, `+1.0` |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+1.0 uncapped vs current practice]]:se_ratio_ci_lower` | 0.7628 | its lower endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+1.0 uncapped vs current practice]]:se_ratio_ci_upper` | 0.8484 | its upper endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ate_policy[+1.0 uncapped vs current practice]]:coverage` | 0.8967 | its coverage |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_learned: ey_policy[+0.5 capped at 5]]:coverage` | 0.9550 | the highest in-sample learned coverage, a policy mean |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+0.5 capped at 5 vs current practice]]:se_ratio` | 1.0515 | cross-fitted SE ratio, `+0.5` capped |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+0.5 capped at 5 vs current practice]]:se_ratio_ci_lower` | 0.9999 | its lower endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+0.5 capped at 5 vs current practice]]:se_ratio_ci_upper` | 1.1099 | its upper endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+0.5 capped at 5 vs current practice]]:coverage` | 0.9625 | its coverage, the highest of the arm |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+0.5 uncapped vs current practice]]:se_ratio` | 1.0251 | cross-fitted SE ratio, `+0.5` uncapped |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+0.5 uncapped vs current practice]]:se_ratio_ci_lower` | 0.9721 | its lower endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+0.5 uncapped vs current practice]]:se_ratio_ci_upper` | 1.0824 | its upper endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+0.5 uncapped vs current practice]]:coverage` | 0.9542 | its coverage |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+1.0 uncapped vs current practice]]:se_ratio` | 1.0382 | cross-fitted SE ratio, `+1.0` |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+1.0 uncapped vs current practice]]:se_ratio_ci_lower` | 0.9853 | its lower endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+1.0 uncapped vs current practice]]:se_ratio_ci_upper` | 1.1000 | its upper endpoint |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ate_policy[+1.0 uncapped vs current practice]]:coverage` | 0.9583 | its coverage |
| `performance[cleverly-point-shift-learned-density/shift_law/crossfit_learned: ey_policy[current practice]]:coverage` | 0.9450 | the lowest cross-fitted coverage, a policy mean |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_oracle: ate_policy[+0.5 capped at 5 vs current practice]]:se_ratio` | 1.0033 | oracle SE ratio, `+0.5` capped |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_oracle: ate_policy[+0.5 capped at 5 vs current practice]]:coverage` | 0.9558 | its coverage, the highest of the arm |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_oracle: ate_policy[+0.5 uncapped vs current practice]]:se_ratio` | 0.9892 | oracle SE ratio, `+0.5` uncapped |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_oracle: ate_policy[+0.5 uncapped vs current practice]]:coverage` | 0.9500 | its coverage |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_oracle: ate_policy[+1.0 uncapped vs current practice]]:se_ratio` | 0.9570 | oracle SE ratio, `+1.0` |
| `performance[cleverly-point-shift-learned-density/shift_law/in_sample_oracle: ate_policy[+1.0 uncapped vs current practice]]:coverage` | 0.9442 | its coverage, the lowest of the arm |
| `properties[interval_calibration/crossfit_learned_capped__declared]:se_ratio` | 1.0419 | calibration SE ratio, `+0.5` capped |
| `properties[interval_calibration/crossfit_learned_capped__declared]:se_ratio_ci_lower` | 1.0285 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/crossfit_learned_capped__declared]:coverage` | 0.9595 | its coverage |
| `properties[interval_calibration/crossfit_learned_uncapped__declared]:se_ratio` | 1.0090 | calibration SE ratio, `+0.5` uncapped |
| `properties[interval_calibration/crossfit_learned_uncapped__declared]:coverage` | 0.9537 | its coverage |
| `properties[interval_calibration/crossfit_learned_one__declared]:se_ratio` | 1.0322 | calibration SE ratio, `+1.0` |
| `properties[interval_calibration/crossfit_learned_one__declared]:coverage` | 0.9542 | its coverage |
| `properties[interval_calibration/crossfit_learned_uncapped__shrunken_se_control]:se_ratio` | 0.7063 | the lowest shrunken-SE control SE ratio |
| `properties[interval_calibration/crossfit_learned_capped__shrunken_se_control]:se_ratio` | 0.7294 | the highest shrunken-SE control SE ratio |
| `properties[interval_calibration/crossfit_learned_uncapped__noise_control]:se_ratio` | 0.7360 | the lowest noise control SE ratio |
| `properties[interval_calibration/crossfit_learned_capped__noise_control]:se_ratio` | 0.7540 | the highest noise control SE ratio |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control applies to each SE |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:subject_failures_allowed` | 12 | failed primary fits that bound allows per setting |
| `bound:crossfit_learned_capped_standard_error` | 0.0099 | the efficiency bound over $\sqrt{n}$ for `+0.5` capped, the noise control's scale |
| `bound:crossfit_learned_uncapped_standard_error` | 0.0097 | the efficiency bound over $\sqrt{n}$ for `+0.5` uncapped, the noise control's scale |
| `bound:crossfit_learned_one_standard_error` | 0.0223 | the efficiency bound over $\sqrt{n}$ for `+1.0`, the noise control's scale |

The ratio SDs quoted under "The size of the interval" are means of the `ratio_sd` column of
`harness.csv.gz` over the 1,200 draws, and the true ratio's SD is `TRUE_RATIO_SD` of the study
module, which the design test recomputes.

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No external package fits this estimator with this density learner. The row rests on accuracy against the known truth and on the calibration cells |
| One law and one sample size | The study fits `shift_dgp()` at n = 3,000 only. The support key holds neither, so a cross-fitted fit of another law or size with the same booster settings also reads `supported` |
| One booster setting and one bin count | The row pins the regularized `HistGradientBoostingClassifier`, every setting but its seed, and 40 bins, so another learner, setting or bin count reads `experimental`. The default booster, cross-fitted with a `HistGradientBoostingRegressor` outcome learner, gave an SE about four times the empirical SD in the probe of the [modified treatment policies](../point-treatment-tmle.md#modified-treatment-policies) reference |
| One outcome learner and one policy kind | The study's degree-2 least-squares outcome model is correct for the law, and its policies are additive shifts. The row reads neither the outcome learner nor the policy kind, so a fit with a flexible outcome learner or a multiplicative or piecewise policy matches it on this evidence alone |
| The outcome range was declared from the law | A cross-fitted fit of a continuous outcome needs `q_bounds`. On real data the range is the analyst's declaration, and an outcome outside it refuses the fit |
| An in-sample learned density is not covered | Its interval is narrow on every contrast and covers 0.8967 for the `+1.0` shift. Such a fit reads `experimental`, and the red cell stays published under `reporting` |
| The cross-fitted interval is slightly wide | The capped contrast's calibration SE-ratio interval lies above 1, inside the band |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_shift_learned_density/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_shift_learned_density/run.log)
records the commit, the clean tree and the runtime. The command
`python -m tests.canonical.point_shift_learned_density.regenerate` reruns the study, and
`python -m tests.studies.point_shift_learned_density_pilot --raw tests/studies/_x121_pilot_raw --summarize`
rebuilds the pilot table
from its committed raw draws. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_shift_learned_density/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_shift_learned_density/properties.csv)
carry every published row.
