# Projection compositions

This study validates four compositions of the projection TMLE of a conditional contrast, the
estimator that [conditional-projection](conditional-projection.md) validates on its own: a missing
treatment fitted in sample through the composite indicator, baseline strata (`strata=`), repeated
cross-fitting (`repeats=3`) and the full-refit bootstrap. The target is the coefficient vector of
the least-squares projection of the CATE on a fixed basis, on the difference scale, and of the log
conditional risk ratio on the `log_ratio` scale. [Conditional effects](../conditional-effects.md)
gives the target, the estimator and its conditions for each composition.

**No canonical implementation is compared.** No maintained package fits a projection TMLE of the
CATE on a declared basis with any of these compositions. A zero-row equivalence artifact records
the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | conditional-projection's `observational` law; `mtreat`: the same outcome and treatment with the treatment recorded with $P(\Delta_A = 1 \mid A, W) = \operatorname{expit}(1 + 2 W_3 - 0.5 A)$, and `mtreat_both` adds outcomes missing at random; `strata`: a stratum $S \sim \text{Bernoulli}(0.4)$ independent of $W$, with the arm-1 logit increment shifted by $0.5 S - 0.4 S W_1$; `hetero_trial`: conditional-projection's randomized law with the risk-scale CATE $0.2 + s W_1$ |
| bases | `Linear(["W1", "W3"])` on $(W_1, W_3)$, and `Linear(["W1"])` on $W_1$ on the `mtreat` laws, so $W_3$, which drives the recording, lies outside the span |
| groups | `insample_2k` and `insample_crr_2k` (in sample, Wald and full-refit bootstrap), `mtreat_4k`, `mtreat_both_4k`, `mtreat_crr_4k` (in-sample composite route), `strata_4k`, `strata_crr_4k` (stacked, `strata=("S",)`; on the log ratio the marginal names come from the unstratified log-ratio group), `rep3_obs_4k`, `rep3_crr_4k` (stacked, `repeats=3`, the median rule), and `boot_stacked_2k` (the cross-fitted bootstrap, diagnostic) |
| learners | a spline GLM per arm for $\mu$ (with the raw columns $S$ and $S W_1$ on the `strata` law) and logistic GLMs for the treatment, recording and observation mechanisms; `mtreat_dr_4k` fits a declared wrong outcome regression |
| truth | Gauss-Legendre quadrature over $U(-1, 1)^3$, checked in the fast tier |
| primary budget | 1,000 replications of `mtreat_4k` at n = 4,000 |
| calibration budget | R = 20,400 per gated group, computed before the run from a 3,200-draw pilot on a disjoint stream (`tests/studies/projection_compositions_pilot.csv`) |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20261401` and `resampling_seed=20261402`; the run, the pilot and the smoke each draw from their own labelled streams, with fold seeds distinct per group |
| policy | `gated` |

The laws, groups, learners, budgets, controls and seeds were fixed before any run, in the P1 plan,
its two design reviews and `tests/studies/projection_compositions.py`.
`tests/unit/test_projection_compositions_design.py` checks the truths, the declared cells, the
pilot rule and the budget.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the missing-treatment law, treatment recorded with P = expit(1 + 2 W3 - 0.5 A), in-sample composite fit, n = 4,000 | `cate_projection[(intercept)]` | the intercept of the least-squares projection of the CATE on (1, W1) | `cleverly` projection TMLE of a conditional contrast: the coefficients of the least-squares projection of the CATE on a fixed basis, targeted jointly, with an influence-curve Wald interval | -0.0030 to 0.000472 | 0.9420 | 0.9913 | pass |
| the missing-treatment law, treatment recorded with P = expit(1 + 2 W3 - 0.5 A), in-sample composite fit, n = 4,000 | `cate_projection[W1]` | the W1 slope of the least-squares projection of the CATE on (1, W1) | `cleverly` projection TMLE of a conditional contrast: the coefficients of the least-squares projection of the CATE on a fixed basis, targeted jointly, with an influence-curve Wald interval | -0.0017 to 0.0043 | 0.9410 | 0.9709 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | for each coefficient of nine gated groups (R = 20,400), does the SE-ratio interval lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98 | the in-sample bootstrap interval shrunk by 0.70 about its median, and the median-rule SE divided by $\sqrt{3}$ (three fold draws read as independent samples), must fall below the band |
| `bias` | on the `mtreat` law with a declared wrong outcome regression (R = 1,000), is each coefficient's standardized-bias interval inside $\pm 0.25$ | each control's interval must lie wholly outside $\pm 0.25$: the complete-case projection on the rows with a recorded treatment, the one-step without the recording probability, and the marginal coefficients read against the stratum-1 truth |
| `type_i_error` | with no heterogeneity and `repeats=3` (n = 2,000, R = 2,400), is the 99% upper end of the df-2 test's rejection rate at most 0.10 | paired with the power cell |
| `power` | at slope 0.10 and `repeats=3` (n = 4,000, R = 1,000), is the rejection rate's lower end at least 0.80 | an inert test fails it |

The cross-fitted bootstrap (`boot_stacked_2k`, R = 1,000) publishes its coverage and SE ratio under
the diagnostic role and states no verdict. It is the evidence the cross-fitted bootstrap's
experimental label cites.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bias` | `complete_case_4k_intercept` | control | the missing-treatment law, n = 4,000: the projection fitted on the rows with a recorded treatment, with the same learners, intercept | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0172 to 0.0179, margin 0.0049 | pass |
| `bias` | `marginal_as_stratum_4k_W1` | control | the two-stratum law, n = 4,000: the marginal coefficient read against the stratum S = 1 truth, W1 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0571 to 0.0580, margin 0.0067 | pass |
| `bias` | `marginal_as_stratum_4k_intercept` | control | the two-stratum law, n = 4,000: the marginal coefficient read against the stratum S = 1 truth, intercept | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0693 to -0.0687, margin 0.0039 | pass |
| `bias` | `mtreat_dr_4k_W1` | positive | the missing-treatment law with a declared wrong outcome regression, n = 4,000: the composite TMLE's W1 slope | the 99% interval of the standardized bias inside +-0.25 | bias -0.0030 to 0.0029, margin 0.0090 | pass |
| `bias` | `mtreat_dr_4k_intercept` | positive | the missing-treatment law with a declared wrong outcome regression, n = 4,000: the composite TMLE's intercept | the 99% interval of the standardized bias inside +-0.25 | bias -0.0027 to 0.000791, margin 0.0053 | pass |
| `bias` | `no_treatment_observation_4k_intercept` | control | the missing-treatment law with a declared wrong outcome regression, n = 4,000: the one-step with the treatment-recording probability replaced by 1, intercept | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0551 to -0.0530, margin 0.0032 | pass |
| `interval_calibration` | `boot_stacked_2k_W1__boot_stacked` | diagnostic | the observational law, stacked cross-fitting with the full-refit bootstrap, n = 2,000: the W1 slope of the projection on (1, W1, W3): the bootstrap percentile interval of the cross-fitted fit, 200 replicates; diagnostic, no verdict | publishes its coverage and SE-ratio intervals; no verdict | coverage 0.9032 to 0.9466, SE ratio 0.9150 to 1.0225 | reported |
| `interval_calibration` | `boot_stacked_2k_W3__boot_stacked` | diagnostic | the observational law, stacked cross-fitting with the full-refit bootstrap, n = 2,000: the W3 slope of the projection on (1, W1, W3): the bootstrap percentile interval of the cross-fitted fit, 200 replicates; diagnostic, no verdict | publishes its coverage and SE-ratio intervals; no verdict | coverage 0.9318 to 0.9677, SE ratio 0.9524 to 1.0697 | reported |
| `interval_calibration` | `boot_stacked_2k_intercept__boot_stacked` | diagnostic | the observational law, stacked cross-fitting with the full-refit bootstrap, n = 2,000: the intercept of the projection on (1, W1, W3): the bootstrap percentile interval of the cross-fitted fit, 200 replicates; diagnostic, no verdict | publishes its coverage and SE-ratio intervals; no verdict | coverage 0.9214 to 0.9602, SE ratio 0.9614 to 1.0720 | reported |
| `interval_calibration` | `insample_2k_W1__boot_diff` | positive | the observational law, in-sample fit, n = 2,000: the W1 slope of the projection on (1, W1, W3): the full-refit bootstrap percentile interval, 200 replicates | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9380 to 0.9464, SE ratio 0.9875 to 1.0130 | pass |
| `interval_calibration` | `insample_2k_W1__shrunken_se_control` | control | the observational law, in-sample fit, n = 2,000: the W1 slope of the projection on (1, W1, W3): the percentile interval shrunk by 0.70 about the replicates' median | the SE-ratio interval must fall below the calibration band | coverage 0.8088 to 0.8229, SE ratio 0.6911 to 0.7089 | pass |
| `interval_calibration` | `insample_2k_W1__wald_diff` | positive | the observational law, in-sample fit, n = 2,000: the W1 slope of the projection on (1, W1, W3): the influence-curve Wald interval | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9459 to 0.9538, SE ratio 0.9833 to 1.0085 | pass |
| `interval_calibration` | `insample_2k_W3__boot_diff` | positive | the observational law, in-sample fit, n = 2,000: the W3 slope of the projection on (1, W1, W3): the full-refit bootstrap percentile interval, 200 replicates | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9369 to 0.9454, SE ratio 0.9799 to 1.0046 | pass |
| `interval_calibration` | `insample_2k_W3__shrunken_se_control` | control | the observational law, in-sample fit, n = 2,000: the W3 slope of the projection on (1, W1, W3): the percentile interval shrunk by 0.70 about the replicates' median | the SE-ratio interval must fall below the calibration band | coverage 0.8061 to 0.8202, SE ratio 0.6856 to 0.7033 | pass |
| `interval_calibration` | `insample_2k_W3__wald_diff` | positive | the observational law, in-sample fit, n = 2,000: the W3 slope of the projection on (1, W1, W3): the influence-curve Wald interval | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9522, SE ratio 0.9751 to 1.0004 | pass |
| `interval_calibration` | `insample_2k_intercept__boot_diff` | positive | the observational law, in-sample fit, n = 2,000: the intercept of the projection on (1, W1, W3): the full-refit bootstrap percentile interval, 200 replicates | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9366 to 0.9451, SE ratio 0.9859 to 1.0118 | pass |
| `interval_calibration` | `insample_2k_intercept__shrunken_se_control` | control | the observational law, in-sample fit, n = 2,000: the intercept of the projection on (1, W1, W3): the percentile interval shrunk by 0.70 about the replicates' median | the SE-ratio interval must fall below the calibration band | coverage 0.8101 to 0.8241, SE ratio 0.6902 to 0.7082 | pass |
| `interval_calibration` | `insample_2k_intercept__wald_diff` | positive | the observational law, in-sample fit, n = 2,000: the intercept of the projection on (1, W1, W3): the influence-curve Wald interval | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9536, SE ratio 0.9823 to 1.0081 | pass |
| `interval_calibration` | `insample_crr_2k_W1__boot_crr` | positive | the observational law on the log-ratio scale, in-sample fit, n = 2,000: the W1 slope of the projection on (1, W1, W3): the full-refit bootstrap percentile interval, 200 replicates | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9377 to 0.9461, SE ratio 0.9926 to 1.0184 | pass |
| `interval_calibration` | `insample_crr_2k_W1__wald_crr` | positive | the observational law on the log-ratio scale, in-sample fit, n = 2,000: the W1 slope of the projection on (1, W1, W3): the influence-curve Wald interval | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9445 to 0.9525, SE ratio 0.9752 to 1.0007 | pass |
| `interval_calibration` | `insample_crr_2k_W3__boot_crr` | positive | the observational law on the log-ratio scale, in-sample fit, n = 2,000: the W3 slope of the projection on (1, W1, W3): the full-refit bootstrap percentile interval, 200 replicates | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9355 to 0.9442, SE ratio 0.9903 to 1.0161 | pass |
| `interval_calibration` | `insample_crr_2k_W3__wald_crr` | positive | the observational law on the log-ratio scale, in-sample fit, n = 2,000: the W3 slope of the projection on (1, W1, W3): the influence-curve Wald interval | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9530, SE ratio 0.9795 to 1.0043 | pass |
| `interval_calibration` | `insample_crr_2k_intercept__boot_crr` | positive | the observational law on the log-ratio scale, in-sample fit, n = 2,000: the intercept of the projection on (1, W1, W3): the full-refit bootstrap percentile interval, 200 replicates | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9360 to 0.9446, SE ratio 0.9936 to 1.0190 | pass |
| `interval_calibration` | `insample_crr_2k_intercept__wald_crr` | positive | the observational law on the log-ratio scale, in-sample fit, n = 2,000: the intercept of the projection on (1, W1, W3): the influence-curve Wald interval | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9517, SE ratio 0.9774 to 1.0024 | pass |
| `interval_calibration` | `mtreat_4k_W1__mtreat` | positive | the missing-treatment law, treatment recorded with P = expit(1 + 2 W3 - 0.5 A), in-sample composite fit, n = 4,000: the W1 slope of the projection on (1, W1): the Wald interval of the composite route | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9396 to 0.9480, SE ratio 0.9704 to 0.9953 | pass |
| `interval_calibration` | `mtreat_4k_intercept__mtreat` | positive | the missing-treatment law, treatment recorded with P = expit(1 + 2 W3 - 0.5 A), in-sample composite fit, n = 4,000: the intercept of the projection on (1, W1): the Wald interval of the composite route | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9436 to 0.9517, SE ratio 0.9773 to 1.0022 | pass |
| `interval_calibration` | `mtreat_both_4k_W1__mtreat_both` | positive | the missing-treatment law with outcomes also missing at random, in-sample composite fit, n = 4,000: the W1 slope of the projection on (1, W1): the Wald interval of the composite route | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9419 to 0.9501, SE ratio 0.9773 to 1.0024 | pass |
| `interval_calibration` | `mtreat_both_4k_intercept__mtreat_both` | positive | the missing-treatment law with outcomes also missing at random, in-sample composite fit, n = 4,000: the intercept of the projection on (1, W1): the Wald interval of the composite route | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9389 to 0.9473, SE ratio 0.9644 to 0.9895 | pass |
| `interval_calibration` | `mtreat_crr_4k_W1__mtreat_crr` | positive | the missing-treatment law on the log-ratio scale, in-sample composite fit, n = 4,000: the W1 slope of the projection on (1, W1): the Wald interval of the composite route | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9417 to 0.9499, SE ratio 0.9682 to 0.9935 | pass |
| `interval_calibration` | `mtreat_crr_4k_intercept__mtreat_crr` | positive | the missing-treatment law on the log-ratio scale, in-sample composite fit, n = 4,000: the intercept of the projection on (1, W1): the Wald interval of the composite route | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9432 to 0.9513, SE ratio 0.9766 to 1.0018 | pass |
| `interval_calibration` | `rep3_crr_4k_W1__rep3_crr` | positive | the observational law on the log-ratio scale, stacked cross-fitting with repeats=3, n = 4,000: the W1 slope of the projection on (1, W1, W3): the median-rule Wald interval over three fold draws | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9469 to 0.9547, SE ratio 0.9876 to 1.0128 | pass |
| `interval_calibration` | `rep3_crr_4k_W3__rep3_crr` | positive | the observational law on the log-ratio scale, stacked cross-fitting with repeats=3, n = 4,000: the W3 slope of the projection on (1, W1, W3): the median-rule Wald interval over three fold draws | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9498 to 0.9574, SE ratio 0.9988 to 1.0241 | pass |
| `interval_calibration` | `rep3_crr_4k_intercept__rep3_crr` | positive | the observational law on the log-ratio scale, stacked cross-fitting with repeats=3, n = 4,000: the intercept of the projection on (1, W1, W3): the median-rule Wald interval over three fold draws | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9522, SE ratio 0.9782 to 1.0032 | pass |
| `interval_calibration` | `rep3_obs_4k_W1__independent_splits_control` | control | the observational law, stacked cross-fitting with repeats=3, n = 4,000: the W1 slope of the projection on (1, W1, W3): the reported SE divided by sqrt(3), as if the three fold draws were independent samples | the SE-ratio interval must fall below the calibration band | coverage 0.7345 to 0.7503, SE ratio 0.5718 to 0.5867 | pass |
| `interval_calibration` | `rep3_obs_4k_W1__rep3` | positive | the observational law, stacked cross-fitting with repeats=3, n = 4,000: the W1 slope of the projection on (1, W1, W3): the median-rule Wald interval over three fold draws | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9481 to 0.9558, SE ratio 0.9907 to 1.0167 | pass |
| `interval_calibration` | `rep3_obs_4k_W3__independent_splits_control` | control | the observational law, stacked cross-fitting with repeats=3, n = 4,000: the W3 slope of the projection on (1, W1, W3): the reported SE divided by sqrt(3), as if the three fold draws were independent samples | the SE-ratio interval must fall below the calibration band | coverage 0.7336 to 0.7494, SE ratio 0.5681 to 0.5829 | pass |
| `interval_calibration` | `rep3_obs_4k_W3__rep3` | positive | the observational law, stacked cross-fitting with repeats=3, n = 4,000: the W3 slope of the projection on (1, W1, W3): the median-rule Wald interval over three fold draws | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9523, SE ratio 0.9837 to 1.0096 | pass |
| `interval_calibration` | `rep3_obs_4k_intercept__independent_splits_control` | control | the observational law, stacked cross-fitting with repeats=3, n = 4,000: the intercept of the projection on (1, W1, W3): the reported SE divided by sqrt(3), as if the three fold draws were independent samples | the SE-ratio interval must fall below the calibration band | coverage 0.7365 to 0.7523, SE ratio 0.5690 to 0.5834 | pass |
| `interval_calibration` | `rep3_obs_4k_intercept__rep3` | positive | the observational law, stacked cross-fitting with repeats=3, n = 4,000: the intercept of the projection on (1, W1, W3): the median-rule Wald interval over three fold draws | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9523, SE ratio 0.9853 to 1.0110 | pass |
| `interval_calibration` | `strata_4k_W1_S0__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the W1 slope of the projection on (1, W1, W3) in stratum S = 0: the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9465 to 0.9544, SE ratio 0.9901 to 1.0157 | pass |
| `interval_calibration` | `strata_4k_W1_S1__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the W1 slope of the projection on (1, W1, W3) in stratum S = 1: the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9449 to 0.9529, SE ratio 0.9813 to 1.0064 | pass |
| `interval_calibration` | `strata_4k_W1__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the W1 slope of the projection on (1, W1, W3): the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9457 to 0.9536, SE ratio 0.9843 to 1.0092 | pass |
| `interval_calibration` | `strata_4k_W3_S0__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the W3 slope of the projection on (1, W1, W3) in stratum S = 0: the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9415 to 0.9497, SE ratio 0.9767 to 1.0016 | pass |
| `interval_calibration` | `strata_4k_W3_S1__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the W3 slope of the projection on (1, W1, W3) in stratum S = 1: the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9536, SE ratio 0.9957 to 1.0208 | pass |
| `interval_calibration` | `strata_4k_W3__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the W3 slope of the projection on (1, W1, W3): the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9468 to 0.9546, SE ratio 0.9870 to 1.0122 | pass |
| `interval_calibration` | `strata_4k_intercept_S0__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the intercept of the projection on (1, W1, W3) in stratum S = 0: the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9444 to 0.9524, SE ratio 0.9842 to 1.0101 | pass |
| `interval_calibration` | `strata_4k_intercept_S1__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the intercept of the projection on (1, W1, W3) in stratum S = 1: the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9448 to 0.9528, SE ratio 0.9839 to 1.0091 | pass |
| `interval_calibration` | `strata_4k_intercept__strata` | positive | the two-stratum law, S ~ Bernoulli(0.4) with a stratum-specific CATE, strata=('S',), stacked cross-fitting, n = 4,000: the intercept of the projection on (1, W1, W3): the Wald interval, marginal and stratum coefficients targeted jointly | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9450 to 0.9530, SE ratio 0.9782 to 1.0033 | pass |
| `interval_calibration` | `strata_crr_4k_W1_S0__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the W1 slope of the projection on (1, W1, W3) in stratum S = 0: the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9457 to 0.9537, SE ratio 0.9801 to 1.0053 | pass |
| `interval_calibration` | `strata_crr_4k_W1_S1__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the W1 slope of the projection on (1, W1, W3) in stratum S = 1: the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9468 to 0.9546, SE ratio 0.9884 to 1.0133 | pass |
| `interval_calibration` | `strata_crr_4k_W1__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the W1 slope of the projection on (1, W1, W3): the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9462 to 0.9541, SE ratio 0.9858 to 1.0118 | pass |
| `interval_calibration` | `strata_crr_4k_W3_S0__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the W3 slope of the projection on (1, W1, W3) in stratum S = 0: the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9486 to 0.9563, SE ratio 0.9942 to 1.0200 | pass |
| `interval_calibration` | `strata_crr_4k_W3_S1__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the W3 slope of the projection on (1, W1, W3) in stratum S = 1: the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9477 to 0.9555, SE ratio 0.9904 to 1.0156 | pass |
| `interval_calibration` | `strata_crr_4k_W3__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the W3 slope of the projection on (1, W1, W3): the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9488 to 0.9565, SE ratio 0.9947 to 1.0197 | pass |
| `interval_calibration` | `strata_crr_4k_intercept_S0__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the intercept of the projection on (1, W1, W3) in stratum S = 0: the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9439 to 0.9520, SE ratio 0.9800 to 1.0054 | pass |
| `interval_calibration` | `strata_crr_4k_intercept_S1__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the intercept of the projection on (1, W1, W3) in stratum S = 1: the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9487 to 0.9564, SE ratio 0.9970 to 1.0223 | pass |
| `interval_calibration` | `strata_crr_4k_intercept__strata_crr` | positive | the two-stratum law on the log-ratio scale, strata=('S',), stacked cross-fitting, n = 4,000: the intercept of the projection on (1, W1, W3): the Wald interval; the stratum coefficients from the stratified fluctuation and the marginal ones from the unstratified log-ratio group | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9453 to 0.9532, SE ratio 0.9837 to 1.0087 | pass |
| `power` | `heterogeneity__rep3_slope_0.10` | positive | the df-2 heterogeneity test of the W1 and W3 slopes of Linear(W1, W3) on a randomized law with the risk-scale CATE 0.2 + s W1: slope s = 0.10, repeats=3, n = 4,000 | the rejection rate's lower end at least 0.80 | rejection 0.9300, 0.9066 to 0.9492 | pass |
| `type_i_error` | `heterogeneity__rep3_sharp_null` | positive | the df-2 heterogeneity test of the W1 and W3 slopes of Linear(W1, W3) on a randomized law with the risk-scale CATE 0.2 + s W1: slope s = 0, so the effect does not vary, repeats=3, n = 2,000 | the 99% upper end of the rejection rate at most 0.10 | rejection 0.0437, 0.0337 to 0.0557 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. Each of the 50
calibration positives has an SE-ratio interval inside 0.93 to 1.07 and a coverage interval inside
0.92 to 0.98, for the missing treatment in sample, both strata scales with their stratum
coefficients, repeats on both scales, and the in-sample Wald and bootstrap intervals on both
scales. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 2 of 2 | `performance-tests.csv` |
| gated property cells passed | 56 of 56 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

The shrunken bootstrap control reads SE ratios near 0.70 and the $\sqrt{3}$ control near 0.58.
The complete-case control is displaced by 0.90 SD, the one-step without the recording probability
by -4.29 SD, and the marginal-as-stratum control by -4.38 and 2.14 SD. The heterogeneity test under
`repeats=3` rejects at 0.044 under the null and 0.930 at slope 0.10. The cross-fitted bootstrap
covers 0.927 to 0.952 with SE ratios of 0.965 to 1.006.

### Rows that read just below 1

Two gated rows pass with 99% SE-ratio intervals that lie wholly below 1: the `mtreat_both_4k`
intercept (0.9765, upper end 0.9895) and the `mtreat_crr_4k` $W_1$ slope (0.9806, upper end
0.9935). The pilot read these rows low, and a diagnosis declared before the run found no defect.
It used fresh draws on its own stream with a paired oracle at the true nuisances as a control
variate. The shortfall is a finite-sample effect of the in-sample composite construction that the
shipped composite ATE shares, and of the in-sample log ratio. It closes as n grows:

| row | n = 1,000 | n = 2,000 | n = 4,000 | n = 8,000 | n = 16,000 |
| --- | --- | --- | --- | --- | --- |
| composite projection intercept / $W_1$ (`mtreat_both`) | 0.949 / 0.955 | | 0.982 / 0.987 | | 0.9925 / 0.9947 |
| shipped composite ATE, same draws | 0.951 | | 0.983 | | 0.993 |
| in-sample log-ratio intercept / $W_1$ (`insample_crr`) | | 0.989 / 0.987 | | 0.998 / 0.997 | |

The reported standard error is within 1.2% of the exact efficient influence function's SD at
n = 4,000 on every diagnosed row. The run's readings, 0.977 to 0.993 on these rows, sit at or above
the diagnosis's floor of 0.965, so they agree with it. The declared reading treated a red on any of
these rows as a defect suspect; none was red.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 4000 | observations per replication |
| `independent_tests_total` | 2 | parameter tests against the truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 56 | gated repeated-sampling property cells |
| `property_cells_passed` | 56 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0594 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9410 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9191 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9192 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0492 | highest bootstrap primary SE-ratio endpoint |
| `properties[interval_calibration/mtreat_both_4k_intercept__mtreat_both]:se_ratio` | 0.9765 | lowest calibration SE ratio |
| `properties[interval_calibration/mtreat_both_4k_intercept__mtreat_both]:se_ratio_ci_upper` | 0.9895 | its upper endpoint |
| `properties[interval_calibration/mtreat_crr_4k_W1__mtreat_crr]:se_ratio` | 0.9806 | composite log-ratio $W_1$ slope |
| `properties[interval_calibration/mtreat_crr_4k_W1__mtreat_crr]:se_ratio_ci_upper` | 0.9935 | its upper endpoint |
| `properties[interval_calibration/insample_2k_W1__shrunken_se_control]:se_ratio` | 0.6999 | shrunken bootstrap control, $W_1$ |
| `properties[interval_calibration/rep3_obs_4k_W3__independent_splits_control]:se_ratio` | 0.5752 | $\sqrt{3}$ control, $W_3$ |
| `properties[bias/complete_case_4k_intercept]:standardized_bias` | 0.8990 | complete-case control |
| `properties[bias/no_treatment_observation_4k_intercept]:standardized_bias` | -4.2863 | recording-probability control |
| `properties[bias/marginal_as_stratum_4k_intercept]:standardized_bias` | -4.3813 | marginal-as-stratum control, intercept |
| `properties[bias/marginal_as_stratum_4k_W1]:standardized_bias` | 2.1444 | marginal-as-stratum control, $W_1$ |
| `properties[bias/mtreat_dr_4k_intercept]:standardized_bias` | -0.0445 | composite route under a wrong outcome regression |
| `properties[type_i_error/heterogeneity__rep3_sharp_null]:rejection_rate` | 0.0437 | heterogeneity test size, `repeats=3` |
| `properties[power/heterogeneity__rep3_slope_0.10]:rejection_rate` | 0.9300 | heterogeneity test power, `repeats=3` |
| `properties[interval_calibration/boot_stacked_2k_W1__boot_stacked]:coverage` | 0.9270 | cross-fitted bootstrap, lowest coverage |
| `properties[interval_calibration/boot_stacked_2k_W1__boot_stacked]:se_ratio` | 0.9652 | cross-fitted bootstrap, lowest SE ratio |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio sanity screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio sanity screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor of the shrunken bootstrap control |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 10 | failed primary fits the bound allows |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope band, upper limit |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a projection TMLE of the CATE with these compositions. The rows rest on accuracy against the truth and on the theory properties |
| The interval covers the projection | The interval covers the coefficients of the least-squares projection of the CATE on the basis, over the study's modifier distribution, and not the CATE where it lies outside the span of the basis |
| One basis per law, consistent learners | Every calibration group uses `Linear(["W1", "W3"])` or `Linear(["W1"])` with learners consistent for the law and one outcome regression per arm. A misspecified treatment or recording model is not measured; the composite route's double robustness under a wrong outcome regression is measured for bias only |
| The missing treatment in sample | The composite route was measured in sample, with the treatment recorded at random given the treatment and covariates, with and without missing outcomes. With cross-fitting a missing treatment refuses |
| Strata | One binary stratum with probability 0.4 and stacked cross-fitting. More strata, a rare stratum, or strata with a missing treatment are not measured |
| Repeats | `repeats=3` with the median rule on stacked cross-fitting, at GLM nuisances, where the split displacement is under 1% of the SE. The median rule's displacement term itself rests on the unit witnesses; another repeat count or a learner whose fold draws differ more reads supported and rests on the theory |
| The bootstrap | The full-refit percentile bootstrap was measured in sample with 200 replicates. The cross-fitted bootstrap reads experimental: on `boot_stacked_2k` it covers 0.927 to 0.952, published without a verdict |
| Laws and sizes | n = 2,000 and 4,000 on smooth laws. The in-sample composite and the in-sample log ratio read SE ratios 1 to 2% below 1 at these sizes, closing with n (see the result above); smaller n reads a larger shortfall |
| Compositions outside the rows | The EP learner (`method="ep"`) refuses a missing treatment and baseline strata. A missing treatment with cross-fitting, an intermediate variable and `model=None` refuse. The EP rule with a declared sieve or final learner reads experimental until the declared `learned-rule-ep-declared` study is registered |

## Reproduction

The declared run is `python -m tests.canonical.projection_compositions.regenerate`, whose docstring
gives the run form and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_compositions/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_compositions/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_compositions/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_compositions/properties.csv)
carry every published row.
