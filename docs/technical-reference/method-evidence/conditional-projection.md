# Conditional-contrast projection

This study validates the projection TMLE of a conditional contrast,
`ConditionalContrast(Modifiers(["W1", "W3"]), model=Linear(["W1", "W3"]))`. The target is the
coefficient vector $\beta_0$ of the least-squares projection of the CATE on the fixed basis
$(1, W_1, W_3)$, on the difference scale, and of the log conditional risk ratio on the
`log_ratio` scale. The coefficients are targeted jointly, and each carries an influence-curve
Wald interval. [Conditional effects](../conditional-effects.md) gives the target, the estimator
and its conditions.

The CATE of every law is not linear in $W_3$, so the $W_3$ slope is a genuine projection
coefficient and not a coefficient of the true model.

**No canonical implementation is compared.** No maintained package fits a projection TMLE of the
CATE on a declared basis. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | $W \sim U(-1, 1)^3$; $\operatorname{logit} \mu(a, W) = -0.3 + 0.6 W_1 - 0.4 W_2 + a (0.3 + 0.5 W_1 + 0.4 \sin(\pi W_3))$. `trial`: $P(A = 1) = 0.5$ known. `observational`: $\operatorname{logit} \pi = 0.4 W_1 - 0.5 W_2 + 0.3 W_3$. Variants add three arms (contrast 2 against 0), sampling weights $0.25 + 1.75 \operatorname{expit}(6 W_3)$, 800 clusters of 5 rows with $W_1$ and the treatment at cluster level and a $N(0, 1)$ effect on the outcome logit, outcomes missing at random with $\operatorname{logit} \Pi = 1.7 + 0.4 W_1 + 0.5 W_3 - 0.4 a$, or a declared score $s = 0.2 + 0.3 W_1$ to assess |
| fit | `study.identify(ConditionalContrast(...)).estimate(...)` with the learners below and every other setting at its default |
| learners | a spline GLM per arm for $\mu$ and a logistic GLM for $\pi$ and $\Pi$, except `weighted_sl_2k`: the ensemble of a spline GLM, gradient boosting and the lasso with a non-negative log-likelihood meta-learner, five outer folds |
| truth | Gauss-Legendre quadrature over $U(-1, 1)^3$ (Gauss-Hermite over the cluster effect), checked in the fast tier against scrambled Sobol integration and a second solver |
| primary budget | 4,800 replications of `obs_4k` at n = 4,000 |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20261105` and `resampling_seed=20261106`; every property group draws from its own labelled stream |
| policy | `gated` |

The laws, groups, learners, budgets, controls and seeds were fixed before any run, in the P1 plan,
its design review and re-review, and `tests/studies/conditional_projection.py`. The pass
probabilities were computed before the run at SE ratios from a pilot on a disjoint stream
(`tests/studies/conditional_projection_pilot.csv`). `tests/unit/test_conditional_projection_design.py`
checks the truths, the declared cells and those design numbers.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000 | `cate_projection[(intercept)]` | the intercept of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of a conditional contrast: the coefficients of the least-squares projection of the CATE on a fixed basis, targeted jointly, with an influence-curve Wald interval | -0.000587 to 0.000563 | 0.9525 | 1.0043 | pass |
| observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000 | `cate_projection[W1]` | the W1 slope of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of a conditional contrast: the coefficients of the least-squares projection of the CATE on a fixed basis, targeted jointly, with an influence-curve Wald interval | -0.000590 to 0.0014 | 0.9523 | 0.9935 | pass |
| observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000 | `cate_projection[W3]` | the W3 slope of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of a conditional contrast: the coefficients of the least-squares projection of the CATE on a fixed basis, targeted jointly, with an influence-curve Wald interval | -0.0016 to 0.000411 | 0.9527 | 0.9886 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | for each coefficient of 11 groups (R = 4,800, `weighted_sl_2k` R = 3,600), does the SE-ratio interval lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98 | the row-level SE of the intercept and the $W_1$ slope on `clustered_4k`, which ignores the clusters, must fall below the band |
| `bias` | on the slow-nuisance ladder (both nuisances off by $5 n^{-1/3}$ on the logit scale, at n = 1,000, 4,000 and 16,000) and on two laws with a declared wrong outcome regression, is each coefficient's standardized-bias interval inside $\pm 0.25$ | each control's interval must lie wholly outside $\pm 0.25$ of its own spread: the IPW projection at every rung, the T-learner, the one-step with unit weights in place of $1/\pi$, the one-step without $1/\Pi$, and the spline-GLM fit that drops the sampling weights |
| `type_i_error` | with no heterogeneity (n = 2,000, R = 2,400), is the 99% upper end of the df-2 test's rejection rate at most 0.10 | paired with the power cell |
| `power` | at slope 0.10 (n = 4,000, R = 1,000), is the rejection rate's lower end at least 0.80 | an inert test fails it |

The ladder claims bias and not interval calibration. Its nuisances are wrong on purpose, and the
plug-in variance of such a fit overstates the spread at 1k, so the band would read a conservative
interval as a defect. Its coverage is published without a gate.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bias` | `ipw_16k_W1` | control | the slow-nuisance ladder at n = 16,000: the projection of (2A - 1) Y / pi_n(A, W), with no outcome fit, W1 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0134 to -0.0124, margin 0.0035 | pass |
| `bias` | `ipw_16k_W3` | control | the slow-nuisance ladder at n = 16,000: the projection of (2A - 1) Y / pi_n(A, W), with no outcome fit, W3 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0137 to -0.0126, margin 0.0036 | pass |
| `bias` | `ipw_1k_W1` | control | the slow-nuisance ladder at n = 1,000: the projection of (2A - 1) Y / pi_n(A, W), with no outcome fit, W1 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0390 to -0.0346, margin 0.0148 | pass |
| `bias` | `ipw_1k_W3` | control | the slow-nuisance ladder at n = 1,000: the projection of (2A - 1) Y / pi_n(A, W), with no outcome fit, W3 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0388 to -0.0343, margin 0.0151 | pass |
| `bias` | `ipw_4k_W1` | control | the slow-nuisance ladder at n = 4,000: the projection of (2A - 1) Y / pi_n(A, W), with no outcome fit, W1 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0218 to -0.0197, margin 0.0071 | pass |
| `bias` | `ipw_4k_W3` | control | the slow-nuisance ladder at n = 4,000: the projection of (2A - 1) Y / pi_n(A, W), with no outcome fit, W3 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0226 to -0.0205, margin 0.0072 | pass |
| `bias` | `missing_dr_4k_W1` | positive | the missing-outcome law with a declared wrong outcome regression correlated with the observation probability, n = 4,000: the projection TMLE's W1 slope | the 99% interval of the standardized bias inside +-0.25 | bias -0.0021 to 0.0028, margin 0.0074 | pass |
| `bias` | `missing_dr_4k_W3` | positive | the missing-outcome law with a declared wrong outcome regression correlated with the observation probability, n = 4,000: the projection TMLE's W3 slope | the 99% interval of the standardized bias inside +-0.25 | bias -0.0026 to 0.0024, margin 0.0077 | pass |
| `bias` | `missing_dr_4k_intercept` | positive | the missing-outcome law with a declared wrong outcome regression correlated with the observation probability, n = 4,000: the projection TMLE's intercept | the 99% interval of the standardized bias inside +-0.25 | bias -0.0015 to 0.0013, margin 0.0043 | pass |
| `bias` | `missing_ignored_4k_W3` | control | the missing-outcome law with a declared wrong outcome regression correlated with the observation probability, n = 4,000: the one-step without the 1 / Pi weight, W3 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0807 to 0.0847, margin 0.0062 | pass |
| `bias` | `missing_ignored_4k_intercept` | control | the missing-outcome law with a declared wrong outcome regression correlated with the observation probability, n = 4,000: the one-step without the 1 / Pi weight, intercept | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0113 to -0.0090, margin 0.0035 | pass |
| `bias` | `no_weight_4k_W1` | control | the wide-overlap law with a declared wrong outcome regression, n = 4,000: the one-step with unit weights in place of 1 / pi_n, W1 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0796 to 0.0816, margin 0.0032 | pass |
| `bias` | `no_weight_4k_intercept` | control | the wide-overlap law with a declared wrong outcome regression, n = 4,000: the one-step with unit weights in place of 1 / pi_n, intercept | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0366 to 0.0379, margin 0.0020 | pass |
| `bias` | `noweight_dr_4k_W1` | positive | the wide-overlap law with a declared wrong outcome regression, n = 4,000: the projection TMLE's W1 slope | the 99% interval of the standardized bias inside +-0.25 | bias -0.0050 to 0.0019, margin 0.0106 | pass |
| `bias` | `noweight_dr_4k_W3` | positive | the wide-overlap law with a declared wrong outcome regression, n = 4,000: the projection TMLE's W3 slope | the 99% interval of the standardized bias inside +-0.25 | bias -0.0049 to 0.0012, margin 0.0094 | pass |
| `bias` | `noweight_dr_4k_intercept` | positive | the wide-overlap law with a declared wrong outcome regression, n = 4,000: the projection TMLE's intercept | the 99% interval of the standardized bias inside +-0.25 | bias -0.0010 to 0.0024, margin 0.0052 | pass |
| `bias` | `slow_16k_W1` | positive | the slow-nuisance ladder at n = 16,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's W1 slope | the 99% interval of the standardized bias inside +-0.25 | bias 0.000445 to 0.0014, margin 0.0033 | pass |
| `bias` | `slow_16k_W3` | positive | the slow-nuisance ladder at n = 16,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's W3 slope | the 99% interval of the standardized bias inside +-0.25 | bias 0.000714 to 0.0017, margin 0.0034 | pass |
| `bias` | `slow_16k_intercept` | positive | the slow-nuisance ladder at n = 16,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's intercept | the 99% interval of the standardized bias inside +-0.25 | bias 0.000448 to 0.0010, margin 0.0020 | pass |
| `bias` | `slow_1k_W1` | positive | the slow-nuisance ladder at n = 1,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's W1 slope | the 99% interval of the standardized bias inside +-0.25 | bias 0.0038 to 0.0078, margin 0.0134 | pass |
| `bias` | `slow_1k_W3` | positive | the slow-nuisance ladder at n = 1,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's W3 slope | the 99% interval of the standardized bias inside +-0.25 | bias 0.0049 to 0.0089, margin 0.0137 | pass |
| `bias` | `slow_1k_intercept` | positive | the slow-nuisance ladder at n = 1,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's intercept | the 99% interval of the standardized bias inside +-0.25 | bias 0.0038 to 0.0062, margin 0.0079 | pass |
| `bias` | `slow_4k_W1` | positive | the slow-nuisance ladder at n = 4,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's W1 slope | the 99% interval of the standardized bias inside +-0.25 | bias 0.0016 to 0.0036, margin 0.0067 | pass |
| `bias` | `slow_4k_W3` | positive | the slow-nuisance ladder at n = 4,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's W3 slope | the 99% interval of the standardized bias inside +-0.25 | bias 0.0018 to 0.0038, margin 0.0068 | pass |
| `bias` | `slow_4k_intercept` | positive | the slow-nuisance ladder at n = 4,000, both nuisances the correct-family logistic GLM with a logit offset 5 n^(-1/3) h(W), h = W1^2 - 1/3 + W1 W3: the projection TMLE's intercept | the 99% interval of the standardized bias inside +-0.25 | bias 0.0017 to 0.0029, margin 0.0038 | pass |
| `bias` | `t_learner_4k_W1` | control | the slow-nuisance ladder at n = 4,000: the projection of mu_n(1) - mu_n(0), with no targeting, W1 slope | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0596 to 0.0615, margin 0.0062 | pass |
| `bias` | `weights_ignored_2k_intercept` | control | the observational law with sampling weights 0.25 + 1.75 expit(6 W3) and the ensemble learner (spline GLM, gradient boosting and lasso), five outer folds, n = 2,000: the same draws fitted with the spline GLM and without the sampling weights, read against the weighted truth, intercept | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0110 to -0.0091, margin 0.0054 | pass |
| `interval_calibration` | `assess_score_2k_calibration_S__correctly_specified` | positive | the observational law with the declared score s = 0.2 + 0.3 W1, n = 2,000: the slope of the calibration line from the score to the CATE: the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9402 to 0.9568, SE ratio 0.9739 to 1.0275 | pass |
| `interval_calibration` | `assess_score_2k_calibration_intercept__correctly_specified` | positive | the observational law with the declared score s = 0.2 + 0.3 W1, n = 2,000: the intercept of the calibration line from the score to the CATE: the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9404 to 0.9570, SE ratio 0.9742 to 1.0275 | pass |
| `interval_calibration` | `assess_score_2k_risk_difference__correctly_specified` | positive | the observational law with the declared score s = 0.2 + 0.3 W1, n = 2,000: the external score's MSE for the CATE minus the constant's: the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9420 to 0.9583, SE ratio 0.9774 to 1.0303 | pass |
| `interval_calibration` | `clustered_4k_W1__correctly_specified` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level and a N(0, 1) cluster effect on the outcome logit, n = 4,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9386 to 0.9554, SE ratio 0.9852 to 1.0372 | pass |
| `interval_calibration` | `clustered_4k_W1__row_variance_control` | control | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level and a N(0, 1) cluster effect on the outcome logit, n = 4,000: the W1 slope of the projection on (1, W1, W3): the same fits with the row-level standard error, which ignores the clusters | the SE-ratio interval must fall below the calibration band | coverage 0.8644 to 0.8890, SE ratio 0.7641 to 0.8039 | pass |
| `interval_calibration` | `clustered_4k_W3__correctly_specified` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level and a N(0, 1) cluster effect on the outcome logit, n = 4,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9465 to 0.9622, SE ratio 0.9873 to 1.0383 | pass |
| `interval_calibration` | `clustered_4k_intercept__correctly_specified` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level and a N(0, 1) cluster effect on the outcome logit, n = 4,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9393 to 0.9560, SE ratio 0.9771 to 1.0301 | pass |
| `interval_calibration` | `clustered_4k_intercept__row_variance_control` | control | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level and a N(0, 1) cluster effect on the outcome logit, n = 4,000: the intercept of the projection on (1, W1, W3): the same fits with the row-level standard error, which ignores the clusters | the SE-ratio interval must fall below the calibration band | coverage 0.8562 to 0.8814, SE ratio 0.7556 to 0.7968 | pass |
| `interval_calibration` | `crr_1k_W1__correctly_specified` | positive | the observational law on the log-ratio scale, n = 1,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9371 to 0.9541, SE ratio 0.9597 to 1.0114 | pass |
| `interval_calibration` | `crr_1k_W3__correctly_specified` | positive | the observational law on the log-ratio scale, n = 1,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9603, SE ratio 0.9716 to 1.0244 | pass |
| `interval_calibration` | `crr_1k_intercept__correctly_specified` | positive | the observational law on the log-ratio scale, n = 1,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9369 to 0.9539, SE ratio 0.9403 to 0.9947 | pass |
| `interval_calibration` | `crr_4k_W1__correctly_specified` | positive | the observational law on the log-ratio scale, n = 4,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9603, SE ratio 0.9877 to 1.0413 | pass |
| `interval_calibration` | `crr_4k_W3__correctly_specified` | positive | the observational law on the log-ratio scale, n = 4,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9407 to 0.9572, SE ratio 0.9662 to 1.0180 | pass |
| `interval_calibration` | `crr_4k_intercept__correctly_specified` | positive | the observational law on the log-ratio scale, n = 4,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9416 to 0.9579, SE ratio 0.9731 to 1.0247 | pass |
| `interval_calibration` | `multi_4k_W1__correctly_specified` | positive | the three-arm law, contrast of arm 2 with arm 0, n = 4,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9427 to 0.9589, SE ratio 0.9732 to 1.0242 | pass |
| `interval_calibration` | `multi_4k_W3__correctly_specified` | positive | the three-arm law, contrast of arm 2 with arm 0, n = 4,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9418 to 0.9581, SE ratio 0.9733 to 1.0257 | pass |
| `interval_calibration` | `multi_4k_intercept__correctly_specified` | positive | the three-arm law, contrast of arm 2 with arm 0, n = 4,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9382 to 0.9551, SE ratio 0.9728 to 1.0258 | pass |
| `interval_calibration` | `obs_1k_W1__correctly_specified` | positive | the observational law, n = 1,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9431 to 0.9593, SE ratio 0.9823 to 1.0343 | pass |
| `interval_calibration` | `obs_1k_W3__correctly_specified` | positive | the observational law, n = 1,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9407 to 0.9572, SE ratio 0.9778 to 1.0300 | pass |
| `interval_calibration` | `obs_1k_intercept__correctly_specified` | positive | the observational law, n = 1,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9380 to 0.9549, SE ratio 0.9697 to 1.0227 | pass |
| `interval_calibration` | `obs_4k_W1__correctly_specified` | positive | the observational law, n = 4,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9355 to 0.9527, SE ratio 0.9501 to 1.0000 | pass |
| `interval_calibration` | `obs_4k_W3__correctly_specified` | positive | the observational law, n = 4,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9614, SE ratio 0.9850 to 1.0382 | pass |
| `interval_calibration` | `obs_4k_intercept__correctly_specified` | positive | the observational law, n = 4,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9393 to 0.9560, SE ratio 0.9689 to 1.0210 | pass |
| `interval_calibration` | `obs_missing_4k_W1__correctly_specified` | positive | the observational law with outcomes missing at random, n = 4,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9440 to 0.9601, SE ratio 0.9761 to 1.0278 | pass |
| `interval_calibration` | `obs_missing_4k_W3__correctly_specified` | positive | the observational law with outcomes missing at random, n = 4,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9418 to 0.9581, SE ratio 0.9753 to 1.0293 | pass |
| `interval_calibration` | `obs_missing_4k_intercept__correctly_specified` | positive | the observational law with outcomes missing at random, n = 4,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9407 to 0.9572, SE ratio 0.9647 to 1.0170 | pass |
| `interval_calibration` | `obs_missing_crr_4k_W1__correctly_specified` | positive | the observational law with outcomes missing at random, on the log-ratio scale, n = 4,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9353 to 0.9526, SE ratio 0.9481 to 1.0004 | pass |
| `interval_calibration` | `obs_missing_crr_4k_W3__correctly_specified` | positive | the observational law with outcomes missing at random, on the log-ratio scale, n = 4,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9409 to 0.9574, SE ratio 0.9852 to 1.0379 | pass |
| `interval_calibration` | `obs_missing_crr_4k_intercept__correctly_specified` | positive | the observational law with outcomes missing at random, on the log-ratio scale, n = 4,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9411 to 0.9576, SE ratio 0.9776 to 1.0309 | pass |
| `interval_calibration` | `trial_2k_W1__correctly_specified` | positive | the randomized trial law with P(A = 1) = 0.5 known, n = 2,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9380 to 0.9549, SE ratio 0.9659 to 1.0186 | pass |
| `interval_calibration` | `trial_2k_W3__correctly_specified` | positive | the randomized trial law with P(A = 1) = 0.5 known, n = 2,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9393 to 0.9560, SE ratio 0.9730 to 1.0266 | pass |
| `interval_calibration` | `trial_2k_intercept__correctly_specified` | positive | the randomized trial law with P(A = 1) = 0.5 known, n = 2,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9427 to 0.9589, SE ratio 0.9777 to 1.0301 | pass |
| `interval_calibration` | `weighted_sl_2k_W1__correctly_specified` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3) and the ensemble learner (spline GLM, gradient boosting and lasso), five outer folds, n = 2,000: the W1 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9345 to 0.9543, SE ratio 0.9630 to 1.0239 | pass |
| `interval_calibration` | `weighted_sl_2k_W3__correctly_specified` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3) and the ensemble learner (spline GLM, gradient boosting and lasso), five outer folds, n = 2,000: the W3 slope of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9381 to 0.9574, SE ratio 0.9578 to 1.0170 | pass |
| `interval_calibration` | `weighted_sl_2k_intercept__correctly_specified` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3) and the ensemble learner (spline GLM, gradient boosting and lasso), five outer folds, n = 2,000: the intercept of the projection on (1, W1, W3): the declared fit with consistent nuisance learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9614, SE ratio 0.9808 to 1.0436 | pass |
| `power` | `heterogeneity__slope_0.10` | positive | the df-2 heterogeneity test of the W1 and W3 slopes of Linear(W1, W3) on a randomized law with the risk-scale CATE 0.2 + s W1: slope s = 0.10, n = 4,000 | the rejection rate's lower end at least 0.80 | rejection 0.9350, 0.9123 to 0.9535 | pass |
| `type_i_error` | `heterogeneity__sharp_null` | positive | the df-2 heterogeneity test of the W1 and W3 slopes of Linear(W1, W3) on a randomized law with the risk-scale CATE 0.2 + s W1: slope s = 0, so the effect does not vary, n = 2,000 | the 99% upper end of the rejection rate at most 0.10 | rejection 0.0433, 0.0333 to 0.0552 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. Each of the 33
calibration positives has an SE-ratio interval inside 0.93 to 1.07 and a coverage interval inside
0.92 to 0.98, on both scales, with three arms, weights and the ensemble, clusters, missing
outcomes and the assessment of an external score. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 3 of 3 | `performance-tests.csv` |
| gated property cells passed | 64 of 64 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

The ladder's standardized bias falls with n, from at most 0.160 at 1k to at most 0.094 at 16k,
and each rung's interval lies inside $\pm 0.25$. The row-level SE controls read SE ratios of
0.78 and 0.78, and the heterogeneity test rejects at 0.043 under the null and 0.935 at slope 0.10.

The support row was declared before the run. It needed every verdict to pass. They did, so the
registered rows of the projection fit cover the compositions these groups measured. Compositions
outside them keep their status, given under Limitations.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4800 | primary replications |
| `n` | 4000 | observations per replication |
| `independent_tests_total` | 3 | parameter tests against the truth |
| `independent_tests_passed` | 3 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 64 | gated repeated-sampling property cells |
| `property_cells_passed` | 64 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0221 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9523 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9438 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9635 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0314 | highest bootstrap primary SE-ratio endpoint |
| `properties[interval_calibration/obs_4k_W1__correctly_specified]:se_ratio_ci_lower` | 0.9501 | lowest calibration SE-ratio endpoint |
| `properties[interval_calibration/weighted_sl_2k_intercept__correctly_specified]:se_ratio_ci_upper` | 1.0436 | highest calibration SE-ratio endpoint |
| `properties[interval_calibration/clustered_4k_intercept__row_variance_control]:se_ratio` | 0.7756 | row-level SE control, intercept |
| `properties[interval_calibration/clustered_4k_W1__row_variance_control]:se_ratio` | 0.7836 | row-level SE control, $W_1$ |
| `properties[bias/slow_1k_intercept]:standardized_bias` | 0.1597 | largest ladder bias, 1k |
| `properties[bias/slow_16k_intercept]:standardized_bias` | 0.0940 | largest ladder bias, 16k |
| `properties[bias/ipw_1k_W1]:standardized_bias` | -0.6200 | IPW control, 1k |
| `properties[bias/t_learner_4k_W1]:standardized_bias` | 2.4272 | T-learner control |
| `properties[bias/weights_ignored_2k_intercept]:standardized_bias` | -0.4628 | dropped-weights control |
| `properties[type_i_error/heterogeneity__sharp_null]:rejection_rate` | 0.0433 | heterogeneity test size |
| `properties[power/heterogeneity__slope_0.10]:rejection_rate` | 0.9350 | heterogeneity test power |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
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
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:subject_failures_allowed` | 48 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a projection TMLE of the CATE on a declared basis. The row rests on accuracy against the truth and on the theory properties |
| The interval covers the projection | The interval covers the coefficients of the least-squares projection of the CATE on the basis, over the study's modifier distribution, and not the CATE itself where it is not in the span of the basis |
| One basis, two modifiers | Every cell uses `Linear(["W1", "W3"])` with continuous modifiers on $[-1, 1]$. The basis is not in the support key, so an `Indicators`, `Cosine`, `BSpline`, `Additive` or `Tensor` basis reads supported and rests on the theory alone. The simultaneous band of `effect_curve(grid=)` was not measured, and its note says so |
| Consistent learners, per arm | Every calibration group uses learners that are consistent for the law, with one outcome regression per arm (`outcome_by_arm=True`); a pooled outcome regression on this axis reads experimental. The ladder measures bias under slow nuisances, and two groups measure it under a declared wrong outcome regression; coverage under a misspecified treatment model was not measured |
| Laws and sizes | n runs from 1,000 to 16,000 on smooth laws with overlap of about 0.25 to 0.77. Weaker overlap, a rare outcome or another n reads supported and rests on the theory alone |
| The assessment of a score | `assess_score` was measured here for one external score on the difference scale, as the TMLE that replaced the one-step. Both estimate the same risk and calibration line, and at n = 2,000 the three cells read coverage and SE ratios within about 0.001 of the one-step run, so they cannot tell the two estimators apart. [`score-assessment`](score-assessment.md) measures the log-ratio scale and the paired comparison `compare=`, and lists which configurations read `supported`. The held-out curve checks of `curve.diagnostics` are descriptive and carry no interval |
| Compositions outside the row | The EP learner (`method="ep"`) is another estimator, measured by [`ep-cate`](ep-cate.md) and [`ep-crr`](ep-crr.md). `model=None` with the default method, `strata=`, an intermediate variable, missing treatment values, repeated out-of-fold fits and the bootstrap refuse before fitting |

## Reproduction

The declared run is `python -m tests.canonical.conditional_projection.regenerate`, whose docstring
gives the run form and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/conditional_projection/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/conditional_projection/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/conditional_projection/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/conditional_projection/properties.csv)
carry every published row.
