# Projection methods (DR-TMLE, per-arm C-TMLE)

This study validates two estimators of the difference-scale projection of a
`ConditionalContrast` on `Linear(["W1"])`, both fitted in sample: the DR-TMLE projection with
`guard=("Q", "g")`, the univariate reduction and cubic reduced regressions, and the per-arm
outcome-adaptive C-TMLE projection. The target is the coefficient vector of the least-squares
projection of the CATE on $(1, W_1)$. [Conditional effects](../conditional-effects.md#the-dr-tmle-projection)
gives the targets, the estimators and their conditions. The DR-TMLE projection reports its
interval only with a declared mechanism; with an estimated mechanism it reports the point under
`reduced_regression_plugin`. The C-TMLE projection reports points only, under
`generated_design_plugin`.

**No canonical implementation is compared.** R `drtmle` fits no projection, and no maintained
package fits a C-TMLE projection. A zero-row equivalence artifact records the absence of a
comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law | law P: $W_1, W_2$ independent standard normal, $g_0(1 \mid W) = \operatorname{expit}(0.6 W_1 + 0.3 W_2)$ and $\bar Q_0(a, W) = \operatorname{expit}(-0.2 + a(0.5 + 0.6 W_1) - W_1 + 2 W_2)$, binary $Y$; truth $\beta = (0.07302, 0.08141)$ |
| fit | in sample, `g_bounds=(0.01, 0.99)`; DR-TMLE with `guard=("Q", "g")`, `reduction="univariate"` and cubic polynomials in the scalar index as the reduced regressions; C-TMLE with `strategy="oat"`, `oat_design="per_arm"` and a logistic per-arm design |
| outcome regressions | wrong: a logistic GLM on $(A, W_1)$ that omits $W_2$ and the modification; correct: a logistic GLM on $(A, W_1, W_2, A W_1)$ |
| mechanisms | declared $g_0$ through `treatment_probabilities=`; the correct logistic GLM on $(W_1, W_2)$; that GLM plus the constructed logit offset $n^{-1/3} W_2 (1 + W_1)$, which breaks the level sets of $g_0$ |
| groups | `dr_known_4k` (declared mechanism, wrong outcome regression, R = 5,600), `dr_both_4k` (both correct, R = 2,400), `dr_glm_4k` (correct GLM mechanism, wrong outcome regression, R = 17,600), `dr_slow_4k`, `dr_slow_16k`, `dr_slow_64k` (the slow mechanism and the wrong outcome regression at n = 4,000, 16,000 and 64,000, R = 1,000 each) and `ctmle_points_4k` (correct outcome regression, R = 2,400); every other group at n = 4,000 |
| truth | Gauss-Hermite quadrature on 120 x 120 points, checked in the fast tier |
| primary budget | 2,400 replications of `dr_known_4k` at n = 4,000, the intercept, on its own stream |
| calibration budget | R = 5,600 for `dr_known_4k`, from X32a's rule over two one-row arms on a 3,200-draw pilot (SE ratios 1.0005 and 0.9795, `tests/studies/projection_methods_pilot.csv`) |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20261531` and `resampling_seed=20261532`; `dr_glm_4k` and the ladder draw on the `property_r3` stream, every other group on `property` |
| policy | `reporting`, declared before the second run so that a red redesigned cell would publish red with its claim withheld |

The law, groups, learners, budgets, controls and seeds were fixed before each run, in the P1 PR 3
plan, its revisions and reviews, and `tests/studies/projection_methods.py`.
`tests/unit/test_projection_methods_design.py` checks the truths, the predicted displacements, the
declared cells, the pilot rule and every budget. [The re-declared cells](#the-re-declared-cells)
gives the redesign between the two declared runs.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000 | `cate_projection[(intercept)]` | the intercept of the least-squares projection of the CATE on (1, W1) | `cleverly` DR-TMLE projection on the difference scale, in sample, with guards Q and g, the univariate reduction and cubic reduced regressions | -0.000964 to 0.000663 | 0.9429 | 0.9791 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | with the mechanism declared and a wrong outcome regression (R = 5,600), does each coefficient's SE-ratio interval lie inside 0.93 to 1.07 and its coverage interval inside 0.92 to 0.98; one coefficient per arm | the reported SE of the intercept times 0.70 must fall below the band |
| `bias` | is each coefficient's standardized-bias interval inside $\pm 0.25$, for the DR-TMLE projection with the mechanism declared and with both nuisances correct, and for the C-TMLE projection | the plug-in of the wrong outcome regression, with no targeting, must lie wholly outside $\pm 0.25$ on both coefficients |
| `one_wrong_inference` | at a correct logistic mechanism and a wrong outcome regression (R = 17,600), is the withheld plug-in interval conservative on the intercept (SE-ratio interval wholly above 1.07, coverage floor, bias inside the margin) and valid on the slope (coverage floor, bias inside the margin, SE ratio inside 0.80 to 1.20); under the slow mechanism, does each rung's bias lie wholly beyond $-0.25$ SD and grow from n = 4,000 to n = 64,000 | the ladder is itself the control: a rung or a growth interval that does not detect the first-order bias is red |

The `spread` rows publish readings with no verdict: the SE read off the projection's own curve
$D^*$ without the mechanism correction on `dr_known_4k`, and the withheld plug-in SE on
`dr_both_4k` and `ctmle_points_4k`.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bias` | `ctmle_points_4k_W1` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the per-arm outcome-adaptive C-TMLE projection with the correct logistic outcome regression and a logistic per-arm design, n = 4,000: the point estimate of the W1 slope of the projection of the CATE on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.000778 to 0.000513, margin 0.0031 | pass |
| `bias` | `ctmle_points_4k_intercept` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the per-arm outcome-adaptive C-TMLE projection with the correct logistic outcome regression and a logistic per-arm design, n = 4,000: the point estimate of the intercept of the projection of the CATE on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.000807 to 0.000501, margin 0.0031 | pass |
| `bias` | `dr_both_4k_W1` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the correct logistic outcome regression and the correct logistic mechanism, n = 4,000: the point estimate of the W1 slope of the projection of the CATE on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.000296 to 0.0012, margin 0.0035 | pass |
| `bias` | `dr_both_4k_intercept` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the correct logistic outcome regression and the correct logistic mechanism, n = 4,000: the point estimate of the intercept of the projection of the CATE on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.000727 to 0.000604, margin 0.0032 | pass |
| `bias` | `dr_known_4k_W1` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the point estimate of the W1 slope of the projection of the CATE on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.0020 to -0.000894, margin 0.0041 | pass |
| `bias` | `dr_known_4k_intercept` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the point estimate of the intercept of the projection of the CATE on (1, W1) | the 99% interval of the standardized bias inside +-0.25 | bias -0.000293 to 0.000759, margin 0.0038 | pass |
| `bias` | `wrong_q_witness_4k_W1` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the plug-in of the wrong outcome regression, with no targeting, for the W1 slope of the projection of the CATE on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0787 to -0.0787, margin 0.000289 | pass |
| `bias` | `wrong_q_witness_4k_intercept` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the plug-in of the wrong outcome regression, with no targeting, for the intercept of the projection of the CATE on (1, W1) | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0834 to 0.0844, margin 0.0039 | pass |
| `interval_calibration` | `dr_known_4k_W1__dr_known_slope` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the influence-curve Wald interval of the W1 slope of the projection of the CATE on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9279 to 0.9449, SE ratio 0.9493 to 0.9974 | pass |
| `interval_calibration` | `dr_known_4k_intercept__dr_known_intercept` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the influence-curve Wald interval of the intercept of the projection of the CATE on (1, W1) | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9414 to 0.9567, SE ratio 0.9673 to 1.0177 | pass |
| `interval_calibration` | `dr_known_4k_intercept__shrunken_se_control` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the reported SE of the intercept times 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8123 to 0.8386, SE ratio 0.6771 to 0.7128 | pass |
| `one_wrong_inference` | `dr_glm__W1__valid` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism, n = 4,000: the plug-in Wald interval of the W1 slope of the projection of the CATE on (1, W1), which the fit's status withholds | coverage clears the floor, bias inside the margin, SE ratio inside the sanity band | bias -0.0019 to -0.0012, coverage 0.9361 to 0.9453, SE ratio 0.9794, 0.9662 to 0.9932 | pass |
| `one_wrong_inference` | `dr_glm__intercept__conservative` | positive | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism, n = 4,000: the plug-in Wald interval of the intercept of the projection of the CATE on (1, W1), which the fit's status withholds | SE-ratio interval wholly above the calibration band, coverage clears the floor, bias inside the margin | bias -0.000160 to 0.000362, coverage 0.9696 to 0.9760, SE ratio 1.1277, 1.1125 to 1.1438 | pass |
| `one_wrong_inference` | `dr_slow__W1__growth` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism plus the constructed logit offset n^(-1/3) W2 (1 + W1): the W1 slope of the projection of the CATE on (1, W1), across the ladder | the lower bound of the 99% interval of the signed growth of bias/SD from n = 4,000 to n = 64,000 lies above zero | growth 0.4066, 0.2781 to 0.5367 | pass |
| `one_wrong_inference` | `dr_slow__intercept__growth` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism plus the constructed logit offset n^(-1/3) W2 (1 + W1): the intercept of the projection of the CATE on (1, W1), across the ladder | the lower bound of the 99% interval of the signed growth of bias/SD from n = 4,000 to n = 64,000 lies above zero | growth 0.7740, 0.6375 to 0.9238 | pass |
| `one_wrong_inference` | `dr_slow_n16000__W1__first_order` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism plus the constructed logit offset n^(-1/3) W2 (1 + W1): the W1 slope of the projection of the CATE on (1, W1), at n = 16,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0066 to -0.0053, margin 0.0020, SE ratio 0.9837 | pass |
| `one_wrong_inference` | `dr_slow_n16000__intercept__first_order` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism plus the constructed logit offset n^(-1/3) W2 (1 + W1): the intercept of the projection of the CATE on (1, W1), at n = 16,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0074 to -0.0063, margin 0.0017, SE ratio 1.0979 | pass |
| `one_wrong_inference` | `dr_slow_n4000__W1__first_order` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism plus the constructed logit offset n^(-1/3) W2 (1 + W1): the W1 slope of the projection of the CATE on (1, W1), at n = 4,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0104 to -0.0079, margin 0.0038, SE ratio 1.0358 | pass |
| `one_wrong_inference` | `dr_slow_n4000__intercept__first_order` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism plus the constructed logit offset n^(-1/3) W2 (1 + W1): the intercept of the projection of the CATE on (1, W1), at n = 4,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0090 to -0.0068, margin 0.0035, SE ratio 1.0618 | pass |
| `one_wrong_inference` | `dr_slow_n64000__W1__first_order` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism plus the constructed logit offset n^(-1/3) W2 (1 + W1): the W1 slope of the projection of the CATE on (1, W1), at n = 64,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0043 to -0.0036, margin 0.000982, SE ratio 1.0280 | pass |
| `one_wrong_inference` | `dr_slow_n64000__intercept__first_order` | control | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the correct logistic mechanism plus the constructed logit offset n^(-1/3) W2 (1 + W1): the intercept of the projection of the CATE on (1, W1), at n = 64,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0048 to -0.0042, margin 0.000839, SE ratio 1.1235 | pass |
| `spread` | `ctmle_points_4k_W1__plugin_reading` | diagnostic | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the per-arm outcome-adaptive C-TMLE projection with the correct logistic outcome regression and a logistic per-arm design, n = 4,000: the plug-in SE beside the spread, withheld by the fit's status, for the W1 slope of the projection of the CATE on (1, W1) | none. The row is a reading and states no verdict | SE ratio 0.9610, coverage 0.9278 to 0.9529 | reported |
| `spread` | `ctmle_points_4k_intercept__plugin_reading` | diagnostic | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the per-arm outcome-adaptive C-TMLE projection with the correct logistic outcome regression and a logistic per-arm design, n = 4,000: the plug-in SE beside the spread, withheld by the fit's status, for the intercept of the projection of the CATE on (1, W1) | none. The row is a reading and states no verdict | SE ratio 0.9732, coverage 0.9324 to 0.9567 | reported |
| `spread` | `dr_both_4k_W1__plugin_reading` | diagnostic | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the correct logistic outcome regression and the correct logistic mechanism, n = 4,000: the plug-in SE beside the spread, withheld by the fit's status, for the W1 slope of the projection of the CATE on (1, W1) | none. The row is a reading and states no verdict | SE ratio 0.9792, coverage 0.9301 to 0.9548 | reported |
| `spread` | `dr_both_4k_intercept__plugin_reading` | diagnostic | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the correct logistic outcome regression and the correct logistic mechanism, n = 4,000: the plug-in SE beside the spread, withheld by the fit's status, for the intercept of the projection of the CATE on (1, W1) | none. The row is a reading and states no verdict | SE ratio 1.0139, coverage 0.9476 to 0.9689 | reported |
| `spread` | `dr_known_4k_W1__d_star_reading` | diagnostic | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the SE read off the projection's own curve D* without the mechanism correction, beside the spread, for the W1 slope of the projection of the CATE on (1, W1) | none. The row is a reading and states no verdict | SE ratio 1.0081, coverage 0.9405 to 0.9559 | reported |
| `spread` | `dr_known_4k_intercept__d_star_reading` | diagnostic | law P (W1, W2 independent standard normal, P(A = 1) = expit(0.6 W1 + 0.3 W2) given W, and E[Y] = expit(-0.2 + A (0.5 + 0.6 W1) - W1 + 2 W2) given A and W), the DR-TMLE projection with guards Q and g and cubic reduced regressions, the wrong outcome regression (a logistic GLM on A and W1) and the declared mechanism (treatment_probabilities=), n = 4,000: the SE read off the projection's own curve D* without the mechanism correction, beside the spread, for the intercept of the projection of the CATE on (1, W1) | none. The row is a reading and states no verdict | SE ratio 1.0334, coverage 0.9506 to 0.9646 | reported |
<!-- /generated -->

## Result

Every primary test and every gated property cell of the second run passed, and every control
failed as designed. No fit failed. Its unchanged groups and the primary reproduced the first run
bit for bit, so the first run's verdicts on them stand as well.

| reading | value | source |
| --- | --- | --- |
| calibration SE ratio, intercept, declared mechanism | 0.9918 | `properties.csv` |
| calibration SE ratio, slope, declared mechanism | 0.9729 | `properties.csv` |
| SE ratio of the withheld interval, intercept, correct GLM mechanism | 1.1277 | `properties.csv` |
| SE ratio of the withheld interval, slope, correct GLM mechanism | 0.9794 | `properties.csv` |
| primary coverage | 0.9429 | `performance-tests.csv` |
| standardized bias of the slow ladder, intercept, n = 4,000 to 64,000 | -0.5659 to -1.3399 | `properties.csv` |
| standardized bias of the slow ladder, slope, n = 4,000 to 64,000 | -0.5951 to -1.0016 | `properties.csv` |

### The interval with a declared mechanism

With the mechanism declared and a wrong outcome regression, the intercept reads an SE ratio of
0.9918 (99% interval 0.9673 to 1.0177) and coverage 0.9495, and the slope 0.9729 (0.9493 to
0.9974) and coverage 0.9368. Both lie inside the calibration bands. The slope's SE-ratio interval
lies wholly below 1, so its SE is about 3% small at n = 4,000, and its point carries a bias of
$-0.0015$, or $-0.089$ SD, whose interval excludes zero and lies inside the margin. The same
slope bias, $-0.095$ SD, appears at the correct GLM mechanism with the same outcome regression,
so it belongs to the wrong regression at this n rather than to the declared mechanism. The
primary test on the intercept, on its own stream, reads coverage 0.9429 and an SE ratio of
0.9791.

The `d_star` reading divides the SE of $D^*$ alone by the spread: 1.0334 on the intercept and
1.0081 on the slope. Both lie inside the calibration band, so the study does not see whether the
fit subtracts the mechanism correction $D_A$. The exact-law curve test in
`tests/unit/test_drtmle_projection.py` checks that term instead.

### The withheld interval at a correct parametric mechanism

With the correct logistic mechanism and a wrong outcome regression the fit withholds its
interval. The study still measured the plug-in interval that the status keeps as a diagnostic.
On the intercept it is conservative, as the theory of the one-wrong case predicts: SE ratio
1.1277 (99% interval 1.1125 to 1.1438), coverage 0.9729. The planning ratio was 1.105, and two
fits of the declared guard at n = 200,000 read 1.133 and 1.108 ([the re-declared
cells](#the-re-declared-cells)). On the slope it is valid: SE ratio
0.9794, coverage 0.9409. The interval stays withheld, because only a declared mechanism can be
checked from a fit ([Conditional effects](../conditional-effects.md#the-dr-tmle-projection)).

### The slow mechanism

Under the constructed slow mechanism the bias is negative on both coefficients at every rung and
grows with n, as the conditioning term predicts. The intercept reads $-0.5659$, $-1.0000$ and
$-1.3399$ SD at n = 4,000, 16,000 and 64,000, and the slope $-0.5951$, $-0.7288$ and
$-1.0016$ SD. The growth from 4,000 to 64,000 rows is 0.7740 SD on the intercept (99% interval
0.6375 to 0.9238) and 0.4066 SD on the slope (0.2781 to 0.5367). Coverage falls with n, to
0.811 and 0.849 at 64,000 rows. A level-set-breaking mechanism error therefore biases the
DR-TMLE projection at the rate the theory gives, and no interval guards against it.

### The points without an interval

With both nuisances correct the DR-TMLE projection's standardized biases are $-0.0049$ and
0.0312. The per-arm C-TMLE projection with a correct outcome regression reads $-0.0123$ and
$-0.0108$. The withheld plug-in SEs read SE ratios of 1.0139 and 0.9792 for the DR-TMLE
projection and 0.9732 and 0.9610 for the C-TMLE projection. These readings decide nothing, since
the status of each fit withholds its interval.

### What the controls show

The shrunken-SE control reads an SE ratio of 0.6943, its interval ending at 0.7128, below 0.93.
The plug-in of the wrong outcome regression, with no targeting, lands 0.0839 above the
intercept's truth and 0.0787 below the slope's, at 5.3237 SD on the intercept. Its slope barely
moves between draws, so its standardized bias reads $-67.9677$. Targeting therefore does all the
work on both coefficients.

The readings do not show the items in this list.

- An estimated mechanism with an interval. The fit withholds it, and no row licenses it.
- Another guard, cross-fitting, repeats, the bootstrap, weights, clusters, three arms or the log
  ratio with either method.
- A treatment model alone correct for the C-TMLE points. The point needs a consistent outcome
  regression.

## The re-declared cells

The first run was declared at commit `32f9d8ac` and ran at commit `c566a4c`. Two gated cells were
red. A diagnosis found no defect in either estimator: each cell measured something other than its
claim. The study re-declared both groups on fresh seed streams before any new draw, and moved no
margin, band, rule, law, offset or threshold. The second run replaced every artifact of the first.
Its unchanged groups and the primary reproduced the first run's rows and verdicts exactly, which
`tests/canonical/projection_methods/first_run.py` checked during the run.
`tests/studies/projection_methods_limit.csv` holds every figure below, copied from the first run's
artifacts or computed by the declared-stream probes of `tests/studies/projection_methods_limit.py`.

### The conservative interval at a correct parametric mechanism

The `dr_glm_4k` group fits the DR-TMLE projection with the correct logistic mechanism and a wrong
outcome regression at n = 4,000. The interval of its intercept is predicted to be conservative,
and the cell requires the 99% interval of the SE ratio to lie wholly above the calibration band's
upper end of 1.07.

| first run, 2,400 replicates | value |
| --- | --- |
| SE ratio (99% interval) | 1.1076 (1.0661 to 1.1514) |
| coverage (99% interval) | 0.9675 (0.957 to 0.976) |
| bias | 0.00075 |
| verdict | **fail** |

The interval is wider than the spread, and its coverage is above 0.95. The ratio's interval
reaches below 1.07, so the cell did not show the excess it is named for. The cell had been sized
from the quadrature of an estimator that solves the reduced equations (8) and (9) only, whose SE
ratio is 1.189. The declared estimator also solves equation (10), whose fluctuation moves the
outcome regression's limit by an amount that does not shrink with n. Its ratio is lower, and it
varies from fit to fit even at n = 200,000.

| fit | SE ratio of the intercept |
| --- | --- |
| quadrature of the limit, equations (8) and (9) | 1.189 |
| n = 200,000, `guard=("Q",)`, replicates 0 and 1 | 1.190 and 1.187 |
| n = 200,000, declared `guard=("Q", "g")`, replicates 0 and 1 | 1.133 and 1.108 |
| first run and the 1,000-replicate declared-stream probe at n = 4,000, pooled | 1.108 |

The redesigned cell read the pooled finite-sample ratio, rounded down to 1.105, as its planning
value. At that value and a spread constant of 0.74, the declared sizing rule gave R = 17,600
replicates on a fresh stream. The second run's cell passed at an SE ratio of 1.1277, so the claim
that the withheld interval is conservative at a correct parametric mechanism rests on it. The
first run's red row stays published here.

### The first-order ladder at 1,000 rows

The `dr_slow` ladder fits the DR-TMLE projection under a mechanism with a constructed logit error
$n^{-1/3} W_2 (1 + W_1)$. Each rung must show a negative bias outside the margin of 0.25 SD. In the
first run the 1,000-row rung read a standardized bias of -0.26 on the intercept and -0.44 on the
slope, and the intercept stayed inside the margin.

| fit at n = 1,000 | intercept (SD) | slope (SD) |
| --- | --- | --- |
| first run, declared offset, 1,000 replicates | -0.26 | -0.44 |
| probe, 400 replicates each: no offset | +0.12 | -0.06 |
| declared offset | -0.19 | -0.31 |
| twice the declared offset | -0.22 | -0.53 |
| three times the declared offset | -0.16 | -0.62 |

Without the offset, the intercept carries a positive finite-sample bias of +0.0036 (SE 0.0015), which works
against the negative first-order term. Larger offsets move the intercept little. At n = 1,000 on
this law, the first-order bias of the intercept under a slow mechanism is therefore masked by an
opposing finite-sample bias, and the rung measured that sum rather than the first-order term. The
ladder was re-declared at 4,000, 16,000 and 64,000 rows, with the factor-4 spacing and factor-16
span of the first ladder, and every rung and both growth cells passed in the second run. The
limitation stands: at n = 1,000 on law P the slow mechanism's bias on the intercept is not
visible.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 4000 | observations per primary replication |
| `independent_tests_total` | 1 | primary tests against the known truth |
| `independent_tests_passed` | 1 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 21 | gated repeated-sampling property cells |
| `property_cells_passed` | 21 | cells whose own and family verdicts pass |
| `property_cells_reported` | 6 | readings, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0097 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9429 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9296 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9421 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0186 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-projection-methods/dr_known_4k/cate_projection[(intercept)]]:coverage` | 0.9429 | primary coverage of the intercept |
| `performance[cleverly-projection-methods/dr_known_4k/cate_projection[(intercept)]]:se_ratio` | 0.9791 | primary SE ratio of the intercept |
| `properties[interval_calibration/dr_known_4k_intercept__dr_known_intercept]:se_ratio` | 0.9918 | calibration SE ratio, intercept |
| `properties[interval_calibration/dr_known_4k_intercept__dr_known_intercept]:se_ratio_ci_lower` | 0.9673 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/dr_known_4k_intercept__dr_known_intercept]:se_ratio_ci_upper` | 1.0177 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/dr_known_4k_intercept__dr_known_intercept]:coverage` | 0.9495 | its coverage |
| `properties[interval_calibration/dr_known_4k_W1__dr_known_slope]:se_ratio` | 0.9729 | calibration SE ratio, slope |
| `properties[interval_calibration/dr_known_4k_W1__dr_known_slope]:se_ratio_ci_lower` | 0.9493 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/dr_known_4k_W1__dr_known_slope]:se_ratio_ci_upper` | 0.9974 | its upper endpoint |
| `properties[interval_calibration/dr_known_4k_W1__dr_known_slope]:coverage` | 0.9368 | its coverage |
| `properties[interval_calibration/dr_known_4k_W1__dr_known_slope]:coverage_ci_lower` | 0.9279 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/dr_known_4k_intercept__shrunken_se_control]:se_ratio` | 0.6943 | shrunken-SE control SE ratio |
| `properties[interval_calibration/dr_known_4k_intercept__shrunken_se_control]:se_ratio_ci_upper` | 0.7128 | its upper endpoint, against 0.93 |
| `properties[bias/dr_known_4k_intercept]:standardized_bias` | 0.0152 | bias of the intercept in SDs, declared mechanism |
| `properties[bias/dr_known_4k_W1]:bias` | -0.0015 | bias of the slope, declared mechanism |
| `properties[bias/dr_known_4k_W1]:standardized_bias` | -0.0887 | the same bias in SDs, against 0.25 |
| `properties[bias/dr_both_4k_intercept]:standardized_bias` | -0.0049 | bias of the intercept in SDs, both nuisances correct |
| `properties[bias/dr_both_4k_W1]:standardized_bias` | 0.0312 | bias of the slope in SDs, both nuisances correct |
| `properties[bias/ctmle_points_4k_intercept]:standardized_bias` | -0.0123 | bias of the intercept in SDs, C-TMLE points |
| `properties[bias/ctmle_points_4k_W1]:standardized_bias` | -0.0108 | bias of the slope in SDs, C-TMLE points |
| `properties[bias/wrong_q_witness_4k_intercept]:bias` | 0.0839 | plug-in of the wrong regression, intercept bias |
| `properties[bias/wrong_q_witness_4k_intercept]:standardized_bias` | 5.3237 | the same bias in SDs |
| `properties[bias/wrong_q_witness_4k_W1]:bias` | -0.0787 | plug-in of the wrong regression, slope bias |
| `properties[bias/wrong_q_witness_4k_W1]:standardized_bias` | -67.9677 | the same bias in its own SDs |
| `properties[one_wrong_inference/dr_glm__intercept__conservative]:se_ratio` | 1.1277 | withheld-interval SE ratio, intercept, correct GLM mechanism |
| `properties[one_wrong_inference/dr_glm__intercept__conservative]:se_ratio_ci_lower` | 1.1125 | its lower endpoint, against 1.07 |
| `properties[one_wrong_inference/dr_glm__intercept__conservative]:se_ratio_ci_upper` | 1.1438 | its upper endpoint |
| `properties[one_wrong_inference/dr_glm__intercept__conservative]:coverage` | 0.9729 | its coverage |
| `properties[one_wrong_inference/dr_glm__W1__valid]:se_ratio` | 0.9794 | withheld-interval SE ratio, slope, correct GLM mechanism |
| `properties[one_wrong_inference/dr_glm__W1__valid]:coverage` | 0.9409 | its coverage |
| `properties[one_wrong_inference/dr_glm__W1__valid]:standardized_bias` | -0.0954 | its bias in SDs |
| `properties[one_wrong_inference/dr_slow_n4000__intercept__first_order]:standardized_bias` | -0.5659 | slow ladder, intercept, n = 4,000 |
| `properties[one_wrong_inference/dr_slow_n16000__intercept__first_order]:standardized_bias` | -1.0000 | slow ladder, intercept, n = 16,000 |
| `properties[one_wrong_inference/dr_slow_n64000__intercept__first_order]:standardized_bias` | -1.3399 | slow ladder, intercept, n = 64,000 |
| `properties[one_wrong_inference/dr_slow_n4000__W1__first_order]:standardized_bias` | -0.5951 | slow ladder, slope, n = 4,000 |
| `properties[one_wrong_inference/dr_slow_n16000__W1__first_order]:standardized_bias` | -0.7288 | slow ladder, slope, n = 16,000 |
| `properties[one_wrong_inference/dr_slow_n64000__W1__first_order]:standardized_bias` | -1.0016 | slow ladder, slope, n = 64,000 |
| `properties[one_wrong_inference/dr_slow_n64000__intercept__first_order]:coverage` | 0.8110 | coverage at n = 64,000, intercept |
| `properties[one_wrong_inference/dr_slow_n64000__W1__first_order]:coverage` | 0.8490 | coverage at n = 64,000, slope |
| `properties[one_wrong_inference/dr_slow__intercept__growth]:growth` | 0.7740 | growth of the intercept's bias, n = 4,000 to 64,000 |
| `properties[one_wrong_inference/dr_slow__intercept__growth]:growth_ci_lower` | 0.6375 | its lower endpoint, against 0 |
| `properties[one_wrong_inference/dr_slow__intercept__growth]:growth_ci_upper` | 0.9238 | its upper endpoint |
| `properties[one_wrong_inference/dr_slow__W1__growth]:growth` | 0.4066 | growth of the slope's bias, n = 4,000 to 64,000 |
| `properties[one_wrong_inference/dr_slow__W1__growth]:growth_ci_lower` | 0.2781 | its lower endpoint, against 0 |
| `properties[one_wrong_inference/dr_slow__W1__growth]:growth_ci_upper` | 0.5367 | its upper endpoint |
| `properties[spread/dr_known_4k_intercept__d_star_reading]:se_ratio` | 1.0334 | SE of D* alone over the spread, intercept |
| `properties[spread/dr_known_4k_W1__d_star_reading]:se_ratio` | 1.0081 | SE of D* alone over the spread, slope |
| `properties[spread/dr_both_4k_intercept__plugin_reading]:se_ratio` | 1.0139 | withheld plug-in SE ratio, intercept, both correct |
| `properties[spread/dr_both_4k_W1__plugin_reading]:se_ratio` | 0.9792 | withheld plug-in SE ratio, slope, both correct |
| `properties[spread/ctmle_points_4k_intercept__plugin_reading]:se_ratio` | 0.9732 | withheld plug-in SE ratio, intercept, C-TMLE |
| `properties[spread/ctmle_points_4k_W1__plugin_reading]:se_ratio` | 0.9610 | withheld plug-in SE ratio, slope, C-TMLE |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control applies to each SE |
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
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 24 | failed primary fits that bound allows per setting |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | R `drtmle` fits no projection and no maintained package fits a C-TMLE projection. The rows rest on accuracy against the known truth and on the theory properties |
| The interval is licensed only with a declared mechanism | With an estimated mechanism the DR-TMLE projection withholds its interval, and row `T49.drtmle` covers the point only. The study measured the withheld interval as conservative on the intercept and valid on the slope at a correct logistic mechanism, which licenses nothing |
| A mechanism error that breaks the level sets of $g_0$ biases the point | The ladder measured a bias that grows with n, to more than one SD at 64,000 rows on both coefficients, and coverage falls with it. At 1,000 rows on law P the intercept's bias is masked by an opposing finite-sample bias ([the re-declared cells](#the-re-declared-cells)) |
| The C-TMLE projection reports points only | Its interval is withheld, and its point needs a consistent outcome regression; a correct treatment model alone does not make it consistent |
| One law, one basis, in sample | The study fits law P, `Linear(["W1"])`, binary outcomes and a binary treatment in sample. The key holds neither the basis nor the law nor n, so another basis, law or n reads supported and rests on the theory alone |
| The slope's SE reads about 3% small at n = 4,000 | With the mechanism declared, the slope's SE-ratio interval lies wholly below 1 and its point carries a bias of about 0.09 SD under the wrong outcome regression, both inside the declared margins |
| No witness of the mechanism correction | The `d_star` reading lies inside the calibration band on both coefficients, so the study cannot see whether the fit subtracts $D_A$. The exact-law curve test in the fast tier checks it |
| Compositions outside the rows | Another guard, cross-fitting, repeats, the bootstrap, weights, clusters, three arms and the log ratio read experimental with either method. The log ratio, the bivariate reduction, `strata=`, missing data, an intermediate, `targeting="one_step"` and `evaluation=` refuse for DR-TMLE |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_methods/manifest.json)
records the seeds, the margins, the estimator configuration, both declarations, the source hashes
and the result hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_methods/run.log)
records the commit, the clean tree, the runtime and the first-run check. The red-cell readings,
declared before each run, are in the docstring of
[`regenerate.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_methods/regenerate.py),
whose command `python -m tests.canonical.projection_methods.regenerate` reruns the study.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_methods/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/projection_methods/properties.csv)
carry every published row.
