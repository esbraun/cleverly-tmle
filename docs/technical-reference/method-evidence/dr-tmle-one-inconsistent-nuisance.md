# DR-TMLE with one inconsistent nuisance

This study measures complete-data and randomized missing-outcome `DRTMLE` when one nuisance
converges to a wrong limit. The [case table](../dr-tmle/theorem.md#the-conditioning-terms-are-linear-in-one-error)
states three cases. The interval is valid for a declared mechanism, conservative for a correctly
specified parametric mechanism, and not established when the consistent nuisance converges slower
than root-n. This study puts each case on a law where the conditioning terms $R_{3,n}$ and
$\tilde R_{5,n}$ do not vanish.

The study has no external comparator. It uses the `reporting` policy.

## What was compared

| setting | value |
| --- | --- |
| law L | $W_1, W_2 \sim N(0, 1)$, $g_0 = \operatorname{expit}(0.6 W_1 + 0.3 W_2)$, $\bar Q_0(a, W) = \operatorname{expit}(-0.2 + 0.5 a - W_1 + 2 W_2)$ |
| law M | the covariates and outcome of law L. $A \sim \operatorname{Bernoulli}(0.5)$ is declared. $Y$ is recorded with probability $\pi_0(a, W) = \operatorname{expit}(0.4 + 0.2 a + 0.6 W_1 + 0.3 W_2)$ |
| truths | $\psi(1) = 0.5422909862$, $\psi(0) = 0.4717756171$, ATE $0.0705153691$, by 120-node Gauss-Hermite quadrature on both laws |
| fit | `DRTMLE(guard=("Q", "g"), reduction="univariate", g_bounds=(0.01, 0.99))`, in sample except the cross-fitted cells |
| reduced regressions | cubic polynomials in the scalar design: least squares for the outcome reductions, Newton logistic with ridge `1e-8` for the mechanism reduction |
| intervals | pointwise 95% Wald from the influence curve |

| setting | outcome regression | mechanism | case |
| --- | --- | --- | --- |
| `q_wrong_g_known` | logistic GLM on $(A, W_1)$, wrong | the declared $g_0$ | valid (H2) |
| `q_wrong_g_glm` | the same | logistic GLM on $(W_1, W_2)$, correct | conservative |
| `q_wrong_g_glm_cf` | the same, cross-fitted over the default folds | the same, cross-fitted | conservative |
| `q_wrong_g_slow` | the same | the correct GLM plus the logit offset $n^{-1/3} W_2$ | not established |
| `q_slow_g_wrong` | the correct GLM plus the logit offset $3 n^{-1/3}(2 W_1 + W_2)/\sqrt 5$ | logistic GLM on $W_2$, wrong | not established (mirror) |
| `mar_q_wrong_pi_slow` | logistic GLM on $(A, W_1)$, wrong | declared $A$. The observation mechanism is the correct GLM plus the logit offset $2 n^{-1/3} W_2$ | not established |

The offsets are constructed errors, not fitted learners. Each one converges at the rate
$n^{-1/3}$, so each conditioning term is of order $n^{-1/3}$. Its root-n scaled contribution grows
by $16^{1/6} = 1.59$ from $n = 1{,}000$ to $n = 16{,}000$. The design test computes each term's
sign by quadrature (`tests/unit/test_drtmle_one_wrong_design.py`).

`q_wrong_g_glm` and `q_wrong_g_known` read the same draws. The primary scenario is
`q_wrong_g_known`, so the accuracy table reads the valid case only.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law L, declared mechanism, outcome regression on A and W1 (wrong), n = 4,000 | `ate` | average treatment effect | `cleverly` DR-TMLE with guards Q and g and the univariate reduction | -0.000676 to 0.000537 | 0.9543 | 1.0229 | pass |
| law L, declared mechanism, outcome regression on A and W1 (wrong), n = 4,000 | `ey0` | counterfactual mean under no treatment | `cleverly` DR-TMLE with guards Q and g and the univariate reduction | -0.000180 to 0.000730 | 0.9513 | 1.0089 | pass |
| law L, declared mechanism, outcome regression on A and W1 (wrong), n = 4,000 | `ey1` | counterfactual mean under treatment | `cleverly` DR-TMLE with guards Q and g and the univariate reduction | -0.000244 to 0.000655 | 0.9505 | 1.0195 | pass |
<!-- /generated -->

The accuracy table reads the primary scenario, `q_wrong_g_known`, the valid case.
No canonical implementation is compared, so the page has no agreement table.

## Rules

| rule | kind | passes when |
| --- | --- | --- |
| C | `conservative` | the 99% bootstrap interval of the SE ratio lies wholly above 1.07, the coverage lower bound clears 0.90, and the bias interval lies inside 0.25 empirical SD |
| V | `valid` | the coverage lower bound clears 0.90, the bias interval lies inside the margin, and the SE ratio lies inside 0.8 to 1.2 |
| F | `first_order` | the bias interval lies wholly beyond the margin on the side of the declared sign, and the SE ratio lies inside 0.1 to 10 |
| G | `growth` | the 99% bootstrap interval of $s \cdot (\text{bias}/\text{SD})(16{,}000) - s \cdot (\text{bias}/\text{SD})(1{,}000)$ has a lower bound above 0, with $s$ the declared sign |

Rule C asserts a standard error that is too large. A projected variance would be a
result-determining change that re-declares the cell. A red C cell is finite-sample only when its
SE-ratio interval contains the planning ratio (1.1251 in sample, 1.138 cross-fitted), its coverage
interval contains the predicted coverage (0.972, 0.974), and the diagnosis finds no defect.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `one_wrong_inference` | `mar_q_wrong_pi_slow__ey0__first_order` | control | arm-0 mean: law M, randomized missing outcome with the declared treatment probability 1/2, outcome regression on A and W1 (wrong), response mechanism a correct GLM plus the constructed logit offset 2 n^(-1/3) W2 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0082 to -0.0060, margin 0.0033, SE ratio 1.0185 | pass |
| `one_wrong_inference` | `mar_q_wrong_pi_slow__ey1__first_order` | control | arm-1 mean: law M, randomized missing outcome with the declared treatment probability 1/2, outcome regression on A and W1 (wrong), response mechanism a correct GLM plus the constructed logit offset 2 n^(-1/3) W2 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0081 to -0.0060, margin 0.0033, SE ratio 1.0065 | pass |
| `one_wrong_inference` | `q_slow_g_wrong__ate__growth` | control | ATE: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong) | the lower bound of the 99% interval of the signed growth of bias/SD from n = 1,000 to n = 16,000 lies above zero | growth 0.4687, 0.3263 to 0.6210 | pass |
| `one_wrong_inference` | `q_slow_g_wrong_n1000__ate__first_order` | control | ATE: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 1,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0245 to -0.0203, margin 0.0064, SE ratio 1.2736 | pass |
| `one_wrong_inference` | `q_slow_g_wrong_n16000__ate__first_order` | control | ATE: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 16,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0091 to -0.0080, margin 0.0016, SE ratio 0.9387 | pass |
| `one_wrong_inference` | `q_slow_g_wrong_n4000__ate__first_order` | control | ATE: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 4,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0145 to -0.0125, margin 0.0032, SE ratio 0.9449 | pass |
| `one_wrong_inference` | `q_wrong_g_glm__ate__conservative` | positive | ATE: law L, outcome regression on A and W1 (wrong), mechanism a correct logistic GLM, in sample | SE-ratio interval wholly above the calibration band, coverage clears the floor, bias inside the margin | bias -0.000966 to 0.000104, coverage 0.9679 to 0.9809, SE ratio 1.1611, 1.1285 to 1.1940 | pass |
| `one_wrong_inference` | `q_wrong_g_glm__ey0__valid` | positive | arm-0 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct logistic GLM, in sample | coverage clears the floor, bias inside the margin, SE ratio inside the sanity band | bias -0.000176 to 0.000687, coverage 0.9579 to 0.9729, SE ratio 1.0646, 1.0361 to 1.0953 | pass |
| `one_wrong_inference` | `q_wrong_g_glm__ey1__valid` | positive | arm-1 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct logistic GLM, in sample | coverage clears the floor, bias inside the margin, SE ratio inside the sanity band | bias -0.000613 to 0.000262, coverage 0.9500 to 0.9664, SE ratio 1.0468, 1.0183 to 1.0771 | pass |
| `one_wrong_inference` | `q_wrong_g_glm_cf__ate__conservative` | positive | ATE: law L, outcome regression on A and W1 (wrong), mechanism a correct logistic GLM, cross-fitted over the default folds | SE-ratio interval wholly above the calibration band, coverage clears the floor, bias inside the margin | bias -0.000472 to 0.000748, coverage 0.9748 to 0.9877, SE ratio 1.1908, 1.1537 to 1.2304 | pass |
| `one_wrong_inference` | `q_wrong_g_glm_cf__ey0__valid` | positive | arm-0 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct logistic GLM, cross-fitted over the default folds | coverage clears the floor, bias inside the margin, SE ratio inside the sanity band | bias -0.000349 to 0.000645, coverage 0.9565 to 0.9739, SE ratio 1.0762, 1.0412 to 1.1130 | pass |
| `one_wrong_inference` | `q_wrong_g_glm_cf__ey1__valid` | positive | arm-1 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct logistic GLM, cross-fitted over the default folds | coverage clears the floor, bias inside the margin, SE ratio inside the sanity band | bias -0.000217 to 0.000790, coverage 0.9551 to 0.9728, SE ratio 1.0600, 1.0264 to 1.0971 | pass |
| `one_wrong_inference` | `q_wrong_g_known__ate__valid` | positive | ATE: law L, outcome regression on A and W1 (wrong), declared mechanism, the draws of q_wrong_g_glm | coverage clears the floor, bias inside the margin, SE ratio inside the sanity band | bias -0.0011 to 0.000112, coverage 0.9404 to 0.9585, SE ratio 1.0085, 0.9808 to 1.0404 | pass |
| `one_wrong_inference` | `q_wrong_g_known__ey0__valid` | positive | arm-0 mean: law L, outcome regression on A and W1 (wrong), declared mechanism, the draws of q_wrong_g_glm | coverage clears the floor, bias inside the margin, SE ratio inside the sanity band | bias -0.000158 to 0.000761, coverage 0.9440 to 0.9614, SE ratio 0.9993, 0.9725 to 1.0276 | pass |
| `one_wrong_inference` | `q_wrong_g_known__ey1__valid` | positive | arm-1 mean: law L, outcome regression on A and W1 (wrong), declared mechanism, the draws of q_wrong_g_glm | coverage clears the floor, bias inside the margin, SE ratio inside the sanity band | bias -0.000662 to 0.000259, coverage 0.9399 to 0.9580, SE ratio 0.9939, 0.9658 to 1.0247 | pass |
| `one_wrong_inference` | `q_wrong_g_slow__ate__growth` | control | ATE: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2 | the lower bound of the 99% interval of the signed growth of bias/SD from n = 1,000 to n = 16,000 lies above zero | growth 0.5166, 0.3810 to 0.6621 | pass |
| `one_wrong_inference` | `q_wrong_g_slow_n1000__ate__first_order` | control | ATE: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 1,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0195 to -0.0151, margin 0.0067, SE ratio 1.0999 | pass |
| `one_wrong_inference` | `q_wrong_g_slow_n16000__ate__first_order` | control | ATE: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 16,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0083 to -0.0072, margin 0.0017, SE ratio 1.1319 | pass |
| `one_wrong_inference` | `q_wrong_g_slow_n4000__ate__first_order` | control | ATE: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 4,000 | bias interval wholly beyond the margin on the predicted negative side, SE ratio between 0.1 and 10 | bias -0.0123 to -0.0102, margin 0.0033, SE ratio 1.1415 | pass |
| `one_wrong_reported` | `mar_q_wrong_pi_slow__ate__reported` | diagnostic | ATE: law M, randomized missing outcome with the declared treatment probability 1/2, outcome regression on A and W1 (wrong), response mechanism a correct GLM plus the constructed logit offset 2 n^(-1/3) W2 | reported: bias, SE ratio and coverage, with no verdict | bias 0.0032 SD, coverage 0.9180 to 0.9577, SE ratio 1.0012 | reported |
| `one_wrong_reported` | `q_slow_g_wrong_n1000__ey0__reported` | diagnostic | arm-0 mean: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 1,000 | reported: bias, SE ratio and coverage, with no verdict | bias 0.5424 SD, coverage 0.8820 to 0.9300, SE ratio 1.1778 | reported |
| `one_wrong_reported` | `q_slow_g_wrong_n1000__ey1__reported` | diagnostic | arm-1 mean: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 1,000 | reported: bias, SE ratio and coverage, with no verdict | bias -0.5700 SD, coverage 0.8864 to 0.9336, SE ratio 1.2761 | reported |
| `one_wrong_reported` | `q_slow_g_wrong_n16000__ey0__reported` | diagnostic | arm-0 mean: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 16,000 | reported: bias, SE ratio and coverage, with no verdict | bias 0.8021 SD, coverage 0.8198 to 0.8788, SE ratio 0.9643 | reported |
| `one_wrong_reported` | `q_slow_g_wrong_n16000__ey1__reported` | diagnostic | arm-1 mean: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 16,000 | reported: bias, SE ratio and coverage, with no verdict | bias -0.8921 SD, coverage 0.8252 to 0.8833, SE ratio 0.9930 | reported |
| `one_wrong_reported` | `q_slow_g_wrong_n4000__ey0__reported` | diagnostic | arm-0 mean: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 4,000 | reported: bias, SE ratio and coverage, with no verdict | bias 0.6386 SD, coverage 0.8676 to 0.9185, SE ratio 0.9754 | reported |
| `one_wrong_reported` | `q_slow_g_wrong_n4000__ey1__reported` | diagnostic | arm-1 mean: law L, outcome regression a correct GLM plus the constructed logit offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong), at n = 4,000 | reported: bias, SE ratio and coverage, with no verdict | bias -0.6917 SD, coverage 0.8490 to 0.9033, SE ratio 0.9742 | reported |
| `one_wrong_reported` | `q_wrong_g_slow_n1000__ey0__reported` | diagnostic | arm-0 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 1,000 | reported: bias, SE ratio and coverage, with no verdict | bias 0.4107 SD, coverage 0.9260 to 0.9636, SE ratio 1.0739 | reported |
| `one_wrong_reported` | `q_wrong_g_slow_n1000__ey1__reported` | diagnostic | arm-1 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 1,000 | reported: bias, SE ratio and coverage, with no verdict | bias -0.4089 SD, coverage 0.9123 to 0.9535, SE ratio 1.0184 | reported |
| `one_wrong_reported` | `q_wrong_g_slow_n16000__ey0__reported` | diagnostic | arm-0 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 16,000 | reported: bias, SE ratio and coverage, with no verdict | bias 0.7519 SD, coverage 0.8709 to 0.9212, SE ratio 1.0355 | reported |
| `one_wrong_reported` | `q_wrong_g_slow_n16000__ey1__reported` | diagnostic | arm-1 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 16,000 | reported: bias, SE ratio and coverage, with no verdict | bias -0.6698 SD, coverage 0.8775 to 0.9265, SE ratio 1.0218 | reported |
| `one_wrong_reported` | `q_wrong_g_slow_n4000__ey0__reported` | diagnostic | arm-0 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 4,000 | reported: bias, SE ratio and coverage, with no verdict | bias 0.5386 SD, coverage 0.9191 to 0.9586, SE ratio 1.0857 | reported |
| `one_wrong_reported` | `q_wrong_g_slow_n4000__ey1__reported` | diagnostic | arm-1 mean: law L, outcome regression on A and W1 (wrong), mechanism a correct GLM plus the constructed logit offset n^(-1/3) W2, at n = 4,000 | reported: bias, SE ratio and coverage, with no verdict | bias -0.5336 SD, coverage 0.9043 to 0.9475, SE ratio 1.0349 | reported |
<!-- /generated -->

## Result

Every cell passed. The table gives the endpoints that each rule reads.

| case | setting | measured | rule |
| --- | --- | --- | --- |
| valid | `q_wrong_g_known`, ATE | SE ratio 1.0085, coverage 0.9500 | V, pass |
| conservative | `q_wrong_g_glm`, ATE | SE ratio 1.1611 (99% interval 1.1285 to 1.1940), coverage 0.9750 | C, pass |
| conservative | `q_wrong_g_glm_cf`, ATE | SE ratio 1.1908 (99% interval 1.1537 to 1.2304), coverage 0.9820 | C, pass |
| not established | `q_wrong_g_slow`, ATE | bias -0.65, -0.86 and -1.16 SD at n = 1,000, 4,000 and 16,000; coverage 0.935, 0.917 and 0.843 | F and G, pass |
| not established | `q_slow_g_wrong`, ATE | bias -0.88, -1.07 and -1.35 SD; coverage 0.825, 0.789 and 0.687 | F and G, pass |
| not established | `mar_q_wrong_pi_slow`, arm means | bias -0.53 SD at arm 0 and -0.54 SD at arm 1; coverage 0.920 and 0.926 | F, pass |

The measured-values table below checks each of these numbers against the committed rows.

The C cells measure the conservative case that the
[projection derivation](../dr-tmle/theorem.md#a-parametric-mechanism-makes-the-interval-conservative)
predicts. Each row also reports a projected SE: the standard error of the influence curve after
its projection on the mechanism GLM's score columns. Its ratio to the empirical SD is 0.994 in
sample and 1.012 cross-fitted, so the projection accounts for the excess. The empirical SD of the
in-sample GLM fit is 0.87 times that of the declared-mechanism fit on the same draws.

Each first-order control's bias has the sign that the design test computes from the population
conditioning term. Each row also reports a remainder proxy:
the conditioning term evaluated on the fit's own rows, over the empirical SD. On law L the proxy
has the sign of the measured bias and the same order. On law M it is -0.90 and -0.83 SD against a
measured -0.53 and -0.54. The arms of the controls are reported, not gated.

Some fits do not converge and stay in their cells, as declared. Most of them stall, and a few
reach the alternation's outer cap. The mirror ladder has the most: 132, 106 and 63 of 1,000 fits at n = 1,000, 4,000 and 16,000. The C and V
cells have at most 4 of 4,000.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4000 | primary replications |
| `n` | 4000 | observations per primary replication |
| `independent_tests_total` | 3 | primary tests against truth |
| `independent_tests_passed` | 3 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 19 | repeated-sampling property cells |
| `property_cells_passed` | 19 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0246 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9505 | lowest measured primary-study coverage |
| `properties[one_wrong_inference/q_wrong_g_known__ate__valid]:se_ratio` | 1.0085 | declared-mechanism ATE SE ratio |
| `properties[one_wrong_inference/q_wrong_g_known__ate__valid]:coverage` | 0.9500 | its coverage |
| `properties[one_wrong_inference/q_wrong_g_glm__ate__conservative]:se_ratio` | 1.1611 | in-sample GLM-mechanism ATE SE ratio |
| `properties[one_wrong_inference/q_wrong_g_glm__ate__conservative]:se_ratio_ci_lower` | 1.1285 | its 99% lower endpoint |
| `properties[one_wrong_inference/q_wrong_g_glm__ate__conservative]:se_ratio_ci_upper` | 1.1940 | its 99% upper endpoint |
| `properties[one_wrong_inference/q_wrong_g_glm__ate__conservative]:coverage` | 0.9750 | its coverage |
| `properties[one_wrong_inference/q_wrong_g_glm__ate__conservative]:projected_se_ratio` | 0.9941 | its projected SE over the empirical SD |
| `properties[one_wrong_inference/q_wrong_g_glm__ate__conservative]:paired_sd_ratio` | 0.8704 | its empirical SD over the declared-mechanism fit's, on the same draws |
| `properties[one_wrong_inference/q_wrong_g_glm_cf__ate__conservative]:se_ratio` | 1.1908 | cross-fitted GLM-mechanism ATE SE ratio |
| `properties[one_wrong_inference/q_wrong_g_glm_cf__ate__conservative]:se_ratio_ci_lower` | 1.1537 | its 99% lower endpoint |
| `properties[one_wrong_inference/q_wrong_g_glm_cf__ate__conservative]:se_ratio_ci_upper` | 1.2304 | its 99% upper endpoint |
| `properties[one_wrong_inference/q_wrong_g_glm_cf__ate__conservative]:coverage` | 0.9820 | its coverage |
| `properties[one_wrong_inference/q_wrong_g_glm_cf__ate__conservative]:projected_se_ratio` | 1.0123 | its projected SE over the empirical SD |
| `properties[one_wrong_inference/q_wrong_g_slow_n1000__ate__first_order]:standardized_bias` | -0.6473 | slow-mechanism bias in SD, n = 1,000 |
| `properties[one_wrong_inference/q_wrong_g_slow_n4000__ate__first_order]:standardized_bias` | -0.8623 | the same, n = 4,000 |
| `properties[one_wrong_inference/q_wrong_g_slow_n16000__ate__first_order]:standardized_bias` | -1.1640 | the same, n = 16,000 |
| `properties[one_wrong_inference/q_wrong_g_slow_n1000__ate__first_order]:coverage` | 0.9350 | slow-mechanism coverage, n = 1,000 |
| `properties[one_wrong_inference/q_wrong_g_slow_n4000__ate__first_order]:coverage` | 0.9170 | the same, n = 4,000 |
| `properties[one_wrong_inference/q_wrong_g_slow_n16000__ate__first_order]:coverage` | 0.8430 | the same, n = 16,000 |
| `properties[one_wrong_inference/q_wrong_g_slow__ate__growth]:growth` | 0.5166 | signed growth of the slow-mechanism bias |
| `properties[one_wrong_inference/q_wrong_g_slow__ate__growth]:growth_ci_lower` | 0.3810 | its 99% lower endpoint |
| `properties[one_wrong_inference/q_slow_g_wrong_n1000__ate__first_order]:standardized_bias` | -0.8794 | mirror bias in SD, n = 1,000 |
| `properties[one_wrong_inference/q_slow_g_wrong_n4000__ate__first_order]:standardized_bias` | -1.0696 | the same, n = 4,000 |
| `properties[one_wrong_inference/q_slow_g_wrong_n16000__ate__first_order]:standardized_bias` | -1.3481 | the same, n = 16,000 |
| `properties[one_wrong_inference/q_slow_g_wrong_n1000__ate__first_order]:coverage` | 0.8250 | mirror coverage, n = 1,000 |
| `properties[one_wrong_inference/q_slow_g_wrong_n4000__ate__first_order]:coverage` | 0.7890 | the same, n = 4,000 |
| `properties[one_wrong_inference/q_slow_g_wrong_n16000__ate__first_order]:coverage` | 0.6870 | the same, n = 16,000 |
| `properties[one_wrong_inference/q_slow_g_wrong__ate__growth]:growth` | 0.4687 | signed growth of the mirror bias |
| `properties[one_wrong_inference/q_slow_g_wrong__ate__growth]:growth_ci_lower` | 0.3263 | its 99% lower endpoint |
| `properties[one_wrong_inference/mar_q_wrong_pi_slow__ey0__first_order]:standardized_bias` | -0.5319 | law M arm-0 bias in SD |
| `properties[one_wrong_inference/mar_q_wrong_pi_slow__ey1__first_order]:standardized_bias` | -0.5378 | law M arm-1 bias in SD |
| `properties[one_wrong_inference/mar_q_wrong_pi_slow__ey0__first_order]:coverage` | 0.9200 | its arm-0 coverage |
| `properties[one_wrong_inference/mar_q_wrong_pi_slow__ey1__first_order]:coverage` | 0.9260 | its arm-1 coverage |
| `properties[one_wrong_reported/q_slow_g_wrong_n1000__ey0__reported]:nonconverged_fits` | 132 | mirror fits that did not converge, n = 1,000 |
| `properties[one_wrong_reported/q_slow_g_wrong_n4000__ey0__reported]:nonconverged_fits` | 106 | the same, n = 4,000 |
| `properties[one_wrong_reported/q_slow_g_wrong_n16000__ey0__reported]:nonconverged_fits` | 63 | the same, n = 16,000 |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | the line that a C cell's SE-ratio interval must clear |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No rule of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No rule of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound. This study has no comparator |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. Rule C reads the upper limit only |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound. This study has no comparator |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No rule of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No rule of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound. This study has no comparator |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No rule of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No rule of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No rule of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No rule of this study reads it |

## Limitations

| limit | what it means for use |
| --- | --- |
| a binary outcome and two Gaussian covariates | the study does not measure other outcome types or covariate laws |
| the univariate reduction with `guard=("Q", "g")` | the bivariate reduction and other guards have no witness here |
| GLM and polynomial learners with constructed offsets | no off-the-shelf flexible learner is measured. An in-sample gradient-boosting mechanism with a wrong outcome regression gave an ATE bias of −2.37 SD and coverage 0.38 at n = 4,000 (`reviews/rm40/hgb_diagnosis.log`). [RM41](../../roadmap.md#rm41-find-the-dr-tmle-bias-with-an-in-sample-flexible-mechanism) owns it |
| no strata, clusters, weights or more arms | those routes inherit the case table without a witness |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_one_wrong/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.drtmle_one_wrong.regenerate --output <scratch>` to
regenerate the artifacts under the declared run form. The
[property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_one_wrong/properties.csv)
carry every published row. The design test, `tests/unit/test_drtmle_one_wrong_design.py`, pins
the truths, the limits, the predicted signs and the pass probabilities.
