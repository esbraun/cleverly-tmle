# Outcome-adaptive per-arm C-TMLE

This study measures `CTMLE(strategy="oat")` with the default `oat_design="per_arm"`. For each arm
$a$, the fit regresses $1\{A = a\}$ on the one column $\bar Q_n(a, W)$. The section
[the per-arm outcome-adaptive design](../collaborative-tmle.md#the-per-arm-outcome-adaptive-design)
states the estimator and its curve. The base result is Theorem 1 of Benkeser, Cai and van der Laan
(2020) for one treatment-specific mean, applied to each arm.

The study pairs the fit with R
[`drtmle`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c)
1.1.2 at commit `538a3a2`, with `adapt_g = TRUE`. The runner calls `drtmle` once per arm on the
binary recode $1\{A = a\}$. Both sides read this package's initial outcome regression, so the pairs
test the design, the mechanism and the targeting step.

The study publishes under the `reporting` policy. Its declared revert cells are red, so every
per-arm fit takes the `generated_design_plugin` status and reports no interval. A shared-design
fit already withheld its interval before this study. The coverage and
standard-error columns below measure `plugin_interval` and `plugin_std_error` as diagnostics.
Their pass labels do not authorize `ci`, `pvalue` or `std_error`.
[The applied revert](../collaborative-tmle.md#the-applied-revert) gives the diagnosis, and
[F19](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) owns every red row.

## What was compared

| setting | `cleverly` | R `drtmle` |
| --- | --- | --- |
| datasets | 800 samples of 1,500 rows for each law | the identical rows |
| law | $W = (W_1, W_2, W_3)$ is independent standard normal. The outcome is binary with $\operatorname{logit} \bar Q_0(a, W) = c_a + u_{a1} W_1 + u_{a2} W_2$, so each arm reads its own index. The treatment law is a softmax in $W_1$ and $W_3$. $W_3$ enters only the treatment law, so it is an instrument | the same |
| scenarios | `binary_active`: two arms. `three_arm_active`: three arms, with arm 0 as the contrast arm | the same |
| reported parameters | at two arms, the arm means, the ATE, RR, OR, PAR and PAF. At three arms, the three arm means and the ATE, RR and OR of arms 1 and 2 against arm 0 | the same |
| outcome regression | an unpenalized logistic regression on the arm indicators, $W$ and $A \times W$. It is correctly specified | this package's initial fit, passed as `Qn` |
| mechanism | for each arm, an unpenalized logistic regression of $1\{A = a\}$ on $(l, l^2, l^3)$, $l = \operatorname{logit} \bar Q_n(a, W)$, with $\bar Q_n$ clipped to $[10^{-6}, 1 - 10^{-6}]$. `g_bounds=(0.01, 0.99)` | `glm_g` with the same cubic, `tolg = 0.01` |
| targeting | one logistic fluctuation for each arm on that arm's clever covariate | the TMLE stage of `drtmle` |
| intervals | pointwise 95% Wald from the per-arm curve, with RR and OR on the log scale. They are diagnostics | the same, from `tmle$cov` |

The `drtmle` result carries no TMLE curve. The runner rebuilds each arm's curve on its
one-dimensional logistic submodel at the reported estimate. It refuses a fit unless the rebuilt
standard error equals the square root of `tmle$cov[1, 1]` to a relative `1e-8`. The contrasts use the rebuilt curves. A call with
`a_0` equal to every level does not fit this design, because `drtmle` then builds a sequential
chain of conditional binary regressions.

The table gives the law-level numbers. A $10^6$-draw Monte Carlo with Gauss-Hermite projections
computes them, in `tests/studies/oat_per_arm_laws.py:design_numbers`. The efficiency ratio is the
variance of Theorem 1's curve over the variance of the efficient influence function.

| quantity | `binary_active` | `three_arm_active` |
| --- | --- | --- |
| arm means | 0.4406, 0.5593 | 0.4178, 0.4999, 0.5834 |
| efficiency ratio, arm means | 0.428, 0.428 | 0.792, 0.178, 0.178 |
| $L_2$ gap between the per-arm and the shared design limits | 0.059, 0.059 | 0.037, 0.114, 0.059 |
| share of rows with the true $g_0$ outside $(0.025, 0.975)$ | 0.029 | 0.004, 0.072, 0.065 |
| $L_2$ distance from the learner limit to the projection | below $10^{-6}$ | below $2 \times 10^{-4}$ |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate` | average treatment effect | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0021 to 0.0031 | 0.9087 | 0.8500 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate` | average treatment effect | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0021 to 0.0031 | 0.9087 | 0.8500 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey0` | counterfactual mean under no treatment | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0024 to 0.0013 | 0.9025 | 0.8526 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey0` | counterfactual mean under no treatment | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0024 to 0.0013 | 0.9025 | 0.8526 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey1` | counterfactual mean under treatment | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0019 to 0.0018 | 0.9062 | 0.8463 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey1` | counterfactual mean under treatment | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0019 to 0.0018 | 0.9062 | 0.8463 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0078 to 0.0136 | 0.9087 | 0.8493 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0078 to 0.0136 | 0.9087 | 0.8493 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `paf` | population attributable fraction | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0024 to 0.0036 | 0.8875 | 0.7835 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `paf` | population attributable fraction | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0024 to 0.0036 | 0.8875 | 0.7835 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `par` | population attributable risk | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0013 to 0.0018 | 0.8875 | 0.7859 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `par` | population attributable risk | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0013 to 0.0018 | 0.8875 | 0.7859 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0038 to 0.0070 | 0.9163 | 0.8517 | **fail** |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0038 to 0.0070 | 0.9163 | 0.8517 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate[1 vs 0]` | difference in counterfactual means, 1 versus 0 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0039 to 0.0030 | 0.8888 | 0.8311 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate[1 vs 0]` | difference in counterfactual means, 1 versus 0 | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0039 to 0.0030 | 0.8888 | 0.8311 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate[2 vs 0]` | difference in counterfactual means, 2 versus 0 | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0039 to 0.0028 | 0.9087 | 0.8631 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate[2 vs 0]` | difference in counterfactual means, 2 versus 0 | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0039 to 0.0028 | 0.9087 | 0.8631 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[0]` | counterfactual mean under treatment arm '0' | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0021 to 0.0027 | 0.9363 | 0.9212 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[0]` | counterfactual mean under treatment arm '0' | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0021 to 0.0027 | 0.9363 | 0.9212 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[1]` | counterfactual mean under treatment arm '1' | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0026 to 0.0023 | 0.8588 | 0.7608 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[1]` | counterfactual mean under treatment arm '1' | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0026 to 0.0023 | 0.8588 | 0.7608 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[2]` | counterfactual mean under treatment arm '2' | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0027 to 0.0022 | 0.8738 | 0.7740 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[2]` | counterfactual mean under treatment arm '2' | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0027 to 0.0022 | 0.8738 | 0.7740 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `or[1 vs 0]` | marginal odds ratio, 1 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0148 to 0.0133 | 0.8900 | 0.8329 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `or[1 vs 0]` | marginal odds ratio, 1 versus 0, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0148 to 0.0133 | 0.8900 | 0.8329 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `or[2 vs 0]` | marginal odds ratio, 2 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0142 to 0.0137 | 0.9113 | 0.8627 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `or[2 vs 0]` | marginal odds ratio, 2 versus 0, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0142 to 0.0137 | 0.9113 | 0.8627 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr[1 vs 0]` | marginal risk ratio, 1 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0081 to 0.0072 | 0.9000 | 0.8434 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr[1 vs 0]` | marginal risk ratio, 1 versus 0, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0081 to 0.0072 | 0.9000 | 0.8434 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr[2 vs 0]` | marginal risk ratio, 2 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE | -0.0073 to 0.0068 | 0.9175 | 0.8817 | **fail** |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr[2 vs 0]` | marginal risk ratio, 2 versus 0, reported on the log scale | R `drtmle` with `adapt_g`, by a binary recode per arm | -0.0073 to 0.0068 | 0.9175 | 0.8817 | **fail** |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate` | average treatment effect | 1.809e-10 | 4.183e-08 | 1.0000 | 0 | 6.837e-09 vs 0.0500 | equivalent |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey0` | counterfactual mean under no treatment | -6.747e-11 | 2.215e-08 | 1.0000 | 0 | 5.179e-09 vs 0.0500 | equivalent |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey1` | counterfactual mean under treatment | 1.134e-10 | 3.691e-08 | 1.0000 | 0 | 3.063e-09 vs 0.0500 | equivalent |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | 1.175e-09 | 4.064e-08 | 1.0000 | 0 | 6.885e-09 vs 0.0500 | equivalent |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `paf` | population attributable fraction | 1.375e-10 | 2.758e-08 | 1.0000 | 0 | 3.761e-09 vs 0.0500 | equivalent |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `par` | population attributable risk | 6.747e-11 | 2.702e-08 | 1.0000 | 0 | 3.051e-09 vs 0.0500 | equivalent |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | 4.586e-10 | 4.061e-08 | 1.0000 | 0 | 7.033e-09 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate[1 vs 0]` | difference in counterfactual means, 1 versus 0 | 2.357e-09 | 4.145e-07 | 1.0000 | 0 | 1.347e-07 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate[2 vs 0]` | difference in counterfactual means, 2 versus 0 | -4.589e-11 | 8.271e-09 | 1.0000 | 0 | 5.229e-09 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[0]` | counterfactual mean under treatment arm '0' | 1.237e-11 | 3.177e-09 | 1.0000 | 0 | 2.161e-09 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[1]` | counterfactual mean under treatment arm '1' | 2.369e-09 | 5.857e-07 | 1.0000 | 0 | 1.805e-07 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[2]` | counterfactual mean under treatment arm '2' | -3.352e-11 | 8.294e-09 | 1.0000 | 0 | 8.831e-09 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `or[1 vs 0]` | marginal odds ratio, 1 versus 0, reported on the log scale | 9.058e-09 | 2.777e-07 | 1.0000 | 0 | 1.341e-07 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `or[2 vs 0]` | marginal odds ratio, 2 versus 0, reported on the log scale | -1.735e-10 | 3.794e-09 | 1.0000 | 0 | 5.147e-09 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr[1 vs 0]` | marginal risk ratio, 1 versus 0, reported on the log scale | 5.324e-09 | 3.524e-07 | 1.0000 | 0 | 1.413e-07 vs 0.0500 | equivalent |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr[2 vs 0]` | marginal risk ratio, 2 versus 0, reported on the log scale | -9.641e-11 | 5.893e-09 | 1.0000 | 0 | 4.436e-09 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_oat` | positive | outcome-adaptive C-TMLE with cross-fitted nuisances and a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.8842 to 1.0686 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3638 to 0.4341 | pass |
| `generated_design` | `binary_active__estimated` | positive | two-arm law with an arm-specific outcome index and an instrument: the same design is estimated from the data, as a real fit does | SE ratio and coverage intervals both inside their calibration bands; the paired deficit is reported only | coverage 0.8892 to 0.9116, SE ratio 0.8217 to 0.8671 | **fail** |
| `generated_design` | `binary_active__oracle_design` | positive | two-arm law with an arm-specific outcome index and an instrument: the outcome-adaptive design is supplied rather than estimated | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9375 to 0.9545, SE ratio 0.9715 to 1.0241 | pass |
| `generated_design` | `three_arm_active__estimated` | positive | three-arm law with an arm-specific outcome index and an instrument: the same design is estimated from the data, as a real fit does | SE ratio and coverage intervals both inside their calibration bands; the paired deficit is reported only | coverage 0.9085 to 0.9290, SE ratio 0.8674 to 0.9131 | **fail** |
| `generated_design` | `three_arm_active__oracle_design` | positive | three-arm law with an arm-specific outcome index and an instrument: the outcome-adaptive design is supplied rather than estimated | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9413 to 0.9578, SE ratio 0.9836 to 1.0360 | pass |
| `interval_calibration` | `binary_active__correctly_specified` | positive | two-arm law with an arm-specific outcome index and an instrument: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.8820 to 0.9170, SE ratio 0.8202 to 0.8874 | **fail** |
| `interval_calibration` | `binary_active__noise_control` | control | two-arm law with an arm-specific outcome index and an instrument: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.7776 to 0.8240, SE ratio 0.6185 to 0.6726 | pass |
| `interval_calibration` | `binary_active__shrunken_se_control` | control | two-arm law with an arm-specific outcome index and an instrument: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7309 to 0.7808, SE ratio 0.5742 to 0.6214 | pass |
| `interval_calibration` | `binary_repeats__correctly_specified` | positive | the two-arm law, the median estimate over three cross-fitting splits: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.8842 to 0.9189, SE ratio 0.8081 to 0.8753 | **fail** |
| `interval_calibration` | `binary_weighted__correctly_specified` | positive | the two-arm law under the declared known weight 0.5 + 1{W1 > 0}: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.8863 to 0.9207, SE ratio 0.8060 to 0.8714 | **fail** |
| `interval_calibration` | `three_arm_active__correctly_specified` | positive | three-arm law with an arm-specific outcome index and an instrument: both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.8928 to 0.9262, SE ratio 0.8509 to 0.9240 | **fail** |
| `interval_calibration` | `three_arm_active__noise_control` | control | three-arm law with an arm-specific outcome index and an instrument: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.7864 to 0.8321, SE ratio 0.6425 to 0.6982 | pass |
| `interval_calibration` | `three_arm_active__shrunken_se_control` | control | three-arm law with an arm-specific outcome index and an instrument: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7433 to 0.7924, SE ratio 0.5952 to 0.6465 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `robustness_contract` | `outcome_correct` | positive | the outcome regression is correct and the mechanism is not | bias interval inside the equivalence margin, SE ratio must remain between 0.1 and 10.0 | bias -0.000703 to 0.0033, margin 0.0061, SE ratio 0.8757 | pass |
| `robustness_contract` | `outcome_wrong` | control | the outcome regression is misspecified | bias interval must fall entirely outside the margin, SE ratio must remain between 0.1 and 10.0 | bias 0.0634 to 0.0681, margin 0.0073, SE ratio 0.9857 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000434, coverage 0.8786 to 0.9357, SE ratio 0.8636 | **fail** |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0044, coverage 0.8721 to 0.9308, SE ratio 0.8275 | **fail** |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000358, coverage 0.8850 to 0.9406, SE ratio 0.8598 | **fail** |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5579 to -0.4843 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5090 to -0.5059 | pass |
| `simultaneous_coverage` | `three_arm_active__pointwise_joint_control` | control | three-arm law with an arm-specific outcome index and an instrument: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6753 to 0.7239 | pass |
| `simultaneous_coverage` | `three_arm_active__simultaneous_band` | positive | three-arm law with an arm-specific outcome index and an instrument: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.8527 to 0.8883 | **fail** |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0963, 0.0712 to 0.1262 | **fail** |
<!-- /generated -->

Before its gated run, the study declared its revert cells.

| revert cells | count |
| --- | --- |
| primary coverage, one cell for each estimand | 16 |
| `interval_calibration` positive cells | 4 |
| `generated_design` cells | 4 |
| the simultaneous band | 1 |

One red revert cell with no defect withholds the per-arm interval.

The two `oracle_design` cells pin $\bar Q$ to the truth. They still estimate the mechanism. Each
passes its own rule, which the table shows. The [red-cell ledger](red-cells.md) counts each one red
by the family's joint clause, because its estimated twin fails. That contrast locates the deficit
in the estimated design.

The simultaneous band cell sets `OAT_PER_ARM_INFERENTIAL` inside the study for that one fit. It
measures the band that the per-arm design would build from its stacked curves. The shipped fit
withholds the band.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications per law |
| `n` | 1500 | observations per primary replication |
| `independent_tests_total` | 32 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 16 | paired comparisons with R `drtmle` |
| `paired_tests_passed` | 16 | of those, passing |
| `property_cells_total` | 25 | repeated-sampling property cells |
| `property_cells_passed` | 12 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0269 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.8588 | lowest primary coverage of the diagnostic interval |
| `min_se_ratio_ci_lower` | 0.7142 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 0.9919 | highest bootstrap primary SE-ratio endpoint |
| `max_margin_utilization` | 5.857e-07 | largest share of the paired margin used |
| `mechanism_fits_with_rows_at_bound` | 157 | primary fits with one or more rows of the per-arm mechanism at a declared bound |
| `mechanism_max_rows_at_bound` | 21 | most rows at a bound in one fit |
| `mechanism_max_gap_off_bound` | 4.658e-07 | largest gap between the two mechanisms in a fit with no row at a bound |
| `mechanism_max_gap` | 0.0080 | largest gap between the two mechanisms in any fit |
| `design_gap[binary_active/ate]:mean_design_gap` | 0.0458 | mean absolute shift from the shared design, in this fit's standard errors, two-arm `ate` |
| `design_gap[three_arm_active/ey[1]]:mean_design_gap` | 1.0825 | the same for the three-arm `ey[1]` |
| `properties[crossfit_overfitting/cross_fitted_oat]:standardized_bias` | 0.9461 | bias of the cross-fitted fit with a fully grown tree, in empirical standard deviations |
| `properties[crossfit_overfitting/cross_fitted_oat]:coverage` | 0.8275 | coverage of that fit's diagnostic interval |
| `properties[root_n_and_efficiency/n_2000]:empirical_se` | 0.0247 | empirical spread of the two-arm `ate` at n = 2,000 |
| `properties[root_n_and_efficiency/n_2000]:mean_std_error` | 0.0213 | mean reported diagnostic standard error at n = 2,000 |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power cell must clear |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:overfit_control_ceiling` | 0.7500 | coverage the in-sample overfitting control must stay under |
| `margin:overfit_coverage_gain` | 0.1500 | coverage gain cross-fitting must show over that control |
| `margin:overfit_se_floor` | 0.8500 | SE-ratio floor of the cross-fitted overfitting cell |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate the null cell must bound |
| `margin:union_model_se_lower` | 0.1000 | robustness-contract SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | robustness-contract SE-ratio screen, upper limit |

The reported standard error is too small, and the estimator is still more precise than the
efficient bound. At n = 2,000 the empirical spread times $\sqrt{n}$ is about 1.10. The per-arm
curve standard deviation of the law is 0.949, and the efficient influence function's is 1.451
(`POWER_CURVE_SD` and `EIF_CURVE_SD` in `tests/studies/ctmle_oat_per_arm_properties.py`). The
study fits no ordinary TMLE on the same draws, so this comparison is reported and not gated.

## Limitations

| limitation | what it means for use |
| --- | --- |
| No cell measures published inference | Every `oat` fit withholds `ci`, `pvalue` and `std_error`. The `std_error`, `ci_lower`, `ci_upper` and `covered` columns read `plugin_std_error` and `plugin_interval`. The coverage, calibration, null-size, band and power cells describe that diagnostic |
| The diagnostic standard error is too small on both laws | The primary SE ratios run from `min_se_ratio_ci_lower` to `max_se_ratio_ci_upper`, and the ladder does not close the gap as n grows. Condition (v) of Theorem 1 fails for an estimated outcome regression, so the remainder $R_{24}$ is first order. R `drtmle` reports the same standard errors, so the pairing cannot detect the gap |
| The points are paired, not the intervals | All 16 paired rows are equivalent, with `max_margin_utilization` of the margin used. The truth rows of both implementations are red alike. The per-arm design moves the three-arm points away from the shared design by `design_gap[three_arm_active/ey[1]]:mean_design_gap` standard errors on `ey[1]` |
| The mechanisms agree only off the bounds | Where no row of the per-arm mechanism sits at a bound, the two mechanisms agree to `mechanism_max_gap_off_bound`. In `mechanism_fits_with_rows_at_bound` fits some rows sit at a bound, and the gap reaches `mechanism_max_gap`. The rows at a bound are reported, not gated |
| A flexible outcome learner biases the point estimate | The `crossfit_overfitting` pair fits a fully grown regression tree. Cross-fitting restores the SE ratio, which is all that pair gates. The cross-fitted estimate still has a bias of `properties[crossfit_overfitting/cross_fitted_oat]:standardized_bias` empirical standard deviations, and its diagnostic interval covers `properties[crossfit_overfitting/cross_fitted_oat]:coverage`. The point-estimate evidence holds for the correctly specified GLM |
| Two parametric laws with a binary outcome | Each law has a normal index for each arm and one instrument. A continuous outcome and more than three arms have fast-tier evidence only |
| Strata and missing data have a study of their own | The laws above have no strata and no missing data. The [strata and missing-data study](outcome-adaptive-per-arm-c-tmle-strata-missing.md) measures those fits. Clusters stay refused |
| One unstratified split does the cross-fitting | The property cells declare `stratify_folds="none"`. The primary fit is in sample, so the pairing says nothing about cross-fitted parity |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm/manifest.json)
records the seeds, the margins, the declared revert cells, the expected red rows, the estimator
configuration and the hashes. Run `python -m tests.canonical.drtmle_oat_per_arm.regenerate` to
regenerate the artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm/equivalence.csv),
[design gap](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm/design-gap.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_oat_per_arm/properties.csv)
carry every published row. `tests/diagnostics/x17_generated_design/` reproduces the diagnosis.

The committed R phase ran under the harness before RM39. RM39's revision row for
`tests/canonical/study_harness.R` covers that hash. `tests/canonical/harness_fixture/determinism.csv`
has no row for `run_drtmle_oat_per_arm.R` yet. The runner itself calls no random number
generator. It fits with `cvFolds = 1`, `glm_g` and `uniroot`.
