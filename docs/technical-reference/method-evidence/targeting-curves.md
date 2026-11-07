# Targeting curves

This study validates the targeting curve of a priority score, from
`IdentifiedEffect.targeting_curve(...)` on an external score and from
`curve.diagnostics.targeting_curve()` on a curve refit on held-out rows. The targets are the
TOC at fixed treated fractions $q$, the AUTOC, the Qini coefficient, two group effects, the
paired difference of two scores, and the sup-t band of the TOC over the grid. Each point is the
substitution estimate of one pooled TMLE fluctuation of the cross-fitted ATE fit, and each
interval is the Wald interval from 2,000 half-sample draws.
[Targeting curves](../conditional-effects.md#targeting-curves) gives the targets, the estimator
and its conditions.

**No canonical implementation is compared.** R `grf`'s `rank_average_treatment_effect`
computes the AUTOC and the Qini of a score from its own forest's scores. A comparison on shared
nuisance scores is pending, and its absence does not block the row. A zero-row equivalence
artifact records it.

## What was tested

| setting | declaration |
| --- | --- |
| laws | $W \sim U(-1, 1)^3$ and $\operatorname{logit} g(W) = 0.5 W_1 - 0.5 W_2 + 0.25 W_3$ for every law. `H`: $Y = 0.5 W_1 - 0.5 W_2 + W_3 + A(1 + W_1) + 0.25\varepsilon$ with the external score $S = W_1 + 0.5 W_3$. `B`: binary, $\operatorname{logit} \mu(a, W) = -0.2 + 0.4 W_2 + a(0.3 + 1.2 W_1)$. `K`: the effect $1 + W_1 + W_1 W_3$ with a main-terms outcome regression, which is wrong for it, and $g$ known. `U`: law `B` with the score $W_3$, so every TOC is 0. `N`: a constant effect and noise SD 0.5, so every TOC is 0. `D`: law `B` with the atomic score $\operatorname{round}(2 W_1)/2$. `S`: law `H`, with the score a B-spline effect curve in $(W_1, W_2, W_3)$ refit on 2,000 training rows and evaluated on 2,000 others. `P`: the effect $1 + 0.46 W_1$ and noise SD 1, at the slope where the band's exact-law power is 0.90. Cell `C` is the paired difference $S - S_B$ with $S_B = W_1 + 1.5 W_3$ on the fits of `H` |
| fit | `effect.targeting_curve("S", computed_from=["W1", "W3"], score_kind="external", n_half_samples=2000)` on `ConditionalContrast(Modifiers(["W1"]))`, and `fit.effect_curve().diagnostics.targeting_curve(n_half_samples=2000)` on cell `S`, with every other setting at its default |
| learners | a spline GLM per arm (cubic splines, five knots) for $\mu$ and a logistic GLM for $g$, except `K`: a main-terms linear regression per arm and the known $g$ |
| truth | Gauss-Legendre quadrature with the score's indicator integrated in closed form; on cell `S`, a fixed scrambled Sobol set of $2^{20}$ points over the curve each replicate fitted |
| primary budget | 6,000 replications of `H` at n = 2,000 |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| property budget | 6,000 replications of each group at n = 2,000, and 800 of `P` at n = 1,000 |
| seeds | `seed=20261107` and `resampling_seed=20261108`; every property group draws from its own labelled stream |
| policy | `reporting`, with one declared reporting row; every other cell gates |

The laws, groups, learners, budgets, controls, seeds and the reporting row were fixed before the
run, in `tests/studies/targeting_curves.py`. The replication count is the smallest multiple of
400 from 3,600 at which every gated calibration row passes together with probability 0.95, at
bootstrap ratios that an oracle computation with the true nuisances
(`tests/studies/targeting_curves_oracle.csv`) and a pilot on a disjoint stream
(`tests/studies/targeting_curves_pilot.csv`) measured first.
`tests/unit/test_targeting_curves_design.py` checks the truths, the cells, the reporting set and
those design numbers.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law H, a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3, a spline GLM per arm and a logistic treatment model, n = 2,000 | `autoc` | the AUTOC, the integral of the TOC over q | `cleverly` targeting curve of an external score: one pooled TMLE fluctuation of the cross-fitted ATE fit along the span of the rank weights, with a Wald interval from 2,000 half-sample draws | -0.000861 to 0.000006 | 0.9490 | 0.9987 | pass |
| law H, a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3, a spline GLM per arm and a logistic treatment model, n = 2,000 | `qini` | the Qini coefficient, the integral of q times the TOC over q | `cleverly` targeting curve of an external score: one pooled TMLE fluctuation of the cross-fitted ATE fit along the span of the rank weights, with a Wald interval from 2,000 half-sample draws | -0.000139 to 0.000115 | 0.9535 | 1.0110 | pass |
| law H, a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3, a spline GLM per arm and a logistic treatment model, n = 2,000 | `toc[0.1]` | the TOC at q = 0.1: the mean effect of the top tenth of S minus the ATE | `cleverly` targeting curve of an external score: one pooled TMLE fluctuation of the cross-fitted ATE fit along the span of the rank weights, with a Wald interval from 2,000 half-sample draws | -0.0019 to 0.000604 | 0.9530 | 1.0222 | pass |
| law H, a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3, a spline GLM per arm and a logistic treatment model, n = 2,000 | `toc[0.5]` | the TOC at q = 0.5: the mean effect of the top half of S minus the ATE | `cleverly` targeting curve of an external score: one pooled TMLE fluctuation of the cross-fitted ATE fit along the span of the rank weights, with a Wald interval from 2,000 half-sample draws | -0.000408 to 0.000507 | 0.9560 | 1.0264 | pass |
| law H, a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3, a spline GLM per arm and a logistic treatment model, n = 2,000 | `toc[0.9]` | the TOC at q = 0.9: the mean effect of the top nine tenths of S minus the ATE | `cleverly` targeting curve of an external score: one pooled TMLE fluctuation of the cross-fitted ATE fit along the span of the rank weights, with a Wald interval from 2,000 half-sample draws | -0.000130 to 0.000152 | 0.9562 | 1.0274 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | for each quantity of seven groups and the paired difference (R = 6,000), does the SE-ratio interval lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98 | draws that keep the full-sample cut, draws that keep the full-sample rank weights and draws that resample two scores separately must each lie above the band; the untargeted one-step draws on `K` must lie below it |
| `bias` | is each point control's standardized-bias interval wholly outside $\pm 0.25$ | the `S >= cut` rule, which puts a whole tied block in the top fraction, on `D`; the curve fitted on its own evaluation rows on `S` |
| `simultaneous_coverage` | does the band's joint coverage interval lie inside 0.92 to 0.98 on `H` | the pointwise intervals read jointly over the grid must cover below 0.95 |
| `type_i_error` | on `U` and `N`, where every TOC is 0, is the 99% upper end of the band's rejection rate at most 0.10 | paired with the power cell |
| `power` | on `P` (n = 1,000, R = 800), is the lower end of the band's rejection rate at least 0.80 | an inert band fails it |

The reporting row is the paired difference of the TOC at 0.1. Before the run, the oracle
computation read its bootstrap SE ratio at 1.063 at n = 2,000, 1.051 at 4,000 and 1.024 at
8,000, a conservative finite-sample excess that shrinks with $n$. The row publishes its band
verdict, and it gates one-sidedly: the lower ends of its 99% SE-ratio and coverage intervals
must reach 0.93 and 0.92, so an interval that became too narrow would fail.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bias` | `D_toc_0.2__whole_block` | control | law D (a binary outcome and the atomic score round(2 W1) / 2, five tied blocks), n = 2,000: the TOC at q = 0.2: the rule S >= cut, which puts the whole tied block at the cut in the top fraction | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias -0.0271 to -0.0251, margin 0.0072 | pass |
| `bias` | `S_autoc__in_sample` | control | law H with the score refit on 2,000 training rows as a B-spline effect curve in (W1, W2, W3) and evaluated on 2,000 other rows (route 2): the AUTOC: the effect curve fitted on the evaluation rows themselves ranks those rows, read against the truth of that curve | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0056 to 0.0064, margin 0.0031 | pass |
| `bias` | `S_toc_0.1__in_sample` | control | law H with the score refit on 2,000 training rows as a B-spline effect curve in (W1, W2, W3) and evaluated on 2,000 other rows (route 2): the TOC at q = 0.1: the effect curve fitted on the evaluation rows themselves ranks those rows, read against the truth of that curve | the 99% interval of the standardized bias must lie wholly outside +-0.25 | bias 0.0121 to 0.0144, margin 0.0087 | pass |
| `interval_calibration` | `B_autoc__correctly_specified` | positive | law B (a binary outcome with a logistic effect in W1 and the external score S = W1 + 0.5 W3), n = 2,000: the AUTOC: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9391 to 0.9542, SE ratio 0.9735 to 1.0197 | pass |
| `interval_calibration` | `B_qini__correctly_specified` | positive | law B (a binary outcome with a logistic effect in W1 and the external score S = W1 + 0.5 W3), n = 2,000: the Qini coefficient: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9395 to 0.9545, SE ratio 0.9724 to 1.0207 | pass |
| `interval_calibration` | `B_toc_0.1__correctly_specified` | positive | law B (a binary outcome with a logistic effect in W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.1: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9414 to 0.9562, SE ratio 0.9956 to 1.0434 | pass |
| `interval_calibration` | `B_toc_0.5__correctly_specified` | positive | law B (a binary outcome with a logistic effect in W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.5: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9453 to 0.9596, SE ratio 0.9906 to 1.0386 | pass |
| `interval_calibration` | `B_toc_0.9__correctly_specified` | positive | law B (a binary outcome with a logistic effect in W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.9: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9398 to 0.9548, SE ratio 0.9802 to 1.0270 | pass |
| `interval_calibration` | `D_autoc__correctly_specified` | positive | law D (a binary outcome and the atomic score round(2 W1) / 2, five tied blocks), n = 2,000: the AUTOC: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9412 to 0.9560, SE ratio 0.9788 to 1.0246 | pass |
| `interval_calibration` | `D_qini__correctly_specified` | positive | law D (a binary outcome and the atomic score round(2 W1) / 2, five tied blocks), n = 2,000: the Qini coefficient: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9380 to 0.9532, SE ratio 0.9758 to 1.0231 | pass |
| `interval_calibration` | `D_toc_0.2__correctly_specified` | positive | law D (a binary outcome and the atomic score round(2 W1) / 2, five tied blocks), n = 2,000: the TOC at q = 0.2: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9411 to 0.9559, SE ratio 0.9814 to 1.0291 | pass |
| `interval_calibration` | `D_toc_0.5__correctly_specified` | positive | law D (a binary outcome and the atomic score round(2 W1) / 2, five tied blocks), n = 2,000: the TOC at q = 0.5: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9400 to 0.9550, SE ratio 0.9725 to 1.0184 | pass |
| `interval_calibration` | `H_S_minus_S_B_autoc__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the AUTOC of S minus that of S_B = W1 + 1.5 W3, on the same draws: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9402 to 0.9551, SE ratio 0.9795 to 1.0271 | pass |
| `interval_calibration` | `H_S_minus_S_B_qini__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the Qini coefficient of S minus that of S_B = W1 + 1.5 W3, on the same draws: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9395 to 0.9545, SE ratio 0.9739 to 1.0198 | pass |
| `interval_calibration` | `H_S_minus_S_B_toc_0.1__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.1 of S minus that of S_B = W1 + 1.5 W3, on the same draws: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands; a reporting row, whose one-sided gate needs the lower ends of its SE-ratio and coverage intervals at or above 0.93 and 0.92 | coverage 0.9559 to 0.9687, SE ratio 1.0350 to 1.0857 | **fail** |
| `interval_calibration` | `H_S_minus_S_B_toc_0.5__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.5 of S minus that of S_B = W1 + 1.5 W3, on the same draws: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9462 to 0.9604, SE ratio 1.0018 to 1.0483 | pass |
| `interval_calibration` | `H_autoc__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the AUTOC: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9368 to 0.9522, SE ratio 0.9699 to 1.0188 | pass |
| `interval_calibration` | `H_fixed_cut_toc_0.5__inflated_se_control` | control | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.5: the same estimate with the SE of draws that keep the full-sample cut | the SE-ratio interval must lie above the calibration band | coverage 0.9810 to 0.9892, SE ratio 1.2164 to 1.2762 | pass |
| `interval_calibration` | `H_fixed_cut_toc_0.9__inflated_se_control` | control | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.9: the same estimate with the SE of draws that keep the full-sample cut | the SE-ratio interval must lie above the calibration band | coverage 0.9967 to 0.9995, SE ratio 1.6790 to 1.7605 | pass |
| `interval_calibration` | `H_fixed_ranks_autoc__inflated_se_control` | control | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the AUTOC: the same estimate with the SE of draws that keep the full-sample rank weights | the SE-ratio interval must lie above the calibration band | coverage 0.9991 to 1, SE ratio 2.6816 to 2.8154 | pass |
| `interval_calibration` | `H_group_effect_0.2__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the mean effect of the top fifth of the score: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9582, SE ratio 0.9819 to 1.0293 | pass |
| `interval_calibration` | `H_group_effect_1__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the mean effect of the bottom fifth of the score: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9494 to 0.9632, SE ratio 0.9988 to 1.0462 | pass |
| `interval_calibration` | `H_qini__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the Qini coefficient: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9423 to 0.9570, SE ratio 0.9733 to 1.0229 | pass |
| `interval_calibration` | `H_toc_0.1__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.1: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9393 to 0.9543, SE ratio 0.9856 to 1.0336 | pass |
| `interval_calibration` | `H_toc_0.5__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.5: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9576, SE ratio 0.9913 to 1.0393 | pass |
| `interval_calibration` | `H_toc_0.9__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.9: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9502 to 0.9638, SE ratio 1.0047 to 1.0551 | pass |
| `interval_calibration` | `H_unpaired_S_minus_S_B_autoc__inflated_se_control` | control | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the AUTOC of S minus that of S_B = W1 + 1.5 W3, on the same draws: the same estimate with the SE of draws that resample the two scores separately | the SE-ratio interval must lie above the calibration band | coverage 0.9991 to 1, SE ratio 2.3888 to 2.5054 | pass |
| `interval_calibration` | `H_unpaired_S_minus_S_B_toc_0.1__inflated_se_control` | control | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the TOC at q = 0.1 of S minus that of S_B = W1 + 1.5 W3, on the same draws: the same estimate with the SE of draws that resample the two scores separately | the SE-ratio interval must lie above the calibration band | coverage 0.9982 to 0.9999, SE ratio 1.7429 to 1.8286 | pass |
| `interval_calibration` | `K_autoc__known_mechanism` | positive | law K (a continuous outcome with effect 1 + W1 + W1 W3, a main-terms outcome regression that is wrong for it, and the treatment mechanism known), n = 2,000: the AUTOC: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with the main-terms outcome regression and the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9345 to 0.9501, SE ratio 0.9488 to 0.9955 | pass |
| `interval_calibration` | `K_qini__known_mechanism` | positive | law K (a continuous outcome with effect 1 + W1 + W1 W3, a main-terms outcome regression that is wrong for it, and the treatment mechanism known), n = 2,000: the Qini coefficient: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with the main-terms outcome regression and the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9414 to 0.9562, SE ratio 0.9703 to 1.0185 | pass |
| `interval_calibration` | `K_toc_0.5__known_mechanism` | positive | law K (a continuous outcome with effect 1 + W1 + W1 W3, a main-terms outcome regression that is wrong for it, and the treatment mechanism known), n = 2,000: the TOC at q = 0.5: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with the main-terms outcome regression and the known treatment mechanism | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9571, SE ratio 0.9729 to 1.0208 | pass |
| `interval_calibration` | `K_untargeted_autoc__shrunken_se_control` | control | law K (a continuous outcome with effect 1 + W1 + W1 W3, a main-terms outcome regression that is wrong for it, and the treatment mechanism known), n = 2,000: the AUTOC: the same TMLE estimate with the SE of draws of the untargeted one-step scores | the SE-ratio interval must lie below the calibration band | coverage 0.8709 to 0.8925, SE ratio 0.7739 to 0.8121 | pass |
| `interval_calibration` | `N_autoc__correctly_specified` | positive | law N (a continuous outcome with a constant effect, so every TOC is 0), n = 2,000: the AUTOC: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9379 to 0.9531, SE ratio 0.9570 to 1.0029 | pass |
| `interval_calibration` | `N_qini__correctly_specified` | positive | law N (a continuous outcome with a constant effect, so every TOC is 0), n = 2,000: the Qini coefficient: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9421 to 0.9568, SE ratio 0.9770 to 1.0239 | pass |
| `interval_calibration` | `S_autoc__correctly_specified` | positive | law H with the score refit on 2,000 training rows as a B-spline effect curve in (W1, W2, W3) and evaluated on 2,000 other rows (route 2): the AUTOC: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9379 to 0.9531, SE ratio 0.9736 to 1.0205 | pass |
| `interval_calibration` | `S_qini__correctly_specified` | positive | law H with the score refit on 2,000 training rows as a B-spline effect curve in (W1, W2, W3) and evaluated on 2,000 other rows (route 2): the Qini coefficient: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9391 to 0.9542, SE ratio 0.9698 to 1.0157 | pass |
| `interval_calibration` | `S_toc_0.1__correctly_specified` | positive | law H with the score refit on 2,000 training rows as a B-spline effect curve in (W1, W2, W3) and evaluated on 2,000 other rows (route 2): the TOC at q = 0.1: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9585, SE ratio 1.0042 to 1.0523 | pass |
| `interval_calibration` | `S_toc_0.5__correctly_specified` | positive | law H with the score refit on 2,000 training rows as a B-spline effect curve in (W1, W2, W3) and evaluated on 2,000 other rows (route 2): the TOC at q = 0.5: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9587, SE ratio 0.9797 to 1.0265 | pass |
| `interval_calibration` | `S_toc_0.9__correctly_specified` | positive | law H with the score refit on 2,000 training rows as a B-spline effect curve in (W1, W2, W3) and evaluated on 2,000 other rows (route 2): the TOC at q = 0.9: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9466 to 0.9607, SE ratio 0.9891 to 1.0358 | pass |
| `interval_calibration` | `U_autoc__correctly_specified` | positive | law U (a binary outcome whose effect does not vary with the external score S = W3, so every TOC is 0), n = 2,000: the AUTOC: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9585, SE ratio 0.9899 to 1.0366 | pass |
| `interval_calibration` | `U_qini__correctly_specified` | positive | law U (a binary outcome whose effect does not vary with the external score S = W3, so every TOC is 0), n = 2,000: the Qini coefficient: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9478 to 0.9618, SE ratio 0.9915 to 1.0383 | pass |
| `interval_calibration` | `U_toc_0.1__correctly_specified` | positive | law U (a binary outcome whose effect does not vary with the external score S = W3, so every TOC is 0), n = 2,000: the TOC at q = 0.1: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9455 to 0.9598, SE ratio 1.0013 to 1.0493 | pass |
| `interval_calibration` | `U_toc_0.5__correctly_specified` | positive | law U (a binary outcome whose effect does not vary with the external score S = W3, so every TOC is 0), n = 2,000: the TOC at q = 0.5: the curve's Wald interval, estimate +- 1.96 times the SD of its 2,000 half-sample draws, with a spline GLM per arm and a logistic treatment model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9466 to 0.9607, SE ratio 0.9983 to 1.0460 | pass |
| `power` | `P_band__slope` | positive | law P (a continuous outcome with effect 1 + 0.46 W1 and noise SD 1), n = 1,000: the sup-t band of the TOC over the nine grid points 0.1 to 0.9: whether the band excludes 0 at some grid point, at the slope where its power is 0.90 | the rejection rate's lower end at least 0.80 | rejection 0.8738, 0.8406 to 0.9023 | pass |
| `simultaneous_coverage` | `H_band__correctly_specified` | positive | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the sup-t band of the TOC over the nine grid points 0.1 to 0.9: whether the band covers the TOC at every grid point at once | the joint coverage interval inside the calibration coverage band 0.92 to 0.98 | joint coverage 0.9482 to 0.9621 | pass |
| `simultaneous_coverage` | `H_band__pointwise_joint` | control | law H (a continuous outcome with effect 1 + W1 and the external score S = W1 + 0.5 W3), n = 2,000: the sup-t band of the TOC over the nine grid points 0.1 to 0.9: whether the pointwise 95% intervals cover the TOC at every grid point at once | the joint coverage interval must lie below 0.95 | joint coverage 0.7472 to 0.7757 | pass |
| `type_i_error` | `N_band__null` | positive | law N (a continuous outcome with a constant effect, so every TOC is 0), n = 2,000: the sup-t band of the TOC over the nine grid points 0.1 to 0.9: whether the band excludes 0 at some grid point | the 99% upper end of the rejection rate at most 0.10 | rejection 0.0423, 0.0359 to 0.0495 | pass |
| `type_i_error` | `U_band__null` | positive | law U (a binary outcome whose effect does not vary with the external score S = W3, so every TOC is 0), n = 2,000: the sup-t band of the TOC over the nine grid points 0.1 to 0.9: whether the band excludes 0 at some grid point | the 99% upper end of the rejection rate at most 0.10 | rejection 0.0407, 0.0344 to 0.0477 | pass |
<!-- /generated -->

## Result

Every primary test passed, and so did every gated property cell. No fit failed. The 33 gated
calibration positives have SE-ratio intervals inside 0.93 to 1.07 and coverage intervals inside
0.92 to 0.98, on both outcome types, under a known mechanism with a wrong outcome regression, on
an atomic score and on a declared split. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 5 of 5 | `performance-tests.csv` |
| property cells passed | 47 of 48 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

The red cell is the reporting row, `H_S_minus_S_B_toc_0.1__correctly_specified`. Its SE ratio
is 1.060 (99% interval 1.035 to 1.086) and its coverage 0.963 (0.956 to 0.969): the interval is
conservative, as the oracle computation predicted, and both lower ends clear the one-sided gate
of 0.93 and 0.92. The other three paired rows pass the band. The
[red-cell ledger](red-cells.md) lists the row under `X92-paired-conservative`.

The band covers the nine grid truths at once in 0.956 of the replications, where the pointwise
intervals read jointly cover in 0.762. It rejects in 0.041 and 0.042 of the null replications of
`U` and `N`, and in 0.874 at the declared slope of `P`. The draws that keep the full-sample cut
read SE ratios of 1.245 and 1.719, the fixed rank weights 2.747, the unpaired draws 1.787 and
2.447, and the untargeted one-step draws 0.793.

The status rule was declared before the run. Every gated verdict passed, so a curve whose
configuration one of the measured cells covers reads `supported`. Configurations outside them
read `experimental`, as Limitations gives.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 6000 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 5 | parameter tests against the truth |
| `independent_tests_passed` | 5 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 48 | repeated-sampling property cells |
| `property_cells_passed` | 47 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0328 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9490 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9412 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9760 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0518 | highest bootstrap primary SE-ratio endpoint |
| `properties[interval_calibration/H_S_minus_S_B_toc_0.1__correctly_specified]:se_ratio` | 1.0602 | reporting row, SE ratio |
| `properties[interval_calibration/H_S_minus_S_B_toc_0.1__correctly_specified]:se_ratio_ci_lower` | 1.0350 | reporting row, SE-ratio lower end, against 0.93 |
| `properties[interval_calibration/H_S_minus_S_B_toc_0.1__correctly_specified]:coverage` | 0.9627 | reporting row, coverage |
| `properties[interval_calibration/H_S_minus_S_B_toc_0.1__correctly_specified]:coverage_ci_lower` | 0.9559 | reporting row, coverage lower end, against 0.92 |
| `properties[interval_calibration/K_autoc__known_mechanism]:se_ratio_ci_lower` | 0.9488 | lowest gated calibration SE-ratio endpoint |
| `properties[interval_calibration/H_toc_0.9__correctly_specified]:se_ratio_ci_upper` | 1.0551 | highest gated calibration SE-ratio endpoint |
| `properties[simultaneous_coverage/H_band__correctly_specified]:coverage` | 0.9555 | band joint coverage |
| `properties[simultaneous_coverage/H_band__pointwise_joint]:coverage` | 0.7617 | pointwise intervals read jointly |
| `properties[type_i_error/U_band__null]:rejection_rate` | 0.0407 | band size, binary null |
| `properties[type_i_error/N_band__null]:rejection_rate` | 0.0423 | band size, constant effect |
| `properties[power/P_band__slope]:rejection_rate` | 0.8738 | band power |
| `properties[interval_calibration/K_untargeted_autoc__shrunken_se_control]:se_ratio` | 0.7931 | untargeted one-step draws |
| `properties[interval_calibration/H_fixed_ranks_autoc__inflated_se_control]:se_ratio` | 2.7465 | fixed rank weights |
| `properties[bias/D_toc_0.2__whole_block]:standardized_bias` | -0.9032 | whole-block tie rule |
| `properties[bias/S_toc_0.1__in_sample]:standardized_bias` | 0.3800 | in-sample curve, TOC(0.1) |
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
| `margin:subject_failures_allowed` | 60 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | The `grf` comparison is pending. The row rests on accuracy against the truth and on the theory properties |
| The measured configurations | An external score on a continuous outcome (`H`, `N`, `P`), on a binary outcome (`B`, `U`, `D`), with a known mechanism on a continuous outcome (`K`), and a declared split on a continuous outcome (`S`), each with a binary treatment, iid unweighted rows, complete outcomes, one split and cross-fitted nuisances. A split on a binary outcome, a known mechanism on a binary outcome or a split, weights, clusters, missing outcomes, a pair of arms of a multi-arm treatment and an ATE fit without cross-fitting read `experimental` |
| The paired difference | Cell `C` measures it on law `H` only. A paired difference of any other configuration reads `experimental`. Its TOC at 0.1 is conservative at n = 2,000 (SE ratio 1.060): the interval is wider than it needs to be, and its coverage is 0.963 |
| Rows in the top fraction | Every calibrated TOC has at least 200 rows in its top fraction (TOC at 0.1 with n = 2,000). A grid point with fewer reads `experimental`, and one with fewer than 50 withholds its interval and the band. Cell `P`'s 100 top rows at n = 1,000 measure the band's power, not an interval's coverage |
| Laws, sizes and learners | Every cell is a smooth law on $[-1, 1]^3$ with treatment probabilities between about 0.22 and 0.78, at n = 2,000, with spline GLM learners (`K` aside). The status does not read the learner, the number of folds or $n$ beyond the top-fraction rule, so a fit on another law, size or learner within a measured configuration reads `supported` and rests on the theory |
| Ties | Cell `D` calibrates cuts strictly inside a tied block, but a cut inside or at the edge of a sample tied block still reads `experimental`: the data cannot tell an interior cut from an edge, where the limit is not normal. Areas on a continuous score with atoms read `experimental` |
| Grid and quantities | The TOC is calibrated at 0.1, 0.2, 0.5 and 0.9, the group effects at the top and bottom fifths on `H`, and the band over the default grid. The other grid points and the cumulative gain are measured only through the band and the shared draws |

## Reproduction

The declared run is `python -m tests.canonical.targeting_curves.regenerate`, whose docstring
gives the run form and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/targeting_curves/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/targeting_curves/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/targeting_curves/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/targeting_curves/properties.csv)
carry every published row.
