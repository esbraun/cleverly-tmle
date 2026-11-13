# Kernel-smoothed dose-response CV-TMLE

This study validates `DoseResponseTMLE` with a `DoseResponse` declaration in
`cleverly.estimators`. The target at each declared bandwidth $h$ and dose $a_0$ is the smoothed
curve $\theta_h(a_0) = \int_D w_{h,a_0}(a)\,\theta(a)\,da$, with $w_{h,a_0}$ the local-linear
equivalent kernel on the declared interval $D$. Each pair's estimate is a cross-validated TMLE
with one pooled fluctuation, each interval is a Wald interval, and the band is the multiplier
max-$t$ band over every declared pair.
[Smoothed dose-response curve](../dose-response.md) gives the estimand, the estimator and its
conditions.

The study has no comparator, so this page publishes no paired verdict. npcausal `ctseff`
estimates a related target, and the separate study
[against npcausal `ctseff`](dose-response-npcausal.md) compares the two on a law where both
targets equal the curve. Three of its six paired rows are red under `reporting`, and that page
diagnoses them.

The declared run at `7f9d5172` passed every gate: 27 of 27 primary rows and 7 of 7 property
cells. An earlier run at `dfd56920` fitted the same seeds and wrote the same replicate rows. It
exited red only because the properties module marked two report-only targeting rows as gated,
an instrument defect fixed before the repeat
([`regenerate.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/dose_response/regenerate.py)
records the history). The numbers below that are not declarations or pilot readings are filled by
`python -m tests.studies.evidence.document --slug dose-response` from the committed artifacts.

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
| property budget | 8,000 fits of the 15-pair binary declaration, which give the band cells and the calibration cell, and 800 replications of each of the three robustness fits, all at n = 4,000 on the binary law |
| failed fits | at most 1% of the replications of any cell |
| seeds | `seed=2026119301` and `resampling_seed=2026119302`. The band fits draw from one labelled stream, the three robustness fits share a second, and the pilot draws from its own |
| policy | `gated`: any red verdict fails the run |

The laws, scenarios, learners, budgets, controls, seeds and red-cell readings were fixed before
the run, in `tests/studies/dose_response.py`, `tests/studies/dose_response_properties.py` and
`tests/canonical/dose_response/regenerate.py`. `tests/unit/test_dose_response_design.py` checks
the truths, the tail dose, the robustness limits, the band design numbers and the sizes.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=0.5,a=0.25]` | the counterfactual outcome mean averaged over doses near 0.25 by the local-linear kernel weight of bandwidth 0.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0018 to 0.0012 | 0.9260 | 0.9996 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=0.5,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 0.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000410 to 0.0011 | 0.9583 | 1.0513 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=0.5,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 0.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000654 to 0.000666 | 0.9497 | 1.0047 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=0.5,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 0.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000564 to 0.000947 | 0.9530 | 1.0205 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=0.5,a=4.75]` | the counterfactual outcome mean averaged over doses near 4.75 by the local-linear kernel weight of bandwidth 0.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0015 to 0.0012 | 0.9375 | 1.0237 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1,a=0.25]` | the counterfactual outcome mean averaged over doses near 0.25 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0018 to 0.0011 | 0.9280 | 1.0047 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000332 to 0.000765 | 0.9595 | 1.0457 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000498 to 0.000473 | 0.9490 | 1.0021 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000671 to 0.000438 | 0.9560 | 1.0136 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1,a=4.75]` | the counterfactual outcome mean averaged over doses near 4.75 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0014 to 0.0013 | 0.9385 | 1.0237 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1.5,a=0.25]` | the counterfactual outcome mean averaged over doses near 0.25 by the local-linear kernel weight of bandwidth 1.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0017 to 0.000926 | 0.9363 | 1.0154 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1.5,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 1.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000352 to 0.000574 | 0.9615 | 1.0451 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1.5,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 1.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000334 to 0.000492 | 0.9527 | 1.0058 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1.5,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 1.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000578 to 0.000375 | 0.9525 | 1.0018 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, 15 (bandwidth, dose) pairs, n = 4,000 | `dose_response[h=1.5,a=4.75]` | the counterfactual outcome mean averaged over doses near 4.75 by the local-linear kernel weight of bandwidth 1.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0015 to 0.0011 | 0.9415 | 1.0212 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, Gaussian outcome with the linear fluctuation, five doses at bandwidth 1.0, n = 4,000 | `dose_response[h=1,a=0.25]` | the counterfactual outcome mean averaged over doses near 0.25 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0044 to 0.0022 | 0.9403 | 1.0076 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, Gaussian outcome with the linear fluctuation, five doses at bandwidth 1.0, n = 4,000 | `dose_response[h=1,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000993 to 0.0016 | 0.9523 | 1.0316 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, Gaussian outcome with the linear fluctuation, five doses at bandwidth 1.0, n = 4,000 | `dose_response[h=1,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000923 to 0.0014 | 0.9563 | 1.0273 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, Gaussian outcome with the linear fluctuation, five doses at bandwidth 1.0, n = 4,000 | `dose_response[h=1,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0018 to 0.000878 | 0.9580 | 1.0274 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, Gaussian outcome with the linear fluctuation, five doses at bandwidth 1.0, n = 4,000 | `dose_response[h=1,a=4.75]` | the counterfactual outcome mean averaged over doses near 4.75 by the local-linear kernel weight of bandwidth 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0052 to 0.0017 | 0.9563 | 1.0794 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, the curve in each stratum of V = 1{W1 > 0} at three doses and bandwidth 1.0, n = 8,000 | `dose_response[h=1,a=0.25][V=0]` | the counterfactual outcome mean averaged over doses near 0.25 by the local-linear kernel weight of bandwidth 1, in stratum V = 0 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000766 to 0.0012 | 0.9560 | 1.0476 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, the curve in each stratum of V = 1{W1 > 0} at three doses and bandwidth 1.0, n = 8,000 | `dose_response[h=1,a=0.25][V=1]` | the counterfactual outcome mean averaged over doses near 0.25 by the local-linear kernel weight of bandwidth 1, in stratum V = 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0014 to 0.0022 | 0.9215 | 0.9346 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, the curve in each stratum of V = 1{W1 > 0} at three doses and bandwidth 1.0, n = 8,000 | `dose_response[h=1,a=2.5][V=0]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 1, in stratum V = 0 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000865 to 0.000120 | 0.9497 | 1.0033 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, the curve in each stratum of V = 1{W1 > 0} at three doses and bandwidth 1.0, n = 8,000 | `dose_response[h=1,a=2.5][V=1]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 1, in stratum V = 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000555 to 0.000328 | 0.9585 | 1.0379 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, the curve in each stratum of V = 1{W1 > 0} at three doses and bandwidth 1.0, n = 8,000 | `dose_response[h=1,a=4.75][V=0]` | the counterfactual outcome mean averaged over doses near 4.75 by the local-linear kernel weight of bandwidth 1, in stratum V = 0 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0022 to 0.000668 | 0.9483 | 1.0501 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, the curve in each stratum of V = 1{W1 > 0} at three doses and bandwidth 1.0, n = 8,000 | `dose_response[h=1,a=4.75][V=1]` | the counterfactual outcome mean averaged over doses near 4.75 by the local-linear kernel weight of bandwidth 1, in stratum V = 1 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0016 to 0.000689 | 0.9517 | 1.0301 | pass |
| a dose with a normal location model nonlinear in three normal covariates, and an outcome whose logit is nonlinear in the dose with a dose-varying covariate slope, binary outcome, bandwidth 0.5 at dose 0.4, where the expected Kish size is 62 just above the support floor of 50, n = 1,000 | `dose_response[h=0.5,a=0.4]` | the counterfactual outcome mean averaged over doses near 0.4 by the local-linear kernel weight of bandwidth 0.5 | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0013 to 0.0035 | 0.9320 | 1.0650 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

None claimed, so no canonical implementation is compared, and a zero-row equivalence artifact
records the absence. npcausal `ctseff` (Kennedy,
Ma, McHugh and Small 2017) smooths against the marginal dose density rather than the Lebesgue
kernel weight of `DoseResponse`, so a paired comparison needs a matched target. The
[comparator study](dose-response-npcausal.md) supplies one: a law where the curve is a line, so
both smoothed curves equal it, and its declared run is registered.

The owner's rule applies: a method merges on its Python evidence, and the comparator study
follows ([Theory and comparators](../../architecture-invariants.md#theory-and-comparators)).

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `simultaneous_coverage` | does the band over the 15 binary pairs cover every truth at once, with its 99% joint-coverage interval inside 0.92 to 0.98 | the pointwise intervals read jointly must cover below 0.95. Their exact joint coverage under the efficient influence correlation is 0.676 |
| `interval_calibration` | at $h = 1.0$, $a_0 = 2.5$, read from each band fit, does the SE-ratio interval lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98, with the empirical SD and the mean reported SE each within 0.90 to 1.10 of the exact efficient SD | the reported SE times 0.70, and one efficient SD of independent noise added to each estimate, must each leave the band |
| `double_robustness` | at $h = 1.0$, $a_0 = 3.5$, with a density that ignores $W$ and the declared outcome sieve, is the 99% bias interval inside $\pm 0.25$ SD. With the dose-only spline outcome and the declared density, does targeting bring the estimate at least 1.5 efficient SEs closer to the truth than its plug-in | with the dose-only spline outcome and the density that ignores $W$, the bias interval must lie wholly outside $\pm 0.25$ SD. An untargeted fit reads a gain of 0 |

The band fits seed their multiplier draws with `random_state=0`, so every replication reads the
same draws, and the band cell measures coverage conditional on that one draw. Since
band-finite-sample the band uses 10,000 Mammen draws on the identity scale, because the curve's
estimates carry no range. At the former 1,000 draws its critical value sat within 0.02 of the
oracle value 2.751, which moves the joint coverage by about 0.003.
The calibration cell reads the pair $(1.0, 2.5)$ of the same fits. Fitted alone on 20 shared
pilot draws, the pair gave the same estimate to within $10^{-6}$ of its SE.

The wrong density is the hazard classifier on the bin indicators alone: the binned marginal
density of the dose. In the population the positive cell's limit is 0.017 efficient SEs off the
truth at n = 4,000, and the control's is 3.03, the unadjusted kernel regression of $Y$ on $A$.
A main-terms logistic outcome on $[A, W]$ was the first control design. It was rejected before
any run, because its limit is 0.40 efficient SEs off the truth at this pair. The three
robustness fits share their draws.

The positive shows that the outcome regression carries the fit when the density is wrong. It is
not a witness of targeting: with the outcome right the fluctuation is near zero, so an
untargeted plug-in, a dropped $1/g$ or $g$ read at $a_0$ would each pass it.

The targeting witness is the fit with the dose-only outcome and the declared density. Its targeted estimate
and its plug-in are published as reported rows, and the family's joint clause gates the gain
between them. With $b_\text{plug-in}$ and $b_\text{targeted}$ their mean errors in efficient SEs
at n = 4,000, both $b_\text{plug-in} - b_\text{targeted}$ (`targeting_gain`) and
$b_\text{plug-in} + b_\text{targeted}$ (`targeting_gain_past_truth`) must reach 1.5. Together they
say $\lvert b_\text{targeted} \rvert \le b_\text{plug-in} - 1.5$: the targeted estimate sits at
least 1.5 SEs closer to the truth than its plug-in, on either side of it. The pilot reads 3.32 and 2.49 from
a plug-in bias of 2.90. A red gain turns the positive and the control red through their
`property_passed`.

The gain is gated rather than a bias interval around the truth because the declared density is
misspecified. Its hazard classifier is a fixed logistic model in the bin indicators, $W$, $W^2$
and their products with the bin index, and the true hazard of a normal location model with a
$\tanh$ mean is not in that model. The fitted density therefore converges to a projection, not
to $g_0$, and the outcome-wrong fit keeps a remainder that does not vanish: the pilot's targeted
bias is $-0.42$ efficient SEs. The exact remainder tests in `tests/unit/test_dose_response.py`
carry the population statement with the true density.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `h1_a3_5__both_wrong` | control | the binary dose-response law, the smoothed curve at dose 3.5 and bandwidth 1.0, n = 4,000: an outcome spline in the dose alone and the binned marginal density, so neither nuisance reads the confounder | bias interval must fall entirely outside the margin | bias 0.0379 to 0.0405, margin 0.0035, SE ratio 0.9832 | pass |
| `double_robustness` | `h1_a3_5__outcome_correct_density_wrong` | positive | the binary dose-response law, the smoothed curve at dose 3.5 and bandwidth 1.0, n = 4,000: the declared outcome sieve, with the dose density replaced by the binned marginal density, which ignores the covariates | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0012 to 0.0013, margin 0.0034, SE ratio 0.9538 | pass |
| `double_robustness` | `h1_a3_5__outcome_wrong_density_declared` | diagnostic | the binary dose-response law, the smoothed curve at dose 3.5 and bandwidth 1.0, n = 4,000: an outcome spline in the dose alone with the declared pooled-hazard density, on the draws of the other robustness cells: the targeted estimate | reported: its bias and the plug-in's give the targeting gain, which the family's joint clause requires to put the estimate at least 1.5 efficient SEs closer to the truth than the plug-in | bias -0.0058 to -0.0033, margin 0.0035, SE ratio 1.0522 | reported |
| `double_robustness` | `h1_a3_5__outcome_wrong_density_declared_plug_in` | diagnostic | the binary dose-response law, the smoothed curve at dose 3.5 and bandwidth 1.0, n = 4,000: the untargeted plug-in of the same fit, carrying the targeted fit's standard error | reported: the bias the targeting step has to remove | bias 0.0384 to 0.0409, margin 0.0033, SE ratio 1.1230 | reported |
| `interval_calibration` | `interior_h1_a2_5__correctly_specified` | positive | the binary dose-response law, the smoothed curve at dose 2.5 and bandwidth 1.0, an interior pair, n = 4,000: the declared outcome sieve and pooled-hazard density, read at this pair from each band fit of the 15-pair declaration with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, and the empirical SD and the mean reported SE each within 0.90 to 1.10 of the exact efficient SD, with both efficiency-ratio intervals inside their bands | coverage 0.9438 to 0.9564, SE ratio 0.9861 to 1.0272, empirical efficiency ratio 0.9962 to 1.0378, reported efficiency ratio 1.0230 to 1.0238 | pass |
| `interval_calibration` | `interior_h1_a2_5__noise_control` | control | the binary dose-response law, the smoothed curve at dose 2.5 and bandwidth 1.0, an interior pair, n = 4,000: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8299 to 0.8511, SE ratio 0.7068 to 0.7354, empirical efficiency ratio 1.3917 to 1.4479, reported efficiency ratio 1.0230 to 1.0238 | pass |
| `interval_calibration` | `interior_h1_a2_5__shrunken_se_control` | control | the binary dose-response law, the smoothed curve at dose 2.5 and bandwidth 1.0, an interior pair, n = 4,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8270 to 0.8483, SE ratio 0.6905 to 0.7185, empirical efficiency ratio 0.9971 to 1.0374, reported efficiency ratio 0.7161 to 0.7167 | pass |
| `simultaneous_coverage` | `binary_curve__pointwise_joint_control` | control | the binary dose-response law, the smoothed curve at doses 0.25, 1.5, 2.5, 3.5 and 4.75 and bandwidths 0.5, 1.0 and 1.5 (15 pairs), n = 4,000: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6658 to 0.6928 | pass |
| `simultaneous_coverage` | `binary_curve__simultaneous_band` | positive | the binary dose-response law, the smoothed curve at doses 0.25, 1.5, 2.5, 3.5 and 4.75 and bandwidths 0.5, 1.0 and 1.5 (15 pairs), n = 4,000: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9310 to 0.9450 | pass |
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
`repeated_sizing` rules. The pilot drew 1,000 samples of each primary scenario, 7,000 of the
calibration pair fitted alone, 300 of the positive and control robustness fits, 400 of the
targeting fit, and 20 samples fitted both ways to compare the calibration pair alone with the
15-pair fit. No pilot fit failed.

| sizing input | value | source |
| --- | --- | --- |
| primary replications | 4,000, from the shared primary rule on the 1,000 pilot draws of each scenario, one bias group per scenario | `tests/studies/dose_response.py` |
| lowest pilot coverage | 0.931, dose 0.25 in stratum $V = 1$ at n = 8,000 (Kish size 141). At n = 4,000 the same cell read 0.908 (Kish size 69, SE ratio 0.94), whose one-sided 80% lower bound lies below the 0.90 floor, so no budget sized it and the scenario moved to n = 8,000 before any run | `tests/studies/dose_response_pilot.csv.gz` |
| boundary pairs | the upper boundary dose 4.75 reads SE ratios 1.10 to 1.12 on the binary outcome, a conservative interval; the lower boundary dose 0.25 reads 0.98 to 0.99 and coverage 0.933 to 0.935 | the same |
| tail pair | coverage 0.935, SE ratio 1.06, mean Kish size 58. The 21% of fits below the floor, which carry the `low_dose_support_plugin` label, covered 0.938, and the rest 0.934. Coverage fell from 0.944 to 0.924 across the quartiles of the fitted Kish size | the same |
| band and calibration fits | 8,000, from the shared X32b calibration rule with the band as its joint-coverage term, at the whole calibration share. It reads the pooled SE ratio 1.023 of the calibration pair over 8,000 pilot fits, the pair's pilot bias, and the one-sided 80% lower bound, 0.935, of the pilot's joint band coverage of 942 in 1,000. The predictive failure probability is 0.0160 at 8,000 and 0.0205 at 7,600, against a budget of 0.0163 | `tests/studies/dose_response_properties.py` |
| calibration cost | reading the calibration cell from the band fits costs no fit. Separate single-pair fits, 3.75 s each, would add 8.3 CPU-hours and change no row | the same |
| band control | the pointwise intervals cover jointly with exact probability 0.676 under the efficient influence correlation (the pilot reads 0.698); the oracle critical value is 2.751 against the pilot's mean 2.733 | `tests/unit/test_dose_response_design.py` |
| efficiency ratios | the pooled pilot's empirical SD is 1.000 of the exact efficient SD and its mean reported SE 1.023 | the same |
| robustness replications | 800. The positive's pilot standardized bias is 0.010 (one-sided 80% upper bound 0.059); the control's is 2.87. The targeting gain's two means are 3.32 and 2.49 efficient SEs, against a floor of 1.5 | the same |

The reported standard error is conservative by a few percent at most interior pairs at
n = 4,000 (mean reported SE 1.02 to 1.06 times the exact efficient SD). The shared rule prices
the pilot noise of the calibration ratio for that reason, and the red-cell route below reads a
red on the upper SE-ratio end alone.

The declared run fits 26,400 samples. At the pilot's seconds per fit it costs about 52
CPU-hours: 33.7 for the primary scenarios (21.2 of them for `strata` at n = 8,000), 16.2 for
the 8,000 band fits and 1.7 for the 2,400 robustness fits.

## Red-cell route

| event | route, declared before any run |
| --- | --- |
| a red primary row of `binary`, `strata` or `gaussian` | a defect suspect of the quadrature, the fluctuation or the influence curve. A red boundary row (dose 0.25 or 4.75) with an equivalent bias and an SE-ratio interval below 1 reads as the variance limit of a boundary weight at n = 4,000, and moves to `reporting` with an owner row if no defect is found |
| a red `tail` row | a finding about the support label, not a reason to raise the floor: in the pilot, coverage did not rise with the fitted Kish size. It moves to `reporting` with the coverage split by the label (`pair-support.csv`) in the owner row, and roadmap X95 takes the relabelling |
| a red band cell below 0.92 while every `binary` row passes | the finite-sample limit of the band over 15 near-collinear curves, moved to `reporting` with the measured coverage. Above 0.98 it is a critical-value defect suspect |
| a control that does not leave its band | an instrument defect, and the run is void |
| a red calibration cell on the upper end of its SE ratio alone, with green coverage and efficiency ratios | a conservative reported SE at n = 4,000, moved to `reporting` with the measured ratio |
| a red calibration cell on the lower end of its SE ratio, on coverage, or on an efficiency ratio | an anti-conservative variance defect suspect |
| a red robustness positive | the outcome regression no longer carries the fit when the density is wrong: a defect suspect of the outcome-side fit or the influence curve |
| a red targeting gain | a defect suspect of the targeting step, the clever covariate or its inverse density |
| a replication that raises | is never redrawn, and the failure record governs. A recordable `DataError` (a zero fitted density inside a kernel's support, or a fluctuation that did not solve its score) costs its cell one replicate, read within the 1% bound by the failure accounting. Any other exception stops the run |
| any verdict | moves no budget, margin, law, learner, fold count or seed |

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4000 | primary replications |
| `n` | 4000 | observations per primary replication |
| `independent_tests_passed` | 27 | truth tests passing |
| `independent_tests_total` | 27 | truth tests reported |
| `property_cells_passed` | 7 | property cells passing |
| `property_cells_total` | 7 | property cells reported |
| `max_standardized_bias` | 0.0308 | largest primary standardized bias |
| `min_coverage` | 0.9215 | lowest primary coverage |
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
| `margin:subject_failures_allowed` | 40 | the same bound, in primary replications |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it, because it has no comparator |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it, because it has no comparator |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it, because it has no comparator |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it, because it has no comparator |

## Limits

| limit | what it means for use |
| --- | --- |
| no comparator | the record names no reference implementation, so the study has no paired verdict. The [comparator study](dose-response-npcausal.md) pairs a triweight fit with npcausal `ctseff` on a linear curve. Its declared run is registered, with three red paired rows under `reporting` |
| one law family | a normal location model for the dose, with an outcome that reads $W$ through one linear index, on $D = [0, 5]$ |
| declared sieve learners | the outcome sieve holds the $W$ part of the law exactly. The default learner library is not measured |
| one sample size per scenario | n = 4,000, n = 8,000 for the stratified scenario and n = 1,000 for the tail pair. Smaller sizes are not claimed |
| the band covers 15 declared pairs, with one multiplier draw | the band cell measures one 15-pair declaration on the binary outcome, conditional on the single multiplier draw that `random_state=0` fixes. A band with strata or on a continuous outcome reads `experimental` ([support label](../dose-response.md#support-label)); the label does not read the number or spacing of the pairs |
| the support label is not validated | the pilot's coverage did not rise with the fitted Kish size. The unlabelled stratified pair at dose 0.25 in $V = 1$ covered 0.908 at n = 4,000 with a mean Kish size of 69 and 2.9% of its fits labelled, so an interval above the floor of 50 can under-cover (roadmap X95). The `tail` cell measures one pair near the floor |
| a wrong outcome is measured as a targeting gain | the declared density is misspecified, so with a wrong outcome the study gates how far targeting moves the estimate, not a bias equivalence |
| no weights, clusters or missing outcomes | the study measures none of them |

## Reproduction

The declared run is `python -m tests.canonical.dose_response.regenerate --output <empty
directory> --jobs <workers>`, whose docstring gives the run form and the red-cell readings. It
starts no reference process, because the study has no comparator. Besides the shared artifacts
it publishes `pair-support.csv`, each primary estimand's coverage, SE ratio and bias split by
the `below_floor` label, which gates nothing.
