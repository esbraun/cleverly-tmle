# Kernel-smoothed dose-response CV-TMLE against npcausal ctseff

This study compares `DoseResponseTMLE` with R `npcausal` `ctseff` (Kennedy, Ma, McHugh and Small
2017) at commit `56a5ac1` on identical samples. The two estimate different smoothed curves in
general. `DoseResponse` smooths $\theta(a) = E\{\bar Q(a, W)\}$ with the local-linear equivalent
kernel on Lebesgue measure over $D = [-1.5, 6.5]$, and `ctseff` regresses a doubly robust
pseudo-outcome on the dose, which smooths $\theta$ against the marginal dose density. On this
study's law $\theta$ is a line, and every local-linear smoother reproduces a line, so both targets
equal $\theta(a_0)$ at every declared dose and bandwidth. The comparison is of target recovery on
that shared value, not of the two smoothing measures, and its kind is "related target".

The declared run's artifacts are in `tests/canonical/dose_response_npcausal/`, and the study is
registered with them. A declared rule read its bandwidths and
its replication count off two pilots, the second of which fitted real `ctseff` on 1,000 draws.
[Results of the declared run](#results-of-the-declared-run) reads the artifacts, including three
red paired rows, one of them outside the pre-registered red set.
[Smoothed dose-response curve](../dose-response.md) gives the estimand and the estimator. The [dose-response study](dose-response.md) holds the
estimator's property evidence.

## What was compared

| setting | declaration |
| --- | --- |
| law | $W \sim N(0, I_3)$; $A = 2.5 + 0.5 W_1 + 0.3 W_2 - 0.3 W_3 + 1.5\varepsilon$; $Y = 0.2 + 0.5 A + (0.8 + 0.1 A)(0.7 W_1 - 0.5 W_3) + \eta$, with standard normal $\varepsilon$ and $\eta$. $W_1$ confounds, and $\theta(a) = 0.2 + 0.5a$ |
| pairs | doses 1.5, 2.5 and 3.5 at two consecutive triweight bandwidths of the candidates 1.0, 1.5, 2.0 and 2.5, which the R pilot selects as 2.0 and 2.5: six pairs, each kernel's support inside $D$ |
| sample | 4,800 replications of n = 4,000, each fitted by both implementations |
| `cleverly` | `DoseResponseTMLE` with the triweight kernel, five folds, the linear fluctuation and `q_bounds=(-25, 25)` |
| `ctseff` | one call per bandwidth, `bw.seq` set to the matched Gaussian bandwidth, so its leave-one-out selection has one candidate. It still computes that candidate's risk |
| matched kernels | `ctseff` receives $h_G = h\,R(K_G)/R(K_T)$, with $R(K) = \int K^2$: 0.3458, 0.5187, 0.6915 and 0.8644 for $h$ = 1, 1.5, 2 and 2.5. At an interior dose both estimators then have the leading variance $R(K)\,E_W\{\sigma^2 / g_0(a_0 \mid W)\}/(nh)$, within 10% of the subject's exact term at $h \le 1.5$ and within 25% at $h = 2.5$ |
| outcome regression | main terms and pairwise products of $(a, W)$ in both: `SL.glm.interaction` in R, `PolynomialFeatures(2, interaction_only=True)` with `LinearRegression` here. The design holds $\bar Q_0$ exactly |
| dose density | `ctseff` fits the dose mean and variance with `SL.glm.interaction` and smooths the standardized residuals with a kernel density estimate. `cleverly` uses the pooled-hazard binned density of the dose-response study |
| evaluation grid | `ctseff` reports its curve on `seq(a.rng[1], a.rng[2], length.out = n.pts)`. The runner widens the observed dose range to multiples of 1/8, so every declared dose is a grid point exactly and the grid spans the data |
| inference | Wald intervals: $z_{0.975}$ times the influence-curve SE for `cleverly`, and `ctseff`'s own `est` $\pm$ 1.96 `se`$/\sqrt n$ |

A single-learner `SuperLearner` gives that learner weight 1, so each R prediction is the learner's
full-sample fit and no R fit depends on the fold draw. $D$ is symmetric about 2.5, so the
reflection $a \mapsto 5 - a$ maps it and the doses to themselves, and the chance that a declared
sample has no dose beyond either edge of $D$, which `DoseResponseTMLE` would refuse, is below
$10^{-6}$.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 1.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000363 to 0.0017 | 0.9625 | 1.0449 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 1.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | R `npcausal` | -0.000636 to 0.0014 | 0.9564 | 1.0240 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 2.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000730 to 0.0012 | 0.9563 | 1.0372 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 2.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | R `npcausal` | -0.000848 to 0.0011 | 0.9526 | 1.0055 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 3.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0015 to 0.000782 | 0.9627 | 1.0679 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 3.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | R `npcausal` | -0.0016 to 0.000577 | 0.9558 | 1.0261 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 1.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000314 to 0.0016 | 0.9604 | 1.0382 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 1.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | R `npcausal` | -0.000570 to 0.0014 | 0.9556 | 1.0283 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 2.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.000720 to 0.0011 | 0.9569 | 1.0397 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 2.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | R `npcausal` | -0.000840 to 0.0010 | 0.9535 | 1.0134 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 3.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | `cleverly` DoseResponseTMLE: CV-TMLE of the kernel-smoothed dose-response curve, one pooled fluctuation per (bandwidth, dose) pair | -0.0013 to 0.000902 | 0.9619 | 1.0550 | pass |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 3.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | R `npcausal` | -0.0015 to 0.000570 | 0.9595 | 1.0370 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
Comparator-backed: a paired comparison with `npcausal` on the same samples. Each row reports estimate equivalence within the declared margin, the RMSE ratio and the coverage difference, and names its comparison kind.

| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | comparison kind | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 1.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | 0.000269 | 0.0646 | 1.0020 | 0.0060 | related target | 0.0084 vs 0.0500 | superior |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 2.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | 0.000094 | 0.0241 | 0.9870 | 0.0035 | related target | 0.0079 vs 0.0500 | **unaccounted** |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 2, which equals the curve at 3.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | 0.000174 | 0.0384 | 1.0392 | 0.0069 | related target | 0.0141 vs 0.0500 | **inconclusive** |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=1.5]` | the counterfactual outcome mean averaged over doses near 1.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 1.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | 0.000251 | 0.0637 | 0.9976 | 0.0048 | related target | 0.0095 vs 0.0500 | equivalent |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=2.5]` | the counterfactual outcome mean averaged over doses near 2.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 2.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | 0.000094 | 0.0255 | 0.9846 | 0.0033 | related target | 0.0087 vs 0.0500 | equivalent |
| a dose with a normal location model linear in three normal covariates, and a Gaussian outcome linear in the dose with a dose-varying covariate slope, so the curve is a line; doses 1.5, 2.5 and 3.5 by triweight bandwidths 2.0 and 2.5, which the R pilot selected from 1.0, 1.5, 2.0 and 2.5, n = 4,000 | `dose_response[h=2.5,a=3.5]` | the counterfactual outcome mean averaged over doses near 3.5 by the local-linear kernel weight of bandwidth 2.5, which equals the curve at 3.5 because the curve is a line; npcausal ctseff estimates the same value with a Gaussian kernel of matched roughness | 0.000272 | 0.0626 | 1.0710 | 0.0023 | related target | 0.0213 vs 0.0500 | **unaccounted** |
<!-- /generated -->

## Results of the declared run

All 12 truth rows pass. `cleverly` covers at 0.956 to 0.963 with SE ratios 1.037 to 1.068, and
`ctseff` covers at 0.953 to 0.960 with SE ratios 1.005 to 1.037. The mean paired differences are
$0.9 \times 10^{-4}$ to $2.7 \times 10^{-4}$, 2% to 6% of the similarity margins, and the RMSE ratios
are 0.976 to 1.038 with 99% upper bounds at most 1.071. The pair at $h = 2$, $a_0 = 1.5$ concludes
superior on coverage, and the pairs at $h = 2.5$, $a_0 = 1.5$ and $a_0 = 2.5$ conclude equivalent.
The RMSE leg at $h = 2.5$, $a_0 = 3.5$, which the pilot projected red at any count, passes.

### Failed replications

`ctseff` failed 6 of 4,800 replications, inside the declared bound of 48, and `cleverly` failed
none. Five failures are `locpoly`'s gridsize error at $h = 2$. The Python transcription fails four
of those calls and counts the fifth, replication 1709, as a knife-edge call, as it did in the
pilot, and it fits all five at $h = 2.5$. The runner fits each replication as one harness group,
so an error at $h = 2$ also removes that replication's $h = 2.5$ rows, and every paired row reads
4,794 replications.

The sixth failure, replication 2540, is a type the pilot never met: `density.default(a.std[1:n])`
stops with "'x' contains missing values". `ctseff` fits the dose variance by regressing the squared
dose residuals on the interaction design with `SL.glm.interaction`, an unconstrained least-squares
fit, and standardizes the dose by the square root of the fitted value. In replication 2540 one row,
with $W = (4.17, -2.38, -0.55)$, gets a fitted variance of $-0.0058$, so its standardized dose is
`NaN`. The transcription rebuilds that input without R and refuses at the same point, "the fitted
dose variance is not positive". The failure comes from `ctseff`'s variance model on this law and
learner, not from its pseudo-outcome or the runner, and it fell inside `study_stage`, so it costs
one replication.

### Why three rows are red

| paired row | complete cases | failure-accounting worst case | conclusion | reading |
| --- | --- | --- | --- | --- |
| $h = 2$, $a_0 = 3.5$ | calibration excess upper bound 0.0558, above 0.05; the other legs pass | not reached | inconclusive | pre-registered: `cleverly`'s mean SE is 1.10 efficient SDs, its SE ratio 1.068 and its coverage 0.963 |
| $h = 2$, $a_0 = 2.5$ | every equivalence leg passes; calibration upper bound 0.0391 | calibration slack 0.0111 falls to $-0.0004$ | unaccounted | post-run diagnosis, outside the pre-registered set |
| $h = 2.5$, $a_0 = 3.5$ | every equivalence leg passes; calibration upper bound 0.0393 | calibration slack 0.0107 falls to $-0.0024$ | unaccounted | post-run diagnosis; the pre-registered dose, not the pre-registered signature |

The failure accounting fills the six replications `ctseff` lost with its declared unfavourable
candidates and withdraws a conclusion that does not survive them. For the calibration excess
$|r_s - 1| - |r_r - 1|$, with $r$ an implementation's mean SE over its empirical SD, the worst
candidate puts `ctseff`'s six missing SEs at the surviving minimum and its six missing estimates
at the surviving minimum, about four empirical SDs below the mean. That raises `ctseff`'s spread
and lowers its mean SE. At $h = 2$, $a_0 = 2.5$ its SE ratio moves from 1.0055 to 0.996 and
crosses 1, so its own error no longer offsets `cleverly`'s 3.7% over-report.

Each imputed failure moves `ctseff`'s SE ratio by 0.0011 to 0.0017. One failure alone lowers the
calibration slack by 0.0026 at $h = 2$, $a_0 = 2.5$ and by 0.0030 at $h = 2.5$, $a_0 = 3.5$, and
the six together lower it by 0.011 to 0.013 in all six cells. Of the five rows that pass on their complete cases,
the two unaccounted ones are those whose complete-case slack, 0.0111 and 0.0107, was smaller than
their shift; the next smallest, 0.0153 at $h = 2.5$, $a_0 = 2.5$, survives with 0.0027. In both, `cleverly`'s truth row is green, its SE
ratio is 1.037 and 1.055, and its mean SE sits at the level the pilot read at these pairs. The
reds do not locate a defect in `DoseResponseTMLE`.

### What the declaration missed

The paired-leg sizing read the pilot's complete cases and priced no failure-accounting imputation.
The R pilot lost 1 of 1,000 draws at the selected pair, so about 5 losses were expected at 4,800
and the run met 6. The shift is not a sampling error that more replications remove: the losses
grow with the count and the extreme survivors move outward, so the shift holds roughly level. Taking
0.012 off the calibration margin, the pilot's projected calibration pass chance at $h = 2$,
$a_0 = 2.5$ falls from 0.99 to 0.89, and at $h = 2.5$, $a_0 = 2.5$ from 0.96 to 0.65. At
$h = 2.5$, $a_0 = 3.5$ it stays at 0.995, because the pilot read an excess of $-0.030$ there
against 0.018 in the run, so the rule would not have flagged that row.

The record is `reporting`, so the three red rows publish as red. The design test admits the
inconclusive row under its pre-registered signature, and admits the two unaccounted rows under a
signature labelled post-run that names those rows only. That signature requires every
complete-case leg to pass, the published conclusion to be unaccounted, the accounting gate to read
"worst case fails" with the equivalence route green on its complete cases, at least one reference
failure within the bound, and `cleverly`'s truth row green. A mutation control breaks each clause
alone and confirms the row is refused. No margin, threshold, count or declared value changed, and
no replication was redrawn.

## Theory properties

None declared: 0/0 cells. The row rests on the comparison and the truth tests, and the
[dose-response study](dose-response.md#theory-properties) holds the estimator's property cells.

## Bandwidth selection

`ctseff` computes its leave-one-out bandwidth risk even when `bw.seq` holds one value. Its
binned local-linear fit (`KernSmooth::locpoly`) cuts the Gaussian kernel at four bandwidths, so
when the lowest or highest dose lies more than $4h_G$ from the next one, the end node's 2x2 system
has one bin or none and its solution is `NaN`. `approx` then returns `NA` at that dose, the risk
is `NA`, `which.min` selects nothing, and the call errors. With a normal dose at n = 4,000 the gap
between the two most extreme doses has a mean near 0.4, so the failure rate falls quickly as the
bandwidth grows.

The candidates 1.0, 1.5, 2.0 and 2.5 were declared before any R output: the two bandwidths first
declared and two wider fallbacks, with $D$ widened from $[0, 5]$ so that every candidate's kernel
lies inside it. The rule takes consecutive pairs in that order and selects the first whose pilot
failures, on the draws where either `ctseff` call failed, keep the chance of exceeding the
declared 1% failure bound at or below 0.01 at the declared count, computed at the failure rate's
one-sided 95% Clopper-Pearson upper bound.

On the 1,000 pilot draws `ctseff` failed 77, 9, 1 and 0 calls at the four candidates, each with
`locpoly`'s error "'bandwidth' must be a scalar or an array of length 'gridsize'", so the rule
selects 2.0 and 2.5. One draw, 215, is lost at bandwidth 2.0, and the chance of exceeding the
failure bound at 4,800 replications is about $10^{-6}$.

The Python transcription reproduces the failure path and fails 74, 7, 1 and 0 of the same calls.
At the highest dose the outcome can hinge on one bit: `locpoly` drops that dose when its bin
index rounds to exactly 400 and keeps it otherwise. When the next dose lies outside the last
node's window, a dropped dose makes the call fail and a kept one does not. The local probe met
one such call, pilot draw 11 at bandwidth 1, which R failed and the transcription fitted. The
transcription counts every such call as failed, which makes 77, 9, 1 and 0, and these are
exactly the calls R failed.

## Sizing

The count is the largest of three readings, each a multiple of 400:

| reading | what it reads | replications |
| --- | --- | --- |
| A | the shared primary rule (`tests/studies/repeated_sizing.py`) at the X32b clause shares, with `cleverly`'s pilot rows for both implementations | 2,400 |
| B | the same rule with each implementation's own pilot rows | 2,400 |
| paired legs | the smallest count at which the similarity, RMSE, coverage and calibration legs of every cell pass together with probability 0.95, by a normal approximation from the pilot's bootstrap spread. The RMSE and calibration legs at dose 3.5 are left out, because the pre-registered red signature admits them | 4,800 |

The paired-leg rule is capped at 8,000, about 83 R CPU-hours at the R pilot's cost. Past the cap a declared straddle
reading governs a calibration red at doses 1.5 and 2.5 instead: the subject's SE ratio in
$[1, 1.07]$, the reference's below 1, both truth rows green. On the pilot, `cleverly` took 2.3
seconds per fit on one core at twelve pairs.

The readings use the R pilot's `ctseff` rows, and the transcription in their place gives the
same three counts. Its binned `locpoly` fit is transcribed from the Fortran, and it differs from
`ctseff` in three declared places. Its
smoothing spline places a knot at every grid point, where R places about 60. The outcome spline
smooths a line, which both reproduce, so only the marginal-density spline can differ. Its
residual density is an exact Gaussian kernel sum, where R bins and convolves, and its
one-learner `SuperLearner` is the least-squares fit of the same design.

## Projected verdicts and red-cell route

At dose 3.5 `cleverly`'s mean reported SE is 1.10 and 1.11 times the exact efficient SD at
bandwidths 2.0 and 2.5, against 1.01 to 1.04 at doses 1.5 and 2.5. The reflected pilot fits move
the law with $A \mapsto 5 - A$, and the excess stays at dose 3.5, so it comes from the
pooled-hazard density, whose bin hazards compound upward through the survival product, and not
from the law or from `ctseff`. The [dose-response study](dose-response.md) sees the same effect at
its upper boundary dose 4.75; this law's wider dose spread brings it to an interior dose. At
bandwidth 2.5 the subject's spread rises to 1.11 times efficient as well, so its SE is calibrated
there, the RMSE ratio against `ctseff` is 1.10, and the RMSE leg projects red at any
count. At bandwidth 2.0 the RMSE ratio is 1.05. The four cells at doses 1.5 and 2.5 pass every
leg with projected probability at least 0.96 at 4,800 replications.

The record is `reporting`, declared before any run, and the design test admits only these
signatures in the run's `equivalence.csv`. Owner rows follow the run.

| red paired row | reading |
| --- | --- |
| dose 3.5, failing only the RMSE leg, the calibration leg, or both, with the subject's mean SE above 1.05 efficient SDs, its coverage at least 0.95 and its truth row green | the pre-registered upper-dose excess of the pooled-hazard density |
| dose 1.5 or 2.5, failing only the calibration leg, when the paired-leg rule hit its cap | the straddle reading above |
| anything else | not admitted. The design test fails, and the row is diagnosed before publication |

The run produced reds of the third kind at $h = 2$, $a_0 = 2.5$ and $h = 2.5$, $a_0 = 3.5$, which
[Why three rows are red](#why-three-rows-are-red) diagnoses.

A red `cleverly` truth row is a defect suspect: the truths are exact, the outcome design holds
$\bar Q_0$ and the doses are interior. A red `npcausal` truth row is a finding about the
reference at this law. It does not turn the subject's row red.

## Local R pilot

The R pilot ran on a machine with Docker. It builds the image of the incremental studies from
`tests/canonical/npcausal_incremental/Dockerfile` on first use and fits through the declared
runner `run_study.R`. Each `ctseff` call is its own harness group, so a failed call is one
failure record with R's message. The probe comes first and the full pilot second:

```console
python -m tests.canonical.dose_response_npcausal.pilot probe --scratch <dir> --jobs <n>
python -m tests.canonical.dose_response_npcausal.pilot reference --scratch <dir> --jobs <n>
```

The probe checks the image, confirms or refutes the failure path on constructed draws with an
isolated extreme dose, compares R with the transcription on 20 pilot draws, reruns two draws for
identical rows, and times a call. Its comparison requires estimates within 0.05 SE, standard
errors within 1%, and the same failed calls. Calls whose outcome hinges on the last bit of the
highest dose's bin are reported and not gated, since only R settles them. The full pilot
works in resumable chunks of 50 draws. It committed
`tests/studies/dose_response_npcausal_pilot_r.csv.gz`, `..._pilot_r_failures.csv` and
`..._pilot_r.json`, measured 18.65 CPU-seconds per call over its 4,000 calls (about 21
CPU-hours), and the design test requires the declared bandwidths and count to equal the rule's
reading of those rows. At that cost the declared reference of 9,600 calls takes about 50
CPU-hours, beside about 3 for the `cleverly` fits.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. The other rows are declared
counts. The run adds the measured rows and the failure bounds it applied.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4800 | paired replications, read off the R pilot |
| `n` | 4000 | observations per replication |
| `independent_tests_total` | 12 | truth tests reported, six pairs for each implementation |
| `paired_tests_total` | 6 | paired comparisons reported |
| `property_cells_total` | 0 | property cells. The [dose-response study](dose-response.md#theory-properties) holds them |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:subject_failure_bound` | 0.0100 | share of `cleverly` fits a cell may lose |
| `margin:subject_failures_allowed` | 48 | `cleverly` fits a cell may lose |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:reference_failure_bound` | 0.0100 | share of `ctseff` fits a cell may lose |
| `margin:reference_failures_allowed` | 48 | `ctseff` fits a cell may lose |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound. No cell of this study reads it |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |

## Limits

| limit | what it means for use |
| --- | --- |
| a linear curve only | the two targets coincide only where $\theta$ is linear on each kernel's support. On a curved $\theta$ they part: on $\sin(2a)$ by up to 0.75 efficient SEs at n = 4,000, which the design test checks. Agreement here says nothing about the smoothing measure |
| interior doses | every kernel's support lies inside $D$. Boundary pairs, where the Lebesgue equivalent kernel is asymmetric, are not compared |
| wide bandwidths | the selected bandwidths are the narrowest at which `ctseff`'s bandwidth risk does not fail on isolated extreme doses. Narrower bandwidths are not compared |
| the triweight against a Gaussian kernel | no shared kernel exists, so the bandwidths are matched by roughness. The Epanechnikov and uniform kernels are not compared |
| in-sample R nuisances | `ctseff` fits in sample and `cleverly` cross-fits over five folds |
| two density estimators | the dose densities come from different estimators, so the comparison is not of one targeting step on shared nuisances |
| a Gaussian outcome | binary outcomes and the logistic fluctuation are not compared |
| no band, strata, weights or clusters | only pointwise Wald intervals are compared |

## Reproduction

The smoke run is `python -m tests.canonical.dose_response_npcausal.regenerate --replicates 20
--output <empty directory outside the repository> --allow-failures --jobs <workers>`. The declared
run is `python -m tests.canonical.dose_response_npcausal.regenerate --output <empty directory>
--jobs <workers>`. Both build the unchanged image of the incremental studies from
`tests/canonical/npcausal_incremental/Dockerfile` and mount `run_study.R`. The declared run
needs a clean tree whose `HEAD` is pushed and every thread variable set to 1, and it copies its
artifacts, `manifest.json` and `run.log` into `tests/canonical/dose_response_npcausal/`. The
smoke draws declared replicates 0 to 19, so no declared value is read from it.
