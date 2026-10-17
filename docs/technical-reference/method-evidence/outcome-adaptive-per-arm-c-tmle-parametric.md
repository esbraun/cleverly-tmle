# Outcome-adaptive per-arm C-TMLE (parametric)

This study measures the interval of `CTMLE(strategy="oat", oat_design="per_arm", cross_fit=False)`
with `ParametricOutcome` and `ParametricMechanism`. The fit is a stacked Z-estimator. Its outcome
GLM, each arm's mechanism, each arm's fluctuation and each arm's mean solve one system. The section
[Parametric per-arm intervals](../collaborative-tmle.md#parametric-per-arm-intervals) states the
construction, its conditions and what the interval covers.

**Python property evidence, no comparator.** No canonical implementation is compared. R
[`drtmle`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c)
with `adapt_g = TRUE` reports the shipped curve of Theorem 1 of Benkeser, Cai and van der Laan
(2020). The stack corrects that curve, so no R program measures the corrected uncertainty. The
point estimate agrees with the pipelines of the registered study
[`ctmle-oat-per-arm`](outcome-adaptive-per-arm-c-tmle.md) to `4e-14`
(`tests/unit/test_parametric_learners.py`), and that study pairs those pipelines with `drtmle`. So
the empty equivalence artifact records the absence of a comparator. The evidence is the exact
truth of each law, the declared calibration bands, and two negative controls that must read red.

The study publishes under the `gated` policy. Every gated row passed, and both negative controls
read red as declared. The [red-cell ledger](red-cells.md) lists no cell of this study.

## What was tested

| setting | declaration |
| --- | --- |
| subject | `CTMLE(strategy="oat", oat_design="per_arm", cross_fit=False, simultaneous=False)` with `ParametricOutcome()` and `ParametricMechanism(degree=3)`. The default `alpha` and `max_iter=100`, `tol=1e-10` |
| primary phase | `binary_active` at $n = 1500$, 800 replications, with `ey0`, `ey1`, `ate`, `rr` and `or`. Each replication must report the status `"influence_curve"` for every estimand |
| property phase | 13 fit groups. Each group fits one shared sample per replicate, and its cells read the same fits |
| truth | a quadrature of each law (`tests/studies/oat_parametric_quadrature.py`), so no truth comes from a simulation. A cell of the pseudo-true law reads the limit $\psi^*$ |
| verdicts | each calibration cell needs its SE-ratio and coverage intervals inside the calibration bands. The type-I cell needs its rejection bound under the ceiling. The power cell needs its rejection lower bound above the minimum power |
| failed replicates | a fit that raises, or a fit whose stack status is not `"influence_curve"`, is a failed replicate. The run records it and draws no replacement. A cell may lose `floor(0.01 R)` replicates, and one more reads `bound exceeded`. No replicate was lost |

The seven laws share $W = (W_1, W_2, W_3)$, three independent standard normals.

| law | role | construction |
| --- | --- | --- |
| `binary_active` | primary and calibration | a binary outcome with an arm-specific index in $W_1$ and $W_2$. The treatment law is a softmax in $W_1$ and $W_3$, so $W_3$ is an instrument. The outcome model is correct |
| `binary_null` | type-I error | the treatment law of `binary_active`, with arm 1 reading the index of arm 0, so the ATE is zero |
| `shared_prognosis` | covariance, `rr` and `or` | the treatment law of `binary_active` and $\operatorname{logit} \bar Q_0(a, W) = \pm 0.3 + 3.5 W_1 + 0.8 W_2$. The two arm estimates correlate |
| `u_shaped` | $\kappa_a \ne 1$ | a mechanism U-shaped in $W_1$, which the cubic basis misses. The study also fits the linear basis |
| `steep` | binding bound | a steep mechanism with `g_bounds=(0.05, 0.95)`, which clips a share of the rows at the limit |
| `bounded` | outcome scale | $Y = 2 + 8B$ with $B \sim \operatorname{Beta}(4 \bar Q_0, 4(1 - \bar Q_0))$ and the $\bar Q_0$ of `binary_active`, fitted with `q_bounds=(2, 10)` |
| `pseudo_true` | wrong outcome model | `binary_active` with $W_1$ dropped from the outcome features. The truth of the cell is the limit $\psi^*$ |

The budgets in the table are the declared replications of each fit group.

| group | law | $n$ | replications |
| --- | --- | ---: | ---: |
| `binary_active` | `binary_active` | 1,500 | 17,200 |
| `ladder_500`, `ladder_2000`, `ladder_8000` | `binary_active` | 500, 2,000, 8,000 | 63,600, 8,400, 9,200 |
| `shared_prognosis` | `shared_prognosis`, with `rr` and `or` | 1,500 | 10,000 |
| `kappa_law` | `u_shaped`, cubic and linear basis | 1,500 | 19,600 |
| `bounds_active` | `steep`, `g_bounds=(0.05, 0.95)` | 1,500 | 12,400 |
| `auto_bounds` | `binary_active`, `g_bounds="auto"` | 1,500 | 9,200 |
| `bounded_outcome` | `bounded`, `q_bounds=(2, 10)` | 1,500 | 16,400 |
| `pseudo_true` | `pseudo_true` | 1,500 | 7,600 |
| `generated_design` | `binary_active`, the estimated outcome and a pinned true outcome | 1,500 | 9,200 |
| `sharp_null` | `binary_null` | 1,000 | 800 |
| `power` | `binary_active` | 800 | 1,200 |

The study sizes each calibration group so that the whole study passes with probability at least
0.95 when nothing is wrong. Each group needs a pass probability of $0.95^{1/12}$. The
declaration counted 12 groups. The study has 11 gated calibration groups, so the study-wise pass
probability is at least $0.95^{11/12}$.

A predictive model sets the size. It centres each row on its
own pilot SE ratio, with variance $s^2 + \tau_{80}^2$. The value $\tau_{80} = 0.0080$ is pooled
over the gated rows of the study.

The pilots raised no separation refusal and no stack failure. They drew 5,000 replications for each
cell, then ran each group at its declared budget on pilot-only seeds. The power cell has an
exact-law power of 0.8475 at $n = 800$. At 1,200 replications, the chance that its lower rejection
endpoint clears 0.80 is 0.957.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate` | average treatment effect | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0025 to 0.0028 | 0.9450 | 1.0004 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey0` | counterfactual mean under no treatment | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0019 to 0.0018 | 0.9450 | 0.9924 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey1` | counterfactual mean under treatment | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0018 to 0.0020 | 0.9437 | 0.9705 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0094 to 0.0121 | 0.9450 | 1.0006 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism | -0.0048 to 0.0060 | 0.9487 | 1.0008 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `generated_design` | `binary_active__estimated` | positive | two-arm law with an arm-specific outcome index and an instrument: the same design is estimated from the data, as a real fit does | SE ratio and coverage intervals both inside their calibration bands; the paired deficit is reported only | coverage 0.9417 to 0.9537, SE ratio 0.9779 to 1.0160 | pass |
| `generated_design` | `binary_active__oracle_design` | positive | two-arm law with an arm-specific outcome index and an instrument: the outcome-adaptive design is supplied rather than estimated | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9414 to 0.9535, SE ratio 0.9742 to 1.0111 | pass |
| `interval_calibration` | `auto_bounds__ate__corrected` | positive | the stacked curve with the default g_bounds of "auto": ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9445 to 0.9563, SE ratio 0.9830 to 1.0211 | pass |
| `interval_calibration` | `auto_bounds__ey0__corrected` | positive | the stacked curve with the default g_bounds of "auto": arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9482 to 0.9596, SE ratio 0.9846 to 1.0226 | pass |
| `interval_calibration` | `auto_bounds__ey1__corrected` | positive | the stacked curve with the default g_bounds of "auto": arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9423 to 0.9542, SE ratio 0.9786 to 1.0163 | pass |
| `interval_calibration` | `binary_active__ate__corrected` | positive | two-arm law with an arm-specific outcome index and an instrument: the stacked curve at n = 1,500: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9462 to 0.9548, SE ratio 0.9883 to 1.0164 | pass |
| `interval_calibration` | `binary_active__ate__noise_control` | control | two-arm law with an arm-specific outcome index and an instrument: a declared scale of independent noise is added to each ATE estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8253 to 0.8401, SE ratio 0.6975 to 0.7171 | pass |
| `interval_calibration` | `binary_active__ate__shipped_curve__shrunken_se_control` | control | two-arm law with an arm-specific outcome index and an instrument: the ATE standard errors come from the shipped adaptive-propensity curve of the same fits, which omits the generated-design term | the SE-ratio interval must fall below the calibration band | coverage 0.8944 to 0.9063, SE ratio 0.8358 to 0.8595 | pass |
| `interval_calibration` | `binary_active__ate__shrunken_se_control` | control | two-arm law with an arm-specific outcome index and an instrument: the ATE standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8206 to 0.8355, SE ratio 0.6920 to 0.7115 | pass |
| `interval_calibration` | `binary_active__ey0__corrected` | positive | two-arm law with an arm-specific outcome index and an instrument: the stacked curve at n = 1,500: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9447 to 0.9534, SE ratio 0.9880 to 1.0158 | pass |
| `interval_calibration` | `binary_active__ey1__corrected` | positive | two-arm law with an arm-specific outcome index and an instrument: the stacked curve at n = 1,500: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9513, SE ratio 0.9833 to 1.0112 | pass |
| `interval_calibration` | `bounded_outcome__ate__corrected` | positive | the stacked curve for an outcome bounded in 2 to 10, fitted with q_bounds on the scaled outcome: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9442 to 0.9531, SE ratio 0.9800 to 1.0082 | pass |
| `interval_calibration` | `bounded_outcome__ey0__corrected` | positive | the stacked curve for an outcome bounded in 2 to 10, fitted with q_bounds on the scaled outcome: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9428 to 0.9518, SE ratio 0.9759 to 1.0040 | pass |
| `interval_calibration` | `bounded_outcome__ey1__corrected` | positive | the stacked curve for an outcome bounded in 2 to 10, fitted with q_bounds on the scaled outcome: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9455 to 0.9543, SE ratio 0.9835 to 1.0116 | pass |
| `interval_calibration` | `bounds_active__ate__corrected` | positive | the stacked curve with g_bounds of (0.05, 0.95), which clip a share of the rows at the limit: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9394 to 0.9501, SE ratio 0.9722 to 1.0043 | pass |
| `interval_calibration` | `bounds_active__ey0__corrected` | positive | the stacked curve with g_bounds of (0.05, 0.95), which clip a share of the rows at the limit: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9414 to 0.9518, SE ratio 0.9797 to 1.0115 | pass |
| `interval_calibration` | `bounds_active__ey1__corrected` | positive | the stacked curve with g_bounds of (0.05, 0.95), which clip a share of the rows at the limit: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9392 to 0.9498, SE ratio 0.9688 to 1.0012 | pass |
| `interval_calibration` | `kappa_cubic__ate__corrected` | positive | the stacked curve on a U-shaped mechanism, with the cubic basis in logit q: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9513, SE ratio 0.9726 to 0.9979 | pass |
| `interval_calibration` | `kappa_cubic__ey0__corrected` | positive | the stacked curve on a U-shaped mechanism, with the cubic basis in logit q: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9381 to 0.9467, SE ratio 0.9519 to 0.9771 | pass |
| `interval_calibration` | `kappa_cubic__ey1__corrected` | positive | the stacked curve on a U-shaped mechanism, with the cubic basis in logit q: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9432 to 0.9515, SE ratio 0.9872 to 1.0129 | pass |
| `interval_calibration` | `kappa_linear__ate__corrected` | positive | the stacked curve on a U-shaped mechanism, with the linear basis in logit q: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9436 to 0.9518, SE ratio 0.9824 to 1.0083 | pass |
| `interval_calibration` | `kappa_linear__ey0__corrected` | positive | the stacked curve on a U-shaped mechanism, with the linear basis in logit q: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9439 to 0.9521, SE ratio 0.9805 to 1.0065 | pass |
| `interval_calibration` | `kappa_linear__ey1__corrected` | positive | the stacked curve on a U-shaped mechanism, with the linear basis in logit q: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9447 to 0.9528, SE ratio 0.9814 to 1.0068 | pass |
| `interval_calibration` | `n_2000__ate__corrected` | positive | the stacked curve at n = 2,000: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9457 to 0.9578, SE ratio 0.9938 to 1.0338 | pass |
| `interval_calibration` | `n_2000__ey0__corrected` | positive | the stacked curve at n = 2,000: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9419 to 0.9545, SE ratio 0.9763 to 1.0161 | pass |
| `interval_calibration` | `n_2000__ey1__corrected` | positive | the stacked curve at n = 2,000: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9453 to 0.9575, SE ratio 0.9946 to 1.0359 | pass |
| `interval_calibration` | `n_500__ate__corrected` | positive | the stacked curve at n = 500: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9443 to 0.9489, SE ratio 0.9895 to 1.0039 | pass |
| `interval_calibration` | `n_500__ey0__corrected` | positive | the stacked curve at n = 500: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9421 to 0.9468, SE ratio 0.9827 to 0.9970 | pass |
| `interval_calibration` | `n_500__ey1__corrected` | positive | the stacked curve at n = 500: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9419 to 0.9466, SE ratio 0.9870 to 1.0014 | pass |
| `interval_calibration` | `n_8000__ate__corrected` | positive | the stacked curve at n = 8,000: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9478 to 0.9592, SE ratio 0.9934 to 1.0311 | pass |
| `interval_calibration` | `n_8000__ey0__corrected` | positive | the stacked curve at n = 8,000: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9455 to 0.9571, SE ratio 0.9887 to 1.0274 | pass |
| `interval_calibration` | `n_8000__ey1__corrected` | positive | the stacked curve at n = 8,000: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9473 to 0.9587, SE ratio 0.9926 to 1.0312 | pass |
| `interval_calibration` | `pseudo_true__ate__corrected` | positive | the stacked curve with W1 dropped from the outcome features, so the interval covers the limit psi*: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9411 to 0.9544, SE ratio 0.9767 to 1.0184 | pass |
| `interval_calibration` | `pseudo_true__ey0__corrected` | positive | the stacked curve with W1 dropped from the outcome features, so the interval covers the limit psi*: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9414 to 0.9546, SE ratio 0.9773 to 1.0197 | pass |
| `interval_calibration` | `pseudo_true__ey1__corrected` | positive | the stacked curve with W1 dropped from the outcome features, so the interval covers the limit psi*: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9560, SE ratio 0.9737 to 1.0156 | pass |
| `interval_calibration` | `shared_prognosis__ate__corrected` | positive | the stacked curve on a law whose two arm means share W1 and W2 in their outcome index: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9543, SE ratio 0.9729 to 1.0085 | pass |
| `interval_calibration` | `shared_prognosis__ate__cross_term_dropped__inflated_se_control` | control | the ATE standard errors are built from the two arm variances without the cross-arm covariance, on a law whose arm estimates correlate | the SE-ratio interval must fall above the calibration band | coverage 0.9811 to 0.9876, SE ratio 1.1958 to 1.2396 | pass |
| `interval_calibration` | `shared_prognosis__log_or__corrected` | positive | the stacked curve on a law whose two arm means share W1 and W2 in their outcome index: log odds ratio | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9543, SE ratio 0.9739 to 1.0095 | pass |
| `interval_calibration` | `shared_prognosis__log_rr__corrected` | positive | the stacked curve on a law whose two arm means share W1 and W2 in their outcome index: log risk ratio | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9540, SE ratio 0.9725 to 1.0088 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.8442, 0.8154 to 0.8701 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0550, 0.0363 to 0.0792 | pass |
<!-- /generated -->

Read the two `generated_design` cells with this study's setting. The shared descriptions come
from the parent study. In this study the `oracle_design` cell pins the true outcome regression
and fits none, so the shipped curve is its full curve. The `estimated` cell fits the outcome
model and reads the stacked curve. The study gates both cells. The phrase "the paired deficit is
reported only" does not apply here.

## Declared reading

The study fixed its reading before the first run, in the module
`tests/studies/canonical_ctmle_oat_per_arm_parametric.py`.

| rule | text |
| --- | --- |
| red calibration row | a finite-sample result only when its 99% SE-ratio interval contains 1, its coverage agrees with 0.95, and the diagnosis finds no defect. Any other red row is a defect suspect, diagnosed before publication |
| negative control | each control must read red. A control row publishes `passed = True` exactly when its SE-ratio interval leaves the calibration band on the declared side. A control that stays inside the band, or straddles it, cannot see its term. The study is then invalid and the design changes before any reading |
| reported only | the quantities of [the next section](#quantities-reported-and-not-gated) gate nothing |
| failures | a failed replicate is recorded and not redrawn. Dropping it would select on success. Substituting the shipped standard error would mix two estimators |

## Negative controls

Both controls read red. The table gives each control, the term it removes, and the measured SE
ratio.

| control | cell | term removed | expected SE ratio | side | measured |
| --- | --- | --- | --- | --- | --- |
| shipped curve | `binary_active__ate__shipped_curve__shrunken_se_control` | the generated-design term. The standard error comes from the shipped adaptive-propensity curve of the same fits | 0.845 | below the band | `properties[interval_calibration/binary_active__ate__shipped_curve__shrunken_se_control]:se_ratio`, with an upper endpoint of `properties[interval_calibration/binary_active__ate__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` |
| cross term dropped | `shared_prognosis__ate__cross_term_dropped__inflated_se_control` | the cross-arm covariance. The ATE standard error is $\sqrt{se_0^2 + se_1^2}$ | 1.228 | above the band | `properties[interval_calibration/shared_prognosis__ate__cross_term_dropped__inflated_se_control]:se_ratio`, with a lower endpoint of `properties[interval_calibration/shared_prognosis__ate__cross_term_dropped__inflated_se_control]:se_ratio_ci_lower` |

The first control uses the same fits as `binary_active__ate__corrected`. Its coverage is
`properties[interval_calibration/binary_active__ate__shipped_curve__shrunken_se_control]:coverage`,
so the shipped curve under-covers where the stacked curve calibrates. The second control uses the
fits of `shared_prognosis__ate__corrected`. The two arm estimates correlate on this
law, so the dropped term is material. The harness also derives a `shrunken_se_control` and a
`noise_control` from the primary ATE row. Both read below the band.

## Quantities reported and not gated

These quantities describe the fit, and no verdict reads them.

| quantity | cell | what it shows | measured |
| --- | --- | --- | --- |
| coverage of $\psi_0$ | `pseudo_true__ey0__corrected`, `pseudo_true__ey1__corrected`, `pseudo_true__ate__corrected` | the interval covers $\psi^*$ and not $\psi_0$ when the outcome model is wrong. The gated coverage of $\psi^*$ in the same cells passes | `properties[interval_calibration/pseudo_true__ey0__corrected]:psi0_coverage`, `properties[interval_calibration/pseudo_true__ey1__corrected]:psi0_coverage` and `properties[interval_calibration/pseudo_true__ate__corrected]:psi0_coverage` |
| covariance of the arm means | `shared_prognosis__ate__corrected` | the Monte Carlo covariance of the two arm estimates against the mean reported covariance | a ratio of `properties[interval_calibration/shared_prognosis__ate__corrected]:covariance_ratio`, with a 99% interval from `properties[interval_calibration/shared_prognosis__ate__corrected]:covariance_ratio_ci_lower` to `properties[interval_calibration/shared_prognosis__ate__corrected]:covariance_ratio_ci_upper` |
| $\sqrt{n}$ SD of the ladder | `n_500__ate__corrected`, `n_2000__ate__corrected`, `n_8000__ate__corrected` | the empirical and reported $\sqrt{n}$ SD of the ATE against the quadrature SD | the table of [the ladder](#the-ladder) |
| clipped share | `bounds_active__ate__corrected` | the mean share of rows at a `g_bounds` limit, over arms and replications | `properties[interval_calibration/bounds_active__ate__corrected]:clipped_share` |

### The ladder

The quadrature gives the asymptotic $\sqrt{n}$ SD of the ATE curve as `properties[interval_calibration/n_500__ate__corrected]:quadrature_sd`.
The table gives the empirical SD and the mean reported standard error, each times $\sqrt{n}$.

| $n$ | empirical $\sqrt{n}$ SD | reported $\sqrt{n}$ SE |
| ---: | --- | --- |
| 500 | `properties[interval_calibration/n_500__ate__corrected]:sqrt_n_empirical_sd` | `properties[interval_calibration/n_500__ate__corrected]:sqrt_n_reported_se` |
| 2,000 | `properties[interval_calibration/n_2000__ate__corrected]:sqrt_n_empirical_sd` | `properties[interval_calibration/n_2000__ate__corrected]:sqrt_n_reported_se` |
| 8,000 | `properties[interval_calibration/n_8000__ate__corrected]:sqrt_n_empirical_sd` | `properties[interval_calibration/n_8000__ate__corrected]:sqrt_n_reported_se` |

### The covariance

The study reports the Monte Carlo covariance and does not gate it. At 10,000 replications the 99%
interval of the ratio runs from `properties[interval_calibration/shared_prognosis__ate__corrected]:covariance_ratio_ci_lower`
to `properties[interval_calibration/shared_prognosis__ate__corrected]:covariance_ratio_ci_upper`.
That interval is wider than the SE-ratio band of `margin:calibration_se_ratio_lower` to
`margin:calibration_se_ratio_upper`, so the SE-ratio band does not transfer to the covariance. The
study declared no covariance band before the run. The ATE row and the cross-term control carry the
covariance instead. The ATE variance contains $-2 \operatorname{Cov}$, and the control moves the
ATE SE ratio out of the band when that term is missing.

## Deviations from the plan

The plan is the X58 plan, section 5. The table lists each deviation and its reason.

| plan | study | reason |
| --- | --- | --- |
| each cell sized to a family-wise pass probability of 0.95 (section 11) | each of the 11 calibration groups sized to $0.95^{1/12}$, so the whole study passes with probability at least 0.95 | per-cell sizing gives a study-wise pass probability near 0.63, as the pre-queue design review found |
| the Monte Carlo covariance of the arm means gated, with a bootstrap interval | the covariance reported and not gated | at 10,000 replications the 99% interval of the ratio is wider than the SE-ratio band, and the study declared no covariance band, as [the covariance](#the-covariance) shows. The ATE row and the cross-term control carry the covariance |
| the ladder in the `root_n_and_efficiency` family and the joint cells in a `joint` family | the ladder and the joint cells are `interval_calibration` cells named `<label>__<estimand>__corrected` | the shared calibration verdict then applies to each estimand. The `sqrt(n)` SD of the ladder stays a reported column |
| the `generated_design` pair a reporting diagnostic, as in the parent study | the pair is a gated family. Both cells pass | the pinned cell fits no outcome model, so the shipped curve is its full curve. The estimated cell reads the stacked curve |

## Result

Every one of the `property_cells_total` cells passes, and so does each of the
`independent_tests_total` primary tests. The lowest primary coverage is `min_coverage`, with a lower
endpoint of `min_coverage_ci_lower`. The primary SE-ratio endpoints run from `min_se_ratio_ci_lower`
to `max_se_ratio_ci_upper`. The largest standardized bias is `max_standardized_bias`.

The stacked curve calibrates on every law the study fits. The sample sizes run from 500 to 8,000.
The mechanism misses the projection on `u_shaped`, a `g_bounds` limit binds on `steep`, the default
`"auto"` bound resolves on `binary_active`, and a bounded outcome fits on its own scale. The
shipped curve fails the same band on `binary_active`, so the correction is what the study measures.

The coverage of $\psi^*$ in the pseudo-true cells is calibrated, and the coverage of $\psi_0$ is
not. A user who reads the interval as an interval for $\psi_0$ under a wrong outcome model gets
the second number. [Parametric per-arm intervals](../collaborative-tmle.md#parametric-per-arm-intervals)
states the claim.

The sharp-null cell rejects at `properties[type_i_error/sharp_null]:rejection_rate`, with an upper
endpoint of `properties[type_i_error/sharp_null]:rejection_ci_upper` against the ceiling
`margin:type_i_ceiling`. The power cell rejects at `properties[power/alternative]:rejection_rate`,
with a lower endpoint of `properties[power/alternative]:rejection_ci_lower` against
`margin:minimum_power`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 1500 | observations per primary replication |
| `independent_tests_total` | 5 | primary tests against truth |
| `independent_tests_passed` | 5 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 41 | repeated-sampling property cells |
| `property_cells_passed` | 41 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0116 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9437 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9194 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9157 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0709 | highest bootstrap primary SE-ratio endpoint |
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
| `margin:minimum_power` | 0.8000 | rejection lower bound the power cell must clear |
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
| `margin:subject_failures_allowed` | 8 | replicates one primary cell may lose |
| `margin:type_i_ceiling` | 0.1000 | the rate the null cell must bound |
| `properties[interval_calibration/binary_active__ate__shipped_curve__shrunken_se_control]:se_ratio` | 0.8476 | SE ratio, shipped-curve control |
| `properties[interval_calibration/binary_active__ate__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` | 0.8595 | its 99% upper endpoint |
| `properties[interval_calibration/binary_active__ate__shipped_curve__shrunken_se_control]:coverage` | 0.9005 | coverage, shipped-curve control |
| `properties[interval_calibration/shared_prognosis__ate__cross_term_dropped__inflated_se_control]:se_ratio` | 1.2173 | SE ratio, cross-term control |
| `properties[interval_calibration/shared_prognosis__ate__cross_term_dropped__inflated_se_control]:se_ratio_ci_lower` | 1.1958 | its 99% lower endpoint |
| `properties[interval_calibration/pseudo_true__ey0__corrected]:psi0_coverage` | 0.5691 | coverage of $\psi_0$, arm-0 mean, wrong outcome model |
| `properties[interval_calibration/pseudo_true__ey1__corrected]:psi0_coverage` | 0.5625 | coverage of $\psi_0$, arm-1 mean, wrong outcome model |
| `properties[interval_calibration/pseudo_true__ate__corrected]:psi0_coverage` | 0.3053 | coverage of $\psi_0$, ATE, wrong outcome model |
| `properties[interval_calibration/shared_prognosis__ate__corrected]:covariance_ratio` | 1.0055 | Monte Carlo covariance over mean reported covariance |
| `properties[interval_calibration/shared_prognosis__ate__corrected]:covariance_ratio_ci_lower` | 0.9269 | its 99% lower endpoint |
| `properties[interval_calibration/shared_prognosis__ate__corrected]:covariance_ratio_ci_upper` | 1.0956 | its 99% upper endpoint |
| `properties[interval_calibration/n_500__ate__corrected]:quadrature_sd` | 1.1234 | asymptotic $\sqrt{n}$ SD of the ATE curve |
| `properties[interval_calibration/n_500__ate__corrected]:sqrt_n_empirical_sd` | 1.1240 | empirical $\sqrt{n}$ SD, $n = 500$ |
| `properties[interval_calibration/n_500__ate__corrected]:sqrt_n_reported_se` | 1.1204 | reported $\sqrt{n}$ SE, $n = 500$ |
| `properties[interval_calibration/n_2000__ate__corrected]:sqrt_n_empirical_sd` | 1.1076 | empirical $\sqrt{n}$ SD, $n = 2000$ |
| `properties[interval_calibration/n_2000__ate__corrected]:sqrt_n_reported_se` | 1.1226 | reported $\sqrt{n}$ SE, $n = 2000$ |
| `properties[interval_calibration/n_8000__ate__corrected]:sqrt_n_empirical_sd` | 1.1097 | empirical $\sqrt{n}$ SD, $n = 8000$ |
| `properties[interval_calibration/n_8000__ate__corrected]:sqrt_n_reported_se` | 1.1231 | reported $\sqrt{n}$ SE, $n = 8000$ |
| `properties[interval_calibration/bounds_active__ate__corrected]:clipped_share` | 0.1440 | share of rows at a `g_bounds` limit, steep law |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0550 | rejection rate at the sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0792 | its 99% upper endpoint |
| `properties[power/alternative]:rejection_rate` | 0.8442 | rejection rate at the alternative |
| `properties[power/alternative]:rejection_ci_lower` | 0.8154 | its 99% lower endpoint |

## Limits

| limit | what it means |
| --- | --- |
| In sample, one split, one draw | The fit is not cross-fitted, and `repeats` is 1. This study measures no cross-fitted or repeated fit. The [cross-fitted study](outcome-adaptive-per-arm-c-tmle-parametric-crossfit.md) of [X58](../../roadmap.md#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) part (c) measures them |
| Two arms | The treatment is binary. The [extensions study](outcome-adaptive-per-arm-c-tmle-parametric-extensions.md) of [X58](../../roadmap.md#x58-per-arm-outcome-adaptive-c-tmle-intervals-with-a-parametric-outcome) part (b) measures three arms, `ey_obs`, `par`, `paf` and the simultaneous band |
| Complete, unweighted, unstratified, unclustered data | The study measures no weight, stratum, missing outcome, missing treatment or cluster. The [extensions study](outcome-adaptive-per-arm-c-tmle-parametric-extensions.md) of X58 part (d) measures a known fixed weight, and X52 holds the rest |
| Two parametric learners only | Only `ParametricOutcome` and `ParametricMechanism` are admitted, as exact classes. A scikit-learn learner keeps the `"generated_design_plugin"` status, and [F19](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) owns its interval |
| The interval covers $\psi^*$ | Under a wrong outcome model the interval covers the limit $\psi^*$ and not $\psi_0$. The pseudo-true cells report the coverage of $\psi_0$ and gate nothing on it. The claim is pointwise, not locally uniform over the nonparametric model |
| Default `alpha` only | Every cell runs at the default `alpha`. No cell sets `alpha` below it. The [clip caveat](../collaborative-tmle.md#parametric-per-arm-intervals) for a user `alpha` such as 0.9 has no study cell |
| Binary outcomes and one bounded outcome | Seven laws on three normal covariates. The bounded outcome is a scaled binary draw. A continuous outcome with a Gaussian family has fast-tier evidence only |
| The covariance is reported, not gated | [The covariance](#the-covariance) explains why. Only the ATE row and the cross-term control test it |
| Intervals only | The bootstrap of an `oat` fit stays a diagnostic. This study does not measure it |

## Reproduction

Run `python -m tests.canonical.ctmle_oat_per_arm_parametric.regenerate` to regenerate the
artifacts. The directory
[`tests/canonical/ctmle_oat_per_arm_parametric`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/canonical/ctmle_oat_per_arm_parametric)
holds `manifest.json`, `run.log` and the committed results. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_oat_per_arm_parametric/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_oat_per_arm_parametric/properties.csv)
carry every published row. The run used 4,788 wall seconds on four cores.
