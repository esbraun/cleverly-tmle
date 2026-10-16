# Outcome-adaptive per-arm C-TMLE (parametric extensions)

This study measures the interval of
`CTMLE(strategy="oat", oat_design="per_arm")` with `ParametricOutcome` and `ParametricMechanism`
in four settings that the two earlier studies leave out. The settings are three arms, a known
observation weight, the population targets `ey_obs`, `par` and `paf`, and the simultaneous band. Each fit is an exact
stacked Z-estimator, in sample or cross-fitted. The weight enters the stack as the normalized mass
of each row.

The section
[Parametric per-arm intervals](../collaborative-tmle.md#parametric-per-arm-intervals) states the
construction, its conditions and what the interval covers. The study
[`ctmle-oat-per-arm-parametric`](outcome-adaptive-per-arm-c-tmle-parametric.md) measures the
binary in-sample form. The study
[`ctmle-oat-per-arm-parametric-crossfit`](outcome-adaptive-per-arm-c-tmle-parametric-crossfit.md)
measures the binary cross-fitted form and the median over draws.

**Python property evidence, no comparator.** No canonical implementation is compared. R
[`drtmle`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c)
with `adapt_g = TRUE` reports the shipped curve of Theorem 1 of Benkeser, Cai and van der Laan
(2020). The stack corrects that curve, so no R program measures the corrected uncertainty. The
empty equivalence artifact records the absence of a comparator. The evidence is the exact truth of
each law, the declared calibration bands, and nine negative controls that must read red.

The study publishes under the `gated` policy. Every gated row passed, and all nine negative
controls read red as declared. The [red-cell ledger](red-cells.md) lists no cell of this study.

## What was tested

| setting | declaration |
| --- | --- |
| subject | `CTMLE(strategy="oat", oat_design="per_arm", n_folds=5, repeats=1)` with `ParametricOutcome(arm_columns=K - 1)` and `ParametricMechanism(degree=3)`. The fit is in sample or cross-fitted, by group. The study sets `g_bounds=(0.01, 0.99)`, `max_iter=100` and `tol=1e-10`, and leaves `alpha` at its default |
| primary phase | in sample at $n = 1500$, 1,600 replications per law. The law `binary_active` reads `par` and `paf`. The law `three_arm_active` fits `reference="0"` and reads `ey`, `ate`, `rr` and `or`. Each replication must report the status `"influence_curve"` for every estimand. The harness gates its rows |
| property phase | 9 fit groups. Each group fits one shared sample per replicate, and its cells read the same fits |
| truth | a quadrature of each law (`tests/studies/oat_multi_arm_quadrature.py`), so no truth comes from a simulation. The truth of a weighted law is the truth under the weight-tilted law |
| verdicts | each calibration cell needs its SE-ratio and coverage intervals inside the calibration bands. A band cell needs its joint-coverage interval inside the coverage band. A type-I cell needs its rejection bound under the ceiling. A control cell needs its SE-ratio interval outside the band on the declared side, or its joint-coverage upper endpoint below 0.95 |
| failed replicates | a fit that raises, or a fit with an estimate whose status is not `"influence_curve"`, is a failed replicate. A band cell loses a replicate when any estimate of its declared band set has another status. The run records the loss and draws no replacement. A cell may lose `floor(0.01 R)` replicates, and one more reads `bound exceeded`. No replicate was lost |

The laws share $W = (W_1, W_2, W_3)$, three independent standard normals.

| law | role | construction |
| --- | --- | --- |
| `binary_active` | binary population targets, two arms | a binary outcome with an arm-specific index in $W_1$ and $W_2$. The treatment law is a softmax in $W_1$ and $W_3$, so $W_3$ is an instrument. The outcome model is correct |
| `binary_active_weighted` | known weight, two arms | `binary_active` under the known weight $0.5 + 1\{W_1 > 0\}$. The truth is the truth under the weight-tilted law |
| `three_arm_active` | three arms | three arm-specific indices in $W_1$ and $W_2$, and a softmax treatment law in $W_1$ and $W_3$. The outcome model is correct |
| `three_arm_shared` | the cross-arm covariance | the mechanism of `three_arm_active` with one strong prognosis, $\text{logit}\, \bar Q_0(a, W) = c_a + 3.5 W_1 + 0.8 W_2$ in every arm. The arm curves correlate at 0.25 to 0.26 |
| `three_arm_active_weighted` | three arms and the known weight | `three_arm_active` under the known weight |
| `three_arm_null` | type-I error | every arm reads the index $c = -0.3$, $u = (0.8, -0.8)$, so every contrast is zero |

The reference arm is a property of the group. The groups `three_arm`, `three_arm_shared`,
`composition` and `three_arm_null` fit `reference="1"`. The groups `three_arm_crossfit`,
`three_arm_generated_design` and the primary phase fit `reference="0"`, the setting of the earlier
three-arm study. The budgets in the table are the declared replications of each fit group.

| group | law | $n$ | replications | fit | estimands | role |
| --- | --- | ---: | ---: | --- | --- | --- |
| primary phase | `binary_active` and `three_arm_active` | 1,500 | 1,600 per law | in sample | `par`, `paf`; `ey`, `ate`, `rr`, `or` | gated against truth |
| `three_arm` | `three_arm_active` | 1,500 | 29,200 | in sample, band fit | `ey`, `ate`, `rr`, `or`, `ey_obs`, `par`, `paf` | calibration, the band, and three controls |
| `three_arm_shared` | `three_arm_shared` | 1,500 | 14,000 | in sample | `ey`, `ate` | calibration, and the cross-term control |
| `binary_population` | `binary_active` | 1,500 | 11,200 | in sample, band fit | `ey0`, `ey1`, `ate`, `ey_obs`, `par`, `paf` | calibration, the band, and three controls |
| `three_arm_crossfit` | `three_arm_active` | 1,500 | 18,800 | cross-fitted, band fit | `ey`, `ate` | calibration, the band, and one control |
| `three_arm_generated_design` | `three_arm_active`, the estimated outcome and a pinned true outcome | 1,500 | 80,000 | cross-fitted | `ate` | the generated-design pair |
| `weighted` | `binary_active_weighted` | 1,500 | 17,600 | in sample | `ey0`, `ey1`, `ate`, `rr`, `or`, `ey_obs`, `par`, `paf` | calibration and one control |
| `weighted_crossfit` | `binary_active_weighted` | 1,500 | 11,600 | cross-fitted | `ey0`, `ey1`, `ate` | calibration |
| `composition` | `three_arm_active_weighted` | 1,500 | 17,600 | cross-fitted | `ey`, `ate` | calibration |
| `three_arm_null` | `three_arm_null` | 1,000 | 1,600 | in sample | `ate` | type-I error |

Each band set is the exact set of estimates that the band must cover. The band of `three_arm` and
`composition` reads `ey[0]`, `ey[1]`, `ey[2]`, `ate[0 vs 1]` and `ate[2 vs 1]`. The band of
`three_arm_crossfit` reads `ey[0]`, `ey[1]`, `ey[2]`, `ate[1 vs 0]` and `ate[2 vs 0]`. The band
of `binary_population` reads `ey0`, `ey1`, `ate`, `ey_obs`, `par` and `paf`.

The study sizes each unit so that the whole study passes with probability at least 0.95 when
nothing is wrong. The count $G$ is 19: eight calibration groups with the generated-design pair
among them, one type-I group, the primary phase and nine negative controls. Each unit needs a
pass probability of $0.95^{1/19}$, which is 0.99730. A predictive model sets the size. It centres
each row on its own pilot SE ratio and adds a pooled spread $\tau_{80}$ over the gated rows. The
[deviations](#deviations-from-the-plan) state where the model gives no size.

The declared-size pilot read every group at its declared budget on pilot-only seeds. The seeds
of the first pass (5,000 replications per group) and of the declared-size pass differ. The
predictive model, pooled over both passes with $\tau_{80} = 0.0106$, gives a study-wise pass
probability of 0.9844. The primary phase pilot passed all 11 rows at 1,600 replications per law.

## Accuracy against known truth
<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `paf` | population attributable fraction | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0024 to 0.0020 | 0.9537 | 1.0084 | pass |
| two arms, an arm-specific outcome index, an instrument, correct outcome regression | `par` | population attributable risk | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0012 to 0.000968 | 0.9525 | 1.0086 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate[1 vs 0]` | difference in counterfactual means, 1 versus 0 | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0021 to 0.0024 | 0.9619 | 1.0271 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ate[2 vs 0]` | difference in counterfactual means, 2 versus 0 | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0022 to 0.0026 | 0.9487 | 0.9899 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[0]` | counterfactual mean under treatment arm '0' | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0018 to 0.0013 | 0.9544 | 0.9984 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[1]` | counterfactual mean under treatment arm '1' | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0018 to 0.0016 | 0.9531 | 1.0077 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `ey[2]` | counterfactual mean under treatment arm '2' | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0017 to 0.0017 | 0.9481 | 1.0060 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `or[1 vs 0]` | marginal odds ratio, 1 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0077 to 0.0106 | 0.9619 | 1.0272 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `or[2 vs 0]` | marginal odds ratio, 2 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0071 to 0.0123 | 0.9525 | 0.9907 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr[1 vs 0]` | marginal risk ratio, 1 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0042 to 0.0057 | 0.9594 | 1.0255 | pass |
| three arms, an arm-specific outcome index, an instrument, correct outcome regression | `rr[2 vs 0]` | marginal risk ratio, 2 versus 0, reported on the log scale | `cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, at three arms, with known weights and for the population targets | -0.0037 to 0.0060 | 0.9544 | 0.9879 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `generated_design` | `three_arm_active__estimated__ate_1_vs_0` | positive | three-arm law with an arm-specific outcome index and an instrument: the same design is estimated from the data, as a real fit does, cross-fitted, ATE, arm 1 versus arm 0 | SE ratio and coverage intervals both inside their calibration bands; the paired deficit is reported only | coverage 0.9483 to 0.9522, SE ratio 1.0031 to 1.0159 | pass |
| `generated_design` | `three_arm_active__estimated__ate_2_vs_0` | positive | three-arm law with an arm-specific outcome index and an instrument: the same design is estimated from the data, as a real fit does, cross-fitted, ATE, arm 2 versus arm 0 | SE ratio and coverage intervals both inside their calibration bands; the paired deficit is reported only | coverage 0.9486 to 0.9526, SE ratio 1.0034 to 1.0168 | pass |
| `generated_design` | `three_arm_active__oracle_design__ate_1_vs_0` | positive | three-arm law with an arm-specific outcome index and an instrument: the outcome-adaptive design is supplied rather than estimated, cross-fitted, ATE, arm 1 versus arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9522 to 0.9561, SE ratio 1.0177 to 1.0309 | pass |
| `generated_design` | `three_arm_active__oracle_design__ate_2_vs_0` | positive | three-arm law with an arm-specific outcome index and an instrument: the outcome-adaptive design is supplied rather than estimated, cross-fitted, ATE, arm 2 versus arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9523 to 0.9561, SE ratio 1.0196 to 1.0327 | pass |
| `interval_calibration` | `binary_population__ey_obs__corrected` | positive | the stacked curve of the population targets on the two-arm law, in sample: observed mean of the outcome | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9381 to 0.9494, SE ratio 0.9759 to 1.0101 | pass |
| `interval_calibration` | `binary_population__paf__corrected` | positive | the stacked curve of the population targets on the two-arm law, in sample: population attributable fraction against reference arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9427 to 0.9535, SE ratio 0.9827 to 1.0165 | pass |
| `interval_calibration` | `binary_population__par__corrected` | positive | the stacked curve of the population targets on the two-arm law, in sample: population attributable risk against reference arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9545, SE ratio 0.9830 to 1.0182 | pass |
| `interval_calibration` | `binary_population__par__noise_control` | control | a declared scale of independent noise is added to each estimate of this contrast | the SE-ratio interval must fall below the calibration band | coverage 0.8226 to 0.8409, SE ratio 0.6950 to 0.7194 | pass |
| `interval_calibration` | `binary_population__par__shipped_curve__shrunken_se_control` | control | the standard errors of population attributable risk against reference arm 0 come from the shipped adaptive-propensity curve of the same fits, which omits the generated-design term | the SE-ratio interval must fall below the calibration band | coverage 0.8608 to 0.8773, SE ratio 0.7544 to 0.7810 | pass |
| `interval_calibration` | `binary_population__par__shrunken_se_control` | control | the standard errors of this contrast are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8209 to 0.8393, SE ratio 0.6882 to 0.7127 | pass |
| `interval_calibration` | `composition__ate_0_vs_1__corrected` | positive | the cross-fitted stacked curve on the three-arm law under the known weight, five folds, reference arm 1: ATE, arm 0 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9452 to 0.9538, SE ratio 1.0000 to 1.0279 | pass |
| `interval_calibration` | `composition__ate_0_vs_1__noise_control` | control | a declared scale of independent noise is added to each estimate of this contrast | the SE-ratio interval must fall below the calibration band | coverage 0.8341 to 0.8484, SE ratio 0.7129 to 0.7327 | pass |
| `interval_calibration` | `composition__ate_0_vs_1__shrunken_se_control` | control | the standard errors of this contrast are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8275 to 0.8420, SE ratio 0.7002 to 0.7199 | pass |
| `interval_calibration` | `composition__ate_2_vs_1__corrected` | positive | the cross-fitted stacked curve on the three-arm law under the known weight, five folds, reference arm 1: ATE, arm 2 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9440 to 0.9526, SE ratio 0.9929 to 1.0203 | pass |
| `interval_calibration` | `composition__ey_0__corrected` | positive | the cross-fitted stacked curve on the three-arm law under the known weight, five folds, reference arm 1: mean of arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9471 to 0.9555, SE ratio 1.0049 to 1.0331 | pass |
| `interval_calibration` | `composition__ey_1__corrected` | positive | the cross-fitted stacked curve on the three-arm law under the known weight, five folds, reference arm 1: mean of arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9440 to 0.9526, SE ratio 0.9955 to 1.0245 | pass |
| `interval_calibration` | `composition__ey_2__corrected` | positive | the cross-fitted stacked curve on the three-arm law under the known weight, five folds, reference arm 1: mean of arm 2 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9413 to 0.9502, SE ratio 0.9969 to 1.0253 | pass |
| `interval_calibration` | `three_arm__ate_0_vs_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: ATE, arm 0 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9431 to 0.9499, SE ratio 0.9808 to 1.0019 | pass |
| `interval_calibration` | `three_arm__ate_0_vs_1__noise_control` | control | a declared scale of independent noise is added to each estimate of this contrast | the SE-ratio interval must fall below the calibration band | coverage 0.8226 to 0.8340, SE ratio 0.6947 to 0.7097 | pass |
| `interval_calibration` | `three_arm__ate_0_vs_1__shrunken_se_control` | control | the standard errors of this contrast are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8225 to 0.8339, SE ratio 0.6868 to 0.7016 | pass |
| `interval_calibration` | `three_arm__ate_2_vs_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: ATE, arm 2 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9449 to 0.9516, SE ratio 0.9894 to 1.0106 | pass |
| `interval_calibration` | `three_arm__ey_0__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: mean of arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9422 to 0.9491, SE ratio 0.9805 to 1.0010 | pass |
| `interval_calibration` | `three_arm__ey_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: mean of arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9493, SE ratio 0.9852 to 1.0066 | pass |
| `interval_calibration` | `three_arm__ey_1__shipped_curve__shrunken_se_control` | control | the standard errors of mean of arm 1 come from the shipped adaptive-propensity curve of the same fits, which omits the generated-design term | the SE-ratio interval must fall below the calibration band | coverage 0.8659 to 0.8760, SE ratio 0.7703 to 0.7868 | pass |
| `interval_calibration` | `three_arm__ey_2__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: mean of arm 2 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9429 to 0.9497, SE ratio 0.9844 to 1.0059 | pass |
| `interval_calibration` | `three_arm__ey_obs__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: observed mean of the outcome | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9455 to 0.9521, SE ratio 0.9887 to 1.0103 | pass |
| `interval_calibration` | `three_arm__log_or_0_vs_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: log odds ratio, arm 0 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9436 to 0.9503, SE ratio 0.9809 to 1.0019 | pass |
| `interval_calibration` | `three_arm__log_or_2_vs_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: log odds ratio, arm 2 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9454 to 0.9521, SE ratio 0.9895 to 1.0103 | pass |
| `interval_calibration` | `three_arm__log_rr_0_vs_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: log risk ratio, arm 0 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9444 to 0.9511, SE ratio 0.9806 to 1.0017 | pass |
| `interval_calibration` | `three_arm__log_rr_2_vs_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: log risk ratio, arm 2 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9458 to 0.9525, SE ratio 0.9894 to 1.0109 | pass |
| `interval_calibration` | `three_arm__paf_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: population attributable fraction against reference arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9426 to 0.9494, SE ratio 0.9850 to 1.0058 | pass |
| `interval_calibration` | `three_arm__par_1__corrected` | positive | the stacked curve on the three-arm law, in sample, reference arm 1: population attributable risk against reference arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9424 to 0.9493, SE ratio 0.9848 to 1.0063 | pass |
| `interval_calibration` | `three_arm_crossfit__ate_1_vs_0__corrected` | positive | the cross-fitted stacked curve on the three-arm law, five folds, reference arm 0: ATE, arm 1 versus arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9482 to 0.9563, SE ratio 1.0021 to 1.0285 | pass |
| `interval_calibration` | `three_arm_crossfit__ate_1_vs_0__noise_control` | control | a declared scale of independent noise is added to each estimate of this contrast | the SE-ratio interval must fall below the calibration band | coverage 0.8335 to 0.8473, SE ratio 0.7102 to 0.7291 | pass |
| `interval_calibration` | `three_arm_crossfit__ate_1_vs_0__shrunken_se_control` | control | the standard errors of this contrast are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8312 to 0.8451, SE ratio 0.7010 to 0.7199 | pass |
| `interval_calibration` | `three_arm_crossfit__ate_2_vs_0__corrected` | positive | the cross-fitted stacked curve on the three-arm law, five folds, reference arm 0: ATE, arm 2 versus arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9469 to 0.9550, SE ratio 1.0018 to 1.0289 | pass |
| `interval_calibration` | `three_arm_crossfit__ey_0__corrected` | positive | the cross-fitted stacked curve on the three-arm law, five folds, reference arm 0: mean of arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9480 to 0.9561, SE ratio 1.0023 to 1.0293 | pass |
| `interval_calibration` | `three_arm_crossfit__ey_1__corrected` | positive | the cross-fitted stacked curve on the three-arm law, five folds, reference arm 0: mean of arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9422 to 0.9507, SE ratio 0.9910 to 1.0184 | pass |
| `interval_calibration` | `three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control` | control | the standard errors of mean of arm 1 come from the shipped adaptive-propensity curve of the same fits, which omits the generated-design term | the SE-ratio interval must fall below the calibration band | coverage 0.8701 to 0.8825, SE ratio 0.7805 to 0.8021 | pass |
| `interval_calibration` | `three_arm_crossfit__ey_2__corrected` | positive | the cross-fitted stacked curve on the three-arm law, five folds, reference arm 0: mean of arm 2 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9521, SE ratio 0.9940 to 1.0211 | pass |
| `interval_calibration` | `three_arm_shared__ate_0_vs_1__corrected` | positive | the stacked curve on a three-arm law whose arms share W1 and W2 in their outcome index, in sample, reference arm 1: ATE, arm 0 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9399 to 0.9499, SE ratio 0.9705 to 1.0011 | pass |
| `interval_calibration` | `three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control` | control | the standard errors of the contrast are built from the two arm variances without the cross-arm covariance, on a law whose arm estimates correlate | the SE-ratio interval must fall above the calibration band | coverage 0.9701 to 0.9771, SE ratio 1.1287 to 1.1649 | pass |
| `interval_calibration` | `three_arm_shared__ate_0_vs_1__noise_control` | control | a declared scale of independent noise is added to each estimate of this contrast | the SE-ratio interval must fall below the calibration band | coverage 0.8179 to 0.8345, SE ratio 0.6839 to 0.7052 | pass |
| `interval_calibration` | `three_arm_shared__ate_0_vs_1__shrunken_se_control` | control | the standard errors of this contrast are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8145 to 0.8312, SE ratio 0.6792 to 0.7006 | pass |
| `interval_calibration` | `three_arm_shared__ate_2_vs_1__corrected` | positive | the stacked curve on a three-arm law whose arms share W1 and W2 in their outcome index, in sample, reference arm 1: ATE, arm 2 versus arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9389 to 0.9490, SE ratio 0.9722 to 1.0031 | pass |
| `interval_calibration` | `three_arm_shared__ey_0__corrected` | positive | the stacked curve on a three-arm law whose arms share W1 and W2 in their outcome index, in sample, reference arm 1: mean of arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9437 to 0.9534, SE ratio 0.9810 to 1.0119 | pass |
| `interval_calibration` | `three_arm_shared__ey_1__corrected` | positive | the stacked curve on a three-arm law whose arms share W1 and W2 in their outcome index, in sample, reference arm 1: mean of arm 1 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9448 to 0.9544, SE ratio 0.9744 to 1.0054 | pass |
| `interval_calibration` | `three_arm_shared__ey_2__corrected` | positive | the stacked curve on a three-arm law whose arms share W1 and W2 in their outcome index, in sample, reference arm 1: mean of arm 2 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9431 to 0.9528, SE ratio 0.9848 to 1.0160 | pass |
| `interval_calibration` | `weighted__ate__corrected` | positive | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9541, SE ratio 0.9905 to 1.0182 | pass |
| `interval_calibration` | `weighted__ate__noise_control` | control | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: a declared scale of independent noise is added to each estimate of this contrast | the SE-ratio interval must fall below the calibration band | coverage 0.8226 to 0.8373, SE ratio 0.6988 to 0.7179 | pass |
| `interval_calibration` | `weighted__ate__shrunken_se_control` | control | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: the standard errors of this contrast are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8211 to 0.8358, SE ratio 0.6934 to 0.7127 | pass |
| `interval_calibration` | `weighted__ey0__corrected` | positive | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9427 to 0.9514, SE ratio 0.9847 to 1.0116 | pass |
| `interval_calibration` | `weighted__ey0__shipped_curve__shrunken_se_control` | control | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: the standard errors of arm-0 mean come from the shipped adaptive-propensity curve of the same fits, which omits the generated-design term | the SE-ratio interval must fall below the calibration band | coverage 0.8829 to 0.8951, SE ratio 0.8077 to 0.8300 | pass |
| `interval_calibration` | `weighted__ey1__corrected` | positive | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9451 to 0.9537, SE ratio 0.9924 to 1.0204 | pass |
| `interval_calibration` | `weighted__ey_obs__corrected` | positive | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: observed mean of the outcome | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9449 to 0.9535, SE ratio 0.9818 to 1.0092 | pass |
| `interval_calibration` | `weighted__log_or__corrected` | positive | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: log odds ratio | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9462 to 0.9547, SE ratio 0.9909 to 1.0182 | pass |
| `interval_calibration` | `weighted__log_rr__corrected` | positive | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: log risk ratio | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9459 to 0.9544, SE ratio 0.9902 to 1.0174 | pass |
| `interval_calibration` | `weighted__paf__corrected` | positive | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: population attributable fraction against reference arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9450 to 0.9535, SE ratio 0.9920 to 1.0196 | pass |
| `interval_calibration` | `weighted__par__corrected` | positive | the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample: population attributable risk against reference arm 0 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9453 to 0.9539, SE ratio 0.9925 to 1.0198 | pass |
| `interval_calibration` | `weighted_crossfit__ate__corrected` | positive | the cross-fitted stacked curve on the two-arm law under the known weight, five folds: ATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9436 to 0.9542, SE ratio 0.9807 to 1.0154 | pass |
| `interval_calibration` | `weighted_crossfit__ate__noise_control` | control | a declared scale of independent noise is added to each estimate of this contrast | the SE-ratio interval must fall below the calibration band | coverage 0.8175 to 0.8357, SE ratio 0.6942 to 0.7180 | pass |
| `interval_calibration` | `weighted_crossfit__ate__shrunken_se_control` | control | the standard errors of this contrast are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8164 to 0.8346, SE ratio 0.6867 to 0.7106 | pass |
| `interval_calibration` | `weighted_crossfit__ey0__corrected` | positive | the cross-fitted stacked curve on the two-arm law under the known weight, five folds: arm-0 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9435 to 0.9541, SE ratio 0.9892 to 1.0231 | pass |
| `interval_calibration` | `weighted_crossfit__ey1__corrected` | positive | the cross-fitted stacked curve on the two-arm law under the known weight, five folds: arm-1 mean | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9479 to 0.9581, SE ratio 0.9856 to 1.0195 | pass |
| `simultaneous_coverage` | `binary_population__pointwise_joint_control` | control | the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8237 to 0.8419 | pass |
| `simultaneous_coverage` | `binary_population__shipped_curve_band_control` | control | the max-t band built from the shipped adaptive-propensity curves of the same estimates | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8597 to 0.8762 | pass |
| `simultaneous_coverage` | `binary_population__simultaneous_band` | positive | the max-t multiplier band over the declared set of stacked estimates, from the same-row stacked influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9365 to 0.9479 | pass |
| `simultaneous_coverage` | `three_arm__pointwise_joint_control` | control | the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8102 to 0.8219 | pass |
| `simultaneous_coverage` | `three_arm__shipped_curve_band_control` | control | the max-t band built from the shipped adaptive-propensity curves of the same estimates | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8465 to 0.8573 | pass |
| `simultaneous_coverage` | `three_arm__simultaneous_band` | positive | the max-t multiplier band over the declared set of stacked estimates, from the same-row stacked influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9403 to 0.9473 | pass |
| `simultaneous_coverage` | `three_arm_crossfit__simultaneous_band` | positive | the max-t multiplier band over the declared set of stacked estimates, from the same-row stacked influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9412 to 0.9498 | pass |
| `type_i_error` | `three_arm_null__ate_0_vs_1` | positive | the three-arm law whose arms share one outcome index, so every contrast is zero: ATE, arm 0 versus arm 1 | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0563, 0.0424 to 0.0728 | pass |
| `type_i_error` | `three_arm_null__ate_2_vs_1` | positive | the three-arm law whose arms share one outcome index, so every contrast is zero: ATE, arm 2 versus arm 1 | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0612, 0.0468 to 0.0784 | pass |
<!-- /generated -->

Read the `generated_design` cells with this study's setting. The shared descriptions come from
the parent study. The `oracle_design` cell pins the true outcome regression and fits none, so the
shipped curve is its full curve. The `estimated` cell fits the outcome model per fold and reads
the stacked curve. The study gates both cells, and the phrase "the paired deficit is reported
only" refers to the deficit between them, which gates nothing.

## Declared reading

The study fixed its reading before the first run, in the module
`tests/studies/canonical_ctmle_oat_per_arm_parametric_extensions.py`.

| rule | text |
| --- | --- |
| red calibration row | a finite-sample result only when its 99% SE-ratio interval contains 1, its coverage agrees with 0.95, and the diagnosis finds no defect. Any other red row is a defect suspect, diagnosed before publication |
| negative control | each control must read red. A control row publishes `passed = True` exactly when its SE-ratio interval leaves the calibration band on the declared side, or, for a joint control, when the upper end of its joint-coverage interval is below 0.95. A control that stays inside, or straddles, cannot see its term. The study is then invalid and the design changes before any reading |
| the nine declared controls | the cells of [the next section](#negative-controls). Four ratio controls read below the band, four joint controls read below 0.95, and one ratio control reads above the band |
| `ey_obs` | the rows `three_arm__ey_obs__corrected`, `binary_population__ey_obs__corrected` and `weighted__ey_obs__corrected` are gated positive cells. The support rows `C11.binary`, `C11.k_arms` and `C11.weighted` license the interval of `ey_obs`, so a measured cell backs each claim |
| reported only | the quantities of [the section after next](#quantities-reported-and-not-gated) gate nothing |
| gate count | $G$ is 19 and the per-unit target is $0.95^{1/19}$, as [What was tested](#what-was-tested) states |
| failures | a failed replicate is recorded and not redrawn. Dropping it would select on success. Substituting the shipped standard error would mix two estimators |

## Negative controls

All nine declared controls read red. The table gives each control, the term it removes, the exact
value from the quadrature of the law, and the measured value. A ratio control reads the SE ratio.
A joint control reads the joint coverage of the declared band set. The exact joint coverage of the
stacked band is 0.950 on every band set.

| control | cell | term removed | exact value | side | measured |
| --- | --- | --- | --- | --- | --- |
| shipped curve | `three_arm__ey_1__shipped_curve__shrunken_se_control` | the generated-design term. The standard error of $\hat\psi_1$ comes from the shipped adaptive-propensity curve of the same fits | SE ratio 0.780 | below the band | `properties[interval_calibration/three_arm__ey_1__shipped_curve__shrunken_se_control]:se_ratio`, with an upper endpoint of `properties[interval_calibration/three_arm__ey_1__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` |
| pointwise joint | `three_arm__pointwise_joint_control` | the band multiplier. The pointwise 95% intervals of the five estimates are read jointly | joint coverage 0.826 | below 0.95 | `properties[simultaneous_coverage/three_arm__pointwise_joint_control]:coverage`, with an upper endpoint of `properties[simultaneous_coverage/three_arm__pointwise_joint_control]:coverage_ci_upper` |
| shipped band | `three_arm__shipped_curve_band_control` | the stacked curves. The max-t band comes from the shipped curves of the same five estimates | joint coverage 0.861 | below 0.95 | `properties[simultaneous_coverage/three_arm__shipped_curve_band_control]:coverage`, with an upper endpoint of `properties[simultaneous_coverage/three_arm__shipped_curve_band_control]:coverage_ci_upper` |
| cross term dropped | `three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control` | the cross-arm covariance. The standard error of `ate[0 vs 1]` is $\sqrt{se_0^2 + se_1^2}$ on a law whose arm curves correlate | SE ratio 1.163 | above the band | `properties[interval_calibration/three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control]:se_ratio`, with a lower endpoint of `properties[interval_calibration/three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control]:se_ratio_ci_lower` |
| shipped curve | `binary_population__par__shipped_curve__shrunken_se_control` | the generated-design term, read on `par` | SE ratio 0.766 | below the band | `properties[interval_calibration/binary_population__par__shipped_curve__shrunken_se_control]:se_ratio`, with an upper endpoint of `properties[interval_calibration/binary_population__par__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` |
| pointwise joint | `binary_population__pointwise_joint_control` | the band multiplier, over the six estimates | joint coverage 0.841 | below 0.95 | `properties[simultaneous_coverage/binary_population__pointwise_joint_control]:coverage`, with an upper endpoint of `properties[simultaneous_coverage/binary_population__pointwise_joint_control]:coverage_ci_upper` |
| shipped band | `binary_population__shipped_curve_band_control` | the stacked curves, over the six estimates | joint coverage 0.872 | below 0.95 | `properties[simultaneous_coverage/binary_population__shipped_curve_band_control]:coverage`, with an upper endpoint of `properties[simultaneous_coverage/binary_population__shipped_curve_band_control]:coverage_ci_upper` |
| shipped curve | `three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control` | the generated-design term, cross-fitted | SE ratio 0.780 | below the band | `properties[interval_calibration/three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control]:se_ratio`, with an upper endpoint of `properties[interval_calibration/three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` |
| shipped curve | `weighted__ey0__shipped_curve__shrunken_se_control` | the generated-design term, under the known weight | SE ratio 0.819 | below the band | `properties[interval_calibration/weighted__ey0__shipped_curve__shrunken_se_control]:se_ratio`, with an upper endpoint of `properties[interval_calibration/weighted__ey0__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` |

The exact values come from the quadrature of each law. The design test
`tests/unit/test_ctmle_oat_per_arm_parametric_extensions_study_design.py` recomputes them. The
cross-fitted group declares the in-sample value 0.780. The shipped band control uses the class `ShippedOutcome`, which the support
predicate refuses. Its fits have the same estimates as the stacked fits and keep the shipped
curve.

The pointwise and shipped band controls of `three_arm_crossfit` are not declared. That group
declares the band cell and one ratio control. The cross-fitted three-arm band is the band of the
earlier three-arm study, and the stacked band covers 0.9497 on its set.

The harness also derives a `shrunken_se_control` and a `noise_control` from one contrast row of
each of seven groups: `three_arm`, `three_arm_shared`, `binary_population`, `three_arm_crossfit`,
`weighted`, `weighted_crossfit` and `composition`. The shrunken control multiplies the standard
error by the declared factor `margin:shrunken_se_factor`. All fourteen derived controls read below
the band. The `noise_control` of `three_arm_shared` uses the asymptotic SD of the contrast from a
20-node Hermite rule. That rule converges slowly on this law, and the bound sits about 0.08%
low. The control reads far inside its side of the band, so the study leaves the rule as it is.

The sharp-null cells of `three_arm_null` can see the stacked term. The cells test the two
contrasts of arm 1 against arms 0 and 2.

## Quantities reported and not gated

These quantities describe the fit, and no verdict reads them.

| quantity | cell | what it shows | measured |
| --- | --- | --- | --- |
| covariance ratio | `three_arm_shared__ate_0_vs_1__corrected` | the Monte Carlo covariance of the two arm means against the mean reported covariance. The ratio shows whether the stack reports the cross-arm term | `properties[interval_calibration/three_arm_shared__ate_0_vs_1__corrected]:covariance_ratio`, with an interval from `properties[interval_calibration/three_arm_shared__ate_0_vs_1__corrected]:covariance_ratio_ci_lower` to `properties[interval_calibration/three_arm_shared__ate_0_vs_1__corrected]:covariance_ratio_ci_upper` |
| paired deficit | `three_arm_active__estimated__ate_1_vs_0` | the SE-ratio deficit of the estimated design against the pinned design, contrast 1 versus 0 | an interval from `properties[generated_design/three_arm_active__estimated__ate_1_vs_0]:se_ratio_deficit_lower` to `properties[generated_design/three_arm_active__estimated__ate_1_vs_0]:se_ratio_deficit_upper` |
| paired deficit | `three_arm_active__estimated__ate_2_vs_0` | the same deficit, contrast 2 versus 0 | an interval from `properties[generated_design/three_arm_active__estimated__ate_2_vs_0]:se_ratio_deficit_lower` to `properties[generated_design/three_arm_active__estimated__ate_2_vs_0]:se_ratio_deficit_upper` |

The deficit is the SE ratio of the estimated design minus that of the pinned design. It is
negative on both contrasts, so the estimated design reads closer to 1. The pinned outcome
regression keeps the shipped curve, which has no generated-design term to omit. The pilot
diagnosis attributes the excess of the pinned design to the shipped curve at about 400 training
rows per arm.

## Deviations from the plan

The plan is the X58 plan, section 13.6. The table lists each deviation and its reason.

| plan | study | reason |
| --- | --- | --- |
| each group is sized to the per-unit target 0.99730 | the group `three_arm_generated_design` has 80,000 replications, and its predicted pass probability is 0.99321, below the target | the pooled rule gives no size for this group. Its four rows sit at SE ratios near 1.01 (estimated) and 1.03 (oracle design). The pooled $\tau_{80}$ of 0.0106 gives a chance near 0.7% that a drawn true ratio passes 1.07, whatever the size. The figure rises by about 0.0002 per 2,000 replications at 80,000. Every row lies inside the band at the declared size. The coordinator set the largest size searched and did not trade quality for cost |
| the budgets follow the pilot step 2 | the budgets follow the pilot steps 2 and 4, pooled over independent seeds | step 2 alone gave no budget up to 80,000 for `three_arm_crossfit` and `three_arm_generated_design`, and a study-wise 0.7675 at the plan budgets. The model carries the sampling error of its own pilot, so the pass probability stops rising with the size. The pooled model gives 0.9844 at the declared budgets |
| `binary_population` has 9,600 replications and `three_arm_crossfit` 17,600 | 11,200 and 18,800 | the pooled pilot gives the smallest multiple of 400 at the target. The predicted values are 0.99738 and 0.99735 |
| `three_arm_null` has 1,200 replications | 1,600 | at 1,200 replications, with every allowed loss counted as a rejection, the two rows pass with probability 0.95525 at a true rate of 0.055. At 1,600 the figure is 0.99490, and 0.99982 at 0.05 |
| the design review states that 1,600 primary replications hold the target down to a true coverage of 0.94 | the primary phase holds the target down to 0.945 | the pass probability of 11 rows at 1,600 replications is 0.99990 at 0.945 and 0.99592 at 0.94, below the target. At 1,200 it is 0.93764 at 0.94. The module comment and the design test state the corrected figures |
| the `ey_obs` interval is reported and gates nothing (plan Q7) | three `ey_obs` cells are gated | three support rows license the Wald interval of `ey_obs`. A supported claim needs a measured cell |
| the pilot pools $\tau_{80}$ over the arms of each group (design review) | the pilot pools $\tau_{80}$ over the study, as the plan and the earlier study do | the study-wise value is 0.0106. The pilot report prints the group-only values |
| the `noise_control` bound uses the exact SD of the contrast | the `three_arm_shared` bound uses a 20-node Hermite rule | the rule converges slowly on this law (0.9203 at 20 nodes, 0.9210 at 30), so the bound sits about 0.08% low. The control reads far from the band edge |
| the clause binding of `interval_calibration, coverage in calibration band` is `bias`, copied from the parametric study, and the two `simultaneous_coverage` clauses are `se_low` | the first clause binds to `affine`, `generated_design, coverage in calibration band` binds to `shift/0.7`, and the two `simultaneous_coverage` clauses keep `se_low` | the binding search on the committed rows chose each perturbation, as [Reproduction](#reproduction) states |

No replicate was lost in either phase, so no deviation concerns the failure bound.

## Result

Every one of the `property_cells_total` cells passes, and so does each of the
`independent_tests_total` primary tests. The lowest primary coverage is `min_coverage`, with a lower
endpoint of `min_coverage_ci_lower`. The primary SE-ratio endpoints run from `min_se_ratio_ci_lower`
to `max_se_ratio_ci_upper`. The largest standardized bias is `max_standardized_bias`. The run lost
`subject_failures` primary fits, and every property cell reports `failed_replicates` of 0 against
its allowance of `floor(0.01 R)`.

The stacked curve calibrates at three arms, in sample and at five folds. It calibrates under the
known weight, in sample and at five folds, and at three arms under the weight at five folds. It
calibrates the population targets `ey_obs`, `par` and `paf`.

The joint coverage of each of the three bands lies inside the coverage band. The pointwise intervals and the shipped band read below
0.95 on the same estimates, so the correction is what the study measures. The shipped curve
reads below the band in all four ratio controls. The interval without the cross-arm covariance
reads above the band.

The generated-design pair passes on both contrasts, and so does the sharp null at three arms.
The oracle-design rows read above the estimated rows. The pilot diagnosis attributes this to the
shipped curve at about 400 training rows per arm.

The band cell of `three_arm` covers `properties[simultaneous_coverage/three_arm__simultaneous_band]:coverage`
against the nominal 0.95. The band cell of `three_arm_crossfit` covers
`properties[simultaneous_coverage/three_arm_crossfit__simultaneous_band]:coverage`, and that of
`binary_population` covers `properties[simultaneous_coverage/binary_population__simultaneous_band]:coverage`.

The sharp-null cells reject at `properties[type_i_error/three_arm_null__ate_0_vs_1]:rejection_rate`
and `properties[type_i_error/three_arm_null__ate_2_vs_1]:rejection_rate`. Their upper endpoints
are `properties[type_i_error/three_arm_null__ate_0_vs_1]:rejection_ci_upper` and
`properties[type_i_error/three_arm_null__ate_2_vs_1]:rejection_ci_upper`, against the ceiling
`margin:type_i_ceiling`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications |
| `n` | 1500 | observations per primary replication |
| `independent_tests_total` | 11 | primary tests against truth |
| `independent_tests_passed` | 11 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 73 | repeated-sampling property cells |
| `property_cells_passed` | 73 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0173 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9481 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9321 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9441 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0738 | highest bootstrap primary SE-ratio endpoint |
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
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
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
| `margin:subject_failures_allowed` | 16 | replicates one primary cell may lose |
| `margin:type_i_ceiling` | 0.1000 | the rate the null cell must bound |
| `properties[interval_calibration/three_arm__ey_1__shipped_curve__shrunken_se_control]:se_ratio` | 0.7785 | SE ratio, shipped-curve control, three arms |
| `properties[interval_calibration/three_arm__ey_1__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` | 0.7868 | its 99% upper endpoint |
| `properties[simultaneous_coverage/three_arm__pointwise_joint_control]:coverage` | 0.8161 | joint coverage, pointwise control, three arms |
| `properties[simultaneous_coverage/three_arm__pointwise_joint_control]:coverage_ci_upper` | 0.8219 | its 99% upper endpoint |
| `properties[simultaneous_coverage/three_arm__shipped_curve_band_control]:coverage` | 0.8520 | joint coverage, shipped band control, three arms |
| `properties[simultaneous_coverage/three_arm__shipped_curve_band_control]:coverage_ci_upper` | 0.8573 | its 99% upper endpoint |
| `properties[interval_calibration/three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control]:se_ratio` | 1.1464 | SE ratio, cross-term control |
| `properties[interval_calibration/three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control]:se_ratio_ci_lower` | 1.1287 | its 99% lower endpoint |
| `properties[interval_calibration/binary_population__par__shipped_curve__shrunken_se_control]:se_ratio` | 0.7677 | SE ratio, shipped-curve control, `par` |
| `properties[interval_calibration/binary_population__par__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` | 0.7810 | its 99% upper endpoint |
| `properties[simultaneous_coverage/binary_population__pointwise_joint_control]:coverage` | 0.8329 | joint coverage, pointwise control, population targets |
| `properties[simultaneous_coverage/binary_population__pointwise_joint_control]:coverage_ci_upper` | 0.8419 | its 99% upper endpoint |
| `properties[simultaneous_coverage/binary_population__shipped_curve_band_control]:coverage` | 0.8681 | joint coverage, shipped band control, population targets |
| `properties[simultaneous_coverage/binary_population__shipped_curve_band_control]:coverage_ci_upper` | 0.8762 | its 99% upper endpoint |
| `properties[interval_calibration/three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control]:se_ratio` | 0.7914 | SE ratio, shipped-curve control, cross-fitted |
| `properties[interval_calibration/three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` | 0.8021 | its 99% upper endpoint |
| `properties[interval_calibration/weighted__ey0__shipped_curve__shrunken_se_control]:se_ratio` | 0.8188 | SE ratio, shipped-curve control, weighted |
| `properties[interval_calibration/weighted__ey0__shipped_curve__shrunken_se_control]:se_ratio_ci_upper` | 0.8300 | its 99% upper endpoint |
| `properties[interval_calibration/three_arm_shared__ate_0_vs_1__corrected]:covariance_ratio` | 1.0314 | Monte Carlo covariance over reported covariance |
| `properties[interval_calibration/three_arm_shared__ate_0_vs_1__corrected]:covariance_ratio_ci_lower` | 0.9457 | its 99% lower endpoint |
| `properties[interval_calibration/three_arm_shared__ate_0_vs_1__corrected]:covariance_ratio_ci_upper` | 1.1327 | its 99% upper endpoint |
| `properties[generated_design/three_arm_active__estimated__ate_1_vs_0]:se_ratio_deficit_lower` | -0.0192 | paired SE-ratio deficit, contrast 1 versus 0, lower endpoint |
| `properties[generated_design/three_arm_active__estimated__ate_1_vs_0]:se_ratio_deficit_upper` | -0.0102 | paired SE-ratio deficit, contrast 1 versus 0, upper endpoint |
| `properties[generated_design/three_arm_active__estimated__ate_2_vs_0]:se_ratio_deficit_lower` | -0.0203 | paired SE-ratio deficit, contrast 2 versus 0, lower endpoint |
| `properties[generated_design/three_arm_active__estimated__ate_2_vs_0]:se_ratio_deficit_upper` | -0.0116 | paired SE-ratio deficit, contrast 2 versus 0, upper endpoint |
| `properties[simultaneous_coverage/three_arm__simultaneous_band]:coverage` | 0.9438 | joint coverage of the band, three arms, in sample |
| `properties[simultaneous_coverage/three_arm_crossfit__simultaneous_band]:coverage` | 0.9456 | joint coverage of the band, three arms, cross-fitted |
| `properties[simultaneous_coverage/binary_population__simultaneous_band]:coverage` | 0.9424 | joint coverage of the band, population targets |
| `properties[type_i_error/three_arm_null__ate_0_vs_1]:rejection_rate` | 0.0563 | rejection rate, arm 0 against arm 1 |
| `properties[type_i_error/three_arm_null__ate_0_vs_1]:rejection_ci_upper` | 0.0728 | its 99% upper endpoint |
| `properties[type_i_error/three_arm_null__ate_2_vs_1]:rejection_rate` | 0.0612 | rejection rate, arm 2 against arm 1 |
| `properties[type_i_error/three_arm_null__ate_2_vs_1]:rejection_ci_upper` | 0.0784 | its 99% upper endpoint |

## Limits

| limit | what it means |
| --- | --- |
| Three arms only | The study measures three arms. A fit at four arms or more matches no support row and is `"experimental"`. The theory covers any arm count (plan 13.2), and no cell measures it |
| Five folds only | The study fits `n_folds=5`. The support rows match any fold count inside the cap of 20 folds, and the preflight labels a fit at another count `"experimental"`. The interval also needs 30 rows in each held-out fold |
| One cross-fitting draw | The study measures `repeats=1`. A fit with `repeats` above 1 and the default `simultaneous=True` raises a `CapabilityError`, because the band would use the curve of a central draw and not the split-adjusted median estimator. A fit with `simultaneous=False` or one draw builds its intervals as the crossfit study states. A fit with `repeats` above 1 under a weight matches no row and is `"experimental"` |
| Known weight only | The study measures a known fixed weight $0.5 + 1\{W_1 > 0\}$, in sample at two arms and cross-fitted at two and three arms. An estimated weight takes the status `"estimated_weight_plugin"` with the label `"experimental"`, and its curve equals the fixed-weight curve. A weighted in-sample three-arm fit matches no row and is `"experimental"` |
| Reference arm by group | The groups fit `reference="1"` or `reference="0"` as the table of [What was tested](#what-was-tested) states. The curve of a contrast does not depend on the label, and no cell reads the other reference |
| The measured band sets | The study measures a band over each set named in [What was tested](#what-was-tested). A band over another subset of these estimates rests on the same multiplier argument (plan 13.2) and has no measured cell. A band over `rr` or `or` is `"experimental"`, because no band cell reads them |
| Complete data, binary outcome | The study measures a binary outcome with no stratum, missing value or cluster. A continuous outcome with declared `q_bounds` has no row and is `"experimental"` |
| Two parametric learners only | Only `ParametricOutcome` and `ParametricMechanism` are admitted, as exact classes. A scikit-learn learner keeps the `"generated_design_plugin"` status, and [F19](../../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) owns its interval |
| Correct outcome model | The laws have a correct outcome model. The pseudo-true behaviour under a wrong model is measured in the [crossfit study](outcome-adaptive-per-arm-c-tmle-parametric-crossfit.md#the-coverage-of-the-limit) at two arms and is not measured at three arms or under a weight |
| Default `alpha` only | Every cell runs at the default `alpha`. No cell sets `alpha` below it |
| Balanced folds | Every studied $n$ is a multiple of 5, so the folds are exactly balanced. The unbalanced fold map is checked by CW2 in `tests/unit/test_oat_crossfit_stack.py`, not by this study |
| Generated-design pair at its size | The group `three_arm_generated_design` has a predicted pass probability of 0.99321, below the per-unit target 0.99730. The pair carries the weakest margin of the study. Its upper 99% SE-ratio bound lies 0.017 under the band edge in the pilot |
| In-sample band below the exact coverage | The in-sample three-arm band covered 0.9402 at pilot step 2 and 0.9439 at step 4, against the exact 0.9505. Both lie inside the coverage band. The [Result](#result) section gives the declared-run value |
| No comparator | The evidence is the exact truth of each law, the calibration bands and the negative controls. No R program reports the corrected curve |

## Reproduction

Run `python -m tests.canonical.ctmle_oat_per_arm_parametric_extensions.regenerate` to regenerate
the artifacts. The directory
[`tests/canonical/ctmle_oat_per_arm_parametric_extensions`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/canonical/ctmle_oat_per_arm_parametric_extensions)
holds `manifest.json`, `run.log` and the committed results. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_oat_per_arm_parametric_extensions/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_oat_per_arm_parametric_extensions/properties.csv)
carry every published row. The run used commit `11b4579b` on a clean tree and 1,750 wall seconds
on four cores. The run log records that the run resumed from saved parts of the same code.

The binding search of the `property-differential` job chose the perturbation of each clause in
`tests/canonical/clause-bindings.csv` for this study. The command is
`python -m tests.canonical.property_differential --bindings <dir> --subjects ctmle-oat-per-arm-parametric-extensions`.
The search bound 11 of 11 clauses with the shared kinds and needed no targeted kind. It took 1,876
seconds.

It found the seeded `affine` map for `interval_calibration, coverage in calibration band`
where the plan copied `bias`, and `shift/0.7` for `generated_design, coverage in calibration band`
where the plan copied `affine`. It kept `se_low` for both `simultaneous_coverage` clauses. The
design review expected that `se_low` could not decide `joint coverage below nominal`. The search
shows that it decides that clause alone on these rows.
