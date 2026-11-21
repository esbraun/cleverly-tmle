# Score assessment compositions

This study validates the assessment of an external score (`assess_score` with `compare=`) in five
compositions that [score-assessment](score-assessment.md) did not reach: sampling weights,
clustered rows, outcomes missing at random, the pair of arms of a three-arm treatment, and an ATE
fit that is not cross-fitted. Each composition is the shipped paired design of score-assessment:
an estimated treatment mechanism, a difference-scale ATE fit, a binary outcome, and the scores
$S = 0.2 + 0.3 W_1$ and $S_B = S + 0.15 \sin(\pi W_3)$. The targets are each score's risk
difference against the best constant, its calibration intercept and slope, and the paired risk
difference of the two scores. [External scores](../conditional-effects.md#external-scores) gives
the targets, the estimator and its conditions.

**No canonical implementation is compared.** No maintained package reports a score's risk
difference or a paired risk difference of two scores with a TMLE interval in these compositions. A
zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | conditional-projection's laws: `weighted` (sampling weights $0.25 + 1.75 \operatorname{expit}(6 W_3)$), `clustered` (800 clusters of 5 rows with $W_1$ and the treatment at cluster level), `missing` (outcomes missing at random), `multi_arm` (three arms, the pair $(2, 0)$) and `observational` |
| groups | `assess_weighted_2k`, `assess_clustered_4k`, `assess_missing_4k`, `assess_arms3_4k`, and `assess_insample_2k` (the ATE fit in sample) |
| arms | each group's rows form two calibration arms: `risk` (the $S$ and $S_B$ risk differences and the paired row) and `cal` (the $S$ and $S_B$ calibration intercepts and slopes) |
| learners | a spline GLM per arm for $\mu$, and logistic GLMs for the treatment and observation mechanisms; the ATE fit is the shipped `Linear(["W1", "W3"])` contrast |
| truth | Gauss-Legendre quadrature over $U(-1, 1)^3$, the weighted law's truth under the target population; checked in the fast tier |
| primary budget | 2,400 replications of `assess_weighted_2k` at n = 2,000, the paired row |
| calibration budget | R = 6,000 per group, computed before the run from a 3,200-draw pilot on a disjoint stream, with the `k3_cal` arm extended by a second 3,200-draw pilot (`tests/studies/score_assessment_compositions_pilot.csv`) |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20261521` and `resampling_seed=20261522`; the run, the pilot and the smoke each draw from their own labelled streams, with fold seeds distinct per group |
| policy | `gated` |

The laws, groups, learners, budgets, controls and seeds were fixed before any run, in the P1 PR 2
plan, its reviews and `tests/studies/score_assessment_compositions.py`.
`tests/unit/test_score_assessment_compositions_design.py` checks the truths, the declared cells,
the pilot rule and the budget.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000 | `risk_difference[S - SB]` | the paired risk difference: S's mean squared error for the CATE minus SB's | `cleverly` score assessment in a composition: one pooled TMLE fluctuation of the ATE fit along the span of the scores' features, with substitution estimates of each score's risk difference, calibration and the paired risk difference, and influence-curve Wald intervals | -0.000217 to 0.000379 | 0.9487 | 1.0078 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | for each of the seven parameters of the five groups (R = 6,000), does the SE-ratio interval lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98 | the reported SE of the weighted $S$ risk difference times 0.70, and the row-level SE that ignores the clusters on the clustered $S$ risk difference and calibration slope, must fall below the band |

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `assess_arms3_4k_SB_calibration_intercept__k3_cal` | positive | the three-arm law, the pair (2, 0) of the ATE fit, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the intercept of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9585, SE ratio 0.9937 to 1.0409 | pass |
| `interval_calibration` | `assess_arms3_4k_SB_calibration_slope__k3_cal` | positive | the three-arm law, the pair (2, 0) of the ATE fit, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the slope of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9487 to 0.9625, SE ratio 0.9987 to 1.0455 | pass |
| `interval_calibration` | `assess_arms3_4k_SB_risk_difference__k3_risk` | positive | the three-arm law, the pair (2, 0) of the ATE fit, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of SB's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9494 to 0.9632, SE ratio 0.9937 to 1.0388 | pass |
| `interval_calibration` | `assess_arms3_4k_S_calibration_intercept__k3_cal` | positive | the three-arm law, the pair (2, 0) of the ATE fit, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the intercept of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9468 to 0.9608, SE ratio 0.9966 to 1.0430 | pass |
| `interval_calibration` | `assess_arms3_4k_S_calibration_slope__k3_cal` | positive | the three-arm law, the pair (2, 0) of the ATE fit, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the slope of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9478 to 0.9618, SE ratio 0.9978 to 1.0450 | pass |
| `interval_calibration` | `assess_arms3_4k_S_risk_difference__k3_risk` | positive | the three-arm law, the pair (2, 0) of the ATE fit, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of S's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9496 to 0.9633, SE ratio 0.9902 to 1.0360 | pass |
| `interval_calibration` | `assess_arms3_4k_paired__k3_risk` | positive | the three-arm law, the pair (2, 0) of the ATE fit, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the paired risk difference, S's MSE for the CATE minus SB's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9478 to 0.9618, SE ratio 0.9964 to 1.0433 | pass |
| `interval_calibration` | `assess_clustered_4k_SB_calibration_intercept__cl_cal` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the intercept of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9379 to 0.9531, SE ratio 0.9711 to 1.0173 | pass |
| `interval_calibration` | `assess_clustered_4k_SB_calibration_slope__cl_cal` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the slope of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9389 to 0.9540, SE ratio 0.9623 to 1.0096 | pass |
| `interval_calibration` | `assess_clustered_4k_SB_risk_difference__cl_risk` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of SB's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9400 to 0.9550, SE ratio 0.9656 to 1.0134 | pass |
| `interval_calibration` | `assess_clustered_4k_S_calibration_intercept__cl_cal` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the intercept of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9391 to 0.9542, SE ratio 0.9684 to 1.0148 | pass |
| `interval_calibration` | `assess_clustered_4k_S_calibration_slope__cl_cal` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the slope of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9384 to 0.9536, SE ratio 0.9648 to 1.0116 | pass |
| `interval_calibration` | `assess_clustered_4k_S_calibration_slope__row_variance_control` | control | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the row-level SE that ignores the clusters, sd(IC) / sqrt(n), for the slope of the calibration line from S to the CATE | the SE-ratio interval must fall below the calibration band | coverage 0.8538 to 0.8766, SE ratio 0.7482 to 0.7848 | pass |
| `interval_calibration` | `assess_clustered_4k_S_risk_difference__cl_risk` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of S's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9416 to 0.9564, SE ratio 0.9694 to 1.0173 | pass |
| `interval_calibration` | `assess_clustered_4k_S_risk_difference__row_variance_control` | control | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the row-level SE that ignores the clusters, sd(IC) / sqrt(n), for S's MSE for the CATE minus the best constant's | the SE-ratio interval must fall below the calibration band | coverage 0.8541 to 0.8770, SE ratio 0.7443 to 0.7814 | pass |
| `interval_calibration` | `assess_clustered_4k_paired__cl_risk` | positive | the clustered law, 800 clusters of 5 rows with W1 and the treatment at cluster level, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the paired risk difference, S's MSE for the CATE minus SB's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9384 to 0.9536, SE ratio 0.9638 to 1.0125 | pass |
| `interval_calibration` | `assess_insample_2k_SB_calibration_intercept__in_cal` | positive | the observational law, an ATE fit that is not cross-fitted, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the intercept of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9416 to 0.9564, SE ratio 0.9759 to 1.0229 | pass |
| `interval_calibration` | `assess_insample_2k_SB_calibration_slope__in_cal` | positive | the observational law, an ATE fit that is not cross-fitted, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the slope of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9377 to 0.9529, SE ratio 0.9666 to 1.0134 | pass |
| `interval_calibration` | `assess_insample_2k_SB_risk_difference__in_risk` | positive | the observational law, an ATE fit that is not cross-fitted, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of SB's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9571, SE ratio 0.9677 to 1.0153 | pass |
| `interval_calibration` | `assess_insample_2k_S_calibration_intercept__in_cal` | positive | the observational law, an ATE fit that is not cross-fitted, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the intercept of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9419 to 0.9567, SE ratio 0.9741 to 1.0219 | pass |
| `interval_calibration` | `assess_insample_2k_S_calibration_slope__in_cal` | positive | the observational law, an ATE fit that is not cross-fitted, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the slope of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9356 to 0.9511, SE ratio 0.9600 to 1.0067 | pass |
| `interval_calibration` | `assess_insample_2k_S_risk_difference__in_risk` | positive | the observational law, an ATE fit that is not cross-fitted, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of S's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9423 to 0.9570, SE ratio 0.9617 to 1.0077 | pass |
| `interval_calibration` | `assess_insample_2k_paired__in_risk` | positive | the observational law, an ATE fit that is not cross-fitted, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the paired risk difference, S's MSE for the CATE minus SB's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9421 to 0.9568, SE ratio 0.9790 to 1.0246 | pass |
| `interval_calibration` | `assess_missing_4k_SB_calibration_intercept__mis_cal` | positive | the observational law with outcomes missing at random, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the intercept of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9441 to 0.9585, SE ratio 0.9743 to 1.0226 | pass |
| `interval_calibration` | `assess_missing_4k_SB_calibration_slope__mis_cal` | positive | the observational law with outcomes missing at random, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the slope of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9434 to 0.9579, SE ratio 0.9753 to 1.0205 | pass |
| `interval_calibration` | `assess_missing_4k_SB_risk_difference__mis_risk` | positive | the observational law with outcomes missing at random, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of SB's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9419 to 0.9567, SE ratio 0.9707 to 1.0163 | pass |
| `interval_calibration` | `assess_missing_4k_S_calibration_intercept__mis_cal` | positive | the observational law with outcomes missing at random, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the intercept of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9459 to 0.9601, SE ratio 0.9841 to 1.0323 | pass |
| `interval_calibration` | `assess_missing_4k_S_calibration_slope__mis_cal` | positive | the observational law with outcomes missing at random, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the slope of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9480 to 0.9619, SE ratio 0.9857 to 1.0332 | pass |
| `interval_calibration` | `assess_missing_4k_S_risk_difference__mis_risk` | positive | the observational law with outcomes missing at random, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of S's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9448 to 0.9591, SE ratio 0.9744 to 1.0216 | pass |
| `interval_calibration` | `assess_missing_4k_paired__mis_risk` | positive | the observational law with outcomes missing at random, S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 4,000: the influence-curve Wald interval of the paired risk difference, S's MSE for the CATE minus SB's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9377 to 0.9529, SE ratio 0.9517 to 0.9963 | pass |
| `interval_calibration` | `assess_weighted_2k_SB_calibration_intercept__w_cal` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the intercept of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9405 to 0.9554, SE ratio 0.9738 to 1.0196 | pass |
| `interval_calibration` | `assess_weighted_2k_SB_calibration_slope__w_cal` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the slope of the calibration line from SB to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9423 to 0.9570, SE ratio 0.9762 to 1.0223 | pass |
| `interval_calibration` | `assess_weighted_2k_SB_risk_difference__w_risk` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of SB's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9576, SE ratio 0.9769 to 1.0247 | pass |
| `interval_calibration` | `assess_weighted_2k_S_calibration_intercept__w_cal` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the intercept of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9391 to 0.9542, SE ratio 0.9740 to 1.0217 | pass |
| `interval_calibration` | `assess_weighted_2k_S_calibration_slope__w_cal` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the slope of the calibration line from S to the CATE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9379 to 0.9531, SE ratio 0.9743 to 1.0217 | pass |
| `interval_calibration` | `assess_weighted_2k_S_risk_difference__shrunken_se_control` | control | the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the reported SE times 0.70, for S's MSE for the CATE minus the best constant's | the SE-ratio interval must fall below the calibration band | coverage 0.8182 to 0.8433, SE ratio 0.6837 to 0.7165 | pass |
| `interval_calibration` | `assess_weighted_2k_S_risk_difference__w_risk` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of S's MSE for the CATE minus the best constant's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9432 to 0.9577, SE ratio 0.9768 to 1.0241 | pass |
| `interval_calibration` | `assess_weighted_2k_paired__w_risk` | positive | the observational law with sampling weights 0.25 + 1.75 expit(6 W3), S = 0.2 + 0.3 W1 against SB = S + 0.15 sin(pi W3), difference scale, n = 2,000: the influence-curve Wald interval of the paired risk difference, S's MSE for the CATE minus SB's | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9393 to 0.9543, SE ratio 0.9780 to 1.0253 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. Each of the 35
calibration positives has an SE-ratio interval inside 0.93 to 1.07 and a coverage interval inside
0.92 to 0.98. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 1 of 1 | `performance-tests.csv` |
| gated property cells passed | 38 of 38 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

The calibration SE ratios run from 0.9734 to 1.0216 and the coverages from 0.9437 to 0.9568. The
shrunken control reads an SE ratio near 0.70, and the row-level SE reads ratios near 0.76 on the
clustered law, with coverage 0.866.

One passing row has a 99% SE-ratio interval wholly below 1: the paired row of
`assess_missing_4k`, at 0.9734 with an upper end of 0.9963. Its pilot read 1.006 on 6,400 draws
of a disjoint stream. If every row's true ratio were 1, the chance that at least one of 35
intervals at 99% excludes 1 would be about 0.30. The row's coverage interval contains 0.95.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 1 | parameter tests against the truth |
| `independent_tests_passed` | 1 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 38 | gated repeated-sampling property cells |
| `property_cells_passed` | 38 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0143 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9487 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9360 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9712 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0479 | highest bootstrap primary SE-ratio endpoint |
| `properties[interval_calibration/assess_missing_4k_paired__mis_risk]:se_ratio` | 0.9734 | lowest calibration SE ratio |
| `properties[interval_calibration/assess_missing_4k_paired__mis_risk]:se_ratio_ci_upper` | 0.9963 | its upper endpoint |
| `properties[interval_calibration/assess_arms3_4k_SB_calibration_slope__k3_cal]:se_ratio` | 1.0216 | highest calibration SE ratio |
| `properties[interval_calibration/assess_insample_2k_S_calibration_slope__in_cal]:coverage` | 0.9437 | lowest calibration coverage |
| `properties[interval_calibration/assess_arms3_4k_S_risk_difference__k3_risk]:coverage` | 0.9568 | highest calibration coverage |
| `properties[interval_calibration/assess_weighted_2k_S_risk_difference__shrunken_se_control]:se_ratio` | 0.6999 | shrunken control |
| `properties[interval_calibration/assess_clustered_4k_S_risk_difference__row_variance_control]:se_ratio` | 0.7625 | row-level SE control, risk difference |
| `properties[interval_calibration/assess_clustered_4k_S_calibration_slope__row_variance_control]:se_ratio` | 0.7661 | row-level SE control, calibration slope |
| `properties[interval_calibration/assess_clustered_4k_S_risk_difference__row_variance_control]:coverage` | 0.8658 | row-level SE control coverage |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio sanity screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio sanity screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor of the shrunken control |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 24 | failed primary fits the bound allows |
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
| There is no cross-implementation evidence | No maintained package reports these parameters with a TMLE interval. The rows rest on accuracy against the truth and on the theory properties |
| One configuration | Every composition was measured on the difference scale, with a binary outcome and an estimated treatment mechanism. The same composition on the log ratio, with a continuous outcome or with a known mechanism reads experimental |
| One composition at a time | Each group adds one composition to the base design. Two together, such as weights with clusters or missing outcomes in sample, read experimental: `score_assessment_codes` lifts a listed composition only when it is the fit's one measured departure (`tests/unit/test_score_assessment_tmle.py`, t20) |
| The measured forms | One weight law, 800 clusters of 5 rows, one response law, the pair $(2, 0)$ of three arms, and the in-sample ATE at n = 2,000. Fewer than 40 clusters read experimental, because the interval then takes a $t(J - 2)$ reference that no study measures |
| Fixed scores | The two scores are fixed functions fitted on other rows. The interval covers one pair named before the fit, not the best of several |
| Consistent learners | Every group fits a spline GLM per arm and logistic mechanisms, consistent for its law; the default learner library is not measured |
| Laws and sizes | n = 2,000 and 4,000 on smooth laws. Smaller n rests on the theory |

## Reproduction

The declared run is `python -m tests.canonical.score_assessment_compositions.regenerate`, whose
docstring gives the run form and the red-cell reading. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment_compositions/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment_compositions/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment_compositions/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/score_assessment_compositions/properties.csv)
carry every published row.
