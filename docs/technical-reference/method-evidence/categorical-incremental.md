# Categorical incremental policies

This study validates the categorical odds tilt of `Incremental({level: factor})`. Family P fits
point `TMLE` on a three-level treatment. Family L fits `IPSI` with two three-level tilt nodes.
The [point estimator](../point-treatment-tmle.md#incremental-propensity-score-interventions) and the
[longitudinal estimator](../longitudinal-tmle.md#incremental-interventions-over-time) state the
construction, its remainder and its exact-law evidence.

No maintained package ships this construction with its influence curve. `npcausal::ipsi` fits two
arms only, and `imtp` 0.1.0 omits the derivative term. So no canonical implementation is compared.
The study rests on exact truths and on repeated-sampling properties. It is Python property
evidence.

## What was fitted

| scenario | law and estimands | folds |
| --- | --- | --- |
| `point_in_sample` | law P: four baseline values, three treatment levels, a continuous outcome. The three tilted means and the two contrasts with the natural course | one |
| `point_crossfit` | law P, the same estimands | five |
| `longitudinal_dropout_folds_5` | law L: a binary baseline covariate, a three-level treatment, dropout, a binary time-varying covariate, a second three-level treatment, dropout and a continuous outcome. The three plan means and the two contrasts | five |
| `longitudinal_folds_5` | law L0: law L without dropout, the same estimands | five |

Plan `T_a` multiplies the odds of the second level by 2 and of the third by 0.5, against the
first. Plan `T_b` multiplies them by 0.5 and 2. The natural course keeps the first level at factor
1. Family L applies each plan at both treatment nodes. Every nuisance is a saturated cell
frequency or a cell mean. The laws are exact finite sums in `tests/discrete_law_categorical_ipsi_point.py`
and `tests/discrete_law_categorical_ipsi_longitudinal.py`.

| law | key numbers |
| --- | --- |
| P | treatment propensities from 0.10 to 0.70. The contrast `T_a` has truth -0.8124 and efficient influence-curve SD 0.9468. The treatment-mechanism term holds 0.62 of the curve variance |
| P0 | law P with the outcome mean replaced by its row mean (sharp null). Every tilted mean is equal |
| Pw | law P with the outcome contrast shrunk to 0.033 of its size. The exact power is 0.8724 at n = 4,000 |
| L | propensities from 0.070 to 0.774 and retention 0.833. The contrast `T_a` has truth 0.9267 and SD 1.0172. With the treatment conditionals frozen at the truth the curve SD falls to 0.808 of this value |
| L0 | law L without dropout, with the same truths and SD 0.9597 |

Two routes give each truth to $10^{-12}$. The longitudinal second route is a g-formula written in
loops from the conditional tables.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two treatment nodes with three levels each, dropout after each node, five folds | `ate_regimen[T_a vs natural course]` | difference in mean outcome between the plans "multiply the odds of the second level by 2 and of the third by 0.5, against the first" against "leave the observed treatment mechanism unchanged" | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.000580 to 0.0012 | 0.9610 | 1.0253 | pass |
| two treatment nodes with three levels each, dropout after each node, five folds | `ate_regimen[T_b vs natural course]` | difference in mean outcome between the plans "multiply the odds of the second level by 0.5 and of the third by 2, against the first" against "leave the observed treatment mechanism unchanged" | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0013 to 0.000838 | 0.9580 | 1.0016 | pass |
| two treatment nodes with three levels each, dropout after each node, five folds | `ey_regimen[T_a]` | mean outcome under the plan multiply the odds of the second level by 2 and of the third by 0.5, against the first | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0019 to 0.0020 | 0.9390 | 0.9633 | pass |
| two treatment nodes with three levels each, dropout after each node, five folds | `ey_regimen[T_b]` | mean outcome under the plan multiply the odds of the second level by 0.5 and of the third by 2, against the first | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0023 to 0.0012 | 0.9390 | 0.9710 | pass |
| two treatment nodes with three levels each, dropout after each node, five folds | `ey_regimen[natural course]` | mean outcome under the plan leave the observed treatment mechanism unchanged | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0021 to 0.0016 | 0.9350 | 0.9486 | pass |
| two treatment nodes with three levels each, no dropout, five folds | `ate_regimen[T_a vs natural course]` | difference in mean outcome between the plans "multiply the odds of the second level by 2 and of the third by 0.5, against the first" against "leave the observed treatment mechanism unchanged" | `cleverly` TMLE and `IPSI` with a categorical odds tilt | 0.000655 to 0.0024 | 0.9460 | 0.9962 | pass |
| two treatment nodes with three levels each, no dropout, five folds | `ate_regimen[T_b vs natural course]` | difference in mean outcome between the plans "multiply the odds of the second level by 0.5 and of the third by 2, against the first" against "leave the observed treatment mechanism unchanged" | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0025 to -0.000496 | 0.9480 | 1.0017 | pass |
| two treatment nodes with three levels each, no dropout, five folds | `ey_regimen[T_a]` | mean outcome under the plan multiply the odds of the second level by 2 and of the third by 0.5, against the first | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.000922 to 0.0028 | 0.9480 | 0.9758 | pass |
| two treatment nodes with three levels each, no dropout, five folds | `ey_regimen[T_b]` | mean outcome under the plan multiply the odds of the second level by 0.5 and of the third by 2, against the first | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0036 to -0.000559 | 0.9590 | 1.0288 | pass |
| two treatment nodes with three levels each, no dropout, five folds | `ey_regimen[natural course]` | mean outcome under the plan leave the observed treatment mechanism unchanged | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0023 to 0.0011 | 0.9470 | 0.9924 | pass |
| three-level treatment, four baseline values, five folds | `ate_ipsi[T_a vs natural course]` | difference in means under the incremental interventions "multiply the odds of the second level by 2 and of the third by 0.5, against the first" against "leave the observed treatment mechanism unchanged" | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0014 to 0.000299 | 0.9480 | 1.0145 | pass |
| three-level treatment, four baseline values, five folds | `ate_ipsi[T_b vs natural course]` | difference in means under the incremental interventions "multiply the odds of the second level by 0.5 and of the third by 2, against the first" against "leave the observed treatment mechanism unchanged" | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.000403 to 0.0014 | 0.9500 | 1.0068 | pass |
| three-level treatment, four baseline values, five folds | `ey_ipsi[T_a]` | mean under the incremental intervention multiply the odds of the second level by 2 and of the third by 0.5, against the first | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0023 to 0.0019 | 0.9410 | 0.9898 | pass |
| three-level treatment, four baseline values, five folds | `ey_ipsi[T_b]` | mean under the incremental intervention multiply the odds of the second level by 0.5 and of the third by 2, against the first | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0011 to 0.0028 | 0.9490 | 0.9964 | pass |
| three-level treatment, four baseline values, five folds | `ey_ipsi[natural course]` | mean under the incremental intervention leave the observed treatment mechanism unchanged | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0017 to 0.0024 | 0.9430 | 0.9903 | pass |
| three-level treatment, four baseline values, nuisances fitted in sample | `ate_ipsi[T_a vs natural course]` | difference in means under the incremental interventions "multiply the odds of the second level by 2 and of the third by 0.5, against the first" against "leave the observed treatment mechanism unchanged" | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.000546 to 0.0012 | 0.9460 | 1.0124 | pass |
| three-level treatment, four baseline values, nuisances fitted in sample | `ate_ipsi[T_b vs natural course]` | difference in means under the incremental interventions "multiply the odds of the second level by 0.5 and of the third by 2, against the first" against "leave the observed treatment mechanism unchanged" | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0013 to 0.000552 | 0.9570 | 1.0173 | pass |
| three-level treatment, four baseline values, nuisances fitted in sample | `ey_ipsi[T_a]` | mean under the incremental intervention multiply the odds of the second level by 2 and of the third by 0.5, against the first | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0014 to 0.0026 | 0.9530 | 1.0295 | pass |
| three-level treatment, four baseline values, nuisances fitted in sample | `ey_ipsi[T_b]` | mean under the incremental intervention multiply the odds of the second level by 0.5 and of the third by 2, against the first | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0019 to 0.0018 | 0.9530 | 1.0316 | pass |
| three-level treatment, four baseline values, nuisances fitted in sample | `ey_ipsi[natural course]` | mean under the incremental intervention leave the observed treatment mechanism unchanged | `cleverly` TMLE and `IPSI` with a categorical odds tilt | -0.0016 to 0.0023 | 0.9510 | 1.0331 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `censoring_complete_case` | `longitudinal_mean_t_a__complete_case` | diagnostic | two-node law L with dropout, mean under T_a: a fit to the rows that never drop out, without a censoring model, beside the exact-law limit | reported only | bias 0.0977 to 0.0996, margin 0.0042, coverage 0.0000 | reported |
| `in_sample_fit` | `longitudinal_dropout_contrast_t_a__one_fold` | diagnostic | two-node law L with dropout and three-level treatment, T_a against the natural course: the same histories fitted with one fold, beside the exact-law truth | reported only | bias -0.000714 to -0.000018, margin 0.0020, coverage 0.9478 | reported |
| `interval_calibration` | `longitudinal_contrast_t_a__correctly_specified` | positive | two-node law L0 without dropout and three-level treatment, T_a against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9429 to 0.9614, SE ratio 0.9819 to 1.0431, empirical efficiency ratio 0.9597 to 1.0194, reported efficiency ratio 1.0006 to 1.0014 | pass |
| `interval_calibration` | `longitudinal_contrast_t_a__noise_control` | control | two-node law L0 without dropout and three-level treatment, T_a against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8096 to 0.8423, SE ratio 0.6870 to 0.7287, empirical efficiency ratio 1.3736 to 1.4569, reported efficiency ratio 1.0006 to 1.0014 | pass |
| `interval_calibration` | `longitudinal_contrast_t_a__shrunken_se_control` | control | two-node law L0 without dropout and three-level treatment, T_a against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8199 to 0.8520, SE ratio 0.6865 to 0.7308, empirical efficiency ratio 0.9589 to 1.0207, reported efficiency ratio 0.7005 to 0.7010 | pass |
| `interval_calibration` | `longitudinal_dropout_contrast_t_a__correctly_specified` | positive | two-node law L with dropout and three-level treatment, T_a against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9360 to 0.9556, SE ratio 0.9637 to 1.0221, empirical efficiency ratio 0.9796 to 1.0389, reported efficiency ratio 1.0009 to 1.0016 | pass |
| `interval_calibration` | `longitudinal_dropout_contrast_t_a__noise_control` | control | two-node law L with dropout and three-level treatment, T_a against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8245 to 0.8562, SE ratio 0.6940 to 0.7372, empirical efficiency ratio 1.3581 to 1.4427, reported efficiency ratio 1.0009 to 1.0016 | pass |
| `interval_calibration` | `longitudinal_dropout_contrast_t_a__shrunken_se_control` | control | two-node law L with dropout and three-level treatment, T_a against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8090 to 0.8418, SE ratio 0.6754 to 0.7160, empirical efficiency ratio 0.9790 to 1.0376, reported efficiency ratio 0.7006 to 0.7011 | pass |
| `interval_calibration` | `point_contrast_t_a__correctly_specified` | positive | three-level law P, T_a against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9441 to 0.9624, SE ratio 0.9773 to 1.0383, empirical efficiency ratio 0.9640 to 1.0241, reported efficiency ratio 1.0001 to 1.0016 | pass |
| `interval_calibration` | `point_contrast_t_a__noise_control` | control | three-level law P, T_a against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8145 to 0.8469, SE ratio 0.6873 to 0.7279, empirical efficiency ratio 1.3750 to 1.4565, reported efficiency ratio 1.0001 to 1.0016 | pass |
| `interval_calibration` | `point_contrast_t_a__shrunken_se_control` | control | three-level law P, T_a against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8133 to 0.8458, SE ratio 0.6847 to 0.7262, empirical efficiency ratio 0.9649 to 1.0233, reported efficiency ratio 0.7000 to 0.7011 | pass |
| `interval_calibration` | `point_contrast_t_b__correctly_specified` | positive | three-level law P, T_b against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9390 to 0.9581, SE ratio 0.9759 to 1.0332, empirical efficiency ratio 0.9690 to 1.0259, reported efficiency ratio 1.0004 to 1.0018 | pass |
| `interval_calibration` | `point_contrast_t_b__noise_control` | control | three-level law P, T_b against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8159 to 0.8482, SE ratio 0.6847 to 0.7262, empirical efficiency ratio 1.3781 to 1.4619, reported efficiency ratio 1.0003 to 1.0018 | pass |
| `interval_calibration` | `point_contrast_t_b__shrunken_se_control` | control | three-level law P, T_b against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8101 to 0.8429, SE ratio 0.6833 to 0.7244, empirical efficiency ratio 0.9672 to 1.0256, reported efficiency ratio 0.7002 to 0.7012 | pass |
| `interval_calibration` | `point_known_t_a__correctly_specified` | positive | law P with the treatment mechanism declared, T_a against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9466 to 0.9644, SE ratio 0.9870 to 1.0490, empirical efficiency ratio 0.9546 to 1.0147, reported efficiency ratio 1.0005 to 1.0022 | pass |
| `interval_calibration` | `point_known_t_a__noise_control` | control | law P with the treatment mechanism declared, T_a against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8219 to 0.8538, SE ratio 0.7029 to 0.7461, empirical efficiency ratio 1.3421 to 1.4244, reported efficiency ratio 1.0005 to 1.0022 | pass |
| `interval_calibration` | `point_known_t_a__shrunken_se_control` | control | law P with the treatment mechanism declared, T_a against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8156 to 0.8480, SE ratio 0.6910 to 0.7346, empirical efficiency ratio 0.9542 to 1.0142, reported efficiency ratio 0.7004 to 0.7015 | pass |
| `interval_calibration` | `point_stratified_t_a_v1__correctly_specified` | positive | law P in baseline stratum V = 1, T_a against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9387 to 0.9579, SE ratio 0.9687 to 1.0295, empirical efficiency ratio 0.9705 to 1.0317, reported efficiency ratio 0.9980 to 1.0007 | pass |
| `interval_calibration` | `point_stratified_t_a_v1__noise_control` | control | law P in baseline stratum V = 1, T_a against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8165 to 0.8488, SE ratio 0.6956 to 0.7381, empirical efficiency ratio 1.3542 to 1.4367, reported efficiency ratio 0.9980 to 1.0007 | pass |
| `interval_calibration` | `point_stratified_t_a_v1__shrunken_se_control` | control | law P in baseline stratum V = 1, T_a against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8093 to 0.8421, SE ratio 0.6787 to 0.7205, empirical efficiency ratio 0.9712 to 1.0305, reported efficiency ratio 0.6986 to 0.7005 | pass |
| `mechanism_targeting` | `point_contrast_t_a__mechanism_targeted` | positive | three-level law P, T_a against the natural course: the treatment mechanism is tilted along the T_a mechanism score by 2 / sqrt(n), then the full targeting runs | bias inside the margin and the SE-ratio interval inside the efficiency-ratio band | bias -0.0020 to -0.000300, margin 0.0038, SE ratio 0.9588 to 1.0371 | pass |
| `mechanism_targeting` | `point_contrast_t_a__outcome_targeted_only` | control | three-level law P, T_a against the natural course: the same tilted mechanism with the outcome fluctuation alone | bias interval must fall above zero and entirely outside the margin | bias 0.0176 to 0.0193, margin 0.0037 | pass |
| `mechanism_targeting` | `point_stratified_t_a_v1__mechanism_targeted` | positive | law P in baseline stratum V = 1, T_a against the natural course: the treatment mechanism is tilted along the T_a mechanism score by 2 / sqrt(n), then the full targeting runs | bias inside the margin and the SE-ratio interval inside the efficiency-ratio band | bias -0.0030 to -0.000502, margin 0.0054, SE ratio 0.9526 to 1.0349 | pass |
| `mechanism_term_necessity` | `point_contrast_t_a__binary_blip` | control | three-level law P, T_a against the natural course: the same estimates with the mechanism term read from the first two levels only | the SE-ratio interval must fall below the calibration band | SE ratio 0.7789 to 0.8261 | pass |
| `mechanism_term_necessity` | `point_contrast_t_a__full_eif` | positive | three-level law P, T_a against the natural course: the reported standard error from the full influence curve | the SE-ratio interval inside the efficiency-ratio band | SE ratio 0.9905 to 1.0511 | pass |
| `mechanism_term_necessity` | `point_contrast_t_a__mechanism_term_dropped` | control | three-level law P, T_a against the natural course: the same estimates with the treatment-mechanism term, on the outcome's own scale, removed from the curve | the SE-ratio interval must fall below the calibration band | SE ratio 0.6091 to 0.6468 | pass |
| `power` | `point_contrast_t_a__alternative` | positive | three-level law P, T_a against the natural course: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.8725, 0.8521 to 0.8911 | pass |
| `single_robustness` | `point_contrast_t_a__mechanism_constant` | diagnostic | three-level law P, T_a against the natural course: constant treatment mechanism and cell-mean outcome regressions, beside the exact-law limit | reported only | bias -0.0653 to -0.0641, margin 0.0026, coverage 0.0000 | reported |
| `single_robustness` | `point_contrast_t_a__mechanism_correct__n_16000` | positive | three-level law P, T_a against the natural course: cell-frequency treatment mechanism and a constant outcome regression, at n = 16,000 | bias inside the margin at n = 8,000 and at n = 16,000 | bias -0.000988 to -0.000125, margin 0.0019, coverage 0.9070 | pass |
| `single_robustness` | `point_contrast_t_a__mechanism_correct__n_8000` | positive | three-level law P, T_a against the natural course: cell-frequency treatment mechanism and a constant outcome regression, at n = 8,000 | bias inside the margin at n = 8,000 and at n = 16,000 | bias -0.0014 to -0.000096, margin 0.0027, coverage 0.8940 | pass |
| `type_i_error` | `point_contrast_t_a__sharp_null` | positive | three-level law P, T_a against the natural course: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0515, 0.0396 to 0.0656 | pass |
<!-- /generated -->

Every property fit uses saturated learners. The tables above print one row for each cell. The
families below state what each cell tests.

| family | cells and fits | what must hold |
| --- | --- | --- |
| `interval_calibration` | six labels with three cells each: the correctly specified fit and two derived controls. Four point labels at n = 4,000. Two longitudinal labels at n = 16,000 | SE-ratio and coverage intervals inside the calibration bands, and both efficiency-ratio intervals inside their band. The shrunken-SE control falls below the band. The noise control rises above the efficiency-ratio band |
| `mechanism_term_necessity` | the `T_a` contrast on law P: the full curve, the curve without the mechanism term, and the curve whose mechanism term reads the first two levels only (the binary blip) | the full curve is calibrated. Each control under-reports and its SE-ratio interval lies below the calibration band |
| `mechanism_targeting` | the `T_a` contrast on law P, with a mechanism tilted along its score by $2/\sqrt n$. The TMLE, unstratified and in stratum V = 1, and the outcome-only targeted plug-in | each TMLE cell has bias inside the margin and an SE-ratio interval inside the efficiency band. The outcome-only control lies above the truth and outside the margin |
| `type_i_error` and `power` | the `T_a` contrast on law P0 and law Pw | the upper endpoint of the rejection interval on law P0 stays at or below the type-I ceiling. The lower endpoint on law Pw clears the minimum power |
| `single_robustness` | the mechanism saturated and the outcome regression constant, at n = 8,000 and 16,000. The mechanism constant with a saturated outcome regression is a diagnostic | bias inside the margin at both sizes. The diagnostic prints its bias beside the exact limit and states no verdict |
| `in_sample_fit` | the longitudinal contrast on law L with one fold, on its own sample stream | no verdict. The cell prints its bias and coverage beside the exact truth |
| `censoring_complete_case` | the mean under `T_a` on law L, fitted to the rows that never drop out, with no censoring model | no verdict. The cell prints its bias beside the exact complete-case limit |

The `mechanism_targeting` family is the study-scale evidence for the softmax update. A saturated
cross-fitted mechanism moves by $O(1/n)$ only. The other cells therefore cannot see a defective
update. The tilted mechanism has a first-order error, and only the mechanism fluctuation removes
it. The outcome-only plug-in sits 1.18 SD above the truth, which is its exact first-order limit.

`tests/unit/test_categorical_incremental_design.py` shows on the exact law that an uncentred
mechanism covariate moves the TMLE by 0.37 SD. A one-versus-rest update is a deliberate mutation.
The fast-tier test `tests/unit/test_categorical_incremental_mutations.py` catches it.

The `mechanism_term_necessity` controls carry a verdict that asserts only the failure each control
exists to show. The `single_robustness` cells gate bias only. A misspecified outcome regression
leaves a first-order variance term, so their intervals are anti-conservative. A bias pass is not
an interval claim.

## Sizing

The sizes follow the pilots in `tests/unit/test_categorical_incremental_design.py`. The study
fixed every size, replication count, margin, law, learner, fold count and seed before any run.

| sizing input | value | source |
| --- | --- | --- |
| primary n | 8,000 for all four scenarios. The plan's rung rule gives 1,000 for family P and 2,000 for family L, where the twenty primary rows pass the bias gate with joint probability about 0.3. At 8,000 the joint probability is 0.99 under a bias that falls as $1/\sqrt n$ | `tests/studies/canonical_categorical_incremental.py` |
| rung pilots | 400 draws per rung from n = 1,000 to 16,000. The largest standardized bias of a primary row is 0.164 at n = 1,000 and 0.054 at n = 16,000 for family P | the same |
| calibration n | 4,000 for the point labels. 16,000 for the longitudinal labels, where the pilot of 1,600 draws read SE ratios 1.009 and 1.016 | `tests/studies/categorical_incremental_properties.py` |
| calibration replications | 3,600. The family-wise pass probability under perfect calibration is 0.966 | the same |
| `mechanism_targeting` replications | 2,000. The family pass probability is 0.995 | the same |
| type I error and power | 2,000 replications at n = 4,000, with exact power 0.8724 | the same |
| single robustness | 2,000 replications at n = 8,000 and 16,000 | the same |
| complete-case cell | 2,000 replications at n = 16,000. The exact limit sits 5.9 standard errors from the truth | the same |

## Red-cell route

| event | route, declared before any run |
| --- | --- |
| a red primary truth row | blocks the merge and gets a diagnosis |
| a red calibration row | is a finite-sample red only when its SE-ratio interval contains 1, its coverage is consistent with 0.95 and the diagnosis finds no defect. It then moves to `reporting` with an owner row before any re-run. Any other red row is a defect suspect. A defect is fixed and the study re-runs under a fresh declaration |
| a red `mechanism_targeting` positive cell | is a defect suspect in the softmax update |
| a control that does not fail | is a defect suspect in the instrument |
| any verdict | moves no budget, margin, law, learner, fold count or seed |

The run found no red cell.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
exact standard errors of the efficient influence curve at the primary n. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 8000 | observations per primary replication |
| `independent_tests_passed` | 20 | truth tests passing |
| `independent_tests_total` | 20 | truth tests reported |
| `property_cells_passed` | 28 | property cells passing |
| `property_cells_total` | 28 | property cells reported |
| `max_standardized_bias` | 0.1422 | largest primary standardized bias |
| `min_coverage` | 0.9350 | lowest primary coverage |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | the rate the upper endpoint of the sharp-null rejection interval must not exceed |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power cell must clear |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `bound:ate_ipsi[T_a vs natural course]_standard_error` | 0.0106 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:ate_ipsi[T_b vs natural course]_standard_error` | 0.0112 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:ey_ipsi[T_a]_standard_error` | 0.0252 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:ey_ipsi[T_b]_standard_error` | 0.0238 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:ey_ipsi[natural course]_standard_error` | 0.0248 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_dropout_folds_5/ate_regimen[T_a vs natural course]_standard_error` | 0.0114 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_dropout_folds_5/ate_regimen[T_b vs natural course]_standard_error` | 0.0133 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_dropout_folds_5/ey_regimen[T_a]_standard_error` | 0.0231 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_dropout_folds_5/ey_regimen[T_b]_standard_error` | 0.0211 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_dropout_folds_5/ey_regimen[natural course]_standard_error` | 0.0215 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_folds_5/ate_regimen[T_a vs natural course]_standard_error` | 0.0107 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_folds_5/ate_regimen[T_b vs natural course]_standard_error` | 0.0123 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_folds_5/ey_regimen[T_a]_standard_error` | 0.0224 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_folds_5/ey_regimen[T_b]_standard_error` | 0.0192 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `bound:longitudinal_folds_5/ey_regimen[natural course]_standard_error` | 0.0206 | exact influence-curve standard error of this estimand at the primary n = 8,000 |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| three levels at each tilt node | the laws have three treatment levels. Four or more levels and the vector-treatment keys are outside this evidence |
| saturated learners | the study does not test a flexible learner. The cross-fitted mechanism moves by $O(1/n)$, which the `mechanism_targeting` family compensates |
| two tilt nodes and one outcome | longer panels, survival and competing-risk outcomes are covered by exact laws and not by a registered study |
| dropout at random given the history | the retention is 0.833 on law L. Heavier and informative dropout are outside this evidence |
| finite-sample bias is sized, not removed | the primary rows pass at n = 8,000. Smaller sizes carry a larger standardized bias, which the rung pilots print |
| single-robustness intervals | the cells gate bias only. The intervals of a misspecified outcome regression are anti-conservative |
| the parametric g-formula | a tilt at a node with other than two levels is refused. This study does not cover it |
| no comparator | no maintained package ships the construction, so the study has no paired verdict |

## Reproduction

`tests/canonical/categorical_incremental/regenerate.py` regenerates the study under the declared
run form. It runs no reference process, because the study has no comparator. The run took 2.7 hours
on four cores.
