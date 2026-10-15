# Point survival curve crossfit

This study validates longitudinal TMLE with `survival_model="hazard"` and five outer folds on a
held design. The design has one baseline treatment, held over every visit, and it reads one
follow-up time and one event code per unit (`TimeToEvent`). Each fold fits the hazards, the
treatment mechanism and the censoring mechanism on its training rows. One pooled targeting step
then reads the stitched out-of-fold predictions. The section
[hazard-based survival targeting](../longitudinal-tmle.md#hazard-based-survival-targeting) states
the estimator.

The primary fit uses `targeting="one_step"`. One walk moves one set of discrete hazards toward the
score equation of every horizon and regimen at once. One calibration cell uses `targeting="iterative"`,
which targets each horizon on its own. The laws are the `survival` and `competing` scenarios of
[point-treatment survival](point-treatment-survival.md). Every truth is an exact finite sum. Every
efficiency bound is the exact standard deviation of the efficient influence function over the
law's support.

No canonical implementation is compared. No maintained package ships hazard targeting with
cross-fitted nuisances for a held baseline treatment. `equivalence.csv` has no rows, and
this study is not pending. The
[theory and comparators rule](../../architecture-invariants.md#theory-and-comparators) lets the
row stand on independent evidence. That evidence is the exact truths, the exact efficiency
bounds, and the controls of the calibration, double-robustness and coverage cells. 
## What was compared

| setting | `cleverly` |
| --- | --- |
| datasets | 1,600 samples of 2,000 rows of the `survival` scenario |
| scenario | `survival`: five visits, the risks of both arms and their difference at visits 1, 3 and 5 |
| property laws | `survival` and `competing`: four visits and two causes, the death incidence difference at visit 4 |
| nuisances | the hazard learner is an unpenalized additive logistic regression on `survival` and the saturated cell fit on `competing`. The treatment and censoring learners are saturated cell fits |
| targeting | `targeting="one_step"` with `step_size=1e-3` and `tol=1e-10`. One calibration cell uses `targeting="iterative"` with `max_iter=100` |
| cross-fitting | five outer folds, one pooled targeting step |
| intervals | pointwise 95% Wald from the stitched influence curve |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.000808 to 0.0011 | 0.9494 | 0.9955 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.0012 to 0.0017 | 0.9494 | 0.9915 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.000614 to 0.0025 | 0.9525 | 1.0122 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.000752 to 0.000721 | 0.9375 | 0.9791 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.0015 to 0.000586 | 0.9506 | 0.9936 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.0018 to 0.000324 | 0.9594 | 1.0478 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.000481 to 0.000714 | 0.9437 | 0.9973 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.0012 to 0.000803 | 0.9450 | 0.9767 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | `cleverly` cross-fitted point-treatment survival with hazard targeting | -0.000936 to 0.0014 | 0.9481 | 0.9924 | pass |
<!-- /generated -->

All nine implementation-estimand tests pass. No replication raised an error in the primary study
or in any property cell (`failed_replicates` is zero in all 15 rows of `properties.csv`).

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `survival_t5__both_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0028 to 0.000648, margin 0.0059, SE ratio 1.0554 | pass |
| `double_robustness` | `survival_t5__both_wrong` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0322 to 0.0358, margin 0.0062, SE ratio 0.9799 | pass |
| `double_robustness` | `survival_t5__mechanism_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000268 to 0.0034, margin 0.0061, SE ratio 1.0223 | pass |
| `double_robustness` | `survival_t5__outcome_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0011 to 0.0025, margin 0.0061, SE ratio 0.9805 | pass |
| `interval_calibration` | `competing_t4__correctly_specified` | positive | difference in the death incidence at visit 4, two causes: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9436 to 0.9597, SE ratio 0.9737 to 1.0259, empirical efficiency ratio 0.9912 to 1.0442, reported efficiency ratio 1.0158 to 1.0177 | pass |
| `interval_calibration` | `survival_t5__correctly_specified` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9412 to 0.9531, SE ratio 0.9890 to 1.0269, empirical efficiency ratio 0.9901 to 1.0280, reported efficiency ratio 1.0164 to 1.0171 | pass |
| `interval_calibration` | `survival_t5__noise_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8349 to 0.8541, SE ratio 0.7057 to 0.7331, empirical efficiency ratio 1.3870 to 1.4408, reported efficiency ratio 1.0164 to 1.0171 | pass |
| `interval_calibration` | `survival_t5__shrunken_se_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8196 to 0.8395, SE ratio 0.6932 to 0.7188, empirical efficiency ratio 0.9901 to 1.0266, reported efficiency ratio 0.7115 to 0.7119 | pass |
| `interval_calibration` | `survival_t5_per_horizon__correctly_specified` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law, targeted one horizon at a time: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9445 to 0.9604, SE ratio 1.0034 to 1.0593, empirical efficiency ratio 0.9597 to 1.0131, reported efficiency ratio 1.0161 to 1.0170 | pass |
| `simultaneous_coverage` | `all_reported__pointwise_joint_control` | control | every parameter the calibration fit reports: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6568 to 0.6816 | pass |
| `simultaneous_coverage` | `all_reported__simultaneous_band` | positive | every parameter the calibration fit reports: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9393 to 0.9513 | pass |
| `simultaneous_coverage` | `curve_arm0__pointwise_joint_control` | control | the arm-0 risk curve alone, at the five visits: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8334 to 0.8527 | pass |
| `simultaneous_coverage` | `curve_arm0__simultaneous_band` | positive | the arm-0 risk curve alone, at the five visits: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9414 to 0.9532 | pass |
| `simultaneous_coverage` | `curve_arm1__pointwise_joint_control` | control | the arm-1 risk curve alone, at the five visits: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8345 to 0.8537 | pass |
| `simultaneous_coverage` | `curve_arm1__simultaneous_band` | positive | the arm-1 risk curve alone, at the five visits: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9405 to 0.9524 | pass |
<!-- /generated -->
All 15 property cells pass. Each positive calibration or coverage cell has a control that fails
the same instrument in the opposite direction.

| family | positive cells | control that must fail |
| --- | --- | --- |
| `interval_calibration` | the visit-5 difference on `survival` (one step), the same difference targeted per horizon, and the death incidence difference at visit 4 on `competing` | reported standard errors scaled by 0.7, and one efficiency-bound unit of added noise, on the one-step `survival` cell only |
| `double_robustness` | the hazard correct, the mechanism correct, or both | both wrong |
| `simultaneous_coverage` | the max-t band over each arm's five-visit risk curve, and over the fifteen reported parameters | the pointwise intervals of the same fits, read jointly |

The per-horizon calibration cell has no control of its own. Its pass rests on the efficiency bound
and the two bands.

The study has no targeting-necessity, power or root-n cell. The
[sibling hazard study](point-survival-hazard.md) holds those cells for the in-sample per-horizon
fit. No cross-fitted cell shows that targeting moves a misspecified hazard, so the study makes no
claim there.

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 9 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 9 | of those, passing |
| `property_cells_total` | 15 | repeated-sampling property cells |
| `property_cells_passed` | 15 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0448 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9375 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9341 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1017 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## Limitations

| limit | what it means for use |
| --- | --- |
| no comparator | no maintained package fits this construction, so no row is paired. The independent evidence is the exact truths, the exact efficiency bounds and the controls |
| one scenario in the primary study | the nine truth tests read the five-visit `survival` law. The `competing` law enters one calibration cell only |
| learners that contain the truth | the evidence covers a correct additive logistic hazard on `survival`, a saturated hazard on `competing`, and saturated mechanisms. A flexible learner is outside it |
| one per-horizon cell | the per-horizon iteration appears only as one calibration cell. It has no double-robustness, coverage-band or control cell |
| one fold count | the study fits five fixed folds. Other fold counts and repeated splits are outside it |
| no targeting-necessity, power or root-n cell | the study does not show that targeting moves a misspecified cross-fitted hazard, or that its rate is root-n |
| no weights, clusters, policies or three arms | each is a separate composition and needs its own row |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_curve_crossfit/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. It records the comparator as absent. Run
`python -m tests.canonical.point_survival_curve_crossfit.regenerate` to regenerate the artifacts.
The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_curve_crossfit/replicates.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_curve_crossfit/properties.csv)
carry every published row.
