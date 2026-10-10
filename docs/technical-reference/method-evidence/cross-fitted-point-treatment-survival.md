# Cross-fitted point-treatment survival

This study validates cross-fitted longitudinal TMLE on the held point-treatment design of
[point-treatment survival](point-treatment-survival.md). The fit uses five outer folds. Each fold
runs an untargeted backward recursion on its training rows. One pooled fluctuation for each node
then targets the stitched out-of-fold predictions (Díaz, Williams, Hoffman and Schenck 2023,
Section 5.2).

No canonical implementation is compared, because no maintained package ships this construction
for a held baseline treatment. The row rests on the truth tests and the property cells.

## What was compared

| setting | `cleverly` |
| --- | --- |
| datasets | 1,600 samples of 2,000 rows for each scenario |
| scenarios | `survival`: the five-visit law of `point-treatment-survival`, the risks of both arms and their difference at visits 1, 3 and 5. `three_arm`: three arms with a binary covariate at each visit on the wide layout, the risks of arms 0 and 2 and the differences of arms 1 and 2 against arm 0 at visits 1 and 3 |
| nuisances | the saturated cell means, which are correct on these finite laws |
| targeting | one pooled fluctuation for each node, over five outer folds |
| intervals | pointwise 95% Wald from the stitched influence curve |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | `cleverly` cross-fitted point-treatment survival | -0.0011 to 0.000826 | 0.9431 | 0.9888 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` cross-fitted point-treatment survival | -0.0018 to 0.0012 | 0.9406 | 0.9681 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | `cleverly` cross-fitted point-treatment survival | -0.0014 to 0.0019 | 0.9394 | 0.9728 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | `cleverly` cross-fitted point-treatment survival | -0.000463 to 0.000986 | 0.9469 | 0.9991 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | `cleverly` cross-fitted point-treatment survival | -0.000297 to 0.0018 | 0.9431 | 0.9875 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | `cleverly` cross-fitted point-treatment survival | -0.0010 to 0.0012 | 0.9500 | 0.9920 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | `cleverly` cross-fitted point-treatment survival | -0.000465 to 0.000743 | 0.9500 | 0.9882 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | `cleverly` cross-fitted point-treatment survival | -0.000505 to 0.0015 | 0.9431 | 0.9761 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | `cleverly` cross-fitted point-treatment survival | -0.000831 to 0.0015 | 0.9487 | 0.9891 | pass |
| three visits, three arms held over every node, a binary L_t at each later visit | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | `cleverly` cross-fitted point-treatment survival | -0.000666 to 0.0015 | 0.9575 | 1.0228 | pass |
| three visits, three arms held over every node, a binary L_t at each later visit | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` cross-fitted point-treatment survival | -0.0019 to 0.0016 | 0.9644 | 1.0481 | pass |
| three visits, three arms held over every node, a binary L_t at each later visit | `ate_regimen[arm2 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 2 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | `cleverly` cross-fitted point-treatment survival | -0.0015 to 0.000888 | 0.9463 | 0.9883 | pass |
| three visits, three arms held over every node, a binary L_t at each later visit | `ate_regimen[arm2 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 2 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` cross-fitted point-treatment survival | -0.0021 to 0.0017 | 0.9519 | 0.9962 | pass |
| three visits, three arms held over every node, a binary L_t at each later visit | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | `cleverly` cross-fitted point-treatment survival | -0.0012 to 0.000404 | 0.9500 | 0.9989 | pass |
| three visits, three arms held over every node, a binary L_t at each later visit | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | `cleverly` cross-fitted point-treatment survival | -0.0013 to 0.0012 | 0.9569 | 1.0229 | pass |
| three visits, three arms held over every node, a binary L_t at each later visit | `risk_regimen[arm2 @ t=1]` | cumulative risk under the plan assign arm 2 at baseline and hold it at horizon t = 1 | `cleverly` cross-fitted point-treatment survival | -0.0015 to 0.000109 | 0.9463 | 1.0156 | pass |
| three visits, three arms held over every node, a binary L_t at each later visit | `risk_regimen[arm2 @ t=3]` | cumulative risk under the plan assign arm 2 at baseline and hold it at horizon t = 3 | `cleverly` cross-fitted point-treatment survival | -0.0017 to 0.0011 | 0.9513 | 0.9995 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `survival_t5__both_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0036 to 0.000022, margin 0.0060, SE ratio 1.0285 | pass |
| `double_robustness` | `survival_t5__both_wrong` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0306 to 0.0343, margin 0.0063, SE ratio 0.9561 | pass |
| `double_robustness` | `survival_t5__mechanism_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0034 to 0.000118, margin 0.0060, SE ratio 1.0450 | pass |
| `double_robustness` | `survival_t5__outcome_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000512 to 0.0031, margin 0.0061, SE ratio 0.9879 | pass |
| `interval_calibration` | `survival_t5__correctly_specified` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9462 to 0.9575, SE ratio 0.9986 to 1.0383, empirical efficiency ratio 0.9816 to 1.0207, reported efficiency ratio 1.0189 to 1.0196 | pass |
| `interval_calibration` | `survival_t5__noise_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8361 to 0.8552, SE ratio 0.7099 to 0.7373, empirical efficiency ratio 1.3824 to 1.4359, reported efficiency ratio 1.0189 to 1.0196 | pass |
| `interval_calibration` | `survival_t5__shrunken_se_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8312 to 0.8505, SE ratio 0.6988 to 0.7258, empirical efficiency ratio 0.9831 to 1.0210, reported efficiency ratio 0.7133 to 0.7137 | pass |
| `interval_calibration` | `three_arm_t3__correctly_specified` | positive | difference of arm 2 and arm 0 at visit 3, three arms with L_t: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9421 to 0.9689, SE ratio 0.9650 to 1.0560, empirical efficiency ratio 1.0101 to 1.1085, reported efficiency ratio 1.0592 to 1.0764 | **fail** |
| `interval_calibration` | `three_arm_t3__noise_control` | control | difference of arm 2 and arm 0 at visit 3, three arms with L_t: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8250 to 0.8716, SE ratio 0.7010 to 0.7659, empirical efficiency ratio 1.3923 to 1.5239, reported efficiency ratio 1.0590 to 1.0764 | pass |
| `interval_calibration` | `three_arm_t3__shrunken_se_control` | control | difference of arm 2 and arm 0 at visit 3, three arms with L_t: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8025 to 0.8517, SE ratio 0.6757 to 0.7382, empirical efficiency ratio 1.0108 to 1.1065, reported efficiency ratio 0.7415 to 0.7533 | pass |
| `simultaneous_coverage` | `all_reported__pointwise_joint_control` | control | every parameter the calibration fit reports: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6652 to 0.6899 | pass |
| `simultaneous_coverage` | `all_reported__simultaneous_band` | positive | every parameter the calibration fit reports: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9418 to 0.9536 | pass |
<!-- /generated -->

The `three_arm_t3__correctly_specified` calibration cell is red on its empirical efficiency band.
The empirical ratio is 1.058, and its 99% interval is 1.010 to 1.108 against an upper bound of
1.10. The SE ratio is 1.008 and the coverage is 0.957, so the interval is calibrated. The reported
standard error is 6.7% above the bound, so the reported and the empirical spreads rise together.

The update is pooled by construction. Each outer fold fits an untargeted recursion, the fit
stitches the held-out predictions, and one fluctuation for each node is solved over every follower
([`_pooled_targeting`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/sequential.py)).
So a fold-local update is not a candidate.

The diagnostic
[`tests/diagnostics/x13_crossfit_three_arm/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x13_crossfit_three_arm)
refits the cell's configuration on fresh draws, with the study's own fit and bound. It separates a
mis-scaled curve, a wrong truth or a wrong bound from a finite-sample cost. Its module records the
seeds and the command. It reads the mean reported standard error over the bound in each design.

| design | reported SE over the bound |
| --- | --- |
| n = 2,000, in sample | 1.0003 |
| n = 2,000, five folds | 1.082 |
| n = 2,000, ten folds | 1.055 |
| n = 8,000, five folds | 1.0085 |

The in-sample fit reaches the bound, so the truth and the bound are right. The excess falls when
the training folds grow, and it nearly disappears at n = 8,000. A mis-scaled curve would keep it
at every size. The cell has many saturated cells in each training
fold, because the three-arm law has a covariate after baseline. The diagnosis finds no defect,
and the cell stays red under `reporting`, with the owner row `X13-finite-sample`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 17 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 17 | of those, passing |
| `property_cells_total` | 12 | repeated-sampling property cells |
| `property_cells_passed` | 11 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0559 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9394 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9267 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0956 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
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
| no comparator | no maintained package fits this construction, so no row is paired |
| saturated cell means only | the evidence covers correct nonparametric learners on finite laws. A flexible learner is outside it |
| one calibration cell is red under `reporting` | the three-arm cell pays a finite-sample cost of cross-fitting at n = 2,000. The interval stays calibrated. The `X13-finite-sample` owner holds it |
| five folds | the fold count is fixed. The probe shows the cost falls with larger training folds |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_crossfit/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.point_survival_crossfit.regenerate` to regenerate the
artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_crossfit/replicates.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_crossfit/properties.csv)
carry every published row.
