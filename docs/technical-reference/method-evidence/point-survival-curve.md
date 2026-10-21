# Point survival curve

This study validates longitudinal TMLE with `survival_model="hazard"` and `targeting="one_step"`
on a held design. The design has one baseline treatment, held over every visit, and it reads one
follow-up time and one event code per unit (`TimeToEvent`). One walk moves one set of discrete
hazards. The walk solves the score equation of every horizon and regimen at once (Cai and van der
Laan 2020, Section 5; Rytgaard and van der Laan 2024, Section 3.4). Every reported risk curve is
then non-decreasing in floating point. The section
[hazard-based survival targeting](../longitudinal-tmle.md#hazard-based-survival-targeting) states
the estimator.

The primary law is the `survival` scenario of
[point-treatment survival](point-treatment-survival.md). Every truth is an exact finite sum. Every
efficiency bound is the exact standard deviation of the efficient influence function over the
law's support.

The comparator is R `tmle3` at commit `ed72f8a`, which runs `tmle_survival` with a one-step
whole-curve hazard fluctuation. The comparator runs are pending, so this study carries the label
"Python property evidence, with comparator runs pending". The runs are queued as Study queue row 21
in `TRACKER.md`. The comparison is not acceptance evidence for any claim until that row closes.

## What was compared

| setting | `cleverly` | R `tmle3` |
| --- | --- | --- |
| status | run and published on this page | pending. No reference process ran, and `equivalence.csv` has no rows |
| datasets | 1,600 samples of 2,000 rows of the `survival` scenario | the identical rows, once the runs exist |
| scenario | `survival`: five visits, the risks of both arms and their difference at every visit | the same |
| treatment model | an unpenalized logistic regression on `W1` and `W2` | `Lrnr_glm` on `W1 + W2` |
| event and censoring hazards | an unpenalized logistic regression on `W1`, `W2` and the arm at each node | `Lrnr_stratified` on the visit over `Lrnr_glm`, one fit for each visit |
| targeting | `targeting="one_step"`, `step_size=1e-3`, stopped at a relative score of 1e-10 | `tmle3_Update_survival` with `fit_method="l2"`, `maxit=100` and `delta_epsilon=0.01`, stopped when every score is at most 1/n in absolute value |
| curve | the risks of both arms and their difference, from one set of hazards | one `tmle_survival` specification for each arm with `target_times = 1:5`. The contrast is the difference of the two per-unit influence curves, and the risk is one minus the reported survival |
| intervals | pointwise 95% Wald from the influence curve | the same |

The pending runs read under the default margins. The comparator is `tmle3` at commit `ed72f8a`
with `sl3` at commit `0e8f236`. It runs in its own image, which adds `glmnet` 5.0 to the `tmle3`
pins. The runner subclasses the updater to repair two defects of its `fit_submodel`, and it changes
nothing else. The vector test `fit_lambda == 1` becomes `length(fit_lambda) == 1`. The retry
assigns `lambda.min.ratio`, so a retry refits a different path.

A replicate with an unsolved score and no nonzero step fails the run. A replicate with a nonzero
step and an unsolved score stays in the paired rows. The run log counts it, and this page will
report it.

The declared unmatched settings are the cause of any difference between the two sides.

| unmatched setting | `cleverly` | `tmle3` |
| --- | --- | --- |
| bounds | its shipped `g_bounds` | `g` at 0.025 and the hazards at 0.025 and 0.005 |
| censoring convention | a unit is censored at node k before its event there can be seen | the censoring hazard is fitted on rows that include the failure row |
| step | the universal least favorable submodel at `step_size = 1e-3` | ridge-penalized steps of at most `delta_epsilon` in norm |
| stopping rule | a relative score of 1e-10 | every score at most 1/n in absolute value |

The censoring convention makes the `cleverly` standard error grow against the `tmle3` one with the
visit. The study module records about 2 percent at visit 5 in its smoke run. A paired row at a late
visit can therefore read as superior on coverage and not as equivalent. Both outcomes pass.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=1]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 1 | `cleverly` | -0.000779 to 0.0011 | 0.9500 | 1.0029 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=2]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 2 | `cleverly` | -0.000939 to 0.0015 | 0.9519 | 1.0039 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=3]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 3 | `cleverly` | -0.000924 to 0.0018 | 0.9531 | 1.0124 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=4]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 4 | `cleverly` | -0.000847 to 0.0021 | 0.9569 | 1.0364 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `ate_regimen[arm1 vs arm0 @ t=5]` | difference in cumulative risk between the plans "assign arm 1 at baseline and hold it" against "assign arm 0 at baseline and hold it" at horizon t = 5 | `cleverly` | -0.0013 to 0.0018 | 0.9544 | 1.0084 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=1]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 1 | `cleverly` | -0.000838 to 0.000594 | 0.9519 | 0.9963 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=2]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000932 to 0.000929 | 0.9575 | 0.9958 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=3]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 3 | `cleverly` | -0.0011 to 0.0010 | 0.9481 | 0.9839 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=4]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 4 | `cleverly` | -0.0016 to 0.000649 | 0.9500 | 0.9851 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm0 @ t=5]` | cumulative risk under the plan assign arm 0 at baseline and hold it at horizon t = 5 | `cleverly` | -0.0017 to 0.000473 | 0.9494 | 0.9830 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=1]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 1 | `cleverly` | -0.000566 to 0.000609 | 0.9425 | 1.0035 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=2]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 2 | `cleverly` | -0.000515 to 0.0011 | 0.9475 | 1.0152 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=3]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 3 | `cleverly` | -0.000492 to 0.0014 | 0.9537 | 1.0310 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=4]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 4 | `cleverly` | -0.000864 to 0.0012 | 0.9500 | 1.0363 | pass |
| five visits, a binary baseline treatment held over every node, visit dropout | `risk_regimen[arm1 @ t=5]` | cumulative risk under the plan assign arm 1 at baseline and hold it at horizon t = 5 | `cleverly` | -0.0015 to 0.000754 | 0.9519 | 0.9936 | pass |
<!-- /generated -->

All 15 implementation-estimand tests pass. The tests are the five risks of each arm and the five
differences, one for each visit. No replication raised an error, so no cell lost a replicate.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `clustered_inference` | `clustered_t5__cluster_robust` | positive | arm-0 risk at visit 5, 25 clusters of 80 that share W: the cluster-robust interval from the cluster-level influence curve of a single-fold hazard fit | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9373 to 0.9543, SE ratio 0.9660 to 1.0208, paired coverage gain 0.0875 to 0.1098 | pass |
| `clustered_inference` | `clustered_t5__iid_control` | control | arm-0 risk at visit 5, 25 clusters of 80 that share W: the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.8341 to 0.8610, SE ratio 0.7158 to 0.7546, paired coverage gain 0.0875 to 0.1098 | pass |
| `clustered_inference` | `clustered_t5_per_horizon__cluster_robust` | positive | arm-0 risk at visit 5, 25 clusters of 80 that share W, targeted one horizon at a time: the cluster-robust interval from the cluster-level influence curve of a single-fold hazard fit | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9413 to 0.9578, SE ratio 0.9595 to 1.0141, paired coverage gain 0.0906 to 0.1140 | pass |
| `clustered_inference` | `clustered_t5_per_horizon__iid_control` | control | arm-0 risk at visit 5, 25 clusters of 80 that share W, targeted one horizon at a time: the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.8348 to 0.8616, SE ratio 0.7114 to 0.7508, paired coverage gain 0.0906 to 0.1140 | pass |
| `double_robustness` | `survival_t5__both_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0026 to 0.0011, margin 0.0061, SE ratio 0.9974 | pass |
| `double_robustness` | `survival_t5__both_wrong` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0326 to 0.0363, margin 0.0061, SE ratio 0.9919 | pass |
| `double_robustness` | `survival_t5__mechanism_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0026 to 0.0012, margin 0.0063, SE ratio 0.9738 | pass |
| `double_robustness` | `survival_t5__outcome_correct` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0011 to 0.0025, margin 0.0061, SE ratio 0.9805 | pass |
| `interval_calibration` | `competing_t4__correctly_specified` | positive | difference in the death incidence at visit 4, two causes: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9458 to 0.9616, SE ratio 0.9867 to 1.0373, empirical efficiency ratio 0.9641 to 1.0134, reported efficiency ratio 0.9991 to 1.0008 | pass |
| `interval_calibration` | `isotonic_t2__correctly_specified` | positive | risk at visit 2 under treatment at both nodes, two-node law, fitted with the isotonic projection: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9240 to 0.9427, SE ratio 0.9391 to 0.9908, empirical efficiency ratio 0.9962 to 1.0500, reported efficiency ratio 0.9831 to 0.9905 | pass |
| `interval_calibration` | `policy_t3__correctly_specified` | positive | difference of a known baseline policy and arm 0 at visit 3, three arms without L_t: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9400 to 0.9566, SE ratio 0.9666 to 1.0175, empirical efficiency ratio 0.9853 to 1.0373, reported efficiency ratio 1.0022 to 1.0031 | pass |
| `interval_calibration` | `rule_t3__correctly_specified` | positive | difference of the rule that treats where W1 is one and arm 0 at visit 3, three arms without L_t: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9357 to 0.9529, SE ratio 0.9660 to 1.0168, empirical efficiency ratio 0.9858 to 1.0378, reported efficiency ratio 1.0017 to 1.0033 | pass |
| `interval_calibration` | `survival_t5__correctly_specified` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9439 to 0.9555, SE ratio 0.9817 to 1.0192, empirical efficiency ratio 0.9831 to 1.0205, reported efficiency ratio 1.0016 to 1.0022 | pass |
| `interval_calibration` | `survival_t5__noise_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8263 to 0.8459, SE ratio 0.7009 to 0.7274, empirical efficiency ratio 1.3774 to 1.4295, reported efficiency ratio 1.0016 to 1.0022 | pass |
| `interval_calibration` | `survival_t5__shrunken_se_control` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8156 to 0.8357, SE ratio 0.6874 to 0.7135, empirical efficiency ratio 0.9830 to 1.0202, reported efficiency ratio 0.7011 to 0.7015 | pass |
| `interval_calibration` | `weighted_t5__correctly_specified` | positive | difference at visit 5 under the weight 1 + W1 / 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9438 to 0.9599, SE ratio 0.9793 to 1.0298, empirical efficiency ratio 0.9727 to 1.0227, reported efficiency ratio 1.0012 to 1.0020 | pass |
| `interval_calibration` | `weighted_t5_per_horizon__correctly_specified` | positive | difference at visit 5 under the weight 1 + W1 / 2, targeted one horizon at a time: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9373 to 0.9543, SE ratio 0.9694 to 1.0225, empirical efficiency ratio 0.9799 to 1.0335, reported efficiency ratio 1.0015 to 1.0024 | pass |
| `isotonic_projection` | `isotonic_t2__moved` | diagnostic | risk at visit 2 under treatment at both nodes, two-node law, fitted with the isotonic projection: per replication, the distance the projection moved the level, against a truth of zero | none. The row is a diagnostic and states no verdict | binding share 0, 0 to 0.0011 | reported |
| `power` | `survival_t5__alternative` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `survival_t5__n_1000` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.0022, coverage 0.9165 to 0.9606, SE ratio 1.0000 | pass |
| `root_n_and_efficiency` | `survival_t5__n_2000` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0013, coverage 0.9107 to 0.9565, SE ratio 0.9990 | pass |
| `root_n_and_efficiency` | `survival_t5__n_8000` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000172, coverage 0.9401 to 0.9767, SE ratio 1.0524 | pass |
| `root_n_rate` | `survival_t5__empirical_sd` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5740 to -0.4845 | pass |
| `root_n_rate` | `survival_t5__reported_se` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5026 to -0.5013 | pass |
| `simultaneous_coverage` | `all_reported__pointwise_joint_control` | control | every parameter the calibration fit reports: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6469 to 0.6719 | pass |
| `simultaneous_coverage` | `all_reported__simultaneous_band` | positive | every parameter the calibration fit reports: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9393 to 0.9513 | pass |
| `simultaneous_coverage` | `curve_arm0__pointwise_joint_control` | control | the arm-0 risk curve alone, at the five visits: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8247 to 0.8443 | pass |
| `simultaneous_coverage` | `curve_arm0__simultaneous_band` | positive | the arm-0 risk curve alone, at the five visits: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9411 to 0.9530 | pass |
| `simultaneous_coverage` | `curve_arm1__pointwise_joint_control` | control | the arm-1 risk curve alone, at the five visits: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8284 to 0.8479 | pass |
| `simultaneous_coverage` | `curve_arm1__simultaneous_band` | positive | the arm-1 risk curve alone, at the five visits: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9394 to 0.9514 | pass |
| `targeting_necessity` | `survival_t5__targeted` | positive | difference of arm 1 and arm 0 at visit 5, five-visit held law: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000485 to 0.0032, margin 0.0062 | pass |
| `targeting_necessity` | `survival_t5__untargeted` | control | difference of arm 1 and arm 0 at visit 5, five-visit held law: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.1405 to 0.1405, margin 0 | pass |
<!-- /generated -->

All 31 scored cells pass, and the isotonic diagnostic reports a binding share of zero.
`properties.csv` has 32 rows, and `failed_replicates` is zero in all of them.

The cells use the same one-step fit as the primary study, except where the text says otherwise.
The learners of every positive cell are correct. The hazard learner is an additive logistic fit on
a one-cause law and the saturated cell fit on a two-cause law. The mechanism learners are the
saturated cell fits.

| family | positive cells | control that must fail |
| --- | --- | --- |
| `interval_calibration` | the visit-5 difference on `survival`; the death incidence difference at visit 4 on `competing`; the rule and the known policy against arm 0 at visit 3 on a three-arm law; the weighted visit-5 difference, one step and per horizon; the visit-2 level of a two-node law after the isotonic projection | reported standard errors scaled by 0.7, and one efficiency-bound unit of added noise, on the one-step `survival` cell only |
| `clustered_inference` | the arm-0 risk at visit 5 on 25 clusters of 80 rows that share `W`, one step and per horizon | the same fit read with independent standard errors |
| `double_robustness` | the hazard correct, the mechanism correct, or both | both wrong |
| `targeting_necessity` | the targeted estimate with a wrong hazard and the correct mechanism | the plug-in of the same fit |
| `root_n_and_efficiency` | n = 2,000 and 8,000 | n = 1,000 |
| `root_n_rate` | the empirical spread and the reported standard error | none. The slope interval must exclude -0.25 |
| `power` | the alternative on `survival` | none |
| `simultaneous_coverage` | the max-t band over each arm's five-visit risk curve, and over the fifteen reported parameters | the pointwise intervals of the same fits, read jointly |

The `weighted_t5` cells use the weight `1 + W1 / 2`. The `rule_t3` cell uses the rule that treats
where `W1` is one. The `policy_t3` cell uses a known baseline policy with probabilities
`(1/4, 1/4 + W1/4, 1/2 - W1/4)` over three arms. Both read one fit of a three-arm law without
`L_t`.

The wrong hazard learner is `InterceptOnly` from the property module of
[point-treatment survival](point-treatment-survival.md). A saturated hazard fitted in sample is the
maximum-likelihood estimate, so its initial score is zero and targeting does not move it. For that
reason the necessity cell uses a wrong hazard. The `untargeted` control predicts one hazard for
each node and both arms, so its plug-in contrast is zero in every sample.

The per-horizon calibration, weighted and clustered cells use `targeting="iterative"` with
`max_iter=100`. They run the per-horizon algorithm with weights and with clusters. The primary
study does not use it. The curve bands are the shipped multiplier band (`simultaneous_bands`, 2,000
rademacher draws, `random_state=0`) over the five risks of one arm. This is the band a fit of that
one regimen reports.

**The isotonic cell does not exercise a binding projection.** The `isotonic_t2` cell fits
`survival_model="iterated_mean"` with `isotonic=True` on a law with a treatment at two nodes. Its
curve rises from 0.375 at node 1 to 0.758 at node 2. That step is about nine standard errors at
n = 2,000. The diagnostic `isotonic_t2__moved` therefore records that the projection moved the level
in none of the 4,800 replications. The cell shows that the projection leaves the inference of a
regular curve unchanged. It is not evidence for an interval after a projection that moves a level.

The property module gives the reason. The increment between two horizons is the event mass at the
later node. The binding share is about $\Phi(-\sqrt{m})$, with $m$ the expected number of observed
events there. A projection binds often only where a node has almost no events. There Westling, van
der Laan and Carone (2020), Theorem 2, does not apply.

## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 15 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 15 | of those, passing |
| `property_cells_total` | 31 | repeated-sampling property cells |
| `property_cells_passed` | 31 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0370 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9425 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9400 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0892 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:iid_control_se_ceiling` | 0.8000 | IID-control SE-ratio ceiling |
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
| `margin:targeting_displacement` | 0.2500 | least targeting must move the estimate |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## Limitations

| limit | what it means for use |
| --- | --- |
| the comparator runs are pending | no `tmle3` output is compared. The `tmle3` rows of Study queue row 21 follow, and this page is Python property evidence until then |
| the declared `tmle3` settings differ | the bounds, the censoring convention, the step and the stopping rule differ by construction. The pending pairs read under the default margins and not as an exactness check |
| one primary scenario | the 15 truth tests read the five-visit `survival` law. The other laws enter the property cells only |
| learners that contain the truth | the primary fit uses unpenalized logistic regressions. The property cells use correct additive or saturated learners, apart from the declared wrong ones. A flexible learner is outside the evidence |
| the isotonic cell does not bind | the diagnostic reports a binding share of zero, so no cell supports an interval after a projection that moves a level |
| the per-horizon algorithm has three cells only | `targeting="iterative"` appears in one calibration, one weighted and one clustered cell. It has no double-robustness or necessity cell here. The [hazard study](point-survival-hazard.md) holds those cells |
| no cross-fitting | every fit uses one fold. The [cross-fitted study](point-survival-curve-crossfit.md) covers five folds |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_curve/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. It records the comparator as pending, with the runner and the image source hashes. Run
`python -m tests.canonical.point_survival_curve.regenerate` to regenerate the artifacts. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_curve/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/point_survival_curve/properties.csv)
carry every published row.
