# Eliminated competing events longitudinal TMLE

This study validates `LTMLE` in `cleverly.longitudinal` with `eliminate=["death"]`: the risk of
relapse had death been prevented. Young, Stensrud, Tchetgen Tchetgen and Hernán (2020) define
the estimand and identify it by their g-formula (23), which treats a prevented death as
censoring. Within a node the order is censoring, death, then relapse. The node retention is the
product of a censoring factor and a prevented-cause factor, each fitted as its own model. The
[eliminated competing events](../longitudinal-tmle.md#eliminated-competing-events) section
states the estimator, its conditions and its exact-law checks.

No canonical implementation is compared. R `lmtp` with a composite censoring column (censoring
or death) is the matching implementation of the time-varying scenarios, and its runs are
pending. The study rests on exact truths and on repeated-sampling properties.

## What was fitted

| scenario | law, engine and estimands | folds |
| --- | --- | --- |
| `ltmle_time_varying` | time-varying law: two binary treatment nodes, a time-varying covariate, censoring, death and relapse at each node. Iterated-mean `LTMLE`; the risk under each of three plans at $t = 1$ and $t = 2$, and two contrasts against `never` | one |
| `ltmle_time_varying_crossfit` | the same law and estimands. No `competing_learner`, so a clone of the censoring learner fits the prevented-cause factor, the default path | two |
| `hazard_held_baseline` | held-baseline law: a three-level treatment held over three nodes with censoring, death and relapse. `LTMLE(survival_model="hazard")`; the risk of each arm at $t = 1, 2, 3$ and the contrasts against arm `a0` | one |

The three plans of the time-varying law are `never`, `always` and `continue_if_l2` (treat first,
then continue if $L_2 = 1$). The laws are in `tests/studies/eliminated_competing_law.py`, built
on `tests/discrete_law_competing.py` and `tests/discrete_law_point_survival.py`. Each truth is the
closed-form g-formula (23) written as quotients of cell masses, and each efficiency bound is the
exact efficient influence curve, a complex-step Gateaux derivative. Every nuisance is a
saturated cell mean, the nonparametric model on these supports, so the one-fold intervals rest
on a finite-dimensional class. The cumulative bounds, 0.001, never bind on these laws.

`SDR` is not measured. With saturated nuisances it returns the iterated `LTMLE` estimate and
standard error on every sample (`tests/unit/test_eliminated_competing_events_design.py`), so its
rows would repeat these. The other engines rest on the exact-law checks of
`tests/unit/test_eliminated_competing_law.py`.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[a1 vs a0 @ t=1]` | difference in the risk of relapse by t = 1, had death been prevented, between arm 1 and arm 0 | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000366 to 0.0012 | 0.9532 | 1.0114 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[a1 vs a0 @ t=2]` | difference in the risk of relapse by t = 2, had death been prevented, between arm 1 and arm 0 | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000584 to 0.0014 | 0.9404 | 0.9770 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[a1 vs a0 @ t=3]` | difference in the risk of relapse by t = 3, had death been prevented, between arm 1 and arm 0 | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000945 to 0.0012 | 0.9432 | 0.9723 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[a2 vs a0 @ t=1]` | difference in the risk of relapse by t = 1, had death been prevented, between arm 2 and arm 0 | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000774 to 0.000686 | 0.9464 | 0.9964 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[a2 vs a0 @ t=2]` | difference in the risk of relapse by t = 2, had death been prevented, between arm 2 and arm 0 | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000980 to 0.000683 | 0.9482 | 0.9864 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[a2 vs a0 @ t=3]` | difference in the risk of relapse by t = 3, had death been prevented, between arm 2 and arm 0 | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000913 to 0.000915 | 0.9454 | 0.9733 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a0 @ t=1]` | risk of relapse by t = 1 under arm 0, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000546 to 0.000441 | 0.9496 | 0.9893 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a0 @ t=2]` | risk of relapse by t = 2 under arm 0, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000691 to 0.000424 | 0.9421 | 0.9929 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a0 @ t=3]` | risk of relapse by t = 3 under arm 0, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000864 to 0.000330 | 0.9400 | 0.9675 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a1 @ t=1]` | risk of relapse by t = 1 under arm 1, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000249 to 0.000965 | 0.9493 | 1.0087 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a1 @ t=2]` | risk of relapse by t = 2 under arm 1, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000538 to 0.0011 | 0.9386 | 0.9761 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a1 @ t=3]` | risk of relapse by t = 3 under arm 1, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.0010 to 0.000705 | 0.9375 | 0.9779 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a2 @ t=1]` | risk of relapse by t = 1 under arm 2, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000648 to 0.000455 | 0.9454 | 0.9705 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a2 @ t=2]` | risk of relapse by t = 2 under arm 2, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000898 to 0.000335 | 0.9521 | 0.9812 | pass |
| three-node law with a three-level baseline treatment held over time, censoring, death and relapse, hazard LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[a2 @ t=3]` | risk of relapse by t = 3 under arm 2, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000960 to 0.000427 | 0.9482 | 0.9735 | pass |
| two-node law with a time-varying covariate, censoring, death and relapse, iterated-mean LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[always vs never @ t=1]` | difference in the risk of relapse by t = 1, had death been prevented, between treat at both times and treat at neither time | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000553 to 0.000633 | 0.9461 | 1.0046 | pass |
| two-node law with a time-varying covariate, censoring, death and relapse, iterated-mean LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[always vs never @ t=2]` | difference in the risk of relapse by t = 2, had death been prevented, between treat at both times and treat at neither time | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.0014 to 0.0011 | 0.9461 | 0.9994 | pass |
| two-node law with a time-varying covariate, censoring, death and relapse, iterated-mean LTMLE with saturated cell means, one fold, n = 16,000 | `ate_regimen[continue_if_l2 vs never @ t=2]` | difference in the risk of relapse by t = 2, had death been prevented, between treat first, then continue if L2 equals one and treat at neither time | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.0017 to 0.000635 | 0.9454 | 1.0124 | pass |
| two-node law with a time-varying covariate, censoring, death and relapse, iterated-mean LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[always @ t=1]` | risk of relapse by t = 1 under treat at both times, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000510 to 0.000438 | 0.9457 | 0.9953 | pass |
| two-node law with a time-varying covariate, censoring, death and relapse, iterated-mean LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[always @ t=2]` | risk of relapse by t = 2 under treat at both times, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.0014 to 0.000716 | 0.9514 | 0.9904 | pass |
| two-node law with a time-varying covariate, censoring, death and relapse, iterated-mean LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[continue_if_l2 @ t=2]` | risk of relapse by t = 2 under treat first, then continue if L2 equals one, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.0016 to 0.000269 | 0.9454 | 1.0028 | pass |
| two-node law with a time-varying covariate, censoring, death and relapse, iterated-mean LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[never @ t=1]` | risk of relapse by t = 1 under treat at neither time, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000429 to 0.000277 | 0.9511 | 1.0170 | pass |
| two-node law with a time-varying covariate, censoring, death and relapse, iterated-mean LTMLE with saturated cell means, one fold, n = 16,000 | `risk_regimen[never @ t=2]` | risk of relapse by t = 2 under treat at neither time, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000855 to 0.000564 | 0.9486 | 0.9855 | pass |
| the same two-node law, two outer folds, the death factor fitted by a clone of the censoring learner, n = 16,000 | `ate_regimen[always vs never @ t=1]` | difference in the risk of relapse by t = 1, had death been prevented, between treat at both times and treat at neither time | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000259 to 0.000898 | 0.9564 | 1.0317 | pass |
| the same two-node law, two outer folds, the death factor fitted by a clone of the censoring learner, n = 16,000 | `ate_regimen[always vs never @ t=2]` | difference in the risk of relapse by t = 2, had death been prevented, between treat at both times and treat at neither time | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | 0.000691 to 0.0038 | 0.9446 | 0.9649 | pass |
| the same two-node law, two outer folds, the death factor fitted by a clone of the censoring learner, n = 16,000 | `ate_regimen[continue_if_l2 vs never @ t=2]` | difference in the risk of relapse by t = 2, had death been prevented, between treat first, then continue if L2 equals one and treat at neither time | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.0012 to 0.0013 | 0.9454 | 0.9914 | pass |
| the same two-node law, two outer folds, the death factor fitted by a clone of the censoring learner, n = 16,000 | `risk_regimen[always @ t=1]` | risk of relapse by t = 1 under treat at both times, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000302 to 0.000614 | 0.9546 | 1.0334 | pass |
| the same two-node law, two outer folds, the death factor fitted by a clone of the censoring learner, n = 16,000 | `risk_regimen[always @ t=2]` | risk of relapse by t = 2 under treat at both times, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | 0.000696 to 0.0034 | 0.9464 | 0.9491 | pass |
| the same two-node law, two outer folds, the death factor fitted by a clone of the censoring learner, n = 16,000 | `risk_regimen[continue_if_l2 @ t=2]` | risk of relapse by t = 2 under treat first, then continue if L2 equals one, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.0012 to 0.000858 | 0.9457 | 0.9694 | pass |
| the same two-node law, two outer folds, the death factor fitted by a clone of the censoring learner, n = 16,000 | `risk_regimen[never @ t=1]` | risk of relapse by t = 1 under treat at neither time, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000522 to 0.000195 | 0.9532 | 1.0034 | pass |
| the same two-node law, two outer folds, the death factor fitted by a clone of the censoring learner, n = 16,000 | `risk_regimen[never @ t=2]` | risk of relapse by t = 2 under treat at neither time, had death been prevented | `cleverly` LTMLE with eliminate=['death']: the relapse risk had death been prevented, with retention fitted as a censoring factor and a prevented-cause factor | -0.000912 to 0.000548 | 0.9468 | 0.9961 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `hb_a1_t2__mechanism_correct` | positive | the held-baseline eliminated-competing-events law, the risk by t = 2 under arm 1, n = 16,000: a prior-only outcome model, with the treatment, censoring and prevented-cause factors correct, so the targeting step does the adjusting | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0014 to 0.0013, margin 0.0040, SE ratio 1.0114 | pass |
| `double_robustness` | `hb_a2_t1__outcome_wrong_competing_wrong` | control | the held-baseline eliminated-competing-events law, the risk by t = 1 under arm 2, n = 16,000: a prior-only outcome model and a prior-only prevented-cause factor, with the treatment and censoring factors correct | bias interval must fall entirely outside the margin | bias 0.0173 to 0.0191, margin 0.0027, SE ratio 0.8879 | pass |
| `double_robustness` | `tv_never_t2__outcome_wrong_competing_wrong` | control | the time-varying eliminated-competing-events law, the risk by t = 2 under the plan treat at neither time, n = 16,000: a prior-only outcome model and a prior-only prevented-cause factor, with the treatment and censoring factors correct | bias interval must fall entirely outside the margin | bias -0.0213 to -0.0190, margin 0.0036, SE ratio 0.9549 | pass |
| `double_robustness` | `tv_rule_t2__mechanism_correct` | positive | the time-varying eliminated-competing-events law, the risk by t = 2 under the plan treat first, then continue if L2 equals one, n = 16,000: a prior-only outcome model, with the treatment, censoring and prevented-cause factors correct, so the targeting step does the adjusting | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000942 to 0.0022, margin 0.0049, SE ratio 1.0222 | pass |
| `interval_calibration` | `hazard_a1_t3__correctly_specified` | positive | the held-baseline eliminated-competing-events law, the risk difference by t = 3 between arms 1 and 0, hazard LTMLE, n = 16,000: every nuisance (outcome, treatment, censoring and prevented-cause factor) is a saturated cell mean with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9362 to 0.9533, SE ratio 0.9669 to 1.0216, empirical efficiency ratio 0.9778 to 1.0327, reported efficiency ratio 0.9970 to 1.0005 | pass |
| `interval_calibration` | `hazard_a1_t3__noise_control` | control | the held-baseline eliminated-competing-events law, the risk difference by t = 3 between arms 1 and 0, hazard LTMLE, n = 16,000: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8173 to 0.8453, SE ratio 0.6922 to 0.7284, empirical efficiency ratio 1.3706 to 1.4426, reported efficiency ratio 0.9969 to 1.0005 | pass |
| `interval_calibration` | `hazard_a1_t3__shrunken_se_control` | control | the held-baseline eliminated-competing-events law, the risk difference by t = 3 between arms 1 and 0, hazard LTMLE, n = 16,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8132 to 0.8415, SE ratio 0.6765 to 0.7141, empirical efficiency ratio 0.9792 to 1.0336, reported efficiency ratio 0.6979 to 0.7003 | pass |
| `interval_calibration` | `ltmle_always_t2__correctly_specified` | positive | the time-varying eliminated-competing-events law, the risk difference by t = 2 between treating at both times and at neither, one-fold iterated LTMLE, n = 16,000: every nuisance (outcome, treatment, censoring and prevented-cause factor) is a saturated cell mean with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9384 to 0.9553, SE ratio 0.9654 to 1.0170, empirical efficiency ratio 0.9813 to 1.0344, reported efficiency ratio 0.9974 to 0.9996 | pass |
| `interval_calibration` | `ltmle_always_t2__noise_control` | control | the time-varying eliminated-competing-events law, the risk difference by t = 2 between treating at both times and at neither, one-fold iterated LTMLE, n = 16,000: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8277 to 0.8550, SE ratio 0.6919 to 0.7296, empirical efficiency ratio 1.3688 to 1.4435, reported efficiency ratio 0.9975 to 0.9995 | pass |
| `interval_calibration` | `ltmle_always_t2__shrunken_se_control` | control | the time-varying eliminated-competing-events law, the risk difference by t = 2 between treating at both times and at neither, one-fold iterated LTMLE, n = 16,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8117 to 0.8401, SE ratio 0.6755 to 0.7109, empirical efficiency ratio 0.9831 to 1.0347, reported efficiency ratio 0.6982 to 0.6997 | pass |
<!-- /generated -->

"Wrong" in the double-robustness cells is a prior-only classifier, which reads no covariate.
The two positive cells take a wrong outcome model with every factor right, so the targeting step
does the adjusting. On the exact laws their untargeted plug-ins sit 1.9 and 9.4 efficient
standard errors off the truth. The two controls also take a wrong prevented-cause factor, and
their exact-law limits sit 1.39 and 1.63 efficient standard errors off, so these cells witness
that the death factor enters the clever covariate. Configurations that coincide with these on a
saturated law (a correct outcome model whatever the death factor, `SequentialIPW`, `SDR`) are
unit identities in the design test, not cells. The calibration cells read the exact efficient
influence-curve spread against the band 0.90 to 1.10.
`tests/studies/eliminated_competing_events_properties.py` declares every cell.

## Sizing

Every size was fixed before the run from the pilot in
`tests/studies/eliminated_competing_events_pilot.py`, on seed streams no declared cell reads.
At n = 4,000 the time-varying scenarios read coverage 0.915 to 0.940 at $t = 2$ with one fold,
and a standardized bias up to 0.23 with two folds, so every scenario moved to 16,000. The
primary replication count follows the shared rule of `tests/studies/repeated_sizing.py` with one
implementation. The calibration count is the smallest multiple of 400 whose family-wise pass
probability reaches 0.984, read from 8,000 pilot draws per cell. The design test recomputes
both.

## Red-cell route

The study is gated, and no cell is red. The declared readings are in
`tests/canonical/eliminated_competing_events/regenerate.py`.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
exact standard errors of the efficient influence curve at the primary n. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2800 | primary replications |
| `n` | 16000 | observations per primary replication |
| `independent_tests_passed` | 31 | truth tests passing |
| `independent_tests_total` | 31 | truth tests reported |
| `property_cells_passed` | 10 | property cells passing |
| `property_cells_total` | 10 | property cells reported |
| `max_standardized_bias` | 0.0733 | largest primary standardized bias |
| `min_coverage` | 0.9375 | lowest primary coverage |
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
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 28 | the same bound, in primary replications |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `bound:ate_regimen[a1 vs a0 @ t=1]_standard_error` | 0.0161 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:ate_regimen[a1 vs a0 @ t=2]_standard_error` | 0.0196 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:ate_regimen[a1 vs a0 @ t=3]_standard_error` | 0.0210 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:ate_regimen[a2 vs a0 @ t=1]_standard_error` | 0.0149 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:ate_regimen[a2 vs a0 @ t=2]_standard_error` | 0.0168 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:ate_regimen[a2 vs a0 @ t=3]_standard_error` | 0.0182 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:ate_regimen[always vs never @ t=1]_standard_error` | 0.0122 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:ate_regimen[always vs never @ t=2]_standard_error` | 0.0257 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:ate_regimen[continue_if_l2 vs never @ t=2]_standard_error` | 0.0241 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a0 @ t=1]_standard_error` | 0.0100 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a0 @ t=2]_standard_error` | 0.0114 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a0 @ t=3]_standard_error` | 0.0119 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a1 @ t=1]_standard_error` | 0.0126 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a1 @ t=2]_standard_error` | 0.0160 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a1 @ t=3]_standard_error` | 0.0174 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a2 @ t=1]_standard_error` | 0.0110 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a2 @ t=2]_standard_error` | 0.0124 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[a2 @ t=3]_standard_error` | 0.0139 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[always @ t=1]_standard_error` | 0.0097 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[always @ t=2]_standard_error` | 0.0213 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[continue_if_l2 @ t=2]_standard_error` | 0.0194 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[never @ t=1]_standard_error` | 0.0074 | exact influence-curve standard error of this estimand at the primary n = 16,000 |
| `bound:risk_regimen[never @ t=2]_standard_error` | 0.0144 | exact influence-curve standard error of this estimand at the primary n = 16,000 |

## Limits

| limit | what it means for use |
| --- | --- |
| saturated cell means at every slot | the study does not test a flexible learner. The one-fold claim rests on the finite-dimensional class |
| discrete laws at n = 16,000 | at 4,000 the time-varying scenarios under-covered, so smaller samples on laws this sparse are not claimed |
| one prevented cause and one remaining cause | more causes, or a prevented cause with a treatment-dependent hazard beyond these laws, are not measured |
| two treatment nodes, or three held nodes | longer panels and later horizons are not claimed |
| sparse influence at $t = 3$ on the hazard path | the fits warn `SparseInfluenceWarning` there; the rows pass, and the warning stays the reader's cue |
| the cross-fitted scenario has no property cell | its rows pass, and its spread sits above the efficient one at $t = 2$ while the reported SE tracks it |
| no comparator | R `lmtp` runs with a composite censoring column are pending, so the study has no paired verdict |
| no weights, clusters or missing outcomes | each is a separate composition and reads experimental |

## Reproduction

`tests/canonical/eliminated_competing_events/regenerate.py` regenerates the study. It runs no
reference process, because the study has no comparator.
