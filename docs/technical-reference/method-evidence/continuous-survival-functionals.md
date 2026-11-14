# Continuous-time survival restricted means and risk ratios

This study validates the restricted means and risk ratios of `ContinuousSurvivalTMLE`: the fit
with `restricted_horizons=(2.32,)`, whose walk targets each arm's restricted mean time lost
(RMTL) to each cause beside the risks, and the risk ratios every fit reports. The section on
[restricted means, ratios and the curve](../continuous-time-survival.md#restricted-means-ratios-and-the-curve)
states the estimands, the influence curves and the exact-law evidence.

No maintained package targets these functionals in continuous time,
so no canonical implementation is compared. The study rests on exact truths and on repeated-sampling properties.
Every gated row passes, and a fit like the one measured reads supported under the conditions of
[limits](#limits).

## The estimand and the law

For arm $a$ and cause $j$ the targets are the risk $F_j(t \mid a) = E\,F_j(t \mid a, X)$, the
risk ratio $F_j(t \mid 1) / F_j(t \mid 0)$, the RMTL $E \int_0^\tau F_j(u \mid a, X)\,du$ and the
restricted mean survival time (RMST) $\tau - \sum_j \mathrm{RMTL}_j$, with the differences of the
two arms. Identification needs no unmeasured confounding of the treatment given $X$, censoring
independent of the event times given $(A, X)$, and positivity of the treatment and of remaining
uncensored through $\tau$.

The law is the base law of the [continuous-time survival study](continuous-time-survival.md):
two causes with constant cause-specific hazards given $(Z, X)$, an arm-specific term that only
the Cox interaction model holds, and exponential censoring at rate 0.15. The restriction time
$\tau = 2.32$ is the upper quartile of the observed event times. Each RMTL truth is
$E\,(h_j / h)\{\tau - (1 - e^{-h\tau}) / h\}$ over $X$, by the quadrature the risks use, in
`tests/studies/continuous_survival_functionals_law.py`.
`tests/unit/test_continuous_survival_functionals_design.py` checks each RMTL truth against the
integral of the risks over $(0, \tau)$ and each RMST against $\tau$ minus the RMTLs.

## The fit

The fit is the registered study's gated configuration: the default hazard and treatment
libraries, cross-fitted with 10 outer folds and selection inside 5 inner folds, at n = 1,000,
with target times at the three event-time quartiles and the restricted horizon added. Every
interval is the one the library ships: range-logit on $(0, 1)$ for a risk and on $(0, \tau)$ for
an RMTL or RMST, log for a ratio, identity for a difference. No maintained package targets these
functionals in continuous time, so the study names no comparator.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=1.12]` | difference in the risk of cause 1 by t = 1.12 between treatment and no treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000842 to 0.000526 | 0.9458 | 0.9989 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `ate_regimen[treated vs control, cause1 @ t=2.32]` | difference in the risk of cause 1 by t = 2.32 between treatment and no treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0011 to 0.000285 | 0.9526 | 1.0165 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=1.12]` | difference in the risk of cause 2 by t = 1.12 between treatment and no treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | 0.000183 to 0.0014 | 0.9349 | 0.9882 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `ate_regimen[treated vs control, cause2 @ t=2.32]` | difference in the risk of cause 2 by t = 2.32 between treatment and no treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000034 to 0.0013 | 0.9449 | 0.9939 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `cif_regimen[control, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under no treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000388 to 0.000751 | 0.9419 | 0.9919 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `cif_regimen[control, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under no treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000329 to 0.000773 | 0.9503 | 1.0116 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `cif_regimen[control, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under no treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000582 to 0.000114 | 0.9525 | 1.0100 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `cif_regimen[control, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under no treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000653 to 0.000211 | 0.9493 | 0.9993 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `cif_regimen[treated, cause1 @ t=1.12]` | risk of cause 1 by t = 1.12 under treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000398 to 0.000445 | 0.9470 | 0.9883 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `cif_regimen[treated, cause1 @ t=2.32]` | risk of cause 1 by t = 2.32 under treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000705 to 0.000310 | 0.9506 | 1.0009 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `cif_regimen[treated, cause2 @ t=1.12]` | risk of cause 2 by t = 1.12 under treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | 0.000043 to 0.0010 | 0.9190 | 0.9595 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `cif_regimen[treated, cause2 @ t=2.32]` | risk of cause 2 by t = 2.32 under treatment, in the restricted fit | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000137 to 0.000999 | 0.9378 | 0.9833 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmst_regimen[control @ t=2.32]` | restricted mean survival time up to 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0011 to 0.000937 | 0.9464 | 1.0005 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmst_regimen[treated @ t=2.32]` | restricted mean survival time up to 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0021 to 0.000023 | 0.9400 | 0.9814 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmst_regimen[treated vs control @ t=2.32]` | difference in the restricted mean survival time up to 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0024 to 0.000474 | 0.9472 | 0.9932 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmtl_regimen[control, cause1 @ t=2.32]` | restricted mean time lost to cause 1 up to 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000523 to 0.0014 | 0.9436 | 0.9968 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmtl_regimen[control, cause2 @ t=2.32]` | restricted mean time lost to cause 2 up to 2.32 under no treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0010 to 0.000261 | 0.9514 | 1.0038 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmtl_regimen[treated vs control, cause1 @ t=2.32]` | difference in the restricted mean time lost to cause 1 up to 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0018 to 0.000602 | 0.9480 | 0.9997 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmtl_regimen[treated vs control, cause2 @ t=2.32]` | difference in the restricted mean time lost to cause 2 up to 2.32 between treatment and no treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | 0.000485 to 0.0026 | 0.9413 | 0.9874 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmtl_regimen[treated, cause1 @ t=2.32]` | restricted mean time lost to cause 1 up to 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.000907 to 0.000643 | 0.9461 | 0.9870 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rmtl_regimen[treated, cause2 @ t=2.32]` | restricted mean time lost to cause 2 up to 2.32 under treatment | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | 0.000269 to 0.0020 | 0.9267 | 0.9636 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rr_regimen[treated vs control, cause1 @ t=1.12]` | ratio of the risks of cause 1 by t = 1.12 under treatment and no treatment, on the log scale | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0017 to 0.0037 | 0.9476 | 0.9982 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rr_regimen[treated vs control, cause1 @ t=2.32]` | ratio of the risks of cause 1 by t = 2.32 under treatment and no treatment, on the log scale | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0028 to 0.000824 | 0.9540 | 1.0167 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rr_regimen[treated vs control, cause2 @ t=1.12]` | ratio of the risks of cause 2 by t = 1.12 under treatment and no treatment, on the log scale | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0151 to -0.0057 | 0.9281 | 0.9692 | pass |
| the continuous-time competing-risks law of the continuous-time survival study; the default Cox libraries and treatment learner, ten outer folds, target times 0.46, 1.12 and 2.32 and one restricted horizon at 2.32, n = 1,000 | `rr_regimen[treated vs control, cause2 @ t=2.32]` | ratio of the risks of cause 2 by t = 2.32 under treatment and no treatment, on the log scale | `cleverly` ContinuousSurvivalTMLE with restricted_horizons=(2.32,): the one-step TMLE whose walk targets each arm's restricted mean time lost to each cause beside the risks (Rytgaard and van der Laan 2024) | -0.0070 to -0.000579 | 0.9417 | 0.9857 | pass |
<!-- /generated -->

Every primary row passes over 14,780 fits; 20 of the 14,800 draws failed with a recordable data
failure, an initial survival that reached zero before $\tau$, inside the declared bound of 148.
Coverage is 0.919 to 0.954 and the SE ratio 0.959 to 1.017. The lowest rows are the sparse
treated arm's risk of cause 2 at $t = 1.12$ (coverage 0.919, SE ratio 0.96), its RMTL at $\tau$
(0.927) and cause 2's risk ratio at $t = 1.12$ (0.928). The largest standardized bias is 0.047,
on that risk ratio. The coverage of each risk and risk difference is within 0.009 of the
registered study's row, which reads independent draws of the same law fitted without the
restricted horizon.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `rmst_rd__correctly_specified` | positive | the continuous-time competing-risks law, the difference in the restricted mean survival time up to 2.32 between treatment and no treatment, in the restricted fit, ten outer folds, n = 1,000: the default libraries, each of which holds the true model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9340 to 0.9551, SE ratio 0.9642 to 1.0302 | pass |
| `interval_calibration` | `rmst_rd__noise_control` | control | the continuous-time competing-risks law, the difference in the restricted mean survival time up to 2.32 between treatment and no treatment, in the restricted fit, ten outer folds, n = 1,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8061 to 0.8411, SE ratio 0.6753 to 0.7195 | pass |
| `interval_calibration` | `rmst_rd__shrunken_se_control` | control | the continuous-time competing-risks law, the difference in the restricted mean survival time up to 2.32 between treatment and no treatment, in the restricted fit, ten outer folds, n = 1,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8817 to 0.9098, SE ratio 0.8202 to 0.8775 | pass |
<!-- /generated -->

The calibration cell reads the RMST difference at $\tau$ on its own 3,200 draws: coverage 0.945
and SE ratio 0.997, inside both bands, with 6 draws lost to the same data failure. Its two
controls are derived from its rows and leave the band as declared: the noise control, which adds
independent noise to each estimate on the scale of the law's exact influence curve, reads SE ratio
0.70, and the shrunken-SE control, which multiplies each standard error by 0.85, reads 0.85. The
cell reads the coverage and SE-ratio bands only. `tests/studies/continuous_survival_functionals_properties.py` declares the cell.

The band row over every risk and risk difference waits for the default band of the band thread's
PR 2 and runs as a second phase under its own declaration. Until then the band of a fit with
restricted means reads experimental, as the registered study's does.

## Reported readings

These readings carry no verdict.

### Targeting necessity

`necessity-cells.csv` reads the RMTLs of a second fit on primary draws 0 to 999 whose walk targets
the risks only, so each RMTL is the integral of the targeted risk curve. Its score
$|P_n D_{\mathrm{RMTL}}| / \hat\sigma$ exceeds the walk's cutoff in 25% to 51% of the 999 fits,
by arm and cause, so the walk without the restricted targets leaves their score equations
unsolved. On this law the plug-in is nonetheless unbiased (standardized bias within 0.022) and
covers 0.937 to 0.965 with SE ratios 0.97 to 1.06. The declared reading applies: this law does
not separate the plug-in from the TMLE, and the claim that the restricted mean must be targeted
rests on the score test of `tests/unit/test_continuous_survival_functionals.py`, not on this
cell.

### The walk

`targeting-readings.csv` gives the steps, the share of fits that met the stopping rule, and the
floor shares. The primary fits met the rule in 99.96% of draws, in 3.8 steps on average and 54 at
most. The score equation held at 46% of 100 random times on average, against 40% in the
registered study.

## Sizing

Every size was fixed before the run from the pilot in
`tests/studies/continuous_survival_functionals_pilot.py`, 3,200 draws on a seed stream no declared
cell reads. The primary count, 14,800, follows the shared rule of `tests/studies/repeated_sizing.py`
with one implementation, set by the treated arm's risk of cause 2 at $t = 1.12$, which covered
0.917 in the pilot. The calibration count, 3,200, meets the at-target rule against the pilot's SE
ratio. `tests/unit/test_continuous_survival_functionals_design.py` recomputes both.

## Red-cell route

The study is gated, and no cell is red. The declared readings are in
`tests/canonical/continuous_survival_functionals/regenerate.py`.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 14800 | primary replications |
| `n` | 1000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 25 | truth tests passing |
| `independent_tests_total` | 25 | truth tests reported |
| `property_cells_passed` | 3 | property cells passing |
| `property_cells_total` | 3 | property cells reported |
| `subject_failures` | 20 | primary fits lost to a recordable data failure |
| `max_standardized_bias` | 0.0467 | largest primary standardized bias |
| `min_coverage` | 0.9190 | lowest primary coverage |
| `properties[interval_calibration/rmst_rd__correctly_specified]:se_ratio` | 0.9970 | SE ratio of the calibration cell |
| `properties[interval_calibration/rmst_rd__correctly_specified]:coverage` | 0.9452 | coverage of the calibration cell |
| `properties[interval_calibration/rmst_rd__noise_control]:se_ratio` | 0.6966 | SE ratio of the noise control |
| `properties[interval_calibration/rmst_rd__shrunken_se_control]:se_ratio` | 0.8474 | SE ratio of the shrunken-SE control |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2000 | standardized-bias margin |
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
| `margin:shrunken_se_factor` | 0.8500 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 148 | the same bound, in primary replications |

## Limits

| limit | what it means for use |
| --- | --- |
| one restriction time, at the last target time | the study restricts at $\tau = 2.32$, the last of three target times. A fit with several restricted horizons, or with a restriction time past its last target time, whose walk runs on a grid extended past it, reads experimental for every estimate; the extension is measured by the exact-law tests alone |
| restriction times by which 67% to 85% of the observed events have happened | the measured restriction time sits at 75% (69% to 80% over the draws). A restricted mean at an earlier restriction time weights the sparse early times, where the standard error of the risks ran low, more heavily, so it reads experimental |
| one law, the default libraries, n = 1,000 | the evidence covers the default Cox libraries and treatment learner, cross-fitted, at n = 1,000, with two causes and the risks' conditions of the [continuous-time survival study](continuous-time-survival.md#limits) |
| risk ratios of a fit with one restricted horizon | the study measures the risk ratio of the fit with $\tau$ only; the risk ratio of a fit without restricted means, an odds ratio and the survival view read experimental |
| no simultaneous band yet | the band row runs as a second phase; until then the band reads experimental |
| no comparator | no external implementation targets these functionals in continuous time |

## Reproduction

Run the pilot first, as `python -m tests.studies.continuous_survival_functionals_pilot --jobs N`.
Run `python -m tests.canonical.continuous_survival_functionals.regenerate --output DIR` with an
empty scratch directory outside the repository to regenerate the artifacts. The red-cell
readings and the budget are declared in
`tests/canonical/continuous_survival_functionals/regenerate.py` before any run.
