# Binding isotonic projection of a survival curve

This study validates the pointwise interval that `LTMLE(isotonic=True)` reports at a projected
level of a survival curve, on a law where the projection moves a level. The projection replaces
the fitted risks by their isotonic regression across horizons and keeps every standard error as
fitted, so the interval is centred at the projected level with the unprojected SE. The
[isotonic projection section](../longitudinal-tmle.md#the-isotonic-projection) states the rule and
its conditions. The study backs the support row `G31.isotonic` beside the `isotonic_t2` cell of
the [point survival curve study](point-survival-curve.md), whose saturated learners cannot bind.

On a two-horizon iterated-mean fit the difference of the two levels is
$D = P_n[H_1\, dY] + P_n[(1 - H_1)\, dQ^*]$ up to the node-1 score residuals, where $H_1$ is the
node-1 clever covariate, $dY = (1 - Y_1) Q^*_2 \ge 0$, and $dQ^*$ is the gap between the two
horizons' node-1 regressions. The projection moves the levels only when $D < 0$, which needs the
second term. Here the horizon-1 node-1 regression is a logistic outcome learner and the
horizon-2 one is a linear pseudo-outcome learner on a continuous covariate, so the gap is
nonzero and the level moves on about a third of draws.

`lmtp`'s `isotonic_projection` is transcribed in `tests/unit/test_isotonic_projection.py` as the
implementation reference, so no canonical implementation is compared. The study reads every
cell against its quadrature truth.

## What was fitted

| law | construction | fits |
| --- | --- | --- |
| LG2 | $W \sim N(0, 1)$; $A_1 \sim \operatorname{Bernoulli}(\operatorname{expit}(1.5 W))$; $C_1 \sim \operatorname{Bernoulli}(\operatorname{expit}(2.5 - 0.5 W))$; $Y_1 \sim \operatorname{Bernoulli}(\operatorname{expit}(-1.5 + 1.6 W - 0.5 A_1))$; $L_2 \sim \operatorname{Bernoulli}(\operatorname{expit}(-0.3 + 0.6 W + 0.4 A_1))$; $A_2 \sim \operatorname{Bernoulli}(\operatorname{expit}(1.5 W + 0.8 L_2 + 0.5 A_1))$; $C_2 \sim \operatorname{Bernoulli}(\operatorname{expit}(2.0 - 0.5 W + 0.5 L_2))$; $Y_2 \sim \operatorname{Bernoulli}(0.01\, \operatorname{expit}(-0.5 + 1.2 W^2 - 0.4 A_2))$, with $C = 1$ observed. The risk is strictly increasing: never treated 0.26074 and 0.26493, always treated 0.19717 and 0.20106 by 120-node Gauss-Hermite quadrature | `LTMLE({"never": 0, "always": 1}, reference="never")`, `survival_model="iterated_mean"`, `isotonic=True`, `simultaneous=False`, `g_bounds=(1e-8, 1.0)`, in sample |
| LG2 with slope 2.0 | the same law with treatment slope 2.0 in place of 1.5 in both treatment nodes, so the projection binds more often and moves further | the same fit, reported only |

The outcome, treatment and censoring learners are `LogisticRegression(C=inf, max_iter=100000,
tol=1e-13)`, and the pseudo-outcome learner is `LinearRegression`. A move is
$|\text{projected} - \text{unprojected}|$ in plug-in SEs of the level. The laws, learners, truths
and the score identity are in `tests/studies/_isotonic_binding_law.py`, and
`tests/unit/test_isotonic_binding_design.py` recomputes every declared number.
`tests/unit/test_isotonic_binding_witness.py` checks the identity at 1e-12 on a binding sample
and on a saturated sample that cannot bind, with mutation controls.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000; logistic outcome and mechanism learners, a linear pseudo-outcome learner | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | `cleverly` | -0.0017 to 0.000844 | 0.9481 | 0.9789 | pass |
<!-- /generated -->

The primary row reads the projected never horizon-1 risk on 1,600 draws of LG2. It passes on its
bias, coverage and SE ratio.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `binding_witness` | `never_t1` | positive | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the share of draws on which the projection moves the risk by visit 1 under never treated by more than 1e-6 SE | the lower limit of the share's exact 99% interval at least 0.25 | share 0.3334 to 0.3747 | pass |
| `interval_calibration` | `projected_t1__correctly_specified` | positive | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the risk by visit 1 under never treated: the interval centred at the projected level with the fit's unprojected SE | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9312 to 0.9515, SE ratio 0.9538 to 1.0133 | pass |
| `interval_calibration` | `projected_t1__noise_control` | control | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the risk by visit 1 under never treated: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.7955 to 0.8292, SE ratio 0.6632 to 0.7067 | pass |
| `interval_calibration` | `projected_t1__shrunken_se_control` | control | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the risk by visit 1 under never treated: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8050 to 0.8381, SE ratio 0.6673 to 0.7094 | pass |
| `material_move` | `never_t1` | positive | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the share of draws on which the projection moves the risk by visit 1 under never treated by at least 0.1 SE | the lower limit of the share's exact 99% interval at least 0.10 | share 0.1767 to 0.2108 | pass |
| `reported_large_move` | `binding_t1` | diagnostic | law LG2 with treatment slope 2.0 (larger moves), in-sample LTMLE with isotonic=True: the share of draws on which the projection moves the risk by visit 1 under never treated by more than 1e-6 SE | none. The row is a diagnostic and states no verdict | share 0.4327 to 0.4757 | reported |
| `reported_large_move` | `projected_t1` | diagnostic | law LG2 with treatment slope 2.0 (larger moves), in-sample LTMLE with isotonic=True: the risk by visit 1 under never treated, at the projected level | none. The row is a diagnostic and states no verdict | bias -0.0027 to -0.000515, coverage 0.9347, SE ratio 0.9426, intervals disagree on 0.0144, coverage difference -0.0081 to 0.0021, largest move 0.9944 SE | reported |
| `reported_large_move` | `stalled_fits` | diagnostic | law LG2 with treatment slope 2.0 (larger moves), in-sample LTMLE with isotonic=True: the share of replications with a logistic fit stopped at max_iter | none. The row is a diagnostic and states no verdict | 2 logistic fits stopped at max_iter | reported |
| `reported_large_move` | `unprojected_t1` | diagnostic | law LG2 with treatment slope 2.0 (larger moves), in-sample LTMLE with isotonic=True: the risk by visit 1 under never treated, at the unprojected level on the same draws | none. The row is a diagnostic and states no verdict | bias -0.000788 to 0.0014, coverage 0.9381, SE ratio 0.9498 | reported |
| `reported_levels` | `always_binding_t1` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the share of draws on which the projection moves the always horizon-1 level by more than 1e-6 SE | none. The row is a diagnostic and states no verdict | share 0.0907 to 0.1171 | reported |
| `reported_levels` | `always_projected_t1` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the risk by visit 1 under always treated, at the projected level | none. The row is a diagnostic and states no verdict | bias -0.000551 to 0.000438, coverage 0.9461, SE ratio 1.0040 | reported |
| `reported_levels` | `always_projected_t2` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the risk by visit 2 under always treated, at the projected level, on the draws that report an SE | none. The row is a diagnostic and states no verdict | bias -0.000130 to 0.0011, coverage 0.9372, SE ratio 0.8645, coverage on draws with an SE 0.9398 | reported |
| `reported_levels` | `always_se_present_t2` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the share of draws on which the always horizon-2 level reports an SE | none. The row is a diagnostic and states no verdict | share 0.8289 to 0.8602 | reported |
| `reported_levels` | `projected_t2` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the risk by visit 2 under never treated, at the projected level, on the draws that report an SE | none. The row is a diagnostic and states no verdict | bias -0.000230 to 0.0016, coverage 0.9486, SE ratio 1.0113, coverage on draws with an SE 0.9489 | reported |
| `reported_levels` | `se_present_t2` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the share of draws on which the never horizon-2 level reports an SE | none. The row is a diagnostic and states no verdict | share 0.9542 to 0.9707 | reported |
| `reported_levels` | `stalled_fits` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the share of replications with a logistic fit stopped at max_iter | none. The row is a diagnostic and states no verdict | 3 logistic fits stopped at max_iter | reported |
| `reported_levels` | `unprojected_t1` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the risk by visit 1 under never treated, the interval at the unprojected level on the same draws | none. The row is a diagnostic and states no verdict | bias -0.000696 to 0.000984, coverage 0.9411, SE ratio 0.9879 | reported |
| `reported_levels` | `unprojected_t2` | diagnostic | law LG2 (two nodes, a continuous baseline covariate, a strictly increasing risk), in-sample LTMLE with isotonic=True, n = 2,000: the risk by visit 2 under never treated, at the unprojected level, on the draws that report an SE | none. The row is a diagnostic and states no verdict | bias -0.0011 to 0.000752, coverage 0.9419, SE ratio 0.9901, coverage on draws with an SE 0.9432 | reported |
<!-- /generated -->

The projection moved the never horizon-1 level on 35% of draws and by at least 0.1 SE on 19%,
by at most 0.56 SE. The interval at the projected level reads coverage 0.942 with SE ratio 0.98,
inside the calibration bands, and both derived controls fail them. The projected and
unprojected intervals disagree on 0.64% of draws, and the paired coverage difference is +0.0008
with a 99% interval of $-0.0027$ to $0.0040$. At these move sizes the cell shows that the
projection costs no measurable coverage. It cannot show a gain.

## Reported readings

These rows carry no verdict.

**The unprojected interval and horizon 2.** On the same draws the unprojected horizon-1 interval
reads coverage 0.941 with SE ratio 0.99. Horizon 2 reports an SE only when a follower has an
event at node 2: 96% of never draws and 85% of always draws. On those draws the projected never
horizon-2 interval reads 0.949. That subset is a selection, so no horizon-2 cell is gated.

**The always regimen.** Its horizon-1 level moved on 10% of draws, and its projected horizon-1
interval reads coverage 0.946.

**The larger-move law.** With treatment slope 2.0 the projection binds on 45% of draws, and the
largest move is 0.99 SE. The projected interval reads coverage 0.935 against 0.938 unprojected,
with SE ratios 0.94 and 0.95, so the unprojected interval itself under-covers there. The paired
coverage difference is $-0.0033$ with a 99% interval of $-0.0081$ to $0.0021$.

**Stalled fits.** Three logistic fits of the 3,600 LG2 replications and two of the 3,600
slope-2.0 replications stopped at `max_iter` at `tol=1e-13`.

## Sizing

The property cells run 3,600 replications, the value of the arm-pooled predictive rule on a
pre-read of 7,200 draws (coverage 0.9468, SE ratio 0.984), and the primary phase runs 1,600.
Each witness share is gated at the lower limit of its exact 99% interval: at least 0.25 for the
binding share and at least 0.10 for the material share. The logistic learner is solved to
`tol=1e-13`. The design test reads a committed table of 300 draws refitted at `tol=1e-10`, in
which every indicator agrees and no move falls between 1e-9 and 1e-5 SE, and it refits three of
those draws. The pilot is `tests/studies/isotonic_binding_pilot.py`.

## Red-cell route

The study is registered under `reporting`, and no cell is red. A red calibration control is a
defect suspect, and a red witness would be a shortfall of `G31.isotonic` that states its share.
The declared reading is in `tests/canonical/isotonic_binding/regenerate.py`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 1 | truth tests passing |
| `independent_tests_total` | 1 | truth tests reported |
| `property_cells_passed` | 5 | gated property cells passing |
| `property_cells_total` | 5 | gated property cells reported |
| `max_standardized_bias` | 0.0219 | largest primary standardized bias |
| `min_coverage` | 0.9481 | lowest primary coverage |
| `properties[interval_calibration/projected_t1__correctly_specified]:coverage` | 0.9419 | coverage of the interval at the projected level |
| `properties[interval_calibration/projected_t1__correctly_specified]:se_ratio` | 0.9818 | SE ratio at the projected level |
| `properties[interval_calibration/projected_t1__correctly_specified]:disagree_share` | 0.0064 | share of draws on which the projected and unprojected intervals disagree |
| `properties[interval_calibration/projected_t1__correctly_specified]:coverage_difference` | 0.000833 | paired coverage difference, projected minus unprojected |
| `properties[interval_calibration/projected_t1__correctly_specified]:coverage_difference_ci_lower` | -0.0027 | its 99% lower limit |
| `properties[interval_calibration/projected_t1__correctly_specified]:coverage_difference_ci_upper` | 0.0040 | its 99% upper limit |
| `properties[interval_calibration/projected_t1__correctly_specified]:max_move` | 0.5638 | largest move, in SEs |
| `properties[binding_witness/never_t1]:coverage` | 0.3539 | share of draws on which the never horizon-1 level moved |
| `properties[binding_witness/never_t1]:coverage_ci_lower` | 0.3334 | its 99% lower limit |
| `properties[material_move/never_t1]:coverage` | 0.1933 | share of draws on which it moved by at least 0.1 SE |
| `properties[material_move/never_t1]:coverage_ci_lower` | 0.1767 | its 99% lower limit |
| `properties[reported_levels/unprojected_t1]:coverage` | 0.9411 | coverage of the unprojected horizon-1 interval |
| `properties[reported_levels/unprojected_t1]:se_ratio` | 0.9879 | its SE ratio |
| `properties[reported_levels/se_present_t2]:coverage` | 0.9631 | share of never draws with a horizon-2 SE |
| `properties[reported_levels/always_se_present_t2]:coverage` | 0.8450 | share of always draws with a horizon-2 SE |
| `properties[reported_levels/projected_t2]:coverage_with_se` | 0.9489 | projected never horizon-2 coverage on draws with an SE |
| `properties[reported_levels/always_binding_t1]:coverage` | 0.1033 | share of draws on which the always horizon-1 level moved |
| `properties[reported_levels/always_projected_t1]:coverage` | 0.9461 | coverage of the projected always horizon-1 interval |
| `properties[reported_levels/stalled_fits]:stalled_fits` | 3 | logistic fits stopped at `max_iter` on LG2 |
| `properties[reported_large_move/binding_t1]:coverage` | 0.4542 | binding share on the slope-2.0 law |
| `properties[reported_large_move/projected_t1]:max_move` | 0.9944 | largest move on the slope-2.0 law, in SEs |
| `properties[reported_large_move/projected_t1]:coverage` | 0.9347 | projected coverage on the slope-2.0 law |
| `properties[reported_large_move/projected_t1]:se_ratio` | 0.9426 | projected SE ratio on the slope-2.0 law |
| `properties[reported_large_move/unprojected_t1]:coverage` | 0.9381 | unprojected coverage on the slope-2.0 law |
| `properties[reported_large_move/unprojected_t1]:se_ratio` | 0.9498 | unprojected SE ratio on the slope-2.0 law |
| `properties[reported_large_move/projected_t1]:coverage_difference` | -0.0033 | paired coverage difference on the slope-2.0 law |
| `properties[reported_large_move/stalled_fits]:stalled_fits` | 2 | logistic fits stopped at `max_iter` on the slope-2.0 law |
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
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
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
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 16 | the same bound, in primary replications |

## Limitations

- **One learner pairing.** The moves arise because the two horizons' node-1 regressions use
  different learner classes. Other pairings are untested, and saturated node regressions with
  untruncated cell-share mechanisms cannot bind once the node-1 score is solved.
- **Moves of at most 0.56 SE.** At these sizes the projected and unprojected intervals disagree
  on 0.64% of draws, so the cell bounds the coverage the projection costs and cannot show a gain.
  The slope-2.0 law reaches 0.99 SE, but there the unprojected interval itself under-covers.
- **A strictly increasing truth.** A flat region of the true curve is untested. There the limit
  law of a projected level is not Gaussian, and Westling, van der Laan and Carone (2020) give no
  pointwise interval.
- **Only the never regimen's horizon 1 is gated.** The always regimen binds on 10% of draws, and
  horizon 2 reports an SE on a selected subset of draws, so both are reported.
- **Two nodes, one size, in sample.** Every cell is at n = 2,000 without cross-fitting.
- **No comparator.** `lmtp`'s projection is transcribed in the fast tier, not run here.

## Reproduction

`tests/canonical/isotonic_binding/regenerate.py` regenerates the study and states the run form.
It runs no reference process, because the study has no comparator. The manifest records the
seeds, the margins, the estimator configuration and the source and result hashes, and the run
log records the commit, the clean tree and the wall time.
