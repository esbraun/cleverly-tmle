# Longitudinal incremental interventions

This study validates `IPSI` in `cleverly.longitudinal`. `IPSI` is the one-step estimator of Kennedy
(2019), Algorithm 1, for incremental propensity-score interventions over time. The
[incremental interventions over time](../longitudinal-tmle.md#incremental-interventions-over-time)
section states the estimator, its conditions and its exact-law evidence.

The evidence label is **Python property evidence, with comparator runs pending**. The study
declares R `npcausal` 0.1.0 at commit `56a5ac11` as its comparator. The comparator runs have not
registered, so this page publishes no paired verdict.
[Comparator runs pending](../../development/method-benchmarking.md#comparator-runs-pending) states
the rule. The [dropout study](longitudinal-incremental-interventions-dropout.md) measures the same
estimator under dropout, survival outcomes, weights and clusters.

## What was compared

| scenario | law and plans | folds |
| --- | --- | --- |
| `ipsi_folds_2` | law `A` of `tests/discrete_law_longitudinal_incremental.py`: a binary baseline covariate, a binary treatment, a binary `L2` and `L3`, and a continuous end-of-study outcome over three nodes. Three plans tilt every node by 0.5, 1 (the natural course) and 2 | two |
| `ipsi_folds_5` | the same law and plans | five |

Each scenario reports five estimands: the three plan means and the two contrasts with the natural
course. The one-fold fit is not a primary scenario. `npcausal` 0.1.0 cannot fit one fold, because
its training rows are empty at `nsplits = 1`. The one-fold fit is the calibration label
`contrast_x2_in_sample` instead.

| setting | `cleverly` | R `npcausal` (declared, pending) |
| --- | --- | --- |
| datasets | 1,000 samples of 96,000 rows per scenario | the identical rows |
| nuisance fits | saturated cell means (`CellMeans`) at every slot | `SL.cellmeans` in `tests/canonical/npcausal_ipsi_adapter.R` |
| folds | the Python fold assignment | the same assignment, which replaces the `npcausal` fold draw |
| treatment model | cell means by node and history | one model over every `(id, time)` row, with the time block and the history cell as design |
| intervals | pointwise 95% Wald from the influence curve | the same, from its influence values (`ifvals`), with no factor between the two standard errors |

With equal cell partitions and equal fallbacks, the two recursions are the same function of the
sample. The pairing gates G1 and G2 in `tests/canonical/npcausal_longitudinal_incremental/gates.py`
test this before the study.

Each gate requires agreement below `PAIRING_GATE` (1e-10) on the
estimate, every influence value and the standard error. The committed `gates.csv` holds 42 plan
rows. All 42 pass, and the largest gap is 2.5e-12, on an influence value. The gates are
pre-declaration checks. They are not paired verdicts.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-node law with a continuous outcome, two folds | `ate_regimen[odds x0.5 vs natural course]` | difference in mean outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | `cleverly` `IPSI`, one-step over time | -0.000593 to -0.000138 | 0.9470 | 1.0251 | pass |
| three-node law with a continuous outcome, two folds | `ate_regimen[odds x2 vs natural course]` | difference in mean outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | `cleverly` `IPSI`, one-step over time | 0.000137 to 0.000632 | 0.9440 | 0.9934 | pass |
| three-node law with a continuous outcome, two folds | `ey_regimen[natural course]` | mean outcome under the plan leave the observed treatment mechanism unchanged | `cleverly` `IPSI`, one-step over time | -0.000693 to 0.000554 | 0.9490 | 1.0030 | pass |
| three-node law with a continuous outcome, two folds | `ey_regimen[odds x0.5]` | mean outcome under the plan multiply the treatment odds by 0.5 | `cleverly` `IPSI`, one-step over time | -0.0010 to 0.000176 | 0.9490 | 1.0012 | pass |
| three-node law with a continuous outcome, two folds | `ey_regimen[odds x2]` | mean outcome under the plan multiply the treatment odds by two | `cleverly` `IPSI`, one-step over time | -0.000297 to 0.000928 | 0.9530 | 1.0119 | pass |
| three-node law with a continuous outcome, five folds | `ate_regimen[odds x0.5 vs natural course]` | difference in mean outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | `cleverly` `IPSI`, one-step over time | -0.000533 to -0.000063 | 0.9600 | 0.9935 | pass |
| three-node law with a continuous outcome, five folds | `ate_regimen[odds x2 vs natural course]` | difference in mean outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | `cleverly` `IPSI`, one-step over time | -0.000010 to 0.000469 | 0.9530 | 1.0282 | pass |
| three-node law with a continuous outcome, five folds | `ey_regimen[natural course]` | mean outcome under the plan leave the observed treatment mechanism unchanged | `cleverly` `IPSI`, one-step over time | -0.000348 to 0.000903 | 0.9350 | 1.0001 | pass |
| three-node law with a continuous outcome, five folds | `ey_regimen[odds x0.5]` | mean outcome under the plan multiply the treatment odds by 0.5 | `cleverly` `IPSI`, one-step over time | -0.000629 to 0.000589 | 0.9450 | 1.0041 | pass |
| three-node law with a continuous outcome, five folds | `ey_regimen[odds x2]` | mean outcome under the plan multiply the treatment odds by two | `cleverly` `IPSI`, one-step over time | -0.000122 to 0.0011 | 0.9480 | 0.9859 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

**Python property evidence, with comparator runs pending.** This section holds no paired table,
because the comparator study has not run. `equivalence.csv` records no row.

| item | state |
| --- | --- |
| comparator | R `npcausal` 0.1.0 `ipsi()` at commit `56a5ac11`, on a digest-pinned R 4.5.2 image |
| reason pending | the `npcausal` R phase runs on the local study queue and registers in a later change |
| paired cell of the grid | 0/0 external comparisons, with the pending label |
| declared margins | the paired margins in the measured table below. No cell reads them yet |
| what the later run adds | five paired tests for each scenario, read against the default `Margins()` |

The owner's rule applies: a method merges on its Python evidence, and the comparator study
follows ([Theory and comparators](../../architecture-invariants.md#theory-and-comparators)). When
the run registers, the paired block and the paired counts replace this section.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `contrast_x2__correctly_specified` | positive | odds-x2 against the natural course, three-node law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9327 to 0.9546, SE ratio 0.9489 to 1.0161, empirical efficiency ratio 0.9848 to 1.0545, reported efficiency ratio 1.0005 to 1.0009 | pass |
| `interval_calibration` | `contrast_x2__noise_control` | control | odds-x2 against the natural course, three-node law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8120 to 0.8476, SE ratio 0.6742 to 0.7210, empirical efficiency ratio 1.3880 to 1.4842, reported efficiency ratio 1.0005 to 1.0009 | pass |
| `interval_calibration` | `contrast_x2__shrunken_se_control` | control | odds-x2 against the natural course, three-node law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8002 to 0.8368, SE ratio 0.6639 to 0.7111, empirical efficiency ratio 0.9851 to 1.0550, reported efficiency ratio 0.7003 to 0.7006 | pass |
| `interval_calibration` | `contrast_x2_in_sample__correctly_specified` | positive | odds-x2 against the natural course, one fold: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9374 to 0.9585, SE ratio 0.9492 to 1.0152, empirical efficiency ratio 0.9851 to 1.0535, reported efficiency ratio 0.9999 to 1.0003 | pass |
| `interval_calibration` | `contrast_x2_in_sample__noise_control` | control | odds-x2 against the natural course, one fold: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8207 to 0.8556, SE ratio 0.6925 to 0.7391, empirical efficiency ratio 1.3532 to 1.4442, reported efficiency ratio 0.9999 to 1.0004 | pass |
| `interval_calibration` | `contrast_x2_in_sample__shrunken_se_control` | control | odds-x2 against the natural course, one fold: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8006 to 0.8371, SE ratio 0.6646 to 0.7111, empirical efficiency ratio 0.9845 to 1.0533, reported efficiency ratio 0.6999 to 0.7002 | pass |
| `interval_calibration` | `known_x2__correctly_specified` | positive | odds-x2 mean with the true treatment probabilities declared: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9385 to 0.9594, SE ratio 0.9785 to 1.0509, empirical efficiency ratio 0.9519 to 1.0223, reported efficiency ratio 1.0002 to 1.0006 | pass |
| `interval_calibration` | `known_x2__noise_control` | control | odds-x2 mean with the true treatment probabilities declared: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8307 to 0.8648, SE ratio 0.7020 to 0.7507, empirical efficiency ratio 1.3328 to 1.4254, reported efficiency ratio 1.0002 to 1.0007 | pass |
| `interval_calibration` | `known_x2__shrunken_se_control` | control | odds-x2 mean with the true treatment probabilities declared: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8224 to 0.8572, SE ratio 0.6858 to 0.7349, empirical efficiency ratio 0.9529 to 1.0211, reported efficiency ratio 0.7001 to 0.7005 | pass |
| `interval_calibration` | `mean_x2__correctly_specified` | positive | odds-x2 mean at every node, three-node law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9392 to 0.9600, SE ratio 0.9834 to 1.0476, empirical efficiency ratio 0.9547 to 1.0170, reported efficiency ratio 1.0001 to 1.0004 | pass |
| `interval_calibration` | `mean_x2__noise_control` | control | odds-x2 mean at every node, three-node law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8144 to 0.8498, SE ratio 0.6815 to 0.7267, empirical efficiency ratio 1.3763 to 1.4677, reported efficiency ratio 1.0001 to 1.0004 | pass |
| `interval_calibration` | `mean_x2__shrunken_se_control` | control | odds-x2 mean at every node, three-node law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8189 to 0.8540, SE ratio 0.6869 to 0.7333, empirical efficiency ratio 0.9548 to 1.0193, reported efficiency ratio 0.7001 to 0.7003 | pass |
| `interval_calibration` | `no_positivity__correctly_specified` | positive | odds-x2 contrast on a law with propensities of exactly 0 and 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9345 to 0.9561, SE ratio 0.9653 to 1.0331, empirical efficiency ratio 0.9680 to 1.0361, reported efficiency ratio 0.9997 to 1.0004 | pass |
| `interval_calibration` | `no_positivity__noise_control` | control | odds-x2 contrast on a law with propensities of exactly 0 and 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8162 to 0.8514, SE ratio 0.6938 to 0.7393, empirical efficiency ratio 1.3527 to 1.4411, reported efficiency ratio 0.9997 to 1.0004 | pass |
| `interval_calibration` | `no_positivity__shrunken_se_control` | control | odds-x2 contrast on a law with propensities of exactly 0 and 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8127 to 0.8483, SE ratio 0.6759 to 0.7234, empirical efficiency ratio 0.9676 to 1.0358, reported efficiency ratio 0.6998 to 0.7003 | pass |
| `interval_calibration` | `per_node__correctly_specified` | positive | odds multipliers 2, 0.5 and 1.5 at the three nodes: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9381 to 0.9591, SE ratio 0.9777 to 1.0445, empirical efficiency ratio 0.9575 to 1.0229, reported efficiency ratio 1.0000 to 1.0003 | pass |
| `interval_calibration` | `per_node__noise_control` | control | odds multipliers 2, 0.5 and 1.5 at the three nodes: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8179 to 0.8530, SE ratio 0.6929 to 0.7402, empirical efficiency ratio 1.3513 to 1.4435, reported efficiency ratio 1.0000 to 1.0003 | pass |
| `interval_calibration` | `per_node__shrunken_se_control` | control | odds multipliers 2, 0.5 and 1.5 at the three nodes: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8207 to 0.8556, SE ratio 0.6852 to 0.7323, empirical efficiency ratio 0.9561 to 1.0217, reported efficiency ratio 0.7000 to 0.7002 | pass |
| `interval_calibration` | `tilt_then_static__correctly_specified` | positive | odds-x2 at nodes 1 and 2, then arm 1 at node 3: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9436 to 0.9636, SE ratio 0.9811 to 1.0474, empirical efficiency ratio 0.9559 to 1.0207, reported efficiency ratio 1.0012 to 1.0016 | pass |
| `interval_calibration` | `tilt_then_static__noise_control` | control | odds-x2 at nodes 1 and 2, then arm 1 at node 3: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8148 to 0.8502, SE ratio 0.6759 to 0.7214, empirical efficiency ratio 1.3881 to 1.4817, reported efficiency ratio 1.0012 to 1.0016 | pass |
| `interval_calibration` | `tilt_then_static__shrunken_se_control` | control | odds-x2 at nodes 1 and 2, then arm 1 at node 3: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8193 to 0.8543, SE ratio 0.6866 to 0.7349, empirical efficiency ratio 0.9538 to 1.0211, reported efficiency ratio 0.7008 to 0.7011 | pass |
| `mechanism_term_necessity` | `contrast_x2__full_eif` | positive | odds-x2 against the natural course, three-node law: the reported standard error from the full influence curve | the SE-ratio interval inside the efficiency-ratio band | SE ratio 0.9619 to 1.0285 | pass |
| `mechanism_term_necessity` | `contrast_x2__mechanism_term_dropped` | control | odds-x2 against the natural course, three-node law: the same estimates with the treatment-mechanism term, on the outcome's own scale, removed from the curve | the SE-ratio interval must fall below the calibration band | SE ratio 0.7524 to 0.8025 | pass |
| `power` | `contrast_x1_5__alternative` | positive | odds-x1.5 against the natural course, small arm effects: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.8900, 0.8708 to 0.9073 | pass |
| `root_n_and_efficiency` | `contrast_x2__n_24000` | control | odds-x2 against the natural course, three-node law: bias, coverage and SE calibration at n = 24,000 | coverage interval lies below nominal or clears the declared floor | bias 0.000729, coverage 0.9389 to 0.9726, SE ratio 1.0254 | pass |
| `root_n_and_efficiency` | `contrast_x2__n_64000` | positive | odds-x2 against the natural course, three-node law: bias, coverage and SE calibration at n = 64,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000343, coverage 0.9295 to 0.9661, SE ratio 1.0070 | pass |
| `root_n_and_efficiency` | `contrast_x2__n_96000` | positive | odds-x2 against the natural course, three-node law: bias, coverage and SE calibration at n = 96,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000094, coverage 0.9283 to 0.9652, SE ratio 0.9911 | pass |
| `root_n_rate` | `contrast_x2__empirical_sd` | positive | odds-x2 against the natural course, three-node law: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5333 to -0.4210 | pass |
| `root_n_rate` | `contrast_x2__reported_se` | positive | odds-x2 against the natural course, three-node law: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5009 to -0.5000 | pass |
| `simultaneous_coverage` | `nine_tilts__pointwise_joint_control` | control | nine odds multipliers from 0.25 to 4, their means and eight contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7651 to 0.8125 | pass |
| `simultaneous_coverage` | `nine_tilts__simultaneous_band` | positive | nine odds multipliers from 0.25 to 4, their means and eight contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9239 to 0.9520 | pass |
| `single_robustness` | `contrast_x2__mechanism_constant` | diagnostic | odds-x2 against the natural course, three-node law: constant treatment mechanism and cell-mean outcome regressions, beside the exact-law limit | reported only | bias 0.1252 to 0.1261, margin 0.0013, coverage 0.0000 | reported |
| `single_robustness` | `contrast_x2__mechanism_correct__n_48000` | positive | odds-x2 against the natural course, three-node law: cell-mean treatment mechanism and linear outcome regressions, at n = 48,000 | bias inside the margin at n = 48,000 and at n = 96,000 | bias -0.000083 to 0.000610, margin 0.0011, coverage 0.9530 | pass |
| `single_robustness` | `contrast_x2__mechanism_correct__n_96000` | positive | odds-x2 against the natural course, three-node law: cell-mean treatment mechanism and linear outcome regressions, at n = 96,000 | bias inside the margin at n = 48,000 and at n = 96,000 | bias 0.000160 to 0.000648, margin 0.000747, coverage 0.9450 | pass |
| `targeting_necessity` | `contrast_x2__one_step` | positive | odds-x2 against the natural course, three-node law: the one-step estimate with linear outcome regressions | bias inside the margin | bias -0.000058 to 0.000424, margin 0.000738 | pass |
| `targeting_necessity` | `contrast_x2__plug_in` | control | odds-x2 against the natural course, three-node law: the mean of the first-node regression before the correction, on the same fits | bias interval outside the margin | bias -0.0016 to -0.0012, margin 0.000717 | pass |
| `type_i_error` | `contrast_x2__sharp_null` | positive | odds-x2 against the natural course, three-node law: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0505, 0.0387 to 0.0645 | pass |
<!-- /generated -->

Every cell passes, and no property fit failed.

Every property fit uses five folds, saturated cell means at every slot and law `A`. The table
names any other setting. `x2` is the plan that doubles the odds of treatment at every node.
`contrast_x2` is `ate_regimen[odds x2 vs natural course]`.
`tests/studies/longitudinal_incremental_properties.py` declares every cell.

| family | cells and fits | what must hold |
| --- | --- | --- |
| `interval_calibration` | seven labels with three cells each: the correctly specified fit and two derived controls. 3,000 draws at n = 48,000 per label | SE ratio and coverage intervals inside the calibration bands, and both efficiency-ratio intervals inside their band. The shrunken-SE control falls below the band. The noise control rises above the efficiency-ratio band |
| `root_n_and_efficiency` | `contrast_x2` at n = 24,000 (the control), 64,000 and 96,000. 1,000 draws each | the shared bias, coverage and SE rules. The smallest rung is the control |
| `root_n_rate` | two rate rows of `contrast_x2`, from the ladder above | the slope interval lies inside the root-n band and excludes -1/4 |
| `type_i_error` | `contrast_x2` on law `A0`, where every tilt has one mean. 2,000 draws at n = 4,000 | the one-sided rejection bound stays under the type-I ceiling |
| `power` | `odds x1.5 vs natural course` on law `Aw`. 2,000 draws at n = 4,000 | the rejection lower bound clears the minimum power |
| `simultaneous_coverage` | the band over nine tilts on `geomspace(0.25, 4, 9)`, with nine means and eight contrasts, and its pointwise control. 2,000 draws at n = 5,000 | the band's joint coverage interval lies inside the calibration coverage band. The pointwise control's joint coverage interval lies below the nominal rate |
| `mechanism_term_necessity` | `contrast_x2` fits on their own draws of law `A`, with the full curve, and with the mechanism term dropped (the control). 3,000 draws at n = 48,000 | the full curve's SE-ratio interval lies inside the efficiency-ratio band. The control's interval lies wholly below the calibration band |
| `single_robustness` | linear outcome regressions with a correct mechanism, at n = 48,000 and 96,000, and a constant mechanism at n = 48,000 (reported). 1,000 draws each | the two positive cells pass the bias-equivalence rule. The pair tests equivalence at two sizes, not a falling bias. The constant-mechanism cell states no verdict |
| `targeting_necessity` | the `single_robustness` fit at n = 96,000 on its own draws: the one-step estimate, and the plug-in mean of the first-node regression (the control). 1,000 draws | the one-step bias lies inside the margin. The plug-in bias lies outside it, and the plug-in mean sits at least 0.25 one-step standard deviations from the one-step mean |

The calibration labels:

| label | law, plan and estimand | efficiency bound |
| --- | --- | --- |
| `mean_x2` | `A`, `ey_regimen[odds x2]` | exact influence-curve standard deviation |
| `contrast_x2` | `A`, `contrast_x2` | the same |
| `contrast_x2_in_sample` | `A`, `contrast_x2`, one fold (Kennedy 2019, Algorithm 2) | the same |
| `known_x2` | `A`, `ey_regimen[odds x2]`, with the law's own $\pi_t$ declared, so the curve has no mechanism term | the known-$\pi$ standard deviation |
| `no_positivity` | `Ax`, exact 0 and 1 propensities and 0.02 cells, `contrast_x2` over two nodes | exact influence-curve standard deviation |
| `per_node` | `A`, the multipliers (2, 0.5, 1.5) at the three nodes | the same |
| `tilt_then_static` | `A`, tilts by 2 at nodes 1 and 2, then arm 1 at node 3 | the same |

The `single_robustness` cells witness that the estimator is not doubly robust. With a correct
mechanism, a linear outcome regression leaves a plug-in that misses the truth, and the weighted
residual terms correct it. With a constant mechanism, the estimate carries a bias that the
reported cell prints beside its exact-law limit,
`constant_mechanism_limit` in `tests/studies/longitudinal_incremental_common.py`.

## Sizing

The cross-fitted contrasts carry a second-order bias of order 1/n, so their standardized bias
falls as $1/\sqrt n$. The bias rule passes a row only when the 99% interval of its mean error lies
inside 0.25 empirical standard deviations. The sizing probe in `tests/diagnostics/x19_sizing/`
measured the bias at n = 8,000, and each size below projects it.

| sizing input | value | source |
| --- | --- | --- |
| standardized bias at n = 8,000, five-fold contrasts | 0.27 | `tests/studies/canonical_longitudinal_incremental.py` |
| standardized bias at n = 8,000, two-fold contrasts | 0.35 | the same |
| primary n | 96,000. The two-fold rows need about 90,000 | the same, and `tests/unit/test_longitudinal_incremental_design.py` |
| pass probability at the projected bias | at least 0.95 for a primary row and a positive root-n rung. At least 0.90 for a calibration cell | `tests/unit/test_longitudinal_incremental_design.py` |
| calibration replications | 3,000. At 2,000 an ideal cell passes with probability 0.935, and the seven labels pass jointly with 0.59. At 3,000 each passes with at least 0.99 and the seven jointly with 0.96 | `tests/studies/longitudinal_incremental_properties.py` |
| calibration n | 48,000. The per-cell rule allows 24,000. The larger size cuts the projected bias by a factor of 1.4 | the same |
| robustness sizes | 48,000 and 96,000. The probe reads biases of 0.20 at n = 6,000 and 0.11 at 24,000, so 24,000 passes with a probability below 0.95 | the same |
| power cell | 2,000 draws. Its rejection rate reads 0.855 at n = 4,000 against an exact-law power of 0.882 | the same |
| band control | the pointwise intervals cover jointly with exact probability 0.811. The control clears the control-power rule at that value plus 0.005 | `tests/unit/test_longitudinal_incremental_design.py` |
| mechanism-term control | the exact SD ratio is 0.78, below the calibration band | `tests/unit/test_longitudinal_incremental_design.py` |

The fallback-read bias bound holds far below these sizes. The study fixed every size, replication
count, margin, law, learner, fold count and seed before any run. After the run, two families
moved to their own sample streams, because they reused the fits of another family. The families
are `mechanism_term_necessity` and `targeting_necessity`. Both were refitted at the declared size
and spliced into the committed rows. `targeted-refit.json` in the study directory records the
baseline, the selection, the replacement fits and the identical carried rows. Both families
passed before and after the refit.

## Red-cell route

| event | route, declared before any run |
| --- | --- |
| a red primary truth row | blocks the merge and gets a diagnosis |
| a red paired row, once the comparator registers | gets a diagnosis |
| a red property cell that reads as a finite-sample limit | is re-registered `reporting` with an owner row before any re-run |
| a replication that raises | is never redrawn. The failing cell drops to its owner with its failure count stated as a page limit, and the run repeats without it |
| any verdict | moves no budget, margin, law, learner, fold count or seed. Controls carry the verdict their family states |

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
exact standard errors of the efficient influence curve at the primary n. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 96000 | observations per primary replication |
| `independent_tests_passed` | 10 | truth tests passing |
| `independent_tests_total` | 10 | truth tests reported |
| `property_cells_passed` | 36 | property cells passing |
| `property_cells_total` | 36 | property cells reported |
| `max_standardized_bias` | 0.1311 | largest primary standardized bias |
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
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it until the comparator runs register |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it until the comparator runs register |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it until the comparator runs register |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it until the comparator runs register |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `bound:ate_regimen[odds x0.5 vs natural course]_standard_error` | 0.0029 | exact influence-curve standard error of this estimand at the primary n = 96,000 |
| `bound:ate_regimen[odds x2 vs natural course]_standard_error` | 0.0030 | exact influence-curve standard error of this estimand at the primary n = 96,000 |
| `bound:ey_regimen[natural course]_standard_error` | 0.0077 | exact influence-curve standard error of this estimand at the primary n = 96,000 |
| `bound:ey_regimen[odds x0.5]_standard_error` | 0.0075 | exact influence-curve standard error of this estimand at the primary n = 96,000 |
| `bound:ey_regimen[odds x2]_standard_error` | 0.0076 | exact influence-curve standard error of this estimand at the primary n = 96,000 |

## Limits

| limit | what it means for use |
| --- | --- |
| one law family with three nodes | the evidence covers law `A` and its variants `A0`, `Aw` and `Ax`, with plans fixed before the run |
| saturated cell means at every slot | the study does not test a flexible learner. The exact pairing and the sizing rest on cell means |
| the `npcausal` comparison is pending | the page publishes no paired verdict. Only the pairing gates G1 and G2 compare the two sides |
| two and five folds, and one fold in one calibration label | the default of ten folds is not tested |
| finite-sample bias is sized, not removed | at n = 8,000 the standardized bias reads 0.27 and 0.35. The primary rows pass at the sizes above, and smaller sizes are not claimed |
| the estimator is not doubly robust | the positive robustness cells hold the mechanism correct. The constant-mechanism cell is reported only |
| no dropout, survival outcome, competing risk, weight, cluster or working model | the [dropout study](longitudinal-incremental-interventions-dropout.md) measures them |
| the band covers nine declared tilts | the study does not test a band over a continuum of $\delta$ |

## Reproduction

`tests/canonical/npcausal_longitudinal_incremental/regenerate.py` regenerates the study. It runs no
reference process while the comparator is pending. The pairing gates need Docker and the
`cleverly-npcausal:0.1.0-56a5ac1` image. Run them with
`python -m tests.canonical.npcausal_longitudinal_incremental.gates`.
