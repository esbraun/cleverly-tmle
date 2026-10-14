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
<!-- /generated -->

TODO(results): after the run, add one sentence on any red cell and the roadmap row that owns it.
Add the count of failed fits per cell, if any.

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
| `mechanism_term_necessity` | the `contrast_x2` calibration fits with the full curve, and with the mechanism term dropped (the control). 3,000 draws at n = 48,000 | the full curve's SE-ratio interval lies inside the efficiency-ratio band. The control's interval lies wholly below the calibration band |
| `single_robustness` | linear outcome regressions with a correct mechanism, at n = 48,000 and 96,000, and a constant mechanism at n = 48,000 (reported). 1,000 draws each | the two positive cells pass the bias-equivalence rule. The constant-mechanism cell states no verdict |
| `targeting_necessity` | the n = 96,000 draws of `single_robustness`: the one-step estimate, and the plug-in mean of the first-node regression (the control) | the one-step bias lies inside the margin. The plug-in bias lies outside it, and the plug-in mean sits at least 0.25 one-step standard deviations from the one-step mean |

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

| quantity | value | source |
| --- | --- | --- |
| standardized bias at n = 8,000, five-fold contrasts | 0.27 | `tests/studies/canonical_longitudinal_incremental.py` |
| standardized bias at n = 8,000, two-fold contrasts | 0.35 | the same |
| primary n | 96,000. The two-fold rows need about 90,000 | the same, and `tests/unit/test_longitudinal_incremental_design.py` |
| pass probability at the projected bias | at least 0.95 for a primary row and a positive root-n rung. At least 0.90 for a calibration cell | `tests/unit/test_longitudinal_incremental_design.py` |
| calibration replications | 3,000. At 2,000 an ideal cell passes with probability 0.935, and the seven labels pass jointly with 0.59. At 3,000 each passes with at least 0.99 and the seven jointly with 0.96 | `tests/studies/longitudinal_incremental_properties.py` |
| calibration n | 48,000. The per-cell rule allows 24,000. The larger size cuts the projected bias by a factor of 1.4 | the same |
| robustness sizes | 48,000 and 96,000. The probe reads biases of 0.20 at n = 6,000 and 0.11 at 24,000, so 24,000 fails | the same |
| power cell | 2,000 draws. Its rejection rate reads 0.855 at n = 4,000 against an exact-law power of 0.882 | the same |
| band control | the pointwise intervals cover jointly with exact probability 0.811. The control clears the control-power rule at that value plus 0.005 | `tests/unit/test_longitudinal_incremental_design.py` |
| mechanism-term control | the exact SD ratio is 0.78, below the calibration band | `tests/unit/test_longitudinal_incremental_design.py` |

The fallback-read bias bound holds far below these sizes. The study fixed every size, replication
count, margin, law, learner, fold count and seed before any run.

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
| `independent_tests_passed` | TODO(results) | truth tests passing |
| `independent_tests_total` | TODO(results) | truth tests reported |
| `property_cells_passed` | TODO(results) | property cells passing |
| `property_cells_total` | TODO(results) | property cells reported |
| `max_standardized_bias` | TODO(results) | largest primary standardized bias |
| `min_coverage` | TODO(results) | lowest primary coverage |
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
