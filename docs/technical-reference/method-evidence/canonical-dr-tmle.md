# Canonical DR-TMLE

This reporting study evaluates the binary complete-data DR-TMLE against the primary simulation
law in Benkeser et al. (2017), Section 4, and against R
[`drtmle`](https://github.com/benkeser/drtmle) 1.1.2 at pinned commit
[`538a3a2`](https://github.com/benkeser/drtmle/tree/538a3a264c1ca984b6d88978ca7f96165f43152c).
The truth and acceptance rules do not come from the R implementation: treatment-specific means
are computed by independent quadrature, and the robustness regimes and corrected influence curve
come from the paper's theorem and simulation design.

This row uses a **reporting** publication policy. Scientific failures remain committed and render
red; incomplete replications, invalid schemas, non-finite fits, active bounds, and broken
provenance still abort publication. Thus the tables can show whether Cleverly established
equivalence, coverage superiority, or an inconclusive result without selecting only favorable runs.

## What was compared

| setting | `cleverly` | R `drtmle` |
| --- | --- | --- |
| data-generating law | binary complete-data law from Benkeser et al. (2017), Section 4 | identical realized rows |
| nuisance regimes | outcome correct, treatment correct, and both correct | identical nuisance predictions supplied as `Qn` and `gn` |
| cross-fitting | one deterministic unstratified ten-fold assignment, drawn by `random_partition` from the sample seed | the identical fold vector through the package's documented `cvFolds` route |
| construction | pooled cross-fitted, univariate reductions, guards `Q` and `g`, R-package update order | native `drtmle()` with the corresponding univariate guards and two Q steps |
| estimands | `ey0`, `ey1`, and `ate` | the same three quantities, with ATE covariance formed from the native joint covariance |
| intervals | pointwise 95% Wald | pointwise 95% Wald |
| alternation rounds | `max_outer = 100` | `maxIter = 100` |
| fit audit | `score_check`'s own bar, `1e-3 x se / sqrt(n)` | the same bar, from the reference's own reported standard errors |

The primary paired route is deliberately difficult. “Superior” requires a positive 99% lower
confidence bound for Cleverly minus R coverage, plus truth validity and non-inferiority in RMSE
and SE calibration. If that route fails, ordinary bounded similarity and non-inferiority can
still establish “equivalent.”

Two rows of that table used to say something else, and both mattered.

The audit bar was `1e-4` on the Cleverly side and `1 / n` on the R side. Each was defended
against R `drtmle`'s default, which is the wrong reference for a claim about Cleverly: the
library ships `1e-3 x se / sqrt(n)`, about `3.4e-7` here. The rule is now imported from
`score_check` rather than restated, so the study cannot drift from the library.

The alternation row said `max_iter: 100`, matching the R runner. That was the cap on the Newton
steps inside one fluctuation. The alternation itself ran at a hard-coded 50 that no caller could
reach. `max_outer` is now a keyword and the value that applied is recorded on the fit.

## What the solver diagnostics do and do not compare

R `drtmle` exposes no convergence flag. The runner had no honest value to write for one, and
wrote `TRUE`. The published study then read “24 Cleverly solver failures against 0” off a column
the reference could not fail.

`solver_reported` now marks which side reports a flag at all, and the reference's `solver_passed`
is left empty. What **is** comparable is the score audit, and at the shared bar it runs the other
way. Each side audits `score_audited_fits` fits of the same replications. Cleverly fails
`subject_score_failures` of them, and R `drtmle` fails `reference_score_failures`. Cleverly's
median score is `2e-11` to `5e-11` against the reference's `7e-9` to `9e-9`.

Every Cleverly failure sits in one regime, the one where the treatment mechanism is misspecified.
The reference fails in all three. The measured table below resolves the three counts against
`tests/canonical/drtmle/fit-diagnostics.csv`.

## The two implementations reach different roots when the outcome regression is wrong

The two sides receive identical rows and identical initial nuisance predictions, agreeing to
`6e-16`. Every difference below is targeting.

| regime | per-replication disagreement in `ate`, as a share of one sampling standard deviation |
| --- | ---: |
| both nuisances correct | 4.6% |
| outcome regression correct | 11.9% |
| treatment mechanism correct | 51.7% |

The last figure is not a convergence failure on either side. It correlates with neither
implementation's score quality, at Spearman `0.003` and `-0.055`. **Cleverly's own two documented
update orders disagree by about the same amount in the same cell**, with both routes solving all
three equations to about `1e-11`. The three equations do not pin down one answer on this law once
the outcome regression is misspecified, and the route decides which answer is reached.
[Targeting and cross-fitting](../dr-tmle/targeting.md#the-update-order) carries that measurement.

This is also what limits the paired calibration test in that regime. The published resolution of
the calibration statistic tracks the disagreement exactly: `0.0032` to `0.0039` where both
nuisances are correct, `0.0094` to `0.0184` where the outcome regression is, and `0.0370` to
`0.0468` where only the mechanism is. Every one of the nine now sits inside the declared `0.05`
margin, so all nine cells conclude `equivalent`. The same table on the treatment-stratified split
this row used before put one of them at `0.0521`, and that cell reported `underpowered` instead.
The margin, the replication count and the statistic did not move. Only the fold draw did, and the
cell sits close enough to the margin that the reader should treat it as resolved once rather than
as an established property.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper binary law, both nuisances correct | `ate` | average treatment effect | `cleverly` | -0.000477 to 0.0030 | 0.9437 | 0.9917 | pass |
| paper binary law, both nuisances correct | `ate` | average treatment effect | R `drtmle` | -0.000472 to 0.0029 | 0.9450 | 0.9934 | pass |
| paper binary law, both nuisances correct | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.0025 to -0.000075 | 0.9575 | 1.0351 | pass |
| paper binary law, both nuisances correct | `ey0` | counterfactual mean under no treatment | R `drtmle` | -0.0025 to -0.000094 | 0.9550 | 1.0359 | pass |
| paper binary law, both nuisances correct | `ey1` | counterfactual mean under treatment | `cleverly` | -0.0014 to 0.0013 | 0.9425 | 0.9765 | pass |
| paper binary law, both nuisances correct | `ey1` | counterfactual mean under treatment | R `drtmle` | -0.0014 to 0.0012 | 0.9450 | 0.9769 | pass |
| paper binary law, outcome regression correct | `ate` | average treatment effect | `cleverly` | 0.000105 to 0.0035 | 0.9413 | 0.9987 | pass |
| paper binary law, outcome regression correct | `ate` | average treatment effect | R `drtmle` | 0.000204 to 0.0036 | 0.9500 | 1.0046 | pass |
| paper binary law, outcome regression correct | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.0015 to 0.000923 | 0.9550 | 1.0235 | pass |
| paper binary law, outcome regression correct | `ey0` | counterfactual mean under no treatment | R `drtmle` | -0.0016 to 0.000864 | 0.9563 | 1.0258 | pass |
| paper binary law, outcome regression correct | `ey1` | counterfactual mean under treatment | `cleverly` | 0.000203 to 0.0028 | 0.9350 | 0.9841 | pass |
| paper binary law, outcome regression correct | `ey1` | counterfactual mean under treatment | R `drtmle` | 0.000239 to 0.0028 | 0.9337 | 0.9853 | pass |
| paper binary law, treatment mechanism correct | `ate` | average treatment effect | `cleverly` | 0.0013 to 0.0049 | 0.9450 | 0.9922 | pass |
| paper binary law, treatment mechanism correct | `ate` | average treatment effect | R `drtmle` | 0.000277 to 0.0039 | 0.9400 | 0.9878 | pass |
| paper binary law, treatment mechanism correct | `ey0` | counterfactual mean under no treatment | `cleverly` | -0.0035 to -0.000789 | 0.9450 | 0.9848 | pass |
| paper binary law, treatment mechanism correct | `ey0` | counterfactual mean under no treatment | R `drtmle` | -0.0031 to -0.000366 | 0.9387 | 0.9744 | pass |
| paper binary law, treatment mechanism correct | `ey1` | counterfactual mean under treatment | `cleverly` | -0.000420 to 0.0023 | 0.9275 | 0.9746 | pass |
| paper binary law, treatment mechanism correct | `ey1` | counterfactual mean under treatment | R `drtmle` | -0.0010 to 0.0018 | 0.9350 | 0.9656 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper binary law, both nuisances correct | `ate` | average treatment effect | -0.000002 | 0.000593 | 1.0058 | -0.0012 | 0.0032 vs 0.0500 | equivalent |
| paper binary law, both nuisances correct | `ey0` | counterfactual mean under no treatment | 0.000018 | 0.0091 | 1.0041 | 0.0025 | 0.0039 vs 0.0500 | equivalent |
| paper binary law, both nuisances correct | `ey1` | counterfactual mean under treatment | 0.000016 | 0.0077 | 1.0050 | -0.0025 | 0.0034 vs 0.0500 | equivalent |
| paper binary law, outcome regression correct | `ate` | average treatment effect | -0.000092 | 0.0329 | 1.0145 | -0.0087 | 0.0184 vs 0.0500 | equivalent |
| paper binary law, outcome regression correct | `ey0` | counterfactual mean under no treatment | 0.000059 | 0.0296 | 1.0086 | -0.0013 | 0.0135 vs 0.0500 | equivalent |
| paper binary law, outcome regression correct | `ey1` | counterfactual mean under treatment | -0.000032 | 0.0151 | 1.0126 | 0.0013 | 0.0094 vs 0.0500 | equivalent |
| paper binary law, treatment mechanism correct | `ate` | average treatment effect | 0.0010 | 0.3362 | 1.0467 | 0.0050 | 0.0403 vs 0.0500 | equivalent |
| paper binary law, treatment mechanism correct | `ey0` | counterfactual mean under no treatment | -0.000415 | 0.1864 | 1.0378 | 0.0062 | 0.0468 vs 0.0500 | equivalent |
| paper binary law, treatment mechanism correct | `ey1` | counterfactual mean under treatment | 0.000590 | 0.2587 | 1.0326 | -0.0075 | 0.0370 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robust_contraction` | `both_wrong_n1500` | control | both nuisances are misspecified, at n = 1,500 | the exact coverage interval must fall below the floor, over the replications this rung declares its own verdict at | coverage 0.0224 to 0.0584, bias 0.2306 | pass |
| `double_robust_contraction` | `both_wrong_n3000` | control | both nuisances are misspecified, at n = 3,000 | the exact coverage interval must fall below the floor, over the replications this rung declares its own verdict at | coverage 0.0148 to 0.0461, bias 0.2321 | pass |
| `double_robust_contraction` | `both_wrong_n6000` | control | both nuisances are misspecified, at n = 6,000 | the exact coverage interval must fall below the floor, over the replications this rung declares its own verdict at | coverage 0.0062 to 0.0300, bias 0.2323 | pass |
| `double_robust_contraction` | `outcome_correct_n1500` | positive | only the outcome regression is correctly specified, at n = 1,500 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9208 to 0.9637, bias 0.0036 | pass |
| `double_robust_contraction` | `outcome_correct_n3000` | positive | only the outcome regression is correctly specified, at n = 3,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9150 to 0.9596, bias 0.0021 | pass |
| `double_robust_contraction` | `outcome_correct_n6000` | positive | only the outcome regression is correctly specified, at n = 6,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9194 to 0.9627, bias 0.0012 | pass |
| `double_robust_contraction` | `rate_both_wrong` | control | the same regression with both nuisances misspecified | slope interval must not establish contraction | slope 0.0037 to 0.0135 | pass |
| `double_robust_contraction` | `rate_outcome_correct` | positive | log absolute bias regressed on log n across three sizes, outcome regression correct, over every replication each rung ran | slope interval entirely below zero, so the bias contracts | slope -1.5345 to -0.4823 | pass |
| `double_robust_contraction` | `rate_treatment_correct` | positive | the same regression with only the treatment mechanism correct | slope interval entirely below zero, so the bias contracts | slope -1.7568 to -1.0221 | pass |
| `double_robust_contraction` | `treatment_correct_n1500` | positive | only the treatment mechanism is correctly specified, at n = 1,500 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.8864 to 0.9385, bias 0.0115 | **fail** |
| `double_robust_contraction` | `treatment_correct_n3000` | positive | only the treatment mechanism is correctly specified, at n = 3,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9179 to 0.9616, bias 0.0041 | pass |
| `double_robust_contraction` | `treatment_correct_n6000` | positive | only the treatment mechanism is correctly specified, at n = 6,000 | the exact coverage interval clears the declared floor, over the replications this rung declares its own verdict at | coverage 0.9165 to 0.9606, bias 0.0019 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000292 to 0.0045, margin 0.0066, SE ratio 1.0071 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.2280 to 0.2330, margin 0.0068, SE ratio 1.8286 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0026 to 0.0075, margin 0.0067, SE ratio 0.9888 | **fail** |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0068 to 0.0121, margin 0.0072, SE ratio 0.9671 | **fail** |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9448 to 0.9667, SE ratio 0.9857 to 1.0570 | pass |
| `root_n_and_efficiency` | `n_1500` | positive | bias, coverage and SE calibration at n = 1,500 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0018, coverage 0.9165 to 0.9606, SE ratio 0.9447 | pass |
| `root_n_and_efficiency` | `n_4500` | positive | bias, coverage and SE calibration at n = 4,500 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000391, coverage 0.9223 to 0.9647, SE ratio 0.9834 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000042, coverage 0.9282 to 0.9688, SE ratio 1.0460 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5236 to -0.4422 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5140 to -0.5070 | pass |
<!-- /generated -->

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications per primary nuisance regime |
| `n` | 3000 | observations per primary replication |
| `subject_tests_passed` | 9 | Cleverly truth tests passing |
| `subject_tests_total` | 9 | Cleverly truth tests reported |
| `paired_tests_passed` | 9 | paired cells concluding equivalent or superior |
| `paired_tests_total` | 9 | paired comparison cells reported |
| `property_cells_passed` | 19 | repeated-sampling property cells passing their own and family verdicts |
| `property_cells_total` | 22 | repeated-sampling property cells reported |
| `score_audited_fits` | 2400 | fits each side audits against the shared score bar |
| `subject_score_failures` | 7 | Cleverly fits that miss that bar |
| `reference_score_failures` | 36 | R `drtmle` fits that miss that bar |
| `properties[double_robust_contraction/rate_outcome_correct]:slope` | -0.9176 | fitted contraction slope, outcome regression correct |
| `properties[double_robust_contraction/rate_outcome_correct]:slope_ci_lower` | -1.5345 | its 99% lower endpoint |
| `properties[double_robust_contraction/rate_outcome_correct]:slope_ci_upper` | -0.4823 | its 99% upper endpoint |
| `properties[double_robust_contraction/rate_outcome_correct]:replicates` | 5600 | replications the contraction slope fits |
| `properties[double_robust_contraction/rate_treatment_correct]:slope` | -1.3194 | fitted contraction slope, treatment mechanism correct |
| `properties[double_robust_contraction/rate_treatment_correct]:slope_ci_lower` | -1.7568 | its 99% lower endpoint |
| `properties[double_robust_contraction/rate_treatment_correct]:slope_ci_upper` | -1.0221 | its 99% upper endpoint |
| `properties[double_robust_contraction/rate_both_wrong]:slope` | 0.0086 | fitted contraction slope of the control arm |
| `properties[double_robust_contraction/rate_both_wrong]:slope_ci_lower` | 0.0037 | its 99% lower endpoint |
| `properties[double_robust_contraction/rate_both_wrong]:slope_ci_upper` | 0.0135 | its 99% upper endpoint |
| `properties[double_robust_contraction/outcome_correct_n1500]:bias` | 0.0036 | bias scale at the first rung, outcome regression correct |
| `properties[double_robust_contraction/treatment_correct_n1500]:replicates` | 800 | replications the first rung reads its own coverage verdict from |
| `properties[double_robust_contraction/treatment_correct_n1500]:coverage` | 0.9150 | exact coverage of the red rung |
| `properties[double_robust_contraction/treatment_correct_n1500]:coverage_ci_lower` | 0.8864 | its 99% coverage lower endpoint |
| `min_coverage` | 0.9275 | lowest implementation-estimand coverage |
| `min_coverage_ci_lower` | 0.9006 | lowest exact 99% coverage lower endpoint |
| `max_standardized_bias` | 0.1557 | largest absolute standardized bias |
| `max_rmse_ratio_upper` | 1.0467 | largest paired 99% RMSE-ratio upper endpoint |
| `min_coverage_difference_lower` | -0.0250 | smallest paired 99% coverage-difference lower endpoint |
| `max_calibration_excess_upper` | 0.0364 | largest paired 99% excess-calibration upper endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples per bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor for the exact coverage lower endpoint |
| `margin:over_coverage_ceiling` | 0.9900 | coverage above this is labeled conservative |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | one-sided type-I error ceiling |
| `margin:paired_difference` | 0.1500 | paired mean-difference margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | paired RMSE-ratio upper limit |
| `margin:coverage_noninferiority` | -0.0250 | paired coverage-difference lower limit |
| `margin:calibration_noninferiority` | 0.0500 | paired calibration-excess upper limit |
| `margin:minimum_power` | 0.8000 | power-control rejection lower bound |
| `margin:root_n_slope` | -0.5000 | predicted root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |

## What a red double-robustness cell means here

The two one-correct `double_robustness` cells exceed the equivalence margin at `n = 1,500`. That
margin is a quarter of an empirical standard deviation, and the test is resolvable at that size:
its 99% half-width is about `0.0025` against margins of `0.0067` and `0.0072`. The bias really is
larger than the margin. The [red-cell ledger](red-cells.md) lists each red cell of this study
with the roadmap ask that owns it.

RM18 read the two red one-correct cells against R `drtmle` on the primary rows at `n = 3,000`,
under a rule declared before the computation. On `outcome_correct`, R `drtmle` shows the same
bias, and the paired interval covers zero, so the reading is `shared`. On `treatment_correct`,
R `drtmle` also shows a bias above zero, and the paired interval of `cleverly` minus R runs
`0.000068` to `0.001942`. Both intervals exclude zero on the side of the bias, so the reading is
`mixed`. That interval carries no
multiplicity adjustment, and a Bonferroni interval over the three paired intervals covers zero.

[RM19](../../roadmap.md#rm19-one-sided-robustness-bias-increment-in-dr-tmle) carries the
increment. {ref}`What the one-sided reading found <what-the-one-sided-reading-found>` gives each
interval.

One size cannot say what kind of failure that is. A second-order remainder that has not yet
decayed and an inconsistent estimator look identical at one `n` and mean opposite things. The
`double_robust_contraction` family fits log absolute bias on log `n` over three sizes to separate
their predicted behavior. A second-order remainder predicts a slope near `-1`, and a first-order
term predicts one near `-1/2`. An inconsistent estimator can produce a slope near `0`. The
`both_wrong` arm is the control that must fail to contract.

Three finite-sample points do not identify which term produced an observed slope.

The measured slopes are `-0.9176` with the outcome regression correct and `-1.3194` with the
mechanism correct. The control sits at `0.0086`, above zero, and its standardized bias *grows*
with `n`, from `8.5` to `12.6` to `16.6`. Both positive biases contract over this ladder, while
the control does not. This result distinguishes contraction from non-contraction. It does not
identify either bias as the theorem's second-order remainder.

Both slope intervals establish contraction. With the mechanism correct the interval runs `-1.7568` to
`-1.0221`. With the outcome regression correct it runs `-1.5345` to `-0.4823`. The second interval
covered zero on the earlier ladder, which ran 800 replications at every rung. The reason was
arithmetic rather than scientific. The bias in that arm is already small at the first rung,
`0.0036`, so three points of a small and noisy quantity gave a wide slope.

`CONTRACTION_REPLICATES` in `tests/studies/drtmle_properties.py` declares the ladder that resolves
the slope. The sizes stay at 1,500, 3,000 and 6,000. The two outer rungs run 2,400 replications
and the middle rung keeps 800. A middle rung carries a centred weight of zero in the fitted slope,
so replications there buy no resolution. The budget rule reads the published bias scale and the
control arm's spread. It reads neither positive rate cell, so the design cannot tune itself to a
verdict.

The declaration predicted the narrowing, and the control arm checks it. A slope interval's width
falls as one over the square root of the replications. Tripling the two outer rungs therefore
narrows the control interval by about the square root of three. The committed control interval
runs `0.0037` to `0.0135`, and that is the predicted narrowing. The git history of
`tests/canonical/drtmle/properties.csv` carries the interval it replaced.

The run that raised the rungs also moved the environment and the package source. SciPy moved from 1.17.1 to 1.18.0, and
Python moved from 3.11.13 to 3.13.7, on that run. `tests/canonical/drtmle/manifest.json` records
the versions the committed rows came from. Its git history records the versions they replaced.

The committed history separates the budget from the source and environment changes together. It
does not separate the source from the environment. The first 800 replications of each rung ran at
both commits, and their estimates agree within 7.3e-5. At that budget the slope rule gives each
interval within 2e-5 at both commits, and
`rate_outcome_correct` still runs `-3.0401` to `+0.1232`. The move below zero therefore comes
from the declared budget. RM18 records the comparison under
{ref}`What the committed history already separates <what-the-committed-history-already-separates>`.

The fitted `-0.9176` sits near the `-1` that a second-order remainder predicts. The budget rule
never read this cell, so the design did not steer that agreement. The interval still covers `-1`
and `-1/2` together. It therefore corroborates the sign of the contraction and not its exponent,
and the Limitations section below says the same.

A reader cannot recompute that slope from the rung rows beside it. The fit reads every replication
each rung ran, which is 2,400 at each outer rung. The `bias` column beside a rung instead reports
the `800` replications that rung reads its coverage verdict from. The two samples differ. The log
ratio of the outer rungs' published `bias` values, divided by the log of four, therefore gives a
slope near `-0.8` and not `-0.9176`. The same ratio over every replication gives the published
slope, and `property-replicates.csv.gz` carries those replications.

The extra draws serve the slope alone. Each rung reads its own coverage verdict from `800`
replications, which `CONTRACTION_VERDICT_REPLICATES` declares beside the budget. A rung's coverage
gate is a one-sided exact interval against a fixed floor. A larger budget narrows that interval
around the true coverage and can resolve an inconclusive verdict. The study refuses the extra
coverage budget because nobody declared it before the run.

The slope budget was declared before the run under the rule above. Each rate cell therefore fits
`5600` replications, while each rung keeps its declared coverage budget. The `both_wrong` control
still has to fail to contract.

Raising the level margin was considered and rejected. The standardized bias under a correct
mechanism runs `0.398`, `0.203` and `0.129` across the ladder, so no affordable size brings the
99% interval inside `0.25`. The level cell stays red and the contraction family says which red it
is.

The ladder also surfaced a result the single-size study could not reach. At `n = 1,500` with the
outcome regression misspecified, the interval does not clear the declared coverage floor: exact
coverage is `0.9150` with a 99% lower endpoint of `0.8864` against a floor of `0.90`. It clears at
`n = 3,000` and at `n = 6,000`. That cell is red, and it is a statement about small samples in one
regime rather than about the construction. Its budget did not move, so the cell stays red under
the ladder that resolved the slope.

## Limitations

The comparison is binary, two-arm, complete-outcome, pointwise, and confined to the paper law and
declared GLMs. It does not establish missing-outcome, multi-arm, weighted, clustered,
simultaneous-inference, broad machine-learning, or practical-positivity parity. Supplying the
same initial nuisance predictions isolates the corrected construction; it does not compare the
two projects' learner wrappers. The study measures finite-sample behavior at its declared sizes
and cannot verify the theorem's unobservable second-order remainder condition for a future fit.

The contraction family fits three points, so it establishes a direction rather than an exponent.
Both positive slope intervals now exclude zero. The outcome-correct interval still covers `-1` and
`-1/2` together, so it does not identify that remainder's order.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/equivalence.csv),
[fit diagnostics](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/fit-diagnostics.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle/properties.csv)
carry the protocol, provenance, and every published row.
