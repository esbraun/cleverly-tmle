# Full-refit bootstrap and derived contrasts

This study measures four post-fit outputs against known truth. The outputs are the log-scale ratio
of two regimen levels (`LongitudinalResult.ratio`), the RMST and its contrast
(`LongitudinalResult.rmst`), the percentile interval of the `LTMLE` full-refit bootstrap, and the
percentile interval of the point-treatment `TMLE` bootstrap.
**No canonical implementation is compared.** No pinned comparator ships these outputs, so the empty
equivalence artifact records the absence of a comparator.

The ratio and RMST are exact delta-method and linear functionals of shipped influence curves.
`tests/unit/test_contrast_conveniences.py` checks their arithmetic on exact laws. Their
log-scale small-sample coverage is a simulation claim, so this study measures it. The bootstrap
has no source for this estimator. The study measures its percentile interval with correctly
specified nuisances at $n = 1000$. No result covers a data-adaptive nuisance, and Cai and van der
Laan (2020) is the warning for that case.

## What was tested

| setting | declaration |
| --- | --- |
| primary scenario | the log risk ratio and log odds ratio of always versus never on the two-node end-of-study law, 1,000 replications at $n = 2000$ |
| laws | the two-node end-of-study law of `tests/discrete_law_longitudinal.py`, the four-node survival law of `tests/studies/survival_grid_law.py`, a clustered draw of the end-of-study law, and the point law of `tests/discrete_law.py` |
| learners | saturated cell means on the end-of-study and point laws. On the four-node law, cell means over the columns each true conditional reads. Every learner is correctly specified |
| bootstrap | 200 full-refit replicates per study replicate. Each cell reads the shipped `estimate.bootstrap.ci` and `estimate.bootstrap.std_error`, and the RMST cell reads `result.rmst(...).bootstrap`. Each fit runs with `n_jobs=1` |
| clustered draw | 60 clusters of 25 rows. A cluster's latent sign moves the outcome probability by $\pm 0.1 (A_1 + A_2 - 1)$, which leaves every regimen mean unchanged |
| replications | 4,000 per property cell, chosen before the run so that a cell whose reported standard error is 2% short passes the calibration rule with probability of at least 0.95 (`calibration_pass_probability`). At 1,000 that probability is about 0.3 |
| verdicts | every cell is an `interval_calibration` cell under the shared margins. A positive cell needs its SE-ratio and coverage intervals inside the calibration bands. A shrunken control multiplies the standard error, or the percentile interval about its median, by 0.70 and must fall below the band |
| failed-replicate cap | a bootstrap cell whose mean failed share of bootstrap replicates exceeds 1% reports `bootstrap_conditional` and claims no coverage |

The cells and their sizes are the table.

| cells | law | folds | n | replications |
| --- | --- | ---: | ---: | ---: |
| log risk and odds ratio, end of study | end of study | 1 | 2,000 | 4,000 |
| log risk ratio and log survival ratio at t = 4, RMST and RMST contrast up to t = 5 | four-node survival | 1 | 2,000 | 4,000 |
| RMST and RMST contrast up to t = 5 | four-node survival | 5 | 2,000 | 4,000 |
| bootstrap of the always mean and the contrast | end of study | 1 and 5 | 1,000 | 4,000 each |
| bootstrap of the risks at t = 2 and 4 and the RMST | four-node survival | 1 | 1,000 | 4,000 |
| cluster bootstrap of the always mean and the contrast | clustered | 1 | 1,500 | 4,000 |
| bootstrap of the point ATE | point | 1 | 1,000 | 4,000 |

The end-of-study ratios appear twice. The primary table measures bias and coverage against the
shared floor at 1,000 replications. The property cells measure calibration at 4,000. The grid row
cites the property cells for the calibration claim.

## Red-cell policy

The policy was declared before the run, in `RED_CELL_POLICY` of
`tests/studies/canonical_full_refit_bootstrap.py`.

| rule | when it applies | what happens |
| --- | --- | --- |
| 1 | always | the study publishes under `reporting`. No budget, margin, law or learner changes after a verdict is seen |
| 2 | a red ratio or RMST cell | the exact-law tests are checked first. A defect is fixed and the study is regenerated once. A red cell that the exact tests cannot explain is published red with owner `X20-derived` |
| 3 | a longitudinal bootstrap cell | a design kind enters `LICENSED_BOOTSTRAP_DESIGNS` only when all its bootstrap cells are green. A red cell keeps its own kind a diagnostic, with owner `X20-bootstrap`, and moves no other kind |
| 4 | a red point-TMLE bootstrap cell | the cell is published red with owner `X20-point-bootstrap`, and rule 3 applies to `TMLEResult` |
| 5 | a control that does not fail | the control is reported as underpowered by design, and its positive cell claims no verdict |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-node end-of-study law, saturated cell-mean learners | `or_regimen[always vs never]` | odds ratio, with inference on the log scale, between the plans "treat at both times" against "treat at neither time" | `cleverly` LTMLE ratio contrast | -0.0176 to 0.0214 | 0.9660 | 1.0171 | pass |
| two-node end-of-study law, saturated cell-mean learners | `rr_regimen[always vs never]` | risk ratio, with inference on the log scale, between the plans "treat at both times" against "treat at neither time" | `cleverly` LTMLE ratio contrast | -0.0104 to 0.0060 | 0.9640 | 1.0202 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `boot_ate_clustered__correctly_specified` | positive | cluster-bootstrap percentile interval of the contrast, 60 clusters: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9123 to 0.9342, SE ratio 0.9496 to 1.0065 | **fail** |
| `interval_calibration` | `boot_ate_crossfit__correctly_specified` | positive | bootstrap percentile interval of the contrast, five folds: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9741 to 0.9857, SE ratio 1.2019 to 1.3336 | **fail** |
| `interval_calibration` | `boot_ate_end_of_study__correctly_specified` | positive | bootstrap percentile interval of the always-never contrast: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9254 to 0.9456, SE ratio 0.9443 to 1.0028 | pass |
| `interval_calibration` | `boot_ate_end_of_study__shrunken_se_control` | control | bootstrap percentile interval of the always-never contrast: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7920 to 0.8243, SE ratio 0.6606 to 0.7022 | pass |
| `interval_calibration` | `boot_ate_point_tmle__correctly_specified` | positive | bootstrap percentile interval of the point-treatment ATE: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9273 to 0.9472, SE ratio 0.9518 to 1.0077 | pass |
| `interval_calibration` | `boot_ey_clustered__correctly_specified` | positive | cluster-bootstrap percentile interval of the always mean, 60 clusters: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9219 to 0.9426, SE ratio 0.9618 to 1.0199 | pass |
| `interval_calibration` | `boot_ey_crossfit__correctly_specified` | positive | bootstrap percentile interval of the always mean, five folds: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9730 to 0.9848, SE ratio 1.1663 to 1.3066 | **fail** |
| `interval_calibration` | `boot_ey_end_of_study__correctly_specified` | positive | bootstrap percentile interval of the always mean, end-of-study law: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9208 to 0.9416, SE ratio 0.9423 to 1.0034 | pass |
| `interval_calibration` | `boot_ey_end_of_study__shrunken_se_control` | control | bootstrap percentile interval of the always mean, end-of-study law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7938 to 0.8260, SE ratio 0.6592 to 0.7009 | pass |
| `interval_calibration` | `boot_risk2_survival__correctly_specified` | positive | bootstrap percentile interval of the always risk at t = 2: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9307 to 0.9502, SE ratio 0.9695 to 1.0289 | pass |
| `interval_calibration` | `boot_risk4_survival__correctly_specified` | positive | bootstrap percentile interval of the always risk at t = 4: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9289 to 0.9486, SE ratio 0.9601 to 1.0178 | pass |
| `interval_calibration` | `boot_rmst_survival__correctly_specified` | positive | bootstrap percentile interval of the always RMST up to t = 5: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9240 to 0.9444, SE ratio 0.9594 to 1.0192 | pass |
| `interval_calibration` | `or_end_of_study__correctly_specified` | positive | log odds ratio of always versus never, end-of-study law: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9367 to 0.9553, SE ratio 0.9477 to 1.0046 | pass |
| `interval_calibration` | `or_end_of_study__shrunken_se_control` | control | log odds ratio of always versus never, end-of-study law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8067 to 0.8380, SE ratio 0.6629 to 0.7030 | pass |
| `interval_calibration` | `rmst_contrast_crossfit__correctly_specified` | positive | RMST contrast up to t = 5, four-node law, five folds: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9628, SE ratio 1.0070 to 1.0655 | pass |
| `interval_calibration` | `rmst_contrast_survival__correctly_specified` | positive | RMST contrast of always versus never up to t = 5, four-node law: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9369 to 0.9555, SE ratio 0.9767 to 1.0349 | pass |
| `interval_calibration` | `rmst_crossfit__correctly_specified` | positive | RMST of always up to t = 5, four-node law, five folds: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9434 to 0.9610, SE ratio 1.0084 to 1.0685 | pass |
| `interval_calibration` | `rmst_survival__correctly_specified` | positive | RMST of always up to t = 5, four-node survival law: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9394 to 0.9575, SE ratio 0.9704 to 1.0256 | pass |
| `interval_calibration` | `rr_end_of_study__correctly_specified` | positive | log risk ratio of always versus never, end-of-study law: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9391 to 0.9573, SE ratio 0.9536 to 1.0105 | pass |
| `interval_calibration` | `rr_end_of_study__shrunken_se_control` | control | log risk ratio of always versus never, end-of-study law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8059 to 0.8373, SE ratio 0.6673 to 0.7080 | pass |
| `interval_calibration` | `rr_survival__correctly_specified` | positive | log risk ratio of always versus never at t = 4, four-node survival law: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9364 to 0.9550, SE ratio 0.9659 to 1.0246 | pass |
| `interval_calibration` | `rr_survival__shrunken_se_control` | control | log risk ratio of always versus never at t = 4, four-node survival law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8188 to 0.8494, SE ratio 0.6760 to 0.7166 | pass |
| `interval_calibration` | `survival_rr_survival__correctly_specified` | positive | log ratio of the survival probabilities of always versus never at t = 4, four-node law: all four required nuisance functions are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9372 to 0.9557, SE ratio 0.9637 to 1.0236 | pass |
<!-- /generated -->

## Result

Every ratio and RMST cell passed, and every shrunken control fell below the band. The
in-sample end-of-study, in-sample survival and point-treatment bootstrap cells passed. So
`end_of_study/in_sample` and `survival/in_sample` enter `LICENSED_BOOTSTRAP_DESIGNS`, and
`POINT_BOOTSTRAP_INFERENTIAL` stays `True`.

Three cells are red, and rule 3 applies to each. The two cross-fitted end-of-study cells
over-cover, with SE ratios near 1.25. Two copies of one unit can fall in different folds of a
replicate, which the bootstrap contract states, and the replicates then spread wider than the
estimator. The cluster contrast covers 0.924, with a 99% interval below the band, while the
cluster mean passes. So `end_of_study/cross_fit` and `end_of_study/cluster` stay diagnostic,
and the roadmap owner `X20-bootstrap` holds the three cells.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | primary ratio tests against truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 23 | repeated-sampling property cells |
| `property_cells_passed` | 20 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0219 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9640 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9460 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.9609 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0809 | highest bootstrap primary SE-ratio endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. A control must rise above it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `properties[interval_calibration/boot_ey_crossfit__correctly_specified]:se_ratio` | 1.2310 | SE ratio, cross-fitted bootstrap of the always mean |
| `properties[interval_calibration/boot_ate_crossfit__correctly_specified]:se_ratio` | 1.2652 | SE ratio, cross-fitted bootstrap of the contrast |
| `properties[interval_calibration/boot_ate_crossfit__correctly_specified]:coverage` | 0.9805 | coverage, cross-fitted bootstrap of the contrast |
| `properties[interval_calibration/boot_ate_clustered__correctly_specified]:coverage` | 0.9237 | coverage, cluster bootstrap of the contrast |
| `properties[interval_calibration/boot_ate_clustered__correctly_specified]:coverage_ci_lower` | 0.9123 | its 99% lower endpoint |
| `properties[interval_calibration/boot_risk4_survival__correctly_specified]:bootstrap_failed_fraction` | 0.0025 | mean failed share of bootstrap replicates, survival |
| `properties[interval_calibration/boot_ate_point_tmle__correctly_specified]:coverage` | 0.9377 | coverage, point-treatment bootstrap of the ATE |

## Limits

| limit | what it means |
| --- | --- |
| correctly specified nuisances only | the bootstrap interval is measured with known-form learners. No result covers a data-adaptive nuisance |
| one size per cell | each bootstrap cell runs at one $n$. A licensed kind is licensed by design, and the study does not measure it at other sizes |
| unmeasured compositions | a working model, competing risks, weights, a dynamic rule, a Gaussian outcome and a cross-fitted survival fit have no licensed kind, so their bootstrap prints as a diagnostic |
| finite laws | every law has binary nodes and finite support |
| static regimens | the study fits never and always. A dynamic rule is not measured by these cells |
| node units | RMST is measured in node units on an equal grid |

## Reproduction

```powershell
uv run --extra dev python -m tests.canonical.full_refit_bootstrap.regenerate
```
