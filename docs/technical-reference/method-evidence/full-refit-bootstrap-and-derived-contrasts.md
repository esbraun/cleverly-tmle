# Full-refit bootstrap and derived contrasts

This study measures four post-fit outputs against known truth. The outputs are the log-scale
ratio of two regimen levels (`LongitudinalResult.ratio`), the RMST and its contrast
(`LongitudinalResult.rmst`), the percentile interval of the `LTMLE` full-refit bootstrap, and the
percentile interval of the point-treatment `TMLE` bootstrap. **No canonical implementation is
compared.** No pinned comparator ships these outputs, so the empty equivalence artifact records
the absence of a comparator.

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
| bootstrap | 400 full-refit replicates per study replicate, two-sided percentile interval, `n_jobs=1` inside each fit |
| clustered draw | 60 clusters of 25 rows. A cluster's latent sign moves the outcome probability by $\pm 0.1 (A_1 + A_2 - 1)$, which leaves every regimen mean unchanged |
| verdicts | every cell is an `interval_calibration` cell under the shared margins. A positive cell needs its SE-ratio and coverage intervals inside the calibration bands. A shrunken control multiplies the standard error, or the percentile interval about its median, by 0.70 and must fall below the band |
| failed-replicate cap | a bootstrap cell whose mean failed share of bootstrap replicates exceeds 1% reports `bootstrap_conditional` and claims no coverage |

The cells and their sizes are the table.

| cells | law | folds | n | replications |
| --- | --- | ---: | ---: | ---: |
| log risk and odds ratio, end of study | end of study | 1 | 2,000 | 2,400 |
| log risk ratio and log survival ratio at t = 4, RMST and RMST contrast up to t = 5 | four-node survival | 1 | 2,000 | 2,400 |
| RMST and RMST contrast up to t = 5 | four-node survival | 5 | 2,000 | 1,200 |
| bootstrap of the always mean and the contrast | end of study | 1 and 5 | 1,000 | 1,000 each |
| bootstrap of the risks at t = 2 and 4 and the RMST | four-node survival | 1 | 1,000 | 1,000 |
| cluster bootstrap of the always mean and the contrast | clustered | 1 | 1,500 | 1,000 |
| bootstrap of the point ATE | point | 1 | 1,000 | 1,000 |

## Red-cell policy

The policy was declared before the run, in `RED_CELL_POLICY` of
`tests/studies/canonical_full_refit_bootstrap.py`.

| rule | when it applies | what happens |
| --- | --- | --- |
| 1 | always | the study publishes under `reporting`. No budget, margin, law or learner changes after a verdict is seen |
| 2 | a red ratio or RMST cell | the exact-law tests are checked first. A defect is fixed and the study is regenerated once. A red cell that the exact tests cannot explain is published red with owner `X20-derived` |
| 3 | a red longitudinal bootstrap cell | `LONGITUDINAL_BOOTSTRAP_INFERENTIAL` becomes `False`, so the percentile interval ships as a diagnostic, with owner `X20-bootstrap` |
| 4 | a red point-TMLE bootstrap cell | the cell is published red with owner `X20-point-bootstrap`, and rule 3 applies to `TMLEResult` |
| 5 | a control that does not fail | the control is reported as underpowered by design, and its positive cell claims no verdict |

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

## Limits

| limit | what it means |
| --- | --- |
| correctly specified nuisances only | the bootstrap interval is measured with known-form learners. No result covers a data-adaptive nuisance |
| one size per cell | each bootstrap cell runs at one $n$. The percentile interval is not licensed at other sizes |
| finite laws | every law has binary nodes and finite support |
| static regimens | the study fits never and always. A dynamic rule is not measured by these cells |
| node units | RMST is measured in node units on an equal grid |

## Reproduction

```powershell
uv run --extra dev python -m tests.canonical.full_refit_bootstrap.regenerate
```
