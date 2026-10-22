# Longitudinal refutation verdict rules

This directory holds a repeated-sampling study of three decision rules that `cleverly.refute`
prints beside its statistics on a longitudinal fit. The X82 plan (rev 5, sections 7.2 and 8.1
to 8.6) declares the design.

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 Markov placebo refits against the exact null, `t` with 19 degrees of freedom | two-sided p below 0.05 |
| subset `rho` | 20 subset refits against the sampling reference of the fit's SE | two-sided `chi^2_19` p below 0.05 |
| negative control `z` | the fit refitted on a declared negative-control outcome | two-sided p below 0.05 |

Each trial draws `n = 1000` units with two treatment nodes and fits one cell.

| cell | configuration | law | size readings |
| --- | --- | --- | --- |
| L1 | `LTMLE` static regimens, GLM, five folds, risk contrast at `t = 2` | S | placebo, subset |
| S1 | `SDR` static regimens, GLM, five folds | E | placebo, subset, negative control |
| I1 | `IPSI`, odds `x2` against the natural course, GLM, five folds | E | placebo, subset, negative control |
| B1 | `SequentialGComputation`, GLM, in sample | E | placebo, subset, negative control |
| B2 | `SequentialIPW` (Hajek), GLM, in sample | E | placebo, subset, negative control |
| B3 | `ParametricGFormula`, GLM models, 10 draws per unit, in sample | E | placebo, subset, negative control |
| SN1 | `StructuralNestedMean`, intercept blips, GLM, five folds | E | placebo, subset, negative control |

Law E is an end-of-study law with a Gaussian outcome, a time-varying confounder `L2`, and an
unmeasured `U` that drives the first treatment and the confounded control only. Law S is the
registered censored survival law of `cleverly.datasets.make_longitudinal_survival`. The module
docstring of `tests/studies/refuter_calibration_longitudinal.py` states both laws.

On law E every mechanism GLM and every static-regimen outcome GLM is correctly specified.
IPSI's iterated regression under the odds tilt is not linear, so I1's outcome model is
misspecified. A red I1 subset cell has that pre-identified cause, and it goes to diagnosis.

SN1 (X82 follow-up SD) fits intercept-only blips, which are correctly specified on law E. Its
nuisance `E[A2 | W, A1]` is not logistic-linear. The influence-curve SE omits a term that is
nonzero when that nuisance is wrong. A red SN1 subset or control cell has that pre-identified
cause. A pre-run check on 400 draws read an SE ratio of 1.033.

L1 reads no negative control. The shipped `refute` refuses a negative-control outcome on a
survival fit, because no replacement keeps the risk sets.

A size cell has 200 trials and passes at no more than 17 rejections. A power cell reads trials
0 to 49 of its size cell under a declared defect and passes at no fewer than 38 detections.

| defect | rule | construction |
| --- | --- | --- |
| stale rows | placebo | a declared share of rows keeps the recorded treatment trajectory |
| half SE, double SE | subset | the same trial's `rho` at the reported SE halved and doubled |
| confounded control | negative control | the control also reads the unmeasured cause of the first treatment |

The sizing probe `tests/diagnostics/x82_sl_sizing/` fits each cell's own estimator on four
samples of 200,000 units of each defective law. `tests/unit/test_refuter_calibration_longitudinal_design.py`
reads it and requires each defect to clear its power floor by three Monte Carlo SDs.

Each trial records the configuration of its fit, read off the fitted result. The run stops if
any axis except the smallest event count varies within a cell. `cells.csv` records each cell's
configuration and the range of its event count.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

Run the declared study from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.refuter_calibration_longitudinal.regenerate --output <empty scratch dir> --jobs <cores>
```

The declarations are in `tests/studies/refuter_calibration_longitudinal.py` and
`tests/studies/refuter_calibration_longitudinal_properties.py`.
