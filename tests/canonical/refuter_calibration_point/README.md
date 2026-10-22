# Point-treatment refutation verdict rules

This directory holds a repeated-sampling study of three decision rules that `cleverly.refute`
prints beside its statistics on a point-treatment fit. The X82 plan (rev 5, sections 8.1 to
8.6) declares the design.

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 placebo refits against the exact null, `t` with 19 degrees of freedom | two-sided p below 0.05 |
| subset `rho` | 20 subset refits against the sampling reference of the fit's SE | two-sided `chi^2_19` p below 0.05 |
| negative control `z` | the fit refitted on a declared negative-control outcome | two-sided p below 0.05 |

Each trial draws `n = 2000` rows with 20 covariates and fits one cell with five-fold
cross-fitting.

| cell | configuration | outcome | size readings |
| --- | --- | --- | --- |
| T1 | `TMLE` ATE, GLM nuisances | Gaussian | placebo, subset, negative control |
| T2 | `TMLE` ATE, GLM nuisances | binary | placebo, subset, negative control |
| T3 | `TMLE` ATE, depth-2 `HistGradientBoosting` | binary | placebo |
| C1 | `CTMLE` greedy, GLM nuisances | binary | placebo |
| D1 | `DRTMLE`, declared mechanism, GLM nuisances | binary | placebo, subset, negative control |
| D2 | `DRTMLE`, estimated mechanism, GLM nuisances | binary | placebo |

A size cell has 200 trials and passes at no more than 17 rejections. A power cell reads trials
0 to 49 of its size cell under a declared defect and passes at no fewer than 38 detections.

| defect | rule | construction |
| --- | --- | --- |
| stale rows | placebo | a declared share of rows keeps the recorded treatment |
| half SE, double SE | subset | the same trial's `rho` at the reported SE halved and doubled |
| confounded control | negative control | the control also reads the unmeasured cause of the treatment |

`tests/unit/test_refuter_calibration_point_design.py` sizes each defect from the declared law
so that its per-trial power is at least 0.95.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

Run the declared study from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.refuter_calibration_point.regenerate --output <empty scratch dir> --jobs <cores>
```

The declarations are in `tests/studies/refuter_calibration_point.py` and
`tests/studies/refuter_calibration_point_properties.py`.
