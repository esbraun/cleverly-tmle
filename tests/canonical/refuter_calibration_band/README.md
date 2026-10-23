# Band-rank placebo verdict rule

This directory holds a repeated-sampling study of the placebo `t` rule that `cleverly.refute`
prints on a continuous dose whose declared support is a function. On such a dose the placebo
moves each dose's position inside its band. `X82-J-plan.md` rev 2 (section 5) declares the design.

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 band-rank placebo refits against the exact null, `t` with 19 degrees of freedom | two-sided p below 0.05 |

Each trial draws `n = 2000` rows and fits one cell with five-fold cross-fitting.

| cell | configuration | law | placebo draw |
| --- | --- | --- | --- |
| PJ1 | `TMLE`, the policy `up`, OLS outcome, classifier ratio with a quadratic logistic | P: band `[L(W), U(W)]` whose location and width read `W` | `band_permute` |
| LJ1 | `LTMLE`, `T = 3`, the regimen `up`, logistic GLM outcome and pseudo-outcome, classifier ratio with a quadratic logistic | L: the pricing shape, each band reads the engagement and the earlier price | `band_markov` |

A size cell has 200 trials and passes at no more than 20 rejections, a size below 0.10. A power cell reads trials
0 to 49 of its size cell under the stale-row defect and passes at no fewer than 38 detections.

| defect | construction |
| --- | --- |
| stale rows | a declared share of units keeps its own dose (PJ1) or its whole observed trajectory (LJ1). The other units keep their band-rank draws |

`tests/diagnostics/x82_sj_sizing/` measures, on 100 (PJ1) and 60 (LJ1) trials at the declared
`n` on seeds outside the study's streams, the band-rank placebo's bias and the stale-row shift in
placebo SD. `tests/unit/test_refuter_calibration_band_design.py` reads it. The design test also
measures the declared classifier's fit to the exact placebo ratio on PJ1.

Each trial records the configuration of its fit, read off the fitted result. The run stops if
any axis varies within a cell. `cells.csv` records each cell's configuration.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

The run costs about 1.2 CPU-hours. LJ1 takes about 1.0 and PJ1 about 0.2. The sizing probe measured
these times on one thread per worker.

Run the declared study from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m tests.canonical.refuter_calibration_band.regenerate --output <empty scratch dir> --jobs <cores>
```

The declarations are in `tests/studies/refuter_calibration_band.py` and
`tests/studies/refuter_calibration_band_properties.py`.
