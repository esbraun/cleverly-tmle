# Repeated cross-fitted longitudinal TMLE, SDR and IPSI

This study measures the repeated cross-fitted fit of `LTMLE`, `SDR`, the survival and clustered
`LTMLE` forms, the MSM form and `IPSI` (`n_repeats = 3`, five outer folds, saturated cell
means, `g_bounds = (0.01, 1)`). The report of each name is the median of the three draws, and
its variance is the median of `sigma_r^2 + (psi_r - median)^2` (Chernozhukov et al. 2018,
equation 3.14). `tests/studies/longitudinal_repeated_crossfit.py` declares the laws, the truths
and the primary phase. `tests/studies/longitudinal_repeated_crossfit_properties.py` declares the
property families, the rules and the red-cell reading.

The study is Python only. No canonical implementation is compared, so `equivalence.csv` is
empty and schema-valid. The publication policy is `reporting`.

## Smoke run

A smoke run passes a `--replicates` other than 1,600 and a scratch `--output` outside the
repository. It skips the property phase.

```bash
python -m tests.canonical.longitudinal_repeated_crossfit.regenerate --replicates 20 --output ~/scratch/x46-smoke
```

## Declared run

Run from a clean, pushed commit with every thread variable set to 1. Set `LOCALAPPDATA` to a
persistent directory: the run keeps its scratch and checkpoints at
`$LOCALAPPDATA/cleverly/runs/longitudinal-repeated-crossfit/<key>`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m tests.canonical.longitudinal_repeated_crossfit.regenerate --jobs 16 --output ~/scratch/x46-run-1
```

A stopped run resumes from the same commit with a new empty `--output`. Every finished primary
and property replicate is a checkpoint, and the run fits only the rest.

## Running in parts

`part.py` fits one part, or its first `K` replicates, into the declared run's scratch. The
declared run then reuses those checkpoints. Every part must run from the same commit and with
the same `LOCALAPPDATA` as the declared run.

```bash
python -m tests.canonical.longitudinal_repeated_crossfit.part --part primary --jobs 16
python -m tests.canonical.longitudinal_repeated_crossfit.part --part calibration_i --first 1500 --jobs 16
```

The parts are `primary`, `calibration_d`, `calibration_i`, `calibration_s`, `calibration_c`,
`calibration_m`, `rate_2000`, `rate_8000`, `rate_32000`, `ipsi_primary`, `stability_d` and
`stability_i`. The final declared run fits nothing that a part finished and then writes the
artifacts.
