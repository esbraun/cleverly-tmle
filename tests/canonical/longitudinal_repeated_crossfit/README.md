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

The resume key holds `HEAD` and the Python, numpy, scipy, pandas and scikit-learn versions.
Build the environment from `uv.lock`, so that a rebuilt environment gives the same key:

```bash
UV_PROJECT_ENVIRONMENT=~/x46-venv uv sync --frozen --extra dev --python 3.13.16
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
~/x46-venv/bin/python -m tests.canonical.longitudinal_repeated_crossfit.regenerate --jobs 4 --output ~/scratch/x46-run-1
```

A stopped run resumes from the same commit with a new empty `--output`. Every finished primary
and property replicate is a checkpoint, and the run fits only the rest. A checkpoint holds a
recorded fitting failure too, so a resume replays it and does not refit it.

Every property loop fits through `fit_properties` (RM39b2). A recorded failure costs the cells
of its fit one replicate each, and the run goes on. The driver writes
`property-failures.csv`, `property-failure-accounting.csv` and
`property-failure-witnesses.csv`, and reads each cell under `FAILURE_BOUND = 0.01` by the
worst case of its clauses in `PROPERTY_RULES`.

## Running in parts

`part.py` fits one part, or its first `K` replicates, into the declared run's scratch. The
declared run then reuses those checkpoints. Every part must run from the same commit and with
the same `LOCALAPPDATA` as the declared run.

```bash
python -m tests.canonical.longitudinal_repeated_crossfit.part --part primary --jobs 4
python -m tests.canonical.longitudinal_repeated_crossfit.part --part calibration_i --first 6000 --jobs 4
```

Copy each part's finished checkpoints (`*.pkl.gz`) and `_key.json` out of the scratch when the
machine can be lost. To resume elsewhere, put them back under the same key, delete `_lock` and
every `.started`, `.kills` and `.tmp` marker, and run the part again.

Each calibration part fits 24,000 replicates. `calibration_i` (n = 80,000) takes about
290 CPU-hours, and each of the other four takes 5 to 15. The whole declared run takes about
360 CPU-hours. Run a calibration part in chunks of 6,000 with `--first`.

The parts are `primary`, `calibration_d`, `calibration_i`, `calibration_s`, `calibration_c`,
`calibration_m`, `band_d`, `band_i`, `band_s`, `band_c`, `band_m`, `increment_s`, `rate_2000`,
`rate_8000`, `rate_32000`, `ipsi_primary`, `stability_d` and `stability_i`. The final declared
run fits nothing that a part finished and then writes the artifacts.

## Rerun with inherited fit sets

The `repeat_stability` family holds three cells per label. `three_repeats` is the three-draw
median. `one_repeat` is draw 0 alone, the base of both ratios. `second_draw_control` is draw 1
alone, read off the three-draw fit. The positive rows pass when the upper endpoint of
`SD(three_repeats) / SD(one_repeat)` is below 1. The control passes when the upper endpoint of
`SD(second_draw_control) / SD(one_repeat)` is at or above 1, because draws 0 and 1 are
exchangeable. All three need the one-draw split variance above the spread floor, and the three
cells of a label share one verdict.

The rename and the new cell change only the two stability fit sets. `inherit.py` checkpoints the
committed rows of every other fit set into the declared run's scratch, so the declared run reuses
them. The stability laws and the primary phase are fitted again.

```bash
python -m tests.canonical.longitudinal_repeated_crossfit.inherit
python -m tests.canonical.longitudinal_repeated_crossfit.part --part stability_i --jobs 4
python -m tests.canonical.longitudinal_repeated_crossfit.part --part stability_d --jobs 4
python -m tests.canonical.longitudinal_repeated_crossfit.regenerate --jobs 4 --output ~/scratch/x79-run
python -m tests.canonical.longitudinal_repeated_crossfit.inherit --check <baseline commit>
```

`inherit.py` refuses a committed row file that its manifest does not hash, a fit set with a
missing replicate, and a row that records a lost replicate. `--check` requires every property row
outside `repeat_stability` to equal the baseline bit for bit. It compares the primary rows, the
kept stability arms (`one_repeat` was `one_repeat_control`) and the summary rows within a relative
tolerance of 1e-12 and an absolute tolerance of 1e-14. `tests/canonical/provenance-revisions.md`
records why the inherited rows are sound under the new code.
