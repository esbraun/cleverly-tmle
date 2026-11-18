# Continuous-time survival refutation verdict rules

This directory will hold a repeated-sampling study of two decision rules that `cleverly.refute`
prints beside its statistics on a continuous-time survival fit. The X101 plan (rev 2, section 5)
declares the design. The declaration is pending and no declared run has happened; see
[Pending](#pending).

| rule | reading | rejects when |
| --- | --- | --- |
| placebo `t` | 20 refits on the treatment permuted among the units, against the exact null 0 of a risk difference, `t` with 19 degrees of freedom | two-sided p below 0.05 |
| subset `rho` | 20 refits on 700 of the 1,000 units, each keeping its outer fold, against the sampling reference of the fit's SE | two-sided `chi^2_19` p below 0.05 |

Each trial draws `n = 1000` rows of the `continuous-time-survival` base law and fits the default
`ContinuousSurvivalTMLE`, cross-fitted with 10 outer and 5 inner folds. The rules read the risk
difference of each of the two causes at the median and upper quartile of the observed event
times (1.12 and 2.32).

| cells | trials | law | readings | pass |
| --- | --- | --- | --- | --- |
| 8 size cells | 200 | base | placebo and subset on each difference, all from the same 41 fits per trial | at most 17 rejections |
| 4 stale-row cells | 50 | power: beta 2.0, cause-2 arm coefficient -1.0 | placebo with 35% of rows keeping the recorded treatment | at least 38 detections |
| 8 rescaled-subset cells | size trials 0 to 49 | base | subset `rho` at the reported SE halved and doubled | at least 38 detections |
| 2 reporting cells | 100 each | dependent censoring; 25% treated | placebo | no verdict |

The plan names only the arm-specific term `beta` for the power law. `beta` alone reaches
4 SE on every difference only at 3.5, where the true treated survival at the last target time
falls below the engine's floor of 1e-12, so the power law also moves the cause-2 arm
coefficient. `tests/unit/test_refuter_calibration_continuous_survival_design.py` checks both
facts from the exact law.

The pilot is `tests/studies/refuter_calibration_continuous_survival_pilot.py`, on its own seed
stream, with its rows in `tests/studies/refuter_calibration_continuous_survival_pilot.csv.gz`:
60 size trials, 12 power trials and 4 trials of each reporting cell.

| reading | value |
| --- | --- |
| size rejections in 57 trials, placebo (c1 1.12, c1 2.32, c2 1.12, c2 2.32) | 4, 2, 1, 1 |
| size rejections in 57 trials, subset (same order) | 5, 3, 5, 3 |
| mean subset `rho` (same order) | 1.03, 1.02, 1.07, 1.04 |
| predictive family pass at 200, 400, 600, 800 trials | 0.12, 0.31, 0.41, 0.47 |
| family pass at the pilot's point rates, 200 and 400 trials | 0.22, 0.80 |
| family pass at exact size with the pilot's correlation | 0.91 |
| stale-row detections, 12 power trials, every difference | 12 of 12 |
| per-trial stale-row power at 0.35 (smallest difference) | 0.998 |
| trials lost to the "initial survival reaches zero" DataError | 3 of 60 size trials, each on a subset refit |
| seconds per size trial, per power or reporting trial | 146, 89 |

## Pending

`DECLARATION_PENDING` in `tests/studies/refuter_calibration_continuous_survival.py` holds two
reasons, and `regenerate.py` refuses a declared run while it is set.

- The plan requires a predictive family pass of at least 0.90 before declaration. The pilot reads
  0.12 at the declared 200 trials, and no multiple of 200 up to 800 reaches 0.90 at the same
  99% Clopper-Pearson ceiling.
- The pilot lost 5% of its size trials to a DataError that X6 PR A removes, against a failure
  bound of 1%. X6 PR A also changes the fit's default SE, which the subset cells read.

After X6 PR A merges, rerun the pilot on more trials and resize from it. The subset cells rerun
if X6 PR A makes its SE the default after a declared run.

## Runner brief

Use this section once `DECLARATION_PENDING` is empty and the design test passes at the commit to
run.

- Branch and commit: the pushed commit that empties `DECLARATION_PENDING`. The guard needs
  `HEAD` equal to its upstream and a clean tree.
- Budget: about 14.5 CPU-hours measured, 19 with a 1.3 margin, about 5 hours on 4 cores.
- Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
  `NUMEXPR_NUM_THREADS` to 1, and pass `nproc --all` as `--jobs`. Each trial runs in its own
  worker process.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m tests.canonical.refuter_calibration_continuous_survival.regenerate \
  --output /tmp/x101-refuter-run-1 --jobs "$(nproc --all)"
```

- Keepalive: the session stays active for the whole run with foreground sleep-and-check polling
  every few minutes, and never ends its turn mid-run.
- Resume: each finished trial is checkpointed in the driver's run scratch, which the driver
  prints at start. After a restart, run the same command with a new empty `--output`
  (`/tmp/x101-refuter-run-2`); `run.log` records the reused trials. Before relaunching, move any
  untracked files the run wrote into this directory out of the tree and check that
  `git status -s` is empty, because the run key hashes the tree.
- Outputs to commit, all in this directory: `replicates.parquet`, `summary.csv`,
  `equivalence.csv`, `performance-tests.csv`, `properties.csv`, `property-replicates.parquet`,
  `cells.csv`, `reporting-cells.csv`, `reporting-replicates.csv`, `manifest.json`, `run.log`, and
  the failure files when the run writes them.
- If `git push` fails server-side with `commit_refs`, push the files with the GitHub
  `push_files` tool.

A verdict is read and registered by the row thread, not the runner. The red-cell readings are
in `regenerate.py`.
