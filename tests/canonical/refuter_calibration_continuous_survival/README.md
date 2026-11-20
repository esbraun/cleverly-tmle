# Continuous-time survival refutation verdict rules

This directory holds a repeated-sampling study of two decision rules that `cleverly.refute`
prints beside its statistics on a continuous-time survival fit. The X101 plan (rev 2, section 5)
declares the design, resized from the second pilot (see [Pilot](#pilot)). The declared run is
committed here; see [Results](#results).

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
| 8 size cells | 600 | base | placebo and subset on each difference, all from the same 41 fits per trial | at most 68 rejections |
| 4 stale-row cells | 50 | power: beta 2.0, cause-2 arm coefficient -1.0 | placebo with 35% of rows keeping the recorded treatment | at least 38 detections |
| 8 rescaled-subset cells | size trials 0 to 49 | base | subset `rho` at the reported SE halved and doubled | at least 38 detections |
| 2 reporting cells | 100 each | dependent censoring; 25% treated | placebo | no verdict |

The plan names only the arm-specific term `beta` for the power law. `beta` alone reaches
4 SE on every difference only at 3.5, where the true treated survival at the last target time
falls below the engine's floor of 1e-12, so the power law also moves the cause-2 arm
coefficient. `tests/unit/test_refuter_calibration_continuous_survival_design.py` checks both
facts from the exact law.

## Pilot

The pilot is `tests/studies/refuter_calibration_continuous_survival_pilot.py`, on its own seed
stream, with its rows in `tests/studies/refuter_calibration_continuous_survival_pilot.csv.gz`:
200 size trials, 12 power trials and 4 trials of each reporting cell. It ran after X6 part A,
which made the switched variance the fit's default and removed the survival-floor DataError that
cost the first pilot (57 usable size trials) 5% of its trials and kept the declaration pending.

| reading | value |
| --- | --- |
| size rejections in 200 trials, placebo (c1 1.12, c1 2.32, c2 1.12, c2 2.32) | 9, 11, 9, 10 |
| size rejections in 200 trials, subset (same order) | 14, 10, 10, 14 |
| mean subset `rho` (same order) | 1.02, 1.00, 1.01, 1.01 |
| predictive family pass at 200, 400, 600, 800 trials | 0.39, 0.82, 0.92, 0.96 |
| stale-row detections, 12 power trials, every difference | 12 of 12 |
| per-trial stale-row power at 0.35 (smallest difference) | 0.998 |
| trials that raised | 0 of 220 |
| seconds per size trial, per power trial, per reporting trial | 122, 78, 74 |

The plan requires a predictive family pass of at least 0.90 before declaration. 600 is the
smallest multiple of 200 that reaches it, so the size cells run 600 trials. Their count, 68, is
the largest whose 99% Clopper-Pearson upper endpoint (0.1506) stays under the ceiling of the X82
rule's 17 of 200 (0.1517). That keeps the interval but loosens the tolerated size: a cell passes
with probability one half at a true rejection rate of about 0.114, against 0.088 at 17 of 200.
At a rate of 0.10 a cell passes with probability 0.875, against 0.285 under the X82 count, and
at 0.15 with 0.006. The sibling X82 studies keep 17 of 200, so a calibrated row from this study
tolerates a larger size than theirs. No count that keeps the X82 tolerance reaches the family
gate on this pilot (0.61 at 55 of 600, 0.70 at 113 of 1,200).

## Results

The declared run ran at `927379cf` on 4 cores in 25,700 s (`run.log`). All 20 property cells pass
(`properties.csv`). One of the 600 size trials was lost to a DataError on subset refit 8 of 20 (an
initial hazard fit put a combined jump of one or more at one time), under the failure bound of
0.01 (`property-failures.csv`), so each size cell reads 599 trials.

| reading | c1 1.12 | c1 2.32 | c2 1.12 | c2 2.32 |
| --- | --- | --- | --- | --- |
| placebo rejections of 599 | 28 | 27 | 33 | 33 |
| subset rejections of 599 | 57 | 48 | 55 | 52 |
| stale-row detections of 50 | 50 | 50 | 50 | 50 |
| halved-SE detections of 50 | 49 | 49 | 50 | 50 |
| doubled-SE detections of 50 | 48 | 48 | 46 | 48 |

The subset rule rejects at 0.080 to 0.095, and every subset cell's 99% interval lies above 0.05.
It passes inside the amended limit of 68 rejections. The pilot read 0.05 to 0.07, so the run moved
the subset reading up. The placebo rule rejects at 0.045 to 0.055. The reporting cells reject in 2
to 9 of their 98 to 100 trials (`reporting-cells.csv`).

Three primary truth rows (`performance-tests.csv`) are red at 200 replications: cause 1 at 2.32
(coverage and SE-ratio screen under-resolved), cause 2 at 1.12 (coverage and bias margin
under-resolved), and cause 2 at 2.32 (bias interval 0.0014 to 0.0129, excluding 0). The roadmap's
`X101-short-study-sanity` row owns them. No rule row reads them.

## Runner brief

`DECLARATION_PENDING` is empty and the design test passes.

- Branch and commit: the pushed commit the row thread names. The guard needs
  `HEAD` equal to its upstream and a clean tree.
- Budget: about 26 CPU-hours from the pilot's timings, 34 with a 1.3 margin, about 8.5 hours on
  4 cores.
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
