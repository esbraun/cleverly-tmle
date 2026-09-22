# RM18 learner-weight standard-error diagnostic

This directory holds the diagnostic that RM18 of `docs/roadmap.md` declares in "The
learner-weight standard-error diagnostic, declared before it runs". That subsection fixes the
selection, the comparison set, the refit, the reproduction control, the recorded statistics, the
five predictions and the reading rule. The code here applies them. It changes no study, no
verdict and no committed row.

| file | what it does |
| --- | --- |
| `select.py` | reads the committed `tests/canonical/weighted_lmtp_ltmle/property-replicates.csv.gz`. It prints the selected and comparison sets, and any difference from the declared sets. It fits nothing |
| `refit.py` | refits each selected and comparison replicate in both arms, records the declared statistics, and reads the predictions. It writes `refit.csv` and `reading.csv` |
| `refit.csv` | one row per replicate, arm and fitted regimen. The replicate-and-arm columns repeat on the three regimen rows of their fit |
| `reading.csv` | the reproduction control, P1 to P5, two supplementary rows, and the reading |
| `run.log` | the command, the preflight, the `pip freeze` digests, the wall times and the exit codes |

`tests/unit/test_rm18_learner_weight_se_diagnostic.py` runs in the fast tier. It checks the
selection against the committed rows, and the reading rule on synthetic tables and their
mutations. It checks that `refit.csv` reproduces the committed rows, and that `reading.csv`
follows from `refit.csv`. Four mutations of the recorded statistics each move the reading. One
fresh refit of selected replicate 17, in one arm, goes through the pool. It must reproduce every
recorded statistic of its three regimen rows in `refit.csv` to a relative difference of 1e-9. It
takes under one second.

## Run

Run from a clean tree at a committed state, so that `run.log` can name the code. Limit every
numerical library to one thread. The pool in `refit.py` is the only parallel layer, and the
`LTMLE` fits inside it run with `n_jobs=1`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m tests.diagnostics.rm18_learner_weight_se.select
python -m tests.diagnostics.rm18_learner_weight_se.refit --output <scratch> --jobs 16
cmp <scratch>/refit.csv tests/diagnostics/rm18_learner_weight_se/refit.csv
cmp <scratch>/reading.csv tests/diagnostics/rm18_learner_weight_se/reading.csv
```

`--output` is required and names a directory, so a bare run cannot overwrite the committed
record. `--read-only` reads an existing `refit.csv` in that directory and fits nothing.

## How the code resolves the declaration

The declaration leaves points open, and the code fixed each one before the run, as this table
states. The one exception is the P5 scope, which the table marks.

The resolutions were fixed in the commit `ff6106b` before the run started. That commit was pushed
together with the run's results, in the same push as `b2924c9`. The commit timestamps and
`run.log` attest the order on this machine. The remote does not attest it.

| point | resolution | reason |
| --- | --- | --- |
| which regimens P1 to P4 read | `always` and `never`, the two regimens of the reported contrast `ate_regimen[always vs never]`. `refit.csv` records `treat_if_l2` as well | the contrast standard error depends on these two fits alone. A floored `treat_if_l2` follower cannot move it. The declaration states no regimen for P1 to P4 |
| which regimens P5 reads | every fitted regimen: `always`, `never` and `treat_if_l2` | the declaration states "every comparison replicate, arm and regimen". The run read P5 over the contrast regimens alone. A review found the mismatch, and `reading.csv` was then rebuilt from the unchanged `refit.csv` with `--read-only` |
| the supplementary rows | P2 over every fitted regimen, and P5 over the contrast regimens, marked "not read" | each one reports the other scope. Neither one enters the reading rule |
| the floored set of the influence-curve share | the union of the floored followers of the `always` and `never` fits | the contrast influence curve is the `always` curve minus the `never` curve |
| the relative difference | `abs(refit - committed) / abs(committed)`, on each of `estimate` and `std_error` | no committed value is zero |
| "one process that uses every core" | one parent process with one `joblib` pool of 16 workers, through `map_parallel`. Each worker uses one thread | the regeneration that wrote the committed rows used the same pool. A single Python process cannot fit on 16 cores |

Each refit calls `_fit_replication` unchanged. `refit.py` wraps the module-level `fit` of
`tests/studies/weighted_longitudinal_properties_common.py` for the duration of one call. The
statistics therefore come from the `LTMLE` result that produced the checked row. `refit_one`
refuses a result whose contrast standard error differs from the row's.

## Runtime

The run uses the runtime that `tests/canonical/weighted_lmtp_ltmle/manifest.json` records for
the committed rows: Python 3.13.7, NumPy 2.4.6, SciPy 1.18.0, pandas 3.0.5 and scikit-learn
1.9.0. The main checkout's virtual environment holds that runtime, and the run reads it without
changes. `PYTHONPATH` puts this tree's `src` and root first, and the preflight checks
`cleverly.__file__`. The choice was fixed before the run. RM18's runtime isolation found that the
pooled code at `0e03a15` reproduces the committed weighted property rows at this runtime. Between
`0e03a15` and the run commit, `src/` changed in `src/cleverly/estimators/ctmle.py` alone.

Before the run, one smoke refit fitted replicate 100 in both arms. That replicate is outside both
sets. `run.log` records it.

## The recorded run

The run started at 2026-09-22 05:42:07 UTC, which is 2026-09-21 22:42 in the local time zone
(UTC-7). `ff6106b` was committed at 05:42:01 UTC. The run used that clean tree, with 16
workers. The 52 refits took 7 seconds of wall time. `run.log` records a first attempt at
2026-09-22 05:34:37 UTC, at `ceb0ce2`, a commit that was later amended into `ff6106b`. That
attempt stopped before any fit returned, because `map_parallel` split each payload into seven
arguments. `refit_all` now passes each payload whole.

| item | result | value |
| --- | --- | --- |
| reproduction control | holds | largest relative difference 1.317e-15 on `estimate` and 1.386e-15 on `std_error`, over 52 refits |
| P1 | holds | 1 to 3 floored followers in each of the 26 selected replicate and arm pairs |
| P2 | holds | no floored follower in the 26 comparison replicate and arm pairs |
| P3 | holds | floored followers carry from 0.999923 to 1.000000 of the squared contrast influence curve |
| P4 | holds | each floored follower has a raw prefix of exactly zero |
| P5 | holds | comparison `max_weight` from 9.51 to 373.78 over all 78 regimen fits, against 2,000 |
| reading | positivity of the out-of-fold saturated mechanism | P1, P3 and P4 hold |

The selected fits count 44 floored followers, and all of them are in the `always` regimen.
No `never` or `treat_if_l2` fit has a floored follower.
Both supplementary rows hold. P2 over every regimen finds no floored follower, and P5 over the
contrast regimens runs from 13.87 to 373.78.
`reading.csv` and `refit.csv` carry every value above.
