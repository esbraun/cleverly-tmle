# R `stremr` points for the sequential baselines

This directory compares `SequentialIPW` and `SequentialGComputation` with R `stremr` on one
long panel. `tests/unit/test_sequential_baselines_stremr.py` reads `reference.csv`.

| file | content |
| --- | --- |
| `Dockerfile`, `install_pinned.R` | the pinned `stremr` image |
| `make_data.py` | draws `long.csv`: 2,000 units, two decision times, an event after each, no censoring |
| `run_stremr.R` | fits `survNPMSM`, `directIPW` and `fit_GCOMP` for always treat and never treat, and writes `reference.csv` and `versions.txt` |

## The pinned image

`stremr` has no CRAN release at the audited commit. The image installs each GitHub source at a
full commit hash and checks the hash after the checkout. Each dependency is pinned to its last
commit at or before the `stremr` commit date, 2026-02-17.

| package | repository | commit |
| --- | --- | --- |
| `stremr` | osofr/stremr | `e075f719d5978c343555d3615e409b30d381e2b9` |
| `sl3` | jeremyrcoyle/sl3 | `0e8f2365bcbe54010b8120c04a7a2dcfc8119227` |
| `gridisl` | osofr/gridisl | `617b1b7f156d99621c05bbd145a4ae2c9a592b9e` |
| `condensier` | osofr/condensier | `6bcc973291cb311f586d3035a43a2468eae541a9` |

`stremr` names `origami@old-stable` (version 0.5) as a remote. `sl3` needs `origami` 1.0.7 or
later, so the image installs `origami` from the CRAN snapshot of the base image. `stremr` calls
`origami` only in its SDR and cross-validated paths, which this comparison does not run.

## What the comparison covers

| `stremr` estimator | cleverly | relation |
| --- | --- | --- |
| `survNPMSM` | `SequentialIPW(normalization="hajek")` | the same estimator |
| `fit_GCOMP`, `stratifyQ_by_rule=TRUE`, node-2 `Qform` interacted with `Aprev` | `SequentialGComputation` | the same estimator |
| `directIPW` | none | a different estimator of the same risk |

`stremr` orders a row as `L, C, A, Y`, and cleverly orders a node as `A, C, Y`. The two packages
therefore factor a censoring node differently, and their weights differ when a unit can leave.
The panel has no censoring, so the factorizations agree. A censored comparison would compare two
different estimators of one target.

`stratifyQ_by_rule=TRUE` fits the node-`t` regression on the rows that follow the plan at `t`
only. At node 2 those rows hold both first arms. cleverly fits node 2 on the units that followed
the plan at both nodes. The node-2 `Qform` therefore interacts every covariate with `Aprev`. The
fit then splits into one fit for each first arm, and the plan's arm is cleverly's fit. Without
the interactions, the 2026-10-14 run gave a horizon-2 risk of 0.2188 for always treat, against
0.2222 for cleverly. A numpy fit on the rows that follow the plan at node 2 only reproduces 0.2188.

The treatment models are fitted per time on the history of cleverly's design. `Lprev` and
`Aprev` carry that history. They are 0 at the first time, where both fits drop them as aliased.
The outcome regressions read the covariates and no arm column, as cleverly's `outcome_design`
does at a label node.

## Regeneration

Run from the repository root, with Docker available.

```powershell
python -m tests.canonical.stremr.make_data
docker build -t cleverly-stremr-reference:e075f71 tests/canonical/stremr
docker run --rm `
  -v "${PWD}/tests/canonical/stremr:/fixture" `
  cleverly-stremr-reference:e075f71 `
  /fixture/run_stremr.R /fixture
```
