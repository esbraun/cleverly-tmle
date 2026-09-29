# RM18 design FW: fixed known weights

This directory holds part FW-A of the design that RM18 of `docs/roadmap.md` at commit
`985849c6` declares in "Design FW". Part FW-B shares its fresh draws with part BD-P, so
[`../rm18_boundary/`](../rm18_boundary/README.md) runs it and publishes its reading. The code
changes no study, no verdict and no committed row.

| file | what it holds |
| --- | --- |
| `run.py` | the harness validation, the fresh ladder and the reading of FW-A |
| `fw-a-validation.csv` | the R4 checks: the refit rows and the recomputed summary row |
| `fw-a-rows.csv.gz` | one row per fresh replicate: the registered row schema, `arm` and `seed` |
| `fw-a-reading.csv` | every declared statistic, `Delta(n)` with its attribution, and the reading |
| `run.log` | the run record that R6 names, and the SHA-256 of each output file |

`tests/unit/test_rm18_fixed_weights_diagnostic.py` runs in the fast tier. It checks the two bounds,
the declared registered interval, the seeds and the collision rule, and the reading and
attribution rules with a mutation at each boundary. It also checks the smoke label, the
failed-fit rule, one registered refit, one fresh fit, and the run guard that every RM18
diagnostic shares. It checks that the committed rows meet the declared budget, rebuilds
`fw-a-reading.csv` from them, and refits one committed fresh row. A missing committed file fails
the test.

The part ran once. The roadmap at commit `985849c6` gives the reading in
"[What design FW found](https://github.com/esbraun/cleverly-tmle/blob/985849c668a18cf800094714ddae3fea0675cb91/docs/roadmap.md#what-design-fw-found)".

## Run

Run a declared part from a clean commit that equals its pushed upstream. Use the main checkout's
virtual environment, which holds the runtime of the committed manifests, with this tree first on
`PYTHONPATH`. Limit every numerical library to one thread. The pool in `run.py` is the only
parallel layer.

```bash
W=<worktree>
export PYTHONPATH="$W/src;$W"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
<main checkout>/.venv/Scripts/python.exe \
  -m tests.diagnostics.rm18_fixed_weights.run --part FW-A --output <scratch>
```

| option | effect |
| --- | --- |
| `--output` | required. The part writes its files there, so a bare run cannot overwrite this record |
| `--read-only` | rebuilds the reading from the rows and the validation record in `--output`, and fits nothing |
| `--replicates K` | caps every budget for a disposable smoke run. Each reading of such a run says `smoke run, not the declared budget` |
| `--jobs` | the pool size. The default is `available_cores()` |

A run without `--replicates` is refused before any work in three cases: `cleverly` is imported
from outside this tree's `src`, the tree has changes, or `HEAD` differs from its upstream. A
structural R4 miss is a refit row with no committed row, or the reverse. It raises, the run exits
with code 1, and the part writes no reading. A fit that raises, or that returns a non-finite
estimate or standard error, stops the part in the same way.

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| arm W | `weighted_longitudinal_properties_common.sample(law.PROBS, n, seed)` with the registered `SELECTION`, and `fit(frame, "both_correct", cross_fit=True)` | the registered calibration cell's draw and fit |
| arm U | the same `sample` with a selection probability of 1 at every support point, so `obs_weight` is 1, and the same `fit` | the declared unweighted twin. The fit still receives the weight column, so the two arms differ in the law alone |
| U bound | `ltmle_crossfit_properties.EFFICIENCY_SD["static"]`, 2.574884 | the bound `lmtp_ltmle` publishes. A test recomputes it as `sqrt(sum(PROBS * eif^2))` |
| row statistics | `replicate_row` on the static contrast `ate_regimen[always vs never]`. `covered` reads the fit's 95% interval | the registered row schema |
| seeds | `tests/diagnostics/rm18_seeds.py` assigns the label `("rm18", "fixed-weights", arm, n, replicate)` in that order, at the declared budget. It starts from the registered seeds of `weighted-ltmle-crossfit` and every BD-P seed up to the 20,000 cap. BD-3 follows FW-A on the same record | the declared collision rule and assignment order. One FW-A label moves, `("U", 2000, 4828)`, which equals a BD-P step 2 seed |
| smoke seeds | a capped run fits the first `K` replicates of each rung, with the declared seeds | a smoke draw is a prefix of the declared draw |
| bootstrap | `ratio_draws` with 10,000 draws and the seed `stream_seed(CROSSFIT, "rm18", "fixed-weights", "bootstrap", f"{arm}__n{n}")`. One index matrix feeds the empirical ratio, the reported ratio and the SE ratio of one arm and size | R5 and the bootstrap-stream rule. `ratio_intervals` takes its percentiles from the same draws |
| `Delta(n)` | the empirical ratio of W minus that of U. Its interval is the 99% percentile interval of the draw-by-draw difference of the two arms' draws | the two-arm rule |
| harness validation | all 2,400 `interval_calibration` payloads through `_fit_replication(CROSSFIT, True, payload)`. Both rows of each payload, static and dynamic, are compared. The study's `summarize_properties` runs once on the committed rows with the refit rows in place, and the `static__correctly_specified` row must match | R4. The dynamic row comes from the same fit, so it is compared as well |
| reading | the empirical efficiency interval of W at 32,000, read from the top of the declared table | the declared rule |

## Sources

| claim | source | locator |
| --- | --- | --- |
| the pooled cross-fitted update, and its limit without weights | Díaz, Williams, Hoffman and Schenck (2023), *JASA* 118(542):846-857, DOI 10.1080/01621459.2021.1955691 | section 5.2 Steps 1-4, pp. 852-853. Theorem 3, p. 853, states no finite-sample rate |
| a known-weight IPCW-TMLE has the weighted influence curve | Rose and van der Laan (2011), *IJB* 7(1) Art. 17, DOI 10.2202/1557-4679.1217 | section 3.2 |
| the R3 budget of the efficiency row | Morris, White and Crowther (2019), *Stat Med* 38(11):2074-2102, DOI 10.1002/sim.8086 | Table 6, the empirical SE row |

## Runtime

The declared run took 1,230.5 s of wall time on 16 logical cores. `run.log` records it. A smoke
run at `--replicates 4` took 16 s.
