# RM18 design FW: fixed known weights

This directory holds part FW-A of the design that RM18 of `docs/roadmap.md` declares in
"Design FW". Part FW-B shares its fresh draws with part BD-P, so
[`../rm18_boundary/`](../rm18_boundary/README.md) runs it and publishes its reading. The code
changes no study, no verdict and no committed row.

| file | what it holds |
| --- | --- |
| `run.py` | the harness validation, the fresh ladder and the reading of FW-A |
| `fw-a-validation.csv` | the R4 checks: the refit rows and the recomputed summary row |
| `fw-a-rows.csv.gz` | one row per fresh replicate: the registered row schema, `arm` and `seed` |
| `fw-a-reading.csv` | every declared statistic, `Delta(n)` with its attribution, and the reading |
| `run.log` | the commit, command, runtime, thread limits, wall time, exit code and output hashes |

`tests/unit/test_rm18_fixed_weights_diagnostic.py` runs in the fast tier. It checks the two bounds,
the declared registered interval, the seeds, the reading and attribution rules with a mutation at
each boundary, one registered refit, and one fresh fit. Once the rows are committed it rebuilds
`fw-a-reading.csv` from them and refits one committed fresh row.

## Run

Run from a clean pushed commit, with every numerical library limited to one thread. The pool in
`run.py` is the only parallel layer.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m tests.diagnostics.rm18_fixed_weights.run --part FW-A --output <scratch>
```

`--output` is required, so a bare run cannot overwrite this record. `--read-only` rebuilds
`fw-a-reading.csv` from the rows and the validation record in `--output` and fits nothing.
`--replicates K` caps every budget for a disposable smoke run. `--jobs` defaults to
`available_cores()`.

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| arm W | `weighted_longitudinal_properties_common.sample(law.PROBS, n, seed)` with the registered `SELECTION`, and `fit(frame, "both_correct", cross_fit=True)` | the registered calibration cell's draw and fit |
| arm U | the same `sample` with a selection probability of 1 at every support point, so `obs_weight` is 1, and the same `fit` | the declared unweighted twin. The fit still receives the weight column, so the two arms differ in the law alone |
| U bound | `ltmle_crossfit_properties.EFFICIENCY_SD["static"]`, 2.574884 | the bound `lmtp_ltmle` publishes. A test recomputes it as `sqrt(sum(PROBS * eif^2))` |
| row statistics | `replicate_row` on the static contrast `ate_regimen[always vs never]`. `covered` reads the fit's 95% interval | the registered row schema |
| seeds | `rm18_shared.ladder_payloads`: the label `("rm18", "fixed-weights", arm, n, replicate)`, assigned arm by arm, then size, then replicate, under the seed-collision rule | R2 and the declared collision rule. No FW-A label collides, so every seed is the declared label's |
| registered seeds | every `_payloads` seed of `weighted-ltmle-crossfit` and every primary replicate seed | "every registered seed of the study" |
| bootstrap | `ratio_draws` with 10,000 draws and the seed `stream_seed(CROSSFIT, "rm18", "fixed-weights", "bootstrap", f"{arm}__n{n}")`. One index matrix feeds the empirical ratio, the reported ratio and the SE ratio of one arm and size | R5 and the bootstrap-stream rule. `ratio_intervals` takes its percentiles from the same draws |
| `Delta(n)` | the empirical ratio of W minus that of U; its interval is the 99% percentile interval of the draw-by-draw difference of the two arms' draws | the two-arm rule |
| harness validation | all 2,400 `interval_calibration` payloads through `_fit_replication(CROSSFIT, True, payload)`. Both rows each payload returns, static and dynamic, are compared. The summary check runs the study's `summarize_properties` on the committed rows with the refit rows in place and compares the `static__correctly_specified` row | R4. The dynamic row comes from the same fit, so it is compared as well |
| reading | the empirical efficiency interval of W at 32,000, read from the top of the declared table | the declared rule |
| failed fit | a fit that raises propagates out of the pool, the run exits 1, and no rows are written | the failed-fit rule |

## Sources

| claim | source | locator |
| --- | --- | --- |
| the pooled cross-fitted update, and its weak convergence without weights | Díaz, Williams, Hoffman and Schenck (2023), *JASA* 118(542):846-857, DOI 10.1080/01621459.2021.1955691 | section 5.2 Steps 1-4, pp. 852-853; Theorem 3, p. 853 |
| a known-weight IPCW-TMLE has the weighted influence curve | Rose and van der Laan (2011), *IJB* 7(1) Art. 17, DOI 10.2202/1557-4679.1217 | section 3.2 |
| the R3 budget of the efficiency row | Morris, White and Crowther (2019), *Stat Med* 38(11):2074-2102, DOI 10.1002/sim.8086 | Table 6 |

## Runtime

Single-core fit times on the idle host, from arbitrary non-declared seeds: 0.034, 0.089 and
0.316 s for arm W at 2,000, 8,000 and 32,000, and 0.036, 0.096 and 0.312 s for arm U. The fresh
ladder costs 8,300 x 0.883 = 7,330 core-seconds, about 11 minutes on 16 logical cores. The
validation adds about 100 core-seconds, and the six bootstraps about a minute. A smoke run at
`--replicates 4` took 9 s of wall time.
