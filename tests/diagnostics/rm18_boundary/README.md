# RM18 design BD, and part FW-B

This directory holds the design that RM18 of `docs/roadmap.md` declares in "Design BD", and part
FW-B of "Design FW", which reads the same fresh paired draws as BD-P. It changes no study, no
verdict and no committed row.

| file | what it holds |
| --- | --- |
| `run.py` | every part, run on its own with `--part`, in the order BD-0, BD-1, BD-2, BD-3, BD-P1, BD-P-pilot, BD-P2 |
| `paired.py` | the paired machinery of BD-P1, BD-P-pilot and BD-P2, and the BD-P and FW-B reading rules |
| `bd-*-validation.csv`, `bd-*-rows.csv.gz`, `bd-*-reading.csv` | per part: the R4 checks, the fresh rows with each sample `seed`, and the reading |
| `pilot.csv` | the pilot's calibration-excess bound and resolution for each mean row and convention. It is committed and pushed before BD-P2 starts |
| `bd-p2-comparisons.csv` | the step 2 framework comparisons, both conventions |
| `run.log` | one block per part |

`tests/unit/test_rm18_boundary_diagnostic.py` runs in the fast tier. It recomputes BD-0 to the
declared three decimals, checks the gate reading of every cell kind with a mutation of each
failing side, the rung rule through the framework on synthetic rows, the `R_p` formula with its
floor and cap, step 1 against the committed `equivalence.csv`, the `hajek` substitution with a
nonzero witness, the BD-P and FW-B rules with a mutation of each clause, the seeds, and two
registered refits of every re-read cell. Once rows are committed it rebuilds each reading.

## Run

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m tests.diagnostics.rm18_boundary.run --part BD-1 --output <scratch>
```

The options are those of [`../rm18_fixed_weights/`](../rm18_fixed_weights/README.md). BD-P-pilot
and BD-P2 need Docker and the pinned image `cleverly-lmtp-crossfit:1.5.4`; each builds it from
`tests/canonical/lmtp_crossfit` if it is absent. Run nothing else on the machine during an R
phase. BD-P2 reads `pilot.csv` from `--output`, and refuses one that differs from the committed
`pilot.csv`. `--read-only` on BD-P2 recomputes `bd-p2-comparisons.csv` from the rows.

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| BD-0, coverage | the smallest count whose 99% Clopper-Pearson lower end is at or above 0.90, and the binomial probability of reaching it when the coverage is 0.95 | the declared exact binomial law |
| BD-0, SE ratio | `P(abs(ratio - 1) < 0.07 - z * s)` for a normal ratio with `s = 1 / sqrt(2R)` and `z = 2.575829` | the declared normal approximation. It reproduces 0.834 |
| BD-1 and BD-2, multi-arm cells | `run_cells([replace(cell, seed=..., replicates=R)], module._estimator)`, with the registered `PropertyCell` of the selector or multi-arm DR module. A replicate that fails stops the part | the declared call |
| BD-1, binary cell | `canonical_drtmle.draw_from_seed(scenario, n, seed)` and `fit_cleverly(frame, scenario).estimates["ate"]`, transcribed by `replicate_row`, as `_property_replicate` does | the declared copy, composed from the imported pieces |
| seeds | multi-arm: the `CoverageStudy` seed `stream_seed(record, "rm18", "boundary", property, cell)`; binary: `stream_seed(BINARY, "rm18", "boundary", property, cell, k)`; BD-3: `stream_seed(CROSSFIT, "rm18", "boundary", "double_robustness", "both_wrong", k)`. The collision rule applies to each. The registered set of a coverage-study module holds every cell's replicate seeds, SL's 73,000 outer-rung seeds and the study's primary seeds | R2 and the declared collision rule. The multi-arm `outcome_correct_n4000` and calibration seeds move to a `"retry"` seed against SL's seeds, and one BD-3 seed moves against a registered seed |
| registered rule | `apply_shared_verdicts(rows, record, rate_labels=())` and `contraction_verdicts`, on a record copy whose `resampling_seed` is `stream_seed(record, "rm18", "boundary", "bootstrap", cell)` | the cell's full registered rule on the diagnostic bootstrap stream. No rate row is fitted, because a re-read cell has one size |
| failing side | root-n: coverage upper below 0.90, `bias_discriminated`, or the SE ratio point outside 0.80 to 1.20. Rung: coverage upper below 0.90. Calibration: the SE-ratio interval or the coverage interval wholly outside its band. Control: `bias_equivalent`, or the SE ratio outside 0.1 to 10 | the declared table |
| BD-3 | the registered `both_wrong` payload through `_fit_replication(CROSSFIT, True, payload)`; the reading reads `static__both_wrong`. `b_inf` is `fit(exact_selected_frame(), "both_wrong", cross_fit=False)` minus the truth, and `SD` is the `ddof = 1` SD of the fresh static estimates | the declared control and exact limit |
| harness validation, BD-1 to BD-3 | every committed replicate of each cell: 400, 2,400, 600, 400, 1,600 and 1,200. The study's own summary recomputes each cell | R4. The binary cell commits 2,400 rows, and all of them are refit |
| paired rows | the three mean rows of both implementations, with the committed or fresh `hajek` standard error beside each `lmtp` row | the declared comparisons read these rows alone. Each framework bootstrap stream is labelled by estimand, so the contrasts' absence moves no stream |
| `hajek` convention | the `lmtp` standard error becomes the `hajek` one, and the interval and `covered` are recomputed with `z = 1.959963984540054`, as `reference_artifacts` computes them | the declared substitution. A test checks the recomputed `covered` against `reference-inference.csv.gz` |
| comparisons | `independent_performance_tests`, `summarize` and `equivalence` on the study record (step 1) or on the fresh record (pilot, step 2), at most three cells at a time | the framework comparison. At 20,000 draws each bootstrap batch gathers 1,000 x 20,000 values per column |
| the fresh path | `draw_both`: `weighted_longitudinal_common._replicate_dispatch` in batches of 500, streamed into one gzip sample file; then one `Reference.run` of the registered runner with `CLEVERLY_R_CORES` equal to `--jobs`; then `reference_artifacts` for the three conventions. The Python and R phases never overlap | the declared `draw_and_fit` and runner, with the samples streamed so that 20,000 draws do not sit in memory |
| a killed R worker | `study_collect` refuses a worker that returned nothing. `draw_both` also refuses the part unless every implementation, replicate and estimand appears exactly once | the orchestrator's warning about `mclapply` |
| BD-P harness validation | BD-P-pilot first refits the registered 800 primary draws through `draw_both`. Both implementations' rows must reproduce `replicates.csv.gz`, and the three conventions `reference-inference.csv.gz`, to R4 | the declared validation |
| `r_pilot` and `R_p` | the largest `calibration_excess_resolution` over the three mean rows and both conventions; `R_p = min(20,000, max(800, ceil(800 * (r_pilot / 0.025)^2)))`. A non-finite `r_pilot`, which only a smoke run of a few draws can give, gives the cap | the declared budget |
| BD-P and FW-B legs | `red_cells.paired_legs` on each comparison row, the rule the ledger uses | the framework `comparison_verdict` |
| FW-B, "the `cleverly` coverage interval lies below 0.92" | the upper end of the subject's 99% Clopper-Pearson coverage interval from `independent_performance_tests` is below 0.92 | the declared clause |
| FW-B, "resolution at or below 0.05" | the `hajek` comparison's `calibration_excess_resolution` | the declared clause |

## Sources

| claim | source | locator |
| --- | --- | --- |
| the `lmtp` native standard error is the Horvitz-Thompson form on the uncentred influence function | `lmtp` 1.5.4 and `ife` 0.2.3 in the pinned image: `lmtp:::theta_dr` passes `eif(...)`, which adds `shifted[, 1]` and does not subtract theta, to `ife::ife`; `ife`'s `std_error` is `sqrt(var(eif * weights) / n)`; `LmtpTask$make_weights` divides the weights by their mean | printed with `Rscript` from `cleverly-lmtp-crossfit:1.5.4`. `run_study.R` asserts native equals the HT formula on every fit |
| the R3 budgets and the RMS-SE supplement | Morris, White and Crowther (2019), *Stat Med* 38(11):2074-2102, DOI 10.1002/sim.8086 | Table 6; section 5.2 |
| the exact coverage interval | the beta-quantile form in `tests/studies/evidence/inference.py` | `clopper_pearson` |

## Runtime

Single-core fit times from the planning probes and this harness: selector `n_500` 0.12 s,
binary `treatment_correct_n1500` 1.0 s, multi-arm `outcome_correct_n4000` 0.8 s, multi-arm
`n_500` 0.4 s, multi-arm calibration 0.7 s, the both-wrong payload 0.043 s, and the primary
`cleverly` fit 0.054 s.

| part | fresh cost | validation cost | wall on 16 logical cores |
| --- | --- | --- | --- |
| BD-0 | none | none | seconds |
| BD-1 | 13,900 core-s | 3,100 core-s | about 26 minutes |
| BD-2 | 11,900 core-s | 1,100 core-s | about 20 minutes |
| BD-3 | 110 core-s | 50 core-s | about 1 minute |
| BD-P1 | none | none | about 1 minute |
| BD-P-pilot | 800 paired draws | 800 registered paired draws | about 30 minutes |
| BD-P2 | `R_p` paired draws | none | about 0.9 s per draw: 4.0 hours at 16,086, 4.9 hours at the cap |

The R phase of 96 registered draws took 93 s of wall time on 16 R workers. Its container peaked
at 3.46 GiB of the 15.46 GiB Docker grants, because `study_stream` reads 64,000 sample lines at a
time. Smoke runs at `--replicates 4` took 0 to 28 s of wall time per part.
