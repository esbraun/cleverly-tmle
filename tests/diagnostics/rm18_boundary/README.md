# RM18 design BD, and part FW-B

This directory holds the design that RM18 of `docs/roadmap.md` at commit `985849c6` declares in
"Design BD", and part FW-B of "Design FW", which reads the same fresh paired draws as BD-P. It changes no study, no
verdict and no committed row.

| file | what it holds |
| --- | --- |
| `run.py` | parts BD-0 to BD-3, and the command line for every part |
| `paired.py` | the paired parts BD-P1, BD-P-pilot and BD-P2, and the BD-P and FW-B reading rules |
| `bd-*-validation.csv`, `bd-*-rows.csv.gz`, `bd-*-reading.csv` | per part: the R4 checks, the fresh rows with each sample `seed`, and the reading |
| `bd-p1-comparisons.csv`, `bd-p2-comparisons.csv` | the framework comparisons of the paired parts, both conventions |
| `pilot.csv` | the pilot's calibration-excess bound and resolution for each mean row and convention. It is committed and pushed before BD-P2 starts |
| `run.log` | one block per part |

`tests/unit/test_rm18_boundary_diagnostic.py` runs in the fast tier. It covers these checks:

- BD-0 to the declared three decimals;
- the gate reading of every cell kind, with a mutation of each failing side, and the rung rule
  through the framework on synthetic rows;
- the `R_p` formula with its floor, its cap and its non-finite refusal;
- BD-P1 against the committed `equivalence.csv`, and the `hajek` substitution with a nonzero
  witness;
- the BD-P and FW-B rules through the reading table, with a mutation of each clause;
- the BD-P1 and pilot comparisons recomputed from their committed paired rows, and the BD-P2
  table rebuilt from a comparisons file;
- the pairwise disjointness of every part's seeds on each record, and their distance from every
  registered seed;
- two registered refits of every re-read cell, and one of the control;
- the harness rule on the committed registered rows of each re-read cell, which reproduces the
  published verdict and every published leg.

For each part, the test checks the declared budget of the committed rows and rebuilds the
reading. A missing committed file fails the test. BD-P2 is the one part whose comparisons the
test does not recompute from rows, because they take about 40 s. Its reading is rebuilt from the
committed `bd-p2-comparisons.csv`.

Each part ran once. The roadmap at commit `985849c6` gives the readings in
"[What design BD found](https://github.com/esbraun/cleverly-tmle/blob/985849c668a18cf800094714ddae3fea0675cb91/docs/roadmap.md#what-design-bd-found)" and
"[What design FW found](https://github.com/esbraun/cleverly-tmle/blob/985849c668a18cf800094714ddae3fea0675cb91/docs/roadmap.md#what-design-fw-found)".

## Run

```bash
python -m tests.diagnostics.rm18_boundary.run --part BD-1 --output <scratch>
```

The interpreter, `PYTHONPATH`, thread limits, options, refusals and failure behaviour are those
of [`../rm18_fixed_weights/`](../rm18_fixed_weights/README.md). BD-P-pilot and BD-P2 also need
Docker and the pinned image `cleverly-lmtp-crossfit:1.5.4`. Each builds the image from
`tests/canonical/lmtp_crossfit` if it is absent. Run nothing else on the machine during an R
phase.

A declared BD-P2 is refused unless the pushed upstream holds
`tests/diagnostics/rm18_boundary/pilot.csv`. It reads that committed pilot, and copies it into
`--output`. A smoke BD-P2 reads the smoke pilot in `--output`. `--read-only` on a paired part
rebuilds its reading from its comparisons file and computes no comparison.

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| BD-0, coverage | the smallest count whose 99% Clopper-Pearson lower end is at or above 0.90, and the binomial probability of reaching it when the coverage is 0.95 | the declared exact binomial law |
| BD-0, SE ratio | `P(abs(ratio - 1) < 0.07 - z * s)` for a normal ratio with `s = 1 / sqrt(2R)` and `z = 2.575829` | the declared normal approximation. It reproduces 0.834 |
| BD-1 and BD-2, multi-arm cells | `run_cells([replace(cell, seed=..., replicates=R)], module._estimator)`, with the registered `PropertyCell` of the selector or multi-arm DR module. A failed replicate stops the part | the declared call |
| BD-1, binary cell | `canonical_drtmle.draw_from_seed(scenario, n, seed)` and `fit_cleverly(frame, scenario).estimates["ate"]`, transcribed by `replicate_row`, as `_property_replicate` does | the declared copy, composed from the imported pieces |
| seeds | `tests/diagnostics/rm18_seeds.py` holds the declared assignment. Multi-arm and selector cells take the `CoverageStudy` seed `stream_seed(record, "rm18", "boundary", property, cell)`, and the binary cell takes `stream_seed(BINARY, "rm18", "boundary", property, cell, k)`. BD-3 takes `stream_seed(CROSSFIT, "rm18", "boundary", "double_robustness", "both_wrong", k)`, after FW-A. A multi-arm cell's taken set holds every cell's replicate seeds, SL's 73,000 outer-rung seeds, the study's primary seeds and each earlier cell's seeds | R2, and the declared collision rule and order. The multi-arm `outcome_correct_n4000` and calibration seeds move against SL's seeds, and one BD-3 seed moves against a registered seed |
| BD-P seeds | the pilot and step 2 seeds are the primary seeds of their records, up to 800 and 20,000. The part stops if one equals a registered seed or another BD-P seed | the declared fixed BD-P seeds |
| registered rule | `apply_shared_verdicts(rows, record, rate_labels=())` and `contraction_verdicts`, on a record copy whose `resampling_seed` is `stream_seed(record, "rm18", "boundary", "bootstrap", cell)` | the cell's full registered rule on the diagnostic bootstrap stream. No rate row is fitted, because a re-read cell has one size |
| failing side | root-n: coverage upper below 0.90, `bias_discriminated`, or the SE ratio point outside 0.80 to 1.20. Rung: coverage upper below 0.90. Calibration: the SE-ratio interval or the coverage interval wholly outside its band. Control: `bias_equivalent`, or the SE ratio outside 0.1 to 10 | the declared table |
| BD-3 | the registered `both_wrong` payload through `_fit_replication(CROSSFIT, True, payload)`. The reading reads `static__both_wrong`. `b_inf` is `fit(exact_selected_frame(), "both_wrong", cross_fit=False)` minus the truth, and `SD` is the `ddof = 1` SD of the fresh static estimates | the declared control and exact limit |
| harness validation, BD-1 to BD-3 | every committed replicate of each cell: 400, 2,400, 600, 400, 1,600 and 1,200. Each study's own summary runs once over its refit cells | R4. The binary cell commits 2,400 rows, and all of them are refit |
| paired rows | the three mean rows of both implementations, with the committed or fresh `hajek` standard error beside each `lmtp` row | the declared comparisons read these rows alone. Each framework bootstrap stream is labelled by estimand, so the contrasts' absence moves no stream |
| `hajek` convention | the `lmtp` standard error becomes the `hajek` one, and the interval and `covered` are recomputed with `z = 1.959963984540054`, as `reference_artifacts` computes them | the declared substitution. A test checks the recomputed `covered` against `reference-inference.csv.gz` |
| comparisons | `independent_performance_tests`, `summarize` and `equivalence` on the study record (BD-P1) or on the fresh record (pilot, BD-P2), at most three cells at a time. Each part computes them once and writes them | the framework comparison. At 20,000 draws each bootstrap batch gathers 1,000 x 20,000 values per column |
| the fresh path | `draw_both` runs `weighted_longitudinal_common._replicate_dispatch` in batches of 500 and streams the samples into one gzip file. It then runs `Reference.run` of the registered runner once, with `CLEVERLY_R_CORES` equal to `--jobs`, and `reference_artifacts` for the three conventions. The Python and R phases never overlap | the declared `draw_and_fit` and runner, with the samples streamed so that 20,000 draws do not sit in memory |
| a killed R worker | `study_collect` refuses a worker that returned nothing. `draw_both` also stops the part unless every implementation, replicate and estimand appears exactly once | `mclapply` drops a killed worker without an error |
| BD-P harness validation | BD-P-pilot first refits the registered 800 primary draws through `draw_both`. Both implementations' rows must reproduce `replicates.csv.gz`, and the three conventions must reproduce `reference-inference.csv.gz`, to R4 | the declared validation |
| `r_pilot` and `R_p` | the largest `calibration_excess_resolution` over the three mean rows and both conventions, with a missing value counted. `R_p = min(20,000, max(800, ceil(800 * (r_pilot / 0.025)^2)))`. A non-finite `r_pilot` stops a declared part. A smoke run takes the cap, which its own cap limits again | the declared budget |
| BD-P and FW-B legs | `red_cells.paired_legs` on each comparison row, the rule the ledger uses | the framework `comparison_verdict` |
| FW-B, "the `cleverly` coverage interval lies below 0.92" | the upper end of the subject's 99% Clopper-Pearson coverage interval from `independent_performance_tests` is below 0.92 | the declared clause |
| FW-B, "resolution at or below 0.05" | the `hajek` comparison's `calibration_excess_resolution` | the declared clause |

## Sources

| claim | source | locator |
| --- | --- | --- |
| the `lmtp` native standard error is the Horvitz-Thompson form on the uncentred influence function | `lmtp` 1.5.4 and `ife` 0.2.3 in the pinned image. `lmtp:::theta_dr` passes `eif(...)`, which adds `shifted[, 1]` and does not subtract theta, to `ife::ife`. `ife`'s `std_error` is `sqrt(var(eif * weights) / n)`. `LmtpTask$make_weights` divides the weights by their mean | printed with `Rscript` from `cleverly-lmtp-crossfit:1.5.4`. `run_study.R` asserts that native equals the HT formula on every fit |
| the R3 budgets and the RMS-SE supplement | Morris, White and Crowther (2019), *Stat Med* 38(11):2074-2102, DOI 10.1002/sim.8086 | Table 6. Section 5.2 |
| the exact coverage interval | the beta-quantile form in `tests/studies/evidence/inference.py` | `clopper_pearson` |

## Runtime

`run.log` records the wall time of each declared run on 16 logical cores. BD-P2 drew
`R_p` = 19,892 paired draws.

| part | wall |
| --- | ---: |
| BD-0 | 0.0 s |
| BD-1 | 4,079.1 s |
| BD-2 | 3,125.3 s |
| BD-3 | 38.1 s |
| BD-P1 | 3.4 s |
| BD-P-pilot | 1,335.2 s |
| BD-P2 | 16,350.0 s |

Smoke runs at `--replicates 4` took 5 to 32 s of wall time per part.
