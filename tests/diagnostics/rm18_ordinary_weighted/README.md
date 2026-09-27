# RM18 design OW: the ordinary weighted study

This directory holds the design that RM18 of `docs/roadmap.md` declares in "Design OW", for the
six red rows of `weighted-ltmle`. It changes no study, no verdict and no committed row.

| file | what it holds |
| --- | --- |
| `run.py` | parts OW-A, OW-B and OW-C, each run on its own with `--part` |
| `ow-a-*`, `ow-b-*`, `ow-c-*` | per part: `validation.csv` (the R4 checks), `rows.csv.gz` (one row per fresh fit, with `seed`) and `reading.csv` |
| `run.log` | one block per part, as R6 names it |

`tests/unit/test_rm18_ordinary_weighted_diagnostic.py` runs in the fast tier. It checks the
exact-law frame, the exact-law precondition and its nonzero witness, `b_inf`, and the
discrimination probability. Dropping the weights moves the untargeted value, which is the
witness. The test also checks the three reading rules at each boundary, the OW-A and OW-B tables
on synthetic ladders with a mutation each, and the family table with a mutation of the control.
It checks the supplementary `p_1200` row, the smoke label, the seeds, and one registered refit
per part. Once a part's rows are committed, it checks the declared budget and rebuilds that
part's reading.

## Run

```bash
python -m tests.diagnostics.rm18_ordinary_weighted.run --part OW-A --output <scratch>
python -m tests.diagnostics.rm18_ordinary_weighted.run --part OW-B --output <scratch>
python -m tests.diagnostics.rm18_ordinary_weighted.run --part OW-C --output <scratch>
```

The interpreter, `PYTHONPATH`, thread limits, options, refusals and failure behaviour are those
of [`../rm18_fixed_weights/`](../rm18_fixed_weights/README.md).

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| OW-A arms, fit and bounds | the FW-A arms with `cross_fit=False`, and the FW-A bounds, 3.045683 for W and 2.574884 for U | the declaration's "as FW-A", on the ordinary fit |
| OW-B law | `ltmle_properties.NULL_PROBS` in both arms. The truth is `NULL_TRUTH`, the contrast of the unselected null law, which both arms target | the registered sharp-null cell's law |
| OW-B statistics | the SE ratio with its bootstrap interval, and the Clopper-Pearson intervals of the coverage and of the rejection rate. `rejected` is the fit's two-sided p-value below 0.05 | the registered `type_i_error` rule reads these |
| OW-C draws | the registered payload `("targeting_necessity", "targeted", k, 2000, 2655, seed, "mechanism_correct")` through `_fit_replication(ORDINARY, False, payload)`. One draw gives all four cells | the registered pair on one frame |
| OW-C family rule | `summarize_cells` and `necessity_verdicts`, with the registered margin 0.25 SD and displacement threshold 0.25. "The smaller displacement" is the minimum over the static and dynamic pairs, as `necessity_verdicts` publishes it | the registered family rule |
| `b_inf` | `rm18_shared.exact_selected_frame` repeats each `W = 0` row of `law.frame()` three times and each `W > 0` row once, with `obs_weight = 1/pi(W)`. The code asserts that the frame's cell shares equal `SELECTED_PROBS` to 1e-15. `fit(frame, "mechanism_correct", cross_fit=False)` supplies the folds, and `untargeted` gives each static regimen. `b_inf` is always minus never minus `law.TRUTH` | the declared exact-law frame |
| exact-law precondition | `follower_mean` weights the outcome of each support point that follows the static regimen at both nodes and stays uncensored by `SELECTED_PROBS * OBS_WEIGHTS`. The absolute difference must be at most 1e-12 for `always` and `never`. It is a harness check in `ow-c-validation.csv` | the declared precondition |
| `SD` and `p_1200` | `SD` is the `ddof = 1` SD of the fresh `static__untargeted` estimates. `p_1200` uses `t(0.995, 1,199)` and R = 1,200, the registered family budget | the declared formula |
| supplementary `p_1200` | `p_1200` at `abs(b_inf)` over each end of the SD's 99% percentile bootstrap interval, 10,000 draws, seed `stream_seed(ORDINARY, "rm18", "ordinary-weighted", "bootstrap", "control SD")`. A lower end of zero, which only a smoke run can give, reads as an infinite standardized bias | the declared supplementary row. It enters no reading |
| seeds | `rm18_seeds.py` assigns OW-A labels `("rm18", "ordinary-weighted", "interval_calibration", arm, n, k)`, then OW-B `(..., "type_i_error", arm, n, k)`, then OW-C `(..., "targeting_necessity", k)`. The taken set starts from every `_payloads` seed and every primary seed of `weighted-ltmle` | R2 and the declared assignment order. Two OW-A labels move: one collides with a registered seed, and one repeats an earlier OW-A seed |
| bootstrap | `ratio_draws` with the label `f"{arm}__n{n}"` on the `ordinary-weighted` design stream | the bootstrap-stream rule |
| harness validation | OW-A refits the 2,400 calibration payloads, OW-B the 800 sharp-null payloads and OW-C the 1,200 targeting payloads. Each compares every row its payload returns, and the study's own summary recomputes the part's cells once | R4 |
| readings | OW-A reads the SE-ratio interval of W at 32,000. OW-B reads the rejection and coverage intervals of W at 4,000. OW-C publishes a population reading and a fresh reading. `Delta_SE(n)` and the U arms are descriptive | the declared rules |

## Sources

| claim | source | locator |
| --- | --- | --- |
| in a different two-stage IPCW-LTMLE, standard variance estimators under-cover and cross-fitted ones do not | Landsiedel, Petersen and van der Laan (2026), arXiv:2607.02702v1, a preprint | abstract |
| the R3 budgets | Morris, White and Crowther (2019), *Stat Med* 38(11):2074-2102 | Table 6. Section 5.3, Equation (1) |

## Runtime

The wall times assume about 11 effective workers on 16 logical cores. The fit times come from one
core on the idle host.

| part | fit times | cost | wall |
| --- | --- | --- | --- |
| OW-A | 0.024, 0.074 and 0.290 s (W), 0.024, 0.079 and 0.305 s (U) at 2,000, 8,000 and 32,000 | 13,530 core-s | about 21 minutes |
| OW-B | 0.040 and 0.146 s (W), 0.042 and 0.151 s (U) at 4,000 and 16,000 | 1,210 core-s | about 2 minutes |
| OW-C | 0.031 s for one targeting payload | 80 core-s, and the exact-law fits | about 1 minute |

Each smoke part at `--replicates 4` took 15 to 17 s of wall time.
