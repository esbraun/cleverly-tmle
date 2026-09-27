# RM18 design OW: the ordinary weighted study

This directory holds the design that RM18 of `docs/roadmap.md` declares in "Design OW", for the
six red rows of `weighted-ltmle`. It changes no study, no verdict and no committed row.

| file | what it holds |
| --- | --- |
| `run.py` | parts OW-A, OW-B and OW-C, each run on its own with `--part` |
| `ow-a-*`, `ow-b-*`, `ow-c-*` | per part: `validation.csv` (the R4 checks), `rows.csv.gz` (one row per fresh fit, with `seed`) and `reading.csv` |
| `run.log` | one block per part: commit, command, runtime, thread limits, wall time, exit code, output hashes |

`tests/unit/test_rm18_ordinary_weighted_diagnostic.py` runs in the fast tier. It checks the
exact-law frame, the exact-law precondition and its nonzero witness (dropping the weights moves
the untargeted value), `b_inf`, the discrimination probability, the three reading rules with a
mutation at each boundary, the family rule on synthetic rows with a mutation of the control, the
seeds, and one registered refit per part. Once a part's rows are committed it rebuilds that
part's reading from them.

## Run

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m tests.diagnostics.rm18_ordinary_weighted.run --part OW-A --output <scratch>
python -m tests.diagnostics.rm18_ordinary_weighted.run --part OW-B --output <scratch>
python -m tests.diagnostics.rm18_ordinary_weighted.run --part OW-C --output <scratch>
```

The options are those of [`../rm18_fixed_weights/`](../rm18_fixed_weights/README.md).

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| OW-A arms, fit and bounds | the FW-A arms with `cross_fit=False`, and the FW-A bounds, 3.045683 for W and 2.574884 for U | the declaration's "as FW-A", on the ordinary fit |
| OW-B law | `ltmle_properties.NULL_PROBS` in both arms. The truth is `NULL_TRUTH`, the contrast of the unselected null law, which both arms target | the registered sharp-null cell's law |
| OW-B statistics | the SE ratio with its bootstrap interval, and the Clopper-Pearson intervals of the coverage and of the rejection rate. `rejected` is the fit's two-sided p-value below 0.05 | the registered `type_i_error` rule reads these |
| OW-C draws | the registered payload `("targeting_necessity", "targeted", k, 2000, 2655, seed, "mechanism_correct")` through `_fit_replication(ORDINARY, False, payload)`. One draw gives all four cells | the registered pair on one frame |
| OW-C family rule | `summarize_cells` and `necessity_verdicts` with the registered margin 0.25 SD and displacement threshold 0.25. "The smaller displacement" is the minimum over the static and dynamic pairs, as `necessity_verdicts` publishes it | the registered family rule |
| `b_inf` | `rm18_shared.exact_selected_frame`: `law.frame()` with each `W = 0` row three times and each `W > 0` row once, and `obs_weight = 1/pi(W)`. The code asserts that the frame's cell shares equal `SELECTED_PROBS` to 1e-15. `fit(frame, "mechanism_correct", cross_fit=False)` supplies the folds, and `untargeted` gives each static regimen. `b_inf` is always minus never minus `law.TRUTH` | the declared exact-law frame |
| exact-law precondition | `rm18_shared.follower_mean`: over the support points that follow the static regimen at both nodes and stay uncensored, the mean of `Y` under `SELECTED_PROBS * OBS_WEIGHTS`. The absolute difference must be at most 1e-12 for `always` and `never`. It is a harness check in `ow-c-validation.csv` | the declared precondition |
| `SD` and `p_1200` | `SD` is the `ddof = 1` SD of the fresh `static__untargeted` estimates. `p_1200` uses `t(0.995, 1,199)` and R = 1,200, the registered family budget | the declared formula |
| seeds | OW-A labels `("rm18", "ordinary-weighted", "interval_calibration", arm, n, k)`, OW-B `(..., "type_i_error", arm, n, k)`, OW-C `(..., "targeting_necessity", k)`. Registered seeds are every `_payloads` seed and every primary seed of `weighted-ltmle` | R2. Under the collision rule two OW-A labels move to a `"retry"` seed: one collides with a registered seed and one repeats an earlier OW-A seed |
| bootstrap | `ratio_draws` with the label `f"{arm}__n{n}"` on the `ordinary-weighted` design stream | the bootstrap-stream rule |
| harness validation | OW-A: the 2,400 calibration payloads; OW-B: the 800 sharp-null payloads; OW-C: the 1,200 targeting payloads. Each compares every row its payload returns, and the study's own summary recomputes the part's cells | R4 |
| readings | OW-A reads the SE-ratio interval of W at 32,000; OW-B the rejection and coverage intervals of W at 4,000; OW-C publishes a population reading and a fresh reading. `Delta_SE(n)` and the U arms are descriptive | the declared rules |

## Sources

| claim | source | locator |
| --- | --- | --- |
| the plug-in influence-curve variance underestimates in finite samples | Tran, Petersen, Schwab and van der Laan (2023), *J Causal Inference* 11(1):20210067, DOI 10.1515/jci-2021-0067 | the published section numbers were not read first-hand, so no locator is cited |
| the same under weights, in a two-stage IPCW-LTMLE | Landsiedel, Petersen and van der Laan (2026), arXiv:2607.02702v1 | preprint section 2.4 |
| the R3 budgets | Morris, White and Crowther (2019), *Stat Med* 38(11):2074-2102 | Table 6 |

## Runtime

Single-core fit times on the idle host: 0.024, 0.074 and 0.290 s (W) and 0.024, 0.079 and
0.305 s (U) at 2,000, 8,000 and 32,000; 0.040 and 0.146 s (W) and 0.042 and 0.151 s (U) on the
null law at 4,000 and 16,000; 0.031 s for one targeting payload. OW-A costs 13,530 core-seconds
(about 21 minutes on 16 logical cores), OW-B 1,210 (about 2 minutes) and OW-C 80. Each smoke part
at `--replicates 4` took 9 s of wall time.
