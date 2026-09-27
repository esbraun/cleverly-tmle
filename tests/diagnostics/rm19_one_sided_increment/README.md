# RM19: the localization design of the one-sided DR-TMLE increment

This directory holds the design that RM19 of `docs/roadmap.md` declares in "The localization
design, declared before it runs". It changes no study, no verdict and no committed row of a
registered study.

| file | what it holds |
| --- | --- |
| `rtrans.py` | the transcription of the binary cross-validated loop of R `drtmle` 1.1.2, with the switches J, P, S and G |
| `run.py` | parts A, B, V4 and C, their reading rules, and the command line |
| `a-*`, `b-*`, `v4-*`, `c-*` | per part: `*-validation.csv`, `*-rows.csv.gz` and `*-reading.csv`. The parts have not run, so no such file exists yet |
| `run.log` | one block per part, as RM18 rule R6 names it |

`tests/unit/test_rm19_one_sided_increment_diagnostic.py` runs in the fast tier. It covers these
checks:

- the budget rule and every declared input, recomputed from the committed rows;
- the declared arms, the Part A draws, and the fresh seeds against every seed on
  `canonical-drtmle`, BD-1's included, with a forced collision that moves to the `"retry"` label;
- `T(0, 0, 0)` and `C` on registered `treatment_correct` draws 1 and 3, against the committed R
  and `cleverly` rows;
- a nonzero witness for each switch, and a mutated tilt that breaks the bracket;
- the primary and localization rules on synthetic tables, with a mutation for each branch, the
  2^3 contrasts against their longhand form, and the Bonferroni levels against SciPy;
- P-ctrl, the Part C scaling rows, the V4 precondition of Part B, and the Part A precondition of
  the later parts;
- V2 and V4 by exit status on synthetic draws: a `maxIter` draw 1e-5 away passes, a `tolIC`
  draw 1e-8 away fails, an exit mismatch fails, and a V4 exit mismatch stops Part C;
- that V4 calls the registered runner with the files the study manifest hashes.

Two tests read the committed record of each part: its validation, its reading rebuilt from its
rows, its declared budget and its seeds. `RECORDED` in the test file lists the parts whose record
is committed. For a part outside it, the same tests assert that the part has no committed file.
The commit that adds a part's files also adds the part to `RECORDED`.

## Run

Run a declared part from a clean commit that equals its pushed upstream. Use the main checkout's
virtual environment with this tree first on `PYTHONPATH`, in the Windows path form. Limit every
numerical library to one thread. The pool in `run.py` is the only parallel layer.

```bash
W=C:/Users/erics/Documents/Projects/cleverly-tmle/.claude/worktrees/<worktree>
export PYTHONPATH="$W/src;$W"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
<main checkout>/.venv/Scripts/python.exe \
  -m tests.diagnostics.rm19_one_sided_increment.run --part A --output <scratch>
```

| option | effect |
| --- | --- |
| `--output` | required. The part writes its files there, so a bare run cannot overwrite this record |
| `--part` | `A`, `B`, `V4` or `C`, in that order |
| `--read-only` | rebuilds the reading from the files in `--output`, and fits nothing |
| `--replicates K` | caps every budget for a disposable smoke run. Each label of such a run says `smoke run, not the declared budget` |
| `--jobs` | the pool size. The default is `available_cores()`. V4 uses at most 7 R workers |

A run without `--replicates` is refused before any work in three cases. `cleverly` is imported
from outside this tree's `src`, the tree has changes, or `HEAD` differs from its upstream. A
declared B is also refused unless the pushed upstream holds `a-validation.csv`, a declared V4
unless it holds `b-rows.csv.gz`, and a declared C unless it holds `a-validation.csv` and
`v4-validation.csv`. Part C carries the V4 exit-status check, so a V4 exit mismatch stops it. A fit that raises, or that returns a non-finite
estimate or standard error, stops the part with exit code 1 and no reading.

The Part B reading needs V4. Part B writes its reading when it ends, and that reading is
`harness not validated, no reading` until V4 holds. Rebuild it with `--read-only` after V4.

V4 needs Docker and the image `cleverly-drtmle-reference:538a3a2`. It runs `docker build` on
`tests/canonical/drtmle/Dockerfile` first, as a regeneration does. Run nothing else on the
machine during a part.

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| `C` | `canonical_drtmle.draw_from_seed(scenario, n, seed)` and `fit_cleverly(frame, scenario)`. The payload adds `qn0`, `qn1` and `gn1` from `repeats[0].nuisance`, as `_replicate` does | the registered fit and payload |
| Part A draws | `replicate_seed(BINARY, scenario, k)` for `k` below 800, in the order `outcome_correct`, `treatment_correct`, `both_correct` | the registered primary draws. A test checks one against `draw_scenario` |
| fresh seeds | `shared.fresh_seeds(BINARY, labels, taken)` with the labels `("rm19", "one-sided-increment", n, k)`, B first, then C at 1,500, then C at 6,000. `taken` is `rm18_seeds.binary_registered()`, BD-1's `rm18_seeds.binary_seeds()` and `2^32 - 1` | the declared rule and order. No label moves at the declared budget |
| R numerics | `glm_binomial` ports the IRLS of `stats::glm.fit` with the clamped logit link of R (`logit_linkinv` and `logit_mu_eta` in `family.c`, `THRESH = 30`). Predictions use the same link. `ols` is `glm(..., gaussian)`, and a constant regressor gives the intercept alone, as `estimateQrn` and `estimategrn` do | the comparator's own arithmetic. A fresh draw whose `Q2` linear predictor passed 30 moved 0.0019 without the clamp |
| switch J | one `glm_binomial` fit of `1(A = 1)` on `(-Qr_0 / (1 - g_1), Qr_1 / g_1)` with offset `logit(g_1)`, to `epsilon = 1e-14` and at most 100 steps, then a clip of `g_1` to `[0.01, 0.99]` | `cleverly`'s two-arm geometry. The clip stands in for `solve_bounded_mechanism`, and `at_bound` counts where it binds |
| switch P | one R-style equation-(8) step of each arm before the loop | the prime of `solve_with_reduction` |
| switch S | the loop tests, after each round, whether every equation meets `1e-10` relatively or `1e-3 / n` absolutely. It stops on the stall rule with factor 0.95 and the joint log-likelihood, or at 100 rounds. Then the closing pass: up to 20 equation-(9) steps at the frozen `Qr` until the relative score is at most `1e-10`, then one two-column solve of equations (8) and (10) per arm at the frozen `gr`. The closing steps take no guard | `_solved`, `_negligible_bar`, the stall rule and `_close_at_frozen_reductions`. The four-column solve separates into one solve per arm, because each arm's columns are zero on the other's rows |
| joint log-likelihood | the outcome log-likelihood at `Q*(A, W)`, plus the binomial log-likelihood of `1(A = 1)` at `g_1` under J = 1, or the sum over arms of `1(A = a)` at `g_a` under J = 0 | the stall rule's objective, with the armwise sum of `solve_armwise_bounded_mechanism` |
| switch G | the skip on `all(H < 1e-7)` and the fallback on a failed or large coefficient are removed. A non-finite coefficient then raises, which stops the part | the declared guard and failed-fit rule |
| `T0+G` | copied from `T(0, 0, 0)` on a draw with no guard event, and fitted otherwise | the guards change no step on such a draw. A test checks both branches |
| standard error | `stats::cov` of `DnoStar - DnQoStar - DngoStar` over the rows, divided by `n`, at the final state. The contrast is `c11 + c22 - 2 c12` | `drtmle.R` under `targeted_se`, and `run_drtmle.R` |
| the six means | equation (8) without its `Q - psi` term, which has mean zero, and equations (10) and (9) as `inf_functions.R` evaluates them. Each scale is `mean(abs(H))` | R's exit statistic, and `cleverly`'s `score_scale` |
| main effects | the mean of the four `T` arms at level 1 minus the mean at level 0, per draw. G is `T0+G - T(0, 0, 0)`, and it enters the family when the `T(0, 0, 0)` rows of the part carry a guard event | the declared contrasts and family |
| two-way contrasts | the four arms whose two levels agree minus the four whose levels differ | the 2^3 design |
| localization side | the side of the primary interval, or the sign of its point estimate when it covers 0. A zero point estimate has no side | the declared rule |
| Part A labels | the primary and localization labels read `committed-draw attribution: <label>`, on `treatment_correct` | the declared Part A reading |
| P-ctrl | `T100 - T000` on each control scenario: the 99% interval covers 0, and its `ddof = 1` SD is at most 0.002223 or 0.000869 | the declared prediction |
| V1 | `compare_rows` of `rm18_shared` on `(scenario, replicate, estimand)`, over `estimate` and `std_error` | R4 |
| V2 and V4, by exit status | one function, `compare_by_exit`, for both. R's exit status is `tolIC` when its `score_max` is at most `1e-8` (from `fit-diagnostics.csv` for V2 and the runner's rows for V4), and `maxIter` otherwise. The status of `T(0, 0, 0)` is its recorded exit, with `cap` read as `maxIter`. It writes three rows: `exit status`, whose `largest_difference` is the number of draws whose statuses differ and which fails on any; `tolIC draws` at the R4 tolerance; and `maxIter draws` at `1e-4`. Each row's `compared` is a count of draws. The `maxIter` count and largest difference also go to `run.log` as a `note:` line, through `rm18_shared.note` | the amended declaration |
| V3 | the payload means against the committed `drtmle-r` `initial_estimate`, scaled difference at most `1e-12` | the declared check |
| V4 | the first 200 Part B draws are drawn again from their committed seeds and refit, and each `C` row must equal its Part B row to R4. `REFERENCE.run` of `tests/canonical/drtmle/regenerate.py` then runs `run_drtmle.R` in a temporary directory inside `--output`, with `CLEVERLY_R_CORES` at most 7. Its rows are compared with the Part B `T(0, 0, 0)` rows by exit status | the declared call, with the runner arguments in one place |
| Part C scaling | `sqrt(n)` times `C - T000` at 1,500 and 6,000 from the C rows, and at 3,000 from the B rows in the same directory, each at `1 - 0.01 / 3` | the declared statistic |
| supplementary context | the `C` bias at 1,500 beside the BD-1 `bias` row of `tests/diagnostics/rm18_boundary/bd-1-reading.csv`, and at 6,000 beside the `double_robust_contraction/treatment_correct_n6000` bias of `properties.csv` | the declared supplementary row |
| instruments | `repeats[0].fluctuations["mean"].reduction` for the `C` rounds, exit and closing steps, and its targeted mechanism for the rows at a bound. `score_max` is the largest score of `score_equations()` for `C`, and the largest of the six means for a `T` arm | the declared instruments |

## Sources

| claim | source | locator |
| --- | --- | --- |
| the R loop, its steps and guards, its exit and its standard error | R `drtmle` 1.1.2 at commit `538a3a264c1c` | `R/drtmle.R` (the loop and `targeted_se`), `R/fluctuate.R` (`fluctuateQ1`, `fluctuateQ2`, `fluctuateG`), `R/estimate.R` (`estimateQrn`, `estimategrn`), `R/inf_functions.R` |
| the IRLS and the clamped logit link | R 4.5.2 in the pinned image | `stats::glm.fit`; `binomial()`'s `linkinv` and `mu.eta`, `src/library/stats/src/family.c` |
| the construction | Benkeser, Carone, van der Laan and Gilbert (2017), *Biometrika* 104(4):863-880, DOI 10.1093/biomet/asx053 | Section 3.2 |
| the budget rule | Morris, White and Crowther (2019), *Stat Med* 38(11):2074-2102, DOI 10.1002/sim.8086 | Table 6 |

## Runtime

Smoke runs on 16 logical cores, from `run.log` in the scratch directory. The projection scales
the 32-draw wall time to the declared budget.

| part | wall at `--replicates 4` | wall at `--replicates 32` | declared draws | projected wall |
| --- | ---: | ---: | ---: | ---: |
| A | 37.9 s | 112.1 s (96 draws) | 2,400 | about 47 min |
| B | 12.0 s | 50.4 s | 2,000 | about 53 min |
| V4 | 27.4 s | 85.3 s | 200 | about 9 min |
| C | 9.4 s | 39.7 s (64 draws) | 4,000 | about 41 min |
