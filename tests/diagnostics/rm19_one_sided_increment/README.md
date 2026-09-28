# RM19: the localization design of the one-sided DR-TMLE increment

This directory holds the design that RM19 of `docs/roadmap.md` declares in "The localization
design, declared before it runs", as amended. It changes no study, no verdict and no committed
row of a registered study. Commit `b7ca9460` holds that declaration.

Every declared part has run once, and its record is committed here. The primary reading, Part B
on fresh draws, is `no increment at the declared resolution`. "What the localization design
found" in RM19 of `docs/roadmap.md` gives each reading and what it does not show.

| file | what it holds |
| --- | --- |
| `rtrans.py` | the transcription of the binary cross-validated loop of R `drtmle` 1.1.2, with the factors J, P, S and K and the guard switch G |
| `run.py` | parts A, B, V4 and C, the step AV, their reading rules, and the command line |
| `count_rounds.R` | the counting wrapper of rule 1. It counts R's loop rounds per draw and sources `tests/canonical/drtmle/run_drtmle.R` unchanged, in the pinned image |
| `fluctuate_g_fallback.R`, `fluctuate-g-fallback.csv` | R's `fluctuateG` on a constructed case where both `glm` attempts fail, run in the pinned image, and its output. A unit test compares the transcription with it |
| `a-*`, `av-*`, `b-*`, `v4-*`, `c-*` | per part: `*-validation.csv`, `*-rows.csv.gz` and `*-reading.csv`. AV and V4 also write `*-twin-rows.csv.gz`, which holds the refit and the twin of rule 2, and their rows are R's counted rows. `RECORDED` in the unit test names the parts with a committed record |
| `run.log` | one block per part, as RM18 rule R6 names it |

`tests/unit/test_rm19_one_sided_increment_diagnostic.py` runs in the fast tier. It covers these
checks:

- the budget rule and every declared input, recomputed from the committed rows;
- each constant that the transcription copies from `cleverly` or the study, against its package
  value;
- the declared arms, the Part A draws, and the fresh seeds against every seed on
  `canonical-drtmle`, BD-1's included, with a forced collision that moves to the `"retry"` label;
- `T(0, 0, 0, 0)` and `C` on registered `treatment_correct` draws 1 and 3, against the committed
  R and `cleverly` rows;
- V5 on four registered draws: `treatment_correct` 14, 25 and 19 (a draw with a guard event of
  `T(0, 0, 0, 0)`) and `outcome_correct` 21 (a draw whose mechanism reaches its bound);
- one deliberate mutation for each part of K that breaks V5, a nonzero witness for each factor,
  and a mutated tilt that breaks the bracket;
- R's `fluctuateG` fallback against `fluctuate-g-fallback.csv`, and the guards of R on a
  constructed draw, which K removes;
- the primary and localization rules on synthetic tables, with a mutation for each branch, the
  2^4 contrasts against their longhand form, and the Bonferroni levels against SciPy;
- P-ctrl on each of its two branches, the Part C scaling and context rows, the K decomposition,
  the V4 precondition of parts A, B and C, and the Part A precondition of the later parts;
- V2 and V4 by round and conditioning on synthetic draws: each branch of rules 3, 4 and 5, with
  a passing and a failing case, the excused count and mean of rule 6, and the `maxIter` drift;
- rule 7: a looser `glm` tolerance on the `gr1` reduction of registered draw 1, which moves one
  refit by less than 1e-8, fails rule 3, through `transcribed_row` and `compare_by_exit`;
- X1: a recorded row 1e-8 from the same run's refit fails the refit row; X2: 2 of 200 and 24 of
  2,400 excused draws hold, and one more fails;
- the twin's QR solve against `gelsd`, the counting step's merge and its stop on a missing
  count, the counting wrapper's image and sourced runner, and V4 on the path of step AV;
- that V4 calls the registered runner with the files the study manifest hashes;
- once AV is recorded, that `a-validation.csv` is V1 to V5 over the committed Part A rows and
  the AV files.

Two tests read the committed record of each part: its reading rebuilt from its rows, its
declared budget and its seeds. `RECORDED` in the test file lists the parts whose record is
committed. For a part outside it, the same tests assert that the part has no committed file. The
commit that adds a part's files also adds the part to `RECORDED`. A part that did not validate is
still recorded, and its reading says `harness not validated, no reading`.

Commit `b51447cb` holds the Part A record under the rule of `520daebd`: `a-reading.csv` with SHA-256
`99c24c48...` and `a-validation.csv` with `24a3b366...`, the hashes of the first `run.log` block.
Commit `6d014540` rebuilt both files under the amended rule (`a-reading.csv` `3a4bba67...`) and
recorded AV, and its `run.log` block describes those AV files. The W4 review then tightened the
rule. The harness commit that follows it returns `a-validation.csv` and `a-reading.csv` to the
bytes of `b51447cb` and removes the AV files, until AV runs again under the tightened rule.
Commit "Record the RM19 AV check under the tightened rule" records that run and the rebuilt
`a-reading.csv`, with SHA-256 `ecac4675...`.

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
| `--part` | `A`, `AV`, `B`, `V4` or `C`, in that order. `AV` recomputes Part A's validation from the committed `a-rows.csv.gz`; a run of `A` takes that step itself |
| `--read-only` | rebuilds the reading from the files in `--output`, and fits nothing |
| `--replicates K` | caps every budget for a disposable smoke run. Each label of such a run says `smoke run, not the declared budget` |
| `--jobs` | the pool size. The default is `available_cores()`. The counting runs of AV and V4 use 7 R workers, below the cap of 8 |

A run without `--replicates` is refused before any work in three cases. `cleverly` is imported
from outside this tree's `src`, the tree has changes, or `HEAD` differs from its upstream.

| declared part | the pushed upstream must hold |
| --- | --- |
| AV | `a-rows.csv.gz` |
| B | `a-validation.csv` |
| V4 | `b-rows.csv.gz` |
| C | `a-validation.csv`, `b-rows.csv.gz` and `v4-validation.csv` |

A declared part reads each of these files from this directory, and a smoke run reads them from
`--output`. A fit that raises, or that returns a non-finite estimate or standard error, stops the
part with exit code 1 and no reading.

The readings of parts A and B need V4. Each part writes its reading when it ends, and the Part B
reading is `harness not validated, no reading` until V4 holds. After V4 is committed, rebuild the
readings of A and B with `--read-only --output tests/diagnostics/rm19_one_sided_increment`, so
that each folds in the V4 record. Part C carries the V4 check of rule 5 from its start.

After AV, copy its files and its `a-validation.csv` into this directory, then rebuild the Part A
reading with `--part A --read-only --output tests/diagnostics/rm19_one_sided_increment`. The
Part A reading reads the AV files from the same directory.

AV and V4 need Docker and the image `cleverly-drtmle-reference:538a3a2`. They run `docker build` on
`tests/canonical/drtmle/Dockerfile` first, as a regeneration does. `r_phase` writes the image ID
to `run.log` for both. Run nothing else on the machine during a part.

The image ID changes with each build, so it differs between the AV and V4 blocks of `run.log`.
The gated provenance is the hash of the `Dockerfile` and of the R runner in the study manifest.
The two AV runs came from two builds and wrote byte-equal R rows.

## How the code resolves the declaration

This table was fixed and committed before the run.

| point | resolution | reason |
| --- | --- | --- |
| `C` | `canonical_drtmle.draw_from_seed(scenario, n, seed)` and `fit_cleverly(frame, scenario)`. The payload adds `qn0`, `qn1` and `gn1` from `repeats[0].nuisance`, as `_replicate` does | the registered fit and payload |
| Part A draws | `replicate_seed(BINARY, scenario, k)` for `k` below 800, in the order `outcome_correct`, `treatment_correct`, `both_correct` | the registered primary draws. A test checks one against `draw_scenario` |
| fresh seeds | `shared.fresh_seeds(BINARY, labels, taken)` with the labels `("rm19", "one-sided-increment", n, k)`, B first, then C at 1,500, then C at 6,000. `taken` is `rm18_seeds.binary_registered()`, BD-1's `rm18_seeds.binary_seeds()` and `2^32 - 1` | the declared rule and order. No label moves at the declared budget |
| R numerics | `glm_irls` ports the IRLS of `stats::glm.fit` with the clamped logit link of R (`logit_linkinv` and `logit_mu_eta` in `family.c`, `THRESH = 30`). Predictions use the same link. `ols` is `glm(..., gaussian)`, and a constant regressor gives the intercept alone, as `estimateQrn` and `estimategrn` do | the comparator's own arithmetic. A fresh draw whose `Q2` linear predictor passed 30 moved 0.0019 without the clamp |
| R's mechanism fallback | `fluctuateG` predicts with `predict(fm, type = "response")` and no `newdata`, so the fitted values of the last `glm`. When both attempts fail, R sets the coefficient to 0 and still predicts with the failed retry's fitted values. `fluctuate_g_armwise` does the same | `fluctuate.R:118-125`, and `fluctuate-g-fallback.csv` from the pinned image |
| factor J | at K = 0: one `glm_irls` fit of `1(A = 1)` on `(-Qr_0 / g_0, Qr_1 / g_1)`, with `g_1` bounded at `[0.01, 0.99]` in the covariate and offset `logit(g_1)`, to `epsilon = 1e-14` and at most 100 steps, then a clip of `g_1` to `[0.01, 0.99]`. At K = 1: see K | `cleverly`'s two-arm geometry |
| J at K = 0, beyond the geometry | J = 1 also changes three settings of the mechanism step of R. The IRLS runs to `epsilon = 1e-14` and at most 100 steps, against `epsilon = 1e-8` and `maxit = 25`. The offset trim is `1e-12`, against `trimLogit(g_a, 0.01)`. The mechanism guard of R is absent. The J main effect at K = 0 includes these three changes | `fluctuate_g_joint` against `fluctuate_g_armwise`, and R3 of the K8 re-review |
| factor P | one equation-(8) step of each arm before the loop, by R's step at K = 0 and by K's outcome solver at K = 1 | the prime of `solve_with_reduction` |
| factor S | the loop tests, after each round, whether every equation meets `1e-10` relatively or `1e-3 / n` absolutely. It stops on the stall rule with factor 0.95, or at 100 rounds. Then the closing pass: up to 20 equation-(9) steps at the frozen `Qr` until the relative score is at most `1e-10`, then one solve of equations (8) and (10) at the frozen `gr`. At K = 0 that solve is one two-column `glm` per arm; at K = 1 it is K's four-column solve. The closing steps take no guard | `_solved`, `_negligible_bar`, the stall rule and `_close_at_frozen_reductions` |
| factor K | four parts, each `cleverly`'s own function at the registered settings. The **outcome solver**: `solve_fluctuation` over both arms at once, `alpha = 0.9995`, `tol = 1e-10`, `max_iter = 100`, which clips `Q*` into `[0.0005, 0.9995]` after every Newton step. The **reduction bounds**: `g` bounded at `[0.01, 0.99]` in the `gr2` target, and `gr1` bounded at `[0.01, 0.99]` where it is read. The **mechanism root**: `solve_bounded_mechanism` at J = 1 and `solve_armwise_bounded_mechanism` at J = 0, with `tol = 1e-10`. The **reduction learners**: `cross_fit_companion` with `LinearRegression` for `Qr` and `gr2` and `ColumnLogistic` for `gr1`, clipped to `[0, 1]` | `fluctuation/iterative.py` (`solve_fluctuation`), `estimators/reduced.py` (`_roles`, `ReducedSet.bounded_gr1`, `REDUCED_FAMILY_SPECS`), `fluctuation/mechanism.py`, `estimators/_nuisance.py` (`cross_fit_companion`) |
| the stall objective | the sum of the outcome and mechanism log-likelihoods. Each term is the value its solver reports when that solver is K's: the last Newton step of the equation-(8) solve, and the unconstrained tilt of the mechanism root. At K = 0 each term is the log-likelihood at the state: `Q*(A, W)`, and `g_1` under J or the armwise sum otherwise | `targeting.py`, `joint = fluctuation.loglik + mechanism.loglik`, and `solve_bounded_mechanism`, which reports `plain.loglik` |
| G and K | the guards of R act only on R's own steps: the outcome steps at K = 0 and the armwise mechanism step at J = 0, K = 0. At K = 1 every guarded step is a `cleverly` solver, so no guard acts | K2 of the amendment |
| switch G | the skip on `all(H < 1e-7)` and the fallback on a failed or large coefficient are removed. A non-finite coefficient then raises, which stops the part | the declared guard and failed-fit rule |
| `T0+G` | `T(0, 0, 0, 0)` without the guards. It is copied from `T(0, 0, 0, 0)` on a draw with no guard event, and fitted otherwise | the guards change no step on such a draw. A test checks both branches |
| the K decomposition | on Part A `treatment_correct`, `T(1, 1, 1, 0)` with one part of K added, for each part. Each row is `T1110[part] - T1110`, beside `T1111 - T1110`, at 99% | K5, supplementary |
| standard error | `stats::cov` of `DnoStar - DnQoStar - DngoStar` over the rows, divided by `n`, at the final state. The contrast is `c11 + c22 - 2 c12` | `drtmle.R` under `targeted_se`, and `run_drtmle.R` |
| the six means | equation (8) without its `Q - psi` term, which has mean zero, and equations (10) and (9) as `inf_functions.R` evaluates them. Each scale is `mean(abs(H))` | R's exit statistic, and `cleverly`'s `score_scale` |
| main effects | the mean of the eight `T` arms at level 1 minus the mean at level 0, per draw. G is `T0+G - T(0, 0, 0, 0)`. G enters the family when the `T(0, 0, 0, 0)` rows of the part, on any scenario, carry a guard event | the declared contrasts and family |
| two-way contrasts | the eight arms whose two levels agree minus the eight whose levels differ, for each of the six pairs | the 2^4 design |
| localization side | the side of the primary interval, or the sign of its point estimate when it covers 0. A zero point estimate has no side | the declared rule |
| Part A labels | the primary and localization labels read `committed-draw attribution: <label>`, on `treatment_correct` | the declared Part A reading |
| P-ctrl | `T1000 - T0000` on each control scenario: the 99% interval covers 0, and its `ddof = 1` SD is at most 0.002223 or 0.000869 | the declared prediction |
| V1 | `compare_rows` of `rm18_shared` on `(scenario, replicate, estimand)`, over `estimate` and `std_error` | R4 |
| V2 and V4, by round and conditioning | one function, `compare_by_exit`, for both, over `classify`. R's exit status is `tolIC` when its `score_max` is at most `1e-8` (from `fit-diagnostics.csv` for V2 and the runner's rows for V4), and `maxIter` otherwise. The status of `T(0, 0, 0, 0)` is its recorded exit, with `cap` read as `maxIter`. R's round count is the `rounds` of the counting run. `abs(T - R)` and the twin's shift are the largest scaled difference over the three estimates and three standard errors. The shift and the twin's change of round or exit are measured against the refit of the same run. It writes the `T0000 refit` row first: every refit of rule 2 equals its recorded `T(0, 0, 0, 0)` row to `1e-15`, in the same round and exit (X1). Then one row per class, in this order: `same round, tolIC draws` at the R4 tolerance (rule 3); `same round, maxIter draws` at `1e-4`, sensitive or not; `rounding-sensitive tolIC draws`, which pass at the R4 tolerance or under rule 4; and `round and exit status`, which pass only when the twin changes the round or the exit, at `1e-4` (rule 5). Each row's `compared` counts the class's draws and `largest_difference` is their largest `abs(T - R)`. Last, the `excused draws` row of rule 6 holds when the excused draws are at most 1% of the draws, and its `largest_difference` is their share (X2). The `maxIter` note and the count and signed mean `ate` difference of the excused draws go to `run.log`. Each is also a supplementary reading row | the declaration amended after the Part A failure. A sensitive draw within `1e-9` passes, because rule 6 defines an excused draw as one that rule 3 fails |
| rule 1 | `count_rounds.R` wraps `drtmle`'s internal `fluctuateG` with a counter and `drtmle` with a writer that keys each count by the `scenario` and `replicate` of `run_drtmle.R`'s `fit_one`, then sources `run_drtmle.R`. `run_drtmle.R` stops when a worker returns no result; the wrapper stops unless each draw has exactly one count. `COUNTING` is `REFERENCE` with this runner and `tests/` mounted at `/fixture`. One scenario per R input, at 7 workers. For V2, `V2 drtmle-r rerun` compares the counted rows with the committed R rows at `1e-15` | W2 and W5 |
| rule 2 | `transcribe(payload, solver="qr")`: every least-squares solve on R's side, the IRLS and the Gaussian `glm`, by `numpy.linalg.qr` and a triangular solve instead of `numpy.linalg.lstsq`. `condition_draw` fits the refit, `T(0, 0, 0, 0)` itself, and the twin in one run. The refit rows carry the arm `T0000` and the twin rows `T0000[qr]` | W2, and X1 of the W4 review |
| step AV | rebuilds each committed Part A draw from its seed, fits `C` for the payload, the refit and the twin, and runs rule 1. `AV payload C refit` checks each `C` against the committed Part A row at R4. It then writes `a-validation.csv` from the committed `a-rows.csv.gz` and the AV files. No other arm is fitted | W3 |
| V3 | the payload means against the committed `drtmle-r` `initial_estimate`, scaled difference at most `1e-12` | the declared check |
| V5 | `compare_rows` of the `T1111` and `C` `estimate` columns of every Part A draw, all three scenarios, at the R4 tolerance | the amended declaration, extended to the controls because K3 found the control differences to be `cleverly` details |
| V4 | the first 200 Part B draws are drawn again from their committed seeds and refit, and each `C` row must equal its Part B row to R4. The step of AV, `r_phase`, then fits the twins and runs `COUNTING`, which sources `run_drtmle.R`, in a temporary directory inside `--output`, with `CLEVERLY_R_CORES` at 7. Its rows are compared with the Part B `T(0, 0, 0, 0)` rows by round and conditioning | the declared call, with the runner arguments in one place |
| the V4 rows that stop the design, in A and C | the Part A reading folds in the V4 rows of `STOPPING`, the refit row, rule 5 and the limit of rule 6, selected by their class constants, when `--output` holds a V4 record. Part C carries it into its own validation record when it starts | F3 of the harness review |
| Part C scaling | `sqrt(n)` times `C - T0000` at 1,500 and 6,000 from the C rows, and at 3,000 from the B rows, each at `1 - 0.01 / 3`. A declared C reads the B rows of this directory, a smoke C those of `--output`, and a missing file stops the part | the declared statistic, and F5 of the harness review |
| supplementary context | the BD-1 `bias` row of `tests/diagnostics/rm18_boundary/bd-1-reading.csv` at 1,500, and the `double_robust_contraction/treatment_correct_n6000` bias of `properties.csv` at 6,000. The `C` bias at each size is the arm summary row | the declared supplementary row |
| instruments | `repeats[0].fluctuations["mean"].reduction` for the `C` rounds, exit and closing steps, and its targeted mechanism for the rows at a bound. `score_max` is the largest score of `score_equations()` for `C`, and the largest of the six means for a `T` arm | the declared instruments |

## Sources

| claim | source | locator |
| --- | --- | --- |
| the R loop, its steps and guards, its exit and its standard error | R `drtmle` 1.1.2 at commit `538a3a2` | `R/drtmle.R` (the loop and `targeted_se`), `R/fluctuate.R` (`fluctuateQ1`, `fluctuateQ2`, `fluctuateG`), `R/estimate.R` (`estimateQrn`, `estimategrn`), `R/inf_functions.R` |
| the IRLS and the clamped logit link | R 4.5.2 in the pinned image | `stats::glm.fit`; `binomial()`'s `linkinv` and `mu.eta`, `src/library/stats/src/family.c` |
| the construction | Benkeser, Carone, van der Laan and Gilbert (2017), *Biometrika* 104(4):863-880, DOI 10.1093/biomet/asx053 | Section 3.2 |
| the law of the draws | the same article, read in PMC5793673 | Section 5.1 |
| the budget rule | Morris, White and Crowther (2019), *Stat Med* 38(11):2074-2102, DOI 10.1002/sim.8086 | Table 6 |

## Runtime

Wall times on 16 logical cores at `--jobs 16`. The projection scales the measured wall time to
the declared budget. Parts A and B were measured on registered draws for the amendment that added
K: Part A with its own arms, and Part B's arms on registered `treatment_correct` draws. V4 and
Part C fit the same arms as before the amendment, so their rows repeat the phase 1 measurement,
which fitted the first 32 fresh draws.

| part | measured | draws | declared draws | projected wall |
| --- | ---: | ---: | ---: | ---: |
| A | 243.8 s | 96 | 2,400 | about 1.7 h |
| A, smoke at `--replicates 4` | 128.0 s | 12 | 2,400 | not projected |
| B | 158.1 s | 32 | 2,000 | about 2.8 h |
| V4 | 85.3 s | 32 | 200 | about 10 min |
| C | 39.7 s | 64 | 4,000 | about 45 min |

The declared runs took these wall times at `--jobs 16`. `run.log` gives each block.

| step | commit | wall time |
| --- | --- | ---: |
| A | `270872b4` | 4,742.4 s |
| AV, first run | `478b0e94` | 4,382.9 s |
| AV, under the tightened rule | `129df59e` | 4,660.6 s |
| B | `56a8b831` | 7,067.0 s |
| V4 | `3f275333` | 419.3 s |
| C | `deca5b73` | 1,554.9 s |
