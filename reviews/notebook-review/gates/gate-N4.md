# Gate N4: `docs/examples/interventions.ipynb` at f62ed489

Reviewed the notebook (every cell and stored output), the callback
`tests/unit/tutorial_semantics/interventions.py`, the probe directory
`reviews/notebook-review/probes/interventions-final/` (`sweep_binary.py`, `sweep_dose.py`,
`summarize.py`, `truth.py`, the CSVs and logs), ledger rows IV-01 to IV-13 and IV-N1 to IV-N3,
plan row N4, `gates/gate2-plan.md` R-6, `docs/development/example-notebooks.md`, gates N1 to N3,
`src/cleverly/datasets/synthetic.py` (`ShiftDGP`, `shift_dgp`, `incremental_truth`),
`src/cleverly/interventions/shift.py`, `src/cleverly/estimators/targeting.py`,
`src/cleverly/fluctuation/submodel.py` (`mtp` builder), the three cited study scripts and their
`summary.csv` files. Scratch for this gate: `.tmp/notebook-review/gate-N4/` (not tracked).

## Verdict

The regime and incremental axes pass: every number reproduces, every repeated-draw claim matches
the committed sweep, the truth is independently confirmed, and the learner choice is disclosed
honestly. The dose axis does not pass. Its shown configuration is biased on every contrast and on
the cap gap, its intervals cover 0.70 to 0.86, and the page explains this away as a property of the
estimated density when the probe shows a configuration the house rules already prescribe covers
at the nominal rate. IV-N3 is reproduced and is a library gap, not a `q_bounds` effect. Four
fixes are required; two are notebook changes with re-execution, one is a library fix with a
failing-first test, one is wording.

## Question 1: the dose axis

All probes below are in sample on `make_shift_dose(n=3000)`, seeds 9000 onward, every
`random_state` set to the seed, the page's four policies, `n_jobs=1`. "rboost" is
`HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0)`,
the booster `example-notebooks.md` rule 147 prescribes below 10,000 rows and the one
`sweep_binary.py` already names. "quad" is `make_pipeline(PolynomialFeatures(2), LinearRegression())`,
the outcome model of the registered study (`QuadraticShiftOutcome`). Coverage is of the 95%
interval; the gap is `uncapped - capped` through `result.contrast`.

| Q | density (bins) | seeds | `+0.5 capped` | `+0.5 uncapped` | `+1.0` | gap | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boosted (page) | bare booster (40) | 120 | 0.800, bias +0.0108 | 0.858, +0.0065 | 0.700, -0.0155, SE/SD 0.61 | 0.692, -0.0043, SE/SD 0.63 | `summary.log` boosted40 |
| quad | bare booster (40) | 120 | 0.867, +0.0005 | 0.850, +0.0012 | 0.725, +0.0005, SE/SD 0.57 | 0.925, +0.0007, SE/SD 0.90 | `summary.log` quad40 |
| quad | oracle, exact law (320) | 60 | 0.933, SE/SD 0.91 | 0.967, 1.05 | 0.967, +0.0032, 1.01 | 0.983, 1.18 | `probe_oracle.log` |
| boosted | oracle, exact law (40) | 60 | 0.933, +0.0058 | 0.967 | 0.967, +0.0007 | 0.733, -0.0042, SE/SD 0.72 | `probe_oracle.log` |
| quad | rboost (40) | 60 | 0.983, +0.0007, SE/SD 1.14 | 0.967, +0.0020, 1.16 | 0.950, +0.0052 (MC SE 0.0037), 0.89 | 0.967, +0.0013, 1.03 | `probe_density.log` |
| quad | rboost (20) | 60 | 0.983 | 0.983 | 0.967, SE/SD 1.13 | 0.933 | `probe_density.log` |
| boosted | rboost (40) | 60 | 0.900, +0.0064 | 0.950 | 0.917, -0.0005, SE/SD 0.83 | 0.717, -0.0037, SE/SD 0.70 | `probe_boostq.log` |

What this settles.

| question | finding |
| --- | --- |
| is the SE shortfall the IC formula? | no. With the exact density the quadratic Q has SE/SD 1.0 and coverage 0.97 on every contrast. The shortfall is the fitted density, as the page says, but it is the *bare* booster's: a regularized booster at 40 bins has SE/SD 0.89 to 1.16 |
| where does the cap-gap bias come from? | the boosted Q. With the exact density it still biases the gap by -0.0042 (coverage 0.73); with the quadratic Q and either density the gap is unbiased and covers 0.93 to 0.98. The page's Step 10 lesson is read on a quantity its own Q biases by 12% of the true gap |
| is the page's choice defensible? | no. Plan row N4 asked for the repair "quadratic Q and more density bins" and the sweep tried only more bins, which raised the variance of an already overfit density (quad80 SD 0.070). The sweep never tried the regularized booster that the house rule requires, and the page's density learner is exactly the bare booster that rule forbids at n = 3000 |
| is an example covering 0.70 acceptable? | not when a rule-compliant configuration covers at the nominal rate. Keeping the shown fit would be showing a misconfigured estimator and describing its symptoms |

## Question 2: IV-N3

Reproduced on 30 seeds (9000 to 9029), boosted Q, bare booster density, 40 bins
(`probe_qbounds.log`); and with the regularized density (`probe_qbounds_rboost.log`).

| configuration | `+1.0` SE/SD | per-fit SE range | `+0.5 uncapped` SE/SD | coverage `+1.0` |
| --- | --- | --- | --- | --- |
| in sample, no `q_bounds` | 0.64 | 0.018 to 0.029 | 0.93 | 0.833 |
| in sample, `q_bounds=(-30, 40)` | 0.64 | identical | 0.93 | 0.800 |
| 3 folds, `q_bounds=(-30, 40)` | 3.62 | 0.039 to 0.235 | 2.40 | 1.000 |
| 3 folds, `q_bounds=(-8, 30)` (snug) | 3.57 | 0.039 to 0.228 | 2.41 | 1.000 |
| 3 folds, regularized density | 1.68 | 0.027 to 0.091 | 1.69 | 0.967 |

`q_bounds` is not the cause: wide and snug bounds give the same SE, and declaring bounds in
sample changes nothing. Cross-fitting is the cause. On seed 9000 the held-out density ratio for
`+1.0` has maximum 104 against 55 in sample, 99% quantile 16.6 against 7.2, and ESS 10.9% against
18.0% (`diag_xfit.py`). The influence curve follows those ratios; the estimate does not (empirical
SD 0.026 cross-fitted against 0.033 in sample).

This is a library gap. `ShiftSet.ratio` is documented as "Untruncated ... the bound belongs to
targeting time so a truncation curve can sweep it" (`interventions/shift.py:188-192`), but
targeting never applies one: `targeting.py:392` passes `nuisance.shifts.design` straight to the
`mtp` builder, and `submodel.py:948-963` divides it by the missingness and intermediate
mechanisms only. The module docstring (`submodel.py:119-123`) lists `g`, pi, and the intermediate
density as bounded before they enter `h`; the shift ratio is absent from that list. The propensity
path truncates `g` with `nuisance_bound`; the shift path has no analogue, and `lmtp` trims the
density ratio at the 0.999 quantile by default (`.trim`). Failing-first test: a cross-fitted
`TMLE(shifts=..., cross_fit=True)` on `make_shift_dose(n=3000, seed=9000)` with the page's
learners must bound `h_r` at targeting time under a declared ratio bound (or the quantile trim),
so that the fitted score weights and the support report's `max ratio` reflect the bound, as the
regime report already distinguishes "before truncation" from "score load reads the truncated
mechanism". The 30-seed SE/SD of 3.6 is the symptom to recheck after the fix, not the unit test.

## Question 3: learner and truth

- The degree-2 logistic `g` is fairly described. The law's logit is
  `0.6 W1 - 0.4 W2^2 + 0.5 W2 W3 + 0.3 1(W4 > 0)` inside `0.05 + 0.90 expit(.)`
  (`synthetic.py:620,638`); squares and products capture all but the step and the squeeze, which
  the page names. The sweep shows its cost honestly (fitted minimum below 0.05 on 120 of 120).
- IPSI truth: Monte Carlo on 4 x 10^7 fresh draws with a different seed, typed from the structural
  equations without `cleverly`: 0.024138 (MC SE 1.4e-6), against `truth.py` 0.024137 (2.4e-6) and
  package quadrature 0.0241378. The ATE and screen agree likewise (0.16286, 0.10546).

## Question 4: readings, sweeps, trust, stress surface, residuals, check

- `scripts/execute_notebook.py docs/examples/interventions.ipynb --check`: "every cell reproduced
  its stored non-image output", exit 0 (`check.log`). `pytest tests/unit/test_documentation_runtime.py
  tests/unit/test_documentation_links.py -k interventions`: 31 passed, exit 0. `ruff format --check`
  and `ruff check` pass. `python -m tests.prose --path docs/examples/interventions.ipynb`: no
  findings, exit 0.
- Every quoted printed number matches a stored output (Steps 2 to 15, including 0.10533, 1205,
  2.664e-06, 87.7 as "about 88", 7.11, 0.046/0.096, 2.5%, 0.036/0.037, 0.0054, 0.0118/0.0116,
  2 and 3, 65.1%/24.4%, 77.9%/36.8%, 0.9 SE, 0.9353, 1.95, 90.8%, -2.3/-6.1/-11.0, 0.0508).
- Every repeated-draw claim matches `summary.log`: 117/117, 1.07/1.10, 116 and 110 of 120,
  +0.00011/+0.00066 (0.00008), "about eight" (8.25), medians 0.93/0.94, 1.000 below the floor,
  96/103/84, +0.0108/+0.0065/-0.0155, 0.81/0.81/0.61, 87 and 57, 0.042 to 0.070, 0.024,
  83 of 120, -0.0043, positive on 120 of 120, 5.6% (0.0558). `sweep_*.log` confirm this
  worktree's `src`.
- Trust rows: 0.9525, 0.9625, 0.94625 match the three `summary.csv` files; n = 2000, in sample,
  `OracleTreatment` for both binary studies, logistic Q (`C=1e6`) and `saturated_discrete_outcome`,
  `OracleShiftDensity` over 320 bins with `QuadraticShiftOutcome` at curvature 0.15 (page law 0.25).
  "None of them estimates g or the density" is correct. R5 items named in each row.
- Stress surface: read as a misclassification stress test, "not a bound"; the printed
  association stays below 0.051; the SE conversions are arithmetic on printed values.
- Score check prints solved/ratio only; no residual appears. `needs attention` was empty for the
  shift fit on 480 of 480 sweep fits, so cell 47's "reports raise no finding" holds.
- Callback: nonzero witnesses present (off-plan smallest g; cap below the largest dose; gap SE below
  each contrast SE; perturbed score leaves [0, 1]); `UNPRINTED_DECIMALS` covers every quoted sweep
  and study number; no refusal pins.

## REQUIRED FIXES

1. Dose axis configuration (`dose-fit`, Step 9 heading, readings of Steps 9 to 11, callback).
   Replace the outcome learner with `make_pipeline(PolynomialFeatures(2), LinearRegression())`
   and the density learner with the regularized booster
   `HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=32)`,
   40 bins, in sample. Say in the Step 9 table that the quadratic model holds `[a, a^2, W]` and
   is correctly specified for this synthetic law, that a real analysis does not know the form, and
   that the registered study uses the same model. Add the two configurations to `sweep_dose.py`
   (`quad_rboost40`; and `quad_oracle320` importing `OracleShiftDensity` and `shift_dgp()`, probe
   only), run 120 seeds, and rewrite the readings on the new sweep: the per-contrast coverage, the
   gap read with its interval, and the SE-shortfall mechanism stated once with the oracle row as
   evidence (exact density: SE/SD about 1.0; bare booster: 0.57 to 0.72). Re-derive the callback's
   draw-specific relations from the new seed-32 output; do not keep "excludes its population value
   by a small margin" unless the new draw shows it. Update `UNPRINTED_DECIMALS`.
2. Step 9 heading, the in-sample rationale. Delete "This score has no finite support, so any bound
   would be a convention" as the reason. A declared convention bound is routine and the scaler
   accepts it; the probe shows bounds change nothing. State the measured reason: a cross-fitted
   shift fit with a flexible density reported standard errors 1.7 to 3.6 times the empirical SD in
   the review probe, because the held-out density ratio is unbounded (fix 3), so this page targets
   in sample as the registered study does. Keep the `q_bounds` requirement sentence as a fact, not
   as the reason.
3. Library: bound the shift density ratio at targeting time, with a failing-first test as
   Question 2 specifies, and make the shift support report say whether `max ratio` is before or
   after the bound. Update the `ShiftSet.ratio` docstring to describe what now happens.
4. Step 10 reading. After fix 1 the gap has a calibrated interval (0.93 to 0.98 across the three
   quadratic-Q probes). Read it as an interval with its sweep coverage, and drop "Read the gap as a
   point contrast" and "too narrow to read as a 95% interval".

## Advisory, not gating

- Cell 30, "The `+1.0` interval still covered on 87 and on 57 of 120 draws": the two counts
  belong to 40 and 80 bins and the sentence does not say so. Moot after fix 1.
- Cell 36 attributes the lower ESS to "the estimated ratio is more dispersed than the true ratio".
  Correct, and the regularized density keeps the +1.0 ESS share at 0.30 against 0.29 for the bare
  booster, so the reading survives the learner change.
- The sibling-sweep row for PT `populations-heading` carries the same "could declare no `q_bounds`"
  reason; fix 2's wording applies there when that page is next touched.

## Re-check of 99cb68d0

Scratch: `.tmp/notebook-review/gate-N4-recheck/` (not tracked). Verdict: PASS.

| check | result |
| --- | --- |
| fix 1, configuration | `dose-fit` uses `make_pipeline(PolynomialFeatures(2), LinearRegression())` and `HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=32)`, 40 bins, in sample. `sweep_dose.py` defines `quad_rboost40` and `quad_oracle320` (`OracleShiftDensity(shift_dgp(), edges)`, 320 bins); `sweep_dose.csv` holds 120 rows per configuration, the 480 earlier rows intact; `sweep_dose_n4.log` names this worktree's `src`. `summarize.py` regenerates `summary.log` byte for byte |
| Step 9 readings | 0.777/0.816/1.746, intervals, widths 0.052/0.048/0.131, 7.11, 2 and 3, fingerprint match the stored output. Sweep: 117/117/114 of 120, -0.0004/+0.0002/-0.0005 (0.0010, 0.0025), 1.17/1.16/0.95; oracle 0.93 to 1.01; quad40 0.57 to 0.73 and 87 to 104 of 120, all in `summary.log` |
| fix 4, the gap | 0.038, SE 0.0047, 0.0133/0.0123, 0.036 match; 0.002 < 1.96 x 0.0047 is arithmetic on printed values; 112 of 120 (0.933), +0.0006 (0.0005), 0.93, excludes zero 1.000 match `quad_rboost40`. "Point contrast" and "too narrow" sentences removed |
| mean ratio | the report's reference is `P(A + delta in support)`; the dose is normal given W, so 1 is correct for every policy. 1.04/1.03/1.08 match the output; sweep means 1.03 to 1.05 (1.045, 1.031, 1.048) and oracle 1.00 to 1.01 (1.002, 1.006, 1.014) match |
| fix 2, in-sample rationale | "no finite support, so any bound would be a convention" removed from Steps 8 and 9. The reason given is the IV-N3 mechanism (the SE reads the estimated ratio; a held-out ratio far from the truth stops it describing the spread); 3.91 is the untrimmed cross-fitted default-booster row and 0.98 the exact-density row of `point-treatment-tmle.md:620-623`, and the shipped estimator applies no trim after 19e91585, so the first row is its behavior. `q_bounds` stays as a fact. No refusal narrated |
| callback | settings pinned; coverage of each contrast and of the gap, `gap.ci[0] > 0`, gap SE below each contrast SE; mean ratios in (1.02, 1.09) with `+1.0` the largest and current practice at exactly 1 as the control; `fold_mean_ratio == (mean_ratio,)`; ESS pins bracket the output. No "small margin" pin on the dose axis. `UNPRINTED_DECIMALS` names each unprinted sweep decimal |
| other axes | cell diff against 99cb68d0^: regime and incremental cells differ only in the cell-17 timestamp; cell 44 changes 24.4% to 25.4% (a dose figure); cell 3 adds the `LinearRegression` import |
| checks | `execute_notebook.py --check` exit 0, "every cell reproduced its stored non-image output" (`check.log`); `pytest tests/unit/test_documentation_runtime.py -k interventions` 5 passed, exit 0 |

Advisory, not gating: in cell 36, "the estimated ratio therefore runs 3% to 8% high on average" is
read from one draw whose `+1.0` value (1.08) lies inside the exact density's own single-draw range
in the sweep (0.949 to 1.113). The next paragraph's sweep means carry the claim; the single-draw
sentence could say so.
