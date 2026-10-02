# Gate N8: `docs/examples/longitudinal-survival.ipynb` at 2692722c

Reviewed every cell and stored output, the callback
`tests/unit/tutorial_semantics/longitudinal_survival.py`, the probe directory
`reviews/notebook-review/probes/longitudinal-survival-final/` (`sweep.py`, `sweep.csv` and
`sweep-confirm.csv` 400 rows each, both logs, `summarize.py`, `summary.log`, `truth.py`,
`truth.log`), ledger rows LS-01 to LS-06 and LS-NF1/NF2, plan row N8,
`docs/development/example-notebooks.md`, `docs/examples/index.md:20`, gate N7,
`src/cleverly/datasets/longitudinal.py` (`make_longitudinal_survival`,
`make_longitudinal_competing`, `_H1`, `_H2`, `_SPLIT1`, `_SPLIT2`, `_L2`, `competing_truth`),
`tests/studies/canonical_ltmle_survival.py`, `tests/studies/canonical_ltmle_competing.py`,
`tests/studies/ltmle_survival_properties.py`, `tests/studies/ltmle_competing_properties.py`,
`tests/canonical/ltmle_survival/{summary,properties}.csv`,
`tests/canonical/lmtp_ltmle_competing/{summary,properties}.csv`, both method-evidence pages'
Limitations tables, `docs/references.md:1422`. Scratch: `.tmp/notebook-review/gate-N8/` (not
tracked). Light probes only: `--check` and the five runtime tests.

## Verdict

PASS with one wording fix. Every printed number reproduces (`--check` exit 0, "every cell
reproduced its stored non-image output"; runtime tests 5 passed including
`test_tutorial_semantics_at_documented_size`; no CRLF). Every repeated-draw number matches the
pooled block of `summary.log`, every population value matches `truth.log` and the generator's
constants, every trust row matches its committed artifact, the censoring and competing-risk
estimands are stated correctly with their untestable assumptions named, and no cell narrates a
refusal. The one gap is a sentence in Step 9 that the trust table contradicts.

## REQUIRED FIXES

1. Step 9 reading (cell 28), last sentence of the double-robustness paragraph: "This page fits
   a correctly specified mechanism, which no registered study covers." The trust table two cells
   later cites `interval_calibration/static_t2__correctly_specified`, whose row says "every
   nuisance fitted by saturated cell means, which are correct". That cell fits a correct mechanism
   from data (`ltmle_survival_properties.py:156-159`, configuration `both_correct`), so the
   sentence is false as written. Scope it to the construction the paragraph is about, for example:
   "No registered competing-risk study fits the mechanisms from data; the survival calibration cell
   in the trust section does, on another law." The trust cell's own sentence, "No registered study
   covers the learned mechanisms of this page", is correct and needs no change. Markdown only.

## Advisory, not gating

- Cell 25: "needs two more assumptions than Step 5 lists." Step 5 already lists exchangeability
  for "remaining tracked" and positivity for "remaining under observation". The Step 8 table
  restates the same two assumptions for disenrollment rather than adding two. "The hypothetical
  target of continued enrollment needs the same two assumptions, now for disenrollment" is the
  exact claim.
- Step 10: the recoded censoring mechanism is `P(C_k = 1) (1 - h_k (1 - s_k))`, which is not a
  main-term logistic in the history, and the outcome regressions do not match the law either.
  Neither nuisance of the death-as-censoring fit is correctly specified. The page claims nothing
  that depends on this (it reports the measured 742 of 800 and SE/SD 0.94), but one sentence
  saying both nuisances are misspecified in this fit would pre-empt the question of why the
  coverage is 0.927.
- `docs/examples/index.md:20`: "Its arm contrast is a controlled direct effect." The notebook
  now writes "a controlled direct effect only under exchangeability for death" and says this law
  has no world without death. The index sentence already adds the identification condition in
  its next sentence, so the two agree; moving the qualifier into the first sentence would match
  the notebook's label word for word.

## OK

- Censoring at plan exit (LS-01). Cell 24 draws `make_longitudinal_competing(n=4_000, seed=53,
  censoring=True)`, declares `censoring=("enrolled_p1", "enrolled_p2")`, and the protocol entry
  reads "Treat disenrollment before readmission or death as censoring, so the target is the
  incidence had every patient stayed enrolled (hypothetical strategy)". Cell 25's table names
  independent censoring (given the recorded history; "no") and positivity for staying enrolled
  ("partly, through the support report"). Counts 451, 457, 253, 2839, 233 reproduce and
  4000 - 451 - 457 - 253 = 2839. Callback pins `censoring_names`, 451, 233, and that disenrolled
  rows carry `NaN` at both event columns.
- Competing-risk estimands. Cell 29: "total effects in the sense of Young and colleagues (2020)";
  cell 31 links `https://doi.org/10.1002/sim.8471` and says the paper defines both estimands and
  their identifying conditions. Young, Stensrud, Tchetgen Tchetgen and Hernan, Statistics in
  Medicine 39(8):1199-1236, 2020 (`docs/references.md:1422-1424`) is the published version, and
  "total effect" and "controlled direct effect" are its terms. The table in cell 31 lists the
  competing-event row under the assumptions of Steps 5 and 8, and the censoring row under "also
  exchangeability and positivity for death as an intervened node, and a well-defined intervention
  that prevents death". The cause-specific incidence formula in `competing_truth` (all-cause
  survival factor, cause-specific numerator) is the total-effect functional.
- Death-as-censoring functional (LS-02, LS-NF2). The recoding `K_k = C_k (1 - D_k)` censors a
  patient at death before the period's readmission is read, which is the `(D_k, Y_k)` ordering;
  the readmission hazard among the kept is `h s / (1 - h (1 - s))`, as `truth.py` writes.
  `truth.log`: death-first never 0.304906, always 0.264627, difference -0.040279 (printed -0.0403);
  readmission-first -0.009290 (printed -0.0093). Generator constants match the `truth.py`
  equations term by term: `_H1 = (-1.1, -0.7, 0.35, -0.25)`, `_H2 = (-1.15, -0.25, -0.8, 0.4,
  0.3, -0.2, kink 0.5)`, `_SPLIT1 = (0.15, 1.1, 0.3, -0.2)`, `_SPLIT2 = (0.1, 0.5, 0.9, -0.25,
  0.2)`, `_L2 = (0.6, 0.9)`. The 4,000,000-unit simulation agrees with the quadrature within
  0.0003 on every checked quantity. The row label "the death-as-censoring g-formula functional,
  which is a controlled direct effect only under exchangeability for death" is the correct
  description; the paragraph "Do not read -0.0180 as a controlled direct effect" gives the two
  real reasons (no world without death in this law; order dependence -0.0403 against -0.0093).
- The hard-coded constant. Cell 29 says the generator returns no value for the shortcut, the
  code writes the value as a constant, and the review probe computes it by quadrature from the
  structural equations; the code comment names the death-first ordering. The callback recomputes
  both plan values from `law._hazard_one`, `law._hazard_two`, `law._relapse_share_one`,
  `law._relapse_share_two`, and `law._L2` to `abs=5e-7`, and rounds the two differences to
  -0.0403 and -0.0093. A reader is told where the number comes from and the fast tier would catch
  a drift of the generator. Not misleading; a generator truth key is a library change the page
  does not need.
- Event-free survival (cell 27). Both causes are exclusive and absorbing, so under the
  continued-enrollment hypothetical the all-cause incidence is the sum of the two cause-specific
  incidences and event-free survival is one minus that sum, with the same standard error. The
  population column uses `1 - cif(relapse) - cif(death)`: 0.6828 = 1 - 0.2467 - 0.0705. Callback
  asserts the sum, the iid standard error from the summed influence curves, the equality with
  `1 - total`, coverage of each population value, and the nonzero witness that one minus one
  cause's incidence differs by more than 0.02.
- Covariate-drop benchmark (Step 12). Follows rule 213 of `example-notebooks.md`. Moves 1.89,
  1.35, 1.46 reproduce; "about seven standard errors below zero" is 0.1576 / 0.0227 = 6.94. The
  reading connects the moves to the question (gate N7's fix): "A move of 1.89 standard errors
  changes its size, not its sign. An unrecorded cause that acts like these covariates would make
  the true reduction larger than the estimate." The direction claims match the generator: `age`
  enters `A1` (+0.3), `L2` (+0.6), `h1` (+0.35), `h2` (+0.3); `baseline_readiness` enters `A1`
  (-0.4), `A2` (-0.2), `h1` (-0.25), `h2` (-0.2); `identified_needs` enters `A2` (+0.5) and `h2`
  (+0.4 and +0.5 tanh). Each sends offers toward higher hazard, so dropping one moves the
  estimate toward zero, which is the sign of all three moves. Probe: all three positive on 800 of
  800 draws. Callback pins the three moves, their signs, and `time_varying == ((), ())` on the
  last refit.
- Trust rows (R5). Survival accuracy row: `ltmle_survival/summary.csv` cleverly
  `ate_regimen[always vs never @ t=2]` n 2000, 1600 replicates, coverage 0.940625, se_ratio
  0.99379; `canonical_ltmle_survival.py` draws `make_longitudinal_survival(n, seed,
  censoring=True)` with no cluster, `QuasiBinomialGLM` outcome and pseudo learners,
  `KnownLongitudinalMechanism` treatment and censoring. Calibration row: `properties.csv`
  `interval_calibration,static_t2__correctly_specified` n 2000, 9600 replicates, coverage CI
  0.92759 to 0.94072, se_ratio CI 0.94113 to 0.97741, configuration `both_correct`, every learner
  `law.CellMeans()` on `tests/discrete_law_survival.py`; the page says "another law: a
  finite-support binary survival law" and "every nuisance fitted by saturated cell means, which
  are correct", and scopes the horizon-two shortfall to "its own law at n = 2,000, with saturated
  fitted nuisances" (LS-04). Competing accuracy row: `lmtp_ltmle_competing/summary.csv`
  cleverly-competing-ltmle death t=2 n 4000, 1600 replicates, coverage 0.951875, se_ratio
  0.98328; `canonical_ltmle_competing.py` uses `tests/discrete_law_competing.py`, `C1`/`C2`
  censoring, `DummyClassifier(strategy="prior")`/`DummyRegressor(strategy="mean")` outcome
  learners, `KnownCompetingMechanism`. Competing DR row: `properties.csv`
  `double_robustness,death_static_t2__mechanism_correct` n 4000, 1200 replicates, bias CI
  -0.002606 to 0.002172, margin 0.008019, se_ratio 0.97776, `mechanism_correct` =
  `KnownCompetingMechanism` with Dummy outcome learners. Limits: survival page row "Inference is
  pointwise" and no mention of clustering; competing page rows "Competing events remain natural"
  and "The data are independent and unweighted" (no clustering). Each row names law,
  construction, outcome type, fold layer, role, and result.
- Repeated-draw numbers against the pooled block of `summary.log` (800 draws): retention 60-day
  coverage 763 (0.954 x 800; 383 + 380), SE/SD 1.025 (page 1.03); competing 12-interval coverage
  min 0.935 (`cif_never_readmission_t2`) and max 0.961 (`cifd_death_t1`); misses per draw 0.638
  (page 0.64) against 12 x 0.05 = 0.60; death-as-censoring covers the functional 742 of 800
  (0.927), the total effect 385 of 800, SE/SD 0.937 (page 0.94), censored below competing 800 of
  800; benchmark moves positive 800 of 800. `sweep.log` and `sweep-confirm.log` name this
  worktree's `src`. `UNPRINTED_DECIMALS` lists every one of these and the eleven study decimals
  with their sources.
- Stored outputs quoted correctly: 463, 739, 2798, 210, 0.2579, 0.4552, 0.1497, 0.3172, -0.1082,
  -0.1380, 26%, 46%; 0.151/-0.143, -0.206/0.202, 0.825/-0.059, -0.0793, -0.1251; fingerprint
  `67439090fc4bcdb0` in Steps 4, 5, 6; 0.1526, 0.3278, 0.2673, 0.4854, 1007, 608, 22.2, 27.4,
  2.424, 1.960; 0.6722 (0.6456, 0.6987), -0.1147 (-0.1423, -0.0872), -0.1576 (-0.2020,
  -0.1131); `9b9cb1dda3683cf0`, 21%, 7%, 0.2443 + 0.2109 = 0.4552; the 8 levels and 4
  differences of Step 9 with distances 0.1 to 1.1, totals 0.3230 and 0.4383, excess 0.0,
  event-free 0.6770 and 0.5617, -0.1404 + 0.0024 = -0.1380; `cd5958d44e088c86`, 0.2285 to 0.2879,
  0.2508 to 0.2698, -0.0180 (0.0210), (-0.0592, 0.0231); `passed`, `completed`, 77.8%, 473 of
  608, 27.4, -0.0679 on never/readmission/2/2; -0.1148, -0.1269, -0.1245.
- Seed-dependent sentences carry "on this draw" where the probe shows the relation below 90%:
  naive understates at t=2 (0.816), 60-day reduction larger (0.936, still marked), readmission
  difference positive at t=2 (0.547), censored interval covers both population values. Relations
  at 800 of 800 (both exit differences negative, death difference negative at both horizons and
  larger at 60 days, censored below competing, moves positive) are stated without the qualifier
  and are law properties.
- LS-03: "The competing-risk study measures that property with the true mechanism supplied" is
  what `ltmle_competing_properties.py:147-159` does. LS-05: the new draw has no miss, so the
  reading describes the 1.1 SE maximum and the expected miss rate. LS-06: cell 0 says the first
  outcome "does not separate death from disenrollment" and cell 28 says the plan-exit risk falls
  "because fewer patients die"; cell 7 names the eligibility files as the source of lost tracking.
  LS-NF1: cell 25 says the two steps use two separate synthetic laws that share event hazards.
- Refusals: zero matches for "refus", "CapabilityError", "unavailable", or "deferred" in the
  notebook; Step 11 prints only the operations that ran, and the callback asserts the exact
  three rows.
- Protocol records: both derived protocols print their changed fields; callback asserts the exact
  sets (seven of the program protocol; four for the event protocol; three for the eliminated
  protocol) and that the disenrollment entry is carried unchanged into the eliminated protocol.
- `--check`: `scripts/execute_notebook.py docs/examples/longitudinal-survival.ipynb --check`
  exit 0 (`.tmp/notebook-review/gate-N8/check.log`); CRLF False; runtime tests 5 passed
  (`.tmp/notebook-review/gate-N8/runtime.log`).
