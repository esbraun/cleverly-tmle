# Gate N7: `docs/examples/longitudinal-tmle.ipynb` at 6714262d

Reviewed every cell and stored output, the callback
`tests/unit/tutorial_semantics/longitudinal_tmle.py`, the probe directory
`reviews/notebook-review/probes/longitudinal-tmle-final/` (`sweep.py`, `sweep.csv` 400 rows,
`sweep.log`, `summarize.py`, `summary.log`, `truth.py`, `truth.log`), ledger rows LT-01 to LT-08
and LT-N1/LT-N2, plan row N7, `docs/development/example-notebooks.md` (rules 69-73, 172-176, 213,
the trust step), gate N6, `src/cleverly/datasets/longitudinal.py` (`make_longitudinal`, `_Y`,
`_L2`), `src/cleverly/longitudinal/sequential.py:804`, `src/cleverly/longitudinal/estimator.py:2554`,
`src/cleverly/validation/longitudinal.py:175-200`, `docs/technical-reference/longitudinal-tmle.md:294`,
`tests/canonical/ltmle/summary.csv` and `properties.csv`, `tests/studies/canonical_ltmle.py`,
`tests/studies/ltmle_properties.py`, `tests/discrete_law_longitudinal.py`. Scratch:
`.tmp/notebook-review/gate-N7/` (not tracked). Light probes only.

## Verdict

PASS with one wording fix. Every printed number reproduces (`--check` exit 0), every repeated-draw
number matches `summary.log`, the truths match the generator's equations, the trust rows match the
committed artifacts, the death coding matches the generator, and no cell narrates a refusal. The
one gap is in Step 10: the benchmark reports moves in standard errors but never connects them to
the sponsor's question, so a practitioner cannot yet use the table.

## REQUIRED FIXES

1. Step 10 reading (cell 36). Add the usable conclusion. The contrast is 0.3689 with SE 0.0172,
   about 21 standard errors from zero, so a move of 2.76 standard errors changes the number and not
   the sign. All three moves are positive because each dropped covariate raises (or, for
   `baseline_readiness`, lowers) both the offers and the top-box probability, so leaving a measured
   confounder out inflates the contrast in this law. Write two sentences after "all three are
   positive here", for example: "Each of these covariates moves the offers and the outcome in the
   same direction, so leaving one out inflates the contrast. An unrecorded cause of the same kind
   would make the true contrast smaller than 0.369, and a move of three standard errors leaves it
   far above zero." Markdown only; the callback already pins the signs and the ordering. Check the
   direction claim against `_Y`, `_L2`, and the `a1`/`a2` indices in `longitudinal.py:380-390`
   (age: +0.3 on `A1`, +0.6 on `L2`, +0.3 on `Y`; readiness: -0.4 on `A1`, -0.2 on `A2`, -0.2 on
   `Y`; engagement: +0.5 on `A2`, +0.4 and +0.5 tanh on `Y`).

## Advisory, not gating

- Cell 36, "an unrecorded cause as strong as `engagement_day7` would move the contrast by 2.76
  standard errors": "would" reads as a prediction. "moved the contrast by 2.76 standard errors on
  this draw" is what the table shows. The next two sentences already say the benchmark is not a
  bound.
- Cell 37 links "double robustness" to `point-treatment-tmle.md`. The longitudinal reference has
  its own statement of the sequential condition; linking there would name the per-node form.

## OK

- LT-01. Cell 37: "The learners are finite-dimensional regressions ... The treatment and
  censoring learners match the law's form. The outcome regressions do not, because the law has a
  `tanh` term." Generator: `A1 = expit(0.3 W1 - 0.4 W2)`, `A2 = expit(0.5 L2 + 0.6 A1 - 0.2 W2)`,
  `C1 = expit(2.2 + 0.3 W1 - 0.3 A1)`, `C2 = expit(2.4 + 0.2 L2)`, all main-term logistic in the
  history; `Y` adds `0.5 tanh(L2)`. The double-robustness sentence states consistency when either
  nuisance is consistent and says the estimate "leans on the mechanisms". Correct.
- LT-02. Cell 33: an in-sample slope near 1 holds for any intercept model, whether or not it is
  correct; the page names the `tanh` term as the reason the outcome models are wrong; cross-fitting
  named as what makes the slope informative. Correct.
- LT-03 and R5. Trust table cites the ordinary end-of-study study (file exists under
  `method-evidence/`). Row 1: `summary.csv` cleverly `ate_regimen[always vs never]` n 2000,
  1600 replicates, coverage 0.94125, se_ratio 0.98895. "Draws the law of this page without teams,
  fits the same three plans in sample" matches `canonical_ltmle.REGIMENS` (never, always, rule);
  "supplies the mechanisms exactly" matches `KnownLongitudinalMechanism`; "quasibinomial rather
  than linear" matches `QuasiBinomialGLM` and `untargeted`'s docstring. Row 2:
  `properties.csv` `double_robustness,static__mechanism_correct` n 2000, 1200 replicates, bias CI
  -0.003633 to 0.005264, margin 0.014934, se_ratio 1.013217. The row names the finite-support law
  of binary variables (`discrete_law_longitudinal.py`), mechanisms by `CellMeans` (saturated,
  fitted from data), and `DummyClassifier`/`DummyRegressor` outcome regressions. No study
  calibration is transferred to the page's clustered fits. Each row names law, construction,
  outcome type, fold layer, role, and result as the trust step requires.
- LT-04 and LT-N1. Cell 17 states the requirement positively ("A longitudinal design with
  `cluster=` needs `CrossFitting(enabled=False)`. The in-sample fit reports a cluster-robust
  variance."). Cell 37: "No registered study covers a clustered longitudinal fit." Zero matches
  for "refus" in the notebook. The library string at `estimator.py:2554-2556` reads "which reports
  a cluster-robust variance, or drop id= from fit."
- LT-05. Cell 13: death before day 30 is tracked with `transition_top_box = 0`; only living
  patients are censored; "This synthetic law draws no deaths." The generator has no death node and
  the protocol summary prints the composite rule. Consistent with gates N5 and N6.
- LT-06. Cell 9 groups by discharge navigation; gaps 0.212 and 0.213 within strata; cell 10 says
  the pooled comparison would mix in the first decision. Probe: both gaps positive 400 of 400,
  within-stratum means 0.2174 and 0.2039. Callback pins both gaps above 0.1 and the stratum
  difference above 0.1 as a nonzero witness.
- LT-07. Cell 33 reads both censoring AUCs (0.599, 0.557), explains the small coefficients, and
  cites "below 0.65 on every draw" and "node 1 on 77 of 400 draws"; both match `summary.log`.
  Callback asserts both below 0.65 and below every other AUC (400 of 400 in the probe).
- LT-08. Callback comment reads "measured 35.2"; stored max weight 35.239.
- Truths. `truth.py` writes the equations of `make_longitudinal` term by term (`_Y`, `_L2`,
  standard-normal `W1, W2, U`); `truth.log` gives 0.780426, 0.418856, 0.361570, 0.740038,
  0.321181, -0.040389, which the stored `truth` dict prints as 0.7804, 0.4189, 0.3616, 0.7400,
  0.3212 and cell 26 as -0.040. The cluster construction keeps the `U` marginal standard normal
  (`make_longitudinal` docstring), so the truth holds on the clustered draw.
- Repeated-draw numbers against `summary.log` (400 draws, seeds 1000-1399): 374 of 400 (0.935),
  SE ratio 0.969 (page 0.97), 362 of 400 without teams, both censoring AUC below 0.65 on 400 of
  400, node 1 lower on 77 of 400, all three moves positive on 400 of 400, engagement largest on
  378 of 400. `sweep.log` names this worktree's `src`. `UNPRINTED_DECIMALS` lists 0.935, 0.97,
  0.65, 0.9413, 0.9890, -0.0036, 0.0053, 0.0149, 1.0132 with sources.
- Stored outputs quoted correctly: 0.980/-0.147, 0.593/0.381/0.770/0.557, 0.489, 0.362, 923,
  1476, 4098, 2426, 0.369/0.017/(0.335, 0.403), 2362/1736, 28.1/35.2, 400 clusters, 0.225/0.137,
  0.413/0.051, 0.007, 0.331/(0.294, 0.368), 0.799, -0.038/(-0.057, -0.018), 0.321, -0.040,
  `4be9fae43366d287` in Steps 4, 5, 6, 8, 35.239, 79.5%, 75 of 11175 (3478 + 2362 + 3599 + 1736),
  0.0019, 0.11, 0.599/0.557, 0.999-1.004, 1.007-1.023, 0.3689/0.0172, 0.4083/2.30,
  0.3806/0.68, 0.4163/2.76.
- Assumptions. Cell 16 names sequential exchangeability (not testable), sequential positivity
  (partly, via the support report), consistency and no interference (not testable); cell 37
  repeats that nothing in the fit validates the causal reading. The terms table defines
  sequential positivity as the cumulative product. The collider row cites What If chapter 20
  (treatment-confounder feedback); the notebook stores `á` as U+00E1.
- Step 6 table. "The recursion clips their predictions to the unit interval":
  `sequential.py:804` `np.clip(scaled, 0.0, 1.0)`. The default bound `(0.01, 1)` caps each weight
  at 100.
- Step 7. The page says the subset conditions on agreement between assignments, so the two point
  fits do not isolate a mediator bias from a confounding bias; the misses are called draw-specific
  and the structural problems law-level. Probe: adjusting below truth 400 of 400, baseline-only
  above 395 of 400.
- Step 8. Rule share 0.799 equals the engaged share among discharge-navigated tracked patients
  (callback witness, sign-flipped rule would give 0.201). `result.contrast` reading as a joint
  influence-curve contrast; probe coverage of rule vs always 385 of 400, interval below zero 399 of
  400.
- Step 9. Status rows `passed` and `completed` are printed tokens (rule 172). "A completed result
  means the calculation ran. It is not a pass." No raw solver residual is printed; the display
  floor is `tolerance * 1e-6`, and the callback asserts the raw residuals are nonzero and below
  it. "A cross-fitted fit also reports `solver` rows only" holds for the public estimator
  (`validation/longitudinal.py:182-190`; the `stitching` row belongs to the engine-level path
  that the public estimator refuses above one fold, `longitudinal-tmle.md:294-296`). Nothing reads
  as a refusal.
- Step 10 design. Follows rule 213 of `example-notebooks.md` (refit without one recorded covariate,
  print each move in standard errors, "a benchmark, not a bound", moves signed). The callback pins
  the three moves, their signs, the ordering, and that the last refit has `time_varying=((), ())`.
  The scale in standard errors is the standard benchmark unit. The signs are a law property
  (every dropped covariate confounds upward), so "positive on 400 of 400" is meaningful and not a
  seed artifact; the page needs only the conclusion above to make it usable.
- `--check`: `scripts/execute_notebook.py docs/examples/longitudinal-tmle.ipynb --check` prints
  "every cell reproduced its stored non-image output", exit 0
  (`.tmp/notebook-review/gate-N7/check.log`).
