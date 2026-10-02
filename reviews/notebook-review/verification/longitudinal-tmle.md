# Verification: `docs/examples/longitudinal-tmle.ipynb`

| item | value |
| --- | --- |
| verifier commit | `5f33902b` (branch `agent/notebook-review`) |
| interpreter | worktree `.venv`, `cleverly.__file__` under this worktree's `src` (printed in `probe.log`) |
| own scripts | `.tmp/notebook-review/verify-longitudinal-tmle/probe.py` -> `probe.log`; notebook dump `nb.txt` |
| compute | single-threaded, under 2 minutes |

Verdicts: CONFIRMED 7, PARTIAL 1 (LT-03, fix needs a correction), REFUTED 0.

## LT-01  "Logistic and linear learners of the law's own form"

- verdict: CONFIRMED. severity: wrong.
- [read] Notebook `trust` cell says exactly: "Logistic and linear learners of the law's own form are
  what makes that condition plausible on this page." The `estimate` cell passes
  `LogisticRegression` for outcome, treatment, and censoring and `LinearRegression()` as
  `pseudo_learner` on raw columns.
- [read] `src/cleverly/datasets/longitudinal.py:139` `_Y = {..., "kink": 0.5}` and `:164`
  `+ _Y["kink"] * np.tanh(l2)`; docstring `:154-155` "so a `glm` nuisance learner here is
  misspecified rather than accidentally exact". The four mechanisms (`:381-391`) are main-term
  logits in the columns the fitted designs contain, so they are of the fitted form.
- [recomputed] `probe.log`, 4e5 uncensored draws: an unpenalized main-term logistic regression of
  `Y` deviates from the true probability by up to 0.091 (RMS 0.015). Adding `tanh(L2)` recovers
  its coefficient as 0.474 (truth 0.5). The node-1 "always" pseudo-outcome regressed on `W`
  gains R2 0.457 -> 0.488 from quadratic terms, so its conditional mean is not linear in `W`.
- [read] The reviewer's second point is also right: `docs/technical-reference/cv-tmle.md:5-11,26`
  ties the data-reuse condition to the empirical-process (Donsker-type) term. Finite-dimensional
  GLM classes satisfy it whether or not they are correctly specified.
- fix: the reviewer's text is correct. Shorter alternative: "The learners are finite-dimensional
  regressions, which makes that condition plausible. The treatment and censoring learners match
  the law's form. The outcome regressions do not, because the law has a `tanh` term in engagement,
  so the estimate leans on the mechanisms." No study moves.

## LT-02  "a maximum-likelihood fit of the correct form reproduces those rows by construction"

- verdict: CONFIRMED. severity: misleading.
- [read] Notebook `reports-reading` quotes match. The library's own docstring already says the
  opposite of the page: `src/cleverly/validation/nuisance.py:28-30` "The slope measures spread, not
  specification: the population slope of any logistic maximum-likelihood limit with an intercept
  is 1, whether the model is correct or not."
- [recomputed] `probe.log`, seed-41 frame, node-2 outcome among the tracked: unpenalized in-sample
  recalibration slope is 1.0000 for the notebook form, for `Y ~ W1`, and for `Y ~ A2`. With
  sklearn's default `C=1` the slopes are 1.0011, 1.0009, 1.0029. Reviewer: 1.0009 / 0.9999.
  The score equations make slope 1 exact for any unpenalized logistic MLE with an intercept.
- fix: the reviewer's sentence is right. Note it covers the logistic rows only. The pseudo-outcome
  `regression_slope` (1.007-1.023) is for a least-squares fit, where an in-sample slope of 1 is
  also exact without penalty, so the same sentence can say "a logistic or least-squares fit with an
  intercept". No study moves.

## LT-03  cross-fitted study named as "the closest"

- verdict: PARTIAL (finding confirmed; part of the proposed fix is wrong). severity: misleading.
- [read] `tests/studies/canonical_ltmle.py`: `make_longitudinal(..., censoring=True)` (`:217`),
  `n_folds=1` and `"cross_fit": False` (`:154,226`), the same three plans including the `L2 > 0`
  rule (`:60-68`), `QuasiBinomialGLM` sequential regressions (`:222-223`, `q_formulas` `:160`), and
  `KnownLongitudinalMechanism` for both mechanisms (`:224-225`). The cross-fitted study
  (`cross-fitted-end-of-study-longitudinal-tmle.md:30-40`) is the same law and plans with five
  outer folds and a pooled fluctuation. The page fits in sample, so the ordinary study is closer.
  Its accuracy table (`ordinary-end-of-study-longitudinal-tmle.md:33-36`) gives coverage 0.9413
  and 0.9387, matching the reviewer. Both studies supply the mechanisms and have no clusters, so the
  page's description stays true after the swap.
- Correction to the fix: the `static__mechanism_correct` / `dynamic__mechanism_correct` cells do
  not run on the page's law. `tests/studies/ltmle_properties.py:15,146-154` draws from
  `tests/discrete_law_longitudinal.py` (all-binary finite-support law) with saturated
  `law.CellMeans()` versus `Dummy*` learners. They show double robustness on a different law, with
  the mechanisms estimated by cell means. Saying they "match this page best" overstates; if cited,
  the page must name the finite-support law.
- fix: link the ordinary end-of-study study as the closest (same law unclustered, same plans, in
  sample, mechanisms supplied, glm sequential regressions). Optionally add: "Its double-robustness
  cells, on a finite-support law, estimate the mechanisms and leave the outcome regression wrong."
  No study moves.

## LT-04  "which is clustered and evidenced"

- verdict: CONFIRMED. severity: misleading.
- [read] Notebook `estimate-heading` contains the quote (ipynb source line 556). The same phrase
  appears only in `src/cleverly/longitudinal/estimator.py:2555` and
  `docs/technical-reference/cv-tmle.md:287` (repo-wide grep, excluding `.venv`, `.tmp`, `reviews`).
- [read] No registered study fits a clustered longitudinal model. The longitudinal method-evidence
  pages are 11 documents (5 cross-fitted, 6 ordinary including the MSM projection); the only
  clustered study is `canonical_clustered_tmle.py` ("clustered point-treatment CV-TMLE", `:1,50`).
  `cluster_size` appears in no longitudinal study module. The ordinary study's limitation row
  (`ordinary-end-of-study-longitudinal-tmle.md:178`) and the cross-fitted one (`:226`) exclude
  clustering. `ltmle_crossfit_properties.py:288` passes `cluster=` to `make_folds` only, with no
  clustered draw. Clustered evidence is e2e only: `tests/e2e/test_ltmle.py:641-670` (clustered SE
  larger than iid, score equations pass) and `:1651-1664` (variance formula identity).
- [read] `docs/technical-reference/longitudinal-tmle.md:314` already uses the defensible wording:
  "The package permits the in-sample clustered fit."
- fix: agree. Library message: "Fit in sample (...), which reports a cluster-robust variance, or
  drop id= from fit." Update `cv-tmle.md:287` to the new message and refresh the prose ledger. No
  test matches the phrase. No study moves.

## LT-05  composite death rule versus censoring

- verdict: CONFIRMED. severity: weak.
- [read] `protocol` output names "death before day 30 as not top box (composite strategy)";
  `protocol-reading` row "loss to tracking ... is not an intercurrent event"; `docs/examples/index.md`
  row "unreturned surveys: the composite score counts as observed. Only living patients who do not
  respond are missing". `make_longitudinal` has no death node. The page never states the coding.
- fix: the reviewer's one sentence is right.

## LT-06  pooled day-seven shares

- verdict: CONFIRMED. severity: weak.
- [executed] seed 41, tracked at day 7: pooled 0.427 / 0.705 (matches page); within
  `navigation_discharge = 0` 0.381 / 0.593, within `= 1` 0.557 / 0.770. Reviewer's numbers match.
  The within-stratum gap (about 0.21) still supports "drives the second", so the conclusion
  survives; only the display confounds.
- fix: agree. Callback `tests/unit/tutorial_semantics/longitudinal_tmle.py:44-45` must change with it.

## LT-07  lowest `auc` attributed to node-2 censoring

- verdict: CONFIRMED (by reading; sweep not re-run). severity: weak.
- [read] Callback `:206-207` asserts `("censoring", 2)` is the minimum, a seed-41 relation. The
  stored values are 0.599 (node 1) and 0.557 (node 2), close enough that seed dependence is
  plausible; the node-2 tracking law `expit(2.4 + 0.2 L2)` is weak, so the explanation is right.
- fix: agree; weaken the callback to "both censoring rows have `auc` below 0.65".

## LT-08  stale "33.3" in the callback

- verdict: CONFIRMED. severity: weak.
- [read] Callback `:161` says "measured 33.3"; stored `support` output gives 35.239 and the summary
  "max weight 35.2". Comment only.
- fix: agree.

## New findings

- N1 (misleading, extends LT-03/LT-04). `trust`: "No registered study covers clustered fits with
  estimated mechanisms." The qualifier implies clustered fits with supplied mechanisms are covered.
  No registered study covers a clustered longitudinal fit of any kind (see LT-04 evidence). Fix:
  "No registered study covers a clustered longitudinal fit."
- N2 (note). The `LinearRegression` pseudo-learner is unbounded on a probability-scale target. Not
  shown to cause harm here (regression slopes 1.007-1.023, no truncation); no action unless a
  sibling page claims the pseudo-outcome stays in [0, 1].
