# Gate F: whole branch `agent/notebook-review` at 255e0e76 against `origin/main` (5f33902b)

Scope: 38 commits, 387 files. Read `plan.md` (pre-registration and "What the run found"),
`progress.md`, `ledger.md`, every gate under `gates/`, every commit message, the full `src` diff,
the shared docs (`docs/examples/index.md`, `docs/development/example-notebooks.md`), the `trust`
cell of all ten notebooks, and the interventions and TWINS notebooks cell by cell with their stored
outputs. Light checks only; the fast suite (13512 passed), docs build, ruff and mypy already passed
at HEAD. No tracked file was edited by this gate.

## Verdict

PASS with two required wording fixes, both markdown only. No library change, no re-execution, and
no study movement is needed. Nothing a hostile expert would correct remains on the two pages read
in full.

## OK

| area | finding |
| --- | --- |
| library diff | every `src` hunk maps to a gated change. S1: `synthetic.py` (`nonlinear_logit`, `g = 0.05 + 0.90 expit(u)`, docstrings), `navigation.py`, `datasets/__init__.py`. S2: `omitted_variable.py` (bias scale, `cf_d`, RV line, benchmark gain form), `validation/drtmle.py` (`cross_fitted` contract line, tolerance comment), `ctmle.py`, `longitudinal/estimator.py`, `methods.py`, `targets/builtin.py`, `population_intervention.py`, `shift.py::_warn_outside_support` (capped-above-max warning). S6: `shift.py` (`mean_ratio`, `fold_mean_ratio`, `ShiftSet.ratio` docstring), `assessment.py` (`folds=` pass-through). The S6 trim (cd2a1452) is reverted bit for bit (19e91585). No numeric estimator path changed outside fix A |
| study regeneration | followed the pre-registration: four runs serial in the declared order, outputs outside the repo, `--skip-reference` where declared, smoke run first, LF artifacts, manifests now plain, `run.log` of the superseded repair deleted, `seed-repair-plan.json` kept. Byte-identical witness held (primary artifacts unchanged in all four). Only `crossfit_overfitting` and `robustness_contract` moved; no verdict changed; `red_cells` updated 0 blocks, so the declared stop did not trigger. S2 waited until S3 ended. The declared section of `plan.md` is unedited; results are appended below it |
| no refusal showcase | grep of all ten notebooks for `refus`, `CapabilityError`, `unavailable`, `not implemented`, `not supported` finds only the `to_frame()` filter code (`~status.isin(["unavailable", "not_applicable"])`) and the TWINS sentence that the protocol "does not support two of" the assumptions, which is about identification, not a capability. `docs/roadmap.md:1522` no longer cites refused cells |
| trust rows (R5) | all ten `trust` cells name law (same or other), nuisance construction, outcome type, fold layer, and role for each cited cell, and quote registered coverage from `properties.csv` or `summary.csv`. TWINS gives the five items in prose. Every repeated-draw number names a committed probe under `reviews/notebook-review/probes/` and says it is not a registered study |
| death strategy | composite on point-treatment, cross-fitting, DR-TMLE, interventions, longitudinal (top box), time-to-event; hypothetical on collaborative, survey, MSM, each with the exchangeability and positivity conditions stated. `index.md` rows 42-46 match page by page. The N5/N6/N9 advisory on `index.md:44` is resolved (the row now distinguishes the two strategies) |
| sensitivity usage | each page runs the analysis the `example-notebooks.md` table names for its kind of fit, with no `try` and no plug-in limits: RV/benchmark/bounds (PT, CF, TWINS), E-value (DR), `simulated_confounding` plus bound on the design-based plain fit (CT), arm-contrast RVs (MSM), tilt and tipping with `use_ci=True` printed (SN), covariate-drop benchmark (LT, LS), treatment-only surface (IV) |
| learners | g: shallow regularized booster (PT, CF, TWINS Super Learner member) or degree-2 logistic (IV regime and incremental), each chosen by a committed sweep and described as regularized or flexible parametric, never "correctly specified" for the bounded law. No bare `HistGradientBoostingClassifier` below 10,000 rows remains; the five remaining bare `HistGradientBoostingRegressor` calls are outcome learners, which the rule permits |
| ledger coverage | every wrong and misleading row is closed by a commit whose message names it, or superseded by refusal removal as the ledger records. The refuted sub-claims (7.77/8.22 nu^2, `CalibratedClassifierCV`, bound-only TWINS fix, constant-Q numbers, "parametric rate suffices", "small fitted probabilities") appear nowhere on the pages. See the two surviving weak-row parts under required fixes |
| shared docs | `index.md` program table matches each page's failure mode, including the N3 rewording (row 16) and the N5 workflow (row 15). `example-notebooks.md` sensitivity and trust tables match the pages after 84d26a6b. Narration harness (S4) checks scientific notation; TWINS no longer quotes machine-noise residuals |
| interventions, read in full | every quoted number is in a stored output or sourced to a probe. Checked by hand: `1/0.0114 = 87.7` ("about 88"); Kish fraction `exp(-delta^2)` for a unit-variance normal shift (0.779, 0.368); mean density ratio reference 1 for both capped-inside-support and normal-support uncapped shifts; `shift_dgp` outcome mean `1 + 0.5a + c a^2 + linear W` is degree 2, so the "correctly specified" quadratic Q claim holds; IPSI weights in `[1/2, 2]` for delta 2; the non-double-robustness statement matches Kennedy (2019); the regime "min g 2.664e-06" row carries zero weight in the offer-to-all equation because its observed arm is 0 |
| TWINS, read in full | g bound `5/(sqrt(n) log n) = 0.004859` at n = 12000 and Q bound `[0.0005, 0.9995]` (`alpha = 0.9995` default) match the package; E-value 26.41 and 17.17 recompute from RR 13.4607 and 8.8426; log-scale half widths equal; stress test uses all 14,870 discordant pairs of the full file with a count-exact bootstrap; citations are the published NeurIPS and QJE entries, also in `references.md`; Section 11 states both sweep counts (55/60 truth, 56/60 naive) and scopes the coverage to the fixed baseline sample |

## REQUIRED FIXES

1. Sibling pattern P3 left on two pages. `docs/examples/interventions.ipynb` cell `data-reading`
   ("The baseline covariates are standardized (mean 0, SD 1)") and
   `docs/examples/longitudinal-survival.ipynb` data-reading table ("standardized baseline
   covariates (mean 0, SD 1)") contradict `index.md` ("The laws do not standardize a sample") and
   the wording of the other eight pages. Both generators draw `rng.standard_normal(n)`
   (`synthetic.py`, `longitudinal.py:623-624`). Write "independent standard-normal draws" on both.
   Markdown only; no stamp or callback change.
2. Ledger rows SN-02 and SN-09, technical-reference parts, unaddressed. The branch changed
   `validation-methods.md` only in the omitted-variable text. The "Missingness tilt and tipping
   gamma" section (line 1625) still does not state (a) that gamma is read on the logit scale of
   the fit's outcome scaling, so `q_bounds` fixes the tilt scale (SN-02, verifier wording), and
   (b) that away from gamma = 0 the curve is a plug-in that reuses the gamma = 0 targeted
   `Q*` (`missingness.py:452`), so its accuracy depends on the outcome model (SN-09). Two
   sentences in that section, then `python -m tests.prose --update` for the ledger.

## Not required, recorded

- `example-notebooks.md:9` calls TWINS "the other published notebook" (pre-existing on `main`;
  ten notebooks are published) and `:152` says the reference notebook takes 12 seconds (now 15 s).
- `progress.md` is modified in the working tree at the time of this gate (the F row); commit it
  with the gate.
- Gate S2's unverified "Appendix E.5, Remark 8" locator was resolved by 753c9992, which pins the
  arXiv v6 numbering. `preprint-and-published-section-numbers-differ` still applies if the
  published locator is ever cited.
- `CorrectionCheck.cross_fitted` defaults to `False`; library callers always set it. A required
  keyword would remove the one way a hand-built check prints "is Theorem 1's estimator" for a
  cross-fitted fit (gate S2 note, still open, not blocking).
