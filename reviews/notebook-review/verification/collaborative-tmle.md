# Verification: `docs/examples/collaborative-tmle.ipynb` (findings CT-01..CT-10)

| item | value |
| --- | --- |
| commit | `5f33902b`, `cleverly.__file__` asserted under this worktree's `src` |
| own sweep | `.tmp/notebook-review/verify-collaborative-tmle/vsweep.py`, seeds 1000-1399 (400 draws, disjoint from the reviewer's 0-299), n = 2,000, notebook configuration (in-sample, greedy, selection_folds 3, inner 2, `Runtime(random_state=44)`), linear Q and constant Q, plain and C-TMLE, plus my own OLS/HC0. Raw `v_1000_1400.csv`, summary `vanalyze.log` |
| reviewer scripts | read `sweep.py`, `probe_crossfit.py`, `analyze.log`; no bug found in the sweep logic. One heavy-tail issue in the constant-Q numbers (see CT-05, CT-07) |

Verdict counts: CONFIRMED 7, PARTIAL 3, REFUTED 0.

## CT-01 — CONFIRMED (core), PARTIAL (one sub-claim). Severity: misleading

- [read] The notebook says it: `protocol-reading` "A standardized score takes its scale from the data, so the analyst can declare no finite support for it"; `estimate-heading` "This page's score is standardized, so it has no known finite support to declare ... Fitting in sample is therefore the honest choice rather than a workaround"; `trust` "The standardized score has no support to declare, so no fit here can hold rows out" (the reviewer did not list the trust cell; it needs the same fix).
- [read] The real reason is the synthetic law: `src/cleverly/datasets/synthetic.py:902-917`, Y mean = 1 + A + 1.5 W1 + 0.8 W3 plus Gaussian noise, so it has no finite range. The e2e docstring states the honest reason: "`instrument_dgp`'s Gaussian outcome has no `q_bounds` a cross-fitted fit could declare" (`tests/e2e/test_ctmle.py:737-738`).
- An affine standardization of a bounded raw instrument stays bounded and its bounds are computable from the published range and the baseline mean/SD (none of which come from held-out outcome rows). So "standardized, therefore no finite support" is a false reason for the applied setting the page names.
- [executed] `CrossFitting(n_folds=5)` with `q_bounds=None` is refused (`CapabilityError ... needs a declared q_bounds`); the same C-TMLE fit with `Targeting(q_bounds=(-20, 20))` is accepted (psi 0.955). So the alternative the fix names exists.
- Sub-claim weakened: "the generated data are not standardized (mean 1.50, SD 2.13)" does not by itself contradict "standardized to the baseline distribution", which scales a follow-up score by baseline moments and need not give mean 0 / SD 1 at follow-up. `data-reading` also says the score "is in synthetic units". I would drop that argument; the support argument stands alone.
- Fix: the reviewer's wording is right in substance ("this synthetic score has no documented range ... a raw patient-reported score usually has one; declare `q_bounds` and cross-fit"). Additions: (1) also fix the `trust` cell sentence; (2) update the callback comment at `tests/unit/tutorial_semantics/collaborative_tmle.py:49-50`, which quotes the sentence (assertions unaffected); (3) changing the protocol `outcome` text changes the printed fingerprint `eb50c4bbf2250f48`, which `protocol-reading` narrates, so the notebook must be re-executed and the narrated fingerprint updated; (4) the same reason appears in siblings (see New findings). No study moves.

## CT-02 — PARTIAL. Severity: weak (reviewer: misleading)

- [read] The quoted text is in `selection-reading`, `failure-reading`, `control-heading`. But the page already flags fragility: "The first two candidates are nearly tied ... A choice between near-tied candidates is fragile", and the plan table says the empty g "is a legitimate candidate". "Step 7 showed the selector stop at ..." is a past-tense statement about the shown draw and is literally true. The page does not claim more than one draw shows; it omits the size of the instability.
- [executed] My 400 seeds vs reviewer's 300: empty g 59.2% vs 55.3%; `social_support` alone 26.2% vs 25.0%; contains `baseline_readiness` 13.5% vs 18.0%; contains `queue_lottery_draw` 1.2% vs 2.7%; |cv gap k1-k0| < 1e-3 on 61.8% vs 64.7%. The reviewer's numbers reproduce in shape.
- Fix: adding a repeated-draw sentence is a real improvement. Use rounded numbers that both sweeps support, e.g. "Over several hundred draws of this law, the selected g was empty on more than half, `social_support` alone on about a quarter, contained the confounder on about one in seven, and contained the draw on a few percent." Do not print 55/25/18/3 as if precise. "On this draw" opener for Step 9 is fine.

## CT-03 — CONFIRMED. Severity: misleading (notebook), weak (technical entry, docstring)

- [read] `control-reading`: "In this known law, the search keeps the variable that the constant outcome model needs. It drops the pure assignment predictor." The previous paragraph says "On this draw", but this sentence is phrased as a property of the law.
- [executed] Constant Q, 400 seeds: `baseline_readiness` kept 100.0% (reviewer 99.7%), `queue_lottery_draw` included 7.0% (reviewer 8.3%), `social_support` kept 98.8% (reviewer 99.0%).
- [read] `docs/technical-reference/collaborative-tmle.md:154` and `src/cleverly/estimators/ctmle.py:222-224` say "in every seed ... still leaves the instrument out". The test behind it uses `SEEDS = (0, 1, 2)`, n = 1,500 (`tests/e2e/test_ctmle.py:741-742`), and the test itself only asserts `included <= 1` for W2 (`test_it_still_leaves_the_instrument_out`, ~line 835). So "in every seed" overstates even what the test pins.
- Fix: correct. Wording for the docs: "on the three e2e seeds (n = 1,500) the search includes the confounder; the test allows the instrument on at most one". For the notebook, "kept the confounder on essentially every draw and left the draw out on about 92-93%". Technical entry edit needs the prose ledger refresh. No study moves.

## CT-04 — CONFIRMED. Severity: misleading

- [read] Texts present: `failure-reading` "can understate the spread of the estimator"; `sensitivity-reading` "On this draw, the plug-in value is smaller than that robust value"; `trust` "The plug-in diagnostic can understate the spread."
- [executed] Linear Q, 400 seeds: C-TMLE plug-in mean SE 0.0451 vs empirical SD 0.0517 (ratio 0.871; reviewer 0.847), plug-in interval coverage 0.917 (reviewer 0.893); plug-in < HC0 on 99.8% (reviewer 98.7%); HC0 against the C-TMLE estimate ratio 1.046, coverage 0.960 (reviewer 1.014, 0.950); |OLS - C-TMLE| < 5e-4 on 86.2% (reviewer 81.7%).
- The understatement is systematic in this configuration. Fix is right; replace "89%" with "about 90%" (my sweep gives 92%). The mechanistic clause ("a g that barely moves ignores the adjustment the outcome regression performs") is acceptable.

## CT-05 — CONFIRMED (direction), PARTIAL (numbers). Severity: misleading

- [read] `control-reading`: "the trust section's limit applies to this ratio. A ratio below one is not a precision gain here".
- [executed] Constant Q, 400 seeds: C-TMLE plug-in mean SE 0.0924 vs empirical SD 0.0582 (ratio 1.59, coverage 0.993); plain empirical SD 0.208, RMSE 0.235 vs C-TMLE RMSE 0.059. The plug-in overstates here, so the "understate" limit does not describe this ratio, and the repeated-sampling gain (SD ratio 0.28) is larger than the printed 0.394. Both points reproduce.
- The constant-Q C-TMLE estimate is heavy-tailed. The reviewer's SD 0.0757 / ratio 1.22 is driven by one seed with |error| 0.724 and by draws that include the instrument: excluding those, their SD is 0.064, mine 0.044. Numbers differ by ~30% between two several-hundred-seed sweeps.
- [read] In the registered study's own `selector_necessity` collaborative cell (bounded twin, cross-fitted), `se_ratio` is 0.80 (`tests/canonical/ctmle_selector/properties.csv` row 13), i.e. understatement. So "the direction depends on the configuration" is well supported.
- Fix: keep the reviewer's structure but do not print 0.076/0.202/1.22. Say: "Over several hundred draws of this law, the selected fit's spread was well under half the plain fit's, and in this configuration the plug-in value overstated the spread. In the registered study's cross-fitted bounded twin it understated it. The direction depends on the configuration."

## CT-06 — CONFIRMED. Severity: weak

- [executed] Linear Q, 400 seeds: empirical SD plain 0.0683 vs C-TMLE 0.0517; RMSE 0.0685 vs 0.0519 (reviewer 0.068 vs 0.053). "Removing only the draw gives the same reduction" rests on the reviewer's `mechanism.log`; I did not reproduce it. Fix is fine; round to "about 0.07 against about 0.05".

## CT-07 — CONFIRMED. Severity: weak

- [read] `tests/studies/ctmle_selector_properties.py:94-119` (`instrument_dgp()`, `DummyRegressor`, n 1500, `RATE_REPLICATES`), `:119` `bounded_twin`, `:138-160` (`cross_fit=True`, `n_folds=5`, `q_bounds=bounded_cv_laws.Q_BOUNDS`); `properties.csv` rows 13-14 (n 1500, 800 replicates, collaborative standardized bias 0.217, bias CI upper 0.00371 vs margin 0.00301; empty_control bias 0.0503; RMSE ratio 0.241). "Repeats this control" overstates sameness. The trust table already says "on its own laws" and that no study covers this score, which is why weak is right.
- Fix: the main sentence is right. Do NOT add the optional "+0.016 bias measured here": my sweep gives +0.0075 (MCSE 0.0029); the reviewer's value is driven by the heavy tail (see CT-05). No study moves.

## CT-08 — CONFIRMED. Severity: weak

- [executed] Constant Q, plain TMLE, 400 seeds: bias +0.111 (MCSE 0.010; reviewer +0.104), SE ratio 1.06 (reviewer 1.105), coverage 0.865 (reviewer 0.887). Cause not isolated by me either. Fix is fine ("about +0.1 over several hundred draws").

## CT-09 — CONFIRMED. Severity: weak

- [read] `tests/unit/tutorial_semantics/collaborative_tmle.py:120` quotes "0.843"; stored output and `failure-reading` say 0.844. Fix right.

## CT-10 — CONFIRMED. Severity: weak

- [read] `src/cleverly/methods.py:764-767` "No strategy reports a confidence interval, a p-value or a standard error ... `oat` the reason that F19 holds" against `:798-799` "For an interval, use `strategy=\"oat\"` or :class:`TMLEMethod`". Technical entry line 232 "The outcome-adaptive path publishes no inference either." Fix ("For an interval, use :class:`TMLEMethod`.") is right.

## New findings

1. Sibling notebooks carry the CT-01 reason. `docs/examples/msm-projections.ipynb` (source lines ~387 "the score is standardized to the baseline distribution, so it has no fixed maximum" and ~497 "standardized score, which has no such support, so `cleverly` refuses a cross-fitted fit of it") and `docs/examples/survey-nonresponse.ipynb` (~491 "Step 4 recorded a standardized score, which has no such support"). Severity: misleading, same fix. [read]
2. Number drift between the e2e test and the docs it backs. `tests/e2e/test_ctmle.py:734-735` says mean absolute error "0.036 for the collaborative fit against 0.695"; `src/cleverly/estimators/ctmle.py:224-225` and `docs/technical-reference/collaborative-tmle.md:155-156` say 0.017 against 0.696. I did not rerun to see which is current. Severity: weak. [read]
3. CT-03 addendum: the test's own assertion (`included <= 1`) shows its author expected the instrument could enter, so "in every seed ... leaves the instrument out" is unsupported by the test, not just by the seed count. [read]
4. Constant-Q C-TMLE estimates are heavy-tailed on this law (a single draw with error 0.72 in 300; draws that include the instrument have bias about +0.10 to +0.13). Any repeated-draw number quoted for the constant-Q configuration should be rounded and qualitative, or come from a larger sweep. [executed]
