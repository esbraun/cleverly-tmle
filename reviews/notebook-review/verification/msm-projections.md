# Verification: `docs/examples/msm-projections.ipynb` (findings MS-01..MS-07)

| item | value |
| --- | --- |
| commit | `5f33902b`, `cleverly.__file__` asserted under this worktree's `src` (script `v.py` line 22) |
| own script | `.tmp/notebook-review/verify-msm-projections/v.py` (independent of the reviewer's `sweep.py`; own seeds 2001-4150; includes an oracle multinomial g built from `multi_arm_dgp.arm_logits`) |
| logs | `fitted3000.log`, `oracle3000.log`, `oracle12000.log`, `fitted12000.log` in the same directory |
| seed 61 check | [executed] slope 0.268753, SE 0.008594, contrast miss -0.208388, Step 7 miss -0.209174, min fitted g 0.000587: matches the notebook and the reviewer |
| reviewer script | [read] `sweep.py` uses the notebook method, an independent truth, and correct coverage logic. No bug found |

## MS-01 — "differs from the Step 7 miss in the fourth decimal"

- verdict: PARTIAL
- severity: weak (reviewer: wrong)
- evidence:
  - [read] The notebook says (cell `failure-reading`): "Its estimate differs from the Step 7 miss in
    the fourth decimal". The printed values are -0.2084 and -0.2092 (outputs of `failure-mode`).
  - [executed] The difference is 0.000786. Its leading digit is in the fourth decimal place. "Differs
    in the fourth decimal" read as "the difference is of order 1e-4" is true on this draw. Read as
    "the first differing printed digit is the fourth", it is false: 0.2084 and 0.2092 differ in the
    third digit. A reader who compares the two printed numbers sees the second reading fail.
  - [recomputed] Over 300 own seeds at n = 3000: |difference| < 0.0005 in 50.7% (reviewer 52.2%),
    < 0.001 in 69.3%, median 0.00049, max 0.0049. Under either reading the sentence holds on only
    some draws. The sentence is not seed-tagged.
  - [read] The callback bound `0.0 < abs(...) < 0.005` (`tests/unit/tutorial_semantics/msm_projections.py:199-201`)
    accepts a third-decimal difference, as the reviewer says.
- fix: The reviewer's text is right: "differs from the Step 7 miss by 0.0008 on this draw, because the
  arm fit and the MSM fit target separately." Pin the callback to the printed magnitude (for example
  `abs(diff - 0.0008) < 5e-5`, or `< 0.001`). No study moves.

## MS-02 — interval coverage of the slope at n = 3000

- verdict: PARTIAL (numbers CONFIRMED, cause partly REFUTED)
- severity: misleading (by omission)
- evidence:
  - [read] The quoted sentences exist. Step 6 (`estimate-arms-heading`): "correctly specified
    parametric learners need no cross-fitting". Trust cell: "Correctly specified parametric learners
    are what makes that condition plausible on this page." Both are true as stated: they concern the
    data-reuse (Donsker) condition, and the shortfall is not caused by in-sample fitting. The page
    claims no nominal coverage in so many words. Step 6 reading says "That is one draw, not a coverage
    result." The trust cell still presents the data-reuse condition as the only remaining interval
    condition, so a reader infers near-nominal intervals.
  - [recomputed] 300 own seeds (2001-2300), n = 3000, the notebook method: slope coverage 0.920
    (MC SE 0.013), mean SE / empirical SD = 0.00896 / 0.00963 = 0.930, bias -0.0001. Reviewer: 0.925
    and 0.921 over 600 seeds. Reproduced.
  - [recomputed] Same seeds with the TRUE multinomial g (oracle learner, validated against the fitted
    g on seed 61, slope SE 0.00908 vs 0.00859): coverage 0.927, SE ratio 0.937. The shortfall stays
    with exact g. It is therefore not caused by fitted-g estimation error, by the multinomial
    learner, or by in-sample fitting. It is a property of the law: true arm probabilities reach about
    1e-4 for `medium` and `high` (reviewer's 2e6-draw minimum), so the influence curve is heavy-tailed
    and its in-sample variance understates the sampling variance at n = 3000.
  - [read] The reviewer's own arm-level numbers support this: SE/SD is 1.005 for `low` (true g at
    least 0.0099), 0.958 for `medium`, 0.944 for `high` (`sweep.log`).
  - [recomputed] Recovery with n: fitted, n = 12000, 150 seeds: coverage 0.960, SE ratio 1.09;
    oracle, n = 12000, 200 seeds: coverage 0.935 (MC SE 0.015), SE ratio 0.978. Consistent with the
    reviewer's 0.947 and 1.001. The shortfall is finite-sample.
  - [recomputed] Splitting own seeds at the median of the minimum fitted g gives 0.913 vs 0.927. That
    split is within noise, so per-draw min g does not predict a miss. The cause is the law's tail, not
    one draw's fitted extreme.
- fix: Add the measured sentence, but change the cause. The reviewer's "Small fitted probabilities of
  `high` and `medium` cause the shortfall" is wrong in its word "fitted": exact g gives the same
  shortfall. Suggested: "Over 600 draws of this law at n = 3000, the slope interval covered the
  projection 92.5% of the time, and 94.7% at n = 12000. Some patients have a true chance of `high` or
  `medium` near 0.0001, so the influence curve has heavy tails at this sample size. The same shortfall
  occurs with the true treatment mechanism." Commit the sweep script (or cite it) per the evidence
  rule; an uncommitted `.tmp` script is not citable evidence. Alternatively, choose a draw size or law
  with better overlap. No registered study moves.

## MS-03 — "That fit reports a standard error that is too small."

- verdict: CONFIRMED
- severity: weak (reviewer: misleading)
- evidence:
  - [read] Notebook `share-weight-reading`: "an estimated function declared as known still fits. That
    fit reports a standard error that is too small." Unconditional.
  - [read] `src/cleverly/msm.py:318-328` (`_ESTIMATED_WEIGHTS`): "The reported standard error can be
    too small, as the RM13 exact-law witness shows; its direction is not universal."
    `src/cleverly/msm.py:330-337` (`_UNDECLARED_WEIGHTS`): "its standard error can be wrong".
  - [read] Witness table, `docs/technical-reference/msm-projections.md:213-222`: ratios 0.742
    (`msm[W]`), 0.913 (intercept), 1.000 (`msm[a]`). One coefficient is not too small. The witness
    never shows a ratio above 1. Theory agrees with the library text: the omitted term D_h adds
    Var(D_h) + 2 Cov(IC_fixed, D_h), and the covariance can be negative, so the direction is not fixed.
  - Severity lowered: the page's instruction (do not pass an estimated weight) is correct whatever the
    direction, and the witnessed direction is "too small or equal".
- fix: The reviewer's text is acceptable. A shorter alternative: "That fit reports a standard error
  that omits the weight's influence-curve term. It can be too small or too large." Sibling:
  `tests/unit/test_msm_projection_weights.py:8` also says "the reported standard error is too small"
  unconditionally, which contradicts `msm.py:323`. No study moves.

## MS-04 — share-weight shift is smaller than one standard error

- verdict: CONFIRMED
- severity: weak
- evidence: [read] slope 0.2657 vs 0.2593 (shift 0.0064) against the fitted SE 0.008594; 0.2593 lies
  inside (0.2519, 0.2856). The prose "only the weight moved the slope" is correct. The step mixes
  population means with sample shares, which the page labels "only an illustration".
- fix: Reviewer's fix is right. Stating that the shift is below one standard error is the minimum.
  Adding the 1:10:1 contrast is optional; if added, the callback must pin its number.

## MS-05 — "one fluctuation equation and one influence-curve row per coefficient"

- verdict: CONFIRMED
- severity: weak
- evidence: [executed] stored output: one `msm fluctuation` row and two influence-curve rows.
  [read] `src/cleverly/validation/score.py:586-588` reports `fluctuation.score_norm` per group;
  `src/cleverly/estimators/targeting.py:351` builds one clever covariate `h_msm{j}` per term, so
  Step 7's "one score equation per coefficient" is right and Step 12's sentence can be parsed as
  contradicting it.
- fix: Reviewer's fix is right.

## MS-06 — trust cell does not name the study's oracle nuisances

- verdict: CONFIRMED (fact); PARTIAL (the reviewer's inference)
- severity: weak
- evidence: [read] `tests/studies/canonical_point_msm.py:180-190` (`OracleOutcomeContinuous`,
  `OracleTreatment`, `g_bounds=G_BOUNDS`), line 31 `G_BOUNDS = (0.01, 0.99)`;
  `docs/technical-reference/method-evidence/point-treatment-msm-projection.md:16`.
  [recomputed] The reviewer's link "which is where MS-02's shortfall arises" is wrong: exact g on this
  page's law gives the same shortfall (MS-02). The relevant difference is the law's overlap, not the
  nuisance construction.
- fix: Add "with exact nuisances and fixed bounds" to the study limits, and do not tie that limit to
  the coverage shortfall.

## MS-07 — callback comment says 1.23%

- verdict: CONFIRMED
- severity: weak
- evidence: [read] `tests/unit/tutorial_semantics/msm_projections.py:214` "1.23% of the units";
  stored output "truncated: 36 unit(s) (1.20%)".
- fix: change the comment to 1.20%.

## New findings

- [read] `tests/unit/test_msm_projection_weights.py:8` (module docstring) states "the reported
  standard error is too small" without qualification, against the library refusal text at
  `src/cleverly/msm.py:323`. Same defect as MS-03, in the test layer. Weak.
- No new material problem found in the notebook. The seed-61 numbers, the population projection,
  and the oracle check reproduce.
