# Gate N6: `docs/examples/survey-nonresponse.ipynb` at 198c79a5

Reviewed every cell and stored output, the callback
`tests/unit/tutorial_semantics/survey_nonresponse.py`, the probe directory
`reviews/notebook-review/probes/survey-nonresponse-final/` (`sweep.py`, `sweep.csv`, `sweep.log`,
`summary.log`, `truth.py`, `truth.log`), ledger rows SN-01 to SN-11 and SN-N1/SN-N2, plan row N6,
`progress.md` (SN-N3 note), `docs/development/example-notebooks.md` (rule 174, the trust step),
`docs/examples/index.md` (row 18, shared design rows 42-45), gate N5 (death strategy wording),
`synthetic.py:1003-1052`, `missingness.py` (`tipping_gamma`, the tilt's reuse of the gamma = 0
targeted model), `tests/canonical/tmle_mar/properties.csv`,
`tests/canonical/tmle_mar_arm_indexed_cvtmle/properties.csv`,
`tests/studies/canonical_mar_arm_indexed_cvtmle.py`. Scratch: `.tmp/notebook-review/gate-N6/`
(not tracked). Light probes only; the Step 9 variance question belongs to the separate sweep.

## Verdict

PASS with two wording fixes. Every printed number reproduces, every repeated-draw number matches
`summary.log` and recomputes from `sweep.csv`, the quadrature truths agree with an independent
Monte Carlo, the protocol is coherent with the collaborative page, and no cell narrates a refusal.
The page states its own under-coverage honestly as a committed probe. One lead sentence in the
trust section overstates which intervals under-cover, and the Step 9 under-coverage has no named
cause yet (SN-N3, under investigation).

## REQUIRED FIXES

1. Trust section, second bold lead (cell 35): "The intervals of this page cover below the nominal
   level on its laws." The probe contradicts the generality: the Step 8 mild-law interval covered
   on 474 of 500 (0.948), and the complete-case interval covered its own respondent target on 470
   of 500 (0.940). Write "The Step 6 and Step 9 intervals cover below the nominal level on their
   laws." Markdown only.
2. Deferred to the SN-N3 result, not to this commit: rule 174 of `example-notebooks.md` asks for
   "the cause that a probe shows" beside a measured under-coverage. Step 6 names a symptom (SE
   0.93 of the spread). Step 9 names nothing; `sweep.csv` holds no Step 9 standard error, so the
   page cannot yet. When the investigation returns, add one sentence per fit naming the cause, and
   extend `sweep.py` to record the Step 9 standard errors if the cause is the SE scale. Until then
   the page is honest: it quotes 461 of 500 as a committed probe and says it is not a registered
   study.

## Advisory, not gating

- Step 4 reading (cell 13): under the hypothetical strategy the death condition also needs
  positivity for survival given the arm and the baseline variables. The collaborative page names
  "exchangeability and positivity for death"; this page names only the exchangeability half. The
  response-positivity assumption of Step 5 covers it in practice because the single indicator
  pools death and non-response, but one clause would align the two pages.
- `docs/examples/index.md:44`, "Only living patients can be missing or censored", is false for the
  two pages that use the hypothetical strategy. Shared with N5; the index row for this page
  (line 18) is correct.

## OK

- Protocol. The hypothetical strategy replaces the program's composite entry (the score has no
  worst value); the other two intercurrent-event entries stay; two rationale entries are added
  (response MAR; death carries no further information about the score had the patient not died).
  The callback pins the exact changed-field set, the entry order, and the fingerprint
  `31e911892577f4bc` on the protocol cell and the Step 6 fit. The death wording matches
  collaborative-tmle cell 13 (hypothetical strategy; the law draws no deaths).
- Composite formula (cell 13). `E[Y^a] = E_W[P(D=1|a,W) y_worst + P(D=0|a,W) E(Y|a,W,D=0,R=1)]`
  is the correct g-formula for a composite score under no unmeasured confounding of `A` and
  `Y ⊥ R | A, W, D=0` (MAR among survivors). The page says this fit has one response indicator
  and does not compute it, and the trust table repeats the scope. Coherent with
  `index.md:42-45`, which state the composite rule and the MAR-among-survivors need.
- Hypothetical-strategy MAR claim (cell 13). "Missing at random for both causes" is right:
  `P(D=0, R=1 | A, W, Y*) = P(D=0 | A, W) P(R=1 | A, W, D=0)` when death and response each carry
  no information about `Y*`, so the single pooled indicator is MAR.
- Complete-case target (cell 22). "Standardized to the respondents' covariate distribution",
  1.428 in the population and 1.409 on this draw; 1.409 equals `1.2 - 0.9 * (-0.232)` by the
  callback. Independent Monte Carlo on 4 x 10^6 rows of the library law: 1.42842 (MC SE 0.00045)
  against the quadrature 1.427936; a 400,000-row draw of `make_missing_outcome` gives 1.42818.
  `truth.py`'s equations match `synthetic.py:1029-1044` term by term. Population ATE 1.2004 and
  response rate 0.7518 also agree.
- Repeated-draw numbers against `summary.log`, recomputed from `sweep.csv` (500 rows): 460,
  461/461/461, 474, 470, +0.0111, SE/SD 0.9346 (page: 0.93), tipping gamma 5-95% 0.9421-1.4534,
  score units 5.38-7.61, `use_ci` mean 1.0097. `sweep.log` names this worktree's `src`.
- Tipping readings (cells 32-34). `tipping_gamma` docstring: the nearest tilt at which the
  estimate (or, with `use_ci=True`, the interval) crosses the null. The largest logit move of a
  mean under shift gamma is `2 expit(gamma/2) - 1` at mean `expit(gamma/2)`: 0.315 at 0.658,
  times the fitted range 20.05 gives 6.32; the `use_ci` value 1.130 gives 5.52; the SD 1.66
  gives 3.8 and 3.3. The callback pins the maximum against a grid and witnesses that the
  mid-range formula would fail. The sample-scale sentence and the `q_bounds` remedy are present.
  "Away from gamma = 0 each row is a plug-in on the targeted outcome model of gamma = 0" matches
  `missingness.py` (`repeat.fluctuations[...].targeted` reused under `_tilted`). Positive gamma
  with `arm_gamma {1: -1}` lowers the navigation-arm mean, and the stored curve falls
  monotonically with the same SE on every row.
- Stored outputs quoted correctly: 0.756, 977, 1.200, 0.763/-0.232, 0.747/0.764, 1.768,
  3023, [0.009532, 0.9905], [0.01, 1], [-6.56, 13.49], 1.189/0.068/(1.056, 1.322), 1.467/
  (1.354, 1.581), 1.409, 1.0, 1.223/(1.152, 1.294), 0.797, 0.1918/1.5139/2.1817, 0.376/0.563,
  0.126/0.138/0.037, 0.1826-0.8240, 0.0351, 0.0075, 0.768, 0.5056, 90.2 (1/90.2 = 0.0111 >
  0.0075, as the reading says), 1.306, 1.130, 0.315, 0.658, 6.32, 5.52, 1.66, 3.8, 3.3,
  2.6619/2.0928/1.1891/0.2387/-0.4125, 0.0679.
- Trust rows. `mar_robustness/mechanisms_correct`: bias CI -0.003853 to 0.002294, margin
  0.010316, n 2000, positive, mechanisms supplied correct and the outcome wrong
  (`mar_tmle_properties.py:73-79`). `missingness_necessity/ate__complete_case_control`:
  -0.122939 to -0.117198, margin 0.009634, control. `interval_calibration/l1_ate__learned_nuisances`:
  coverage 0.936086-0.961718, SE ratio 0.941324-1.019810, n 2000, `N_FOLDS = 10`, depth-five
  trees (`canonical_mar_arm_indexed_cvtmle.py:40, 224-235`). Each row names law, construction,
  outcome type, fold layer, and role, as the trust step requires. The probe is cited as "a review
  probe" with its committed script and "not a registered study". R5 is closed: no study
  calibration is transferred to the page's own fits.
- SN-10. Zero matches for "refus" in the notebook. Cell 31 reads the `unavailable` rows as "not
  implemented for a fit with a response mechanism" and points at `to_frame()`; the callback pins
  that every omitted-variable row and `evalue` is `unavailable` and that `missingness` and
  `tipping_gamma` returned. Step 10 (PAF) is gone; Steps 11 and 12 are now 10 and 11, and the
  learning table points at them.
- SN-01 to SN-09, SN-N1, SN-N2 delivered as the ledger specifies: no "standardized" score, no
  "worst transition score" in the summary, contact attempts framed as an estimator limit beside
  death, the one-SE distance called sampling variation with the 0.011 mean shift, the plug-in
  sentence, the `use_ci` value printed.
- SN-N3 honesty. The page states Step 9 coverage as 461 of 500 from a committed probe and makes
  no calibration claim for it; the callback asserts only this-draw coverage. My recompute from
  `sweep.csv`: Step 9 `ate` bias +0.00013, SD 0.0202, so the under-coverage is not bias, which is
  consistent with the editor's finding. The page claims nothing it does not have.
- `--check`: `scripts/execute_notebook.py docs/examples/survey-nonresponse.ipynb --check` prints
  "every cell reproduced its stored non-image output", exit 0 (`.tmp/notebook-review/gate-N6/check.log`).
