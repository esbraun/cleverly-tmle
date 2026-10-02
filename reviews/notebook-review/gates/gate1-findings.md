# Review gate 1: findings

Commit `5f33902b`. Inputs: `ledger.md`, `sibling-sweep.md`, `refusal-inventory.md`, the per-notebook
`findings/` and `verification/` files, the notebooks (full read of point-treatment-tmle,
survey-nonresponse, longitudinal-survival), `src/cleverly/datasets/{synthetic,navigation,longitudinal}.py`,
`src/cleverly/sensitivity/{omitted_variable,evalue}.py`, the tutorial callbacks, and the orchestrator's
learner experiment under `.tmp/notebook-review/learner-exp/`. Scratch: `.tmp/notebook-review/gate1/`
(`ptn1.py` recomputes PT-N1 and the squeezed-law moments; `pt.txt`, `sn.txt`, `ls.txt` are cell dumps).

Verdict vocabulary: CONFIRMED (real, fix right), FIX-WRONG (real, fix needs the stated change),
NOT-REAL (with evidence). Evidence tags: [recomputed] by this gate, [executed] by a reviewer or by
the orchestrator's experiment and read here, [read] source or stored output.

## 1. Orchestrator decision 1: PT-N1 (E[1/g] infinite on the `navigation_data` law)

**CONFIRMED** [recomputed]. `navigation_data` wraps `make_nonlinear_bounded`
(`navigation.py:108`), whose propensity is `nonlinear_dgp().propensity` (`synthetic.py:696`), the
logit `0.6 W1 - 0.4 W2^2 + 0.5 W2 W3 + 0.3 1(W4 > 0)` (`synthetic.py:608`). With W ~ N(0, I):

- `1/g = 1 + exp(-logit)`. Integrating W1, W3, W4 out gives `E[exp(-logit) | W2] = c exp(0.525 W2^2)`
  with `c = exp(0.18) (1 + exp(-0.3)) / 2`. Against the N(0, 1) density the integrand is
  `exp(0.025 W2^2)`, so the integral diverges.
- Partial integrals over `|W2| <= 4, 8, 16, 32` (`ptn1.py`): 4.83, 13.8, 696, 7.0e10. These match
  the ledger's figures.
- `E[1/(1 - g)]` is finite (opposite sign in the exponent), so the tail sits in the offer arm.

Two consequences the ledger does not state:

1. The divergence is slow (`exp(0.025 W2^2)`), so at n = 3,000 the truncated estimator is well
   behaved. The orchestrator's 153-seed run on the **unsqueezed** law (`learner-exp/summary.csv`,
   config `point`) gives, with the true g (`e_logit_true`): bias -0.0001, SE/SD 1.02, coverage
   0.94, truncation on every seed. With regularised HGB or logistic g: SE/SD 1.04 to 1.05,
   coverage 0.92 to 0.95. The infinite bound is an asymptotic fact about the untruncated target.
   It does not make the pages' ATE intervals wrong.
2. The same run refutes the refusal inventory's 20-seed flag that a calibrated propensity "halves
   the interval width" to SE/SD 0.64 to 0.71 [executed, read]. Over 152 to 153 seeds the calibrated
   learners give SE/SD 1.04 to 1.12 and coverage 0.94 to 0.95. The PT and CF verifiers were right.
   Drop the inventory's "outcome-side problem" hypothesis from the plan.

What the infinite `nu_0^2` does break is the omitted-variable bound (P18): Lemma 3 and Theorem 4 of
CCNSS need a finite second moment, and every "true nu^2" quoted from Monte Carlo (7.77, 8.22, and
the 7.75 pinned in `dr_tmle.py:144`) is a sample statistic of a divergent expectation. That is the
real reason to change the law.

## 2. Orchestrator decision 2 (revised): squeeze the propensity inside `nonlinear_dgp` and regenerate

Judged in the revised form: `g = 0.05 + 0.90 expit(logit)` goes into `nonlinear_dgp` itself, so
`nonlinear_bounded_dgp`, `make_nonlinear_ate`, `make_nonlinear_bounded`, and `navigation_data`
all change, and `cvtmle_properties`, `bounded_cv_laws`, `canonical_properties`, and
`ctmle_oat_properties` are regenerated.

**Sound, and the right shape.** It removes most traps of the earlier "change `navigation_data`
only" form: the trust cells keep "same law", `test_datasets.py:531-539` (row-for-row equality with
`make_nonlinear_bounded`) still holds, and the three direct uses of `nonlinear_bounded_dgp()`
outside the studies (`interventions.ipynb:184` for `screen_truth` and `incremental_truth`,
`dr_tmle.py:142`, `point_treatment_tmle.py:31`) follow automatically. Checks:

| item | finding | evidence |
| --- | --- | --- |
| no registered study calls `navigation_data` | confirmed; `grep navigation_data tests/studies` is empty. The four studies import the DGPs | [read] |
| ATE truth unchanged | confirmed; depends on the outcome law and the W marginal only. `strong_truth.log`: 0.162858 before and after | [executed] |
| squeezed-law moments | `E[1/g] = 2.75`, `E[1/(1-g)] = 2.08`, `nu_0^2 = 4.83`, `P(A=1) = 0.460` (was 0.456); `E[1/g] <= 20` holds trivially | [recomputed], [executed] |
| "a GLM is misspecified for g" stays true | the mixture `0.05 + 0.9 expit(eta)` is logistic in no polynomial, so the law's story (flexible learners earn their cost) survives. Record the squeeze and the strong-positivity range in the `nonlinear_dgp` docstring | reasoning |

Traps that survive the revised form and must be in the plan:

| trap | detail | evidence |
| --- | --- | --- |
| **ATT and ATC truths change** | 0.179 -> 0.177 and 0.149 -> 0.151 (`strong_truth.log`). PT `data`, `data-reading`, and `association` print and narrate them. The cb pins (`att > ate + 0.01`, unadjusted closer to ATT) still hold with a 0.0145 margin | [executed] |
| **the IPSI estimand changes, and its learner fix is unverified on the new law** | the incremental truth carries g, and the IPSI remainder is not doubly robust in g (Kennedy 2019). The degree-2 logistic that repaired IV-01 was swept on the old law and is misspecified on the new one. Re-sweep the incremental and regime axes (at least 60 seeds) before adopting any learner there | reasoning, [read] |
| **truncation lessons go flat** | with g >= 0.05 the default bound 0.0114 never bites a calibrated fit (`strong_truth.log`: min fitted g 0.07 to 0.08; the bare HGB still truncates 3 rows). PT Step 9's truncation curve, PT-08, DR-09 ("0.43% of the population below 0.0147" becomes 0%), and the P6 and P16 support readings need a new lesson, not only the PT-N1 sentence. One option: keep the bare HGB as the Step 9 diagnostic fit and show the calibrated refit beside it | [executed], [read] |
| **study verdicts can move, so declare the reporting policy first** | `cvtmle_properties` scores an in-sample tree control (coverage 0.4875 against 0.9325, PT-07 and CF-N1). Bounding g away from 0 shrinks the overfit tree's weight explosion, so the control gap can narrow and the CF trust-cell lesson can weaken. Decide before the run how a changed or red cell is reported; do not tune margins afterwards | [read] |
| **pinned numbers outside the notebooks** | `make_nonlinear_*` is used by `tests/conftest.py`, `tests/e2e/test_backends_and_api.py`, `test_ipsi.py`, `test_variants.py`, `tests/unit/test_fold_policy_rules.py`, `test_split_plan.py`, `test_parallel_invariance.py`, `test_intervention_load_diagnostics.py`, and the doctest at `estimators/tmle.py:64-72`. Every pinned estimate there moves; truths with `ate` keys do not. Run the fast suite and the doctests before regenerating anything | [read] |
| **method-evidence pages quote the old runs** | the technical-reference pages for the four studies print bias, coverage, and SE ratios. Regenerate, then rewrite those numbers; the fast suite recomputes verdicts from artifacts. `tests/canonical/provenance-revisions.md` is not the route, because this change is result-determining | [read] `CLAUDE.md` |
| regeneration order | the four studies and the notebooks share the generator; regenerate once, after every other generator edit (including the MSM one below) is final | memory rule |
| the learner experiment | `summary.csv` is the old law (152 to 153 seeds). `rows_strong.csv` does not exist; `run.log` stops at 2300/3200; the strong-law rows carry `diag:KeyError:'cal_slope'`, so no calibration slope was recorded. Fix the column lookup and finish or discard the run before citing a 300-seed result | [read] |
| learner presentation | present the chosen g as the analyst's flexible parametric choice (PT-02 verifier). Never write "correctly specified" for a polynomial logistic (section 5, item 1) | reasoning |

**MS-02 generator fix (MSM near-zero cadence probabilities).** Sound in the same way: the shortfall
(slope coverage 0.92 at n = 3,000, the same with the oracle g) is a positivity tail of the law, so
the law is the right place to fix it. The law lives in `make_multi_arm` (`synthetic.py:1671`), and
`tests/studies/multi_arm_common.py` is the only study user [read], so the MSM studies built on it
regenerate. Traps: every MS truth (arm means, slope, the 1:10:1 contrast in MS-04) and every stored
MS number moves; the MS-07 comment (1.20% truncated) and the `G_BOUNDS = (0.01, 0.99)` study
setting lose their bite; re-sweep the slope coverage on the new law before writing the trust cell,
and declare the reporting policy before the run. Check that no MS lesson depends on the rare arm.

## 3. Orchestrator decision 3: sensitivity replacements

| page | replacement | verdict | notes |
| --- | --- | --- | --- |
| PT, CF | default doubly robust bound on a calibrated refit | CONFIRMED, conditional on section 2 | finite `nu_0^2 = 4.83` puts the pages inside Lemma 3 and Theorem 4; one-sided limits become publishable |
| DR-TMLE | E-value | CONFIRMED with wording | the continuous path standardises by the weighted SD of the observed outcome (`evalue.py:546-557`), the marginal SD, which includes the between-arm shift. The chain Chinn `d = log(OR)/1.81` then `RR ~ sqrt(OR)` equals VanderWeele and Ding's `exp(0.91 d)` (`_SMD_TO_LOG_RR = 1.81/2`). Label it approximate. Say it reads the estimate and interval only, so it does not test the doubted g, which is the page's premise |
| interventions | treatment-only stress surface | CONFIRMED | qualitative; the reading must not call it a bound |
| C-TMLE | point and selection; plain TMLE interval; `simulated_confounding` | CONFIRMED | `methods.py:798` docstring fix (CT-10) in the same change |
| LT, LS | leave-one-covariate-out refits as a benchmark | CONFIRMED | say each move is signed and specific to the dropped covariate, and that a refit without a confounder is a biased estimator of the same estimand. Dropping `engagement_day7` removes a node-2 history variable from the treatment mechanism and the outcome regression together, which is the right analogue of omission |
| MSM | second known design coded by cadence step | CONFIRMED | two working fits, two questions |
| survey | delete the PAF step | CONFIRMED | renumber; the `title` learn row goes |
| TWINS | sensitivity robust across pair samples | CONFIRMED | "stable configuration" means a propensity library whose cross-fitted g stays away from 0 on a seed sweep, or the full 71,345 pairs (TW-06). Not a `try` block |

## 4. Material ledger rows and sweep patterns

Rows marked wrong or misleading, plus weak rows whose fix I would change.

| ID | verdict | evidence and notes |
| --- | --- | --- |
| PT-01 | CONFIRMED | [read] `omitted_variable.py:1196-1202`: multiplier `abs(rho) sqrt(cf_y cf_d / (1 - cf_d))`, 2.85 at 0.9/0.9. `max_bias` is a scale |
| PT-02 | CONFIRMED | [executed] 153 seeds: bare HGB SE/SD 1.83 to 1.86, coverage 1.00, calibration slope 0.47; regularised HGB SE/SD 1.04, coverage 0.94. Quote these, not the 20-seed figures |
| PT-08, PT-N1 | CONFIRMED; fix changes under section 2 | [recomputed] |
| PT-03, PT-04, PT-05 | CONFIRMED (verifier wording) | [read] the gain form is the paper's; keep the clip disclosure if the benchmark stays |
| CF-01, CF-N1 | CONFIRMED | [executed, read] "0.27" becomes "about 0.7"; lead with the coverage gap, which the regenerated study must re-supply |
| CT-01, CT-N1, SN-03, IV-04 (P2) | CONFIRMED | [read] `navigation.py:177` base outcome is a share of the maximum; the synthetic score is Gaussian |
| CT-03, CT-04, CT-05 | CONFIRMED | [read] systematic plug-in understatement; qualitative constant-Q wording |
| DR-01 | CONFIRMED | verifier executed `n_folds=2` pooled |
| DR-02 | CONFIRMED (verifier wording) | no coverage advantage shown; a registered study adds, does not move |
| DR-04, DR-06 | CONFIRMED | [read] Theorem 1 covers the non-cross-fitted estimator under Donsker conditions |
| DR-07 | CONFIRMED | superseded by the E-value. The `7.75` witness in `dr_tmle.py:138-146` is a sample mean of a divergent expectation and must not be called "true"; it changes under the squeeze anyway |
| IV-01, IV-09 | **FIX-WRONG (wording)** | the ledger and P1 call the degree-2 logistic "correctly specified". The logit has `0.3 1(W4 > 0)`, which no polynomial represents; the IV verifier wrote "near-correctly specified" (`verification/interventions.md:40`). Write "a flexible parametric g that carries the law's quadratic and interaction terms". The empirical repair (coverage 0.93 to 0.98) stands on the old law; re-sweep on the new one (section 2) |
| IV-02 | CONFIRMED | state the measured under-coverage, or repair the density and re-sweep |
| IV-03, IV-05 | CONFIRMED | [read] `OracleShiftDensity`; `P(A < 0) = 0.056` |
| SN-01 | CONFIRMED | derivation: death forces `R = 1` and `Y = worst`, so `R` depends on `Y` given `(A, W)` unless `P(death | A, W)` is 0 or 1. `E[Y | a, W, R = 1]` then over-weights deaths by the survivors' non-response rate. The fix (two-part formula, "assumes no deaths", `index.md:42-43`) is right |
| SN-04 | CONFIRMED | a post-assignment variable in the response model needs the sequential-regression form; the page states an estimator limit as a scientific rule |
| SN-02 | CONFIRMED (verifier) | the tipping arithmetic itself is right [recomputed]: with `arm_gamma[1] = -1` the largest move is `2 expit(gamma/2) - 1 = 0.315`, at fitted mean `expit(gamma/2) = 0.658` |
| SN-N1 | CONFIRMED | resolves with P2 |
| LT-01, LT-02, LT-03, LT-04, LT-N1 | CONFIRMED | [read]; finite dimension, not correctness, is the Donsker argument (P15) |
| MS-02 | CONFIRMED (verifier cause) | heavy tails from true probabilities near 1e-4; oracle g shows the same. Generator fix judged in section 2 |
| TW-01, TW-02, TW-03, TW-04 | CONFIRMED | [read] stored tables are fixed point; `final_score == 0.0` |
| LS-01 | CONFIRMED (weak) | adding censoring is sound; the recoding `K_k = C_k (1 - D_k)` is well defined because death and readmission are exclusive at a node (`longitudinal.py:834, 845`) |
| LS-02, LS-NF2 | CONFIRMED | label the row as the death-as-censoring functional; `index.md:20` too |
| LS-03, LS-04 | CONFIRMED | [read] the LS treatment and censoring logits are linear in history (`longitudinal.py:627-647`, `826-840`), so "the page models the mechanisms correctly" is true; the study supplies them |
| P1 | FIX-WRONG (wording) | as IV-01: "correctly specified (degree-2 logistic)" in the summary row and the recommendation |
| P2, P3, P4, P5, P7, P8, P12, P13, P15 | CONFIRMED | |
| P6, P16 | CONFIRMED; shape changes | under section 2 the narration becomes "strong positivity by construction; the support report still describes only the fitted model" |
| P18 | CONFIRMED | resolved by the squeeze |

NOT-REAL: none of the material rows. The refuted sub-claims table in `ledger.md` is right as it
stands.

## 5. Items the reviewers missed

1. "Correctly specified degree-2 logistic" is false on the old law (step term in W4) and further
   off on the new one. Section 4, IV-01 and P1.
2. The refusal inventory's cross-cutting flag (calibrated g gives SE/SD 0.64 to 0.71, "an
   outcome-side problem") is refuted by the orchestrator's 153-seed run. Remove it from the plan's
   premises.
3. The IPSI truth and learner fix depend on g; the interventions page is the one sibling where the
   squeeze changes the estimand itself. Section 2.
4. PT Step 9's truncation curve and DR-09 lose their subject under the squeeze; the plan needs a
   replacement lesson, not a sentence.
5. `cvtmle_properties`' in-sample control is the evidence behind the CF trust cell. The squeeze can
   narrow it. Declare the reporting policy before regenerating.
6. `dr_tmle.py:144` pins "true nu^2 = 7.75" by a sample mean; rename the comment and variable with
   the squeeze.
7. `run2.py` reads `cal_slope` from a wrong column, so no calibration slopes exist for the new law.
8. DR E-value: the standardising SD is the marginal observed-outcome SD. Say "approximate" and name
   the chain.
9. Pinned numbers in eight test modules and one doctest move with the generator (section 2 table).
10. The spot-check of point-treatment, survey-nonresponse, and longitudinal-survival found no further
    wrong scientific statement. Checked and sound: the PT clever-covariate sentence, the default
    bound `5 / (sqrt(n) log n) = 0.0114`, the calibration-slope direction, the SN tipping
    arithmetic, the SN OR-versus-RR reading, the LS survival-scale bound swap, and the LS claim that
    the mechanisms are correctly specified.

## 6. What the orchestrator must fix before planning

1. Plan the squeeze as a generator change with its full cost: docstring, ATT/ATC numbers,
   the IV re-sweep, a new Step 9 lesson, pinned tests and the doctest, the four method-evidence
   pages, and a declared reporting policy for the regenerated verdicts. Regenerate once, last.
2. Treat the MSM generator fix the same way: re-sweep slope coverage on the new law and declare the
   reporting policy before regenerating.
3. Replace every "correctly specified" claim about a polynomial logistic g with "flexible
   parametric".
4. Strike the inventory's "calibrated g halves the interval" flag; cite the 153-seed run.
5. Finish or discard the strong-law learner run before citing a 300-seed result; fix its
   `cal_slope` lookup first.
