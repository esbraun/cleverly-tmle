# Verification: `docs/examples/interventions.ipynb` (IV-01..IV-13)

- Worktree commit `5f33902b`, `.venv` Python 3.11. `vsweep.py` asserts that `cleverly.__file__` lies under this worktree's `src`.
- Own scripts are in `.tmp/notebook-review/verify-interventions/`: `vsweep.py`, `vsum.py`, `v_inc.csv`, and `v_sh.csv`. I wrote them independently of the reviewer's `sweep.py`.
  - Seeds **101-160** (60 seeds), which do not overlap the reviewer's seeds 1-60. n = 3000. The sweep uses the notebook learners and settings, with the learner and Runtime seeds equal to the data seed. Every process ran single-threaded.
  - Truths come from the library's own quadrature (`law.incremental_truth`, the `make_shift_dose` truth dict), not from the reviewer's Monte Carlo. Incremental truth 0.02338. Shift truths 0.7761, 0.8125, 1.7500, and gap 0.0364.
  - Sanity check: seed 31 gives psi 0.025800 (notebook 0.026, z 2.6). Seed 32 gives 0.7880, 0.8248, and 1.8002, which match the stored outputs.
- I read the reviewer's `sweep.py` and found no bug that would move its numbers. Its hard-coded truths agree with the library truths to 4 decimals.

## Summary table

| ID | verdict | severity |
| --- | --- | --- |
| IV-01 | CONFIRMED (fix needs a correction) | wrong |
| IV-02 | CONFIRMED | misleading |
| IV-03 | CONFIRMED | misleading |
| IV-04 | CONFIRMED | wrong |
| IV-05 | CONFIRMED | misleading (borderline weak) |
| IV-06 | CONFIRMED | weak |
| IV-07 | CONFIRMED | weak |
| IV-08 | CONFIRMED | weak |
| IV-09 | CONFIRMED (read, numbers not re-run) | weak |
| IV-10 | PARTIAL | weak |
| IV-11 | CONFIRMED (two docstrings, not one) | weak |
| IV-12 | CONFIRMED | weak |
| IV-13 | PARTIAL | weak (borderline none) |

## IV-01: the incremental miss is systematic
- verdict: **CONFIRMED**. severity: **wrong**.
- [read] Cell `incremental-fit-reading` says: "A 95% interval misses on about one draw in 20, so one miss is consistent with sampling variation." It also says: "One draw cannot show whether the miss comes from g. Try a better-calibrated treatment learner before you report this axis." The quotes are exact.
- [executed] Own sweep results, seeds 101-160, three treatment learners. Only the treatment learner changes. The outcome learner, folds, and bounds are the notebook's.

  | treatment learner | bias (MC se) | emp. SD | mean SE | SE ratio | coverage | mean z | z > 0 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | notebook `HistGradientBoostingClassifier` | +0.00176 (0.00013) | 0.00099 | 0.00089 | 0.89 | **27/60 = 0.45** | +1.98 | 57/60 |
  | `StandardScaler` + degree-2 `PolynomialFeatures` + `LogisticRegression` | +0.00012 (0.00011) | 0.00084 | 0.00089 | 1.06 | **59/60 = 0.98** | +0.12 | 37/60 |
  | `CalibratedClassifierCV(HGB, method="sigmoid", cv=3)` | +0.00138 (0.00012) | 0.00094 | 0.00089 | 0.94 | **38/60 = 0.63** | +1.56 | 56/60 |

  The reviewer reported 0.48 for the notebook learner and 0.93 for the polynomial logistic learner. My figures are 0.45 and 0.98. The `nuisance_models` warning fires on 60/60 seeds.
- Conclusion: the bias is about 2 SE and has the same sign on 57/60 seeds. It is not sampling variation. Swapping **only** the treatment learner for the near-correctly specified degree-2 logistic model removes the bias. So the answer to the focus question is yes.
- Correction to the proposed fix: option (a) lists "a regularized or `CalibratedClassifierCV` booster" as equivalent to degree-2 logistic. It is not. Sigmoid recalibration of the booster leaves 80% of the bias, and coverage reaches only 0.63. The calibration slope that the warning reports is a symptom. Recalibrating to that slope does not repair g's error in the direction the IPSI functional needs. Recommend the degree-2 logistic learner, or another learner that has been checked with a seed sweep. Do not recommend recalibration.
- Under option (b), the notebook should state the probe result: "with the degree-2 logistic g the bias falls to about 0.0001". The notebook should not imply that any better-calibrated learner fixes the miss. The sentence "Try a better-calibrated treatment learner" is itself a slightly misleading pointer, by this evidence.
- The callback assertions `not covers(tilt, ...)` and `2.55 <= z < 2.65` (`tests/unit/tutorial_semantics/interventions.py:183-184`) must change with either option. No registered study moves.
- The coordinator note about the house rule at `docs/development/example-notebooks.md:161` is correct. The rule makes "sampling variation" the default explanation.

## IV-02: the shift-axis intervals are miscalibrated
- verdict: **CONFIRMED**. severity: **misleading**.
- [read] The quotes exist. `dose-fit-reading` says "One draw is not a coverage result either way". `failure-reading` says "On this draw, the interval contains the population gap and excludes zero". `shift-support-reading` already concedes "the influence-curve standard error can itself be too small". The page hedges, but it attributes the +1.0 miss to in-sample fitting and strain on a draw that sits *above* the truth.
- [executed] Own sweep results, seeds 101-160, notebook shift configuration:

  | contrast | bias (MC se) | emp. SD | mean SE | SE ratio | coverage (mine) | coverage (reviewer) |
  | --- | --- | --- | --- | --- | --- | --- |
  | +0.5 capped at 5 | +0.0098 (0.0017) | 0.0129 | 0.0118 | 0.92 | 51/60 = 0.85 | 0.83 |
  | +0.5 uncapped | +0.0035 (0.0019) | 0.0145 | 0.0116 | 0.80 | 53/60 = 0.88 | 0.93 |
  | +1.0 uncapped | **-0.0244** (0.0040) | 0.0312 | 0.0207 | 0.66 | **39/60 = 0.65** | 0.60 |
  | gap uncapped - capped | -0.0063 (0.0009) | 0.0070 | 0.0047 | 0.68 | **34/60 = 0.57** | 0.60 |

- The reviewer's numbers reproduce on disjoint seeds. The +1.0 bias is significantly negative (about 6 MC se), while the notebook draw sits +0.050 above the truth. The capped +0.5 bias is significantly positive (about 6 MC se). The gap interval undercovers badly.
- I did not re-run the reviewer's correct-Q and cross-fit probes. I read their design and found it sound. Their attribution, that density error dominates and in-sample fitting explains only part, therefore rests on their evidence alone.
- Fix: endorse the reviewer's options. The cheapest option that keeps the lesson is a parametric (quadratic) Q, which the generator docstring says represents the law exactly (`synthetic.py:1820-1826`), together with more density bins. Re-sweep before the page reads any interval as calibrated. At minimum, the gap reading should present the gap as a point contrast and state the measured undercoverage. The callback relations at lines 132-142 and 160 change. No registered study moves.

## IV-03: the trust table omits that the studies supply g or the density exactly
- verdict: **CONFIRMED**. severity: **misleading**.
- [read] `deterministic-point-treatment-regimes.md` Limits: "the treatment mechanism is supplied exactly".
- [read] `incremental-propensity-interventions.md` Limits: "The treatment mechanism is supplied exactly to `cleverly`".
- [read] The MTP study supplies the density through an oracle: `tests/studies/canonical_shift_policies.py:54-55`, `class OracleShiftDensity`, "Exact pooled-hazard probabilities for the study's conditional normal dose". It uses `PRIMARY_DENSITY_BINS = 320` (line 51). This is stronger evidence than the page wording the reviewer cites.
- No registered study therefore covers an estimated g or an estimated density. That gap is exactly where IV-01 and IV-02 fail.
- Fix: correct as proposed. Add "treatment mechanism (density) supplied exactly" to each row.

## IV-04: the outcome is called "standardized"
- verdict: **CONFIRMED**. severity: **wrong**.
- [read] The notebook says "This law records a standardized score, which takes its scale from the data" (`dose-identify-heading`). The protocol says `outcome="Patient-reported transition score, standardized to the baseline distribution"`. `dose-fit-heading` says "a standardized score with no known finite support".
- [read] `src/cleverly/datasets/synthetic.py:1828-1844` gives Y = 1 + 0.5a + 0.25a^2 + W1 - 0.5W2 + 0.2W3 + N(0, 0.8^2). The notebook itself prints `ey_shift[current practice] 3.3950` and Y values up to 5.417. Nothing is standardized, and nothing takes its scale from the data.
- Additional nuance: "a cross-fitted fit could declare no `q_bounds`" is too strong. A cross-fitted fit can declare a wide, conservative range, and the reviewer's own probe ran with `q_bounds=(-30, 40)`. The correct reason is that Gaussian noise has no finite support, so any declared bound is a convention rather than a measurement fact.
- Fix: correct as proposed. Also soften the "could declare no `q_bounds`" sentence. The protocol fingerprint changes, and the callback is otherwise unaffected.

## IV-05: the intensity units contradict each other
- verdict: **CONFIRMED**. severity: **misleading**. It is borderline weak: synthetic data, with the negative share disclosed.
- [read] `dose-data-reading` says "synthetic units with no physical zero, so 5.1% of the values are below zero". The protocol says "Intensity records in navigator contact units support consistency", and its versions say "Navigator contacts at ...".
- [recomputed] Marginally, A ~ N(2, 0.49 + 0.09 + 1 = 1.58 variance). P(A < 0) = Phi(-2/1.257) = 0.056.
- Fix: endorse the notebook-only fix (an "intensity index"). Changing `make_shift_dose` moves the MTP study law, so avoid it.

## IV-06: the capped shift skips the support warning
- verdict: **CONFIRMED**. severity: **weak**.
- [read] `src/cleverly/interventions/shift.py:403-406`: `_warn_outside_support` returns immediately when `shift.cap is not None`. Any cap therefore silences the check, including a cap above max(A). The answer to the focus question is yes.
- [read] The module docstring at lines 16-19 says "the cap is what makes the parameter well defined without a positivity assumption nobody can check: Qbar is never evaluated outside the range the data covers". That holds only if u <= max observed A, or more precisely u lies inside the conditional support. The library does not check it.
- The notebook statement "The capped policy raises no warning" is literally true. The page never claims the absence of a warning is a check, but it places the sentence next to warnings that are such a check.
- Fix: endorse the notebook sentence and the library wording change. The optional warning when cap > max(A) is a reasonable library follow-up that needs a failing-first test. The stored `dose-identify` output changes only if the identification text changes.

## IV-07: the ESS gap attributed to "positivity strain"
- verdict: **CONFIRMED**. severity: **weak**.
- [read] For the normal location shift, the true ratio has E[r^2] = exp(delta^2), so the Kish fraction is exp(-delta^2). The page computes this itself.
- The gap between that fraction and the estimated-ratio ESS is the excess dispersion of the estimated density ratio, an estimation effect. The population strain is the exp(-delta^2) term itself. The proposed rewording is correct.

## IV-08: "about twice as wide"
- verdict: **CONFIRMED**. severity: **weak**.
- [executed] SE ratio (+1.0)/(+0.5 uncapped) over my 60 seeds: mean 1.77, range 1.58-2.27, above 1.8 on 15/60 = 25%. The reviewer reported mean 1.77, range 1.55-2.12, and 28%.
- The callback's `> 1.8` (lines 132-133) pins a minority outcome. Fix as proposed: quote the printed widths.

## IV-09: the regime intervals are too wide
- verdict: **CONFIRMED** on reading. I did not re-run the regime fits. severity: **weak**.
- [read] The reviewer's `summary.log` shows SE ratios of 1.80 and 1.46 and coverage of 60/60. Their degree-2 logistic run brings coverage to 0.93 and 0.90.
- The text claim "The regime estimate is double robust" is about consistency and is not false. Folding this into the IV-01 learner change is the right fix.

## IV-10: the sensitivity step yields no usable outcome-side information
- verdict: **PARTIAL**. severity: **weak**.
- [read] The facts are right. The reading also states the limitation candidly: "This grid therefore shows one treatment-side movement and no outcome-side movement."
- A tutorial may legitimately show a refusal mode, so dropping the step is a judgement call rather than a correction. A one-cell treatment-only grid plus one sentence about the outcome axis would cost 2 fewer refits. The robustness-value refusal is sound.

## IV-11: the `ShiftDGP` docstring states a different policy
- verdict: **CONFIRMED**, with a wider scope than the finding states. severity: **weak** (library documentation).
- [read] `src/cleverly/datasets/synthetic.py:1712`, class docstring: "for d(a, w) = min(a + delta, u)".
- [read] `synthetic.py`, `shifted()` docstring line: "``d(a, w) = min(a + delta, cap)``, holding a unit at its own dose past the cap". This line contradicts itself.
- [read] The code: `np.where(moved > cap, dose, moved)`. It holds the unit at its own dose and does not truncate at the cap.
- [read] `shift.py:7-12` documents the hold-at-own-dose form correctly.
- Fix: correct **both** docstrings. The fix is documentation only and moves no study.

## IV-12: the implemented incremental policy needs an estimated g
- verdict: **CONFIRMED**. severity: **weak**. One sentence in the protocol reading suffices.

## IV-13: "budget navigator hours"
- verdict: **PARTIAL**. severity: **weak**, borderline none.
- The point is real: the hours scale with the number offered, about 1521/3000. However, "budget" in a program office loosely covers benefit per cost. The proposed rewording is clearer and harmless.

## New findings
1. **The IV-01 fix as written could introduce a new error.** `CalibratedClassifierCV` recalibration of the booster does not fix the incremental bias: coverage 0.63, bias +0.0014 [executed]. Any notebook text such as "a better-calibrated treatment learner fixes this" would be wrong. The existing sentence "Try a better-calibrated treatment learner" already points readers in a direction that does not work by itself. Name the probe that works: a correctly specified logistic g.
2. **The `shifted()` docstring contradicts itself** (see IV-11). The reviewer cited only the class docstring.
3. **IV-04 nuance**: the claim that a cross-fitted fit "could declare no `q_bounds`" is too strong. Wide declared bounds run, and the reviewer's probe used them. Their 3.6x SE inflation under wide bounds deserves the separate library look the reviewer suggested. I did not reproduce it.
