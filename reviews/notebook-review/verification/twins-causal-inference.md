# Verification: `docs/examples/twins-causal-inference.ipynb` (findings TW-01..TW-12)

- Commit `5f33902b`, worktree venv Python 3.11.13; `cleverly.__file__` asserted under this worktree's `src`
  (`common.py` asserts it on every run).
- Scratch: `.tmp/notebook-review/verify-twins-causal-inference/` (`common.py` is an independent rebuild of
  sections 2, 6, 7 from the cached pinned CSVs; `sweep_gap.py`, `two_cov.py`, `sens.py`, `sens_g2.py`; logs and
  `sweep_gap_2026.csv`). All runs single-threaded, `n_jobs=1`.
- Verdicts: 5 CONFIRMED, 7 PARTIAL, 0 REFUTED. The most important correction is in TW-01: the symptom is real,
  but the mechanism the reviewer gives is wrong. The proposed fix would not repair the assertion.

## TW-01: hand-vs-package agreement holds only for the seed shown

verdict: **PARTIAL** (symptom confirmed; mechanism and fix refuted). severity: **misleading**

- [read] The notebook asserts `abs(package_gap) < 0.1 * abs(targeting_shift)` (cell `production-fit`). Its prose
  says the two constructions "agree well within that move".
- [recomputed] `sweep_gap.py`, pair-sample seeds 0..29 (learner seeds follow): the assertion fails at
  **11/30** (seeds 0, 1, 3, 10, 13, 14, 18, 20, 22, 24, 29). Ratio 4.97 at seed 1, 0.92 at seed 14, 0.80 at seed 10,
  0.76 at seed 24. The reviewer reports the same 11/30 and the same ratios. Seed 2026 reproduces the stored gap
  -1.130e-06 and shift 5.114e-04.
- [recomputed] The reviewer says "the whole gap is therefore the bound". **That is wrong.**
  - Four failing seeds (3, 10, 20, 29) have no in-sample propensity outside [0.01, 0.99], so the 0.01 clip and
    the package bound give identical `g`.
  - When the hand-built TMLE uses the package bound 5/(sqrt(n) log n), the assertion still fails in **12/31**
    (seed 1: ratio 4.55 rather than 4.97). The fix as proposed does not repair it.
- [recomputed] `two_cov.py`: the remaining gap comes from the targeting construction. The package targets each
  arm mean (A/g and (1-A)/(1-g)); the hand version uses one ATE clever covariate. A hand-built two-covariate
  logistic fluctuation at the package bound reproduces the package fit to within 1.1e-08 at seed 1 (gap was
  1.5e-03) and to 1e-6 or better at seeds 0, 3, 10, 24 and 2026. Seed 14 still differs by 1.5e-04, probably
  because of a package stopping rule; I did not investigate it.
- The ratio criterion is also ill-conditioned. The denominator, the targeting move, can be close to zero: 1.2e-05
  at seed 3 and 5.1e-06 at seed 10. A tiny absolute gap then fails the check.
- Fix recommendation: make the hand-built step match the package construction. Use the package bound and two
  arm-specific clever covariates; the hand result then serves as a real witness at about 1e-6. Alternatively,
  keep one covariate but state that one- and two-covariate TMLE differ at finite n. In that case compare
  `|gap|` to the standard error, not to the targeting move (max |gap|/SE over 31 seeds = 0.39, at seed 1).
  Update the prose in `production-heading` and `production-fit-reading` to match. `check_stored()` would need its
  0.1 relation changed. No registered study is affected.

## TW-02: sensitivity section raises on some samples

verdict: **CONFIRMED** (cause attribution PARTIAL). severity: **misleading**

- [executed] `sens.py 11`: the assessment records `omitted_confounding` as `unavailable`, because the
  doubly robust nu^2 is -10.0006. `assessment.report("omitted_confounding")` raises
  `KeyError: "assessment operation 'omitted_confounding' did not run"`. `benchmark(...)` raises `CapabilityError`.
- [executed] `sens.py 5`: the assessment completes (RV 0.2661). `benchmark(...)` raises `CapabilityError` with
  nu^2 = -10.3091 in the short refit. The reviewer's numbers match exactly.
- [executed] `sens_g2.py`: the cause at seed 11 is a treated row with a **cross-fitted** Super Learner propensity
  of 0.00129, so the Riesz weight is about 777. The cause is not literally the in-sample cells of TW-01, though
  rare categorical cells are a plausible common source. At seed 5 the full fit's minimum cross-fitted g is
  0.0176, and the failure is only in the benchmark refit. I did not re-measure the 2/30 rate.
- Fix: the reviewer's prose fix is right. Attribute the refusal to extreme cross-fitted propensities (a large
  Riesz representer), not to the hand-built model's in-sample cells. Coarsening rare race and region levels is a
  reasonable optional change, but it moves every stored output. No registered study is affected.

## TW-03: trust row misdescribes the clustered CV-TMLE study

verdict: **CONFIRMED**. severity: **wrong**

- [read] The notebook's `trust` cell says the study uses "a continuous outcome, a linear outcome regression, and
  the exact propensity". It says the uncovered part is "the binomial outcome and the Super Learner used here,
  which no registered study covers".
- [read] The study page `docs/technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md` says
  otherwise:
  - :3-4: "binary-outcome clustered law with ten observations per cluster", 200 clusters.
  - :9: "the Gaussian clustered law this row replaces". Commit `93800e86` is "Regenerate clustered CV-TMLE on a
    declared binary law", so the notebook row describes the superseded law.
  - :21-23: five grouped folds, exact propensity, "unpenalized logistic regression, `C=1e6`".
  - Limitations :168-178: one cluster size (ten); "Exact treatment mechanism"; "Main-effects outcome learners ...
    does not establish flexible learner-library parity"; "Pointwise identity-scale intervals ... does not cover
    ... ratios".
- (The reviewer cites `validation-grid.md:31`. That file does not exist at this path, but the claim stands on the
  study page.)
- Fix: the reviewer's rewrite is correct. Also add "risk-ratio (log-scale) intervals" to the uncovered list,
  because section 10 fits a `RiskRatio` and the study's limitation row excludes ratios. Notebook only.

## TW-04: quoted decimals not in stored output; scientific-notation flip

verdict: **CONFIRMED**. severity: **misleading**

- [read] The notebook's `targeting-reading` says "5.0875e-04 ... 2.9606e-19 ... 4.2286e-03". The stored
  `manual-estimators` output prints `0.0005`, `0.0000`, `0.0042`.
- [recomputed] With pandas 3.0.5 and `display.precision=4`, the ladder column prints in fixed notation only when
  `final_score == 0.0` exactly. Any nonzero value below 1e-4 (tested: 2.9606e-19, 5.9212e-19, 3e-5) switches the
  **whole** column to scientific notation, including the estimates (`5.6500e-02`). So the stored run had
  `final_score == 0.0`, and 2.9606e-19 came from some other execution.
- [recomputed] At seed 2026 in this venv, `final_score` = 5.921189e-19, which matches `check.log`. Over 31 seeds,
  `final_score` is exactly 0.0 in 7. Nonzero values are small multiples of 1.48e-19 (about 2^-49/n), which is
  summation round-off. The value depends on the BLAS and the numpy reduction order.
- [executed] `_DECIMAL` (`tests/unit/tutorial_semantics/__init__.py:253`) returns only `0.0565` from a test
  sentence containing `5.0875e-04`, `2.9606e-19` and `4.2286e-03`, so the harness gap is real. Note for the harness
  fix: the tolerance `0.5 * 10**-places` uses mantissa places. A scientific literal needs a relative tolerance on
  its mantissa, so the extension is not just a regex change.
- Robust print (recommended):
  - Split the table into estimates (fixed `.4f`) and targeting quantities.
  - Print epsilon and the initial score with `f"{x:.4e}"`. Their magnitudes (4.2e-03, 5.1e-04) are stable.
  - Print the post-targeting score as a pass/fail against the asserted bound, for example
    `print(f"|score after targeting| < 1e-10: {abs(final_score) < 1e-10}")`. Do not print the raw value.
  - In the prose, write "below 1e-10" instead of a machine-noise value.

## TW-05: platform noise in stored outputs

verdict: **PARTIAL**. severity: **weak**

- [read] The `check.log` diffs are confirmed: `ltmle-diagnostics` solver scores at 1e-18, and the reproducibility
  Python line (3.13.7 stored, 3.11.13 in this worktree's venv).
- The Python line is not a defect. The cell exists to record the interpreter. The reading says only that a rerun
  "with the same versions" reproduces the outputs, and 3.11 is not the same version. It does mean that `--check`
  can never pass on a different interpreter. Keep the line. The checker or the maintainer should treat that diff
  as expected.
- The 1e-18 solver scores are real noise. Print `passed` and `|score| / threshold` rounded, or omit the raw score.
  The section 8 score-equation table (3.791e-13) did not move in this run, but it has the same exposure.

## TW-06: stress-test interval excludes zero only for this sample

verdict: **CONFIRMED** (by reading the reviewer's script and log; no new sweep). severity: **weak**

- [read] `data_check.py` draws 6,000-pair samples from the full file and reuses the notebook's paired bootstrap.
  I found no bug.
- The figure does show [0.0008, 0.0171], clear of zero. The prose makes no significance claim.
- Fix: compute the stress test on all 71,345 pairs (it needs no model). That option is better than a caveat.

## TW-07: eligibility omits the equal-weight exclusion

verdict: **CONFIRMED** (read). severity: **weak**

- [read] The notebook's eligibility is "Both twins have a recorded birth weight and first-year outcome". The
  reviewer's README quote and the 0 equal-weight pairs are consistent with twin 0 always being lighter.
- Fix as proposed. The protocol fingerprint changes, so the fingerprint strings in the readings and the
  `check_stored()` regex flow change with it.

## TW-08: "this ordinary-TMLE value" names the wrong fit

verdict: **CONFIRMED** (read). severity: **weak**

- [read] `sensitivity-reading` says "Interpret this ordinary-TMLE value". The value comes from
  `flexible_result` (the Super Learner CV-TMLE). Section 7 uses "ordinary package TMLE" for the
  non-cross-fitted fit.
- The fix is right.

## TW-09: cf_d paraphrased in its partially linear special case

verdict: **PARTIAL** (real, minor). severity: **weak**

- [read] `sensitivity-heading` says "cf_d is the share of the residual treatment variation". The library
  docstring describes the gain in the Riesz representer.
- The reviewer's replacement sentence is acceptable. Its second sentence ("would sharpen the treatment
  prediction") is informal; the first sentence is enough. Leave the library RV text to a separate change.

## TW-10: "about four times" and "cf_y rounds to 0" are sample-specific

verdict: **PARTIAL**. severity: **weak**

- [read] The narration matches the stored output (0.05/0.0139 = 3.6). The `check_stored()` band pins this
  artifact, not a generalization, and the narration does not claim generality.
- The reviewer's sweep rates were not re-run.
- The fix is acceptable as optional. Leading with the cf_y point is a good edit.

## TW-11: synthetic law barely separates LTMLE from the naive contrast

verdict: **PARTIAL**. severity: **weak**

- [recomputed] W2 share = 0.02867 in the sample (0.02945 in the full file).
- [read] `ltmle_sweep.log` matches the reviewer: 37/40 coverage, naive inside the CI in 36/40, mean naive error
  +0.0149 against SE 0.0134. Its seed 0 reproduces the stored draw exactly (-0.0641 [-0.0892, -0.0390], naive
  -0.0443), so the sweep is wired correctly.
- The reviewer's remark that W2 is too rare is beside the point. The naive bias runs mainly through the
  time-varying L2 path, which is the design's purpose.
- The reading is honest about the limitation. Stating the 36/40 figure is the cheap fix. Strengthening the
  L2 paths changes the hard-coded truth assertion.

## TW-12: preprint and working-paper citations

verdict: **CONFIRMED**. severity: **weak**

- [read] The published version is Advances in Neural Information Processing Systems 30 (NIPS 2017). The
  proceedings page is
  https://proceedings.neurips.cc/paper/2017/hash/94b5bde6de888ddf9cde6748ad2523d1-Abstract.html (fetched). The
  reviewer's extracted proceedings PDF has "4.3 Binary treatment outcome on Twins" (nips.pdf.txt:398), with
  11984 pairs under 2 kg, the heavier-twin treatment, and first-year mortality. So the locator is the same.
- [read] Almond, Chay and Lee: the notebook links NBER w10552. The DOI 10.1162/003355305774268228 resolves to MIT
  Press (QJE 120(3):1031-1083, as `references.md:1818-1821` records).
- The fix is right. Update `docs/references.md` too: it still says "In arXiv v2" and links arXiv.

## New findings

1. The TW-01 mechanism is wrong (see above). The agreement check fails mainly because the hand-built TMLE uses one
   ATE clever covariate and the package uses arm-specific covariates, and because the targeting move can be near
   zero. The truncation bound is a minor contributor. Any sibling notebook that copies "use the package bound" as
   the repair would still fail.
2. The trust table omits the risk-ratio interval from the uncovered list. The study's limitation row excludes
   ratios (see TW-03).
3. On a fresh run, the TW-04 column flip also reformats the four estimate rows of `manual-estimators`
   (`5.6493e-02`). The prose numbers 0.0565, 0.0571 and 0.0570 then match only by tolerance, not textually.
