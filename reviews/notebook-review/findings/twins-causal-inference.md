# Review: `docs/examples/twins-causal-inference.ipynb`

- Commit: `5f33902b` (branch `agent/notebook-review` = origin/main). `cleverly.__file__` resolved under this worktree's `src`.
- Module: `tests/unit/tutorial_semantics/twins_causal_inference.py` (`UNPRINTED_DECIMALS`, `check_stored()` only).
- `scripts/execute_notebook.py docs/examples/twins-causal-inference.ipynb --check`: **exit 1**, wall time 53 s (network worked; the
  download succeeded). Differences in `manual-estimators`, `ltmle-diagnostics`, and `reproducibility` (see TW-04, TW-05). Log:
  `.tmp/notebook-review/twins-causal-inference/check.log`.
- Scratch scripts (all under `.tmp/notebook-review/twins-causal-inference/`):
  - `data_check.py` - independent recomputation from the pinned CSVs (no `cleverly`), plus a 200-sample sweep of the within-pair stress test.
  - `ltmle_sweep.py` - independent LTMLE truth (enumeration and 1.2M-draw Monte Carlo) and a 40-draw sweep of the synthetic draw.
  - `point_sweep.py`, `point_sweep2.py`, `point_sweep3.py` - sections 6 to 9 rerun over 30 pair-sample seeds (0..29; learner seeds follow the sample seed). Combined result `point_sweep_all.csv`.
  - `g_range.py` - in-sample propensity range of the hand-built logistic model over the same 30 seeds plus 2026.
  - `gap_probe.py` - why the package and hand-built TMLE differ at seed 1.
- PR 211 open items: all three are **closed**. The outcome is `mortality_1y`, a `StudyProtocol` is written and carried
  (fingerprint `0643c45a6ab43754`), and a typed `RiskRatio` is fitted.
- Note for the coordinator: the brief describes a "heavier twin" design with a twin-based truth. The notebook no longer does that.
  Treatment is birth weight below 2,500 g over all sampled children, there is no real-data truth, and the only known truth is the
  section 11 semi-synthetic LTMLE law. The review below follows the notebook as it is.
- I could not read `docs/development/example-notebooks.md`: the tool permission layer denied the read.

## Findings

### TW-01 - `production-fit` (and `manual-estimators`): the hand-vs-package agreement holds only for the seed shown

Claim/code: `assert abs(package_gap) < 0.1 * abs(targeting_shift)`; the prose says the package "bounds the propensity at the package
default rather than at 0.01" and "the two constructions agree well within that move".

Problem. The hand-built TMLE clips the in-sample propensity at [0.01, 0.99]. The package bounds it at 5/(sqrt(n) log n) = 0.004859.
The two bounds differ whenever an in-sample propensity leaves [0.01, 0.99]. That happens often in this data, because rare
categorical cells (one-hot race and region levels) give the penalized logistic model near-deterministic fits. When the bounds bind,
the counterfactual updates `eps / g1` and `eps / (1 - g1)` differ by up to a factor of two on those rows, and the gap exceeds the
targeting move. The notebook then stops with an `AssertionError`.

Severity: **misleading** (true only for the seed shown; the notebook aborts at 11 of 30 other samples).

Evidence:
- [recomputed] `g_range.py`: at seed 2026 itself the in-sample propensity reaches 0.99751, and 4 rows lie outside both [0.01, 0.99]
  and the package bound. The agreement at 2026 holds only because those 4 rows carry a tiny q0 (gap -1.13e-06).
- [recomputed] Over pair-sample seeds 0..29, the 0.01 clip binds in 13/30 samples. Extremes: in-sample g down to 0.00095 (seed 26)
  and up to 0.99945 (seed 1).
- [executed] `point_sweep_all.csv`: `|package - hand| / |hand - gcomp|` is below 0.1 in 19/30 samples. It is 4.97 at seed 1, 0.92 at
  seed 14, 0.80 at seed 10, and 0.76 at seed 24. `gap_probe.py` (seed 1) shows identical nuisances (max |g_pkg - g_hand| = 7.5e-15).
  The whole gap is therefore the bound.

Proposed fix: make the hand-built calculation use the package bound (`5 / (np.sqrt(n) * np.log(n))`, or read it from the result).
The remaining gap then measures only one clever covariate against two. State in the prose that the agreement depends on the
bound. Optionally print the in-sample propensity range of the ordinary fit. The overlap section shows only the cross-fitted
Super Learner propensities (0.2384 to 0.9389), and those hide the near-deterministic in-sample cells. Files:
`docs/examples/twins-causal-inference.ipynb` (cells `manual-estimators`, `production-fit`, readings); `check_stored()` keeps the
0.1 relation. Library bug: no. Registered study result can move: no.

### TW-02 - `diagnostics` / `sensitivity`: the sensitivity section fails outright on some samples

Code: `assessment.report("omitted_confounding")` and `flexible_result.sensitivity.benchmark(covariates=("preterm", "tobacco"), ...)`.

Problem. The doubly robust estimate of nu^2 goes negative on some samples. The library then refuses the bound, which is correct
behaviour (`omitted_variable.py:699`). The notebook does not anticipate this. At seed 11 the assessment records
`omitted_confounding` as `unavailable`, and `report("omitted_confounding")` raises `KeyError`. At seed 5 the benchmark's short refit
raises `CapabilityError` (nu^2 = -10.31). The reading's caveat ("Interpret this ... value only if the fitted assignment model
consistently estimates the full assignment mechanism") describes this risk. The notebook does not show that the risk materializes
in about 1 sample in 15.

Severity: **misleading** (a reader's rerun on another sample crashes in 2/30 samples; the stored run looks routine).

Evidence: [executed] `point_sweep.log` (seed 5 traceback: "the 'doubly_robust' estimator of nu^2 returned -10.3091");
`point_sweep_c.log` (seed 11: "returned -10.0006", status `unavailable`). Seed 11 has in-sample g down to 0.0010.

Proposed fix: in the reading, say that the bound needs a nonnegative nu^2 and that the library refuses otherwise. Point to the
near-deterministic cells of TW-01 as the cause, and consider coarsening the rare race and region levels. Files: the notebook.
Library bug: no (an explicit refusal). Study result can move: no.

### TW-03 - `trust`: the evidence row describes the clustered CV-TMLE study incorrectly

Claim: "the clustered point-treatment CV-TMLE study, with a continuous outcome, a linear outcome regression, and the exact
propensity | the binomial outcome and the Super Learner used here, which no registered study covers".

Problem. The registered study uses a **binary** outcome and an unpenalized **logistic** outcome regression. Commit `93800e86`
regenerated it on a declared binary law. So the binomial outcome *is* covered. What is not covered: an estimated propensity, the
Super Learner, penalized learners, clusters of size two, and three folds.

Severity: **wrong**.

Evidence: [read] `docs/technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md:3-4` ("binary-outcome clustered law
with ten observations per cluster"), `:23` ("unpenalized logistic regression, `C=1e6`"), `:173`; and `validation-grid.md:31`.

Proposed fix: rewrite the row to read "binary outcome, cluster-robust variance, grouped folds, exact propensity, clusters of ten" |
"an estimated propensity, the Super Learner, penalized logistic learners, clusters of two, three folds". File: the notebook
(`trust` cell). Library bug: no. Study result can move: no.

### TW-04 - `targeting-reading`: three quoted numbers appear in no stored output, and one cannot have come from the stored run

Claim: "Before targeting, the residual score is 5.0875e-04. After targeting it is 2.9606e-19 ... The fluctuation epsilon is
4.2286e-03".

Problem. The stored `manual-estimators` table prints `0.0005`, `0.0000`, and `0.0042`. pandas switches the whole column to
scientific notation whenever any nonzero value is below 1e-4. The stored fixed-point table therefore means that the stored run had
`final_score == 0.0` exactly. The value 2.9606e-19 comes from some other execution. A fresh run prints scientific notation with
5.9212e-19, so `--check` fails on this cell. The narration test does not see the problem, because `_DECIMAL` in
`tests/unit/tutorial_semantics/__init__.py:255` rejects a mantissa followed by `e` (`(?!\w...)`). Every scientific-notation decimal
in prose is invisible to it.

Severity: **misleading** (the reading quotes values the artifact does not support).

Evidence: [executed] `check.log` (fresh: `5.0875e-04`, `4.2286e-03`, `5.9212e-19`; stored: `0.0005`, `0.0042`, `0.0000`).
[recomputed] the pandas behaviour with `display.precision=4`: 2.96e-19 prints `2.9600e-19`, and 0.0 prints `0.0000`.
[recomputed] Over 30 samples, `final_score` is exactly 0.0 in 7/30, so the table format is a coin toss.

Proposed fix: print each ladder value with an explicit format (`f"{value:.4e}"` for the scores and epsilon). Say "below 1e-10"
rather than quoting machine noise. In the harness, extend `_DECIMAL` (or add a second pattern) to cover scientific notation.
Files: the notebook; `tests/unit/tutorial_semantics/__init__.py` (harness gap, needs a failing-first test that a narrated
`1.23e-04` with no printed match is reported). Library bug: no. Study result can move: no.

### TW-05 - `ltmle-diagnostics`, `reproducibility`: stored outputs print platform noise

Problem. `--check` also differs on `ltmle-diagnostics`, where the solver scores print at the 1e-18 level (3.6637e-18 vs 3.1826e-18),
and on `reproducibility` (Python 3.13.7 stored vs 3.11.13 here). The reading "A rerun with the same versions, commit, and seed
reproduces them" is untestable when the table prints round-off. Severity: **weak**. Evidence: [executed] `check.log`.
Proposed fix: print `abs(score) < threshold` or the ratio rounded, not the raw 1e-18 score. Files: the notebook.

### TW-06 - `association-stress-test` / `comparison-figure`: the stress-test interval excludes zero only for this sample

Claim/output: "within threshold-discordant pairs 0.0085 [0.0008, 0.0171]"; the figure shows this interval clear of zero.

Problem. The notebook computes a model-free paired contrast on a random 6,000 of 71,345 pairs. On the full file (14,870 discordant
pairs) the contrast is 0.0047. Over 200 random 6,000-pair samples it has mean 0.0051 and SD 0.0037. Its bootstrap interval excludes
zero in only 24% of them. The shown sample is about one SD high. The prose does not claim significance. The figure, though, invites
the reading of a positive within-pair effect that most samples would not show.

Severity: **weak** (seed-favourable display; no explicit false claim).

Evidence: [recomputed] `data_check.py` / `data_check.log`.

Proposed fix: compute the stress test on all pairs (it needs no model), or state that its interval reflects a 6,000-pair subsample.
Files: the notebook.

### TW-07 - `protocol` / `data-heading`: eligibility omits the source file's exclusions

Claim: `target_population="Children from same-sex US twin births in 1989-1991"`,
`eligibility=("Both twins have a recorded birth weight and first-year outcome",)`.

Problem. The pinned README says "I removed all pairs with exactly the same weight". The file holds 71,345 pairs and has no missing
weights or outcomes. The real eligibility is therefore "same-sex pairs with unequal birth weights, as preprocessed by CEVAE". Every
equal-weight pair is concordant for the 2,500 g threshold, so the exclusion changes the population. The random 6,000-pair subsample
is also not stated as part of the design. Severity: **weak**.

Evidence: [read] `ReadmeTwins` at the pinned commit. [recomputed] 0 missing values in T and Y; 0 equal-weight pairs; twin 0 lighter in
100% of pairs.

Proposed fix: add the unequal-weight exclusion to `eligibility`, and mention it in the data-heading prose. Files: the notebook; the
fingerprint and its `check_stored()` assertions change with it.

### TW-08 - `sensitivity-reading`: "this ordinary-TMLE value" names the wrong fit in this notebook's vocabulary

Problem. The notebook uses "ordinary package TMLE" for the non-cross-fitted fit of section 7. The sensitivity value comes from the
Super Learner CV-TMLE. The library's "ordinary TMLE" (as opposed to C-TMLE or DR-TMLE) covers both fits, but a reader of this
notebook will think the sentence refers to the section 7 fit. Severity: **weak**. Evidence: [read] the notebook `diagnostics` cell
(`flexible_result.assess`), and `validation-methods.md:997` for the library sense.
Proposed fix: write "Interpret this Super Learner CV-TMLE value only if ...".

### TW-09 - `sensitivity-heading`: cf_d is paraphrased in its partially linear special case

Claim: "Its strength cf_d is the share of the residual treatment variation."

Problem. For the nonparametric ATE, cf_d is the gain in the Riesz representer's variation from adding the confounder
(Chernozhukov et al., the `1 - R^2_{alpha ~ alpha_s}` term). It equals a share of residual treatment variation only in the partially
linear model. The library docstring says "the corresponding gain in the Riesz representer"
(`src/cleverly/sensitivity/omitted_variable.py:22`). The library's RV sentence ("explaining 14.2% of the residual variation in BOTH
the outcome and treatment") uses the same simplification. Severity: **weak**. Evidence: [read].
Proposed fix: write "cf_d is the share of the Riesz representer's variation that the confounder explains. For a binary treatment it
measures how much the confounder would sharpen the treatment prediction." Consider the same fix in the library's RV text, which
would need a failing-first test on the summary string.

### TW-10 - `sensitivity-reading`: "about four times the benchmark" and "cf_y rounds to 0" are sample-specific

Problem. The reading narrates the stored output correctly. The narrated relations do not generalize, and the `check_stored()`
assertion `3.5 < 0.05 / cf_d < 4.5` pins a seed-specific value. Over 28 samples where the benchmark ran, 0.05/cf_d had median 3.7
and range 0.09 to infinity (cf_d prints 0.0000 in 4 samples). It falls in (3.5, 4.5) in only 5/28 (18%). The implied cf_y prints
0.0000 in 57% of samples, but it is at most 0.0019 in all of them. So the stable conclusion is "the benchmark gives no scale for
cf_y". Severity: **weak**. Evidence: [executed] `point_sweep_all.csv`.
Proposed fix: keep the narration but add that the benchmark cf_d varies widely with the sample. Lead with the stable cf_y point.

### TW-11 - `semisynthetic-data` / `semisynthetic-fit`: the synthetic law cannot separate LTMLE from the naive contrast

Problem. The reading concedes that one draw does not show bias removal, which is honest. The law, however, makes the check
uninformative in almost every draw. The naive bias is about +0.015, roughly 1.1 standard errors. W2 (diabetes or chronic
hypertension) is 1 for only 2.87% of rows, so baseline confounding through W2 is negligible. Severity: **weak**.

Evidence: [recomputed] `ltmle_sweep.py`: exact truth -0.064643 (matches the notebook's -0.06464265; Monte Carlo -0.06528 with
1.2M draws per regime). [executed] Over 40 synthetic draws: LTMLE coverage of the truth 37/40; the interval contains the naive
contrast in 36/40; LTMLE is closer to the truth than the naive contrast in 29/40. Mean LTMLE error -0.0036, mean naive error +0.0149,
SD of the LTMLE estimate 0.0122, mean SE 0.0134.

Proposed fix: strengthen the L2 to A2 and L2 to Y paths, or use a larger synthetic n, so that the naive contrast lies outside the
interval in most draws. Alternatively, state the 36/40 figure. Files: the notebook, and the hard-coded truth assertion if the law
changes.

### TW-12 - `data-heading`, `sensitivity-reading`: citations point to the preprint and the working paper

Problem. Louizos et al. is cited as "(2017, arXiv v2, section 4.3)" with the arXiv link. The published NIPS 2017 version has the same
locator: section 4.3 "Binary treatment outcome on Twins", 11,984 pairs below 2 kg, treatment "being born the heavier twin",
first-year mortality, and one twin selectively hidden. The project convention is to cite the published version. Almond, Chay and
Lee "(2005)" links NBER w10552 (a 2004 working paper). `references.md` gives the QJE DOI, but the inline link does not.
Severity: **weak**. Evidence: [read] both PDFs (NeurIPS proceedings and arXiv v2) extracted to `nips.pdf.txt` / `arxiv.pdf.txt`.
[recomputed] 11,984 pairs with both twins < 2,000 g in the pinned file.
Proposed fix: link the NeurIPS proceedings page and write "section 4.3". Link the QJE DOI inline. Files: the notebook and
`docs/references.md:1818-1821`.

## Checked and sound

- [recomputed] Sample: 12,000 children, 6,000 pairs, LBW share 0.5145, mortality 0.0359, observed risks 0.0043 / 0.0658, crude RD
  0.0615, 1,290 discordant pairs, paired RD 0.0085. All match the stored output, recomputed without `cleverly`.
- [recomputed] 47 adjustment columns = 7 binary + 4 missingness indicators (diabetes, chyper, and phyper share one pattern: 4,240
  NaN each) + 36 dummies. The "one indicator per distinct missingness pattern" description is correct.
- [read]+[recomputed] Twin 0 is the lighter twin in every pair, and no pair has equal weights. The outcome is first-year (infant)
  mortality: the NBER linked infant-death source and Louizos's "3.5% first-year mortality" agree with the file's 0.0359.
- [read] `preterm` = "Previos pre-term or small" (covar_desc). gestat10 is excluded, as stated. birth order is excluded.
- [read] The three URLs are pinned to commit `9081f863`. `SEED` drives the pair sample, the bootstrap, the learners, the folds, and
  the assessment. `SEED + 1` drives the synthetic draw. No unseeded randomness was found.
- [recomputed] The default bound 5/(sqrt(12000) ln 12000) = 0.004859. The maximum clever covariate 12.82 = 1/(1 - 0.9220).
- [recomputed] E-value: 13.72 + sqrt(13.72 x 12.72) = 26.93 (stored 26.94). The limit 8.92 gives 17.33. The log-scale symmetry of the
  RR interval holds: 0.4307 on each side.
- [executed] Over 30 samples, stable claims: both calibration flags fire (30/30); `nuisance_models` is the only attention row
  (30/30); the sign survives at cf = 0.05 wherever the bound runs (29/29); the targeting move / initial score ratio lies in
  [0.994, 1.152], which supports "close to the initial score"; the CV-TMLE estimate has mean 0.0580 and SD 0.0037 across samples; the
  CI lower limit is above 0 in 30/30.
- [read] Cluster handling: grouped outer folds (`StratifiedGroupKFold`, `crossfit.py:1191`), and `groups` is passed into the
  Super Learner inner CV (`super_learner.py:188-242`). `simulated_confounding` is refused for clustered fits
  (`_simulated_confounding_request.py:216`), as the reading says.
- [recomputed] LTMLE truth: independent enumeration gives -0.064643, which matches. The DAG table matches the structural equations.
- [read] The identification statement, the interference caveat, the protocol's stated failures of exchangeability and consistency,
  and the "not a causal finding" framing are accurate. The reading avoids claiming double robustness or coverage from one sample.
- [read] The binomial outcome uses the logistic iterative fluctuation, which is appropriate for a 3.6% outcome.

## Patterns for siblings

1. Hand-built versus package agreement checks must use the package's own truncation bound. A different clip makes the agreement
   seed-dependent whenever the in-sample propensity approaches 0 or 1, and in-sample fits do so far more often than the cross-fitted
   ones a notebook displays.
2. pandas `display.precision` switches a whole column to scientific notation when any nonzero value is below 10^-precision. A table
   that mixes estimates with machine-zero scores therefore changes format between runs. Format scores explicitly.
3. The narration regex (`_DECIMAL`) ignores scientific-notation literals in prose. Any notebook that quotes `x.xxxxe-yy` is
   unchecked.
4. One-hot rare categories with a penalized logistic model give near-deterministic in-sample propensities. Omitted-variable bounds
   can then refuse (negative DR nu^2). Notebooks that call `report("omitted_confounding")` or `benchmark()` unconditionally will
   crash on some samples.
5. Evidence-row text in "How far to trust this" tables goes stale when a study is regenerated on a new law. Cross-check against the
   study page.
6. Semi-synthetic laws should be strong enough that the naive contrast falls outside the estimator's interval in most draws.
   Otherwise the "check against an exact truth" demonstrates little.
