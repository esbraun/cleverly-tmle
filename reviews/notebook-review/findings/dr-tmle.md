# Review: `docs/examples/dr-tmle.ipynb`

| item | value |
| --- | --- |
| notebook | `docs/examples/dr-tmle.ipynb` (callback `tests/unit/tutorial_semantics/dr_tmle.py`) |
| commit | `5f33902b` (branch `agent/notebook-review`), `cleverly.__file__` under this worktree's `src` |
| `--check` | exit 0, wall 17 s, "every cell reproduced its stored non-image output" (`.tmp/notebook-review/dr-tmle/check.log`) |
| independent truth | `.tmp/notebook-review/dr-tmle/truth.py` -> `truth.log` (4e6 Monte Carlo draws, no `cleverly` import) |
| seed sweep | `.tmp/notebook-review/dr-tmle/sweep.py` (seeds 1000-1119 requested, 86 completed before the compute budget stop), summary `analyze.py` -> `analyze.log`; seed 55 reproduces the notebook exactly (`seed55.csv`) |
| probes | `.tmp/notebook-review/dr-tmle/probe_folds.py` -> `probe_folds.log` |

The sweep repeats Steps 3, 6, 7, 8 and 9 at n = 2,000. For each seed `s` it uses `navigation_data(seed=s)`,
`Runtime(random_state=s)`, and `random_state=s` on every learner, as the notebook does with 55.

## Findings

### DR-01, cell `estimate` (Step 6 settings table)

> "`CrossFitting(n_folds=3)` | three folds | ... The default reduced cross-fitting refuses fewer than three folds"

- Problem: The default `reduced_crossfit="pooled"` accepts two folds. Only `reduced_crossfit="nested"` refuses fewer than three.
- Severity: **wrong**
- Evidence: [read] `src/cleverly/estimators/drtmle.py:685-698`. The `n_folds < 3` refusal is inside `if self.reduced_crossfit == "nested"`, and the default is `"pooled"` (`drtmle.py:494`, `methods.py:919`). The docstring at `drtmle.py:335` says the same. [executed] `probe_folds.py`: `DRTMLEMethod(cross_fitting=CrossFitting(n_folds=2), targeting=Targeting(q_bounds=(0,1)))` on the seed-55 frame returns `ACCEPTED 0.1648...`.
- Proposed fix: Replace the sentence with "Three folds leave each nuisance fit two thirds of the rows. `reduced_crossfit="nested"`, a diagnostic setting, refuses fewer than three folds." You can also delete it. Files: `docs/examples/dr-tmle.ipynb`. This is not a library bug. No study result can move.

### DR-02, cells `plan`, `ordinary-reading`, `trust` (the page's premise on its own law)

> "DR-TMLE | ... it stays asymptotically linear" (plan); "The program's question has a conditional answer. DR-TMLE reports the interval (0.15129, 0.17801)" (trust)

- Problem: On this law with these learners, DR-TMLE is not better than the ordinary TMLE. Over 86 seeds it has more bias, a higher RMSE, and slightly lower coverage. The guarded estimate moves systematically away from the truth. The page frames DR-TMLE as the remedy for the analyst's doubt and never says that its own setting shows no benefit. The single draw cannot reveal this, and the Step 7 reading ("That resemblance says nothing about either interval's coverage") is accurate but leaves the comparison open.
- Severity: **misleading**
- Evidence: [executed] 86 seeds, truth 0.16286:

  | fit | coverage of 95% CI | bias (MC se) | emp. SD | mean SE | RMSE |
  | --- | --- | --- | --- | --- | --- |
  | ordinary TMLE | 78/86 = 0.907 | -0.00243 (0.00078) | 0.00728 | 0.00689 | 0.00763 |
  | DR-TMLE, spline SL reductions | 76/86 = 0.884 | -0.00401 (0.00076) | 0.00708 | 0.00690 | 0.00810 |
  | DR-TMLE, constant reductions | 78/86 = 0.907 | -0.00242 (0.00078) | 0.00727 | 0.00686 | 0.00763 |

  The DR shift relative to the ordinary TMLE, in ordinary SEs, has a mean of -0.23 and a 10-90% range of -0.54 to +0.02. The notebook's -0.20 is typical, so the shift is systematic and not a property of seed 55. DR-TMLE lands nearer the truth than the ordinary TMLE in 37/86 = 0.43 of the seeds. Both intervals undercover at n = 2,000 (exact 95% binomial intervals include 0.95 only marginally for the ordinary fit). The coverage difference is within Monte Carlo error. The bias difference, -0.0016, is not.
- Proposed fix: Add a sentence to "How far to trust this" with the measured numbers. For example: "On this law with these learners, a review sweep of 86 draws at n = 2,000 gave 95% interval coverage of 0.88 for DR-TMLE and 0.91 for the ordinary TMLE, and the DR-TMLE estimate ran 0.0016 lower on average. This page therefore shows how to run and read the variant, not that it helps on this law." The better fix is to register a small study for this law (DR-TMLE vs TMLE, these learners) and cite it. Another option is to change the law or the learners so that the ordinary TMLE's interval demonstrably fails where DR-TMLE's holds. Files: notebook, possibly a new study under `tests/studies/`. This is not a library bug by itself. It may deserve a look, because the guarded fit adds bias in the direction of the existing bias. No registered study can move.

### DR-03, cell `plan` ("Why this method" table, ordinary TMLE row)

> "stays consistent. If the outcome fit converges too slowly, the remainder can dominate the root-n scale."

- Problem: When g converges to the wrong limit, the remainder is first order in the outcome error. So "too slowly" means slower than n^(-1/2), which every nonparametric outcome learner is, including this page's gradient boosting. The conditional "if" reads as an occasional risk when the theory says it is the norm. The page's own reference says "If one error does not shrink, the remainder is first order in the other".
- Severity: **weak**
- Evidence: [read] `docs/technical-reference/dr-tmle/index.md` "What this solves" (remainder `R_{2,a}` and the sentence quoted above).
- Proposed fix: "stays consistent. With the assignment model wrong, the remainder is first order in the outcome-fit error, so the interval needs the outcome fit to converge at the parametric rate. A flexible learner such as gradient boosting does not." Files: notebook. Not a library bug.

### DR-04, cell `trust`

> "That study has a cell for this page's case, a correct outcome regression with a wrong assignment model."

- Problem: The canonical cell is not this page's case. Its "outcome correct" regression is a correctly specified parametric GLM, which converges at root-n. In that regime the ordinary TMLE's remainder is already negligible, so the cell does not exercise the rate conditions that DR-TMLE relaxes. It also uses a binary outcome, GLM reductions (not a spline Super Learner), and ten folds. The page's case is a flexible, slower-than-root-n outcome fit on a bounded outcome.
- Severity: **misleading**
- Evidence: [read] `tests/studies/canonical_drtmle.py:181-184` ("correct": "unpenalized logistic GLM with W1:W2"; "misspecified": main-effects logistic; reductions Gaussian/binomial GLM), `:332-360` (`reduced_outcome_learner=LinearRegression()`, `reduced_treatment_learner=ColumnLogistic()`). [read] `docs/technical-reference/method-evidence/canonical-dr-tmle.md` "What was compared" (ten folds, binary law).
- Proposed fix: "Its nearest cell pairs a correctly specified parametric outcome GLM with a main-effects assignment model, on a binary outcome, with GLM reductions and ten folds. A parametric outcome fit does not need the protection DR-TMLE adds, so that cell does not test this page's flexible outcome fit." Files: notebook. No study result moves.

### DR-05, cell `trust`

> "At n = 1,500 the bias exceeded the equivalence margin."

- Problem: The point bias, 0.0051, is inside the 0.0067 margin. What failed is equivalence: the 99% bias interval, 0.0026 to 0.0075, reaches past the margin. In the contraction family at the same n, the outcome-correct rung's bias is 0.0036.
- Severity: **weak**
- Evidence: [read] `tests/canonical/drtmle/properties.csv`, row `double_robustness,outcome_correct` (n = 1500, bias 0.005076, CI 0.002611 to 0.007540, margin 0.006749, `passed=False`), and row `double_robust_contraction,outcome_correct_n1500` (coverage interval 0.9208 to 0.9637, pass). The coverage claim at 1,500, 3,000, and 6,000 is correct.
- Proposed fix: "At n = 1,500 the bias test did not establish equivalence. The 99% bias interval, 0.0026 to 0.0075, extends past the 0.0067 margin." Files: notebook.

### DR-06, cells `plan`, `trust`, `assessment-reading`

> "Under Theorem 1 of Benkeser et al. (2017), it stays asymptotically linear, given rate conditions on the outcome fit and the reduced regressions" / "The coverage of that interval rests on the rate conditions for the outcome fit and the reduced regressions."

- Problem: The page's configuration is outside Theorem 1 as published, and the page does not say so. It cross-fits the primary nuisances with pooled reduced cross-fitting, and its reductions are Super Learners with cross-validated ensemble weights. The technical reference says "Cross-fitting is not in the theorem". It also names condition (S), L2-continuity of the reduction fit, as "the open condition", "not free for anything that *selects* structure from the data: ... a CV-chosen candidate". So the interval rests on an additional, unproven condition that this page's learner choice triggers. The library's `contract: theorem -- ... this fit is Theorem 1's estimator` line, which the page repeats as "The contract is `theorem`", strengthens the impression.
- Severity: **misleading**
- Evidence: [read] `docs/technical-reference/dr-tmle/targeting.md:166-215`. [read] `src/cleverly/learners/super_learner.py` (`meta_learner="auto"` gives NNLS/NNloglik convex weights, i.e. data-selected weights). [executed] stored output of cell `assessment` ("contract: theorem -- none of the 3 truncations is active, so this fit is Theorem 1's estimator").
- Proposed fix: In `plan`, write "Theorem 1 ... covers the non-cross-fitted estimator. This page's pooled cross-fitted construction adds a continuity condition on the reduction learner, which [targeting](../technical-reference/dr-tmle/targeting.md#reduced-regression-cross-fitting) states and leaves open for cross-validated selection." Add the same condition to the trust table. Alternatively, use fixed-basis spline reductions (`spline(LinearRegression())`, no Super Learner), which the reference says satisfy (S). Then Step 9 loses the Super Learner table, so the page would need a different diagnostic. For the library: consider renaming the `contract` wording "Theorem 1's estimator" for cross-fitted fits. That is a string change. A failing-first test would assert the wording for `cross_fit=True`. No study moves.

### DR-07, cells `sensitivity`, `sensitivity-reading`

> "the library builds the representer from the fitted assignment model, and this page doubts that model ... A number that errs toward the reassuring answer is worse than no number, so `cleverly` refuses it here."

- Problem: The same doubted assignment model sits under the ordinary TMLE of Step 7, and `cleverly` does compute the bound for that fit. Its nu^2 is too small by the same mechanism. The page states the reason as the doubt about g, so a reader can conclude that falling back to the ordinary fit gives a usable bound. It does not.
- Severity: **misleading**
- Evidence: [executed] The callback pins the ordinary fit's `nu2 = 4.316` against a sample true nu^2 of 7.75 on seed 55 (`dr_tmle.py:137-146`). Over 86 seeds, ordinary-fit `nu2` / sample true nu^2 has a median of 0.674 (range 0.118 to 0.801), so it is always optimistic. [recomputed] Population true nu^2 = E[1/g0 + 1/(1-g0)] = 8.23. The main-effects logistic limit g* gives E[1/g* + 1/(1-g*)] = 4.39, and E[2 m(alpha*) - alpha*^2] = 4.39 at that limit, which confirms the refusal's Riesz identity and direction.
- Proposed fix: Add to `sensitivity-reading`: "The ordinary TMLE of Step 7 uses the same assignment model, and `cleverly` does compute the bound for it. On this draw its nu^2 is 4.32 where the true value is 7.75, so that bound is too narrow for the same reason. Do not use it as a substitute." For the library: this is a design question, not a bug. The refusal keys on the estimator, so an ordinary fit with a doubted g gets an optimistic bound silently. Maintainers may want a note in the omitted-variable docs. No study moves.

### DR-08, cells `title`, applied question

> "The program logged every common cause ... The recorded assignment rule has a squared term, an interaction, and a threshold. A main-effects logistic model of that rule is therefore misspecified."

- Problem: If the assignment rule is recorded and its form is known, a competent analyst models that form, or computes the propensity from the logged rule. The analyst would not fit a main-effects logistic model and then doubt it. The setting undercuts the premise that g is "the doubtful one".
- Severity: **weak** (practicality)
- Evidence: [read] cells `title`, `data-reading`. [recomputed] The main-effects limit has coefficients (0.553, 0.001, 0.001, 0.113), intercept -0.189, and E|g* - g0| = 0.10. It ignores the logged W2^2 and W2*W3 terms that a rule-aware analyst would include.
- Proposed fix: Describe assignment as navigator discretion informed by the logged covariates, with an unknown functional form. Keep the generator. Say that the analyst chooses a main-effects model by convention or for stability. Files: notebook. No study moves.

### DR-09, cell `assessment` / `assessment-reading`

> "validation support: maximum truncated fraction 0.0%; minimum effective-sample-size ratio 90.3%" (output, not interpreted)

- Problem: The support row comes from the doubted, misspecified g, which cannot see the -0.4 W2^2 term that drives the true propensity toward 0. The row reads as reassuring overlap evidence, but the true law has some near-violations. The reading table does not mention the row.
- Severity: **weak**
- Evidence: [recomputed] The true g has a minimum near 0, and 0.42% of the population lies below the 0.01471 truncation bound. True nu^2 is 8.23 against 4.39 under the fitted limit (`truth.log`).
- Proposed fix: Add a row to the Step 9 table: "`support` | computed from the fitted assignment model. When that model is wrong, as here, it can understate overlap problems." Files: notebook.

### DR-10, cell `assessment-reading`

> "the spline candidate has the lowest cross-validated risk in four of the six `gr1` fits ... In most of the `gr1` fits, the data favor a nonlinear reduction."

- Problem: The Super Learner is a convex ensemble, so the reduction actually used is a weighted combination and not the "best" candidate. The table reports selection by risk, not the fitted reduction. Across seeds, the gr1 statement holds in most draws. The other two families are mostly linear-best, and the page leaves that implicit.
- Severity: **weak**
- Evidence: [executed] gr1 has spline best in at least 4 of 6 fits in 70/86 = 0.81 of seeds. qr reaches that in 0/86, and gr2 in 16/86 = 0.19. [read] `super_learner.py` (`meta_learner="auto"`, convex weights).
- Proposed fix: "The candidate with the lowest cross-validated risk is the spline in four of the six `gr1` fits. The fitted reduction is a weighted combination of both candidates." Optionally, print `fit.weights` or the equivalent attribute. Files: notebook.

## Checked and sound

- [recomputed] Truth: E[Y^1] = 0.56725, E[Y^0] = 0.40439, ATE = 0.16285 (MC se 3e-5), matching the printed 0.567 / 0.404 / 0.163. ATT 0.179 and ATC 0.149 differ, as the generator docstring says.
- [recomputed] The propensity has a squared term, an interaction, and a threshold, so a main-effects logistic model is misspecified (W2 and W3 coefficients are about 0 at the limit). The outcome mean is nonlinear. Population AUC is 0.716 for the true g and 0.650 for the limit g*.
- [recomputed] The population unadjusted difference is 0.2011 (+0.038 over the ATE). E[W1|A=1] = 0.28 and E[W1|A=0] = -0.23. [executed] Unadjusted > ATE + 0.005 and a risk gap > 0.4 hold in 86/86 seeds.
- [recomputed] A logistic model with an intercept has a population calibration slope of 1 whether it is correct or not. The score equations for (intercept, slope) at the MLE limit are solved by (0, 1).
- [executed] `guard=()` equals the ordinary TMLE exactly (seed 55, by construction in `drtmle.py`). The callback asserts `ordinary.psi == empty_guard.psi`.
- [executed] Failure-mode mechanism: constant reductions pass score and correction checks in 86/86 seeds, and they change psi by a median of 0.05 ordinary SE. The spline reductions change it by a median of 0.24. Targeting solves whatever equations the reductions pose. The page's hedges ("on this draw", "not evidence") are correct: spline beats constant on |error| in only 41/86 seeds.
- [executed] The guarded fit's score and correction checks pass in 85/86 seeds. `contract == "theorem"` holds in 78/86 and `needs attention` is empty in 85/86. The page labels these "on this draw", which is fair.
- [executed] The nuisance report says "look reasonable" in 86/86 seeds, which supports the page's point that it cannot detect the misspecified g.
- [read] Reduced-regression definitions and guard crossing (`qr` -> "Q", `gr1`/`gr2` -> "g") match `theorem.md` and Benkeser et al. (2017) Section 3.2.
- [read] Propensity truncation 0.01471 = 5/(sqrt(2000) ln 2000).
- [executed] `q_bounds=None` is refused for a cross-fitted continuous outcome (`probe_folds.log`), as Step 6 says.
- [executed] The E-value row: the sample SD(Y) is 0.2345 and the conversion uses RR = exp(0.905 d), which gives 3.18 / 2.99 (`evalue.py:632`). No Riesz representer is involved, as the page says.
- [read] The refusal text's Riesz identity E[2 m(a_hat) - a_hat^2] = nu0^2 - E[(a_hat - a0)^2] is correct. [recomputed] the direction is confirmed above.
- [read] Citation: Benkeser, Carone, van der Laan & Gilbert, *Biometrika* 104(4):863-880, DOI 10.1093/biomet/asx053. Theorem 1 is in Section 3.2 of the published version (PMC5793673, fetched). The notebook names Theorem 1 without a section locator, which is correct.
- [read] drtmle vignette at the pinned commit `538a3a2` uses `SL_gr = c("SL.glm", "SL.gam")` and `SL_Qr = c("SL.glm", "SL.gam")` in its first example (fetched), as Step 6 says.
- [read] Canonical study coverage claim: the `outcome_correct_n1500/3000/6000` coverage intervals clear the floor (`properties.csv`).
- [executed] Every learner and `Runtime` seed is set, and the run is deterministic: seed 55 in the sweep script reproduces every stored decimal. No deprecation warnings appear in the outputs.
- [read] The protocol: ATE, population, outcome scale, and 30-day horizon match the identified estimand. The only changed field is the first assumption rationale.

## Patterns for siblings

- **Single-draw comparisons with no sweep behind the framing.** The notebook hedges each number "on this draw", but nowhere does it state what happens across draws on its own law. A page that motivates a method by its setting should cite repeated-sampling evidence for that setting, or say that none exists and what a quick sweep shows. Check every sibling whose trust cell says "No registered study covers this bounded law with flexible learners".
- **Registered-study cell vs. page case.** "A cell for this page's case" claims should be checked against the study's learner classes. Parametric-correct versus flexible fits are not the same regime. This likely recurs in the C-TMLE and CV-TMLE pages.
- **Diagnostics computed from a doubted nuisance** (support/positivity, nu^2, calibration). Any page that misspecifies g on `nonlinear_bounded_dgp` inherits the hidden near-violations from the -0.4 W2^2 term (0.42% of the population below 0.0147). Its support row will look clean.
- **Omitted-variable bounds on an ordinary TMLE with a misspecified g** are optimistic (nu^2 at about 0.67x truth on this law). Sibling notebooks that run `robustness_value` on `navigation_data` with a main-effects logistic g report a too-small nu^2.
- **"Default X refuses Y" statements**: check them against the default branch. Here the refusal belongs to the non-default `nested` option.
- **Library string `contract: theorem -- ... this fit is Theorem 1's estimator`** appears on every cross-fitted DR-TMLE fit, including the multi-arm and missing-outcome pages, even though the technical reference says cross-fitting is not in the theorem.
