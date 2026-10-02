# Verification: `docs/examples/dr-tmle.ipynb` (findings DR-01..DR-10)

| item | value |
| --- | --- |
| commit | `5f33902b`, `cleverly.__file__` asserted under this worktree's `src` |
| own sweep | `.tmp/notebook-review/verify-dr-tmle/vsweep.py`, seeds 2000-2099 (disjoint from the reviewer's 1000-1119), n = 2,000, 100/100 completed, `vanalyze.py` -> `vanalyze.log`. Seed 55 reproduces the notebook's ordinary and DR-TMLE psi, se and CI exactly (`s55.csv`). |
| probes | `folds2.py` (DR-01); population nu^2 by 2e6 Monte Carlo draws (inline) |

Own sweep, truth 0.16286, 100 seeds:

| fit | coverage (exact 95% CI) | bias (MC se) | emp. SD | mean SE | RMSE |
| --- | --- | --- | --- | --- | --- |
| ordinary TMLE | 96/100 = 0.96 (0.90, 0.99) | -0.00208 (0.00060) | 0.00603 | 0.00684 | 0.00635 |
| DR-TMLE, notebook config | 94/100 = 0.94 (0.87, 0.98) | -0.00346 (0.00060) | 0.00597 | 0.00686 | 0.00687 |

Paired DR - ordinary: mean -0.00139 (MC se 0.00015); shift in ordinary SEs median -0.20, 10-90% -0.52 to +0.05. DR-TMLE closer to truth in 41/100.

---

### DR-01 (Step 6, "The default reduced cross-fitting refuses fewer than three folds")
- verdict: **CONFIRMED**, severity **wrong**
- [read] Notebook cell `estimate` settings table contains the quoted sentence verbatim.
- [read] `src/cleverly/estimators/drtmle.py:494` default `reduced_crossfit="pooled"`; the `n_folds < 3` refusal is at `:693`, inside `if self.reduced_crossfit == "nested":` (`:685`). Docstring `:334-335` attaches the refusal to `"nested"` only. `methods.py:919` default `"pooled"`.
- [executed] `folds2.py`: `DRTMLEMethod(cross_fitting=CrossFitting(n_folds=2), ...)` -> `pooled ACCEPTED 0.1499`; `reduced_crossfit="nested"` -> `REFUSED ValueError ... needs at least three`.
- Fix: reviewer's fix is correct. Simplest: delete the sentence (three folds is a page choice, not a library constraint). If kept, attribute the refusal to `reduced_crossfit="nested"`, which the docstring calls "a diagnostic keyword". No study moves.

### DR-02 (DR-TMLE no better than ordinary TMLE on this law)
- verdict: **PARTIAL**, severity **misleading** (kept, but the evidence must be restated)
- [read] The page never claims DR-TMLE beats TMLE on this draw or in general. Step 7 says "That resemblance says nothing about either interval's coverage"; Step 8 says "One draw cannot rank two reductions". The `plan` table makes only the conditional theoretical claim (Theorem 1 under rate conditions) and says the ordinary interval "need not attain nominal coverage". So nothing on the page is false. The problem is that the framing (DR-TMLE as the answer to the analyst's doubt) has no repeated-sampling support on this law, and the page does not say so.
- [executed] What survives in my independent sweep: the systematic DR shift (mean -0.00139, MC se 0.00015; reviewer -0.0016; median shift -0.20 SE vs reviewer mean -0.23), larger DR bias (-0.0035 vs -0.0021) and larger RMSE (0.00687 vs 0.00635), DR closer in only 41/100 (reviewer 37/86). DR-TMLE moves the estimate further in the direction of the existing bias.
- What does NOT survive: the reviewer's "both intervals undercover at n = 2,000" (0.907 / 0.884). My seeds give 0.96 / 0.94, with exact CIs containing 0.95, and emp. SD 0.0060 vs the reviewer's 0.0073. The reviewer's sweep script has no bug I could find (same law, same config, seed 55 reproduces); the difference is seed-to-seed Monte Carlo variation, which 86-100 draws cannot resolve for coverage. Pooled over both sweeps (186 draws): ordinary about 0.94, DR about 0.92; the difference is within MC error.
- Fix: the reviewer's proposed sentence quotes coverage 0.88 vs 0.91 as if established; do not use those numbers. Better wording: "On this law with these learners, a review sweep of 186 draws at n = 2,000 found no coverage advantage for DR-TMLE, and its estimate ran about 0.0014 below the ordinary TMLE's on average. The page shows how to run and read the variant, not that it helps on this law." Better still: a small registered study, since an unregistered sweep is not citable evidence under house rules. Note the paper's own simulations (Benkeser et al. 2017, Section 5.1, as rendered by PMC) use a cross-validated-bandwidth Nadaraya-Watson estimator for the consistent nuisance and a main-terms logistic model for the inconsistent one, which is structurally this page's case; so the absence of benefit here is a genuine gap worth flagging to maintainers, not just page wording. No registered study moves.

### DR-03 (plan, ordinary TMLE row "If the outcome fit converges too slowly")
- verdict: **CONFIRMED** (weak), but the proposed fix contains an error
- [read] With g converging to g* != g0, the ordinary TMLE remainder is first order in the outcome error: roughly P0[(Qn - Q0)(g0 - g*)/g*]. For the usual IC interval this must be o_p(n^-1/2). A parametric (root-n) outcome fit gives O_p(n^-1/2), which is NOT negligible: the estimator is still root-n and normal, but the IC variance is wrong.
- Fix: the reviewer's text "the interval needs the outcome fit to converge at the parametric rate" is wrong (parametric rate is not enough). Use: "With the assignment model wrong, the remainder is first order in the outcome-fit error. The usual interval then needs that error to vanish faster than root-n, which no fitted outcome model attains in general." Severity weak.

### DR-04 (trust, "That study has a cell for this page's case")
- verdict: **PARTIAL**, severity **misleading**; the rationale and the proposed fix are partly wrong
- [read] `tests/studies/canonical_drtmle.py:180-184`: outcome "correct" is an unpenalized logistic GLM with W1:W2; misspecified g is main-effects logistic; reductions are Gaussian/binomial GLMs (`:351-352`, `LinearRegression()` and `ColumnLogistic()`); ten unstratified folds (`:176-178`); binary law. So the cell differs from the page in outcome type, learner class (parametric vs boosting), reductions (GLM vs Super Learner with spline), and folds. "This page's case" overstates the match. Confirmed.
- Refuted part: the reviewer says that with a root-n parametric outcome fit "the ordinary TMLE's remainder is already negligible", and the fix says "A parametric outcome fit does not need the protection DR-TMLE adds". That is false (see DR-03): with g inconsistent, a root-n outcome error leaves an O_p(n^-1/2) remainder that invalidates the ordinary IC variance; DR-TMLE's correction is relevant there. The parametric cell tests the variance correction, not the slower-than-root-n regime.
- Fix: "Its nearest cell pairs a correctly specified parametric outcome model with a main-effects assignment model, on a binary outcome, with GLM reductions and ten folds. It does not test a flexible outcome fit that converges slower than root-n, which is this page's case." Drop the "does not need the protection" sentence. No study moves.

### DR-05 (trust, "At n = 1,500 the bias exceeded the equivalence margin")
- verdict: **CONFIRMED**, severity **weak**
- [read] `tests/canonical/drtmle/properties.csv` row `double_robustness,outcome_correct`: bias 0.005076, 99% CI 0.002611 to 0.007540, margin 0.006749, `bias_equivalent=False`. The point bias is inside the margin; the interval crosses it. Coverage rows at 1,500/3,000/6,000 confirm the page's coverage sentence.
- Fix: reviewer's fix is correct.

### DR-06 (Theorem 1 coverage of the page's configuration)
- verdict: **CONFIRMED**, severity **misleading**
- [read] The page: "Under Theorem 1 of Benkeser et al. (2017), it stays asymptotically linear, given rate conditions on the outcome fit and the reduced regressions" and trust cell "rests on the rate conditions". The library's own reference says otherwise: `docs/technical-reference/dr-tmle/targeting.md` ("Cross-fitting is not in the theorem"; the pooled construction is "specifically unaddressed"; condition (S) "is the open condition ... not free for ... a CV-chosen candidate"). `theorem.md:252` adds that the appendices also need a P0-Donsker condition on the estimated curve, which the page omits and which gradient boosting does not obviously meet (that is why the page cross-fits). [web] PMC5793673: Theorem 1 conditions are the score conditions plus the second-order terms in Appendix B being o_p(n^-1/2); Appendix A uses a P0-Donsker class condition. No cross-fitted version is proved.
- [read] `super_learner.py:404-413`: `meta_learner="auto"` uses NNLS / NNloglik weights fitted on CV predictions; the spline candidate uses `knots="quantile"` (data-chosen knots). Both are data-selected structure in (S)'s sense.
- [read] `src/cleverly/validation/drtmle.py:600` prints "this fit is Theorem 1's estimator" based only on truncation activity, for cross-fitted fits too.
- Fix: reviewer's fix is right in substance. Name both gaps: Theorem 1 covers the non-cross-fitted estimator under Donsker conditions; the pooled cross-fitted construction adds condition (S), open for Super Learner reductions. The `contract` string is a library wording issue worth a separate item (string change; no study moves). The alternative fix (fixed-basis spline reductions) still uses quantile knots; use `knots="uniform"` or fixed knots if that route is taken.

### DR-07 (sensitivity, omitted-variable bound on the ordinary fit)
- verdict: **CONFIRMED** (mechanism partly misattributed), severity **misleading**
- [read] Notebook `sensitivity-reading`: "the library builds the representer from the fitted assignment model, and this page doubts that model ... so `cleverly` refuses it here." The callback `tests/unit/tutorial_semantics/dr_tmle.py:137-150` already pins ordinary nu2 4.316 vs sample truth 7.75 and calls it "the optimism the DR-TMLE refusal prevents", but the notebook never tells the reader.
- [executed] `ordinary.sensitivity.robustness_value()` returns on 100/100 of my seeds (seed 55: rv 0.426). Ordinary nu2 / sample true nu2: median 0.696, range 0.123-0.807 (reviewer 0.674, 0.118-0.801). Always optimistic.
- [recomputed] Correction to the mechanism: population nu^2 = 8.22 untruncated, but 5.61 when g0 is clipped at the fit's own [0.01471, 0.9853] bounds (0.43% of the population lies below the floor). Against the clipped truth, the ordinary nu2 ratio is median 0.777 (range 0.72-0.84). So roughly 2.6 of the 3.8 gap between 8.2 and the misspecified limit 4.4 comes from near-positivity violations and truncation, which would afflict even a correctly specified g at these bounds; only about 1.2 comes from the misspecification the page doubts. The reviewer's "too small by the same mechanism" is only partly right.
- Fix: add to `sensitivity-reading` that the ordinary fit of Step 7 uses the same assignment model and `cleverly` does compute the bound for it, with nu2 4.32 against a sample true value of 7.75, so it is not a substitute. Do not attribute the whole gap to the misspecified g; say the true propensity also has near-zero values that the truncated fit cannot represent. Design question for maintainers stands.

### DR-08 (title: recorded rule with known form, yet a main-effects model)
- verdict: **CONFIRMED**, severity **weak**
- [read] `title` says "The recorded assignment rule has a squared term, an interaction, and a threshold", which implies a known form; the protocol rationale says only that inputs were logged. Fix is fine: describe the inputs as logged and the functional form as unknown.

### DR-09 (support row computed from the doubted g)
- verdict: **CONFIRMED**, severity **weak**
- [recomputed] 0.43% of the population has true g below 0.01471 (my 2e6 draws; reviewer 0.42%). The support row reports truncated fraction 0.0% from the fitted g. Fix is correct; this fact also underlies DR-07.

### DR-10 (gr1 "best candidate" read as the fitted reduction)
- verdict: **CONFIRMED**, severity **weak**
- [read] `super_learner.py:404-413` convex weights; the notebook prints `fit.best`, the lowest-CV-risk candidate. "The data favor a nonlinear reduction" is a reasonable gloss but the fitted reduction is a mixture. Fix is correct. I did not recompute the 70/86 cross-seed share.

## New findings

- **N1 (weak).** The reviewer's "Patterns for siblings" entry for omitted-variable bounds attributes the optimism (0.67x) to misspecified g. Much of it is truncation of a law with near-zero propensities (see DR-07); sibling pages with a *correct* g on `navigation_data` will still report nu^2 near 5.6 against 8.2. Siblings should be checked for that too.
- **N2 (weak).** The `plan` table's DR-TMLE row omits the Donsker condition that `theorem.md:252` lists alongside the rate conditions; covered by the DR-06 fix.
- No other material problem found.
