# Verification: `docs/examples/point-treatment-tmle.ipynb`

| item | value |
| --- | --- |
| findings | `reviews/notebook-review/findings/point-treatment-tmle.md` (PT-01..PT-09) |
| commit | `5f33902b`, worktree `.venv` (Python 3.11.13), `cleverly.__file__` asserted under this worktree's `src` |
| own sweep | `.tmp/notebook-review/verify-point-treatment-tmle/vsweep.py`, seeds 200-239 (disjoint from the reviewer's 100-159), n = 3000, single thread. Seed 21 reproduces the page (`v21.csv`: psi 0.17181, SE 0.009814, slope 0.4620, cf_y 1.0, cf_d 0.42779, bounds (-0.0211, 0.3647)). Results `vsweep.csv`, summary `vsum.py` -> `vsum.log` |
| own law computations | `eff.py` (efficient SE, truncated), `nu2tail.py` / `nu2short.py` (closed form + quadrature for nu_0^2 of the long and the W3-dropped law) |
| literature | arXiv 2112.13398v6 HTML (fetched, `paper.txt`): Eq. (14), Theorem 2, Appendix E.1-E.5 incl. Remark 8. Publication in REStat corroborated by web search (NBER w30302 page); DOI and published locators not checked against the published PDF |

Verdicts: 4 CONFIRMED (PT-01, 06, 07, 09), 5 PARTIAL (PT-02, 03, 04, 05, 08), 0 REFUTED. One library-level recommendation of the reviewer (PT-04, share-form benchmark) is refuted outright and must not be acted on. One new material finding (N1) overturns figures the reviewer used in PT-03, PT-08 and "Checked and sound".

## PT-01. "The largest bias is sqrt(sigma^2 nu^2)"

- verdict: **CONFIRMED**. severity: **wrong**.
- [read] Notebook cell `sensitivity-heading`, row `$\nu^2$`: "The largest bias is $\sqrt{\sigma^2 \nu^2}$, and every number below scales with it". Quoted exactly.
- [read] `src/cleverly/sensitivity/omitted_variable.py:1195-1202`: multiplier `abs(rho) * sqrt(cf_y * cf_d / (1 - cf_d))`; `cf_d` in [0, 1) so `cf_d/(1-cf_d)` is unbounded.
- [recomputed] Page values: S = sqrt(0.01748 * 23.762) = 0.6445; factor sqrt(0.1198 * 0.4278 / 0.5722) = 0.2993; 0.17181 +/- 0.1929 = (-0.021, 0.365), the printed bounds. At cf_y = cf_d = 0.9, rho = 1 the factor is 2.85 > 1, so the bias bound exceeds S.
- [read] arXiv v6: S = sqrt(E(Y-g_s)^2 E alpha_s^2) is the "identifiable scaling factor", and C_D^2 = (1-R^2)/R^2 is unbounded.
- Fix: the reviewer's wording is correct. The same "max bias" label sits in the module docstring (`omitted_variable.py:14`), the `max_bias` field (DoubleML's name), `docs/technical-reference/validation-methods.md:833,856` ("maximal bias"), and `twins-causal-inference.ipynb`. Renaming the field is a library change; the notebook fix alone needs no study.

## PT-02. Overfit default propensity inflates the interval

- verdict: **PARTIAL** (direction confirmed, magnitude overstated, proposed fix (a) risky). severity: **misleading** (low end).
- [read] The page says in `estimate-reading` "The estimate is 0.172 with a standard error of 0.010" with no caveat; in `assessment-reading` "The finding asks you to review the learner. It is not a verdict on the estimate."
- [executed] Own sweep, 40 seeds, default learners: calibration slope < 1 on 40/40 (0.383 to 0.543); DR nu^2 refused on 38/40; coverage 40/40; mean SE 0.01156 against empirical SD of psi 0.00740 (robust SD 0.0076). Ratio **1.56**, against the reviewer's 2.1 (their SD 0.0057 over seeds 100-159). Pooling both sweeps (100 seeds) gives SD about 0.0064 and ratio about 1.8. "About twice" should be "1.6 to 2.1 times".
- [recomputed] SE of the efficient influence curve with g clipped at 0.0114: 0.00572 at n = 3000 (`eff.py`), matching the reviewer's 0.0057. The reviewer's "untruncated efficient SE 0.0076" is wrong: see N1, the untruncated bound is infinite.
- [executed] Regularised learner (`max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0`), same 40 seeds: slope 1.004 (0.94 to 1.07), DR nu^2 never refused, coverage 38/40 = 0.95, mean SE 0.00565 against SD 0.00646 (ratio 0.87; the reviewer had 1.06). So it removes the over-width; its SE is near the truncated efficient SE.
- Is it overfit? Yes in the calibration sense: the out-of-fold propensities are too extreme on every seed (slope about 0.47). The default `HistGradientBoostingClassifier` has no early stopping below 10,000 rows.
- Judgment of fix (a): it would correct Step 6, but it introduces two problems. (1) The hyperparameters were chosen with knowledge of the truth and a seed sweep; a reference tutorial should show a data-driven choice (`early_stopping=True` on validation log-loss, a small CV grid, or a stacked library), or say the values are illustrative. (2) With the regularised g the plug-in nu^2 is 4.6 to 5.6, which equals the nu^2 of the law with `medication_burden` dropped (4.79, `nu2short.py`): the smoother does not fit the `W2*W3` tail where positivity fails (N1). The DR nu^2 would then pass, and Step 10 would publish DR confidence limits and RVa on a law whose nu_0^2 is infinite, outside the regularity of Lemma 3 / Theorem 4. That would be a new error. Prefer fix (b), with the measured range "1.6 to 2.1 times the sampling SD across two 40-60 seed sweeps", and qualify Step 9's "It is not a verdict on the estimate" with "it does make the interval conservative". If (a) is chosen, Step 10 must also state N1. No registered study moves for (a) or (b).

## PT-03. `medication_burden` benchmark story is seed-specific

- verdict: **PARTIAL** (seed-specificity confirmed; the reviewer's law-level mechanism is wrong; the page already carries a hedge). severity: **weak**.
- [read] The reading quoted by the reviewer is on the page. The same cell also says "Read the numbers as a worked example of the vocabulary. Do not report them as a bound for this fit." The reviewer does not mention this hedge.
- [executed] Own 40 seeds: bounds contain zero on 6/40 (15%, the reviewer had 9/60 = 15%); the relation `cf_y < 0.2 < 0.4 < cf_d` holds on 3/40 (7.5%, reviewer 8/60); cf_d median 0.064, zero on 14/40, 90th percentile 0.36, max 0.65; the page's 0.428 is above my 90th percentile. med_lower median 0.103.
- [recomputed] The reviewer's mechanism ("In the law, W3 enters g only through `0.5*W2*W3`", implying a weak treatment role) is wrong. In the law, nu_0^2 = E[1/g + 1/(1-g)] is infinite (N1), while the law without W3 has nu_s^2 = 4.79 (`nu2short.py`, converged at |W2| <= 8 and 12). The population gain G_D = nu^2/nu_s^2 - 1 of `medication_burden` is therefore infinite. The page's "much of the remaining treatment variation" is true of the law. The instability is that the estimate targets an infinite quantity, not only that g is overfit; the reviewer's "3.1x nu_0^2 = 7.77" is a Monte Carlo value of a divergent integral.
- Fix: prefix "On this draw", and say "Across draws the implied cf_d varies from 0 to about 0.7, and the bounds contain zero on about one draw in seven". Do not attribute the instability only to the overfit propensity. The callback pin `benchmark.cf_y < 0.2 < 0.4 < benchmark.cf_d` is a draw relation and may stay.

## PT-04. `cf_y = 1.0000` read as "the whole of the remaining outcome variation"

- verdict: **PARTIAL** (the clip is real and hidden; the "wrong" reading and the library concern are refuted). severity: **weak**.
- [read] `omitted_variable.py:1686-1691`: `cf_y = clip((R2_long - R2_short)/(1 - R2_long), 0, 1)` = sigma2_short/sigma2_long - 1; `cf_d = clip(nu2_long/nu2_short - 1, 0, 1)`.
- [executed] Raw gain at seed 21 = 0.040486/0.01748 - 1 = 1.316 (reviewer 1.316); over my 40 seeds 1.151 to 1.444, clipped on 40/40; share 0.535 to 0.591.
- [read] arXiv v6 Appendix E.5, Eq. (29)-(30) and Remark 8: the paper defines the gain metrics G_Y,j = sigma^2_{s,-j}/sigma^2_s - 1 and G_D,j = nu^2_s/nu^2_{s,-j} - 1, and states that under k_Y,j = k_D,j = 1 they "serve as proxies for the sensitivity parameters eta^2_{Y~U|DX} and 1 - R^2_{alpha~alpha_s}", i.e. for the share-form cf_y and cf_d of the bound. The library and DoubleML follow the paper exactly. The reviewer's open question is answered: the gain-form inputs are the published convention, and the share-form alternative (0.107, 0.300, factor 0.214) would contradict the source. Do not implement it, and do not call the existing bounds "too wide".
- Consequence for the page text: the implied share of a confounder "as strong as `discharge_risk`" is 1.316, which exceeds the largest possible share, so a clip to 1 does mean "explains all remaining outcome variation, and more is impossible". The sentence "which is the whole of the remaining outcome variation ... would leave the outcome model nothing else to explain" is a defensible reading. What is missing is that the printed 1.0000 is a clip of 1.316.
- [read] "The bound formula needs a cf_y below that ceiling" is a library rule, not the paper's: C_Y^2 = eta^2 lies in [0, 1] and the bound is finite at C_Y^2 = 1. The refusal is `_confounding_strength`, `omitted_variable.py:1197-1199` (`0 <= cf_y < 1`), pinned in the callback (`point_treatment_tmle.py:164`).
- Fix (notebook): "The implied `cf_y` is clipped at 1.0000. Dropping `discharge_risk` raised the residual outcome variance by 132%, and the benchmark uses that gain as the hidden confounder's share. A share cannot exceed 1, so `cleverly` clips it, and it accepts no `cf_y` of 1 in a bound. This covariate therefore calibrates no bound here." Fix (library, optional): expose the unclipped gains or a clip flag on `BenchmarkResult`; reconsider whether `cf_y = 1` should be refused. No study moves.

## PT-05. `cf_d` definition

- verdict: **PARTIAL**. severity: **weak**.
- [read] Page: "`cf_d` | the share of the remaining treatment variation that a hidden confounder explains". Paper (Eq. 11): for the ATE, 1 - R^2_{alpha~alpha_s} is the relative gain in average precision of the treatment model; it equals eta^2_{D~U|X} (a share of residual treatment variation) only in the partially linear model. The page's wording is the PLM reading, so it is imprecise for a binary-treatment ATE. The reviewer's second point (benchmark cf_d is a gain, so it "does not match") is the paper's proxy convention (PT-04), not a defect.
- Fix: "`cf_d` | the share of the Riesz representer's second moment, here the average inverse-propensity weight, that only the hidden confounder would add. For the ATE it is the gain in precision of the treatment model." The reviewer's text ("how much a hidden confounder would raise the second moment") reads as a gain, which mismatches the share the bound takes; use the share wording.

## PT-06. No-interference row

- verdict: **CONFIRMED**. severity: **weak**.
- [read] Page row: "one patient's assignment does not change another patient's offer or outcome"; printed assumption: "one unit's potential outcome does not depend on other units' treatment assignments". The "offer" clause adds an assignment-mechanism condition that is not part of no interference. (Dependent assignment does matter for the i.i.d. influence-curve variance, but that is a different assumption.)
- Fix: the reviewer's text is correct.

## PT-07. Trust cell

- verdict: **CONFIRMED**. severity: **weak**.
- [read] `tests/studies/cvtmle_properties.py:50-76`: the cross-fitted tree is the positive cell; `in_sample_control` has `role="control"`. `tests/canonical/tmle3_cvtmle/properties.csv` rows 2-3: truth 0.1628580 (the page's ATE), n = 500, 400 replicates, coverage 0.9325 (CI 0.894 to 0.961) for `stacked_cvtmle`, 0.4875 for the control. The page's "It adds a cross-fitted tree-learner control" mislabels the role.
- Fix: the reviewer's text is correct. Note that the positive cell also shows a bias of -0.0062 (CI -0.0085 to -0.0039), which is consistent with N1's truncation bias; cite coverage, not unbiasedness.

## PT-08. Strong positivity fails

- verdict: **PARTIAL** (the finding is real and worse than stated; its key evidence is wrong). severity: raise to **misleading** together with N1.
- [recomputed] P(g < 0.0114) = 0.00323 (reviewer 0.0032). The reviewer's "untruncated efficient variance is finite (0.173, SE 0.0076)" is wrong: see N1. My Monte Carlo of the same quantity in 1e6 chunks gave chunk SDs as large as the mean (0.236 +/- 0.27), the signature of a divergent integral.
- Fix: see N1.

## PT-09. "standardized"

- verdict: **CONFIRMED**. severity: **weak**.
- [read] `synthetic.py:85-88` and `DGP.sample`: independent standard-normal latents, not sample-standardised. Fix as proposed; it applies to all `navigation_data` pages.

## New findings

### N1. The law's ATE has an infinite efficiency bound and nu_0^2 = infinity (misleading; pages: this one and every `navigation_data` sibling)

- [recomputed] Closed form. The propensity logit is 0.6 W1 - 0.4 W2^2 + 0.5 W2 W3 + 0.3 1(W4>0). For fixed W2, E[1/g | W2] >= exp(0.18)(0.5 + 0.5 e^{-0.3}) exp(0.4 W2^2) E[exp(-0.5 W2 W3)] = c exp(0.525 W2^2). Against the N(0,1) density exp(-0.5 W2^2), the integrand grows as exp(0.025 W2^2), so E[1/g] = infinity. Numerically (`nu2short.py`): the partial integral is 4.8 at |W2| <= 4, 13.8 at 8, 696 at 16, 7e10 at 32. Because Var(Y | A, W) = m(1-m)/13 is bounded below, E[Var(Y|1,W)/g(W)] is also infinite.
- Consequences:
  - The untruncated ATE is not regularly root-n estimable on this law. The Step 6 interval is valid for the 0.0114-truncated target, whose efficient SE is 0.0057 at n = 3000; truncation bias shrinks slowly as the bound shrinks with n.
  - The page's Step 10 explanation "That estimator equals the true nu_0^2 minus the squared error of the fitted representer" is infinity minus infinity here. The plug-in and DR nu^2 estimate a truncation-dependent finite quantity, not nu_0^2. Every sensitivity number (S, rv, benchmarks) on this law is a regularisation artefact. The page's own warning "Do not report them as a bound for this fit" is correct, but its stated reason (an overfit propensity) is not the only one.
  - `medication_burden` is exactly the covariate that makes nu^2 infinite (without it nu_s^2 = 4.79).
- Fix options: (i) narrate it. Add to the data table "the true propensity approaches 0 fast enough that E[1/g] is infinite; the fit's default bound makes the target estimable, and Step 9's truncation curve measures that choice". (ii) Change the generator, for example `tanh(W2*W3)`, so strong positivity holds. Option (ii) moves registered studies: `nonlinear_dgp().propensity` is shared by `tests/studies/cvtmle_properties.py`, `bounded_cv_laws.py`, `canonical_properties.py`, and `ctmle_oat_properties.py`, so those studies would need regeneration. Prefer (i) for this review.

### N2. The reviewer's "Checked and sound" bullet on the DR nu^2 identity (nu_0^2 = 7.77, representer error about 15.7) is invalid for the same reason (weak; affects only the review record).
