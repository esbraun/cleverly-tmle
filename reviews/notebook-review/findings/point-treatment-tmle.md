# Review: `docs/examples/point-treatment-tmle.ipynb`

| item | value |
| --- | --- |
| notebook | `docs/examples/point-treatment-tmle.ipynb` (callback `tests/unit/tutorial_semantics/point_treatment_tmle.py`) |
| commit | `5f33902b` (branch `agent/notebook-review`), Python 3.11.13, `cleverly.__file__` under this worktree's `src` |
| `--check` | exit 0, wall 16 s. Log: `.tmp/notebook-review/point-treatment-tmle/check.log` ("every cell reproduced its stored non-image output") |
| independent truth | `.tmp/notebook-review/point-treatment-tmle/truth.py` -> `truth.log` (numpy only, structural equations re-implemented, MC n = 4e6) |
| seed sweep | `.tmp/notebook-review/point-treatment-tmle/sweep.py` (60 seeds, 100-159, n = 3000, every cell's fits; seed 21 reproduces the page exactly: `seed21.csv`), results `sweepA.csv`, `sweepB.csv`, aggregated by `summarize.py` -> `summary.log` |
| probes | `probes.py` -> `probes.log` (q_bounds refusal, regularised propensity, both-linear bias at n = 1e5, true-propensity tail); `sweep_reg.py` -> `sweep_reg.log` (40 seeds, regularised propensity learner) |

Totals: 2 wrong, 3 misleading, 4 weak.

## Findings

### PT-01. Cell `sensitivity-heading`. The bias bound is not a maximum.

- Claim: "$\nu^2$ | ... The largest bias is $\sqrt{\sigma^2 \nu^2}$, and every number below scales with it".
- Problem: the bound is $|\rho|\, c_Y \sqrt{c_D^2/(1-c_D^2)}\, \sqrt{\sigma^2\nu^2}$. The factor $\sqrt{c_D^2/(1-c_D^2)}$ is unbounded, so the bias can exceed $\sqrt{\sigma^2\nu^2}$. Chernozhukov et al. call $S$ a "scaling factor", not a maximal bias (Eq. 14, arXiv v6 = REStat 2026, DOI 10.1162/REST.a.1705).
- Severity: **wrong**.
- Evidence: [read] `src/cleverly/sensitivity/omitted_variable.py:1195-1202` computes `abs(rho) * sqrt(cf_y * cf_d / (1 - cf_d))`. [recomputed] For cf_y = cf_d = 0.9 and rho = 1 the multiplier is sqrt(0.9 * 9) = 2.85. The page's own benchmark gives 0.299, which reproduces the printed bounds 0.1718 +/- 0.6445 * 0.299 = (-0.021, 0.365). [read] arXiv v6 HTML: S is the "scaling factor".
- Fix: "$\sqrt{\sigma^2\nu^2}$ is the scale of the bound. The bound multiplies it by a factor set by `cf_y`, `cf_d`, and `rho`, and that factor can exceed 1." Files: this notebook. Not a numeric library bug. The same wording also appears in the module docstring's `\underbrace{...}_{\text{max bias}}` (`omitted_variable.py:14`), in the `max_bias` field name, which follows DoubleML, and in `docs/examples/twins-causal-inference.ipynb` ("maximal bias sqrt(sigma^2 nu^2)"). No study moves.

### PT-02. Cells `estimate`, `estimate-reading`, `assessment-reading`. The overfit propensity makes the headline interval about twice as wide as it should be.

- Claim: Step 6 fits "TMLE with explicit learners, and read[s] its interval". `HistGradientBoostingClassifier(random_state=21)` is the propensity learner with default hyperparameters. Step 9 says of the calibration warning: "It is not a verdict on the estimate."
- Problem: the default boosted classifier overfits g on every seed tested. The influence-curve SE then overstates the sampling SD by about 2.1x, and the interval over-covers. The page never tells the reader that the headline interval is inflated by this learner choice. It presents the configuration as the method to copy. An analyst would tune or regularise this learner, or use a library, before reporting. The same overfit fit drives the Step 10 refusal (PT-03, PT-04).
- Severity: **misleading**.
- Evidence:
  - [executed] Over 60 seeds the calibration slope is below 1 on 60/60 (range 0.41 to 0.54). Mean reported SE is 0.0120 against an empirical SD of psi of 0.0057. Coverage is 60/60. The doubly robust nu^2 is negative on 56/60 (93%), and the `support` check also warns on 14/60.
  - [executed] Probe at seed 21 with `HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0)`: calibration slope 1.010, no attention rows, positive DR nu^2, SE 0.0053. The default learner gives 0.0098.
  - [executed] 40-seed sweep of that learner (`sweep_reg.log`): coverage 0.95, empirical SD 0.0054 against mean SE 0.0057, DR nu^2 never refused.
  - [recomputed] The efficient SE at n = 3000 is 0.0076 for the untruncated law and 0.0057 with g clipped at 0.0114.
- Fix: choose one of two options.
  - (a) Give Step 6 a regularised or early-stopped propensity learner, or a small super learner. Then move the default overfit learner into Step 9 as the diagnostic lesson, with the sentence: "With this learner the influence-curve SE was about twice the sampling SD across 60 seeds, so the interval is conservative, not exact." This changes every printed number, so Step 10 must be re-narrated, because DR nu^2 no longer refuses.
  - (b) Keep the learner and add to `estimate-reading` and `assessment-reading`: "This propensity learner is overfit (Step 9). An overfit g inflates the influence-curve variance, so this interval is wider than an efficient one. Tune the learner before you report."
- Files: notebook, callback. Not a library bug. No registered study moves.

### PT-03. Cell `sensitivity-reading`. The `medication_burden` benchmark story holds on the shown seed only.

- Claim: "This covariate explains little of the remaining outcome variation and much of the remaining treatment variation. Its role is the mirror image of the first covariate's." Also: "The range contains zero. A confounder as strong as `medication_burden`, at the worst alignment, would be enough to explain this estimate away."
- Problem: seed 21 is an outlier. On most draws `medication_burden` is a weak treatment benchmark, and the worst-case bounds exclude zero. The large cf_d comes from the plug-in nu^2 of the overfit propensity (PT-02). In the law, W3 enters g only through `0.5*W2*W3`. The page frames nothing here as specific to this draw.
- Severity: **misleading** (true only for the seed shown).
- Evidence:
  - [executed] Over 60 seeds the bounds contain zero on 9/60 (15%). `cf_y < 0.2 < 0.4 < cf_d` holds on 8/60. cf_d has median 0.094, at least 25% of seeds clip it to 0, and its range is 0 to 0.72. The page's 0.428 is near the 90th percentile. med_lower has median 0.097 and range -0.163 to 0.174.
  - [recomputed] The plug-in nu^2 on the page is 23.76 against the true nu_0^2 = E[1/g + 1/(1-g)] = 7.77, about 3.1x. Across seeds the plug-in nu^2 ranges from 12.6 to 31.1.
- Fix: prefix the reading with "On this draw". Add: "The benchmark refit is unstable here. Across 60 draws the implied `cf_d` ranged from 0 to 0.72, and the bounds contained zero on 15% of them, because the plug-in nu^2 rests on the overfit propensity." Fix PT-02 first, then re-narrate. Files: notebook, callback (its `benchmark.cf_y < 0.2 < 0.4 < benchmark.cf_d` relation pins the outlier). No study moves.

### PT-04. Cell `sensitivity-reading`. `cf_y = 1.0000` is a clipped gain ratio, not "the whole" of the variation.

- Claim: "The implied `cf_y` reaches 1.0000, which is the whole of the remaining outcome variation. A hidden confounder as strong as the recorded risk score would leave the outcome model nothing else to explain."
- Problem: `benchmark()` computes `cf_y = clip((R2_long - R2_short)/(1 - R2_long), 0, 1)` = (sigma2_short - sigma2_long)/sigma2_long. That value is a gain ratio, which is not bounded by 1. It is 1.316 on this fit and is silently clipped to 1. The share of the short model's residual variance that `discharge_risk` explains is 0.568. The sentence therefore misreads the printed number. The reason "this covariate calibrates no bound" is the clip, not that the covariate explains all the remaining variation.
- Severity: **wrong**.
- Evidence:
  - [read] `omitted_variable.py:1686-1691`.
  - [executed] The raw gain is 1.316 at seed 21 and 1.18 to 1.41 over 60 seeds, clipped on 60/60. The share is 0.568, and 0.54 to 0.58 over the seeds.
  - [read] DoubleML `doubleml/utils/gain_statistics.py` uses the same two formulas, and the module docstring says it follows DoubleML.
  - [read] In the paper, C_Y^2 = R^2_{Y-g_s ~ g-g_s} is a share bounded by 1. C_D^2 = (1 - R^2_{alpha~alpha_s})/R^2_{alpha~alpha_s} is a gain.
  - The bound slot `cf_d` is converted with cf_d/(1 - cf_d), yet the benchmark returns cf_d in gain form, (nu2_long - nu2_short)/nu2_short.
  - I did not verify the Online Appendix's benchmark definitions, so I do not assert that the DoubleML convention is wrong.
- Fix (notebook): "The implied `cf_y` is clipped at 1.0000. Dropping `discharge_risk` raised the residual outcome variance by 132%. That gain exceeds the largest value the bound accepts, so this covariate calibrates no bound here."
- Fix (library, needs a source check before any failing-first test):
  - Make `BenchmarkResult` report the unclipped gains, or flag a clip.
  - Check against the Online Appendix whether gain-form `cf_y` and `cf_d` are the right inputs to a bound whose `cf_y` is a share and whose `cf_d` is converted with cf_d/(1 - cf_d). If they are not, every benchmark-calibrated bound in the docs is too wide. On this page, the share-form inputs (0.107, 0.300) give a factor of 0.214, not 0.299.
- Studies: none registered for benchmark values as far as I found.

### PT-05. Cell `sensitivity-heading`. The `cf_d` definition does not match either form the code uses.

- Claim: "`cf_d` | the share of the remaining treatment variation that a hidden confounder explains".
- Problem: in the bound, `cf_d` is 1 - E[alpha_s^2]/E[alpha^2]. That is a share of Riesz-representer variation (precision gain), not of treatment variation. The benchmark's `cf_d` is the gain (nu2_long - nu2_short)/nu2_short (PT-04).
- Severity: **weak**.
- Evidence: [read] `omitted_variable.py:21-23` ("the corresponding gain in the Riesz representer") and `omitted_variable.py:1691`.
- Fix: "`cf_d` | how much a hidden confounder would raise the second moment of the Riesz representer, the inverse-propensity weight. It is large when the confounder predicts the offer." Files: notebook. Sibling pages define the same terms.

### PT-06. Cell `identify-reading`. The no-interference row adds a condition about offers.

- Claim: "no interference | one patient's assignment does not change another patient's offer or outcome".
- Problem: no interference constrains potential outcomes only. One unit's outcome may not depend on others' treatment. Dependence of one patient's offer on another's is a feature of the assignment mechanism, and capacity rationing makes it common. That dependence does not violate this assumption. The printed assumption says "potential outcome ... does not depend on other units' treatment assignments".
- Severity: **weak**.
- Evidence: [read] stored output of cell `identify`.
- Fix: "one patient's offer does not change another patient's score". Files: notebook. The same row may be shared with sibling pages.

### PT-07. Cell `trust`. The tree cell is mislabelled, and the evidence on this exact law goes uncited.

- Claim: "It adds a cross-fitted tree-learner control. No registered study covers this fit's boosted learners..."
- Problem: in the study, the cross-fitted tree is the positive cell (`stacked_cvtmle`). The in-sample tree is the control. The page also misses that the study's overfitting cells sample this page's law, with truth 0.16286 = the navigation ATE. Those cells use n = 500, a `DecisionTreeRegressor` Q, a logistic g, 10 folds, and g bounds 0.025.
- Severity: **weak**.
- Evidence: [read] `tests/studies/cvtmle_properties.py:50-76`. [read] `tests/canonical/tmle3_cvtmle/properties.csv` rows `crossfit_overfitting`: truth 0.16286, cross-fit coverage 0.9325 (99% CI 0.894 to 0.961), in-sample coverage 0.4875. [read] The page's statements about GLM learners, ten folds, and bounds 0.025 to 0.975 match `manifest.json`.
- Fix: "It also runs a cross-fitted regression tree on this page's own law at n = 500, with an in-sample tree as the control. Cross-fitting restored coverage to 0.93." Files: notebook.

### PT-08. Cells `data-reading` and `estimate-heading`. Strong positivity fails in the law, and the page does not say so.

- Claim: the plan table says "positivity must hold". The data table says only that "the four baseline covariates drive assignment".
- Problem: the true propensity has infimum 0, because the `-0.4*W2^2` term is unbounded. Weak positivity (g > 0 a.s.) holds, but the strong positivity that the Wald-interval theory uses does not. The default bound clips about 0.3% of the population (P(g < 0.0114) = 0.0032), so it adds a small truncation bias. The truncation-curve reading speaks of "truncation bias" generically. It does not say that the law itself needs truncation.
- Severity: **weak**.
- Evidence: [recomputed] true g quantiles: 0% = 0.0000, 1% = 0.032, 99% = 0.829. P(g < 0.0114) = 0.32%. The untruncated efficient variance is finite (0.173, SE 0.0076 at n = 3000).
- Fix: add a row to the data table: "some profiles almost never receive an offer | the true propensity approaches 0 for extreme `prior_utilization`, so the default bound clips a few rows". Files: notebook.

### PT-09. Cell `data-reading`. "Standardized" misdescribes the covariates.

- Claim: "The four baseline covariates are standardized (mean 0, SD 1)".
- Problem: the covariates are independent standard-normal draws (`DGP.n_latent`, `synthetic.py:85-88`). They are not sample-standardized, and they are mutually independent, which no real risk score, utilization, medication count, and age would be.
- Severity: **weak**.
- Evidence: [read] `src/cleverly/datasets/synthetic.py:77-120`.
- Fix: "The four baseline covariates are independent standard-normal draws, so a negative `age` is below the average age. Real covariates would be correlated." Files: notebook. This applies to every `navigation_data` sibling.

## Checked and sound

- [recomputed] Truth: ey1 0.5673, ey0 0.4044, ate 0.1629, att 0.1791, atc 0.1493 (MC 4e6, SE about 3e-5). These match the printed 0.567/0.404/0.163/0.179/0.149. Simulating beta counterfactual draws gives an ATE of 0.1626, and Y lies in (0, 1).
- [recomputed] Spread law: ate 1.0003, att 1.6918, atc 0.3088. These match the printed 1.000/1.691/0.309.
- [recomputed] "a higher `discharge_risk` raises both the chance of an offer and the size of the effect": E[tau | W1 decile] rises from 0.082 to 0.244, and E[g | W1 decile] rises from 0.239 to 0.681.
- [recomputed] "would also score higher ... under usual support alone": E[Y0|A=1] - E[Y0|A=0] = 0.0219.
- [executed] Unadjusted > ATE on 60/60 seeds, and closer to the ATT than to the ATE on 60/60 (mean 0.2013, which is about ATT + selection bias). The page's "On this draw" hedge is conservative.
- [read] All four covariates are confounders: each enters both g and Q in `nonlinear_dgp`/`nonlinear_bounded_dgp`.
- [read] Both nuisances are nonlinear, so a GLM is misspecified for each. The linear learners in Step 8 omit `W2^2`, `W2*W3`, `1(W4>0)`, `sin`, and `tanh` terms.
- [executed] "With q_bounds=None ... cleverly refuses that fit": `CapabilityError` (probe 1).
- [read] The clever covariate is described correctly as A/g - (1-A)/(1-g). The logistic fluctuation keeps the targeted predictions in the bounds.
- [read] The fitted bound 0.0114 = 5/(sqrt(3000) log 3000) = 0.01140.
- [executed] Step 7: `att > ate > atc` estimates on 60/60. Coverage is 58/60, 58/60, and 59/60. The claim that the logistic g is correctly specified holds (`heterogeneous_dgp`).
- [executed] Step 8 mechanism, both nuisances misspecified:
  - At n = 1e5, the both-linear error is -0.0266, -0.0249, and -0.0245 on 3 seeds, a persistent asymptotic bias.
  - At n = 3000 over 60 seeds, the mean error is -0.0264 (SD 0.0058) and the interval covers on 1/60.
  - The fits with one flexible learner have mean errors of -0.0018 and -0.0013. Coverage is 58/60 and 60/60.
  - The both-linear error exceeds 2x the larger one-flexible error on 58/60 seeds (median ratio 5.4). The "about three times" on the shown seed is hedged "On this draw".
- [executed] Truncation curve: SE decreases monotonically from the fitted bound on 60/60 seeds. A retarget at the fitted bound reproduces Step 6 (callback).
- [executed] The DR nu^2 refusal and its explanation are correct. The identity E[2m(alpha_hat) - alpha_hat^2] = nu_0^2 - ||alpha_hat - alpha_0||^2 follows from the Riesz property. With nu_0^2 = 7.77 [recomputed], the page's -7.97 implies a representer error of about 15.7 in mean square. The page's warning not to report the plug-in bounds is warranted: the plug-in 23.76 is 3.1x nu_0^2, which widens the bounds by about 1.75x.
- [recomputed] Robustness value 0.233 solves rv^2/(1 - rv) = (psi/S)^2 with S = sqrt(0.01748 * 23.762). It lies in (0.2, 0.33) on 55/60 seeds.
- [read] The locators "Lemma 3 and Theorem 4", "Equation (14) in Section 4", and "Theorem 2" match arXiv v6 (Lemma 3: DML for the bound components; Theorem 4: confidence bounds). arXiv lists the published version as REStat 2026, DOI 10.1162/REST.a.1705. I did not check the locators against the published PDF.
- [read] The trust cell's claims about GLM learners, ten folds, and bounds 0.025 to 0.975 match `tests/canonical/tmle3_cvtmle/manifest.json`. All accuracy, agreement, and property rows read `pass`.
- [executed] Save and restore: the fingerprint and replayability flags match. `--check` reproduces every output.
- [read] Every effective seed is set: the generator, `Runtime`, and each learner's `random_state`. `LinearRegression` has none. Benchmark refits inherit the fit seed.

## Patterns for siblings

- "The largest bias is sqrt(sigma^2 nu^2)" and "maximal bias" wording (PT-01). Check `twins-causal-inference.ipynb`, `cross-fitting.ipynb`, and the `omitted_variable.py` docstring.
- Benchmark `cf_y = 1.0000` read as a share (PT-04). Any page that benchmarks a strong covariate hits the silent clip. The gain-versus-share convention is a library-level question.
- Default `HistGradientBoostingClassifier` as the propensity learner at n = 3000 (PT-02). It overfits g (calibration slope about 0.47), inflates the IC SE about 2x, and makes the DR nu^2 negative. Check every sibling that copies the Step 6 `flexible` method, especially `cross-fitting.ipynb`, whose stored output carries the same plug-in refusal.
- Sensitivity narratives pinned on one draw's benchmark values (PT-03). The plug-in nu^2 from an overfit g makes the benchmarks very unstable.
- The `navigation_data` covariates are independent N(0, 1), and the law's propensity has infimum 0 (PT-08, PT-09). This is shared by every tutorial that draws `make_nonlinear_bounded`.
- The no-interference row wording (PT-06), if it is shared.
