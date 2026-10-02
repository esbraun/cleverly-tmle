# Review: docs/examples/cross-fitting.ipynb

- Notebook: `docs/examples/cross-fitting.ipynb`. Callback: `tests/unit/tutorial_semantics/cross_fitting.py`.
- Commit: `5f33902b` (branch `agent/notebook-review`). The worktree venv is Python 3.11.13, and `cleverly.__file__` resolves under the worktree `src`.
- `scripts/execute_notebook.py docs/examples/cross-fitting.ipynb --check`: **exit 0, 34 s wall time**. The output was "every cell reproduced its stored non-image output" (`.tmp/notebook-review/cross-fitting/check.log`).
- Scratch scripts (`.tmp/notebook-review/cross-fitting/`):
  - `truth.py`: independent Monte Carlo truths from the structural equations, N = 4e6. Log: `truth.log`.
  - `sweep_main.py` + `analyze_main.py`: Steps 5, 6, 11 and 13 over 60 seeds (100 to 159) at n = 3000, with the notebook's learners and folds. It also runs mechanism probes. Logs: `analyze_main.log`, `main_[abc].csv`. Seed 34 reproduces the notebook exactly (`main_34.csv`).
  - `sweep_teams.py`: Step 7 over 60 seeds at n = 3000 with cluster_size 15. Log: `teams.log`.
  - `probe_glearner.py`: the Step 5/6 comparison with a regularised boosted propensity, over 40 seeds (`probe_g_[ab].csv`).
- All runs used `n_jobs=1` and `OMP/OPENBLAS/MKL_NUM_THREADS=1`.

## Findings

### CF-01: the cross-fitted interval that the page treats as the honest yardstick is about twice too wide
- Cells: `failure-mode`, `failure-mode-reading`, `trust` (and implicitly `estimate-reading`).
- Claims:
  - "The in-sample standard error is 0.27 times the cross-fitted one, which is less than a third."
  - "without splitting, the boosted fit reported a standard error 0.27 times the cross-fitted one" (trust table).
  - The step heading "the failure mode, an interval that is too narrow".
- Problem: the page measures the in-sample SE against the cross-fitted SE, so the reader infers that the in-sample SE understates the truth by about 3.7x. On this configuration, the cross-fitted SE is itself inflated by about 2.1x. The cause is that the default `HistGradientBoostingClassifier` propensity is badly miscalibrated out of fold. The page prints this signal itself: calibration slope 0.49 (Step 11) and a doubly robust nu^2 of -9.03 (Step 13). The page never connects either signal to the interval it reports.
- Severity: **misleading**.
- Evidence `[executed]` + `[recomputed]`, 60 seeds at n = 3000 (`analyze_main.log`). The truth is the independent MC value 0.16287. The efficient SE recomputed from the law is 0.00661.

  | fit | mean bias | empirical SD | mean reported SE | SE/SD | 95% coverage |
  | --- | --- | --- | --- | --- | --- |
  | cross-fitted, notebook learners | -0.0018 | 0.0057 | 0.0120 | 2.12 | 60/60 (1.000) |
  | in-sample, notebook learners | -0.0010 | 0.0053 | 0.0033 | 0.62 | 0.833 |

  - The in-sample SE / cross-fitted SE ratio ranges from 0.17 to 0.34 (median 0.28). The ratio is below 1/3 on 87% of seeds, so the "less than a third" wording happens to hold most of the time. The correct sampling-SD yardstick gives 0.62, not 0.27.
  - Mechanism probes (`[executed]`, same seeds):

    | probe | finding |
    | --- | --- |
    | Q = linear, g = boosted, cross-fitted | SE/SD 2.73, coverage 1.00 |
    | Q = boosted, g = logistic, cross-fitted | SE/SD 1.15, coverage 0.967 |
    | regularised boosted g (`max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40`), Q unchanged, cross-fitted, 40 seeds | SE 0.0059 against SD 0.0053, coverage 0.95, calibration slope 0.94 (mean) |
    | same regularised g, in-sample | coverage 0.80, in-sample/cross-fitted SE median 0.67 |

    The first two rows show that the inflation comes from the out-of-fold boosted propensity. The last two rows show that the page's real lesson survives a well-calibrated g: in-sample undercovers, and cross-fitting restores calibration.
  - Calibration slope < 0.6 on 60/60 seeds (mean 0.474). The DR nu^2 is negative on 56/60 seeds. So the miscalibration belongs to the configuration, not to seed 34.
- Proposed fix (choose one):
  - (a) Preferred. Give the treatment learner explicit regularisation, for example `HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40, random_state=34)`. Then re-narrate Steps 5, 6, 11, 12 and 13. The Step 11 calibration warning and the Step 13 DR refusal will likely disappear, so those readings and the callback assertions (`cal_slope < 0.6`, `-9.03194`, `truncated["count"] == 16`, the 0.27 ratio) must change.
  - (b) Keep the learners and add a qualification to the Step 6 reading and the trust table. Suggested wording: "The cross-fitted standard error is not the truth either. With this propensity learner, whose out-of-fold calibration slope is 0.49 (Step 11), the cross-fitted interval is conservative. In 60 draws its standard error was about twice the spread of the estimate, and the in-sample standard error was 0.62 times that spread." That wording is evidence from a scratch sweep, so the page would also need a registered source for it.
  - Files: `docs/examples/cross-fitting.ipynb`, `tests/unit/tutorial_semantics/cross_fitting.py`, and `tests/prose-report.md` (ledger).
  - Not a library bug. No registered study moves: the cited study uses a tree Q and a logistic g at n = 500.

### CF-02: "Cross-fitting addresses only the Donsker condition" invites a reading that the shown failure does not support
- Cells: `plan` ("It addresses no other condition."), `where-next` ("Cross-fitting addresses only the Donsker condition."), `failure-mode-reading`.
- Problem: on the notebook's configuration, the in-sample failure is in the variance estimate, not the point estimate. In-sample bias is -0.0010 against -0.0018 cross-fitted, with similar empirical SDs. The in-sample IC variance is too small because in-sample g (and Q) are evaluated on the rows they were fit to. Cross-fitting repairs that variance estimate as well. The page does hedge ("These values do not isolate the empirical-process term"), but the "only the Donsker condition" framing still steers the reader to read the 0.27 ratio as the empirical-process term.
- Severity: **weak**.
- Evidence `[executed]`: CF-01 table; probe "Q = linear, g = boosted": in-sample/cross-fitted SE median 0.345, against 0.754 for "Q = boosted, g = logistic". So the narrowing runs mainly through the in-sample propensity. `[read]`: `docs/technical-reference/cv-tmle.md` "What this solves" assigns IC L2 convergence to "your learners".
- Proposed fix: in Step 6, add one sentence: "Here the narrowing comes mostly from the in-sample propensity, whose weights sit near 1 on the rows it was fit to. The estimate itself is not more biased than the cross-fitted one at this sample size." In `where-next`, replace "addresses only the Donsker condition" with "addresses only the data-reuse conditions: the empirical-process term and the in-sample evaluation of the influence curve". If the reference keeps its four-row table, align `cv-tmle.md` too. Not a library bug.

### CF-03: Step 11 attributes the support pattern away from the learner, but the learner's miscalibration drives much of it
- Cell: `assessment-reading`. Claims: "Cross-fitting did not cause the support pattern, and it does not repair it. The 22.9% describes the estimated mechanism, not the population."
- Problem: the second sentence is true. The page gives no reference value, so a reader cannot tell how much of the 22.9% is the law and how much is the learner. With the true g, the treated-arm ESS ratio is 38.9% (and the true g reaches below 0.01 for 0.28% of patients). The out-of-fold boosted g gives 26.5% on average. That gap is miscalibration (slope 0.47), not support. The in-sample 91.6% is an overfitting artifact in the other direction.
- Severity: **weak**.
- Evidence `[recomputed]`: `truth.log` gives a population treated ESS ratio of 0.389 with true g and true nu^2 = 7.77. `[executed]`: cross-fitted minimum ESS ratio mean 0.265 (range 0.186 to 0.418), in-sample 0.91 (0.885 to 0.922), 60 seeds. Treated is the minimum arm on 59/60 seeds.
- Proposed fix: add "Part of the concentration comes from the learner: the slope of 0.49 says the out-of-fold propensities are too extreme, which makes the weights more uneven than the treatment law itself would." This addition is moot if CF-01 fix (a) is taken.

### CF-04: the team law is described as a modest, realistic team difference, but nearly half of the teams have a harmful effect
- Cells: `team-clusters-heading`.
- Claims:
  - "Teams differ in ways the recorded covariates do not capture, such as supervisor practice and local follow-up quality."
  - "In this law the team effect changes the benefit but not who receives an offer."
  - "the team law measures the same program on a coarser scale."
- Problems:
  1. The law's arm-by-team coefficient is 8.0 on the logit scale (`src/cleverly/datasets/synthetic.py`, `clustered_dgp`, `binomial_mean`). It was chosen to hit a design effect of about 1.95. Under it, 46% of teams have a negative mean effect of navigation. The 5th percentile is -0.32 and the 95th is +0.52, around an ATE of 0.104. An analyst who saw that would report the team heterogeneity, not only a pooled ATE.
  2. The shared team latent also has an additive main effect (`0.6 * w[:, 2]`), so it changes the baseline as well as the benefit.
  3. "Same program on a coarser scale" suggests the top-box indicator dichotomises the Step 2 score. It does not. The team law has two different covariates and its own truth (0.104 against 0.163). The covariate renames map `W2` to `medication_burden`, which is `W3` in the main law.
- Severity: **weak** (practicality / precision).
- Evidence `[recomputed]`: team-effect quantiles from 200,000 simulated teams of 50 (`team_effects.log`). `[read]`: generator source.
- Proposed fix:
  - Replace "measures the same program on a coarser scale" with "is a separate synthetic law, with its own covariates and truth".
  - Add: "The shared team factor shifts both the baseline score and the benefit. It is strong: in this law, nearly half the teams have a negative average effect, so the pooled ATE hides large team differences."
  - Optionally, a gentler coefficient with a comparable design effect is a generator/study change. It would move the clustered study, so do not do it here.

### CF-05: the plain-meaning entry for `cf_d` does not match the quantity the bound uses
- Cell: `sensitivity-heading`. Claim: "`cf_d` | the share of the remaining treatment variation that a hidden confounder explains".
- Problem: the bound uses `|rho| sqrt(cf_y cf_d / (1 - cf_d))` (`src/cleverly/sensitivity/omitted_variable.py:1196-1202`). The module docstring (line 22) defines `cf_d` as the gain in the Riesz representer, following Chernozhukov et al. That equals a treatment partial R^2 only in the partially linear model. For the ATE with a binary treatment, "share of treatment variation" is a loose analogy.
- Severity: **weak**.
- Evidence `[read]`.
- Proposed fix: "`cf_d` | how much a hidden confounder would add to the variation of the Riesz representer, which is the inverse-propensity weight here. It plays the role of a partial R^2 with the treatment". The same row exists in `point-treatment-tmle.ipynb` (line 1041) and probably `twins-causal-inference.ipynb`.

### CF-06: the reason Step 7 gives for choosing a binary outcome is opaque
- Cell: `team-clusters-heading`. Claim: "A binary outcome needs no declared support, so these fits keep cross-fitting and Step 8 can read the realized split."
- Problem: this reads as if a continuous team outcome would force cross-fitting off. The real constraint is that the Gaussian clustered law has unbounded support, so no `q_bounds` can be declared, and the package refuses that cross-fitted fit. The sentence explains a design choice of the notebook's authors, not anything the analyst controls.
- Severity: **weak**.
- Evidence `[read]`: `src/cleverly/estimators/tmle.py:1775`, and the callback comment at lines 72-74.
- Proposed fix: "The outcome is binary, so its support is known, and the team fits can cross-fit without a declared support."

## Checked and sound

- `[recomputed]` Population ATE of the Step 2 law: 0.16287 ± 0.00003 (MC, N = 4e6, from the structural equations). Matches the printed 0.163.
- `[recomputed]` Team-law ATE: 0.10413 ± 0.00018 (independent latent), 0.10426 with explicit size-15 clusters. Matches 0.104. Cluster size does not move the truth.
- `[executed]` `--check` reproduces every stored output (exit 0, 34 s). The seed-34 scratch rerun matches the notebook to all printed digits.
- `[read]` The API is current: no deprecated parameters, and no warnings in the outputs except the deliberate four-team fold-reduction warning. Every seed is set: generator `seed=34`, `Runtime(random_state=34)`, learner `random_state=34`. The HGB `random_state` is inert, since there is no early stopping at n < 10000.
- `[read]` The estimand (the ATE of the offer on the 30-day score) matches the protocol. The protocol fingerprint is the point-treatment tutorial's. The "interference unit" reasoning (clustering is dependence, not interference) is correct.
- `[read]` Untestable assumptions (consistency, no interference, no unmeasured confounding) are stated as assumptions. No diagnostic is claimed to check them. Step 13 says folds cannot check confounding.
- `[executed]` Step 6 "the two point estimates are close": |in-sample − cross-fitted| < 0.01 on 60/60 seeds.
- `[executed]` The in-sample interval undercovers at this configuration (0.833 over 60 seeds). The "one miss can be sampling variation" hedge is appropriate. The 2.5-SE miss at seed 34 is on the unlucky side.
- `[executed]` Step 7: the teams/patients SE ratio is 1.50 to 1.70 over 60 seeds (mean 1.59), against the printed 1.54. Coverage over 60 seeds:

  | fit | coverage | SE |
  | --- | --- | --- |
  | patients as units | 0.867 | 0.0191 |
  | teams as units | 0.950 | 0.0304 |

  The empirical SD is 0.0275. Both intervals covered together on 87% of seeds, so "Both intervals contain the true ATE ... on this draw" is correctly hedged. The page could usefully state that the unclustered interval undercovers.
- `[executed]` "With singleton teams it equals the ordinary formula": the variance is identical to machine precision (ratio 1.0) for an in-sample fit with one cluster per row.
- `[read]` Fewer than 40 clusters gives `few_cluster_plugin`, and unequal cross-fitted cluster sizes give `unequal_cluster_plugin` (`src/cleverly/_inference_status.py`, `docs/technical-reference/inference.md#clusters`). `[executed]` The four-team fit warns and realises 4 folds (stored output, callback).
- `[read]` Teams are contiguous blocks in the generator. Grouped folds keep each team in one fold (stored output "spans: 1"). The unclustered fit spans several folds (callback witness).
- `[read]` Step 10: equal folds of 600 make the fold-evaluated point equal the stacked point. The page's single-draw ranking caveat is present.
- `[read]` Step 12: the median-of-(variance + squared displacement) rule matches the callback's reconstruction. "About a third of this standard deviation" is 0.00101/0.0034 = 0.30.
- `[executed]` Step 11 property claims hold across seeds:
  - calibration slope < 0.6 on 60/60
  - treated is the minimum-ESS arm on 59/60
  - the in-sample ESS ratio exceeds the cross-fitted ratio on 60/60
- `[executed]` Step 13: the DR nu^2 is refused on 56/60 seeds, and the plug-in bounds at the default strengths exclude 0 on 60/60. The plug-in RV is 0.189 to 0.281, against the printed 0.200. The page qualifies the plug-in reliance correctly. The true nu^2 is 7.77, so the DR estimate of -9.03 is far off, consistent with "a fitted treatment mechanism far from the treatment law".
- `[read]` The registered study citation is accurate. `tests/canonical/tmle3_cvtmle/properties.csv`, rows `crossfit_overfitting`:

  | cell | coverage |
  | --- | --- |
  | `in_sample_control` | 0.4875 |
  | `stacked_cvtmle` | 0.9325 |

  The design matches: 400 replicates, n = 500, `DecisionTreeRegressor(min_samples_leaf=1)` Q, logistic g, 10 folds, `q_bounds=(0, 1)`, and `nonlinear_bounded_dgp(concentration=12)`, which is the page's law (truth 0.16286). "Both fall below 95%, so ... relative recovery" is an honest reading. The study also shows a stacked-arm bias of -0.0062 (CI excludes 0) at n = 500, which the page does not mention but does not contradict.
- `[read]` "No registered row covers five boosted folds exactly" is true and important. CF-01 shows that this gap matters.
- Not verified: the "Lemma 3 and Theorem 4" locators in the library's F26 refusal text, which a stored output prints. The paper is Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis, "Long Story Short", reported as published in *Review of Economics and Statistics* (2026). I could not check the locators against the published version. The text comes from library code, so it is shared with sibling notebooks.

## Patterns for siblings

- **Default `HistGradientBoostingClassifier` as a propensity learner** gives out-of-fold calibration slopes near 0.47 on `nonlinear_bounded`. That inflates the IC-based SE about 2x and drives the DR nu^2 negative. Any notebook that uses `boosted = ModelSpec(...HGB...)` on `navigation_data` inherits this, including likely `point-treatment-tmle.ipynb` and the DR-TMLE and C-TMLE pages. Check whether their intervals are compared against a yardstick that is itself conservative.
- **A ratio of two SEs presented as the failure size.** Any "A's SE is k times B's" claim needs the empirical SD as the reference, not the other estimator.
- **The `cf_d` plain-meaning row** ("share of the remaining treatment variation") is shared with `point-treatment-tmle.ipynb` and probably `twins-causal-inference.ipynb`.
- **`make_clustered(family="binomial")`** has an arm-by-team logit coefficient of 8.0. About 46% of clusters have a negative effect, and the shared latent also shifts the baseline. Any page that narrates it as a mild team difference overstates realism.
- **"Measures the same program"** wording for a different synthetic law, with its own covariates and truth, under renamed columns.
- **The library refusal text cites "Lemma 3 and Theorem 4 ... (2026)"**. One reviewer should verify the locators against the published RESTAT version once, for all pages.
