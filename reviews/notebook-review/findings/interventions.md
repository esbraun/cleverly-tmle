# Review: `docs/examples/interventions.ipynb`

- Notebook: `docs/examples/interventions.ipynb`. Callback: `tests/unit/tutorial_semantics/interventions.py`.
- Commit: `5f33902b` (branch `agent/notebook-review`). `cleverly.__file__` resolved to this worktree's `src`. Python 3.11.
- `scripts/execute_notebook.py docs/examples/interventions.ipynb --check`: **exit 0**, wall time **15 s**, "every cell reproduced its stored non-image output" (`.tmp/notebook-review/interventions/check.log`).
- Scratch scripts (all under `.tmp/notebook-review/interventions/`):
  - `truths.py` / `truths.log`: independent truths. I re-implemented the structural equations and used 2x10^7 (binary law) and 4x10^6 (dose law) Monte Carlo draws. No library truth helper was called.
  - `sweep.py` / `sw1-3.csv` / `summarize.py` / `summary.log`: a 60-seed sweep (seeds 1-60, n = 3000) of all three axes. The sweep uses the notebook's learners, folds, bins, and bounds, and learner and Runtime seeds equal to the data seed. Seed 31 reproduces the notebook's regime estimate (0.171645), and seed 32 reproduces its shift estimates (1.800, 0.788). For the incremental and regime fits, the sweep also ran a degree-2 polynomial logistic treatment learner as a probe.
  - `shift_probe.py` / `pq.csv`, `pc1.csv`, `pc2.csv`: two probes of the shift-axis undercoverage over 40 seeds each. One uses a correctly specified quadratic outcome regression in sample. The other uses the notebook learners with 3-fold cross-fitting and declared wide `q_bounds=(-30, 40)`.
- Total compute was about 14 minutes wall time. Each process ran single-threaded, with at most 3 processes at once.

## Findings

### IV-01: the incremental-intervention miss is systematic, not sampling variation
- Cell: `incremental-fit-reading`. The claim: "The estimate lies 2.6 standard errors above the population value. A 95% interval misses on about one draw in 20, so one miss is consistent with sampling variation." Also: "One draw cannot show whether the miss comes from g. Try a better-calibrated treatment learner before you report this axis."
- Problem: the notebook's configuration (default `HistGradientBoostingClassifier` propensity, 3-fold CV-TMLE) is biased upward for `ate_ipsi[double odds vs current odds]`. Its 95% interval covers the truth on about half of the draws. The reading attributes the miss to chance. A probe does show the cause: the miscalibrated g. The notebook recommends that probe and does not run it.
- Severity: **wrong**. The "one draw in 20" premise is false for this configuration.
- Evidence:
  - [recomputed] Truth is 0.02338 (MC; the notebook prints 0.023).
  - [executed] 60 seeds with the notebook configuration gave these results. Mean estimate 0.0252, bias +0.0018 (MC se 0.0001), which is about 2 SE of a single fit. **Coverage 29/60 = 0.48**. The mean z is +2.03, z > 0 on 56/60 seeds, and |z| >= 2.6 on 35% of seeds. SE ratio (mean reported SE / empirical SD) 0.78.
  - [executed] Probe with only the treatment learner swapped for `make_pipeline(PolynomialFeatures(2), LogisticRegression())`, on the same seeds. Bias +0.0002, **coverage 56/60 = 0.93**, SE ratio 0.88. On seed 31 the estimate moves from 0.0258 (miss) to 0.0243 (covers).
  - [executed] The `nuisance_models` calibration warning fires on 60/60 seeds (slope about 0.46 on seed 31). The warning is real and has consequences.
  - [read] `src/cleverly/interventions/incremental.py` module docstring. The remainder carries (g_hat - g) in every term, so the estimand is not doubly robust. This matches the notebook's own Step 12 reading, which makes a sampling-variation explanation the less likely one.
- Proposed fix: Option (a) changes the treatment learner for the binary-treatment axes to a calibrated one, such as degree-2 logistic or a regularized or `CalibratedClassifierCV` booster, and re-reads the outputs. Option (b) keeps the booster as a deliberate failure and narrates the probe: show the poly-logistic refit beside it, and say the miss comes from g. Either way, delete "consistent with sampling variation". Update the callback assertion `2.55 <= z < 2.65` and the `not covers` assertion. Files: the notebook and the callback. Not a library bug. No registered study moves.
- Note for the coordinator: `docs/development/example-notebooks.md` (rule "Explain an interval that misses the true value as sampling variation ... Name another cause only when a probe or a study supports it") pushed the author toward this phrasing. That rule needs "run a seed sweep before invoking sampling variation".

### IV-02: the shift-axis intervals are miscalibrated in this configuration, and the reading frames the misses as one-draw events
- Cells: `dose-fit-reading` ("One draw is not a coverage result either way") and `failure-reading` ("On this draw, the interval contains the population gap and excludes zero").
- Problem: with in-sample boosted Q, a 40-bin boosted density, and n = 3000, the `+1.0 uncapped` interval covers its truth on 60% of draws. The capped `+0.5` contrast covers 83%, and the highlighted cap gap covers 60%. The notebook's seed shows +1.0 *above* the truth (+0.050). The systematic bias is *downward* (-0.0275). The reading's attribution to in-sample fitting is only partly right.
- Severity: **misleading**. The hedge is technically true. The page still presents these intervals, and the gap interval in particular, as usable inference.
- Evidence:
  - [recomputed] Truths are 0.7759 / 0.8123 / 1.7495 by MC, matching the printed 0.7761 / 0.8125 / 1.7500. Closed forms: uncapped 1.5 delta + 0.25 delta^2, which gives 0.8125 and 1.75. E[Y] = 3.395.
  - [executed] 60-seed sweep, notebook configuration:

    | contrast | bias | emp. SD | mean SE | SE ratio | coverage |
    | --- | --- | --- | --- | --- | --- |
    | +0.5 capped at 5 | +0.0086 | 0.0140 | 0.0118 | 0.84 | 50/60 = 0.83 |
    | +0.5 uncapped | +0.0022 | 0.0132 | 0.0116 | 0.88 | 56/60 = 0.93 |
    | +1.0 uncapped | -0.0275 | 0.0342 | 0.0206 | 0.60 | 36/60 = 0.60 |
    | gap uncapped - capped | -0.0065 | 0.0074 | 0.0047 | 0.64 | 36/60 = 0.60 |

  - [executed] Probe with a correct outcome regression (degree-2 polynomial OLS) in sample, 40 seeds. +1.0 bias -0.018, coverage 0.775. Capped and uncapped +0.5 coverage 0.90 and 0.875. Most of the +1.0 bias survives a correct Q, which points to the 40-bin boosted density under strain.
  - [executed] Probe with notebook learners and 3-fold cross-fitting with declared `q_bounds=(-30, 40)`, 40 seeds. +1.0 bias -0.030 persists. The reported SE inflates about 3.6x (mean SE 0.088 against emp. SD 0.024), so coverage is 1.0 for the wrong reason. Cross-fitting does not repair the bias. The "in-sample" explanation in the reading accounts only for the SE shortfall.
  - [read] The callback pins seed-specific relations: `not covers(... +1.0 ...)` and covers for both +0.5 rows.
- Proposed fix: state the measured miscalibration, or change the configuration so that the intervals the page reads are calibrated. Options: declare a parametric Q for this lesson, use more bins, or drop `+1.0` and use a smaller shift. In the failure step, say that the gap estimate is valid as a point contrast. Do not present its interval as calibrated without evidence. Files: the notebook and the callback. Not a demonstrated library bug. The cross-fit SE inflation with wide declared bounds may deserve a separate look, because 3.6x is large. No registered study moves.

### IV-03: "How far to trust this" omits that every cited study supplies the mechanism or density exactly
- Cell: `trust`, the per-axis table ("ordinary, non-cross-fitted targeting, pointwise intervals only, a binary outcome, and no flexible learners", and so on).
- Problem: each study's Limits section names a gap that is load-bearing here. In the regime and incremental studies, the treatment mechanism is supplied exactly. In the MTP study, `cleverly` evaluates the known law through its pooled-hazard representation. No registered study therefore covers an *estimated* g or density, which is where IV-01 and IV-02 fail. "No flexible learners" does not convey this. For the incremental axis the gap is the central one, because g is inside the estimand.
- Severity: **misleading**.
- Evidence:
  - [read] `docs/technical-reference/method-evidence/deterministic-point-treatment-regimes.md`, Limits: "the treatment mechanism is supplied exactly".
  - [read] `incremental-propensity-interventions.md`, Limits: "The treatment mechanism is supplied exactly to `cleverly`".
  - [read] `continuous-modified-treatment-policies.md`: "`cleverly` evaluates the law through its pooled-hazard density representation".
  - [read] `docs/references.md` (Kennedy entry): Kennedy "does not establish inference for the realized learned-density target".
  - [read] Committed `tests/canonical/{lmtp_regimes,lmtp_shift,npcausal_incremental}/summary.csv` agree with the generated tables the page links.
- Proposed fix: add "the treatment mechanism (density) is supplied exactly, not estimated" to each row of the table. Files: the notebook. No study moves.

### IV-04: the intensity-law outcome is called "standardized", and it is not
- Cells: `dose-identify-heading` ("This law records a standardized score, which takes its scale from the data, so no finite support can be declared for it"), `dose-identify` (protocol `outcome="... standardized to the baseline distribution"`), and `dose-fit-heading`.
- Problem: `shift_dgp` draws Y = 1 + 0.5a + 0.25a^2 + W1 - 0.5W2 + 0.2W3 + N(0, 0.8^2). Its population mean is 3.395, and the printed head shows values up to 5.417. It is not standardized, and its scale does not come from the data. Finite support is undeclarable because the noise is Gaussian.
- Severity: **wrong**. A false statement about the data, contradicted by the cell's own output.
- Evidence: [read] `src/cleverly/datasets/synthetic.py:1815-1844` (`shift_dgp`, `noise_scale=0.8`). [recomputed] E[Y] = 3.395, consistent with the printed `ey_shift[current practice] 3.3950`.
- Proposed fix: describe the outcome as a continuous score on an unbounded scale, for example "points on a composite scale", in the protocol and in the prose. Keep the reasoning: an unbounded outcome has no declarable `q_bounds`. Files: the notebook. The callback is unaffected except for the stored protocol fingerprint, which changes.

### IV-05: the intensity units contradict each other
- Cells: `dose-data-reading` ("synthetic units with no physical zero, so 5.1% of the values are below zero") against `dose-identify` ("Intensity records in navigator contact units support consistency", "declared capacity of 5 units").
- Problem: a count of navigator contacts cannot be negative. The protocol, which is the reviewable scientific record, says contact units. A program office would read "+0.5 contacts" and "capacity 5 contacts" literally. Practicality is weak.
- Severity: **misleading**.
- Evidence: [recomputed] The population P(A < 0) is 5.6% (A ~ N(2, 1.58) marginally). The sample shows 5.1%.
- Proposed fix: call the exposure an intensity index (standardized hours, say) in the protocol, or shift the generator's dose mean so that the support is plausibly positive. Files: the notebook, and optionally `make_shift_dose`. The latter would move the MTP study's law, so the notebook-only fix is preferred.

### IV-06: "The capped policy raises no warning" is not evidence about support, and the printed assumption overstates what a constant cap secures
- Cells: `dose-fit-reading` ("The capped policy raises no warning") and the `dose-identify` output ("... is exactly what the cap is declared to secure").
- Problem: `_warn_outside_support` returns before any check whenever `cap is not None`, so any cap silences the warning, even `cap=1e6`. A constant cap u also secures g(d(a,w)|w) > 0 only when u lies inside the conditional support of A for every w. Haneuse and Rotnitzky (2013) use a covariate-dependent u(w) for this reason. The claim holds here only because the conditional law is normal (full support).
- Severity: **weak**. Literally true on this law.
- Evidence: [read] `src/cleverly/interventions/shift.py:403-406` (early return) and lines 16-19 of the module docstring ("The cap is what makes the parameter well defined without a positivity assumption nobody can check").
- Proposed fix: in the notebook, add "the library does not check a capped policy against the observed range; the support report of Step 11 is the check". In the library, reword the identification text: a declared cap secures support only when it lies within the conditional support at every w. Optionally, warn when the cap exceeds the observed maximum. A library text change alters the stored `dose-identify` output. No study result moves.

### IV-07: the ESS gap is attributed to "positivity strain"; it measures density-estimation dispersion
- Cell: `shift-support-reading`. The claim: "The estimated 40-bin density gives 65.1% ... and 24.4% ... That gap is practical positivity strain in a finite sample."
- Problem: the true-density Kish fraction exp(-delta^2) already measures the population positivity strain. The shortfall of the estimated-ratio ESS below it comes from the noisier estimated density ratio, not from the support. The sentence conflates the two.
- Severity: **weak**.
- Evidence:
  - [recomputed] The MC Kish fraction under the true ratio is 0.779 (+0.5) and 0.368 (+1.0), matching exp(-delta^2). It is 0.769 for the capped +0.5.
  - [executed] Sweep means of the estimated ESS: 0.605 (+0.5) and 0.302 (+1.0). The estimated ESS is below exp(-delta^2) on 60/60 and 54/60 seeds.
- Proposed fix: "The estimated ratio is more dispersed than the true ratio, so the fit's effective sample size is lower than the population strain alone implies." Files: the notebook.

### IV-08: "about twice as wide" is near the top of the seed range
- Cell: `dose-fit-reading`. The claim: "the `+1.0 uncapped` interval is about twice as wide as either `+0.5` interval."
- Problem: the seed shown gives 2.1. Across seeds the ratio is 1.55 to 2.12 (mean 1.77), and it exceeds 1.8 on only 28% of seeds. The callback pins `> 1.8`. The empirical SD ratio is 2.6, so the reported widths also understate the true difference (see IV-02).
- Severity: **weak**. The notebook hedges "On this draw".
- Evidence: [executed] Sweep `width_ratio_one_vs_unc`.
- Proposed fix: quote the printed widths without a generalizing multiplier, or say "wider, 0.096 against 0.046". Files: the notebook and the callback.

### IV-09: the regime intervals are about 1.5 to 1.8 times too wide, which the calibration warning explains and the page does not say
- Cells: `regime-fit-reading` and `regime-support-reading`.
- Problem: with the miscalibrated boosted propensity, the influence-curve SE overstates the sampling SD. The SE ratio is 1.80 for offer-to-all and 1.46 for the screen, and coverage is 60/60 for both. The page reads the `nuisance_models` warning without stating any consequence for the regime fit. It also says "The regime estimate is double robust", which is about consistency, not interval width.
- Severity: **weak**. No false statement, but readers learn the wrong lesson that the warning is harmless here.
- Evidence: [executed] Sweep: offer-to-all empirical SD 0.0070, mean SE 0.0125. Screen empirical SD 0.0054, mean SE 0.0079. Both biases are within MC error, +-0.001.
- Proposed fix: fold into IV-01's learner change, or add one sentence that names the cost. Files: the notebook.

### IV-10: the sensitivity step produces no usable sensitivity information
- Cell: `sensitivity` and its reading.
- Problem: the outcome axis is designed to fail whenever `q_bounds` is declared. The reading itself says the one treatment-only cell shows misclassification, not confounding. An analyst would not run or report a grid whose informative cells cannot run. The step consumes 3 refits to show a limitation that one sentence could state.
- Severity: **weak** (practicality).
- Evidence: [executed] The stored output and the `--check` run. [read] `src/cleverly/sensitivity/simulated_confounding.py:583-587` (the flip threshold matches the notebook's recomputation `latent >= inv_cdf(0.9)`, so that code is correct).
- Proposed fix: drop the outcome-strength column and state the limitation, or run the surface on a fit where the outcome axis is defined. Keep the robustness-value refusal, which is sound. Files: the notebook and the callback.

### IV-11: the `ShiftDGP` docstring states a different policy from the one it computes
- Location: `src/cleverly/datasets/synthetic.py`, `ShiftDGP` class docstring: "for d(a, w) = min(a + delta, u)". The notebook cites `shift_dgp` in `dose-data-reading`.
- Problem: `ShiftDGP.shifted` and `Shift.apply` both hold a unit at its own dose past the cap. They do not truncate at u. The two policies have different truths. The notebook's prose is correct ("keeps the current intensity"), so a reader who follows the citation finds the wrong formula.
- Severity: **weak** (library documentation).
- Evidence: [read] `synthetic.py` ShiftDGP docstring against `shifted()` (`np.where(moved > cap, dose, moved)`).
- Proposed fix: correct the docstring to the hold-at-own-dose form. Library doc only. No failing-first test is needed. No study moves.

### IV-12: implementing the incremental policy needs an estimated propensity, and the protocol does not say so
- Cell: `incremental-identify` (rationale "The program can run a randomized offer at the doubled odds") and its reading.
- Problem: to double each patient's odds, the program must know the current g(W). In practice it would use an estimated g, so the implemented policy is q built from g_hat, not the estimand q built from g. A regional plan's analyst should be told this. The notebook says only that "the observed mechanism defines it".
- Severity: **weak** (practicality).
- Evidence: [read] `docs/references.md`, Kennedy entry ("does not establish inference for the realized learned-density target").
- Proposed fix: add one table row or sentence. Files: the notebook.

### IV-13: "The office needs that number to budget navigator hours"
- Cell: `data-reading`.
- Problem: hours depend on how many patients are offered (about 51% flagged), not on the mean gain. The gain informs the value per hour.
- Severity: **weak**.
- Evidence: [read] The printed group counts, 1521/3000 flagged.
- Proposed fix: "The office needs that number to weigh the gain against the hours an offer to about half the patients costs." Files: the notebook.

## Checked and sound
- [recomputed] ATE 0.1629 (printed 0.163), screen-vs-none 0.1055 (0.105), share kept 0.6475 (0.648), double-odds IPSI contrast 0.02338 (0.023). Shift contrasts 0.7759 / 0.8123 / 1.7495 (MC) and closed forms 0.8125 and 1.75 match the printed 0.7761 / 0.8125 / 1.7500. E[Y] = 3.395.
- [recomputed] "the effect of an offer grows with `discharge_risk`": E[tau | W1] rises from about 0.08 (W1 < -1) to about 0.26 (W1 > 2). It is flat in the lower tail.
- [recomputed] Cap holds back 2.3% in the population (2.5% printed). P(A < 0) is 5.6% in the population (5.1% sample).
- [recomputed] The Kish fraction under the true ratio is exp(-delta^2): 0.779 and 0.368 by MC.
- [recomputed] Failure mechanism probe: with the cap removed (cap = 100) the population gap is 0. A ceiling inside the bulk separates the policies further: cap 4 gives 0.156, cap 3 gives 0.396, cap 2.5 gives 0.531. The reading's claim holds.
- [recomputed] Truncation 0.0114 = 5/(sqrt(n) ln n) at n = 3000. 1/0.0114 = 87.7, so "about 88" is right.
- [read] The MTP positivity statement (g(d(a,w)|w) > 0 wherever g(a|w) > 0) is the standard condition. The capped clever covariate in `shift.py:_ratio` is the correct pushforward ratio.
- [read] The IPSI definition matches Kennedy (odds of q_delta = delta x odds of g). The clever covariate lies in [1/delta, delta] = [0.5, 2]. "No positivity assumption" is correct (Kennedy abstract: "avoid positivity assumptions entirely"). "Not doubly robust" matches the library's remainder derivation. I could not extract the text of Kennedy's paper to confirm its exact locator for the non-DR statement.
- [read] The Kennedy citation is the published JASA version: 114(526):645-656, DOI 10.1080/01621459.2017.1422737.
- [executed] Sweep relations that hold on most or all seeds: 2 PositivityWarnings (60/60); the `nuisance_models` warning on regime and incremental fits (60/60); shift `needs attention` empty (60/60); offer-to-all has the smallest ratio ESS (57/60) and the largest ratio (56/60); the gap interval excludes zero (60/60); the gap SE is below half of each contrast's SE (53/60).
- [read] Off-plan rows equal zero-load rows. This is deterministic by construction.
- [read] Every effective seed is set: generator seeds 31 and 32, learner `random_state`, and `Runtime(random_state, n_jobs=1)`. No deprecated parameters. The only warnings shown are the intended PositivityWarnings.
- [read] The registered-study pages agree with the committed `summary.csv` artifacts. "320-bin density, +0.25 and capped +0.5 only" and "targets in sample" are correct for the MTP study.
- [read] The robustness-value refusal reading follows the house rule ("not implemented", not "not defined"). The Riesz-representer claim for a deterministic regime mean is correct.

## Patterns for siblings
1. **"Sampling variation" readings of a miss.** The house rule in `example-notebooks.md` defaults to this explanation. Here the miss is systematic, with coverage 0.48. Any sibling that narrates a miss should be seed-swept before the explanation stands.
2. **Default `HistGradientBoostingClassifier` propensity on `navigation_data` (n = 3000)** triggers the calibration warning on every seed. It inflates DR/regime SEs (SE ratio 1.5 to 1.8) and biases g-dependent estimands. Siblings that use the same learner on the same law (point-treatment, cross-fitting, survey, DR) probably show the over-wide intervals.
3. **"How far to trust" tables omit "nuisance supplied exactly".** Several registered studies supply g or the density exactly. A notebook that estimates them inherits no validation for that step.
4. **Callbacks pin seed-specific coverage relations**, such as `not covers` and width multipliers. They certify the stored draw, not the method.
5. **The absence of a library warning is read as a passed check.** The shift positivity warning is skipped for any declared cap.
6. **Protocol text that contradicts the synthetic law**: "standardized" outcome, "contact units" with negative values. Check each protocol field against the generator.
