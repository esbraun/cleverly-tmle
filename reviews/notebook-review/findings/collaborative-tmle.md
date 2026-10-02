# Review: `docs/examples/collaborative-tmle.ipynb`

| item | value |
| --- | --- |
| notebook | `docs/examples/collaborative-tmle.ipynb` (callback `tests/unit/tutorial_semantics/collaborative_tmle.py`) |
| commit | `5f33902b` (branch `agent/notebook-review`), `cleverly.__file__` verified under this worktree's `src` |
| `--check` | exit 0, wall 9 s. "every cell reproduced its stored non-image output" (`.tmp/notebook-review/collaborative-tmle/check.log`) |
| seed sweep | `.tmp/notebook-review/collaborative-tmle/sweep.py` (seeds 0-299, n = 2,000, the notebook's exact four fits plus OLS/HC0), summarized by `analyze.py` into `analyze.log`; raw rows `sweep_0_300.csv` |
| mechanism probe | `.tmp/notebook-review/collaborative-tmle/mechanism.py` (150 seeds, fixed candidate g sets, no search) -> `mechanism.log` |
| truth / refusal probe | `.tmp/notebook-review/collaborative-tmle/probe_crossfit.py` -> `probe_crossfit.log` |

Counts: wrong 0, misleading 5, weak 5.

## Findings

### CT-01, cells `estimate-heading`, `protocol-reading`, `data-reading` (and protocol in `protocol`)

> "A standardized score takes its scale from the data, so the analyst can declare no finite support
> for it." / "This page's score is standardized, so it has no known finite support to declare ...
> Fitting in sample is therefore the honest choice rather than a workaround." / protocol outcome:
> "Patient-reported transition score, standardized to the baseline distribution"

- Problem: Standardizing to a fixed baseline distribution is an affine map. A patient-reported
  instrument with a bounded raw scale (every item-summed score is) stays bounded after it, with a
  support the analyst can compute from the published range and the baseline mean and SD. The
  page's central reason for fitting in sample is therefore false for the applied setting it names.
  The generated data also are not standardized: the score has mean 1.50 and SD 2.13 (control-arm
  mean 0.52), so the protocol field does not describe the data the page fits. The in-sample fit is
  a consequence of a library refusal (which also refuses `Targeting(fluctuation="linear")`), so
  "rather than a workaround" overstates it.
- Severity: misleading.
- Evidence: [recomputed] 10^6 draws from the structural equations: Y mean 1.497, SD 2.135.
  [executed] `probe_crossfit.log`: `CrossFitting(n_folds=5)` without `q_bounds` is refused with
  `CapabilityError` for `TMLEMethod` and `CollaborativeTMLEMethod`, under both `logistic` and
  `linear` fluctuation. [read] `src/cleverly/datasets/synthetic.py:908-914` (Y = 1 + A + 1.5 W1 +
  0.8 W3 + N(0, 1), no standardization).
- Proposed fix: State the real premise. For example: "This synthetic score has no documented
  range. A raw patient-reported score usually has one, and then the analyst declares `q_bounds`
  and cross-fits. Here no bound exists, `cleverly` refuses a cross-fitted fit without one, and the
  page fits in sample." Change the protocol outcome to "Patient-reported transition score on a
  scale with no documented bounds" (or similar) and drop "standardized" and "honest choice rather
  than a workaround". Files: the notebook only (callback asserts `changed_fields` and the
  in-sample summary, both unaffected). Not a library bug. No registered study can move.

### CT-02, cells `selection-reading`, `failure-reading`, `control-heading`

> "With a correctly specified outcome model, Step 7 showed the selector stop at a variable that does
> not move assignment." / "The search kept it, and it left out both the confounder and the
> instrument." / "Its g holds one variable that does not move assignment."

- Problem: The page notes the near-tie, but never says how unstable the selection is, and Step 9
  restates the draw's selection as what the selector does. For the page's own question ("which
  approved variables belong in the assignment model?") that instability is the answer a reader
  needs.
- Severity: misleading.
- Evidence: [executed] 300 seeds, notebook configuration (`analyze.log`): selected g empty 55.3%,
  `social_support` alone 25.0%, any set containing `baseline_readiness` 18.0%, any set containing
  `queue_lottery_draw` 2.7% (8 seeds; those fits have ~6% of g in each tail). |cv risk(k=1) -
  cv risk(k=0)| < 1e-3 on 64.7% of seeds. The first greedy step is `social_support` on 274/300.
- Proposed fix: Add after the near-tie paragraph: "Over 300 draws of this law, the selected g was
  empty on 55%, `social_support` alone on 25%, contained the confounder on 18%, and contained the
  draw on 3%." Reword Step 9's opener to "On this draw, Step 7 showed ...". Notebook only; not a
  library bug; no study moves.

### CT-03, cell `control-reading`

> "In this known law, the search keeps the variable that the constant outcome model needs. It drops
> the pure assignment predictor."

- Problem: Stated as a property of the law, but it holds on most seeds, not all. The technical
  entry and the `ctmle.py` module docstring say "in every seed", and those seeds are three
  (`tests/e2e/test_ctmle.py:741`, `SEEDS = (0, 1, 2)`, n = 1,500).
- Severity: misleading.
- Evidence: [executed] 300 seeds, constant Q, greedy C-TMLE: `baseline_readiness` kept 99.7%,
  `queue_lottery_draw` left out 91.7% (25/300 seeds include it), `social_support` kept 99.0%.
  [read] `docs/technical-reference/collaborative-tmle.md:154`,
  `src/cleverly/estimators/ctmle.py:222-224`.
- Proposed fix: "On this draw the search keeps ... Over 300 draws it kept the confounder on all
  but one and left the draw out on 92%." In the technical entry and the module docstring, replace
  "in every seed" with "on all three e2e seeds (n = 1,500)". Files: notebook,
  `docs/technical-reference/collaborative-tmle.md` (prose ledger refresh needed),
  `src/cleverly/estimators/ctmle.py` docstring. Not a library bug; no study moves.

### CT-04, cells `collaborative-reading`, `failure-reading`, `sensitivity-reading`, `trust`

> "the curve behind that value can understate the spread" / "On this draw, the plug-in value is
> smaller than that robust value." / "The plug-in diagnostic can understate the spread."

- Problem: In this law with a correct Q the understatement is systematic, not a possibility or a
  feature of one draw. A near-constant g makes the plug-in converge to sigma * sqrt((1/p + 1/(1-p))/n),
  while the estimator (about the OLS coefficient) has variance sigma^2 / (n E[Var(A|W)]), and
  Var(A|W) < p(1-p) whenever W predicts A. The HC0 standard error, by contrast, is calibrated for
  the C-TMLE estimate on this law.
- Severity: misleading (true wording that invites "this happened to be smaller here").
- Evidence: [recomputed] analytic limits at n = 2,000 from 10^6 draws: plug-in 0.0447, OLS SD
  0.0542 (ratio 0.82). [executed] 300 seeds: mean plug-in SE 0.0452 against empirical SD 0.0534
  (ratio 0.847), plug-in interval coverage 0.893; plug-in < HC0 on 98.7% of seeds; HC0 against
  the C-TMLE estimate: ratio 1.014, coverage 0.950. This agrees with the RM12 probe the evidence
  page quotes (ratio 0.844, coverage 0.92).
- Proposed fix: In `sensitivity-reading` replace "On this draw, the plug-in value is smaller" with
  "The plug-in value is smaller. In this law that holds on almost every draw, because a g that
  barely moves ignores the adjustment the outcome regression performs. Over 300 draws the plug-in
  interval covered 89%." In the trust section, "understates the spread in this law" instead of
  "can understate". Notebook only; no study moves.

### CT-05, cell `control-reading`

> "The selected g also leaves out a variable of the true assignment mechanism, so the trust
> section's limit applies to this ratio. A ratio below one is not a precision gain here, because the
> numerator carries no coverage claim."

- Problem: In the constant-Q configuration the plug-in errs in the other direction (conservative),
  so "the trust section's limit" (understatement) does not describe this ratio. And the selected
  fit is, in repeated sampling, much more precise than the plain fit; "not a precision gain" reads
  as a denial of a gain that the sweep shows is larger than the printed ratio suggests.
- Severity: misleading.
- Evidence: [executed] 300 seeds, constant Q: C-TMLE plug-in SE mean 0.0925 against empirical SD
  0.0757 (ratio 1.22, plug-in coverage 0.977); plain empirical SD 0.2024, RMSE 0.227, against
  C-TMLE RMSE 0.077. Empirical SD ratio 0.37 against the printed 0.394.
- Proposed fix: "The ratio is 0.394. It compares a diagnostic with a standard error, so it does
  not measure the precision gain. Over 300 draws of this law the selected fit's spread was 0.076
  against 0.202 for the plain fit, and in this configuration the plug-in value overstated the
  spread (ratio 1.22). The direction of the plug-in error depends on the configuration." Notebook
  only; no study moves.

### CT-06, cell `failure-reading`

> "Do not read the smaller plug-in value as a precision gain."

- Problem: Correct as a caution about the number, but the page never says whether leaving the
  draw out actually helps, which is the method's selling point. It does, by less than the plug-in
  suggests.
- Severity: weak.
- Evidence: [executed] 300 seeds: empirical SD 0.0682 (plain) against 0.0534 (C-TMLE), RMSE
  0.0681 against 0.0533; C-TMLE closer to the truth on 63.7% of seeds. Mechanism probe
  (`mechanism.log`, 150 seeds, fixed g, no search): g on {W1, W2, W3} SD 0.0728, g on {W1, W3} SD
  0.0580, g on {W1} SD 0.0579, identical bias (-0.005).
- Proposed fix: Add one sentence: "Over 300 draws the spread of the estimate was 0.068 for the
  plain fit and 0.053 for the collaborative fit. Removing only the draw from g gives the same
  reduction." Notebook only.

### CT-07, cell `control-reading` (registered-study paragraph) and `trust` table

> "The registered study repeats this control over many draws."

- Problem: The study's `selector_necessity` cells run on the bounded (beta-outcome) twin of the
  instrument law, at n = 1,500, cross-fitted with five outer folds, with 800 replicates. They score
  bias and RMSE, not which covariates are kept. "Repeats this control" implies the same law,
  configuration, and selection outcome. The study's red free-selector bias (standardized 0.217)
  also reproduces at this page's size, which the page could say.
- Severity: weak.
- Evidence: [read] `tests/studies/ctmle_selector_properties.py:94-119,138-160` (`instrument_dgp`,
  `DummyRegressor`, `bounded_twin`, `cross_fit=True`, `n_folds=5`); `tests/canonical/ctmle_selector/properties.csv`
  rows 13-14 (n 1500, 800 replicates; empty_control bias 0.0495-0.0511 vs margin 0.0022;
  collaborative bias CI upper 0.0037 vs margin 0.0030; RMSE ratio 0.241). [executed] constant-Q
  C-TMLE bias on this page's law +0.0160 (MCSE 0.0044, 0.21 empirical SD).
- Proposed fix: "The registered study runs a similar control on a bounded twin of this law,
  cross-fitted at n = 1,500 over 800 draws. It scores bias and error, not the selected set."
  Optionally add the +0.016 bias measured here. Notebook only; no study moves.

### CT-08, cell `stress-control` / `control-reading`

> "constant Q, plain psi= 1.077 se=0.2343 CI=(0.618, 1.537)" used as the reference standard error
> of the ratio, and "The plain interval ... contain 1.000."

- Problem: The plain constant-Q comparator is itself biased and under-covers at this size, which
  the page does not mention while using its standard error as the denominator of a headline ratio.
- Severity: weak.
- Evidence: [executed] 300 seeds: bias +0.104 (MCSE 0.012, about half an SD), SE ratio 1.105,
  coverage 0.887. I did not isolate the cause (candidate: truncation at 0.0147 with about 1.6% of
  rows truncated and extreme inverse weights); not verified.
- Proposed fix: One sentence: "The plain constant-Q fit is itself biased on this law (about +0.10
  over 300 draws), so it is a stress comparator, not a reference." Notebook only. If the bias is
  confirmed to come from truncation, it is a known property, not a library bug.

### CT-09, `tests/unit/tutorial_semantics/collaborative_tmle.py:120`

> `# "The plain g has an AUC of 0.843, so it predicts the offer well."`

- Problem: The quoted token is stale. The notebook prints and narrates 0.844. The assertion
  (`> 0.8`) is unaffected.
- Severity: weak.
- Evidence: [read] stored output `plain TMLE propensity AUC: 0.844`; [executed] AUC range over 300
  seeds 0.807-0.861, so the `> 0.8` assertion is seed-robust.
- Proposed fix: Update the comment to 0.844. Callback only.

### CT-10, `src/cleverly/methods.py:798` (library docstring the page's Step 6 configuration uses)

> "No strategy reports a confidence interval, a p-value or a standard error." ... Notes: "For an
> interval, use `strategy="oat"` or :class:`TMLEMethod`."

- Problem: The `CollaborativeTMLEMethod` docstring contradicts itself. `oat` reports no interval
  either (technical entry, "The outcome-adaptive path publishes no inference either").
- Severity: weak (outside the notebook, but on the class the notebook teaches).
- Evidence: [read] `src/cleverly/methods.py:765-767` and `:798-799`;
  `docs/technical-reference/collaborative-tmle.md` "outcome-adaptive path" paragraph.
- Proposed fix: "For an interval, use :class:`TMLEMethod`." Docstring only; not a numerical bug.

## Checked and sound

- [recomputed] Truth: 10^6 counterfactual draws from my own implementation of the structural
  equations give ATE = ATT = 1.000 exactly (constant effect), matching the printed `ate/att/atc`.
- [recomputed] Unadjusted bias is driven by readiness alone: 1.5 x (0.321 + 0.288) + 0.8 x (0.042
  + 0.035) = 0.975, plus the effect 1 gives 1.975 against the printed 1.977. The draw gap adds
  nothing, as the `association-reading` says.
- [read] Roles: W1 in both equations (0.8, 1.5), W2 in assignment only (1.5), W3 in outcome only
  (0.8) (`synthetic.py:908-914`). "The draw moves assignment more strongly than readiness" holds
  (1.5 against 0.8). "Assignment logit is linear in the three declared covariates" holds.
- [executed] Step 8 mechanism: removing only W2 from a fixed g cuts the empirical SD from 0.0728
  to 0.0580 with unchanged bias. "An instrument in g ... precision falls" is the true mechanism.
- [executed] A `discrete` C-TMLE whose candidate is the full set is numerically identical to plain
  TMLE on 150/150 seeds (matches the technical entry's identity claim).
- [executed] Plain-fit tails: share below 0.1 and above 0.9 both > 0.05 on 300/300 seeds; plain
  AUC 0.807-0.861. Collaborative tails both zero on 92.3%; ESS ratio > 0.99 on 80.3% (the reading
  is phrased about this draw, which is correct).
- [executed] `plugin_std_error < plain std_error` on 99.3% of seeds; constant-Q ratio < 0.5 on
  76.7% (mean 0.433). Both are this draw's readings and the page frames them as such.
- [executed] "The regression gives the same estimate": |OLS - C-TMLE| < 5e-4 on 100% of seeds
  where g is `social_support` alone, 81.7% overall. The FWL/HC0 formula in `sensitivity` is the
  correct HC0 for the offer coefficient (HC0 vs OLS empirical SD ratio 1.012, coverage 0.953).
- [executed] Cross-fitting refusal without `q_bounds` is real, for both methods and both
  fluctuations (the rationale around it is CT-01).
- [read] Untestable assumptions: no unmeasured confounding and the exclusion restriction are both
  named as untestable ("The data cannot check it", "The data cannot establish this exclusion
  restriction"). No diagnostic is presented as checking them.
- [read] C-TMLE scope (`ate`, `ey`, `ey1`, `ey0`, `rr`, `or`) matches the technical entry; the ATT
  refusal text matches the stored output.
- [read] Greedy description (one extra targeting step, then the best addition, `risk` can rise)
  matches `src/cleverly/estimators/ctmle.py:35-46,378-384`. [executed] `train_risk` rises
  somewhere on 99.3% of seeds.
- [read] Jensen argument in `sensitivity-reading` (conditional expectation of the representer has
  a smaller second moment) is correct. The cited unit test exists and pins `nu2` and `max_bias`
  below the plain values (`tests/unit/test_omitted_variable_refusals.py:688-722`).
- [read] Registered study numbers quoted on the page (empty-path bias outside margin, RMSE ratio
  0.241 "well below", free-selector cell red) match `properties.csv` rows 13-14 and the evidence
  page. "Agreement rows use a binary-outcome law" matches.
- [executed] Seeds: generator `seed=44`, `Runtime(random_state=44)`, `LogisticRegression(random_state=44)`;
  `LinearRegression` and `DummyRegressor` are deterministic. No warnings in stored outputs; no
  deprecated parameters.
- [read] Literature: the notebook itself cites no paper. It routes sources to the technical entry,
  whose references point to the published versions (van der Laan and Gruber 2010, IJB, DOI
  10.2202/1557-4679.1181; Ju et al. 2019 SMMR). I did not re-verify theorem locators in those
  papers.

## Patterns for siblings

- Draw-specific selection or ranking stated as a property of the law ("the search keeps X and drops
  Y"). Any notebook that narrates a selector, a stopping index, or a winner needs a seed-sweep
  share beside it. The technical entry and `ctmle.py` say "in every seed" from three seeds.
- Directional claims about a diagnostic standard error ("can understate"), presented per draw when
  the direction is systematic, and applied to a configuration where the direction reverses
  (CT-04, CT-05). Siblings that print `plugin_std_error` (outcome-adaptive C-TMLE, DR-TMLE pages
  that print diagnostic spreads) should state the measured direction and size for their own law.
- "Not a precision gain" wording that denies a real repeated-sampling gain while it means "this
  number does not measure the gain".
- In-sample fits justified by "the outcome has no support to declare". Any page on
  `navigation_protocol()` that switches to a continuous score should check whether the outcome
  really is unbounded, and whether the generated data match the protocol's outcome description.
- Constant-Q stress comparators used as a reference denominator without checking their own bias
  and coverage.
- Callback comments that quote stored numbers drift from the stored output (0.843 vs 0.844); the
  numbers in comments are not checked by the gate.
