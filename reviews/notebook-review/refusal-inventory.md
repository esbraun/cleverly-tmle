# Refusal inventory for the ten example notebooks

Rule from the user: the examples show end-to-end workflows that work. A failure-mode demonstration
that runs and returns numbers stays. A cell that calls something to print a `CapabilityError`,
`DataError`, or `ValueError`, prints a capability list with `available=False`, or narrates
"`cleverly` refuses X" as a lesson goes.

Evidence tags: **[executed]** means a probe ran at the notebook's size, seed, and learners on
`origin/main` 5f33902b (`cleverly.__file__` asserted under the worktree `src`). The probe executed
the notebook's own code cells up to the named cell (`.tmp/notebook-review/refusals/nbrun.py`), then
ran the replacement. **[read]** means source or stored output only. Scratch probes and logs are in
`.tmp/notebook-review/refusals/`.

Prose-only mentions (class P below) are not showcases. They still read as refusal lessons. Rewrite
each as the positive requirement, instruction first. For example, write "Declare `q_bounds` for a
cross-fitted continuous outcome", not "`cleverly` refuses a cross-fitted fit that declares none".

## Cross-cutting finding: the negative doubly robust nu² comes from the boosted propensity

[executed] `pt.py`, `pt2.py`, `pt3.py`; logs `pt_seeds*.txt`. `navigation_data(n=3000)`, five
folds, `q_bounds=(0, 1)`, and the boosted outcome learner of the notebook. Seeds 1 to 20 share the
same 20 draws across rows.

| propensity learner | default `robustness_value()` works | reported SE / empirical SD of psi | CI covers truth | seed 21 | seed 34 |
| --- | --- | --- | --- | --- | --- |
| `HistGradientBoostingClassifier(random_state=s)` (notebook) | 1 of 20 (nu² from -0.16 to -12.2) | 1.34 | 20 of 20 | refused, -7.96655 | refused, -9.03194 |
| HGB `max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0` | 20 of 20, rv 0.383 to 0.454 | 0.71 | 17 of 20 | rv 0.444, rva 0.424 | rv 0.407, rva 0.385 |
| HGB `max_depth=3, min_samples_leaf=100, learning_rate=0.05, max_iter=150` | 20 of 20, rv 0.388 to 0.460 | 0.70 | 17 of 20 | not run | rv 0.405, rva 0.384 |
| `LogisticRegression(max_iter=1000)` | 20 of 20, rv 0.399 to 0.466 | 0.65 | 17 of 20 | rv 0.455, rva 0.436 | rv 0.420, rva 0.401 |
| degree-2 polynomial logistic | 20 of 20, rv 0.351 to 0.521 | 0.70 | 16 of 20 | rv 0.415 | rv 0.374 |
| degree-2 polynomial logistic, with a regularized HGB outcome learner | 20 of 20 | 0.64 | 16 of 20 | not run | not run |

Each fit takes 1.2 to 3 seconds. The calibrated learners fix the refusal on every seed, and the
propensity calibration slope moves to about 1.0. Seed 21 gives 1.0095 (SE 0.060), AUC 0.688, and
`nuisance fits look reasonable`.

**Flag.** Every calibrated propensity halves the interval width (mean SE about 0.0057 against 0.0112).
On these 20 draws, the reported SE is 0.64 to 0.71 of the empirical SD of psi (0.0080 to 0.0091).
The overfit boosted g gives 1.22 to 1.34. Twenty shared draws give the SD a relative error of
about 16%, so this is a signal, not evidence. Before a tutorial adopts a calibrated propensity, run
a coverage check on this law with at least 200 seeds. The registered stacked CV-TMLE study uses GLM
learners and ten folds, so it does not cover this configuration. If the check confirms
under-coverage, the wide boosted interval was hiding an outcome-side problem. The tutorial then
needs a different outcome learner, not only a different propensity.

## point-treatment-tmle (data seed 21)

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `sensitivity` (code) | calls `robustness_value()` to print the doubly robust nu² refusal (-7.96655). Then it uses `nu2_estimator="plugin"` and prints the `ci_lower` refusal (`limit_refusal`) | Step 9 flags the propensity, so refit once with the regularized HGB propensity and the same folds and seed. Then run the default (doubly robust) `robustness_value()`, both benchmarks, and `omitted_confounding(..., rho=1.0)` on the refit, and print `bounds.ci_lower` / `ci_upper`. Do not use `nu2_estimator` or `try`/`except` | [executed] `pt.py nb` | refit psi 0.1706, CI (0.1602, 0.1810), truth 0.163. rv 0.444, rva 0.424. `medication_burden` benchmark cf_y 0.1194, cf_d 0.0855. Bounds at rho=1 are (0.140, 0.201), with limits (0.131, 0.210). The `discharge_risk` benchmark has cf_y 1.0000, the same ceiling the page reads now. 9.9 s with both benchmarks and `assess()` |
| `sensitivity-heading` (md) | "The code first calls the robustness value with that default, which refuses here" | say that the step refits with a calibrated propensity, and say why: Step 9's slope of 0.46 | [read] | none |
| `sensitivity-reading` (md) | paragraphs 1, 2, and 7 narrate the refusal and the plug-in qualification | delete them. Read rv, rva, the benchmarks, and the one-sided limits of the refit. Give one sentence on the cost: the refit's interval is narrower, which the coverage flag above qualifies | [read] | none |
| `assessment-reading` (md) | "Step 10 reads the refusal" | "Step 10 runs the omitted-variable analysis on a refit" | [read] | none |
| `trust` (md) | sensitivity row: "...or that the plug-in nu² of Step 10 belongs to the treatment law" | name the refit's calibrated propensity instead | [read] | none |
| `estimate-heading` (md), class P | "...and `cleverly` refuses that fit" | "Declare `q_bounds` whenever you cross-fit a continuous outcome" | [read] | none |

Alternative: put the regularized learner in the Step 6 fit itself. That removes the calibration
lesson of Step 9, so the refit route keeps more of the page. Callback:
`tests/unit/tutorial_semantics/point_treatment_tmle.py` lines 143 to 180 (`nu2_refusal`,
`assert_plugin_limits_refuse`) need rewriting.

## cross-fitting (data seed 34)

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `reuse-split` (code) | applies the plan to `navigation_data(seed=35)` and prints the `DataError` | delete the other-data block. Use the plan for its stated purpose, a comparison of two methods on the same folds: the boosted fit and a fit with a calibrated propensity. This also builds the model that Step 13 uses | [executed] `p_cf.py` | regularized HGB g on `plan`: fold fingerprint `c7f4eafaaf319249` (equal to Step 5), psi 0.15633, SE 0.00561, CI (0.14534, 0.16732), calibration slope 0.980, AUC 0.698, no attention items, 2.5 s. Logistic g: psi 0.15530, CI (0.14494, 0.16566), 1.5 s |
| `reuse-split-reading` (md) | the `refused on other data` table row, and "A hand-built assignment ... `CrossFitting` refuses it" | delete both. Keep "Take the plan from a result". Say once, positively, that a plan belongs to the data it was drawn on | [read] | none |
| `sensitivity` (code) | prints the default nu² refusal (-9.03194), then uses the plug-in and prints the `ci_lower` refusal | run the default `robustness_value()` and `omitted_confounding()` on the Step 9 calibrated fit, and print the limits | [executed] `p_cf.py` | regularized HGB: rv 0.407, rva 0.385. Default strengths 0.03 give bounds (0.147, 0.165) and limits (0.138, 0.175). Logistic: rv 0.420, rva 0.401, bounds (0.147, 0.164), limits (0.138, 0.173) |
| `sensitivity-heading`, `sensitivity-reading` (md) | narrate both refusals | rewrite around the refit. Keep the "cross-fitting does not make sensitivity unnecessary" paragraph | [read] | none |
| `assessment-reading` (md) | "Step 13 reads the refusal" | "Step 13 runs the analysis on the calibrated fit" | [read] | none |
| `team-clusters-reading` (md), class P | "`ci`, `pvalue`, and `std_error` raise `CapabilityError`" | "Declare at least 40 teams for an interval. Fewer teams give a point estimate only" | [read] | none |
| `team-folds` (code), borderline | fits 4 teams and 60 rows, which return a point estimate and no interval (no exception is printed) | keep the folds-per-team count. Drop the 4-team fit, or keep only its fold-reduction warning and leave out the "no interval" sentence | [read] | none |
| `estimate-heading`, `constructions-heading`, `repeated-folds-heading`, `repeated-folds-reading` (md), class P | "refuses a cross-fitted fit...", "lists what each one refuses", "a repeated fit refuses a simultaneous band" | state the requirement or the scope instead | [read] | none |

Callback: `cross_fitting.py` lines 113 to 125 (the `refusal` `DataError`) and lines 221 to 238
(`nu2_refusal`, `assert_plugin_limits_refuse`). Once both tutorials drop the helper, delete
`assert_plugin_limits_refuse` from `tests/unit/tutorial_semantics/__init__.py` lines 45 and 162 to
200. The anchor `point-treatment-tmle.ipynb#step-10-sensitivity-what-the-fit-cannot-check` in
`cross-fitting.ipynb` must follow any heading change. The anchor
`cv-tmle.md:430 -> cross-fitting.ipynb#step-9-reuse-the-same-outer-split` must keep the Step 9
heading.

## collaborative-tmle (data seed 44)

**Answer to the key question.** No `CollaborativeTMLEMethod` configuration reports an interval. The
greedy, ordered, and discrete paths take `working_mechanism_plugin` (F18). `oat` takes
`generated_design_plugin` (F19) [executed `p_ctmle2.py`: `oat` `.ci` raises `CapabilityError`]. The
bootstrap prints `bootstrap sd` and a `percentile range` as diagnostics only
(`collaborative-tmle.md` lines 278 to 283).

**Docstring defect.** The Notes section of `CollaborativeTMLEMethod` at
`src/cleverly/methods.py:787` says "For an interval, use `strategy="oat"` or `TMLEMethod`". The
`strategy` parameter text, the reference, and the executed probe all show that `oat` reports no
interval. Fix the Notes to "For an interval, fit `TMLEMethod`".

**Working workflow.** Report C-TMLE as a point estimate with its selection path. Take the reported
interval from the plain TMLE fit on the declared adjustment set, which the page already fits in
Step 8. Run sensitivity analysis on both fits: the omitted-variable bound on the plain fit, and the
simulated stress surface on the collaborative fit.

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `identify` (code) | prints every method with `available=False` and its reason, then the ATT catalog refusal | print `effect.summary(...)` and only the available names: `[m.name for m in effect.available_methods() if m.available]`. Drop the ATT lookup | [executed] `p_ctmle.py` | `tmle`, `collaborative_tmle`, `drtmle` |
| `identify-reading` (md) | the last paragraph on the two unavailable methods and the ATT refusal | one sentence: `collaborative_tmle` is listed for this ATE. Point to the technical entry for the scope | [read] | none |
| `title` (md) | learn row "check that a method is available before you fit it" | keep it. It is now a positive check | [read] | none |
| `collaborative` (code) | `try: point.ci` / `except CapabilityError` prints `inference_refusal` | delete the `try` block. Print psi, `point.inference`, and the plug-in pair under its diagnostic labels, as the cell does now | [executed] prefix run | psi 0.955, plug-in SE 0.0440, plug-in (0.869, 1.041) |
| `collaborative-reading` (md) | the `.ci` / `refused` table row, and "The refusal states the reason" | replace the row with "`plain TMLE` interval (Step 8): the interval to report". Keep the reason as a property of the fit: "the fit reports no interval, because no result gives the influence curve after selection (F18)" | [read] | none |
| `failure-reading` (md) | "That is the reason `cleverly` refuses the interval rather than printing it" | "Report the plain TMLE interval, (0.758, 1.004), which covers the declared adjustment set" | [read] | plain psi 0.881, SE 0.0626 |
| `sensitivity` (code) | `collaborative.sensitivity.elements(...)` refusal (`bound_refusal`) | delete the `try` block. Keep the plain-fit rv/rva row and the HC0 comparison. Add `collaborative.sensitivity.simulated_confounding(estimand="ate", grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.5)))` | [executed] `p_ctmle2.py` | four cells and 0 failures in 0.6 s. Outcome 0.5: 1.0232 (+0.0683). Treatment 0.1: 0.63882 (-0.3162). Both: 0.60972 (-0.3453). Induced association -0.0480 and -0.0252. The plain rv 0.227 and rva 0.204 are unchanged |
| `sensitivity-heading`, `sensitivity-reading` (md) | narrate the refusal. The Jensen argument and the unit-test pointer explain it | delete the refusal narration and the unit-test paragraph. Keep one sentence: the omitted-variable bound is read on the plain fit, because its nu² comes from the full assignment model. Read the surface as qualitative movement, as its footer says | [read] | none |
| `assessment-reading` (md) | "Step 11 shows the refusal and its reason" | "Step 11 runs the sensitivity analysis on both fits" | [read] | none |
| `trust` (md) | "`cleverly` therefore refuses `.ci`, `.pvalue`, and `.std_error`...", and the table row "...which `cleverly` refuses" | "This fit reports a point estimate and two plug-in diagnostics. Report the Step 8 TMLE interval". Table row: the surface establishes movement under one latent cause on the collaborative fit, and not a bound | [read] | none |
| `where-next` (md), class P | "The library refuses longitudinal and incremental-target C-TMLE" | "C-TMLE in `cleverly` covers point-treatment targets. The technical entry lists them" | [read] | none |
| `estimate-heading` (md), class P | "`cleverly` refuses a cross-fitted fit of it" | "A cross-fitted fit needs a declared support, and this score has none, so the page fits in sample" | [read] | none |

Option, not recommended: a protocol-declared TMLE without `queue_lottery_draw` reports psi 1.0122
and CI (0.921, 1.104), with rv 0.381 [executed]. It relies on the exclusion restriction that the data
cannot check. The plain fit stays valid if the draw is a confounder, so it is the safer reported
interval.

Callback: `collaborative_tmle.py` lines 64 to 66 (unavailable methods), 91 to 100
(`inference_refusal`), and 158 to 173 (`bound_refusal`). The surface needs a new relation pin, for
example that the anchor cell equals `point.psi` and that the movement signs hold.

## dr-tmle (data seed 55)

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `identify` (code) | prints all five methods with `False` entries, then the ATT catalog refusal | print the available names only. Drop the ATT lookup | [read] | `tmle`, `collaborative_tmle`, `drtmle` |
| `identify-reading` (md) | the catalog-outcome table (available / unavailable / refused at fit time) | delete the table. Keep "DR-TMLE relaxes none of the four assumptions" | [read] | none |
| `sensitivity` (code) | prints five `unavailable` ledger rows and the `robustness_value` refusal | `print(guarded.sensitivity.evalue())`. It reads only the estimate and its interval, so it does not use the doubted g. Optional: a treatment-only `simulated_confounding` grid. `q_bounds` is declared, so an outcome axis would fail, as in interventions | [executed] `p_dr.py` | RR 1.888 [1.793, 1.988] (approximate conversion, sd(Y) = 0.2345). E-value 3.1824 for the point and 2.9853 for the limit. Under 0.1 s. The ordinary fit gives 3.2037 / 3.0030 |
| `sensitivity-heading`, `sensitivity-reading` (md) | narrate the refusal and the nu² direction-of-error argument | read the E-value. Name the conversion chain that the output prints (SMD, Chinn's log OR / 1.81, the square root for a common outcome). Say that it is approximate. Say that it does not use the fitted propensity, which is why it suits this page | [read] | none |
| `title` (md) | learn row "...and why `cleverly` refuses the omitted-variable bounds here" and intro "holds ... the refusals" | "say which assumption DR-TMLE does not relax, and read an E-value for it". The intro names "the scope" | [read] | none |
| `assessment-reading` (md) | "Step 10 gives the reason for the refusal" | "Step 10 reads the E-value" | [read] | none |
| `trust` (md) | the "refused omitted-variable operations" row | the E-value row: how strong a confounder on the risk-ratio scale explains the estimate away, under an approximate conversion | [read] | none |
| `estimate-heading`, `where-next` (md), class P | "refuses fewer than three folds", "`cleverly` refuses that fit", "DR-TMLE raises that refusal at fit time" | state each requirement positively. "The two methods do not compose" can stay | [read] | none |

Callback: `dr_tmle.py` line 58 (the catalog) and lines 126 to 140 (`bound_refusal` and its witness).
The witness about the representer gap (7.75) moves to a unit test, if it is not one already.

## interventions (data seeds 31 and 32)

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `sensitivity` (code) | prints the `robustness_value` refusal for the regime axis. The surface has two `failed` outcome cells, with the `ValueError` message printed | drop the `try` block. Run the surface with a treatment-only grid, `ConfounderStrengthGrid(treatment=(0.0, 0.05, 0.1, 0.2), outcome=(0.0,))`, so every cell returns. Drop the "why each cell failed" print | [executed] `p_int.py` | 0.11645, 0.10693 (-0.0095), 0.086852 (-0.0296), 0.066025 (-0.0504). Induced association -0.0001, +0.0292, +0.0442, +0.0574. 0 failures, 6.1 s |
| `sensitivity-reading` (md) | the refusal reading, the "not implement that case" paragraph, the two failed rows, and the support paragraph | read the four cells. Keep "qualitative, not a bound". Say that the grid has no outcome axis, because the fit declares `q_bounds` and the surface perturbs the score on an unbounded scale | [read] | none |
| `assessment` (code), borderline | the cross-axis status table prints seven `unavailable` rows per axis | print only the rows whose status is not `unavailable` or `not_applicable`, or drop the sensitivity rows | [read] | none |
| `assessment-reading` (md) | the `unavailable` row, and "also refuses `truncation_curve`" | delete both | [read] | none |
| `trust` (md) | "...which this fit's declared support refuses" | "the grid has no outcome axis" | [read] | none |
| `identify-heading`, `dose-fit-heading` (md), class P | "The `Rule` class refuses a rule...", "refuses that fit" | state the requirement | [read] | none |

No implemented omitted-variable bound covers the regime, shift, or ipsi axes. Each reason is
recorded in its capability row [executed]. `evalue` is `not_applicable`. The surface is the only
supported sensitivity operation for the three axes. Callback: `interventions.py` lines 202 and 213
to 214.

## survey-nonresponse (data seeds 71 and 72)

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `refusal-heading`, `refusal`, `refusal-reading` | Step 10 asks for `PopulationAttributableFraction` under `missingness=` and prints the `CapabilityError` | delete Step 10 and renumber Steps 11 to 12. The page keeps its missing-outcome sensitivity in Step 12 | [read] | none |
| (optional new Step 10) | none | `box_study.identify(NaturalCourseMean()).estimate(method=box_method)`: the top-box share that the program delivered, with non-respondents handled under MAR. No supported contrast with `ey0` exists, so do not print one beside it | [executed] `p_survey.py` | `ey_obs` 0.47384, SE 0.009135, CI (0.45594, 0.49174). Under 1 s. The generator returns no truth for `ey_obs` |
| `title` (md) | learn row "recognize a refused composition / Step 10" | delete the row | [read] | none |
| `top-box-heading` (md), class P | "`cleverly` refuses `stratify_by="treatment"` on every fit that draws a split" and its reason | delete. The fit uses the default | [read] | none |
| `data-reading`, `estimate-heading` (md), class P | "`CausalStudy` raises `DataError` for a missing outcome...", "...and `clev[erly refuses]`" | "Declare the response indicator with `missingness=`", and "declare `q_bounds`" | [read] | none |
| `assessment-reading` (md) | the `unavailable` row narration | keep the list, but drop any "refuses" wording | [read] | none |

Callback: `survey_nonresponse.py` lines 26, 202 to 207, 222 to 224, and 236 (the PAF refusal and
the `stratify_by` refusal pins).

## longitudinal-tmle (data seed 41)

No sensitivity operation runs on a longitudinal fit. All nine report `unavailable`, and `refute` is
`unavailable` too [executed]. The working replacement is a measured-covariate benchmark by refit.
Drop one recorded covariate, refit with the same method, and read how far the estimate moves. It is
not a bound. It calibrates "a cause as strong as X" in the Cinelli–Hazlett benchmarking sense,
with no formula.

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `sensitivity` (code) | prints every sensitivity item as `unavailable`, plus `refute` and `corrections`, and `available_methods()` with `False` rows | for each of `age`, `baseline_readiness`, and `engagement_day7`, refit `CausalStudy(frame, design=replace(study.design, ...), protocol=protocol).identify(plan).estimate(method=sequential)`. Print psi, the move, and the move in SE | [executed] `p_ltmle.py` | full history 0.3689 (SE 0.0172). Without `age`: 0.4083 (+0.0395, +2.30 SE). Without `baseline_readiness`: 0.3806 (+0.0118, +0.68 SE). Without `engagement_day7`: 0.4163 (+0.0475, +2.76 SE). Truth 0.3616. About 0.1 s each |
| `sensitivity-heading` (md) | "Step 10: sensitivity, and what the library refuses" | "Step 10: how much a recorded cause moves the estimate" | [read] | none |
| `sensitivity-reading` (md) | the whole reading is about the refusals | read the three moves. Say that an unrecorded cause as strong as `engagement_day7` would move the contrast by about 2.8 SE. Say that the comparison is a benchmark, not a bound, and that no longitudinal bound is implemented (one sentence) | [read] | none |
| `title` (md) | learn row "... and the refusals" | "... and a covariate benchmark" | [read] | none |
| `assessment-reading` (md) | "lists ten `unavailable` operations" | shorten to the operations that ran | [read] | none |
| `estimate-heading`, `rule-heading` (md), class P | "`cleverly` refuses that ...", "The fit refuses an undeclared rule" | state the requirement | [read] | none |

Callback: `longitudinal_tmle.py` lines 159 to 160 (the `refused` method set).

## longitudinal-survival (data seeds 52 and 53)

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `estimate-retention` (code) | `except ValueError as refusal` for `horizons=(1, 3)` | delete the `try` block. Optional: print `sorted({k.horizon for k in exit_result.parameter_keys.values()})` | [executed] `p_surv.py` | `[1, 2]` |
| `estimate-heading` / `estimate-reading` (md) | "The code also asks for a third horizon...", "The last line is a refusal ... refuses a horizon outside 1..2" | "Horizons index the fit's own time points, 1 and 2, not days" | [read] | none |
| `estimate-competing` (code) | `except ValueError` prints the `curve(scale="survival")` refusal | compute event-free survival from `incidence_totals`: `1 - total` with the same `std_err`. The refusal message itself names this route | [executed] `p_surv.py` | always t=1 0.8592, t=2 0.7020. Never t=1 0.7526, t=2 0.5855. SE 0.0079, 0.0112, 0.0100, 0.0165 |
| `competing-estimate-reading` (md) | "...therefore refuses `curve(scale="survival")`" | "Event-free survival is one minus the sum: 0.7020 under always and 0.5855 under never at 60 days" | [read] | none |
| `sensitivity` (code) | prints ten `unavailable` rows and the `robustness_value` refusal | the same benchmark-by-refit as longitudinal-tmle, on the 60-day always-minus-never risk difference. Drop `age`, `baseline_readiness`, and `identified_needs` in turn | [executed] `p_surv.py` | full -0.1576 (SE 0.0227). Without `age`: -0.1148 (+1.89 SE). Without `baseline_readiness`: -0.1269 (+1.35 SE). Without `identified_needs`: -0.1245 (+1.46 SE). About 0.1 s each |
| `sensitivity-heading`, `sensitivity-reading`, `trust` (md) | narrate the unavailability, and the trust row says "nothing for a longitudinal fit" | read the benchmark. The trust row names a benchmark, not a bound | [read] | none |
| `failure-heading`, `failure-reading` (md), class P | "runs without a refusal", "`cleverly` refuses a direct request ... bypasses that refusal" | "`cleverly` implements no intervention that eliminates a competing event. The recoding fits one anyway, because no design check can see the intent". This keeps the failure-mode lesson | [read] | none |
| `assessment-reading` (md) | "ten operations are `unavailable`. Step 12 reads their reasons" | "Step 12 benchmarks the estimate against recorded covariates" | [read] | none |

Callback: `longitudinal_survival.py` line 66 (the horizon refusal) and line 283
(`levels.curve(scale="survival")` refusal). Pin `1 - total` against the cause sum instead.

## msm-projections (data seed 61)

| cell id | current refusal | replacement | evidence | numbers |
| --- | --- | --- | --- | --- |
| `msm-linear-refusal` (code) and its heading and reading | `MSM.linear()` on text labels prints a `DataError` | rename the id to `msm-step-coding`. Fit a second known design with the cadence step coded `{low: 0, medium: 1, high: 2}`, and print both slopes. The lesson that "the mapping is the program's decision" then rests on two working fits that answer two questions | [executed] `p_msm.py` | per cadence step: 0.7341 (SE 0.0224, CI 0.6902 to 0.7779). Per contact: 0.2688 (CI 0.2519 to 0.2856). Intercepts -0.0343 and -0.1068. Under 0.1 s |
| `sensitivity` (code) | the `robustness_value` refusal for the slope (`sensitivity_refusal`) | drop the `try` block. Keep the arm-contrast robustness values | [executed] prefix run | `ate[medium vs low]` rv 0.204 (rva 0.181). `ate[high vs low]` rv 0.420 (rva 0.394) |
| `sensitivity-heading`, `sensitivity-reading` (md) | "and where it stops", and two paragraphs on the refusal | read the two contrasts. Say once that the bound is read on the arm contrasts, because no omitted-variable bound for an MSM coefficient is implemented | [read] | none |
| `title` (md) | learn row "say why `cleverly` refuses to read a label as a dose" | "code the arms for the question the board asks" | [read] | none |
| `estimate-arms-heading`, `share-weight-reading`, `where-next` (md), class P | "`cleverly` refuses a cross-fitted fit of it", "The `MSM` class refuses a `weights=` function unless...", "`MSM.linear` is refused there too" | state each requirement positively | [read] | none |
| `assessment-reading` (md) | "Step 13 shows why" | "Step 13 reads the bound on the arm contrasts" | [read] | none |

`simulated_confounding` is `unavailable` for this multi-arm treatment (F8) [executed], so the arm
contrasts are the only bound. Callback: `msm_projections.py` lines 87 to 88 (`refusal`) and 242 to
247 (`sensitivity_refusal`).

## twins-causal-inference

No refusal showcase [read: scan of every cell]. `diagnostics-reading` says that
`simulated_confounding` is unavailable for a clustered fit. That is a status-table reading, so it can
stay. **Working-workflow concern:** the section 9 sensitivity runs only because the doubly robust nu²
is positive on the stored sample. Findings TW-02 record that seed 11 leaves `omitted_confounding`
`unavailable`, so `report(...)` raises `KeyError`, and that seed 5 raises in `benchmark`
(nu² -10.3). This is the same boosted-propensity mechanism as above. The notebook is static, so the
stored outputs stay valid. A rerun on a new pair sample can fail, though, so the section depends on
its sample.

## House-rule changes

### `docs/development/example-notebooks.md`

| location | current text | proposed text |
| --- | --- | --- |
| cell outline, row 11 | "the sensitivity analysis that applies, or the refusal that explains why none does" / "the robustness value, the bounds, or the refusal message" | "a sensitivity analysis that runs on this fit. The next table names one for each kind of fit" / "the robustness value, the bounds, the E-value, the surface, or the benchmark refits" |
| reading rule, row 6 | "Describe a refusal as what `cleverly` does not implement. Say that a quantity does not exist only when a source shows it" | "Show only operations that run. Do not call an operation to print its refusal, and do not print a capability list with unavailable rows. When the page must name a limit, write the requirement in one sentence. Say that a quantity does not exist only when a source shows it" / reason: "a tutorial is a workflow that works. The technical reference lists every limit" |
| reading rule, row 5 | "...Otherwise print it, state which derivation is missing, and do not interpret the number" | "Read a sensitivity output as robustness only when the technical reference derives it for the fitted estimator. Otherwise choose the operation that the next table names for that fit" |
| the "not derived" table and its two rows | DR-TMLE and C-TMLE rows explain why the bounds are not derived | replace it with a table of the sensitivity analysis that works for each fit (below). Move the DR-TMLE and C-TMLE derivation notes into the technical reference, which already holds them (`collaborative-tmle.md`, and `dr-tmle/`) |
| rules for code cells (new row) | none | "Do not catch `CapabilityError`, `DataError`, or `ValueError` in a tutorial. A cell that needs a `try` block shows a workflow that does not work" / reason: "the user's rule. The callback then has no refusal text to pin" |
| rules for code cells (new row) | none | "Choose learners for which the default doubly robust nu² is positive on the tutorial's seed. Check the propensity calibration slope first" / reason: "an overfit propensity makes the default estimator negative, and the bound refuses" |

Proposed replacement table:

| fitted estimator or axis | the sensitivity analysis to run | evidence |
| --- | --- | --- |
| point-treatment TMLE, arm-indexed `ate`, `att`, `atc`, `ey` | `robustness_value()`, `benchmark()`, and `omitted_confounding()` with the default nu² | the omitted-variable bounds section of `validation-methods.md` |
| DR-TMLE | `evalue()`. It reads the estimate and its interval only | `dr-tmle.ipynb` Step 10 |
| C-TMLE | the bound on the plain TMLE fit, and `simulated_confounding()` on the collaborative fit | `collaborative-tmle.ipynb` Step 11 |
| regime, shift, or incremental axis | `simulated_confounding()`. Use a treatment-only grid when the fit declares `q_bounds` | `interventions.ipynb` Step 15 |
| MSM coefficient | the bound on the arm contrasts of the saturated fit | `msm-projections.ipynb` Step 13 |
| missing outcomes | `missingness` and `tipping_gamma` | `survey-nonresponse.ipynb` Step 12 |
| longitudinal fit | refit without one recorded covariate at a time, and read the move in standard errors. Call it a benchmark, not a bound | `longitudinal-tmle.ipynb` Step 10 |

### `docs/examples/index.md`

It has no refusal text. The "failure mode it demonstrates" column names statistical failures that
run and return numbers. Optional sentence for "The program" section: "Each tutorial runs end to
end. The technical reference states the limits of each method."

### Outside the guidance

| file | change | reason |
| --- | --- | --- |
| `src/cleverly/methods.py:787` (Notes of `CollaborativeTMLEMethod`) | "For an interval, use `strategy="oat"` or `TMLEMethod`" becomes "For an interval, fit `TMLEMethod`" | `oat` raises on `.ci` [executed]. The `strategy` text and `collaborative-tmle.md` lines 233 to 242 agree |
| `tests/unit/tutorial_semantics/__init__.py` | delete `assert_plugin_limits_refuse` | no tutorial reads a plug-in limit after the change |
| `tests/prose-report.md` | refresh the ledger after the rewrites | the CLAUDE.md procedure |
