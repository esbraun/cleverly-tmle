# Sensitivity and validation methods

This page accounts for every instrument `cleverly` gives you to review a method. Each entry answers
three questions in the same order. Why do you use it? What does it tell you? How does it tell you
that?

The instruments fall into four layers, and the layers see different mistakes. Read them in order.
A layer does not replace the layer above it.

| layer | the question it answers | what it cannot answer |
| --- | --- | --- |
| [Diagnostics on the fit you have](#diagnostics-on-the-fit-you-have) | did this fit, on this sample, do what the estimator asked of it? | whether the estimator asked for the right thing |
| [Sensitivity to untestable assumptions](#sensitivity-to-untestable-assumptions) | how wrong would an assumption have to be to change the conclusion? | whether the assumption is in fact wrong |
| [Refutation and simulation you run](#refutation-and-simulation-you-run) | does the fitted workflow behave as it must under a known answer? | anything the law you simulated does not contain |
| [How the library certifies itself](#how-the-library-certifies-itself) | is the derivation this library implements the correct one? | how your own data behaves |

## Diagnostics on the fit you have

These read the artifacts a completed fit already holds. They are cheap, and they run without
refitting a nuisance model.

### Positivity and overlap

**Why.** Most treatment-intervention clever covariates divide by an estimated density. A small
fitted denominator can create a large covariate. Targeting cannot create support that the data does
not contain.

**What it tells you.** The report describes fitted overlap, truncation, and concentration in
inverse-probability weights. It also describes concentration in each fitted score equation's
absolute clever-covariate load. These sample diagnostics do not prove population positivity.

**How.** `result.diagnostics.support()` returns a `PositivityReport` from
[`sensitivity/positivity.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/positivity.py).
It reports six separate quantities, because they fail in different places.

| quantity | how it is computed | what it describes |
| --- | --- | --- |
| per-arm weight concentration | Kish-equivalent rows and top-load shares from inverse arm weights | how unevenly the fitted arm weights are distributed |
| truncation | the units whose fitted mechanism values the bound changes | where the fit uses bounded rather than raw mechanism values |
| per-arm overlap | fitted treatment probabilities, reported by arm | empirical regions with limited fitted overlap |
| maximum clever covariate | the largest reconstructed absolute covariate value | the scale of the largest reported covariate |
| per-equation load concentration | Kish-equivalent rows and top-load shares from $|w_i H_{ij}|$ | how concentrated one fitted score equation's absolute load is |
| other mechanisms | fitted observation and intermediate probabilities when applicable | concentration outside the treatment mechanism |

The report is per arm. A multi-arm fit reads its arms from the parameter's structured index rather
than assuming two.

Missing-outcome and controlled-direct-effect reports add a derived denominator row. Its raw
quantiles describe the complete factor product. Its Kish summaries use the factorwise bounded
product on residual-contributing rows. They also include observation weights. This order matches
the targeting construction. The diagnostic never applies a separate bound to the product.

Every mechanism row reads each unit at the arm the unit received. `Propensity` names that column.
A fit with more than two arms therefore weights each unit by its own denominator.

A derived row counts a clipped cell when any factor moves. Either bound can produce that count.
The verdict therefore names both truncation curves. It reads truncation from the factor rows and
the propensity. It reports product-weight concentration separately.

The derived row covers the marginal-mean groups. These are `mean`, `regime`, and `msm`, whose
covariate divides by the product. A fit that targets only `att`, `atc`, or an incremental
intervention gets no derived row. Those covariates divide by another quantity.

The report states that exclusion rather than leaving the row out in silence. `composed_excluded`
lists every targeted group the derived row does not describe. The summary names the alternative
denominator for each group. The list is empty when every targeted group forms the product. It is
also empty when no fitted factor stands beside `g`.

The fit retains exact absolute score weights for every reported group and equation.
`group_leverage` reads that fitted artifact. The attribute name is retained for compatibility.
Interpret its values as load concentration, not statistical leverage.

For score equation $j$, $w_i H_{ij}$ is row $i$'s multiplier on the targeted residual. The report
summarizes its magnitude, $|w_i H_{ij}|$, for each equation separately. It does not include the
residual itself. The group row reports the equation with the smallest `targeted_ratio`.

The calculation does not collapse several equations through an L1 norm. Kish ratios and top-load
shares stay unchanged when a caller rescales one equation. The raw `max_load` retains that
equation's units.

The report does not rebuild a clever covariate from nuisance predictions. Each row pairs the
retained residual-multiplier load with the bound used by that fitted group. The `att` and `atc`
groups record `g_bounds_conditional`. The `mean` and `msm` groups record `g_bounds`. An ordinary
fit also records the bound's clipping count. If targeting changed the treatment mechanism and its
exact clipping mask was not retained, the count is unavailable and `group_leverage_omissions`
records why.

The public `diagnostics.support()` route provides group rows as follows.

| fitted group | support report | group row available |
| --- | --- | --- |
| `mean`, `att`, `atc`, or `msm` | `PositivityReport` | yes |
| regime | intervention-specific regime report | one row per regime equation |
| shift | intervention-specific shift report | one row per shift equation |
| incremental | intervention-specific incremental report | one row per outcome equation |
| longitudinal | `LongitudinalDiagnostics` from `support()` | no. A longitudinal fluctuation retains no absolute score weights |

The longitudinal row is a disclosed limitation and not an oversight. A longitudinal result returns
per-node leverage through `diagnostics.support()`. The direct `stagewise()` call remains a
compatibility alias for the same payload. A combined run retains only the `support` row.

The route reaches neither `positivity_report` nor an intervention report. A longitudinal
fluctuation also retains no `absolute_score_weights`. No fitted artifact therefore exists for a
generic or intervention load row.

The intervention-specific reports keep their ratio and support fields. Their `score_load` field
adds the fitted equation load without replacing those quantities. The report matches columns to
policies in declaration order and retains the fitted equation name. It does not parse generated
equation labels.

An incremental fit solves a second equation for the treatment mechanism. `score_load` describes
only the retained outcome residual-multiplier equation. The mechanism fluctuation does not retain
an equivalent row-level absolute load, so the report does not claim to summarize it.

The combined support row reports the most concentrated intervention equation separately. It does
not add that ratio to the mechanism effective-sample-size minimum or use it to assign status.

An older fitted artifact might not retain exact absolute score weights. The report does not
reconstruct an approximation. `group_leverage_omissions` names each omitted generic group and its
reason. Each intervention row uses `score_load_omission` for the same purpose.

The reason strings are machine-readable, so they are part of the API. Both paths accept an artifact
through `cleverly.data.weighting.validate_score_loads` and record the constants that module
declares. A given condition therefore produces the same text whichever report holds it.

| constant | what the report means by it |
| --- | --- |
| `SCORE_LOAD_MISSING` | no artifact arrived, so there are no exact absolute score weights |
| `SCORE_LOAD_NO_EQUATION` | an artifact arrived and the fit recorded no score equation for it |
| `SCORE_LOAD_SHAPE_MISMATCH` | the artifact is not one column per recorded score equation |
| `SCORE_LOAD_NOT_FINITE` | a load is negative, infinite, or `nan` |
| `SCORE_LOAD_EMPTY_MASK` | the fitted score mask selected no rows |
| `SCORE_LOAD_MASK_TOO_LARGE` | the mask holds more rows than the fitted data |
| `SCORE_LOAD_PREDATES` | the report unpickled from before the score-load fields existed |

The table order is the guard order. A caller can be in the first two states at once, because
`check_support`, `check_shift_support`, and `check_incremental_support` all default to no artifact
and no equation names. That caller reads `SCORE_LOAD_MISSING`.

Two of those rows are behavior changes. The empty-mask and oversized-mask conditions were one reason
before, and they say opposite things: a fit that targeted nothing, and a malformed block. The
generic group path also had no size refusal at all. A mask taller than the fitted data was described
rather than omitted, and the row published a `total_ratio` above one. A direct-call table in
`tests/unit/test_intervention_load_diagnostics.py` reaches each condition, under
`test_every_refused_score_artifact_names_the_one_condition_it_failed`.

Every generic group row and intervention `score_load` row carries these concentration keys.

| key | what it holds |
| --- | --- |
| `equation` | the score equation with the smallest `targeted_ratio` |
| `n_total` | all rows in the fitted data |
| `n_targeted` | rows in the fitted score mask, including structural zeros in the selected equation |
| `effective` | Kish-equivalent rows for the selected equation's absolute load |
| `targeted_ratio` | `effective` divided by `n_targeted` |
| `total_ratio` | `effective` divided by `n_total` |
| `top_1pct` | the share of the load the largest 1% of rows hold |
| `top_5pct` | the share of the load the largest 5% of rows hold |
| `max_load` | the selected equation's largest absolute residual multiplier |
| `zero_load` | score-mask rows with a zero residual multiplier in the selected equation |
| `reported_repeat`, `n_repeats` | draw 01 and the total draw count, on intervention rows only |
| `lower_bound`, `upper_bound` | the exact treatment-mechanism bound for the group |
| `clipped_count`, `clipped_fraction` | units that bound changes, or unavailable when the exact targeted-mechanism mask was not retained |

`top_1pct` and `top_5pct` count at least one row each. Each key takes the largest
`max(1, ceil(fraction * n_targeted))` loads. At 20 score-mask rows or fewer the two keys therefore
report the same number, which is the largest single row's share. Read them as separate quantities
only above that size. The rounding rule lives in `cleverly.data.weighting.top_weight_share`.

The bound and clipping fields apply only to generic group rows. The repeat fields apply only to
intervention rows. A repeated fit's intervention load describes draw 01 and not its
coordinatewise median-combined estimate.

A generic group row carries no repeat fields. Its draw count comes from
`PositivityReport.n_repeats`, which the report records once for the whole fit. Every reader renders
the draw from the row or from the report, and no reader writes the count as a fixed string. The test
`test_the_group_load_row_counts_the_same_draws_the_report_does` in
`tests/unit/test_repeated_crossfit.py` holds the combined `support` row and
`PositivityReport.summary()` to one total.

The per-equation tests recompute these fields from the fitted score artifacts. They also use
multi-column controls with different equation scales and signs.

`max_load` and the maximum clever covariate answer different questions. The first includes score
weights and uses targeted rows. The second is unweighted and uses every row. The two need not agree.

Arm, mechanism, and group concentration are different descriptive quantities. The combined
`support` row reports them separately. It does not pool them into one minimum.

The truncation load counts units and not cells. A unit counts once when the bound moves any arm of
its mechanism. `Propensity` owns the rule the count follows. It clips a two-arm mechanism through
`g1` and takes arm 0 as the complement. It clips a mechanism with more arms column by column.

**What it grades, and what it only reports.** The report grades the truncated fraction and fitted
zero support. These findings describe the fitted procedure. They do not establish a population
positivity violation. The report does not grade any Kish ratio or top-load share. No published
result supplies a universal cutoff for these descriptive quantities.

The combined `support` row is `completed` when no graded finding applies. Do not read that status
as positivity clearance. Inspect the fitted probabilities, truncation, and concentration tables.

### Truncation stability

**Why.** A truncation bound is a finite-sample choice. A conclusion that survives only at one bound
is a conclusion about the bound.

**What it tells you.** How far the estimate moves as the bound moves.

The bound regularises the fitted procedure. It does not change the requested estimand or define an
overlap-population estimand. A moving curve shows extrapolation sensitivity through the
second-order remainder.

**How.** `result.diagnostics.truncation_curve()` sweeps the `g_bounds` level and **retargets** the
cached nuisances at each level through `TMLE.retarget`. On an ordinary, collaborative, or unguarded
doubly-robust fit it refits no nuisance model, so it is a retarget operation.

The returned frame records both ends of every evaluated pair. Scalar treatment-mechanism values
remain symmetric shorthand for `[bound, 1 - bound]`; observation- and intermediate-mechanism
values use `[bound, 1]`. It also records each parameter's own fitted pair and estimate, plus the
signed difference from that estimate. The default grid includes every exact fitted pair, without
rounding, so every parameter has one fitted marker. An explicit grid is not expanded: when it omits
the fitted configuration, the fitted columns retain that configuration and every marker is false.

The combined diagnostic row reports the signed movement range for each parameter or working-model
coefficient. The retained frame keeps the fitted pair and the per-bound values.

`estimands=` restricts the emitted rows to the parameters you name. It accepts two forms.

| you name | the curve emits |
| --- | --- |
| a reported parameter, such as `ey[high]` | the row for that parameter |
| a registered target, such as `ey` | one row for each reported alias of that target, in report order |

The two forms coincide on a two-armed fit. There the target and its one reported alias share a
spelling. The curve refuses any other name, and it names the parameters the fit carries. It also
refuses an empty selection, which would emit no row. Each refusal comes before the first retarget.

The `truncated_fraction` column counts the units the bound moves. It follows the same rule
`Propensity` applies.

The count belongs to the bound pair the curve evaluated that row at. The `support()` report always
reads `config.g_bounds`, so the two agree on a row that carries that pair. An `att` or `atc` row
carries `config.g_bounds_conditional` instead, and its count then differs. A shift fit clips no arm
probability, so the column reports no value there.

A guarded DR-TMLE fit is the exception. Its targeting step alternates against the reduced-dimension
regressions, so each bound refits them, and the missing-outcome construction receives the swept
bounds because they define two of its regression targets. The capability row for such a fit
declares `refit` and asks for `include_refits`. The primary outcome regression and the propensity
are still cached, so the curve can cost less than a fit. `LTMLE` refuses it: `g_bounds` enters the
pseudo-outcome of every earlier node through the backward recursion, so changing it changes what
the earlier regressions were fitted to, and the whole pass has to run again.

### Nuisance model quality

**Why.** A nuisance model can predict well and remain miscalibrated. The clever covariate divides by
the predicted probability itself, so a miscalibrated fit moves every weight.

**What it tells you.** The report gives prediction loss and calibration for each retained nuisance
fit. It also gives the candidate weights and risks when a Super Learner supplies them. For C-TMLE,
it retains the selector or outcome-adaptive fit. For repeated cross-fitting, it reports parameter
movement across split draws.

**How.** `result.diagnostics.nuisance_models()` returns a report for the fitted result family.
Point-treatment results return `NuisanceDiagnostics` from
[`validation/nuisance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/nuisance.py).
That report contains propensity AUC, logistic calibration, outcome fit metrics, and Super Learner
candidate details. The metrics are out-of-fold when the fit cross-fitted its nuisances. They are
in-sample when it used one fold.

Longitudinal results return `LongitudinalNuisanceDiagnostics`. The report includes only models the
estimator fitted. Treatment and censoring models appear once per node because each model serves all
regimens. Outcome and pseudo-outcome models appear for each fitted regimen, cause, horizon, and
node.

| role | evaluation target | reported loss |
| --- | --- | --- |
| treatment | observed arm under the observed history | weighted negative log likelihood |
| censoring | observed retention under the observed treatment history | weighted negative log likelihood |
| outcome | final target in each fitted recursion | weighted Brier loss for a binary target, or mean squared error otherwise |
| pseudo-outcome | each earlier target in the fitted recursion | weighted mean squared error |

The treatment and censoring predictions come from the fitted observed-law pass. The report does not
refit those learners or reconstruct observed histories from regimen predictions. Each row uses
`evaluation` to label its loss as `out_of_fold` or `in_sample`.

`loss_name` is `log_loss`, `brier`, or `mse`. `loss` holds that role-specific value, while
`reported_loss` preserves the MSE from an older regression row. The nested `model` is a
`NuisanceModelReport`. It retains calibration and learner-library details.

The `mse` field stays the legacy column, and it answers about node regressions alone. An outcome
row or a pseudo-outcome row reports its square loss there. A treatment row or a censoring row
reports `nan`, because a mechanism fit has no square loss.

Two further changes affect a weighted fit. The `mse` value now averages under the observation
weights, where an older release averaged without them. The frame also admits an empty value in
`regimen`, `cause`, and `horizon`, because a mechanism row carries no regimen identity. That row
stores `None`, which a dataframe renders as `None` or as `NaN` by column type.

For a categorical treatment, `log_loss` is the observed-class multinomial negative log likelihood.
The nested report uses `kind="multinomial probability"` and retains no binary calibration table.
Armwise calibration requires a separate report design because no treatment arm is privileged.

`LongitudinalNuisanceDiagnostics.omissions` holds typed `LongitudinalNuisanceOmission` records.
Each record names `role`, `time`, and `reason`. An older artifact uses
`LONGITUDINAL_MECHANISM_PREDICTIONS_MISSING` when it lacks an observed-law mechanism prediction. A
complete-data design uses `LONGITUDINAL_CENSORING_NOT_FITTED` because it made no censoring fit.

`to_frame()` starts with the row identity, evaluation, loss, model name, and model kind. It then
adds the union of metrics that the nested model reports.

Read the propensity AUC as a positivity signal and not as a score. A higher AUC means the treatment
is more predictable, which means the arms overlap less. Higher is not better here.

One exception changes the interpretation. A C-TMLE propensity is a selected working mechanism, not
the estimated treatment law given the complete adjustment set. The report records this role in
`treatment_role` and retains the exact method artifact in `selection`. `TMLEResult.ctmle_selection`
returns that same artifact, typed. The report keeps every AUC and calibration value.

The role drops exactly two claims. Both claims read a treatment law that a collaborative fit never
estimates.

| claim the report can make about the propensity | on a collaborative fit | why |
| --- | --- | --- |
| an AUC below 0.55 means overlap is excellent and confounding by these covariates is limited | dropped | an intercept-only candidate gives an AUC near chance by construction, so the claim describes a model nobody fitted |
| a calibration slope outside 0.7 to 1.4 means the predicted probabilities are systematically off | dropped | the same reason. A selected working mechanism has no calibration target |
| an AUC above 0.9 signals a positivity problem | kept | `CTMLE._nuisances` puts the selected mechanism on `nuisance.propensity`, so it is the denominator the clever covariate divides by. An AUC near one there means the fitted weights are near-degenerate |
| a super learner weight above 0.8 on the marginal mean means the model contributes little | kept | the note states a fact about a learner library, not an interpretation of the treatment law |

`tests/e2e/test_ctmle.py` holds both halves of this rule. Its
`test_the_role_suppresses_two_claims_and_no_others` mutates the retained report and asserts that
each dropped claim returns under `estimated_treatment_law`. Its
`test_a_mean_only_learner_library_is_reported_for_a_working_model_too` raises the mean weight on a
working model and requires the note. Use `support()` to inspect the denominator the selected
mechanism creates.

The selector artifact retains every candidate, risk, fitted fold, and the selected index. The
outcome-adaptive artifact retains its outcome-prediction features and treatment risk. Each artifact
renders one sentence through `describe()`. The nuisance summary and the combined assessment row
both print that one sentence.

| artifact | what `describe()` renders |
| --- | --- |
| `CTMLESelection` | `C-TMLE greedy selected candidate 2 of 4 for ate` |
| `CTMLEOutcomeAdaptiveFit` | `C-TMLE outcome-adaptive fit used 2 Qbar feature(s)` |

On a repeated fit these objects describe draw 1, because the result retains method-specific state
for that draw. The nuisance summary and the combined assessment row render the scope through
`cleverly.utils.text.format_draw`, as `draw 1 of 3`. The score-load rows above print the
zero-padded `draw 01 of 03` instead. Those rows pad every style, so a table cell and the
sentence that describes it agree on one width.

`repeat_spread` contains one `RepeatSpreadRow` per reported parameter when the fit uses two or more
draws. The row pairs the split standard deviation across every retained draw with the
median-combined result's standard error.

Both quantities are on the inference scale of the estimand. That scale is the log scale for a ratio,
and the outcome scale otherwise, so `ratio_to_standard_error` divides two like quantities.
`test_a_ratio_takes_its_spread_on_the_scale_its_standard_error_lives_on` in
`tests/unit/test_repeated_crossfit.py` reads `ate`, `rr`, and `or` off one fit. The ratio is
descriptive and has no pass threshold.

A one-draw fit retains an empty tuple instead of reporting zero. A table cell with no finite value
prints `-`. `repeat_spread_frame()` follows the input dataframe backend.

`repeat_spread_omission` names the cause of every missing spread, and `selection_omission` does the
same for an absent artifact. Both read a constant that `cleverly.validation.nuisance` declares, so
one condition always produces one text.

| constant | what the report means by it |
| --- | --- |
| `SPREAD_SINGLE_DRAW` | the fit drew the split once, so there is no between-draw spread |
| `SPREAD_UNAVAILABLE_DRAWS` | some draw reported no estimate for the named parameters |
| `SPREAD_NOT_FINITE` | the spread exists and is not finite for the named parameters |
| `SPREAD_NO_PARAMETERS` | the result reports no parameter to take a spread of. No fitted result reaches this state, because every fit reports at least one estimand |
| `NUISANCE_SELECTION_MISSING` | the fit declares itself collaborative and retains no `ctmle` artifact |

The summary and the combined row print a spread reason only above one draw. `SPREAD_SINGLE_DRAW` is
the ordinary state of an ordinary fit, and printing it would put an "unavailable" line under every
report this package produces.

### Score equations

**Why.** TMLE is defined by the equation its fluctuation solves. A fit that stopped early solves it
approximately, and the reported influence curve is then not mean zero.

**What it tells you.** Whether the fluctuation reached the root of the equation the library posed.

**How.** `result.diagnostics.score_equations()` recomputes $P_n \hat{D}^*(O)$ from the fitted
influence curves and compares it against a tolerance scaled to the score's own units
([`validation/score.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/score.py)).
A point-treatment fit compares the score in the outcome's units against
`tolerance * se / sqrt(n)`. A longitudinal fit bounds each node's relative score.

**What it does not tell you.** `score_check` is necessary and not sufficient. A clever covariate
that is wrong in the same way in both the targeting step and the reported curve solves its own
equation exactly. That the equation is the right one is a claim about the library, and
[how the library certifies itself](#how-the-library-certifies-itself) is where that claim is
tested.

### Correction identities

**Why.** `DRTMLE` reports a curve assembled from three separate score equations. The curve is only
mean zero if each correction the curve subtracts is the correction whose equation the fit solved.

**What it tells you.** Two different failures, told apart. An **identity residual** means the
software solved one expression and reported another. A **correction score** means the fit did not
converge.

**How.** `result.diagnostics.corrections()` recomputes each correction's empirical mean from the
exact returned state and compares it with the score the solver recorded
([`validation/drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/drtmle.py)).
It reports the two residuals and a clipping bias term separately.

This instrument found a real defect. One clipped row in six hundred left the reported curve
uncentred at `2e-04` while all three fluctuation rows reported `1e-11`.

### Intervention support

**Why.** Arm positivity, regime support, shift support, and incremental support are four different
questions. One propensity histogram answers none of them.

**What it tells you.** Whether the declared intervention puts mass where the data has none.

**How.** Each intervention class exposes its own report through
[`interventions/support.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/support.py).
A shift fit asks whether the density *ratio* stays bounded. A per-arm propensity table has no rows
on a continuous dose, so `diagnostics.support()` dispatches to the question that applies rather
than returning an empty table.

### Design weights

**Why.** Observation weights tilt the population. Their cost to precision is separate from the
clever covariate's cost, and adding the two together hides both.

**What it tells you.** The effective sample size the declared weights leave, before any positivity
cost.

**How.** `WeightReport` in
[`data/weighting.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/data/weighting.py)
reports the weighted effective sample size and warns when the weights concentrate.

### The status contract

Every assessment row returns one of seven states, and the states are part of the contract rather than a
presentation choice.

| status | what it means |
| --- | --- |
| `passed` | the check ran and its condition holds |
| `failed` | the check ran and its condition does not hold |
| `warning` | the check ran and an explicit diagnostic rule requires qualification |
| `completed` | a descriptive analysis ran and defines no pass or fail rule |
| `deferred` | the operation can run after the caller supplies a required choice or cost opt-in |
| `not_applicable` | no such analysis exists for this scientific question |
| `unavailable` | the analysis is meaningful, but its method, derivation, fitted artifact, or requested variant cannot run it |

`AssessmentReport.summary()` prints three sections under these headings, in this order.

| section | the rows it holds | its shape |
| --- | --- | --- |
| `Returned results` | every `completed` row | one row per operation, which gives the surface, the operation, and the returned result in full text |
| `Checks` | `failed`, then `warning`, then `passed` | the first row of a status group states the status and the count. Each row names one surface-qualified operation |
| `Not run` | `deferred`, then `unavailable`, then `not_applicable` | the same compact shape as `Checks` |

An empty section states `none`. The summary ends with two footer lines. They name `to_frame()`,
`next_steps()`, and `report(...)`.

`_SUMMARY_CHECK_ORDER` and `_SUMMARY_OMISSION_ORDER` in `src/cleverly/assessment.py` pin the two
status orders above. The test
`test_the_battery_summary_expands_results_before_compact_checks_and_omissions` in
`tests/unit/test_post_fit_assessment_battery.py` builds one report that carries all seven
statuses. It reads the three section headings, the status order, and the count on each group. It
also reads the surface qualification of a duplicated name, and it asserts that the long check
detail stays out of the summary.

**The summary leads with returned results, and that order is a recorded decision.** A completed
analysis carries the numbers a reader interprets, so it expands first. A `failed` check therefore
prints below several full-width result rows.
[Pull request 198](https://github.com/esbraun/cleverly-tmle/pull/198) records the decision. Two
independent reviews found that its first draft over-prioritized warnings, and the result-first
revision resolved their findings.

The reader still finds every actionable row in the `Checks` section. That section gives one line to
each failure and each warning. Read `AssessmentReport.attention` for those rows as objects. Call
`next_steps()` for the follow-up action on each row that carries one.

`AssessmentReport.to_frame()` remains the complete row ledger, including warning and refusal
details. The three report surfaces retain the original rows. `next_steps()` retains follow-up
actions. `report(...)` returns a retained payload from an operation that ran.

The summary prints no next step for any row, and `next_steps()` returns them. Deferred rows retain
the required argument or flag in their next step. They also retain the request the caller made, and
the combined run's seed when the operation accepts one.

Known unsupported omissions carry the capability's reason and remain `unavailable` or
`not_applicable`. Such a row retains no arguments, because this fit refuses it before any request
is considered. An expected refusal after invocation becomes `unavailable` and retains the bound
invocation arguments. The report continues with other accepted operations. Structural errors still
propagate.

#### The ambiguous default estimand

Each operation below answers for one parameter, and takes that parameter from a default. A fit that
reports a bare `ate` settles the `"ate"` default. A multi-arm fit reports `ate[high vs low]` and
`ate[medium vs low]`, so it settles nothing. The caller then owns the choice. The row reports
`deferred`, and its next step names `estimand`.

| operation | facade | default estimand |
| --- | --- | --- |
| `omitted_confounding` | sensitivity | `"ate"` |
| `robustness_value` | sensitivity | `"ate"` |
| `elements` | sensitivity | `"ate"` |
| `benchmark` | sensitivity | `"ate"` |
| `contour` | sensitivity | `"ate"` |
| `tipping_gamma` | sensitivity | `"ate"` |
| `refute` | diagnostics | `"ate"` |
| `evalue` | sensitivity | `None` |

The library reads that default from the routed signature. `benchmark` takes `covariates` first and
`estimand` by keyword, and the keyword default is as ambiguous as a positional one. `evalue`
selects its own contrast from a `None` default, and defers when two contrasts qualify.

The count of eligible reported parameters selects the answer.

| eligible parameters | the combined report |
| --- | --- |
| the fit reports a bare `ate` | runs the operation |
| exactly one, on a sensitivity route | supplies that name, then runs the operation |
| exactly one, on `refute` | reports `deferred` and names `estimand` |
| two or more | reports `deferred` and names `estimand` |
| none | keeps the fit's own refusal, which is `not_applicable` or `unavailable` |

The sensitivity facade fills in a sole eligible name. `refute` fills in nothing, because it refits
under the name it is given. Its row therefore defers for one eligible alias, and the caller names
that alias to run it.

`simulated_confounding` shares the same default and is not in the table. It declares `estimand` in
`requires_arguments` beside `grid`, so the required-argument gate defers its row and names both.

An estimand the caller names is never a deferral. A name the fit never reported, an unsupported
contrast, a missing derivation, and a missing replay artifact all report `unavailable`. The facade
resolves the deferral for each request, so a supplied `estimand` makes the row available again.

`ASSESSMENT_CAPABILITIES` in
[`assessment.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/assessment.py)
declares, for each operation and each result family, the answer, the required artifacts, the cost,
and the execution class. The two costly classes are disjoint and are named separately.

A **refit**
operation fits new nuisance models. A **retarget** operation re-solves the fluctuation against
cached ones. `run_all` includes cheap retargets, such as an E-value derivation, by default.
It excludes refits and moderate retargets. Each deferred row names the flag that runs it.
The `arguments` mapping supplies required analyst choices for named operations.

An operation can decline caller-supplied arguments after its capability precheck. The combined
report records that refusal as an omission and continues with later operations. A direct call still
raises `CapabilityError` with the full refusal. Structural exceptions still stop the report.

The top-level `random_state` reaches `refute`, `benchmark`, and `simulated_confounding` only.
A seeded fit supplies its fit seed when this value is absent. An unseeded fit draws and records a
seed. Supplying the seed both places is an error.

## Sensitivity to untestable assumptions

Identification rests on assumptions that no diagnostic can test. These instruments do not test
them either. Some derive a formal scale. The simulated surface reports estimate movement
under a declared perturbation instead. Its `movement_scale` field names the scale of that
movement, which is additive for a difference and logarithmic for a ratio.

| instrument | the assumption it stresses | the number it reports | what it assumes to report it |
| --- | --- | --- | --- |
| omitted-variable bounds | no unmeasured confounding | the largest bias an unmeasured confounder of declared strength can produce | the confounder acts through the outcome regression and the treatment mechanism, with declared partial-$R^2$ strength in each |
| robustness value | no unmeasured confounding | the single strength at which the conclusion flips | that the two strengths are equal |
| benchmark | no unmeasured confounding | the strength of a confounder "as strong as" a named observed covariate | that dropping the covariate and refitting calibrates the scale |
| simulated common cause | no unmeasured confounding | estimate displacement across a declared strength grid, on the additive scale or the log scale | a supported latent perturbation family and plausible declared strengths |
| E-value | no unmeasured confounding | the minimum risk-ratio association with both treatment and outcome that explains away the effect | a risk-ratio scale |
| missingness tilt | outcomes missing at random | how the estimate moves as the unobserved outcomes are tilted away from the observed ones | the tilt is a constant on the logit scale |
| tipping gamma | outcomes missing at random | the tilt at which the conclusion changes | as above |

### Omitted-variable bounds, robustness value, benchmark, and contours

**How.** [`sensitivity/omitted_variable.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/omitted_variable.py)
implements Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022). The bound is

$$
|\text{bias}| \le |\rho| \sqrt{\frac{c_D^2}{1-c_D^2}}\; c_Y \sqrt{\sigma^2 \nu^2},
$$

with $\sigma^2 = E[(Y - \bar{Q})^2]$ and $\nu^2$ the second moment of the Riesz representer. The
two primitives are exposed as `elements()`. A median-combined repeated fit refuses this analysis
because the median bound needs its own influence function.
`robustness_value()` inverts the bound for the single strength that flips the conclusion.
`benchmark()` drops each named observed covariate, refits, and calibrates the strength scale
against what that covariate was worth. `contour()` returns the grid a contour plot needs.

`benchmark()` is the only member of this group that refits.

### Simulated common-cause stress surface

**Why.** An analyst can inspect whether a fitted parameter moves under a plausible latent
common cause. Sharma and Kiciman (2020) name this procedure. Sharma et al. (2021) state its
qualitative limits.

**What it tells you.** `simulated_confounding()` reports the point estimate and its displacement
at every declared strength pair. `movement_scale` states how to read that displacement.
The frame from `to_frame()` carries it as a column on every surface, additive and ratio alike.
Additive parameters and attributable fractions use `estimate_difference`. Risk and odds ratios
use `log_ratio` and the signed displacement

$$
\log(\widehat\psi_{\mathrm{refit}})-\log(\widehat\psi_{\mathrm{original}}).
$$

Its exponential is the refitted ratio divided by the original ratio. Each ratio cell keeps its
point estimate on the ratio scale. Gruber and van der Laan (2012), Section 2.1 and Section 2.7,
define the ratios and their log-scale intervals. Appendix A gives their log-scale influence
curves. The pinned `tmle3` `R/delta_functions.R` source uses the same contrasts.

Neither source defines a simulated-confounding displacement. The log difference is this package's
descriptive movement convention, not a paper-derived sensitivity measure.

Each cell also reports `induced_treatment_association`. This value is the realised correlation
between the latent vector and the treatment of that cell. The operation gives no corrected
estimate, bound, p-value, confidence interval, robustness value, threshold, or pass/fail result.

`target_measure` records how the operation reads the analysis rows. It is `unweighted` when the
fit declares no weights. It is `fixed_empirical_tilt` when the fit declares fixed weights.
`weight_report` carries the weight kind, column, scale, effective size, and concentration measures.
The label follows the declared weight column, so read `weight_report.is_weighted` to learn whether
the realized tilt is nonconstant.

**How.** The operation draws one row-level standard-normal latent vector. It reuses that vector
and one refit seed across the complete grid. Each cell starts from the original data.

For binary treatment strength $k_A$, the operation flips treatment when
$U \geq \Phi^{-1}(1-k_A)$. A continuous treatment uses $A' = A+k_AU$. Gaussian outcomes use
$Y' = Y-k_YU$. Binomial outcomes use the same tail-flip construction as binary treatment.
Flip strengths range from zero through 0.5. Continuous treatment strengths are signed finite
coefficients.

**Target populations.** The surface accepts one exact reported alias. Structured parameter metadata
identifies its arms, policy, and baseline stratum. The following contracts distinguish its populations.

| reported field | population contract |
| --- | --- |
| `stratum`, `strata_names` | The selected baseline values and their column names. A marginal parameter has no selected stratum |
| `population="baseline"` | Arm, regime, incremental, attributable, modified-policy, and MSM parameters use fixed baseline membership |
| `population="perturbed_treatment_group"` | ATT and ATC use observed-treatment membership after the cell's perturbation |
| `conditioning_arm` | The observed arm defining ATT or ATC. Other targets report `None` |
| `target_population_fraction` in each cell | The conditioning group's weighted share within the baseline population. Other targets report one |
| `association_population` | Correlation uses the selected baseline stratum, or all fitted rows for a marginal parameter. Both arms remain eligible |
| `calibration_population` | Numeric calibration uses the complete original fitted population |
| `refit_population` | Every non-anchor cell fits the complete perturbed dataset |

For baseline strata, the operation holds $S=f(W)$ fixed and changes only treatment and outcome.
The existing perturbation therefore acts on the conditional empirical law within each stratum.
One latent vector spans all rows and all strata. The operation never subsets rows before nuisance
fitting and never draws a new latent vector for each stratum. Two calls with the same seed and
different stratum aliases use the same perturbed datasets.

Ordinary TMLE supports conditional binary arm means, ATE, ratios, PAR, PAF, ATT, ATC, regime
parameters, and identity-link MSM coefficients. It also supports conditional modified-policy
means and contrasts. MSM strata require binary treatment and the identity link. Binary C-TMLE supports conditional arm means, ATE, and
ratios. These paths retain fixed observation weights and estimator-owned repeat aggregation. A
DR-TMLE fit refuses `strata=` when it fits, so no stratified DR-TMLE result exists for the surface
to replay.

ATT and ATC condition on the arm that the structured key names. Each cell recomputes that group's
membership from its perturbed treatment. Displacement combines changes in the fitted outcome
contrast and changes in population composition. It does not assess the effect among the original
treated or control rows held fixed. `target_population_fraction` makes the changing group share
visible. `CTMLE` and `DRTMLE` refuse ATT and ATC when they estimate, so this surface receives
ordinary-TMLE results only.

The DoWhy refit at `2116d5c` preserves effect modifiers and the target-unit selector. Its
propensity-weighting estimator recomputes ATT and ATC indicators from the perturbed data. This
supplies a finite-sample population convention, not a new sensitivity theorem. See the
[DoWhy source locators](../references.md#sensitivity-analysis). van der Laan (2010),
[Part I, Section 4](../references.md#point-treatment-and-stochastic-interventions), describes the
observed-treatment conditional target.

The post-fit assessment battery reads `target_population_fraction`. It reports `warning` when
`population` is `"perturbed_treatment_group"` and the smallest cell keeps less than half the anchor
cell's fraction. The anchor is the cell whose treatment strength and outcome strength are both
zero, so its fraction is the unperturbed share of the conditioning arm. The rule compares each
surface against its own anchor, so a small treated arm alone raises no warning.

The minimum covers every cell that recorded a fraction, and not the successful cells alone. A cell
whose conditioning arm empties fails its refit with `DataError`, and that cell holds the hardest
collapse on the surface. The surface records the fraction before it refits, so the failed cell
still carries the fraction.

A surface whose `population` is `"baseline"` never warns under this rule, because every cell
reports a fraction of one. The detail line carries the minimum fraction and the anchor fraction on
every surface. One test witnesses each part of the rule.

| claim | witness |
| --- | --- |
| the detail line carries both fractions | `test_post_fit_assessment_battery.py::test_descriptive_interpreters_complete_without_inventing_a_verdict` |
| the one-half threshold, from both sides | `test_simulated_confounding_populations.py::test_the_assessment_row_warns_when_the_conditioning_group_halves` |
| a failed collapsing cell still reports the collapse | `test_simulated_confounding_populations.py::test_empty_perturbed_conditioning_population_retains_a_failed_cell` |
| a `"baseline"` surface never warns | `test_post_fit_assessment_battery.py::test_a_baseline_surface_reports_no_population_collapse` |

Every file in that table sits under `tests/unit/`.

The diagnostic population differs from the ATT or ATC conditioning group. Treatment is constant
inside that group, so its correlation with the latent variable is undefined there. The surface
instead measures association across both arms within the selected baseline population. Numeric
calibration stays global; a conditional surface does not introduce a conditional calibration formula.

`tests/unit/test_simulated_confounding_populations.py` checks these contracts with complete refits
and nonzero population witnesses. Existing target instruments in the
[evidence manifest](evidence.md) establish the reused estimands. This wrapper adds no interval or
repeated-sampling claim and changes no estimator equations.

**The binary treatment axis.** The misclassification analysis below, its closed form, and the
treated-fraction table describe the binary tail flip only. The continuous law follows in its own
block.

The treatment law is non-differential misclassification. It flips a treated row and an untreated
row in the same latent tail. The association it induces between $U$ and the treatment therefore
depends on the treated fraction $\pi$. Write $q = \pi + (1-2\pi)k_A$ for the perturbed treated
fraction. When $A$ is drawn independently of $U$, the induced correlation is

$$
\operatorname{corr}(A', U) = \frac{(1-2\pi)\,\phi(\Phi^{-1}(1-k_A))}{\sqrt{q(1-q)}}.
$$

| treated fraction $\pi$ | $k_A = 0.1$ | $k_A = 0.3$ | $k_A = 0.5$ |
| --- | --- | --- | --- |
| 0.2 | +0.240 | +0.430 | +0.479 |
| 0.35 | +0.108 | +0.210 | +0.239 |
| 0.5 | +0.000 | +0.000 | +0.000 |
| 0.65 | -0.108 | -0.210 | -0.239 |
| 0.8 | -0.240 | -0.430 | -0.479 |

A 500,000-row simulation reproduces every entry to within 0.005. That bound is the Monte Carlo
error at that sample size. On a balanced design the treatment axis induces no association with
$U$. It moves the estimate through misclassification of the treatment alone. Above a treated
fraction of one half the sign reverses.

Sharma and Kiciman (2020) prescribe this construction, so the law does not change. The surface
instead reports what the law achieved on your data. Every cell carries the realised correlation
between the latent vector and its own treatment, in `induced_treatment_association`. Read that
value before you read a movement along the treatment axis as confounding. A cell near the anchor
value moved the estimate by misclassification of the treatment alone. The table above gives the
population value each cell approaches.

**The continuous treatment axis.** Under the continuous law, $U$ changes the dose by construction,
so $\operatorname{corr}(U, A')$ grows with $k_A$. A confounding path also needs $U$ to enter the
outcome. That happens only when the outcome strength is nonzero. A cell in the zero
outcome-strength column therefore carries no confounding path, whatever its association. Its
movement reports dose perturbation alone.

**The zero treatment-strength column.** A cell at $k_A=0$ leaves $U$ out of the treatment, so it
carries no confounding path either. Its movement reports the outcome perturbation alone. The
Gaussian law $Y'=Y-k_YU$ is a level shift, and the surface draws $U$ uncentred. An
`ate_shift[...]` contrast subtracts one policy mean from the other, so it removes most of that
level and the column stays small. An `ey_shift[...]` policy mean keeps it, so read the $k_A=0$
column of a policy-mean surface as an artifact of the outcome law.
`tests/unit/test_simulated_confounding.py::test_continuous_policy_mean_runs_a_real_ordinary_tmle_refit`
measures both columns on one fit.

**The reported association.** The surface measures the correlation on the analysis data. For binary
treatment, the value carries the treated fraction of your own fit. For continuous treatment, the
value reports what $A'=A+k_AU$ achieved on your dose distribution. The `(0, 0)` anchor cell
measures the original treatment, which gives the null level of the same data. A cell with constant
treatment reports no association, because its correlation is undefined. A failed cell keeps the
association of the treatment the surface built for it.

The `(0, 0)` cell returns the original estimate without a refit. A failed replacement or refit
remains visible as a `ReplicationFailure`. Successful cells report their displacement from the
original estimate, on the scale `movement_scale` names.

The surface supports these compositions with one or more cross-fitting draws.
Risk ratios, odds ratios, and PAF require a binomial outcome. The other rows support Gaussian and binomial outcomes.

| treatment | parameter | replayed estimator |
| --- | --- | --- |
| binary | backdoor-identified marginal ATE | ordinary TMLE, collaborative TMLE, or complete-outcome DR-TMLE |
| binary | one explicitly named marginal `ey1`, `ey0`, or `ey[...]` counterfactual mean | ordinary TMLE, collaborative TMLE, or complete-outcome DR-TMLE |
| binary | marginal `rr` risk ratio or `or` odds ratio | ordinary TMLE, collaborative TMLE, or complete-outcome DR-TMLE |
| binary | marginal or baseline-stratum `par` population attributable risk or `paf` population attributable fraction | exact ordinary TMLE |
| binary | baseline-stratum arm mean, ATE, risk ratio, or odds ratio | ordinary TMLE or collaborative TMLE |
| binary | marginal or baseline-stratum ATT or ATC | exact ordinary TMLE |
| binary | marginal or baseline-stratum `ey_regime[...]` mean or `ate_regime[...]` contrast | exact ordinary TMLE; fixed `Static`, `Rule`, or `Stochastic` intervention |
| binary or continuous | marginal `msm[...]` coefficient; baseline strata for binary identity-link MSMs only | exact ordinary TMLE; built-in identity, log, or logit link with fixed projection measure |
| binary | marginal `ey_ipsi[...]` mean or `ate_ipsi[...]` contrast | exact ordinary TMLE; fixed odds multipliers with a refitted mechanism |
| continuous | one explicitly named marginal or baseline-stratum `ey_shift[...]` policy mean | exact ordinary TMLE |
| continuous | one explicitly named marginal or baseline-stratum `ate_shift[...]` contrast | exact ordinary TMLE |

Every row accepts fixed probability weights under its listed estimators.

**Fixed policies and projections.** A regime surface holds its declared density $q(a\mid W)$
fixed. It refits the observed treatment mechanism and outcome regression after each perturbation.
It then targets the same functional, $E_W\sum_a q(a\mid W)Q(a,W)$, on that cell's empirical law.
Static assignments and baseline rules are degenerate densities under this definition.

An MSM surface holds its grid design, link, and projection measure fixed.
The estimator recomputes the projection from the perturbed data. A coefficient uses additive
movement on its original scale. Projection weights define the working approximation; observation
weights define the empirical population. The surface preserves both without combining their roles.

The operation checks typed identification, parameter keys, replay configuration, and every fitted
repeat before it draws the latent vector. It evaluates the original policy or projection callbacks
and compares their output with the stored arrays. A disagreement raises `CapabilityError`.
The replay freezes baseline policy and grid arrays after validation.
Continuous observed-dose evaluations use the declared deterministic callbacks on each new dose vector.
This also prevents an assessment from changing the original estimator's configuration.

| input | what stays fixed |
| --- | --- |
| `Static` | the assigned treatment label |
| `Rule` | the assignment at each original baseline row |
| `Stochastic` | the complete probability vector at each original baseline row |
| finite-arm `MSM` | terms, link, arm-specific design, and known projection weights |
| continuous-dose `MSM` | terms, link, integration grid, grid design, and raw grid weights |
| `Incremental` | odds multiplier, intervention name, and reference label |
| baseline strata and observation weights | original row membership and normalized weight |

The regime functional and influence curve follow the existing
[known-regime contract](point-treatment-tmle.md#known-regimes), which cites Díaz and van der Laan (2013).
Van der Laan and Gruber (2010), Section 6, supplies the working projection.
The pinned DoWhy refuter preserves the target specification when it refits perturbed data.
Pinned `lmtp` and `tmle3` supply policy and projection implementation provenance.
The [source locators](../references.md#sensitivity-analysis) distinguish these roles.
None of these sources supplies a sensitivity-adjusted interval for this composition.

`tests/unit/test_simulated_confounding_policies.py` compares nonzero cells with independent complete
refits. It also checks static-arm identities, fixed densities, projection weights, callback drift,
metadata mutations, assessment routing, and persistence. The registered deterministic-regime,
stochastic-regime, and MSM studies validate the unchanged estimator paths.

**Incremental interventions.** Each cell preserves the declared multiplier $\delta$ and refits the treatment mechanism $g$.
The estimator rebuilds $q_\delta=\delta g/(\delta g+1-g)$ and retains its mechanism targeting contribution.
The surface therefore compares the same multiplier rule under different fitted laws.
It does not preserve the original intervention probabilities.
Kennedy (2019), equation (1) and Corollaries 1–2, supplies the parameter and influence curve.
The [source audit](../references.md#sensitivity-analysis) separates that result from the qualitative perturbation.

A multiplier-one mean equals the natural-course mean and is refused before a random draw.
A contrast against that reference remains supported. Means and contrasts report additive movement.
The operation validates every stored repeat against its own initial, untruncated propensity predictions.
It checks intervention names, multipliers, densities, clever weights, derivatives, and the reference.

**Nonlinear and continuous MSMs.** The surface preserves built-in identity, log, and logit links.
It invokes the existing complete projection and targeting solver in each cell.
Movement remains a coefficient difference, including coefficients on log or logit scales.
It never substitutes a ratio-scale coefficient view.

For continuous treatment, the declared integration grid stays fixed after $A'=A+k_AU$.
The replay freezes raw grid weights. The estimator applies its existing trapezoid rule once.
The observed-dose design, raw weights, and support mask are recomputed at $A'$.
Freezing these observed arrays would evaluate the score at the original treatment.

Deterministic callbacks must accept the new dose vector. Callback and outcome-support failures remain recorded in their cells.
Custom MSM links require their own replay audit and remain refused.

A continuous MSM has no implemented dose-grid support diagnostic.
Its combined assessment records that row as unavailable and continues to the requested surface.
Outcome diagnostics remain available; integration-grid positions never become treatment-arm calibration rows.
`tests/unit/test_continuous_msm_assessment.py` checks these reporting boundaries.

| witness | what it checks |
| --- | --- |
| `tests/unit/test_simulated_confounding_incremental.py` | nonzero complete refits, mechanism rebuilding, natural-course refusals, repeated provenance, and persisted assessment |
| `tests/unit/test_simulated_confounding_msm.py` | built-in links, fixed quadrature, observed-dose rebuilding, support masks, and corrupted cached arrays |
| existing incremental and MSM estimator tests | unchanged targeting, influence curves, and projections |

These are deterministic diagnostic checks, not new repeated-sampling or interval claims.
The registered incremental and point-MSM artifacts remain unchanged.
The estimator refuses baseline strata for incremental targets and nonlinear or continuous MSMs.
Roadmap item X8 tracks those targeting extensions before any replay audit.
The point-MSM study covers identity-link finite-arm estimation; it supplies no nonlinear or continuous-MSM interval validation.

For `repeats > 1`, each non-anchor cell calls the replay estimator once. The estimator owns all
repeat draws and reports their coordinatewise median. Additive displacement compares the refitted
median estimate with the original median estimate. Ratio displacement compares their median log
estimates. The result records `n_repeats` and `repeat_aggregation="coordinatewise_median"`.

The root seed gives every non-anchor cell the same repeat seed sequence. It does not preserve
realised folds after the perturbation changes a stratification variable. Treatment-stratified or
outcome-stratified splitting can therefore assign different rows to folds under the same seed.

The root seed reproduces the folds of the original fit only when it equals that fit's seed. The
helper `resolve_assessment_seed` returns an explicit `random_state` first, the fit's own seed
second, and a fresh seed for an unseeded fit. A refit through `TMLE.refit` reuses the estimator's own seed
sequence only for a seed equal to its `random_state`. Under any other seed each non-anchor cell
rebuilds its folds, and the anchor keeps the original ones. Movement near the anchor can therefore
carry a fold artifact.

The binary mean path keeps the named counterfactual arm fixed. Each cell replaces only the
observed treatment and outcome before the complete refit. A sole reported mean needs only the
grid. A fit that reports several means also needs one explicit alias.

The binary ratio path keeps the numerator and denominator arms fixed. It validates their direction
and the stored log estimate before the latent draw. A ratio-only result needs only the grid.

**Population attributable parameters.** Let $\mu$ be the natural-course outcome mean and $\mu_a$
the counterfactual mean at the declared reference arm. Both means describe the selected baseline
population and use its fixed weights when declared.

| parameter | cell estimate | displacement scale |
| --- | --- | --- |
| `par` | $\widehat\mu_{\mathrm{cell}}-\widehat\mu_{a,\mathrm{cell}}$ | estimate difference |
| `paf` | $1-\widehat\mu_{a,\mathrm{cell}}/\widehat\mu_{\mathrm{cell}}$ | estimate difference |

Each non-anchor cell recomputes both means from its complete refit. Its natural-course mean uses
the perturbed outcome, rather than the original outcome. The reference arm and baseline membership
remain fixed. A sole PAR or PAF alias needs only the grid. A stratified result needs an explicit alias.

Hubbard and van der Laan (2008), Sections 1 and 3, define population-intervention contrasts and
their observed-outcome contribution. PAR reverses their intervention-minus-observed difference.
PAF takes the complement of their intervention-to-observed ratio. The pinned `tmle3`
`tmle3_Spec_PAR.R` composes the same two means, and `delta_functions.R` supplies both contrasts.
The [source locators](../references.md#point-treatment-and-stochastic-interventions) distinguish
that parameter construction from the descriptive stress surface.

PAF uses `movement_scale="estimate_difference"`, even though its definition contains a ratio.
A negative PAF remains valid and is not clipped or logged. A cell with zero observed outcome risk
cannot define PAF; the surface retains its failure. These conventions do not add sensitivity-adjusted inference.

The identified effect's method catalog refuses PAR and PAF under C-TMLE and DR-TMLE before fitting.
This includes outcome-adaptive C-TMLE. Their joint observed-law and intervention construction
needs separate estimator evidence. Low-level mean-group arithmetic does not remove that gate.
The surface preserves the same refusal and replays exact ordinary TMLE only for these targets.

The tests in `tests/unit/test_simulated_confounding_attributable.py` check the composition directly.

| instrument | test |
| --- | --- |
| weighted marginal and conditional cells equal a complete refit | `test_attributable_cells_equal_complete_weighted_refits` |
| observed and reference components remain distinct from frozen-mean, sign, denominator, and arm errors | `test_observed_and_reference_components_move_with_the_cell` |
| negative fractions retain identity movement, repeated fitting, and persistence | `test_negative_fraction_uses_identity_movement_and_repeat_aggregation` |
| zero observed risk retains a failed cell | `test_zero_observed_risk_retains_a_failed_fraction_cell` |
| the pandas and polars backends report equal cells, and a sole PAR or PAF alias reaches the surface with no named estimand | `test_attributable_backend_parity_and_sole_alias_facade` |
| C-TMLE and DR-TMLE refuse a PAR or PAF fit, and the surface withholds and refuses a forged result of either kind | `test_unevidenced_attributable_fit_refuses_upstream`, `test_unreplayable_attributable_estimators_are_withheld` |
| the surface refuses the natural-course mean, and no suggested alias offers it | `test_natural_course_refuses_and_is_absent_from_suggestions` |
| every registered point target is supported or refused for a stated reason | `test_registered_point_targets_have_an_explicit_surface_disposition` |
| inconsistent result, identity, or registry metadata fails before randomness | `test_attributable_metadata_refuses_before_randomness`, `test_attributable_identity_and_registry_metadata_refuse_before_draws` |
| PAF refuses a Gaussian outcome before the first draw | `test_fraction_family_refuses_before_draws` |

The continuous path keeps the fitted modified treatment policies fixed. Each cell replaces only
the observed dose and outcome before the complete refit.

**Fixed observation weights.** Ordinary-TMLE surfaces accept declared fixed probability weights
for every row in the support table. Binary complete-outcome collaborative-TMLE and DR-TMLE
surfaces also accept them. The weights define $dP_w=w\,dP/E_P[w]$. The operation keeps each
normalized weight on its original row during both replacements and every complete refit.

The latent value stays independent and standard normal. Each weight depends only on its own
observed row, so the joint law factorizes:

$$
d(P_w\times\Phi)(o,u)=\frac{w(o)}{E_P[w]}\,dP(o)\,d\Phi(u).
$$

Benkeser et al. (2017), Theorem 1, supplies the complete-outcome DR-TMLE corrected curve and its
remainder conditions.
`tests/unit/test_remainder_drtmle.py::TestAWeightedFitTransportsToTheTiltedLaw`
transports its conditional expectations, mechanism, marginal means, and scores to $P_w$. It keeps
a wrong-transport control with a nonzero remainder.
`tests/unit/test_simulated_confounding.py::test_fixed_weight_drtmle_witnesses_the_weight_on_the_reduced_regressions`
adds the applied witness. It removes the weight from the reduced regressions alone, and it
requires the cell estimate to move.

Van der Laan and Gruber (2010), Sections 2, 5.1, and 6, define C-TMLE for a generic law and its
empirical distribution. Replacing that law by $P_w$ makes $P_{n,w}$ the empirical measure.
Selector strategies use the same normalized row mass in nuisance fits, targeting, outcome loss,
the influence-curve penalty, cross-validated risk, and the plug-in. The outcome-adaptive strategy
uses it in the outcome fits, categorical mechanism fit, targeting, and plug-in.

The fixed-weight C-TMLE tests cover greedy selection, both data-adaptive ordered preorders, an
explicit ordering, discrete selection, and outcome-adaptive fitting. They reconstruct a selected
path's weighted squared loss, weighted log-likelihood loss, influence-curve penalty, and nested
cross-validated risk term by term. They do not read the value that a production scoring method
returns. The nested reconstruction does call the production path search inside each training fold.
That search runs the production propensity fits, the targeting step, the loss, and the penalty. The
reconstruction discards the risk each candidate carries, and scores that candidate again on the
held-out rows.

The same test runs the production fold splitter on the surface's root seed. It compares the result
against the fold assignment the refit stored. Component mutations strip the weights from the loss,
the influence curve, the targeting step, the fold nuisance fit, the intercept mechanism, or the
candidate mechanism. Each mutation moves the stored nested-risk array.

Three tests ask whether a search decision reads the observation weights. The candidate risk is the
loss plus the penalty. A greedy search on a binomial fit and a logistic preorder on a Gaussian fit
each lose the weights from the loss. Each one moves the risk it is scored on. The
partial-correlation preorder never reaches the loss, so a third test removes the weights from that
ranking and asserts the order it produces. An explicit ordering carries no witness, because the
reader declares that order.

A fourth test pins the selected adjustment set, and not only the order the search visited. A moved
order that leaves the selection alone leaves the fitted mechanism identical. The fixture carries
two candidates that serve opposite weight blocks, in a mass ratio of nine to one. The weighted
search selects one candidate and the unweighted search selects the other, while the visited order
stays equal.

End-to-end mutations drop every selector weight or drop the outcome-adaptive
mechanism weights. Each mutation moves a manual refit that runs on the surface's seeds at strengths
0.2 and 0.3. Neither control reads a surface cell, because its grid holds the anchor alone and the
anchor runs no refit. Each control requires a move above one part in a thousand of the unmutated
refit's estimate.

The tests rebuild the vector-target penalty and the ratio penalty once each. The `ey1`, `ey0`, and
`or` estimands reuse the same arm-curve and delta-method arithmetic, and they carry no separate
dropped-weight mutation.

The pinned R `drtmle` 1.1.2 implementation accepts no observation-weight argument. It supplies no
weighted numerical comparator. This surface therefore claims no weighted R parity and broadens no
interval claim.

The canonical R `ctmle` 0.1.2 source at commit `18de559` also accepts no observation-weight
argument. The `ctmleDiscrete` entry point, which the pinned selector-parity study calls, takes
none, and neither do its `stage2` and `cv` helpers. The `ctmleGeneral`, `stage2_general`, and
`cv_general` paths use unweighted operations. The archived `ctmle3` source supplies
outcome-adaptive control flow but no weighted comparison. The fixed-weight C-TMLE surface
therefore claims parity with neither implementation.

Conditional on the observed rows and fixed weights, the simulation draws each latent value
independently from a standard normal law without using either. After that draw, the realized
weighted empirical distribution is discrete. Its latent marginal need not have weighted mean zero
or variance one, and chance association with the original treatment need not vanish. The anchor's
reported association records that finite-sample imbalance. Each cell reports the same parameter
functional on its perturbed weighted empirical law. This statement does not claim that the
operation reproduces the sampling or selection mechanism.

Hartman and Huang (2024) treat sensitivity to a confounder that the weighting model omits. They
report a bound, a robustness value, and a benchmark. This surface reports none of those, so their
method does not govern it.

The induced treatment association uses weighted means, variances, and covariance. Numeric
calibration also uses weighted scaling, model fitting, prediction-change fractions, and moments.
Constant weights use the exact unweighted calculation path. A common weight scale leaves the cell
estimates, the displacements, the induced associations, and the calibration strengths numerically
unchanged. The agreement is numerical rather than bitwise, because `check_weights` normalizes each
weight vector to mean one.
`tests/unit/test_simulated_confounding.py::test_fixed_weight_surface_is_invariant_to_a_common_weight_scale`
pins ordinary TMLE to 1e-12.
`tests/unit/test_simulated_confounding.py::test_fixed_weight_drtmle_surface_is_invariant_to_a_common_weight_scale`
applies the same tolerance to the DR-TMLE cells. The supplied scale is itself a descriptive
measurement, so
`weight_report.scale` does change with it.
`tests/unit/test_simulated_confounding.py::test_fixed_weight_ctmle_surface_is_invariant_to_a_common_weight_scale`
applies the same tolerance to selector-based and outcome-adaptive C-TMLE cells.

**Refusals.** `_FIT_WIDE_RULES`, in
`src/cleverly/sensitivity/_simulated_confounding_request.py`, is one ordered table. It names every
boundary that neither the requested estimand nor the strength grid can move. `_fit_wide_refusal`
walks that table and returns the first reason that applies. Six missing-science stops apply in
this order: longitudinal, multi-arm, missing-outcome, intermediate, estimated-weight, and
clustered fits. Each one carries a roadmap item, and the refusal table below gives it a row.

Twelve provenance and shape stops read the object you hold rather than the state of the science.
The `result_type` stop runs second, before the multi-arm stop, because every later rule reads a
field that only a `TMLEResult` declares. The other eleven follow the clustered stop. The table
below gives all twelve in contract order.

| rule | refuses |
| --- | --- |
| `result_type` | an artifact that is not exactly a point-treatment `TMLEResult` |
| `missing_estimator` | a restored or legacy result that stores no replay estimator |
| `repeat_provenance` | a stored draw count, a `config.crossfit.repeats`, and a replay estimator `repeats` that disagree |
| `binary_estimator` | a binary fit made by an estimator other than `TMLE`, `CTMLE`, or `DRTMLE` |
| `continuous_estimator` | a continuous fit made by anything but exact ordinary `TMLE` |
| `outcome_family` | an outcome family other than `gaussian` or `binomial` |
| `weight_kind` | a weight kind other than `probability` |
| `weight_provenance` | a `weights_name` and a `WeightSpec` name that disagree |
| `undeclared_weights` | nonconstant observation weights with no declared weight column |
| `identification` | a legacy fit that records no identification metadata |
| `functional` | an identified functional other than a backdoor mean contrast |
| `provider` | backdoor provenance from anything but a registered explicit adjustment set |

`_validate_request` applies the first reason before calibration, a random draw, or a refit. The
assessment capability walks the same table, so `available`, `status`, `reason`, and direct
execution name the same stop for all eighteen. A fit that reaches the `provider` stop reports
`reason="simulated_confounding needs registered explicit-adjustment backdoor provenance"` on its
capability row.

The rows below record the six missing-science stops, plus the estimands and compositions the
surface refuses. The
`kind` column uses the vocabulary of
[how to read a refusal](scope-and-refusals.md#how-to-read-a-refusal), plus
`waiting on published theory` from the [roadmap's eligibility rules](../roadmap.md#eligibility).

| refused | kind | why |
| --- | --- | --- |
| a longitudinal result | waiting on published theory | no time-indexed latent law covers treatments, censoring, histories, outcomes, and contrasts. See [F13](../roadmap.md#f13-longitudinal-simulated-confounding-replay) |
| multi-arm treatment | waiting on published theory | no source-backed category-valued perturbation defines the contrast. See [F8](../roadmap.md#f8-multi-arm-simulated-confounding-stress-surface) |
| a missing outcome | waiting on published theory | no joint observation, treatment, and outcome perturbation law has identified refit semantics. See [F12](../roadmap.md#f12-missing-outcome-simulated-confounding-replay) |
| a controlled direct effect, or any fit that carries an intermediate variable | waiting on published theory | no ordered treatment, intermediate, observation, and outcome law has a controlled contrast contract. See [F15](../roadmap.md#f15-controlled-direct-effect-simulated-confounding-replay) |
| estimated observation weights | waiting on published theory | replay lacks stored model provenance, target-population semantics, and a regeneration rule. See [F11](../roadmap.md#f11-estimated-weight-simulated-confounding-replay) |
| a clustered fit | waiting on published theory | no source chooses a row-level, cluster-level, or mixed latent cause. See [F9](../roadmap.md#f9-clustered-simulated-confounding-stress-surface) |
| identification other than a backdoor mean contrast with explicit adjustment | not written yet | the surface reads registered explicit-adjustment provenance |
| ATT or ATC under C-TMLE or DR-TMLE | not written yet | `CTMLE` and `DRTMLE` refuse these functionals when they estimate, so no such fitted result exists |
| a requested baseline stratum under DR-TMLE | not written yet | a DR-TMLE fit refuses `strata=` when it fits, so no such fitted result exists. The guard keys on the requested stratum, not on `data.has_strata`. See [X8](../roadmap.md#x8-stratified-incremental-and-msm-targeting) |
| baseline strata with an incremental target or nonlinear or continuous MSM | not written yet | ordinary TMLE lacks stratified alternating equations or continuous-dose targeting for these groups. See [X8](../roadmap.md#x8-stratified-incremental-and-msm-targeting) |
| a custom MSM link | not written yet | only built-in identity, log, and logit links have a replay audit |
| a custom intervention type | not written yet | only exact `Static`, `Rule`, `Stochastic`, and `Incremental` declarations have an audited baseline-input contract |
| a regime, incremental target, or MSM under C-TMLE or DR-TMLE | not written yet | the method catalog lacks the corresponding collaborative score or reduced-dimension correction. See [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| continuous-treatment C-TMLE and DR-TMLE | not written yet | both estimators refuse a modified-policy functional. See [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| PAR or PAF under C-TMLE or DR-TMLE | not written yet | the method catalog lacks a score or correction for these observed-law contrasts. See [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| `NaturalCourseMean`, reported as `ey_obs` | wrong by construction | $E[Y]$ has no counterfactual treatment term. Outcome perturbation alone cannot make it a confounding diagnostic |
| the policy mean of a zero-delta shift | wrong by construction | $d_0(a, w) = a$ on both branches, so the policy is the natural course. Its mean is $E[Y]$, and no counterfactual treatment dependence remains for a common cause to move. The `ate_shift[...]` contrast that uses this policy as its reference is still accepted |
| an incremental mean at multiplier one | wrong by construction | its mean is $E[Y]$. Contrasts with this reference remain supported |
| a categorical benchmark covariate | waiting on published theory | no logical-covariate calibration maps categories to these perturbation strengths. See [F10](../roadmap.md#f10-logical-categorical-confounder-calibration) |

Some rows above record an upstream limit rather than a surface limit. The identified effect's
method catalog refuses unsupported variant targets at `estimate()`, before it builds an estimator.
`DRTMLE` itself refuses a baseline stratum during the fit. No fitted result reaches the surface
guard in these cases. `_replay_refusal` keeps that guard as defence in depth.

| composition | layer that refuses | when | message |
| --- | --- | --- | --- |
| ATT or ATC under C-TMLE or DR-TMLE | the identified effect's method catalog | `estimate()` | `method 'collaborative_tmle' cannot estimate ATT: no collaborative score is evidenced for this functional`. `DRTMLE` names its reduced-dimension correction instead |
| a modified-treatment policy under C-TMLE or DR-TMLE | the identified effect's method catalog | `estimate()` | `method 'drtmle' cannot estimate ModifiedTreatmentPolicy: no reduced-dimension correction is evidenced for this functional` |
| a baseline stratum under DR-TMLE | `DRTMLE`, in the shared targeting loop of `src/cleverly/estimators/tmle.py` | the fit | `baseline strata are not yet combined with the 'mean' group's alternating targeting equations`. `needs_reduction` holds because a DR-TMLE fit always builds reduced regressions |
| PAR or PAF under C-TMLE or DR-TMLE | the identified effect's method catalog | `estimate()` | `method 'collaborative_tmle' cannot estimate PopulationAttributableRisk: no collaborative score is evidenced for this functional`. DR-TMLE names its reduced-dimension correction; PAF names its own type |
| regime or MSM under C-TMLE | the identified effect's method catalog | `estimate()` | `method 'collaborative_tmle' cannot estimate RegimeMean: no collaborative score is evidenced for this functional`. Other targets name their own type |
| regime or MSM under DR-TMLE | the identified effect's method catalog | `estimate()` | `method 'drtmle' cannot estimate RegimeMean: no reduced-dimension correction is evidenced for this functional`. Other targets name their own type |

The surface also refuses a constant benchmark covariate. That check reads the requested covariate
rather than the fitted result, so it sits outside the fit-wide table.

`test_fit_wide_refusals_have_one_order_and_match_assessment` checks the shared order and exact
capability reasons. `test_real_mar_fit_refuses_before_calibration_draw_or_refit` uses a fitted
ordinary MAR estimator. `test_randomized_missing_outcome_fit_refuses_simulated_confounding_before_work`
uses the separate randomized DR-TMLE construction. Both tests fail if refusal reaches calibration,
the latent draw, or a refit.

`tests/unit/test_simulated_confounding.py::test_weight_refusals_and_provenance_tampering_precede_draws_and_refits`
pins the three weight stops before the first draw and refit. It also tampers with stored
provenance, so a passing run witnesses each refusal on a real fitted result.

Numeric calibration follows the maintained DoWhy source as secondary implementation provenance.
For a binary variable, it reports the class-prediction change after one standardized column is set
to zero. For a Gaussian outcome or continuous dose, it reports `corr(W_j, V) * sd(V)`.
All terms use $P_w$ when the fit declares fixed probability weights.

The Gaussian calibration is signed. It carries the covariate's own direction of association with
the outcome or with the dose. Each axis converts a calibrated value into a declared strength by its
own rule, because the two perturbation laws carry opposite signs.

| axis | law | conversion |
| --- | --- | --- |
| outcome | $Y'=Y-k_YU$ | the law subtracts, so an outcome strength of $k_Y$ calibrates at $-k_Y$. To match a covariate that calibrates at $+c$, declare an outcome strength of $-c$ |
| treatment, continuous dose | $A'=A+k_AU$ | the law adds, so a treatment strength of $k_A$ calibrates at $+k_A$. To match a covariate that calibrates at $+c$, declare a treatment strength of $+c$ |

A binary treatment or a binomial outcome has no such conversion. Its calibration is the
class-prediction change fraction, which carries no sign.

Calibration does not select or modify the grid. It is not partial R-squared and does not reuse the
omitted-variable `benchmark()` scale.

The maintained source is pinned at revision `2116d5c`. The implementation does not copy its
cumulative cell mutations, schedule-dependent random draws, non-exact zero refit, automatic
ranges, categorical encoded-column deletion, or unstructured failure behavior.

### E-value

**How.** [`sensitivity/evalue.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/evalue.py)
implements VanderWeele and Ding (2017): $E = RR + \sqrt{RR(RR-1)}$. It computes the E-value for the
point estimate and, separately, for the confidence limit, because the second is the one an
adversarial reader asks for.

The selected path depends on the reported contrast and retained artifacts.

| request | E-value path |
| --- | --- |
| reported risk ratio | use it directly and mark the result exact |
| unambiguous default binary marginal ATE or odds ratio from ordinary TMLE | retarget cached nuisances to the matching risk ratio and mark the result exact; combined runs include this cheap retarget by default |
| explicit reported odds ratio, or default odds ratio without exact retarget support | use the common-outcome approximation $\sqrt{OR}$ and mark the result approximate |
| binary ATE without exact retarget support, with a usable reported reference-arm mean | hold the baseline risk fixed and mark the result approximate; includes DR-TMLE, collaborative TMLE, and CV evaluation |
| Gaussian ATE, ATT, or ATC | standardize by the observed outcome standard deviation, weighted on a weighted fit, and mark the result approximate |
| binomial ATT or ATC | refuse because the conditional baseline risk and conditional ratio target are absent |
| level or non-arm parameter | report `not_applicable` because no supported two-arm contrast exists |
| binomial ATE without exact retarget support or a usable reported baseline; controlled direct effect needing derivation | report `unavailable` and name the missing evidence, artifact, or target |
| several eligible contrasts and no explicit estimand | report `deferred` and name `estimand` in the next step |

Several eligible contrasts require an explicit alias, which is the `deferred` row above.
[The status contract](#the-status-contract) states that rule for every operation that shares it.
Combined runs select availability and cost from the alias before they apply the cost flags.

Raw results compose arm identities forward from fitted treatment metadata.
Explicit structured keys remain authoritative. No routing step parses display aliases.
The fixed-baseline approximation ignores baseline sampling error and requires the contrast's matching reference arm.
Its availability does not depend on persistence or estimator retention.

A usable reference-arm mean is finite, positive, and separated from zero by its own standard error.
The conversion divides by that mean, so a mean at zero leaves the ratio without a stable denominator.
The library also refuses a risk difference at or below the negative of that mean.
Such a difference implies a nonpositive risk in the contrast arm, and no risk ratio describes one.
Both refusals report `unavailable` and name the two reported numbers.

The conversion is affine, so the lower interval bound can leave the parameter space while the point ratio stays inside it.
Only the lower bound can leave it. `normal_ci` gives `high >= psi`, and the refusals above force `baseline.psi > 0` and `baseline.psi + psi > 0`.
The report truncates the lower bound at 0, records the untruncated value in `truncated_bound`, and repeats it in the note.
The `to_dict` mapping and the battery row both carry that value, so no surface presents the 0 as a converted confidence limit.

What the truncation means for the confidence limit depends on the side of the null the point ratio is on.

| point ratio | bound the E-value reads | effect of the truncation |
| --- | --- | --- |
| at or above the null | the truncated lower bound | the interval covers the null, so the confidence-limit E-value is 1 |
| below the null | the untruncated upper bound | none. The interval can still exclude the null, and the confidence-limit E-value can exceed 1 |

For rare outcomes, the odds ratio itself approximates the risk ratio. The square-root
transformation addresses common outcomes and can lie above or below the risk ratio.
See [VanderWeele's analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC5617805/).

The Gaussian path first applies Chinn's odds-ratio to standardized-mean-difference relation.
It then applies the common-outcome square-root approximation before calculating the E-value.
The resulting conversion is $RR \approx \exp((1.81 / 2)d) = \exp(0.905d)$.
This retains an approximate analysis, not an exact continuous-outcome risk ratio.

A weighted fit standardizes by the weighted outcome standard deviation.
The estimate targets the population the observation weights describe, and the standardizing scale describes the same population.
The weighted form divides by the reliability-weight correction $\sum w - \sum w^2 / \sum w$.
That correction is $n - 1$ when every weight is one, so an unweighted fit reports the plain sample standard deviation.

### Missingness tilt and tipping gamma

**How.** [`sensitivity/missingness.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/missingness.py)
implements the Scharfstein, Rotnitzky and Robins (1999) tilt. It sets

$$
\bar{Q}^{\text{miss}}_\gamma = \operatorname{expit}\{\operatorname{logit} \bar{Q}^* + \gamma\}
$$

for the unobserved outcomes, and mixes it with $\bar{Q}^*$ by the estimated missingness
probability. At $\gamma = 0$ it reproduces the missing-at-random estimate by construction, which is
the control that says the tilt is wired in. `arm_gamma=` gives per-arm tilt directions and must
name every arm. `tipping_gamma()` inverts the tilt for the value at which the conclusion changes.

This is a retarget operation and not a refit.

### The scope rule

A point-treatment sensitivity formula is not reused on longitudinal data. `LTMLE` reports these
operations `unavailable` with the reason its own capability row declares, rather than borrowing a
derivation. Stagewise support, scores, and nuisance loss are supported longitudinally, because each
has its own derivation.

Tan (2025) derives population sensitivity bounds for binary, static longitudinal strategies, but
no sample estimator for them. [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation)
states the contracts that source leaves open.

## Refutation and simulation you run

These fit new models. They cost what a fit costs, multiplied by the number of draws.

### Refutation operations

**Why.** A diagnostic reads the fit you have. A refuter constructs a case whose answer is known and
checks that the workflow returns it.

**What each tells you.** [`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py)
ships seven operations. Six test the implementation. The negative-control outcome tests a design,
and the paragraph below states its boundary.

| refuter | what it does | what must happen | what it tests |
| --- | --- | --- | --- |
| `placebo` | permutes the treatment column and refits | the estimate goes to zero | the pipeline, not the data |
| `random_common_cause` | adds an irrelevant covariate and refits | the estimate does not move | the adjustment set is not sensitive to noise |
| `subset` | refits on random subsamples | the scatter is about one standard error | the reported standard error is the right size |
| `negative_control_outcome` | refits on an outcome the treatment cannot affect | the estimate goes to zero | the design, under the control assumptions the paragraph below states |
| `dummy_outcome` | draws independent Gaussian noise and refits | the empirical draws include zero | outcome replacement and the full estimator pipeline |
| `simulated_outcome` | draws `f(W) + effect * A + epsilon` and refits | the empirical draws include the declared effect | adjustment and treatment terms in the full pipeline |
| `bootstrap_measurement_error` | bootstraps, perturbs declared adjustment variables, and refits | the empirical draws include the original estimate | stability under the declared measurement error |

A refuter refits the nuisance models once for each replication. The three default operations use
five replications each, so `refute()` costs about 15 fits. Empirical refuters use 100 draws by
default because their rule reads a distribution. `run_all(include_refits=True)` runs `refute()`.

A fit that declared `split_plan=` refuses two of these operations. A supplied plan labels the rows
it was realized on, by position. It cannot label a refit that drops rows or draws them with
replacement. So `refute()` raises `CapabilityError` for `subset` and for
`bootstrap_measurement_error` before it refits anything, and it names both the plan and the test.

A `bootstrap_measurement_error` draw holds the declared number of rows, so only the request
identifies it. The `placebo` and `random_common_cause` operations still run, because each one
replaces a column and leaves every row in its position. A `run_all(include_refits=True)` battery
records the refusal as an `unavailable` row and continues to the rest of the battery. See
[reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) for the table.

`refute()` draws its randomization from the seed of the fit, unless the caller passes
`random_state`. A fit that carries a seed gives the same refutation on every call. A fit that
carries no seed gives a different refutation on every call.

The report records the seed under `random_state`. Pass that value back to `refute()` to obtain
the report again. The seed governs the perturbations and the refits they feed, so it repeats
the report of a fit that carries no seed of its own. The seed applies to a copy of the
estimator, so a refutation never changes the fit it examines.

Each empirical draw derives a child seed from the recorded root seed. The perturbation and its
full refit use that child seed. `report.draws_frame(name)` reports every child seed, estimate,
standard error, family, and failure. `GeneratedOutcomeRecord` remains an alias for the shared
`EmpiricalRefitRecord`.

`EmpiricalInclusionRule` uses a two-sided empirical rank and inclusive half-ties. It passes only
when the empirical probability strictly exceeds alpha, matching the maintained DoWhy convention
that a probability at or below alpha fails. The default rule uses alpha 0.05, requires 40
successful draws, and fails when any refit fails. A failure stays in the report as the shared
`ReplicationFailure` record, so the operation never reports only the conditional distribution of
successful fits.

The rule count and the draw budget are separate defaults. `DEFAULT_OUTCOME_REPLICATES` is 100 and
sets `n_replicates`. `EmpiricalInclusionRule()` requires 40 successful draws and sets no budget.
The operation refuses a budget below the rule's minimum before it refits anything.

Bootstrap measurement error uses the same rule against the original estimate. It draws an iid or
whole-cluster sample through the inference bootstrap design. It perturbs numeric variables with
mean-zero Gaussian noise after sampling. The noise scale is the declared multiplier times the
selected variable's bootstrap-sample standard deviation.

Each sampled cluster occurrence receives a distinct code before perturbation. Repeated draws of
one source cluster therefore remain separate during the refit.

The categorical path recovers original logical levels from `CausalData.encodings`. Each changed
row draws uniformly from the other levels. The operation then rebuilds the complete drop-first
indicator block. Boolean covariates use the same two-level path.

The operation checks every condition in the table below before the first refit. Each row is one
`CapabilityError` branch of `_validate_measurement_error_eligibility` in
[`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py),
and each message names the variable it rejects.

| what the operation refuses | what it requires instead |
| --- | --- |
| a result whose data is not point-treatment `CausalData` | a point-treatment `CausalData` result |
| `resampling="cluster"` on data that carries no cluster ids | declared cluster ids, or iid resampling |
| a selected strata variable | a variable outside the strata, because the operation cannot perturb target metadata coherently |
| a generated indicator name | the original categorical variable that the indicator encodes |
| a name that is not an original adjustment variable | one of the original names the message lists |
| a categorical variable whose encoded block lost an indicator to duplicate-column removal | a retained indicator for every level the encoding generated |
| a selected variable whose values are all 0 or 1 and that carries no `CategoricalEncoding` | the same variable declared to `CausalData.from_frame` as a boolean or categorical column, because relative Gaussian noise makes an undeclared indicator real-valued |
| a selected numeric variable that is constant | a variable with nonzero spread, because a relative noise scale of zero leaves every draw unperturbed |

The report records the declaration, requested draw count, resolved mode, rule, estimates,
standard errors, child seeds, and failures.

**Alpha is a width here, and not a false-alarm rate.** The rule decides one question. Does the
declared effect lie inside the central `1 - alpha` of the refit estimates? At the default alpha
that is the central 95%.

A correct estimator centres the refit distribution on the declared effect, so each draw falls
above it with probability about one half. Under 100 draws the rule then fails only when at most
two draws fall on one side. That probability is `2 * (1 + 100 + 4950) / 2**100`, which is about
`8e-27`. More draws do not buy power. They stabilise the quantile the rule reads.

The default minimum of 40 draws is the smallest budget at alpha 0.05 that can fail on anything
short of a one-sided sweep. With 40 draws, one estimate on the minority side gives `2 / 40`, which
equals alpha and fails. With 39 draws, one estimate on the minority side gives `2 / 39`, which
exceeds alpha and passes. The rule therefore refuses a declaration whose `minimum_draws * alpha`
is below 2, because such a rule fails only when every draw falls on one side.

The first process catalog covers Gaussian outcomes and additive `ate`, `att`, or `atc` contrasts.
`refute()` and its `_validate_generated_eligibility` helper in
[`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py)
check every condition in the table before the operation refits anything. Each row is one
`CapabilityError` branch, and each message names what is missing.

| what the operation refuses | what it requires instead |
| --- | --- |
| an inclusion rule of any other type | the exact registered `EmpiricalInclusionRule` declaration |
| a process declaration of any other type | the exact registered `GaussianIndependentOutcome` or `GaussianAdjustmentOutcome` |
| a process whose family is not `"gaussian"` | the Gaussian family, which is the one with an implemented effect derivation |
| a legacy fit that carries no identification metadata | identification metadata on the result |
| a functional that is not `BackdoorMeanContrast` | a backdoor-identified additive mean contrast |
| a provider that is not `ExplicitAdjustmentProvider` | registered backdoor provider provenance |
| an original outcome family that is not `"gaussian"` | an original family equal to the process family, so a binomial fit is refused |
| an estimator configured with a family other than `"auto"` or `"gaussian"` | a configured family that accepts the declared process |
| a saved outcome learner with no regression-capable route | a regression-capable outcome learner |
| a longitudinal functional | a point-treatment functional |
| a treatment that is not binary | code one against code zero |
| an intermediate node or a controlled direct effect | no intermediate node |
| a fit with missing outcomes | complete outcomes |
| an MSM functional or the `msm` axis | the `arm` axis |
| an intervention-indexed functional | the `arm` axis with no declared interventions |
| a parameter key that is not a structured `ParameterKey` | a structured key for the selected estimand |
| a ratio estimand, or any other non-additive contrast | `ate`, `att`, or `atc` |
| disagreement between the functional target, the identified estimand, and the parameter key | one estimand named by all three |
| a missing registered identification artifact | the artifact `TARGETS` records for that target |
| an arm contrast that does not resolve to two distinct arms | a resolvable contrast of two distinct arms |
| a draw budget below the rule's minimum | `n_replicates` of at least `minimum_draws` |

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate
from cleverly.validation import GaussianAdjustmentOutcome, GaussianNoise

frame, _ = make_linear_ate(n=200, seed=21)
study = CausalStudy(
    frame,
    design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
)
result = study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=21,
)
process = GaussianAdjustmentOutcome(effect=0.5, noise=GaussianNoise())
report = result.diagnostics.refute(
    tests=("simulated_outcome",),
    simulated_outcome=process,
    random_state=21,
)
print(report.summary())
```

Sharma and Kiciman (2020) define the refutation framework. The maintained DoWhy dummy refuter,
pinned at
[`2116d5c`](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_refuters/dummy_outcome_refuter.py),
supplies secondary evidence for independent noise and `f(W) + h(A)`. That implementation uses a
normal rule below 100 draws, which `perform_normal_distribution_test` in
`dowhy/causal_refuter.py` applies. It declares no failure policy at all. The pinned file contains
no `try` block, and its refits run under `joblib.Parallel`, so one failed refit aborts the whole
refutation. `cleverly` keeps each failed refit as a `ReplicationFailure` record and fails the
refutation under the rule stated above.

The maintained DoWhy bootstrap refuter at the same revision supplies secondary control-flow
evidence for measurement-error sampling and refitting. `cleverly` does not copy three defects in
that source. It uses logical metadata for numeric and categorical variables. It does not reuse a
Boolean probability array for another categorical variable. It derives a distinct child seed for
each draw.

A negative-control outcome must have no causal path from treatment. It must also share the relevant
confounding structure with the primary outcome. A non-null result flags residual bias or a bad
control, and the refuter cannot tell you which. A null result does not establish that unmeasured
confounding is absent. See [negative controls](../references.md#negative-controls).

### Coverage studies

**Why.** Coverage, bias, and standard-error calibration are claims about repeated sampling. One fit
contains no information about any of them.

**What it tells you.** Three numbers, from
[`validation/simulation.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/simulation.py).

| number | how it is computed | how to read it |
| --- | --- | --- |
| coverage | the share of replications whose interval contains the truth | sustained undercoverage beyond Monte Carlo uncertainty indicates invalid intervals on that law |
| root-n bias | $\sqrt{n}$ times the mean error | bounded values support a negligible first-order bias claim. They do not establish efficiency |
| SE ratio | mean reported standard error over the empirical standard deviation of the estimates | one means the reported uncertainty matches the real spread |

**How.** `CoverageStudy` draws from a generator with a known truth, runs the complete estimator on
each draw, and summarises through `summarize_replications`. A failed draw is retained as a
`ReplicationFailure` record carrying its index, its seed, and its exception. A study that silently
replaced failed draws would report the distribution of the draws that happened to work.

The generators live in
[`datasets/`](https://github.com/esbraun/cleverly-tmle/tree/main/src/cleverly/datasets) and each
one carries an exact `truth`.

**A simulated law is an instrument, and it can be wrong.** A coverage study is evidence only if the
number it calls the truth is the number an adjusted fit estimates. Two shipped clustered generators
once failed that test, and
[evidence.md](evidence.md#a-simulated-law-is-an-instrument-too-and-it-can-be-wrong-the-same-way)
records what went wrong and what each generator now asserts about itself.

### Variable importance

`variable_importance` gives each candidate covariate the treatment role in its own fit, and
reports the target-relevant change with multiplicity-adjusted p-values
([`variable_importance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/variable_importance.py)).
It is an assessment of the fitted causal workflow. It is not a predictive feature-importance score,
and it introduces no new influence function.

## How the library certifies itself

The three layers above review *your fit*. This layer reviews *the implementation*. It is the
evidence that the equation `score_check` solved is the right equation.

The instruments go blind in different places, and the differences are the reason there are six of
them. [evidence.md](evidence.md) records which instrument covers which registered estimand, in both
directions, and it is a test rather than a note.

| instrument | why it exists | what it tells you | how it tells you | what it cannot see |
| --- | --- | --- | --- | --- |
| **exact oracle law** | an estimator has to recover a parameter that was computed rather than estimated | the reported number is the parameter, exactly | a finite-support law whose every cell probability is a multiple of $1/N$, so an $N$-row frame **is** the law. Handed oracle nuisances, the fit is exactly right and $\epsilon$ is zero | nothing about a term that is zero at the truth |
| **Gateaux comparison** | the influence curve is what every interval is built from | the reported curve is the pathwise derivative of the parameter | complex-step differentiation of an independently written functional, compared at about `1e-14` absolute with `rtol=0` | a sign on any block that vanishes at correct nuisances, and any counterfactual block, because $\epsilon$ is zero there |
| **second-order remainder** | double robustness *is* the remainder carrying both nuisance errors | one wrong nuisance still leaves the remainder second order | the von Mises expansion evaluated at nuisances that are wrong on purpose, against a longhand form of the exact remainder | a first-order error that cancels inside the remainder |
| **exact identity** | some mistakes are algebraic and cheap to catch | a relation that holds by definition holds bit for bit | relabelling the arms, a null outcome model giving zero, weights scaling out, the one-step and iterative solvers agreeing | anything symmetric in whatever the identity is symmetric in |
| **theorem check** | the anchor the others need | the implementation agrees with the source's own theorem | evaluation at values where the quantity does **not** vanish | nothing the theorem does not state |
| **deliberate mutation** | a passing test proves nothing unless a wrong version fails it | each plausible way of building the thing wrong is shown to fail | the component is broken on purpose and the suite is required to go red | a mistake nobody thought to make |

Three supporting rules make the table mean what it says.

- **The oracle laws share no code with the library.** `tests/unit/test_oracle_independence.py`
  asserts that the oracle modules never import `cleverly`. A shared helper would move both sides of
  the comparison equally.
- **A heading is not enough.** `tests/unit/test_registry.py::TestEvidenceManifest` checks the
  evidence table against the target registry in both directions, checks that every module named
  there exists, and checks that the oracle-law column names the law whose functional really has the
  branch.
- **Cross-fitting is checked without a tolerance.**
  `tests/unit/test_crossfit_leakage.py` rigs a law in which one covariate is constant within a
  cluster and the outcome *is* that covariate with no noise. A nearest-neighbour learner then
  reproduces a held-out row bit for bit if and only if a same-cluster row was in its training set.
  The assertions are array equality, so leakage is not a matter of degree.

### The oracle-law gate

Registering a target whose reported parameters have no branch in an oracle law's `functional` is a
test failure rather than an oversight caught in review. The evidence this package offers that an
influence curve is correct is that it agrees with one obtained by complex-step differentiation of
an independently written functional on an exactly representable law. An estimand without that has
no such evidence.

The gate walks the *parameter* names a target reports rather than the target name, so a per-arm
target needs an oracle for each arm. A target intended for more than two arms needs one on the
three-armed law, because two arms cannot distinguish code that keys by arm from code that has two
columns and calls them 0 and 1. The gate runs in both directions. An oracle branch that no target
reports is dead code, so a law and the registry must cover each other exactly.

### Registered repeated-sampling studies

The instruments above ask whether each parameter is implemented correctly. A registered study asks
the complementary question. Apply a complete estimator to samples from a known law, and does its
bias and its uncertainty behave as its source theory predicts?

The design rules are in
[method benchmarking strategy](../development/method-benchmarking.md). The grid is in
[the technical reference index](method-evidence/validation-grid.md). The test-by-test results
are in [the implementation validation studies](method-evidence/index.md).

Three properties of the harness are worth stating here, because they are what make a green study
mean something.

- **A verdict is bounded by a margin declared before the run.** No rule tests whether a discrepancy
  is exactly zero.
  [The verdict rules](method-evidence/how-to-read.md#the-verdict-rules) give the argument and list
  every rule with its own control.
- **Every positive claim carries a control that must fail.** Double robustness carries a
  both-wrong-nuisance control. A type-I error cell carries a power cell, so an inert test cannot
  pass by never firing. An interval-calibration cell carries deliberately invalid inference.
- **A replication is a fixed sample.** Seeds spawn on the study's own record and on the replication
  index, so replication *k* is the same draw whatever the study's budget. A two-replication probe
  redraws exactly the published first two.

Matching a canonical R implementation is a separate and weaker claim. Two implementations
descended from one source share transcription errors, so agreement localises a discrepancy and does
not certify either one. Every study therefore tests each implementation against known truth first,
and separately.
