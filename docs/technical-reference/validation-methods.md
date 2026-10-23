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

A fit with a declared missing treatment adds the factor row `P(Delta_A=1|W)` and the composite
row `P(A=a,Delta_A=1,Delta=1|W)`, or `P(A=a,Delta_A=1|W)` without `delta=`. The treatment and
response factors are read on the rows whose treatment is recorded. The composite row also reports
two numbers: its minimum, and the share of unit-arm cells below the propensity floor `g_lo`. That
floor is the scale at which R `drtmle`'s `tolg` floors the same product. The fit itself tilts the
composite inside the product of the factor floors. That product is `1e-6` at
`g_bounds=(0.01, 0.99)` and `nuisance_bound=0.01`, as
[the composite contract](dr-tmle/theorem.md#observational-missing-data-the-composite-indicator)
states.

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
`group_score_load` reads that fitted artifact.

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
exact clipping mask was not retained, the count is unavailable and `group_score_load_omissions`
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
per-node leverage through `diagnostics.support()`.

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

A fluctuation built by hand might not retain exact absolute score weights. The report does not
reconstruct an approximation. `group_score_load_omissions` names each omitted generic group and its
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

The table order is the guard order. A caller can be in the first two states at once, because
`check_support`, `check_policy_support`, and `check_incremental_support` all default to no artifact
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

**How.** Point-treatment `result.diagnostics.truncation_curve()` sweeps the `g_bounds` level and
**retargets** cached nuisances through `TMLE.retarget`. Ordinary, collaborative, and unguarded
doubly-robust fits refit no nuisance model, so their curves are retarget operations.

A guarded DR-TMLE fit retargets its cached primary nuisances and refits its reduced regressions at
each bound. Its row reports the `refit` cost and requires the `retarget_cached_nuisances` replay
slot, because the curve does not call `refit()`. The function `truncation_refusal` first runs the
checks of the reduction settings that the retarget reads. It also compares `guard`, `reduction`
and `reduced_crossfit` with the fitted construction. When a fit refuses those settings or they
differ from the fitted construction, the row reads `unavailable`. The facade call and the module
call then each raise `CapabilityError`.

| test | configuration | outcome |
| --- | --- | --- |
| `TestGuardedTruncationUsesCachedNuisanceReplay` in `tests/unit/test_capability_row_predicates.py` | a stratified fold policy | the curve runs, and the three surfaces agree |
| `TestTheGuardedCurveReadsItsReductionSettings` in `tests/unit/test_summary_and_message_accuracy.py` | `targeting="one_step"` with `reduced_crossfit="nested"` | the three surfaces refuse |
| `TestTheGuardedCurveReadsItsReductionSettings` | a changed `guard`, `reduction` or `reduced_crossfit` | the three surfaces refuse |

`mechanism=None` sweeps the observation mechanism on a fit that estimates no treatment law, and
`g(W)` on every other fit. `truncation_refusal` in `sensitivity/positivity.py` then applies three
rules in order. The module call raises its sentence, and the capability row quotes it for the same
request.

| order | rule | refuses |
| --- | --- | --- |
| 1 | `observation_axis` | `mechanism=True` on a fit with no missingness and no intermediate mechanism |
| 2 | `incremental` | the treatment axis of an incremental fit, because $g$ is inside its estimand |
| 3 | `natural_course` | the treatment axis of a fit that estimates no treatment mechanism |

An incremental fit with missing outcomes reads `deferred` on `mechanism`, because
`mechanism=True` runs. `TestTheTruncationRowResolvesEachRequest` in
`tests/unit/test_capability_row_predicates.py` checks each request on five kinds of fit.

The returned frame records both ends of every evaluated pair. Scalar treatment-mechanism values
remain symmetric shorthand for `[bound, 1 - bound]`; observation- and intermediate-mechanism
values use `[bound, 1]`. It also records each parameter's own fitted pair and estimate, plus the
signed difference from that estimate. The default grid includes every exact fitted pair, without
rounding, so every parameter has one fitted marker. An explicit grid is not expanded: when it omits
the fitted configuration, the fitted columns retain that configuration and every marker is false.

Longitudinal TMLE uses a different normalization and execution path. Its grid is always explicit.
A scalar `b` means the cumulative lower-only pair `(b, 1)`. An explicit `(lower, upper)` pair keeps
both limits. Point-treatment scalar treatment bounds instead mean `(b, 1 - b)`.

For each longitudinal pair, the diagnostic keeps the realized data, resolved plans, folds,
unbounded treatment and censoring predictions, weights, clusters, and estimand structure fixed. It
recomputes each bounded cumulative mechanism prefix. It then reruns the complete backward TMLE
recursion. It refits every bound-dependent outcome or pseudo-outcome regression and every targeting
update. It reuses no earlier-node outcome fit or prediction.

The diagnostic first replays the fitted pair. Every retained estimate, regimen fit, and MSM fit
must match exactly. This preflight runs even when the requested grid omits the fitted pair. A
mismatch makes the operation unavailable.

Each row is a descriptive ordinary LTMLE point estimate under one fixed bound. It is not a
cached-nuisance retarget and does not change the causal estimand. The curve reports no standard
error, confidence interval, preferred bound, selected-bound correction, or pass threshold.

The longitudinal frame carries one row per evaluated pair and reported parameter. The curve
resolves every grid entry first and drops a repeat after that, so a repeated pair is replayed once
and reported once. A scalar `b` and the pair `(b, 1)` resolve to the same entry, so a grid carrying
both collapses to one row.

| column | what it holds |
| --- | --- |
| `lower_bound`, `upper_bound` | the cumulative pair this row was evaluated at |
| `estimand` | the reported parameter this row describes |
| `psi` | the replayed point estimate under that pair |
| `fitted_lower_bound`, `fitted_upper_bound` | the pair the fit itself used |
| `fitted_psi` | the retained estimate under that fitted pair |
| `is_fitted_bound` | whether this row's pair is the fitted pair |
| `delta_from_fitted` | `psi` minus `fitted_psi` |
| `evaluated_score_cells` | the mechanism cells the contributing fits consumed |
| `truncated_score_cells` | how many of those cells the bound moved |

The longitudinal frame names its lower endpoint `lower_bound`. The point-treatment frame above
names the same endpoint `bound`. Read the endpoint under the name its own frame uses, because
indexing a longitudinal frame at `bound` raises `KeyError`.

Each count reads only the fits that contribute to that row's parameter.

| the parameter | the fits it counts |
| --- | --- |
| a level | its own regimen fit |
| a contrast | its two regimen fits |
| an MSM coefficient | every regimen and horizon cell for its cause |

`evaluated_score_cells` counts one cell for each row in each contributing node's reporting score
mask. That mask is the node's `trained_on` set, which is the rows whose clever covariate is nonzero.
Of those cells, `truncated_score_cells` counts the ones whose bounded cumulative product differs
from the raw product.

Neither count is a count of regression rows. On a single-fold fit, the node regression is fitted on
`fitted_on`, which is that score mask. On a cross-fitted fit, each fold's untargeted regression is
fitted on the score mask inside its training rows, and the pooled fluctuation solves over the whole
score mask.

A cross-fitted replay counts one out-of-fold mechanism pair, so every scored row counts once. The
pooled fluctuation at each node divides by that pair and by nothing else. The untargeted fold
regressions read no mechanism. The count therefore has one denominator for a single-fold fit and a
cross-fitted one.

`tests/unit/test_longitudinal_truncation_refit.py` pins the counts on a two-fold fixture, measured
under the package's drawn unstratified folds. At the bound 0.12 it reports 3 truncated cells of 112
for `ey_regimen[always]`, and 22 of 112 at the bound 0.3. The same module pins the movement of that
estimate at the bound 0.3. The replayed value is 0.74038, against a fitted value of 0.56304. A
longhand recomputation with its own fold recursions and Newton solves reproduces both values.

An MSM coefficient reads every regimen and horizon cell for its cause, and each cell is its own
backward pass. Its count adds one pass per cell to the same total. On a survival MSM that total can
exceed the sample size times the node count, which bounds one complete pass. A longitudinal MSM runs
at one fold, so no fold factor enters that comparison.

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

Guarded DR-TMLE and longitudinal TMLE are refit exceptions. Guarded DR-TMLE refits its reduced
regressions because the swept bounds define two regression targets. Its primary outcome regression
and propensity stay cached. Longitudinal TMLE refits the complete recursion because each later
targeted prediction becomes the preceding regression's response. Their capability rows declare
`refit` and ask for `include_refits=True`. The longitudinal row also requires explicit `bounds`.

Longitudinal replay needs a stored recipe, a cloneable outcome learner, and a cloneable
pseudo-outcome learner. The recipe stores no fitted outcome model. Cross-fitted replay rebuilds
the out-of-fold mechanism pair at each bound, because the pooled fluctuation divides by that pair.
[Replay-only unavailability](scope-and-refusals.md#replay-only-unavailability) states the
`random_state` rule each learner must satisfy, and lists every omission code that makes the
diagnostic unavailable.

### Nuisance model quality

**Why.** A nuisance model can order the rows well and still predict probabilities that are too
extreme or too moderate. The clever covariate divides by the predicted probability itself, so the
spread of the predictions matters as well as their order.

**What it tells you.** The report gives prediction loss and calibration for each retained nuisance
fit. It also gives the candidate weights and risks when a Super Learner supplies them. For C-TMLE,
it retains the selector or outcome-adaptive fit. For repeated cross-fitting, it reports parameter
movement across split draws.

**How.** `result.diagnostics.nuisance_models()` returns a report for the fitted result family.
Point-treatment results return `NuisanceDiagnostics` from
[`validation/nuisance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/nuisance.py).
That report contains propensity AUC, the calibration slope and its standard error, outcome fit
metrics, and Super Learner candidate details. The metrics are out-of-fold when the fit cross-fitted
its nuisances. They are in-sample when it used one fold, and `evaluation` records which. The
[calibration-slope rule](#calibration-slope-rule) below reads the slope.

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

`loss_name` is `log_loss`, `brier`, or `mse`. `loss` holds that role-specific value. The nested
`model` is a `NuisanceModelReport`. It retains calibration and learner-library details.

The `mse` field answers about node regressions alone. An outcome row or a pseudo-outcome row
reports its square loss there. A treatment row or a censoring row reports `nan`, because a
mechanism fit has no square loss. On a weighted fit, the `mse` value averages under the
observation weights.

The frame admits an empty value in `regimen`, `cause`, and `horizon`, because a mechanism row
carries no regimen identity. That row stores `None`, which a dataframe renders as `None` or as
`NaN` by column type.

For a categorical treatment, `log_loss` is the observed-class multinomial negative log likelihood.
The nested report uses `kind="multinomial probability"` and retains no binary calibration table.
Armwise calibration requires a separate report design because no treatment arm is privileged.

`LongitudinalNuisanceDiagnostics.omissions` holds typed `LongitudinalNuisanceOmission` records. Each
record names `role`, `time`, and `reason`. An outcome or pseudo-outcome record also names
`regimen`, `cause`, and `horizon`, and a fold-level record names its `folds`. The reasons are in
the table.

| reason | role and time | why the row is absent |
| --- | --- | --- |
| `LONGITUDINAL_CENSORING_NOT_FITTED` | `censoring`, no time | the design has no censoring columns, so the fit made no censoring fit |
| `LONGITUDINAL_NO_CENSORING` | `censoring` at one node | no eligible unit was censored at the node, so its factor is exactly one and no learner ran |
| `LONGITUDINAL_NO_CENSORING_IN_FOLD` | `censoring` at one node | one training fold had no censored unit and predicts one. Its held-out fold holds every censored unit at the node. The node's model row stays |
| `LONGITUDINAL_HELD_DECISION` | `treatment` at a later node | the node holds the baseline decision, so its factor is exactly one and no model ran |
| `LONGITUDINAL_CLASSIFIER_RATIO` | `treatment` at a continuous node | the classifier ratio route fits no density |
| `LONGITUDINAL_KNOWN_MECHANISM` | `treatment` or `censoring` at one node | the data declares the node's factor known, so the fit read the declaration and no learner ran |
| `LONGITUDINAL_CONSTANT_TARGET` | `outcome` or `pseudo_outcome` at one node | every follower held 0, such as no event at the node, or every follower held 1. The regression is that value and no learner ran |
| `LONGITUDINAL_CONSTANT_TARGET_IN_FOLD` | `outcome` or `pseudo_outcome` at one node | one or more training folds held 0, or held 1, while the sample did not. Those folds fitted no learner, and `folds` names them. The node's model row stays |

`to_frame()` starts with the row identity, evaluation, loss, model name, and model kind. It then
adds the union of metrics that the nested model reports. A binary row reports
`calibration_slope` and `calibration_slope_se`, from the fit's folds and cluster codes. An outcome
or pseudo-outcome row with a continuous target reports its linear slope as `regression_slope`. The
longitudinal report applies no calibration rule.

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
| a calibration-slope finding says the predictions are more extreme or more moderate than the observed rates | dropped | the same reason. A selected working mechanism has no calibration target. An intercept-only candidate has no slope at all |
| an AUC above 0.9 signals a positivity problem | kept | `CTMLE._nuisances` puts the selected mechanism on `nuisance.propensity`, so it is the denominator the clever covariate divides by. An AUC near one there means the fitted weights are near-degenerate |
| a super learner weight above 0.8 on the marginal mean means the model contributes little | kept | the note states a fact about a learner library, not an interpretation of the treatment law |

`tests/e2e/test_ctmle.py` holds both halves of this rule. Its
`test_the_role_suppresses_two_claims_and_no_others` mutates the retained report and asserts that
each dropped claim returns under `estimated_treatment_law`. Its
`test_a_mean_only_learner_library_is_reported_for_a_working_model_too` raises the mean weight on a
working model and requires the note. Use `support()` to inspect the denominator the selected
mechanism creates.

The selector artifact retains every candidate, risk, fitted fold, and the selected index. The
outcome-adaptive artifact retains its design, its outcome-prediction features and its treatment
risk, and under the per-arm design the risk of each arm. Each artifact
renders one sentence through `describe()`. The nuisance summary and the combined assessment row
both print that one sentence.

| artifact | what `describe()` renders |
| --- | --- |
| `CTMLESelection` | `C-TMLE greedy selected candidate 2 of 4 for ate` |
| `CTMLEOutcomeAdaptiveFit`, `oat_design="per_arm"` | `C-TMLE outcome-adaptive fit used one Qbar feature per arm in 2 binary mechanisms` |
| `CTMLEOutcomeAdaptiveFit`, `oat_design="shared"` | `C-TMLE outcome-adaptive fit used 2 Qbar features in one categorical mechanism` |

On a repeated fit these objects describe draw 1, because the result retains method-specific state
for that draw. The nuisance summary and the combined assessment row render the scope through
`cleverly.utils.text.format_draw`, as `draw 1 of 3`. The score-load rows above print the
zero-padded `draw 01 of 03` instead. Those rows pad every style, so a table cell and the
sentence that describes it agree on one width.

A repeated longitudinal fit follows the same rule. `support()` and `nuisance_models()` describe
draw 1 and carry `n_repeats` and `reported_repeat`. `score_equations()` reports every draw and adds
a one-based `draw` column. `tests/unit/test_longitudinal_repeats.py` checks both reports.

`repeat_spread` contains one `RepeatSpreadRow` per reported parameter when the fit uses two or more
draws. The row pairs the split standard deviation across every retained draw with the
median-combined result's standard error. `LongitudinalNuisanceDiagnostics.repeat_spread` holds the
same rows for a longitudinal fit, and `LongitudinalResult.repeat_spread()` returns the standard
deviations.

Both quantities are on the inference scale of the estimand. That scale is the log scale for a ratio,
and the outcome scale otherwise, so `ratio_to_standard_error` divides two like quantities.
`test_a_ratio_takes_its_spread_on_the_scale_its_standard_error_lives_on` in
`tests/unit/test_repeated_crossfit.py` reads `ate`, `rr`, and `or` off one fit. The ratio is
descriptive and has no pass threshold.

A selector-path collaborative fit supplies no standard error, so its row pairs the spread with the
plug-in diagnostic. `repeat_spread_frame()` then names the last two columns
`plugin_standard_error` and `ratio_to_plugin_standard_error`. The printed table heads them
`plugin se` and `sd/plugin se`. The `RepeatSpreadRow` fields keep their names.
`cleverly.inference.influence.spread_name` holds each renamed column.

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

### Calibration-slope rule

**Why.** A learner can predict probabilities that are too extreme or too moderate for the observed
rates. The calibration slope measures that spread. A finding asks the analyst to review the
learner. It is not evidence that the estimate is biased.

**What it tells you.** The table defines the statistic and the rule.
`cleverly.validation.nuisance` implements both.

| term | definition |
| --- | --- |
| calibration slope | the last coefficient of a weighted logistic regression of the label on one indicator for each validation fold and on $\operatorname{logit} \hat p$, with no other intercept. A slope below 1 means the predictions are more extreme than the observed rates. A slope above 1 means they are more moderate. An in-sample fit has one fold, which gives the pooled recalibration of Cox (1958) |
| standard error | the sandwich. With $B = X^\top \operatorname{diag}(w \mu (1 - \mu)) X$ and the score $s_i = w_i (y_i - \mu_i) x_i$, the curve of the slope is $n (B^{-1} s_i)_{\mathrm{slope}}$. `influence_variance` reads the curve with the fit's cluster codes |
| omission | `NuisanceModelReport.calibration_omission` names why a report has no slope, and `summary()` prints it under the table. `to_frame()` has no column for it. `CALIBRATION_CONSTANT_WITHIN_FOLDS` is the state of a mean-only learner or an intercept-only mechanism. `CALIBRATION_SEPARATED` means that in every fold one threshold on $\operatorname{logit} \hat p$ splits the labels, the same way in each fold. The likelihood then rises without bound in the slope. A Newton fit that does not converge, a singular information matrix, or a standard error that is not finite also removes the slope |
| tested models | every probability report with a finite slope and standard error, except a C-TMLE working mechanism, on an out-of-fold fit. An in-sample fit tests none |
| level | with $K$ tested models, the interval is the slope plus or minus $z_{1 - 0.05 / (2K)}$ standard errors. That is a Bonferroni split of `CALIBRATION_FAMILY_ALPHA` over the report |
| finding | the interval lies above 0 and excludes 1 |
| no finding | the interval contains 1, reaches 0, or lies below 0 |
| message | the model, the direction, the slope, the interval, the AUC, and for `propensity`, `propensity[<arm>]` and `missingness` the largest untruncated inverse weight. That weight is the largest $1 / \hat p$ of the observed label over the rows whose weight uses it, before any truncation bound applies |

The fold intercepts matter. Cross-fitting moves the level of each fold's predictions against that
fold's labels. With one pooled intercept that movement drives the slope, and a prediction that is
constant in each fold reads close to $-(V - 1)$. At three folds,
`test_w4_a_fold_constant_prediction_has_no_slope` pins that pooled slope within 0.05 of $-2$.

Four facts shape the rule. Each has its source or its witness.

| fact | source or witness |
| --- | --- |
| the slope measures spread, not specification. The limit $\eta^*$ of a logistic model with an intercept and main effects solves $E[(A - \operatorname{expit} \eta^*)(1, \eta^*)] = 0$, so its population slope is 1 whether the model is correct or not | the score equations of the model. [Riley et al. (2021)](../references.md#calibration-of-prediction-models), Section 2.1.2, state the in-sample case |
| estimation noise can put the out-of-fold slope of a correct model below 1 when the signal is weak or the covariates are many | Riley et al. (2021), Section 2.1.2. The [calibration-slope study](method-evidence/calibration-slope-warning.md) measures the mean slope of a correct model on four covariates |
| an unpenalized logistic model with an intercept has an in-sample slope of exactly 1 | Riley et al. (2021), Section 2.1.2. `test_w10_an_in_sample_fit_carries_no_test` |
| the slope measures the spread of the predictions, and it is read with the calibration intercept, not alone | [Stevens and Poppe (2020)](../references.md#calibration-of-prediction-models), Sections 3 and 4, and [Van Calster et al. (2019)](../references.md#calibration-of-prediction-models), section "How to assess calibration?" |

No read source gives a fixed band or a rule for cross-fitted nuisance predictions. The interval, the
Bonferroni split and the gate at 0 are therefore design decisions of this package. The gate at 0
refuses to read a spread where the slope shows no association between the predictions and the
label.

**Limits.** A prediction at 0 or 1 has a logit near $\pm 27.6$ after the clip, and a few such rows
can move the slope toward 0. When the interval reaches 0, the rule gives no finding. Read the
calibration table and the AUC beside the slope.

| witness | what it checks |
| --- | --- |
| `tests/unit/test_calibration_slope_rule.py` | the slope and its standard error on weighted data with three folds equal an independent fit to 1e-10. The weights, the clusters and the fold intercepts each move the result. The file also checks the truth table of the rule, the Bonferroni divisor, the gate at 0 and the in-sample gate. A correct weak-signal fit at a seed where the old band fired gives no finding, and a doubled or halved logit gives one |
| [calibration-slope warning](method-evidence/calibration-slope-warning.md) | the standard error of the slope for a known propensity, the rule's false-warning rate on five laws, the old band's rate on the same fits, and detection on two tempered learners, over 10,000 samples each |

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
| omitted-variable bounds | no unmeasured confounding | the largest bias an unmeasured confounder of declared strength can produce | the confounder acts through the outcome regression and the Riesz representer. `cf_y` is its share of the residual outcome variation, and `cf_d` is its share of the representer's second moment |
| robustness value | no unmeasured confounding | the single strength at which the conclusion flips | that the two strengths are equal |
| benchmark | no unmeasured confounding | the strength of a confounder "as strong as" a named observed covariate | that dropping the covariate and refitting calibrates the scale |
| simulated common cause | no unmeasured confounding | estimate displacement across a declared strength grid, on the additive scale or the log scale | a supported latent perturbation family and plausible declared strengths |
| E-value | no unmeasured confounding | the equal-strength threshold for treatment and outcome confounder associations that explains away the effect | a risk-ratio scale |
| marginal sensitivity model bounds | no unmeasured confounding | bounds on the arm means and the ATE when `U` moves the odds of treatment by at most $\Gamma$ | Tan's model on the treatment odds |
| missingness tilt | outcomes missing at random | how the estimate moves as the unobserved outcomes are tilted away from the observed ones | the tilt is a constant on the logit scale |
| tipping gamma | outcomes missing at random | the tilt at which the conclusion changes | as above |

### Omitted-variable bounds, robustness value, benchmark, and contours

**How.** [`sensitivity/omitted_variable.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/omitted_variable.py)
implements the bounds of Equation (14) in Section 4 of Chernozhukov, Cinelli, Newey, Sharma and
Syrgkanis (2026). Their Theorem 2 gives the bias of a linear functional. The bound is

$$
|\text{bias}| \le |\rho| \sqrt{\frac{c_D^2}{1-c_D^2}}\; c_Y \sqrt{\sigma^2 \nu^2},
$$

with $\sigma^2 = E[R (Y - \bar{Q})^2]$ and $\nu^2$ the second moment of the Riesz representer.
$R$ is the indicator of the rows the representer reads. On complete data $R$ is one on every row.
[Second mechanisms and strata](#omitted-variable-bounds-robustness-value-benchmark-and-contours) defines $R$ for the other
fits. The two primitives are exposed as `elements()`.

The estimate of $\nu^2$ reads the fitted treatment mechanism, so it needs a consistent assignment
model. On a wrong model, the doubly robust estimate of $\nu^2$ falls below $\nu_0^2$ by
$\lVert \hat\alpha - \alpha_0 \rVert^2$. The bounds then read too narrow, and the robustness value
reads too large. The plug-in estimate can fall on either side of $\nu_0^2$. Witness 5 in
`tests/unit/test_omitted_variable_refusals.py` gives the plug-in 5.32 against $\nu_0^2 = 4.4$.

A plain `TMLE` fit with `treatment_adjustment=` is admitted. The fitted representer uses the
mechanism $g_n(W_g)$, which is the fit of the restricted model. So the bound is claimed only under
the declared [statistical-model restriction](point-treatment-tmle.md#treatment-mechanism-covariates).
The fit's summary prints that restriction. `tests/unit/test_treatment_covariates_collaborative.py` fits the
bound on such a fit and asserts the restriction in its summary. A collaborative fit stays refused,
including a fit that selects nothing.

`cleverly` refuses every operation in this group on the fits below. The
refusal runs in `sensitivity_elements()` before any computation. Every entry point and the
matching capability row therefore carry one reason.

| refused fit | the result the package does not have |
| --- | --- |
| a `drtmle` fit | an estimate of $\nu^2$ for an estimator that does not assume a consistent treatment mechanism. $E[2 m(\hat\alpha) - \hat\alpha^2]$ equals $\nu_0^2 - \lVert \hat\alpha - \alpha_0 \rVert^2$, so it falls where that mechanism is wrong. `assume_consistent_mechanism=True` gives a point bound under that assumption |
| a `collaborative_tmle` fit | the same estimate. The working mechanism conditions on a function $V$ of the covariates. $V$ is the selected set $W_S$ on the greedy, ordered, and discrete paths, and the fitted outcome regression under `strategy="oat"`. Where the working mechanism is $P(A \mid V)$ in the limit, the representer is $E[\alpha_W \mid A, V]$. Its second moment cannot exceed that of $\alpha_W$, while $\sigma^2$ reads every declared covariate. `CTMLE(strategy="derivative")` is not refused. Its mechanism reads every declared covariate, as a `TMLE` mechanism does. `assume_consistent_mechanism=True` gives a point bound on a selecting fit |
| a longitudinal fit | any registered longitudinal derivation. See [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) |
| an `rr_tilt` parameter axis | an implementation. A risk-ratio tilt mean is a linear functional of the outcome regression and has a Riesz representer, so the bound is well posed here |
| an `msm` axis with a continuous dose or a link other than `"identity"` | a representer. A continuous dose reads the treatment density at quadrature doses, and under another link the coefficient is not linear in the outcome regression |
| a `policy` axis with a randomized, fold-tuned, or joint policy set | a combination of the component representers |
| a `policy` axis on a continuous dose, where an observed or a shifted dose is not a declared dose atom | a derived representer for a fitted dose density, and a guarantee for $\hat\nu^2$. On a binned density, the doubly robust $\hat\nu^2$ is often negative |
| a `learned_rule` parameter axis | a bound for the frozen fold rules. A benchmark refit would relearn the rules and change the target. See [X11](../roadmap.md#x11-learned-policy-follow-ups) part (n), which waits on X82 part G |
| an `ipsi` parameter axis | a bound that covers it. An incremental intervention tilts the treatment mechanism, so the mechanism is part of the estimand rather than a nuisance |
| an arm-indexed fit that reports no counterfactual mean and no linear contrast, such as a ratio-only, PAR, PAF, or `ey_obs` fit | a parameter to bound. One bound is the second moment of one contrast's own Riesz representer. On a fit that reports `rr` or `or`, the sentence names `sensitivity.evalue()` |

#### Second mechanisms and strata of the bound

The bound also covers the fits in the table below. Each representer is the fit's own clever
covariate times the indicator $R$ of the rows that it reads. The submodel applies $\Delta$ through
the mask of the fluctuation, so the bound multiplies it into the representer. $\sigma^2$ is the
mean over every row of the residual that $R$ masks.

| fit | representer $\alpha$ | $m(O, \alpha)$ | $R$ | feature |
| --- | --- | --- | --- | --- |
| a response mechanism | $\Delta 1\{A = a\} / (g_a \pi_a)$ | $1 / (g_a \pi_a)$ at each arm | $\Delta$ | `response` |
| a declared missing treatment | $C_a / g_{c,a}$, with $C_a = \Delta_A \Delta 1\{A = a\}$ | $1 / g_{c,a}$ at each arm | $\Delta_A \Delta$ | `missing_treatment` |
| an intermediate variable at the level $z$ | $\Delta 1\{A = a\} 1\{Z = z\} / (g_a q_z \pi_a)$ | $1 / (g_a q_z \pi_a)$ at each arm | $\Delta 1\{Z = z\}$ | `intermediate` |
| a stratum-conditional parameter | the marginal $\alpha$ times $1\{S = s\} / p$ | the marginal $m$ times $1\{S = s\} / p^2$ | the marginal $R$ times $1\{S = s\}$ | `stratum` |

For a mean or a difference of means in a stratum, $p = P_w(S = s)$. For an ATT or an ATC in a
stratum, $p = P_w(A = c, S = s)$, and it replaces the marginal share $P_w(A = c)$. The composite
mechanism $g_{c,a}$ is the one that the
[composite construction](point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) targets.

Each functional is linear in the regression on the rows that $R$ marks, so Theorem 2 of the cited
paper covers it. Only $R$ and the representer differ from the shipped bound. On complete data
$R = 1$, and each quantity equals the shipped one. The bias reads $Y$ only where $R = 1$, so $c_Y$
is the share of the residual variation of $Y$ on those rows.

On the first three rows, $c_D$ measures a joint strength over the treatment mechanism and a second
mechanism. The summaries of the bound and of the `benchmark()` record name that second mechanism. `benchmark()` refits every mechanism without the named covariates, so it measures the
same joint strength. The curve of $\nu^2$ in a stratum carries the influence of the share $p$, as
the [standard error](#standard-error-of-the-omitted-variable-bound) states. Under
`nu2_estimator="plugin"` these fits report the point bounds. Their limits refuse with the key
`configuration`, because the mechanism stack carries the score of the treatment mechanism alone.

| witness in `tests/unit/test_omitted_variable_part_g.py` | what it checks |
| --- | --- |
| `TestTheRieszIdentityOnAnExactLaw` | $\hat\nu^2 = \nu_0^2 - E[(\hat\alpha - \alpha_0)^2]$ to a relative tolerance of $10^{-12}$ on each configuration. The mechanisms are wrong, so the error term is not zero |
| `TestTheOracleRepresenter` | at the law's own mechanisms, the representer equals $\alpha_0$ on every row. One mutation drops each weight: $\Delta$, $1\{Z = z\}$, $\Delta_A$, and $1\{S = s\}$ |
| `TestTheResidualVariance` | $\sigma^2$ masks $R$ and averages every row. A mutation removes $1\{Z = z\}$ from $R$ |
| `TestTheStratumShareCurve` | the curve of $\hat\nu^2$ equals a finite-difference Gateaux derivative on every row. A mutation removes the share term |
| `TestOneStratumIsTheMarginalBound` | a stratum that holds every row gives the marginal bound |

#### Regime, shift, and MSM parameters

The bound covers the regime, shift, and MSM parameters in the table below. Each parameter is a
linear functional of the outcome regression. The representer is the fit's own clever covariate
times $R$, as in the table above. In each row, $m(O, \alpha)$ reads the counterfactual blocks of
the clever covariate.

| axis | parameter | representer $\alpha$ | $m(O, \alpha)$ | feature |
| --- | --- | --- | --- | --- |
| `regime` | `ey_regime`, `ate_regime` | $g^\star_r(A \mid W) / g(A \mid W)$, over $\pi$ and $q_z$ where the fit has them | $\sum_a g^\star_r(a \mid W) \alpha(a, W)$ | `regime` |
| `policy` | `ey_policy`, `ate_policy` | the shift's ratio of dose probabilities $h_s(A, W)$, on declared dose atoms | $\alpha(d_s(A, W), W)$, at the dose the policy assigns | `shift` |
| `msm` | `msm[term]` | $e_j^\top M^{-1} h(A, V) \varphi(A, V) / g(A \mid W)$ | $\sum_a e_j^\top M^{-1} h(a, V) \varphi(a, V) \alpha(a, W)$ | `msm` |

$M = P_n \sum_a h \varphi \varphi^\top$ is the observation-weighted Gram matrix that the
coefficient inverts. A contrast takes the difference of two representers and of two $m$ terms.

The MSM estimate of $\nu^2$ reads $\hat M$. Its curve therefore adds the influence of $\hat M$,
$-2 w [\sum_a h_a (z^\top \varphi_a)(\varphi_a^\top b) - \nu^2]$, with $b = M^{-1} e_j$ and
$z = M^{-1} K b$. Here $K$ is the matrix with $\nu^2 = b^\top K b$. The term is zero on every row
of a marginal MSM, because each row's Gram contribution is $M$. The package derives this term,
and no cited source gives it.

A shift representer is a ratio of dose probabilities. The bound covers a shift only when
`dose_atoms` holds every observed dose and every shifted dose. A shift on a continuous dose refuses
with `implementation_gap` under both estimators of $\nu^2$.
`test_a_continuous_dose_shift_is_refused` in `tests/unit/test_omitted_variable_refusals.py` checks
the refusal.

`benchmark()` refits without the named covariates. The refit evaluates the declared regime,
policy, or working model again. When a dropped covariate changes or breaks that declaration, the
call refuses, because the short model would estimate another target.

`tests/unit/test_omitted_variable_axes.py` checks these claims:

- The Riesz identity holds on exact laws for each axis to `rtol=1e-12`, with wrong mechanisms as
  the nonzero witness. A stochastic regime's density has a mutation control.
- A static regime and a saturated MSM give the arm bound's $\nu^2$.
- The MSM curve equals the Gateaux derivative on a design that reads $V$. A mutation that drops
  the $\hat M$ term departs from it.

#### The bound after repeated cross-fitting

A fit with `repeats` above 1 has one bound for each draw of the cross-fitting split. Draw $r$ has
its own estimate $\psi_r$, bias scale $B_r$, and influence curve. The reported quantities combine
the draws as the table states. One draw reproduces the shipped bound exactly.

| quantity | the combination over draws |
| --- | --- |
| each bound | the median of $\psi_r \mp S B_r$, with $S$ the confounding strength |
| `max_bias` | the median of $B_r$ |
| each one-sided limit | the median edge $\tilde t$ of the draws, extended by the quantile times $\sqrt{\operatorname{median}_r\{\hat\sigma_r^2 + (t_r - \tilde t)^2\}}$ |
| `rv` | the strength at which the median bound reaches the null |
| the confidence-limit robustness value | the strength at which the combined limit reaches the null |
| each `contour` cell | the median bound at that cell |
| `benchmark()` | the medians of $\sigma^2$ and $\nu^2$ over draws, before and after the refit |
| `sensitivity_elements()` | one draw, named by `repeat=`. A bare request refuses, and its capability row requires `repeat` |

On several draws, `max_bias` is the median of $B_r$, and a bound is not $\psi \mp S$ times that
median. Read each bound from the result, and do not recompute it from `max_bias`.

The standard error rule is the one that
[`repeats`](cv-tmle.md#variations) applies to the estimate. Chernozhukov et al.
(2018, equation 3.14) give it. The rule assumes that $\hat\sigma^2$ and $\hat\nu^2$ are
asymptotically linear on each draw, with limits common to the draws.
[X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) part (b) gives that
argument for the estimate.

Each draw keeps its point estimate and not its curve, so the bound
retargets each draw from its own nuisances. The retarget must reproduce the stored estimate of the
draw, or the call raises `RuntimeError`. The feature `repeats` marks these limits descriptive.
`tests/unit/test_omitted_variable_repeats.py` checks each row against the fit with one draw at the
seed of that draw.

#### The point bound under an assumed consistent mechanism

`assume_consistent_mechanism=True` lifts the refusal of a `drtmle` fit and of a selecting
`collaborative_tmle` fit. `omitted_variable_bounds()`, `robustness_value()`, and `contour_data()`
accept it. The keyword states an assumption that the estimator itself does not make. The bound then
reads the treatment mechanism as consistent, which is the assumption of the published $\nu^2$.

| quantity | `drtmle` fit | selecting `collaborative_tmle` fit |
| --- | --- | --- |
| centre | the fit's estimate | the fit's estimate |
| $\sigma^2$ | the fit's targeted outcome regression | the fit's targeted outcome regression |
| the mechanism of $\nu^2$ | the primary mechanism, which the ordinary TMLE also fits | one refit of the fit's treatment learner on every mechanism covariate, on the fit's folds. A `strategy="lasso"` fit refits the unpenalized member of its path, `L1Logistic(0.0)` |
| limits and the confidence-limit robustness value | refused. The remainder is linear in the error of the mechanism that the estimator does not assume consistent | refused. The selected mechanism has no derived curve for $\nu^2$ |
| a declared missing treatment | admitted | refused, because no refit of the composite mechanism is implemented |
| a per-arm outcome-adaptive fit (`strategy="oat"`, `oat_design="per_arm"`) | not applicable | refused with the reason `implementation_gap`. The fit declares only a learner that reads one outcome prediction, so no learner exists for the mechanism on every covariate ([X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator)). Test: `test_a_per_arm_fit_refuses_the_refit` |

The limits refuse with the key `consistent_mechanism` and the reason `inference_gap`. The bound
keeps the point bounds and the point robustness value. `SensitivityBounds.assumption` holds the
assumption, the summary prints it, and `to_dict()` adds it. On another fit the keyword raises
`ValueError`. A collaborative fit that selects nothing keeps its redirection to `TMLE`. Every other rule of the refusal table still applies. A fit with `repeats` above 1
takes the median of the draws' point bounds.

`tests/unit/test_omitted_variable_consistent_mechanism.py` checks that the DR-TMLE $\nu^2$ equals the
ordinary TMLE $\nu^2$ with one learner and seed. It checks the same equality for the C-TMLE refit.
A mutation control that reads the selected mechanism instead of the refit gives another $\nu^2$.

#### Omitted-variable limits and their coverage evidence

Each bound asks the evidence registry about its one-sided limits, with the instrument `ovb_limit`.
The rule is the features of the fit, joined with `+`, for example `response+stratum`.
[Omitted-variable bound limit coverage](method-evidence/ovb-limit-coverage.md) registers one L2 row
for each single-feature rule in the table below. Each row covers only the configuration that the
study measured.

| rule | covered configuration |
| --- | --- |
| `response` | in-sample `TMLE`, `ate` or `ey1`, binary outcome missing at random, at least 250 events |
| `intermediate` | in-sample `TMLE`, the CDE at a level of a binary intermediate, at least 250 events |
| `regime` | in-sample `TMLE`, a regime mean or contrast, at least 325 events |
| `msm` | in-sample `TMLE`, an MSM coefficient, at least 325 events |
| `shift` | in-sample `TMLE`, a shift mean or contrast on a continuous dose |

Every row also requires a binary outcome, $n \geq 2{,}000$, influence-curve inference, and the
saturated logistic nuisances of the study. Each event floor lies between the minimum and the 5th
percentile of the smallest arm event count over the cell's 1,600 primary draws.
`tests/unit/test_sensitivity_limit_coverage_registry.py` recomputes that range from the declared
seeds. A fit outside a row prints its limits as descriptive.
A joined rule, such as `response+stratum`, has no row. `SensitivityBounds.limit_reading` holds the
resolution, and `SensitivityBounds.limit_descriptive` is true for a descriptive limit. `to_dict()`
adds `features`, `limit_rung`, `limit_licensed`, and `limit_sentence`. `robustness_value()` adds
`limit_licensed` and `limit_sentence`. The shipped bound asks the registry nothing, and its output
does not change.

`nu2_estimator=` accepts `"auto"`, `"doubly_robust"`, and `"plugin"`. `"auto"` resolves to
`"doubly_robust"`. A value outside those three raises `ValueError` before the table above refuses a
fit. `benchmark()` checks two things before it, as the order below states. A
`"doubly_robust"` estimate that is not positive raises `CapabilityError` and names the estimator.
The package returns no plug-in value in its place, because the plug-in squares the same fitted
representer. The plug-in estimator reports no confidence limits. The
[standard error of the omitted-variable bound](#standard-error-of-the-omitted-variable-bound) gives the reason.

For the ATT and the ATC, the doubly robust score weights the contrast by $1\{A = c\} / P(A = c)$.
Here $c$ is the arm the estimand conditions on. This weight is the observed arm indicator of the
ATT and the ATU in Example 2 of Online Appendix A of the cited paper, not the fitted $\hat g_c$.
Item (2) of Theorem 5 there writes $m$ and the representer. Only the observed
indicator keeps the Riesz identity above. `TestTheConditionalEffectScoreReadsTheObservedArm` in
`tests/unit/test_omitted_variable_refusals.py` checks both conditional effects on an exact law.

`robustness_value()` inverts the bound for the single strength that flips the conclusion.
`benchmark()` drops each named observed covariate, refits, and calibrates the strength scale
against what that covariate was worth. `contour()` returns the grid a contour plot needs.

`benchmark()` reports two gain statistics. The table defines them and the share that each
stands for. Here "short" is the refit without the named covariates.

| strength | `BenchmarkResult` field | bound parameter it stands for |
| --- | --- | --- |
| `cf_y` | `gain_y` $= \sigma^2_{\text{short}} / \sigma^2 - 1$, clipped to $[0, 1]$ | $\eta^2_{Y \sim U \mid D, X}$, in $[0, 1]$ |
| `cf_d` | `gain_d` $= \nu^2 / \nu^2_{\text{short}} - 1$, clipped to $[0, 1]$ | $1 - R^2_{\alpha \sim \alpha_s}$, in $[0, 1)$ |

Pass the gains to the bound unchanged. Remark 8 of Appendix E.5 of the cited paper states that the
gains serve as proxies for the two shares when $k_Y = k_D = 1$. DoubleML uses the same
convention. Do not convert a gain to the form $G / (1 + G)$. That form is a narrower calibration
that the cited paper does not give.

`omitted_variable_bounds(result, benchmark=record)` reads `cf_y` and `cf_d` from the record. It
refuses a record of another estimand or another fit, and it refuses explicit `cf_y=` or `cf_d=`
beside the record. The benchmark does not set `rho`. Pass `rho=record.rho` to use its alignment.

| benchmark | what the bound does |
| --- | --- |
| `gain_y` above 1 | uses `cf_y = 1`. The bound is finite at $\eta^2 = 1$, which is the largest share. The summary of the record prints the unclipped gain |
| `gain_d` of 1 or more | raises `ValueError`. The bound is unbounded at `cf_d = 1`, so this record calibrates no finite bound |

`tests/unit/test_omitted_variable_benchmark_handoff.py` checks the hand-off against explicit
strengths and against the share form.

`benchmark()` is the only member of this group that refits. It reads the refusals above before it
refits, so a refused fit pays for no second fit.

`benchmark()` checks a request in this order, before the refit. The function and
`result.sensitivity.benchmark()` both call `_benchmark_names` first, so a malformed name comes
before every refusal.

| order | request | what happens |
| --- | --- | --- |
| 1 | an empty `covariates`, an indicator column of an encoded covariate, or a name that the fit does not adjust for | `DataError`. The message for an unknown name lists the covariates of the fit, with an encoded covariate once by its column name |
| 2 | a result that holds no estimator | `CapabilityError` |
| 3 | a `nu2_estimator` outside the three accepted values | `ValueError` |
| 4 | a fit that the table above refuses | `CapabilityError` with that table's sentence |
| 5 | an estimand that the bound does not cover, or that the fit did not report | `CapabilityError` from `resolve_parameter` |
| 6 | every covariate of the fit | `CapabilityError` from `benchmark_refusal`. The short model would adjust for nothing, and no estimator here fits without a covariate |

Name an encoded categorical covariate by its column. The refit then drops the whole block of
indicator columns, because one indicator alone does not define a covariate. The class
`TestBenchmarkNamesAreLogicalCovariates` in `tests/unit/test_summary_and_message_accuracy.py`
checks the dropped block. A boolean covariate is one column under its own name, and a benchmark
names it directly. The facade and a combined report read a one-shot iterator argument once, into
a tuple, and the assessment cache keys the answer by that tuple. A generator of names therefore
gives the same result as a list.

The `benchmark` capability row reads the same predicate for each request. On a fit with one
covariate, the bare row reads `unavailable`, because no value runs. On a wider fit, the bare row
keeps its `covariates` argument. `TestTheBenchmarkRowResolvesEachRequest` in
`tests/unit/test_capability_row_predicates.py` checks both fits.

### Standard error of the omitted-variable bound

Each bound has a one-sided confidence limit on each end. Theorem 4 in Section 4 of Chernozhukov,
Cinelli, Newey, Sharma and Syrgkanis (2026) gives them. The curve of a limit is the curve of the
estimate plus or minus the strength times the curve of the bias scale. The standard error is the
square root of `influence_variance` of that curve, which sums the rows of each cluster on a
clustered fit. The table gives each curve per row, with weights $w$ of mean one.

Theorem 4 inherits Lemma 3's DML conditions. They include cross-fitting, the cited DML
assumptions, and $o_P(n^{-1/4})$ $L^2$ rates for the outcome regression and representer. The
bias scale $S = \sqrt{\sigma^2\nu^2}$ must be positive. $S$ is a scale and not the largest bias,
because the strength that multiplies it can exceed 1. An in-sample TMLE also needs
conditions that control the empirical-process remainder. The registered study checks correct GLMs
on its declared law; it does not establish coverage for arbitrary learners.

The theorem's DML result uses independent rows. The cluster sum and fixed-weight calculations use
the same row curve, but the registered study does not measure their coverage.

When $\sigma^2=0$, the point bias bound is zero. The ordinary curve in the table divides by
$S=0$, so the package keeps the point bounds and refuses their limits. A robustness threshold is
`None` if no equal strength below one reaches the null. If the baseline confidence limit already
includes the null, its robustness threshold is zero.

| quantity | curve per row | source |
| --- | --- | --- |
| estimate | the estimate's own influence curve. For the ATT and the ATC it includes the share term of $\theta_s$ | Online Appendix A, "Statistical Inference", Equation (15) |
| $\sigma^2$ | $w (R (Y - \bar Q^*)^2 - \sigma^2)$ | Lemma 3 |
| $\nu^2$, doubly robust | $w (2 m(O, \alpha) - \alpha^2 - \nu^2) - 2 \nu^2 w (I - p) / p$ | Lemma 3 for the first term. The second term is this package's derivation |
| $\nu^2$, plug-in | $w (\alpha^2 - \nu^2) + \psi^\gamma - 2 \nu^2 w (I - p) / p$. The mechanism term $\psi^\gamma$ is zero for a declared known mechanism. The last term applies to the ATT and the ATC only | Saul and Hudgens (2020), Section 2, Equation (1), for $\psi^\gamma$. The share term is this package's derivation, as for the doubly robust curve. [Plug-in limits](#plug-in-limits-and-the-learner-rule) defines $\psi^\gamma$ |
| bias scale $S$ | $(\sigma^2 \psi_{\nu^2} + \nu^2 \psi_{\sigma^2}) / (2 \sqrt{\sigma^2 \nu^2})$ | Theorem 4 |

The second term of the $\nu^2$ curve applies to the ATT, the ATC, and a stratum-conditional
parameter. $I$ is the indicator of the rows the parameter conditions on, and $p = P_w(I = 1)$.

| parameter | $I$ |
| --- | --- |
| an ATT or an ATC | $1\{A = c\}$ |
| a mean or a difference of means in the stratum $s$ | $1\{S = s\}$ |
| an ATT or an ATC in the stratum $s$ | $1\{A = c, S = s\}$ |

Here $c$ is the arm the parameter conditions on: the contrast arm for an ATT and the reference for
an ATC. With more than two arms each contrast has its own $c$.

The representer and $m(O, \alpha)$ both divide by $p$, so the estimate is $\hat F / \hat p^2$. The
Riesz identity makes $\hat F$ insensitive to the fitted mechanism at first order, so its curve is
the Lemma 3 score. The curve of $\hat p$ is $w (I - p)$, and the derivative of $F / p^2$ in $p$ is
$-2 \nu^2 / p$. The delta method gives the second term.

The paper gives no such term for $\nu^2$. Equation (15) of Online Appendix A gives the term for the
estimate $\theta_s$ only. The R package `dml.sensemakr` carries the same $\nu^2$ term since commit
`d5293ecb`, and DoubleML omits it. The term sums to zero over the weighted rows, so a check of a
mean or of a point estimate cannot see it.

| witness | what it checks |
| --- | --- |
| `tests/unit/test_omitted_variable_standard_error.py` | on the two exact laws, weighted and unweighted, each curve of $\hat\nu^2$ equals the Gateaux derivative of $\nu^2$ row by row. The whole lower bound and its clustered standard error match the exact curve. Seven committed mutations fail it, and an unchanged control (M0) passes |
| [omitted-variable bound standard error](method-evidence/omitted-variable-bound-standard-error.md) | the ratio of the reported standard error to the sampling spread of each bound, over 10,000 samples, and two controls that read the curve without the share term |

Omitting the term does not always widen the limits. The table gives the lower-bound standard
error on the two-arm exact law, with the curve before RM22 and with the exact curve.

| parameter and sample | without the term | exact | effect of the omission |
| --- | ---: | ---: | --- |
| ATT, iid | 0.03882 | 0.03544 | wider |
| ATT, 50 clusters of 20 rows | 0.02129 | 0.02303 | narrower |
| ATC, 50 clusters of 20 rows | 0.02572 | 0.02413 | wider |

The direction depends on the parameter and on the sampling design. This is one exact law, so the
table shows that either direction occurs and not how often.

`nu2_estimator=` accepts `"auto"`, `"doubly_robust"` and `"plugin"`. The value `"auto"` resolves
to `"doubly_robust"` for every functional the bound covers. Any other value raises `ValueError`
before a refusal or a computation. The doubly robust estimator gives the limits for any mechanism
learner. The plug-in estimator gives them only for the fits that the
[plug-in learner rule](#plug-in-limits-and-the-learner-rule) admits.

The table gives each source this section and the plug-in section cite, and the version read.

| source, and the version read | what it gives |
| --- | --- |
| Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026), arXiv 2112.13398v6 of 21 September 2026 | Theorem 2 in Section 3.2 gives the bias of a linear functional. Equation (14) in Section 4 gives the bounds. Lemma 3 gives the $\nu^2$ score, and Theorem 4 the curve of each bound and its one-sided limits |
| the same paper, the published Online Appendix A, read through the Internet Archive | Example 2 gives the ATT and the ATU weights $\ell(W_s)$. Item (2) of Theorem 5 writes $m$ and the representer. "Statistical Inference", Equation (15), gives the share term of $\theta_s$ |
| `dml.sensemakr`, `R/short-parameters.R`, commit `d5293ecb` | the same $\nu^2$ share term, in code, with no derivation cited |
| DoubleML, `doubleml/irm/irm.py`, commit `b69f86ef` | `_sensitivity_element_est` builds the $\nu^2$ curve without the share term |
| Saul and Hudgens (2020), *Journal of Statistical Software* 92(2), doi:10.18637/jss.v092.i02 | Section 2, Equation (1), states the estimating equation, and Equation (2) the empirical sandwich variance. Section 2 also covers independent rows that are not identically distributed. Section 3.4 takes the independent clusters as the units. Section 3.5, Equation (4), stacks a propensity-model nuisance with the target. The paper refers to Boos and Stefanski (2013), Section 7.2, for the regularity conditions |

The publisher's site refused access to the published main text, so this package did not read it.
The main-text locators follow v6. The published appendix cites "C.7. Proof of Theorem 2" and
"C.10. Proof of Lemma 3 and Theorem 4", which match the numbering of v6.

### Plug-in limits and the learner rule

**Why.** The plug-in $E_n[\hat\alpha^2]$ squares the fitted Riesz representer. It moves at first
order with the fitted treatment mechanism, because
$E[\hat\alpha^2] - \nu_0^2 = 2E[\alpha_0(\hat\alpha - \alpha_0)] + E[(\hat\alpha - \alpha_0)^2]$.
The curve $w(\hat\alpha^2 - \nu^2)$ alone has no term for that fit. The doubly robust
$\nu^2$ is insensitive to the fitted mechanism at first order, so it needs no such term.

**What it tells you.** The limits open when the fit's own treatment mechanism is declared known,
or is a qualifying scikit-learn `LogisticRegression`, fitted in sample or cross-fitted. Every
other fit keeps the point bounds and refuses the limits. A qualifying fit on which the
second-order correction does not hold refuses them too. The table gives both outcomes.

| quantity | qualifying mechanism, correction holds | any other fit, or correction fails |
| --- | --- | --- |
| `lower`, `upper`, `max_bias`, `bias`, `robustness_value` | reported | reported |
| `benchmark()`, `contour()`, and `nu2` and `sigma2` of `elements()` | reported | reported |
| `ci_lower`, `ci_upper`, `robustness_value_ci`, and the three `plugin_interval_*` names | reported | `CapabilityError`, reason `inference_gap`. The inference status is checked first |
| `to_dict()` | carries `nu2_estimator` and the three limit keys | carries `nu2_estimator` and `limit_refusal`, and omits the three limit keys |
| `robustness_value()` | carries `nu2_estimator` and `rva` | carries `nu2_estimator`, `rv`, `max_bias` and `limit_refusal`, and no `rva` under any name |
| `psi_nu2` and `psi_max_bias` of `elements()` | the curves below | `None` |
| `summary()` | names the mechanism whose score the curve carries | prints the clause that refused |

**How.** The curve of the plug-in $\nu^2$ adds the score of the mechanism to the sample mean.
The reported $\nu^2$ subtracts a second-order bias, as the paragraph on the second-order correction states.
The curve comes from a stacked estimating equation. It holds the mechanism score and the
second-moment equation $w\{\alpha(O; \gamma)^2 - \nu^2\} = 0$ (Saul and Hudgens 2020, Section 2,
Equations (1) and (2)). The mechanism score does not involve $\nu^2$ or $p$, so the stacked Jacobian is
block-triangular. The delta method of Theorem 4 then carries the curve into the bound.

$$
\psi_{\nu^2, i} = w_i(\alpha_i^2 - \nu^2) + \psi^{\gamma}_i - 2\nu^2 w_i (1\{A_i = c\} - p)/p,
\qquad
\psi^{\gamma}_i = n \sum_k D_k J_k^{+} u_{k,i}.
$$

The last term of $\psi_{\nu^2, i}$ applies to the ATT and the ATC only, as in the table above. The
table defines the symbols of $\psi^\gamma$ for fold $k$.

| symbol | meaning |
| --- | --- |
| $\theta_k$ | the coefficients of the fold's model in reference form. The lowest arm is the reference, and $\theta_m = \beta_m - \beta_r$ for scikit-learn's symmetric `coef_` and `intercept_` |
| $x_i$ | the design row after the pipeline's scaler steps, with the intercept first when the model fits one |
| $u_{k,i}$ | the estimating function of row $i$: $w_i\, x_i \otimes (e_{A_i} - g_k(x_i))_{-r}\, 1\{i \in \mathrm{train}_k\} - (w_i/n)\, P\theta_k/C$. The second term allocates the fixed L2 penalty under the mean-one weight convention |
| $J_k$ | the information $H_k + P/C$. Here $H_k = \sum_{j \in \mathrm{train}_k} w_j [\mathrm{diag}(g_{-r}) - g_{-r} g_{-r}^\top] \otimes x_j x_j^\top$. With no penalty, $P/C = 0$ |
| $J_k^{+}$ | the pseudo-inverse on the range of the design, so an exactly collinear design passes |
| $D_k$ | $n^{-1} \sum_{j \in \mathrm{test}_k} w_j\, \partial_\theta \alpha_j^2 |_{\theta_k}$, the derivative of the fold's contribution to $\nu^2$ |

An in-sample fit is one fold with train and test equal to all rows. A cross-fitted fit stacks $K$
fold scores for a fixed $K$. Saul and Hudgens
(2020), Section 2, cover independent rows that are not identically distributed, which a fixed
partition produces. A cluster is the unit of the
variance, as in Section 3.4 of the same paper. Section 3.5, Equation (4), stacks a
propensity-model nuisance with the target.

A declared known mechanism (`treatment_probabilities=`) is the degenerate case. The fit reads
$\hat\alpha = \alpha(g_0)$, so $E_n[\alpha(g_0)^2]$ is a sample mean of a known function and
$\psi^\gamma = 0$. The fit refuses `g_bounds` that move a known probability. The bound repeats
the check and refuses with the key `known_truncated` and reason `invalid_declaration`.

**The clip.** The derivative of $\alpha$ in $g$ is zero on every cell that the truncation bounds
moved. With fixed bounds, the curve is the exact curve of the truncated plug-in functional
$\gamma \mapsto E[\alpha_b(g_\gamma)^2]$. The limits cover $E[\alpha_b^2]$. That value equals
$\nu_0^2$ only when the truncation is asymptotically inactive.

With `g_bounds="auto"` the bound of the mean family,
$5/(\sqrt{n_{\mathrm{eff}}} \log n_{\mathrm{eff}})$, moves with $n$. The bound of the ATT and the
ATC is fixed at 0.025. The second-order correction is a smooth expansion, and a clip puts a kink
in $\alpha(g)^2$. The limits therefore refuse on an active clip, with the key `truncation_active`.
The doubly robust path does not refuse so.
[Truncation stability](#truncation-stability) reports the units that the bounds move.

**The learner rule.** The bound reads the fit's own $\hat g$. Only the fit's own fitted models
supply the score. The rule reads the fitted fold models that the fit recorded. A record survives
a pickle round trip and `save`. The table compares the two classes of fit.

| attribute | qualifying mechanism | refused as another learner |
| --- | --- | --- |
| model | a declared known mechanism, or `LogisticRegression` | `SuperLearner`, even with `LogisticRegression` in its library; `LogisticRegressionCV`; any other learner |
| treatment classes | two arms, or more than two | none |
| penalty | none (`C=numpy.inf` or `penalty=None`), or L2 with a fixed C | L1 or elastic net |
| solver and weights | any solver except `liblinear`; `class_weight=None` | `solver="liblinear"`; any `class_weight` |
| pipeline | none, or `StandardScaler` and `"passthrough"` steps before the model | any other step; `screen_treatment=True` |

The bound checks the fit in this order. The fit-wide rules come first, then a declared known
mechanism, then the keys in the table below. The reason of a learner key is `inference_gap`, and
its roadmap item is [F26](../roadmap.md#f26-confidence-limits-of-the-plug-in-omitted-variable-bound).
Each key gives the clause that the refusal quotes after the sentence "This fit does not
qualify". The remedy of a key without its own remedy is `nu2_estimator="doubly_robust"`.

| key | the refusal fires when | remedy |
| --- | --- | --- |
| `unrecorded` | the nuisance holds no per-fold record of the fitted treatment learner. No roadmap item | `nu2_estimator="doubly_robust"` |
| `treatment_design` | the treatment model read a covariate set that differs from the design the stack rebuilds. No roadmap item | `nu2_estimator="doubly_robust"` |
| `screened` | `screen_treatment=True` and the screen kept every covariate. A screen that drops a covariate gives `treatment_design` | `nu2_estimator="doubly_robust"` |
| `cv_penalty` | the final model is `LogisticRegressionCV`, which selects C by cross-validation | pass `LogisticRegression` with a fixed C |
| `super_learner` | the learner is a `SuperLearner` | `nu2_estimator="doubly_robust"` |
| `other_learner` | the final model is any other learner | `nu2_estimator="doubly_robust"` |
| `pipeline` | a step before the final model is not exactly `StandardScaler` or `"passthrough"` | transform the covariates before the fit, or pass the bare model |
| `pipeline_intercept` | a `StandardScaler` centers the covariates and the model sets `fit_intercept=False` | set `fit_intercept=True` |
| `l1_penalty` | the penalty resolves to neither none nor L2. A finite C with `penalty="l1"`, or an elastic net, resolves so | use `l1_ratio=0` for L2, or `C=numpy.inf` |
| `liblinear` | `solver="liblinear"`, which penalizes the intercept | use another solver |
| `class_weight` | `class_weight` is not `None`, which changes the score | declare observation weights with `weights=` |

The fit-level keys read the record of the first fold. A numerical check then runs on each fold in
fold order. The numerical keys carry the reason `inference_gap` and no roadmap item, because F26
does not track a declared model that failed a check. The three correction keys follow them.

| key | the refusal fires when | remedy |
| --- | --- | --- |
| `missing_arm` | the model of a fold has classes that differ from the fit's arms, for example because the training rows of the fold hold no unit of an arm | `nu2_estimator="doubly_robust"` |
| `score_not_zero` | the Newton step from the captured coefficients to the root of the penalized score exceeds `ROOT_TOLERANCE`, 0.05 standard errors of a coefficient. The clause adds a note when the solver stopped at `max_iter` | lower `tol`, raise `max_iter`, or use `solver="newton-cholesky"` |
| `singular` | the information $H_k + P/C$ loses rank against the design, or its condition number on the design's range exceeds `CONDITION_LIMIT`, $10^{10}$ | remove the covariates that separate the arms, or use a fixed C |
| `mismatch` | the softmax of the recomputed linear predictor differs from the stored mechanism by more than `MISMATCH_TOLERANCE`, $10^{-8}$, or the fit holds a different number of records than folds | `nu2_estimator="doubly_robust"` |
| `truncation_active` | a bound clips a cell that the representer reads, or a fitted logit lies within `TRUNCATION_MARGIN`, 3, standard errors of a bound | pass `g_bounds` that the fitted mechanism does not approach, or use `nu2_estimator="doubly_robust"` |
| `correction_unstable` | the predicted remainder $\hat B^2/\hat\nu^2$ exceeds `CORRECTION_STABILITY`, 0.25, of the standard error of $\nu^2$ | collect more rows, or use `nu2_estimator="doubly_robust"` |
| `corrected_nonpositive` | the corrected $\nu^2$ is not positive | `nu2_estimator="doubly_robust"` |

**The second-order correction.** The plug-in $E_n[\hat\alpha^2]$ has a bias of order $1/n$. Its
constant is large, because $\alpha^2$ grows like $g^{-2}$. Out of fold the bias reached 1.44
standard deviations at $n = 1000$
([`tests/diagnostics/x26_crossfit_bias/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x26_crossfit_bias)).
A fixed L2 penalty adds to it.

The root of the stacked estimating equation has the bias
$\beta = Q\{\sum_c \nabla Z_c\, Q Z_c + \tfrac12 \sum_c \nabla^2 Z_c[V] + \mathrm{pen}(\eta)\}$,
with $Q = -(\sum_c \nabla Z_c)^{-1}$ and $V = Q S Q^\top$ (Rilstone, Srivastava and Ullah 1996,
Proposition 3.2; Newey and Smith 2004, Theorem 4.1). The $\nu$ coordinate of $\beta$ is $\hat B$.

The bound reports $\hat\nu^2 - \hat B$. The curve is unchanged, and the variance is the influence
variance of that curve. The package makes no correction of the variance. In a design pilot at
$n = 1000$, a second-order correction of the variance estimate took a large share of the
cross-fitted variance, so its expansion was not trusted.

The correction of the point assumes a smooth mechanism, a small remainder and a positive result.
The three correction keys refuse the limits when one assumption fails. The point bounds remain.

The property study of the corrected limits has run. The reported standard error is calibrated in 46
of 48 positive cells, and the other two read conservative. The
[study page](method-evidence/omitted-variable-bound-plugin-standard-error.md) states the result.

**Conditions.** The limits inherit the conditions of Theorem 4, and add the conditions in the
table. The first two conditions have no check.

| condition | how the package treats it |
| --- | --- |
| a correctly specified mechanism model | no check. Otherwise the plug-in estimates $\nu^2(\gamma^*) \neq \nu_0^2$, and the curve stays the curve of that functional |
| truncation asymptotically inactive | checked at the fit, key `truncation_active`. The check reads the fitted logits, not the limit |
| $E[\alpha^4] < \infty$ | no check. The curve of $\nu^2$ then has a finite variance |
| a score root at the captured coefficients | checked on each fold, key `score_not_zero` |
| a well-conditioned information matrix | checked on each fold, key `singular` |
| the TMLE conditions of the bounded estimate | unchanged |
| a smooth mechanism and a small remainder | checked, keys `truncation_active` and `correction_unstable` |
| a positive corrected $\nu^2$ | checked, key `corrected_nonpositive` |

**The ATE and `ey` identity.** For $\alpha = \sum_a c_a 1\{A = a\}/g_a$, the derivative of the
square is $\partial_\theta \alpha^2 = -2\alpha^2 \partial_\theta \log g_A$. Here
$\partial_\theta \log g_A$ is the score $s$ of the model, so $D = -2E[\alpha^2 s]$. At a correct
model $J = E[s s^\top]$, and $\psi^\gamma = -2\Pi(\alpha^2 \mid \mathrm{span}\, s)$. Here $\Pi$ is
the $L^2$ projection. The map $h \mapsto h - 2\Pi h$ is a reflection, so it keeps the variance of
$h$.

The standard error of a limit of an ATE or of an `ey` then equals the standard error without
the mechanism term. This is projection geometry and not Neyman orthogonality, because
$D \neq 0$.

The identity concerns the curve. The bound reports the influence variance of that curve.

The identity holds under four conditions: a correct model, constant weights, independent and
identically distributed rows, and no active clip. The ATT and the ATC lack the form. The identity
has one consequence for the evidence. A control that drops the mechanism term cannot detect the
omission for an ATE or an `ey`. The exact-law witnesses carry that term.

| witness | what it checks |
| --- | --- |
| `TestTheExactLawCurve` in `tests/unit/test_omitted_variable_plugin_limits.py` | on finite laws, two-arm and three-arm, with and without a ridge and a clipped cell, each row of $\psi_{\nu^2}$ equals the complex-step Gateaux derivative of the plug-in functional, for `ey`, `ate`, `att` and `atc`. A saturated model gives the doubly robust curve |
| `TestTheKnownMechanism` and `TestTheClusteredLimit` | the known mechanism term is zero, and a clustered standard error sums the exact curve |
| `TestTheInfinitesimalJackknife` | on a simulated fit with fixed folds and weights, the curve of each row equals $n\, d\hat\nu^2/d\epsilon_i$ from an independent refit of every fold, binary, multinomial, ridge, clipped and cross-fitted |
| `TestTheFiniteDifferenceJacobians` and `TestTheNonzeroWitness` | the analytic information, $D_k$ and $\partial g/\partial\theta$ equal finite differences, and the mechanism term is not zero on the witness laws |
| `TestTheDesignInvariances` and `TestTheReflectionIdentity` | a duplicated column and a scaler before an unpenalized model leave the curve unchanged. The reflection identity holds on a saturated model and fails on a misspecified one |
| `TestTheNumericalSandwich` | the standard error equals the numerical stacked sandwich of Saul and Hudgens (2020), Equation (2), on a binary, a multinomial and a ridge fit |
| `TestTheWitnessHasTeeth` | thirteen mutations, M0 to M12, each fail the witnesses that name them. M0 changes nothing and passes |
| `tests/unit/test_omitted_variable_plugin_second_order.py` | the closed forms against central differences, the Cox and Snell bias of a saturated model, and $n\hat B$ against an exact binomial enumeration, in sample and cross-fitted. `TestTheCorrectionRefuses` checks the three correction keys. Eight mutations, M13 to M20, each fail the witnesses that name them |
| `tests/unit/test_omitted_variable_plugin_refusals.py` | each key has a full-string test with its reason, roadmap item and remedy, a qualifying twin of each learner key, and a test of the capture |
| [omitted-variable bound plug-in standard error](method-evidence/omitted-variable-bound-plugin-standard-error.md) | the registered study `omitted-variable-bound-plugin-se`. It compares the reported standard error of each bound with its sampling spread in eight scenarios, in sample and cross-fitted. The corrected $\nu^2$ is measured and not judged. The page states the declaration, the result and the two conservative cells |

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

**Point laws of X82 part F.** Eight fit shapes take a declared law. The table gives each law
and the test module that checks it.

| fit | law | witness |
| --- | --- | --- |
| a multi-arm treatment | the binary tail flip applied to the two arms of a declared pair $(a, b)$: rows of arm $a$ in the tail move to $b$, and rows of $b$ move to $a$. Other arms keep their rows. A contrast alias such as `ate[a vs b]` declares the pair. Any other parameter takes `arm_pair=(a, b)`. A known mechanism becomes $(1-s)g_a + s g_b$ for arm $a$, the mirror for arm $b$, and $g$ elsewhere. The association reads the indicator of arm $a$ on the rows in arm $a$ or $b$ | `test_simulated_confounding_point_laws.py`: the pair order gives identical cells, the third arm keeps its rows, and a one-way swap and a cycle through every arm fail those checks |
| a vector treatment with a categorical component of more than two levels | the same swap on that component, declared as `arm_pair={component: (a, b)}` | `test_vector_simulated_confounding.py::test_a_three_level_component_swaps_its_declared_pair_and_keeps_its_third_level` |
| a clustered fit | the caller declares `latent_level`. `"row"` draws one latent per row. `"cluster"` draws one per cluster. `"mixed"` draws $\sqrt{\rho}\,U_c + \sqrt{1-\rho}\,e$ with a declared `latent_rho`, so each latent stays standard normal. A treatment column that is constant within every cluster refuses `"row"` and `"mixed"` when a cell moves that column. A vector reads each component alone. An outcome-only grid runs at every level | `test_simulated_confounding_point_laws.py`: a cluster latent is constant within each cluster, the mixed latent equals its declared composition, and an outcome-only row latent leaves the treatment unchanged. `test_vector_simulated_confounding.py`: a moved cluster-level flag refuses, and a moved dose runs |
| a categorical benchmark covariate | one block over every encoded indicator. A binary target uses the class-prediction change of an unpenalized logistic fit when the whole block is set to its mean. A Gaussian target uses $R \cdot \mathrm{sd}$, with $R$ the multiple correlation of the block. An indicator name is refused with its logical covariate | `test_simulated_confounding_point_laws.py`: a different reference level gives the same strength to $10^{-8}$, and the default L2 fit does not |
| estimated observation weights | every cell holds the supplied weights on their rows, so it targets the fit's weighted population $E[w\,h(W, A', Y')]$. `weights_estimated` records the declaration | `test_simulated_confounding_point_laws.py`: on a saturated binary law each cell equals the weighted plug-in, and the unweighted plug-in differs by more than $10^{-3}$ |
| a missing outcome or a declared missing treatment | each cell perturbs the recorded values only and keeps every observation indicator. Outcome-axis cells need no declaration, because the latent is independent of the data. A cell that moves the treatment needs `observation_reads=`, the adjustment covariates that every observation mechanism reads, so that $\Delta \perp (A', Y') \mid W$. The report states that assumption, and calibration reads the recorded rows of each target | `test_simulated_confounding_missing_laws.py`: recorded outcomes follow the law and placeholders stay, the indicators never change, and a declaration that names the treatment is refused |
| a controlled direct effect | each cell holds the treatment, flips the binary intermediate in the latent tail with strength `ConfounderStrengthGrid(intermediate=...)`, and moves the outcome. The refit targets the same intermediate level. A nonzero treatment strength is refused | `test_simulated_confounding_intermediate_laws.py`: the treatment is identical in every refit, the intermediate equals the published flip, and the cell reports `induced_intermediate_association` |
| a Gaussian fit whose estimator declares `q_bounds` | `bounded_outcome_law="logit"`, the default there, shifts $\mathrm{logit}((Y - l)/(h - l))$ by $-k_Y U$, and a recorded row at or beyond a bound never moves. `"normal_score"` shifts $\Phi^{-1}(\mathrm{rank}/(m + 1))$, maps it back through the empirical quantile at the Weibull plotting positions, and clips the result to $[l, h]$. It depends on the sample. Every cell reports `induced_outcome_association`, because $k_Y$ is no longer in outcome SD units | `test_simulated_confounding_intermediate_laws.py`: every outcome stays in the bounds at a strength where the linear shift leaves them, rows at a bound stay, and both laws are the identity at zero strength. The normal-score law matches a hand computation at tied values, and its clip is a mutation control |

**Longitudinal latent of X82 part F.** One standard-normal latent value per unit is shared by
every treatment node and the outcome. The contract covers `LTMLE`, `SDR`,
`SequentialGComputation`, `SequentialIPW`, and `ParametricGFormula` fits. The panel has no
censoring, an end-of-study outcome, no time-varying covariate after the first node, binary or
continuous nodes, and no cluster. The table gives the law and the test module that checks it.

| item | law | witness |
| --- | --- | --- |
| node law | a binary node flips on $U \ge \Phi^{-1}(1 - s_t)$, and a dose adds $k_t U$. A one-axis grid applies each strength at every node, and a mapping keys strengths by node column | `test_simulated_confounding_longitudinal.py`: a two-node discrete panel fitted by saturated sequential regressions gives a cell equal to the hand g-formula on the perturbed table. A law that moves node 1 alone gives another value |
| nested flips | every node reads one latent, so the flipped row sets are nested in strength order | the flipped rows of the weaker node sit inside those of the stronger node |
| later nodes | each cell reports, node by node, the weighted correlation of $U$ and $A'_t$. It includes the path through $A'_1, \dots, A'_{t-1}$, because the data generated $A_t$ under the original history | the association of each node equals a hand computation |
| held decision | node 1 moves, and the identity nodes repeat it | the refit data repeat the flipped node 1 at node 2 |
| declared mechanism | a declared node-1 factor becomes $(1 - s)\,g + s\,g(1 - a)$. A declared later node is refused | the refit carries the flipped factor |
| outcome | the point outcome law, with the bounded laws on a `q_bounds` fit | a cross-fitted bounded fit keeps its folds and its bounds |

Each refit passes the original draws' seeds when the root seed is the fit's own, so the refit at
zero strength reproduces the estimate. An explicit seed other than the fit's own redraws them,
and `refit_folds="redrawn"` then says that each cell's movement includes fold noise.

Calibration reads each named baseline covariate against each moved node and the outcome. It conditions on
the other baseline covariates, not on the earlier nodes. A benchmark names a baseline covariate.
A covariate measured before node 1 in the time-varying block is not a benchmark. `longitudinal_fit_refusal`, in
`src/cleverly/sensitivity/_simulated_confounding_longitudinal.py`, applies the ordered
longitudinal stops below. The point rule table hands a longitudinal result to it before any
point rule runs.

| stop | refuses | roadmap |
| --- | --- | --- |
| `estimator` | an `IPSI` fit, whose tilt reads the earlier perturbed nodes | [F13](../roadmap.md#f13-longitudinal-simulated-confounding-replay) |
| `outcome_kind` | a survival, competing-risk, or repeated outcome | [F13](../roadmap.md#f13-longitudinal-simulated-confounding-replay) |
| `censoring` | a censoring node | [F13](../roadmap.md#f13-longitudinal-simulated-confounding-replay) |
| `later_covariates` | a time-varying covariate after the first node | [F13](../roadmap.md#f13-longitudinal-simulated-confounding-replay) |
| `node_law` | a node with more than two levels, or a vector node | [X66](../roadmap.md#x66-simulated-confounding-beyond-x82-part-f) part (j) |
| `cluster` | a clustered panel | [X66](../roadmap.md#x66-simulated-confounding-beyond-x82-part-f) part (k) |
| `known_later_node` | a declared mechanism at a node after the first | [F13](../roadmap.md#f13-longitudinal-simulated-confounding-replay) |

Every binary calibration uses the unpenalized fit, numeric covariates included, so one report
reads one fit. Complete or quasi-complete separation leaves no maximum-likelihood fit (Albert and
Anderson 1984). The surface tests for it with the linear programme of Konis (2007), extended to
$K$ classes. A separated fit reports a `nan` strength. A fit that stops at its iteration limit
also reports `nan`, with its own method text. `summary()` counts both kinds of row. A block
strength is unsigned and a numeric Gaussian strength is signed, and `summary()` says so.

Hartman and Huang (2024) place a latent inside the weight model. The estimated-weight law does not
do that, and regenerated weights would define a population that the fit never targeted. The
[hard-stop review](hard-stop-review.md) records both verdicts (row F11).

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

Ordinary TMLE supports conditional binary arm means, ATE, ratios, PAR, PAF, ATT, ATC, and regime
parameters. It also supports conditional incremental means and contrasts, MSM coefficients under
every built-in link, and modified-policy means and contrasts. Binary C-TMLE and DR-TMLE support
conditional arm means, ATE, and ratios. These paths retain fixed observation weights and estimator-owned repeat aggregation.

A DR-TMLE fit replays a requested stratum at every `guard`, through a complete refit.

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
`ate_policy[...]` contrast subtracts one policy mean from the other, so it removes most of that
level and the column stays small. An `ey_policy[...]` policy mean keeps it, so read the $k_A=0$
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
| binary | backdoor-identified marginal ATE | ordinary TMLE, collaborative TMLE, complete-outcome DR-TMLE, or `AdaptiveIPTW` |
| binary | one explicitly named marginal `ey1`, `ey0`, or `ey[...]` counterfactual mean | ordinary TMLE, collaborative TMLE, complete-outcome DR-TMLE, or `AdaptiveIPTW` |
| binary | marginal `rr` risk ratio or `or` odds ratio | ordinary TMLE, collaborative TMLE, complete-outcome DR-TMLE, or `AdaptiveIPTW` |
| binary | marginal or baseline-stratum `par` population attributable risk or `paf` population attributable fraction | exact ordinary TMLE |
| binary | baseline-stratum arm mean, ATE, risk ratio, or odds ratio | ordinary TMLE, collaborative TMLE, complete-outcome DR-TMLE, or `AdaptiveIPTW` |
| binary | marginal or baseline-stratum ATT or ATC | exact ordinary TMLE |
| binary | marginal or baseline-stratum `ey_regime[...]` mean or `ate_regime[...]` contrast | exact ordinary TMLE; any intervention whose density is declared known, frozen per row |
| binary, multi-arm, or continuous | marginal or baseline-stratum `msm[...]` coefficient | exact ordinary TMLE; any link registered in `cleverly.msm.LINKS`, with fixed projection measure |
| binary or multi-arm | marginal or baseline-stratum `ey_ipsi[...]` mean or `ate_ipsi[...]` contrast | exact ordinary TMLE; fixed odds multipliers with a refitted mechanism |
| continuous | one explicitly named marginal or baseline-stratum `ey_policy[...]` policy mean | exact ordinary TMLE |
| continuous | one explicitly named marginal or baseline-stratum `ate_policy[...]` contrast | exact ordinary TMLE |

Every row accepts fixed probability weights under its listed estimators.

A binary row admits an estimator whose class declares `simulated_confounding_replay = True`. `TMLE`, `CTMLE`, `DRTMLE` and `AdaptiveIPTW` declare it. `tests/unit/test_simulated_confounding_gate.py` collects every declaring class and checks a zero-strength refit and one nonzero cell for each.

An `AdaptiveIPTW` cell refits the stored estimator on the perturbed data (X82 part I). On a fit
with a declared mechanism, a cell that flips the treatment declares the mechanism of the flipped
treatment, $(1-k_A)\,g(a\mid W)+k_A\,g(1-a\mid W)$. The flip reads only the latent, so that
is the exact mechanism of the cell, and the refit keeps its interval. A regime or MSM of an
`AdaptiveIPTW` fit is refused ([X81](../roadmap.md#x81-adaptiveiptw-follow-ups) part (d)).
`tests/unit/test_adaptive_iptw_refute.py` holds the zero-strength reproduction, a nonzero cell,
and the check that a known-mechanism cell equals a direct fit with the flipped mechanism.

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
through `RegimeSet.evaluate` or `MSMSet.evaluate`. Each evaluator checks the declaration of each
`Stochastic` density, or of the `MSM` design and weight, before it runs a callback. The operation
compares the callback output with the stored arrays. A refused declaration or a disagreement raises
`CapabilityError`.

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

The summary states this reading on its `incremental reading` line.
On a multi-arm fit, $q_\delta(a \mid W) = \delta_a g(a \mid W) / \sum_b \delta_b g(b \mid W)$, and the cells swap the declared `arm_pair`.
Kennedy (2019), equation (1) and Corollaries 1–2, supplies the parameter and influence curve.
The [source audit](../references.md#sensitivity-analysis) separates that result from the qualitative perturbation.

A multiplier-one mean equals the natural-course mean and is refused before a random draw.
A contrast against that reference remains supported. Means and contrasts report additive movement.
The operation validates every stored repeat against its own initial, untruncated propensity predictions.
It checks intervention names, multipliers, densities, clever weights, derivatives, and the reference.

**Learned rules.** A refit of a learned-rule fit relearns the rule inside each training fold.
A perturbed refit would then estimate the value of other rules.

The replay keeps the outer folds of the fit.
It also keeps the realized rule $d_v(W_i)$ of each row, which `result.nuisance.regimes` stores.
The perturbation changes no covariate, so each frozen rule value stays defined.
Each cell evaluates $\sum_v V^{-1} n_v^{-1} \sum_{i \in v} \bar Q(d_v(W_i), W_i)$ on the perturbed table.
The weights are the $1/V$ fold weights of the learned-rule record.
The [learned-rule contract](point-treatment-tmle.md#learned-rules) defines this target.

The summary states the reading on its `learned-rule reading` line.
A cell reads confounding of the value of these fixed rules, not of the learned-rule procedure.
`FrozenLearnedRule`, in `src/cleverly/interventions/learned.py`, carries the frozen values.

| check | witness in `tests/unit/test_simulated_confounding_learned_rule.py` |
| --- | --- |
| zero strength | the cell equals the reported estimate exactly |
| nonzero cells | each cell equals a manual frozen-rule refit of its perturbed table |
| frozen rules | a perturbed refit keeps every row's rule and the folds, and a relearning refit changes some rules |
| fold weights | on folds of 81 and 80 rows, the cell equals the mean of the fold values and not the row-weighted mean |
| mutation controls | a replay that relearns the rule, or that redraws the folds, fails the nonzero cell check |

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
[X65](../roadmap.md#x65-sensitivity-analyses-beyond-the-shipped-scope) part (j) holds it.
Its combined assessment records that row as unavailable and continues to the requested surface.
Outcome diagnostics remain available; integration-grid positions never become treatment-arm calibration rows.
`tests/unit/test_continuous_msm_assessment.py` checks these reporting boundaries.

| witness | what it checks |
| --- | --- |
| `tests/unit/test_simulated_confounding_incremental.py` | nonzero complete refits, mechanism rebuilding, natural-course refusals, repeated provenance, and persisted assessment |
| `tests/unit/test_simulated_confounding_msm.py` | built-in links, fixed quadrature, observed-dose rebuilding, support masks, and corrupted cached arrays |
| `tests/unit/test_simulated_confounding_off_axis.py` | on a saturated finite law, a multi-arm incremental cell, a stratum incremental cell and a multi-arm MSM coefficient each equal the hand g-formula on the refitted table. A frozen-mechanism control misses the incremental cell. A stratum continuous-MSM cell equals a manual refit |
| existing incremental and MSM estimator tests | unchanged targeting, influence curves, and projections |

These are deterministic diagnostic checks, not new repeated-sampling or interval claims.
The registered incremental and point-MSM artifacts remain unchanged.
Stratified incremental, linked-MSM and continuous-MSM fits replay too, as do stratified
DR-TMLE fits at every `guard`. Every replay is a complete refit, which targets the stratum blocks
as the fit did. `tests/unit/test_simulated_confounding_strata.py` checks that the zero-strength
cell of a stratum alias equals the fit's own stratum estimate, and that every other cell equals
the manual refit of its replaced data.
The point-MSM study covers identity-link finite-arm estimation; it supplies no nonlinear or continuous-MSM interval validation.

For `repeats > 1`, each non-anchor cell calls the replay estimator once. The estimator owns all
repeat draws and reports their coordinatewise median. Additive displacement compares the refitted
median estimate with the original median estimate. Ratio displacement compares their median log
estimates. The result records `n_repeats` and `repeat_aggregation="coordinatewise_median"`.

The root seed gives every non-anchor cell the same repeat seed sequence. The perturbation cannot
move a fold. The draw reads the row count, the cluster labels and the seed, and it balances
neither the treatment nor the outcome, so the realized partition is the same under the same seed.

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
uses it in the outcome fits, the mechanism fits, targeting, and plug-in.

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

End-to-end mutations drop every selector weight or drop the shared outcome-adaptive
mechanism weights. Each mutation moves a manual refit that runs on the surface's seeds at
strengths 0.2 and 0.3. Neither control reads a surface cell, because its grid holds the anchor
alone and the anchor runs no refit. Each control requires a move above one part in a thousand of
the unmutated refit's estimate.

The per-arm outcome-adaptive design has its own weight controls in
`tests/unit/test_outcome_adaptive_per_arm.py`. Integer weights must equal duplicated rows, and a
cross-fitted longhand with fixed weights must agree to `1e-8`. Dropping the weights from the
per-arm mechanism, in sample or in each fold, fails these checks.

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
walks that table and returns the first reason that applies.

Two named stops apply in this order: longitudinal and risk-ratio tilt fits. The longitudinal stop waits on a declared law.
The tilt stop waits on a typed study request ([X85](../roadmap.md#x85-typed-study-requests-for-the-risk-ratio-tilt)). The longitudinal stop applies
the ordered stops of the longitudinal latent, and a longitudinal result that passes them meets no
point rule. Each one has a row in the
refusal table below, and each one carries a roadmap item. The missing-outcome, missing-treatment,
and intermediate stops read the grid, so they run in `_validate_request` and not in this table.

Twelve provenance and shape stops read the object you hold rather than the state of the science.
The `result_type` stop runs second, before the risk-ratio tilt stop, because every later rule reads a
field that only a `TMLEResult` declares. The other eleven follow the risk-ratio tilt stop. The table
below gives all twelve in contract order.

| rule | refuses |
| --- | --- |
| `result_type` | an artifact that is not exactly a point-treatment `TMLEResult` |
| `missing_estimator` | a result that stores no replay estimator |
| `repeat_provenance` | a stored draw count, a `config.crossfit.repeats`, and a replay estimator `repeats` that disagree |
| `binary_estimator` | a binary fit made by an estimator other than `TMLE`, `CTMLE`, or `DRTMLE` |
| `continuous_estimator` | a continuous fit made by anything but exact ordinary `TMLE` |
| `outcome_family` | an outcome family other than `gaussian` or `binomial` |
| `weight_kind` | a weight kind other than `probability` |
| `weight_provenance` | a `weights_name` and a `WeightSpec` name that disagree |
| `undeclared_weights` | nonconstant observation weights with no declared weight column |
| `identification` | a fit that records no identification metadata |
| `functional` | an identified functional other than a backdoor mean contrast |
| `provider` | backdoor provenance from anything but a registered explicit adjustment set |

`_validate_request` applies the first reason before calibration, a random draw, or a refit. The
assessment capability walks the same table, so `available`, `status`, `reason`, and direct
execution name the same stop for all fourteen. A fit that reaches the `provider` stop reports
`reason="simulated_confounding needs registered explicit-adjustment backdoor provenance"` on its
capability row.

The rows below record the two fit-wide named stops and the three grid stops, plus the estimands and compositions the
surface refuses. The `code` column holds the reason code of the raise site. The codes are those of
[how to read a refusal](scope-and-refusals.md#how-to-read-a-refusal). A row that cites an `F` item
of the roadmap also has a part that waits on published theory, as the
[roadmap's eligibility rules](../roadmap.md#eligibility) define the term.

| refused | code | why |
| --- | --- | --- |
| a longitudinal result outside the time-fixed latent | `implementation_gap` | no time-indexed latent law covers censoring, survival outcomes, a time-varying covariate after the first treatment node, a declared later-node mechanism, an `IPSI` tilt, or a time-varying latent. See [F13](../roadmap.md#f13-longitudinal-simulated-confounding-replay). A node with more than two levels, a vector node, and a clustered panel wait on [X66](../roadmap.md#x66-simulated-confounding-beyond-x82-part-f) parts (j) and (k) |
| a risk-ratio tilt fit | `implementation_gap` | no typed study request reports a tilt, so no tilt fit carries the backdoor identification that the surface reads. See [X85](../roadmap.md#x85-typed-study-requests-for-the-risk-ratio-tilt) |
| a cell that moves the treatment of a fit with a missing outcome, and no `observation_reads=` | `implementation_gap` | the observation mechanism may read the treatment, and then the flip breaks missing at random. Outcome-axis cells run. See [F12](../roadmap.md#f12-missing-outcome-simulated-confounding-replay) |
| a cell that moves a declared missing treatment, and no `observation_reads=` | `implementation_gap` | as the missing-outcome row. See [F12](../roadmap.md#f12-missing-outcome-simulated-confounding-replay) |
| a cell that moves the treatment of a controlled direct effect | `implementation_gap` | a flip of the treatment makes the latent confound the intermediate and the outcome by an unnamed path. The intermediate and outcome axes run. See [F15](../roadmap.md#f15-controlled-direct-effect-simulated-confounding-replay) |
| identification other than a backdoor mean contrast with explicit adjustment | `invalid_declaration` | the surface reads registered explicit-adjustment provenance |
| ATT or ATC under C-TMLE or DR-TMLE | `implementation_gap` | `CTMLE` and `DRTMLE` refuse these functionals when they estimate, so no such fitted result exists |
| a binary fit under an estimator class that does not declare `simulated_confounding_replay = True` | `implementation_gap` | the surface refits the stored estimator, and the declaration states that its `refit` refits every nuisance. The class itself must declare it, so a subclass does not inherit it. See [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) part F, further replays (e) |
| a regime, incremental target, or MSM under C-TMLE or DR-TMLE | `implementation_gap` | the method catalog lacks the corresponding collaborative score or reduced-dimension correction. See [X50](../roadmap.md#x50-further-dr-tmle-compositions) and [X51](../roadmap.md#x51-c-tmle-point-estimates-for-further-targets) for regimes and MSMs, and [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) for incremental targets |
| continuous-treatment C-TMLE and DR-TMLE | `implementation_gap` | both estimators refuse a modified-policy functional. See [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| PAR or PAF under C-TMLE or DR-TMLE | `implementation_gap` | the method catalog lacks a score or correction for these observed-law contrasts. See [X66](../roadmap.md#x66-simulated-confounding-beyond-x82-part-f) part (h), which depends on [X50](../roadmap.md#x50-further-dr-tmle-compositions), [X51](../roadmap.md#x51-c-tmle-point-estimates-for-further-targets) and [X62](../roadmap.md#x62-other-dr-tmle-and-c-tmle-compositions) |
| `NaturalCourseMean`, reported as `ey_obs` | `invalid_declaration` | $E[Y]$ has no counterfactual treatment term. Outcome perturbation alone cannot make it a confounding diagnostic |
| the policy mean of a zero-delta shift | `invalid_declaration` | $d_0(a, w) = a$ on both branches, so the policy is the natural course. Its mean is $E[Y]$, and no counterfactual treatment dependence remains for a common cause to move. The `ate_policy[...]` contrast that uses this policy as its reference is still accepted |
| an incremental mean at multiplier one | `invalid_declaration` | its mean is $E[Y]$. Contrasts with this reference remain supported |
| a row-level or mixed latent that moves a treatment constant within every cluster | `invalid_declaration` | the latent varies within a cluster, so it would move single rows of a cluster-level assignment. `latent_level="cluster"` and an outcome-only grid run |
| a generated indicator of a categorical covariate as a benchmark | `invalid_declaration` | one indicator does not define a covariate's strength. The message names the logical covariate, which calibrates as a block |

Some rows above record an upstream limit rather than a surface limit. The identified effect's
method catalog refuses unsupported variant targets at `estimate()`, before it builds an estimator.
`DRTMLE` itself refuses a baseline stratum at a non-empty `guard` before any learner. No fitted
result reaches the surface guard in these cases, and `_replay_refusal` keeps that guard as defence
in depth.

A `DRTMLE` fit at `guard=()` is the exception. It fits strata, so a request for one of its
stratum-specific aliases reaches `_replay_refusal`. The stratum guard is the only refusal of that
request. The surface covers marginal DR-TMLE targets only, and the message tells the caller to fit
the ordinary TMLE, which `guard=()` is.

| composition | layer that refuses | when | message |
| --- | --- | --- | --- |
| ATT or ATC under C-TMLE or DR-TMLE | the identified effect's method catalog | `estimate()` | `method 'collaborative_tmle' cannot estimate ATT: no collaborative score is evidenced for this functional`. `DRTMLE` names its reduced-dimension correction instead |
| a modified-treatment policy under C-TMLE or DR-TMLE | the identified effect's method catalog | `estimate()` | `method 'drtmle' cannot estimate ModifiedTreatmentPolicy: no reduced-dimension correction is evidenced for this functional` |
| PAR or PAF under C-TMLE or DR-TMLE | the identified effect's method catalog | `estimate()` | `method 'collaborative_tmle' cannot estimate PopulationAttributableRisk: no collaborative score is evidenced for this functional`. DR-TMLE names its reduced-dimension correction; PAF names its own type |
| regime or MSM under C-TMLE | the identified effect's method catalog | `estimate()` | `method 'collaborative_tmle' cannot estimate RegimeMean: no collaborative score is evidenced for this functional`. Other targets name their own type |
| regime or MSM under DR-TMLE | the identified effect's method catalog | `estimate()` | `method 'drtmle' cannot estimate RegimeMean: no reduced-dimension correction is evidenced for this functional`. Other targets name their own type |

The surface also refuses a constant benchmark covariate. That check reads the requested covariate
rather than the fitted result, so it sits outside the fit-wide table. The latent-level, observation,
intermediate, and bounded-law checks read the request too.

`simulated_confounding_refusal` returns the refusal of one request before any draw. It reads the
fit-wide table and then the parameter and calibration checks that the call runs. The capability
row reads it for each request, with the request's `benchmark_covariates` held fixed.

| request | row |
| --- | --- |
| a refused parameter, such as `ey_obs` or a zero-delta policy mean | `unavailable`, with the call's sentence |
| a categorical indicator or a constant benchmark covariate | `unavailable`, with the call's sentence |
| no `estimand` on a natural-course fit | `unavailable`, because no reported parameter runs |
| a malformed argument | unchanged. The call raises `ValueError` or `TypeError` |

A malformed argument is a duplicate, non-string, or unknown covariate name, an unknown estimand, or
a grid strength out of range. A request on a continuous fit that names no policy parameter is
malformed too. The call reports a malformed name before any parameter or calibration refusal.
`TestTheSimulatedConfoundingRowResolvesEachRequest` in
`tests/unit/test_capability_row_predicates.py` checks four study fits.

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

The E-value is the threshold when the two confounder associations have equal strength.
Unequal strengths can trade off under the bounding factor of
[Ding and VanderWeele (2016)](https://arxiv.org/html/1507.03984).
For $RR=2$, the E-value is $3.414$. Associations of $3$ and $4$ give a bounding factor of
$3\times4/(3+4-1)=2$, so both associations need not reach $3.414$.
The E-value does not decide whether a confounder of either strength is plausible.

The selected path depends on the reported contrast and retained artifacts.

| request | E-value path |
| --- | --- |
| reported risk ratio | use it directly and mark the result exact |
| unambiguous default binary marginal ATE or odds ratio from `TMLE`, `CTMLE` or `DRTMLE` | retarget cached nuisances to the matching risk ratio and mark the result exact; combined runs include this cheap retarget by default |
| explicit reported odds ratio, or default odds ratio without exact retarget support | use the common-outcome approximation $\sqrt{OR}$ and mark the result approximate |
| binary ATE without exact retarget support, with a usable reported reference-arm mean | hold the baseline risk fixed and mark the result approximate; includes CV evaluation |
| Gaussian ATE, ATT, or ATC with every outcome observed | standardize by the observed outcome standard deviation, weighted on a weighted fit, and mark the result approximate |
| Gaussian ATE, ATT, or ATC on a fit with a response mechanism | standardize by the missing-at-random plug-in of the population `sd(Y)` and mark the result approximate. The plug-in fits one more outcome regression, so the default assessment defers it. A fit that also misses treatments reports `unavailable`, because an unrecorded row has no $A$ for the regression |
| a contrast within a baseline stratum, from `strata=` | apply the path of the marginal contrast to the stratum's rows. The derived risk ratio and the standardizing `sd(Y)` read the stratum |
| two reported risks at one horizon of a `TimeToEvent` fit, `estimand=(numerator, denominator)` | read their risk ratio from `result.ratio(numerator, denominator, kind="rr")` and mark the result exact. Both regimens must be static |
| binomial ATT or ATC | refuse because the conditional baseline risk and conditional ratio target are absent |
| level or non-arm parameter | report `not_applicable` because no supported two-arm contrast exists |
| two-arm contrast on a fit whose status supplies no inference: a collaborative fit at any `strategy`, a `DRTMLE` fit with a `guard` and varying weights declared estimated, or a clustered fit under `"few_cluster_plugin"` | report `unavailable`. Each branch reads the estimate's interval or the reference arm's standard error, and this fit supplies neither. The reason of the status follows |
| any request on a learned-rule fit | report `unavailable` before the estimand is resolved. No E-value of a regime contrast ships, and a learned-rule contrast is not a contrast of two exposure levels. See [F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) |
| any request on a controlled-direct-effect fit with a discrete treatment, which is such a fit with `intermediate=` | take the path of the same request without the intermediate variable, then invert the per-arm product below. A result that names only the column or only the level reports `unavailable` |
| a `TimeToEvent` fit without an explicit pair | report `deferred` and name the reported risks |
| any other longitudinal fit, or a `TimeToEvent` pair with a dynamic regimen or with two horizons | report `unavailable` and name [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) or [F34](../roadmap.md#f34-e-values-beyond-the-total-population-arm-contrast) |
| binomial ATE without exact retarget support or a usable reported baseline | report `unavailable` and name the missing evidence, artifact, or target |
| several eligible contrasts and no explicit estimand | report `deferred` and name `estimand` in the next step |

`_select_evalue` in `cleverly.sensitivity.evalue` runs four fit-wide checks before it resolves
the estimand. The table gives them in the order they run.

| check | status |
| --- | --- |
| a longitudinal fit, other than a `TimeToEvent` pair | `unavailable`, or `deferred` on a `TimeToEvent` fit |
| a continuous treatment | `not_applicable` |
| a learned-rule fit | `unavailable` |
| a result that names only one half of an intermediate variable | `unavailable` |

So a continuous-treatment fit with an intermediate variable reports `not_applicable`.

The status refusal follows the `not_applicable` check of the request. A request for `ey1` on a
collaborative fit therefore still reports `not_applicable`. Several eligible contrasts require an
explicit alias, which is the `deferred` row above.
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
Only the lower bound can leave it. `wald_ci` gives `high >= psi`, and the refusals above force `baseline.psi > 0` and `baseline.psi + psi > 0`.
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

#### Controlled direct effect

A fit with `intermediate=` reports the risk ratio of arms $a$ and $a'$ at the fixed level $z$. The
bias of that ratio splits into one factor per arm. The split is an identity. Ding and VanderWeele
(2016), arXiv:1507.03984v1, eAppendix Proposition A.2, bounds each factor by its own bounding
factor. Smith, Mathur and VanderWeele (2021), arXiv:2005.02908v1, Result 1, uses the same per-arm
product.

| term | definition |
| --- | --- |
| treated-arm strength | $\sup_u P(u \mid A = a, Z = z, W) / P(u \mid W)$. It mixes the association of $U$ with $A$ and with $Z$ |
| control-arm strength | $\sup_u P(u \mid W) / P(u \mid A = a', Z = z, W)$, the reciprocal direction |
| outcome strength | one $RR_{UY}$ for each arm |
| conditions | $Y_{az} \perp A \mid W, U$ and $Y_{az} \perp Z \mid A, W, U$; $U$ is a baseline latent that $A$ does not affect; positivity given $(W, U)$; no confounder of $Z$ and $Y$ that $A$ affects |
| inversion | with four equal strengths the product is $BF(E, E)^2$, so the E-value is the ordinary E-value of $\sqrt{RR}$ |

The square root comes after the path's own conversion. An odds ratio therefore enters as
$OR^{1/4}$, and a Gaussian contrast as $\exp(0.905d)^{1/2}$. The confidence limit takes the root
of the limit nearest the null. The report states the four strengths and the conditions. The
derived risk ratio passes the fitted level of $Z$ to the retarget.

`tests/unit/test_evalue_part_g.py` checks the bound on two finite laws. On the first law each
per-arm factor alone, and the treated factor squared, is below the bias, and the product bounds it.
The second law explains its ratio away with every strength at most 9. The per-arm E-value of that
ratio is 4.76, and an inversion without the square root asks for 13.6.

VanderWeele (2010) assumes that the latent confounds only the intermediate and the outcome, so it is
context and not the base. The table below lists the other sources that the review read.

| source, and the version read | what it covers | what it does not cover |
| --- | --- | --- |
| VanderWeele (2010), [PMC4231822](https://pmc.ncbi.nlm.nih.gov/articles/PMC4231822/) | Theorem 1 and Appendix 1 give controlled-direct-effect bias formulas when the latent confounds only the intermediate and the outcome | a latent that also confounds the treatment. The [hard-stop review](hard-stop-review.md) records it as context for X82 part G |
| Smith and VanderWeele (2019), [author PDF](https://www.louisahsmith.com/publications/smith2019mediational.pdf), pages 835–837, and [supplement](https://download.lww.com/wolterskluwer_vitalstream_com/PermaLink/EDE/B/EDE_2019_07_22_SMITH_EDE18-0696R2_SDC1.pdf), Sections 1–2 | an approximate mediational E-value for natural direct and indirect effects under its stated confounding model | no mapping in this source to the fitted population controlled direct effect at fixed $z$ |
| Ding and VanderWeele (2016), *Biometrika*, [arXiv 1601.05155](https://arxiv.org/html/1601.05155), Sections 2–4 and Appendix D.4 | bounds for natural direct and indirect effects; Appendix D.4 drops treatment-confounder independence and changes the target to the unexposed population | no stated mapping to this package's population controlled direct effect at fixed $z$ |
| Ding and VanderWeele (2016), *Epidemiology*, [arXiv 1507.03984](https://arxiv.org/html/1507.03984), Sections 2–3 and Appendix 2.5 | a bounding factor for an exposure contrast with general, possibly multivariate confounders and averaging over measured covariates | a composite exposure such as $(A,Z)$ needs an explicit confounding and standardization mapping before the factor describes the fitted target |
| Gilbert, Fong, Kenny and Carone (2023), [*Biostatistics*](https://academic.oup.com/biostatistics/article/24/4/850/7320953), Sections 2.1 and 4.1 | a controlled direct effect and an E-value for a marker contrast within a fixed vaccination arm | its E-value compares marker interventions within one vaccination arm, not treatment arms at fixed marker value |
| Mathur, Ding, Riddell and VanderWeele (2018), [PMC6066405](https://pmc.ncbi.nlm.nih.gov/articles/PMC6066405/) | software for ordinary E-values and effect-scale conversions | no mapping of that software's ordinary conversion to this fitted controlled direct effect |
| Tingley, Yamamoto, Hirose, Keele and Imai (2014), [*Journal of Statistical Software*](https://www.jstatsoft.org/article/view/v059i05) | the `mediation` package and its mediation sensitivity analysis | no E-value bound for this fitted controlled direct effect is supplied by its software account |
| the R package `EValue` 4.1.4, [CRAN manual](https://cran.r-project.org/web/packages/EValue/EValue.pdf) | software for confounding, selection bias, and measurement error | its lack of a mediation API is a software limit, not a mathematical impossibility |

The search did not read these sources.

| source | title |
| --- | --- |
| VanderWeele and Ding (2017), appendix | the online appendix of *Sensitivity analysis in observational research: introducing the E-value* |
| VanderWeele, Ding and Mathur (2019) | [*Technical considerations in the use of the E-value*](https://biostats.bepress.com/harvardbiostat/paper215/), *Journal of Causal Inference*. The abstract was available; the full text was not |
| VanderWeele (2015) | *Explanation in Causal Inference: Methods for Mediation and Interaction*, a book |

The Smith and VanderWeele supplement was read in the present review. The 2017 appendix and
the full text of the 2019 technical discussion remain unread. A search that did not locate a
target-specific theorem does not establish that none exists.

A weighted fit standardizes by the weighted outcome standard deviation.
The estimate targets the population the observation weights describe, and the standardizing scale describes the same population.
The weighted form divides by the reliability-weight correction $\sum w - \sum w^2 / \sum w$.
That correction is $n - 1$ when every weight is one, so an unweighted fit reports the plain sample standard deviation.

#### Confidence limits and their coverage evidence

Each report asks the evidence registry before it prints a confidence-limit E-value, with the
instrument `evalue_limit` and one rule per configuration. Two configurations on one fit join with
`+`, for example `cde+stratum`.
[E-value limit coverage](method-evidence/evalue-limit-coverage.md) registers L2 rows for two rules.

| rule | configuration | registry row |
| --- | --- | --- |
| `cde` | a controlled direct effect | none |
| `stratum` | a contrast within a baseline stratum | in-sample `TMLE` risk ratios `rr[V=s]`, at least 200 events |
| `response_sd` | the standardized path with a response mechanism | none |
| `derived_collaborative_tmle`, `derived_drtmle` | the derived risk ratio of a `CTMLE` or `DRTMLE` fit | none |
| `time_to_event` | the risk ratio of a `TimeToEvent` pair | in-sample hazard `LTMLE`, five nodes, at least 100 events |

Each row also requires a binary or survival outcome, $n \geq 2{,}000$, and the logistic learners of
the study. A fit outside a row prints its limit as descriptive. `EValue.limit_reading` holds the
resolution, and `EValue.limit_descriptive` is true for a descriptive limit. A `DRTMLE` fit with an
estimated mechanism also states that its remainder is linear in the mechanism error.

### Marginal sensitivity model bounds

**How.** [`sensitivity/marginal.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/marginal.py)
bounds the arm means and the ATE of a two-arm fit under the marginal sensitivity model of Tan
(2006, Section 4.2, eq. (9)). The parameter $\Gamma \ge 1$ bounds the factor by which an
unobserved $U$ moves the odds of treatment, given $W$:

$$
\Gamma^{-1} \le \frac{P(A = 1 \mid W, U) / P(A = 0 \mid W, U)}{e(W) / (1 - e(W))} \le \Gamma .
$$

$\Gamma = 1$ is no unmeasured confounding. Call `result.sensitivity.marginal_bounds(gamma=...)` or
`cleverly.sensitivity.marginal_sensitivity_bounds(result, gamma=...)`.

| term | definition |
| --- | --- |
| $\tau$ | $\Gamma / (1 + \Gamma)$ for an upper bound, $1 / (1 + \Gamma)$ for a lower bound |
| $q(W)$ | the $\tau$-quantile of $Y$ given $A = 1$ and $W$ |
| $\varphi_\Gamma(u)$ | $\Gamma u_+ - \Gamma^{-1} u_-$ |
| $R(q)$ | $q + \varphi_\Gamma(Y - q)$ |
| $\rho(W)$ | $E[R(q) \mid A = 1, W]$, the largest value of $E[Y(1) \mid A = 0, W]$ that the model allows |

The model allows $E[Y(1) \mid A = 0, W] = E[wY \mid A = 1, W]$ for any weight $w$ in
$[\Gamma^{-1}, \Gamma]$ with conditional mean one. The largest such value is a linear program.
Its dual is $\min_q \, q + E[\varphi_\Gamma(Y - q) \mid A = 1, W]$, and the $\tau$-quantile solves
it.

Any other $q$ gives a larger value, so it gives a valid bound that is not sharp. These
statements hold for a continuous and for a binary outcome. The upper bound of $E[Y(1)]$ is
$E[AY + (1 - A)\rho(W)]$. The lower bound applies the same step to $-Y$. Arm 0 exchanges the arms.

| quantity | value |
| --- | --- |
| upper bound of the ATE | upper bound of $E[Y(1)]$ minus lower bound of $E[Y(0)]$ |
| lower bound of the ATE | lower bound of $E[Y(1)]$ minus upper bound of $E[Y(0)]$ |
| bounds at $\Gamma = 1$ | the cross-fitted AIPW estimate |

The arm-mean bounds are sharp. The ATE set is the componentwise difference, and this page makes
no sharpness claim for it.

**Estimator.** The bound is the cross-fitted mean of
$AY + (1 - A)\hat\rho(W) + A\,\frac{1 - \hat e(W)}{\hat e(W)}\bigl(R(\hat q) - \hat\rho(W)\bigr)$.

| nuisance | continuous outcome | binary outcome |
| --- | --- | --- |
| $\hat e$ | the fit's out-of-fold propensity, bounded at its `g_bounds` | the same |
| $\hat q$ | a quantile regression on $W$ among the arm, cross-fitted on the fit's folds | $1\{\hat\mu > 1 - \tau\}$ |
| $\hat\rho$ | $\hat\mu / \Gamma + (1 - 1/\Gamma)\,\hat c$. $\hat\mu$ is the fit's out-of-fold outcome regression, and $\hat c$ regresses $q + (Y - q)_+ / (1 - \tau)$ on $W$ | $\min(\Gamma\hat\mu, 1 - (1 - \hat\mu)/\Gamma)$ |

For each predicted fold, the quantile trains on one half of the other folds, and the tail
regression trains on the other half. So no training pseudo-outcome reads a quantile fitted on
its own row. The authors' code uses the same split (`DVDS.R`, `semiadaptive = FALSE`). The
default quantile learner selects a linear quantile regression or a shallow boosted quantile
regression by out-of-fold pinball loss. The default tail learner selects a linear regression or
a shallow booster by out-of-fold squared loss. `quantile_learner=` and `pseudo_outcome_learner=`
replace them. A fit with fewer than three folds gets its own five-fold split for these regressions.

| property | condition |
| --- | --- |
| valid bound | $\hat e$ is consistent, or both $\hat\mu$ and $\hat c$ are consistent |
| sharp bound | also, $\hat q$ converges to the quantile. For a binary outcome, $\hat\mu$ puts little mass near $1 - \tau$ (a margin condition) |
| second-order quantile error | the outcome is continuous at the quantile, so the derivative of $E[R(q) \mid A = 1, W]$ in $q$ is zero there |

The formulas match the authors' replication code for Dorn, Guo and Kallus,
[`CausalML/DoublyValidDoublySharpSensitivity`](https://github.com/CausalML/DoublyValidDoublySharpSensitivity)
at `26cbbf6e`, `simplified_dvds.R` (`phi.continuous`, `dvds.binary`). This page cites no theorem
of that paper, because the published version was not read.

**Interval.** The set interval is $[\text{lower} - z\,\text{se}_-, \text{upper} + z\,\text{se}_+]$
with $z = \Phi^{-1}(1 - (1 - \text{level})/2)$. By the union bound it covers the identified set
with asymptotic probability at least `level`. Each standard error is the variance of the side's
influence curve, summed within clusters on a clustered fit. A fit with `repeats > 1` applies the
median rule to each side. The interval of Imbens and Manski (2004) for the parameter alone is
narrower and is not shipped.

**Breakdown.** `breakdown` is the smallest grid $\Gamma$ whose bounds contain zero.
`limit_breakdown` reads the set interval. An arm mean has no null, so both are `None`.

**Scope.**

| fit | result |
| --- | --- |
| `TMLE` (CV-TMLE included) or `DRTMLE`, two arms, Gaussian or binary outcome | ships; clusters and repeats are supported |
| more than two arms, weights, strata, missing outcomes, `CTMLE`, `AdaptiveIPTW` | `implementation_gap`, [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) part H2 |
| a controlled direct effect, a missing treatment | `implementation_gap`, [X87](../roadmap.md#x87-marginal-sensitivity-model-bounds-that-need-a-second-sensitivity-parameter) |
| a longitudinal fit | `implementation_gap`, [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) |
| a continuous treatment, a policy, a regime, an incremental intervention, an MSM, a structural nested mean model | `different_estimand` |
| a fit whose estimate has no influence-curve inference | `inference_gap` |

**Evidence.** `tests/unit/test_marginal_sensitivity.py` solves the linear program on a finite law
with `scipy.optimize.linprog`. On the same law, rows in exact proportion to the probabilities make
the estimator's mean equal to the population value. The binary branch and the continuous branch,
with oracle nuisances, each reproduce the sharp bound to `rtol=1e-12`. Mutation controls move the
quantile to the other side, change the correction weight to $1/\hat e$, and replace both
nuisances. Each moves the value off the sharp bound. At $\Gamma = 1$ the bounds equal an
independent AIPW computation.

On a Gaussian law, a quantile error $\delta$ moves the continuous side by $O(\delta^2)$. The
control at the quantile of the other side moves it by $O(\delta)$.
A parametrized test refuses each composition in the scope table before a learner fits.

**Confidence limits.** The limits resolve through the evidence registry with the instrument
`msm_bound_limit`. The rule is `continuous` or `binary`. A continuous outcome with a
caller-supplied quantile or tail learner adds `+custom_learners`, and a clustered fit adds
`+clustered`. The registry has no `msm_bound_limit` row, so the limits print as descriptive
until study SH registers its coverage cells.

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

The tilt applies $\gamma$ on the logit of the outcome scale that the fit uses.
A binary outcome uses that scale directly.
A continuous outcome maps onto $[0, 1]$ through the declared `q_bounds`.
Without `q_bounds`, `OutcomeScaler.from_outcome` takes the observed range and widens it by 10% on each side.
Declare `q_bounds` to fix what one unit of $\gamma$ means for a continuous outcome.

Away from $\gamma = 0$, the curve is a plug-in value.
`_tilted_value` reuses the targeted $\bar{Q}^*$ of the $\gamma = 0$ fit and does not target again at each $\gamma$.
The accuracy of the curve away from $\gamma = 0$ therefore depends on the outcome model.

This is a retarget operation and not a refit.

The tilt re-mixes each reported parameter in the table below. Each row repeats the arithmetic of
the target that reported the parameter on the tilted regressions. Each curve therefore passes
through the reported estimate at $\gamma = 0$. $\bar{Q}_\gamma(a, W)$ is the mixture above, and
$\gamma_a$ is the tilt of arm $a$.

| parameter | point curve at $\gamma$ | spread at $\gamma \neq 0$ |
| --- | --- | --- |
| an arm mean, a contrast, `att`, `atc` | the same functional of $\bar{Q}_\gamma(a, W)$ | the reported standard error under `std_err`, `ci_lower` and `ci_upper` |
| `rr`, `or` | the ratio of two tilted arm means | the reported log-scale standard error, centred on the log of the tilted point |
| `ey_obs` | $P_n[\pi(A, W)\,\bar{Q}^*(A, W) + \{1 - \pi(A, W)\}\,\operatorname{expit}\{\operatorname{logit}\bar{Q}^*(A, W) + \gamma_A\}]$ | the reported standard error, centred on the tilted point |
| `par`, `paf` | the tilted `ey_obs` and the tilted reference-arm mean | as above |
| a regime mean or contrast | $P_n[\sum_a g^\star(a \mid W)\,\bar{Q}_\gamma(a, W)]$ | as above |
| an MSM coefficient | the projection of $\bar{Q}_\gamma(a, W)$ onto the working model, with its design, weights and link | as above |

Every row except the first publishes its spread at $\gamma \neq 0$ under `plugin_std_err`,
`plugin_interval_lower` and `plugin_interval_upper`. The reported influence curve is the
missing-at-random curve, so this spread is a diagnostic and not an interval for the tilted
parameter. At $\gamma = 0$ each row publishes the fit's own spread. A frame that mixes the two
kinds carries both sets of columns, with NaN where a row has no value.

The `ey_obs` row reads the natural-course fluctuation at the received arm, and the mechanism at
that arm. It replaces the observed $\Delta Y$ by its regression $\pi \bar{Q}^*$. That form and
$P_n[\Delta Y + (1 - \Delta)\bar{Q}_\gamma(A, W)]$ estimate one functional. Only the regression
form passes through the reported estimate at $\gamma = 0$, because targeting solves the weighted
residual equation and not the unweighted one.

`tipping_gamma(use_ci=True)` follows a confidence limit, so it refuses every row but the first.
`tipping_interval_refusal` gives the sentence, and the argument-aware capability row quotes it.
The default point search answers for every row.

`tests/unit/test_missingness_tilt_part_g.py` checks each row against the reported estimate at
$\gamma = 0$ to `rtol=1e-12`, and against a hand computation at $\gamma = 1$. Two mutations move
the curve: a mechanism that tilts the observed rows too, and a mechanism that tilts no row. A
two-arm saturated MSM matches the arm-mean contrast. An MSM with an effect modifier matches a
stacked weighted least squares.

`fit_wide_tilt_refusal` holds every refusal that no argument lifts. `missingness_tilt()` and
`tipping_gamma()` raise its sentence first, and both capability rows quote it. The table is
ordered, and each rule assumes that the rules above it passed.

| order | rule | refuses | row status |
| --- | --- | --- | --- |
| 1 | `longitudinal` | a longitudinal result | `unavailable` |
| 2 | `learned_rule` | a learned-rule fit, which refuses missing outcomes before any learner. [F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) holds that refusal | `unavailable` |
| 3 | `missing_outcome` | a fit with no missing outcome | `not_applicable` |
| 4 | `continuous` | a continuous dose | `unavailable` |
| 5 | `incremental` | an incremental fit | `unavailable` |
| 6 | `tiltable_parameters` | a fit that reports no parameter of the first table | `unavailable` |

On a composite fit with a missing treatment and a missing outcome, the tilt re-mixes the targeted
regression with the response mechanism $\pi(a, W) = P(\Delta = 1 \mid A = a, \Delta_A = 1, W)$.
Both are defined there, so the formula does not change. The tilt holds the treatment-missingness
condition fixed: it moves the outcome's recording away from missing at random and leaves the
treatment's recording at it.

A bare `tipping_gamma()` on a fit with several tiltable parameters reads `deferred` and requires
`estimand`. `tests/unit/test_attributable_mar_capabilities.py` checks the natural-course rows.

The omitted-variable bound also answers on a fit with a response mechanism, with a joint strength
over the treatment and response mechanisms
([second mechanisms and strata](#omitted-variable-bounds-robustness-value-benchmark-and-contours)). The two analyses
answer different questions. The tilt moves the recording of the outcome away from missing at
random. The bound measures an omitted confounder. `TestTheTiltRowsReadTheCallsPredicate` in
`tests/unit/test_capability_row_predicates.py` checks the tilt rows.

The curve is a point-estimate sweep, so a selector-based C-TMLE fit receives all of it. Only the
three spread columns read an interval that those paths refuse. Such a fit therefore receives
`plugin_std_err`, `plugin_interval_lower`, and `plugin_interval_upper` in place of `std_err`,
`ci_lower`, and `ci_upper`, which is the swap the truncation curve above already makes.
`tipping_gamma()` keeps its default point-estimate search on that fit. It refuses `use_ci=True`,
because it returns one float and a float carries no column name.
[Collaborative TMLE](collaborative-tmle.md) states the refusal these columns come from.

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
and the paragraph below states its boundary. Each operation prints a statistic against a stated
reference. It prints `passed` only where the evidence registry covers the fit.

| refuter | what it does | statistic and reference | what it tests |
| --- | --- | --- | --- |
| `placebo` | permutes the treatment column and refits. A dose whose declared support is a function moves each dose's position within its band instead (below) | one-sample t of the deviations from the exact null, on R - 1 degrees of freedom | the pipeline, not the data |
| `random_common_cause` | adds an irrelevant covariate and refits | the paired shift against the original fit, in standard errors | how far the learners move when a noise column joins the adjustment set |
| `subset` | refits on random subsamples | the scatter ratio rho against `SE * sqrt((n/m - 1) * n/(n - 1))`; `(R - 1) * rho^2` follows a chi-square law on R - 1 degrees of freedom | the reported standard error is the right size |
| `negative_control_outcome` | refits on an outcome the treatment cannot affect | z against the exact null | the design, under the control assumptions the paragraph below states |
| `dummy_outcome` | draws an outcome independent of `W` and `A` and refits | bias, standard-error ratio, and coverage against equivalence bands | outcome replacement and the full estimator pipeline |
| `simulated_outcome` | draws `Y` from a declared mean `m(a, W)` and refits | the same three readings, against the exact truth of the target | adjustment and treatment terms in the full pipeline |
| `plasmode` | resamples units, draws `A` from a frozen treatment model, draws `Y` from `m(a, W)`, and refits | bias, standard-error ratio, and coverage against the same bands | operating characteristics on a law fitted to resemble the data |
| `bootstrap_measurement_error` | bootstraps, perturbs declared adjustment variables, and refits; each copy of a unit keeps the unit's original fold | how far the estimate moves from the original; no verdict | stability under the declared measurement error |
| `bootstrap_stability` | bootstraps units that keep their original folds and seeds, and refits | the spread, and the shift of the mean against the original fit; no verdict | how far the estimate moves under resampling |

The placebo reads an MSM coefficient of a point treatment against its value under the placebo
law. The table names the two cases. The code reads the case from the realised design and weight
arrays.

| case | reference | reading |
| --- | --- | --- |
| identity link, and the coefficient is 0 for every function of the covariates | exactly 0 | the `t` rule |
| any other coefficient, weight, or link | the projection of the fitted regression `sum_a g(a given W) Qbar(a, W)` | descriptive, with no verdict |

A product weight and a tensor design whose arm factor holds an intercept give the first case. The
`TestMSMPlacebo` witnesses in `tests/unit/test_refutation_gate.py` enumerate a finite law for each
case.

**The band-rank placebo.** A declared support `Support(lower, upper)` with a callable bound gives
each unit its own band `[L(h), U(h)]`. A plain permutation ignores the receiving unit's band, so
`refute()` draws each dose's position within its own band and maps it into the receiving unit's
band. The refit then meets the same declared support as the fit. Source:
[`validation/_band_placebo.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/_band_placebo.py).

| band on the rows that record a dose | position `u` | placebo dose |
| --- | --- | --- |
| both bounds finite | `(A - L) / (U - L)` | `L' + u (U' - L')` |
| only the lower bound finite | `A - L` | `L' + u` |
| only the upper bound finite | `U - A` | `U' - u` |
| zero width | none: the row keeps its bound and gives no position | `L'` |
| a declared atom | the atom | the atom |

The primed bounds are the receiving unit's, read on the frame that the refit reads. A point dose
reads its covariates. A vector component reads the covariates, the donor's categorical components,
and the placebo doses of the components before it.

The positions are a permutation (or an i.i.d.
draw with `placebo_mode="draw"`) of the position column. They are independent of every
non-treatment variable, and the placebo dose is a function of the unit's own frame and its
position. So `E[Y | A', W] = E[Y | W]` on the pooled placebo law, and every policy contrast is
exactly 0. A support with constant bounds keeps the plain permutation, bit for bit.

The placebo dose depends on the covariates through the band, so the reading adds one sentence. A
rejection can also show a treatment model that cannot follow the band. `refute()` draws every
placebo data set first and runs the refit's own declaration checks on each. A draw that a check
refuses stops the call before any refit, with the check's reason code. The support check also runs
with every row at the smallest and the largest pooled position. The banded placebo reads
descriptively until study SJ of [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator)
registers its rows.

| case | `refute()` | reason code |
| --- | --- | --- |
| a bound finite on some rows and infinite on others, or both bounds infinite on every row | refuses | `implementation_gap` |
| a declared atom outside the band of a row that records a dose | refuses | `implementation_gap` |
| an atom or a zero-width band on a banded vector component | refuses | `implementation_gap` |
| a callable bound that is not a row-wise function of its frame, such as a closure over the observed dose | refuses | `invalid_declaration` |
| a policy that leaves the band on a placebo draw, or at a pooled extreme position | refuses, naming the draw | the refusal of the check |

`tests/unit/test_band_placebo.py` enumerates all 720 permutations of a six-row law whose band
reads `W` in its location and its width. The pooled law gives `E[Y | A', W] = E[Y | W]` within
`1e-12`. Its mutation controls are a draw that reads `Y`, and the permutation restricted to
in-band raw doses, which keeps a row's own dose with probability above `1 / n`.

The plasmode follows Franklin et al. (2014). Each draw takes `n` units from the empirical law of
`W`. It draws `A` from the known mechanism of the fit, or from a logistic model fitted once.

A model with more than two arms is multinomial. A two-arm model is clipped to the `g` bounds of
the fit, or to `[0.01, 0.99]` when the fit records none. A multi-arm model floors each
probability at the lower bound and renormalizes each row. The report prints the smallest arm
probability.

The plasmode draws `Y` from the `simulated_outcome` declaration, with the covariate term
standardized on the original data. The truth is a finite sum over the `n` support points, so all
three readings have an exact reference.

The plasmode draws each observation process of the fit from a frozen logistic model with the
same clip. The table lists the models.

| fit structure | plasmode draw | truth |
| --- | --- | --- |
| missing outcome | `Delta*` given `(W, A*)` | unchanged: the outcome is missing at random given `(W, A)` |
| missing treatment | the recording indicator given `W`. The treatment model is fitted on the recorded rows | unchanged |
| intermediate | `Z*` given `(W, A*)` | the contrast at the declared value of `Z` |
| clusters | whole clusters with replacement. Each drawn copy gets its own cluster id | unchanged: each support point has expected count 1 in a draw |
| baseline strata | each unit keeps its stratum | the sum over the support points of the stratum |

A Gaussian process with `cluster_sd > 0` adds one mean-zero intercept to each drawn cluster, so
no truth changes. The plasmode refuses weights, because rows drawn from the unweighted
empirical law that keep their weights describe a mixed law. It also refuses a fit with two of the
three observation processes, because each frozen model reads `(W, A)` only.
`tests/unit/test_plasmode_refutation.py` holds the witness and a mutation that keeps `W` fixed.

#### Generated point outcomes

A point process declares a mean `m(a, W)` and an arm term `c(a)`. The scalar `effect` gives
`c(a) = effect * code(a)`. The mapping `arm_effects` gives each arm label its own term. `f(W)` is
the standardized covariate sum over the square root of the column count, times
`adjustment_scale`.

| process | fit family | mean `m(a, W)` | draw |
| --- | --- | --- | --- |
| `GaussianIndependentOutcome` | Gaussian, `q_bounds=None` | the noise mean | Gaussian |
| `GaussianAdjustmentOutcome` | Gaussian, `q_bounds=None` | `f(W) + c(a)` plus the noise mean | Gaussian |
| `BernoulliAdjustmentOutcome`, `link="logit"` | binomial | `expit(intercept + f(W) + c(a))` | Bernoulli |
| `BernoulliAdjustmentOutcome`, `link="identity"` | binomial | `intercept + adjustment_scale * (expit(s(W)) - 1/2) + c(a)` | Bernoulli |
| `BoundedAdjustmentOutcome` | Gaussian with `q_bounds=(lo, hi)` | `lo + (hi - lo) * u(a, W)`, with `u` the Bernoulli mean | `lo + (hi - lo) * Beta(u * phi, (1 - u) * phi)` |

`s(W)` is the standardized sum with scale 1, and `phi` is `precision`. The Beta law has mean `u`
exactly. `dummy_outcome` takes the Gaussian independent process, or a Bernoulli or bounded process
with `effect=0` and `adjustment_scale=0`. A declaration whose mean leaves `[0, 1]` on a row is
refused before any refit. A bounded mean must lie strictly inside `q_bounds`.

The truth of each target is a finite sum of `m` over rows. The held design sums over the fit's
rows with the fit's weights. The plasmode sums over the support points, and it weights a row of
the ATT or ATC population by the frozen probability of that population's arm.

| target | truth | held-design readings |
| --- | --- | --- |
| `ate`, for each arm pair | the mean of `m(v, W) - m(r, W)` over every row | three readings when `m(v, W) - m(r, W)` is constant over the rows; else bias only |
| `att`, `atc` | the same mean over the rows of arm `v` (`att`) or arm `r` (`atc`) | the same rule |
| `ey`, `ey1`, `ey0` | the mean of `m(a, W)` | three readings when `m(a, W)` is constant over the rows; else bias only |
| `rr`, `or` | the ratio of the two levels, or of their odds | three readings when both levels are constant over the rows; else bias only. The readings use the log scale |
| `ey_obs` | the mean of `m(A, W)` at each row's own arm | three readings when `m(A, W)` is constant over the rows; else bias only |
| `par` | `ey_obs` minus the level of the reference arm `r` | three readings when `m(A, W) - m(r, W)` is constant over the rows; else bias only |
| `paf` | `par` divided by `ey_obs` | three readings when `m(A, W)` and `m(r, W)` are constant over the rows; else bias only |
| `ey_regime` | the mean of `sum_a q(a \| W) m(a, W)`, with `q` the declared regime | three readings when that sum is constant over the rows; else bias only |
| `ate_regime` | `ey_regime` of the value regime minus that of the reference regime | three readings when the difference of the two sums is constant over the rows; else bias only |
| `ey_ipsi`, `ate_ipsi` | as `ey_regime` and `ate_regime`, with `q` the tilt of the true mechanism | refused. The plasmode reads them |
| `ey_policy`, `ate_policy` | as `ey_regime` and `ate_regime`, with `q` the policy applied to the natural treatment | three readings when the sum, or the difference of the two sums, is constant over the rows; else bias only |
| `msm` | one coefficient of the projection of `m(a, W)` onto the working model | three readings when the coefficient's covariate part is constant over the rows; else bias only |

The code reads the constancy from the generated law. A Gaussian process and an identity link have
a constant arm contrast. A logit link with a nonzero effect does not. The plasmode reads all three
readings for every target.

A ratio reads `log psi` with the fit's log-scale standard error against
`log T`. A refit with a ratio that is not positive is a retained failure. The six exact registry
rows cover every point target, so no study is needed.

The held-design truth of `ey_obs`, `par` and `paf` reads the held arm of each row. The plasmode
truth reads the frozen arm probabilities. A stratum target, such as `ate[S=1]`, sums over the
rows of its stratum only.

The intervention targets need the law `q(a | W)` of the arms at each row. The table gives its
source.

| intervention | `q(a \| W)` | held design | plasmode |
| --- | --- | --- | --- |
| `Static`, `Rule`, known `Stochastic` regime | the declared density, evaluated on the fitted rows | yes | yes, on the support points |
| `Incremental` tilt with factors `f` | `f_a g(a \| W) / sum_b f_b g(b \| W)` (Kennedy 2019) | no, because the held design does not know the true `g` | yes, with the frozen treatment model as `g` |
| modified treatment policy `d` on arms | `sum_s P(A = s \| W) T(s -> a \| W)`, with `T` the probability that `d` moves arm `s` to arm `a` (Díaz et al. 2023, Section 4) | yes, with the held arm of each row as `A` | yes, with the frozen treatment model as `P(A = s \| W)` |
| modified treatment policy `d` on a continuous dose | the branch probability on the dose `d(A, W)` of each branch | yes, with the held dose of each row as `A` | no, because the plasmode has no frozen law of a dose |

An MSM coefficient solves the working-model normal equations of `cleverly.msm` with `m(a, W)` in
place of the outcome regression. `Projection` in `validation/_point_generated.py` solves them by
its own Gauss-Newton iteration for the identity, log and logit links. The covariate part of a
coefficient is `e_j' M^-1 sum_a h m' phi (m(a, W) - m(a, V; beta))`. Under a Gaussian process and
the linear model `MSM.linear()`, the slope's part is 0 at every row, so the slope reads
all three. The intercept's part is `f(W)` minus its mean, so the intercept reads bias only.

A shift `a + delta` under a Gaussian process has the contrast `effect * delta` at every row, so
`ate_policy` against the natural course reads all three. A cap that keeps some rows at their own
dose makes the contrast vary over the rows, so the target reads bias only.
`tests/unit/test_generated_policies.py` checks both cases and a randomized policy.

The refit draws a new outcome, so a policy that reads the outcome law cannot apply. The table
gives the policy requests that are refused, and the roadmap id of each.

| request | refusal | roadmap |
| --- | --- | --- |
| a policy target on a fit with a missing treatment, in the held design | the held design reads the held arm as the natural arm. An unrecorded row has none | [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) |
| a cap that a tuner such as `DoseQuantile` resolves | the truth applies the declared map. A tuned cap has no map until a fit resolves it | [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) |
| `arm_effects` on a continuous treatment | a dose has no arms. The scalar `effect` gives the term `effect * dose` | none. Declare `effect` |
| a risk-ratio tilt | the tilt reads the true mechanism, so only a plasmode can read it. That plasmode is not built | [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) part E (c) |
| a learned rule | no generated truth | [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) |

The held design keeps every observation indicator and the intermediate `Z`. The draw replaces
the outcome only, so a missing outcome stays missing at random. The table gives the rules for
each structure.

| structure | rule |
| --- | --- |
| controlled direct effect | each process adds `intermediate_effect * z` inside its link, with no arm interaction. The truth sets `z` to the declared value of the effect. A Gaussian contrast is then `c(v) - c(r)` |
| missing treatment | an unrecorded row takes the generated outcome of the first recorded row. A Gaussian fit scales the outcome by its range over every row, so the copy keeps that range a function of the recorded rows |

`TestUnrecordedRows` in `tests/unit/test_generated_observation.py` shows that `tmle`, `drtmle`
and `adaptive_iptw` give a bitwise equal estimate when these rows take other values inside the
range. A value outside the range moves the `tmle` estimate.

A held-design `att` or `atc` on a fit
with a missing treatment is refused. An unrecorded row's arm is unknown, so the population is
unknown. A continuous treatment has no arm target and is refused.
`tests/unit/test_generated_point_families.py` checks each truth against a closed form. It also
holds a mutation for the contrast direction, the population, the weights and the ratio scale.
`tests/unit/test_generated_observation.py` does the same for the natural-course targets, the
intermediate value and the stratum mask. Its mutations drop the stratum mask, set the wrong
intermediate value, and hide the clusters from the refit.

The default tests are `placebo`, `random_common_cause`, and `subset`. Each runs 20 replications by
default, so `refute()` costs about 60 fits. A check that needs more than 20 refits is opt-in.
The generated-outcome tests default to `DEFAULT_OUTCOME_REPLICATES` (400) draws. Pass them in
`tests=` to run them. `run_all(include_refits=True)` runs `refute()` with the default tests.

#### Adaptive-propensity IPTW refutation

`refute()` runs every point test on an `AdaptiveIPTW` fit. The estimator fits no outcome
regression, so three readings differ from a TMLE fit.

| reading | on `AdaptiveIPTW` |
| --- | --- |
| `placebo` with a declared mechanism | the refit takes the constant arm shares as its mechanism. In sample, the targeted and one-step forms then equal the arm-mean contrast of the permuted data |
| `random_common_cause` with a declared mechanism and complete data | no fitted function reads the noise column, so the paired shift is exactly 0 |
| `negative_control_outcome` | the control refits with the same estimator and no outcome bounds, because the estimator scales no outcome |

A fit with an estimated mechanism reports no interval, and takes the status
`estimated_mechanism_plugin`. Its readings use `plugin_std_error`, a diagnostic. The subset
detail names that spread. The generated-outcome tests and the plasmode then print the bias
reading only. This holds for every status that withholds an interval, so a C-TMLE `oat` fit
also reads bias only.

The placebo witness enumerates all 252 assignments of 5 treated among 10 units. In sample, each
refit equals the arm-mean contrast to `1e-12`, and the mean over the assignments is 0 to `1e-12`.
Cross-fitted, the arm labels swap to the complement assignment and negate the ATE. A mutation
that keeps two rows at their observed arm moves the mean. With weights, a refit equals the
weighted arm-mean contrast, and the mean is not exactly 0, because a weighted mean is a ratio.
`tests/unit/test_adaptive_iptw_refute.py` holds these witnesses.

The registry axis `method` reads `AdaptiveIPTW`, or `AdaptiveIPTW[one_step]` for the one-step
form. The learner axis fingerprints the reduced-regression learner. A row for these fits waits on
the follow-up study SI, so every study-backed reading prints descriptively until then.

#### Longitudinal refutation

A longitudinal result stores an unfitted copy of its estimator in `result.estimator`. The copy
fits no bootstrap and no simultaneous band. `refute()` refits it with the original fold labels and
repeat seeds. The `refute` row reads the `rerun_configuration` slot of `replayability()`. That slot needs
the stored copy and the regimen and MSM declarations, and it does not need seeded learners. The
table lists the tests that run on `LTMLE`, `SDR`, `IPSI`, the baselines,
and the survival, competing-risk, repeated-outcome, and MSM fits.

A test prints a verdict only
on a fit inside a calibrated scope. Every other test prints its numbers with `passed=None`. The
[longitudinal refutation study](method-evidence/refuter-calibration-longitudinal.md) calibrates
the placebo, subset and negative-control rules on seven configurations.

| test | longitudinal draw | reference |
| --- | --- | --- |
| `placebo` | a Markov treatment trajectory per unit, masked by the observed at-risk pattern | 0 for a difference, 1 for a ratio, 0 for an MSM coefficient whose operator row vanishes |
| `random_common_cause` | one baseline noise column; with `noise_nodes=True`, also one noise column per node | the paired shift against the original fit |
| `subset` | units drawn without replacement, with their original fold labels | `SE * sqrt((n/m - 1) * n/(n - 1))` |
| `bootstrap_stability` | units drawn with replacement; each copy keeps the unit's fold | no reference |
| `negative_control_outcome` | the control in place of the outcome. The refit keeps the original split and the fit seed | 0 for a blip coefficient and for a regimen contrast. A level refuses the control |
| `negative_control_outcome` | the control in place of the outcome, with the original folds and seeds | the no-effect value |
| `future_treatment` | a point-treatment TMLE of a control on the treatment at node `t + 1` | 0 |
| `dummy_outcome`, `simulated_outcome` | a generated outcome on the held history | the sample-conditional truth |
| `plasmode` | resampled baselines, and a history drawn from frozen node models | the g-formula value of the frozen law |
| `bootstrap_measurement_error` | units drawn with replacement; each copy keeps the unit's fold. Each declared column takes noise at every node where it appears | no reference |

The placebo does not permute trajectories. A permuted trajectory gives a unit at risk the missing
treatment of a unit that left earlier. A permutation within each follow-up pattern makes the
treatment depend on the event process. `refute()` refuses `placebo_mode="permute"` and
`placebo_mode="draw"` on a longitudinal fit for this reason.

The placebo fits a covariate-free first-order kernel `K` to the observed treatments. It draws a
complete trajectory for each unit from `K`, independently of the panel, and then applies the mask.
So the placebo law factorizes as `P(O-) x K`, where `O-` is every non-treatment variable. Every
policy then has the value `E[Q_1(W)]`, and every policy contrast is exactly 0.

| node | draw |
| --- | --- |
| 1 | a node-1 treatment of a unit at risk |
| `t >= 2`, categorical previous node | the node-`t` treatment of a donor with the same previous treatment; the node-`t` marginal when no donor has it |
| `t >= 2`, continuous previous node | the node-`t` treatment of the donor with the nearest previous dose |
| a node whose declared support is a function | the within-band position of one of the 5 donors with the nearest previous value, mapped into the unit's band on the placebo history |
| held decision | the node-1 draw |

The exact null needs a positive kernel probability for each regimen. When `K` gives a static
regimen probability 0, the placebo prints that reason and runs no refit. A treatment that is
constant within each cluster draws one trajectory per cluster. A node that
the fit declares known keeps a known mechanism, which is the law of `K` given the drawn previous
treatment. A vector node with a continuous component has no kernel, and the placebo refuses it.
A parametric g-formula fit refuses `noise_nodes=True`, because the forward simulation would need a
law for the new covariates.

An MSM coefficient has an exact placebo value under the identity link only. The coefficient is 0
when its row of `M^-1 sum_c h_c phi_c` vanishes for each unit and horizon. Any other coefficient
prints its numbers against no reference.

`tests/unit/test_longitudinal_refutation.py` enumerates the placebo law of a two-node survival
panel and finds each contrast equal to 0 within `1e-12`. Its mutation control fits the kernel
within each follow-up pattern, and the contrast moves by more than `1e-3`.

A node whose declared support is a function takes the band-rank placebo of the point section. The
kernel runs on the within-band positions at that node. Each node's band is read on the placebo
policy frame `H'_t`, which holds the placebo doses of the earlier nodes. So a band that reads
`A_{t-1}` stays consistent.

Given the data, the drawn positions are independent of `O-`, and the
placebo dose at node `t` is a function of the history and the drawn positions, so the null holds
by induction. A cluster draws one position. The cluster's band is the intersection of its rows'
bands, read on the placebo frames. `refute()` refuses these cases before any refit.

| case | reason code |
| --- | --- |
| a held node with a callable band | `implementation_gap` |
| a declared atom at a banded node | `implementation_gap` |
| a cluster whose rows' bands share no dose on a placebo history | `implementation_gap` |
| a band that a placebo history makes inverted, `NaN`, or of another form | `invalid_declaration` |

`tests/unit/test_band_placebo.py` enumerates the placebo law of a two-node panel with censoring
and a node-2 band that reads `L_2` and `A_1`. The node-2 regression is free of the placebo
treatment within `1e-12`. Its mutation control maps node 2 on the observed `A_1`. The doses then
leave the band that the refit reads, and the regression moves by more than 0.1.

#### The longitudinal plasmode

The plasmode resamples the baseline rows of the fit. It draws the rest of each history from models
that it fits on the panel once and then holds fixed. The table gives the model of each node.

| variable | frozen model | design |
| --- | --- | --- |
| a time-varying column with values 0 and 1 | logistic | `W`, the earlier columns, and `A_1` to `A_{t-1}` |
| any other time-varying column | Gaussian linear, with the residual standard deviation | the same |
| `A_t` | logistic, clipped to the fit's `g` bounds | `W`, `L_1` to `L_t`, and `A_1` to `A_{t-1}` |
| censoring after node `t` | logistic | `W`, `L_1` to `L_t`, and `A_1` to `A_t` |
| outcome | `mu + f(W) + effect * (A_1 + ... + A_T) + history_effect * h(L) + e` | `h` is the root-count standardized sum of every time-varying column |
| `Y_t` of a repeated outcome | the outcome row, with `t` in place of `T` and `h` over the columns of nodes 1 to `t` | the drawn `Y_t` is also the copy that opens the block of node `t + 1` |
| event at node `t` of a survival or competing-risk fit | the declared hazard, with `history_effect * h_t` added to each log-odds | `h_t` is the same sum over the columns of nodes 1 to `t` |

Each model fits on the units present at its node. Such a unit is uncensored and has no event
before the node. The draw orders a node as `L_t`, `A_t`, `C_t` and then the event, as the
container does.

The truth is the sequential g-formula of the frozen law under the regimen, with no censoring.
The table gives the two forms of the truth.

| law | truth |
| --- | --- |
| every time-varying column is binary, and at most 4,096 branches per support point | enumerates every branch of every history, so it is exact |
| any other law | averages `N` simulated histories per support point |

A pilot of 64 histories sizes `N` so that the Monte Carlo standard error is at most 2% of the
fit's standard error, with `N` at most 2,048. The report prints `N` and that standard error. The
bias and coverage readings add that standard error to each interval in quadrature. The plasmode
resamples `W`, so every target reads all three readings.

A survival truth at horizon `h` is the mean of `sum_{t <= h} S(t - 1) p_j(t)` over the histories,
where `S` is the probability of no event through a node. The truth carries `S` on each history,
so only `L` and `A` branch.

Only the plasmode reads `history_effect`. The held design keeps `L`, whose law under a regimen
it does not know, so `simulated_outcome` refuses a nonzero `history_effect`. A point fit also
refuses it.

`tests/unit/test_longitudinal_plasmode.py` works the g-formula by hand on a two-node panel with a
binary `L_2` that reads `A_1`. It checks a static regimen, a rule, and a stochastic node against
the enumerated truth to `1e-12`. Its mutation control holds `L_2` at its law under the observed
`A_1`, and that naive truth differs by more than 0.05. A static contrast and a rule pass at
R = 400 on that panel, and a contrast passes on a panel with a Gaussian `L_2`.

Two tests in the same file lower the limits. Past the branch limit, the truth samples. At a draw
cap of 4, the bias interval widens.

`tests/unit/test_repeated_plasmode.py` checks the Monte Carlo truth of `Y_2` against its
closed form within four standard errors. In that panel, `L_2` is Gaussian linear in `W`, `A_1`
and `Y_1`. The naive truth holds `Y_1` at its observed value and differs by more than 0.02. A
repeated contrast passes at R = 400.

`tests/unit/test_survival_plasmode.py` works the same g-formula for a survival hazard that reads
`L_2`. It checks static regimens, a rule, each horizon, and both causes of a competing-risk fit to
`1e-12`. The naive truth differs by more than 0.01. A survival contrast passes at R = 400 with
n = 1,500. The mean of 40 competing-risk refits lies within four standard errors of the truth.

The table gives the panels that the plasmode refuses, with roadmap id
[X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator).

| panel | reason |
| --- | --- |
| a held baseline decision, or a node with no treatment decision | the draw draws one treatment at each node |
| a declared known mechanism | the frozen model replaces the declared one |
| a clustered or weighted panel | resampled rows do not carry the cluster or weight law |
| a regimen with a tuned cap or an odds tilt | the truth has no frozen map for the node |
| a treatment node with levels other than 0 and 1 | the outcome adds the effect once per treated node |

#### Generated outcomes on the held history

The draw keeps `W`, `L`, `A` and `C` of each unit and replaces the outcome. The outcome at node
`t` is `f(W) + effect * (A_1 + ... + A_t) + e_t`, with fresh Gaussian noise `e_t`. The sum runs
over the decision nodes, so a held design adds the effect once. `dummy_outcome` sets `f` and
`effect` to 0. An end-of-study outcome is the node-`T` value. A repeated outcome draws every node,
and its copies in the node histories keep the original values.

The outcome reads no `L`, so each static regimen `a` has an exact level. That level is the
weighted sample mean of `f(W)`, plus the noise mean, plus `effect` times the sum of `a` up to the
horizon. A difference contrast is the difference of two levels. The independent process gives
every policy contrast the value 0. The held censoring reads `L`, and the generated outcome does not,
so the censoring stays at random.

| target | `dummy_outcome` | `simulated_outcome` |
| --- | --- | --- |
| a static-regimen difference contrast | bias, standard-error ratio, coverage | bias, standard-error ratio, coverage |
| a static-regimen level | bias, standard-error ratio, coverage | bias |
| a dynamic, MTP, stochastic or incremental policy | bias, standard-error ratio, coverage | refused, `IMPLEMENTATION_GAP` |

A level of `simulated_outcome` has a covariate part `f(W) - mean f(W)` in its influence curve. With
`W` held, that part is fixed, so the standard-error ratio and the coverage have no held-design
reference (X82 plan 5.5.1). A contrast cancels `f(W)`. Coverage also needs a correct censoring
model, so the report prints the uncensored share at each node. A policy that reads `L` or the
natural treatment has an unknown truth when `L` is held.

A survival or competing-risk fit replaces its events, not one outcome. Its declarations are hazards.

| fit | declaration | event probability at node `t` for a unit at risk |
| --- | --- | --- |
| survival | `LogisticHazardOutcome(intercept, effect, adjustment_scale, history_effect)` | `expit(intercept + f(W) + effect * d_t)` |
| competing risks | `MultinomialHazardOutcome(intercepts, effects, adjustment_scale, history_effect)` | `exp(eta_j) / (1 + sum_k exp(eta_k))` for cause `j`, with `eta_j = intercepts[j] + f(W) + effects[j] * d_t` |

`d_t` is the number of treated decision nodes up to `t`. The cause probabilities and the
probability of no event sum to 1, which independent per-cause logistic hazards do not ensure.
`dummy_outcome` needs a hazard with every effect 0 and `adjustment_scale=0`. The default Gaussian
declarations are refused on these fits. The held design refuses a nonzero `history_effect`, which
only the plasmode reads.

The draw recodes the exit before it draws the events (X82 plan 5.5.2).

1. An original event at node `t` becomes censoring at `t`. The generated outcome is not observed
   at that node.
2. Original censoring stays.
3. While a unit is under observation, it has the generated event with the declared probability
   and fresh noise. After an event, its later nodes are missing.

The container orders a node as `L_t`, `A_t`, `C_t`, `Y_t`, and a unit censored at `t + 1` must
hold `L_{t+1}` and `A_{t+1}`. A unit with an original event at `t` holds neither, so the recode
censors it at `t`. The recoded censoring reads the original outcome. The generated event reads
`W`, the treatments and fresh noise, so the censoring stays at random given the recorded history.
The report prints the uncensored share after the recode at each node, because the recode censors
every original event.

The truth of a static regimen `a` at horizon `h` is the weighted sample mean of
`sum_{t <= h} S(t - 1) p_j(t)`, where `S` is the probability of no event through a node. A
hazard that reads `W` gives each risk and contrast a covariate part, so these read bias only. An
independent hazard gives every policy contrast the value 0. A fit that declares a known censoring
mechanism is refused, because the recode changes the censoring law.

`tests/unit/test_longitudinal_generated_events.py` enumerates a two-node law whose original event
reads `L_2`. The g-formula of the recoded and censored law equals the uncensored closed-form risk
within `1e-12`. Its mutation control lets the generated hazard read `L_2`, and the two values
differ by more than `1e-2`.

The registered `generated_outcome` rows cover point fits only. Their bands were measured there,
and sequential regressions, censoring weights and per-node targeting change the variance. So each
longitudinal reading prints its numbers with `passed=None` until a longitudinal study registers a
row.

On an end-of-study or repeated fit, the tests refuse a non-Gaussian outcome, declared `q_bounds`,
and a classifier outcome learner. A cross-fitted Gaussian panel must declare `q_bounds`, so these
tests run on a fit with `n_folds=1`. Every longitudinal fit refuses an MSM coefficient and a node
whose levels are not 0 and 1.
`tests/unit/test_longitudinal_generated_outcomes.py` checks each truth with a nonzero effect, and
checks that the mean of 40 refits lies within four Monte Carlo standard errors of the truth.

#### Negative controls

A negative-control outcome has the shape of the primary outcome. An end-of-study fit takes one
value per unit. A repeated-outcome fit takes one value per unit and node. The copies of the primary
outcome in the node histories do not change, so each adjustment set stays the same. A survival or
competing-risk fit refuses the control, because the event indicators define the risk sets.

A control fit is an auxiliary fit, so `refute()` sets its outcome bounds. A binary control takes
no bounds. A continuous control takes its observed range on a cross-fitted fit, or on a fit that
declares `q_bounds`. This is the usual TMLE choice (Gruber and van der Laan 2010). The report
prints the bounds. A continuous control on a fit with a classifier outcome learner uses the
pseudo-outcome learner of the fit.

`negative_control_kind="pre_treatment"` declares a control that precedes the first treatment. The
form applies to point and longitudinal fits. `refute()` refuses the control in two cases.

| refusal | reason |
| --- | --- |
| the control equals an adjustment column | the control fit drops a measured confounder |
| the cross-fitted residual variance share is below `negative_control_floor` | the estimate is 0 because the control has no free variation |

The adjustment set is the covariates of a point fit and the baseline of a longitudinal fit. The
share is the residual variance of a 2-fold cross-fitted linear regression of the control on the
adjustment set, over the control variance. The folds use a fixed seed, so the `refute` row and the
call compute the same share. The default floor is 0.01.

The `future_treatment` test reads a control `C` that the fit does not use. Under no unmeasured
confounding of `A_{t+1}`, the control is independent of `A_{t+1}` given the history `H_{t+1}`. That
history is `W`, the earlier treatments, `L_1` to `L_{t+1}`, and each earlier repeated outcome. The
test fits a point-treatment TMLE of `C` on `A_{t+1}`, adjusted for `H_{t+1}`, on the units at risk at
node `t + 1`. `future_treatment_node=t` selects the node, and the default is 1.

The test uses the outcome and treatment learners and the fold count of the fit. The bounds rule
and the learner rule of the negative-control outcome apply to the control.

The reading is valid only when `C` is measured before `A_{t+1}`, has no effect on `A_{t+1}`, and gets no feedback from `A_{t+1}`. The
code cannot check this condition, so the report states it and stays at `L0`. `refute()` refuses
these requests.

| request | reason |
| --- | --- |
| a point-treatment fit | no later treatment node exists |
| one decision, or a held design | no later treatment exists |
| `future_treatment_node` at the last node | no node `t + 1` exists |
| a control equal to a column of `H_{t+1}` | the fit adjusts for it by design |
| a control whose cross-fitted residual share on `H_{t+1}` is below `negative_control_floor` | it is nearly a function of the history, so it is associated with `A_{t+1}` by design |
| a control constant on the units at risk | its estimate is 0 by construction |
| a later node that is not binary | `IMPLEMENTATION_GAP`, roadmap X82 |

`tests/unit/test_longitudinal_negative_controls.py` holds the witnesses. An unmeasured `U` that
moves `A_2` and the control gives `|z| > 6`. The same control without that path gives `|z| < 3`. A
control that `L_2` drives reads 0 with `L_2` in the history. With `L_2` removed, the same
control gives `|z| > 6`.

#### Structural nested mean refutation

A `StructuralNestedMean` result declares the `structural` assessment family. It keeps a copy of
its settings, its one outer split, and its fit seed in `result.config.fit_seed`. An unseeded fit
draws that seed once and uses it for the split and for every ensemble that inherits a seed.
`refute()` runs the five tests below. The placebo, subset and negative-control rules print a
verdict on a fit inside the SN1 scope of the
[longitudinal refutation study](method-evidence/refuter-calibration-longitudinal.md). That scope
is the regimen contrast, intercept-only blips fitted with GLM nuisances, five folds, two nodes, and
$n \ge 1000$. Every other reading prints its numbers with `passed=None`.

| test | structural draw | reference |
| --- | --- | --- |
| `placebo` | a Markov treatment trajectory per unit, as for a longitudinal fit, with no mask | 0 for a blip coefficient and for a regimen contrast. A level has no reference and refuses the placebo |
| `random_common_cause` | one baseline noise column; with `noise_nodes=True`, also one per node. Each refit keeps the original split and the fit seed | the paired shift against the original fit |
| `subset` | units drawn without replacement, with their original fold labels | `SE * sqrt((n/m - 1) * n/(n - 1))`, with the plug-in SE of a fit that reports no interval |
| `bootstrap_stability` | units drawn with replacement; each copy keeps the unit's fold | no reference |
| `negative_control_outcome` | the control in place of the outcome. The refit keeps the original split and the fit seed | 0 for a blip coefficient and for a regimen contrast. A level refuses the control |

The placebo law factorizes as `P(O-) x K`. No covariate and not the outcome reads the drawn
treatment, so `E[Y | H_t, A_t]` is free of `A_t`. Every true blip is then 0, and `psi = 0` solves
the population g-estimating equation of each node. A regimen contrast is linear in the blips, so
it is 0 without a positivity condition on the regimen.

`tests/unit/test_structural_refutation.py`
runs the package g-estimation on an enumerated two-node law. The observed law gives the blips 0.74
and 0.6. The placebo law gives 0 within `1e-12`. Its mutation control lets the drawn node-2
treatment read the observed one, and the node-2 blip moves to 0.387.

A subset draw that leaves a fold with fewer than 2 units of a treatment level at some node is
refused before the first subset refit. The g-estimating equation of that node would have no
treatment variation in the fold. Raise `subset_fraction`, or drop `subset`.

The negative-control outcome follows the negative-control rules above, and the
pre-treatment form reads the baseline as its adjustment set. A control outside {0, 1} is a
regression target, which `regression_learner` fits. When that setting is `None` and the outcome
learner only classifies, or is a binomial `GLM`, `refute()` refuses the control before any refit.
The witness test fits a control that `A1` moves beyond its history, and its z exceeds 5. A
control independent of the treatment gives a z below 2.

The other operations of the family do not run. The table gives the reason of each.

| operation | status | reason |
| --- | --- | --- |
| `support`, `nuisance_models` | `unavailable`, `api_redirection` | the fit records its positivity and learner diagnostics in `result.diagnostics` |
| `score_equations`, `corrections`, `truncation_curve` | `not_applicable` | g-estimation solves its equations by a linear solve, uses no correction system, and applies no mechanism bound |
| omitted-variable analyses, `evalue` | `unavailable`, `implementation_gap` | no Riesz representer or risk-ratio scale of a blip is derived. [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) tracks the bound |
| `simulated_confounding` | `unavailable`, `implementation_gap` | [F13](../roadmap.md#f13-longitudinal-simulated-confounding-replay) |
| `missingness`, `tipping_gamma` | `not_applicable` | the estimator refuses missing outcomes and censoring |

The generated outcomes, the future-treatment control, and the measurement-error bootstrap of a
structural fit are refused with the roadmap id
[X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator).

#### Verdicts and evidence

A rule prints a verdict only when evidence backs it. `VERDICT_EVIDENCE` in
[`validation/_interpretation.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/_interpretation.py)
holds that evidence as frozen `VerdictEvidence` rows. Each row names an instrument, a rule, a
rung, the theory behind it, and a `CalibrationScope`. A fit outside the scope of every row gets the
numbers and `passed=None`. The report prints the configuration that the nearest row covers.

| rung | what backs the reading |
| --- | --- |
| `L0` | a description of what moved, with no reference law |
| `L1` | a stated reference law whose size or power has no measurement |
| `L2` | an exact reference law, or a measured size and power on the configuration of the fit |
| `L3` | an `L2` reading with a comparator or an external truth beside it |

A verdict needs `L2` or above. A `CalibrationScope` has these axes: method, estimand kind, outcome
family, treatment kind, learner fingerprints, cross-fitting, inference status, known mechanism, arm
count, bounds on n, covariates per row, minimum event count, and cluster count. An unset axis
covers every value.

`passed` has three values. `RefutationResult.passed` is `False` when any test failed. It is `None`
when any test is descriptive and none failed. It is `True` when every test passed. `bool(report)`
raises `TypeError` for `None`, so a descriptive report never reads as a pass.

| test | rule | rung today |
| --- | --- | --- |
| `dummy_outcome`, `simulated_outcome` | bias band | `L2` on a point fit, at 400 draws or more |
| `dummy_outcome`, `simulated_outcome` | standard-error ratio band and coverage band | `L2` on a point fit where the covariate part of the influence curve is zero, and at 400 draws or more. Not printed on a fit whose status withholds an interval |
| `dummy_outcome`, `simulated_outcome` on a longitudinal fit | all three bands | descriptive until a registered study adds a row |
| `placebo`, `subset` on a longitudinal fit | t, rho | `L2` on the seven configurations of the [longitudinal refutation study](method-evidence/refuter-calibration-longitudinal.md): `LTMLE`, `SDR`, `IPSI`, the three baselines and `StructuralNestedMean`, two nodes, GLM nuisances, $n \ge 1000$ |
| `negative_control_outcome` on a longitudinal fit | z | `L2` on the six end-of-study configurations of the same study, for a Gaussian control |
| `placebo` on a point fit | t | `L2` on the six configurations of the [point refutation study](method-evidence/refuter-calibration-point.md): `TMLE` with GLM nuisances on a Gaussian and a binary outcome, `TMLE` with a depth-2 booster, greedy `CTMLE`, and `DRTMLE` with a declared and an estimated mechanism. Each covers the ATE of two arms, five folds and $n \ge 2000$ |
| `subset`, `negative_control_outcome` on a point fit | rho, z | `L2` on the T1, T2 and D1 configurations of the same study: `TMLE` with GLM nuisances on either outcome, and `DRTMLE` with a declared mechanism |
| `placebo`, `subset`, `random_common_cause`, `negative_control_outcome`, `future_treatment` on any other fit | t, rho, paired shift, z | descriptive until a registered study adds a row |

The equivalence bands are `|bias| <= 0.20` standard deviations of the draws, a standard-error ratio
in [0.85, 1.15], and a coverage in [0.88, 0.99]. A reading passes when its 95% interval lies inside
its band. The bias interval is exact because the draws are i.i.d. given the held design. The
coverage interval is the Clopper-Pearson interval.

A budget below 400 draws still runs, and its
readings print at `L1`. A budget below 10 draws is refused, because an interval needs a spread.
At 400 draws and no bias, the three readings pass together with probability 0.93. At 100 draws the
probability is 0.003. `tests/unit/test_refutation_gate.py` checks the bands and the gate.

A `placebo` fit that declared a known treatment mechanism keeps one on the placebo law. The
placebo law draws treatment independently of the covariates, so the exact mechanism is the arm
shares of the placebo column, constant in `W`. A refit that re-estimated the mechanism would
refute another configuration of the estimator.

A `subset` refit keeps the original fold label of each drawn row, in the original repeats. The
refit redraws no fold. A subset of whole clusters keeps whole clusters on a clustered fit. The
reference counts sampling units, and the scatter reads the units that the draw kept.

A `random_common_cause` refit reuses the seed of the original fit, so the original fit is the
paired baseline. `seed_spread=True` adds refits that omit the noise column, to measure the spread
across seeds. That option doubles the refit count, so it is opt-in.

`random_common_cause` adds one column of independent noise, and the refit must accept that column.

| estimator | what the refit runs |
| --- | --- |
| `TMLE` | the estimator |
| `CTMLE` with an explicit `ordering=` | a copy whose ordering is the declared ordering, then each added covariate |
| `DRTMLE` with an `evaluation=` companion that lacks the added column | a copy without the companion |

An explicit ordering ranks covariates by their prior relevance. Independent noise has none, so the
noise goes last. `CausalData.with_extra_covariate` also appends the column last. The companion
enters no fit, fold, or score, so the refit reports the same estimate without it.
`tests/unit/test_drtmle_companion.py` checks that bit for bit. Neither copy changes the fitted
estimator. `TestAnAddedCovariateGoesAfterTheDeclaredOrdering` and
`TestARefitDropsACompanionThatLacksACovariate` in `tests/unit/test_capability_row_predicates.py`
check both cases.

`refute_refusal` in `validation/refute.py` holds every refusal of a request before any refit.
`refute()` first checks its argument values with `_validated_tests`. An unknown test name and an
invalid `n_replicates` raise `ValueError`. `result.diagnostics.refute()` runs the same check before
it reads its capability row. The call then raises the first refusal of this table as
`CapabilityError`.

| order | rule | refuses |
| --- | --- | --- |
| 1 | `estimator` | a result with no estimator |
| 2 | `estimand` | an estimand that the fit did not report |
| 3 | `no_effect_null` | `placebo` or `negative_control_outcome` on an outcome or policy mean, or a target with no registered null. `negative_control_outcome` also refuses an MSM coefficient. `placebo` on a point-treatment MSM coefficient refuses only a continuous dose grid |
| 4 | `row_set_under_plan` | `bootstrap_measurement_error` or `bootstrap_stability` on a fit given `split_plan=` |
| 5 | `generated_rule` | a generated-outcome test under bands that are not exactly `EquivalenceBands` |
| 6 | `generated_eligibility` | a generated-outcome test whose fit, estimand, or process has no derivation, such as a direct fit that records no identification |
| 7 | `measurement_declaration` | `bootstrap_measurement_error` without exactly a `BootstrapMeasurementError` declaration |
| 8 | `measurement_rule` | `bootstrap_measurement_error` under a rule that is not exactly `EmpiricalInclusionRule` |
| 9 | `measurement_eligibility` | a measurement-error declaration that the data of the fit cannot carry |
| 10 | `stability_eligibility` | `bootstrap_stability` on data that is not point-treatment `CausalData`, or with fewer than 2 draws |
| 11 | `measurement_budget` | a `bootstrap_measurement_error` draw count below the floor of its rule |
| 12 | `generated_budget` | a generated-outcome draw count below 10 |

On a learned-rule fit, `random_common_cause`, `subset`, `bootstrap_stability` and `bootstrap_measurement_error` hold the realized rule of each original row fixed.
A subset or a resample carries the rule of each kept row, so a duplicated row carries it twice.
The measurement-error draw perturbs the covariates after the resample, so each unit keeps the rule at its recorded covariates.
Each refit then estimates the value of the fit's own rules, as the [learned-rule replay](#simulated-common-cause-stress-surface) does.
The placebo refits as the fit did, because the placebo law gives every rule the value `E[Y]`.
The placebo reads the learned-rule level against the weighted mean of the outcome.
`tests/unit/test_refute_learned_rule.py` recomputes each refit by hand and relearns the rule as a mutation control.

Rules 6, 9 and 10 call the validators that the tests run, so the table copies no check. The argument
check also requires an outcome array for `negative_control_outcome`. It raises `ValueError` before
any refit when that array is absent.

The `refute` capability row reads the same table for each request. An omitted `tests` means the
default tests. When the default tests meet a refusal and a single default test runs, the row reads
`deferred` on `tests`, with the call's sentence. A request that names only tests that run lifts the
deferral. `TestTheRefuteRowResolvesEachRequest` in the same file checks both fits below.

A refutation also runs on a selector-path collaborative fit. The `placebo`, `subset`,
`random_common_cause`, and `negative_control_outcome` statistics scale by a standard error. Each
reads that scale through `plugin_std_error`, so the statistic makes no coverage claim. On a fit
that supplies no inference, each `RefutationTest` carries `inference`. `RefutationTest.to_frame()`
and `RefutationResult.draws_frame()` then name the standard-error column `plugin_std_error` in
place of `std_error`.

A fit that declared `split_plan=` refuses `bootstrap_measurement_error`. A supplied plan labels the
rows it was realized on, by position. A bootstrap draw holds the declared number of rows, so the
labels would land on resampled units. `refute()` raises `CapabilityError` before it refits
anything, and it names both the plan and the test. The `subset` operation runs under a supplied
plan, because each kept row carries its original label. The `placebo` and `random_common_cause`
operations replace a column and leave every row in its position. See
[reusable outer split plans](cv-tmle.md#reusable-outer-split-plans) for the table.

The two no-effect operations read the registered target through the reported parameter key.
Differences and the attributable fraction use zero. Risk ratios and odds ratios use one on their
reported scale. Outcome and policy means have no fixed no-effect value. An arbitrary MSM
coefficient has none either.

Both operations refuse these parameters before any refit. The `subset` and
`random_common_cause` operations keep their descriptive interpretations. All ratio statistics use
log ratios because their standard errors use the log scale. Reports keep the refit ratios on their
reported scale.

A ratio placebo or noise test reports the geometric mean of its refits. Its `values` remain the
individual reported ratios. A ratio subset reports log-scale scatter in `refuted_sd` and its
expectation. Its `values` also remain reported ratios.

The default tests include `placebo`, so `estimand="ey_obs"` defers on `tests`. A request with
`tests=("random_common_cause", "subset")` runs. `TestReportedScaleNoEffectNull` checks the
reported null, refusal, and capability row.

`refute()` draws its randomization from the seed of the fit, unless the caller passes
`random_state`. A fit that carries a seed gives the same refutation on every call. A fit that
carries no seed gives a different refutation on every call.

The report records the seed under `random_state`. Pass that value back to `refute()` to obtain
the report again. The seed governs the perturbations and the refits they feed, so it repeats
the report of a fit that carries no seed of its own. The seed applies to a copy of the
estimator, so a refutation never changes the fit it examines.

Each empirical draw derives a child seed from the recorded root seed. The perturbation and its
full refit use that child seed. `report.draws_frame(name)` reports every child seed, estimate,
standard error, family, and failure. The report stores each draw as an
`EmpiricalRefitRecord`.

Each generated-outcome draw derives a child seed from the recorded root seed. A refit that fails
stays in the report as the shared `ReplicationFailure` record. The readings need every draw, so a
failure leaves the test without a verdict. The operation never reports only the conditional
distribution of successful fits.

The draw budget has two thresholds. `DEFAULT_OUTCOME_REPLICATES` is 400 and sets the default
`n_replicates`. `EquivalenceBands().minimum_draws` is also 400, the calibrated minimum. The
operation refuses a budget below 10 before it refits anything.

Bootstrap measurement error reports the empirical probability of the original estimate under
`EmpiricalInclusionRule` and states no verdict. It draws an iid or
whole-cluster sample through the inference bootstrap design. It perturbs numeric variables with
mean-zero Gaussian noise after sampling. The noise scale is the declared multiplier times the
selected variable's bootstrap-sample standard deviation.

Each sampled cluster occurrence receives a distinct code before perturbation. Repeated draws of
one source cluster therefore remain separate during the refit.

A declared strata variable keeps its recorded values, so each reported stratum keeps its rows.
The operation perturbs the other declared variables, and the report names the held strata
variables. A declaration of strata variables only is refused.

A longitudinal result perturbs each declared column at every node where the column appears. A
baseline name moves once. A long-panel name such as `l` moves at each node, where the panel
names it `l@t`. The noise scale of each node is the multiplier times the column's standard
deviation at that node, over the recorded rows. A missing value stays missing. The report names
the nodes of each column.

A panel carries no categorical metadata, so the operation refuses a
column whose values are all 0 or 1. It also refuses an outcome of the panel, a constant column,
and a name that is not a baseline or node covariate. `tests/unit/test_measurement_error_panels.py`
holds the witnesses.

The categorical path recovers original logical levels from `CausalData.encodings`. Each changed
row draws uniformly from the other levels. The operation then rebuilds the complete drop-first
indicator block. Boolean covariates use the same two-level path.

The operation checks every condition in the table below before the first refit. Each row is one
`CapabilityError` branch of `_validate_measurement_error_eligibility` in
[`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py),
and each message names the variable it rejects.

| what the operation refuses | what it requires instead |
| --- | --- |
| a result whose data is neither `CausalData` nor `LongitudinalData` | a point-treatment or longitudinal result |
| `resampling="cluster"` on data that carries no cluster ids | declared cluster ids, or iid resampling |
| a declaration of strata variables only | at least one adjustment variable outside the strata, because the strata keep their recorded values |
| a generated indicator name | the original categorical variable that the indicator encodes |
| a name that is not an original adjustment variable | one of the original names the message lists |
| a categorical variable whose encoded block lost an indicator to duplicate-column removal | a retained indicator for every level the encoding generated |
| a selected variable whose values are all 0 or 1 and that carries no `CategoricalEncoding` | the same variable declared to `CausalData.from_frame` as a boolean or categorical column, because relative Gaussian noise makes an undeclared indicator real-valued |
| a selected numeric variable that is constant | a variable with nonzero spread, because a relative noise scale of zero leaves every draw unperturbed |

The report records the declaration, requested draw count, resolved mode, rule, estimates,
standard errors, child seeds, and failures.

**Why the bands are equivalence bands.** A significance test of zero bias fails an estimator that
is biased by a trivial amount once the draws are many, and it passes a biased estimator when the
draws are few. An equivalence band asks the question that the user needs answered. Does the
interval of the reading lie inside a stated tolerance? The 95% interval of the bias has a width
of about 0.20 standard deviations at 400 draws, which is why 400 is the calibrated minimum.

The first process catalog covers Gaussian outcomes and additive `ate`, `att`, or `atc` contrasts.
`refute()` and its `_validate_generated_eligibility` helper in
[`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py)
check every condition in the table before the operation refits anything. Each row is one
`CapabilityError` branch, and each message names what is missing.

| what the operation refuses | what it requires instead |
| --- | --- |
| bands of any other type | the exact registered `EquivalenceBands` declaration |
| a process declaration of any other type | the exact registered `GaussianIndependentOutcome` or `GaussianAdjustmentOutcome` |
| a process whose family is not `"gaussian"` | the Gaussian family, which is the one with an implemented effect derivation |
| a fit that carries no identification metadata | identification metadata on the result |
| a functional that is not `BackdoorMeanContrast` | a backdoor-identified additive mean contrast |
| a provider that is not `ExplicitAdjustmentProvider` | registered backdoor provider provenance |
| an original outcome family that is not `"gaussian"` | an original family equal to the process family, so a binomial fit is refused |
| an estimator configured with a family other than `"auto"` or `"gaussian"` | a configured family that accepts the declared process |
| a saved outcome learner with no regression-capable route | a regression-capable outcome learner |
| a longitudinal functional | a point-treatment functional |
| a treatment that is not binary | code one against code zero |
| an intermediate node or a controlled direct effect | no intermediate node |
| a fit with missing outcomes | complete outcomes |
| a modified-policy, risk-ratio-tilt or learned-rule axis | the `arm`, `regime`, `ipsi` or `msm` axis |
| an `ipsi` target in the held design | the plasmode |
| an MSM with a link other than identity, log or logit | one of those three links |
| a parameter key that is not a structured `ParameterKey` | a structured key for the selected estimand |
| a ratio estimand, or any other non-additive contrast | `ate`, `att`, or `atc` |
| disagreement between the functional target, the identified estimand, and the parameter key | one estimand named by all three |
| a missing registered identification artifact | the artifact `TARGETS` records for that target |
| an arm contrast that does not resolve to two distinct arms | a resolvable contrast of two distinct arms |
| a draw budget below 10 | `n_replicates` of at least 10 |

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
    cross_fit=False,
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
refutation. `cleverly` keeps each failed refit as a `ReplicationFailure` record and withholds the
verdict.

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

A study of a selector-path collaborative estimator measures the plug-in diagnostic. Each
replication reads `plugin_interval`, `plugin_std_error`, and the same p-value arithmetic that
`pvalue` refuses on that fit. The records carry the inference status, and the table changes as
follows.

| output | ordinary estimator | selector-path collaborative estimator |
| --- | --- | --- |
| `to_frame()` columns | `mean_std_error` | `inference` and `mean_plugin_std_error` |
| `summary()` | the table and the verdict | a line above the table that says the columns describe the plug-in diagnostic |
| `verdict()` with no finding | coverage and bias are consistent with a correctly working estimator | bias is consistent, and the coverage column certifies no confidence interval |

A clustered status depends on the draw, so the replicates of one study can take different
statuses. A cluster count that straddles 4 gives that mix. Every record measures the plug-in
numbers, and on an inferential fit those numbers equal `ci` and `std_error`. So
`summarize_replications` summarizes a mix under the status that `precedent_status` gives. The
table gives what the summary adds.

| output | a study whose replicates took more than one status |
| --- | --- |
| `EstimandSummary.status_counts` | each status with its replicate count. It is empty when every replicate took one status |
| `to_dict()` and `to_frame()` | a `mixed_statuses` column, for example `few_cluster_plugin in 4, influence_curve in 4` |
| `summary()` | the line above the table says that the estimator "supplies no interval on some replicates". One more line per mixed estimand names its statuses and counts |

`tests/unit/test_simulation_summary.py` runs a study whose cluster count straddles 4. It holds
the mixed witness, a mutation that labels the mix `influence_curve`, and a uniform control.

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

An estimator whose status supplies no inference is refused with `CapabilityError` before the
first fit. That includes a collaborative estimator at any `strategy`, and a `DRTMLE` with a
`guard` and varying weights declared estimated. It also includes a clustered `TMLE` or `DRTMLE`
under `"few_cluster_plugin"`. The procedure adjusts one p-value per candidate with
Benjamini and Hochberg, and those fits report no p-value. The check prepares each candidate's data
and asks the estimator's own `_inference_status()`, which is the status its estimates carry.

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
