# Check a fitted result

A fitted result can check itself: how well the outcome and treatment models predict, how extreme
the weights are, whether the correction step solved its equations, and how strong a missing
confounder would have to be to change the answer. `assess()` runs these checks together and prints
one report. This page reads that report, then the checks of the models and the weights behind it.

## Run the assessment battery

Call `assess()` to collect validation, diagnostics, and sensitivity in one report.

```python
battery = result.assess()
print(battery.summary())
support = battery.report("support")
```

The summary opens with counts: how many checks ran, how many need attention, and how many did not
run. `Results` gives one line for each analysis that ran without a pass or fail rule, such as the
E-value, with its main finding. `Checks` lists the checks that pass or fail, failures and warnings first.
`Not run` lists what was skipped and why, such as an analysis that needs an argument you have not
given. The notes list the suggested next steps. Print `battery.summary("full")` to see each row's
whole finding or reason.

Call `battery.to_frame()` for the complete row ledger. It carries the `surface`, `check`, `status`,
`detail`, and `next_steps` columns. Call `battery.next_steps()` for the follow-up actions as a
list.

```python
ledger = battery.to_frame()
deferred = ledger[ledger["status"] == "deferred"]
print(deferred[["surface", "check", "next_steps"]])

for step in battery.next_steps():
    print(step)
```

`battery.attention` contains failures and warnings. `battery.omissions` contains deferred,
not-applicable, and unavailable analyses. Use `battery.report(...)` to retrieve the detailed
payload from an operation that ran, and `battery.report(name, surface="diagnostics")` to retrieve a
diagnostic that also appears under validation.

Pass analyst choices through `arguments`. Opt in to each expensive work class by name.

```text
battery = result.assess(
    include_refits=True,
    include_retargets=True,
    random_state=21,
    arguments={
        "benchmark": {"covariates": ("W1", "W2")},
        "simulated_confounding": {"grid": grid},
    },
)
```

The common seed reaches `refute`, `benchmark`, and `simulated_confounding`. A seeded fit reuses
its fit seed when you omit this argument. An unseeded fit draws a seed and records it on the
returned row.

`assess()` refuses `arguments` for `score_equations`. The validation battery owns that name and
runs it argument-free. The battery presents the validation row for it, so it would compute your
answer and then hide it. Call `result.diagnostics.run_all(arguments=...)`, or call the operation
itself.

The battery also owns `support` and `nuisance_models`, but neither one accepts an argument. An
argument for either is a `TypeError` from the signature. An empty mapping applies nothing, so
`assess()` accepts it for all three names.

### What runs by default and what needs a flag

`assess()` with no flags reads what the fit stored and refits no model. It runs every summary of
stored numbers, such as the E-value, and every cheap re-estimate. Each check declares its cost in
`result.diagnostics.capability(name).execution` (or `result.sensitivity.capability(name)`):
`"summarize"` reads stored numbers, `"retarget"` reruns the correction step on the stored models,
and `"refit"` fits new models. `include_retargets=True` runs the retarget rows, such as the curve
over caps on extreme weights (`truncation_curve()`), `missingness()` and `tipping_gamma()`.
`include_refits=True` runs the refit rows: the refutation tests (`refute()`), `benchmark()`,
`simulated_confounding()`, which also needs a grid, and `marginal_bounds()`, which fits extra
regressions on the fit's folds without refitting the estimator.

The cost belongs to the fit, not to the name of the check. The weight-cap curve of a `DRTMLE` fit
with a `guard` refits its reduced regressions at every cap, and that of a longitudinal fit reruns
its whole backward recursion, so both ask for `include_refits=True`. An E-value that needs one
more outcome regression is a refit too. Read the class from the capability row rather than from a
list. On a longitudinal fit, an operation without a published derivation reads `unavailable`
rather than borrowing point-treatment formulas, and its reason says why.

To run every diagnostic, including the refits and the retargets, pass both flags:

```python
everything = result.diagnostics.run_all(include_refits=True, include_retargets=True)
```

Pass required choices with `arguments={"operation": {...}}`. The report row retains the effective
arguments and the returned object. Use `report.report(name)` to retrieve that object.

A refusal names the first thing that is wrong. An operation that needs a choice you did not pass
names that choice, and not a cost flag. No flag supplies the covariates for `benchmark`, so the
row says so under every flag.

### Rows that did not run

Each row of a combined report carries one status, and
[the status contract](../technical-reference/validation-methods.md#the-status-contract) names
every status. `passed` and `failed` belong to checks with a pass or fail rule, `completed` means an
analysis ran without one, and `warning` uses an existing diagnostic rule. `deferred` means the
same check can run once you supply a choice or pass a cost flag. The row keeps its next step and
the request you made, with the seed a combined run supplies, and stays in `omissions`. Pass that
mapping back to the direct call to run the row you deferred.

A check that answers for one estimate defers on a fit that reports several, and its next step
names `estimand`; pass the estimate's name through `arguments`. The status contract's subsection
"The ambiguous default estimand" lists those checks and their defaults. A sensitivity analysis
picks the estimate itself when the fit reports exactly one eligible estimate. `refute()` does not,
so on a fit whose one estimate is not named `ate` its row defers as well.

Four rows can also defer on another argument. The row defers when the call refuses the default
value and another value runs. It reads `unavailable` when no value runs. The row quotes the
sentence that the call raises.

| operation | argument | the row defers when |
| --- | --- | --- |
| `refute()` | `tests` | the fit refuses one default test and admits another, such as `placebo` on a fit of `NaturalCourseMean` |
| `truncation_curve()` | `mechanism` | the fit refuses the default axis and admits the other, such as the treatment axis of an incremental fit with missing outcomes |
| `benchmark()` | `covariates` | always, because the call needs the argument. A fit with one covariate reads `unavailable`, because the call cannot drop every covariate |
| `simulated_confounding()` | `estimand` | the call refuses the default estimand, and another estimand that the fit reports runs |

An expected refusal after invocation becomes `unavailable`. An estimand you name that the fit never
reported is such a refusal. The aggregate run then continues with other accepted diagnostics.
Direct calls still raise the precise refusal. `not_applicable` and `unavailable` also appear in
`omissions`.

## Diagnostics

```python
validation = result.validate()
support = result.diagnostics.support()
nuisance = result.diagnostics.nuisance_models()
scores = result.diagnostics.score_equations()
influence = result.diagnostics.influence()
corrections = (
    result.diagnostics.corrections()
    if result.diagnostics.capability("corrections").available
    else None
)
stability = result.diagnostics.truncation_curve()
all_cached = result.diagnostics.run_all()
```

`truncation_curve()` re-estimates at other limits on extreme weights. Print
`stability.summary()` to see each estimate's change from the fitted one and the share of units the
limits capped. Call `stability.to_frame()` for the dataframe.

`influence()` reports how few rows drive each estimate, read from how much each row moves it. It
gives the Kish count (an effective number of rows behind the standard error), the skewness, the
kurtosis, the relative standard deviation of the variance estimate, and the coverage loss of the
Wald interval that a skewness correction (the Edgeworth term) predicts. This operation is a direct
call. `run_all()` and the assessment battery do not include it.

The correction diagnostic is available only for a DR-TMLE fit whose guard actually subtracts a
correction term. Ordinary and collaborative TMLE report it as `not_applicable`.

### Check covariate balance

Weighting by the fitted treatment model should make the treated and untreated groups look alike
on the covariates you adjusted for. `covariate_balance(result)` measures how far it succeeded. For
each covariate it gives the standardized mean difference, the gap between the two groups' means
in standard deviations, before and after weighting. A gap near zero after weighting says the
weights balanced that covariate; one above 0.1, the usual threshold, says the treatment model
missed something about it. The check describes the data: it has no interval or test, and it says
nothing about covariates you did not adjust for.

```python
from cleverly.sensitivity import covariate_balance

balance = covariate_balance(result)
print(balance.summary())
fig = balance.plot()
```

When treatment is decided at several times, as in a longitudinal or time-to-event fit, the same
call returns a balance over time, a `BalanceOverTime`, following Jackson (2016). At each treatment
time it compares the covariates recorded so far across that treatment, among the people who had
the same earlier treatments. Only people still in the study at that time count, and on a survival
fit only those still free of the event. After weighting, each person counts as one over the
fitted chance of the treatments and the follow-up they had up to that time. The summary averages
each comparison over the groups of earlier treatments, each group counting by its size, and
`to_frame()` keeps every group. Leaving the study is checked as one more treatment.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean
from cleverly.datasets import make_longitudinal
from cleverly.sensitivity import covariate_balance

frame, _ = make_longitudinal(n=600, seed=0)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    ),
)
regimes = study.identify(RegimeMean({"always": 1, "never": 0})).estimate(
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    random_state=0,
)
over_time = covariate_balance(regimes)
print(over_time.summary())
per_time = over_time.average(over="history")
fig = over_time.plot(diagnostic="confounding", view="panels")
```

The feedback check compares a later covariate across an earlier treatment. After weighting, a
difference there means the treatment changed the covariate; before weighting it also mixes in
confounding. Weighting is not meant to remove it. When that
covariate also affects later treatment, adjusting for it in an ordinary regression gives a biased
effect, which is why these fits weight or model each time in turn instead. A survival fit has no feedback check,
because its later covariates are recorded only for people still free of the event.

A continuous-time survival fit has one treatment, at baseline. It compares the covariates across
the arms at baseline. At each target time it also compares the arms among the people whose status
is known (an event seen, or follow-up past that time), and those people with their whole arm. The
[plotting reference](../api/plotting.md#fits-over-time-and-with-external-data) describes each view.

### Read a calibration-slope finding

The point-treatment report tests the calibration slope of each probability model. The slope comes
from a logistic recalibration of the out-of-fold predictions with one intercept per fold. A slope
below 1 means the predictions are more extreme than the observed rates. A slope above 1 means they
are more moderate. The [calibration-slope rule](../technical-reference/validation-methods.md#calibration-slope-rule)
gives the statistic and its standard error.

The report warns only when the slope's interval lies above 0 and excludes 1. The interval splits a
level of 0.05 over the tested models. A finding reads like this line from a learner whose logit is
doubled:

```text
propensity: the out-of-fold predictions are more extreme than the observed rates (calibration
slope 0.50, 97.5% interval 0.44 to 0.56, AUC 0.756, largest untruncated inverse weight 220).
Estimation noise can cause this even in a correctly specified model when the signal is weak or
the model has many covariates. Review the learner before you change the model
```

| question | answer |
| --- | --- |
| what does a finding ask you to do? | review the learner. For the `propensity` model of a binary point treatment, `result.diagnostics.support()` describes the weights that model creates |
| can a correct model warn? | yes. With a weak signal or many covariates, estimation noise can make out-of-fold predictions more extreme than the truth |
| does a finding mean that the model is misspecified? | no. A misspecified logistic model with an intercept also has a population slope of 1, so the slope cannot detect it |
| does a finding mean that the estimate is biased? | no. The rule measures the spread of the predictions and does not measure the estimate |
| why is there no finding on an in-sample fit? | the report tests out-of-fold predictions only. A fit without cross-fitting has `nuisance.evaluation` equal to `in_sample`, and the report tests nothing there. An unpenalized logistic model with an intercept has an in-sample slope of exactly 1, so that slope says nothing about new rows |
| why is a slope missing? | `nuisance["propensity"].calibration_omission` names the cause for that model, for example predictions that are constant within every fold. `nuisance.summary()` prints each cause under the table. `nuisance.to_frame()` has no column for it |

The [calibration-slope study](../technical-reference/method-evidence/calibration-slope-warning.md)
measures the rule's false-warning rate and its detection rate for a binary propensity.

## Go deeper

### More on rows that did not run

Known omissions carry the capability's reason. Examples include an E-value without a supported
contrast and a missingness analysis without missing outcomes. Such a row records no arguments,
because the fit refuses it before the report considers your request. A row that your request
refuses is different. It records your arguments with the signature defaults filled in, and its
reason names the refused value. An example is `truncation_curve` with `mechanism=False` on an
incremental fit with missing outcomes.

An operation can also refuse after invocation, such as omitted-confounding sensitivity when the
doubly robust $\nu^2$ is not positive. That row becomes an `unavailable` omission, retains its
invocation arguments, and names the direct call. Other accepted diagnostics still run.

Structural errors still stop the report. Examples are an invalid argument name, an unknown `refute`
test, and a malformed `benchmark` request. An empty `covariates`, an indicator column of an encoded
covariate, and an unknown covariate are malformed. A direct call raises the same error.

### The model report

For a longitudinal result, `nuisance.rows` covers every retained nuisance fit. Each fitted
treatment and censoring model appears once per time node. Outcome and pseudo-outcome rows also
identify their regimen, cause, and horizon when those fields apply.

The `evaluation` field is `out_of_fold` when the fit used more than one outer fold. It is
`in_sample` for a one-fold fit. The `loss_name` field is `log_loss`, `brier`, or `mse`. The `loss`
field holds the named value.

Treatment and censoring rows report weighted negative log likelihood. An outcome row reports
weighted Brier loss for a binary target, or mean squared error otherwise. A pseudo-outcome row
reports weighted mean squared error.

The `mse` field answers about node regressions alone. A treatment row and a censoring row report
`nan` there. On a weighted fit the value averages under the observation weights.

The nested `model` retains calibration and Super Learner details when the learner supplies them.
`nuisance.omissions` retains typed reasons for unavailable mechanism rows, such as the censoring
model of a fit with complete data.
[Nuisance model quality](../technical-reference/validation-methods.md#nuisance-model-quality)
lists every reason.

A categorical treatment row uses `kind="multinomial probability"` and an empty calibration table.
The report does not convert top-class confidence into binary calibration for a privileged arm.

A binary row reports `calibration_slope` and `calibration_slope_se`. An outcome or pseudo-outcome
row with a continuous target reports its linear slope as `regression_slope`. The longitudinal
report applies no calibration rule.

`nuisance.selection` retains the complete C-TMLE selector or outcome-adaptive artifact. For these
fits, `nuisance.treatment_role` is `collaborative_working_model`. The propensity values describe
the selected denominator and not treatment given the complete adjustment set.

The role drops the two claims that read a treatment law the fit never estimated: that an AUC below
0.55 means overlap is excellent, and the calibration-slope finding. The report keeps every other
claim it makes about the propensity, and
[nuisance model quality](../technical-reference/validation-methods.md#nuisance-model-quality)
states the rule.

### Repeated cross-fitting

A fit with `repeats` above 1 runs cross-fitting several times on new folds and combines the
estimates. On such a fit, `nuisance.repeat_spread` contains one typed row per reported parameter.
Each row gives `standard_deviation`, `reported_standard_error`, and `ratio_to_standard_error`.

Both `standard_deviation` and `reported_standard_error` are on the inference scale of the estimand.
That scale is the log scale for `rr` and `or`, and the outcome scale otherwise, so the two are
comparable. The ratio stays descriptive and carries no pass threshold.

Use `nuisance.repeat_spread_frame()` for a dataframe in the input backend. A one-draw fit retains no
spread row. `nuisance.repeat_spread_omission` then names the cause, and every absent or non-finite
spread gets its own machine-readable reason. A table cell with no finite value prints `-`.

Nuisance models and C-TMLE selection describe draw 1. Split-spread rows read every retained draw.
The model check states the scope in its header, as `Split shown: split 1 of 3`.

A repeated longitudinal fit uses the same rows. `result.diagnostics.nuisance_models().repeat_spread`
holds one row per reported parameter, and `result.repeat_spread()` returns the standard deviations
alone. The nuisance rows describe draw 1, and the report records `n_repeats` and
`reported_repeat`. The score report has one row per node and draw, with a `draw` column.

The combined assessment row states the draw beside the C-TMLE selection fact alone. An ordinary
repeated fit therefore carries no draw text in that row.

### Weight load per score equation

The `group_score_load` table reads each fitted score equation separately. For row $i$ and equation
$j$, $w_i H_{ij}$ multiplies the targeted residual. The table summarizes its magnitude, not the
residual contribution. The group row selects the equation with the smallest Kish load ratio among
score-mask rows. Its `equation` field identifies that equation.

```python
mean_load = support.group_score_load["mean"]
print("equation:", mean_load["equation"])
print("Kish load ratio among targeted rows:", mean_load["targeted_ratio"])
print("top 5% load share:", mean_load["top_5pct"])
print("fitted bound:", (mean_load["lower_bound"], mean_load["upper_bound"]))
```

Read this row as a concentration diagnostic. It is not an effective sample size for the estimate.
It does not include the targeted residual, so it is not an influence contribution. `n_targeted`
counts score-mask rows, including structural zeros in the selected equation. The row also does not
prove or disprove population positivity.

The overlap report provides these group rows for `mean`, `att`, `atc`, and `msm` fits. Regime,
shift, and incremental fits return their own overlap reports instead, with the matching
fitted equation in each policy row's `score_load`; read those rather than the generic positivity
function. A longitudinal fit carries no load row. Missing-outcome and controlled-direct-effect
fits add a product row for the combined denominator, and the report names each estimand that row
does not describe.
[Positivity and overlap](../technical-reference/validation-methods.md#positivity-and-overlap)
gives the product row, the policy rows and each omission.

### The weight-cap curve

`truncation_curve()` re-estimates at other caps on extreme weights. The operation varies a
finite-sample regularisation choice. It does not change the requested estimand. Movement shows
sensitivity to extrapolation in regions where estimated support is limited.

The curve records `upper_bound`, the parameter-specific fitted bound pair and estimate, and
`delta_from_fitted`. Its default grid contains each fitted pair exactly. An explicit `bounds=` grid
keeps the requested grid cardinality and can therefore have no fitted marker. The fitted columns
still name the configuration the grid omitted.

A longitudinal curve has no default grid, so supply `bounds=`. A scalar `b` there means the pair
`(b, 1)`, not `(b, 1 - b)`. Each row reruns the whole backward recursion under one fixed bound on
the cumulative treatment and censoring probabilities, and reports no standard error or interval.
Before the grid runs, a check at the fitted bound must reproduce every reported estimate exactly,
or the operation is unavailable. The frame names the lower endpoint `lower_bound`, where the
point-treatment frame names it `bound`.
[Diagnostics](longitudinal.md#diagnostics) on the longitudinal page gives the rest, and
[truncation stability](../technical-reference/validation-methods.md#truncation-stability) holds the
column table and the two counting rules.

`estimands=` restricts the rows to the parameters you name. Name a reported parameter, such as
`ey[high]`, to select its own row. Name a registered target, such as `ey`, to select one row for
each arm of that target the fit reports. The two forms coincide on a two-armed fit. The curve
refuses any other name, and it refuses an empty selection.

A combined report gives one signed movement range for each reported parameter or working-model
coefficient. It measures that movement from the parameter's own fitted estimate. It also counts the
parameters whose fitted pair the grid skipped. Read the fitted estimate and the fitted pair from the
retained curve.

### Score equation tolerance

`score_equations(tolerance=...)` gates both result families, but on the scale each one’s score
lives on. A point-treatment fit compares the score in the outcome’s own units against
`tolerance * se / sqrt(n)`; a longitudinal fit bounds each node’s relative score, the quantity
the sequential targeting loop itself gates on, and can only tighten the fit’s own convergence
verdict rather than overturn it.
