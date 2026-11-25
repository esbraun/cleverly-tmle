# Check a fitted result

A fitted result can check itself. An effect estimate rests on two models: the outcome model, which
predicts the outcome from the treatment and the covariates, and the treatment model, which predicts
each row's chance of the treatment it got. TMLE, the default method, starts from the outcome model
and adds a correction step that uses the treatment model to remove the bias the outcome model
leaves. That step weights each row by one over its predicted chance of its own treatment, so a row
with a near-zero chance gets an extreme weight.

The checks on this page ask whether the two models predict well, whether the weights are extreme,
whether the correction step finished its job, and how strong a missing confounder would have to be
to change the answer. A confounder is a variable that affects both the treatment and the outcome,
and a missing one is a confounder the data do not record. `assess()` runs the checks together and
prints one report.

A clean report says that these checks found no problem with how the estimate was computed. It does
not show that the effect is causal. That needs every confounder to be among the covariates you
adjusted for, and no check can see a variable the data do not hold. [Sensitivity
analysis](sensitivity.md) asks how strong such a variable would have to be.

## Run every check at once

Call `assess()` on your fitted result:

```python
battery = result.assess()
print(battery.summary())
```

The [quickstart](../getting-started/quickstart.md) fit of the navigation program prints this report,
shortened here:

```text
                            Assessment of the fit
==============================================================================
Result:      nothing needs attention    Need attention:                      0
Checks run:  8                          Not run:                             8
==============================================================================
Results
analysis                       finding
------------------------------------------------------------------------------
overlap                        0.0% of units were capped at [0.015, 0.985], so
                               the estimate does not lean on the cap.
outcome and treatment models   The models show no problem these metrics can
                               detect.
unmeasured confounding         Left-out variables with both strengths at 0.514
                               could, at worst, move the estimate to 0; weaker
                               ones, with both strengths below 0.514, could
                               not.
E-value                        An unmeasured confounder would need a risk
                               ratio of 3.52 with both the treatment and the
                               outcome, beyond the measured covariates, to
                               explain away the estimate; 3.34 to move the
                               interval to the null.
==============================================================================
Checks
check              result   finding
------------------------------------------------------------------------------
equations solved   passed   The correction step solved every estimating
                            equation to within its allowed error.
==============================================================================
Not run
check                        why
------------------------------------------------------------------------------
cap on extreme weights       not run by default because it retargets the fit
refutation tests             not run by default because it refits the models
covariate benchmark          needs an explicit covariates argument, which a
                             combined report has no basis to choose
worst-case bounds            not run by default because it refits the models
```

Look first at the header. `Need attention` counts the checks that failed or warned, and they print
first under `Checks`; any count above 0 is the thing to read before you report. The finding of such
a row and the notes under the report say what to look at next. For a manager, the header is the
summary: how many checks ran and how many need attention.

`Results` gives one line for each analysis that has no pass or fail rule. The overlap check asks
whether every kind of row had a real chance of either treatment, read from the predicted chances and
the weights. The E-value measures how strong a missing confounder would need to be. Read those
numbers yourself: [Sensitivity analysis](sensitivity.md) explains the two confounding rows.

`Not run` lists what was skipped and why. A skipped row is not a failure: it waits for a flag or an
argument, and the notes under the report name the call that runs it.

On this fit nothing needs attention and the weights needed no cap. The correction step works by
solving one equation per estimate, and it solved them; that is needed for the standard error to
describe the estimate, though it does not show that the standard error is right. The report covers
three kinds of check: checks of the fitted models and the correction step, further checks such as
the refutation tests and the weight-cap curve, and the missing-confounder analyses. Print
`battery.summary("full")` to see each row's whole finding, and call `battery.to_frame()` for one row
per check with its status and next step. `battery.report("support")` returns the full overlap report
behind the `overlap` row.

### What runs by default and what needs a flag

`assess()` with no flags reads what the fit stored and refits no model, so it is fast. It runs every
summary of stored numbers, such as the E-value, and every cheap re-estimate, and it skips two slower
kinds of check. One reruns the correction step on the stored models (a retarget), such as the curve
over caps on extreme weights (`truncation_curve()`), `missingness()` and `tipping_gamma()`;
`include_retargets=True` runs those. The other fits new models (a refit): the refutation tests
(`refute()`), which refit the analysis about 60 times, the covariate benchmark (`benchmark()`), and
the planted confounder (`simulated_confounding()`). `marginal_bounds()`, the "worst-case bounds"
row, is a refit too: it fits extra regressions on the fit's folds (the groups of rows that
cross-fitting holds out in turn) without refitting the estimator. `include_refits=True` runs the
refits.

Before you report a result, run at least the refutation tests and the covariate benchmark. Pass both
flags, and give the checks that need a choice their arguments:

```python
from cleverly.sensitivity import ConfounderStrengthGrid

full = result.assess(
    include_refits=True,
    include_retargets=True,
    random_state=21,
    arguments={
        "benchmark": {"covariates": ("W1", "W2")},
        "simulated_confounding": {
            "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.5)),
        },
    },
)
print(full.summary())
```

Replace `W1` and `W2` with covariates of your own. The benchmark asks what a missing confounder as
strong as those covariates could do, and the grid sets the strengths of a confounder that
`simulated_confounding()` plants in the data before it refits; [Sensitivity
analysis](sensitivity.md) explains both. `random_state` seeds every check that draws at random. A
seeded fit reuses its fit seed when you omit it, and an unseeded fit draws a seed and records it on
the returned row. `result.diagnostics.run_all(include_refits=True, include_retargets=True)` runs the
model and weight checks, including the refutation tests, without the missing-confounder analyses.

The cost belongs to the fit, not to the name of the check, so read it from
`result.diagnostics.capability(name).execution` (or `result.sensitivity.capability(name)`), which
holds `"summarize"`, `"retarget"` or `"refit"`. [Cost classes](#cost-classes-by-fit) gives the cases
where the same check costs more.

### Rows that did not run

Each row carries one status, and [the status
contract](../technical-reference/validation-methods.md#the-status-contract) names every status.
`passed` and `failed` belong to checks with a pass or fail rule, `completed` means an analysis ran
without one, and `warning` uses an existing diagnostic rule; `failed` and `warning` are the two that
need your attention. `deferred` means the same check can run once you supply a choice or pass a cost
flag. The row keeps its next step and the request you made, with the seed a combined run supplies.
Pass that request back to the direct call to run it. `not_applicable` means the check has no meaning
for this question, such as a missing-outcome analysis on complete data, and `unavailable` that the
check has meaning but cannot run on this fit.

Some checks answer for one estimate. The estimand is the quantity the fit estimates, named by its
row in the summary, such as `ate`. Such a check defers on a fit that reports no bare `ate` and two
or more estimates it could read, such as a fit with three arms, and its next step names `estimand`.
Pass the estimate's name, as `result.estimates` lists it, through `arguments`.

The status contract's subsection "The ambiguous default estimand" lists those checks and their
defaults. A sensitivity analysis picks the estimate itself when the fit reports exactly one eligible
estimate. `refute()` does not, so on a fit whose one estimate is not named `ate` its row defers as
well. [More on rows that did not run](#more-on-rows-that-did-not-run) gives the other cases.

## Run one check at a time

Each row of the report is also a call of its own, which returns the full report of that check.

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

`validate()` runs the three quick checks: whether the correction step solved its equations
(`score_equations()`), the overlap check of the predicted chances and weights (`support()`), and how
well the outcome and treatment models predict (`nuisance_models()`). `truncation_curve()`
re-estimates at other caps on extreme weights; print `stability.summary()` to see each estimate's
change from the fitted one and the share of units each cap changed. A large change means the
estimate leans on a few rows with extreme weights. `corrections()` applies only to a `DRTMLE` fit (a
TMLE variant whose interval aims to stay valid when one of the two models is wrong) whose guard
subtracts a correction term; ordinary and collaborative TMLE report it as `not_applicable`.
`run_all()` with no flags runs every check of `result.diagnostics` that reads only what the fit
stored.

`influence()` reports how few rows drive each estimate. It is a direct call, which `run_all()` and
the combined report do not include. On the quickstart fit it prints:

```text
estimate   effective rows   skewness   SE uncertainty   coverage loss   flag
------------------------------------------------------------------------------
ate                 612.8       0.02            0.034            .000      -
```

`effective rows` (the Kish count) is how many rows with equal pull would give the same spread. `SE
uncertainty` is the relative standard deviation of the variance estimate, and the column to watch:
above 0.125, close to 64 effective rows, the fit warns that its standard error is itself noisy
(`SparseInfluenceWarning`). `coverage loss` is the shortfall of the 95% interval that a correction
for skewness and heavy tails (the Edgeworth term) predicts.

### Check covariate balance

Weighting by the fitted treatment model should make the treated and untreated groups look alike on
the covariates you adjusted for. `covariate_balance(result)` measures how far it succeeded. For each
covariate it gives the standardized mean difference, the gap between the two groups' means in
standard deviations, before and after weighting.

```python
from cleverly.sensitivity import covariate_balance

balance = covariate_balance(result)
print(balance.summary())
fig = balance.plot()
```

On the quickstart fit the table reads:

```text
covariate           before weighting   after weighting
------------------------------------------------------------------------------
discharge_risk                 0.272            -0.001
prior_utilization             -0.129             0.000
medication_burden              0.100             0.000
age                           -0.050             0.001
```

Read the `after weighting` column. A gap near zero says the weights balanced that covariate; one
above 0.1, the usual threshold, says the treatment model missed something about it. Here
`discharge_risk` starts at 0.27 and ends near zero. The check describes the data: it has no interval
or test, and it says nothing about covariates you did not adjust for. A treatment decided at several
times gets a balance over time ([balance over time](#balance-when-treatment-repeats-over-time)).

### Read a calibration-slope finding

A model's calibration slope says whether its predicted chances are too extreme or too timid. The
report regresses the observed yes/no values on the model's out-of-fold predictions, on the log-odds
scale, with one intercept per fold (a logistic recalibration), and reads the slope. A slope below 1
means the predictions are more extreme than the observed rates; a slope above 1 means they are more
moderate. The report tests the slope of each model that predicts a probability, and names the
treatment model `propensity`. The [calibration-slope
rule](../technical-reference/validation-methods.md#calibration-slope-rule) gives the statistic and
its standard error.

The report warns only when the slope's interval lies above 0 and excludes 1. The interval's level is
corrected for testing several models at once: it splits a level of 0.05 over the tested models. A
finding reads like this line, from a learner whose predictions on the log-odds scale are doubled:

```text
propensity: the out-of-fold predictions are more extreme than the observed rates (calibration
slope 0.50, 97.5% interval 0.44 to 0.56, AUC 0.756, largest untruncated inverse weight 220).
Estimation noise can cause this even in a correctly specified model when the signal is weak or
the model has many covariates. Review the learner before you change the model
```

A finding asks you to review the learner. For the `propensity` model of a yes/no treatment given
once, `result.diagnostics.support()` describes the weights that model creates. A correct model can
warn too: with a weak signal or many covariates, estimation noise can make out-of-fold predictions
more extreme than the truth. The slope cannot detect a wrong model, because a misspecified logistic
model with an intercept also has a population slope of 1. It measures the spread of the predictions,
so a finding says nothing about bias in the estimate.

A fit without cross-fitting gets no finding. The report tests out-of-fold predictions only, and
there `nuisance.evaluation` equals `in_sample`. An unpenalized logistic model with an intercept has
an in-sample slope of exactly 1, so that slope says nothing about new rows. When a slope is missing,
`nuisance["propensity"].calibration_omission` names the cause for that model, for example
predictions that are constant within every fold. `nuisance.summary()` prints each cause under the
table, and `nuisance.to_frame()` has no column for it. The [calibration-slope
study](../technical-reference/method-evidence/calibration-slope-warning.md) measures the rule's
false-warning rate and its detection rate for a yes/no treatment.

## Go deeper

### More on the combined report

```python
ledger = battery.to_frame()
deferred = ledger[ledger["status"] == "deferred"]
print(deferred[["surface", "check", "next_steps"]])

for step in battery.next_steps():
    print(step)
```

`battery.to_frame()` carries the `surface`, `check`, `status`, `detail`, and `next_steps` columns.
The `surface` column names the group a check belongs to: `validation` for the quick checks,
`diagnostics` for further checks such as the refutation tests and the weight-cap curve, and
`sensitivity` for the missing-confounder analyses. `battery.next_steps()` lists the follow-up calls.
`battery.attention` contains failures and warnings, and `battery.omissions` contains deferred,
not-applicable, and unavailable analyses. `battery.report(name)` retrieves the full report of a
check that ran, and `battery.report(name, surface="diagnostics")` a diagnostic that also appears
under validation. Each row keeps the arguments it ran with and the returned object, and
`report.report(name)` on any combined report retrieves that object.

`assess()` refuses `arguments` for `score_equations`. The combined report runs it under validation
with no arguments and shows that row, so it would compute your answer and then hide it. Call
`result.diagnostics.run_all(arguments=...)`, or call the check itself. The combined report also runs
`support` and `nuisance_models` under validation. Neither accepts an argument, so an argument for
either is a `TypeError` from the signature. An empty mapping applies nothing, so `assess()` accepts
it for all three names.

A refusal names the first thing that is wrong. A check that needs a choice you did not pass names
that choice, and not a cost flag. No flag supplies the covariates for `benchmark`, so the row says
so under every flag.

### Cost classes by fit

The weight-cap curve of a `DRTMLE` fit with a guard refits its smaller (reduced) regressions at
every cap, and that of a treatment over time reruns its whole backward recursion (the step-by-step
fit from the last time point to the first), so both ask for `include_refits=True`. An E-value that
needs one more outcome regression is a refit too. On a fit of a treatment over time, a check whose
formulas have not been worked out for that kind of fit reads `unavailable`, and its reason says why;
the package does not reuse the formulas for a treatment given once.

### More on rows that did not run

Four rows can also defer on another argument. The row defers when the call refuses the default value
and another value runs. It reads `unavailable` when no value runs. The row quotes the sentence that
the call raises.

| check | argument | the row defers when |
| --- | --- | --- |
| `refute()` | `tests` | the fit refuses one default test and admits another, such as `placebo` on a fit of `NaturalCourseMean` |
| `truncation_curve()` | `mechanism` | the fit refuses the default axis and admits the other, such as the treatment axis of an incremental fit with missing outcomes |
| `benchmark()` | `covariates` | always, because the call needs the argument. A fit with one covariate reads `unavailable`, because the call cannot drop every covariate |
| `simulated_confounding()` | `estimand` | the call refuses the default estimand, and another estimand that the fit reports runs |

A refusal the call raises after it starts becomes `unavailable`. An estimand you name that the fit
never reported is such a refusal. The combined run then continues with the other checks, and a
direct call still raises the precise refusal. `not_applicable` and `unavailable` rows also appear in
`omissions`.

Known omissions carry the reason the check gives. Examples include an E-value without a supported
contrast and a missingness analysis without missing outcomes. Such a row records no arguments,
because the fit refuses it before the report considers your request. A row that your request refuses
is different. It records your arguments with the signature defaults filled in, and its reason names
the refused value. An example is `truncation_curve` with `mechanism=False` on an incremental fit
with missing outcomes.

A check can also refuse after it starts, such as the missing-confounder bound when its weight term
(the doubly robust $\nu^2$) is not positive. That row becomes an `unavailable` omission, keeps its
arguments, and names the direct call. The other checks still run.

Structural errors still stop the report. Examples are an invalid argument name, an unknown `refute`
test, and a malformed `benchmark` request. An empty `covariates`, an indicator column of an encoded
covariate, and an unknown covariate are malformed. A direct call raises the same error.

### Balance when treatment repeats over time

When treatment is decided at several times, as in a longitudinal or time-to-event fit, the same call
returns a balance over time, a `BalanceOverTime`, following Jackson (2016). At each treatment time
it compares the covariates recorded so far across that treatment, among the people who had the same
earlier treatments (the same treatment history). Only people still in the study at that time count,
and on a survival fit only those still free of the event. After weighting, each person counts as one
over the fitted chance of the treatments and the follow-up they had up to that time. The summary
averages each comparison over the groups of earlier treatments, each group counting by its size, and
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
confounding. Weighting is not meant to remove it. When that covariate also affects later treatment,
adjusting for it in an ordinary regression gives a biased effect, which is why these fits weight or
model each time in turn instead. A survival fit has no feedback check, because its later covariates
are recorded only for people still free of the event.

A continuous-time survival fit has one treatment, at baseline. It compares the covariates across the
arms at baseline. At each target time it also compares the arms among the people whose status is
known (an event seen, or follow-up past that time), and those people with their whole arm. The
[plotting reference](../api/plotting.md#fits-over-time-and-with-external-data) describes each view.

### The outcome and treatment model report

For a fit of a treatment over time, `nuisance.rows` covers every stored model fit. Each fitted
treatment and censoring model appears once per time point (node). Outcome rows, and the rows of the
regressions a sequential fit runs on its own predicted outcomes (pseudo-outcomes), also name their
regimen, cause, and horizon when those fields apply.

The `evaluation` field is `out_of_fold` when the fit used more than one outer fold, and `in_sample`
for a one-fold fit. The `loss_name` field is `log_loss`, `brier`, or `mse`, and the `loss` field
holds the named value. Treatment and censoring rows report weighted negative log likelihood. An
outcome row reports weighted Brier loss for a yes/no target, or mean squared error otherwise, and a
pseudo-outcome row reports weighted mean squared error.

The `mse` field answers about the outcome regressions alone, and a treatment row or a censoring row
reports `nan` there. On a weighted fit the value averages under the observation weights. The nested
`model` keeps calibration and Super Learner details when the learner supplies them.
`nuisance.omissions` keeps a typed reason for each model row that is missing, such as the censoring
model of a fit with complete data. [Nuisance model
quality](../technical-reference/validation-methods.md#nuisance-model-quality) lists every reason.

A categorical treatment row uses `kind="multinomial probability"` and an empty calibration table.
The report does not turn the top class's predicted chance into a yes/no calibration for one favoured
arm. A yes/no row reports `calibration_slope` and `calibration_slope_se`. An outcome or
pseudo-outcome row with a continuous target reports its linear slope as `regression_slope`. The
report of a treatment over time applies no calibration rule.

On a collaborative TMLE fit or an outcome-adaptive fit, which choose the treatment model's
covariates from the data, `nuisance.selection` keeps the whole record of that choice, and
`nuisance.treatment_role` is `collaborative_working_model`. The propensity values then describe the
chosen model, which predicts treatment from the chosen covariates only. The role drops the two
claims that read a treatment model the fit never estimated: that an AUC below 0.55 means overlap is
excellent, and the calibration-slope finding. The report keeps every other claim it makes about the
propensity, and [nuisance model
quality](../technical-reference/validation-methods.md#nuisance-model-quality) states the rule.

### Repeated cross-fitting

A fit with `repeats` above 1 runs cross-fitting several times on new folds and combines the
estimates. On such a fit, `nuisance.repeat_spread` contains one typed row per reported estimate,
with `standard_deviation` (the spread across the repeats), `reported_standard_error`, and
`ratio_to_standard_error`. The first two are on the scale the interval is built on: the log scale
for `rr` and `or`, and the outcome scale otherwise, so the two are comparable. The ratio describes
and carries no pass threshold.

Use `nuisance.repeat_spread_frame()` for a dataframe in the input backend. A one-draw fit keeps no
spread row. `nuisance.repeat_spread_omission` then names the cause, and every absent or non-finite
spread gets its own machine-readable reason. A table cell with no finite value prints `-`.

The model rows and the collaborative selection describe draw 1, and the spread rows read every
stored draw. The model check states the scope in its header, as `Split shown: split 1 of 3`. A
repeated fit of a treatment over time uses the same rows:
`result.diagnostics.nuisance_models().repeat_spread` holds one row per reported estimate,
`result.repeat_spread()` returns the standard deviations alone, and the report records `n_repeats`
and `reported_repeat`. Its score report has one row per node and draw, with a `draw` column. The
combined report states the draw only beside the collaborative selection, so an ordinary repeated fit
carries no draw text in that row.

### Weight load per score equation

The correction step solves one equation per estimate (a score equation). The `group_score_load`
table reads each of them separately. For row $i$ and equation $j$, $w_i H_{ij}$ multiplies the row's
residual after the correction step, and the table summarizes the size of that multiplier. The group
row picks the equation whose weights concentrate most, the smallest Kish load ratio (the effective
share of rows) among the rows the equation reads. Its `equation` field names it.

```python
mean_load = support.group_score_load["mean"]
print("equation:", mean_load["equation"])
print("Kish load ratio among targeted rows:", mean_load["targeted_ratio"])
print("top 5% load share:", mean_load["top_5pct"])
print("fitted bound:", (mean_load["lower_bound"], mean_load["upper_bound"]))
```

Read this row as a measure of how concentrated the weights are. It does not include the residual, so
it describes neither the estimate's sample size nor any row's pull on the estimate. `n_targeted`
counts the rows the equation reads, including rows whose term is zero by construction. The row also
cannot prove or rule out overlap in the population.

The overlap report provides these group rows for `mean`, `att`, `atc`, and `msm` fits. Regime,
shift, and incremental fits return their own overlap reports instead, with the matching fitted
equation in each policy row's `score_load`. A fit of a treatment over time carries no load row.
Missing-outcome and controlled-direct-effect fits add a product row for the combined denominator,
and the report names each estimate that row does not describe. [Positivity and
overlap](../technical-reference/validation-methods.md#positivity-and-overlap) gives the product row,
the policy rows and each omission.

### The weight-cap curve

`truncation_curve()` re-estimates at other caps on extreme weights. A cap is a choice made for a
finite sample, and changing it leaves the question the same. A large movement shows that the
estimate leans on rows whose predicted chances lie near 0 or 1, where the data hold few similar
rows.

The curve records `upper_bound`, the fitted limits and estimate of each parameter, and
`delta_from_fitted`. Its default grid contains each fitted pair of limits exactly. An explicit
`bounds=` grid keeps the number of limits you asked for and can therefore have no fitted marker; the
fitted columns still name the limits it left out.

A curve of a treatment over time has no default grid, so supply `bounds=`. A single number `b` there
means the pair `(b, 1)`. Each row reruns the whole backward recursion under one fixed bound on the
cumulative treatment and censoring probabilities, and reports no standard error or interval. Before
the grid runs, a check at the fitted bound must reproduce every reported estimate exactly, or the
curve is unavailable. Its frame names the lower endpoint `lower_bound`, where the frame of a
treatment given once names it `bound`. [Diagnostics](longitudinal.md#diagnostics) on the
longitudinal page gives the rest, and [truncation
stability](../technical-reference/validation-methods.md#truncation-stability) holds the column table
and the two counting rules.

`estimands=` restricts the rows to the estimates you name. Name a reported estimate, such as
`ey[high]`, to select its own row. Name a whole family, such as `ey`, to select one row for each arm
the fit reports. The two forms coincide on a two-armed fit. The curve refuses any other name, and it
refuses an empty selection.

A combined report gives one signed movement range for each reported estimate or working-model
coefficient, measured from that estimate's own fitted value. It also counts the estimates whose
fitted pair of limits the grid skipped. Read the fitted estimate and the fitted pair from the stored
curve.

### Score equation tolerance

`score_equations(tolerance=...)` sets how small each equation's error must be, on the scale each
kind of fit uses. A fit of a treatment given once compares the error in the outcome's own units
against `tolerance * se / sqrt(n)`. A fit of a treatment over time bounds each node's relative
error, the same quantity its step-by-step correction loop checks, so the tolerance can only make the
fit's own convergence check stricter.
