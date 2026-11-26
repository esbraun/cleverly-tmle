# Check a fitted result

An effect estimate rests on two models, and a fitted result stores what it takes to check them. The
outcome model predicts the outcome from the treatment and the covariates, and the treatment model
predicts each row's chance of the treatment it got. TMLE, the default method, starts from the
outcome model and adds a correction step that uses the treatment model to remove most of the bias
the outcome model leaves, when the treatment model is close to right. That step weights each row by
one over its predicted chance of its own treatment, so a row with a near-zero chance gets an extreme
weight.

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

The [quickstart](../getting-started/quickstart.md) fit of a care-navigation offer prints this
report:

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
robustness value               Left-out variables with both strengths at 0.514
                               could, at worst, move the estimate to 0; weaker
                               ones, with both strengths below 0.514, could
                               not.
confounding bound inputs       see report.report('elements')
confounding grid               see report.report('contour')
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
simulated confounder         needs an explicit grid argument, which a combined
                             report has no basis to choose
worst-case bounds            not run by default because it refits the models
DR-TMLE corrections          the fitted method does not use the correction
                             system
missing not at random        the identified functional has no observation
                             mechanism: missingness_tilt requires a fit with
                             missing outcomes
missing-data tipping point   the identified functional has no observation
                             mechanism: missingness_tilt requires a fit with
                             missing outcomes
==============================================================================
No check needs attention.
Notes:
[1] Suggested next steps: call result.diagnostics.truncation_curve() directly,
    or pass include_retargets=True; call result.diagnostics.refute() directly,
    or pass include_refits=True; call result.sensitivity.benchmark() directly
    with covariates; call result.sensitivity.simulated_confounding() directly
    with grid; call result.sensitivity.marginal_bounds() directly, or pass
    include_refits=True.
[2] report.to_frame() lists every row with its detail, report.next_steps() the
    suggested calls, and report.report(name) the full report of one check.
```

Look first at the header. `Need attention` counts the checks that failed or warned, and they print
first under `Checks`; any count above 0 is the thing to read before you report. The finding of such
a row and the notes under the report say what to look at next; in the printed text, `report` stands
for your report object, here `battery`. For a manager, the header is the summary: 8 checks ran and
none needs attention.

`Results` gives one line for each analysis that has no pass or fail rule. The `overlap` row reads
the cap on predicted chances: the fit limits each row's predicted chance of treatment to between
0.015 and 0.985, so that no row gets an extreme weight. A share of 0.0% means the cap changed no
row, which is what you want; a large share means the estimate leans on the cap, and the weight-cap
curve below shows how much. The `outcome and treatment models` row says whether either model
predicts poorly; a calibration-slope warning there is explained under [Read a calibration-slope
finding](#read-a-calibration-slope-finding).

The `equations solved` check reads the correction step. That step adjusts the outcome model's
predictions until their weighted errors average to about zero, one such estimating equation per
estimate, and it solved them here. That is needed for the standard error to describe the estimate,
though it does not show that the standard error is right.

`Not run` lists what was skipped and why; a skipped row is no failure. The first five wait for a
flag or an argument, and note 1 names the call that runs each; [Run the slower checks before you
report](#run-the-slower-checks-before-you-report) covers them. The last three do not apply to this
fit: `DR-TMLE corrections` applies only to `DRTMLE`, a variant of the method, and the two
missing-data rows need a data set with some outcomes missing. Print `battery.summary("full")` to see
each row's whole finding, and call `battery.to_frame()` for one row per check with its status and
next step.

### Read the confounding rows

The `unmeasured confounding` row describes a missing confounder by two strengths, each a share
between 0 and 1. The outcome-side strength is the share of the outcome's variation, left after
adjusting for the covariates, that the confounder would explain; the treatment-side strength is how
much it would add to predicting who got the treatment. Here a confounder would need both strengths
at about half (0.514) to move the estimate to 0. Near 0 a weak confounder could erase the effect;
near 1 only an overwhelming one could. The `robustness value` row repeats that number on its own,
and the `confounding bound inputs` and `confounding grid` rows hold tables you open with
`battery.report("elements")` and `battery.report("contour")`.

The `E-value` row measures strength as a risk ratio, the chance of something in one group divided by
the chance in another. To explain the estimate away, a missing confounder would need to make both
the treatment and the outcome about 3.5 times as likely between its levels, beyond the measured
covariates. On a continuous outcome like this one, the effect is first converted to a risk ratio,
approximately. An E-value of 1 means no confounding is needed.

Neither number has a fixed cutoff for "robust enough". For the strengths and the robustness value,
the package's yardstick is a covariate you did record: the covariate benchmark asks what a missing
confounder as strong as that covariate could do. The E-value has no such yardstick in the package;
judge it against how strongly your recorded covariates relate to the treatment and the outcome.
[Sensitivity analysis](sensitivity.md) shows both.

### Run the slower checks before you report

`assess()` with no flags reads what the fit stored and fits no new model, so it is fast. Two flags
add the slower checks. `include_retargets=True` reruns the correction step on the stored models at
other caps on extreme weights (the `cap on extreme weights` row). `include_refits=True` adds the
checks that fit new models:

- the refutation tests (`refute()`), which change the problem so that the answer is known, such as
  shuffling the treatment so that the effect should be about zero, and refit about 60 times;
- the "worst-case bounds" row (`marginal_bounds()`), the range of effects if a missing confounder
  could change each row's odds of treatment by up to a set factor, with no limit on its link to the
  outcome;
- the covariate benchmark (`benchmark()`) and the planted confounder (`simulated_confounding()`),
  which adds a made-up confounder to the data and refits. These two also need a choice from you,
  so they run only when you pass their arguments as well.

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

Replace `W1` and `W2` with covariates of your own that you understand well. The grid sets the
planted confounder's strengths: a treatment strength of 0.1 switches the treatment of the 10% of
rows highest on the made-up confounder, and on a continuous outcome a strength of 0.5 lowers the
outcome by 0.5, in its own units, for each standard deviation of the made-up confounder. The grid
above is a fair first try. Its report, `full.report("simulated_confounding")`, measures your named
covariates on the same scales, so widen the grid until it reaches them ([Sensitivity
analysis](sensitivity.md) shows an example). `random_state` seeds every check that draws at random.

On the quickstart fit, with its own covariates `discharge_risk` and `prior_utilization`, the header
reads `Checks run: 13` and `Not run: 3`, and the new rows include:

```text
cap on extreme weights         Across these caps the estimate moves by at most
                               0 (0 std. errors).
refutation tests               No test failed, and none of the three tests
                               gives a pass or fail for this kind of fit, so
                               compare each mean with the expected column.
covariate benchmark            see report.report('benchmark')
worst-case bounds              see report.report('marginal_bounds')
```

The cap row says the estimate did not move at any cap tried, so it does not lean on rows with
extreme weights. A refutation test prints `passed` or `failed` only where a simulation study backs
it for this kind of fit; otherwise it prints the numbers for you to compare with what is expected,
as here. Healthy looks like a shuffled-treatment average near 0 and refits that spread about as much
as the standard error predicts. Open the full tables with `full.report("refute")`,
`full.report("benchmark")` and `full.report("marginal_bounds")`. On this fit the worst-case range
stays above 0 even when a missing confounder could double or halve each row's odds of treatment.
[Try to break the result](refutation.md) reads the refutation table, and [Sensitivity
analysis](sensitivity.md) reads the benchmark and the bounds.

### What each status means

Each row carries one status, and [the technical
reference](../technical-reference/validation-methods.md#the-status-contract) lists them all.
`passed` and `failed` belong to checks with a pass or fail rule, and `completed` means an analysis
ran without one. `warning` means a check's fixed rule found something you should qualify in your
report. `failed` and `warning` are the two that need your attention. `deferred` means the check can
run once you supply a choice or pass one of the two flags above, and its `next_steps` column names
the call. `not_applicable` means the check has no meaning for this question, such as a
missing-outcome analysis on complete data, and `unavailable` that the check has meaning but cannot
run on this fit.

A check that reads one estimate needs to know which. On a fit with several estimates and no average
treatment effect named `ate`, such as a fit with three treatment groups, it defers and asks for
`estimand`, the name of the estimate as `result.estimates` lists it, which you pass through
`arguments`. [More on rows that did not run](#more-on-rows-that-did-not-run) gives the details.

## Run one check at a time

Most rows of the report are also calls of their own, which return the full report of that check.

```python
validation = result.validate()
scores = result.diagnostics.score_equations()
stability = result.diagnostics.truncation_curve()
influence = result.diagnostics.influence()
```

`validate()` runs the three quick checks: whether the correction step solved its equations
(`score_equations()`), the overlap check of the predicted chances and weights
(`result.diagnostics.support()`), and how well the outcome and treatment models predict
(`result.diagnostics.nuisance_models()`). `truncation_curve()` re-estimates at other caps on extreme
weights; print `stability.summary()` to see each estimate's change from the fitted one and the share
of rows each cap changed. A large change means the estimate leans on a few rows with extreme
weights. [Other single checks](#other-single-checks) lists the rest.

`influence()` is not in the report. It shows whether a few rows drive the estimate, and on the
quickstart fit it prints:

```text
estimate   effective rows   skewness   SE uncertainty   coverage loss   flag
------------------------------------------------------------------------------
ate                 612.8       0.02            0.034            .000      -
```

The estimate is an average of one contribution per row. `effective rows` is how many equal
contributions would give the same spread (the Kish count), and `skewness` measures how lopsided the
contributions are. `SE uncertainty` is how noisy the standard error is itself, and the column to
watch: above 0.125, close to 64 effective rows, the `flag` column marks the estimate, because its
standard error is itself noisy. A fit of a treatment over time also warns when it is fitted
(`SparseInfluenceWarning`). `coverage loss` is how much the 95% interval is predicted to fall short
because of lopsided or heavy-tailed contributions. Here 0.034 is far below 0.125 and the predicted
loss is zero, so a few rows do not drive this estimate.

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

Read the `after weighting` column. A gap near zero says the weights balanced that covariate. One
larger than 0.1 in size, the threshold the summary holds against, says the weights left that
covariate imbalanced: the treatment model may have missed something about it, or few rows overlap.
Here `discharge_risk` starts at 0.27 and ends near zero. The check describes the data: it has no
interval or test, and it ignores covariates you did not adjust for. A treatment decided at several
times gets a balance over time ([balance over time](#balance-when-treatment-repeats-over-time)).

When the data declare each row's chances (`treatment_probabilities=`), no treatment model is
fitted: the weights read the declared chances, and the summary says so. A gap past 0.1 is then a
chance difference or a sign that treatment did not follow those chances, as the
[randomized pilot](../examples/randomized-pilot.ipynb) example discusses.

## Go deeper

### Read a calibration-slope finding

A model's calibration slope says whether its predicted chances are too extreme or too timid. The
report fits a small logistic regression of the observed yes/no values on the model's out-of-fold
predictions (each row's prediction from a model trained without it), expressed as log-odds (the
logarithm of chance over one minus chance), with a separate intercept for each fold, and reads its
slope. A slope below 1 means the predictions are more extreme than the observed rates; a slope above
1 means they are more moderate. The report tests the slope of each model that predicts a
probability, and names the treatment model `propensity`. The [calibration-slope
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

The AUC measures how well the model ranks treated rows above untreated ones (0.5 is chance, 1 is
perfect), and the largest untruncated inverse weight is the biggest weight any row would get before
the cap. To review the learner, look at its settings (a learner that overfits gives extreme
predictions), and for the `propensity` model of a yes/no treatment given once, read
`result.diagnostics.support()`, which describes the weights that model creates, to see whether its
extreme predictions give some rows very large weights. The slope cannot detect a wrong model,
because a misspecified logistic model with an intercept also has a population slope of 1. It
measures the spread of the predictions, so a finding says nothing about bias in the estimate.

A fit without cross-fitting gets no finding. The report tests out-of-fold predictions only, and such
a fit's model report records `nuisance.evaluation` as `in_sample` (`nuisance` is the outcome and
treatment model report from `result.diagnostics.nuisance_models()`). An unpenalized logistic model
with an intercept has an in-sample slope of exactly 1, so that slope cannot show how the model does
on new rows. When a slope is missing, `nuisance["propensity"].calibration_omission` names the cause
for that model, for example predictions that are constant within every fold. `nuisance.summary()`
prints each cause under the table, and `nuisance.to_frame()` has no column for it. The
[calibration-slope study](../technical-reference/method-evidence/calibration-slope-warning.md)
measures the rule's false-warning rate and its detection rate for a yes/no treatment.

### Other single checks

```python
support = result.diagnostics.support()
nuisance = result.diagnostics.nuisance_models()
corrections = (
    result.diagnostics.corrections()
    if result.diagnostics.capability("corrections").available
    else None
)
all_cached = result.diagnostics.run_all()
```

`support()` and `nuisance_models()` return the overlap and model reports behind the first two
`Results` rows, and `battery.report("support")` returns the same overlap report from a combined run.
`corrections()` applies only to a `DRTMLE` fit (a TMLE variant whose interval aims to stay valid
when one of the two models is wrong) fitted with a guard, an extra term the fit subtracts to protect
that interval; ordinary and collaborative TMLE report it as `not_applicable`. `run_all()` with no
flags runs every check of `result.diagnostics` that reads only what the fit stored, and
`result.diagnostics.run_all(include_refits=True, include_retargets=True)` runs the model and weight
checks, including the refutation tests, without the missing-confounder analyses. `random_state`
on `assess()` defaults to the fit's seed on a seeded fit, and an unseeded fit draws a seed and
records it on the returned row.

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

### Checks that cost more on some fits

The cost belongs to the fit, not to the name of the check, so read it from
`result.diagnostics.capability(name).execution` (or `result.sensitivity.capability(name)`), which
holds `"summarize"`, `"retarget"` or `"refit"`. Besides the weight-cap curve, the missing-outcome
analyses `missingness()` and `tipping_gamma()` are retargets on a fit with missing outcomes.

The weight-cap curve of a `DRTMLE` fit with a guard refits the smaller regressions that the guard
needs (its reduced regressions) at every cap, and that of a treatment over time reruns its whole
backward recursion (the step-by-step fit from the last time point to the first), so both ask for
`include_refits=True`. An E-value that needs one more outcome regression is a refit too. On a fit of
a treatment over time, a check whose formulas have not been worked out for that kind of fit reads
`unavailable`, and its reason says why; the package does not reuse the formulas for a treatment
given once.

### More on rows that did not run

A deferred row keeps its next step and the arguments you passed, with the seed a combined run
supplies, in its `arguments` field (each record in `battery.omissions` has one); pass those
arguments to the direct call it names to run it. The status contract's subsection "The ambiguous
default estimand" lists the checks that read one estimate and their defaults. A sensitivity analysis
picks the estimate itself when the fit reports exactly one it can read. `refute()` does not, so on a
fit whose one estimate is not named `ate` its row defers as well.

Four rows can also defer on another argument: the row defers when the call refuses the default
value and another value runs, and reads `unavailable` when no value runs, quoting the sentence the
call raises. `NaturalCourseMean` below is the mean outcome under the treatments people actually got,
and an incremental fit multiplies each row's odds of treatment.

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
(the estimate of $\nu^2$, the average squared weight, built from the treatment model) is not
positive. That row becomes an `unavailable` omission, keeps its
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

### The model report on other fits

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

### How concentrated the weights are in each equation

The correction step solves one estimating equation per estimate. In each, every row's error after
the correction step is multiplied by a weight built from the treatment model: for row $i$ and
equation $j$ that multiplier is $w_i H_{ij}$, which for a mean grows as one over the row's predicted
chance of its own treatment. The `group_score_load` table summarizes how unevenly those multipliers
are spread. Its group row picks the equation whose multipliers concentrate most: the one with the
smallest Kish load ratio, the effective share of rows, among the rows the equation reads (its
targeted rows). Its `equation` field names it.

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
equation in each policy row's `score_load`. Read those rows, not the generic overlap function. A fit
of a treatment over time carries no load row. Missing-outcome and controlled-direct-effect fits add
a product row for the combined denominator, and the report names each estimate that row does not
describe. [Positivity and
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
means the pair `(b, 1)`, not `(b, 1 - b)`. Each row reruns the whole backward recursion under one
fixed bound on the cumulative treatment and censoring probabilities, and reports no standard error
or interval. Before the grid runs, a check at the fitted bound must reproduce every reported
estimate exactly, or the curve is unavailable. Its frame names the lower endpoint `lower_bound`,
where the frame of a treatment given once names it `bound`.
[Diagnostics](longitudinal.md#diagnostics) on the longitudinal page gives the rest, and [truncation
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

### The checks in the tutorials

| tutorial | what it runs |
| --- | --- |
| [One treatment, one outcome](../examples/point-treatment-tmle.ipynb) | `assess()` and the weight-cap curve |
| [Flexible models and shared teams](../examples/cross-fitting.ipynb) | `assess()` and repeated cross-fitting |
| [Readmission by day 30, 60 and 90](../examples/discharge-to-readmission.ipynb) | `assess()`, balance over time and the weight-cap curve on a survival fit |
| [Weekly offer policies](../examples/sequential-offers.ipynb), [Weekly orders under repeated offers](../examples/weekly-outcomes.ipynb) | balance over time and the weight-cap curve on longitudinal fits |
| [Plot results](../examples/plots.ipynb) | covariate balance and the weight-cap curve as figures |
