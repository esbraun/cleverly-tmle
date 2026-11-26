# Workflow

This page runs one complete analysis: whether offering people a health program changes the chance
that they complete an annual check-up, on a simulated table of 400 people in 40 households. The
program raises that chance by an estimated 8 percentage points, with a 95% interval from about -8 to
24 points, so these data are too few to tell whether the program works, which is not evidence that it fails;
the simulation's true effect is about 10 points.
The answer is causal only if the recorded characteristics include every confounder, meaning every
factor that drives both who was offered the program and the outcome, and no data can check that.
The [quickstart](getting-started/quickstart.md) is the
shorter first read.

## Write down the question

Write the question first, because no estimator can fix a vague question. State the population, the
treatment strategies you compare, the outcome, the time horizon, and the
contrast (a difference or a ratio). A label such as "ATE" (average treatment effect: the mean outcome if everyone were
treated minus the mean if no one were) is not enough when the comparison arm or the population is
unclear. Write the description down in code (`StudyProtocol`), so the fitted result carries it:

```python
from cleverly import StudyProtocol

protocol = StudyProtocol(
    target_population="Adults in households registered with participating clinics",
    eligibility=("Adult at the offer date", "Registered with a participating clinic"),
    time_zero="Program offer date",
    treatment_strategies=("Offer the program", "No offer"),
    treatment_versions=("Home visit and two reminder calls", "Usual clinic reminders"),
    outcome="Annual check-up completed (yes/no)",
    horizon="12 months after the offer date",
    intercurrent_event_handling=("Count a person who moves away as not completing",),
    interference_unit="Person",
    assumption_rationale=(
        "Age and baseline score cover the measured common causes of the offer and the check-up",
        "Household members have similar outcomes, but one member's offer does not change another's",
    ),
)
```

Every field is free text: replace each string with your study's, and put your argument for the
conditions the answer rests on in `assumption_rationale`. Time zero is the moment when eligibility is judged, the treatment is decided and follow-up starts.
Treatment versions say what each strategy involves in
practice. An intercurrent event is something after time zero, such as moving away, that changes how
the outcome is read or whether it can be measured. The interference unit is the group within which
one person's treatment may change another's outcome; here each person is their own group.

The fields borrow from the [target-trial](https://miguelhernan.org/whatifbook)
framework (describe the randomized trial you would run if you could), the
[ICH E9(R1)](https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf)
guideline on stating what a study estimates, and work on
[treatment versions](https://pmc.ncbi.nlm.nih.gov/articles/PMC4219328/). The record is a checklist,
not a complete protocol or proof of ICH compliance, and it changes no estimate: code the data to
match it, such as recording a person who moved away as 0. The fitted result keeps it, and
`summary(detail="full")` prints it.

## Describe the data

The code below calls your table `data`. This page's table has a yes/no `outcome` (the check-up), a
0/1 `treatment` (the offer), two baseline covariates, a sampling weight (how many people in the target population each row
stands for), and a `household` column.
People in one household have similar outcomes, so their rows are not independent; that is
different from interference, where one person's offer changes another's outcome. To run the page as
written, build a simulated table first.

:::{dropdown} Build a practice table
```python
import numpy as np
from cleverly.datasets import make_clustered

# The second value holds the true effects, known only for simulated data.
data, _ = make_clustered(n=400, cluster_size=10, seed=7, family="binomial")
data = data.rename(
    columns={
        "Y": "outcome",
        "A": "treatment",
        "W1": "age",
        "W2": "baseline_score",
        "cluster": "household",
    }
)
# Random weights stand in for survey weights.
data["sampling_weight"] = np.random.default_rng(7).uniform(0.5, 1.5, len(data))
```
:::

A design object says what each column is for: the outcome, the treatment, the covariates to adjust
for, and here the weights and the household. One treatment given once is a `PointTreatment`. On your
own data, change the column names, and drop `weights=` or `cluster=` if you have none. Adjust for columns
recorded before the treatment that affect the treatment or the outcome, so that every common cause
of the two is included, and leave out anything the treatment could change:

```python
from cleverly import CausalStudy, PointTreatment

study = CausalStudy(
    data,
    design=PointTreatment(
        outcome="outcome",
        treatment="treatment",
        adjustment=("age", "baseline_score"),
        weights="sampling_weight",
        cluster="household",
    ),
    protocol=protocol,
)
print(study.summary())
```

The summary shows a weighted treated share (the weighted fraction offered the program) of 0.49,
so the groups are about the same size, and an
effective sample size of 369 of 400 rows, which is what the weighted rows are worth in equally
weighted rows. Because household members have similar outcomes, `cluster=` makes the interval
count households instead of people; without it, the interval would treat the 400 people as
independent and come out too narrow. With 40 households, this fit uses the usual 1.96 multiplier. A treated
share near 0 or 1, or an effective sample size far below the row count, means few rows carry the
comparison.

Other column types, such as missing-outcome indicators, subgroups, doses and known treatment
chances, are under [other columns](#other-columns); treatments over time, event times and trials
pooled with outside data have their own designs, under [over time and
survival](#over-time-and-survival).

## Name the quantity you want

The question becomes an object that carries what it needs, such as the treatment level to compare
against or the horizon. `ATE(reference=0)` compares
treatment level 1 with the reference level 0, no offer:

```python
from cleverly import ATE

question = ATE(reference=0)
```

Other questions, such as the effect among the treated, a risk ratio, a dose policy or a treatment
rule, each have their own object, listed under [other questions](#other-questions).

## List the conditions the answer rests on

`study.identify` checks that the question fits the design, without fitting a model, and lists the
conditions the answer rests on and the methods that can estimate it:

```python
effect = study.identify(question)
print(effect.summary())
print(effect.available_methods().summary())
```

Here it lists four conditions: a well-defined treatment (every way of getting the offer, or of not
getting it, has the same effect, which the treatment versions above describe), no interference, no hidden confounding, and
overlap (everyone had some chance of each option). The data cannot check the first three. Overlap is
partly checkable: the overlap check after the fit can show where it fails, but cannot prove it holds.
The methods summary lists the methods that can answer the question, with TMLE, described below,
first. `summary(detail="full")` gives each condition in formal wording and why
a method does not fit, and a program reads the same statements from `effect.identification`.

If a question is well posed but the package cannot answer it correctly, it raises a
`CapabilityError` that names what is missing instead of estimating a different quantity
([capabilities and refusals](user-guide/capabilities.md)).

## Fit the effect

Targeted maximum likelihood estimation (TMLE), the default method, fits an outcome model, which
predicts the check-up from the offer and the covariates, and a treatment model, which predicts each
person's chance of the offer. The models in these two slots are called learners. TMLE then adjusts the outcome model's
predictions using the treatment model. When the treatment model is close to right, the adjustment removes most of the bias a
slightly wrong outcome model would leave. For an auditable record, write the settings as one
object; on your own data, the learners and fold counts (folds are explained below) are the lines to revisit:

```python
from sklearn.linear_model import LogisticRegression
from cleverly import CrossFitting, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5),
    runtime=Runtime(random_state=17, n_jobs=1),
)
print(effect.planned_fits(method=method))
result = effect.estimate(method=method)
```

`planned_fits` counts the model fits before you run them, here 10, two models times
five folds ([count the model fits](user-guide/methods-learners.md#count-the-model-fits-before-a-long-run)).
The same settings also work as keyword arguments to `estimate()`.

Any scikit-learn model fits either slot. Left out, each slot gets a combination of three models
(gradient boosting, a random forest and a lasso), which often takes half a minute to two minutes
per fit. Cross-fitting fits the models on other folds of the data than the rows they predict, so
overfitting does not leak into the estimate or make the interval too narrow, as it can in-sample. The default is 10 folds, plus 5 inner
folds that the default combination uses to weigh its models; this page uses 5 to stay fast. On
your own data keep the default unless the run is too slow.

This page's yes/no outcome needs nothing more. A cross-fitted fit of a continuous outcome also
needs the outcome's known range.

:::{dropdown} Details: a continuous outcome
Pass the range as `q_bounds=(lower, upper)` to `estimate()`, or
`targeting=Targeting(q_bounds=...)` on the method object. Without it the fit would read the range
off every observed outcome, including the rows a model predicts but was not fitted on, so the
package refuses it. An outcome with no
range you can state is fitted on every row instead, with `CrossFitting(enabled=False)` and simple
models ([fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)).
:::

The methods summary also lists collaborative TMLE, which helps when one covariate almost decides
who is treated. At its default settings it refuses clustered data like this page's, and its default
search reports no interval. [Method
presets](user-guide/methods-learners.md#method-presets) says when to switch.

## Check the fit and its sensitivity

`result.assess()` runs the default checks in one call: overlap, the outcome and treatment models,
whether TMLE's adjustment did its job, and the sensitivity analyses that refit nothing. The
adjustment should make the outcome model's errors, weighted using the treatment model, average to
zero, and the interval relies on that; the Checks table reports it as the `equations solved` row.
A sensitivity analysis asks how strong a missing confounder would have to be to change the answer.
It gives the confounder two strengths, each from 0 (none) to 1 (all): how much of the outcome it
would explain beyond the covariates, and how much it would add to predicting who got the offer.

```python
print(result.assess().summary())
```

Here nothing needs attention. No one's predicted offer chance reached the probability limits, the
bounds the package puts on each chance so that no one gets an extreme weight. The
outcome and treatment models row reports no problem its metrics can detect (a problem would be named
there), and the `equations solved` row passed, which does not show that either model is right.

A missing confounder with both strengths at 0.075 could move the estimate to 0, a low bar, as
expected for an estimate whose interval already includes 0. Eight checks did not run. Three need
missing outcomes or another method. The other five are deferred because
they refit the models, redo TMLE's adjustment under several weight caps, or need an argument such as
a covariate name, and the first note gives the call for each.

Two checks worth running by hand are the refutation tests, which refit after breaking the data on
purpose, and the benchmark, which measures how strong a missing confounder would be if it mattered
as much as a covariate you recorded:

```python
print(result.diagnostics.refute().summary())
print(result.sensitivity.benchmark(covariates=["age"]).summary())
```

A shuffled treatment should give an effect near 0, and an added noise column should leave the
estimate where it was. Here both match the expected value each row prints. Leaving out age moves
the estimate from 8 to 10 points, toward the raw 15-point difference between the groups, as expected for a covariate that raises both
the offer and the check-up. A missing confounder as strong as age would have strengths of about
0.015 and 0.007, both below 0.075, so it could not move the estimate to 0. In words for a report: a hidden factor as
strong as age on both sides would fall far short of erasing the estimate. To run both inside `assess()`, pass `include_refits=True` and
`arguments={"benchmark": {"covariates": ["age"]}}`; `random_state=` seeds the refits
([check a fitted result](user-guide/assessment.md)).

An overlap plot draws each group's predicted chance of the offer. Where one group has people and the
other has none, or chances pile up near 0 or 1, the comparison there rests on extrapolation. Here
every chance lies between 0.10 and 0.88, and the two groups cover the same range:

```python
from cleverly.plotting import overlap

fig = overlap(result)
```

## Report the result and its limits

```python
print(result.to_frame())
fig = result.plot()
```

`result.to_frame()` gives one row per estimate, with its interval and its support label, which says
whether the package's own simulation studies back a fit set up like this one. The `ate` row holds
the answer given at the top: about 8 points, interval about -8 to 24. The raw weighted difference
between the groups is about 15 points, larger than the true effect, because age and baseline score
raise both the chance of the offer and the chance of a check-up.

The support label reads `supported` when those studies cover the fit and passed, as here, and
`experimental` when they do not or a condition the summary's notes name applies; an experimental
result can still be reported, with that label and its reason
([read the support label](user-guide/results-assessment.md#read-the-support-label)). Report it
with the question, the population, the conditions, the learners and fold settings, the estimate and
interval, and the overlap and sensitivity checks. Report also what `result.provenance` records: the
`cleverly`, NumPy and scikit-learn versions and fingerprints of the data and the folds. The package
records no git commit; set `Runtime(run_id=...)` to tie the fit to your own commit or job.
`result.plot()` draws a forest plot for most results; a figure carries no notes, and
`cleverly.plotting.notes(fig)` returns them ([plotting](api/plotting.md)).

:::{dropdown} Details: intervals with few households
Every fold keeps a household whole, and the interval rests on the number of households. With 40 or
more, a fit like this one uses the usual 1.96 multiplier. With fewer, or for an estimate evaluated
on held-out folds (`cv_evaluation=True`) at any count, it uses a Student t multiplier, which is
larger. Below 10 households, for a fit like this one, it labels the interval experimental, and with
fewer than three it reports none. [Few clusters](technical-reference/inference.md#few-clusters)
gives the thresholds for each method, and
[small-sample scale](technical-reference/inference.md#small-sample-scale) covers fits without
clusters.
:::

## Save the result

Save the result when an audit trail matters:

```python
result.save("analysis.joblib")
```

The file holds the complete fitted result, including its settings and fitted models; load only
files you trust, in a compatible Python environment. `cleverly.load()` warns with
`VersionMismatchWarning` when another `cleverly` version saved the file, then loads it as saved.
On the loaded result, `replayability.summary()` says what it can redo
([what a loaded result can redo](user-guide/results-assessment.md#what-a-loaded-result-can-redo)).

## Go deeper

### Other columns

- `weights=`: sampling weights that make the rows represent a target population; the estimate then
  describes that population ([observation weights](user-guide/data-design.md#observation-weights-are-not-estimand-weights)).
- `missingness=` and `treatment_missingness=`: a 0/1 column saying whether the outcome, or the
  treatment, was recorded ([missing outcomes](user-guide/data-design.md#missing-outcomes-and-missing-treatments)).
- `strata=`: subgroups to report separately; each must also be an adjustment column.
- `treatment_kind="continuous"`: a dose instead of arms ([continuous treatment](user-guide/data-design.md#continuous-treatment)).
- `outcome_family=`: the outcome's type when the package should not infer it.

An empty adjustment set claims that nothing confounds the treatment. Set `randomized=True` when
randomization justifies that claim; the treatment model is still estimated. A trial that recorded
each person's chance of treatment declares it with `treatment_probabilities=`
([known treatment mechanism](technical-reference/point-treatment-tmle.md#known-treatment-mechanism)),
and [known chances beside weights](user-guide/data-design.md#known-chances-beside-weights) says
what changes when the rows also carry unequal weights. [Data and study
design](user-guide/data-design.md) covers every column type.

### Other questions

| your question | object | guide |
| --- | --- | --- |
| the average effect, or the effect among the treated or the untreated | `ATE`, `ATT`, `ATC`, `RiskRatio`, `OddsRatio`, `CounterfactualMean` | [arm-indexed questions](user-guide/estimands.md#arm-indexed-questions) |
| current practice compared with no one treated | `NaturalCourseMean`, `PopulationAttributableRisk`, `PopulationAttributableFraction` | [arm-indexed questions](user-guide/estimands.md#arm-indexed-questions) |
| a policy that shifts a dose or raises the chance of treatment | `ModifiedTreatmentPolicy`, `IncrementalEffect` | [modified treatment policies](user-guide/estimands.md#modified-treatment-policies), [incremental interventions](user-guide/estimands.md#incremental-propensity-score-interventions) |
| a curve over a continuous dose, or a trend across levels | the dose-curve call, `MSMProjection` | [a curve over a continuous dose](user-guide/estimands.md#a-curve-over-a-continuous-dose), [marginal structural models](user-guide/estimands.md#marginal-structural-models) |
| the value of a treatment rule, learned or optimal | `LearnedRuleValue`, `OptimalRuleValue` | [learned treatment rules](user-guide/estimands.md#learned-treatment-rules) |
| an effect that changes with a characteristic such as age | `ConditionalContrast` | [conditional effects](user-guide/conditional-effects.md) |
| the effect with a later step fixed for everyone | `ControlledDirectEffect` | [controlled direct effects](user-guide/estimands.md#controlled-direct-effects) |
| a trial's effect, helped by outside records | `TrialATE` | [a trial with external data](user-guide/trial-with-external-data.md) |
| plans over time, risks by a time, restricted mean survival, competing events | `RegimeContrast`, `RegimeMean` | [longitudinal treatment](user-guide/longitudinal.md), [time-to-event outcomes](user-guide/survival.md) |

### Over time and survival

The steps from naming the quantity on are the same for every design: `identify`, `estimate`,
`assess`, the diagnostics, the sensitivity analyses and `save` work on each result. What changes is
the design and the question.

| your data | design | question | guide |
| --- | --- | --- | --- |
| a treatment at several times, with covariates measured between them | `LongitudinalTreatment` (wide table), `LongPanel` (one row per person and time) | `RegimeContrast` of treatment plans | [longitudinal treatment](user-guide/longitudinal.md) |
| one treatment at the start and a time to an event, with censoring | `TimeToEvent` | `RegimeContrast` with `horizons=` for risks by each time | [time-to-event outcomes](user-guide/survival.md) |
| a randomized trial plus outside records | `TrialWithExternalData` | `TrialATE` | [a trial with external data](user-guide/trial-with-external-data.md) |

### Where next

- One treatment, flexible models: [point-treatment TMLE](examples/point-treatment-tmle.ipynb) and
  [flexible models and shared teams](examples/cross-fitting.ipynb).
- An effect that varies across people: [conditional effects](user-guide/conditional-effects.md).
- Choosing learners and folds: [methods and learners](user-guide/methods-learners.md).
- Reading every table the package prints: [read cleverly's output](user-guide/reading-output.md).
- Figures: [plot results](examples/plots.ipynb).
- Every worked analysis by situation: [examples](examples/index.md).
