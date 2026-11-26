# A small trial with external data

Use this page when you ran a randomized trial that is too small for a tight interval, and you also
have outcomes for other patients on the same treatment question, such as a registry or the records
of an earlier study. Those external patients were not randomized, so a simple pooled comparison is
biased. `cleverly` can learn how their outcomes differ from the trial's, remove that difference,
and give an interval for the trial's own effect that is often narrower than the trial alone gives.

The interval answers a narrower question than the usual one. It is valid for the effect inside a
simple model of how the external patients differ, which the fit chooses from your data. It is also valid
for the trial's effect itself when that model is rich enough to describe the difference and how the
effect varies, and the data cannot confirm that. The fit labels every result experimental.

A [simulation
study](../technical-reference/method-evidence/adaptive-tmle-trial.md) checks how often the 95%
interval covers its target, including a stress setting where the outside patients differ in a way
the fit's model cannot describe. The study page holds the numbers. The label stays because of the
condition above, which no data check confirms, whatever models and sample size a fit uses.

## What the data must look like

Put the trial rows and the external rows in one table with a column that marks which is which. The
external rows can have both treatment arms or only controls. Every row needs the outcome, the
treatment (two values), and the covariates you adjust for.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    AdaptiveTMLEMethod,
    CausalStudy,
    CrossFitting,
    ModelSpec,
    PointTreatment,
    Runtime,
    TMLEMethod,
    TrialATE,
    TrialWithExternalData,
)

# For your own data, change the file and the four column names below.
frame = pd.read_csv("pooled_rows.csv")

design = TrialWithExternalData(
    outcome="score",
    treatment="program",
    adjustment=("risk", "age", "prior_visits"),
    source="in_trial",  # 1 on trial rows (or the value of trial=), any other on external rows
)
study = CausalStudy(frame, design=design)
```

The study description states what you claim. The treatment was randomized in the trial, and every kind of
patient in the trial could get either arm. You claim nothing about the external rows. The fit
learns that difference as a function of the adjustment columns, so include the covariates that
explain it: the simpler the difference looks in them, the more the external rows help. Also include
any covariate the trial's randomization depended on.

## Fit and read it

Pick the effect first; `population=` has no default. `TrialATE(population="trial")` is the effect
among the trial's patients. `TrialATE(population="pooled")` is the trial's effect averaged over the
covariates of every row, trial and external. Choose `"trial"` unless you want the trial's effect
carried over to the mix of patients in the whole table, and then read the note on the external rows
below. The two answers differ when the effect varies with a covariate that also predicts who is in
the trial.

```python
effect = study.identify(TrialATE(population="trial"))

fast = ModelSpec(
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
result = effect.estimate(
    method=AdaptiveTMLEMethod(models=fast, runtime=Runtime(random_state=1)),
)
print(result.summary())
```

`effect.estimate()` with no method already picks this method for a trial design, with the default
library of learners. That library takes minutes at a few thousand rows, so pass learners as above
while you explore, and widen them once the workflow runs. Any scikit-learn regressor and classifier
works. The method fits three prediction models, for the outcome, for who gets the treatment, and
for who is in the trial, and each prediction for a row comes from models trained without that
row. The trial-membership model uses the treatment learner unless you pass
`AdaptiveTMLEMethod(source_learner=...)`.

Simple learners are for trying the workflow, so report a fit with flexible learners, such as
gradient boosting or the default library. Main-terms learners give each covariate one straight-line
term, as `LinearRegression` does. In a simulation by
[Karim (2026)](https://arxiv.org/abs/2607.02787), their 95% intervals missed the effect far too
often when trial enrollment depended on the covariates and the external rows differed in a large,
irregular way. Flexible learners restored coverage. In this library's
[selective-enrollment study](../technical-reference/method-evidence/adaptive-tmle-selective.md),
one simulated design with 3,200 rows, flexible learners covered close to 95%. Main-terms learners
did too in the default form but fell short in the faster pooled form, and the study records them as
a side measurement rather than a test, so one design cannot clear them in general.

In the printed summary, read the estimate and its interval first. For a continuous outcome such
as `score`, the interval uses a t distribution, with its degrees of freedom in the `df` column,
and it can stretch further on one side for a skewed outcome. So it need not be centred on the
estimate, and neither need the trial-alone interval below. A yes/no outcome gets the usual
symmetric interval and no `df` column
([small-sample scale](../technical-reference/inference.md#small-sample-scale)).

Then read the note on the external rows, which gives the lowest estimated chance that an external
row could have been a trial row. For the pooled effect, a low value means the trial has no
patients like some external ones, so the effect is extrapolated there and the fit warns. For the
trial effect the note is information only. The fit also warns when 20 or fewer external rows got
the treatment, or 1% of the external rows if that is more. It then learns their difference from the
trial from few patients ([the technical page](../technical-reference/adaptive-tmle.md#refusals)
lists this warning with the refusals).

The last note counts the terms the fit used for the effect and for the external difference (the
bias). The fit holds out each
of five groups of rows in turn and selects terms for each:

```python
for model in result.working_models:
    print(model.fold, model.effect_columns, model.bias_columns)
```

The lists differ because each group selects its own terms, and the choice can swing a lot between
groups: on data where the effect is the same for every patient, one group may still keep a dozen or
more effect terms. Read the lists to see what the fit used, not as a test of whether the effect
varies or of how the external rows differ. In the names, `A` stands for the
treatment column (`program` here), so `A*age` lets the age trend of the external difference change
with the arm. `age_hinge3` is a bent-line piece of `age`, zero below a cut point and rising in a
straight line above it.

To see the note on the external rows as a picture, plot the predicted chances:

```python
from cleverly.plotting import overlap

fig = overlap(result)
```

The upper panel shows each trial patient's predicted chance of the treatment, by the arm they got.
The lower panel shows every row's estimated chance of being a trial row, trial rows above the axis
and external rows below. The dash-dotted line at 0.01 is the floor the warning checks, with the
share of external rows below it printed at its foot. The dotted lines are the limits the fit puts on
both chances. These are the chances before the fit adjusts the trial-membership model, and
`cleverly.plotting.notes(fig)` says so.

## Compare with the trial alone

The reason to borrow external data is a narrower interval, so check that it narrowed. Fit the
trial rows by themselves with an ordinary estimate and compare the two intervals.

```python
trial_only = CausalStudy(
    frame[frame["in_trial"] == 1],
    design=PointTreatment(
        outcome="score", treatment="program", adjustment=("risk", "age", "prior_visits")
    ),
)
alone = trial_only.identify(ATE()).estimate(
    method=TMLEMethod(models=fast, cross_fitting=CrossFitting(enabled=False)),
)
print(alone.estimate, result.estimate)
```

If the two estimates sit far apart relative to their intervals, the external rows pulled the answer
away from the trial, and the bias model did not fully explain why. A gain is not guaranteed. The
interval narrows when the external rows differ from the trial by the same amount in both arms, in a
way simple enough for the model to learn, such as a constant offset or a trend in one covariate. If
the difference itself depends on the arm, the external rows add little.

The pooled fit estimates each patient's chance of treatment from all rows, so it does not take the
trial's known chance. When you know that chance, give it to the trial-alone fit with
`PointTreatment(treatment_probabilities=...)`. Its interval then needs no model of who got treated,
as long as no outcome is missing and the outcome model is fitted on other rows than it predicts
(cross-fitting) or is a fixed formula such as `LinearRegression`. Knowing the chance does not
improve the best precision any method can reach for this effect.
[Known treatment mechanism](../technical-reference/point-treatment-tmle.md#known-treatment-mechanism)
gives the conditions.

## Options

| situation | what to set |
| --- | --- |
| outcomes are missing for some rows | name the indicator column in `TrialWithExternalData(missingness=...)` and the fit weights recorded outcomes. `ModelSpec(missingness_learner=...)` sets the learner that predicts who has a recorded outcome; without it the slow default library runs. The design assumes the trial's missing outcomes depend only on the covariates and the arm, and assumes nothing about the external ones |
| the external rows are controls only | nothing. The fit notices and learns the difference on the control rows |
| a binary outcome | `TrialWithExternalData(outcome_family="binomial")`, or leave `"auto"` |
| a quick check | `AdaptiveTMLEMethod(form="pooled")` selects the working models on all rows at once and runs faster than the default cross-validated form |
| the external arms may differ from the trial by a different amount each | `AdaptiveTMLEMethod(force_arm=True)` always lets the two external arms differ by a free shift. It matches the R package `atmle`, and the external rows then add no information on the effect's level, so the interval is no narrower than the trial's own |
| your own list of candidate terms | `working_basis=` takes a scikit-learn transformer that maps covariates to the candidate terms |

The [worked example](../examples/trial-with-external-data.ipynb) runs each of these options on
simulated data under its "Go deeper" heading, beside the known answer.

By default, a covariate with nine or more distinct values becomes its standardized value
(rescaled to mean 0 and standard deviation 1) and five bent-line pieces. A covariate with fewer
values, such as a yes/no site flag or a short count, becomes one on/off column for each value above
the smallest. Every pair of covariates also gets its product. The fit drops a bent-line, on/off or
product column that is nonzero on fewer than `max(10, ceil(sqrt(n)/2))` rows in either of the two
sets it compares, treated and control or external and trial. Here `n` is the number of rows the
terms are selected on, and the screen always keeps the standardized values.
[The dictionary](../technical-reference/adaptive-tmle.md#the-dictionary) on the technical page
gives the full rule.

## What the method refuses

It estimates a two-valued treatment on the difference scale, and it refuses a treatment with three
or more levels, a ratio, and a continuous dose, each with a reason. It refuses a known chance of
treatment in the trial, `treatment_probabilities=`, because the pooled fit has no form that uses it.
When you declare the design, it refuses sampling weights (`weights=`), clusters (`cluster=`),
effects within subgroups (`strata=`), a direct effect with a middle-step variable held fixed
(`intermediate=`) and rows with an unrecorded treatment (`treatment_missingness=`). Each refusal
points to the trial-alone `PointTreatment` fit, which takes them. The method also refuses rerunning
the five-group split with fresh groups and combining the fits (`CrossFitting(repeats=...)` above
1), and it refuses every other estimator on a trial design, because they would treat the pooled rows
as one population. [Scope and refusals](../technical-reference/scope-and-refusals.md#adaptive-tmle-refusals)
lists each refusal with its message.

It also refuses the sensitivity analyses of `cleverly.sensitivity` and the refutation and truncation
curve of `cleverly.validation`. No bound is derived for this kind of fit, and each refit picks its
own simple description of the difference, so the refits do not estimate the same thing.

Four checks run on the fit:

- `result.diagnostics.support()`: the trial and external row counts, and the lowest chance of
  trial membership among the external rows;
- `score_equations()`: whether the correction step, which adjusts the fitted models to remove the
  bias they leave in the estimate, solved its equations;
- `nuisance_models()`: the learners, and the terms each group kept;
- `corrections()`: how many passes adjusted the trial-membership model, and whether they met the
  stopping rule.

`result.assess()` collects the four and lists each refused check with its reason, and
`result.diagnostics.run_all()` gives the four plus the two refused refits. Neither runs
`result.diagnostics.influence()`. It counts the effective rows, how many equal contributions would
give the same spread, and flags the estimate `noisy` when the standard error is itself uncertain
enough that the interval may miss its 95% level ([checking a fit](assessment.md) explains the
columns). On a small trial it can flag `noisy` while `run_all()` reports nothing that needs
attention, so read a `noisy` interval with caution.

:::{dropdown} Details: what the interval covers
The fit estimates a projection of the trial effect on a model that it selects with a lasso, a
regression that sets unneeded coefficients to zero. The standard error is for that projection. The
projection equals the trial effect when the selected model contains the true conditional effect and
the true difference between external and trial outcomes.
It also equals the trial effect when the chances of treatment and of trial membership change little
across patients. The authors' theorem requires the selected model to approximate both well enough
that the leftover bias shrinks faster than the standard error, and no check on the data confirms
that. The [technical page](../technical-reference/adaptive-tmle.md) states the estimator, the
conditions, and the evidence, and the method is from van der Laan, Qiu, Tarp and van der Laan,
*Journal of Causal Inference* (2026), doi:10.1515/jci-2024-0025.
:::

For a full worked analysis on simulated hospital data, with the known answer beside the estimates,
see [a small pilot with outside records: adaptive TMLE](../examples/trial-with-external-data.ipynb).
