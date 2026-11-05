# A small trial with external data

Use this page when you ran a randomized trial that is too small for a tight interval, and you also
have outcomes for other patients on the same treatment question, such as a registry or the records
of an earlier study. Those external patients were not randomized, so a simple pooled comparison is
biased. `cleverly` can learn how their outcomes differ from the trial's, remove that difference,
and give an interval for the trial's own effect that is often narrower than the trial alone gives.

The interval answers a narrower question than the usual one. It is valid for the effect inside a
simple model of how the external patients differ, which the fit chooses from your data. It is valid
for the trial's effect itself only if that model is rich enough to describe the difference, and
the data cannot confirm that. The fit labels every result experimental for this reason, which means
no simulation study checks this exact setup.

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
    source="in_trial",  # 1 on trial rows, any other value on external rows
)
study = CausalStudy(frame, design=design)
```

The study description states what you claim. The treatment was randomized in the trial, and every kind of
patient in the trial could get either arm. You claim nothing about the external rows. The
adjustment columns must include every covariate that explains why the external outcomes differ
from the trial's, because the fit learns that difference as a function of them.

## Fit and read it

Pick the effect first. `TrialATE(population="trial")` is the effect among the trial's patients.
`TrialATE(population="pooled")` is the trial's effect averaged over the covariates of every row,
trial and external. Choose `"trial"` unless you want the effect for the external patients' kind of
patient, and then read the note on the external rows below. The two answers differ when the effect varies with
a covariate that also predicts who is in the trial.

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
for who is in the trial, and each prediction for a row comes from models trained without that row.

In the printed summary, read the estimate and its interval first. Then read the note on the
external rows, which gives the lowest estimated chance that an external row could have been a
trial row. For the pooled effect, a low value means the trial has no patients like some external
ones, so the effect is extrapolated there and the fit warns. For the trial effect the note is
information only. The last note says how many terms the fit used to describe the effect and the
bias, and `result.working_models` lists them for each of the five groups of rows that the fit holds out in turn.

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
interval narrows when the external rows differ from the trial in a way simple enough for the model
to learn, such as a constant offset or a trend in one covariate.

## Options

| situation | what to set |
| --- | --- |
| outcomes are missing for some rows | name the indicator column in `TrialWithExternalData(missingness=...)` and the fit weights recorded outcomes. The design assumes the trial's missing outcomes depend only on the covariates and the arm, and assumes nothing about the external ones |
| the external rows are controls only | nothing. The fit notices and learns the difference on the control rows |
| a binary outcome | `outcome_family="binomial"`, or leave `"auto"` |
| fewer rows, or a quick check | `AdaptiveTMLEMethod(form="pooled")` selects the working models on all rows at once and runs faster than the default cross-validated form |
| your own list of candidate terms | `working_basis=` takes a scikit-learn transformer that maps covariates to the candidate terms |

The default candidate terms are each covariate, five bent-line pieces of it (each is zero below
a cut point and rises in a straight line above it), and every product of two covariates. The fit keeps only terms
that have enough rows in both groups to be estimated.

## What the method refuses

It estimates a two-valued treatment on the difference scale, and it refuses a treatment with three
or more levels, a ratio, and a continuous dose, each with a reason. It refuses every other
estimator on a trial design, because they would treat the pooled rows as one population. It also
refuses the sensitivity and refutation analyses of the ordinary workflow, which assume one
population. [Scope and refusals](../technical-reference/scope-and-refusals.md) lists each one.

:::{dropdown} Details: what the interval covers
The fit estimates a projection of the trial effect on a model that it selects with a lasso, a
regression that sets unneeded coefficients to zero. The standard error is for that projection. The
projection equals the trial effect when the selected model contains the true conditional effect and
the true difference between external and trial outcomes, or in some cases when it contains only
one of the two. The authors' theorem requires the selected model to approximate both well enough
that the leftover bias shrinks faster than the standard error, and no check on the data confirms
that. The [technical page](../technical-reference/adaptive-tmle.md) states the estimator, the
conditions, and the evidence, and the method is from van der Laan, Qiu, Tarp and van der Laan,
*Journal of Causal Inference* (2026), doi:10.1515/jci-2024-0025.
:::

For a full worked analysis on simulated hospital data, with the known answer beside the estimates,
see [a small pilot with external records](../examples/trial-with-external-data.ipynb).
