# Rank candidate exposures

You have several things that might change an outcome, such as exercise, diet and smoking, and
you want to know which of them have an effect you can detect. `variable_importance` fits one causal
effect for each of these *candidate exposures*, each in its own fit with that candidate as the
treatment. It then adjusts the p-values together, because testing many candidates at once makes
a small p-value appear by chance more often.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly import variable_importance, variable_importance_plan
from cleverly.estimators import TMLE

rng = np.random.default_rng(0)
n = 1_000
age = rng.normal(size=n)
exercise = rng.binomial(1, 1 / (1 + np.exp(-age)))
diet = rng.binomial(1, 0.5, size=n)
smoking = rng.binomial(1, 0.3, size=n)
risk = -0.5 + 0.8 * exercise + 0.4 * diet + 0.6 * age
recovered = rng.binomial(1, 1 / (1 + np.exp(-risk)))
frame = pd.DataFrame(
    {"age": age, "exercise": exercise, "diet": diet, "smoking": smoking, "recovered": recovered}
)

arguments = dict(
    outcome="recovered",
    candidates=["exercise", "diet", "smoking"],
    covariates=["age"],  # adjusted for in every candidate's fit
    estimator=TMLE(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        cross_fit=False,
    ),
)
variable_importance_plan(frame, **arguments)  # counts the model fits before any is made
ranking = variable_importance(frame, **arguments)
ranking.summary()
ranking.to_frame()  # one row per candidate: estimate, interval, p-value, adjusted p-value
```

In this simulation exercise raises the chance of recovery, diet raises it a little and smoking
does nothing. Exercise comes first, with an adjusted p-value far below 0.05. Diet's small effect
is not detected at this sample size, and smoking's estimate sits near zero.

The rows are sorted by adjusted p-value, so the first row is the candidate whose effect is
clearest. That is not a ranking by size: a small effect measured precisely can come before a
large one measured loosely. Compare the `psi` column, the estimated effect, and its interval
(`ci_lower`, `ci_upper`) to judge size.

## What each row estimates

Each row is the effect of setting that candidate while holding the adjustment set fixed. With the
default `estimand="ate"` and a binary outcome, it is the difference in the chance of the outcome
between everyone having the candidate and no one having it, the average treatment effect. Pass
another `estimand=`, such as `"rr"` for the risk ratio, to fit a ratio instead; a ratio is tested
against 1 rather than 0.

Each candidate is fitted as a treatment with a few levels. A candidate with more than two levels
gets one row for each level against the lowest, so a three-level candidate gives two rows.

A row is not that candidate's share of one combined effect. It is a separate causal question,
and it needs its own conditions: no unmeasured cause of both that candidate and the outcome, and
every kind of person having some chance of each level (*overlap*). The summary repeats this under
the table.

This is not a predictive feature-importance score, such as a random forest's. A candidate can
predict the outcome well and still have no effect on it, and the
[method reference](../technical-reference/validation-methods.md#variable-importance) says what
the procedure computes.

## Choose what each fit adjusts for

`covariates=` lists the columns every fit adjusts for. By default each candidate's fit also adjusts
for the other candidates (`adjust_for_other_candidates=True`). That answers "what if we change
exercise, holding diet and smoking as they are", and the `adjustment_set` column of `to_frame()`
shows what each row adjusted for.

Set `adjust_for_other_candidates=False` when one candidate can cause another, or is itself
caused by the outcome. Adjusting for a column that sits on the path from a candidate to the
outcome removes part of the effect you want to measure. Adjusting for candidates that move
together can also leave some people with almost no chance of one level, which weakens overlap.

## The adjusted p-values

The p-values are two-sided. They are adjusted together, over every row, with the method of
Benjamini and Hochberg. That method controls the *false discovery rate*: among the candidates you
call significant, the expected share that have no effect stays at or below the level you choose,
such as 5%. The guarantee needs the tests to be independent or to move together in the same
direction.

Here every fit uses the same rows, so the tests are related, and nothing guarantees that
condition. No registered study has yet measured the false discovery rate of these adjusted
p-values, so treat that claim as experimental; the roadmap item
[X125](../roadmap.md#x125-registered-study-of-variable-importance) plans one. The intervals and the
unadjusted p-values are each candidate's own and are not adjusted.

Each candidate's own estimate carries the label of its own fit. A `support` column appears in
`to_frame()` only when some candidate's estimate is experimental, and
[read the support label](results-assessment.md#read-the-support-label) explains the labels.
That column describes each candidate's estimate, not the false discovery claim above.

## What the data must look like

- `data` is a pandas or polars dataframe.
- Each candidate is a column with a few levels, at most 20, and no missing values. A continuous
  candidate, such as minutes of exercise, is refused; cut it into levels first if that answers
  your question.
- The outcome is not one of the candidates.
- Each candidate's fit adjusts for at least one column. With
  `adjust_for_other_candidates=False`, `covariates=` must not be empty.

The default `TMLE` estimator splits the rows into folds and fits each model on the other folds
(*cross-fitting*). With a continuous outcome it then needs the known range of the outcome, passed
as `q_bounds=(lower, upper)`, as [cross-validated TMLE](../technical-reference/cv-tmle.md)
explains. The example above uses a binary outcome and simple logistic models without folds
(`cross_fit=False`), which suits models that cannot overfit. Keep the folds with flexible
learners.

An estimator that would report no interval for some candidate is refused before the first fit,
because the adjustment needs one p-value per row. The
[method reference](../technical-reference/validation-methods.md#variable-importance) lists which
settings do that.

## Large runs

`variable_importance_plan`, called with the same arguments apart from `keep_fits`, counts the model
fits the run would make, one row per candidate, without fitting anything.

By default the result keeps every candidate's fit in `ranking.fits`, so memory grows with the
number of candidates. `keep_fits=False` releases each candidate's data and fit before the next
one starts. The estimates, the adjusted p-values, the table and the summary are the same. What
needs a fit then refuses: `ranking.fits`, the checks below and the model plots.

## Check each candidate's fit

`ranking.diagnostics`, `ranking.sensitivity`, `ranking.validate()` and `ranking.assess()` run each
check on every candidate's own fit and report each candidate separately. A sensitivity bound or a refutation
of a candidate reads that candidate's own estimate and unadjusted interval, not the adjusted test.
To plot a candidate's models, pass its fit, such as
`cleverly.plotting.overlap(ranking.fits["exercise"])`.
[Checking a result](results-assessment.md) describes each check.
