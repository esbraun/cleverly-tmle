# Rank candidate exposures

You have several things that might change an outcome, such as exercise, diet and smoking. You
want to know which of them have an effect you can detect. `variable_importance` estimates one
causal effect for each of these *candidate exposures*, each in its own fit with that candidate as
the treatment. It then adjusts the p-values together, because a small p-value turns up by chance
more often when you test many candidates at once.

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
ranking.to_frame()  # one row per candidate here: estimate, interval, p-values
```

In this simulation exercise raises the chance of recovery, diet raises it a little and smoking
does nothing. Exercise comes first: it raises the chance of recovery by about 0.16 (95% interval
0.09 to 0.23), with an adjusted p-value far below 0.05. Diet's small effect is not detected at this
sample size (adjusted p-value 0.23), and smoking's estimate sits near zero. The summary marks the
adjustment's claim experimental, for the reason [the adjusted p-values](#the-adjusted-p-values)
gives.

The [discharge practices example](../examples/ranking-exposures.ipynb) runs a full ranking, with
its checks, on simulated data where one practice causes another.

To record what the study is about, pass a `StudyProtocol` as `protocol=`. The estimates do not
change. The result keeps the record, `ranking.summary(detail="full")` prints it as the study
description, and each candidate's fit stores its fingerprint, as a fit from a `CausalStudy` does.

The rows are sorted by adjusted p-value, so the first row is the candidate whose effect is
clearest. That is not a ranking by size: a small effect measured precisely can come before a
large one measured loosely. Compare the `psi` column, the estimated effect, and its interval
(`ci_lower`, `ci_upper`) to judge size.

## What each row estimates

Each row is the effect of setting that candidate for everyone, with the other columns left as they
were, after adjusting for the columns its fit adjusts for. With the default `estimand="ate"` and a
binary outcome, it is the difference in the chance of the outcome between everyone having the
candidate and no one having it, the average treatment effect. Pass another `estimand=`, such as
`"rr"` for the risk ratio, to fit a ratio instead; a ratio is tested against 1 rather than 0.

Each candidate is fitted as a treatment with a few levels. A candidate with more than two levels
gets one row for each level against the lowest (the first in sorted order for text), so a
three-level candidate gives two rows. The `parameter` column names the levels, such as
`ate[2 vs 0]`.

A row is not that candidate's share of one combined effect. It is a separate causal question,
and it needs its own conditions: no unmeasured cause of both that candidate and the outcome, and
every kind of person having some chance of each level (*overlap*). The summary repeats this under
the table.

This is not a predictive feature-importance score, such as a random forest's. A candidate can
predict the outcome well and still have no effect on it. The
[method reference](../technical-reference/validation-methods.md#variable-importance) says what the
procedure computes.

## Choose what each fit adjusts for

`covariates=` lists the columns every fit adjusts for. By default each candidate's fit also adjusts
for the other candidates (`adjust_for_other_candidates=True`). That answers "what if we change
exercise, with diet and smoking as they are", and the `adjustment_set` column of `to_frame()`
shows what each row adjusted for.

Set `adjust_for_other_candidates=False` when one candidate can cause another, or is itself
caused by the outcome. Adjusting for a column that sits on the path from a candidate to the
outcome removes part of the effect you want to measure. When you turn it off, list the candidate
that comes first in `covariates=` as well, so the fits of the candidates it affects still adjust
for it; a candidate is always left out of its own fit's adjustment. Adjusting for candidates that
move together can also leave some people with almost no chance of one level, which weakens
overlap.

## The adjusted p-values

The p-values are two-sided, and by default they are adjusted together, over every row, so that
the chance of *any* false finding stays at or below the level you choose, such as 5%. That chance
is the *family-wise error rate*. Keeping it at 5% also keeps the *false discovery rate*, the
expected share of rows without an effect among those you call significant, at or below 5%.

The adjustment is a *step-down max-T* (`adjustment="max_t"`, the default). Every candidate's fit
uses the same rows, so the estimates move together. The adjustment measures how strongly they move
together from each fit's per-row contributions to its estimate, then simulates 10,000 sets of test
statistics with that pattern and no real effects (`multiplier_draws=`, seeded by
`multiplier_seed=0`, so a rerun gives the same p-values). A row's adjusted p-value is the share of
those simulated sets whose largest statistic, among the rows not yet called significant, reaches
the row's own. The guarantee is a large-sample one: it holds as the number of people grows, with the
list of candidates fixed, when each candidate's own test is correct. Because the p-values come from
a simulation, they carry a small random error, about 0.002 near 0.05 with 10,000 draws, and
another `multiplier_seed` can move them by that much.

`ranking.adjustment_support` labels that claim. It reads *experimental* for every run until the
simulation study that measures the family-wise error rate has run; the study is declared and its
settings are fixed in advance. After it runs, the label covers only runs like the study's own,
which the [method reference](../technical-reference/validation-methods.md#variable-importance)
lists. One of them matters for this example: a run with fewer than 4,000 rows keeps the label,
because at 2,000 rows the per-candidate tests rejected a little too often in the study's design
probe. The summary states the reason.

`adjustment="bh"` applies the Benjamini-Hochberg adjustment instead, which aims at the false
discovery rate. Its claim always reads experimental: its proof needs tests that are independent or
that depend on each other in a particular positive way. Two-sided tests of related estimates have
that property only for special patterns of correlation, and nothing guarantees one here, so no
proof covers a run like this. With some patterns the false discovery rate exceeds the level you
chose, as the [method reference](../technical-reference/validation-methods.md#variable-importance)
shows with its sources.

The step-down max-T refuses weights estimated by an earlier model (`weights_estimated=True`) and
clustered data with fewer than 40 clusters before any fit; `adjustment="bh"` runs in both cases.
The [method reference](../technical-reference/validation-methods.md#variable-importance) lists
every refusal. The intervals and the unadjusted p-values are each candidate's own and are not
adjusted.

`cleverly` marks an estimate *experimental* when no simulation study covers a fit like it, so its
interval has no checked coverage. Each candidate's estimate takes the mark of its own fit. A
`support` column appears in `to_frame()` only when some candidate's estimate is experimental, and
[read the support label](results-assessment.md#read-the-support-label) explains the labels. That
column describes each candidate's estimate, not the adjustment's claim, which
`ranking.adjustment_support` describes.

## What the data must look like

- `data` is a pandas or polars dataframe.
- Each candidate is a column with a few levels, at most 20, and no missing values. A continuous
  candidate, such as minutes of exercise, is refused; cut it into levels first if that answers
  your question.
- The outcome is not one of the candidates.
- Each candidate's fit adjusts for at least one column. With
  `adjust_for_other_candidates=False`, `covariates=` must not be empty.

The [method reference](../technical-reference/validation-methods.md#variable-importance) lists
these refusals and the order in which they run.

The default `TMLE` estimator splits the rows into folds and fits each model on the other folds
(*cross-fitting*). With a continuous outcome it then needs the known range of the outcome, passed
as `q_bounds=(lower, upper)`, as [cross-validated TMLE](../technical-reference/cv-tmle.md)
explains. The example above uses a binary outcome and fits without folds (`cross_fit=False`),
which suits simple models with few covariates, such as these logistic regressions. Keep the folds
with flexible learners.

An estimator that would report no interval for some candidate is refused before the first fit,
because either adjustment needs one p-value per row. The
[method reference](../technical-reference/validation-methods.md#variable-importance) names those
settings, and the [inference reference](../technical-reference/inference.md) says when each fit
reports an interval.

## Large runs

`variable_importance_plan`, called with the same arguments apart from `keep_fits`, counts the model
fits the run would make, one row per candidate, without fitting anything.

By default the result keeps every candidate's fit in `ranking.fits`, so memory grows with the
number of candidates. `keep_fits=False` releases each candidate's data and fit before the next
one starts. Each estimate still keeps its per-row contributions, which the adjustment reads, so
memory grows by one column of numbers per row of the table rather than by one fit per candidate.
The estimates, the adjusted p-values and the table are the same, and so is the summary, except
that any fit warnings are written into it. `ranking.fits`, each `diagnostics` and
`sensitivity` call, and the model plots then refuse, and `validate()` and `assess()` return reports
that mark every check unavailable, as the
[method reference](../technical-reference/validation-methods.md#variable-importance) records.

## Check each candidate's fit

`ranking.diagnostics` and `ranking.sensitivity` run each check on every candidate's own fit and
return one report per candidate. A sensitivity bound, how strong a missing confounder would have
to be to erase the effect, reads that candidate's own estimate and unadjusted interval, not the
adjusted test. `ranking.validate()` and `ranking.assess()` run on every candidate's fit and return
one row per check, naming each candidate that needs attention. To plot a candidate's models, pass
its fit, such as `cleverly.plotting.overlap(ranking.fits["exercise"])`.
[Checking a result](results-assessment.md) describes each check.
