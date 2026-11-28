# Treatments over time

These pages cover a treatment decided at several times, such as a weekly offer, where earlier
treatments change later covariates and later decisions. They describe the data, choose the plans to
compare, and fit them. Intervals, diagnostics, refutation, sensitivity analysis and saving work as
for any fit, and each of those pages has a section for fits over time. An outcome that is a time to
an event has its own pages under [time-to-event outcomes](time-to-event.md).

| your question | page |
| --- | --- |
| how to declare the treatment times, the covariates known before each one, censoring and the outcome, read a long panel, or declare repeated outcomes, clusters, survey weights or known treatment chances | [describe data over time](../longitudinal.md) |
| what to compare: a rule that reads the history, a random draw from known chances, a shift in a continuous dose, a change in the odds of treatment, or a trend line across plans | [plans and policies over time](../longitudinal-plans.md) |
| which estimator to fit: longitudinal TMLE, the sequentially doubly robust estimator, sequential g-computation and IPW, structural nested mean models or the parametric g-formula, and their fitting options | [estimators over time](../longitudinal-estimators.md) |

```{toctree}
:hidden:
:maxdepth: 1

Describe data over time </user-guide/longitudinal>
Plans and policies over time </user-guide/longitudinal-plans>
Estimators over time </user-guide/longitudinal-estimators>
```
