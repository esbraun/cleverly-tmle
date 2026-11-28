# Time-to-event outcomes

These pages cover an outcome that is whether and when an event happens, such as readmission or
churn, when some people leave follow-up. They declare the data, choose what to report, and fit it.
Intervals, diagnostics, refutation, sensitivity analysis and saving use the general pages, which
note what differs for these fits.

| your question | page |
| --- | --- |
| the risk of the event by each of a grid of check times under each plan, its differences and ratios, the average event-free time up to a cut-off, a treatment that changes over time, or a trend line through the risk curves | [risks at set check times](../survival.md) |
| the same risks from exact event and censoring times, without a grid | [exact event times](../survival-exact-times.md) |
| an event that can come first, such as death before readmission, the risk had it been prevented, or a treatment with two parts that act on the two events | [competing events](../survival-competing-events.md) |
| how the fit gets there: a risk curve that dips although a risk cannot fall over time, data-chosen treatment and censoring models, an interval that holds when one set of models is wrong, or other estimators to compare | [fitting choices for event times](../survival-fitting.md) |

```{toctree}
:hidden:
:maxdepth: 1

Risks at set check times </user-guide/survival>
Exact event times </user-guide/survival-exact-times>
Competing events </user-guide/survival-competing-events>
Fitting choices for event times </user-guide/survival-fitting>
```
