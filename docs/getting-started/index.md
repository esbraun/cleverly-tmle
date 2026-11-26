# Getting started

Install the package, then run the quickstart: it estimates the effect of one treatment on one
outcome from simulated data with a known answer, so you can check each step. The quickstart uses
the package's synthetic data only so the column roles and true effect are known; the same
workflow accepts pandas, polars, Arrow-backed pandas, and `pyarrow.Table` inputs
([installation](installation.md) says what each one needs).

```{toctree}
:maxdepth: 1

installation
quickstart
```

## Pick your path

The quickstart covers one treatment given once and one outcome measured later. If your question
has another shape, read the quickstart for the shared steps, then go to the guide and the example
that match it.

| your question | guide | example |
| --- | --- | --- |
| the time until an event, such as readmission, when people leave follow-up or another event comes first | [time-to-event outcomes](../user-guide/survival.md) | [readmission when death can come first](../examples/longitudinal-survival.ipynb) |
| a treatment repeated over time, where earlier treatment changes later choices | [longitudinal treatment](../user-guide/longitudinal.md) | [weekly offer policies](../examples/sequential-offers.ipynb) |
| a policy rather than "everyone against no one", such as offering only high-risk patients or raising each patient's chance of an offer | [estimands and interventions](../user-guide/estimands.md) | [three offer policies](../examples/interventions.ipynb) |
| a continuous dose, such as a discount from 0% to 40% | [a curve over a continuous dose](../user-guide/estimands.md#a-curve-over-a-continuous-dose) | [renewal discounts](../examples/dose-response.ipynb) |
| an effect that changes with a characteristic such as age, or a rule for who should be treated | [effects that change with a baseline column](../user-guide/conditional-effects.md), [learned treatment rules](../user-guide/estimands.md#learned-treatment-rules) | [who gains most from a coupon](../examples/conditional-effects.ipynb), [who should get the offer](../examples/treatment-rules.ipynb) |
| a small randomized trial plus outside records on the same outcome | [a trial with external data](../user-guide/trial-with-external-data.md) | [a small pilot with outside records](../examples/trial-with-external-data.ipynb) |
| the effect with a later step, such as a follow-up visit, fixed for everyone (a controlled direct effect) | [controlled direct effects](../user-guide/estimands.md#controlled-direct-effects) | [gain over current practice](../examples/population-impact.ipynb) |

The [examples](../examples/index.md) list every analysis by situation. The
[workflow](../workflow.md) walks through the stages of one analysis, and
[read and save results](../user-guide/results-assessment.md) explains how to read a
fitted result.
