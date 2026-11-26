# Choosing models and checking results

These tutorials choose the prediction models inside a fit, and keep the interval honest when the
models are flexible or patients share a team. The last one checks and explains a result with
figures.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| which prediction models to put inside the fit, such as a blend of a straight line, splines and gradient boosting, or many recorded columns to screen | [choosing the prediction models](../learner-choice.ipynb) | a screen that picks the treatment model's columns by a straight-line fit to the outcome can drop real confounders whose link to the outcome is curved, leaving them to the outcome model alone |
| flexible models such as gradient boosting, or patients who share a care team | [flexible models and shared teams](../cross-fitting.ipynb) | a model that predicts the same rows it was trained on looks more accurate than it is, and an analysis that treats teammates as independent ignores that their results move together. Both make the interval too narrow |
| a fitted result that you want to look at, check, and explain with figures | [plot results](../plots.ipynb) | a table of estimates hides which patients had almost no chance of an offer, how the estimate moves when a setting changes, and a method whose intervals all miss on one side |

```{toctree}
:hidden:
:maxdepth: 1

/examples/learner-choice
/examples/cross-fitting
/examples/plots
```
