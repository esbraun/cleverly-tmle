# Choosing models and checking results

These tutorials choose the prediction models inside a fit, such as gradient boosting. They keep the
interval honest when the models are flexible, when patients share a team, or when one variable
nearly decides who gets the treatment. The last one checks and explains a result with figures.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| which prediction models to put inside the fit, such as a blend of a straight line, splines and gradient boosting, or many recorded columns to screen | [choosing the prediction models](../learner-choice.ipynb) | a screen that picks the treatment model's columns by a straight-line fit to the outcome can drop real confounders whose link to the outcome is curved, leaving them to the outcome model alone |
| flexible models such as gradient boosting, or patients who share a care team | [flexible models and shared teams](../cross-fitting.ipynb) | a model that predicts the same rows it was trained on looks more accurate than it is, and an analysis that treats teammates as independent ignores that their results move together. Both make the interval too narrow |
| one recorded variable strongly predicts who gets the treatment | [a strong predictor of treatment](../collaborative-tmle.ipynb) | some patients become nearly certain or nearly unable to get the treatment, so the few exceptions get very large weights. In 2,000 simulated repeats, the 95% interval of standard TMLE contained the truth only 93% of the time, so it was slightly too narrow |
| a fitted result that you want to look at, check, and explain with figures | [plot results](../plots.ipynb) | a table of estimates hides which patients had almost no chance of an offer, how the estimate moves when a setting changes, and a method whose intervals all miss on one side |

```{toctree}
:hidden:
:maxdepth: 1

Choosing the prediction models </examples/learner-choice>
Flexible models and shared teams </examples/cross-fitting>
A strong predictor of treatment </examples/collaborative-tmle>
Plot results </examples/plots>
```
