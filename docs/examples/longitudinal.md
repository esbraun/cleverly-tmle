# Longitudinal regimen analysis

This example estimates the effect of two treatment strategies over two time points in the presence
of time-varying confounding and censoring.

## Design and regimens

```python
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=3_000, seed=41)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
question = RegimeContrast(
    {"always": 1, "never": 0},
    reference="never",
)
effect = study.identify(question)
print(effect.summary())
```

At the second node, `L2` is a pre-treatment history variable for `A2`. Censoring is modeled at both
nodes. The regimen contrast is “always treat” minus “never treat.”

## Fit

```python
result = effect.estimate(
    outcome_learner="glm",
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=3,
    learner_folds=3,
    random_state=41,
)

print(result.summary())
for alias, key in result.parameter_keys.items():
    print(alias, key.value, key.reference, key.horizon, key.cause)
```

## Stagewise assessment

```python
support = result.diagnostics.support()
scores = result.diagnostics.score_equations()
nuisance = result.diagnostics.nuisance_models()

print(support.summary())
print(scores.summary())
print(nuisance.summary())
```

The reports retain node-level details. See [Longitudinal TMLE](../technical-reference/longitudinal.md)
for the sequential regression, cumulative clever covariate, event-process extensions, external
provenance, and evidence.
