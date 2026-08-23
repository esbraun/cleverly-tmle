# Longitudinal regimen analysis

This example estimates the effect of two treatment strategies over two time points in the presence
of time-varying confounding and censoring.

## Design and regimens

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
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
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
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
print(result.diagnostics.run_all().summary())

print(result.diagnostics.support().to_frame())
print(result.diagnostics.score_equations().to_frame())
print(result.diagnostics.nuisance_models().to_frame())
```

The combined report summarizes as text. The three stage reports are tabular rather than
printed prose: each carries one row per regimen and node, so `to_frame()` returns them in the
backend the study was built from. The reports retain node-level details. See [Longitudinal TMLE](../technical-reference/longitudinal-tmle.md)
for the sequential regression, cumulative clever covariate, event-process extensions, external
provenance, and evidence.
