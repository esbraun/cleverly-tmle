# Point-treatment analysis

This workflow estimates a nonlinear observational ATE, inspects the identification statement,
uses an explicit method configuration, and compares the estimate with the known simulation truth.

## Data and question

```python
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=3_000, seed=21)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)
effect = study.identify(ATE(reference=0))

print(effect.summary())
print(effect.identification.assumptions)
```

The estimand compares the mean outcome under treatment 1 with the mean under treatment 0 in the
population represented by `frame`. The causal interpretation requires consistency, conditional
exchangeability given the four adjustment variables, and treatment positivity.

## Estimation

```python
from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(outcome_learner="glm", treatment_learner="glm"),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
    inference=Inference(alpha=0.05),
    runtime=Runtime(random_state=21, n_jobs=1),
)
result = effect.estimate(method=method)

estimate = result["ate"]
print(result.summary())
print("estimate:", estimate.psi)
print("95% CI:", estimate.ci)
print("simulation truth:", truth)
```

The truth comparison is available because this is a data generator; it is not part of an applied
analysis. In real data, identification and diagnostic arguments replace truth inspection.

## Diagnostics and persistence

```python
print(result.diagnostics.support().summary())
print(result.diagnostics.nuisance_models().summary())
print(result.diagnostics.score_equations().summary())

result.save("nonlinear-ate.npz")
```

See [Point-treatment TMLE](../technical-reference/point-treatment.md) for the parameter and
influence function.
