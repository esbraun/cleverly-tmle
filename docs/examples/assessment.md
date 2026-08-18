# Post-fit assessment and replay

This example treats assessment as part of the analysis rather than an afterthought. It uses only
fitted artifacts until an operation is explicitly documented as a refit.

## Fit and inspect capabilities

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_weak_overlap

frame, truth = make_weak_overlap(n=2_500, seed=51)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
    ),
)
result = study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=51,
)

for capability in result.diagnostics.capabilities:
    print(capability.operation, capability.cost, capability.execution)
```

## Run cache-only assessment

```python
validation = result.validate()
diagnostics = result.diagnostics.run_all()
sensitivity = result.sensitivity.run_all()

print(validation.summary())
print(diagnostics.summary())
print(sensitivity.summary())
```

The combined reports keep `not_applicable` separate from `unavailable` and exclude refits by
default. Weak overlap should be interpreted through the target-specific support report and
truncation behavior, not from one universal propensity threshold.

## Save, restore, and replay

```python
from cleverly import load

result.save("assessed-result.joblib")
restored = load("assessed-result.joblib")

print(restored.replayability)
print(restored.diagnostics.run_all().summary())
assert restored.parameter_keys == result.parameter_keys
```

The whole result, including its nuisance estimator templates, is restored. Load only trusted
joblib artifacts and use compatible dependency versions. See
[Inference, diagnostics, and sensitivity](../technical-reference/inference-assessment.md).
