# Quickstart

## Ask for an average treatment effect

The design owns the observed columns. The estimand owns the causal question.

```python
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate
from sklearn.linear_model import LinearRegression, LogisticRegression

frame, truth = make_linear_ate(n=2_000, seed=7)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)
effect = study.identify(ATE(reference=0))
```

Identification happens before fitting. Inspect it before you estimate anything.

```python
print(effect.summary())
print(effect.functional)
print(effect.identification.assumptions)
print(effect.available_methods())
```

These fields state the estimand, the assumptions for its causal reading, the required nuisance
functions, and the supported methods.

## Estimate and inspect

Both nuisance functions of `make_linear_ate` are linear, so the linear learners below are
correctly specified. On data you did not simulate, use flexible learners.

```python
result = effect.estimate(
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=7,
)

print(result.summary())
estimate = result["ate"]
print(estimate.psi, estimate.std_error, estimate.ci)
print("population ATE:", truth["ate"])
```

The fit cross-fits both nuisances over 10 folds by default. Each result entry is a
`ParameterEstimate` with the point estimate, standard error, interval, p-value, and influence
curve.

## Assess the fitted result

```python
assessment = result.assess()
print(assessment.summary())
print("needs attention:", tuple(item.name for item in assessment.attention))
print(assessment.report("support").summary())
```

`assess()` collects validation, diagnostics, and sensitivity in one report. It refits no nuisance
model unless you opt in. Each row carries one of seven statuses, and `completed` is not a pass. The
[status contract](../technical-reference/validation-methods.md#the-status-contract) defines them.

## Choose the next path

- Read the [workflow](../workflow.md) before adapting the quickstart to real observational data.
- Read [point-treatment TMLE](../examples/point-treatment-tmle.md) for a complete applied analysis.
- Use the [estimands and interventions guide](../user-guide/estimands.md) to choose a question.
- Use the [methods and learners guide](../user-guide/methods-learners.md) before replacing defaults.
- Consult the [technical reference](../technical-reference/index.md) for equations, assumptions,
  citations, implementation provenance, and correctness evidence.
