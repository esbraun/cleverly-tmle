# Quickstart

## Ask for an average treatment effect

The design owns the observed columns. The estimand owns the causal question.

```python
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2_000, seed=7)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)
effect = study.identify(ATE())
```

Identification happens before fitting. Inspect it when the analysis is consequential:

```python
print(effect.summary())
print(effect.functional)
print(effect.identification.assumptions)
print(effect.available_methods())
```

These fields state what is being estimated, the assumptions under which it has a causal
interpretation, the nuisance functions required, and the supported estimation methods.

## Estimate and inspect

```python
result = effect.estimate(random_state=7)

print(result.summary())
estimate = result["ate"]
print(estimate.psi, estimate.std_error, estimate.ci)
```

An ordinary fit returns a `CausalResult` directly. Each entry is a `ParameterEstimate` containing
the point estimate, standard error, interval, p-value, influence curve, and inference scale. For a
multi-parameter result, `result.parameter_keys` preserves arm, regimen, horizon, cause, and MSM term
as structured fields rather than parsing display aliases.

## Assess the fitted result

```python
validation = result.validate()
support = result.diagnostics.support()
scores = result.diagnostics.score_equations()
sensitivity = result.sensitivity.run_all()

print(validation.summary())
print(support.summary())
print(scores.summary())
print(sensitivity.summary())
```

Cache-only assessment never refits the estimator. Each capability reports whether it passed,
failed, raised a warning, did not apply to the question, or lacked a required fitted artifact.

## Choose the next path

- Read the [workflow](../workflow.md) before adapting the quickstart to real observational data.
- Use the [estimands and interventions guide](../user-guide/estimands.md) to choose a question.
- Use the [methods and learners guide](../user-guide/methods-learners.md) before replacing defaults.
- Consult the [technical reference](../technical-reference/index.md) for equations, assumptions,
  citations, implementation provenance, and correctness evidence.
