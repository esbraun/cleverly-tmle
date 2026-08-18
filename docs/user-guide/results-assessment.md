# Results, inference, and assessment

## Result structure

`CausalResult` is a read-only mapping from stable aliases to `ParameterEstimate` objects.

```python
names = list(result.estimates)
table = result.to_frame()
covariance = result.covariance(names)
point = result[names[0]]

print(point.psi, point.std_error, point.ci, point.pvalue)
```

`result.parameter_keys` maps each alias to a structured `ParameterKey`. Use those fields for
programmatic selection; display aliases are not a serialization format.

## Contrasts and simultaneous inference

The result retains influence curves and their joint covariance, so smooth contrasts and
simultaneous bands do not refit nuisance models.

```python
if len(names) >= 2:
    difference = result.contrast(lambda values: values[0] - values[1], names[:2])
```

Use cluster roles in the study design for cluster-robust variance. Use `Inference(simultaneous=True)`
when the reported family, rather than each interval separately, needs error control.

## Diagnostics

```python
validation = result.validate()
support = result.diagnostics.support()
nuisance = result.diagnostics.nuisance_models()
scores = result.diagnostics.score_equations()
all_cached = result.diagnostics.run_all()
```

Combined reports distinguish five states: `passed`, `failed`, `warning`, `not_applicable`, and
`unavailable`. “Not applicable” means the scientific question has no such analysis; “unavailable”
means the question is meaningful but the fit lacks a derivation or saved artifact.

## Sensitivity analysis

```python
all_sensitivity = result.sensitivity.run_all()
print(all_sensitivity.summary())
```

Available point-treatment analyses include positivity summaries, E-values where meaningful,
omitted-variable bounds, missingness tilts, and refit-based benchmarking. A combined cache-only
call excludes refits by default. Longitudinal operations without a published derivation are
reported unavailable rather than borrowing point-treatment formulas.

## Persistence and replayability

```python
from cleverly import load

result.save("analysis.npz")
restored = load("analysis.npz")
assert restored.parameter_keys == result.parameter_keys
assert restored.method == result.method
```

The format allow-lists structured metadata and arrays; it does not pickle arbitrary objects.
Named learner libraries round-trip exactly. Custom estimator objects and callables are recorded by
identity so cached assessment can replay, but their restored slots refuse a new fit rather than
substituting a default. `result.replayability` explains what remains available.
