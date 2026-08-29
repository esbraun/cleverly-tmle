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
programmatic selection. Display aliases are not a serialization format.

`result.estimate` is the sole `ParameterEstimate` only when the result holds one parameter, and
`result.psi()` is its numeric point value. With several parameters, index the alias or pass a name
to `psi`.

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
corrections = result.diagnostics.corrections()
stability = result.diagnostics.truncation_curve()
all_cached = result.diagnostics.run_all()
```

Combined reports distinguish five states: `passed`, `failed`, `warning`, `not_applicable`, and
`unavailable`. “Not applicable” means the scientific question has no such analysis; “unavailable”
means the question is meaningful but the fit lacks a derivation or saved artifact.

An `unavailable` row reached that state one of two ways, and its `detail` says which. A row whose
capability declaration already refuses it carries that declaration’s own reason. A row that had to
inspect the fit before it could refuse is prefixed `refused on inspection:`, and it names the
operation to call directly for the refusal in full. Two such rows are an E-value on a fit that
reported no contrast, and a missingness tilt on a fit with no missing outcomes. Nothing else is
caught: an error a capability did not declare propagates, because a report that renders a bug as
“unavailable” states a scientific conclusion about the fit that nobody established.

A combined report runs only the operations that summarise stored artifacts. The two costlier
classes are named separately because they are disjoint. `refute()` and `benchmark()` refit
nuisance models, while `truncation_curve()`, `missingness()` and `tipping_gamma()` retarget cached
ones:

```python
everything = result.diagnostics.run_all(include_refits=True, include_retargets=True)
```

`run_all` passes no seed, so `refute()` draws from the seed of the fit. Give the fit a seed if
you need the same refutation twice.

`score_equations(tolerance=...)` gates both result families, but on the scale each one’s score
lives on. A point-treatment fit compares the score in the outcome’s own units against
`tolerance * se / sqrt(n)`; a longitudinal fit bounds each node’s relative score, the quantity
the sequential targeting loop itself gates on, and can only tighten the fit’s own convergence
verdict rather than overturn it.

## Sensitivity analysis

```python
all_sensitivity = result.sensitivity.run_all()
print(all_sensitivity.summary())
```

Support and truncation stability live under `result.diagnostics`. Point-treatment sensitivity
methods are explicit: `omitted_confounding()`, `robustness_value()`, `elements()`, `benchmark()`,
`contour()`, `evalue()`, `missingness()`, and `tipping_gamma()`. A combined call excludes refits and
retargets by default, and each skipped row names the flag that would run it. Longitudinal
operations without a published derivation are reported unavailable rather than borrowing
point-treatment formulas, and each refusal gives the reason its own capability row declares.

## Persistence and replayability

```python
from cleverly import load

result.save("analysis.joblib")
restored = load("analysis.joblib")
assert restored.parameter_keys == result.parameter_keys
assert type(restored.method) is type(result.method)
assert type(restored.method.models.outcome_learner) is type(result.method.models.outcome_learner)
```

The joblib artifact contains the complete result graph, including nuisance estimator templates,
so successfully restored results retain refit-based assessment. Joblib uses pickle internally:
loading can execute arbitrary code, so load only trusted artifacts in an environment with
compatible cleverly, sklearn, Python, and third-party estimator versions. Legacy `.npz` results
must be loaded with the cleverly version that created them.
