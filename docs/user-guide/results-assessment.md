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

Use generated outcomes to test the full pipeline against a declared effect. The dummy operation
uses independent Gaussian noise and declares zero. The simulated operation uses a standardized
adjustment function, an additive treatment term, and Gaussian noise.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate
from cleverly.validation import GaussianAdjustmentOutcome

gaussian_frame, _ = make_linear_ate(n=200, seed=21)
gaussian_study = CausalStudy(
    gaussian_frame,
    design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
)
ate_result = gaussian_study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=21,
)
generated = ate_result.diagnostics.refute(
    tests=("simulated_outcome",),
    simulated_outcome=GaussianAdjustmentOutcome(effect=0.5),
    n_replicates=40,
    random_state=21,
)
draws = generated.draws_frame("simulated_outcome")
```

That call reports `includes 0.5 yes` on 40 successful refits. Each row in `draws` records the
child seed, estimate, standard error, family, or refit failure.

Use bootstrap measurement error to assess estimate stability under declared covariate error.
The operation samples rows first. It then perturbs original adjustment variables and refits the
complete estimator.

```python
from cleverly.validation import BootstrapMeasurementError, RelativeGaussianNoise

measurement = ate_result.diagnostics.refute(
    tests=("bootstrap_measurement_error",),
    bootstrap_measurement_error=BootstrapMeasurementError(
        variables=("W1", "W2"),
        numeric_noise=RelativeGaussianNoise(standard_deviation=0.1),
    ),
    n_replicates=40,
    random_state=21,
)
measurement_draws = measurement.draws_frame("bootstrap_measurement_error")
```

The numeric multiplier uses each variable's standard deviation in its bootstrap sample.
Categorical changes act on the original variable and rebuild its complete indicator block.
Set `categorical_change_probability` on the declaration to control those changes.

The operation compares the original estimate with the empirical refit distribution.
It can pass or fail because it measures stability under the declared error.
It stays outside `DEFAULT_TESTS` because every draw resamples, perturbs, and refits.
Plain bootstrap inference does not add measurement error.

Two defaults govern generated-outcome calls, and they are separate objects. `n_replicates` defaults to
`DEFAULT_OUTCOME_REPLICATES`, which is 100 draws. `outcome_rule` defaults to
`EmpiricalInclusionRule()`, which needs 40 successful draws, uses alpha 0.05, and fails when any
refit fails. The simulated-outcome example asks for 40 draws, which is the smallest budget the
default rule accepts.

Read alpha as a width and not as a false-alarm rate. The rule passes when the declared effect
lies inside the central 95% of the refit estimates, which
[the technical reference](../technical-reference/validation-methods.md#refutation-operations)
derives. Generated outcomes apply only to identified additive contrasts for binary point
treatments.

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
`contour()`, `evalue()`, `missingness()`, `tipping_gamma()`, and `simulated_confounding()`. A
simulated surface needs an explicit strength grid, so `run_all()` never starts it.

```python
from cleverly.sensitivity import ConfounderStrengthGrid

grid = ConfounderStrengthGrid(
    treatment=(0.0, 0.05, 0.10),
    outcome=(0.0, 0.25, 0.50),
)
surface = ate_result.sensitivity.simulated_confounding(
    grid=grid,
    benchmark_covariates=("W1", "W2"),
    random_state=21,
)
movements = surface.to_frame()
calibration = surface.calibration_frame()
```

The operation draws one shared latent variable. It flips the treatment in the upper latent tail. A
Gaussian outcome subtracts the outcome strength times the latent value. A binomial outcome flips in
the same tail.

The treatment flip is non-differential misclassification. The association it induces between the
latent variable and the treatment depends on the treated fraction. That association is zero on a
balanced design. The technical reference
[measures it](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface).

The operation refits the complete estimator at each nonzero strength pair. The zero cell equals the
original estimate exactly.

Read the output as a qualitative stress surface. It is not a corrected estimate, bound, p-value,
confidence interval, robustness value, or pass/fail result. Calibration reports model-dependent
source conventions for numeric covariates. It does not change the declared grid.

A combined call excludes refits and retargets by default. Each skipped row names the flag that
would run it. Longitudinal
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
