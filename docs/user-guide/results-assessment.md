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

An ordinary-TMLE fit can declare fixed probability weights. The surface keeps each normalized
weight on its row during every treatment replacement, outcome replacement, and complete refit.
Read `surface.target_measure` to distinguish `unweighted` from `fixed_empirical_tilt`.
`surface.weight_report` records the weight column, kind, supplied scale, and concentration.
The label follows the declared weight column, so read `surface.weight_report.is_weighted` to learn
whether the realized tilt is nonconstant.

Under `fixed_empirical_tilt`, each cell evaluates its parameter on the perturbed weighted
empirical law. This operation does not reproduce the original sampling or selection mechanism.
It conditions on the fixed weights that the fit stores.

The surface keeps the fitted estimator's `repeats` setting. Each non-anchor cell runs one complete
refit, and the estimator combines its draws with a coordinatewise median. Read `surface.n_repeats`
and `surface.repeat_aggregation` to confirm that provenance. `repeat_aggregation` names the rule
for more than one draw. A single draw needs no aggregation. The shared root seed gives every
non-anchor cell the same repeat seed sequence. A treatment or outcome perturbation can still
change stratified folds.

Pass the seed of the original fit, or pass no `random_state` at all. The surface reproduces the
folds of that fit under the seed of that fit alone. A different seed gives every non-anchor cell new folds, and
a fit that declared no seed has the same effect. The anchor keeps the original folds, so movement
near the anchor can carry a fold artifact.

The operation draws one shared latent variable. For binary treatment, it flips the treatment in the
upper latent tail. A Gaussian outcome subtracts the outcome strength times the latent value. A
binomial outcome flips in the same tail.

A binary `CounterfactualMean(treatment=...)` fit reports one `ey1` or `ey0` mean. The facade selects
that sole mean when you pass the grid. A `CounterfactualMean()` fit reports `ey[...]` for both arms.
Pass one exact alias through `estimand=` for that multi-mean result. The intervention arm stays fixed
while the operation perturbs the observed treatment and outcome.

A binary `RiskRatio` or `OddsRatio` fit also needs only the grid. Each cell reports the refitted
ratio in `estimate`. Its `displacement` is the refitted log ratio minus the original log ratio.
The surface and its frame record `movement_scale="log_ratio"`. Exponentiate a displacement to get
the refitted ratio divided by the original ratio.

A continuous-dose fit uses signed treatment strengths. Pass one exact modified-policy mean or
contrast alias because a continuous result has no bare `ate` parameter.

```python
from cleverly import ModifiedTreatmentPolicyEffect
from cleverly.datasets import make_shift_dose
from cleverly.interventions import Shift

shift_frame, _ = make_shift_dose(n=200, seed=21)
shift_study = CausalStudy(
    shift_frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
        treatment_kind="continuous",
    ),
)
shift_result = shift_study.estimate(
    ModifiedTreatmentPolicyEffect(
        shifts=(
            Shift(0.0, cap=3.0, name="natural course"),
            Shift(0.5, cap=3.0, name="up half"),
        )
    ),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=2,
    learner_folds=2,
    random_state=21,
    simultaneous=False,
)
shift_alias = "ate_shift[up half vs natural course]"
shift_surface = shift_result.sensitivity.simulated_confounding(
    estimand=shift_alias,
    grid=ConfounderStrengthGrid(
        treatment=(-0.25, 0.0, 0.25),
        outcome=(0.0, 0.25, 0.50),
    ),
    benchmark_covariates=("W1", "W2"),
    random_state=21,
)
```

A `ModifiedTreatmentPolicy` fit reports policy means instead. Select one mean with an alias such
as `ey_shift[up half]`. The surface reports that mean and its signed displacement in each cell.

Select a policy whose `delta` is nonzero. A zero-delta shift assigns every unit its own dose, so
its mean is $E[Y]$ and no common cause can move it through the treatment. The surface refuses that
mean by name. It still accepts an `ate_shift[...]` contrast that uses the zero-delta policy as its
reference, because the contrast keeps its treatment dependence.

The continuous treatment law is $A'=A+k_AU$. It keeps the declared modified treatment policies
fixed during each ordinary-TMLE refit. The outcome laws and common-randomness contract stay the
same as the binary surface.

On a binary fit, the treatment flip is non-differential misclassification. The association it
induces between the latent variable and the treatment depends on the treated fraction. That
association is zero on a balanced design. The technical reference
[measures it](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface).

Each cell reports its own realised association in `induced_treatment_association`. The frame
carries the same value in a column of that name, and `summary()` prints it. On a binary fit, check
the column before you read a treatment movement as confounding. A value near the anchor can
reflect misclassification alone.

For a fixed-weight fit, the association and numeric calibration use the weighted empirical law.
Calibration weights feature scaling, model fitting, prediction-change fractions, correlations,
and standard deviations. A common scale change to the weights leaves these values unchanged.

On a continuous fit, the latent variable changes the dose by construction, so the association
grows with the treatment strength. A confounding path also needs the latent variable to enter the
outcome, and only a nonzero outcome strength puts it there. A cell in the zero outcome-strength
column therefore carries no confounding path, whatever its association. Its movement reports the
dose perturbation alone. The technical reference states the same
[reading rule](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface).

The zero treatment-strength column carries no confounding path either. Its movement reports the
outcome perturbation alone. The Gaussian outcome law subtracts a level from every row, and an
`ate_shift[...]` contrast removes most of that level. An `ey_shift[...]` policy mean keeps it, so
read the zero treatment-strength column of a policy-mean surface as an artifact of the outcome
law.

The operation refits the complete estimator at each nonzero strength pair. The zero cell equals the
original estimate exactly. Additive surfaces record `movement_scale="estimate_difference"`.

Read the output as a qualitative stress surface. It is not a corrected estimate, bound, p-value,
confidence interval, robustness value, or pass/fail result. Calibration reports model-dependent
source conventions for numeric covariates. It does not change the declared grid.

The operation refuses estimated weights before it draws the latent vector. The fitted result does
not store the weight model needed after a perturbation. It also refuses fixed weights with
collaborative TMLE or DR-TMLE. Their weighted composition lacks estimator-specific source support.
Clustered fits remain a theory stop. No cited source defines the latent cause at the row, cluster,
or mixed level.

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
