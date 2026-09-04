# Results, inference, and assessment

## Run the assessment battery

Call `assess()` to collect validation, diagnostics, and sensitivity in one report.

```python
battery = result.assess()
print(battery.summary())
support = battery.report("support")
```

The default call reads stored artifacts and runs the cheap E-value retarget when applicable. It does not refit nuisance models. `battery.attention` contains explicit failures and warnings. `battery.omissions`
contains analyses that were not applicable or were unavailable.

Pass analyst choices through `arguments`. Opt in to each expensive work class by name.

```text
battery = result.assess(
    include_refits=True,
    include_retargets=True,
    random_state=21,
    arguments={
        "benchmark": {"covariates": ("W1", "W2")},
        "simulated_confounding": {"grid": grid},
    },
)
```

The common seed reaches `refute`, `benchmark`, and `simulated_confounding`. A seeded fit reuses
its fit seed when you omit this argument. An unseeded fit draws a seed and records it on the
returned row.

`assess()` refuses `arguments` for `score_equations`. The validation battery owns that name and
runs it argument-free. The battery presents the validation row for it, so it would compute your
answer and then hide it. Call `result.diagnostics.run_all(arguments=...)`, or call the operation
itself.

The battery also owns `support` and `nuisance_models`, but neither one accepts an argument. An
argument for either is a `TypeError` from the signature. An empty mapping applies nothing, so
`assess()` accepts it for all three names.

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
corrections = (
    result.diagnostics.corrections()
    if result.diagnostics.capability("corrections").available
    else None
)
stability = result.diagnostics.truncation_curve()
all_cached = result.diagnostics.run_all()
```

The correction diagnostic is available only for a DR-TMLE fit whose guard actually subtracts a
correction term. Ordinary and collaborative TMLE report it as `not_applicable`.

Combined reports distinguish six states. `passed` and `failed` belong to checks with an explicit
verdict. `completed` means a descriptive analysis ran without an inferential verdict. `warning`
uses an existing diagnostic rule. An expected refusal becomes `unavailable`.
The aggregate run then continues with other accepted diagnostics. Direct calls still raise the
precise refusal. `not_applicable` and `unavailable` appear in `omissions`.

Known omissions carry the capability's reason. Examples include an E-value without a supported
contrast and a missingness analysis without missing outcomes. An operation can also refuse after
invocation, such as omitted-confounding sensitivity on median-combined repeats. That row becomes
an `unavailable` omission, retains its invocation arguments, and names the direct call. Other accepted
diagnostics still run. Structural errors, such as invalid argument names, still stop the report.

A combined report runs summaries and cheap retargets by default. The two costlier
classes are named separately because they are disjoint. `refute()` and `benchmark()` refit
nuisance models, while `truncation_curve()`, `missingness()` and `tipping_gamma()` retarget cached
ones. These moderate retargets require `include_retargets=True`. The E-value retarget is cheap and runs by default.
Explicit odds-ratio and reported risk-ratio E-values summarize existing estimates:

```python
everything = result.diagnostics.run_all(include_refits=True, include_retargets=True)
```

Pass required choices with `arguments={"operation": {...}}`. The report row retains the effective
arguments and the returned object. Use `report.report(name)` to retrieve that object.

A refusal names the first thing that is wrong. An operation that needs a choice you did not pass
names that choice, and not a cost flag. No flag supplies the covariates for `benchmark`, so the
row says so under every flag.

A completed `tipping_gamma` search can return `None` when its interval contains no tipping point.
Use `battery.report(name, surface="diagnostics")` to retrieve a diagnostic that also appears under validation.

The E-value row records its source alias, such as `estimand="ate"`, for replay.
A derived ratio can have a different output alias, such as `rr`.
For a default odds-ratio source, the row keeps `estimand=None` because an explicit alias chooses the approximation.
The returned E-value records that source in `source_estimand`.

The omitted-confounding row says "at the default strengths" when you omit both strength arguments.
It names the omitted strength when you supply only one. The effective arguments retain both numeric values.

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

An ordinary-TMLE fit can declare fixed probability weights. Binary complete-outcome collaborative
TMLE and DR-TMLE fits can also declare them. The surface keeps each normalized weight on its row
during every treatment replacement, outcome replacement, and complete refit.
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

A binary `RiskRatio` or `OddsRatio` fit without baseline strata needs only the grid. Each cell reports the refitted
ratio in `estimate`. Its `displacement` is the refitted log ratio minus the original log ratio.
The surface and its frame record `movement_scale="log_ratio"`. Exponentiate a displacement to get
the refitted ratio divided by the original ratio.

For a fit with baseline `strata=`, pass the complete conditional alias from `result.estimates`.
The operation holds the baseline stratum fixed and refits all rows at every non-anchor cell.
`surface.stratum` records the selected values. `surface.association_population` identifies the
population used for the treatment correlation. Numeric calibration always describes the full
original fitted population, as `surface.calibration_population` records.

Ordinary-TMLE ATT and ATC surfaces use the treated or control group after each perturbation.
Their movement includes population change. Read `surface.conditioning_arm` and each cell's
`target_population_fraction` before interpreting that movement. A baseline stratum remains fixed
even when treatment-group membership changes within it. C-TMLE and DR-TMLE still refuse ATT and ATC.

This example selects one baseline stratum through its structured key.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATT, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate
from cleverly.sensitivity import ConfounderStrengthGrid

population_frame, _ = make_linear_ate(n=160, seed=21)
population_frame["V"] = np.where(population_frame["W1"] > 0, "high", "low")
population_result = CausalStudy(
    population_frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4", "V"),
        strata=("V",),
    ),
).estimate(
    ATT(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=2,
    random_state=21,
    simultaneous=False,
)
population_alias = next(
    alias for alias, key in population_result.parameter_keys.items() if key.stratum == ("high",)
)
population_surface = population_result.sensitivity.simulated_confounding(
    estimand=population_alias,
    grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.25)),
)
population_cells = population_surface.to_frame()
```

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
not store the weight model needed after a perturbation.

A fixed-weight collaborative fit reruns its selector or outcome-adaptive treatment fit at each
cell. It uses the normalized row weights in every nuisance fit, loss, penalty, targeting step, and
plug-in.

The R `ctmle` and archived `ctmle3` sources provide no weighted comparator. The fixed-weight
collaborative surface makes no numerical parity claim with those implementations.

Clustered fits remain a theory stop. No cited source defines the latent cause at the row, cluster,
or mixed level.

A combined call excludes refits and retargets by default. A row skipped for cost alone names the
flag that would run it. A row that also needs a choice names the choice instead. Longitudinal
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

Read `result.diagnostics.capabilities` before you run an operation on a restored artifact. Each
row declares the replay it needs, and `available` is `False` when the artifact cannot supply it.
The row says which slot is missing, so you learn the answer before the operation runs.

The saved artifact carries the assessment cache. A result you derive with `dataclasses.replace`
does not. The cache key records the operation and its arguments, and it records nothing about the
result that answered them, so a derived result starts with an empty cache of its own.

The joblib artifact contains the complete result graph, including nuisance estimator templates,
so successfully restored results retain refit-based assessment. Joblib uses pickle internally:
loading can execute arbitrary code, so load only trusted artifacts in an environment with
compatible cleverly, sklearn, Python, and third-party estimator versions. Legacy `.npz` results
must be loaded with the cleverly version that created them.
