# Migrating to the causal-question API

The foundational API release is a clean alpha break. The final commit containing the former root
constructors is tagged `pre-public-api-redesign`; use that tag to reproduce an old analysis while
you migrate it. The new API deliberately has no runtime aliases for `TMLE`, `LTMLE`, `CTMLE`,
`DRTMLE`, `tmle`, or `ltmle` at the package root.

The replacement is not a new estimator. It is a typed route to the same evidenced point and
longitudinal engines:

```text
roles -> PointTreatment or LongitudinalTreatment
question -> typed estimand
assumptions -> IdentifiedEffect
algorithm -> TMLEMethod or a named preset
output -> CausalResult directly
```

## Ordinary point-treatment fit

Before:

```python
from cleverly import TMLE

fit = TMLE(
    estimands=("ate",),
    outcome_learner="glm",
    treatment_learner="glm",
    n_folds=5,
    random_state=7,
).fit(
    frame,
    outcome="Y",
    treatment="A",
    covariates=("W1", "W2", "W3"),
)
result = fit.single()
```

After:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
    ),
)
result = study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=5,
    random_state=7,
)
```

There is no replacement call for `.single()`: the new fit already returns the object it used to
unwrap. Do not change `result = fit.single()` to `result = fit.estimate`. On a causal result,
`.estimate` is the sole `ParameterEstimate`, and `.psi()` is its numeric point value.

## Inspect identification before fitting

The concise `study.estimate(...)` path still creates and stores an identified effect. Split it
when the assumptions should be reviewed or logged first:

```python
effect = study.identify(ATE())
print(effect.summary())
print(effect.available_methods())
result = effect.estimate(method="tmle", random_state=7)
```

## Point constructor argument map

| Former location | New location | Meaning |
| --- | --- | --- |
| `.fit(outcome=)` | `PointTreatment(outcome=)` | Outcome column |
| `.fit(treatment=)` | `PointTreatment(treatment=)` | Treatment column |
| `.fit(covariates=)` | `PointTreatment(adjustment=)` | Declared adjustment set |
| `.fit(delta=)` | `PointTreatment(missingness=)` | Outcome-observation indicator |
| `.fit(intermediate=)` | `PointTreatment(intermediate=)` | Binary intermediate for a controlled direct effect |
| `.fit(weights=)` | `PointTreatment(weights=)` | Observation-weight column |
| `.fit(weights_type=)` | `PointTreatment(weights_type=)` | Weight interpretation |
| `.fit(weights_estimated=)` | `PointTreatment(weights_estimated=)` | Whether weights were estimated |
| `.fit(id=)` | `PointTreatment(cluster=)` | Independent-unit/cluster column |
| `.fit(strata=)` | `PointTreatment(strata=)` | Baseline subgroup axes |
| `.fit(family=)` | `PointTreatment(outcome_family=)` | Outcome family |
| `.fit(treatment_kind=)` | `PointTreatment(treatment_kind=)` | Discrete arm or continuous dose |
| `estimands=("ate",)` | `ATE()` | Average treatment effect |
| `estimands=("att",)` | `ATT()` | Effect among comparison-arm recipients |
| `estimands=("atc",)` | `ATC()` | Effect among reference-arm recipients |
| `estimands=("ey",)` | `CounterfactualMean()` | Mean under each arm |
| `estimands=("ey1",)` | `CounterfactualMean(treatment=1)` | Mean under the treated arm |
| `estimands=("ey0",)` | `CounterfactualMean(treatment=0)` | Mean under the untreated arm |
| `estimands=("ey_obs",)` | `NaturalCourseMean()` | Observed/natural-course mean |
| `estimands=("par",)` | `PopulationAttributableRisk()` | Population attributable risk |
| `estimands=("paf",)` | `PopulationAttributableFraction()` | Population attributable fraction |
| `estimands=("rr",)` | `RiskRatio()` | Counterfactual risk ratio |
| `estimands=("or",)` | `OddsRatio()` | Counterfactual odds ratio |
| `reference=` | `reference=` on the estimand | Reference arm or intervention |
| `interventions=` | `RegimeMean(...)` or `RegimeContrast(...)` | Regimen means or contrasts |
| `shifts=` | `ModifiedTreatmentPolicy(...)` or `ModifiedTreatmentPolicyEffect(...)` | Shift means or contrasts |
| `incremental=` | `IncrementalMean(...)` or `IncrementalEffect(...)` | Incremental means or contrasts |
| `msm=` | `MSMProjection(model)` | Working-model projection |

Roles are validated once when `CausalStudy` is constructed. Estimation options cannot reassign
them later.

## Missing outcomes and controlled direct effects

Before, `intermediate=` changed both the design and the estimand and produced a result set keyed by
intermediate level. After, the level is part of the typed question:

```python
from cleverly import CausalStudy, ControlledDirectEffect, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        missingness="Delta",
        intermediate="Z",
    ),
)
at_zero = study.estimate(ControlledDirectEffect(intermediate=0.0))
at_one = study.estimate(ControlledDirectEffect(intermediate=1.0))
```

This prevents code from choosing an intermediate level by accident. It does not reinterpret the
two controlled direct effects as a direct/indirect decomposition.

## Longitudinal fit

Before:

```python
from cleverly import LTMLE

result = LTMLE(
    {"always": 1, "never": 0},
    reference="always",
    outcome_learner="glm",
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=3,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=("A1", "A2"),
    baseline=("W1", "W2"),
    time_varying=((), ("L2",)),
    censoring=("C1", "C2"),
)
```

After:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast

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
result = study.estimate(
    RegimeContrast({"always": 1, "never": 0}, reference="always"),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=3,
    random_state=0,
)
```

## Longitudinal argument map

| Former location | New location |
| --- | --- |
| `LTMLE(regimens, ...)` | `RegimeMean(regimens, ...)`, `RegimeContrast(regimens, ...)`, or `MSMProjection(model, regimens=...)` |
| `LTMLE(reference=)` | `reference=` on `RegimeMean`/`RegimeContrast` |
| `LTMLE(horizons=)` | `horizons=` on the longitudinal estimand |
| `LTMLE(msm=)` | `MSMProjection(model, regimens=..., horizons=...)` |
| `.fit(outcome=)` | `LongitudinalTreatment(outcome=)` |
| `.fit(treatment=)` | `LongitudinalTreatment(treatment=)` |
| `.fit(baseline=)` | `LongitudinalTreatment(baseline=)` |
| `.fit(time_varying=)` | `LongitudinalTreatment(time_varying=)` |
| `.fit(censoring=)` | `LongitudinalTreatment(censoring=)` |
| `.fit(id=)` | `LongitudinalTreatment(cluster=)` |
| `.fit(weights=)` | `LongitudinalTreatment(weights=)` |
| `.fit(family=)` | `LongitudinalTreatment(outcome_family=)` |

An outcome sequence still declares survival. A cause-to-sequence mapping still declares competing
risks. Their numeric estimates, influence curves, standard errors, and confidence intervals use
the unchanged longitudinal engine.

## Method and runtime argument map

Most common estimator keywords remain valid as convenience shortcuts, but they normalize into an
immutable method object:

| Former constructor argument | Configuration field |
| --- | --- |
| `outcome_learner` | `ModelSpec.outcome_learner` |
| `treatment_learner` | `ModelSpec.treatment_learner` |
| `missingness_learner` | `ModelSpec.missingness_learner` |
| `intermediate_learner` | `ModelSpec.intermediate_learner` |
| `pseudo_learner` | `ModelSpec.pseudo_learner` |
| `censoring_learner` | `ModelSpec.censoring_learner` |
| `density_bins` | `ModelSpec.density_bins` |
| `cross_fit` | `CrossFitting.enabled` |
| `n_folds` | `CrossFitting.n_folds` |
| `learner_folds` | `CrossFitting.learner_folds` |
| `repeats` | `CrossFitting.repeats` |
| `stratify_folds` | `CrossFitting.stratify_by` |
| `targeting_scheme` | `CrossFitting.targeting_scheme` |
| `cv_evaluation` | `CrossFitting.fold_evaluation` |
| `fluctuation` | `Targeting.fluctuation` |
| `targeting` | `Targeting.algorithm` |
| `g_bounds` | `Targeting.g_bounds` |
| `q_bounds` | `Targeting.q_bounds` |
| `alpha` | `Targeting.submodel_alpha` only through the typed field |
| `alpha_sig` | `Inference.alpha` |
| `simultaneous` | `Inference.simultaneous` |
| `n_multiplier` | `Inference.n_multiplier` |
| `random_state` | `Runtime.random_state` |
| `run_id` | `Runtime.run_id` |
| `n_jobs` | `Runtime.n_jobs` |

The convenience spelling intentionally differs for the two alpha values: `alpha=` now means
`Inference.alpha`, the significance level, and `submodel_alpha=` means the logistic-submodel
bound. The ambiguous old `alpha_sig=` shortcut is rejected. A typed `TMLEMethod` makes both fields
explicit.

`CollaborativeTMLEMethod` replaces root `CTMLE`; `DRTMLEMethod` replaces root `DRTMLE`. Their
variant-only arguments are fields on those typed method objects.

## Learner objects and persistence

Nuisance slots no longer accept `"glm"`, `"fast"`, `"default"`, or `"rich"`. Pass an
sklearn-compatible estimator object. Use `LinearRegression()` for mean regressions and
`LogisticRegression()` for treatment, censoring, missingness, and other probability mechanisms.
Omitting a learner, or passing `SuperLearner()`, uses the concrete histogram-gradient-boosting,
random-forest, and lasso ensemble. The former `"fast"` and `"rich"` sets have no exact alias;
construct the desired `SuperLearner(library=[...])` explicitly.

Saved results now use whole-result joblib artifacts:

```python
from cleverly import load

result.save("analysis.joblib")
restored = load("analysis.joblib")
```

Legacy `.npz` results require the cleverly version that wrote them. Joblib loading can execute
arbitrary Python code, so load only trusted artifacts under compatible dependency versions.

## Changed defaults and validations

Post-fit assessment has one authoritative path. The alpha API removes the legacy
`result.validation` suite and the callable longitudinal `result.diagnostics()` spelling:

| previous call | alpha call |
| --- | --- |
| `result.validation.score_check()` | `result.diagnostics.score_equations()` |
| `result.validation.correction_check()` | `result.diagnostics.corrections()` |
| `result.validation.nuisance()` | `result.diagnostics.nuisance_models()` |
| `result.validation.refute(...)` | `result.diagnostics.refute(...)` |
| `result.diagnostics()` on LTMLE | `result.diagnostics.stagewise().to_frame()` |
| `result.sensitivity.positivity()` and axis-specific support methods | `result.diagnostics.support()` |
| `result.sensitivity.truncation_curve(...)` | `result.diagnostics.truncation_curve(...)` |
| `result.sensitivity.omitted_variable(...)` | `result.sensitivity.omitted_confounding(...)` |
| `result.sensitivity.missingness_tilt(...)` | `result.sensitivity.missingness(...)` |

`result.validate()` remains the inexpensive, no-refit battery. Sensitivity exposes only the
capability-aware explicit methods documented in the results and assessment guide; removed aliases
are not retained as runtime compatibility shims.

- An empty adjustment set is refused unless `PointTreatment(randomized=True)` is declared.
- Ordinary estimation returns a causal result directly; no `.single()` wrapper remains.
- `alpha=` is the inference significance level. Use `submodel_alpha=` for the targeting bound.
- Longitudinal `g_bounds="auto"` still has no scientific meaning; the adapter normalizes the
  established longitudinal default `(0.01, 1.0)`.
- Unknown method names and unsupported method/estimand combinations fail before fitting.
- A point design with `intermediate=` requires `ControlledDirectEffect`; it cannot accidentally
  turn an ATE request into another parameter.
- Structured parameter metadata survives persistence. Alias parsing is not a migration strategy.

## Static migration audit

Run the bundled audit on Python files or directories:

```bash
python scripts/migrate_public_api.py analysis.py src/
```

It reports root imports, one-call helpers, `.single()` calls, and estimator `.fit(...)` role
arguments with line numbers. It does not rewrite scientific intent: a string such as `"ate"`,
`"ey"`, or `"ate_shift"` must be mapped to the typed question the analysis actually means.
The command exits nonzero while migration findings remain, so it can be used as a temporary local
gate.

## Advanced implementation imports

The old engines remain internal adapters under `cleverly.estimators` and `cleverly.longitudinal`
so the package can preserve its tested arithmetic. They are not a supported alternative public
workflow. Importing them directly is appropriate only for package development, scientific oracle
tests, and migration comparison—not for new applied analyses.
