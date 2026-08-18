# User guide

The public workflow has four visible stages:

```text
CausalStudy -> identify(typed estimand) -> IdentifiedEffect -> estimate(method) -> CausalResult
```

The study design owns column roles. The estimand owns the causal question. The method owns
learning, targeting, inference, and runtime settings. Separating them is what lets `cleverly`
validate unsupported combinations before it starts fitting nuisance models.

All examples below use ordinary supported behavior, but documentation examples are explanatory;
the corresponding behavioral guarantees live in the fast and slow test tiers.

## Point treatment

```python
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2_000, seed=3)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)

effect = study.identify(ATE())
print(effect.summary())
result = effect.estimate(random_state=3)
print(result.summary())
```

Identification is inspectable before estimation. `effect.functional` states the observed-data
functional, `effect.identification` states its assumptions and required nuisances, and
`effect.available_methods()` reports supported and refused methods with reasons.

An ordinary fit returns a causal result directly. `result.estimate` is the sole
`ParameterEstimate` only when the result has one parameter; `result.psi()` is its numeric point
value. With several parameters, index the stable alias or pass a name to `psi`.

## Arm-indexed estimands

Use typed objects rather than target strings:

```python
from cleverly import (
    ATC,
    ATT,
    CounterfactualMean,
    NaturalCourseMean,
    OddsRatio,
    PopulationAttributableFraction,
    PopulationAttributableRisk,
    RiskRatio,
)

questions = (
    ATT(),
    ATC(),
    CounterfactualMean(),
    CounterfactualMean(treatment=1),
    NaturalCourseMean(),
    PopulationAttributableRisk(),
    PopulationAttributableFraction(),
    RiskRatio(),
    OddsRatio(),
)

for question in questions:
    fitted = study.estimate(question, random_state=3, simultaneous=False)
    print(question.name, list(fitted.estimates))
```

For a multi-valued treatment, the original arm labels and the chosen reference appear in
`ParameterKey.value` and `ParameterKey.reference`. They are never reconstructed from the printed
alias.

```python
from cleverly.datasets import make_multi_arm

frame, truth = make_multi_arm(n=2_000, seed=9)
multi = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
    ),
)
result = multi.estimate(ATE(reference="low"), random_state=9)
for alias, key in result.parameter_keys.items():
    print(alias, key.value, key.reference)
```

## Randomized studies and strata

An empty adjustment set is an identification claim, so it must be declared:

```python
randomized = CausalStudy(
    frame[["Y", "A"]],
    design=PointTreatment(outcome="Y", treatment="A", randomized=True),
)
result = randomized.estimate(
    ATE(),
    outcome_learner="glm",
    treatment_learner="glm",
    n_folds=3,
)
```

Use `strata=` for subgroup parameters and `cluster=` for cluster-robust inference.

## Observation weights, and which population they define

Observation weights are design roles too:

```python
design = PointTreatment(
    outcome="Y",
    treatment="A",
    adjustment=("W1", "W2", "W3"),
    weights="sampling_weight",
    cluster="household",
    strata=("region",),
)
```

Weights define the target population and flow through nuisance losses, targeting, influence
curves, and covariance. Estimand or MSM weights are different objects and are not interchangeable.

## Dynamic and stochastic regimes

Point-treatment regimens are explicit intervention objects:

```python
from cleverly import RegimeContrast, RegimeMean
from cleverly.interventions import Rule, Static, Stochastic

plans = (
    Static(1, name="treat all"),
    Static(0, name="treat none"),
    Rule(lambda data: (data["W1"] > 0).astype(float), name="treat if W1 positive"),
    Stochastic(0.6, name="assign with probability 0.6"),
)

means = study.estimate(RegimeMean(plans, reference="treat none"), random_state=3)
effects = study.estimate(RegimeContrast(plans, reference="treat none"), random_state=3)
```

The mean and contrast are different typed questions even though the analytic engine can share
nuisance work internally.

## Shifting a continuous dose

A shift is a modified treatment policy for a continuous dose, not an arm contrast:

```python
from cleverly import ModifiedTreatmentPolicy, ModifiedTreatmentPolicyEffect
from cleverly.datasets import make_shift_dose
from cleverly.interventions import Shift

frame, truth = make_shift_dose(n=2_000, seed=5)
dose = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
        treatment_kind="continuous",
    ),
)
policies = (Shift(0.0, cap=None), Shift(0.25, cap=5.0), Shift(0.5, cap=5.0))
means = dose.estimate(ModifiedTreatmentPolicy(policies), density_bins=40, random_state=5)
effects = dose.estimate(ModifiedTreatmentPolicyEffect(policies), density_bins=40, random_state=5)
```

Arm-indexed estimands on a continuous-dose design are refused during identification.

## Tilting the odds of treatment

Incremental interventions multiply the observed treatment odds by `delta` while preserving
support:

```python
from cleverly import IncrementalEffect, IncrementalMean
from cleverly.interventions import Incremental

tilts = (Incremental(1.0), Incremental(2.0), Incremental(0.5))
means = study.estimate(IncrementalMean(tilts), random_state=3)
effects = study.estimate(IncrementalEffect(tilts), random_state=3)
```

These parameters have their own score equations and one-sided nuisance robustness; they are not
ordinary ATEs evaluated at different propensities.

## Marginal structural models

An MSM is a projection of counterfactual means onto a declared working model:

```python
import numpy as np

from cleverly import MSMProjection
from cleverly.msm import MSM

dose_by_arm = {"low": 0.0, "medium": 1.0, "high": 2.0}
model = MSM(
    design=lambda arm, data: np.column_stack(
        [np.ones(len(data)), np.full(len(data), dose_by_arm[arm])]
    ),
    terms=("(intercept)", "dose"),
)
result = multi.estimate(MSMProjection(model), random_state=9)
```

The result reports structured MSM terms. A coefficient is a projection parameter even when the
working model is misspecified; it is not silently relabeled as an arm effect.

## Missing outcomes and controlled direct effects

### Missing outcomes, an intermediate, and weights on a dose

Missingness and intermediate columns belong to the design:

```python
from cleverly import ControlledDirectEffect
from cleverly.datasets import make_cde

frame, truth = make_cde(n=2_000, seed=4)
cde_study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
        intermediate="Z",
    ),
)
at_zero = cde_study.estimate(ControlledDirectEffect(intermediate=0.0), random_state=4)
at_one = cde_study.estimate(ControlledDirectEffect(intermediate=1.0), random_state=4)
```

A design with `intermediate=` must identify a `ControlledDirectEffect` explicitly. The two levels
are distinct causal parameters, not a direct/indirect decomposition.

## Treatment given over time

`LongitudinalTreatment` records the temporal roles; the estimand records the regimens:

```python
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=2_000, seed=11)
longitudinal = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
result = longitudinal.estimate(
    RegimeContrast({"always": 1, "never": 0}, reference="always"),
    outcome_learner="glm",
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=3,
    learner_folds=3,
    random_state=0,
)
```

Dynamic rules receive only history available at their treatment node. The resolved assignment
matrix is shared by nuisance fitting, follower masks, targeting, and reporting.

### A survival outcome

An outcome sequence declares one absorbing event at each horizon:

```python
from cleverly.datasets import make_longitudinal_survival

frame, truth = make_longitudinal_survival(n=2_000, seed=12)
survival = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome=("Y1", "Y2"),
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
risks = survival.estimate(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)),
    outcome_learner="glm",
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=3,
    learner_folds=3,
)
```

### Competing risks

A mapping declares competing causes:

```python
from cleverly.datasets import make_longitudinal_competing

frame, truth = make_longitudinal_competing(n=2_000, seed=13)
competing = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")},
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
incidence = competing.estimate(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)),
    outcome_learner="glm",
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=3,
    learner_folds=3,
)
```

Every longitudinal alias has a `ParameterKey` carrying regimen, horizon, and cause. Survival and
competing-risk arithmetic remain the evidenced longitudinal engine; the public adapter only
normalizes the contract and selects the requested parameter family.

## Collaborative TMLE

Estimator variants are typed methods:

```python
from cleverly import CollaborativeTMLEMethod, DRTMLEMethod

collaborative = effect.estimate(method=CollaborativeTMLEMethod(strategy="greedy"), random_state=3)
doubly_robust = effect.estimate(method=DRTMLEMethod(), random_state=3)
```

Method capability is checked against the identified functional before engine construction.
Collaborative TMLE does not claim support for longitudinal, shift, incremental, or MSM functionals
without a derivation.

## Doubly-robust inference

`DRTMLEMethod` selects the existing doubly-robust-inference variant for a compatible identified
effect. Its extra score equations weaken the nuisance conditions for inference; they do not promise
a narrower interval. The complete scientific contract is in [drtmle.md](drtmle.md).

## Cross-fitting and CV-TMLE

Cross-fitting policy is a named immutable group rather than a set of constructor side effects.

## Immutable configuration

```python
from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod, Targeting

method = TMLEMethod(
    models=ModelSpec(outcome_learner="glm", treatment_learner="glm"),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3, repeats=1),
    targeting=Targeting(g_bounds="auto", algorithm="iterative"),
    inference=Inference(alpha=0.05, simultaneous=False),
    runtime=Runtime(random_state=3, n_jobs=1),
)
result = effect.estimate(method=method)
```

Common keyword shortcuts normalize into these same objects. `alpha=` is the confidence-interval
significance level; `submodel_alpha=` is the separate logistic-submodel bound. Unknown options
raise `MethodConfigurationError`, a `CleverlyError`, rather than being forwarded to an
implementation constructor. A setting that one engine cannot implement is rejected at the same
boundary: longitudinal estimation refuses point-only controls such as `n_bootstrap=` instead of
accepting and discarding them. `cross_fit=False` is supported and translates explicitly to
`n_folds=1`.

## Results, contrasts, and persistence

All point and longitudinal causal results share mapping, inference, summary, frame, contrast, and
persistence operations:

```python
from cleverly import load

names = list(result.estimates)
frame = result.to_frame()
covariance = result.covariance(names)
if len(names) >= 2:
    difference = result.contrast(lambda values: values[0] - values[1], names[:2])

result.save("analysis.npz")
restored = load("analysis.npz")
assert restored.parameter_keys == result.parameter_keys
```

The result stores `identified_effect`, normalized `method`, provenance, influence curves, and
structured parameter keys. A restored effect is fitted metadata, not a hidden copy of the source
data; construct a new `CausalStudy` to estimate it again.

Assessment is post-fit and capability-aware:

```python
validation = result.validate()  # inexpensive and never refits
support = result.diagnostics.support()
nuisance = result.diagnostics.nuisance_models()
scores = result.diagnostics.score_equations()
all_cached = result.diagnostics.run_all()

print(validation.summary())
print(result.sensitivity.run_all().summary())
```

Each combined report labels an item `passed`, `failed`, `warning`, `not_applicable`, or
`unavailable`; the last two remain distinct after serialization. Every diagnostic capability
names its required fitted artifacts, cost, execution mode, saved-result determinism, and
method-specific interpretation. `run_all()` and `validate()` exclude refits by default;
`result.diagnostics.refute()` and `result.sensitivity.benchmark()` are explicit refit operations.

Point-treatment diagnostics reuse the established score, nuisance, and support report objects and
their numerical results. Longitudinal results expose immutable stagewise reports for cumulative
support, targeting scores, and node regression loss. Sensitivity operations without a published
longitudinal derivation remain present as `unavailable` capabilities rather than disappearing or
borrowing point-treatment arithmetic.

Completed assessments are cached by operation plus normalized arguments. The cache, along with
`result.replayability`, is persisted separately from the headline estimates, so restoring a result
can replay cache-only reports without changing or recomputing its estimate.

## Migrating old code

The old root constructors and one-call helpers were removed in the clean alpha break. See the
[migration guide](migration.md) for before/after examples, the complete argument map, changed
defaults, the static audit script, and the stable tag containing the final old API.

## Extending the package

The public protocols are `Estimand`, `IdentificationProvider`, `EstimationMethod`, and
`CausalResult`. A custom provider should produce an `IdentifiedEffect` only when it can name the
observed-data functional and assumptions. A custom method must declare support and return a result
with stable parameter keys. Adding a new registered analytic target still requires the independent
oracle, Gateaux, remainder, and mutation evidence described in [evidence.md](evidence.md); a typed
wrapper is not scientific certification.
