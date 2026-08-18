# Intervention analysis

This example contrasts three ways to intervene: a known treatment rule, a continuous modified
treatment policy, and an incremental propensity-score intervention. They are different causal
parameters even when their result tables look similar.

## A dynamic rule

```python
from cleverly import CausalStudy, PointTreatment, RegimeContrast
from cleverly.datasets import make_nonlinear_ate
from cleverly.interventions import Rule, Static

frame, truth = make_nonlinear_ate(n=2_500, seed=31)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)
plans = (
    Static(0, name="treat none"),
    Static(1, name="treat all"),
    Rule(lambda data: (data["W1"] > 0).astype(float), name="treat if W1 positive"),
)
rule_result = study.estimate(
    RegimeContrast(plans, reference="treat none"),
    random_state=31,
)
print(rule_result.summary())
```

The dynamic rule is fixed before fitting. It needs support only for the arm it assigns at each
covariate value.

## A continuous-dose shift

```python
from cleverly import ModifiedTreatmentPolicyEffect
from cleverly.datasets import make_shift_dose
from cleverly.interventions import Shift

dose_frame, dose_truth = make_shift_dose(n=2_500, seed=32)
dose_study = CausalStudy(
    dose_frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
        treatment_kind="continuous",
    ),
)
policies = (
    Shift(0.0, cap=None),
    Shift(0.25, cap=5.0),
    Shift(0.50, cap=5.0),
)
shift_result = dose_study.estimate(
    ModifiedTreatmentPolicyEffect(policies),
    density_bins=40,
    random_state=32,
)
print(shift_result.summary())
print(shift_result.diagnostics.support().summary())
```

Here `cap=5.0` is part of the declared intervention and ensures that the shifted dose remains in
the intended support region.

## An incremental odds tilt

```python
from cleverly import IncrementalEffect
from cleverly.interventions import Incremental

tilts = (Incremental(0.5), Incremental(1.0), Incremental(2.0))
incremental_result = study.estimate(
    IncrementalEffect(tilts),
    random_state=31,
)
print(incremental_result.summary())
```

Because the incremental target is defined through the observed treatment mechanism, its inference
is conditional on estimating that mechanism consistently. See [stochastic interventions](../technical-reference/interventions.md).
