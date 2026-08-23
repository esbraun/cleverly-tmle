# Estimands and interventions

Choose the object that names the scientific quantity, not the estimator you expect to use.

Every fit on this page shares one parametric configuration, so the estimand is the only thing
changing between them:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import CrossFitting, ModelSpec, TMLEMethod

quick = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(), treatment_learner=LogisticRegression(max_iter=1000)
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
)
```

Passing no method instead inherits the default Super Learner library over ten outer folds, which
is the right default for an analysis and slow enough to be surprising in an example. See
[methods and learners](methods-learners.md).

## Arm-indexed questions

| question | typed estimand | reported quantity |
| --- | --- | --- |
| Average treatment effect | `ATE` | mean under an arm minus mean under the reference arm |
| Effect among treated / controls | `ATT`, `ATC` | contrast in the observed treatment-defined population |
| Counterfactual level | `CounterfactualMean` | mean outcome under one or all arms |
| Relative effect | `RiskRatio`, `OddsRatio` | arm-to-reference ratio on risk or odds scale |
| Natural course | `NaturalCourseMean` | mean observed outcome |
| Population impact | `PopulationAttributableRisk`, `PopulationAttributableFraction` | natural-course mean compared with a reference intervention |

```python
from cleverly import ATE, CounterfactualMean, RiskRatio

ate = study.estimate(ATE(reference=0), method=quick, random_state=3)
levels = study.estimate(CounterfactualMean(), method=quick, random_state=3)
relative = study.estimate(RiskRatio(reference=0), method=quick, random_state=3)
```

Multi-valued treatments preserve original arm labels in `ParameterKey.value` and
`ParameterKey.reference`.

## Known regimes

Use `Static`, `Rule`, or `Stochastic` interventions with `RegimeMean` or `RegimeContrast`.

```python
import numpy as np

from cleverly import RegimeContrast
from cleverly.interventions import Rule, Static, Stochastic

plans = (
    Static(1, name="treat all"),
    Static(0, name="treat none"),
    Rule(lambda data: (data["W1"] > 0).astype(float), name="treat if W1 positive"),
    Stochastic(
        lambda data: np.column_stack([np.full(len(data), 0.4), np.full(len(data), 0.6)]),
        name="assign with probability 0.6",
    ),
)
result = study.estimate(RegimeContrast(plans, reference="treat none"), method=quick, random_state=3)
```

A rule is a known function of observed history. An intervention estimated from the treatment
mechanism is a different parameter.

`Stochastic` takes the assignment *density*, not a scalar probability: a function of the covariate
frame returning one column per arm in `data.treatment_levels` order, with rows summing to one. It
must be a fixed function of the covariates. The influence curve reported for a regime carries no
term for a $g^\star$ that depends on $P$.

## Modified treatment policies

For a continuous dose, `Shift` maps each observed dose to a policy dose. `cap=` is part of the
policy and its support argument; it is not estimated from the sample maximum.

```python
from cleverly import ModifiedTreatmentPolicyEffect
from cleverly.interventions import Shift

policies = (Shift(0.0, cap=None), Shift(0.25, cap=5.0), Shift(0.5, cap=5.0))
result = dose_study.estimate(
    ModifiedTreatmentPolicyEffect(policies),
    method=quick,
    density_bins=40,
    random_state=5,
)
```

## Incremental propensity-score interventions

`Incremental(delta)` multiplies the observed treatment odds by `delta`. It avoids a conventional
positivity requirement, but the target depends on the true treatment mechanism and therefore does
not have ordinary two-sided double robustness.

```python
from cleverly import IncrementalEffect
from cleverly.interventions import Incremental

tilts = (Incremental(1.0), Incremental(2.0), Incremental(0.5))
result = study.estimate(IncrementalEffect(tilts), method=quick, random_state=3)
```

## Marginal structural models

`MSMProjection` defines a projection of counterfactual means onto a declared working model. A
coefficient remains a projection parameter when the working model is misspecified.

## Controlled direct effects

`ControlledDirectEffect(intermediate=z)` compares treatment arms while setting the declared
intermediate to `z`. Fits at two intermediate levels are two controlled direct effects, not an
automatic direct / indirect decomposition.

See the [technical implementation matrix](../technical-reference/index.md) for assumptions and
citations for every family.
