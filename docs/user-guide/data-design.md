# Data and study design

## Accepted table backends

`CausalStudy` accepts pandas, polars, Arrow-backed pandas, and `pyarrow.Table` inputs through
narwhals. Table-returning output comes back in the backend the study was built from, so
`result.to_frame()` on a `pyarrow.Table` study returns a `pyarrow.Table`. Column-role and dtype
validation happens before nuisance fitting.

The input must have one row per independent observational unit unless `cluster=` declares the unit
at which inference is independent. A cross-fitted fit needs each treatment arm in at least two of
those units, because a split moves whole units. It also needs the outcome's known support when the
outcome is continuous.
[Fold and outcome-scale rules](../technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
gives both requirements and their refusals. Missing values are supported only in roles whose design
explicitly models missingness; a missing adjustment or treatment value is not silently imputed.

## Point treatment

```python
from cleverly import CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "region"),
        weights="sampling_weight",
        cluster="household",
        strata=("region",),
    ),
)
```

- `adjustment` is the measured pre-treatment set used by the identification argument.
- `weights` defines the target population and flows through nuisance losses, targeting, influence
  curves, and covariance.
- `cluster` selects cluster-robust inference, and it selects a grouped fold draw that keeps each
  cluster whole. It is not another adjustment variable. A cluster is then the independent unit a
  cross-fitted fit counts, so each treatment arm must appear in at least two clusters. C-TMLE
  refuses `cluster=` at every setting, and longitudinal TMLE refuses it above one fold.

  A TMLE, DR-TMLE, or longitudinal TMLE fit reports no interval when it has fewer than 40
  clusters with positive weight mass. For TMLE and DR-TMLE, this also applies to one reported
  stratum. A cross-fitted fit reports none when its clusters differ in rows or, on a weighted
  fit, weight mass, overall or within a reported stratum
  ([clusters](../technical-reference/inference.md#clusters)).
- `strata` requests subgroup parameters and preserves the stratum in structured parameter keys.
  A stratum variable must also appear in `adjustment`: it conditions the reported parameter, so a
  design that stratified on a variable it did not adjust for is refused rather than fitted.
- `intermediate` and explicit missingness roles activate supported controlled-direct-effect and
  missing-outcome compositions. The next section gives the two missingness roles.

For a randomized study with no adjustment variables, declare the design:

```python
randomized = CausalStudy(
    frame[["Y", "A"]],
    design=PointTreatment(outcome="Y", treatment="A", randomized=True),
)
```

## Missing outcomes and missing treatments

Declare each missingness role with a 0/1 indicator column. The column is 1 where the value is
recorded. The package does not infer missingness from a missing value, so an accidental gap is
refused rather than analyzed.

| role | `PointTreatment` | `fit()` and `CausalData` | `tmle()` |
| --- | --- | --- | --- |
| outcome recorded | `missingness=` | `delta=` | `Delta=` |
| treatment recorded | `treatment_missingness=` | `treatment_delta=` | `DeltaA=` |

A declared missing treatment supports the arm means and their contrasts on in-sample `TMLE` and
`DRTMLE` fits. Each arm uses the composite indicator: the treatment and the outcome are recorded,
and the treatment is that arm. The recording of the treatment may depend on the treatment, so the
ATT, the ATC and the population-intervention targets are not identified and are refused.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, DRTMLEMethod, PointTreatment
from cleverly.datasets import make_missing_outcome

observed, _ = make_missing_outcome(n=600, seed=1)
rng = np.random.default_rng(1)
observed["A_recorded"] = (rng.random(len(observed)) < 0.85).astype(float)
observed.loc[observed["A_recorded"] == 0.0, "A"] = np.nan

missing_study = CausalStudy(
    observed,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
        missingness="Delta",
        treatment_missingness="A_recorded",
    ),
)
effect = missing_study.estimate(
    ATE(),
    method=DRTMLEMethod(
        reduced_outcome_learner=LinearRegression(),
        reduced_treatment_learner=LogisticRegression(max_iter=1000),
    ),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    missingness_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
    random_state=0,
)
print(effect.extra["missing_data"])
```

The fit records the construction it ran as `composite`. The
[composite contract](../technical-reference/dr-tmle/theorem.md#observational-missing-data-the-composite-indicator)
states the conditions, and [scope and refusals](../technical-reference/scope-and-refusals.md)
lists the refused compositions.

## Continuous treatment

Set `treatment_kind="continuous"` when the target is a modified treatment policy. Arm-indexed
estimands are refused on this design.

```python
dose_study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="dose",
        adjustment=("W1", "W2"),
        treatment_kind="continuous",
    ),
)
```

## Longitudinal treatment

A longitudinal design records one ordered treatment role per node, the history available at each
node, censoring, and either an end-of-study outcome or event processes. Continue with the
[longitudinal guide](longitudinal.md).

## Observation weights are not estimand weights

Sampling weights change the population represented by the empirical distribution. MSM projection
weights change the definition of the projection. Clever covariates weight observations inside an
estimating equation. These three roles are deliberately separate.
