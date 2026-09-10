# Data and study design

## Accepted table backends

`CausalStudy` accepts pandas, polars, Arrow-backed pandas, and `pyarrow.Table` inputs through
narwhals. Table-returning output comes back in the backend the study was built from, so
`result.to_frame()` on a `pyarrow.Table` study returns a `pyarrow.Table`. Column-role and dtype
validation happens before nuisance fitting.

The input must have one row per independent observational unit unless `cluster=` declares the unit
at which inference is independent. Missing values are supported only in roles whose design
explicitly models missingness; a missing adjustment or treatment value is not silently imputed.

## The study protocol record

Pass `protocol=StudyProtocol(...)` to `CausalStudy` when the analysis needs a durable record of its
scientific context. The record runs from the target population to the assumption rationale, and
`StudyProtocol` documents each of its ten fields. The argument is optional, and every other part of
the workflow behaves the same way without it.

`CausalStudy` stamps the record onto the identified effect and onto each fitted result, so a summary
reports the study context beside the estimate. The record describes the study and configures
nothing. The design still owns the column roles, and the typed estimand still owns the contrast.
[Step 1 of the workflow](../workflow.md#1-formulate-the-causal-question) shows a complete record,
and [study protocol vocabulary](../references.md#study-protocol-vocabulary) names its sources.

The constructor strips surrounding whitespace from each text field and then validates it. The strip
is part of the canonical form, so a padded field and the same text without padding give one digest.
A later change to that normalization raises the schema version.

The constructor refuses a wrong type with `TypeError`, such as one string where a sequence of
strings belongs. It refuses content the schema cannot state with `DataError`. Blank text, a control
character, a line separator, and a strategy list and a version list of different lengths are all
refused that way.

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
- `cluster` selects cluster-robust inference; it is not another adjustment variable.
- `strata` requests subgroup parameters and preserves the stratum in structured parameter keys.
  A stratum variable must also appear in `adjustment`: it conditions the reported parameter, so a
  design that stratified on a variable it did not adjust for is refused rather than fitted.
- `intermediate` and explicit missingness roles activate supported controlled-direct-effect and
  missing-outcome compositions.

For a randomized study with no adjustment variables, declare the design:

```python
randomized = CausalStudy(
    frame[["Y", "A"]],
    design=PointTreatment(outcome="Y", treatment="A", randomized=True),
)
```

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
