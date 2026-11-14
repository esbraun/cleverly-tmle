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
  refuses `cluster=` at every setting except `strategy="derivative"`
  ([scope and refusals](../technical-reference/scope-and-refusals.md)). Longitudinal TMLE
  cross-fits with whole-cluster folds.

  Clusters may differ in size. A clustered fit with fewer than 40 clusters with positive weight mass reports a Student t interval with the cluster count minus 2
  degrees of freedom. Below 10 such clusters, or 20 for longitudinal TMLE, the interval carries
  the `experimental` label, and below 3 the fit reports no interval. For TMLE and DR-TMLE, these
  rules also apply to one reported stratum. [Clusters](../technical-reference/inference.md#clusters)
  gives the rules and the estimators they cover.

  A fit that averages one estimate per cross-fitting fold (`cv_evaluation=True`) uses the cluster
  count minus the number of folds, at every cluster count. Outside learned treatment rules, that
  fit needs clusters of equal size, or of equal total weight on a weighted fit, and the effect
  among the treated also needs the same number of treated rows in each cluster. Otherwise it
  refuses and points to the default, which estimates from all folds at once
  ([clusters of unequal mass](../technical-reference/cv-tmle.md#clusters-of-unequal-mass)).
- `strata` requests subgroup parameters and preserves the stratum in structured parameter keys.
  A stratum variable must also appear in `adjustment`: it conditions the reported parameter, so a
  design that stratified on a variable it did not adjust for is refused rather than fitted.
  Arm, regime, shift, incremental, MSM and natural-course targets fit strata, and so does
  DR-TMLE at every `guard`. Fold targeting and fold evaluation refuse them
  ([weights, strata, and clusters](../technical-reference/point-treatment-tmle.md#weights-strata-and-clusters)).
- `intermediate` and explicit missingness roles activate supported controlled-direct-effect and
  missing-outcome compositions. The next section gives the two missingness roles.

For a randomized study with no adjustment variables, declare the design:

```python
randomized = CausalStudy(
    frame[["Y", "A"]],
    design=PointTreatment(outcome="Y", treatment="A", randomized=True),
)
```

A trial that records the probability with which each unit was assigned each arm declares that
known mechanism. Name the frame column of each arm's probability. In this example, every unit had
probability one half:

```python
trial = CausalStudy(
    frame.assign(p_control=0.5, p_treated=0.5),
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        treatment_probabilities={0: "p_control", 1: "p_treated"},
    ),
)
```

- Every method that accepts the design divides by the declared probabilities and fits no
  treatment learner. On complete
  data, the estimate is then consistent for any outcome learner, and the interval is not
  conservative. With missing outcomes, the fit still estimates the missingness mechanism, and the
  outcome learner must also be consistent.
  [Known treatment mechanism](../technical-reference/point-treatment-tmle.md#known-treatment-mechanism)
  gives each case.
- The probability columns are not adjustment variables, and they can depend on the adjustment
  set, as a stratified allocation does.
- `CollaborativeTMLEMethod` is unavailable on such a design, because there is no mechanism to
  select.
- A fit refuses a bound pair that would move a declared value, before any learner. Pass
  `g_bounds=` wide enough to hold every declared probability.

[Known treatment mechanism](../technical-reference/point-treatment-tmle.md#known-treatment-mechanism)
gives the forms, the curves and the refusals. `LongitudinalTreatment` takes the same declaration
per node ([known node mechanisms](../technical-reference/longitudinal-tmle.md#known-node-mechanisms)).

### Known chances beside weights

With `weights=`, the fit estimates the effect in the population the weights describe, so it needs
the chance of treatment in that population. The weights can change that chance when they depend on
the treatment or the outcome. A design that declares chances beside weights that are not all
equal must therefore say which population the chances describe, with `mechanism_law=`. Without it,
the fit refuses before it fits any model.

Pass `mechanism_law="target"` when the declared chances already hold in the weighted population.
That is the case when the treatment was randomized before the sampling or loss that the weights
undo. In a case-cohort study drawn from a trial, the trial randomized everyone, then measured a
subsample, and the weights restore the whole trial. A non-response weight is the same case:
everyone was randomized, people answered the follow-up survey at rates that differ by arm, and
each answer is weighted by one over that person's chance of answering, so the answers stand for
everyone randomized. It is also the case when the weights depend only on the adjustment columns.
The fit then divides by the declared chances, as on unweighted data:

```python
case_cohort = CausalStudy(
    frame.assign(p_control=0.5, p_treated=0.5),
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        weights="sampling_weight",
        treatment_probabilities={0: "p_control", 1: "p_treated"},
        mechanism_law="target",
    ),
)
```

Pass `mechanism_law="sample"` when the declared chances hold in the rows you have, and you add
weights afterwards that depend on the treatment and are not meant to restore a larger randomized
group. For example, every row was randomized with the declared chances, and the analysis weights
each row by its arm and covariates to stress the treated patients at the highest baseline risk.
The weighted rows then no longer have the declared chances. The fit learns how the weights shift
the chances, by a classification of the treatment that starts from the declared chances, and
treats the result as an estimated treatment model:

```python
analysis_weighted = CausalStudy(
    frame.assign(p_control=0.5, p_treated=0.5),
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        weights="analysis_weight",
        treatment_probabilities={0: "p_control", 1: "p_treated"},
        mechanism_law="sample",
    ),
)
```

The two readings give different guarantees. With `"target"`, the interval holds even when the
outcome model is wrong. With `"sample"`, the estimate is still consistent when either the outcome
model or the treatment learner is right, but the interval holds only when both are close to the
truth, as for any estimated treatment model. The treatment learner must accept `sample_weight`. The
`"sample"` fit also does not use the declared chances as efficiently as a method built for known
chances could.

When the weights depend on the outcome, the weighted population is a causal target
only if the weights describe how that population was sampled, with known sampling fractions.
With `"sample"`, a weight that depends on the outcome also undoes the randomization in the
weighted rows, so the estimate is not a causal effect without a further argument. A weight that
depends only on the treatment and the adjustment columns keeps the causal reading.
[Known treatment mechanism](../technical-reference/point-treatment-tmle.md#known-treatment-mechanism)
gives the formulas, and
[known node mechanisms](../technical-reference/longitudinal-tmle.md#known-node-mechanisms) gives
the same choice at each node of a `LongitudinalTreatment`, `TimeToEvent` or `LongPanel` design.

## Missing outcomes and missing treatments

Declare each missingness role with a 0/1 indicator column. The column is 1 where the value is
recorded. The package does not infer missingness from a missing value, so an accidental gap is
refused rather than analyzed.

| role | `PointTreatment` | `fit()` and `CausalData` | `tmle()` |
| --- | --- | --- | --- |
| outcome recorded | `missingness=` | `delta=` | `Delta=` |
| treatment recorded | `treatment_missingness=` | `treatment_delta=` | `DeltaA=` |

A declared missing treatment supports the arm means and their contrasts on in-sample `TMLE` and
`DRTMLE` fits. In-sample `TMLE` fits also support regime means and arm-indexed `msm=` coefficients. Each arm uses the composite indicator: the treatment and the outcome are recorded,
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

## A trial with external data

`TrialWithExternalData` declares a randomized trial pooled with external rows on the same outcome.
It takes the outcome, a binary treatment, the adjustment columns, and `source=`, the column that
marks trial rows. [A small trial with external data](trial-with-external-data.md) shows the
workflow.

## Longitudinal treatment

A longitudinal design records one ordered treatment role per node, the history available at each
node, censoring, and either an end-of-study outcome or event processes. Continue with the
[longitudinal guide](longitudinal.md).

## Change one field of a design

A design object cannot be changed in place. To try another adjustment set or another known
mechanism, copy the design with `dataclasses.replace`:

```python
import dataclasses

narrower = dataclasses.replace(study.design, adjustment=("W1", "W2", "region"))
```

Pass the copy to a new `CausalStudy`. The copy equals the design built directly from the same
arguments. This works on
`PointTreatment`, `TimeToEvent`, `LongitudinalTreatment` and `LongPanel`
(`tests/unit/test_design_replace.py`). A mapping field passed as pairs that repeat a key, or that
use an unhashable key, raises `DataError`.

## Write down the study description

`StudyProtocol` stores the study description in words. It records who is in the study, when
follow-up starts, what each option means, and how the outcome is measured. Three of its fields are
easy to misread.

`treatment_versions` says what each option delivers in practice, such as "a bedside plan and two
scheduled calls", because two programs with one name can differ. `intercurrent_event_handling`
says how the analysis treats events after the start that change what the outcome means, such as a
readmission before the score is measured.
A "composite strategy" folds such an event into the
outcome, for example by counting a death as the worst score. `interference_unit` names the group
within which one person's treatment can change another's outcome, such as a household. The
[workflow page](../workflow.md) shows a complete description and the sources of these terms.

## Observation weights are not estimand weights

Sampling weights change the population represented by the empirical distribution. MSM projection
weights change the definition of the projection. Clever covariates weight observations inside an
estimating equation. These three roles are deliberately separate.
