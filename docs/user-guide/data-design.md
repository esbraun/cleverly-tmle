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
- `weights` names a column of sampling or survey weights. They set the population the answer
  describes: every model, the correction step that removes the outcome model's remaining bias,
  and the standard error use them. If the weights came from a model fitted to these same data,
  such as a nonresponse model, also pass `weights_estimated=True`. That changes no estimate or
  standard error, because cleverly never sees the weight model and treats the weights as fixed
  numbers. The standard error therefore leaves out the error of the weight model, and the
  summary's Weights line says the weights were estimated. Some fits then carry the
  `experimental` label (the status table in [inference](../technical-reference/inference.md)).
- `cluster` selects cluster-robust inference, and it selects a grouped fold draw that keeps each
  cluster whole. It is not another adjustment variable. A cluster is then the independent unit a
  cross-fitted fit counts, so each treatment arm must appear in at least two clusters.
  Collaborative TMLE (C-TMLE), a variant that fits its treatment model in light of the outcome
  model, refuses `cluster=` at every setting except `strategy="derivative"`
  ([scope and refusals](../technical-reference/scope-and-refusals.md)). Longitudinal TMLE
  cross-fits with whole-cluster folds.

  Clusters may differ in size. A clustered fit with fewer than 40 clusters with positive weight mass reports a Student t interval with the cluster count minus 2
  degrees of freedom. Below 10 such clusters, or 20 for longitudinal TMLE, the interval carries
  the `experimental` label, and below 3 the fit reports no interval. For TMLE and DR-TMLE, these
  rules also apply to one reported stratum. [Clusters](../technical-reference/inference.md#clusters)
  gives the rules and the estimators they cover.

  A fit that averages one estimate per cross-fitting fold (`CrossFitting(fold_evaluation=True)`)
  uses the cluster count minus the number of folds, at every cluster count. Outside learned
  treatment rules, that fit needs clusters of equal size, or of equal total weight on a weighted
  fit, and the effect among the treated also needs the same number of treated rows in each
  cluster. Otherwise it refuses and points to the default, which estimates from all folds at once
  ([clusters of unequal mass](../technical-reference/cv-tmle.md#clusters-of-unequal-mass)).
- `strata` requests subgroup parameters and preserves the stratum in structured parameter keys.
  A stratum variable must also appear in `adjustment`: it conditions the reported parameter, so a
  design that stratified on a variable it did not adjust for is refused rather than fitted.
  Arm, regime, shift, incremental, MSM and natural-course targets fit strata, and so does
  DR-TMLE at every `guard`.

  A fit that averages one estimate per cross-fitting fold (`CrossFitting(fold_evaluation=True)`)
  also fits strata. It
  reports the mean under each treatment option and their differences, the means under fixed plans,
  and a learned rule's value, on complete data without `cluster=`. A
  [simulation study](../technical-reference/method-evidence/fold-evaluated-strata.md) checks the
  option means and the average effect with a single draw of the folds (`repeats=1`), so those read
  `supported`. The means under plans, more than one draw, and the band that covers every estimate
  at once read `experimental`. Running the correction step separately inside each fold
  (`targeting_scheme="fold"`) refuses strata.
  [Point-treatment TMLE](../technical-reference/point-treatment-tmle.md), section "Weights,
  strata, and clusters", lists every case and refusal.
- `outcome_family` is the outcome's type. The default `"auto"` reads an outcome with only the
  values 0 and 1 as yes-or-no (`"binomial"`) and anything else as continuous (`"gaussian"`).
  Declare it when the sample could mislead, such as a score that happens to take only 0 and 1 in
  a small pilot. A continuous outcome, declared or read, needs its known range (`q_bounds`) for a
  cross-fitted fit, as the opening section says. A declared `"binomial"` checks that the recorded
  outcomes are 0 or 1.
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
- For the effect on the treated (ATT) or on the controls (ATC), pass
  `conditional_form="known_score"` to `TMLE` or `Targeting`. It averages the effect over every
  unit, weighted by its declared chance of being in the treated (or control) arm, instead of over
  the units that landed there. With an outcome learner close to the truth, its interval is
  no wider than the default, and narrower when the effect varies with the covariates. It answers the
  question for the treated population; for the treated units actually in this sample, keep the
  default.

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
chances could. A [simulation study](../technical-reference/method-evidence/known-mechanism-sample-weights.md)
checked the `"sample"` claims on one point-treatment and one two-time-point design with logistic
and linear models: the estimate stayed on target with either model wrong, and the interval covered
when both were right.

When the weights depend on the outcome, the weighted population is a causal target
only if the weights describe how that population was sampled, with known sampling fractions.
With `"sample"`, a weight that depends on the outcome also undoes the randomization in the
weighted rows, so the estimate is not a causal effect without a further argument. A weight that
depends only on the treatment and the adjustment columns keeps the causal reading.
[Known treatment mechanism](../technical-reference/point-treatment-tmle.md#known-treatment-mechanism)
gives the formulas, and
[known node mechanisms](../technical-reference/longitudinal-tmle.md#known-node-mechanisms) gives
the same choice at each node of a `LongitudinalTreatment`, `TimeToEvent` or `LongPanel` design.

### Let the treatment model read fewer columns

When you know that the treatment was assigned from only some of the adjustment columns, name them
in `treatment_adjustment`. The treatment model then reads only those columns, and the outcome
model keeps every column of `adjustment`. The declaration is an assumption about how treatment was
assigned. If it holds, the estimate stays on target when either model is right; if it fails, only
a right outcome model keeps it on target. The interval needs the assumption and both models to be
reasonably accurate. To leave out a column that predicts who is treated and has no other path to
the outcome, drop it from `adjustment` instead, which removes it from both models.

```python
restricted = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3"),
        treatment_adjustment=("W1",),
    ),
)
```

A fit's full summary (`summary(detail="full")`) lists the assumption under Assumptions. A design
refuses `treatment_adjustment` beside `treatment_probabilities=`, which already fixes the treatment
model, and a question that tilts the odds of treatment (`IncrementalMean` or `IncrementalEffect`)
refuses it too ([scope and refusals](../technical-reference/scope-and-refusals.md), the rows on
`treatment_adjustment=`). A
[simulation study](../technical-reference/method-evidence/option-treatment-adjustment.md) checks
the declaration, and [point-treatment TMLE](../technical-reference/point-treatment-tmle.md),
section "Treatment-mechanism covariates", states its conditions.

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

With a missing treatment, collaborative TMLE with `strategy="oat"` (`CollaborativeTMLEMethod`)
also gives in-sample point estimates of the arm means and their contrasts. Its treatment model reads only the outcome
model's prediction for each arm, so these estimates are on target only when the outcome model
is right. They come with no interval
([collaborative TMLE](../technical-reference/collaborative-tmle.md), section "Baseline strata and
missing data in the per-arm design", and the status table in
[inference](../technical-reference/inference.md)).

## Continuous treatment

Set `treatment_kind="continuous"` when the treatment is a dose, such as a price or a discount.
The design answers a modified treatment policy, a working model over doses (`MSMProjection`)
and the smoothed dose-response curve ([`DoseResponseCurve`](estimands.md#a-curve-over-a-continuous-dose)).
Arm-indexed estimands are refused on this design.

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

If you know the range the dose can take, declare it with `treatment_support=Support(lower, upper)`
(from `cleverly.interventions`). If many rows share one dose, such as zero, list it in
`dose_atoms=`. The section "Modified treatment policies" of [estimands](estimands.md) says what
each declaration changes. A trend line over doses (`MSMProjection`) refuses both declarations, so
fit it on a study without them (the refusal table of
[MSM projections](../technical-reference/msm-projections.md)).

## Several treatment columns

Pass a list to `treatment=` when the treatment has several parts, such as a coupon offered or not
and a discount amount. When every part takes a few values, cleverly treats each combination as one
option (up to 20), and the questions about options, such as `ATE` and `CounterfactualMean`,
report one estimate per combination. When a part is an amount, name it in `continuous_treatment=`.
Such a treatment has no fixed options. It answers policy questions that change one or more parts
(`JointPolicy` inside `ModifiedTreatmentPolicy` or `ModifiedTreatmentPolicyEffect`), and the
questions about options, such as `ATE`, refuse
([point-treatment TMLE](../technical-reference/point-treatment-tmle.md), section "Vector
treatments").

```python
bundle = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment=["A", "dose"],
        adjustment=("W1", "W2"),
        continuous_treatment=["dose"],
    ),
)
```

`treatment_support=` and `dose_atoms=` then take a mapping from the amount's column to its
declaration, such as `treatment_support={"dose": Support(0.0, 10.0)}`.

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
