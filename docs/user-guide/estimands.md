# Average effects and fixed plans

An estimand is the quantity you ask for, written as an object such as `ATE()`. Choose the object
that names the scientific quantity, not the estimator you expect to use. This page covers questions
about fixed treatment levels and plans set in advance: average effects and their ratios, effects
among the treated, the mean under a rule or a random assignment, a trend line across plans, and a
controlled direct effect. [Estimands](topics/estimands.md) lists the pages for the other kinds of
question.

The examples use a `study` with a binary treatment `A`, a binary outcome `Y` and covariates `W1`
to `W3`, built as in the [quickstart](../getting-started/quickstart.md).

Most fits on this page share one configuration, `quick`, so only the question changes between
them:

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

Passing no method instead inherits the default Super Learner library over ten unstratified outer
folds, which is the right default for an analysis and slow enough to be surprising in an example.
This study's outcome is binary. A cross-fitted fit of a continuous outcome also needs a declared
`Targeting(q_bounds=(lower, upper))`, which
[fold and outcome-scale rules](../technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
states. See
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

`NaturalCourseMean()` can also target the mean before outcome nonresponse when the study declares
`missingness=`. That path uses the outcome regression and the response mechanism. It uses no
treatment model.

`cleverly` supports two scalar TMLE configurations for this target. An ordinary fit uses
`CrossFitting(enabled=False)`. It accepts a binary outcome or a continuous outcome with fixed
`q_bounds`. A stacked CV-TMLE fit is supported for a binary outcome, and runs a continuous outcome
under the experimental label. It uses this cross-fitting declaration:

```python
stacked_mar = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        missingness_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(
        n_folds=10,
        repeats=1,
        stratify_by="none",
        targeting_scheme="pooled",
        fold_evaluation=False,
    ),
)
```

Pass `stacked_mar` when you estimate `NaturalCourseMean()` from a study with `missingness=`. The
fit generates one near-balanced outer partition. Both configurations are supported with iterative
unweighted logistic targeting; other targeting settings run under the experimental label. The
treatment can have two or more arms. The ordinary fit also accepts fixed weights and clusters. The
stacked fit is supported on unweighted iid rows, and runs weights or clusters under the
experimental label.

`PopulationAttributableRisk` and `PopulationAttributableFraction` accept `missingness=` too. The
fit stacks the natural-course mean and the reference-arm mean on the same rows, so it also uses a
treatment model:

```python
import numpy as np

from cleverly import CausalStudy, PointTreatment, PopulationAttributableRisk

rng = np.random.default_rng(7)
observed = frame.assign(Delta=(rng.random(len(frame)) < 0.85).astype(int))
observed.loc[observed["Delta"] == 0, "Y"] = np.nan
mar_study = CausalStudy(
    observed,
    design=PointTreatment(
        outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), missingness="Delta"
    ),
)
in_sample_mar = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(enabled=False),
)
attributable = mar_study.estimate(PopulationAttributableRisk(reference=0), method=in_sample_mar)
```

The [PAR and PAF contract](../technical-reference/point-treatment-tmle.md#par-and-paf-with-missing-outcomes)
gives the stack, its conditions, and the compositions it refuses.

[Missing-outcome natural-course contracts](../technical-reference/scope-and-refusals.md#missing-outcome-natural-course-contracts)
lists every refused composition for both configurations. See the
[missing-outcome construction and evidence](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects).

`ATE`, `CounterfactualMean`, `RiskRatio`, and `OddsRatio` also accept `missingness=`. Under
cross-fitting with missing outcomes, `TMLEMethod` fits these targets with an interval.
`CollaborativeTMLEMethod` fits them only with `strategy="oat"` and the default `oat_design="per_arm"`, and reports a
point estimate only. Its other strategies and `DRTMLEMethod` are refused. `ATT` and `ATC` run under
the experimental label under cross-fitting with missing outcomes. Two further requirements are not special to this surface.
Every cross-fitted fit draws `CrossFitting(stratify_by="none")` folds, which is the default, and
every cross-fitted continuous outcome needs `Targeting(q_bounds=(lower, upper))` equal to its known
support.

The [arm-indexed contract](../technical-reference/scope-and-refusals.md#missing-outcome-arm-indexed-contract)
lists every refusal. Without a known finite support, fit in sample with
`CrossFitting(enabled=False)`. The in-sample fit keeps whichever fold policy you declare, because
one fold balances nothing.

[One treatment, one outcome](../examples/point-treatment-tmle.ipynb) runs an `ATE` from start to
finish, and its "Go deeper" section fits the `ATT`, the `ATC` and both counterfactual means. [Gain
over current practice](../examples/population-impact.ipynb) adds risk and odds ratios and the
population-impact estimands.

## Known regimes

Use `Static`, `Rule`, or `Stochastic` interventions with `RegimeMean`, which reports the mean
outcome under each plan, or `RegimeContrast`, which reports each plan's difference from the
reference plan. With two competing events, `Separable` sets the part of a treatment that acts on
each event to its own level, as
[A medicine with two effects on readmission and death](../examples/separable-effects.ipynb)
shows.

```python
import numpy as np

from cleverly import RegimeContrast
from cleverly.interventions import Rule, Static, Stochastic

plans = (
    Static(1, name="treat all"),
    Static(0, name="treat none"),
    Rule(
        lambda data: (data["W1"] > 0).astype(float),
        name="treat if W1 positive",
        rule_kind="known",
    ),
    Stochastic(
        lambda data: np.column_stack([np.full(len(data), 0.4), np.full(len(data), 0.6)]),
        name="assign with probability 0.6",
        density_kind="known",
    ),
)
result = study.estimate(RegimeContrast(plans, reference="treat none"), method=quick, random_state=3)
```

Write a `Rule` as a fixed rowwise function of the covariates, and declare `rule_kind="known"`.
Choose the rule independently of the analysis sample. `Rule` raises `CapabilityError` for a rule
with no declaration and for a rule declared `"estimated"`. For a threshold at a sample mean, the
fixed-rule curve omits the derivative through that mean. Code cannot inspect a closure, so the
fit accepts a false `"known"` declaration. The [scope page](../technical-reference/scope-and-refusals.md#wrong-by-construction)
gives the size of that error on one exact law.

`Stochastic` takes the assignment *density*, not a scalar probability. The density is a function
of the covariate frame. It returns one column per arm in `data.treatment_levels` order, and its
rows sum to one. For this API, choose it independently of the analysis sample, and declare
`density_kind="known"`.

`Stochastic` raises `CapabilityError` for a density with no declaration and for a density declared
`"estimated"`. For a population-law target whose $g^\star$ depends on $P$, the regime curve
omits the derivative of that dependence. A realized learned density is a different target whose
inference needs conditions this API does not check. For the population odds tilt of the treatment
mechanism, use an [incremental intervention](policies.md#incremental-propensity-score-interventions).

On a point-treatment design, each typed estimand holds one intervention kind.
`CausalStudy.identify` raises `CapabilityError` when a set holds another kind, such as an
`Incremental` in `RegimeContrast`. The message names the typed estimands for that item.
`identify` raises `DataError` when the set is a mapping, an iterator, a string, or a single item.
Write one item as a one-item tuple.
[X57](../roadmap.md#x57-joint-intervention-axes-and-tilt-compositions) tracks a fit with two kinds.

## Marginal structural models

`MSMProjection` summarizes the mean outcomes under the treatment options with a simple formula
you choose, such as a straight line in the treatment (a marginal structural model, MSM).
`MSM.linear()` builds that line and needs no declaration; a formula you write as a function needs
`design_kind="known"`. The coefficients describe the formula's best fit to the true means, even
when the true means do not follow it.
[MSM projections](../technical-reference/msm-projections.md), section "Variations", gives the
refusals.

```python
from cleverly import MSMProjection
from cleverly.msm import MSM

trend = study.estimate(MSMProjection(MSM.linear(modifiers=("W1",))), method=quick, random_state=3)
dose_trend = dose_study.estimate(
    MSMProjection(MSM.linear(doses=(1.0, 2.0, 3.0))), method=quick, random_state=5
)
```

On the yes-or-no treatment, `msm[a]` is the line's effect at `W1` equal to 0, and `msm[a:W1]` is
how the line's effect changes per unit of `W1`. On the dose, `msm[a]` is the slope of the best
straight line through the mean outcomes at the doses 1, 2 and 3, with the two end doses weighted
half as much as dose 2.

On a continuous dose, `MSMProjection` needs every outcome observed. `CausalStudy.identify`
raises `CapabilityError` when an outcome is missing. The refused table of
[MSM projections](../technical-reference/msm-projections.md#variations) lists this refusal.
[X10](../roadmap.md#x10-continuous-dose-msm-with-a-second-mechanism) tracks that composition.

[A trend across three contact schedules](../examples/msm-projections.ipynb) projects the arm means
onto a line with `MSMProjection`.

## Controlled direct effects

A controlled direct effect asks what the treatment does when a later yes-or-no variable is held
at one value for everyone. The variable is one the treatment can change, such as whether a patient
was readmitted. Declare it with `PointTreatment(intermediate="Z")` and ask for
`ControlledDirectEffect(intermediate=z)`, with `z` equal to 0 or 1. Its `contrast=` takes `ATE()`
(the default), `ATT()`, `ATC()`, `RiskRatio()`, `OddsRatio()`, or a `ConditionalContrast` from
[effects that differ between people](conditional-effects.md).

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, ControlledDirectEffect, PointTreatment
from cleverly.datasets import make_cde

visits, _ = make_cde(n=1000, seed=2)
cde_study = CausalStudy(
    visits,
    design=PointTreatment(
        outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), intermediate="Z"
    ),
)
held = {
    z: cde_study.estimate(
        ControlledDirectEffect(intermediate=z, contrast=ATE()),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        intermediate_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        random_state=0,
    )
    for z in (0.0, 1.0)
}
```

The example fits without cross-fitting because its outcome has no known range. With a known
range, pass `q_bounds=(lower, upper)` and keep the default cross-fitting.

The two levels are two different effects, and together they do not split the total effect into a
direct and an indirect part. Each needs no unmeasured common cause of the treatment and the
outcome, given the covariates. It also needs none of the intermediate and the outcome once the
treatment and the covariates are known. Every patient must have had a real chance of each
treatment, and of the held value of the intermediate under either treatment.
[Conditional effects](../technical-reference/conditional-effects.md), section "Controlled direct
effects", states the conditions. [Gain over current practice](../examples/population-impact.ipynb)
reports one beside the population-impact estimands, and [effects by group](../examples/subgroup-effects.ipynb)
fits one for each living situation, with a `ConditionalContrast` as its contrast.

See the [technical implementation matrix](../technical-reference/index.md) for assumptions and
citations for every family.
