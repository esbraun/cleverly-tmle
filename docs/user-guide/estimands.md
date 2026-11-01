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
`q_bounds`. A stacked CV-TMLE fit accepts a binary outcome and uses this cross-fitting declaration:

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
fit generates one near-balanced outer partition. Both configurations require iterative unweighted
logistic targeting. The treatment can have two or more arms. The ordinary fit also accepts fixed
weights and clusters. The stacked fit requires unweighted iid rows.

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
cross-fitting with missing outcomes, only `TMLEMethod` fits these targets.
`CollaborativeTMLEMethod` and `DRTMLEMethod` are refused. `ATT` and `ATC` are refused under
cross-fitting with missing outcomes. Two further requirements are not special to this surface.
Every cross-fitted fit draws `CrossFitting(stratify_by="none")` folds, which is the default, and
every cross-fitted continuous outcome needs `Targeting(q_bounds=(lower, upper))` equal to its known
support.

The [arm-indexed contract](../technical-reference/scope-and-refusals.md#missing-outcome-arm-indexed-contract)
lists every refusal. Without a known finite support, fit in sample with
`CrossFitting(enabled=False)`. The in-sample fit keeps whichever fold policy you declare, because
one fold balances nothing.

## Known regimes

Use `Static`, `Rule`, or `Stochastic` interventions with `RegimeMean` or `RegimeContrast`.

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
mechanism, use an [incremental intervention](#incremental-propensity-score-interventions).

On a point-treatment design, each typed estimand holds one intervention kind.
`CausalStudy.identify` raises `CapabilityError` when a set holds another kind, such as an
`Incremental` in `RegimeContrast`. The message names the typed estimands for that item.
`identify` raises `DataError` when the set is a mapping, an iterator, a string, or a single item.
Write one item as a one-item tuple.
[X57](../roadmap.md#x57-joint-intervention-axes-and-tilt-compositions) tracks a fit with two kinds.

## Learned treatment rules

Use `LearnedRuleValue` to estimate the value of a treatment rule that the fit learns from the data.
The fit learns one rule inside each training fold and reports an average of the values of those
rules, weighted by fold (equal weights in the example below). The interval can under-cover when the effect is zero or near zero for a share of units.
[Learned rules](../technical-reference/point-treatment-tmle.md#learned-rules) states the
conditions.

Two registered studies measure the 95% interval over 6,000 replications at n = 2,000. The table
gives its coverage at each law.

| law | coverage | reading |
| --- | --- | --- |
| no effect for any unit | 0.8938, with a 99% interval of 0.8832 to 0.9038 | `under-covers at the exceptional law`, in the [boundary study](../technical-reference/method-evidence/learned-rule-cvtmle-boundary.md) |
| a weak effect, $0.15 W_1$ on the logit scale | 0.8948, with a 99% interval of 0.8842 to 0.9048 | `under-covers at the weak_blip law`, in the same study |
| two laws that meet the published conditions, one with a correctly specified outcome learner and one with a misspecified one | 0.9465 and 0.9413 | every verdict passes, in the [gated study](../technical-reference/method-evidence/learned-rule-cvtmle.md) |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from cleverly import CrossFitting, LearnedRuleValue, ModelSpec, TMLEMethod

fold_evaluated = TMLEMethod(
    models=ModelSpec(
        outcome_learner=make_pipeline(
            PolynomialFeatures(2, interaction_only=True, include_bias=False),
            LogisticRegression(max_iter=1000),
        ),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5, fold_evaluation=True),
)
learned = study.estimate(LearnedRuleValue(), method=fold_evaluated, random_state=3)
record = learned.extra["learned_rule"]
(split,) = record.splits  # one split into five folds
shares = split.treated_shares
```

The rule of fold $v$ treats a unit when the outcome regression fitted on the other folds predicts
a higher outcome under treatment. A tie assigns control. The outcome learner therefore sets the
rule, so give it the treatment-by-covariate terms that the rule can depend on.

The reported value is a data-adaptive target. It depends on the realized split and on the fitted
rules. It is not the value of one rule fitted on all rows (the in-sample fit below), and it is not
the value of the optimal rule. The table gives what the result holds.

| where | content |
| --- | --- |
| `learned.estimates` | one parameter, `ey_learned_rule[learned rule]` |
| `learned.extra["learned_rule"].splits` | for each split into folds, the fold sizes, the fold estimates, the treated share of each fold, and quantiles of the estimated blip. Under the default `CrossFitting()` the fold estimates describe each fold's part of one pooled estimate, not separate estimates |
| `learned.nuisance.regimes.values[:, 1, 0]` | the rule of each row, `1.0` for treat |
| `learned.summary()` | a line that states the target |

Use `CrossFitting(n_folds=5, fold_evaluation=True)`, as above. It scores each fold's rule on that
fold alone and gives every fold equal weight, and the studies above measure that setting. The
default `CrossFitting()` also runs: it scores all held-out rows together and weights each fold by
its share of the rows (of the total weight, when you pass `weights=`). Its result is marked
experimental until a study measures it.

`CrossFitting(enabled=False)` runs too, and it answers a different question. It learns one rule
from all the rows and reports the value of that one rule, scored on the same rows that chose it.
Its interval rests on two things: the rule settles as the data grow, and the models are simple
enough not to chase noise in the rows that chose the rule. A model with a fixed set of terms and
nothing selected from the data is simple enough. cleverly reads this from the model's class: a
`LogisticRegression`, a `LinearRegression`, a `GLM`, or a class marked
`cleverly_learner_family = "parametric"` passes, and any other model, the pipeline above
included, makes the fit warn with `InSampleLearnerWarning`.

To declare that the pipeline above has fixed terms, subclass it and add the mark:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from cleverly import CrossFitting, LearnedRuleValue, ModelSpec, TMLEMethod


class FixedTerms(Pipeline):
    cleverly_learner_family = "parametric"


in_sample = TMLEMethod(
    models=ModelSpec(
        outcome_learner=FixedTerms(
            [
                ("terms", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
                ("fit", LogisticRegression(max_iter=1000)),
            ]
        ),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(enabled=False),
)
full_sample = study.estimate(LearnedRuleValue(), method=in_sample, random_state=3)
```

The mark is your statement that the model has fixed terms, and cleverly does not test it. With
it the fit gives no warning. No study measures this fit yet, so its result is still marked
experimental.

Every one of these fits needs, among other things, a binary treatment, complete outcomes and no
intermediate variable, and a cross-fitted fit needs at least two folds. A fit that lacks one of
them refuses before any model is fitted. The refusal table under
[Learned rules](../technical-reference/point-treatment-tmle.md#learned-rules) lists the rest.

A refit that relearned the rules would estimate the value of other rules. So the `subset`,
`random_common_cause`, `bootstrap_stability` and `bootstrap_measurement_error` tests of `refute`,
and `simulated_confounding`, keep the outer folds and the rule of each row, and read the value of
the fit's own rules. The `placebo` test refits as the fit did: under random treatment every rule
has the value of the mean outcome, which is what it reads against. The generated-outcome tests
refuse, because a learned rule has no generated truth yet.

The omitted-variable bound runs for this target. It gives a point bound under cross-fitting, and
in sample it also gives its one-sided limits. `benchmark()`, the E-value and the missingness tilt
read `unavailable`. The support report and `truncation_curve` run.
[X11](../roadmap.md#x11-learned-policy-follow-ups) and
[F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) hold the refused
requests.

The [treatment-rules example](../examples/treatment-rules.ipynb) runs this fit end to end.

## Modified treatment policies

For a continuous dose, `Shift` maps each observed dose to a policy dose. `cap=` is part of the
policy and its support argument; it is not estimated from the sample maximum. `q_bounds` follows
the same rule for the outcome. Declare it from known support, and never from the observed range.

```python
from cleverly import ModifiedTreatmentPolicyEffect
from cleverly.interventions import Shift

policies = (Shift(0.0, cap=None), Shift(0.25, cap=4.0), Shift(0.5, cap=4.0))
result = dose_study.estimate(
    ModifiedTreatmentPolicyEffect(policies),
    method=quick,
    density_bins=40,
    random_state=5,
)
```

A shifted dose above the cap holds the row at its own dose. The example cap of 4.0 lies inside
the observed doses. The fit issues a `PositivityWarning` when a declared cap lies above the largest
observed dose and the policy assigns a dose above that maximum.

The correction step needs the ratio of the density of each policy dose to the density of the
observed dose. By default the fit estimates the dose density in `density_bins` bins. With
`ratio="classifier"`, the treatment learner instead learns to tell each observed row from its
policy copy, and the ratio comes from its predicted odds. The classifier route fits no density,
so it suits a flexible classifier when the bins are too coarse.
[Modified treatment policies](../technical-reference/point-treatment-tmle.md#modified-treatment-policies)
states both routes.

A cap can also come from the data. `Shift(1.0, cap=DoseQuantile(0.9), target="population")` holds
every dose whose shift would pass the 90th percentile of the observed doses.
`SupportBoundary(LinearQuantileRegression(features=...), q=0.9)` sets the cap from a quantile
regression of the dose on the named covariates, so the cap varies by row. `DoseQuantile`,
`SupportBoundary` and `LinearQuantileRegression` come from `cleverly.interventions`.

A tuned cap needs `target=`, and the fit refuses one without it. `target="population"` estimates
the mean under the cap that the whole population would give, and the interval includes the error
of estimating that cap. It needs an outcome model that is accurate near the cap.
`target="fold_average"` averages the means under each fold's own cap. The [tuned-cap study](../technical-reference/method-evidence/mtp-tuning.md) checks the
population target over two decisions. A tuned cap at one decision rests on exact-law tests only
(`tests/unit/test_point_tuned_caps.py`).

Other maps of the dose use the same `policies=` axis. The table names each class.

| class | use it for |
| --- | --- |
| `Scale(factor, cap=...)` | a dose multiplied by a factor, held at its own value above the cap |
| `Piecewise(pieces)` | a `Shift` or a `Scale` on each interval of the dose |
| `ModifiedPolicy(name, pieces=..., policy_kind="known")` | your own map, with its inverse and its derivative on each piece |
| `ModifiedPolicy(name, apply=..., policy_kind="known")` | a map of the levels of a categorical treatment |
| `RiskRatioTilt(delta)` | the `ipsi` tilt of `lmtp` on a 0/1 treatment, reported as `ey_rr_tilt`. Fit it with `TMLE(policies=...)`. The typed study requests refuse it |
| `JointPolicy(name, components={column: policy}, policy_kind="known")` | a vector treatment of several columns. It sets one policy for each component, as the [vector treatments](../technical-reference/point-treatment-tmle.md#vector-treatments) section states |

```python
from cleverly.interventions import ModifiedPolicy, Piece, Scale

halve_below_3 = ModifiedPolicy(
    "halve below 3",
    pieces=(
        Piece(
            -float("inf"),
            3.0,
            lambda a, h: (a + 3.0) / 2.0,
            lambda b, h: 2.0 * b - 3.0,
            lambda b, h: 2.0,
        ),
        Piece(3.0, float("inf")),
    ),
    policy_kind="known",
)
policies = (Shift(0.0, cap=None), Scale(1.25, cap=5.5, name="x1.25"), halve_below_3)
```

A `Piece` without a map leaves its doses unchanged. The fit checks that every observed dose lies in
exactly one piece. It also checks that each declared inverse inverts its map, and that each declared
derivative matches the map. In a longitudinal fit, a policy is a node of
a `DynamicRegimen` plan. The
[longitudinal reference](../technical-reference/longitudinal-tmle.md#modified-treatment-policies-at-a-node)
gives the details, and
[limits of the continuous-policy evidence](../technical-reference/longitudinal-tmle.md#limits-of-the-continuous-policy-evidence)
lists what the studies do not cover. The [monthly prices](../examples/sequential-pricing.ipynb)
example raises a price at each of three months.

## Incremental propensity-score interventions

`Incremental(delta)` multiplies the observed treatment odds by `delta`. It avoids a conventional
positivity requirement, but the target depends on the true treatment mechanism and therefore does
not have ordinary two-sided double robustness.

A treatment with three or more levels takes one factor for each level. Declare the factors as a
mapping from the level label to the factor. A level that the mapping does not name keeps the
factor 1. The odds of level $a$ against level $b$ are multiplied by the ratio of their factors.

| declaration | meaning |
| --- | --- |
| `Incremental(2.0)` | two levels only: the odds of the second sorted level times 2 |
| `Incremental({"high": 2.0})` | the odds of `"high"` against every other level times 2 |
| `Incremental({"high": 2.0, "low": 0.5})` | the factors of `"high"` and `"low"`, and 1 for the other levels |

A scalar on a treatment with other than two levels is refused before any learner, and so is a key
that is not a level. Keys are the labels that the data reports, and a position is not a label.
A vector treatment whose columns are all categorical is one treatment over its observed tuples.
Its keys are the tuples. In a longitudinal fit, one `Incremental` applies to every node, and its
keys are read against the levels of each node in turn.

A fit with three or more levels carries the label of the registered study
[categorical incremental policies](../technical-reference/method-evidence/categorical-incremental.md)
when it matches that study's rows. Any other such fit is labelled `experimental`.

```python
from cleverly import IncrementalEffect
from cleverly.interventions import Incremental

tilts = (Incremental(1.0), Incremental(2.0), Incremental(0.5))
result = study.estimate(IncrementalEffect(tilts), method=quick, random_state=3)
```

## Marginal structural models

`MSMProjection` defines a projection of counterfactual means onto a declared working model. A
coefficient remains a projection parameter when the working model is misspecified. A written
`design=` function needs `design_kind="known"`. A design that `MSM.linear` builds needs no
declaration.
[MSM projections](../technical-reference/msm-projections.md#variations) gives the refusals.

On a continuous dose, `MSMProjection` needs every outcome observed. `CausalStudy.identify`
raises `CapabilityError` when an outcome is missing. The refused table of
[MSM projections](../technical-reference/msm-projections.md#variations) lists this refusal.
[X10](../roadmap.md#x10-continuous-dose-msm-with-a-second-mechanism) tracks that composition.

## Controlled direct effects

`ControlledDirectEffect(intermediate=z)` compares treatment arms while setting the declared
intermediate to `z`. Fits at two intermediate levels are two controlled direct effects, not an
automatic direct / indirect decomposition.

See the [technical implementation matrix](../technical-reference/index.md) for assumptions and
citations for every family.
