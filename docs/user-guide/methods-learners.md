# Methods, learners, and cross-fitting

## Method presets

`"tmle"` is the ordinary default. `CollaborativeTMLEMethod` and `DRTMLEMethod` select compatible
estimator variants; they do not change the causal question.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import CollaborativeTMLEMethod, DRTMLEMethod, ModelSpec

parametric = ModelSpec(
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)

collaborative = effect.estimate(
    method=CollaborativeTMLEMethod(models=parametric, strategy="greedy"),
    random_state=3,
)
doubly_robust = effect.estimate(
    method=DRTMLEMethod(models=parametric),
    random_state=3,
)
```

Call `effect.available_methods()` before fitting when method availability matters. An unavailable
method comes with a capability reason.

`SDRMethod` selects the sequentially doubly robust estimator for a longitudinal functional. A
point-treatment functional lists it as unavailable. The estimator fits no fluctuation, so it
refuses a non-default `submodel_alpha`, `max_iter`, `tol`, `stopping`, `step_bound`, `step_search`
or `iterate`. Its `pseudo_learner` fits a
transform that can leave $[0, 1]$, so give it a learner of a real-valued target, such as
`LinearRegression()`. The default is a regression Super Learner with no clip.
[The sequentially doubly robust estimator](../technical-reference/longitudinal-tmle.md#the-sequentially-doubly-robust-estimator)
states its contract.

`IPSIMethod` selects the incremental estimator of Kennedy (2019) for `IncrementalMean` or
`IncrementalEffect` on a longitudinal design. A point-treatment functional lists it as
unavailable. It fits no fluctuation either, so it refuses the same settings. At a tilt
node the treatment mechanism defines the plan, and the estimator is not doubly robust. Give
`treatment_learner` a learner that can fit the mechanism at every node.
[Incremental interventions over time](../technical-reference/longitudinal-tmle.md#incremental-interventions-over-time) states
its contract.

The collaborative fit above reports no confidence interval, no p-value, and no standard error. The
`"greedy"` strategy is the default. `"ordered"`, `"discrete"`, `"lasso"`, `"learners"`, and `"oat"`
refuse the same three accessors. Each estimate keeps `plugin_std_error` and `plugin_interval` as
a diagnostic. For an interval, use `TMLEMethod` or `strategy="derivative"`.
[Collaborative TMLE](../technical-reference/collaborative-tmle.md) gives the reason for each
strategy.

### C-TMLE0: a derivative covariate with an interval

`strategy="derivative"` fits one TMLE with an extra covariate for each arm. That covariate is the
derivative of the clever covariate along the lasso penalty. The strategy selects nothing, so it
reports the TMLE interval. Declare two penalties in `lambdas=`: the mechanism first, then its
partner. Omit `lambdas=` to choose the mechanism penalty by cross-validation.

```python
derivative = effect.estimate(
    method=CollaborativeTMLEMethod(models=parametric, strategy="derivative", lambdas=(0.02, 0.01)),
    random_state=3,
)
print(derivative.extra["ctmle"].describe())
```

The fit prints the penalty pair and the number of training sets. The default cross-fit has 10. The
[C-TMLE0 section](../technical-reference/collaborative-tmle.md#c-tmle0-the-derivative-covariate)
lists the other two sources of the pair and the refusals.

### A lasso grid that starts at a cross-validated penalty

`strategy="lasso"` takes its penalties from `lambdas=`. To start the grid at a cross-validated
penalty, fit `LogisticRegressionCV` with an L1 penalty on standardized covariates. Then convert its
`C` with `lambda = 1 / (C n)`. Use the `saga` solver. The `liblinear` solver penalizes the
intercept, and the lasso candidates do not.

`l1_ratios=(1.0,)` selects the L1 penalty in scikit-learn 1.8 and later. `cleverly` requires
scikit-learn 1.8 or later. Earlier releases ignore `l1_ratios` without `penalty="elasticnet"` and
fit an L2 penalty. `use_legacy_attributes=False` makes `C_` one number.

```python
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

scaled = StandardScaler().fit_transform(frame[["W1", "W2", "W3"]])
search = LogisticRegressionCV(
    Cs=10,
    l1_ratios=(1.0,),
    solver="saga",
    scoring="neg_log_loss",
    max_iter=5000,
    use_legacy_attributes=False,
)
search.fit(scaled, frame["A"])
start = 1.0 / (search.C_ * len(frame))
lambdas = start * np.geomspace(1.0, 0.01, 6)
lasso = effect.estimate(
    method=CollaborativeTMLEMethod(models=parametric, strategy="lasso", lambdas=lambdas),
    random_state=3,
)
```

`StandardScaler` divides by the population standard deviation, as the lasso candidates do.
`tests/unit/test_ctmle_candidate_sequences.py::TestConfiguration::test_the_public_lambda_cv_recipe`
runs this recipe. It also checks that the search sets some coefficients to exactly 0, which an L2
search never does.

### Declared learner candidates

`strategy="learners"` reads `treatment_candidates=` in the declared order, then appends
`treatment_learner` on every covariate. Order the candidates from the least to the most adaptive.
Each candidate must fit the treatment better than the candidate before it. The search tries the
candidates in this order, so an unordered list can stop the search before a confounder enters.

Nested candidates are the safest declaration. Make each candidate read the covariates of the
candidate before it, and add more. The table shows a declaration on the instrument law of
`tests/studies/ctmle_sequence_properties.py`, where `W1` is the confounder and `W2` the instrument.

| declaration | candidates, then the terminal | treatment loss |
| --- | --- | --- |
| nested | `W2`, then `W1` and `W2`, then every covariate | decreases, except for ties |
| not nested | `W2`, then `W1`, then every covariate | rises at the `W1` candidate |

The fit warns with a `UserWarning` when the treatment negative log likelihood of a candidate rises
by more than 1% above the candidate before it. The warning names each such candidate by its index
and label. The fit continues, because a nested candidate with an uninformative covariate can rise
by sampling noise under the propensity bounds. The 1% threshold separates the two cases on the
instrument law. There the nested terminal rises by about 0.2%, and the `W1` candidate by about 19%.
`CTMLESelection.treatment_risk_increases` holds the same indices.
`tests/unit/test_ctmle_candidate_sequences.py::TestLearners` checks the warning.

Both variants are named here with a parametric learner on purpose. Neither fits a fixed number of
nuisance models: collaborative TMLE fits one per candidate along its selection path, and DR-TMLE
re-fits its reduced regressions on every round of the alternation that solves its extra score
equations, because those regressions are regressions *on* the current targeted pair. Each of those
fits is then multiplied by the cross-fitting folds and again by the learner library. Leaving both
at the default library is the difference between seconds and hours on the same data, so decide the
learner deliberately for these two rather than inheriting it.

## Immutable configuration

```python
from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod, Targeting

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3, repeats=1),
    targeting=Targeting(g_bounds="auto", algorithm="iterative"),
    inference=Inference(alpha=0.05, simultaneous=False),
    runtime=Runtime(random_state=3, n_jobs=1),
)
result = effect.estimate(method=method)
```

Keyword shortcuts such as `n_folds=`, `alpha=`, and `random_state=` normalize into these same
groups. A method object makes the normalized configuration serializable and reviewable.

This example fits a binary outcome, so its `Targeting` declares no `q_bounds`. Read [Two fold layers](#two-fold-layers) below before you copy the shape onto a continuous
outcome, which needs its support declared there.

## Learner choices

Every nuisance slot takes an sklearn-compatible estimator object. Strings such as `"glm"` and
`"default"` are rejected so the configuration always identifies the actual model being fitted.

When learners are omitted, cleverly constructs a task-appropriate `SuperLearner` over three
concrete sklearn candidates: histogram gradient boosting, a random forest, and lasso (`LassoCV`
for a mean regression or L1 logistic cross-validation for a probability). No optional model
package is imported. Install XGBoost or LightGBM yourself and pass its estimator object when that
is the model you want.

Construct an ensemble explicitly by passing model objects, optionally paired with report names:

```python
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from cleverly import SuperLearner

outcome_model = SuperLearner(
    library=[
        ("boost", HistGradientBoostingRegressor(random_state=3)),
        ("forest", RandomForestRegressor(n_jobs=1, random_state=3)),
        ("linear", LinearRegression()),
    ],
    n_folds=3,
    random_state=3,
)
```

The estimator clones each candidate before fitting. The objects supplied by the caller therefore
remain unfitted and can be reused in another method configuration.

Library size multiplies with both fold layers below, so it is the first thing to reach for when a
fit is slower than expected.

Outcome regressions need predictions on the outcome's mean scale. Binary treatment and censoring
mechanisms need probability predictions. Continuous treatment policies use a conditional density
model. `cleverly` validates learner task compatibility and sample-weight support.

## Use GLM for the sequential baselines

`cleverly.learners.GLM` is an unpenalized maximum-likelihood fit with a canonical link. Its score
and information are public, so `SequentialGComputation` and `SequentialIPW` stack them into an
interval. See [Report the sequential baselines](longitudinal.md#report-the-sequential-baselines).

| behavior | what `GLM` does |
| --- | --- |
| `family="binomial"` on a target with three or more labels | raises `ValueError`. Pass `family="multinomial"` or `"auto"` |
| a design column that is constant, or a combination of other columns, on the fit rows | aliases the column. It gets no coefficient |
| `alias_` | holds each aliased column as a combination of the active columns |
| `outside_span(x)` | returns `True` for each row where an aliased column departs from that combination |

A prediction through an aliased column assumes that the column has no effect. On a row that
departs from the fit rows, the data cannot support that assumption. Sequential g-computation
refuses such a fit with `DataError`, because no follower has that history. The technical page lists this refusal for
g-computation only.

```python
import numpy as np

from cleverly.learners import GLM

x = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
model = GLM().fit(x, np.array([0, 1, 1, 0]))
print(model.active_)
print(model.outside_span(np.array([[0.0, 1.0], [1.0, 1.0]])))
```

The first column is constant on the fit rows, so the fit aliases it with the intercept. The row
`[0.0, 1.0]` departs from that relation.

## Use GLM for the parametric g-formula

`ParametricGFormula` fits a `GLM` for every covariate, outcome and hazard model by default. The
fit chooses the family of each model from its response: binomial for a binary column, multinomial
for a categorical one and Gaussian for a normal one. A `GLM` with another family is refused.
The forward draw of a Gaussian column needs the residual variance of the fit, so a Gaussian
covariate accepts a `GLM` only. A data-adaptive learner for any other model gives the status
`adaptive_baseline_plugin`. See
[Simulate the parametric g-formula](longitudinal.md#simulate-the-parametric-g-formula).

Choose `ParametricGFormula` over `SequentialGComputation` when the plan fixes a continuous dose,
or when you need the forward-simulation form. Choose `SequentialGComputation` when you doubt a
covariate law, because it models no covariate.

## Two fold layers

- `n_folds` defines the outer cross-fitting split used to keep each observation out of the nuisance
  fit that predicts it. Default 10.
- `learner_folds` defines internal model-selection folds for learners such as a Super Learner.
  Default 5.
- `repeats` repeats the outer split and reports the median of the draws. Default 1. The value must
  be at least 1, and a value above 1 requires `enabled=True`. A repeated fit refuses a simultaneous
  band, because the band needs the joint influence curves of one draw. Pass `simultaneous=False` to
  a repeated fit.
- `enabled=False` sets a one-fold analysis with no cross-fitting.
- `stratify_by="none"` draws unstratified folds, so the split reads neither the treatment nor the
  outcome. This is the default, and it is the only policy a fit that draws a split accepts.
- `stratify_by="treatment"` and `stratify_by="treatment+outcome"` are refused when the fit draws
  folds. A selector-based collaborative fit refuses them at every setting, because its search draws
  selection folds without cross-fitting. In-sample outcome-adaptive C-TMLE draws no selection folds
  and accepts an unused policy.

`CrossFitting` raises `MethodConfigurationError` when you construct a declaration its own checks
refuse. Four inputs earn one: `repeats` below 1, `repeats` above 1 with `enabled=False`,
`enabled=True` with fewer than two folds, and a `stratify_by` above. The engine raises `ValueError`
with the same reason. [Fold and outcome-scale rules](../technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
gives every message and the audit behind it.

A cross-fitted fit of a continuous outcome needs a declared `Targeting(q_bounds=(lower, upper))`
equal to the outcome's known support. With `q_bounds=None` the scale would come from every observed
outcome, held-out rows included. The fit is refused before any learner. Declare the support, or fit
in sample with `CrossFitting(enabled=False)`. Never read the interval off the sample.

Because the split reads no treatment, it cannot keep a rare arm in every training complement. The
fit checks the realized draw instead, before the first learner. Each arm must appear in at least
two independent units, which are rows for iid data and *clusters* under `id=`. Each training
complement must hold every arm, and both observed outcome classes of a binary outcome. A package
classification `SuperLearner` needs two rows of each class of its target in each complement, which
the default library makes the common case.

None of these refusals names a redraw. The remedy is `CrossFitting(enabled=False)`, or more
observations at the rare level.

| missing-outcome fit | what it needs beyond the rules above | contract table |
| --- | --- | --- |
| binary `NaturalCourseMean`, stacked | a binary outcome, pooled targeting, and one repeat | [natural-course contracts](../technical-reference/scope-and-refusals.md#missing-outcome-natural-course-contracts) |
| arm-indexed means and contrasts, cross-fitted | ordinary TMLE, and no `ATT` or `ATC` target | [arm-indexed contract](../technical-reference/scope-and-refusals.md#missing-outcome-arm-indexed-contract) |
| arm-indexed means and contrasts, per-arm `CTMLE(strategy="oat")` | a respondent in each arm and stratum of each training complement, and a nonrespondent in each complement. The fit reports a point estimate only | [per-arm design contract](../technical-reference/collaborative-tmle.md#baseline-strata-and-missing-data-in-the-per-arm-design) |
| shift, incremental, regime, MSM, and controlled-direct-effect targets, cross-fitted | no contract. The fit refuses them before the first learner | [not written yet](../technical-reference/scope-and-refusals.md#not-written-yet) |

To fit any target in the table in sample, set `CrossFitting(enabled=False)`. One fold balances nothing, so
the fold policy makes no difference there.

The two layers multiply: one nuisance fit at the defaults is `10 × 5` model fits per library
candidate, before an estimator variant multiplies it again. The examples in this documentation set
`n_folds=5, learner_folds=3` to stay quick to run, which is not a recommendation for an analysis.

Keep all rows from one declared cluster in the same fold. A grouped draw does that by permuting the
distinct cluster labels and cutting them into near-equal parts. Ordinary TMLE, DR-TMLE and
longitudinal TMLE accept `id=` under cross-fitting. C-TMLE refuses `id=` at every setting, except `strategy="derivative"`.
Cross-fitting reduces empirical-process bias. It does not establish positivity, correct
identification, or nuisance consistency.

Four fold declarations are refused by name. A blocked temporal split needs a row-level time
ordering that no design role carries. A rolling-origin split nests its training sets, so no single
fold holds out each row, and `Folds` stores exactly one fold per row. The rolling-origin refusal
would survive a time index; it asks for a different storage contract. The two stratified policies
are refused for a third reason, which the fold and outcome-scale rules give.

## Reuse an outer split

Every point-treatment result records its realized outer folds as an immutable `split_plan`. Most
point-treatment methods accept that plan to compare methods on the same validation rows.

A fit accepts a plan only when the plan records how `random_partition` drew each repeat. Before
the first learner runs, the fit draws each repeat again from that record and compares the labels.
A plan whose labels no recorded draw produces could have been chosen by reading the outcome, and
the fit refuses it. Two plans carry the record: the plan `SplitPlan.from_folds` copies off
`random_partition` draws, and `result.split_plan` from a cross-fitted fit. A plan you build from
labels alone carries no record, and neither does the one-fold plan of an in-sample fit.

```python
from cleverly import SplitPlan
from cleverly.learners import random_partition

plan = SplitPlan.from_folds([random_partition(result.data.n, 5, seed=3)])
reused_method = TMLEMethod(
    models=method.models,
    cross_fitting=CrossFitting(
        n_folds=5,
        learner_folds=3,
        repeats=1,
        split_plan=plan,
    ),
    targeting=method.targeting,
    inference=method.inference,
    runtime=method.runtime,
)
reused_result = effect.estimate(method=reused_method)
assert reused_result.provenance.fold_fingerprint == plan.fingerprint
```

Set `repeats` to the repeat count the plan records. Set `n_folds` to the count the plan was
declared under, which can exceed the plan's own fold count when the data capped it. A declaration
the plan cannot serve raises `MethodConfigurationError` instead of changing the plan.

A plan read off a result is bound to the rows that produced it, by position. Reuse it on those
rows, in that order. Call `plan.unbound()` to reuse the labels on other rows deliberately. The
method keeps the generator record and drops the binding.

The two stacked missing-outcome contracts do not accept `split_plan=`. They cover the binary
`NaturalCourseMean` and the cross-fitted arm-indexed means and contrasts. Each audited contract
covers package-generated near-balanced folds only. The result still records the realized plan for
provenance. The
[natural-course table](../technical-reference/scope-and-refusals.md#missing-outcome-natural-course-contracts)
and the
[arm-indexed table](../technical-reference/scope-and-refusals.md#missing-outcome-arm-indexed-contract)
list the other refusals.

[Reusable outer split plans](../technical-reference/cv-tmle.md#reusable-outer-split-plans) states
the whole contract: the counts, the row binding, what validation checks, and every refusal.

## Targeting and bounds

`g_bounds="auto"` chooses target-aware treatment-mechanism truncation. The bound changes the
finite-sample estimating procedure and should be reported with support diagnostics. The logistic
submodel bound, `submodel_alpha`, is separate from the confidence interval's `alpha`.

Four `Targeting` options choose how the targeting step runs. Each default runs the shipped step.

| option | what it chooses |
| --- | --- |
| `stopping="scaled_var"` or `"sample_size"` | the `tmle3` rule that decides the score equation is solved. The result prints the score residual in standard errors |
| `step_bound=` and `step_search=` | the longest step of one iteration, and whether a line search sets its length |
| `iterate="best"` | the iterate of smallest score norm, in place of the last |

`ModelSpec(outcome_by_arm=True)` fits one outcome regression for each treatment arm.
`CollaborativeTMLEMethod(patience=p)` stops the scoring of the candidate path early.
`PointTreatment(treatment_adjustment=[...])` lets the treatment mechanism read a subset of the
adjustment set. That declaration restricts the statistical model.
[Stopping rules, step bound and best iterate](../technical-reference/point-treatment-tmle.md#stopping-rules-step-bound-and-best-iterate)
and [treatment-mechanism covariates](../technical-reference/point-treatment-tmle.md#treatment-mechanism-covariates)
state the contracts.

Unknown settings and settings an engine cannot use raise `MethodConfigurationError` before fitting
rather than being ignored. A longitudinal fit refuses a point-only control such as `repeats=`
instead of accepting it and discarding it.
