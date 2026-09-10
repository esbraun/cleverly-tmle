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

## Two fold layers

- `n_folds` defines the outer cross-fitting split used to keep each observation out of the nuisance
  fit that predicts it. Default 10.
- `learner_folds` defines internal model-selection folds for learners such as a Super Learner.
  Default 5.
- `repeats` repeats the outer split and reports the median of the draws. Default 1. A repeated fit
  that would build a simultaneous band refuses instead. The band needs a single draw's joint
  influence curves. Pass `simultaneous=False` to a repeated fit.
- `cross_fit=False` explicitly sets a one-fold, no-cross-fit analysis.

The two layers multiply: one nuisance fit at the defaults is `10 × 5` model fits per library
candidate, before an estimator variant multiplies it again. The examples in this documentation set
`n_folds=5, learner_folds=3` to stay quick to run, which is not a recommendation for an analysis.

Keep all rows from one declared cluster in the same fold. Cross-fitting reduces empirical-process
bias. It does not establish positivity, correct identification, or nuisance consistency.

Two fold schemes are refused by name, for two different reasons. A blocked temporal split needs a
row-level time ordering that no design role carries. A rolling-origin split nests its training
sets, so no single fold holds out each row, and `Folds` stores exactly one fold per row. The
rolling-origin refusal would survive a time index; it asks for a different storage contract.

## Reuse an outer split

Every point-treatment result records its realized outer folds as an immutable `split_plan`.
Supply that plan to compare methods on exactly the same validation rows.

```python
reused_method = TMLEMethod(
    models=method.models,
    cross_fitting=CrossFitting(
        n_folds=5,
        learner_folds=3,
        repeats=1,
        split_plan=result.split_plan,
    ),
    targeting=method.targeting,
    inference=method.inference,
    runtime=method.runtime,
)
reused_result = effect.estimate(method=reused_method)
assert reused_result.provenance.fold_fingerprint == result.provenance.fold_fingerprint
```

Set `n_folds` and `repeats` explicitly to the counts recorded by the plan. A mismatch raises
`MethodConfigurationError` instead of changing the plan.

Plan labels align by input row position, not by pandas or Polars index metadata. Reuse a plan only
with the same rows in the same order. With a cluster role, every row from one cluster must share a
fold within each repeat.

A supplied plan controls only the outer nuisance split. `learner_folds` still controls inner model
selection. Collaborative TMLE still constructs its selection folds from the repeat seed.

Validation runs before nuisance fitting. It checks row and repeat counts, fold labels, required
training arms, declared strata, and cluster integrity. The estimator refuses supplied plans for
longitudinal fits and targeted bootstrap inference.

Saving a result preserves `split_plan`. Provenance also records a fingerprint over every repeat's
assignments. Exact generated-versus-reused tests cover point, multi-arm, clustered, repeated,
pandas, Polars, serialization, and `n_jobs` behavior. Those tests establish computational identity,
not new statistical guarantees.

## Targeting and bounds

`g_bounds="auto"` chooses target-aware treatment-mechanism truncation. The bound changes the
finite-sample estimating procedure and should be reported with support diagnostics. The logistic
submodel bound, `submodel_alpha`, is separate from the confidence interval's `alpha`.

Unknown settings and settings an engine cannot use raise `MethodConfigurationError` before fitting
rather than being ignored. A longitudinal fit refuses a point-only control such as `n_bootstrap=`
instead of accepting it and discarding it.
