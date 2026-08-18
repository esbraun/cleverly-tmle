# Methods, learners, and cross-fitting

## Method presets

`"tmle"` is the ordinary default. `CollaborativeTMLEMethod` and `DRTMLEMethod` select compatible
estimator variants; they do not change the causal question.

```python
from cleverly import CollaborativeTMLEMethod, DRTMLEMethod, ModelSpec

parametric = ModelSpec(outcome_learner="glm", treatment_learner="glm")

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
    models=ModelSpec(outcome_learner="glm", treatment_learner="glm"),
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

Four named libraries are available, each a Super Learner over the listed candidates:

| name | candidates | use it for |
| --- | --- | --- |
| `"glm"` | mean, linear/logistic | examples, debugging, and deliberately parametric analyses |
| `"fast"` | mean, glm, spline GAM, gradient boosting | a flexible fit when runtime is the binding constraint |
| `"default"` | mean, glm, elastic net, spline GAM, gradient boosting | the unnamed default |
| `"rich"` | the `"default"` five plus a random forest | the widest library shipped |

`SuperLearner` may be constructed directly, and a scikit-learn-compatible object may be passed in
place of a name. The boosting candidate uses LightGBM when the `boost` extra is installed and
scikit-learn's histogram gradient boosting otherwise, so the library keeps its shape either way.

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
- `repeats` repeats the outer split and aggregates estimates. Default 1.
- `cross_fit=False` explicitly sets a one-fold, no-cross-fit analysis.

The two layers multiply: one nuisance fit at the defaults is `10 × 5` model fits per library
candidate, before an estimator variant multiplies it again. The examples in this documentation set
`n_folds=5, learner_folds=3` to stay quick to run, which is not a recommendation for an analysis.

Keep all rows from one declared cluster in the same fold. Cross-fitting reduces empirical-process
bias; it does not establish positivity, correct identification, or nuisance consistency.

## Targeting and bounds

`g_bounds="auto"` chooses target-aware treatment-mechanism truncation. The bound changes the
finite-sample estimating procedure and should be reported with support diagnostics. The logistic
submodel bound, `submodel_alpha`, is separate from the confidence interval's `alpha`.

Unknown settings and settings an engine cannot use raise `MethodConfigurationError` before fitting
rather than being ignored.
