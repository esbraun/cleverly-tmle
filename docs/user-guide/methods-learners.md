# Methods, learners, and cross-fitting

## Method presets

`"tmle"` is the ordinary default. `CollaborativeTMLEMethod` and `DRTMLEMethod` select compatible
estimator variants; they do not change the causal question.

```python
from cleverly import CollaborativeTMLEMethod, DRTMLEMethod

collaborative = effect.estimate(
    method=CollaborativeTMLEMethod(strategy="greedy"),
    random_state=3,
)
doubly_robust = effect.estimate(method=DRTMLEMethod(), random_state=3)
```

Call `effect.available_methods()` before fitting when method availability matters. An unavailable
method comes with a capability reason.

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

Named libraries supply conservative defaults. `"glm"` is useful for examples, debugging, and
parametric analyses; flexible presets and `SuperLearner` are available when the scientific design
requires them. A scikit-learn-compatible object may also be passed directly.

Outcome regressions need predictions on the outcome's mean scale. Binary treatment and censoring
mechanisms need probability predictions. Continuous treatment policies use a conditional density
model. `cleverly` validates learner task compatibility and sample-weight support.

## Two fold layers

- `n_folds` defines the outer cross-fitting split used to keep each observation out of the nuisance
  fit that predicts it.
- `learner_folds` defines internal model-selection folds for learners such as a Super Learner.
- `repeats` repeats the outer split and aggregates estimates.
- `cross_fit=False` explicitly sets a one-fold, no-cross-fit analysis.

Keep all rows from one declared cluster in the same fold. Cross-fitting reduces empirical-process
bias; it does not establish positivity, correct identification, or nuisance consistency.

## Targeting and bounds

`g_bounds="auto"` chooses target-aware treatment-mechanism truncation. The bound changes the
finite-sample estimating procedure and should be reported with support diagnostics. The logistic
submodel bound, `submodel_alpha`, is separate from the confidence interval's `alpha`.

Unknown settings and settings an engine cannot use raise `MethodConfigurationError` before fitting
rather than being ignored.
