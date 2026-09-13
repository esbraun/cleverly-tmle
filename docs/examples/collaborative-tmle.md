# Collaborative TMLE: which baseline variables belong in the assignment model?

Read [Collaborative TMLE](../technical-reference/collaborative-tmle.md) for the candidate paths, the
selection loss, and the fold structure.

## The applied question

Clinical and operations staff approve three baseline variables. They confirm that each is measured
before assignment and that none is a collider. The review cannot certify each variable's causal
role. An assignment model that uses all three predicts assignment best. Predictive accuracy is the wrong criterion for that model.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| an approved baseline set | an assignment model selected by cross-validated loss on the targeted outcome regression, so an instrument can be left out | one nuisance fit per candidate along the selection path |
| the outcome regression is already good | the empty assignment model is a legitimate candidate | selecting it is not evidence that the search discriminates |

The selector chooses a nuisance model inside the approved set. It does not discover a causal
adjustment set.

## The data

The generator is `make_instrument`. Its covariates are standardized (mean 0, SD 1), and the score
is in synthetic units.

```python
from cleverly.datasets import make_instrument

frame, truth = make_instrument(n=2_000, seed=44)
frame = frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "baseline_readiness",
        "W2": "queue_lottery_draw",
        "W3": "social_support",
    }
)
print(frame.head())
print("population ATE:", truth["ate"])
```

| column | role in the law | what it is in the program |
| --- | --- | --- |
| `baseline_readiness` | confounder | navigators prioritize patients ready to engage, so higher readiness raises the chance of an offer and the transition score |
| `queue_lottery_draw` | instrument | an encounter-ID hash sets a queue draw. A higher draw strongly raises the chance of an offer and has no other path to the score |
| `social_support` | outcome predictor | it moves the transition score and does not move assignment |

The queue draw is an instrument only under three conditions. The program fixes the hash before
assignment, prevents staff overrides, and verifies that the draw changes no other service. The
data cannot establish this exclusion restriction.

An instrument in the assignment model pushes propensity scores toward zero and one without removing
confounding, so precision falls. That argument assumes exchangeability. If an unmeasured common
cause remains, a strong instrument can also amplify residual bias. C-TMLE does not turn the queue
draw into a design-based instrument estimator.

The effect is constant in this law, so the population `ate`, `att`, and `atc` all equal one.

## Design and identification

The design holds all three approved baseline columns. In this synthetic scenario,
`baseline_readiness` alone closes the common-cause path. C-TMLE selects terms for the assignment
nuisance. It does not revise that identification decision.

```python
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("baseline_readiness", "queue_lottery_draw", "social_support"),
    ),
)
effect = study.identify(ATE(reference=0))
print(effect.summary())
```

Check method availability before fitting. Collaborative TMLE covers the point-treatment `ate`,
`ey`, `ey1`, `ey0`, `rr`, and `or` targets only. Each refusal carries its reason.

```python
for method in effect.available_methods():
    print(method.name, method.available, method.reason or "")
```

## Estimate

The plain fit in the next section uses the same learners and cross-fitting. Only the choice of
assignment model differs.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CollaborativeTMLEMethod, CrossFitting, ModelSpec, Runtime, TMLEMethod

models = ModelSpec(
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
folds = CrossFitting(n_folds=3)
runtime = Runtime(random_state=44, n_jobs=1)

collaborative = effect.estimate(
    method=CollaborativeTMLEMethod(
        models=models,
        cross_fitting=folds,
        runtime=runtime,
        strategy="greedy",
        selection_folds=3,
        selection_inner_folds=2,
    )
)
print(collaborative.summary())
print("population ATE:", truth["ate"])
```

The nuisance report retains the selection path. It lists the candidate models in order and marks
the one the cross-validated loss chose.

```python
selection = collaborative.diagnostics.nuisance_models().selection
print("candidate path:", selection.path)
print("selected covariates:", selection.selected_covariates)
```

## The failure mode: an instrument in the assignment model

First look at what including everything does to the propensity scores.

```python
plain = effect.estimate(method=TMLEMethod(models=models, cross_fitting=folds, runtime=runtime))
print(plain.diagnostics.support().summary())
```

More than 5% of the fitted propensity scores lie below 0.1, and more than 5% lie above 0.9. That is
the queue lottery at work. High draws have propensities near one, and low draws have propensities
near zero. No confounding was removed in exchange.

Now compare the two estimators.

```python
def show(label, result):
    point = result["ate"]
    low, high = point.ci
    print(
        f"{label:22s} psi={point.psi:6.3f}  se={point.std_error:6.4f}  CI=({low:.3f}, {high:.3f})"
    )


show("plain TMLE", plain)
show("collaborative TMLE", collaborative)
print("population ATE:", truth["ate"])
```

### Why this comparison needs a control

With a correctly specified outcome model, the selector chooses the **empty** assignment model on
this law. That choice can minimize the targeted cross-validated loss. It does not show that the
search can tell a confounder from an instrument. A selector that always chose the empty model would
give the same result.

### The comparison that does discriminate

Use a deliberate stress control where selecting nothing is wrong. Reduce the outcome model to a
constant, so the assignment model must carry the adjustment. This tests the selector. It is not a
recommended production outcome model.

```python
from sklearn.dummy import DummyRegressor

weak_models = ModelSpec(
    outcome_learner=DummyRegressor(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
weak_plain = effect.estimate(
    method=TMLEMethod(models=weak_models, cross_fitting=folds, runtime=runtime)
)
weak_collaborative = effect.estimate(
    method=CollaborativeTMLEMethod(
        models=weak_models,
        cross_fitting=folds,
        runtime=runtime,
        strategy="greedy",
        selection_folds=3,
        selection_inner_folds=2,
    )
)
show("constant Q, plain", weak_plain)
show("constant Q, C-TMLE", weak_collaborative)

weak_selection = weak_collaborative.diagnostics.nuisance_models().selection
print("selected covariates:", weak_selection.selected_covariates)
print("population ATE:", truth["ate"])
```

On this draw the selector now includes `baseline_readiness`, the confounder, and leaves
`queue_lottery_draw`, the instrument, out. The standard error is less than half that of the plain
fit that used all three.

In this known synthetic law, the search keeps the variable that the reduced outcome model needs. It
drops the pure assignment predictor. The selection result does not prove that either variable has
its declared causal role.

## How far to trust this

Start with the combined assessment. Sensitivity analysis cannot tell whether the selector chose a
useful assignment model, so inspect the selection and support reports next.

```python
assessment = collaborative.assess()
print(assessment.summary())
print(assessment.report("support").summary())
nuisance = assessment.report("nuisance_models")
print("treatment role:", nuisance.treatment_role)
print(nuisance.summary())
print("selected covariates:", nuisance.selection.selected_covariates)
print(nuisance.selection.summary())
```

The role prints as `collaborative_working_model`. On this draw the selected model is intercept
only, so its AUC is about 0.5. Its calibration slope is not meaningful for a constant model. These
metrics describe the working denominator, not assignment given the complete adjustment set.
[Nuisance model quality](../technical-reference/validation-methods.md#nuisance-model-quality)
defines the retained findings.

The support report describes the selected denominator only. Compare it with the plain fit's report
above to see the tails that selection removed. The selection footer states the loss that chose the
candidate. It assigns no causal role to an omitted variable.

Two limits belong in every report of a collaborative fit.

**Post-selection coverage has not been established.** The data chose the candidate model. The
reported Wald interval treats that model as if it had been fixed in advance.

**The influence curve can omit a first-order term.** This happens when the selected model is not
consistent for the true assignment mechanism. The empty model here is such a case, so its reported
standard error can be too small. The
[technical entry](../technical-reference/collaborative-tmle.md#validation-issues-special-to-this-method)
records both limits and links the registered studies. No diagnostic on the fit repairs them.

| layer | establishes | does not establish |
| --- | --- | --- |
| the combined assessment | which cached checks need attention and which costly operations did not run | selection uncertainty or the causal role of a candidate variable |
| the support report | how far the selected denominator reaches into the tails | that the selected model is the right one |
| the nuisance report | selected-model metrics, model role, and the retained selection | whether low AUC means limited confounding after collaborative selection |
| the retained selection path | which candidates the search considered and selected | calibrated inference for the selected candidate |

## Where to go next

Collaborative TMLE addresses *selection*. If your worry is the *inference* instead, because you
expect one nuisance to be inconsistent however you choose it, read [DR-TMLE](dr-tmle.md). If the
adjustment set is small and you would include all of it, the plain
[point-treatment TMLE](point-treatment-tmle.md) is the right entry.

The library refuses longitudinal and incremental-target C-TMLE. The technical entry gives the
reasons.
