# Collaborative TMLE: which baseline variables belong in the assignment model?

The network holds many baseline variables. Only some help the model of who receives navigation.
This test of change shows what happens when a variable that predicts assignment very well
goes into that model, and how a collaborative selector keeps it out.

Read [Collaborative TMLE](../technical-reference/collaborative-tmle.md) for the candidate paths, the
selection loss, and the fold structure.

## The applied question

The program office wants to evaluate transition navigation. Before fitting, clinical and operations
staff classify three baseline variables with a causal diagram and a protocol review. They exclude
all descendants of assignment and all colliders.

The three approved variables can all enter the outcome regression. A model of assignment that uses
all three predicts assignment
best, and predictive accuracy feels like the right criterion. It is not.

The selector chooses a nuisance model inside this approved baseline set. It does not discover a
causal adjustment set from the data. The [cross-fitting tutorial](cross-fitting.md) adds shared
navigator teams.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| a large approved baseline set | an assignment model selected by cross-validated loss on the targeted outcome regression, so an instrument can be left out | one nuisance fit per candidate along the selection path |
| near-positivity failure driven by strong assignment predictors | a less adaptive model when its targeted loss is better | selection is data-dependent, and the reported interval does not account for it |
| the outcome regression is already good | the empty assignment model is a legitimate candidate | selecting it is not evidence that the search discriminates |

Variable roles guide the causal review. They do not mechanically determine the denominator that a
collaborative selector chooses.

| role | predicts assignment? | affects the score? | how to handle it |
| --- | --- | --- | --- |
| confounder | yes | yes | include it in the study design. A collaborative denominator can omit it only when the outcome regression handles the residual bias |
| instrument | yes, strongly | no | do not include it for confounding control. It can reduce precision |
| outcome predictor | no | yes | use it in the outcome regression. It need not enter the assignment model |

An instrument does not close a common-cause path. A strong instrument can push fitted propensity
scores toward zero and one without removing confounding. The resulting clever covariate can become
more variable and reduce precision.

A model chosen by treatment-prediction loss tends to take a strong instrument. Collaborative TMLE
scores its candidates against the targeted outcome-regression loss instead.

## The data

The generator is `make_instrument`. Its three covariates have cleanly separated roles, which is what
makes the demonstration readable.

```python
from cleverly.datasets import make_instrument

frame, truth = make_instrument(n=2_000, seed=44)
frame = frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "baseline_readiness",
        "W2": "queue_lottery_position",
        "W3": "social_support",
    }
)
print(frame.head())
print("population ATE:", truth["ate"])
```

| column | role in the law | what it is in the program |
| --- | --- | --- |
| `baseline_readiness` | confounder | lower readiness increases the chance of an offer and predicts the transition score |
| `queue_lottery_position` | instrument | an encounter-ID hash sets queue position, which strongly predicts an offer and has no path to the outcome except through the offer |
| `social_support` | outcome predictor | it moves the transition score and does not move assignment |

The queue lottery is an instrument only because the program fixes the hash before assignment,
prevents staff overrides, and verifies that queue position changes no other service. If any of those
conditions fails, the variable loses that role. The data cannot establish the exclusion restriction.

This variance argument assumes exchangeability already holds. If an unmeasured common cause
remains, adding a strong instrument can amplify residual bias. C-TMLE does not turn the queue
variable into a design-based instrument estimator.

The effect is constant in this law, so the population `ate`, `att`, and `atc` all equal one. The
analyst does not know any of this.

## Design and identification

The design holds all three approved baseline columns. The causal review established that
`baseline_readiness` is sufficient for the common-cause path in this synthetic scenario. C-TMLE
then selects terms for the assignment nuisance; it does not revise that identification decision.

```python
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("baseline_readiness", "queue_lottery_position", "social_support"),
    ),
)
effect = study.identify(ATE(reference=0))
print(effect.summary())
```

Method availability is checked before any model is fitted. Collaborative TMLE has no longitudinal
derivation, and it covers only the arm-axis targets, so it is worth asking first.

```python
for method in effect.available_methods():
    print(method.name, method.available)
```

## Estimate

Both fits below share their learners and their cross-fitting, so the only difference between them is
how the assignment model was chosen.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CollaborativeTMLEMethod, CrossFitting, ModelSpec, Runtime, TMLEMethod

models = ModelSpec(
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
folds = CrossFitting(n_folds=3, learner_folds=2)
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

The nuisance report retains the selection path. It says which candidate models were considered,
in order, and which one the cross-validated loss chose.

```python
selection = collaborative.diagnostics.nuisance_models().selection
print("candidate path:", selection.path)
print("selected covariates:", selection.path[selection.selected])
```

## The failure mode: an instrument in the assignment model

First look at what including everything does to the propensity scores.

```python
plain = effect.estimate(method=TMLEMethod(models=models, cross_fitting=folds, runtime=runtime))
print(plain.diagnostics.support().summary())
```

The support report shows the propensity distribution reaching far into both tails. That is the
queue lottery at work. Early queue positions have propensities near one, and late positions have
propensities near zero. No confounding was removed in exchange.

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
search can distinguish a confounder from an instrument, because a selector that always chose the
empty model would give the same result.

### The comparison that does discriminate

Use a deliberate stress control where selecting nothing is wrong. Reduce the outcome model to a
constant, so the assignment model must carry the adjustment. This is a test of the selector, not a
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

weak_selection = weak_collaborative.ctmle_selection
print("selected covariates:", weak_selection.path[weak_selection.selected])
print("population ATE:", truth["ate"])
```

At the documented sample size the selector now includes `baseline_readiness`, the confounder, and
leaves `queue_lottery_position`, the instrument, out. The standard error falls by a large factor
against the plain fit that used all three.

In this known synthetic law, the search retains the variable needed by the deliberately reduced
outcome model and drops the pure assignment predictor. A real analysis still needs the causal review
above. The selection result does not prove that either variable has its declared causal role.

## How far to trust this

Start with the combined assessment. Sensitivity analysis cannot determine whether the selector
chose a useful assignment model, so inspect the selection and support reports next.

```python
assessment = collaborative.assess()
print(assessment.summary())
print(assessment.report("support").summary())
nuisance = assessment.report("nuisance_models")
print("treatment role:", nuisance.treatment_role)
print(nuisance.summary())
print(nuisance.selection.summary())
```

The role prints as `collaborative_working_model`. The AUC and calibration values describe the
selected working denominator, not assignment given the complete adjustment set. Read them with the
selection path and support report. [Nuisance model quality](../technical-reference/validation-methods.md#nuisance-model-quality)
defines the retained findings.

One limitation is structural and belongs in every report of a collaborative fit.

**The reported interval does not account for the selection.** The data chose the candidate model.
The influence curve is then computed as if that model had been fixed in advance. The technical
entry records this limit, and no diagnostic on the fit can repair it.

| layer | establishes | does not establish |
| --- | --- | --- |
| the diagnostic overview | which cached checks need attention and which costly operations did not run | selection uncertainty or the causal role of a candidate variable |
| the support report | how far the propensity reached into the tails, before and after selection | that the selected model is the right one |
| the nuisance report | selected-model metrics, model role, and the retained selection | whether low AUC means limited confounding after collaborative selection |
| the retained selection path | which candidates the search considered and selected | calibrated inference for the selected candidate |

The [collaborative TMLE technical entry](../technical-reference/collaborative-tmle.md) links the
registered studies and states their limits.

## Where to go next

Collaborative TMLE addresses *selection*. If your worry is the *inference* instead, because you
expect one nuisance to be inconsistent however you choose it, read [DR-TMLE](dr-tmle.md). If the
adjustment set is small and you would include all of it, the plain
[point-treatment TMLE](point-treatment-tmle.md) is the right entry.

The current library refuses two compositions. It has no registered longitudinal C-TMLE
implementation. It also has no validated selector path for an incremental target, whose definition
depends on the treatment mechanism. These are current support boundaries, not claims that no method
can be derived.
