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
| a large approved baseline set | an assignment model selected by cross-validated loss on the targeted estimate, so an instrument can be left out | one nuisance fit per candidate along the selection path |
| near-positivity failure driven by strong assignment predictors | a less adaptive model, when the data says a less adaptive one estimates better | selection is data-dependent, and the reported interval does not account for it |
| the outcome regression is already good | the empty assignment model is a legitimate choice, and the selector will make it | that is not evidence the search discriminates |

Three variable roles matter, and only one belongs in the assignment model for free.

| role | predicts assignment? | affects the score? | belongs in the assignment model? |
| --- | --- | --- | --- |
| confounder | yes | yes | yes. Omitting it biases the estimate |
| instrument | yes, strongly | no | **no**. Including it inflates variance and removes no bias |
| outcome predictor | no | yes | it helps the outcome regression, not the assignment model |

An instrument is the dangerous one. It does not confound, so including it leaves the bias exactly
where it was. It does predict assignment, so including it pushes propensity scores toward zero and
one. The clever covariate divides by those propensities. A small denominator makes a large clever
covariate, and the variance of the estimate follows.

A model chosen by predictive loss will take the instrument, because an instrument is precisely what
predicts assignment best. Collaborative TMLE scores its candidates against the targeted estimate
instead.

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

The selection path is stored on the fit. It says which candidate models were considered, in order,
and which one the cross-validated loss chose.

```python
selection = collaborative.extra["ctmle"]
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

### Why this comparison is not yet evidence

Look again at the selected covariate set printed above. With a correctly specified outcome model,
the selector chose the **empty** assignment model.

That is the right choice rather than a defect. When the outcome regression already captures the
conditional mean, an empty assignment model minimizes the cross-validated loss, and the collaborative
criterion says so. It also means the comparison above proves nothing about whether the search can
demonstrate useful selection. A selector that always selects nothing would win it too.

The technical entry makes this point about its own validation, and the same caution belongs in an
applied reading.

### The comparison that does discriminate

Test the search where selecting nothing is wrong. Reduce the outcome model to a constant. The
assignment model now has to carry the whole adjustment, so omitting the confounder is no longer
harmless.

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

weak_selection = weak_collaborative.extra["ctmle"]
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

```python
print(collaborative.diagnostics.support().summary())
print(collaborative.diagnostics.score_equations().summary())
print(collaborative.validate().summary())
```

One limitation is structural and belongs in every report of a collaborative fit.

**The reported interval does not account for the selection.** The candidate model was chosen using
the data, and the influence curve is computed as if it had been fixed in advance. The technical
entry records this among its declared limits, and no diagnostic on the fit can repair it.

| layer | establishes | does not establish |
| --- | --- | --- |
| the support report | how far the propensity reached into the tails, before and after selection | that the selected model is the right one |
| the score-equation report | the pooled targeting continued from the selected candidate and converged | anything about the selection |
| the registered studies | the selectors recover known truths and match R `ctmle` where a comparator exists | calibrated inference while selection is load-bearing. No cell asks for it |

The evidence rows are
[selector-based point-treatment C-TMLE](../technical-reference/method-evidence/selector-based-point-treatment-c-tmle.md)
and
[outcome-adaptive point-treatment C-TMLE](../technical-reference/method-evidence/outcome-adaptive-point-treatment-c-tmle.md).
Both declare their limits in their own rows, including that parity is binary, two-arm, and not
cross-fitted.

## Where to go next

Collaborative TMLE addresses *selection*. If your worry is the *inference* instead, because you
expect one nuisance to be inconsistent however you choose it, read [DR-TMLE](dr-tmle.md). If the
adjustment set is small and you would include all of it, the plain
[point-treatment TMLE](point-treatment-tmle.md) is the right entry.

Two compositions are refused rather than approximated. Collaborative TMLE has no longitudinal
derivation, so it cannot evaluate navigation across two decision times. It is also wrong by construction
on an incremental fit, because each candidate assignment model would define a different estimand, and
the search would then select between estimands rather than between estimators.
