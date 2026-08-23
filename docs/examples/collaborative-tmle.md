# Collaborative TMLE: which ward characteristics belong in the adoption model?

The network holds a lot of data about its wards. Only some of it belongs in the model of who adopted
rounding. This test of change shows what happens when a variable that predicts adoption very well
goes into that model, and how a collaborative selector keeps it out.

Read [Collaborative TMLE](../technical-reference/collaborative-tmle.md) for the candidate paths, the
selection loss, and the fold structure.

## The applied question

The program office wants to evaluate hourly rounding using everything it has. Three ward-level
variables reach the analyst's desk, one from the quality system, one from the training office, and
one from finance. Nobody has drawn a causal diagram.

The temptation is to adjust for all three. A model of adoption that uses all three predicts adoption
best, and predictive accuracy feels like the right criterion. It is not.

This page analyses one hospital's patients, so patients are treated as independent. The
[cross-fitting tutorial](cross-fitting.md) adds the ward layer.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| a large adjustment set with unknown structure | an adoption model selected by cross-validated loss on the targeted estimate, so an instrument is left out | one nuisance fit per candidate along the selection path |
| near-positivity failure driven by strong adoption predictors | a less adaptive model, when the data says a less adaptive one estimates better | selection is data-dependent, and the reported interval does not account for it |
| the outcome regression is already good | the empty adoption model is a legitimate choice, and the selector will make it | that is not evidence the search discriminates |

Three variable roles matter, and only one of them belongs in the adoption model for free.

| role | predicts adoption? | affects the score? | belongs in the adoption model? |
| --- | --- | --- | --- |
| confounder | yes | yes | yes. Omitting it biases the estimate |
| instrument | yes, strongly | no | **no**. Including it inflates variance and removes no bias |
| outcome predictor | no | yes | it helps the outcome regression, not the adoption model |

An instrument is the dangerous one. It does not confound, so including it leaves the bias exactly
where it was. It does predict adoption, so including it pushes propensity scores toward zero and
one. The clever covariate divides by those propensities. A small denominator makes a large clever
covariate, and the variance of the estimate follows.

A model chosen by predictive loss will take the instrument, because an instrument is precisely what
predicts adoption best. Collaborative TMLE scores its candidates against the targeted estimate
instead.

## The data

The generator is `make_instrument`. Its three covariates have cleanly separated roles, which is what
makes the demonstration readable.

```python
from cleverly.datasets import make_instrument

frame, truth = make_instrument(n=2_000, seed=44)
frame = frame.rename(
    columns={
        "Y": "experience_score",
        "A": "hourly_rounding",
        "W1": "baseline_experience",
        "W2": "training_cohort_slot",
        "W3": "case_mix",
    }
)
print(frame.head())
print("population ATE:", truth["ate"])
```

| column | role in the law | what it is on the ward |
| --- | --- | --- |
| `baseline_experience` | confounder | wards scoring poorly last year were pushed hardest to adopt, and last year's score also predicts this year's |
| `training_cohort_slot` | instrument | which wave the ward's manager was scheduled into. Scheduling was driven by room availability, and it strongly predicts adoption |
| `case_mix` | outcome predictor | it moves the experience score and it does not move adoption |

The training cohort is the instrument, and it is a realistic one. Program offices schedule training
by convenience, then use attendance as a variable because it is recorded and it predicts adoption
almost perfectly. That is exactly the wrong reason to include it.

The effect is constant in this law, so the population `ate`, `att`, and `atc` all equal one. The
analyst does not know any of this.

## Design and identification

The adjustment set holds all three columns. That is the honest starting point, because the analyst
cannot tell them apart. The design is a statement about what was measured, not a claim about roles.

```python
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="experience_score",
        treatment="hourly_rounding",
        adjustment=("baseline_experience", "training_cohort_slot", "case_mix"),
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
how the adoption model was chosen.

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

## The failure mode: an instrument in the adoption model

First look at what including everything does to the propensity scores.

```python
plain = effect.estimate(method=TMLEMethod(models=models, cross_fitting=folds, runtime=runtime))
print(plain.diagnostics.support().summary())
```

The support report shows the propensity distribution reaching far into both tails. That is the
training cohort at work. Wards in an early slot have propensities near one, and wards in a late slot
have propensities near zero. No confounding was removed in exchange.

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
the selector chose the **empty** adoption model.

That is the right choice rather than a defect. When the outcome regression already captures the
conditional mean, an empty adoption model minimises the cross-validated loss, and the collaborative
criterion says so. It also means the comparison above proves nothing about whether the search can
tell a confounder from an instrument. A selector that always selects nothing would win it too.

The technical entry makes this point about its own validation, and the same caution belongs in an
applied reading.

### The comparison that does discriminate

Test the search where selecting nothing is wrong. Reduce the outcome model to a constant. The
adoption model now has to carry the whole adjustment, so omitting the confounder is no longer
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

At the documented sample size the selector now includes `baseline_experience`, the confounder, and
leaves `training_cohort_slot`, the instrument, out. The standard error falls by a large factor
against the plain fit that used all three.

That is the claim the method makes. The search distinguishes a variable that removes bias from a
variable that only predicts adoption, because it scores candidates against the targeted estimate
rather than against adoption prediction.

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
[selector-based point-treatment C-TMLE](../technical-reference/method-evidence.md#selector-based-point-treatment-c-tmle)
and
[outcome-adaptive point-treatment C-TMLE](../technical-reference/method-evidence.md#outcome-adaptive-point-treatment-c-tmle).
Both declare their limits in their own rows, including that parity is binary, two-arm, and not
cross-fitted.

## Where to go next

Collaborative TMLE addresses *selection*. If your worry is the *inference* instead, because you
expect one nuisance to be inconsistent however you choose it, read [DR-TMLE](dr-tmle.md). If the
adjustment set is small and you would include all of it, the plain
[point-treatment TMLE](point-treatment-tmle.md) is the right entry.

Two compositions are refused rather than approximated. Collaborative TMLE has no longitudinal
derivation, so it cannot evaluate rounding across two admissions. It is also wrong by construction
on an incremental fit, because each candidate adoption model would define a different estimand, and
the search would then select between estimands rather than between estimators.
