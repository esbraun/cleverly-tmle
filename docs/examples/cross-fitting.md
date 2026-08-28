# CV-TMLE: an honest interval across navigator teams

This is the same navigation question as the [first test of change](point-treatment-tmle.md), asked at
network scale. Two things change. The plan now has rich electronic-record features, so the analyst
wants a flexible learner. Patients also share navigator teams, so their outcomes are not independent.

Each change breaks the interval in its own way, and folds are the answer to both.

Read [CV-TMLE and cross-fitting](../technical-reference/cv-tmle.md) for the two constructions and
their fold arithmetic.

## The applied question

The program is preparing a network-wide report on transition navigation. An analyst objects to the
independent-row analysis on two grounds.

The first objection is about the models. The nuisance functions are not linear, and a GLM cannot fit
them. Use a gradient-boosted model for both, the analyst says, and the estimate will be better.

The second objection is about the rows. Patients assigned through the same navigator team share
staff, workflow, and local practice. Treating three thousand patients from two hundred teams as three
thousand independent observations claims more information than the network holds.

The analyst is right about the point estimate and wrong about the interval, twice over.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| a flexible learner for either nuisance | the empirical-process term is controlled without a Donsker condition on the nuisance estimators | one nuisance fit per fold, times the learner library |
| you want the package default | cross-fitting is on by default, at ten outer folds and five learner folds | the two fold layers multiply |
| patients nested in navigator teams | teams stay intact in every split, and the variance is computed on team totals | fewer effective folds than the row count suggests |

The reason for the first row is not overfitting in the ordinary sense. A boosted model that predicts
held-out patients well can still break the interval.

The interval comes from an asymptotic argument with an empirical-process term in it. That term is
negligible when the nuisance estimators live in a class that is not too rich. A gradient-boosted
model does not satisfy that condition. The term stops vanishing, the variance estimate loses a
piece, and the interval shrinks below its nominal width.

Cross-fitting removes the assumption instead of hoping it holds. Every nuisance prediction used for
a patient comes from a model that never saw that patient.

**Folds do not buy the rest of efficiency.** Four conditions stand behind a valid interval, and
folds address one.

| condition | what supplies it |
| --- | --- |
| the empirical-process term is negligible | cross-fitting |
| positivity bounds the clever covariate | the truncation, and your design |
| the estimated influence curve converges | your learners |
| the second-order remainder vanishes fast enough, by a product rate on both nuisances | your learners, and nothing the fluctuation can do |

The last condition is the one a variant can weaken. That variant is [DR-TMLE](dr-tmle.md).

## The data

The law is the one from the first test of change. Its nuisance functions are nonlinear, which is
exactly the situation that invites a flexible learner.

```python
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=3_000, seed=34)
frame = frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "discharge_risk",
        "W2": "prior_utilization",
        "W3": "medication_burden",
        "W4": "age",
    }
)
print("population ATE:", truth["ate"])
```

## Design and identification

The design and the estimand do not change. Cross-fitting is an estimation choice, and it must not
touch the question.

Clustering is not an interference adjustment. The program still needs reserved per-patient capacity
and access controls so one offer does not change another patient's protocol. If patients compete for
slots, potential outcomes can depend on other assignments. A cluster-robust standard error cannot
repair that causal-design failure.

```python
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("discharge_risk", "prior_utilization", "medication_burden", "age"),
    ),
)
effect = study.identify(ATE(reference=0))
print(effect.summary())
```

This separation is the point of the two-step API. A reader comparing this page with the previous one
can see that only the method changed.

## Estimate

Cross-fitting is configured as a named group. `n_folds` splits the sample. `learner_folds` tunes a
model inside one outer training set. The two layers are separate, and neither borrows the other's
count.

```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod

boosted = ModelSpec(
    outcome_learner=HistGradientBoostingRegressor(random_state=34),
    treatment_learner=HistGradientBoostingClassifier(random_state=34),
)
cross_fitted = effect.estimate(
    method=TMLEMethod(
        models=boosted,
        cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
        runtime=Runtime(random_state=34, n_jobs=1),
    )
)
print(cross_fitted.summary())
print("population ATE:", truth["ate"])
```

The summary names the construction it used. It reports stacked CV-TMLE, the number of outer folds,
and the targeting scheme.

## The failure mode: an interval that is too narrow

Now fit the same learners with the splitting turned off. `cross_fit=False` is not a tuning knob. It
is a different estimator, and it is the ordinary TMLE of the first test of change.

```python
in_sample = effect.estimate(
    method=TMLEMethod(
        models=boosted,
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=34, n_jobs=1),
    )
)


def show(label, result, target):
    point = result["ate"]
    low, high = point.ci
    print(
        f"{label:22s} psi={point.psi:6.3f}  se={point.std_error:6.4f}  "
        f"CI=({low:.3f}, {high:.3f})  covers={low <= target <= high}"
    )


show("no cross-fitting", in_sample, truth["ate"])
show("cross-fitted", cross_fitted, truth["ate"])
print("population ATE:", truth["ate"])
```

At the documented sample size the two point estimates are close. The standard errors are not. The
in-sample fit reports a much smaller standard error, and its interval excludes the population value.
The cross-fitted interval covers it.

Read the direction of the error carefully. The in-sample fit is not merely imprecise. It is
confident and wrong. Its nuisance models were partly fitted to the patients they are being evaluated
on, so the residual variation those patients contribute is too small. The variance estimate inherits
the optimism.

This is the failure that no amount of held-out predictive accuracy will warn you about. The boosted
model may score well on a validation set and still produce this interval, because the problem is in
the influence curve rather than in the prediction.

## The second failure mode: patients are not independent

Two hundred navigator teams, fifteen patients each. Teams differ in ways the recorded covariates do
not capture: supervisor practice, workflow, and local follow-up quality. A shared difference that
changes how much navigation helps makes the influence values of patients on that team
move together.

The `cluster=` role says so. A second law carries a genuine shared team effect, and it enters
interacted with the exposure, which is what correlates the influence curves rather than merely
shifting them.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_clustered

team_frame, team_truth = make_clustered(n=3_000, seed=34, cluster_size=15)
team_frame = team_frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "discharge_risk",
        "W2": "medication_burden",
        "cluster": "navigator_team",
    }
)
simple = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
    runtime=Runtime(random_state=34, n_jobs=1),
)


def team_fit(cluster, label):
    design = PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("discharge_risk", "medication_burden"),
        cluster=cluster,
    )
    fitted = (
        CausalStudy(team_frame, design=design).identify(ATE(reference=0)).estimate(method=simple)
    )
    show(label, fitted, team_truth["ate"])
    return fitted


ignoring = team_fit(None, "patients as units")
clustered = team_fit("navigator_team", "teams as units")
print("navigator teams:", clustered.data.n_clusters)
print("population ATE:", team_truth["ate"])
```

At the documented sample size the two point estimates are nearly identical and the standard errors
are not. Declaring the navigator team widens the interval substantially.

Neither interval is wrong about the estimand. The unclustered one is wrong about how much
information three thousand patients from two hundred teams carry. Cluster-robust inference sums the
influence values inside a team first, then takes the variance across teams. With singleton teams it
collapses back to the ordinary formula, so nothing is lost by declaring a structure that turns out
not to matter.

Folds change too. Each team must land in one fold, and the number of teams bounds the fold count.

| what happens | why it matters |
| --- | --- |
| a team lands entirely in one fold | otherwise a patient's nuisance prediction comes from a model trained on their own team, which is leakage through the team effect |
| an externally supplied fold assignment that splits a team is refused | buying more folds that way shrinks the standard error in exactly the direction the cluster role was declared to prevent |
| with fewer teams than folds, the fold count is reduced and warns | the number of teams, not the number of patients, is what bounds the split |

## A second construction over the same folds

`cleverly` ships two estimators over these folds, and they are not the same estimator.

| construction | how it is selected | what it does |
| --- | --- | --- |
| stacked CV-TMLE | `targeting_scheme="pooled"`, the default | stacks all out-of-fold predictions, fits one targeting regression, evaluates the plug-in on the whole sample |
| fold-evaluated CV-TMLE | `cv_evaluation=True` | averages the fold plug-ins, with a cross-validated variance |

```python
fold_evaluated = effect.estimate(
    method=TMLEMethod(
        models=boosted,
        cross_fitting=CrossFitting(n_folds=5, learner_folds=3, fold_evaluation=True),
        runtime=Runtime(random_state=34, n_jobs=1),
    )
)
show("fold-evaluated", fold_evaluated, truth["ate"])
print("cross-validated targeting recorded:", fold_evaluated.cv_targeting is not None)
```

At equal fold sizes the two variance formulas nearly coincide, so the numbers sit close together
here. They are still different estimators with different registered evidence, and neither row
inherits the other's result.

Two refusals apply to fold evaluation. A nonlinear fold aggregate has a fold-varying gradient, so
risk ratios, odds ratios, and MSM coefficients are refused rather than given an interval whose
reported curve has a nonzero score.

## How far to trust this

Start with the fit's own diagnostics.

```python
print(cross_fitted.diagnostics.nuisance_models().summary())
print(cross_fitted.diagnostics.score_equations().summary())
print(cross_fitted.validate().summary())
```

The nuisance report is the one that matters most here. It is computed out of fold, so it measures
the models on patients they did not see. An in-sample version of the same report would flatter a
gradient-boosted learner.

Then the fold draw itself. One split is one draw, and a nervous analyst can average over several.

```python
repeated = effect.estimate(
    method=TMLEMethod(
        models=boosted,
        cross_fitting=CrossFitting(n_folds=5, learner_folds=3, repeats=3),
        inference=Inference(simultaneous=False),
        runtime=Runtime(random_state=34, n_jobs=1),
    )
)
show("three fold draws", repeated, truth["ate"])
print(repeated.repeat_spread())
```

`repeats=` is the same estimator over several draws rather than a new estimator. It reports the
median point and includes split displacement in the variance. `repeat_spread()` reports how much
the answer moved between draws. A large spread says the fold draw is doing work that the sample
size should be doing.

Three things constrain what this page establishes.

| layer | establishes | does not establish |
| --- | --- | --- |
| the two comparisons above | that in-sample nuisances and undeclared teams each gave a narrower interval on this draw | the coverage rate of any of these estimators |
| the out-of-fold nuisance report | how the learners performed on unseen patients | that the learners converge fast enough for the remainder condition |
| the registered studies | stacked CV-TMLE matches R `tmle3` on identical realized folds, and both constructions recover known truths | that folds fix a product-rate failure. They do not |

Leakage is checked separately, and without a tolerance. A test rigs a law where a nearest-neighbour
learner reproduces a held-out row exactly if and only if a same-cluster row was in its training set.
The assertions are array equality and array inequality, so leakage is not a matter of degree.

The evidence rows are
[stacked point-treatment CV-TMLE](../technical-reference/method-evidence/stacked-point-treatment-cv-tmle.md)
and
[fold-evaluated point-treatment CV-TMLE](../technical-reference/method-evidence/fold-evaluated-point-treatment-cv-tmle.md).
The second has no canonical comparator, and its study says so in its own cell rather than borrowing
a surrogate.

## Where to go next

Cross-fitting controls one of the four conditions. If you doubt one nuisance and still want an
interval, the condition you are worried about is the product rate, and the variant for it is
[DR-TMLE](dr-tmle.md). If your worry is instead which baseline variables belong in the assignment
model, read [collaborative TMLE](collaborative-tmle.md).
