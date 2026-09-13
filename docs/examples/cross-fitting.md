# CV-TMLE: an honest interval across navigator teams

Read [CV-TMLE and cross-fitting](../technical-reference/cv-tmle.md) for the constructions and
their fold arithmetic.

## The applied question

The [point-treatment tutorial](point-treatment-tmle.md) fit boosted learners. A reviewer of the
network report asks why the sample was split, given that the learners predict held-out patients
well. The reviewer also asks whether patients from two hundred navigator teams are independent
observations.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| a flexible learner for either nuisance | an interval without a Donsker restriction, under the [remaining conditions](../technical-reference/cv-tmle.md#what-this-solves) | one nuisance fit per outer fold, ten by default |
| patients nested in navigator teams | teams stay intact in every split, and the variance uses team totals | fewer effective folds than the row count suggests |

Classical interval proofs control an empirical-process term with a Donsker condition. Rich, tuned
learners need not satisfy it, however well they predict. Cross-fitting predicts each patient from a
model that never saw that patient. It addresses no other condition.

## The data

The law is the point-treatment tutorial's law, drawn with a new seed. Its nuisance functions are
nonlinear. Covariates are standardized (mean 0, SD 1), and the score is in synthetic units.

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

This page keeps the [shared study design](index.md#the-shared-study-design) and changes only the
estimator. Cross-fitting is an estimation choice, and it must not touch the question.

Clustering is not an interference adjustment. If patients compete for slots, potential outcomes can
depend on other assignments. A cluster-robust standard error cannot repair that causal-design
failure.

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

## Estimate

Cross-fitting is configured as a named group. The `n_folds` count splits the sample for out-of-fold
nuisance prediction. The bare boosted learners below have no inner learner folds, so
`learner_folds` does not apply.

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
        cross_fitting=CrossFitting(n_folds=5),
        runtime=Runtime(random_state=34, n_jobs=1),
    )
)
print(cross_fitted.summary())
print("population ATE:", truth["ate"])
```

The summary names the construction it used. It reports stacked CV-TMLE, the number of outer folds,
and the targeting scheme.

## The failure mode: an interval that is too narrow

Now fit the same learners with the splitting turned off. `CrossFitting(enabled=False)` is not a
tuning knob. It selects ordinary TMLE rather than CV-TMLE.

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

On this draw the two point estimates are close. The in-sample standard error is less than a third
of the cross-fitted one.

One draw does not establish a coverage rate. The registered
[stacked CV-TMLE study](../technical-reference/method-evidence/stacked-point-treatment-cv-tmle.md#theory-properties)
fit this law 400 times at n = 500, with a fully grown regression tree and ten folds. The in-sample
control covered 65.0% of the time and the cross-fitted fit 89.5%. Both fall below 95%, so the study
supports relative recovery, not nominal coverage.

## The second failure mode: patients are not independent

Two hundred navigator teams, fifteen patients each. Teams differ in ways the recorded covariates do
not capture, such as supervisor practice and local follow-up quality. A shared difference that
changes how much navigation helps makes the influence values of one team move together.

The `cluster=` role declares the teams. A second law carries a shared team effect interacted with
the exposure. The two failure modes use separate laws, so the team fit uses linear learners that
are correct for its two covariates.

In this law the team effect changes the benefit but not who receives an offer. A team factor that
also drives offers is an unmeasured confounder, and no cluster declaration repairs it.

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
    cross_fitting=CrossFitting(n_folds=5),
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

On this draw the two point estimates are nearly identical. Declaring the navigator team makes the
standard error about 1.7 times larger.

Neither interval is wrong about the estimand. The unclustered one is wrong about how much
information three thousand patients from two hundred teams carry. Cluster-robust inference sums the
influence values inside a team, then takes the variance across teams. With singleton teams it
equals the ordinary formula.

The large-sample argument counts teams, not patients. The interval uses a normal reference with no
small-sample cluster correction, so with few teams it can still be too narrow.

Folds change too.

| what happens | why it matters |
| --- | --- |
| each team lands entirely in one outer fold | a patient's nuisance prediction cannot use outcomes from the patient's own team |
| with fewer teams than folds, the fit reduces the fold count and warns | the number of teams, not the number of patients, bounds the split |

The registered
[clustered CV-TMLE study](../technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md#theory-properties)
compares cluster-robust and independent-row variances on the same influence curves.

## Reuse the same outer split

Return to the unclustered `cross_fitted` result above. A second fit can run on exactly its
validation rows. Use this when you compare two methods rather than two splits.

Every point-treatment result exposes its realized outer folds as `split_plan`. Pass that plan back
through `CrossFitting`. The fits below use a different runtime seed, so only the plan can hold the
folds fixed.

```python
import numpy as np

new_seed = Runtime(random_state=35, n_jobs=1)
redrawn = effect.estimate(
    method=TMLEMethod(models=boosted, cross_fitting=CrossFitting(n_folds=5), runtime=new_seed)
)
reused = effect.estimate(
    method=TMLEMethod(
        models=boosted,
        cross_fitting=CrossFitting(n_folds=5, repeats=1, split_plan=cross_fitted.split_plan),
        runtime=new_seed,
    )
)
assert redrawn.provenance.fold_fingerprint != cross_fitted.provenance.fold_fingerprint
assert reused.provenance.fold_fingerprint == cross_fitted.provenance.fold_fingerprint
assert reused["ate"].psi == cross_fitted["ate"].psi
assert np.array_equal(reused["ate"].influence_curve, cross_fitted["ate"].influence_curve)
show("new seed, new folds", redrawn, truth["ate"])
```

The new seed alone draws new folds. With the plan, the fit repeats the folds, the point estimate,
and the influence curve of the first fit exactly. That is reproducible wiring. It adds no evidence
about bias, coverage, or efficiency.

A plan labels rows by position, and it records the data it came from. A fit on other data is
refused. [Reusable outer split plans](../technical-reference/cv-tmle.md#reusable-outer-split-plans)
states the contract.

## Other constructions over the same folds

`cleverly` ships three constructions over these folds. They are different estimators, and each has
its own registered evidence in the construction column.

| construction | how it is selected | what it does |
| --- | --- | --- |
| [stacked CV-TMLE](../technical-reference/method-evidence/stacked-point-treatment-cv-tmle.md) | `targeting_scheme="pooled"`, the default | stacks all out-of-fold predictions, fits one targeting regression, evaluates the plug-in on the whole sample |
| [fold-evaluated CV-TMLE](../technical-reference/method-evidence/fold-evaluated-point-treatment-cv-tmle.md) | `CrossFitting(fold_evaluation=True)` | keeps the pooled update, averages the fold plug-ins, and uses a cross-validated variance |
| [fold-targeted CV-TMLE](../technical-reference/method-evidence/fold-targeted-point-treatment-cv-tmle.md) | `CrossFitting(targeting_scheme="fold")` | fits one targeting regression inside each validation fold |

Pass `split_plan=cross_fitted.split_plan` to compare them on identical folds. The
[variations table](../technical-reference/cv-tmle.md#variations) lists what each one refuses.

## How far to trust this

Start with the fit's own diagnostic overview. It uses `diagnostics.run_all()` rather than
`assess()` because sensitivity analysis does not answer a question about cross-fitting.

```python
diagnostics = cross_fitted.diagnostics.run_all()
print(diagnostics.summary())
print(diagnostics.report("nuisance_models").summary())
print(diagnostics.report("support").summary())
print("in-sample support:", in_sample.diagnostics.run_all()["support"].detail)
```

The overview warns that the out-of-fold propensity model is poorly calibrated. That report measures
the models on patients they did not see.

The support row reports about 1% of units truncated and a minimum effective-sample-size ratio of
about 25%. It does not grade that ratio. The in-sample fit reports a ratio above 85%, although the
true assignment mechanism is the same.

Cross-fitting did not cause this strain, and it does not repair it. An in-sample propensity predicts
its own training rows, so the observed-arm weights collapse toward one. The 25% describes the
estimated mechanism, not the population.

Then the fold draw itself. One split is one draw. `repeats=` reports the median over several.

```python
repeated = effect.estimate(
    method=TMLEMethod(
        models=boosted,
        cross_fitting=CrossFitting(n_folds=5, repeats=3),
        inference=Inference(simultaneous=False),
        runtime=Runtime(random_state=34, n_jobs=1),
    )
)
show("three fold draws", repeated, truth["ate"])
repeated_diagnostics = repeated.diagnostics.run_all()
repeated_nuisance = repeated_diagnostics.report("nuisance_models")
print(repeated_nuisance.repeat_spread_frame())
```

`repeats=` is the same estimator over several draws rather than a new estimator. It reports the
median point and includes split displacement in the variance. The median is coordinatewise, so a
repeated fit refuses a simultaneous band and this call sets `simultaneous=False`.

The retained nuisance report gives the spread across draws, the reported standard error, and their
ratio. Read the ratio descriptively. The report defines no threshold for it.

| layer | establishes | does not establish |
| --- | --- | --- |
| the two comparisons above | that in-sample nuisances and undeclared teams each gave a narrower interval on this draw | the coverage rate of any of these estimators |
| the out-of-fold nuisance report | how the learners performed on unseen patients | that the learners converge fast enough for the remainder condition |
| the retained support report | truncation, effective sample size, and clever-covariate leverage | that cross-fitting repairs poor support. It does not |
| the retained split-spread rows | how much the estimate moved across the three declared fold draws | whether that amount is acceptable or another draw would help |

No registered row covers five boosted folds exactly. The
[technical entry](../technical-reference/cv-tmle.md#validation-issues-special-to-this-method)
links the evidence for each construction.

## Where to go next

Cross-fitting addresses only the Donsker condition. If you doubt one nuisance and still want an
interval, the condition you are worried about is the
[product rate](../technical-reference/cv-tmle.md#what-this-solves). The variant for it is
[DR-TMLE](dr-tmle.md). If your worry is instead which baseline variables belong in the assignment
model, read [collaborative TMLE](collaborative-tmle.md).
