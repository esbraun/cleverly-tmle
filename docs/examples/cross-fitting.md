# CV-TMLE: an honest interval across navigator teams

This is the same navigation question as the [first test of change](point-treatment-tmle.md), asked at
network scale. Two things change. The plan now has rich electronic-record features, so the analyst
wants a flexible learner. Patients also share navigator teams, so their outcomes are not independent.

Each change can invalidate the usual interval in a different way. Out-of-fold prediction and
cluster-aware splitting and inference address the two problems separately.

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

The two objections require different checks. Flexible learners can reduce nuisance-model bias, but
they do not guarantee a better point estimate. Cross-fitting and cluster-aware inference address
different parts of the interval calculation.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| a flexible learner for either nuisance | cross-fitting can avoid a Donsker restriction under its remaining conditions | one nuisance fit per outer fold; a Super Learner also fits its candidates on inner folds |
| you want the package default | cross-fitting is on by default, at ten outer folds | a Super Learner uses five additional inner folds unless configured otherwise |
| patients nested in navigator teams | teams stay intact in every split, and the variance is computed on team totals | fewer effective folds than the row count suggests |

The reason for the first row is not overfitting in the ordinary sense. A boosted model that predicts
held-out patients well can still break the interval.

The interval comes from an asymptotic argument with an empirical-process term in it. Classical
proofs control that term with complexity conditions such as a Donsker condition. Rich, adaptively
tuned learners need not satisfy those conditions, so in-sample nuisance evaluation does not provide
the required argument.

Cross-fitting avoids that empirical-process reliance under its remaining conditions. Every initial
nuisance prediction used for a patient comes from a model that never saw that patient.

**Cross-fitting does not buy the rest of efficiency.** Four conditions stand behind a valid
interval, and folds address one of them. The reference table
[names each condition and what supplies it](../technical-reference/cv-tmle.md#what-this-solves).
The study design and your learners supply the other three.

The product-rate condition is the one a variant can weaken. That variant is [DR-TMLE](dr-tmle.md).

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

This separation is the point of the two-step API. A reader comparing this page with the previous one
can see that only the method changed.

## Estimate

Cross-fitting is configured as a named group. The `n_folds` count splits the sample for out-of-fold
nuisance prediction. The `learner_folds` count reaches only the Super Learner that `cleverly` builds
when you pass no learner. An explicitly supplied `SuperLearner` keeps its own `n_folds`. The bare
boosted learners below have no inner learner folds.

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

At the documented sample size the two point estimates are close. The standard errors are not. In
this fixed draw, the in-sample fit reports a smaller standard error and excludes the population
value. The cross-fitted interval contains it.

Read this as a failure demonstration, not a universal ordering. Here the nuisance models reuse the
rows on which their influence values are evaluated. The smaller standard error is compatible with
overfit nuisance residuals, but one draw does not establish its cause or coverage rate.

Predictive accuracy alone does not establish the product-rate and influence-curve conditions. A
learner can predict well and still leave those inferential conditions unsupported.

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
| generated outer folds keep teams intact | the nuisance prediction cannot use outcomes from the patient's own team |
| with fewer teams than folds, the fold count is reduced and warns | the number of teams, not the number of patients, is what bounds the split |

`tests/unit/test_crossfit_leakage.py` names the evidence for the second row. Its
`test_a_clustered_fit_keeps_every_cluster_in_one_fold` case asserts that each declared cluster lands
in exactly one generated fold. A companion case rigs a noiseless law. A nearest-neighbour learner
there reproduces a held-out row exactly when a same-cluster row trained it, and never otherwise.
Those assertions are array equality and array inequality, so leakage is not a matter of degree.

## Reuse the same outer split

Return to the unclustered `cross_fitted` result above. It drew one outer split. A second fit can
run on exactly those validation rows, which is what you want when you compare two methods rather
than two splits.

Every point-treatment result exposes its realized outer folds as `split_plan`. Pass that plan back
through `CrossFitting`, with a declaration the plan can serve.

```python
import numpy as np

reused = effect.estimate(
    method=TMLEMethod(
        models=boosted,
        cross_fitting=CrossFitting(
            n_folds=5,
            repeats=1,
            split_plan=cross_fitted.split_plan,
        ),
        runtime=Runtime(random_state=34, n_jobs=1),
    )
)
assert reused.provenance.fold_fingerprint == cross_fitted.provenance.fold_fingerprint
assert reused["ate"].psi == cross_fitted["ate"].psi
assert np.array_equal(
    reused["ate"].influence_curve,
    cross_fitted["ate"].influence_curve,
)
```

The three assertions hold exactly. The reused fit repeats the folds, the point estimate, and the
influence curve of the first fit, digit for digit.

That is reproducible wiring for this fit. It adds no evidence about bias, coverage, or efficiency.

A plan labels rows by position, so it belongs to these rows in this order. The plan also records
which data it came from. A fit on other data is refused rather than given the wrong labels.
[Reusable outer split plans](../technical-reference/cv-tmle.md#reusable-outer-split-plans) states
the contract and every refusal, including the two refutation tests a supplied plan refuses.

## A second construction over the same folds

`cleverly` ships two estimators over these folds, and they are not the same estimator.

| construction | how it is selected | what it does |
| --- | --- | --- |
| stacked CV-TMLE | `targeting_scheme="pooled"`, the default | stacks all out-of-fold predictions, fits one targeting regression, evaluates the plug-in on the whole sample |
| fold-evaluated CV-TMLE | `CrossFitting(fold_evaluation=True)` | averages the fold plug-ins, with a cross-validated variance |

```python
fold_evaluated = effect.estimate(
    method=TMLEMethod(
        models=boosted,
        cross_fitting=CrossFitting(n_folds=5, fold_evaluation=True),
        runtime=Runtime(random_state=34, n_jobs=1),
    )
)
show("fold-evaluated", fold_evaluated, truth["ate"])
print("cross-validated targeting recorded:", fold_evaluated.cv_targeting is not None)
```

At equal fold sizes the two variance formulas nearly coincide, so the numbers sit close together
here. They are still different estimators with different registered evidence, and neither row
inherits the other's result.

Two refusals apply to fold evaluation. A nonlinear fold aggregate has a fold-varying gradient.
Risk ratios, odds ratios, and MSM coefficients are therefore refused instead of receiving an
interval whose reported curve has a nonzero score.

## How far to trust this

Start with the fit's own diagnostic report. This page does not run the sensitivity battery because
its question is cross-fitting behavior.

```python
diagnostics = cross_fitted.diagnostics.run_all()
print(diagnostics.summary())
print(diagnostics.report("nuisance_models").summary())
print(diagnostics.report("support").summary())
```

The nuisance report is the one that matters most here. It is computed out of fold, so it measures
the models on patients they did not see. An in-sample version of the same report would flatter a
gradient-boosted learner.

The support row reports 1.0% truncation and a 25.2% minimum effective sample size, and completes
without grading that ratio. Compare it with the in-sample fit above, which reports 91.6%. Nothing
about the true assignment mechanism changed between the two. Cross-fitting neither caused this
strain nor repaired it. It revealed it: an in-sample propensity predicts its own training rows, so
the weights there collapse toward one and flatter the effective sample size.

Read the 25.2% as a property of the estimated mechanism rather than of the population. A
well-specified propensity on this same law would strain less. The overview warns on the nuisance
models, which is the row this page is about, and the retained report gives the numbers behind both.
Its largest clever covariate is about 83.

Then the fold draw itself. One split is one draw. A nervous analyst can take the median of several.

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
median point and includes split displacement in the variance. The retained nuisance report gives
the spread of each parameter across draws, the reported standard error, and their ratio.

Both the spread and the standard error are on the inference scale, which is the log scale for a
ratio. The ratio therefore divides two like quantities. The report defines no threshold for it.

Two things follow from the median rule. The report is coordinatewise, so this call sets
`simultaneous=False`. A repeated fit reports no simultaneous band.

Read the retained ratio descriptively rather than as a pass threshold. The registered study covers
the three-draw reporting rule on its declared laws. It does not guarantee that another repeat count
will reduce split sensitivity in a new sample.

Three things constrain what this page establishes.

| layer | establishes | does not establish |
| --- | --- | --- |
| the two comparisons above | that in-sample nuisances and undeclared teams each gave a narrower interval on this draw | the coverage rate of any of these estimators |
| the diagnostic overview | which cached checks need attention and which costly operations did not run | the detailed severity of every retained report |
| the out-of-fold nuisance report | how the learners performed on unseen patients | that the learners converge fast enough for the remainder condition |
| the retained support report | truncation, effective sample size, and clever-covariate leverage | that cross-fitting repairs poor support. It does not |
| the retained split-spread rows | how much the estimate moved across the three declared fold draws | whether that amount is acceptable or another draw would help |
| the two construction studies | that stacked CV-TMLE matches R `tmle3` on identical realized folds, and that both constructions recover known truths | that folds fix a product-rate failure. They do not |
| the repeated cross-fitting study | that the three-draw median reduced fold-seed spread against the paired first-draw control, on one fixed binary-law sample | the same reduction on another sample, law, or repeat count. The study declares that limit |

The evidence rows are
[stacked point-treatment CV-TMLE](../technical-reference/method-evidence/stacked-point-treatment-cv-tmle.md),
[fold-evaluated point-treatment CV-TMLE](../technical-reference/method-evidence/fold-evaluated-point-treatment-cv-tmle.md),
and
[repeated point-treatment cross-fitted TMLE](../technical-reference/method-evidence/repeated-cross-fitting.md).
Each page states the construction and limits its evidence to the folds and reporting rule it tested.

## Where to go next

Cross-fitting controls one of the four conditions. If you doubt one nuisance and still want an
interval, the condition you are worried about is the product rate, and the variant for it is
[DR-TMLE](dr-tmle.md). If your worry is instead which baseline variables belong in the assignment
model, read [collaborative TMLE](collaborative-tmle.md).
