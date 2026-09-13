# DR-TMLE: an interval when the recorded assignment rule is hard to model

An ordinary TMLE stays consistent when one nuisance is consistent. Its interval needs both
nuisances to converge fast enough. This page shows what DR-TMLE changes when the assignment model is
the doubtful one. The [DR-TMLE reference](../technical-reference/dr-tmle/index.md) holds the
theorem, the refusals, and the release claim.

## The applied question

The navigation evaluation is repeated with 2,000 discharges. The program logged every common cause
of assignment and outcome, and a flexible model can learn the outcome. The assignment rule has
thresholds and interactions, so a main-effects logistic model of it is misspecified. An unrecorded
common cause would be a different failure, and DR-TMLE does not repair it.

## Why this method

| estimator | when the assignment model converges to the wrong limit |
| --- | --- |
| ordinary TMLE | stays consistent. If the outcome fit converges too slowly, the remainder can dominate the root-n scale. The usual influence-curve interval then need not attain nominal coverage |
| DR-TMLE | solves two extra score equations built from reduced-dimension regressions. Under Theorem 1 of Benkeser et al. (2017), it stays asymptotically linear, given [rate conditions](../technical-reference/dr-tmle/theorem.md#the-remainder-terms-and-the-rate-conditions) on the outcome fit and the reduced regressions |

When both nuisances are consistent, the corrections converge to zero. DR-TMLE then has no asymptotic
advantage and adds finite-sample cost.

## The data

The law is `make_nonlinear_ate` again, with standardized covariates (mean 0, SD 1). Both of its
nuisance functions are nonlinear.

```python
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2_000, seed=55)
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

DR-TMLE targets the same parameter under the same assumptions. This page keeps the
[shared study design](index.md#the-shared-study-design) and every common cause in the adjustment
set.

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

for method in effect.available_methods():
    print(method.name, method.available)
```

The catalog lists `drtmle` as available for this ATE. An `ATT`, `ATC`, MSM, or intervention-axis
effect lists it as unavailable, and selecting it raises before any nuisance is fitted. Other
refusals raise at fit time.
[Refused by name](../technical-reference/dr-tmle/supported-estimands.md#refused-by-name) lists them.

## Estimate

The primary nuisances are the analyst's: a flexible outcome regression and a crude assignment
model. The reduced regressions each have one input, so a spline can fit them fast. Each reduction
below is a Super Learner over a linear and a spline candidate. The
[`drtmle` vignette](https://github.com/benkeser/drtmle/blob/538a3a264c1ca984b6d88978ca7f96165f43152c/vignettes/using_drtmle.Rmd)
uses the same pairing, `SL.glm` and `SL.gam`.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from cleverly import CrossFitting, DRTMLEMethod, ModelSpec, Runtime, SuperLearner, TMLEMethod

models = ModelSpec(
    outcome_learner=HistGradientBoostingRegressor(random_state=55),
    treatment_learner=LogisticRegression(max_iter=1000),
)
folds = CrossFitting(n_folds=3)
runtime = Runtime(random_state=55, n_jobs=1)


def spline(final):
    return make_pipeline(SplineTransformer(n_knots=5, knots="quantile"), final)


reduced_outcome = SuperLearner(
    library=[("linear", LinearRegression()), ("spline", spline(LinearRegression()))],
    task="regression",
    n_folds=3,
)
reduced_treatment = SuperLearner(
    library=[
        ("logistic", LogisticRegression()),
        ("spline", spline(LogisticRegression(max_iter=1000))),
    ],
    task="classification",
    n_folds=3,
)
drtmle = DRTMLEMethod(
    models=models,
    cross_fitting=folds,
    runtime=runtime,
    reduced_outcome_learner=reduced_outcome,
    reduced_treatment_learner=reduced_treatment,
)
guarded = effect.estimate(method=drtmle)
print(guarded.summary())
```

| keyword | what it controls |
| --- | --- |
| `reduced_outcome_learner` | the conditional means $Q_r$ and $g_{r,2}$ |
| `reduced_treatment_learner` | the probability $g_{r,1}$ |
| `guard=` | the extra equations. Both are on by default. `guard=("g",)` guards only against the assignment model and reports $D = D^{*} - D^{*}_{Q}$ |

## What one fit can show

An empty guard solves no extra equation, so it must reproduce the ordinary TMLE exactly.

```python
from dataclasses import replace

ordinary = effect.estimate(method=TMLEMethod(models=models, cross_fitting=folds, runtime=runtime))
empty_guard = effect.estimate(method=replace(drtmle, guard=()))
print("identical:", ordinary["ate"].psi == empty_guard["ate"].psi)


def show(label, result):
    point = result["ate"]
    low, high = point.ci
    print(f"{label:14s} psi={point.psi:.3f}  se={point.std_error:.4f}  CI=({low:.3f}, {high:.3f})")


show("ordinary TMLE", ordinary)
show("DR-TMLE", guarded)
print("population ATE:", truth["ate"])
```

The equality fixes what the variant is: the same estimator plus extra equations, not a different
target. The guarded estimate moves because the extra fluctuations change the targeted fits. On
this draw it moves by less than one standard error, and the standard errors are similar. That
resemblance says nothing about either interval's coverage.

Next, run the combined assessment and open the reports that bear on the nuisances.

```python
assessment = guarded.assess()
print(assessment.summary())
print(assessment.report("corrections").summary())
print(assessment.report("nuisance_models").summary())

reduced = guarded.extra["drtmle"].diagnostics
for family, fits in reduced.items():
    print(family, "best candidate per arm and fold:", [fit.best for fit in fits])
```

| report | what it shows on this draw |
| --- | --- |
| the assessment summary | score equations and corrections pass. The omitted-confounding rows address exchangeability, which DR-TMLE does not relax |
| the corrections report | each extra equation per arm, its solved score, and the statement that no truncation was active |
| the nuisance report | held-out fit that looks reasonable for both primary models, including the misspecified assignment model |
| the reduced-regression diagnostics | the spline candidate has the lowest cross-validated risk in every $g_{r,1}$ fit, so the data favor a nonlinear reduction there |

## The failure mode: solved scores do not certify the nuisances

Every score above is approximately zero. The held-out nuisance report also looks reasonable for an
assignment model the analyst knows has the wrong form. Neither result shows that any nuisance
converges at the rate the theorem needs.

Score convergence is a property of the targeting step. Held-out risk compares candidates, but it
cannot measure distance from the true function. The reduced-regression table is the one place the
fit shows a choice that the theorem's conditions depend on.
[Solved scores do not establish nuisance consistency](../technical-reference/dr-tmle/diagnostics.md#solved-scores-do-not-establish-nuisance-consistency)
gives the exact-law test behind this rule.

## How far to trust this

| layer | establishes | does not establish |
| --- | --- | --- |
| `guard=()` equality | the variant reduces exactly to the ordinary estimator | anything about the guarded fit |
| the score and corrections reports | the targeting solved all three equations, with no active truncation | the rate conditions behind the interval |
| the nuisance and reduced-regression reports | which candidate fit best under cross-validated risk | that any fitted function is consistent |
| the omitted-confounding rows | how much hidden confounding would move the estimate | that no hidden confounder exists |

DR-TMLE ships under **conditional validity**. No registered study covers this continuous law with
flexible learners, and the [DR-TMLE evidence](../technical-reference/dr-tmle/validation-programme.md)
shows where the interval fell short of nominal coverage.

## Where to go next

If your question is which baseline variables belong in the assignment model, read
[collaborative TMLE](collaborative-tmle.md). The two methods do not compose. A reduced regression
conditions on the fitted assignment mechanism as a covariate, and C-TMLE's mechanism is deliberately
not an estimate of the true one. DR-TMLE raises that refusal at fit time.

When outcomes are also missing, double robustness takes a different shape. Read
[survey non-response](survey-nonresponse.md).
