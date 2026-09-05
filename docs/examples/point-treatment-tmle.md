# Point-treatment TMLE: did transition navigation improve the experience score?

This is the first test of change in the network's patient-experience program. It estimates one
average treatment effect from observational data, and it demonstrates the property the method is
named for. A TMLE stays consistent when the outcome regression is wrong **or** the treatment
mechanism is wrong.

Read [Point-treatment TMLE](../technical-reference/point-treatment-tmle.md) for the parameter, the
influence curve, and the algorithm.

## The applied question

A regional health plan offers adults a **standard transition-navigation protocol** before they
leave a hospital for home. Assignment means an offer of a bedside plan and two scheduled contacts
within 30 days. It does not mean that the patient completed every contact.

Nobody randomized the rollout. Discharge teams used a recorded risk process, and patients who got
an offer were not a random sample. The plan measured discharge risk, prior utilization, medication
burden, and age before assignment.

The program wants one number. How much would the average 30-day transition score change if every
eligible discharge received the navigation offer, compared with usual discharge support?

That question names an estimand before any model exists. It is the average treatment effect. It
contrasts two counterfactual means over the same population. It is not the coefficient on a
navigation indicator. It is not a comparison of offered patients with patients who got usual care.

The outcome is a standardized 30-day patient-reported transition score. This first synthetic law
treats discharges as independent. The [cross-fitting tutorial](cross-fitting.md) adds shared
navigator teams.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| observational data, confounders measured | a doubly-robust estimate. It is consistent if the outcome regression **or** the treatment mechanism is consistent | you must name the estimand first |
| the nuisance functions are not linear | flexible learners fit both nuisances, and the estimate stays a plug-in | a valid interval needs a product rate on the two nuisances |
| you want an interval you can report | the interval comes from the targeted influence curve | positivity must hold, and the support report is where you check it |

Two familiar alternatives fail here, for different reasons.

A regression of the transition score on navigation and the four covariates reports a coefficient. That
coefficient equals the average treatment effect only if the outcome model is correct and the effect
is constant. Neither holds here. The number changes when you add an interaction term, and nothing in
the output tells you which version answers the question.

Inverse-probability weighting avoids the outcome model. It replaces it with a mean whose variance is
set by the patient with the smallest propensity score. One unlucky row moves the answer.

TMLE uses both models. It starts from the outcome regression. It then moves that regression along a
submodel chosen so the estimate solves the efficient score equation. That step is what makes one
wrong model survivable.

## The data

The generator is `make_nonlinear_ate`. Its own documentation calls it the process that exercises
double robustness. A linear model is wrong for the treatment mechanism and wrong for the outcome
regression, and the effect varies with the covariates.

```python
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=3_000, seed=21)
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
print(frame.head())
print("population ATE:", truth["ate"])
```

The renaming is cosmetic. It keeps the prose about patients rather than about `W1`.

| feature of the law | what it means in this program |
| --- | --- |
| the four baseline covariates drive assignment and the outcome | higher-risk patients receive different offers and report different outcomes. All four are confounders in the synthetic law |
| both nuisance functions are nonlinear | a GLM is misspecified for each one, which is the condition this page exploits |
| the effect varies with the covariates | `ate`, `att`, and `atc` differ, so the estimand must be named rather than inferred |

The `truth` dictionary exists because this is a generator. A real program has no such column. Every
comparison against `truth` below is a teaching device.

## Design and identification

The design says which column plays which role. The estimand says which contrast you want. The two
are separate on purpose, so changing the estimator later cannot change the question.

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
for assumption in effect.identification.assumptions:
    print("-", assumption)
```

`identify` returns the assumptions that carry the causal reading. Four apply here.

| assumption | what it means for this program | can the data check it? |
| --- | --- | --- |
| consistency | an offer always means the declared bedside plan and two scheduled contacts | no |
| no interference | one patient's assignment does not change another patient's offer or outcome | no |
| no unmeasured confounding | the recorded baseline variables block every common cause of assignment and the score | no |
| positivity | each baseline profile has some chance of an offer and of usual support | partly, through the support report |

Only positivity leaves a direct trace in the observed treatment support. The program supports
consistency with one protocol and version log. It reserves navigator capacity and records
contamination to support no interference.

Exchangeability needs a causal argument. For example, an unrecorded discharge-team judgement that
affects both assignment and recovery would violate it. No estimator on this page repairs that
failure. Restrict eligibility or redesign assignment when the argument is not credible.

The synthetic law needs only four covariates. A real protocol should also evaluate pre-assignment
site, navigator-team, calendar, language-access, and discharge-destination causes. Add them when the
causal review places them on a common-cause path.

## Estimate

The configuration is written out in full rather than left to defaults. Each group answers one
question, so a reader sees every choice that was made.

```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod

flexible = TMLEMethod(
    models=ModelSpec(
        outcome_learner=HistGradientBoostingRegressor(random_state=21),
        treatment_learner=HistGradientBoostingClassifier(random_state=21),
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
    inference=Inference(alpha=0.05),
    runtime=Runtime(random_state=21, n_jobs=1),
)
result = effect.estimate(method=flexible)

estimate = result["ate"]
print(result.summary())
print("estimate:", estimate.psi)
print("standard error:", estimate.std_error)
print("95% CI:", estimate.ci)
print("population ATE:", truth["ate"])
```

Both nuisances use a gradient-boosted learner, because the law is nonlinear. Cross-fitting is on,
because a flexible learner needs it. The [cross-fitting tutorial](cross-fitting.md) shows what
happens when you leave it off.

The interval is built from the targeted influence curve. It is not the outcome model's own standard
error. It already accounts for the fact that both nuisances were estimated.

## Which population is the number about?

The average treatment effect answers a question about every patient. A spread decision asks
something narrower, and the difference is not cosmetic.

| estimand | the question it answers | who asks it |
| --- | --- | --- |
| ATT | what did patients who received an offer gain from assignment? | the teams reviewing the pilot |
| ATE | what would the eligible population gain if everyone received an offer? | the program sponsor |
| ATC | what would patients who received usual support gain from an offer? | whoever is deciding on spread |

These are three parameters, not three estimates of one. A second law makes the gap visible, because
its effect modification is aligned with the propensity: patients most likely to receive an offer are
that benefit most.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATC, ATT
from cleverly.datasets import make_heterogeneous

spread_frame, spread_truth = make_heterogeneous(n=3_000, seed=23)
spread_frame = spread_frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "discharge_risk",
        "W2": "medication_burden",
    }
)
spread_study = CausalStudy(
    spread_frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("discharge_risk", "medication_burden"),
    ),
)
simple = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
    runtime=Runtime(random_state=23, n_jobs=1),
)
for estimand, key in (
    (ATT(reference=0), "att"),
    (ATE(reference=0), "ate"),
    (ATC(reference=0), "atc"),
):
    point = spread_study.identify(estimand).estimate(method=simple)[key]
    low, high = point.ci
    print(
        f"{key}: {point.psi:6.3f}  CI=({low:.3f}, {high:.3f})  population {spread_truth[key]:.3f}"
    )
```

At the documented sample size the three sit in a strict order, and their intervals do not overlap.
The ATT is the largest and the ATC is the smallest.

Read that as a warning about spread. The pilot's own result is the ATT, and it is the number a
successful pilot reports. Patients who received usual support would get the ATC, which here is a small
fraction of it. A program that budgets the network rollout against the pilot's number will
overpromise.

## The failure mode: one wrong model is survivable

Double robustness is a claim about *or*, not about *and*. The clearest way to see it is to break one
nuisance at a time and watch the estimate.

A gradient-boosted learner is approximately right for this law. A linear learner is wrong for it, by
construction. Three more fits therefore span the cases that matter.

```python
def fit(outcome_learner, treatment_learner, label):
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=outcome_learner, treatment_learner=treatment_learner),
        cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
        runtime=Runtime(random_state=21, n_jobs=1),
    )
    point = effect.estimate(method=method)["ate"]
    low, high = point.ci
    covered = low <= truth["ate"] <= high
    print(f"{label:24s} psi={point.psi:6.3f}  CI=({low:.3f}, {high:.3f})  covers={covered}")


fit(
    HistGradientBoostingRegressor(random_state=21),
    LogisticRegression(max_iter=1000),
    "outcome right, g wrong",
)
fit(
    LinearRegression(),
    HistGradientBoostingClassifier(random_state=21),
    "outcome wrong, g right",
)
fit(LinearRegression(), LogisticRegression(max_iter=1000), "both wrong")
print("population ATE:", truth["ate"])
```

At the documented sample size the pattern is the one the theory predicts. The first two fits sit
near the population value, and their intervals cover it. The third sits well below it, and its
interval excludes it.

That is the whole argument for the method. A single misspecified nuisance is repaired by the other
one during the targeting step. Two misspecified nuisances leave nothing to repair with.

Three cautions belong with the demonstration.

| caution | why |
| --- | --- |
| "wrong" here means wrong in a known way | you do not know which of your models is wrong in a real analysis. Double robustness buys two chances, not a diagnosis |
| the intervals differ in width, not only in position | the standard error under a wrong treatment mechanism is not the efficient one |
| one sample is one draw | coverage is a repeated-sampling property. The next section measures it |

## How far to trust this

Start with one assessment battery. It presents validation, diagnostics, and sensitivity together.
It also identifies rows that need attention and operations that did not run.

```python
assessment = result.assess()
print(assessment.summary())
print("needs attention:", tuple(item.name for item in assessment.attention))

support = assessment.report("support")
nuisance = assessment.report("nuisance_models")
scores = assessment.report("score_equations")
print(support.summary())
print(nuisance.summary())
print(scores.summary())
```

The overview routes attention and follow-up work. The retained reports provide the tables needed to
interpret each row. A `completed` sensitivity row means the calculation ran. It is not a pass.

The support report is where positivity becomes visible. It gives the propensity quantiles, the
effective sample size per arm, and the share of rows the truncation touched.

**Read its verdict on this fit.** The gradient-boosted model separates offered patients from the
usual-support group well. The price is a set of extreme propensity scores. The effective sample
size in the navigation arm falls to about a fifth of those patients, and the largest clever covariate
is in the tens. The verdict reports that fifth and does not grade it. Grading it is your job, and on
this contrast a fifth is thin.

Two lessons follow, and both are general.

The first is that a better-predicting assignment model is not automatically a better assignment model
for this purpose. Prediction accuracy and estimand-relevant behaviour are different criteria, and
[collaborative TMLE](collaborative-tmle.md) is the entry that chooses between models on the second
one.

The second is what the support row does not say. It reports 0.9% truncation and a 20.1% minimum
effective sample size, and its status is `completed` rather than `passed`. That is deliberate. The
report grades the truncated fraction, because a clipped row contributes extrapolation instead of
data. It does not grade the effective-sample-size ratio, because no published result fixes a cutoff
on a Kish ratio, and a threshold invented here would read as a positivity clearance this package
cannot give.

So the 20.1% is yours to judge. It says the estimate leans on an effective fifth of the rows in its
narrowest arm. That is a lot of strain for a small contrast, and it is the reason the truncation
curve below is worth reading. A `passed` on this row would have told you nothing you should act on.

The overview names the follow-up itself.

```python
retargeted = result.assess(include_retargets=True)
curve = retargeted.report("truncation_curve")
print(curve)
```

The curve retargets the estimate at a range of truncation bounds. It does not refit the nuisance
models. A point estimate that moves across the range depends on patients constrained by the bound.

Here it does not move much, and the interval narrows as the bound tightens. Read that as good news
of a limited kind. The positivity strain is showing up in the variance rather than in the location
of the estimate. It would still be wrong to report the number without the report.

The nuisance report adds one more finding. The boosted assignment model is poorly calibrated, at a
calibration slope well below one, while the outcome regression fits well. That combination is the
one this page has been describing from the other side. The estimate survives it because the outcome
regression is good, and double robustness is what makes that survival possible.

The sensitivity section comes next. It addresses assumptions that the observed data cannot test.

Unmeasured confounding is invisible to every diagnostic, because the data holds no record of it.
Sensitivity analysis asks a different question. How strong would an unmeasured confounder have to be
to explain the result away? An unrecorded discharge-team judgement is the concrete threat to hold
against the answer.

Then refutation, which perturbs the analysis and checks that it responds as it should.

```python
refitted = result.assess(include_refits=True)
print(refitted.report("refute").summary())
```

A placebo exposure should give roughly zero. A random common cause should change nothing. A subset
refit should scatter around the original estimate. A test that fails is evidence of a problem. A
test that passes is not evidence of correctness.

This fit sets `random_state=21`, so the refuter inherits that seed and repeats the same report.

Last comes a repeated-sampling check. This is the only instrument on the page that measures the
estimator rather than one fit.

```python
from cleverly.datasets import linear_dgp
from cleverly.estimators import TMLE
from cleverly.validation import CoverageStudy

check = CoverageStudy(
    dgp=linear_dgp(),
    estimator=lambda: TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        estimands=("ate",),
        simultaneous=False,
    ),
    n=400,
    n_replicates=40,
    seed=7,
)
print(check.run().summary())
```

This runs on `linear_dgp`, where a GLM is correctly specified for both nuisances. It reports a bias
near zero and a coverage near 0.95. That is the baseline. With correct nuisances the estimator
recovers the truth, and its interval means what it says.

Forty replications is a demonstration rather than evidence. The registered study behind this method
runs 1,600 replications on two laws. It is published test by test in the
[implementation validation grid](../technical-reference/method-evidence/validation-grid.md) and
in
[canonical point-treatment TMLE](../technical-reference/method-evidence/canonical-point-treatment-tmle.md).

### What each layer can and cannot establish

| layer | establishes | does not establish |
| --- | --- | --- |
| assessment overview | which stored checks need attention, and which operations did not run | the detail needed to interpret each retained report |
| retained diagnostics | that targeting converged, and how far positivity was strained | that the nuisance models are right |
| sensitivity analysis | how large an unmeasured confounder would need to be | that no such confounder exists |
| refutation | the estimate responds correctly to perturbations with a known answer | that the estimate is correct |
| the registered study | the implementation recovers known truths and behaves as its theory predicts | that your identification assumptions hold on your data |

Nothing in this list validates the causal reading. That rests on consistency, no interference, and
no unmeasured confounding. All three are arguments about the program rather than about the fit.

## Keeping the result

A fit is an artifact. It carries the nuisance models, the influence curves, and the provenance
stamp, so the assessment above replays without refitting. A program that reports quarterly needs
that.

```python
for capability in result.diagnostics.capabilities:
    print(capability.operation, capability.cost, capability.execution)

result.save("transition-navigation-ate.joblib")
```

```python
from cleverly import load

restored = load("transition-navigation-ate.joblib")
print(restored.replayability)
print(restored.assess().summary())
```

`replayability` says which operations the restored artifact can still perform. The saved assessment
cache retains reports that ran before the save. A new nuisance refit still needs the analysis data.
Load only joblib files you trust, and keep the dependency versions compatible.

## Where to go next

| the next question | read |
| --- | --- |
| flexible learners, and patients nested in navigator teams | [CV-TMLE and cross-fitting](cross-fitting.md) |
| which baseline variables belong in the assignment model | [collaborative TMLE](collaborative-tmle.md) |
| an interval when the assignment model is known to be crude | [DR-TMLE](dr-tmle.md) |
| a rule, a dose change, or an odds tilt instead of "offer to everyone" | [intervention axes](interventions.md) |
| most patients never returned the survey | [survey non-response](survey-nonresponse.md) |
| navigation at more than one decision time | [longitudinal TMLE](longitudinal-tmle.md) |
| leaving the plan is the outcome, and one cause is administrative | [retention and competing risks](longitudinal-survival.md) |
| three navigation cadences summarized as a trend | [MSM projections](msm-projections.md) |
