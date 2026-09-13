# Point-treatment TMLE: did transition navigation improve the experience score?

This page estimates one average treatment effect from observational data with TMLE.
[Point-treatment TMLE](../technical-reference/point-treatment-tmle.md) gives the parameter, the
influence curve, and the algorithm.

## The applied question

A regional health plan offers adults a **standard transition-navigation protocol** when a discharge
home is ordered. The offer is a bedside plan and two scheduled contacts within 30 days. Nobody
randomized it, and discharge teams used a recorded risk process. How much would the mean 30-day
transition score change if every eligible discharge received the offer, rather than usual support?
That question is the average treatment effect, not a regression coefficient.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| observational data, confounders measured | double robust point consistency: either consistent nuisance model can supply it, under positivity and regularity conditions | you must name the estimand first |
| the nuisance functions are not linear | flexible learners fit both nuisances, and the estimate stays a plug-in | a valid interval needs a product rate on the two nuisances |
| you want an interval you can report | the interval comes from the targeted influence curve | positivity must hold, and a support report cannot verify it |

A regression coefficient and an inverse-probability-weighted mean each rest on one model. TMLE
targets the outcome regression with the treatment mechanism, so it uses both.

## The data

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

The baseline covariates are standardized (mean 0, SD 1), so a negative `age` is below the
average age. The transition score is in synthetic units. Its effect is larger than a real navigation
program would expect, so each fit shows its behavior clearly. Every discharge is independent here,
and the [cross-fitting tutorial](cross-fitting.md) adds shared navigator teams.

| feature of the law | what it means in this program |
| --- | --- |
| the four baseline covariates drive assignment and the outcome | higher-risk patients are more likely to receive an offer and report different outcomes. All four are confounders in the synthetic law |
| both nuisance functions are nonlinear | a GLM is misspecified for each one, which is the condition this page exploits |
| the effect varies with the covariates | `ate`, `att`, and `atc` differ, so the estimand must be named rather than inferred |

A real program has no `truth`. Every comparison against it below is a teaching device.

## Design and identification

The design says which column plays which role. The estimand says which contrast you want. The two
are separate on purpose, so changing the estimator later cannot change the question.

```python
from cleverly import ATE, CausalStudy, PointTreatment, StudyProtocol

protocol = StudyProtocol(
    target_population=(
        "Adults with a discharge-home order at a participating hospital during the enrollment period"
    ),
    eligibility=(
        "Age 18 years or older",
        "Discharge home ordered at a participating hospital",
    ),
    time_zero="Discharge-home order, after baseline measurement and before the navigation offer",
    treatment_strategies=(
        "Offer standard transition navigation",
        "Provide usual discharge support",
    ),
    treatment_versions=(
        "Bedside transition plan and two scheduled navigator contacts within 30 days",
        "No access to the transition-navigation offer",
    ),
    outcome="Patient-reported transition score",
    horizon="30 days after discharge",
    intercurrent_event_handling=(
        "Use the transition score regardless of readmission",
        "Analyze the offer regardless of completed contacts",
        "The protocol scores death before day 30 as the worst transition score (composite strategy)",
    ),
    interference_unit="Individual patient",
    assumption_rationale=(
        "The recorded baseline variables cover the measured common causes",
        "The standardized offer and version records support consistency",
        "Reserved navigator capacity and access controls support no interference",
    ),
)

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("discharge_risk", "prior_utilization", "medication_burden", "age"),
    ),
    protocol=protocol,
)
effect = study.identify(ATE(reference=0))

print(effect.summary())
for assumption in effect.identification.assumptions:
    print("-", assumption)
```

Time zero, eligibility, and the offer coincide at the discharge-home order. Eligibility therefore
cannot depend on the offer. The identification summary renders the stored protocol, and the typed
`ATE` owns the contrast and its reference arm.

`identify` returns the assumptions that carry the causal reading. Four apply here.

| assumption | what it means for this program | can the data check it? |
| --- | --- | --- |
| consistency | an offer always means the declared bedside plan and two scheduled contacts | no |
| no interference | one patient's assignment does not change another patient's offer or outcome | no |
| no unmeasured confounding | the recorded baseline variables block every common cause of assignment and the score | no |
| positivity | each baseline profile has some chance of an offer and of usual support | partly, through the support report |

The [shared study design](index.md#the-shared-study-design) states how the program supports
consistency and no interference. This page changes nothing in it.

No unmeasured confounding needs a causal argument. For example, an unrecorded discharge-team
judgment that affects both assignment and recovery would violate it. No estimator repairs that
failure. The sensitivity analysis below asks how strong such a judgment would need to be.

The synthetic law needs only four covariates. A real protocol should also evaluate pre-assignment
site, navigator-team, calendar, and language-access causes. Add them when the causal review places
them on a common-cause path.

## Estimate

The configuration is written out in full, so a reader sees every choice.

```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod

flexible = TMLEMethod(
    models=ModelSpec(
        outcome_learner=HistGradientBoostingRegressor(random_state=21),
        treatment_learner=HistGradientBoostingClassifier(random_state=21),
    ),
    cross_fitting=CrossFitting(n_folds=5),
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

The result summary repeats the protocol and its fingerprint, and it reports the method
configuration separately. It names the construction as stacked CV-TMLE. Cross-fitting separates
each nuisance prediction from the row used to evaluate it. The
[cross-fitting tutorial](cross-fitting.md) shows why that separation matters for flexible learners.

The interval is built from the targeted influence curve. It is not the outcome model's own standard
error. Its validity remains conditional on support, nuisance convergence, the product-rate
condition, and the declared dependence structure.

## Which population is the number about?

The average treatment effect answers a question about every patient. A spread decision asks
something narrower.

| estimand | the question it answers | who asks it |
| --- | --- | --- |
| ATT | what did patients who received an offer gain from assignment? | the teams reviewing the rollout |
| ATE | what would the eligible population gain if everyone received an offer? | the program sponsor |
| ATC | what would patients who received usual support gain from an offer? | whoever is deciding on spread |

These are three parameters, not three estimates of one. A second law makes the gap visible, because
its effect modification is aligned with the propensity. Patients most likely to receive an offer
benefit most.

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
    cross_fitting=CrossFitting(n_folds=5),
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

The population law has `att > ate > atc` by construction. Its propensity is exactly logistic, so
the treatment learner is correct. The linear outcome learner omits the law's navigation-by-risk
interaction, so these fits rest on the treatment model. Do not use interval overlap as a test of
the differences between these parameters.

Read that as a warning about spread. The offered patients' gain is the ATT. Patients who received
usual support would get the ATC, which here is a small fraction of it. A program that budgets the
network rollout against the ATT will overpromise.

## The failure mode: both nuisance models misspecified

Double robustness is a claim about *or*, not about *and*. Use the known synthetic law to compare
learner combinations. Treat the result as an illustration, not as validation evidence.

A gradient-boosted learner can represent the law's nonlinear features. The linear learners omit
those features by construction. Three more fits show the resulting finite-sample pattern.

```python
dr_points = {}


def fit(outcome_learner, treatment_learner, label):
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=outcome_learner, treatment_learner=treatment_learner),
        cross_fitting=CrossFitting(n_folds=5),
        runtime=Runtime(random_state=21, n_jobs=1),
    )
    point = effect.estimate(method=method)["ate"]
    dr_points[label] = point
    low, high = point.ci
    print(f"{label:22s} psi={point.psi:6.3f}  CI=({low:.3f}, {high:.3f})")


fit(
    HistGradientBoostingRegressor(random_state=21),
    LogisticRegression(max_iter=1000),
    "flexible Q, linear g",
)
fit(
    LinearRegression(),
    HistGradientBoostingClassifier(random_state=21),
    "linear Q, flexible g",
)
fit(LinearRegression(), LogisticRegression(max_iter=1000), "both linear")
print("population ATE:", truth["ate"])
```

On this draw, the fit with both linear learners misses the population value by several times more
than either fit with one flexible learner. This output does not establish nuisance consistency or
repeated-sampling coverage.

| caution | why |
| --- | --- |
| the linear models omit known terms | a real analysis does not reveal which nuisance model is consistent |
| one draw is not a coverage result | coverage is a repeated-sampling property |
| one nuisance is inconsistent | the point estimate can stay consistent, but the influence-curve interval need not be valid. [DR-TMLE](dr-tmle.md) addresses that case |
| point consistency and interval validity differ | Wald inference needs the stated product-rate and regularity conditions |

## How far to trust this

Start with the combined assessment. It presents validation, diagnostics, and sensitivity together.
Read `assessment.attention` for the failure and warning rows.

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

Call `assessment.to_frame()` for the complete row ledger. A `completed` row means the calculation
ran. It is not a pass.

On this draw, `needs attention` names `nuisance_models`, because the boosted propensity is poorly
calibrated. The support report clips a small share of rows at the truncation bound. It describes
fitted overlap and cannot verify population positivity.

Two lessons follow. First, a better-predicting assignment model is not automatically a better one
for this purpose. [Collaborative TMLE](collaborative-tmle.md) chooses the assignment model by its
effect on the targeted estimate. Second, the report gives no positivity verdict, because no
universal cutoff applies. Inspect the retained tables and the truncation curve before reporting.

```python
retargeted = result.assess(include_retargets=True)
curve = retargeted.report("truncation_curve")
print(curve)
```

The curve retargets the estimate at a range of truncation bounds without refitting the nuisance
models. Movement shows sensitivity to this regularization choice. Limited movement does not verify
positivity.

Diagnostics cannot see unmeasured confounding. Sensitivity analysis asks how strong a confounder
would need to be to explain the result away. The assessment above already holds the robustness
value. The benchmark calibrates that strength against `discharge_risk`, the covariate behind the
recorded risk process.

```python
robustness = assessment.report("robustness_value")
benchmark = result.sensitivity.benchmark(covariates=("discharge_risk",))
bounds = result.sensitivity.omitted_confounding(cf_y=benchmark.cf_y, cf_d=benchmark.cf_d)
print("robustness value:", robustness)
print(benchmark)
print("bias-adjusted 95% CI at the benchmark strength:", (bounds.ci_lower, bounds.ci_upper))
```

The benchmark refits the nuisances once without `discharge_risk`. The call goes to the facade
because `assess(include_refits=True)` would also run the costlier refutations.

The robustness value assumes worst-case alignment (`rho=1`). It is the equal outcome-side and
treatment-side strength that moves the point estimate to zero. On this draw it is about 0.26. At the
benchmark strength and `rho=1`, the bias-adjusted interval still excludes zero. The review must
decide whether an unrecorded team judgment could be stronger than the recorded risk score. The
[omitted-variable bounds](../technical-reference/validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours)
section defines each quantity.

`result.assess(include_refits=True)` adds placebo, noise, and subsampling refutations. A stable
refutation is not evidence of correctness.

The [stacked point-treatment CV-TMLE study](../technical-reference/method-evidence/stacked-point-treatment-cv-tmle.md)
validates this construction with GLM learners. No registered study covers the boosted learners
used here.

### What each layer can and cannot establish

| layer | establishes | does not establish |
| --- | --- | --- |
| assessment overview | which stored checks need attention, and which operations did not run | the detail needed to interpret each retained report |
| retained diagnostics | that targeting converged, and how concentrated the fitted weights are | that the nuisance models are right |
| sensitivity analysis | how strong an unmeasured confounder would need to be | that no such confounder exists |
| the registered study | the implementation recovers known truths and behaves as its theory predicts | that your identification assumptions hold on your data |

Nothing in this list validates the causal reading. That rests on consistency, no interference, and
no unmeasured confounding. All three are arguments about the program rather than about the fit.

## Keeping the result

A fit is an artifact. It carries the nuisance models, the influence curves, and the provenance
stamp, so the assessment above replays without refitting. A program that reports quarterly needs
that.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from cleverly import load

with TemporaryDirectory() as directory:
    saved = Path(directory) / "transition-navigation-ate.joblib"
    result.save(saved)
    restored = load(saved)
    restored_protocol = restored.identified_effect.protocol
    assert restored_protocol is not None
    print("\n".join(restored_protocol.summary_lines()))
    assert restored_protocol.fingerprint == restored.provenance.protocol_fingerprint
    print(restored.replayability)
    print(restored.assess().summary())
```

`replayability` says which operations the restored artifact can still perform. A new nuisance
refit still needs the analysis data. Use a maintained path for a real audit artifact. Load only
joblib files you trust, and keep the dependency versions compatible.

## Where to go next

This page treated discharges as independent rows. Read
[CV-TMLE and cross-fitting](cross-fitting.md) for the same question at network scale, where patients
share navigator teams. If your worry is instead which baseline variables belong in the assignment
model, read [collaborative TMLE](collaborative-tmle.md).

The [examples index](index.md#the-program) lists every tutorial in the program.
