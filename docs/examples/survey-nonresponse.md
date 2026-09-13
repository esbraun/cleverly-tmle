# Survey non-response: the patients who never answered

About a quarter of patients never return the 30-day transition survey. This page asks the navigation
question again with that fact declared. It shows what a complete-case analysis estimates, and what
declaring the response indicator recovers.
[Missing outcomes](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects)
gives the clever covariate and the identification argument.

## The applied question

The survey goes out 30 days after discharge and closes on day 45. The program still wants the ATE
for **every eligible patient**, not only respondents. Discharge risk and prior utilization affect
response, and navigation can change willingness to reply.

The response model conditions on the arm and the baseline adjustment set. A baseline predictor of
response, such as language, must enter that set. Contact attempts after assignment cannot, because
navigation can change them. The protocol scores death before day 30 as the worst transition score
(composite strategy). That score counts as observed, so only living patients who do not respond
are missing.

## Why this method

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| outcomes missing for reasons recorded at baseline | the full-population estimand, identified under missingness at random given the baseline variables and the arm | a response model on top of the assignment model |
| response depends on the exposure | the two mechanisms compose into one factor in the clever covariate | identification needs the **product** of the two mechanisms to be positive |
| you want double robustness | you keep it, in a different shape | it becomes "the outcome regression is right, **or** the product of the assignment and response mechanisms is right" |

## The data

The generator is `make_missing_outcome`. Covariates are standardized (mean 0, SD 1), and scores are
in synthetic units. `strength=2.0` adds outcome curvature and an effect that varies with discharge
risk. It also makes response depend more strongly on discharge risk.

```python
from cleverly.datasets import make_missing_outcome

frame, truth = make_missing_outcome(n=4_000, seed=71, strength=2.0)
frame = frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "discharge_risk",
        "W2": "age",
        "W3": "prior_utilization",
        "Delta": "responded",
    }
)
print(frame.head())
print("response rate:", frame["responded"].mean())
print("population ATE:", truth["ate"])
```

The score is missing wherever `responded` is zero. The container refuses a missing outcome that
carries no indicator.

| feature of the law | what it means for the survey |
| --- | --- |
| response falls as discharge risk rises, and rises with prior utilization | respondents are not a random slice of the eligible population |
| response also depends on the arm | navigation changes who answers |
| the effect shrinks as discharge risk rises | respondents, who have lower risk, have a larger average effect than the eligible population |
| the outcome surface has curvature a main-effects model cannot reach | a linear regression fitted to respondents extrapolates the wrong shape |

About three quarters of patients respond in this law. That rate is higher than many real transition
surveys. Response rate alone does not determine selection bias.

## Design and identification

The response indicator is a **design role**, like the outcome and the exposure. Declaring it tells
the estimator that the missing rows are part of the population.

```python
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=("discharge_risk", "age", "prior_utilization"),
        missingness="responded",
    ),
)
effect = study.identify(ATE(reference=0))
print(effect.summary())
```

The summary adds missingness at random, response positivity, and the missingness mechanism as a
required nuisance.

| assumption | what it means here |
| --- | --- |
| missingness at random | given the baseline adjustment set and the arm, response carries no further information about the unobserved score |
| positivity, in its new form | every kind of patient had some chance of both arms **and** of responding. Identification needs the product to be positive. Stable inference also needs it away from zero |

Missingness at random is not testable. A patient who ignores the survey because navigation failed
violates it directly. No diagnostic can detect that violation from the observed data. The
sensitivity analysis below stresses one declared departure instead.

## Estimate

The response model gets its own learner slot.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5),
    runtime=Runtime(random_state=71, n_jobs=1),
)
full = effect.estimate(method=method)
print(full.summary())
print("population ATE:", truth["ate"])
```

These learners deliberately use the second route to double robustness. The linear outcome model is
wrong for this law. The two main-effects logistic models match the law's assignment and response
mechanisms exactly. One covered interval on one draw is not evidence of coverage.

## The failure mode: complete cases estimate a different population

Now do what the program office would do without thinking about it. Drop the patients who never
answered, forget the indicator, and run the ordinary analysis.

```python
respondents = frame[frame["responded"] == 1].drop(columns=["responded"])
complete_case = (
    CausalStudy(
        respondents,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=("discharge_risk", "age", "prior_utilization"),
        ),
    )
    .identify(ATE(reference=0))
    .estimate(method=method)
)


def show(label, result, target):
    point = result["ate"]
    low, high = point.ci
    covered = low <= target <= high
    print(
        f"{label:22s} psi={point.psi:6.3f}  se={point.std_error:6.4f}  "
        f"CI=({low:.3f}, {high:.3f})  covers={covered}"
    )


show("complete cases only", complete_case, truth["ate"])
show("missingness declared", full, truth["ate"])
print("population ATE:", truth["ate"])
print("rows used:", len(respondents), "of", len(frame))
```

On this draw the complete-case fit sits above the population value, and its interval excludes it.
The fit that declares the response mechanism covers it.

The complete-case fit averages over respondents only. Under missingness at random, it estimates the
effect standardized to the respondents' covariate distribution. That equals the eligible-population
ATE when the effect does not vary with the predictors of response. Here respondents have lower
discharge risk and a larger average effect.

Among respondents neither of its nuisance models is correct, so double robustness gives it no
protection. Its linear outcome model is wrong. Its assignment model is wrong too, because response
depends on the arm. `P(A = 1 | W, responded = 1)` is then not a main-effects logistic function. The
fit solved its score equation, but it answers a question about a different population.

### A law where complete cases agree

This demonstration does not show that complete-case analysis is always wrong.

```python
mild_frame, mild_truth = make_missing_outcome(n=4_000, seed=71, strength=1.0)
mild_frame = mild_frame.rename(
    columns={
        "Y": "transition_score",
        "A": "transition_navigation",
        "W1": "discharge_risk",
        "W2": "age",
        "W3": "prior_utilization",
        "Delta": "responded",
    }
)
mild_respondents = mild_frame[mild_frame["responded"] == 1].drop(columns=["responded"])
mild = (
    CausalStudy(
        mild_respondents,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=("discharge_risk", "age", "prior_utilization"),
        ),
    )
    .identify(ATE(reference=0))
    .estimate(method=method)
)
point = mild["ate"]
low, high = point.ci
print(f"mild law, complete cases: psi={point.psi:6.3f}  CI=({low:.3f}, {high:.3f})")
print("population ATE:", mild_truth["ate"])
```

At `strength=1.0` the outcome is linear and the effect is constant. The respondents' average effect
then equals the population's, and the linear model is correct. On this draw the complete-case
interval contains the truth. Do not expect that agreement when effects vary with response
predictors.

## Reading it as a top-box rate

Program scorecards often report the share of patients above a declared transition threshold. They
also compare that share as a ratio. One shipped law carries both a binary outcome and a response
indicator, so it checks that reading against a known truth.

```python
from cleverly import OddsRatio, RiskRatio
from cleverly.datasets import make_missing_outcome_binary

box_frame, box_truth = make_missing_outcome_binary(n=4_000, seed=72)
box_frame = box_frame.rename(
    columns={
        "Y": "top_box",
        "A": "transition_navigation",
        "W1": "discharge_risk",
        "W2": "age",
        "W3": "prior_utilization",
        "Delta": "responded",
    }
)
box_study = CausalStudy(
    box_frame,
    design=PointTreatment(
        outcome="top_box",
        treatment="transition_navigation",
        adjustment=("discharge_risk", "age", "prior_utilization"),
        missingness="responded",
    ),
)
box_method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5),
    runtime=Runtime(random_state=72, n_jobs=1),
)
for estimand, key in (
    (ATE(reference=0), "ate"),
    (RiskRatio(reference=0), "rr"),
    (OddsRatio(reference=0), "or"),
):
    point = box_study.identify(estimand).estimate(method=box_method)[key]
    low, high = point.ci
    print(f"{key}: {point.psi:6.4f}  CI=({low:.3f}, {high:.3f})  population {box_truth[key]:.4f}")
```

Three readings of one comparison. The difference is in percentage points of top-box. The risk ratio
is the multiplicative version a program scorecard uses. The odds ratio lies farther above one than
the risk ratio, and the gap is large for a common outcome. Reporting it as a risk ratio would
overstate the change.

The ratio intervals are built on the log scale, so they are asymmetric around the point estimate.

The population attributable fraction is refused under `missingness=`.

```python
from cleverly import CapabilityError, PopulationAttributableFraction

try:
    box_study.identify(PopulationAttributableFraction(reference=0)).estimate(method=box_method)
except CapabilityError as error:
    refusal = str(error)
else:
    raise AssertionError("the attributable fraction should be refused under missingness=")
print("refused:", refusal)
```

It needs a joint outcome and response score equation that is not yet derived.
[Missing-outcome natural-course contracts](../technical-reference/scope-and-refusals.md#missing-outcome-natural-course-contracts)
lists what the natural-course mean supports.

## How far to trust this

```python
assessment = full.assess(
    include_retargets=True,
    arguments={
        "missingness": {
            "gamma": (-2.0, -1.0, 0.0, 1.0, 2.0),
            "arm_gamma": {0: 0.0, 1: -1.0},
        },
        "tipping_gamma": {
            "arm_gamma": {0: 0.0, 1: -1.0},
        },
    },
)
print(assessment.summary())

support = assessment.report("support")
nuisance = assessment.report("nuisance_models")
missingness_curve = assessment.report("missingness")
tipping_gamma = assessment.report("tipping_gamma")

print(support.summary())
print(nuisance.summary())
print(missingness_curve)
print("tipping gamma:", tipping_gamma)
```

The combined assessment collects validation, diagnostics, and sensitivity in one object. It also
retains the detailed reports.

Positivity is now a statement about the product of two mechanisms. A patient with a middling
chance of navigation and response can still have a small product. The clever covariate divides by
that product. The support report shows each fitted factor and the product, with its effective
sample size and top-weight concentration.

The missingness curve is a pattern-mixture sensitivity analysis. Within each arm and covariate
profile, it shifts the mean of the unobserved scores away from the respondents' mean.

| element | meaning |
| --- | --- |
| `gamma` | the shift on the logit of the score rescaled to [0, 1]. The fit summary prints the range |
| `arm_gamma={0: 0.0, 1: -1.0}` | the control arm stays at MAR. Positive `gamma` lowers the unobserved navigation-arm mean |
| `gamma=0` | reproduces the MAR estimate by construction |
| tipping gamma | the point estimate reaches zero near `gamma=1.3` on this draw. That shift moves an unobserved navigation-arm mean by at most about 0.29 of the score range, at mid-range. The ATE moves less, because respondents keep their observed scores |
| `ci_lower`, `ci_upper` | reuse the MAR standard error, so they ignore uncertainty about `gamma` |

This curve does not detect why patients did not answer. It shows how far one stated departure must
move before the conclusion changes.

| layer | establishes | does not establish |
| --- | --- | --- |
| the support report | overlap for each fitted mechanism, plus joint ESS, concentration, and maximum leverage | that the response model is correct or missingness at random holds |
| the nuisance report | held-out fit and calibration measures for the treatment, response, and outcome models | that any nuisance model is correctly specified |
| the score-equation check in the summary | the targeting solved the composed score | that missingness at random holds |
| the MNAR tilt and tipping gamma | estimate movement under one declared arm-specific departure | that the departure describes why patients did not respond |
| the mild-law comparison | complete cases agree when the effect is constant and the outcome model is correct | whether the same agreement holds in another population |
| the [ordinary missing-outcome study](../technical-reference/method-evidence/ordinary-missing-outcome-tmle.md) | repeated-sampling truth, R `tmle` agreement, three-nuisance robustness, calibration, and a complete-case control on a binary law | coverage for this continuous law, or that missingness at random holds in this survey |

The strongest assumption on this page remains untestable. A survey whose non-response is driven by
the experience itself breaks missingness at random. The tilt shows consequences under one departure,
but it cannot establish which departure is plausible. Treat the estimate as conditional on an
argument about the mailing process.

## Where to go next

Non-response is one mechanism removing patients from view. Plan exit is another, and it acts over
time rather than once. Read [time-to-event outcomes](longitudinal-survival.md) for the version
where plan exit is the outcome.
