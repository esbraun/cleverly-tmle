# Survey non-response: the patients who never answered

Every result so far treated the experience score as if the program had it for every patient. It does
not. HCAHPS goes out after discharge, and most patients never send it back.

This test of change asks the rounding question again with that fact declared. It shows what a
complete-case analysis is actually estimating, and why that is a different parameter from the one
the program wants.

Read
[Missing outcomes and controlled direct effects](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects)
for the clever covariate and the identification argument.

## The applied question

The plan mails the HCAHPS survey after discharge. A response counts toward the ward's score if it
arrives inside a **49-day window**. After that the case closes.

Who answers is not random. Older patients answer more. Patients with a long, difficult stay answer
less. And a ward that rounds hourly may leave patients more willing to answer at all, which makes
response depend on the exposure as well as on the patient.

The program still wants the same estimand as the [first test of change](point-treatment-tmle.md).
What would the average experience score be if every discharged patient had been cared for under
hourly rounding, compared with nobody? That question is about **every discharged patient**, not
about the quarter who replied.

A complete-case analysis answers a different question. It estimates an effect in the population of
people who would have responded, and it is that substitution, not the loss of sample size, that this
page is about.

One honest note about the 49-day window. It is the definition of a response here, and nothing on
this page models time-to-response. The estimator sees a binary indicator, and the window is what
determines its value.

This page analyses one hospital's patients, so patients are treated as independent.

## Why this method

Missingness is not a nuisance to be dropped. It is a second mechanism, and it enters the same clever
covariate as the exposure.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| outcomes missing for reasons you recorded | the full-population estimand, identified under missingness at random given the recorded variables and the arm | a response model on top of the adoption model |
| response depends on the exposure | the two mechanisms compose into one factor, so the arm-dependence is handled rather than assumed away | positivity is now needed for the **product** of the two mechanisms, not for either alone |
| you want double robustness | you keep it, in a different shape | it becomes "the outcome regression is right, **or** the adoption model and the response model are both right" |

That last row is the one to read twice. Double robustness in the complete-data case gives you two
independent chances. Here the second chance requires two models to be right together, so it is a
weaker guarantee than it looks.

The alternative that fails is the obvious one. Dropping the non-respondents and fitting the usual
analysis is consistent only when a correctly specified outcome regression can extrapolate from
respondents to everyone. When the outcome surface has curvature the model cannot reach, and
respondents carry a shifted covariate distribution, the extrapolation is wrong in a direction
nothing in the fit reveals.

## The data

The generator is `make_missing_outcome`. Its `strength` argument controls exactly the combination
described above: curvature in the outcome surface, plus a response mechanism that sharpens on the
same covariate.

```python
from cleverly.datasets import make_missing_outcome

frame, truth = make_missing_outcome(n=4_000, seed=71, strength=2.0)
frame = frame.rename(
    columns={
        "Y": "experience_score",
        "A": "hourly_rounding",
        "W1": "acuity",
        "W2": "age",
        "W3": "prior_admissions",
        "Delta": "responded",
    }
)
print(frame.head())
print("response rate:", frame["responded"].mean())
print("population ATE:", truth["ate"])
```

The score is missing wherever `responded` is zero. That is the shape the container expects, and it
refuses a missing outcome that carries no indicator.

| feature of the law | what it means for the survey |
| --- | --- |
| response depends on acuity and on prior admissions | respondents are not a random slice of the discharged population |
| response also depends on the arm | rounding changes who answers, so the respondent pool differs between the two groups being compared |
| the outcome surface has curvature a main-effects model cannot reach | a linear regression fitted to respondents extrapolates the wrong shape to everyone else |

The response rate in this law is higher than a real HCAHPS rate. What drives the bias is the
mechanism rather than the rate, so the demonstration holds at a realistic rate too and is only
noisier.

## Design and identification

The response indicator is a **design role**, like the outcome and the exposure. Declaring it is what
tells the estimator that the missing rows are part of the population.

```python
from cleverly import ATE, CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="experience_score",
        treatment="hourly_rounding",
        adjustment=("acuity", "age", "prior_admissions"),
        missingness="responded",
    ),
)
effect = study.identify(ATE(reference=0))

print(effect.summary())
for assumption in effect.identification.assumptions:
    print("-", assumption)
```

The assumptions gain one, and one of the old ones changes shape.

| assumption | what it means here |
| --- | --- |
| missingness at random | given the recorded variables and the arm, who responds carries no further information about what they would have scored |
| positivity, in its new form | every kind of patient had some chance of both arms **and** some chance of responding. The product is what must stay away from zero |

Missingness at random is not testable, and it is a strong claim. The recorded variables must explain
away every reason a patient's own score would predict whether they answered. A patient who was
furious about their stay and threw the survey away violates it directly, and no diagnostic on this
page detects that.

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
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
    runtime=Runtime(random_state=71, n_jobs=1),
)
full = effect.estimate(method=method)
print(full.summary())
print("population ATE:", truth["ate"])
```

## The failure mode: a complete-case analysis changes the question

Now do what the program office would do without thinking about it. Drop the patients who never
answered, forget the indicator, and run the ordinary analysis.

```python
respondents = frame[frame["responded"] == 1].drop(columns=["responded"])
complete_case = (
    CausalStudy(
        respondents,
        design=PointTreatment(
            outcome="experience_score",
            treatment="hourly_rounding",
            adjustment=("acuity", "age", "prior_admissions"),
        ),
    )
    .identify(ATE(reference=0))
    .estimate(method=method)
)


def show(label, result):
    point = result["ate"]
    low, high = point.ci
    covered = low <= truth["ate"] <= high
    print(f"{label:22s} psi={point.psi:6.3f}  CI=({low:.3f}, {high:.3f})  covers={covered}")


show("complete cases only", complete_case)
show("missingness declared", full)
print("population ATE:", truth["ate"])
print("rows used:", len(respondents), "of", len(frame))
```

At the documented sample size the complete-case fit sits well above the population value, and its
interval excludes it. The fit that declares the response mechanism covers it.

Note what the complete-case interval looks like. It is narrow and it is confident. Nothing in that
fit is broken. It solved its score equation, its diagnostics pass, and it is a perfectly good
estimate **of the effect among people who would respond**. It answers a question nobody asked.

### The honest boundary

This demonstration is not a licence to say complete-case analysis is always wrong. It is not.

```python
mild_frame, mild_truth = make_missing_outcome(n=4_000, seed=71, strength=1.0)
mild_frame = mild_frame.rename(
    columns={
        "Y": "experience_score",
        "A": "hourly_rounding",
        "W1": "acuity",
        "W2": "age",
        "W3": "prior_admissions",
        "Delta": "responded",
    }
)
mild_respondents = mild_frame[mild_frame["responded"] == 1].drop(columns=["responded"])
mild = (
    CausalStudy(
        mild_respondents,
        design=PointTreatment(
            outcome="experience_score",
            treatment="hourly_rounding",
            adjustment=("acuity", "age", "prior_admissions"),
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

At `strength=1.0` the outcome mean is linear, a GLM is correctly specified for it, and the
complete-case fit lands on the truth. Missingness at random does **not** by itself bias a
complete-case analysis. What biases it is missingness plus an outcome model that cannot fit the
surface it is extrapolating.

That is the useful rule for a program. Modelling the response mechanism is a matter of efficiency
when the outcome model is right, and a matter of bias when it is not. Since nobody knows which case
they are in, declare the indicator.

## Reading it as a top-box rate

Wards do not report standardized composites. They report the share of patients who answered "Always"
on nurse communication, and they compare that share as a ratio. One shipped law carries both a
binary outcome and a response indicator, so it is the only place to check that reading against a
known truth.

```python
from cleverly import OddsRatio, RiskRatio
from cleverly.datasets import make_missing_outcome_binary

box_frame, box_truth = make_missing_outcome_binary(n=4_000, seed=72)
box_frame = box_frame.rename(
    columns={
        "Y": "top_box",
        "A": "hourly_rounding",
        "W1": "acuity",
        "W2": "age",
        "W3": "prior_admissions",
        "Delta": "responded",
    }
)
box_study = CausalStudy(
    box_frame,
    design=PointTreatment(
        outcome="top_box",
        treatment="hourly_rounding",
        adjustment=("acuity", "age", "prior_admissions"),
        missingness="responded",
    ),
)
box_method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        missingness_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
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
is the multiplicative version a ward scorecard uses. The odds ratio is larger than the risk ratio
here, as it always is when the outcome is common, and reporting it as though it were a rate ratio
would overstate the change.

The ratio parameters are built on the log scale, so their intervals are asymmetric around the point
estimate. That is correct rather than a display artefact.

## How far to trust this

```python
print(full.diagnostics.support().summary())
print(full.diagnostics.nuisance_models().summary())
print(full.diagnostics.score_equations().summary())
print(full.validate().summary())
```

The support report is doing more work here than on any other page. Positivity is now a statement
about the product of two mechanisms. A patient with a middling chance of hourly rounding and a
middling chance of responding can still have a small product, and it is the product the clever
covariate divides by.

One family of estimands is refused under `missingness=`, and the refusal is worth knowing before you
plan a report. The attributable fraction needs a binary outcome, so ask it of the top-box study.

```python
from cleverly import PopulationAttributableFraction

try:
    box_study.identify(PopulationAttributableFraction(reference=0)).estimate(method=box_method)
except NotImplementedError as error:
    print("refused:", error)
```

The natural-course mean, the population attributable risk, and the attributable fraction all involve
the observed law. Under missingness at random each would need its own outcome and response score
equation, and the complete-case mean is a different parameter. So "what fraction of poor experience
is attributable to the old rounding practice" and "most patients did not answer" cannot be asked in
one fit.

| layer | establishes | does not establish |
| --- | --- | --- |
| the support report | whether the product of the two mechanisms stays away from zero | that the response model is correctly specified |
| the score-equation report | the targeting solved the composed score | that missingness at random holds |
| the mild-law comparison | that the bias needs curvature plus a sharpening mechanism, not merely missingness | which of the two cases your own data is in |
| the evidence manifest | exact-law and remainder checks for the randomized missing-outcome construction | a repeated-sampling study. Missing outcomes have no row in the validation grid |

The strongest assumption on this page is the one with no diagnostic at all. Missingness at random
says the recorded variables explain every reason a patient's own score would predict whether they
answered. A survey whose non-response is driven by the experience itself breaks it, and that is the
most plausible failure in patient experience work. Treat the estimate as conditional on an argument
you make from what you know about the mailing, not from the fit.

## Where to go next

Non-response is one mechanism removing patients from view. Disenrollment is another, and it acts
over time rather than once. Read [longitudinal TMLE](longitudinal-tmle.md) for the version where
members leave the plan between admissions.
