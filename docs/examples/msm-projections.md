# MSM projections: three navigation cadences read as one trend

The network did not run one test of change. It assigned three cadences side by side. Reporting a
separate counterfactual mean for each is honest and unreadable. A marginal structural model gives
one coefficient instead.

This page fits one, and shows what the coefficient means when the working model does not fit.

Read [Marginal structural model projections](../technical-reference/msm-projections.md) for the
projection, its clever covariate, and the longitudinal version.

## The applied question

Eligible patients received one of three standardized navigation cadences.

| tier | assigned protocol | contacts in 30 days |
| --- | --- | --- |
| `low` | one transition-planning contact | 1 |
| `medium` | planning plus two follow-up contacts | 3 |
| `high` | planning plus five follow-up contacts | 6 |

Nobody randomized cadence. Recorded baseline support need influenced assignment and outcome.

The program board does not want three numbers. It wants to know whether experience improves with
navigation intensity, and by how much per assigned contact. That is a slope.

A slope is a summary, and it must be defined before it is estimated. The question the board is
really asking is this. Among all straight lines in assigned contacts, which one comes closest to the
true counterfactual response surface? That line is the estimand.

Each cadence uses the same script, contact window, and access rules. Only the number of assigned
contacts changes. This restriction supports consistency. Reserved capacity supports no interference.

## Why this method

An MSM in `cleverly` is a **projection**. It is the best approximation of the true counterfactual
surface within a working model you declared. It is not a claim that the working model is the true
causal response surface, and the estimand is well defined whether or not the working model fits.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| many arms, or many regimens | one coefficient vector instead of one mean per level | the coefficients mean what the working model says they mean |
| a cadence you want to summarize as a trend | a slope, with an influence curve and an interval | a model linear in the arm reads the arm as a dose, so non-numeric labels are refused |
| effect modification by a baseline variable | an interaction term in the working design | the design must be full rank on the realized cells |

The alternative that fails here is the familiar one. Fitting a linear regression of the experience
score on assigned contacts and the covariates gives a slope whose meaning depends on that regression
being correct. If it is wrong, the coefficient is not the projection of anything, and it changes
when you add a term.

The projection is different. It is a functional of the true law, defined by the working model and a
weight. It has a value whether or not the working model is close, and that value is what the
estimator targets.

## The data

The generator is `make_multi_arm`. It produces a three-armed confounded process with known
counterfactual means, and its arm labels are the three tiers.

```python
from cleverly.datasets import make_multi_arm

frame, truth = make_multi_arm(n=3_000, seed=61)
frame = frame.rename(
    columns={
        "Y": "transition_score",
        "A": "cadence",
        "W1": "discharge_risk",
        "W2": "age",
        "W3": "baseline_support_need",
    }
)
print(sorted(frame["cadence"].unique()))
for name, value in truth.items():
    print(f"{name:26s} {value:.4f}")
```

The three counterfactual means are not on a straight line. They rise with cadence, and the step from
`medium` to `high` is larger than the step from `low` to `medium`. That is what makes this a useful
law for the page. A working model linear in assigned contacts is misspecified here, on purpose.

The arm labels stay as the generator writes them, because the published truths are keyed by those
labels. The table at the top of this page is what maps a label to a cadence.

## Design and identification

The design is an ordinary point-treatment design. A multi-arm exposure is an exposure with more than
two levels, and nothing about the design changes.

```python
from cleverly import CausalStudy, CounterfactualMean, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="cadence",
        adjustment=("discharge_risk", "age", "baseline_support_need"),
    ),
)
arms = study.identify(CounterfactualMean())
print(arms.summary())
```

Start with the per-arm report, because the projection is a summary *of it*. A board that cannot
interpret the arm means cannot interpret their projection either.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=3, learner_folds=2),
    runtime=Runtime(random_state=61, n_jobs=1),
)
arm_result = arms.estimate(method=method)
print(arm_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

## Estimate: the working model is a declaration

The working model is written out. `design` is handed one arm label and the covariate frame, and
returns the design matrix for that arm. `terms` names the columns, and those names appear in the
report.

```python
import numpy as np

from cleverly import MSMProjection
from cleverly.msm import MSM

CONTACTS_30D = {"low": 1.0, "medium": 3.0, "high": 6.0}
trend = MSM(
    design=lambda arm, data: np.column_stack(
        [np.ones(len(data)), np.full(len(data), CONTACTS_30D[arm])]
    ),
    terms=("(intercept)", "assigned contacts"),
)
trend_result = study.identify(MSMProjection(trend)).estimate(method=method)
print(trend_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The slope is in transition-score points per additional assigned contact. It exists only because the
program declared what the three tiers mean in contacts.

That mapping is the program's decision, not the estimator's. `low`, `medium`, and `high` are labels.
`cleverly` refuses to guess a number for them.

```python
from cleverly.exceptions import CleverlyError

try:
    study.identify(MSMProjection(MSM.linear())).estimate(method=method)
except CleverlyError as error:
    print("refused:", error)
```

`MSM.linear` is the shorthand for a plain dose-response line. It reads the arm label as a number, and
these labels are not numbers. The refusal names what is missing rather than falling back on sort
order, because the sort order of `{"high", "low", "medium"}` is not one anybody chose. Silently
coding the tiers `0, 1, 2` would also have been wrong here, because the real spacing is not even.

## The failure mode: a projection is not a fitted curve

The working model above is wrong. The true response surface is not linear in assigned contacts. Read
what the coefficient still means.

```python
contacts = np.array([CONTACTS_30D[arm] for arm in ("low", "medium", "high")])
population = np.array([truth[f"ey[{arm}]"] for arm in ("low", "medium", "high")])
design = np.column_stack([np.ones(3), contacts])
projection, *_ = np.linalg.lstsq(design, population, rcond=None)
print("population arm means:", population)
print("population projection (intercept, slope):", projection)
print("estimated:")
print(trend_result.to_frame()[["estimand", "psi"]])
```

The estimated coefficients sit near the projection of the *population* means onto the same working
model. They do not sit near any straight line through the true surface, because no straight line
passes through all three points.

That is the whole idea. The estimand is the projection, and the estimator recovers it. Read the
slope as "the best linear summary of the cadence response under a uniform weight", not as "the
causal effect of one more contact".

Three consequences follow, and they are what a board should be told.

| consequence | why |
| --- | --- |
| the coefficient depends on the working model | change the terms and you change the estimand, not just the estimate |
| the coefficient depends on the weight | the projection minimises a weighted squared error, and the weight is part of the declaration |
| misspecification is not a bug in the fit | the parameter is well defined either way. Interval validity still needs the causal and nuisance conditions |

The applied warning follows from the first row. A board that reads the slope as "each extra contact
buys this much" will extrapolate it to ten contacts, which no patient in the study was assigned.
The projection says nothing about a cadence nobody used.

## The control: a saturated working model

A saturated model has one free parameter per arm. It cannot be misspecified, and it must therefore
reproduce the per-arm report exactly.

```python
saturated = MSM(
    design=lambda arm, data: np.column_stack(
        [
            np.ones(len(data)),
            np.full(len(data), float(arm == "medium")),
            np.full(len(data), float(arm == "high")),
        ]
    ),
    terms=("(intercept)", "medium vs low", "high vs low"),
)
saturated_result = study.identify(MSMProjection(saturated)).estimate(method=method)
print(saturated_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print(arm_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The intercept equals the `low` arm's mean. The two remaining coefficients equal the two contrasts
against it. The agreement holds at the point estimate and at the influence curve, so the intervals
match as well.

This is the check that says the projection machinery is a reparameterisation rather than a different
analysis. When the working model can represent the surface exactly, the projection *is* the surface.

It is also the practical fallback. A board that does not want to commit to a spacing can report the
saturated model and get the arm report under a single heading.

## How far to trust this

```python
print(trend_result.diagnostics.support().summary())
print(trend_result.diagnostics.score_equations().summary())
print(trend_result.validate().summary())
```

With the identity link the clever covariate has one column per term, so the score equation is one
per coefficient rather than one per arm. The score report reflects that, and it is the right place
to confirm the fluctuation solved every one of them.

Positivity is a three-arm statement here. Every patient needs a positive probability of each cadence
the working model reads. A support report showing a near-empty cell means the trend is being carried
by extrapolation into a cadence that kind of patient never received.

| layer | establishes | does not establish |
| --- | --- | --- |
| the saturated control | the projection reproduces the arm report when it can | that a non-saturated working model is a good summary |
| the score-equation report | one solved score per coefficient | that the working model resembles the truth |
| the support report | whether every cadence the design reads was actually observed | that the coefficient answers the board's question |
| the evidence manifest | exact-law, rank, and pooled-design witnesses for the point and longitudinal projections | a repeated-sampling study. MSM projections have no row in the validation grid |

The evidence for this family is the point `msm` row and the longitudinal MSM rows in the
[evidence manifest](../technical-reference/evidence.md#the-table), with rank and pooled-design
witnesses. R `ltmleMSM` differs in projection scale and is not treated as a parity oracle.

One composition is refused. `msm=` cannot be combined with `interventions=` or `shifts=`, because one
fluctuation solves one set of score equations. A fit reporting parameters from two axes would put two
of them under one heading.

## Where to go next

The same projection works over regimens and horizons in a longitudinal fit, where the design callable
receives the horizon as well as the label. `MSM.linear` is refused there too, and for a stronger
reason: a regimen is a sequence of decisions, and no arithmetic on its name summarizes it. Read
[longitudinal TMLE](longitudinal-tmle.md) first, because the projection summarizes the parameters
that page estimates one at a time.
