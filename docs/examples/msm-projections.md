# MSM projections: three navigation cadences read as one trend

The network assigned three navigation cadences side by side. A marginal structural model (MSM)
summarizes their counterfactual means as one trend. This page shows what the slope means when the
working model does not fit.

## The applied question

| tier | assigned protocol | contacts in 30 days |
| --- | --- | --- |
| `low` | one transition-planning contact | 1 |
| `medium` | planning plus one follow-up contact | 2 |
| `high` | planning plus five follow-up contacts | 6 |

Nobody randomized cadence. Discharge risk and age influenced assignment and the outcome. Each
cadence uses the same script and access rules, which supports consistency. The rest of the
[shared study design](index.md#the-shared-study-design) is unchanged.

The program board asks how much experience changes per assigned contact. The estimand is the
straight line in assigned contacts closest to the three counterfactual means, with each cadence
weighted equally.

## Why this method

An outcome regression coefficient is an observed-data projection. The MSM coefficient is a
functional of the counterfactual means, and it has a value whether or not the line fits. Read the
[technical entry](../technical-reference/msm-projections.md) for the projection and its clever
covariate.

## The data

`make_multi_arm` draws a three-arm confounded law with known counterfactual means. Its covariates
are standardized (mean 0, SD 1), and the outcome is in synthetic units.

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

The published truths are keyed by the generator's labels, so the labels stay unchanged. Baseline
support need influences only the outcome. The gain per contact falls from 0.60 between `low` and
`medium` to 0.21 between `medium` and `high`. A working model linear in assigned contacts is
therefore misspecified on purpose.

## Design and identification

The design is an ordinary point-treatment design with a three-level treatment.

```python
from cleverly import ATE, CausalStudy, CounterfactualMean, PointTreatment

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

The printed positivity condition names `high`, `low`, and `medium`. Each level needs a positive
probability at every covariate value where the target places mass.

## Estimate

Start with the per-arm report, because the projection is a summary of it. The outcome and
multinomial treatment models below are correctly specified for this law.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=3),
    runtime=Runtime(random_state=61, n_jobs=1),
)
arm_result = arms.estimate(method=method)
print(arm_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The page writes the working model out. The `design` callable receives one arm label and the
covariate frame, and returns the design matrix for that arm. The `terms` tuple names the columns.

```python
import numpy as np

from cleverly import MSMProjection
from cleverly.msm import MSM

ARMS = ("low", "medium", "high")
CONTACTS_30D = {"low": 1.0, "medium": 2.0, "high": 6.0}
trend = MSM(
    design=lambda arm, data: np.column_stack(
        [np.ones(len(data)), np.full(len(data), CONTACTS_30D[arm])]
    ),
    terms=("(intercept)", "assigned contacts"),
)
trend_effect = study.identify(MSMProjection(trend))
print(trend_effect.summary())
trend_result = trend_effect.estimate(method=method)
print(trend_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The slope is in transition-score units per additional assigned contact. This declaration leaves
`weights` unset, which sets the projection weight h(a, V) to 1 for every cadence. The slope is then
the fixed contrast (-2 E[Y(low)] - E[Y(medium)] + 3 E[Y(high)]) / 14 of the three means.

An inverse-probability-weighted regression with stabilized weights uses h(a) = P(A = a) instead.
Its slope is a different estimand whenever the cadence shares are unequal.

The contact mapping is the program's decision, not the estimator's. `cleverly` refuses to guess a
number for a label.

```python
from cleverly.exceptions import DataError

try:
    study.identify(MSMProjection(MSM.linear())).estimate(method=method)
except DataError as error:
    refusal = str(error)
    print("refused:", refusal)
else:
    raise AssertionError("MSM.linear accepted text cadence labels")
```

`MSM.linear` reads each arm label as a number. The refusal does not fall back on sort order,
because nobody chose the order of `{"high", "low", "medium"}`. A `0, 1, 2` coding would also be
wrong here, because the real spacing is not even.

## The failure mode: a projection is not a fitted curve

The working model above is wrong. Compare the line with the arm means, first in the population and
then in the fits.

```python
contacts = np.array([CONTACTS_30D[arm] for arm in ARMS])
population = np.array([truth[f"ey[{arm}]"] for arm in ARMS])
design = np.column_stack([np.ones(3), contacts])
projection, *_ = np.linalg.lstsq(design, population, rcond=None)
names = ("msm[(intercept)]", "msm[assigned contacts]")
estimated_line = design @ np.array([trend_result[name].psi for name in names])
arm_means = np.array([arm_result[f"ey[{arm}]"].psi for arm in ARMS])
print("population projection (intercept, slope):", projection.round(4))
print("population line minus population means:", (design @ projection - population).round(4))
print("estimated line minus estimated arm means:", (estimated_line - arm_means).round(4))
print(arm_result.to_frame()[["estimand", "ci_lower", "ci_upper"]])
```

The population line misses `medium` by about 0.19. On this draw the estimated line falls well below
the `medium` interval, and each coefficient interval contains the population projection. The line
minimizes the total squared deviation from the three means under the uniform weight.

The estimand is the projection, and the estimator recovers it. Read the slope as "the best linear
summary of the cadence response under a uniform weight". Do not read it as "the causal effect of
one more contact".

| consequence | why |
| --- | --- |
| the coefficient depends on the working model | change the terms and you change the estimand, not just the estimate |
| the coefficient depends on the weight | the projection minimizes a weighted squared error, and the weight is part of the declaration |
| misspecification is not a bug in the fit | the parameter is well defined either way. Interval validity still needs the causal and nuisance conditions |

A board that reads the slope as "each extra contact buys this much" will extrapolate to ten
contacts. No patient in the study was assigned ten contacts, and the projection says nothing about
that cadence.

## The control: a saturated working model

A saturated model has one free parameter per arm. It cannot be misspecified, so it represents the
three arm means exactly.

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
arm_contrasts = study.identify(ATE(reference="low")).estimate(method=method)
print(saturated_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print(arm_contrasts.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The intercept matches the `low` arm mean. The other coefficients match the `medium` versus `low`
and `high` versus `low` contrasts, with the same intervals. The `ATE` uses the identification that
`arms.summary()` printed.

This check shows that the projection is a reparameterization, not a different analysis. It is also
the practical fallback. A board that does not want to commit to a spacing can report the saturated
model.

## How far to trust this

```python
assessment = trend_result.assess(include_retargets=True)
print(assessment.summary())

support = assessment.report("support")
scores = assessment.report("score_equations")
curve = assessment.report("truncation_curve")

print(support.summary())
print(scores.summary())
print(
    curve.loc[
        curve["estimand"] == "msm[assigned contacts]",
        [
            "bound",
            "upper_bound",
            "psi",
            "delta_from_fitted",
            "ci_lower",
            "ci_upper",
            "truncated_fraction",
        ],
    ]
)
```

`include_retargets=True` runs the truncation curve. It shows how each coefficient moves as the
fitted treatment mechanism is bounded more tightly. The score report has one row per coefficient,
not one per arm. The support report reads its verdict from the share of units the fit truncated.

| layer | establishes | does not establish |
| --- | --- | --- |
| the saturated control | the projection reproduces the arm report when it can | that a non-saturated working model is a good summary |
| the score-equation report | one solved score per coefficient | that the working model resembles the truth |
| the support report | how the fitted mechanism spreads over the three cadences, and how many units it truncated | that the true mechanism is bounded away from zero, or that the coefficient answers the board's question |
| the truncation curve | whether the slope is stable across declared mechanism bounds | support for a cadence outside the observed range |

These checks do not establish exchangeability or a well-chosen working model. The
[technical entry](../technical-reference/msm-projections.md#validation-issues-special-to-this-method)
links the registered evidence and its limits.

## Where to go next

The same projection works over regimens and horizons in a longitudinal fit. Its design callable
also receives the horizon, and `MSM.linear` is refused there too. Read
[longitudinal TMLE](longitudinal-tmle.md) and [time-to-event outcomes](longitudinal-survival.md)
first. The projection summarizes the parameters those pages estimate one at a time.
