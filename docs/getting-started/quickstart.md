# Quickstart

This page estimates one average treatment effect from simulated observational data. It follows the
order of a real analysis: the situation, the question, the method, the fit, the checks, and the
reading. The data come from a law with a known answer, so you can compare the estimate with the
truth.

## The situation

A regional health plan offers **care-transition navigation** to adults who leave hospital for home.
The [examples index](../examples/index.md#the-shared-study-design) describes the program. Discharge
teams decide who receives the offer. Nobody randomizes it. Each row of the data is one discharged
adult.

| column | role | measured |
| --- | --- | --- |
| `discharge_risk`, `prior_utilization`, `medication_burden`, `age` | baseline covariates `W` | before the offer |
| `transition_navigation` | the treatment `A`: 1 if the team offered navigation, 0 for usual support | at discharge |
| `transition_score` | the outcome `Y`: a 30-day transition score, in synthetic units | at day 30 |

The synthetic score has no worst value, so this page uses the hypothetical strategy for death: the
score had the patient not died. The law draws no deaths.

Each patient has two counterfactual scores. $Y^1$ is the score had the team offered navigation.
$Y^0$ is the score under usual support. The data show only one of them for each patient.

The causal graph below states what the analysis assumes about these variables.

```text
   W = (discharge_risk, prior_utilization, medication_burden, age)
          |                      |
          v                      v
          A  ------------------> Y
```

| arrow or absence | what it asserts |
| --- | --- |
| `W -> A` | the baseline covariates change the chance of an offer. In this law, `age` has no such arrow |
| `W -> Y` | the baseline covariates change the score. A covariate on both paths is a confounder |
| `A -> Y` | the effect the question asks about |
| no other node | no unmeasured variable affects both `A` and `Y`. No variable measured after the offer enters `W` |

`W` contains only baseline variables. A mediator, such as completed navigator contacts, occurs after
the offer. Adjustment for it would remove part of the effect.

## The question and the estimand

The program sponsor asks one question. How much would the mean 30-day score change if every
eligible adult received the offer, rather than usual support? The estimand is the average treatment
effect (ATE) in the discharged population:

$$
\psi = E[Y^1] - E[Y^0].
$$

The difference in observed means does not answer this question. It compares patients who received
the offer with different patients who did not. Confounders make those groups differ before the
offer. The average effect on the treated (ATT) answers a different question: the effect among the
patients the teams chose. In this law the effect is constant, so the ATT equals the ATE. On real
data they can differ.

Four assumptions identify $\psi$ from the observed data.

| assumption | what it requires | how you check it |
| --- | --- | --- |
| consistency | the observed score equals the counterfactual score under the received offer | scientific judgment: one well-defined offer protocol |
| no interference | one patient's offer does not change another patient's score | scientific judgment: the program design |
| no unmeasured confounding | $Y^a$ is independent of $A$ given `W` | scientific judgment. No test on these data can confirm it |
| positivity | each kind of patient has a nonzero chance of each arm | partly from the data: the fitted offer probabilities |

Under these assumptions, $\psi = E_W[E(Y \mid A=1, W) - E(Y \mid A=0, W)]$. The code declares the
design and identifies the estimand before it fits anything.

```python
import numpy as np
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate
from sklearn.linear_model import LinearRegression, LogisticRegression

frame, truth = make_linear_ate(n=2_000, seed=7)
columns = {
    "Y": "transition_score",
    "A": "transition_navigation",
    "W1": "discharge_risk",
    "W2": "prior_utilization",
    "W3": "medication_burden",
    "W4": "age",
}
frame = frame.rename(columns=columns)

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

The summary states the question, the four conditions the answer rests on (with whether the data
can check each), and the methods that can answer it. The fit needs two helper models. The outcome
regression is $Q(A, W) = E(Y \mid A, W)$. The treatment mechanism is $g(W) = P(A = 1 \mid W)$. This page records no `StudyProtocol`, a written description of the
study's population, timing, and treatments. The [workflow](../workflow.md) shows how to record one.

## Why TMLE with these settings

Targeted maximum likelihood estimation (TMLE) fits $Q$, then updates it with a term built from $g$.
The update makes the plug-in estimate solve the efficient influence-curve equation.

The Wald interval from that influence curve is valid under three conditions. Positivity holds. The
nuisance estimates converge fast enough. Either the nuisance models are simple enough (a Donsker
class), or the fit is cross-fitted. [Point-treatment TMLE](../technical-reference/point-treatment-tmle.md) states the conditions.

Each setting of the fit below matches a feature of this problem.

| setting | the feature of the problem |
| --- | --- |
| `LinearRegression()` for $Q$ | `make_linear_ate` draws $Y$ from a linear model in `A` and `W`, so this learner is correctly specified |
| `LogisticRegression(C=np.inf)`, with no penalty, for $g$ | the law draws `A` from a logistic model in `W`, so this learner is correctly specified |
| `cross_fit=False` | a parametric model class is a Donsker class, so an in-sample fit keeps the interval valid. This outcome has no known support, and a cross-fitted continuous fit needs one declared as `q_bounds=(lower, upper)` |

On real data, the nuisance models are not known. Use flexible learners and cross-fitting there. Declare
`q_bounds` when the support of a continuous outcome is known. The
[methods and learners guide](../user-guide/methods-learners.md) explains both.

## A comparison with a known cause

In this law, `discharge_risk` raises both the offer probability and the score. Two other covariates
push the other way. The unadjusted difference in means therefore estimates
$E(Y \mid A=1) - E(Y \mid A=0)$, which is not $\psi$. The code computes that population quantity
from one million draws of the same law, so sampling noise cannot explain the gap.

```python
population, _ = make_linear_ate(n=1_000_000, seed=8)
treated = population["A"] == 1
association = population.loc[treated, "Y"].mean() - population.loc[~treated, "Y"].mean()
print(f"population difference in means: {association:.2f}")
print(f"population ATE:                 {truth['ate']:.2f}")
```

The difference in means is 1.62, and the ATE is 1.50. The gap of 0.12 is confounding bias. It is a
property of the law, not of one sample.

## Estimate the effect

```python
result = effect.estimate(
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(C=np.inf, max_iter=1000),
    cross_fit=False,
    random_state=7,
)
print(result.summary())
estimate = result["ate"]
print(f"TMLE: {estimate.psi:.3f}, 95% CI [{estimate.ci[0]:.3f}, {estimate.ci[1]:.3f}]")
print(f"population ATE: {truth['ate']:.3f}")
```

The header names the two models and shows the limits on the predicted offer probabilities,
[0.01471, 0.9853]. The package keeps $g$ away from 0 and 1. The assessment below reports that no
patient was capped at a limit. The package also maps the score to $[0, 1]$ for the
logistic update, and maps the estimate back. The TMLE estimate is 1.513, with a 95% interval from 1.422 to 1.603. The
interval contains the true value of 1.5.

## Check the fit

One interval that contains the truth does not show that the method works. Evidence for the
interval comes from repeated sampling.

| check | what it shows | what it does not show |
| --- | --- | --- |
| the truth of the simulated law | this estimate is 0.013 from $\psi$ | the behavior of the interval over repeated samples |
| the equations-solved check, `result.diagnostics.score_equations()` | the targeting step solved the efficient influence-curve equation | that $Q$ or $g$ is correct |
| the overlap check, `result.diagnostics.support()` | no patient's $g(W)$ reaches the bound. The effective-sample-size ratio is above 0.97 in each arm | structural positivity in the population |
| the [canonical point-treatment TMLE study](../technical-reference/method-evidence/canonical-point-treatment-tmle.md) | the ATE interval covers the true ATE in 94.81% of 1,600 replications at n = 1,000, with GLM nuisances and no cross-fitting | coverage on this law. The study law has a bounded outcome and effect modification |
| the omitted-variable sensitivity analysis | the confounder strength that would move the estimate to zero | whether such a confounder exists |

`assess()` runs the overlap, equations-solved, and sensitivity checks on the fitted result. It
refits no nuisance model unless you opt in.

```python
assessment = result.assess()
print(assessment.summary())
```

Only the Checks table passes or fails. The Results rows report findings that have no pass or fail
rule, so a row there is not a pass. The Not run table names each check the default call skips and
why; the first note gives the call that runs it. `assessment.to_frame()` lists every row with its
status, and the [status contract](../technical-reference/validation-methods.md#the-status-contract) defines
each one.

The unmeasured-confounding row reports 0.514. A left-out variable whose strengths
with the offer and with the score are both 0.514 could, at worst, move the point estimate to zero;
with both strengths below 0.514 it could not.
[Omitted-variable bounds](../technical-reference/validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours)
define the two strengths.

## What the result means

If every eligible adult received the navigation offer, the mean 30-day score would rise by about
1.5 synthetic units, compared with usual support. The 95% interval is 1.42 to 1.60. The estimate
describes the discharged population that the sample represents. It describes the offer, not
completed navigation.

This page establishes less than a real analysis must.

| conclusion | status here |
| --- | --- |
| the four identification assumptions hold | true by construction of the law. On real data, each needs a design argument |
| the nuisance models are correct | true by construction. On real data, flexible learners and cross-fitting replace this |
| the interval has 95% coverage | partly supported. The registered study covers in-sample GLM TMLE on a bounded-outcome law. No study covers this unbounded law, and one sample cannot |
| the effect is the same for every patient | not supported. The ATE is an average |
| the effect holds in another health plan | not supported. The estimand is specific to its population |

## Choose the next path

- Read the [workflow](../workflow.md) before adapting the quickstart to real observational data.
- Read [point-treatment TMLE](../examples/point-treatment-tmle.ipynb) for an analysis with nonlinear
  nuisances and flexible learners.
- Use the [estimands and interventions guide](../user-guide/estimands.md) to choose a question.
- Use the [methods and learners guide](../user-guide/methods-learners.md) before you change the
  default learners.
- Consult the [technical reference](../technical-reference/index.md) for equations, assumptions,
  citations, implementation provenance, and correctness evidence.
