# Quickstart

A health plan wants to know whether offering patients a navigation program after hospital
discharge raises their 30-day transition score. On simulated data where the true effect is 1.5
points, this page estimates it at 1.51 points, with a 95% confidence interval from 1.42 to 1.60.
The answer is causal only if the recorded patient characteristics include everything that drove
both the offer and the score, and no data can check that. Because the data are simulated, you can
compare each number with the truth.

The page needs a data-frame library: install `cleverly[all]` or `cleverly[pandas]`
([installation](installation.md)).

## The situation

A regional health plan offers **care-transition navigation** to adults who leave hospital for home.
Discharge teams decide who receives the offer, and nobody randomizes it, so the offered and
other patients can differ before the offer. Each row of the data is one discharged adult. The
[examples index](../examples/index.md#the-shared-study-design) describes the program.

| column | role | measured |
| --- | --- | --- |
| `discharge_risk`, `prior_utilization`, `medication_burden`, `age` | patient characteristics (covariates) | before the offer |
| `transition_navigation` | the treatment: 1 if the team offered navigation, 0 for usual support | at discharge |
| `transition_score` | the outcome: a 30-day transition score, in simulated points | at day 30 |

Each patient has two possible scores: the score had the team offered navigation, and the score
under usual support. The data show only one of them for each patient, so the analysis has to
estimate the other from similar patients who got the other option.

A covariate that changes both the chance of an offer and the score is a confounder. Adjust only
for characteristics measured before the offer. Completed navigator contacts, for example, happen
after the offer and are part of how it works; adjusting for them would remove part of the effect.

:::{dropdown} Details: the causal graph and deaths
With $W$ the four covariates, $A$ the offer and $Y$ the score, the analysis assumes this graph:

```text
   W = (discharge_risk, prior_utilization, medication_burden, age)
          |                      |
          v                      v
          A  ------------------> Y
```

$W \to A$: the covariates change the chance of an offer (in this simulation `age` does not).
$W \to Y$: the covariates change the score. $A \to Y$: the effect the question asks about. No other
node: no unmeasured variable affects both $A$ and $Y$, and no variable measured after the offer is
in $W$. The two possible scores are written $Y^1$ (offer) and $Y^0$ (usual support).

The score has no worst value, so a death cannot be scored as the worst outcome. The page uses the
hypothetical strategy: the score the patient would have had had they not died. The simulation
draws no deaths, so the choice does not change any number here.
:::

## The question to answer

The program sponsor asks one question: how much would the mean 30-day score change if every
eligible adult received the offer, rather than usual support? That quantity is the average
treatment effect (ATE) in the discharged population. A simple difference between offered and
other patients does not answer it, because confounders make those groups differ before the offer.
The average effect on the treated (ATT) answers another question, the effect among the patients
the teams chose. In this simulation the effect is the same for everyone, so the two are equal;
on real data they can differ.

Four conditions let the data answer the question. The code prints them under the names in the
first column; textbooks use the names in parentheses.

| condition | what it requires | how you check it |
| --- | --- | --- |
| well-defined treatment (consistency) | the observed score equals the score the patient would have had under the offer they received | judgment: one well-defined offer |
| no interference | one patient's offer does not change another patient's score | judgment: the program design |
| no hidden confounding (no unmeasured confounding) | among patients with the same covariates, who got the offer is unrelated to the scores they would have had | judgment; no test on these data can confirm it |
| overlap (positivity) | each kind of patient had some chance of each option | partly from the data: the fitted offer chances |

:::{dropdown} Details: the estimand and its identification
The estimand is $\psi = E[Y^1] - E[Y^0]$. Under the four conditions,
$\psi = E_W[E(Y \mid A=1, W) - E(Y \mid A=0, W)]$, so it can be computed from the observed data.
:::

The code first describes the data: which column is the outcome, which is the treatment, and which
covariates to adjust for (`PointTreatment`, inside a `CausalStudy`). It then names the question
(`ATE(reference=0)`, the offer compared with usual support, coded 0) and checks that the data can
answer it before any model is fitted (`study.identify`).

```python
import numpy as np
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate
from sklearn.linear_model import LinearRegression, LogisticRegression

# With your own data, load your table here, such as pd.read_csv(...), and skip the renaming.
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

The Conditions table repeats the four conditions, and its `checkable` column says the data cannot
check three of them; that is expected, not a warning. The Methods table lists the four estimators
that can answer this question ([other methods for the same question](#other-methods-for-the-same-question)
compares them). [Data and study design](../user-guide/data-design.md) gives the rules for each
column role, including weights, clusters and missing outcomes.

The fit needs two helper models, which the package calls nuisance models: an outcome model, which
predicts the score from the offer and the covariates, and a treatment model, which predicts the
chance of an offer from the covariates. Neither is the answer; both are steps toward it.

## Why a simple comparison misleads

In the simulation, `discharge_risk` raises both the chance of an offer and the score, while
`prior_utilization` and `medication_burden` push the other way. The code below compares the mean
scores of offered and other patients in one million simulated patients, so sampling noise cannot
explain any gap.

```python
# A fresh draw of the simulated data, with its original column names (A: offer, Y: score).
population, _ = make_linear_ate(n=1_000_000, seed=8)
treated = population["A"] == 1
association = population.loc[treated, "Y"].mean() - population.loc[~treated, "Y"].mean()
print(f"population difference in means: {association:.2f}")
print(f"population ATE:                 {truth['ate']:.2f}")
```

The simple difference is 1.62 points, against a true effect of 1.50. The extra 0.12 points, about
8% of the effect, come from who got the offer, and more data would not remove them.

## Estimate the effect

Targeted maximum likelihood estimation (TMLE), the default method, fits the outcome model, then
adjusts its predictions using the treatment model, giving more weight to patients whose offer was
hard to predict. The adjustment removes the bias a slightly wrong outcome model would leave. Here
both models are simple, a linear regression and an unpenalized logistic regression (`C=np.inf`),
because that is how the data were simulated, and each model is fitted on every row
(`cross_fit=False`). [Why TMLE with these settings](#why-tmle-with-these-settings) gives the
reasons, and [on your own data](#on-your-own-data) shows what to change.

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

Read the `ate` row: the estimate is 1.513 points, and the 95% interval runs from 1.422 to 1.603,
which contains the true 1.5. The header's probability limits are the bounds the package puts on
each predicted offer chance, so that no patient whose offer was almost certain or almost
impossible gets an extreme weight. The note under the table says the interval allows for a skewed
score; these data show almost no skew, so the interval is centred on the estimate.
[Read cleverly's output](../user-guide/reading-output.md) explains each label and note.

:::{dropdown} Details: how the interval and the bounds are built
For a continuous outcome without clusters, the package uses a t distribution and widens the
interval on the side where a skewed outcome makes misses more likely. Here the two sides differ by
less than 0.0001, and with this many rows the t distribution is close to the normal one. For a
yes/no outcome, the interval of a mean or of the ATE is built differently, so that it stays inside
the possible range. [Small-sample scale](../technical-reference/inference.md#small-sample-scale)
gives both constructions. The probability limits are [0.01471, 0.9853]. The package also rescales
the score to between 0 and 1 for the adjustment step and maps the estimate back.
:::

The `support` column of `result.to_frame()` says whether the package's own simulation studies
cover a fit set up like this one. `supported` means they do. `experimental` means they do not, or
a stated condition applies; the estimate is computed the same way, but no study backs its
interval, so report it with that caveat.

```python
print(result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper", "support", "interval_scale"]])
```

This fit reads `supported`. [Read the support label](../user-guide/results-assessment.md#read-the-support-label)
says how to find the study behind a label.

## Check the fit

`assess()` runs the standard checks on the fitted result: overlap, the two helper models, whether
the adjustment step did its job, and how strong a missing confounder would have to be. It refits no
model unless you ask.

```python
assessment = result.assess()
print(assessment.summary())
```

The header reads "nothing needs attention", and the only pass-or-fail check, equations solved,
passed: the adjustment step reached its target. That shows the step ran correctly, not that either
helper model is right. The overlap row says no patient's offer chance reached a limit, so the
estimate does not lean on the bounds. The Not run table lists the checks the default call skips;
for each check deferred because it is slow or needs an argument, the first note gives the call.
[Run the assessment battery](../user-guide/results-assessment.md#run-the-assessment-battery)
explains the three tables.

The unmeasured-confounding row reports 0.514. A missing confounder's strength here is the share of
what the covariates leave unexplained, in the score and in the offer weights, that it would
explain. A confounder with both strengths at 0.514 could, at worst, move the estimate to zero; one
with both below 0.514 could not.
[Omitted-variable bounds](../technical-reference/validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours)
give the exact definitions.

Whether 0.514 is large depends on your covariates. The benchmark measures how strong a missing
confounder would be if it mattered as much as one you recorded, and the bounds say how far it could
move the estimate:

```python
# Name a covariate you recorded and understand well.
benchmark = result.sensitivity.benchmark(covariates=["discharge_risk"])
print(benchmark.summary())
print(result.sensitivity.omitted_confounding(benchmark=benchmark).summary())
```

Leaving out `discharge_risk` would move the estimate from 1.51 to 1.78. Its strength with the score
is 0.870, but it adds little to the offer weights once the other covariates are known (0.020). A
missing confounder that strong could move the estimate to anywhere from 1.24 to 1.79, which stays
well above zero; whether such a variable exists is a question about your data. The E-value row says
the same on an approximate risk-ratio scale: a missing confounder would need a risk ratio of about
3.5 with both the offer and the score to explain the effect away
([E-value](../technical-reference/validation-methods.md#e-value)).
[Sensitivity analysis](../user-guide/results-assessment.md#sensitivity-analysis) covers the other
checks, and [plot results](../examples/plots.ipynb) draws the overlap, balance and sensitivity
figures.

:::{dropdown} Details: what each check shows and what backs the interval
| check | what it shows | what it does not show |
| --- | --- | --- |
| the true effect of the simulation | this estimate is 0.013 from it | how the interval behaves over repeated samples |
| equations solved, `result.diagnostics.score_equations()` | the adjustment step solved the efficient influence-curve equation | that either helper model is correct |
| overlap, `result.diagnostics.support()` | no patient's fitted offer chance reaches a bound, and the effective sample size (what the weighted data are worth in equally weighted patients) is above 97% of each arm | overlap in the whole population |
| the [canonical point-treatment TMLE study](../technical-reference/method-evidence/canonical-point-treatment-tmle.md), behind this fit's `supported` label | the symmetric normal interval covers the true ATE in 94.81% of 1,600 replications at n = 1,000, with linear and logistic helper models and no cross-fitting. The printed interval contains that symmetric interval, because it only widens it | coverage on this simulation. The study's outcome has a bounded range and an effect that varies |
| the [asymmetric interval study](../technical-reference/method-evidence/asymmetric-wald-intervals.md) | the skew-widened t interval keeps close to 95% coverage for subgroup estimates of skewed and symmetric continuous outcomes | coverage on this simulation |
| the omitted-variable sensitivity analysis | the confounder strength that would move the estimate to zero | whether such a confounder exists |

The E-values are 3.52 for the estimate and 3.34 for the interval's end nearest zero. For a
continuous score the package converts the effect to an approximate risk ratio first.
:::

## What the result means

If every eligible adult received the navigation offer, the mean 30-day score would rise by about
1.5 points compared with usual support, with a 95% interval of 1.42 to 1.60. The estimate describes
the discharged population the sample represents, and the offer, not completed navigation.

The limits a reader reaches for are these. The four conditions hold here because the data were
simulated that way; on real data each needs an argument from how the program ran. The ATE is an
average, so it says nothing about which patients gain most, and it belongs to this population, not
to another health plan. Finally, one simulated sample cannot show that the interval covers 95% of
the time: the package's label rests on a study of in-sample linear and logistic models on an
outcome with a bounded range, which measured the symmetric interval.

## On your own data

On your own data the true models are unknown, so use flexible models and keep the default
out-of-fold fitting (cross-fitting: each row's prediction comes from models fitted without it).
With a yes/no outcome, such as a successful transition, that takes no other setting:

```python
from cleverly.datasets import make_binary_outcome
from sklearn.ensemble import HistGradientBoostingClassifier

transitions, _ = make_binary_outcome(n=2_000, seed=11)
binary_study = CausalStudy(
    transitions,
    design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
)
binary_effect = binary_study.identify(ATE(reference=0))
print(binary_effect.planned_fits(random_state=7))
binary_result = binary_effect.estimate(
    outcome_learner=HistGradientBoostingClassifier(max_iter=50, random_state=0),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=7,
)
print(binary_result.summary())
```

The summary's header says the predictions came from 10 folds, and the `ate` row gives the effect on
the chance of a successful transition, about 0.22, with its interval. The fence uses gradient
boosting for the outcome and a logistic regression for the treatment to keep it fast; any
scikit-learn model fits either slot.

When you leave out both learners, the package combines three scikit-learn models (gradient
boosting, a random forest and a lasso), weighted by how well each predicts on held-out folds. The
first print, `planned_fits`, counts the work before you start: 360 candidate fits for this
question, which often take half a minute to two minutes
([count the model fits](../user-guide/methods-learners.md#count-the-model-fits-before-a-long-run)).

A continuous outcome needs its known range when it is cross-fitted, such as `q_bounds=(0, 100)` for
a 0-to-100 score, passed to `estimate()`. Without it the package would have to read the range off
every row, the held-out rows included. No published result covers that, so it refuses the fit
([fold and outcome-scale rules](../technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)).
An outcome with no range you can state from how it is measured, like this page's score, has no
cross-fitted route: fit it on every row, as this page does, and use simple models such as linear
and logistic regressions, because the interval of an in-sample fit is valid only for such models.

## Go deeper

### Why TMLE with these settings

TMLE fits the outcome model, then updates it with a term built from the treatment model, so that
the estimate solves the efficient influence-curve equation. Its interval is valid in large samples
under three conditions: overlap holds, the helper models converge fast enough, and either the
models are simple enough (a Donsker class, which includes parametric models) or the fit is
cross-fitted. [Point-treatment TMLE](../technical-reference/point-treatment-tmle.md) states the
conditions. On this page `make_linear_ate` draws the score from a linear model and the offer from a
logistic model, so `LinearRegression()` and the unpenalized `LogisticRegression(C=np.inf)` are
correctly specified, and an in-sample fit keeps the interval valid. The score has no known range,
so the cross-fitted route, which needs `q_bounds`, is not available.

### Other methods for the same question

The Methods table lists four estimators of the same ATE; they differ in how they handle a hard
case. This page uses `tmle`. Collaborative TMLE (`collaborative_tmle`) chooses how much of the
treatment model to use, which helps when one covariate almost decides who gets the offer
([a strong predictor of treatment](../examples/collaborative-tmle.ipynb)). With its default search
it reports the estimate without an interval; the
[inference status](../technical-reference/inference.md#inference-status) table names the
settings that report one. DR-TMLE (`drtmle`) adds corrections meant to keep the interval valid
when one helper model is wrong; the package has shown this only when the treatment probabilities
are known ([DR-TMLE](../technical-reference/dr-tmle/index.md)). Adaptive-propensity weighting
(`adaptive_iptw`) fits no outcome model and reports an interval only when the treatment
probabilities are known, as in a trial
([adaptive-propensity IPTW](../technical-reference/adaptive-iptw.md)).
`effect.available_methods()` lists every method with the reason the others do not fit, and the
[methods and learners guide](../user-guide/methods-learners.md#method-presets) shows how to choose
one.

## Choose the next path

- Read the [workflow](../workflow.md) before adapting the quickstart to real observational data. It
  also shows how to record a written study description (`StudyProtocol`).
- Read [data and study design](../user-guide/data-design.md) to map your columns to roles.
- Read [point-treatment TMLE](../examples/point-treatment-tmle.ipynb) for an analysis with nonlinear
  helper models and flexible learners.
- Use the [estimands and interventions guide](../user-guide/estimands.md) to choose a question.
- Use the [methods and learners guide](../user-guide/methods-learners.md) before you change the
  default learners.
- Read [capabilities and refusals](../user-guide/capabilities.md) for what the package can estimate,
  and why it refuses a request it cannot answer correctly.
- Consult the [technical reference](../technical-reference/index.md) for equations, assumptions,
  citations, implementation provenance, and correctness evidence.
