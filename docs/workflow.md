# Workflow

`cleverly` makes the order of a causal analysis visible. The estimator cannot rescue an undefined
question or an implausible identification argument, so those decisions precede learner selection.

## 1. Formulate the causal question

State the population, treatment strategy, outcome, time horizon, and contrast. Choose the target
family. It can be an arm contrast, a regime mean, a treatment policy, an incremental intervention,
an MSM projection, or a longitudinal quantity. A label such as “ATE” is insufficient when the
reference arm or target population is ambiguous.

Store the study context in a `StudyProtocol` when the analysis needs a durable record:

```python
from cleverly import StudyProtocol

protocol = StudyProtocol(
    target_population="Adults discharged home from participating hospitals",
    eligibility=("Adult at discharge", "Discharged alive", "Discharged home"),
    time_zero="Hospital discharge, before navigation assignment",
    treatment_strategies=("Offer navigation", "Provide usual discharge support"),
    treatment_versions=("Bedside plan and two scheduled contacts", "No navigation offer"),
    outcome="Thirty-day patient-reported transition score",
    horizon="30 days after discharge",
    intercurrent_event_handling=("Use the score regardless of readmission",),
    interference_unit="Patient",
    assumption_rationale=(
        "Baseline adjustment covers the measured common causes of assignment and outcome",
        "Reserved navigator capacity supports the no-interference argument",
    ),
)
```

This record combines selected vocabulary from the
[target-trial components](https://miguelhernan.org/whatifbook),
[ICH E9(R1)](https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf),
and the literature on
[treatment versions](https://pmc.ncbi.nlm.nih.gov/articles/PMC4219328/). It is not a complete
target-trial protocol or proof of ICH estimand compliance.

## 2. Declare the observed-data design

Use `PointTreatment` for one treatment node and `LongitudinalTreatment` for an ordered treatment
history. Column roles are part of the design rather than learner arguments.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import CausalStudy, PointTreatment

study = CausalStudy(
    data,
    design=PointTreatment(
        outcome="outcome",
        treatment="treatment",
        adjustment=("age", "baseline_score", "site"),
        weights="sampling_weight",
        cluster="household",
    ),
    protocol=protocol,
)
```

An empty adjustment set is an identification claim. Set `randomized=True` when randomization, not
omission, justifies it. `randomized=True` does not give the fit the allocation probabilities: the
treatment mechanism is still estimated. A trial that records its allocation probabilities declares
them with `treatment_probabilities=`, and every method then divides by them
([known treatment mechanism](technical-reference/point-treatment-tmle.md#known-treatment-mechanism)).

Each public object keeps one responsibility:

| object | owns | does not own |
| --- | --- | --- |
| `StudyProtocol` | the population, eligibility, timing, strategies, versions, outcome, follow-up, intercurrent-event handling, interference unit, and assumption rationale | the estimand contrast or analysis configuration |
| study design | observed column roles and treatment-time structure | the scientific rationale for those choices |
| typed estimand | the mathematical contrast, intervention, and target-specific metadata | observed column roles or learner choices |
| identification | the observed-data functional, assumptions, nuisances, and remainder condition | the evidence that assumptions hold in this study |
| typed method | learners, cross-fitting, targeting, inference, and runtime settings | the causal question or identification argument |

`CausalStudy` stamps the optional protocol onto the identified effect and each fitted result. A
summary prints the analysis by default. `summary(detail="full")` also prints the protocol's
fields, or says that no study protocol was recorded.

## 3. Choose a typed estimand

```python
from cleverly import ATE

question = ATE(reference=0)
```

Typed estimands carry their required design and intervention information. They prevent a
continuous-dose policy from being interpreted as an arm contrast and keep longitudinal regimen,
horizon, and cause metadata intact.

## 4. Inspect identification

```python
effect = study.identify(question)
print(effect.summary())
print(effect.available_methods().summary())
```

The question's summary lists the conditions the answer rests on, such as no hidden confounding
and overlap. It says whether the data can probe each one, and it names the estimation methods
that fit. The methods summary lists every method, with yes or no. Pass `detail="full"` to
`summary()` to see each condition in its formal wording and why a method does not fit. A program
reads the same statements from `effect.identification`.

The identified effect is the boundary between scientific and computational choices. If a
composition is well posed but unsupported, `cleverly` raises a capability error with the missing
derivation or implementation rather than estimating a convenient different parameter.

## 5. Configure estimation

Start with the ordinary `"tmle"` preset. Use an immutable method object when the analysis needs an
auditable configuration:

```python
from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(), treatment_learner=LogisticRegression(max_iter=1000)
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3, repeats=1),
    inference=Inference(alpha=0.05, simultaneous=False),
    runtime=Runtime(random_state=17, n_jobs=1),
)
result = effect.estimate(method=method)
```

Common keyword shortcuts normalize into the same objects. Unknown or inapplicable options raise
`MethodConfigurationError` before an engine is constructed.

The example above fits a binary outcome. A cross-fitted fit of a *continuous* outcome also needs
`Targeting(q_bounds=(lower, upper))`, holding the outcome's known support. Without it the fit
would read its scale off every observed outcome, held-out rows included, and `cleverly` refuses it
before any learner runs. If the transition score has a fixed scale, declare its minimum and
maximum. An unbounded outcome has no such range, so fit it in sample with
`CrossFitting(enabled=False)` instead.
[Fold and outcome-scale rules](technical-reference/cv-tmle.md#fold-and-outcome-scale-rules)
gives the messages.

## 6. Assess estimation quality

`result.assess()` runs the default checks in one call: overlap, the nuisance models, the
estimating equations, and the sensitivity analyses that need no extra argument and refit nothing. Its summary has a header with the count of
findings that need attention, then tables. Only the Checks table passes or fails. The Results table
reports findings that have no pass or fail rule. The Not run table names each check the default
skips and why, and for a deferred check the first note gives the call that runs it. `include_refits=True` adds the checks that refit the
models, such as the refutation tests, and `include_retargets=True` adds the truncation curve.

```python
print(result.assess().summary())
```

A solved estimating equation is evidence about the targeting step, not proof that the nuisance
models are consistent or that the identification assumptions hold. Each check also has its own
call, with a longer report:

```python
print(result.diagnostics.support().summary())
print(result.diagnostics.nuisance_models().summary())
print(result.diagnostics.score_equations().summary())
print(result.sensitivity.run_all().summary())
```

## 7. Report the result and its limits

Report the parameter definition, population, identification assumptions, learner and cross-fitting
configuration, estimate and interval, support diagnostics, sensitivity analysis, and package
version or commit. This page's fit is weighted and clustered by household, every fold keeps a
household whole, and the households differ in size. The rule below counts households with
positive weight, and this page's 40 keep the normal reference. From 3 to 39 households the interval uses a
Student t reference. From 3 to 9 households the fit still reports the interval, labelled
experimental, and `result.inference_status` is `"few_cluster_plugin"`. Below 3 households it
withholds the interval: report the point estimate and the reason the status gives in its place
([few clusters](technical-reference/inference.md#few-clusters)).

Save structured results when an audit trail matters:

```python
result.save("analysis.joblib")
```

The joblib artifact stores the complete fitted result, including estimator configuration and
nuisance-model objects. Load only trusted artifacts in a compatible Python environment.
`cleverly.load()` warns with `VersionMismatchWarning` when another `cleverly` version saved the
artifact. It then loads the artifact as saved, with no migration. See
[results, inference, and assessment](user-guide/results-assessment.md) for replayability rules.
