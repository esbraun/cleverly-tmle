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

The record combines selected vocabulary from the target-trial literature, the ICH E9(R1) estimand
addendum, and the work on treatment versions.
[Study protocol vocabulary](references.md#study-protocol-vocabulary) names the three sources and
states the scope of the record.

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
omission, justifies it.

### Who owns what

Each public object keeps one responsibility:

| object | owns | does not own |
| --- | --- | --- |
| `StudyProtocol` | the population, eligibility, timing, strategies, versions, outcome, follow-up, intercurrent-event handling, interference unit, and assumption rationale | the estimand contrast or analysis configuration |
| study design | observed column roles and treatment-time structure | the scientific rationale for those choices |
| typed estimand | the mathematical contrast, intervention, and target-specific metadata | observed column roles or learner choices |
| identification | the observed-data functional, assumptions, nuisances, and remainder condition | the evidence that assumptions hold in this study |
| typed method | learners, cross-fitting, targeting, inference, and runtime settings | the causal question or identification argument |

`CausalStudy` stamps the optional study protocol onto the identified effect and onto each fitted
result. The identification summary and the result summary both render the record, one field per
line. The result summary also prints `protocol digest <digest>` in its provenance block, beside the
data and fold digests of the same fit. Both summaries print `causal study protocol: absent` when the
study carried no record.

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
print(effect.functional)
print(effect.identification.assumptions)
print(effect.identification.required_nuisances)
print(effect.available_methods())
```

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

## 6. Assess estimation quality

Inspect support, nuisance performance, targeting scores, and sensitivity in the context of the
identified estimand. A solved score is evidence about the targeting step, not proof that nuisance
models are consistent or that identification assumptions hold.

```python
print(result.diagnostics.support().summary())
print(result.diagnostics.nuisance_models().summary())
print(result.diagnostics.score_equations().summary())
print(result.sensitivity.run_all().summary())
```

## 7. Report the result and its limits

Report the parameter definition, population, identification assumptions, learner and cross-fitting
configuration, estimate and interval, support diagnostics, sensitivity analysis, and package
version or commit. Save structured results when an audit trail matters:

```python
result.save("analysis.joblib")
```

The joblib artifact stores the complete fitted result, including estimator configuration and
nuisance-model objects. An artifact written before `StudyProtocol` still loads, and its summary
states `causal study protocol: absent`. See
[persistence and replayability](user-guide/results-assessment.md#persistence-and-replayability)
for the trust boundary, the version rules, and the replay each restored artifact supports.
