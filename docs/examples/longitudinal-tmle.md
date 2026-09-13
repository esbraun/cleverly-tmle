# Longitudinal TMLE: navigation at two decisions

Navigation assigned twice is not one assignment with extra columns. This page shows why a
point-treatment analysis of two decisions fails. The
[technical entry](../technical-reference/longitudinal-tmle.md) gives the sequential regression.

## The applied question

The plan offers navigation at discharge and again on day seven. Before the second decision, the
program records an engagement score from attendance, medication pickup, and portal activity. What
share would report a top-box transition score if every patient received both offers, compared with
neither? Patients lost from outcome tracking are censored.

## Why this method

Engagement is a time-varying confounder affected by prior treatment. No single outcome regression
handles it.

| engagement in this law | what one regression does with it |
| --- | --- |
| discharge navigation raises it, and it raises the top-box probability | adjusting for it blocks the part of the discharge effect that runs through engagement |
| it raises day-seven navigation | leaving it out leaves day-seven navigation confounded |
| an unmeasured cause of engagement and the score would make it a collider | adjusting for it would link discharge navigation to that cause (Hernán and Robins, *What If*, chapter 20). This law has no such cause |

Sequential regression works backward. Each node's regression conditions on the history available
*at that node*, and the recursion averages that history under the plan, which evaluates the
g-formula.

## The data

The generator `make_longitudinal` returns a wide frame with one row per patient. Its columns follow
the node order. `cluster_size` puts patients into navigator teams, and each team shares part of the
engagement noise.

```python
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=8_000, seed=41, cluster_size=20)
frame = frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_discharge",
        "C1": "tracked_day7",
        "L2": "engagement_day7",
        "A2": "navigation_day7",
        "C2": "tracked_day30",
        "Y": "transition_top_box",
        "id": "navigator_team",
    }
)
print(frame.head())
for name, value in truth.items():
    print(f"{name:56s} {value:.4f}")
```

The baseline covariates are standardized (mean 0, SD 1). In this synthetic law, the top-box share is
42% with no navigation and 78% with both offers. A real program would expect a smaller effect.

| column | node | role |
| --- | --- | --- |
| `age`, `baseline_readiness` | before discharge | baseline covariates |
| `navigation_discharge` | discharge | the first navigation assignment |
| `tracked_day7` | after discharge | 1 if the patient remains observable at day seven |
| `engagement_day7` | day seven, before the decision | responds to `navigation_discharge`, and drives `navigation_day7` |
| `navigation_day7` | day seven | the second navigation assignment |
| `tracked_day30` | after day seven | 1 if the transition outcome remains observable |
| `transition_top_box` | day 30 | the survey outcome |
| `navigator_team` | fixed | the cluster |

Nodes after a patient is lost from tracking are missing. The estimator expects that shape. A
complete-case frame would discard those patients before any model sees them.

## Design and identification

The design places every column at its node. `time_varying` has one entry per treatment node. The
empty first entry says no time-varying covariate precedes discharge.

```python
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast, StudyProtocol

protocol = StudyProtocol(
    target_population=(
        "Adults discharged home from a participating hospital during the enrollment period"
    ),
    eligibility=(
        "Age 18 years or older",
        "Discharged alive",
        "Discharged home from a participating hospital",
    ),
    time_zero="Hospital discharge, after baseline measurement and before first assignment",
    treatment_strategies=(
        "Offer navigation at discharge and day seven",
        "Offer no navigation at either decision",
    ),
    treatment_versions=(
        "The declared discharge and day-seven navigation contacts",
        "Usual discharge support without navigation contacts",
    ),
    outcome="Top-box patient-reported transition score",
    horizon="30 days after discharge",
    intercurrent_event_handling=(
        "Use the transition score regardless of readmission",
        "The protocol scores death before day 30 as not top box (composite strategy)",
        "Analyze each navigation offer regardless of completed contacts",
    ),
    interference_unit="Individual patient",
    assumption_rationale=(
        "Recorded history covers the measured common causes at each decision",
        "Version records support consistency at both navigation decisions",
        "Reserved navigator capacity supports no interference between patients",
    ),
)

study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="transition_top_box",
        treatment=("navigation_discharge", "navigation_day7"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("engagement_day7",)),
        censoring=("tracked_day7", "tracked_day30"),
        cluster="navigator_team",
    ),
    protocol=protocol,
)
plan = RegimeContrast({"always": 1, "never": 0}, reference="never")
effect = study.identify(plan)
print(effect.summary())
```

The identification summary renders the stored protocol. The typed `RegimeContrast` owns the
mathematical comparison between the two plans.

The placement of `engagement_day7` is the scientific decision on this page. It is time-varying at
the second node. The estimator therefore conditions on it when it models day-seven navigation. It
averages over it when it carries the discharge effect backward.

The `censoring=` role models observation at each node. Observation enters the same cumulative
product as navigation.

The assumptions change shape from the point-treatment case.

| assumption | what it becomes here |
| --- | --- |
| exchangeability | sequential. At every node, navigation and remaining tracked are independent of the potential outcomes, given the recorded history |
| positivity | sequential. At every node, each history that the plan can reach has a positive probability of the plan's arm and of remaining tracked |
| consistency | each decision uses the declared protocol version, and later treatment remains defined under maintained follow-up |
| no interference | one patient's assignments do not change another patient's protocol or outcome |

Identification needs each conditional probability to be positive. Estimation divides by their
product. That product can be small when no single factor is small, which is a practical positivity
problem.

## Estimate

Four learner slots correspond to four kinds of nuisance. The pseudo-outcome learner fits the
intermediate regressions. Their targets are continuous even when the final outcome is binary, and
the recursion clips their predictions to the unit interval.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CrossFitting, ModelSpec, Runtime, TMLEMethod

sequential = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=41),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=3),
    runtime=Runtime(random_state=41, n_jobs=1),
)
result = effect.estimate(method=sequential)
print(result.summary())
print("population contrast:", truth["ate_regimen[always vs never]"])

for alias, key in result.parameter_keys.items():
    print(alias, "|", key.value, "vs", key.reference, "| horizon:", key.horizon)
```

The result summary renders the protocol and its fingerprint. The method settings are a separate
record. Every reported parameter carries a structured key. Use the key rather than parsing the
alias, because the analyst chooses each regimen name.

## The failure mode: a point-treatment analysis of the same data

Now do it the wrong way, twice. Keep patients observed through day 30, keep those whose two
assignments agreed, and treat "received navigation at both decisions" as one exposure.

```python
observed = frame[(frame["tracked_day7"] == 1) & (frame["tracked_day30"] == 1)]
consistent = observed[observed["navigation_discharge"] == observed["navigation_day7"]].copy()
consistent = consistent.rename(columns={"navigation_discharge": "navigation_throughout"})
print("patients kept:", len(consistent), "of", len(frame))
```

That subsetting already loses information. It discards patients with missing follow-up rather than
modeling them. It discards patients whose assignments differed, because a point treatment has no
place for them.

```python
from cleverly import ATE, PointTreatment

target = truth["ate_regimen[always vs never]"]
point_method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=41),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=3),
    runtime=Runtime(random_state=41, n_jobs=1),
)


def report(label, point):
    low, high = point.ci
    print(f"{label:26s} psi={point.psi:6.3f}  CI=({low:.3f}, {high:.3f})")
    return point


def naive(adjustment, label):
    naive_study = CausalStudy(
        consistent,
        design=PointTreatment(
            outcome="transition_top_box",
            treatment="navigation_throughout",
            adjustment=adjustment,
            cluster="navigator_team",
        ),
    )
    point = naive_study.identify(ATE(reference=0)).estimate(method=point_method)["ate"]
    return report(label, point)


adjusted = naive(("age", "baseline_readiness", "engagement_day7"), "adjusting for engagement")
baseline_only = naive(("age", "baseline_readiness"), "baseline only")
report("sequential regression", result["ate_regimen[always vs never]"])
print("population contrast:", target)
```

On this draw the two shortcuts miss the population value in opposite directions. One draw shows the
structure. It is not a coverage claim.

| analysis | result on this draw | structural problem |
| --- | --- | --- |
| adjusting for engagement | more than 0.1 below the population value | the regression conditions on a post-discharge variable and blocks the path through engagement |
| baseline only | about 0.05 above the population value | the regression omits a cause of day-seven navigation |
| sequential regression | within one standard error of the population value | each node conditions on its own history, and the recursion averages that history under the plan |

The shortcut also conditions on agreement between the two assignments. Agreement depends on the
time-varying history, so the selected sample is not the target population. The two point fits
therefore do not isolate a pure mediator bias from a pure confounding bias.

## A rule instead of a plan

A dynamic rule reads the history available at its node. This rule assigns navigation at discharge.
It then continues day-seven navigation only for patients with a positive engagement score.

A plan has one entry per node. An entry is an arm for everybody, or a callable that receives that
node's history frame.

```python
from cleverly.datasets import RULE_LABEL

rule_result = study.identify(
    RegimeContrast(
        {
            "never": 0,
            "continue if engaged": (
                1,
                lambda history: (history["engagement_day7"] > 0).astype(float),
            ),
        },
        reference="never",
    )
).estimate(method=sequential)
print(rule_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print(rule_result.diagnostics.support().to_frame()[["regimen", "time", "share_assigned_1"]])
print("population contrast:", truth[f"ate_regimen[{RULE_LABEL} vs never]"])
```

The `share_assigned_1` column gives the share of followers that the rule treats at each node. A
program with scarce navigator time can consider this policy. No static plan expresses it, and two
rules are two parameters.

## How far to trust this

Start with the combined assessment. This call adds a truncation curve, which refits the recursion at
each cumulative bound.

```python
assessment = result.assess(
    include_refits=True,
    arguments={"truncation_curve": {"bounds": [0.01, 0.05, 0.1]}},
)
print(assessment.summary())

support = assessment.report("support").to_frame()
curve = assessment.report("truncation_curve")
print(support[["regimen", "time", "n_followed", "max_weight", "effective_n", "share_truncated"]])
print(curve[["lower_bound", "psi", "delta_from_fitted", "truncated_score_cells"]])
```

Every sensitivity operation is `unavailable`, because no longitudinal sensitivity derivation is
registered. Refutation is also `unavailable`, and corrections are `not_applicable`.

The support table shows cumulative positivity.

| column | what it says |
| --- | --- |
| `n_followed` | how many patients followed the plan and stayed tracked through that node |
| `max_weight` | the largest cumulative clever covariate among those patients |
| `effective_n` | the Kish effective sample size of those weights |
| `share_truncated` | the share of scored rows whose cumulative probability the bound replaced |

The fit uses the default cumulative bound `(0.01, 1)`, which caps each weight at 100. On this draw
the bound replaces no row. A lower bound of 0.1 truncates some rows. It moves the estimate by less
than 0.005, a small fraction of its standard error. The curve is descriptive and carries no
interval. The [truncation stability](../technical-reference/validation-methods.md#truncation-stability)
section defines its columns.

| layer | establishes | does not establish |
| --- | --- | --- |
| the support report and the truncation curve | how many patients followed each plan, how heavy the weights are, and how far a bound moves the estimate | that sequential exchangeability holds at every node |
| the score-equation report | each fold solved its equation, and the stitched residual is compatible with sampling | that the node regressions are correctly specified |
| the nuisance report | retained loss and calibration for each fitted nuisance role | that any nuisance model is correct, or that causal identification holds |

No registered study covers clustered fits with estimated mechanisms. The
[technical entry](../technical-reference/longitudinal-tmle.md#validation-issues-special-to-this-method)
lists the evidence and its limits.

[Collaborative TMLE](collaborative-tmle.md) and [DR-TMLE](dr-tmle.md) both refuse a longitudinal
design, and `available_methods()` says so before any model is fitted.

## Where to go next

This page reported one parameter per plan, with the transition score as the outcome and loss to
tracking as a nuisance. [Time-to-event outcomes](longitudinal-survival.md) makes an event the
outcome. It reports a cumulative risk per horizon and then splits the risk by cause.
