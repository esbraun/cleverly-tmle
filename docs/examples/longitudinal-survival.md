# Longitudinal TMLE for time-to-event outcomes

A patient who leaves the plan is not a patient with a missing outcome. This page makes plan exit
the outcome, then splits readmission and death as competing causes. It reuses the design vocabulary
of [longitudinal TMLE](longitudinal-tmle.md). The
[technical entry](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) gives
the event-process recursion.

## The applied question and the data

Does repeated navigation keep patients enrolled? Time zero is hospital discharge, and each period
is 30 days. The program offers navigation at the start of each period, so the second decision is on
day 31 rather than day seven. The event is plan exit for any reason, including death.

```python
from cleverly import CausalStudy, LongitudinalTreatment, RegimeMean
from cleverly.datasets import make_longitudinal_survival

exit_frame, exit_truth = make_longitudinal_survival(n=4_000, seed=52, cluster_size=20)
exit_frame = exit_frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_p1",
        "C1": "tracked_p1",
        "Y1": "plan_exit_p1",
        "L2": "identified_needs",
        "A2": "navigation_p2",
        "C2": "tracked_p2",
        "Y2": "plan_exit_p2",
        "id": "navigator_team",
    }
)
```

| column | node | role |
| --- | --- | --- |
| `age`, `baseline_readiness` | before discharge | standardized baseline covariates (mean 0, SD 1) |
| `navigation_p1`, `navigation_p2` | start of each period | the navigation offer |
| `tracked_p1`, `tracked_p2` | during each period | 1 if the plan can observe exit in that period |
| `plan_exit_p1`, `plan_exit_p2` | end of each period | the absorbing event, carried forward after exit |
| `identified_needs` | end of period 1, among patients at risk | raised by navigation, raises day-31 navigation and the exit hazard |
| `navigator_team` | fixed | the cluster |

In this synthetic law, 26% of patients would exit within 30 days and 46% within 60 days under no
navigation. Real plans see lower rates. The high rates make each fit's behavior visible.

## Why this method

After a patient exits, the later navigation and needs nodes do not exist. A plan can therefore
intervene only on patients who are still at risk. Each horizon is its own parameter, and each
node's regression uses the patients at risk entering that node.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| the outcome is an event that can happen at more than one time | one cumulative risk per horizon, each on its own risk set | one regression per node per horizon |
| patients can experience one of two competing events | a cause-specific cumulative incidence, with the competing cause left in the history | cause-specific incidences sum to all-cause risk, not to one |
| you want the retention scale rather than the risk scale | `curve(scale="survival")`, which mirrors each level and its interval | nothing. It is the same fit read the other way |

## Design and identification

The `outcome=` role carries the event. The `censoring=` role carries whether the plan could observe
the patient in that period. A patient the plan lost track of is censored. A patient who left the
plan had the event. One outcome column per time point declares an absorbing event.

```python
exit_study = CausalStudy(
    exit_frame,
    design=LongitudinalTreatment(
        outcome=("plan_exit_p1", "plan_exit_p2"),
        treatment=("navigation_p1", "navigation_p2"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("identified_needs",)),
        censoring=("tracked_p1", "tracked_p2"),
        cluster="navigator_team",
    ),
)
exit_effect = exit_study.identify(RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)))
print(exit_effect.summary())
```

| assumption | what it becomes here |
| --- | --- |
| exchangeability | sequential. At every node, navigation and remaining tracked are independent of the potential outcomes, given the recorded history |
| positivity | sequential, and per horizon. A later horizon passes through every earlier node, so its cumulative product has more factors |
| consistency | each period uses the declared protocol version, and the recorded event is the event under that protocol |
| no interference | one patient's assignments do not change another patient's protocol or outcome |

**Horizons are the fit's own time points, not days.** `horizons=(1, 2)` names the two periods. The
fit refuses a horizon outside `1..T` rather than interpolating it.

## Estimate the retention curve

Every fit on this page uses the same parametric learners, fitted on all rows without cross-fitting.
Parametric GLM learners satisfy the Donsker condition. The in-sample fits therefore keep
influence-curve inference valid under the
[remaining conditions](../technical-reference/cv-tmle.md#what-this-solves).

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
    cross_fitting=CrossFitting(enabled=False),
    runtime=Runtime(random_state=41, n_jobs=1),
)


def difference(result, first, second):
    """Delta-method difference of two estimates from one fit."""
    return result.contrast(lambda psi: psi[0] - psi[1], [first, second], name=f"{first} - {second}")


def show(label, estimate, truth):
    low, high = estimate.ci
    print(f"{label:16s} {estimate.psi:7.4f}  CI=({low:.4f}, {high:.4f})  population {truth:.4f}")


exit_result = exit_effect.estimate(method=sequential)
survival_curve = exit_result.curve(scale="survival")
print(survival_curve[["estimand", "parameter", "regimen", "time", "psi", "ci_lower", "ci_upper"]])

exit_differences = {
    horizon: difference(
        exit_result, f"risk_regimen[always @ t={horizon}]", f"risk_regimen[never @ t={horizon}]"
    )
    for horizon in (1, 2)
}
for horizon, estimate in exit_differences.items():
    show(f"exit t={horizon}", estimate, exit_truth[f"ate_regimen[always vs never @ t={horizon}]"])
```

The generator includes a nonlinear second-period hazard. These compact learners are therefore not a
claim that each nuisance regression is correctly specified.

`curve()` returns one row per regimen per horizon, with a `time` column. On the survival scale a
level row reports $1 - F$, and its interval bounds swap. The `estimand` column names the survival
quantity, and the `parameter` column names the risk estimate behind it.

`difference()` uses the influence curves that the fit already holds. It gives the same estimate and
standard error as a separate `RegimeContrast` fit, without a second backward recursion.

On this draw both risk differences are negative, and the 60-day difference is larger. Under the
always plan, more of the population would remain enrolled at both horizons. The printed population
values have the same pattern.

## Two competing clinical events

A patient can be readmitted, or can die before readmission. Either event ends the patient's time at
risk for the other. The two causes are mutually exclusive and absorbing. A mapping of cause to one
indicator column per time point declares competing risks.

| cause | what it is | can the program move it? |
| --- | --- | --- |
| readmission | the first unplanned readmission after discharge | possibly. Navigation can change access and adherence |
| death | death before a recorded readmission | possibly. Any claim needs a plausible pathway and the same identification review |

```python
from cleverly.datasets import make_longitudinal_competing

event_frame, event_truth = make_longitudinal_competing(n=4_000, seed=53, censoring=False)
event_frame = event_frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_p1",
        "L2": "identified_needs",
        "A2": "navigation_p2",
        "D1": "death_p1",
        "D2": "death_p2",
        "R1": "readmission_p1",
        "R2": "readmission_p2",
    }
)
event_study = CausalStudy(
    event_frame,
    design=LongitudinalTreatment(
        outcome={
            "readmission": ("readmission_p1", "readmission_p2"),
            "death": ("death_p1", "death_p2"),
        },
        treatment=("navigation_p1", "navigation_p2"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("identified_needs",)),
    ),
)
event_effect = event_study.identify(RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)))
print(event_effect.summary())
event_levels = event_effect.estimate(method=sequential)
print(event_levels.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])

generator_cause = {"readmission": "relapse", "death": "death"}
event_differences = {}
for cause, source in generator_cause.items():
    for horizon in (1, 2):
        estimate = difference(
            event_levels,
            f"cif_regimen[always, {cause} @ t={horizon}]",
            f"cif_regimen[never, {cause} @ t={horizon}]",
        )
        event_differences[cause, horizon] = estimate
        truth = event_truth[f"ate_regimen[always vs never, {source} @ t={horizon}]"]
        show(f"{cause} t={horizon}", estimate, truth)

incidence_totals = event_levels.incidence_total()
print(incidence_totals[["regimen", "time", "total", "std_err", "excess"]])
```

This section runs without censoring, so the causes are the only way to leave the risk set. The
generator has no cluster option, so this study declares no navigator teams. In this synthetic law,
60-day death risk is 21% under no navigation and 7% under navigation at both periods.

On this draw the readmission difference is negative at 30 days and shrinks toward zero by 60 days.
The death difference is negative at both horizons and larger at 60 days. The population readmission
difference crosses zero between the two horizons.

Navigation lowers death risk in this law, so more navigated patients remain at risk of readmission.
The readmission difference therefore includes a pathway through death. It is not a negative
control.

`incidence_total()` sums the causes per regimen per horizon. It does not renormalize them, because
each cause solves its own score equation. Event-free survival is one minus that **sum**. A fit that
declares two or more causes therefore refuses `curve(scale="survival")`.

## The failure mode: coding death as censoring

The estimates above are total effects (Young et al., 2020). Each readmission incidence counts every
pathway from navigation, including the pathway through death. A common shortcut codes death as a
censoring event. The fit then runs without a refusal, but it targets a different estimand.

```python
alive_through_p1 = event_frame["death_p1"].eq(0) & event_frame["readmission_p1"].eq(0)
death_as_censoring = event_frame.assign(
    alive_p1=1.0 - event_frame["death_p1"],
    alive_p2=(1.0 - event_frame["death_p2"]).where(alive_through_p1),
    readmission_p1=event_frame["readmission_p1"].where(event_frame["death_p1"].eq(0)),
    readmission_p2=event_frame["readmission_p2"].where(event_frame["death_p2"].eq(0)),
)
eliminated = (
    CausalStudy(
        death_as_censoring,
        design=LongitudinalTreatment(
            outcome=("readmission_p1", "readmission_p2"),
            treatment=("navigation_p1", "navigation_p2"),
            baseline=("age", "baseline_readiness"),
            time_varying=((), ("identified_needs",)),
            censoring=("alive_p1", "alive_p2"),
        ),
    )
    .identify(RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)))
    .estimate(method=sequential)
)
eliminated_t2 = difference(eliminated, "risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]")
print("death as censoring, t=2:", round(eliminated_t2.psi, 4))
print("total effect, t=2:      ", round(event_differences["readmission", 2].psi, 4))
```

| analysis | estimand | what it needs |
| --- | --- | --- |
| death as a competing event | the total effect on the cumulative incidence of readmission | the assumptions in the table above |
| death as censoring | the controlled direct effect: readmission risk if death were eliminated | exchangeability and positivity for death as an intervened node, and a well-defined intervention that prevents death |

On this draw the censored analysis reports a larger reduction in readmission. More patients die
under the never plan. Eliminating death returns them to the risk set, which raises readmission risk
most under that plan. No population value exists for this estimand, because the generator defines
no world without death. [Young et al. (2020)](https://doi.org/10.1002/sim.8471) define both
estimands and their identifying conditions.

## How far to trust this

Assess the retention and competing-risk fits separately, because their risk sets have different
shapes.

```python
exit_assessment = exit_result.assess()
event_assessment = event_levels.assess()
print(exit_assessment.summary())
print(exit_assessment.report("support").to_frame())
print(event_assessment.report("support").to_frame())
```

Every sensitivity operation and refutation is `unavailable` for a longitudinal fit. The competing
support table adds `cause` and `horizon` columns, which name the risk set of each row.
[Longitudinal TMLE](longitudinal-tmle.md#how-far-to-trust-this) explains the support columns and
shows the truncation curve.

These reports describe weights and fitted equations. They do not establish sequential
exchangeability or correct nuisance models. The
[technical entry](../technical-reference/longitudinal-tmle.md#validation-issues-special-to-this-method)
lists the survival and competing-risk evidence and its limits.

## Where to go next

This page reported one parameter per plan per horizon, and then one per cause as well. A program
comparing many navigation plans wants a summary instead. That is
[MSM projections](msm-projections.md), and the same projection works over regimens and horizons.
