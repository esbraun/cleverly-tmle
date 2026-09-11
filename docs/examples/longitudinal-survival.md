# Longitudinal TMLE for time-to-event outcomes

A patient who leaves the plan is not a patient with a missing outcome. This page first makes plan exit
the outcome. It then uses relapse and death to show a competing-risk outcome with two causes.

Read [longitudinal TMLE](longitudinal-tmle.md) first. It declares the same program at two decision
points, and this page reuses its design vocabulary. Read
[Longitudinal TMLE](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) for
the event-process recursion and the cause-specific construction.

## The applied question

The plan first asks whether repeated transition navigation keeps patients enrolled. A clinical team
then asks how the same repeated offer changes cumulative relapse and death after discharge.
All-cause incidence answers the overall event-free question, but it cannot separate the two causes.

## Why this method

An absorbing event needs the risk set to shrink as patients leave it. A single end-of-study
regression cannot do that, because a patient who left in the first period is not available to have
the event again in the second.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| the outcome is an event that can happen at more than one time | one cumulative risk per horizon, each on its own risk set | one regression per node per horizon, so the fit is longer than an end-of-study one |
| patients can experience one of two competing events | a cause-specific incidence that leaves the competing cause in the history | cause-specific incidences sum to all-cause risk, not one. Renormalising them would break the score equation |
| you want the retention scale rather than the risk scale | `curve(scale="survival")`, which mirrors the interval correctly | nothing. It is the same fit read the other way |

## Design and identification

Two roles look alike on an event process and do different jobs. The `outcome=` role carries the
event. The `censoring=` role carries whether the plan could observe the patient in that period. A
patient the plan lost track of is censored. A patient who left the plan had the event.

The assumptions keep the sequential shape of
[longitudinal TMLE](longitudinal-tmle.md#design-and-identification). Positivity now has one
statement per horizon.

| assumption | what it becomes here |
| --- | --- |
| exchangeability | sequential. It must hold at every node, given the recorded history at that node |
| positivity | cumulative, and now per horizon. A later horizon is reached through every node before it, so its product carries more factors |
| consistency | each period uses the declared protocol version, and the recorded event is the event under that protocol |
| no interference | one patient's assignments do not change another patient's protocol or outcome |

Each fit prints its own assumptions. `summary()` lists them under `assumptions:`, and
`identification.assumptions` holds the same entries.

## The shared configuration

The fits use the same parametric learner families where their nuisance roles match. The
competing-risk fit later uses fewer folds because one cause has few events.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, CrossFitting, LongitudinalTreatment
from cleverly import ModelSpec, RegimeContrast, Runtime, TMLEMethod

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
```

These compact learners keep the tutorial quick. The generator includes a nonlinear second-period
hazard, so this configuration is not a claim that each nuisance regression is correctly specified.

## Plan exit as the outcome: a retention curve

In [longitudinal TMLE](longitudinal-tmle.md), loss from outcome tracking was a nuisance. Plan
exit is different. Here it is the question: does repeated transition navigation keep patients in the
plan?

This page uses two declared plan periods rather than the discharge and day-seven clock. Time zero
precedes the first period. The treatment at each node is an offer of the same navigation
protocol, and the event is plan exit by the end of that period.

The declaration changes shape rather than gaining a keyword. Passing **one outcome column per time
point** says the outcome is an absorbing event, and the fit reports cumulative risk at each horizon.

```python
from cleverly import RegimeMean
from cleverly.datasets import make_longitudinal_survival

exit_frame, exit_truth = make_longitudinal_survival(n=4_000, seed=52, cluster_size=20)
exit_frame = exit_frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_p1",
        "C1": "tracked_p2",
        "Y1": "plan_exit_p1",
        "L2": "unresolved_transition_issues",
        "A2": "navigation_p2",
        "C2": "tracked_close",
        "Y2": "plan_exit_p2",
        "id": "navigator_team",
    }
)
exit_study = CausalStudy(
    exit_frame,
    design=LongitudinalTreatment(
        outcome=("plan_exit_p1", "plan_exit_p2"),
        treatment=("navigation_p1", "navigation_p2"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("unresolved_transition_issues",)),
        censoring=("tracked_p2", "tracked_close"),
        cluster="navigator_team",
    ),
)
exit_effect = exit_study.identify(RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)))
print(exit_effect.summary())
exit_result = exit_effect.estimate(method=sequential)
print(exit_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

In this frame `plan_exit_*` fills the **event** role, and `tracked_*` fills the **censoring** role.

Read the same fit on the retention scale.

```python
survival_curve = exit_result.curve(scale="survival")
print(survival_curve[["estimand", "parameter", "scale", "view"]])
print(survival_curve[["regimen", "time", "psi", "ci_lower", "ci_upper"]])
```

`curve()` returns one row per regimen per horizon, with a `time` column that `to_frame()` does not
carry. The two prints show the same rows in the same order. The survival scale is not a
relabelling. For a level it reports $1 - F$, mirroring the estimate and the interval. For a
contrast it negates the estimate, and it negates **and swaps** the interval bounds. The standard
error is the same either way.

Four columns name each row, and each one answers a different question.

| column | what it records | on this frame |
| --- | --- | --- |
| `estimand` | the name of the quantity the row reports | `survival_regimen[always @ t=1]` |
| `parameter` | the key of the estimate the row comes from | `risk_regimen[always @ t=1]` |
| `scale` | `level` or `difference`, the vocabulary `to_frame()` also uses | `level` |
| `view` | the view you asked for | `survival` |

The level rows show why the frame carries two names. This fit never estimated a parameter called
`survival_regimen[always @ t=1]`, so `exit_result[row.estimand]` raises a `KeyError`. It did
estimate `risk_regimen[always @ t=1]`, so `exit_result[row.parameter]` returns that estimate. A
contrast row keeps its generic `ate_regimen[...]` name under both views. A risk difference and a
survival difference are the same parameter up to a sign. On a contrast row the two columns
therefore hold the same key.

At the documented sample size the retention curve separates. Patients assigned navigation in both
periods stay enrolled at a higher rate by the second period.

**Horizons are the fit's own time points, not days.** `horizons=(1, 2)` names the two declared plan
periods. Asking for a horizon outside `1..T` is refused rather than interpolated.

```python
exit_contrast = exit_study.identify(
    RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
).estimate(method=sequential)
print(exit_contrast.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
for name, value in exit_truth.items():
    if name.startswith("ate_"):
        print(f"{name:52s} {value:.4f}")
```

The contrast is negative, because navigation reduces the cumulative risk of plan exit. Its size
grows between the two horizons.

## Two competing clinical events

A patient can experience relapse or die before relapse. Either event removes the patient from the
risk set for the other.

| cause | what it is | can the program move it? |
| --- | --- | --- |
| relapse | the first recorded recurrence after discharge | possibly. Navigation can change access and adherence pathways |
| death | death before a recorded relapse | possibly. Any claim needs a plausible pathway and the same identification review |

The two are mutually exclusive and absorbing, which is what the container requires. Competing risks
are declared by the **shape** of `outcome=`: a mapping of cause to its indicator columns, one per
time point.

```python
from cleverly.datasets import make_longitudinal_competing

event_frame, event_truth = make_longitudinal_competing(n=4_000, seed=53, censoring=False)
event_frame = event_frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_p1",
        "L2": "unresolved_transition_issues",
        "A2": "navigation_p2",
        "D1": "death_p1",
        "D2": "death_p2",
        "R1": "relapse_p1",
        "R2": "relapse_p2",
    }
)
event_study = CausalStudy(
    event_frame,
    design=LongitudinalTreatment(
        outcome={
            "relapse": ("relapse_p1", "relapse_p2"),
            "death": ("death_p1", "death_p2"),
        },
        treatment=("navigation_p1", "navigation_p2"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("unresolved_transition_issues",)),
    ),
)
event_result = event_study.identify(
    RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
).estimate(
    method=TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=1000, random_state=53),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        ),
        cross_fitting=CrossFitting(n_folds=2),
        runtime=Runtime(random_state=53, n_jobs=1),
    )
)
print(event_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
for cause in ("relapse", "death"):
    for horizon in (1, 2):
        key = f"ate_regimen[always vs never, {cause} @ t={horizon}]"
        print(f"population {cause} contrast at t={horizon}: {event_truth[key]:.4f}")
```

Two operational notes, because both are the kind of thing that stops a competing-risks fit rather
than biasing it.

The fold count drops to two here. Each cause needs a regression at each node, and the rarer cause is
thin. A fold whose training rows happen to contain no event of one cause leaves its learner with a
single class, and the fit stops. Fewer, larger folds is the fix. The number of **events of the
rarest cause**, not the number of patients, is what bounds the split.

This section also runs without censoring, so the causes are the only way to leave the risk set. The
censoring machinery is unchanged from the retention curve above. This generator ships no cluster
variant, so navigator teams are not declared here.

At the documented sample size the two causes behave differently. The estimated relapse contrast
shrinks toward zero by the second horizon, and its interval covers zero there. The population
relapse contrast crosses zero between the two horizons. The estimated death contrast is negative at
both horizons, and it grows.

The generator makes treatment affect both the all-cause hazard and the split between causes. The
relapse result is therefore not a negative control. Its small second-horizon contrast is a feature
of this law, not evidence that navigation has no path to that cause.

The contrast fit reports differences. To see the levels the differences are built from, and to add
them up, ask for the means.

```python
event_method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=53),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=2),
    runtime=Runtime(random_state=53, n_jobs=1),
)
event_levels = event_study.identify(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2))
).estimate(method=event_method)
print(event_levels.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
incidence_totals = event_levels.incidence_total()
print(incidence_totals[["regimen", "time", "total", "std_err", "excess"]])
```

`incidence_total()` sums the causes per regimen per horizon. It does not renormalise them onto a
simplex, because that would move each cause off the score equation the fit just solved. The `excess`
column is what a renormalisation would have hidden.
The standard error uses the covariance of the summed cause-specific influence curves. The
calculation preserves the fit's cluster structure.

Event-free survival is the complement of the **sum** of the cause-specific incidences. The
complement of one cause-specific incidence is a different quantity. A fit that declares two or more
causes therefore refuses `curve(scale="survival")`, and this fit declares two. Use
`incidence_total()` to inspect the sum across causes. A fit that declares one cause reports the
survival view, because there the sum is that one incidence.

## The failure mode: asking the fit to remove a competing cause

The tempting next question is what relapse risk would be if death were prevented. That is not a
setting on this estimand. It is a different estimand, and it is refused by name.

```python
from cleverly.longitudinal import LTMLE

try:
    LTMLE({"always": 1, "never": 0}, eliminate="death")
except TypeError as error:
    print(error)
```

The refusal explains itself. This fit reports cause-specific cumulative incidence with the competing
causes left alone. Removing a cause would make it an intervened node, with a further factor per node
in the denominator and its own exchangeability and positivity assumptions.

The refusal is also the scientifically cautious answer. Eliminating a competing event needs a new
causal intervention and its own identification argument. This fit does not create either one.

## How far to trust this

Inspect the retention and competing-risk fits separately. Their risk sets have different shapes.
This page calls `.diagnostics.run_all()` rather than `.assess()`, because every sensitivity
operation that `.assess()` adds is unavailable for a longitudinal fit.

```python
exit_diagnostics = exit_result.diagnostics.run_all()
event_diagnostics = event_result.diagnostics.run_all()

print("Retention diagnostics")
print(exit_diagnostics.summary())
print("Competing-risk diagnostics")
print(event_diagnostics.summary())

print(exit_diagnostics.report("support").to_frame())
print(event_diagnostics.report("support").to_frame())
print(exit_diagnostics.report("score_equations").to_frame())
print(exit_diagnostics.report("nuisance_models").to_frame())
```

The support table is where cumulative positivity becomes visible on an event process. A horizon
is reached through every node before it, so `effective_n` falls faster here than on an
end-of-study fit. [Longitudinal TMLE](longitudinal-tmle.md#how-far-to-trust-this) reads those three
columns in full.

The direct `stagewise()` method remains a compatibility alias for the same support report. The
diagnostic report keeps only the `support` name.

The retention report has six stage rows. The competing-risk report has twelve because it also
separates causes. Its `cause` and `horizon` columns show which risk set each row describes.

Cross-fitting gives each fitted stage a `solver` row and a `stitching` row. The solver row checks
the equations fitted outside each reporting fold. The stitching row checks the pooled residual
against its sampling scale. The aggregate verifies all competing-risk rows without printing a
second large score table.

The nuisance table covers every fitted role. Treatment and censoring models appear once per node.
Outcome and pseudo-outcome rows also identify their regimen, cause, horizon, and node.

Treatment and censoring rows report weighted negative log likelihood. An outcome row reports
weighted Brier loss for a binary target, or mean squared error otherwise. A pseudo-outcome row
reports weighted mean squared error. The `evaluation` column is `out_of_fold` for these fits.

Each diagnostic report defers its truncation curve until the caller supplies bounds and permits
refits. Refutations remain unavailable. A `completed` nuisance row means the retained losses
exist, not that the models are correct.

| layer | establishes | does not establish |
| --- | --- | --- |
| the support report | how many patients were still at risk at each node, and how hard the weights worked | that sequential exchangeability holds at every node |
| the score-equation report | each fold solved its equation, and the stitched residual is compatible with sampling | that the node regressions are correctly specified |
| the nuisance report | retained loss and calibration for each fitted nuisance role | that any nuisance model is correct, or that causal identification holds |
| the registered event-process studies | ordinary and cross-fitted fits recover known two-horizon survival and competing-risk truths | MSMs, weights, clustering, eliminated competing events, or simultaneous bands |

The survival curve rests on two registered rows, the
[ordinary](../technical-reference/method-evidence/ordinary-survival-curve-longitudinal-tmle.md) and
the
[cross-fitted](../technical-reference/method-evidence/cross-fitted-survival-curve-longitudinal-tmle.md)
study. Both are pointwise, two-horizon, one-cause rows.

The competing-risk fit has separate
[ordinary](../technical-reference/method-evidence/ordinary-competing-risk-longitudinal-tmle.md) and
[cross-fitted](../technical-reference/method-evidence/cross-fitted-competing-risk-longitudinal-tmle.md)
rows. Each row covers two causes, two horizons, censoring, and static and dynamic plans. The rows
also test targeting and the all-cause risk-set recursion with nonzero controls.

These rows do not cover simultaneous bands, learned mechanisms, active truncation, weights,
clustering, or elimination of a competing event. Read those compositions as unsupported by this
study rather than as inherited from a nearby row.

## Where to go next

This page reported one parameter per plan per horizon, and then one per cause as well. A program
comparing many navigation plans wants a summary instead. That is
[MSM projections](msm-projections.md), and the same projection works over regimens and horizons.
