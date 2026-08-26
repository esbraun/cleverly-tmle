# Longitudinal TMLE for members who leave: retention and competing risks

A member who leaves the plan is not a member with a missing outcome. This test of change makes
leaving the question rather than the nuisance, and then splits it by cause. One cause the program
can move. The other it cannot, and asking the fit to remove it is refused by name.

Read [longitudinal TMLE](longitudinal-tmle.md) first. It declares the same program at two decision
points, and this page reuses its design vocabulary. Read
[Longitudinal TMLE](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) for
the event-process recursion and the cause-specific construction.

## The applied question

The plan wants to know whether repeated transition navigation keeps members enrolled. Members leave
for two reasons. Some choose another plan in the annual open-enrollment window. Others lose
eligibility because their employer group terminated.

Only the first reason is something the navigation program can affect. The program still needs a
number for both, because a cumulative incidence that pools them answers neither question.

## Why this method

An absorbing event needs the risk set to shrink as members leave it. A single end-of-study
regression cannot do that, because a member who left in the first period is not available to have
the event again in the second.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| the outcome is an event that can happen at more than one time | one cumulative risk per horizon, each on its own risk set | one regression per node per horizon, so the fit is longer than an end-of-study one |
| members can leave for reasons you cannot intervene on | a cause-specific incidence that leaves the competing cause in the history | the causes do not sum to one, and renormalising them would break the score equation |
| you want the retention scale rather than the risk scale | `curve(scale="survival")`, which mirrors the interval correctly | nothing. It is the same fit read the other way |

## The shared configuration

Every fit below uses the same parametric method, so the estimand is the only thing that changes.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, CrossFitting, LongitudinalTreatment
from cleverly import ModelSpec, RegimeContrast, Runtime, TMLEMethod

sequential = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=3, learner_folds=2),
    runtime=Runtime(random_state=41, n_jobs=1),
)
```

## Churn as the outcome: a retention curve

In [longitudinal TMLE](longitudinal-tmle.md), loss from outcome tracking was a nuisance. Plan
exit is different. Here it is the question: does repeated transition navigation keep members in the
plan?

This page uses two declared plan periods rather than the discharge and day-seven clock. Time zero
precedes the first period. The treatment at each node is an offer of the same navigation
protocol, and the event is plan exit by the end of that period.

The declaration changes shape rather than gaining a keyword. Passing **one outcome column per time
point** says the outcome is an absorbing event, and the fit reports cumulative risk at each horizon.

```python
from cleverly import RegimeMean
from cleverly.datasets import make_longitudinal_survival

churn_frame, churn_truth = make_longitudinal_survival(n=4_000, seed=52, cluster_size=20)
churn_frame = churn_frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_p1",
        "C1": "tracked_p2",
        "Y1": "disenrolled_p1",
        "L2": "unresolved_transition_issues",
        "A2": "navigation_p2",
        "C2": "tracked_close",
        "Y2": "disenrolled_p2",
        "id": "navigator_team",
    }
)
churn_study = CausalStudy(
    churn_frame,
    design=LongitudinalTreatment(
        outcome=("disenrolled_p1", "disenrolled_p2"),
        treatment=("navigation_p1", "navigation_p2"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("unresolved_transition_issues",)),
        censoring=("tracked_p2", "tracked_close"),
        cluster="navigator_team",
    ),
)
churn_result = churn_study.identify(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2))
).estimate(method=sequential)
print(churn_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

Two roles that look alike are doing different jobs here. `disenrolled_*` is the **event**, and
`tracked_*` is the **censoring**: whether the plan could observe the member's status in that period
at all. A member the plan lost track of is censored. A member who left the plan had the event.

Read the same fit on the retention scale.

```python
print(churn_result.curve(scale="survival"))
```

`curve()` returns one row per regimen per horizon, with a `time` column that `to_frame()` does not
carry. The survival scale is not a relabelling. For a level it reports $1 - F$, mirroring the
estimate and the interval. For a contrast it negates the estimate, and it negates **and swaps** the
interval bounds. The standard error is the same either way.

At the documented sample size the retention curve separates. Members assigned navigation in both
periods stay enrolled at a visibly higher rate by the second period.

**Horizons are the fit's own time points, not days.** `horizons=(1, 2)` names the two declared plan
periods. Asking for a horizon outside `1..T` is refused rather than interpolated.

```python
churn_contrast = churn_study.identify(
    RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
).estimate(method=sequential)
print(churn_contrast.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
for name, value in churn_truth.items():
    if name.startswith("ate_"):
        print(f"{name:52s} {value:.4f}")
```

The contrast is negative, because navigation reduces cumulative disenrollment. Its size grows between
the two horizons.

## Two ways to leave, and one of them the program cannot touch

A member leaves through one of two events, and the event type decides what the program can
claim.

| cause | what it is | can the program move it? |
| --- | --- | --- |
| open enrollment | the member chose another plan in the annual window | yes. This is the exit that patient experience is supposed to affect |
| other | the employer group terminated, or a qualifying life event ended eligibility | no. It is administrative, and it removes the member before they could ever choose |

The two are mutually exclusive and absorbing, which is what the container requires. Competing risks
are declared by the **shape** of `outcome=`: a mapping of cause to its indicator columns, one per
time point.

```python
from cleverly.datasets import make_longitudinal_competing

exit_frame, exit_truth = make_longitudinal_competing(n=4_000, seed=53, censoring=False)
exit_frame = exit_frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_p1",
        "L2": "unresolved_transition_issues",
        "A2": "navigation_p2",
        "D1": "open_enrollment_exit_p1",
        "D2": "open_enrollment_exit_p2",
        "R1": "other_exit_p1",
        "R2": "other_exit_p2",
    }
)
exit_study = CausalStudy(
    exit_frame,
    design=LongitudinalTreatment(
        outcome={
            "open enrollment": ("open_enrollment_exit_p1", "open_enrollment_exit_p2"),
            "other": ("other_exit_p1", "other_exit_p2"),
        },
        treatment=("navigation_p1", "navigation_p2"),
        baseline=("age", "baseline_readiness"),
        time_varying=((), ("unresolved_transition_issues",)),
    ),
)
exit_result = exit_study.identify(
    RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2))
).estimate(
    method=TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(max_iter=1000),
            pseudo_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        ),
        cross_fitting=CrossFitting(n_folds=2, learner_folds=2),
        runtime=Runtime(random_state=53, n_jobs=1),
    )
)
print(exit_result.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

Two operational notes, because both are the kind of thing that stops a competing-risks fit rather
than biasing it.

The fold count drops to two here. Each cause needs a regression at each node, and the rarer cause is
thin. A fold whose training rows happen to contain no event of one cause leaves its learner with a
single class, and the fit stops. Fewer, larger folds is the fix. The number of **events of the
rarest cause**, not the number of members, is what bounds the split.

This section also runs without censoring, so the causes are the only way to leave the risk set. The
censoring machinery is unchanged from the retention curve above. This generator ships no cluster
variant, so navigator teams are not declared here.

At the documented sample size the two causes behave differently, and the contrast is the point.
Navigation in both periods cuts the cumulative incidence of **open-enrollment** exits at both
horizons, and both intervals exclude zero. For **other** exits the effect is small, and by the
second horizon its interval contains zero.

The administrative exit can serve as a negative-control outcome only under further design
conditions. Navigation must have no path to employer eligibility, the outcome must share relevant
confounding with voluntary exit, and selection must not create a new path. A large estimate would
flag possible bias. A small estimate cannot prove exchangeability.

The contrast fit reports differences. To see the levels the differences are built from, and to add
them up, ask for the means.

```python
exit_method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
    ),
    cross_fitting=CrossFitting(n_folds=2, learner_folds=2),
    runtime=Runtime(random_state=53, n_jobs=1),
)
exit_levels = exit_study.identify(RegimeMean({"always": 1, "never": 0}, horizons=(1, 2))).estimate(
    method=exit_method
)
print(exit_levels.to_frame()[["estimand", "psi", "ci_lower", "ci_upper"]])
print(exit_levels.incidence_total())
```

`incidence_total()` sums the causes per regimen per horizon. It does not renormalise them onto a
simplex, because that would move each cause off the score equation the fit just solved. The `excess`
column is what a renormalisation would have hidden.

## The failure mode: asking the fit to remove a cause the program cannot intervene on

The tempting next question is what the open-enrollment loss would be if nobody's group had
terminated. That is not a setting on this estimand. It is a different estimand, and it is refused by
name.

```python
from cleverly.longitudinal import LTMLE

try:
    LTMLE({"always": 1, "never": 0}, eliminate="other")
except TypeError as error:
    print(error)
```

The refusal explains itself. What this fit reports is the cause-specific cumulative incidence with
the competing causes left alone, so an administrative exit is part of the history and enters the
clever covariate's indicator. Removing it would make it an intervened node, with a further factor
per node in the denominator, and its own exchangeability and positivity assumptions to state.

The refusal is also the scientifically right answer. A group termination is not something a navigation
program intervenes on, so a counterfactual world without them is not a world the program could
create.

## How far to trust this

Both fits above report one row per regimen per horizon, so read their diagnostics the same way.

```python
print(churn_result.diagnostics.run_all().summary())
print(churn_result.diagnostics.stagewise().to_frame())
```

The stagewise table is where cumulative positivity becomes visible on an event process. A horizon
is reached through every node before it, so `effective_n` falls faster here than on an
end-of-study fit. [Longitudinal TMLE](longitudinal-tmle.md#how-far-to-trust-this) reads those three
columns in full.

```python
print(churn_result.diagnostics.score_equations().to_frame())
print(churn_result.validate().summary())
```

| layer | establishes | does not establish |
| --- | --- | --- |
| the stagewise report | how many members were still at risk at each node, and how hard the weights worked | that sequential exchangeability holds at every node |
| the score-equation report | every node's fluctuation converged | that the node regressions are correctly specified |
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
