# When another event can come first

A patient who dies can no longer be readmitted. Death is then a competing event. Declare the causes with a mapping from
cause labels to event columns on `LongitudinalTreatment`, or with `causes=` on `TimeToEvent`,
which maps each nonzero event code to a label. Without `causes=`, codes `0` and `1` declare one
event and any other codes declare competing causes.

The fit reports, for each cause, its cumulative incidence: the share of people who have that
event first by each time, with the other events left as they are. Cause, horizon and plan stay
in the result keys, such as `cif_regimen[arm1, relapse @ t=4]`. A treatment that raises
deaths can lower readmissions in this sense, because fewer people stay alive to be readmitted.
[Readmission when death can come first](../examples/longitudinal-survival.ipynb) works through it,
and [Survival and competing risks](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks)
gives the recursion, which keeps people with an earlier competing event out of later risk sets.
[Readmission or death with exact event times](../examples/exact-event-times.ipynb) does the same
when each event's exact time is recorded.

## Ask what the risk would be had the other event been prevented

Another question asks what the risk of readmission would be in a world where nobody died, under
each plan. Declare it by naming the causes to prevent with `eliminate=` on the design. The fit
then treats a prevented event as censoring. Within one time step it counts censoring first, then
the prevented event, then the remaining causes.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, RegimeContrast, TimeToEvent
from cleverly.datasets import make_point_survival

two_causes, _ = make_point_survival(n=1_000, seed=4, n_times=4, causes=2)
prevented = CausalStudy(
    two_causes,
    design=TimeToEvent(
        time="time",
        event="event",
        treatment="A",
        baseline=("W1", "W2"),
        grid=(1, 2, 3, 4),
        causes={1: "relapse", 2: "death"},
        eliminate=["death"],
    ),
).estimate(
    RegimeContrast({"arm1": 1, "arm0": 0}, reference="arm0", horizons=(4,)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    competing_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
print(prevented.summary())
```

The fit models staying in follow-up and staying free of the prevented event as two separate
steps. `result.diagnostics.support()` then shows whether staying in follow-up or staying free of
death has probabilities near zero, which would make the estimate rest on few people. `competing_learner=` sets the
model of the second step, and it defaults to the censoring learner.

This answer needs two conditions beyond those of the competing-event fit. No unrecorded factor may
drive both the prevented event and the event you study, given the recorded history. And every kind
of person must have had a real chance of staying free of the prevented event under each plan. No
data check can test the first. The question also needs a world where the event could be
prevented, which may not exist for death, and on a coarse grid the answer depends on the order of
the two events within a step.

The fit above prints `experimental`, which means no simulation study has yet checked its
intervals on that kind of fit.

:::{dropdown} Details: where eliminate= works, and its study
With one cause left, `ratio()` and `rmst()` work on the fit.
`eliminate=` works on a `TimeToEvent` design with a grid, `LongitudinalTreatment` and the long
layout `LongPanel`, and on every longitudinal estimator except structural nested means. The
exact-time design `grid="continuous"` refuses it (roadmap
[X6](../roadmap.md#x6-continuous-time-survival-and-competing-risks)), and so do the collaborative and doubly robust
survival fits (the [collaborative TMLE refusals](../technical-reference/scope-and-refusals.md#collaborative-tmle-of-a-risk-curve)
and [doubly robust TMLE on time-to-event outcomes](../technical-reference/dr-tmle/theorem.md#time-to-event-outcomes)).
A [learned plan](../technical-reference/longitudinal-tmle.md#learned-regimens) and a
[separable effect](../technical-reference/scope-and-refusals.md#separable-effects) refuse it too.

The [registered study](../technical-reference/method-evidence/eliminated-competing-events.md)
covers iterated-mean longitudinal TMLE (`LTMLE`) on two binary treatment decisions and hazard
`LTMLE` on a three-level baseline treatment, both at n = 16,000 with saturated models. Those fits
print `supported`, in-sample and with two-fold cross-fitting; every other fit with `eliminate=` prints `experimental`.
[Eliminated competing events](../technical-reference/longitudinal-tmle.md#eliminated-competing-events)
gives the exact quantity from Young et al. (2020) and each check.
:::

## Separate a treatment's effect on two events

A drug might lower relapse through one ingredient and raise the risk of death through another. A separable effect asks what the risk of each event would be if the part acting on
relapse were given while the part acting on death were not. Declare it with `Separable`, beside
an ordinary arm that serves as the reference. Fit it with `survival_model="hazard"`, which models
the hazard (the chance of each event at a time among people still at risk), and
`targeting="one_step"`. TMLE's correction step is what `cleverly` calls targeting; the one-step form corrects the hazards for every
time at once.

```python
from cleverly.longitudinal import Separable

split = Separable(arms={"relapse": 1, "death": 0}, first="death")
learner = LogisticRegression(max_iter=1000)
separable = CausalStudy(
    two_causes,
    design=TimeToEvent(
        time="time",
        event="event",
        treatment="A",
        baseline=("W1", "W2"),
        grid=(1, 2, 3, 4),
        causes={1: "relapse", 2: "death"},
    ),
).estimate(
    RegimeContrast({"arm0": 0, "split": split}, reference="arm0", horizons=(4,)),
    survival_model="hazard",
    targeting="one_step",
    outcome_learner=learner,
    treatment_learner=learner,
    censoring_learner=learner,
    cross_fit=False,
)
print(separable["ate_regimen[split vs arm0, relapse @ t=4]"].ci)
```

The printed interval is for the change in relapse risk by time 4 from giving only the part that
acts on relapse, against giving neither part. `arms=` gives each part its treatment level: the part acting on relapse at 1 (given), the part
acting on death at 0 (not given). `first="death"` says that, within one time step, a death is counted before a relapse. On a coarse
time grid this choice can change the answer, so it is part of the question you ask. The estimate
also needs two conditions no data can check unless the two parts were given apart in a trial:
each part reaches the other event only through the event it acts on, and no unmeasured cause of
both events exists beyond the covariates you adjust for.
[Separable effects](../technical-reference/longitudinal-tmle.md#separable-effects) gives the exact
estimand and what a separable fit refuses. The
[separable effects tutorial](../examples/separable-effects.ipynb) works a full analysis: a
simulated water pill whose fluid-clearing part keeps heart-failure patients out of hospital while
its potassium-lowering part raises the risk of death.
