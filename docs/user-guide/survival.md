# Time-to-event outcomes

Use this page when the outcome is whether and when an event happens, such as readmission or
relapse. `cleverly` estimates the risk of the event by each time you name, under each plan you
compare: for example, the share of
patients readmitted by day 30, 60 and 90 had everyone been offered a program, and had no one. It
also reports differences, ratios, and the average event-free time up to a cut-off. The
estimators are targeted maximum likelihood estimation (TMLE) for data over time: they fit
prediction models, then correct the estimate so its confidence interval is valid when those models are good enough.

`TimeToEvent` takes one row per person with a follow-up time and an
event code, and a treatment given once at the start. `LongitudinalTreatment` takes the wide
layout of [Longitudinal treatment](longitudinal.md), with one event column per time point, and
also fits treatments that change over time. A person who leaves follow-up is censored: their
event is unseen from then on.

The answer is causal only under the four conditions of
[What must hold for a causal answer](../examples/index.md#what-must-hold-for-a-causal-answer).
One of them, that no unrecorded factor drove both the treatment and the event, no data check can
test. Censoring adds two more: who leaves follow-up depends only on recorded information, and
every kind of person had a real chance of staying in follow-up.

## One treatment at the start and a time to an event

Use `TimeToEvent` when each person has one follow-up time, one event code and one treatment given
at baseline, the start of follow-up. Code `0` means censored. You choose a grid of check times, such as
days 30, 60 and 90, and the design holds the treatment over every check.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import CausalStudy, RegimeContrast, TimeToEvent
from cleverly.datasets import make_point_survival

frame, truth = make_point_survival(n=1_000, seed=4, n_times=4)
# For your own data, change the column names, grid=, the plans and horizons=.
# baseline= lists the confounders: covariates measured before treatment that drove
# both who got it and the event. Each plan value is a value of the treatment column.
design = TimeToEvent(
    time="time", event="event", treatment="A", baseline=("W1", "W2"), grid=(1, 2, 3, 4)
)
curve = CausalStudy(frame, design=design).estimate(
    RegimeContrast({"arm1": 1, "arm0": 0}, reference="arm0", horizons=(2, 4)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
print(curve["ate_regimen[arm1 vs arm0 @ t=4]"].ci, truth["ate_regimen[arm1 vs arm0 @ t=4]"])
```

`ate_regimen[arm1 vs arm0 @ t=4]` is the difference in the risk of the event by grid time 4
between treating everyone (`arm1`) and treating no one (`arm0`), and the printed pair is its 95%
confidence interval. Only simulated data come with the true value. `cleverly` calls each plan a regimen, and
`horizons=` names the times at which to report.

`outcome_learner` predicts the event. The method then
works backward in time, regressing each step's predictions on what was known one step earlier,
and `pseudo_learner` is that regression model, with a continuous target. `treatment_learner`
predicts who got the treatment. `censoring_learner`, left out here, predicts who stays in
follow-up; when you leave it out, `cleverly` picks a default model, and the summary names it.
Plain models keep the example fast; on real data use flexible learners, such as gradient
boosting, or leave the learners out to use the slower built-in library, and
keep the default `cross_fit=True`, which fits each prediction on rows other than its own.

An event counts at the first grid time at or after it, so a readmission on day 45 counts by day
60. A censoring time must sit on a grid time or after the last one. The fit refuses a censoring
between two grid times, because it cannot be ordered against that interval's events. Build the grid from the times at
which follow-up was checked, such as scheduled calls; a finer grid costs more time.

:::{dropdown} Details: all the input rules
| rule | reason |
| --- | --- |
| Declare `grid=` at the check times. Omit it only for integer times, and the grid is then `1, 2, ...` up to the largest event time | a grid read off the sample would change with the sample |
| Record a censoring time as a grid time, or as a time after the last grid time | a censoring between two grid times cannot be ordered against that interval's events, so the fit refuses it |
| Give `horizons=` and the `tau` of `rmst` as grid times | the parameter name keeps the position on the grid, and `curve()` and `to_frame()` report the grid time in a `time` column |
| Expect $K(K+1)/2$ regressions per regimen and cause for $K$ grid times | a default grid on day-level times can hold thousands of time points |
| Expect a risk of zero with no interval at a grid time before any person in an arm had the event | with no event there is nothing to vary, so the fit reports no interval and warns |
:::

An event after the last grid time leaves the person event-free at the end of the grid. For a
continuous dose, declare `continuous_treatment=True` and give each plan as a change to each
person's dose, such as `Shift(0.5, cap=None)`, which adds 0.5.

:::{dropdown} Details: one baseline treatment on the wide layout
On the wide layout, declare the same design with one column name: `treatment="A"` on
`LongitudinalTreatment`. The number of time points then comes from the outcome columns, else
from `censoring=`, else from `time_varying=`. The wide layout also takes a covariate measured
after baseline, and an end-of-study outcome after several censoring time points. A
regimen is one arm, a known rule, a known stochastic policy, or a modified treatment policy, all
at baseline. A plan that changes the treatment after baseline is refused, because the design
holds one decision.
[One baseline treatment held over the nodes](../technical-reference/longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes)
gives the contract and its evidence.
:::

[Readmission by day 30, 60 and 90](../examples/discharge-to-readmission.ipynb) works a full
analysis with this design.

## Ratios and the restricted mean survival time

A fitted risk curve also gives ratios and the restricted mean survival time (RMST), the average
event-free time up to a cut-off. The fit below uses `RegimeMean`, which reports each plan's risk
at the `horizons=` given; `RegimeContrast` above reports only differences. `ratio` and `rmst`
refit nothing, and `.psi` is the point estimate.

```python
from cleverly import RegimeMean

risks = CausalStudy(frame, design=design).estimate(
    RegimeMean({"arm1": 1, "arm0": 0}, horizons=(1, 2, 3, 4)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
risk_ratio = risks.ratio("risk_regimen[arm1 @ t=4]", "risk_regimen[arm0 @ t=4]")
survival_ratio = risks.ratio(
    "risk_regimen[arm1 @ t=4]", "risk_regimen[arm0 @ t=4]", view="survival"
)
rmst = risks.rmst("arm1", 4, versus="arm0")
print(risk_ratio.ci, survival_ratio.psi, rmst.psi, rmst.ci)
```

| call | what it reports |
| --- | --- |
| `ratio(a, b)` | the risk ratio of two plans at one time, with its interval on the log scale |
| `ratio(a, b, view="survival")` | the ratio of the two survival probabilities |
| `ratio(a, b, kind="or")` | the odds ratio of the two plans |
| `rmst(d, tau)` | the RMST of plan `d` up to time `tau` |
| `rmst(d, tau, versus=b)` | the difference of two RMSTs |
| `rmtl(d, tau, cause)` | the restricted mean time lost to one cause |

:::{dropdown} Details: what `rmst` needs and its units
`rmst` needs the risk at every time from the first to the one before `tau`, since for unit-spaced times it is
$\tau - \sum_{t<\tau} F_d(t)$ with $F_d(t)$ the risk by `t` under plan `d`. On a `TimeToEvent`
grid with other spacing, each risk is weighted by the gap to the next grid time. On a `TimeToEvent`
grid the times are grid times. On the wide layout they count time points, so for points spaced
$\Delta$ apart, multiply by $\Delta$ to get calendar time.
:::

## Exact event times, without a grid

When you know the exact time of each event and of each censoring, `grid="continuous"` skips the
check-time grid: no event is moved to a later check time. It handles a two-level treatment given
once at the start, and one event type or several competing ones, and reports each risk at the
times in `horizons=`.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 1_000
x = rng.normal(size=n)
treated = rng.binomial(1, 1 / (1 + np.exp(-x)))
relapse_rate = np.exp(-1.5 + 0.5 * x - treated)
death_rate = np.exp(-2.0 - 0.5 * x - 0.5 * treated)
event_time = rng.exponential(1 / (relapse_rate + death_rate))
cause = np.where(rng.uniform(size=n) < relapse_rate / (relapse_rate + death_rate), 1, 2)
dropout = rng.exponential(1 / 0.15, size=n)
exact = pd.DataFrame(
    {
        "time": np.minimum(event_time, dropout),
        "event": np.where(event_time <= dropout, cause, 0),  # 0 means censored
        "treated": treated,
        "x": x,
    }
)
exact_design = TimeToEvent(
    time="time",
    event="event",
    treatment="treated",
    baseline=("x",),
    grid="continuous",
    causes={1: "relapse", 2: "death"},
)
exact_risks = CausalStudy(exact, design=exact_design).estimate(
    RegimeContrast({"treated": 1, "control": 0}, reference="control", horizons=(0.5, 2.0)),
    random_state=0,
)
for time in (0.5, 2):
    difference = exact_risks[f"ate_regimen[treated vs control, relapse @ t={time}]"]
    print(time, round(difference.psi, 3), difference.support.label)
```

Each line is the difference in the risk of relapse between treating everyone and no one, by
time 0.5 and by time 2, with its label. The fit models each event's rate over time with Cox
regressions and the treatment with logistic regression, picks among a few of each by
cross-validation, and fits each person's predictions on the other rows. It then nudges the
predicted rates, in small steps, until the risks satisfy the equations that make their intervals
valid, the method of Rytgaard and van der Laan (2024). It needs no learners, and a fit takes
about ten seconds at 1,000 people.

A [simulation study](../technical-reference/method-evidence/continuous-time-survival.md) with two
competing events at 1,000 people found the risks unbiased and their 95% intervals covering 92% to
95% of the time, at the times by which half and three quarters of the events had happened. At the
time by which a quarter had happened the intervals were too narrow, and some covered only 88% of
the time. So an estimate reads `supported` only at a time by which 40% to 85% of the events of
any cause have happened, not counting people who left follow-up. By time 0.5 about a fifth had,
so the first line reads `experimental`. The other conditions of the label are in the study's
[limits](../technical-reference/method-evidence/continuous-time-survival.md#limits).

When some people had almost no chance of one treatment, the intervals of a study cell at 4,000 people
covered as little as 37% of the time.
[Continuous-time survival and competing risks](../technical-reference/continuous-time-survival.md)
gives the method, its conditions and what it refuses.

### Restricted means and risk ratios with exact times

With exact times, call `ContinuousSurvivalTMLE` directly to get the restricted mean survival time
(RMST), the average time without either event up to a cut-off, and the restricted mean time lost
to one event. Name each cut-off in `restricted_horizons=` before the fit.

```python
from cleverly.survival import ContinuousSurvivalTMLE

exact_fit = ContinuousSurvivalTMLE(
    (0.5, 2.0),
    restricted_horizons=(2.0,),
    arms={"treated": 1, "control": 0},
    reference="control",
    causes={1: "relapse", 2: "death"},
    random_state=0,
).fit(exact, time="time", event="event", treatment="treated", baseline=["x"])
gained = exact_fit.rmst("treated", 2.0, versus="control")
lost = exact_fit.rmtl("treated", 2.0, "relapse", versus="control")
ratio = exact_fit.ratio(
    "cif_regimen[treated, relapse @ t=2]", "cif_regimen[control, relapse @ t=2]"
)
print(round(gained.psi, 3), round(lost.psi, 3), round(ratio.psi, 2), ratio.support.label)
```

The first number is the time without relapse or death that treating everyone adds, on average,
over the first 2 time units. The second is the change in time lost to relapse over the same
window, negative when treatment saves time. The third
is the ratio of the two risks of relapse by time 2. Each has an interval in `.ci`. All three read
`experimental`, because no simulation study has measured them yet.

The fit adjusts its predicted event rates until each reported number satisfies the equation its
interval rests on. A cut-off it was not told about was never part of that adjustment, so `rmst`
and `rmtl` refuse one, and `ratio` and `curve` refuse a time missing from the first argument, the
target times. The "What is refused" table of
[continuous-time survival](../technical-reference/continuous-time-survival.md) lists each case.
The extra cut-offs join the adjustment, so the risks of this fit differ slightly
from those of a fit without them.

## Treatments over time with an event outcome

When the treatment can change at each time point, declare the event as a sequence of outcome
columns on `LongitudinalTreatment`. Each column records whether the event has happened by that
point, and once the event happens it stays.

```python
from cleverly import LongitudinalTreatment
from cleverly.datasets import make_longitudinal_survival

panel, panel_truth = make_longitudinal_survival(n=2_000, seed=12)
panel_study = CausalStudy(
    panel,
    design=LongitudinalTreatment(
        outcome=("Y1", "Y2"),  # the event by each time point
        treatment=("A1", "A2"),  # the treatment at each time point
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),  # L2 is measured before the second treatment
        censoring=("C1", "C2"),  # 1 while the person is still followed, 0 once censored
    ),
)
panel_risks = panel_study.estimate(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
print(panel_risks.curve()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

Other plans of [Longitudinal treatment](longitudinal.md), such as rules that read a covariate,
work the same way.

## When another event can come first

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

### Ask what the risk would be had the other event been prevented

Another question asks what the risk of readmission would be in a world where nobody died, under
each plan. Declare it by naming the causes to prevent with `eliminate=` on the design. The fit
then treats a prevented event as censoring. Within one time step it counts censoring first, then
the prevented event, then the remaining causes.

```python
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
`eliminate=` works on `TimeToEvent`, `LongitudinalTreatment` and the long layout `LongPanel`, and
on every longitudinal estimator except structural nested means. A learned plan and a separable
effect refuse it.
The [registered study](../technical-reference/method-evidence/eliminated-competing-events.md)
covers iterated-mean longitudinal TMLE (`LTMLE`) on two binary treatment decisions and hazard
`LTMLE` on a three-level baseline treatment, both at n = 16,000 with saturated models. Those fits
print `supported`, in-sample and with two-fold cross-fitting; every other fit with `eliminate=` prints `experimental`.
[Eliminated competing events](../technical-reference/longitudinal-tmle.md#eliminated-competing-events)
gives the exact quantity from Young et al. (2020) and each check.
:::

### Separate a treatment's effect on two events

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
estimand and what a separable fit refuses.

## Keep the risk curve from dipping

A risk by day 60 cannot be lower than the risk by day 30, but the default fit corrects each
horizon on its own, so its curve can dip slightly. Two options prevent that.

| option | what it does | what it needs |
| --- | --- | --- |
| `survival_model="hazard", targeting="one_step"` | fits the hazard at every time point and corrects one set of hazards for every horizon together | one categorical treatment at baseline, held over every time point, and no covariate after baseline |
| `isotonic=True` | projects each fitted curve and its band limits onto a curve that never decreases | a longitudinal TMLE (`LTMLE`) survival or competing-risk fit without `msm=`. The other longitudinal estimators have no `isotonic=` |

Pass the hazard learner as `outcome_learner`. A hazard fit does not use `pseudo_learner`.

```python
hazard_curve = CausalStudy(frame, design=design).estimate(
    RegimeMean({"arm1": 1, "arm0": 0}),
    survival_model="hazard",
    targeting="one_step",
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
print(hazard_curve.curve()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

The risks of each arm do not decrease over the four grid times. The Settings block of
`summary(detail="full")` says whether the correction step converged; if it did not, the estimate
is not fully corrected, so say so when you report it. With the default `targeting="iterative"`, a hazard curve can
dip too.

To keep the default fit and project its curve, pass `isotonic=True`.

```python
projected = CausalStudy(frame, design=design).estimate(
    RegimeMean({"arm1": 1, "arm0": 0}),
    isotonic=True,
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
print(projected.isotonic_moves)
```

`isotonic_moves` lists each risk the projection moved. On this sample it is empty, because the
fitted curve already does not decrease. A pointwise interval at a moved risk is valid only where the true
curve strictly increases.

:::{dropdown} Details: what the projection keeps
The projection keeps every influence curve and standard error of the fit; only the point
estimates and band limits move.
[The isotonic projection](../technical-reference/longitudinal-tmle.md#the-isotonic-projection)
states each claim and its condition.
:::

:::{dropdown} Details: what the hazard model refuses, and its studies
The hazard model refuses a treatment at several time points, a covariate after baseline, a
continuous treatment, and a modified treatment policy. Each message points to the default model,
which fits those designs.
[Hazard-based survival targeting](../technical-reference/longitudinal-tmle.md#hazard-based-survival-targeting)
gives the contract, both algorithms, and the evidence.

Three registered studies measure hazard fits on a design with one baseline treatment:
[per-horizon targeting](../technical-reference/method-evidence/point-survival-hazard.md),
[one-step targeting](../technical-reference/method-evidence/point-survival-curve.md) and
[five-fold cross-fitting](../technical-reference/method-evidence/point-survival-curve-crossfit.md).
The first two agree with R packages fitted to the same samples (`survtmle` and `tmle3`). The
cross-fitted one has no comparison, because no package fits that construction.

The one-step study also holds one cell that fits the default iterated-mean model with
`isotonic=True` on a law with treatment at two time points. It fits every model by group
averages, and with such models the projection cannot move a risk. A second study,
[binding isotonic projection](../technical-reference/method-evidence/isotonic-binding.md), fits
logistic and linear models with a continuous covariate, so the projection moves a risk on about a
third of samples. There the 95% interval around the moved risk contained the true risk on 94% of
samples, inside the range the study accepts for a 95% interval. The moves it measured were at
most about half a standard error.
:::

## Let the data choose the treatment and censoring models

With many covariates, a treatment model that uses all of them can give some people a tiny
predicted chance of the treatment they got. Their weights then grow large and the estimate
becomes noisy. `method=CollaborativeTMLEMethod()` on a `TimeToEvent` design with a two-level
treatment builds the treatment and censoring models one covariate at a time instead, keeping a
covariate only when it improves the fit of the event data. It reports point estimates of the
risk, with no confidence interval.

A [simulation study](../technical-reference/method-evidence/survival-ctmle.md) on the laws of
Stitelman and van der Laan (2010) found the points unbiased when the event model is correct. With
a covariate that strongly predicts treatment but not the event, their typical error was 40% to
65% of that of the ordinary hazard fit with every covariate in the treatment model. When the
event model was wrong, some points were biased by up to about one standard deviation of the
estimate. [Methods and learners](methods-learners.md#method-presets) gives the settings, and
[Time-to-event outcomes](../technical-reference/collaborative-tmle.md#time-to-event-outcomes)
gives the algorithm and what it refuses.

## Keep the interval when one set of models is wrong

The ordinary hazard fit gives a valid confidence interval when the event model and the treatment
and censoring models are all close to the truth. `method=DRTMLEMethod()` adds a correction, from Díaz (2019), that
keeps the interval valid when only one side is right: the event model, or the treatment and
censoring models together. It fits a `TimeToEvent` design with a two-level treatment and one
event type. The side that is wrong still has to keep the ordering of people by
risk, or by chance of treatment, that its truth gives, and no fit can check that. When you know
the treatment and dropout probabilities, as in a randomized trial, give them to `TimeToEvent`
(`treatment_probabilities=` and `censoring_probabilities=`): the interval
then holds for any event model, as long as no one's chance of treatment or of staying in the
study is tiny.

A [simulation study](../technical-reference/method-evidence/survival-dr-tmle.md) with logistic
models on one law with five check times measured this. With one side wrong the points stayed
unbiased at 16,000 people, and with every model right the intervals covered 94% to 95% of the
time at 1,000 to 16,000 people. The correction loop did not always settle: in about 11% of the fits at 1,000 people
it stopped at its iteration cap with a warning, and those fits are part of the measured coverage.
Fit flexible treatment and censoring models with cross-fitting, the default: fitted on all rows
beside a wrong event model, the points were badly biased. [Time-to-event outcomes](../technical-reference/dr-tmle/theorem.md#time-to-event-outcomes)
gives the estimator, the conditions and what it refuses.

## Bootstrap intervals

Pass `n_bootstrap=` to refit the whole estimator on resamples of the data. Each estimate then
carries a percentile interval beside its usual interval.

```python
bootstrapped = panel_study.estimate(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_bootstrap=200,
    random_state=0,
)
print(bootstrapped.summary())
```

`ratio` and `rmst` carry the bootstrap of their inputs. Unless a simulation study covers the fit,
a note under the summary says the bootstrap interval is experimental. Report the usual interval
as the main result. When the fit warns with `SparseInfluenceWarning` that an estimate rests on
few people, show the bootstrap interval beside it.

:::{dropdown} Details: which bootstrap fits a study covers
The registered study covers an in-sample, single-cause survival fit on the wide layout with one
treatment decision at each time point, static plans, and no weights, measured with correctly
specified cell-mean models on finite binary laws at $n = 1000$. A `TimeToEvent` fit, whose one
treatment is held over the time points, and a competing-risk fit print an experimental bootstrap. A fit with fewer than 20 clusters (groups such as care teams whose members' outcomes move
together) prints its bootstrap standard deviation and percentile range as diagnostics.
[The full-refit bootstrap](../technical-reference/longitudinal-tmle.md#the-full-refit-bootstrap)
states what each resample redraws and refits, and
[Bootstrap for sparse regimens](longitudinal.md#bootstrap-for-sparse-regimens) covers the BCa
interval and the band for plans that few people follow.
:::

## Check sensitivity to a missing confounder

Two sensitivity analyses run on a `TimeToEvent` fit. Each asks how a confounder that the data do
not record could change the answer.

| call | what it does | where it runs |
| --- | --- | --- |
| `result.sensitivity.evalue((numerator, denominator))` | reads the risk ratio of two reported risks at one time and says how strongly a missing confounder would need to be tied to both treatment and event to explain it away | two static plans at one horizon. Name the risks, numerator first, such as `("risk_regimen[arm1 @ t=4]", "risk_regimen[arm0 @ t=4]")` |
| `result.sensitivity.simulated_confounding(...)` | plants a made-up confounder of chosen strengths and refits. It is a stress test, not a bound | a survival or competing-risk fit whose treatment is one baseline decision held over follow-up, censored or not |

`simulated_confounding` returns a surface: one estimate for each pair of confounder strengths in
the grid you pass. To read it, find the weakest confounder at which the estimate crosses zero, and judge
whether an unrecorded factor that strong is plausible in your setting. Each cell also reports
which share of the replayed rows came from the fitted models; a high share means the replay leans
on the models more than on the data.

:::{dropdown} Details: how the replay works
On a censored fit, the simulated confounder raises or lowers each person's chance of the event at
every time point, and the follow-up is replayed. Censoring is drawn again where the treatment
changed, and a person whose event the confounder removes stays in follow-up on fitted models. A
grid cell with both strengths at zero gives the original estimate.
[The stress surface](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)
gives the replay and its checks.
:::

Both analyses, and the checks that generate new outcomes (`dummy_outcome`, `simulated_outcome`,
`plasmode`), refuse a fit with `eliminate=` or a separable effect; the E-value reads only two ordinary arms. The
placebo, subset and random-common-cause checks of [Diagnostics](longitudinal.md#diagnostics) run
on these fits.
