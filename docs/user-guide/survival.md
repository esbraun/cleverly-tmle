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

To check overlap, plot each person's predicted chance of treatment at baseline:

```python
from cleverly.plotting import notes, overlap

fig = overlap(curve, censoring=True)
print(notes(fig))
```

The top panel shows that chance for the treated above the axis and the untreated below. Where one
group has people and the other has none, the comparison leans on the model there rather than on
data. The fit limits the product of that chance and the chance of staying in follow-up, not the
chance alone, so `censoring=True` adds that product for each plan up to the last horizon, with the
limit as a dotted line and the share of people beyond it printed at its foot.

[Readmission by day 30, 60 and 90](../examples/discharge-to-readmission.ipynb) works a full
analysis with this design.

## Ratios and the restricted mean survival time

The restricted mean survival time (RMST) up to a cut-off `tau` is the average event-free time
over the window from 0 to `tau`: a person who has the event at time 1.5 counts 1.5, and a person
still event-free at `tau` counts `tau`. Declare each cut-off in `restricted_horizons=` on the
plan comparison, and read the difference of the two plans with `rmst`. `.psi` is the point
estimate.

```python
gain = CausalStudy(frame, design=design).estimate(
    RegimeContrast(
        {"arm1": 1, "arm0": 0}, reference="arm0", horizons=(4,), restricted_horizons=(4,)
    ),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
rmst = gain.rmst("arm1", 4, versus="arm0")
print(rmst.name, rmst.ci, rmst.support.label)
```

`rmst_regimen[arm1 vs arm0 @ t=4]` is the event-free time that treating everyone adds, on
average, over the first 4 grid time units. `RegimeMean(..., restricted_horizons=(4,))` reports
each plan's own RMST instead, and `rmtl(arm, tau, cause)` the restricted mean time lost to one
cause, the average time spent after that event within the window. The fit needs the risk at every
grid time before `tau`, so it adds those times to the fit and reports only the ones in
`horizons=`. That costs $K(K+1)/2$ regressions per plan and cause for $K$ grid times up to `tau`
on the default model. On `survival_model="hazard"` with `targeting="one_step"`, the added times
also join the adjustment step and move the reported risks slightly.

The RMST is read off the grid: an event counts at the next check time, so this average is never
below the one from exact event times, and exceeds it by less than the longest gap between check
times, counting from time 0. The name keeps the cut-off's position on the grid, as for a risk,
and the `time` column of `to_frame()` gives `tau` itself. Its interval is the plain symmetric
interval of a sum of risks, unlike the exact-time one below, which stays inside 0 to `tau`.
This example's interval reads `experimental`, because no simulation study checked four check
times fitted in sample. The RMST difference reads `supported` on the five-time design that the
[point-treatment survival](../technical-reference/method-evidence/point-treatment-survival.md)
study checked, and [Read and save results](results-assessment.md) explains the labels.

A fit with every grid time in `horizons=` also gives ratios, and computes the RMST of any cut-off
from its risks without declaring it. Neither call refits anything.

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
print(risk_ratio.ci, survival_ratio.psi, risks.rmst("arm1", 3).psi)
```

| call | what it reports |
| --- | --- |
| `ratio(a, b)` | the risk ratio of two plans at one time, with its interval on the log scale |
| `ratio(a, b, view="survival")` | the ratio of the two survival probabilities |
| `ratio(a, b, kind="or")` | the odds ratio of the two plans |
| `rmst(d, tau)` | the RMST of plan `d` up to time `tau` |
| `rmst(d, tau, versus=b)` | the difference of two RMSTs |
| `rmtl(d, tau, cause)` | the restricted mean time lost to one cause |

To see the risks over time, draw the fit that reports every grid time:

```python
import cleverly.plotting

cleverly.plotting.curve(risks)
```

The top panel shows each plan's risk at every grid time with its 95% interval, held flat until
the next grid time, and the rows below count, for each plan, the people who received it and are
still followed. The paler band around each interval, drawn as brackets at the last time, is the
simultaneous band, which covers every risk at once; no study has measured it yet, so it reads
`experimental`. On a `RegimeContrast` fit, which keeps no single plan's risk, the figure draws
only the differences.

:::{dropdown} Details: what `rmst` needs and its units
For unit-spaced times the RMST is $\tau - \sum_{t<\tau} F_d(t)$, with $F_d(t)$ the risk by `t`
under plan `d`. On a `TimeToEvent` grid with other spacing, each risk is weighted by the gap to
the next grid time. On a `TimeToEvent` grid the times are grid times. On the wide layout they
count time points, so for points spaced $\Delta$ apart, multiply by $\Delta$ to get calendar
time. For a `tau` not in `restricted_horizons=`, `rmst` needs the risk at every grid time before `tau` in
the fit, and a `RegimeContrast` fit keeps no risk of a single plan, so it then refuses and names
the field.
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
a few seconds at 1,000 people.

The fit works through the people in small groups, so its memory grows with the number of people
and of estimates, not with the number of distinct follow-up times. With one kind of event, 100,000
people and two horizons, it needs about 0.7 GB. About 0.4 GB of that is working space for the
default standard errors (`variance="switched"`), which does not grow
with the number of people; `variance="influence_curve"` skips that step and needs about 0.3 GB.
Before fitting anything, the fit checks that the machine has the memory it needs free and stops
with an error when it does not; on macOS, which does not report free memory, it checks only a
limit you set. Between passes it keeps as many groups as the free memory holds and recomputes the
rest, so a larger limit makes a large fit faster and never changes its result. Pass
`max_memory_bytes=` to `estimate()` to set the limit, as the
[Memory](../technical-reference/continuous-time-survival.md#memory) section explains.

A [simulation study](../technical-reference/method-evidence/continuous-time-survival.md) with two
competing events at 1,000 people found the risks unbiased and their 95% intervals covering 94% to
95% of the time, at the times by which half and three quarters of the events had happened. At the
time by which a quarter had happened, a
[second study](../technical-reference/method-evidence/continuous-survival-variance.md) on five
other data-generating setups found some intervals still too narrow, and a
[third study](../technical-reference/method-evidence/continuous-survival-causes.md) measured fits
with one and with three kinds of event. The
share in the table counts every observed event of any kind together, and leaves out people who
left follow-up.

| the fit, with the default models and settings | an estimate reads `supported` when |
| --- | --- |
| two kinds of event, `cross_fit=True` | n is at least 500, 40% to 85% of the events have happened by its time, and its interval does not lean on a few people (its `variance_rel_sd` is at most 0.246) |
| one or three kinds of event, `cross_fit=True` | the same, from 1,000 people, with no restricted mean |
| two kinds of event, `cross_fit=False` | the same, from 1,000 people, with no restricted mean |
| two kinds of event, `cross_fit=True`, one cut-off in `restricted_horizons=` | the same, from 1,000 people; `rmst` and `rmtl` also need 67% to 85% of the events by the cut-off, and the risk ratio of the two arms reads `supported` only in such a fit |

Anything else you change from the defaults, such as the models, `n_folds=` or
`variance="influence_curve"`, makes the estimate `experimental`, and so does a fit with four or
more kinds of event. Each estimate's `support.reason` names the condition it missed, and the
[Evidence](../technical-reference/continuous-time-survival.md#evidence) section gives every
condition. In the example, about a fifth of the events had happened by time 0.5, so the first line
reads `experimental`. The treated arm's risk of death by time 2, and the difference in that risk,
read `experimental` too, because their intervals lean on a few people.

`exact_risks.assess()` checks the fit after it runs. It reports how far the predicted chances of
each treatment, and of staying in follow-up, come from 0, which models the fit chose, and whether
the step that adjusts the risks met the stopping rule its intervals rely on. Three checks refit the
analysis: the refutation tests (placebo, random common cause and subset), the truncation curve
(how the estimates move as the floor on small chances is raised), and the simulated-confounder
stress test. They run when you call them, and the report lists them as waiting for that call;
the refutation also needs you to name the estimate to check, and the stress test an estimate and
a grid of confounder strengths. The E-value of a risk ratio waits too, until you name the two
risks, as [Check sensitivity to a missing confounder](#check-sensitivity-to-a-missing-confounder)
shows. The omitted-variable bound, the marginal bound and the missingness tilt are refused for this
fit, each with its reason; the [Assessment](../technical-reference/continuous-time-survival.md#assessment)
table lists every check.

The placebo test shuffles the treatment among the people and refits, so the refitted difference
should sit near zero:

```python
placebo = exact_risks.diagnostics.refute(
    estimand="ate_regimen[treated vs control, relapse @ t=2]",
    tests=("placebo",),
    n_replicates=2,
)
print(placebo.summary())
```

Two refits keep the example fast. The default runs the three tests 20 times each, 60 refits, which
takes minutes at 1,000 people; `exact_risks.diagnostics.planned_refits(estimand=...)` gives the
count before you start. A [refutation study](../technical-reference/method-evidence/refuter-calibration-continuous-survival.md)
set pass and fail rules for the placebo and subset tests on a risk difference, from a fit like the
one it ran: two kinds of event, at least 1,000 people, the default models and settings, the
default 20 refits, and a time by which 43% to 79% of the events have happened. Even there, the
subset test fails about one fit in eleven with nothing wrong. Every other reading prints its
numbers with no pass or fail, as this two-refit example does.

The plots draw the risks and the same checks from what the fit kept:

```python
from cleverly.plotting import curve, learners, overlap, weights

curve(exact_risks)
overlap(exact_risks, censoring=True)
weights(exact_risks)
learners(exact_risks)
```

`curve` draws each plan's risk of each event, one column per event, and the difference below.
Each risk is a point at a time in `horizons=` with its 95% interval, and nothing is drawn between
those times, because the fit estimates no risk there. Brackets at each time mark the
simultaneous band, which covers all the reported risks and differences at once. A study fit
computes it by default, and a direct `ContinuousSurvivalTMLE` fit needs `simultaneous=True`. No
study has measured this band yet, so it reads `experimental`. `curve(..., band="none")` leaves it
out, and `band="bootstrap"` draws the bootstrap band of a fit with `n_bootstrap=`.

`overlap` shows each person's predicted chance of being treated. With `censoring=True` it adds,
for each arm, the chance of that arm times the chance of staying in follow-up just before the last
time in `horizons=`. The fit puts a floor on that product, drawn as a dotted line.

`weights` draws one over the product at each time in `horizons=`, for each person under the arm
they received. Before the floor caps it, no weight the fit gives that person by then is larger.
Each box holds everyone given that arm, whether or not they were still followed at that time.
`learners` shows the cross-validated loss of each candidate model and marks the one the fit chose.

When some people had almost no chance of one treatment, the intervals of a study cell at 4,000 people
covered as little as 37% of the time.

Such people can also push the event-rate model past what it can predict. A person whose covariates
lie outside the range seen in one treatment group can get, under that treatment, a predicted chance
of an event at a single time of one or more. The fit then caps that chance just below one, keeping
each event type's share of it, and goes on. It warns with `PositivityWarning`, naming the treatment
and how many people it capped, and marks every estimate `experimental`, because no study measured a
fit where this happened. `result.record.jump_bound_share` gives the share of person and treatment
pairs it capped; check `overlap` and consider a model without the treatment interaction.
When the cap acts at many times for one person, that person's predicted chance of staying free
of every event shrinks past what the computer can hold, and the fit still stops with a
`DataError` that names the cap.
[Continuous-time survival and competing risks](../technical-reference/continuous-time-survival.md)
gives the method, its conditions and what it refuses.

### Restricted means and risk ratios with exact times

With exact times, declare each cut-off in `restricted_horizons=` on the plan comparison to get the
restricted mean survival time (RMST), the average time without either event up to the cut-off,
and the restricted mean time lost to one event.

```python
exact_fit = CausalStudy(exact, design=exact_design).estimate(
    RegimeContrast(
        {"treated": 1, "control": 0},
        reference="control",
        horizons=(0.5, 2.0),
        restricted_horizons=(2.0,),
    ),
    random_state=0,
)
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
is the ratio of the two risks of relapse by time 2. Each has an interval in `.ci`.

The ratio reads `supported`. A [simulation study](../technical-reference/method-evidence/continuous-survival-functionals.md)
with one cut-off at the last target time found the restricted means unbiased and their 95%
intervals covering 94% to 95% of the time. It measured a cut-off by which three quarters of the
events had happened, and a mean up to an earlier cut-off leans more on the early times, where
intervals ran narrow. So `rmst` and `rmtl` read `supported` only at a cut-off by which 67% to 85%
of the events have happened. By time 2 here 62% had, so the first two read `experimental`. The
study's [limits](../technical-reference/method-evidence/continuous-survival-functionals.md#limits)
give the other conditions, among them that a cut-off past the last target time reads
`experimental`.

The fit adjusts its predicted event rates until each reported number satisfies the equation its
interval rests on. A cut-off it was not told about was never part of that adjustment, so `rmst`
and `rmtl` refuse one, and `ratio` and `curve` refuse a time missing from `horizons=`. The "What
is refused" table of
[continuous-time survival](../technical-reference/continuous-time-survival.md) lists each case.
The extra cut-offs join the adjustment, so the risks of this fit differ slightly
from those of a fit without them.

`ratio`, `rmst` and `rmtl` work as in the grid fits above, except that `rmst` and `rmtl` read
only a cut-off declared in `restricted_horizons=`. A fit with exact times also reports the
risk ratio of the two arms beside each difference, under `rr_regimen[...]`, and these read
`supported` only in a fit with one cut-off. Its other calls:

| call | what it gives |
| --- | --- |
| `curve()` | a frame of every risk with its interval, one row per arm, kind of event and time |
| `contrast(function, names)` | any smooth function of reported estimates, with its interval; it always reads `experimental` |
| `to_frame()`, `to_dict()` | every estimate as a table or a dictionary |

With two or more kinds of event, `curve(scale="survival")` and `ratio(..., view="survival")`
refuse, because one minus one event's risk is not the chance of being free of every event. The
risk of any event is the sum of the risks of each kind, which `contrast` gives, and the chance of
staying free of every event is one minus that sum:

```python
any_event = exact_fit.contrast(
    lambda risks: risks[0] + risks[1],
    ["cif_regimen[treated, relapse @ t=2]", "cif_regimen[treated, death @ t=2]"],
    name="any event, treated @ t=2",
)
print(round(any_event.psi, 3), any_event.ci)
```

:::{dropdown} Details: the same fit without a study
The study builds this call to `ContinuousSurvivalTMLE`, which takes the cut-offs as
`restricted_horizons=` and returns the same estimates:

```python
from cleverly.survival import ContinuousSurvivalTMLE

engine_fit = ContinuousSurvivalTMLE(
    (0.5, 2.0),
    restricted_horizons=(2.0,),
    arms={"treated": 1, "control": 0},
    reference="control",
    causes={1: "relapse", 2: "death"},
    random_state=0,
).fit(exact, time="time", event="event", treatment="treated", baseline=["x"])
print(round(engine_fit.rmst("treated", 2.0, versus="control").psi, 3))
```
:::

### Choose the event-rate models

By default the fit chooses each event's rate model, and the model of leaving follow-up, from three
Cox regressions: one on the treatment alone, one on the treatment and the covariates, and one that
adds every treatment-by-covariate product. It picks among them by cross-validation. Pass
`hazard_library=` and `censoring_library=` to `estimate()` to choose others:

```python
from cleverly.survival import CoxHazard, StratifiedNelsonAalen

own_models = CausalStudy(exact, design=exact_design).estimate(
    RegimeContrast({"treated": 1, "control": 0}, reference="control", horizons=(0.5, 2.0)),
    hazard_library=(CoxHazard("main"), CoxHazard("interaction", penalty=1.0)),
    censoring_library=(StratifiedNelsonAalen(),),
    random_state=0,
)
print(own_models["ate_regimen[treated vs control, relapse @ t=2]"].support.label)
```

`CoxHazard(terms, penalty=)` takes `"treatment"`, `"main"` or `"interaction"`, and a ridge
`penalty=` that shrinks the coefficients. `StratifiedNelsonAalen()` estimates the rate within
each arm with no covariates, which suits dropout that depends on the arm alone. A library with
more than one model must hold `CoxHazard` models only, because the cross-validation compares their
fits on the same people at risk; mixing in `StratifiedNelsonAalen` is refused. Any library other
than the default makes every estimate `experimental`, as this one prints.

A Cox fit can fail to settle when a covariate sorts the event times perfectly, or a kind of
event is missing from one arm in a training fold. The fit may then warn with a
`ConvergenceWarning` naming the coefficients that may be infinite; a check of this kind catches
most such cases, not all. Its message gives the two remedies: a `penalty=` on `CoxHazard`, or
`StratifiedNelsonAalen(by=...)` stratified on a discrete column that separates the data.
[Initial fits](../technical-reference/continuous-time-survival.md#initial-fits) gives the
selection rule, and [Settings](settings.md#models-and-learners-modelspec) lists both arguments.

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
| `survival_model="hazard", targeting="one_step"` | fits the hazard at every time point and corrects one set of hazards for every horizon together | one treatment at baseline, held over every time point, or a treatment with categories at each time point. A baseline treatment can also be a dose under a plan that changes each person's dose. A covariate measured after baseline needs `targeting="one_step"` |
| `isotonic=True` | projects each fitted curve and its band limits onto a curve that never decreases | a longitudinal TMLE (`LTMLE`) survival or competing-risk fit without `msm=`. The other longitudinal estimators have no `isotonic=` |

Pass the hazard learner as `outcome_learner`. A hazard fit uses `pseudo_learner` only when the
design has a covariate measured after baseline, as the next section explains.

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

### Covariates measured after baseline

A covariate measured after baseline is one recorded again at later time points, such as a lab
value at each visit, declared with `time_varying=`. When it predicts both dropout and the event, the
fit must adjust for it. With one treatment at baseline, the hazard model takes such covariates under
`targeting="one_step"`. At each time point it fits the chance of the event, then a regression of what
happens later given the history so far, so it needs no model of the covariate itself.
`pseudo_learner` fits that regression. By default it reuses the `outcome_learner` library, so pass
`pseudo_learner` yourself when `outcome_learner` is a single classifier.

The answer it targets is the same as the default fit's, and when the models fit well both have the
same precision in large samples. What the hazard fit adds is a curve that never dips, and with
competing events, risks that add up to at most one for each person. With well-fitting models the gap
between the two estimates shrinks faster than their standard errors as the sample grows, so on a
large sample they nearly agree.

A treatment with categories at each time point fits beside such covariates too. A plan can fix the
treatment at each time point, follow a known rule that reads the latest covariate, or draw the
treatment with known probabilities that may read the covariate and the earlier treatment. At each
time point the fit predicts the event for every treatment level at each person's own history, so it
never builds the treatment sequences a plan could follow. A
[validation study](../technical-reference/method-evidence/hazard-time-varying-multinode.md)
measures this fit with a yes-or-no treatment at three time points and one yes-or-no covariate
measured before the treatment at each later time point, with one event type, in sample and with
cross-fitting.

A covariate declared at the first time point only is measured before the treatment, so it is an
ordinary confounder. The hazard model fits it, and gives the same estimate as declaring it in
`baseline=`. Under `markov_order=` the hazard model, like the default fit, drops that covariate
at later time points. `SurvivalCTMLE`, `SurvivalDRTMLE` and a `Separable` regimen refuse it; declare it
in `baseline=` for those. The
[covariates-after-baseline study](../technical-reference/method-evidence/hazard-time-varying.md)
measures the hazard fit with one yes-or-no covariate per later time point, for two fixed
treatments with one or two event types, in sample and with cross-fitting, and for a policy that
moves everyone up one treatment level.
[Covariates after baseline on the hazard path](../technical-reference/longitudinal-tmle.md#covariates-after-baseline-on-the-hazard-path)
gives the exact estimand, the settings it refuses, and its checks.

:::{dropdown} Details: what the hazard model refuses, and its studies
With a treatment at several time points and a covariate measured after baseline, the hazard model
refuses a plan that moves people to another treatment level, or one that multiplies the odds of
treatment when you supplied the treatment probabilities yourself
([H27 in the hazard refusals](../technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)).
Its message points to the default model, which fits that plan.

A treatment with categories at each time point also fits on the hazard model, when no covariate
is measured after baseline. A plan can fix the treatment at each time point, follow a rule that
reads the baseline covariates, or draw the treatment with known probabilities that may depend on
the earlier treatments. The fit corrects the hazards of every treatment sequence the plan can
follow. A plan that draws at many time points has many sequences, and a fit whose predicted
memory use passes 2 GiB is refused before any model is fitted. On a machine with more memory,
`max_memory_bytes=` on `LTMLE` or on `estimate()` sets a different limit, higher or lower.

If a trial design fixes the
treatment probabilities, you can pass them for the first time point only, as a mapping from the
first treatment column to its levels, such as
`treatment_probabilities={"A1": {0: "p0", 1: "p1"}}` with one probability column per level. An
array that covers every time point is refused, even with the later columns left empty
([H26 in the hazard refusals](../technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)).
The fit
estimates the probabilities of later treatment and of dropout, because it needs them along
sequences that nobody followed.

The
[multi-node hazard study](../technical-reference/method-evidence/hazard-multinode.md)
measures this fit with a yes-or-no treatment at three and at four time points, one event type and
no cross-fitting, under the default `targeting="iterative"`. Those fits are labelled `supported`.
Fits outside that, including the same regimens with `targeting="one_step"`, are labelled
`experimental`.
[A treatment at several nodes on the hazard path](../technical-reference/longitudinal-tmle.md#a-treatment-at-several-nodes-on-the-hazard-path)
gives the exact estimand and its checks.

A plan that changes each person's dose, such as `Shift(1.0, cap=2.0)`, also fits on the hazard
model. The treatment can be categories, a dose, or a dose recorded beside a yes/no treatment.
Each such plan gets its own correction of the hazards, so with `targeting="one_step"` its curve
never dips. The [hazard policy study](../technical-reference/method-evidence/hazard-mtp.md)
measures such fits in sample and with cross-fitting. With `ratio="density"`, which bins the
dose to compare the shifted and the observed dose, it covers only a dose with a fixed range,
because on an unbounded dose the outer bins are wider than the shift.

A few settings cannot be combined with such a plan, and
[the hazard refusals](../technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)
list them.
[A modified treatment policy on the hazard path](../technical-reference/longitudinal-tmle.md#a-modified-treatment-policy-on-the-hazard-path)
gives the exact estimand and its checks.
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

## Summarize the risk curves with a trend line

A trend line can summarize the risk curves too: fit a straight line in time through each plan's
risk at every check time, and read how fast the risk climbs and how much treatment slows it. As in
[Longitudinal treatment](longitudinal.md#summarize-several-plans-with-a-trend-line), you choose the
line's form (a working model, `MSM`) and pass it to `MSMProjection`. Its coefficients estimate
the best line through the true risks, with each plan and check time weighted equally, whether or
not the risks fall on one; when they bend, as here, the slope is the average rise over the check
times. This example uses the hazard model from
[Keep the risk curve from dipping](#keep-the-risk-curve-from-dipping), with five check times.

```python
from cleverly import MSMProjection
from cleverly.msm import MSM

visits, visit_truth = make_point_survival(n=1_000, seed=4, n_times=5)
visit_design = TimeToEvent(
    time="time", event="event", treatment="A", baseline=("W1", "W2"), grid=(1, 2, 3, 4, 5)
)


def risk_line(plan, time, covariates):
    n, treated = len(covariates), float(plan == "arm1")
    return np.column_stack(
        [np.ones(n), np.full(n, treated), np.full(n, time), np.full(n, treated * time)]
    )


risk_model = MSM(
    design=risk_line, terms=("intercept", "arm1", "time", "arm1 x time"), design_kind="known"
)
risk_trend = CausalStudy(visits, design=visit_design).estimate(
    MSMProjection(risk_model, regimens={"arm1": 1, "arm0": 0}, horizons=(1, 2, 3, 4, 5)),
    survival_model="hazard",
    targeting="one_step",
    outcome_learner=LogisticRegression(max_iter=1000),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
print(risk_trend.coefficients()[["estimand", "psi", "ci_lower", "ci_upper"]].round(3))
```

The `msm_regimen[time]` row says that without treatment the risk rises by 0.117 per time unit
(0.104 to 0.130). The `msm_regimen[arm1 x time]` row reads -0.029 (-0.047 to -0.012): treating
everyone slows that rise by about 0.029 per unit. In the simulated data the best line through the
true risks in `visit_truth` rises by 0.106 per unit without treatment and by 0.024 less with it, both inside their intervals.

The design function receives each plan's name, the check time and the covariates, and returns one
column per term, here a constant, the treated plan, the time, and their product. A
[simulation study](../technical-reference/method-evidence/hazard-msm.md) measured this setup: the
hazard model with `targeting="one_step"`, no cross-fitting, one kind of event, five check times
and a treatment given at the start. So these coefficients carry the `supported` label, and the
same fit on any other number of check times, or with cross-fitting, reads `experimental`. Pass `link="logit"` to `MSM` to fit the line
on the log-odds scale, and `coefficients(scale="ratio")` then reports odds ratios. With competing
events each cause gets its own coefficients, as the subsection "A working model over the hazard
curve" of
[hazard-based survival targeting](../technical-reference/longitudinal-tmle.md#hazard-based-survival-targeting)
explains.

The hazard model refuses a trend line under the default `targeting="iterative"`, beside a plan
that changes each person's dose, and on a design with a covariate measured after baseline.
`isotonic=True` refuses any trend line. The default model, `survival_model="iterated_mean"`, takes
a trend line on every design, including those two; pass it a regression as `pseudo_learner=`.
[The hazard refusals](../technical-reference/scope-and-refusals.md#hazard-and-isotonic-survival-refusals)
give each message (H5, H14, H19 and H29).

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
estimate. `cleverly.plotting.learners(result)` draws each arm's sequence of added covariates with
the one the search chose marked, and `cleverly.plotting.overlap(result)` draws each arm's chosen
treatment model. [Methods and learners](methods-learners.md#method-presets) gives the settings, and
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

A fit with exact times, `grid="continuous"`, takes `n_bootstrap=` too. Each resample refits every
model and the risks, so it costs about as long as the fit itself. A person drawn twice into a
resample stays in one group when the fit splits the rows to fit predictions on the other rows, so
a copy never predicts its twin. The bootstrap interval of each risk sits in `estimate.bootstrap`
and in the `bootstrap_ci_lower` and `bootstrap_ci_upper` columns of `result.to_frame()`;
`bootstrap_interval="bca"` gives the BCa interval, which corrects the
percentile interval for skew. If more than 1% of the refits fail, the fit warns, drops every
bootstrap interval and keeps the usual ones. Below that share the bootstrap intervals use only the
refits that worked, without a warning; the summary's bootstrap note gives the count, and
`result.bootstrap_failures` says why the others failed.

On these fits the bootstrap interval always reads
`experimental`. With the default cross-fitting no theory yet shows that the bootstrap is valid,
and for a fit without cross-fitting the conditions of the theory that would cover it have not
yet been checked. [The full-refit bootstrap](../technical-reference/continuous-time-survival.md#the-full-refit-bootstrap)
gives both reasons and the study planned to measure it.

:::{dropdown} Details: which bootstrap fits a study covers
The registered study covers an in-sample, single-cause survival fit on the wide layout with one
treatment decision at each time point, static plans, and no weights, measured with correctly
specified cell-mean models on finite binary laws at $n = 1000$. A `TimeToEvent` fit, whose one
treatment is held over the time points, and a competing-risk fit print an experimental bootstrap. A fit with fewer clusters than its floor (groups such as care teams whose members' outcomes move
together; the floor is 20 for most longitudinal fits) prints its bootstrap standard deviation and percentile range as diagnostics.
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
| `result.sensitivity.evalue((numerator, denominator))` | reads the risk ratio of two reported risks at one time and says how strongly a missing confounder would need to be tied to both treatment and event to explain it away | two static plans at one horizon. Name the risks, numerator first, such as `("risk_regimen[arm1 @ t=4]", "risk_regimen[arm0 @ t=4]")`. On a fit with exact times, name the two arms' risks of one event at one target time, such as `("cif_regimen[treated, relapse @ t=2]", "cif_regimen[control, relapse @ t=2]")` |
| `result.sensitivity.simulated_confounding(...)` | plants a made-up confounder of chosen strengths and refits. It is a stress test, not a bound | a survival or competing-risk fit whose treatment is one baseline decision held over follow-up, censored or not, and a fit with exact times |

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

On a fit with exact times, the confounder scales each person's event rate of every event type
up or down, and new event and censoring times are drawn from models the analysis fits for this
purpose. Censoring is again drawn afresh only where the treatment changed, and both strengths at
zero give back the recorded times and the original estimate. Each grid cell other than that one
refits the whole analysis, so a 3 by 3 grid costs 8 refits: seconds at a few hundred people, and
minutes for a large cross-fitted fit. The surface prints numbers, not a pass or fail.
[Continuous-time survival replay](../technical-reference/validation-methods.md#simulated-common-cause-stress-surface)
gives the law and its checks.
:::

Both analyses, and the checks that generate new outcomes (`dummy_outcome`, `simulated_outcome`,
`plasmode`), refuse a fit with `eliminate=` or a separable effect; the E-value reads only two ordinary arms. The
placebo, subset and random-common-cause checks of [Diagnostics](longitudinal.md#diagnostics) run
on these fits.
