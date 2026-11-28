# Exact event times, without a grid

When you know the exact time of each event and of each censoring, `grid="continuous"` skips the
check-time grid: no event is moved to a later check time. It handles a two-level treatment given
once at the start, and one event type or several competing ones, and reports each risk at the
times in `horizons=`. The tutorial [readmission or death with exact event
times](../examples/exact-event-times.ipynb) runs this analysis from start to finish on simulated
patients whose true risks are known.

```python
import numpy as np
import pandas as pd

from cleverly import CausalStudy, RegimeContrast, TimeToEvent

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
risks, as [Check sensitivity to a missing confounder](sensitivity.md#sensitivity-analysis-of-a-time-to-event-fit)
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

## Restricted means and risk ratios with exact times

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

## Choose the event-rate models

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
