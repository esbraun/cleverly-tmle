# Risks at set check times

Use this page when the outcome is whether and when an event happens, such as readmission or
relapse, and you report the risk at a grid of check times. `cleverly` estimates the risk of the event by each time you name, under each plan you
compare: for example, the share of
patients readmitted by day 30, 60 and 90 had everyone been offered a program, and had no one. It
also reports differences, ratios, and the average event-free time up to a cut-off. The
estimators are targeted maximum likelihood estimation (TMLE) for data over time: they fit
prediction models, then correct the estimate so its confidence interval is valid when those models are good enough.

`TimeToEvent` takes one row per person with a follow-up time and an
event code, and a treatment given once at the start. `LongitudinalTreatment` takes the wide
layout of [describe data over time](longitudinal.md), with one event column per time point, and
also fits treatments that change over time. A person who leaves follow-up is censored: their
event is unseen from then on. [Exact event times](survival-exact-times.md) skips the grid,
[competing events](survival-competing-events.md) covers an event that can come first, and
[fitting choices for event times](survival-fitting.md) covers the estimation options.

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
study checked, and [the support label](reading-output.md#read-the-support-label) explains the labels.

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

Other plans of [plans and policies over time](longitudinal-plans.md), such as rules that read a covariate,
work the same way.

### Treatments that change at irregular visits

When the treatment and the covariates change at visits that fall at any time, rather than at
shared time points, `ContinuousLongitudinalTMLE` estimates each regime's risk with no time grid.
A simulation study with known answers checked its intervals for the default models, with at
least 1,000 people and two competing kinds of event. An estimate reads `supported` only inside
the setting that study checked, and `experimental` otherwise, with the reason. The example below
fits 300 people with models of its own, so its estimates read `experimental`.

Give it one row per stretch of time between visits. A person's first row starts at 0 and each
later row starts at a visit, where the covariate `l` is measured and the treatment `a` decided;
the last row ends at the person's event or last follow-up, with the event code (0 when follow-up
ends without an event).

```python
import numpy as np
import pandas as pd

from cleverly.survival import ContinuousLongitudinalTMLE, CoxHazard

rng = np.random.default_rng(0)
rows = []
for person in range(300):
    w, l, a = rng.integers(2), rng.integers(2), rng.integers(2)
    start, visit = 0.0, rng.exponential(3.0)
    end = rng.exponential(1 / (0.1 * np.exp(0.5 * l - 0.5 * a)))
    if visit < min(end, 8.0):  # a visit before the event: l is measured again, a decided
        rows.append((person, 0.0, visit, 0, w, l, a))
        l = rng.integers(2)
        a = int(rng.random() < 0.3 + 0.4 * l)
        start = visit
        end = visit + rng.exponential(1 / (0.1 * np.exp(0.5 * l - 0.5 * a)))
    rows.append((person, start, min(end, 8.0), int(end < 8.0), w, l, a))
visits = pd.DataFrame(rows, columns=["id", "start", "stop", "event", "w", "l", "a"])

cox = [CoxHazard("main")]
visit_risks = ContinuousLongitudinalTMLE(
    [2.0, 6.0],  # the times at which to report each risk
    regimes={"always": 1, "never": 0},  # the treatment held at every visit
    reference="never",
    hazard_library=cox,
    visit_library=cox,
    censoring_library=cox,
    random_state=0,
).fit(
    visits,
    id="id",
    start="start",
    stop="stop",
    event="event",
    treatment="a",
    baseline=["w"],
    time_varying=["l"],
)
print(visit_risks.curve()[["estimand", "psi", "ci_lower", "ci_upper"]])
```

A regime can also be a rule: a function that takes a frame of the visits, with each covariate
by name, and returns the treatment at each. Wrap it as
`Rule(function, "treat when l is high", rule_kind="known")` from `cleverly.interventions`, which
declares that the rule was fixed before you saw the data; a bare function is refused. The estimate assumes that the models read a
person's history only through the last visit's covariates and treatment, the one before it, and
the visit's time and count. The [technical reference](../technical-reference/continuous-time-longitudinal.md)
states this assumption, the cap on the number of visits modelled, what the fit refuses, and every
condition of the `supported` label.

## Summarize the risk curves with a trend line

A trend line can summarize the risk curves too: fit a straight line in time through each plan's
risk at every check time, and read how fast the risk climbs and how much treatment slows it. As in
[a trend line over time](longitudinal-plans.md#summarize-several-plans-with-a-trend-line), you choose the
line's form (a working model, `MSM`) and pass it to `MSMProjection`. Its coefficients estimate
the best line through the true risks, with each plan and check time weighted equally, whether or
not the risks fall on one; when they bend, as here, the slope is the average rise over the check
times. This example uses the hazard model from
[Keep the risk curve from dipping](survival-fitting.md#keep-the-risk-curve-from-dipping), with five check times.

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
