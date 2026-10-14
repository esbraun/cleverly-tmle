# Longitudinal treatment

## Declare temporal roles

`LongitudinalTreatment` aligns treatment nodes, time-varying histories, censoring indicators, and
outcome processes. At treatment node `t`, a dynamic rule sees only history available by `t`.
For the reported influence-curve inference, the rule must be fixed before the fit and rowwise. A
callable may vectorize over the history frame, but it must not estimate a threshold from the
sample or otherwise make one row's assignment depend on other rows. The rule needs a declaration,
which [Declare a dynamic rule](#declare-a-dynamic-rule) shows.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=2_000, seed=11)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
result = study.estimate(
    RegimeContrast({"always": 1, "never": 0}, reference="always"),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=3,
    learner_folds=3,
    random_state=0,
)
```

The resolved regimen matrix is shared by nuisance fitting, follower masks, targeting, and report
keys. A regimen cannot look ahead or change interpretation between stages. An empirically learned
policy needs inference for the policy-learning step and is outside this estimator's contract.

With `n_folds > 1`, each outer training fold runs an untargeted backward regression sequence. One
pooled fluctuation per node then targets the out-of-fold predictions over every follower. The fit
stores the realized assignment in `result.folds`.
[Cross-fitting the recursion](../technical-reference/longitudinal-tmle.md#cross-fitting-the-recursion)
gives the steps and the published result they follow.

Each node's fluctuation solves its score over every follower, as on a single-fold fit.
`res.diagnostics.score_equations()` reports one `solver` row per node.

The [cross-fitted end-of-study study](../technical-reference/method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md)
measured standard-error ratios from 0.9797 to 1.0130 over 1,600 replications at `n=2000`. Those
results apply only to that study's law and estimator settings. They do not establish calibrated
variance for other laws, weights, clusters, survival outcomes, or sample sizes.

## Read a long panel

Customer data usually arrives as one row per customer and decision time. Each row holds the
covariates known before the decision. It also holds the decision and the observation indicator.
`LongPanel` and `LongitudinalData.from_long_panel` reshape such a frame into the trajectory
container that `LongitudinalTreatment` builds from a wide frame. The reshape ends in
`LongitudinalData.from_frame`, so every check of the wide layout runs on the result.

| argument | the column holds | wide name |
| --- | --- | --- |
| `unit=` | the customer identifier. One value is one trajectory | the row of the container, kept in `unit_ids` |
| `time=` | the decision time. Values must sort | the node index `t`, counted from 1 |
| `treatment=` | the decision at that row. A list of columns is one vector node | `"offer@t"` |
| `baseline=` | unit-level covariates, measured before the first decision | the original name |
| `time_varying=` | covariates measured at that row, before its decision | `"eng@t"` |
| `censoring=` | `1` while the customer stays observed after the row's decision, `0` on the last row of a customer who left | `"active@t"` |
| `outcome=` | an end-of-study outcome, one value per customer, on any of its rows | the original name |
| `event=` | a survival event indicator on each row. A mapping of cause to column declares competing risks | `"event@t"` |
| `times=` | the decision times, in any order. Omit it to use the times the panel holds | the node order |
| `cluster=` | an account or store identifier, constant within a customer | `id=` of the cluster contract |
| `weights=` | a unit-level probability weight | `weights=` |
| `treatment_probabilities=`, `censoring_probabilities=` | the known mechanism, as `P(A_t = level \| past)` and `P(C_t = 1 \| past)` on each row | `"column@t"` |

Declare exactly one of `outcome=` and `event=`. One customer trajectory is one observation for the
inference model. Repeated rows of one customer add no independent observations. Declare
`cluster=` when customers depend on each other, such as the customers of one account.

A generated name is `"column@t"`, where `t` is the node index and not the time label. The labels
stay in `data.panel`, a `PanelLayout`, which `data.panel.decision_times` lists. A rule at node `t`
reads `"column@s"` for `s <= t`. The summary of a fit on an adapted panel adds a `long panel:`
line. The line names the unit column, the decision times, and any column left out at a node.
`from_long_panel` takes rows in any order. The container follows the sorted unit identifiers.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongPanel, RegimeContrast

rng = np.random.default_rng(5)
rows, bought = [], {}
for customer in range(400):
    age, eng, offers = rng.normal(), rng.normal(), 0
    for week in (10, 20, 30):
        offer = rng.random() < 1 / (1 + np.exp(-0.5 * eng))
        stays = week == 30 or rng.random() < 0.9
        rows.append((customer, week, age, eng, "offer" if offer else "none", float(stays)))
        if not stays:
            break
        offers += offer
        eng = 0.6 * eng + 0.5 * offer + rng.normal(scale=0.7)
    else:
        p = 1 / (1 + np.exp(-(0.5 * eng + 0.4 * offers - 0.5)))
        bought[customer] = float(rng.random() < p)
panel = pd.DataFrame(rows, columns=["customer", "week", "age", "eng", "offer", "active"])
panel["bought"] = panel["customer"].map(bought)

long_design = LongPanel(
    unit="customer",
    time="week",
    treatment="offer",
    baseline=["age"],
    time_varying=["eng"],
    censoring="active",
    outcome="bought",
)
long_study = CausalStudy(panel, design=long_design)
print(long_study.data.history_names(2), long_study.data.panel.decision_times)
```

The panel holds 400 customers over weeks 10, 20 and 30. A customer who leaves has `active` equal to
0 on its last row, and no outcome. `history_names(2)` returns `('age', 'eng@1', 'eng@2')`. The
design hands the frame to `CausalStudy`, and `from_long_panel` takes the same arguments if you
want the container first.

### Put each measurement on the right row

A `time_varying` column is measured before that row's decision. Code cannot read the order of a
measurement from the data, so the declaration states it. A measurement taken after the decision,
such as the response to the offer, belongs to the next row. Move it with one line of pandas:

```python
after = panel.assign(response=panel["eng"])  # recorded after each row's decision
after["response"] = after.groupby("customer")["response"].shift()
```

The first row of each customer then holds a missing value. A leading run of missing values is not
an error. The column starts at its first recorded node, and `data.panel.omitted` lists each column
and node left out. A column that is missing at a later node, after it was first recorded, is
refused. Leaving it out there would remove a confounder from that node's adjustment set.

A column that holds the previous treatment is not a leak. Among the units that follow a plan, it
equals the arm that the plan assigned. Under a policy, the earlier observed arm is what
`policy_frame` already exposes to a density.

### Write one rule for every node

`each_node(rule, n_times)` returns one rule per node. Node `t` calls `rule(history, t)`, so the
rule reads `f"eng@{t}"` and nothing later. Put the result in a `DynamicRegimen`, as
[Declare a dynamic rule](#declare-a-dynamic-rule) states.

```python
from cleverly.longitudinal import DynamicRegimen, each_node

engaged = DynamicRegimen(
    "engaged",
    each_node(lambda history, t: np.where(history[f"eng@{t}"] > 0, "offer", "none"), 3),
    rule_kind="known",
)
long_result = long_study.estimate(
    RegimeContrast({"never": ("none",) * 3, "engaged": engaged}, reference="never"),
    outcome_learner=LogisticRegression(max_iter=1000),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    n_folds=2,
    random_state=0,
)
print(long_result.summary())
```

This plan offers at node `t` when the customer's engagement at node `t` is positive. The fit
raises `DataError` for a rule that names a column from a later node. It raises the same error when
the column is absent at the node, for example because the node screened it out as constant. It
never reads an earlier node instead.

| test | what it checks |
| --- | --- |
| `test_a_rule_reads_its_own_node_and_never_a_later_one` | `each_node` equals a rule written by hand, a rule that reads the previous node assigns different arms, and a rule that reads a later node raises |
| `test_a_screened_out_node_raises_rather_than_read_an_earlier_node` | a rule raises when its column is absent at its node |
| `test_ltmle_on_the_adapted_panel_equals_the_wide_fit` | `LTMLE` on the adapted panel equals the fit on the equivalent wide frame, to a relative difference of 1e-12 |

### Go back to a long panel

`data.to_long_panel()` returns one row per customer and node at which the customer is still in the
study. Rows are sorted by unit and time, in the backend that the data came from.

```python
back = long_study.data.to_long_panel()
print(back.head(3))
```

The result differs from the input in three ways.

| difference | reason |
| --- | --- |
| the weights have mean one | the container normalizes the weights |
| a covariate screened out at a node is missing at that node | the node holds no column for it |
| the known probability columns are not returned | the container keeps them as mechanisms, not as covariates |

`to_long_panel` raises `DataError` on a container built from a wide frame.

### What the reader refuses

Each refusal raises `DataError` before any learner, and the message names the unit and the time.

| the panel holds | why the reader refuses it |
| --- | --- |
| a missing unit or time, a duplicated `(unit, time)` pair, a time that `times=` does not list | a long panel holds one row per unit and decision time. `times=` names every decision time and cannot coarsen a grid |
| a customer whose first row is after the first decision, or a gap between two rows | late entry is a different observation scheme. Intermittent observation is not censoring |
| rows after a customer left, or rows that stop with no censoring 0 and no event | a customer who is censored or has the event leaves for good. Declare `censoring=` |
| an event value on a row whose censoring is 0 | the order at a node is $A_t, C_t, Y_t$. Record an event as censoring 1 and event 1. Record a loss after an event-free interval as censoring 0 and a missing event |
| a `baseline`, `cluster` or `weights` column that varies within a customer, or is recorded only after the first row | a later value would let a rule at an earlier node read the future |
| an `outcome` that varies within a customer | an end-of-study outcome holds one value per unit. A per-row outcome is `event=` |
| one column in two roles, or a covariate that equals the outcome or the row's event | a rule could read the outcome under another name |
| a time-varying column with no value at a node after its first recorded node | the column would leave that node's adjustment set |
| a treatment that never changes within a customer | that is one baseline decision held over every node. Build the wide frame and declare `treatment="A"` |
| a column named like a generated name, such as `eng@1` | the name collides with a generated column. Rename it |

The wide layout refuses the same event on a censored row. `LongitudinalData.from_frame` used to
drop it without a message, and it now raises `DataError`.

### Long horizons

The registered evidence for `LTMLE` and `SDR` covers two decision nodes. The
[roadmap](../roadmap.md#x39-longitudinal-evidence-over-three-to-five-decisions) plans laws with
three to five. Every $L_t$ block accumulates into the history design, so a long panel adds columns
at each node and needs more customers. `times=` cannot coarsen a grid. To model fewer decisions,
aggregate the frame to the coarser decision times before you read it.

## Declare a dynamic rule

Write each plan that holds a rule as a `DynamicRegimen`, and declare `rule_kind="known"`. The
declaration states that every callable node is a fixed rowwise function of the history.

```python
from cleverly.longitudinal import DynamicRegimen

adaptive = DynamicRegimen(
    "adaptive",
    (1, lambda history: (history["L2"] > 0).astype(int)),
    rule_kind="known",
)
rule_result = study.estimate(
    RegimeContrast({"never": 0, "adaptive": adaptive}, reference="never"),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=3,
    learner_folds=3,
    random_state=0,
)
```

This plan treats at the first node. At the second node it treats each unit whose `L2` is positive.
`DynamicRegimen` raises `CapabilityError` for `rule_kind=None` and for `"estimated"`. The fit
refuses a callable written inline in `regimens=`, such as `{"adaptive": (1, rule)}`. An inline
callable carries no declaration. A plan of labels alone, such as `{"never": 0}`, needs no
declaration. Code cannot inspect a closure, so the fit accepts a false `"known"` declaration.

## Declare a stochastic policy

A node can draw its arm from a known distribution. Write the node as a `Stochastic` node with
`density_kind="known"`, inside a `DynamicRegimen`. The density function receives the history and
the earlier treatments, and it returns one column per treatment level, in sorted level order.

```python
import numpy as np
from cleverly.interventions import Stochastic


def half(history):
    return np.full((len(history), 2), 0.5)


def by_marker(history):
    treat = np.where(history["L2"] > 0, 0.75, 0.25)
    return np.column_stack([1.0 - treat, treat])


mostly = DynamicRegimen(
    "mostly",
    (
        Stochastic(half, "half", density_kind="known"),
        Stochastic(by_marker, "marker", density_kind="known"),
    ),
)
policy_result = study.estimate(
    RegimeContrast({"never": 0, "mostly": mostly}, reference="never"),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_folds=3,
    learner_folds=3,
    random_state=0,
)
```

This plan treats half of the units at the first node. At the second node it treats three quarters
of the units whose `L2` is positive and a quarter of the others. The parameter is the mean outcome
if every unit drew its arms this way.

A rule node cannot read an earlier treatment, because the history it receives holds none. To
continue the arm that an earlier policy node drew, write a one-hot `Stochastic` node that reads
it. The fit resolves a one-hot density as the rule it equals.

```python
def continue_drawn(history):
    drawn = (history["A1"] == 1).to_numpy(dtype=float)
    return np.column_stack([1.0 - drawn, drawn])


continued = DynamicRegimen(
    "continued",
    (
        Stochastic(half, "half", density_kind="known"),
        Stochastic(continue_drawn, "continue", density_kind="known"),
    ),
)
```

Positivity is required wherever the policy puts mass. A policy that can draw a level that no unit
on the plan received at that node raises `LongitudinalError` before any learner.
[Known stochastic policies](../technical-reference/longitudinal-tmle.md#known-stochastic-policies)
gives the estimator, its conditions, and its evidence. Code cannot inspect a closure, so the fit
accepts a false `"known"` declaration.

## Survival outcomes

An outcome sequence declares one absorbing event process and makes horizon part of the estimand.

```python
from cleverly import RegimeMean
from cleverly.datasets import make_longitudinal_survival

frame, truth = make_longitudinal_survival(n=2_000, seed=12)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome=("Y1", "Y2"),
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
risks = study.estimate(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
```

Each node fit uses the relevant uncensored, event-free risk set. The result key preserves regimen
and horizon.

Read ratios and the restricted mean survival time (RMST) off the fitted curve. They refit nothing.

```python
risk_ratio = risks.ratio("risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]")
survival_ratio = risks.ratio(
    "risk_regimen[always @ t=2]", "risk_regimen[never @ t=2]", view="survival"
)
rmst = risks.rmst("always", 3, versus="never")
print(risk_ratio.ci, survival_ratio.psi, rmst.psi, rmst.ci)
```

| call | what it reports |
| --- | --- |
| `ratio(a, b)` | the risk ratio of two levels at one horizon, with its interval on the log scale |
| `ratio(a, b, view="survival")` | the ratio of the two survival probabilities |
| `ratio(a, b, kind="or")` | the odds ratio of the two levels |
| `rmst(d, tau)` | the RMST up to node `tau`, $\tau - \sum_{t<\tau} F_d(t)$, in node units |
| `rmst(d, tau, versus=b)` | the difference of two RMSTs |
| `rmtl(d, tau, cause)` | the restricted mean time lost to one cause |

`rmst` needs the risk at every horizon from 1 to `tau - 1`. For nodes spaced $\Delta$ apart,
multiply by $\Delta$ to get calendar time.

Pass `n_bootstrap=` to refit the whole estimator on resamples. Each estimate then carries a
percentile interval beside its influence-curve interval.

```python
bootstrapped = study.estimate(
    RegimeMean({"always": 1, "never": 0}, horizons=(1, 2)),
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    n_bootstrap=200,
    random_state=0,
)
print(bootstrapped.summary())
```

The [bootstrap contract](../technical-reference/longitudinal-tmle.md#the-full-refit-bootstrap)
states what each replicate resamples and refits. The registered study measures the percentile
interval with correctly specified cell-mean nuisances on finite binary laws, at $n = 1000$. A
design is licensed as inference only after its cells are green. Until then the summary prints
`bootstrap sd` and a percentile range, a diagnostic. `ratio` and `rmst` carry the bootstrap of
their inputs.

## Competing risks

A mapping from cause labels to absorbing outcome sequences declares competing risks. Cause,
horizon, and regimen remain separate key fields. The engine uses a cause-specific recursion while
keeping all prior competing events out of later risk sets.

## A baseline treatment and a time-to-event outcome

Use `TimeToEvent` when each unit has one follow-up time, one event code and one treatment given at
baseline. Code `0` means censored. The design bins the times onto a grid of visit times and holds
the treatment over every visit.

```python
from cleverly import TimeToEvent
from cleverly.datasets import make_point_survival

frame, truth = make_point_survival(n=1_000, seed=4, n_times=4)
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

Follow these rules for the input.

| rule | reason |
| --- | --- |
| Declare `grid=` at the visit times. Omit it only for integer times, and the grid is then `1, 2, ...` up to the largest event time | a grid read off the sample would change with the sample |
| Record a censoring time as a grid time, or as a time after the last grid time | a censoring between two grid times cannot be ordered against that interval's events, so the fit refuses it |
| Give `horizons=` and the `tau` of `rmst` as grid times | the parameter name keeps the node index, and `curve()` and `to_frame()` report the grid time in a `time` column |
| Expect $K(K+1)/2$ regressions per regimen and cause for $K$ grid times | a default grid on day-level times can hold thousands of nodes |
| Expect a risk of zero with no interval at a grid time before any follower of an arm had the event | the regression of a node with no event is zero, its maximum-likelihood hazard. That node adds no variance to the curve, so the risk and its contrasts report no interval, and the fit warns. `msm=` refuses such a cell |

An event after the last grid time leaves the unit event-free at the end of the grid. `causes=`
maps each nonzero code to a cause label. Without it, codes `0` and `1` declare one event and any
other codes declare competing causes. Declare `continuous_treatment=True` for a continuous dose,
and give each regimen as a modified treatment policy such as `Shift(0.5, cap=None)`.

On the wide layout, declare the same design with one column name: `treatment="A"` on
`LongitudinalTreatment`. The node count then comes from the outcome columns, else from
`censoring=`, else from `time_varying=`. The wide layout also takes a covariate measured after
baseline, and an end-of-study outcome after several censoring nodes. A regimen is one arm, a known
rule, a known stochastic policy, or a modified treatment policy, all at baseline. A plan that
changes the treatment after baseline is refused, because the design holds one decision.
[One baseline treatment held over the nodes](../technical-reference/longitudinal-tmle.md#one-baseline-treatment-held-over-the-nodes)
gives the contract and its evidence.

## Fit the sequentially doubly robust estimator

Pass `method="sdr"` to `estimate()` to fit the sequentially doubly robust estimator. It reports
the same parameters as the default method. It stays consistent when, at each node, either the
mechanism or the regression of the node's transform is consistent.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal

frame, _ = make_longitudinal(n=300, seed=0)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
result = study.identify(RegimeContrast({"always": 1, "never": 0}, reference="never")).estimate(
    method="sdr",
    outcome_learner=LogisticRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(),
    censoring_learner=LogisticRegression(),
    n_folds=2,
    random_state=0,
    simultaneous=False,
)
print(result.summary())
```

Give `pseudo_learner` a learner of a real-valued target. A transform can leave $[0, 1]$, so a
level can leave the outcome range too. The fit then warns with `ParameterSpaceWarning`, and
`ratio()` refuses a scale that is undefined at that level.
[The sequentially doubly robust estimator](../technical-reference/longitudinal-tmle.md#the-sequentially-doubly-robust-estimator)
states the contract and what differs from the default method.

## Estimate an incremental intervention over time

Pass `IncrementalEffect` to `identify()` and `method="ipsi"` to `estimate()`. Each
`Incremental(delta)` multiplies every unit's odds of treatment by `delta` at every node.
`Incremental(1.0)` is the natural course, and the example uses it as the reference.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, IncrementalEffect, LongitudinalTreatment
from cleverly.datasets import make_longitudinal
from cleverly.interventions import Incremental

frame, _ = make_longitudinal(n=300, seed=0)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
effect = study.identify(IncrementalEffect([Incremental(1.0), Incremental(2.0)]))
result = effect.estimate(
    method="ipsi",
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(),
    censoring_learner=LogisticRegression(),
    n_folds=2,
    random_state=0,
    simultaneous=False,
)
assert result.config.estimator == "ipsi"
assert list(result.estimates) == ["ate_regimen[odds x2 vs natural course]"]
print(result.summary())
```

The `ipsi` method fits the one-step estimator of Kennedy (2019), with the dropout weights of Kim,
Kennedy and Naimi (2021). It needs no treatment positivity at a tilt node. It is not doubly
robust, so give `treatment_learner` a learner that can fit the mechanism at every node.
`available_methods()` lists `tmle` and `sdr` as unavailable for an estimated tilt, with the
reason.

| task | how |
| --- | --- |
| fit the engine directly | `IPSI([Incremental(1.0), Incremental(2.0)], ...)` from `cleverly.longitudinal` |
| use one tilt per node | `DynamicRegimen("ramp", (Incremental(1.5), Incremental(2.0)))` |
| tilt one node and fix another | `DynamicRegimen("tilt then stop", (Incremental(2.0), 0))` |
| declare a known treatment mechanism | pass `treatment_probabilities=` to the design. The tilt is then a known policy |

A level can leave the outcome range, because the estimate is an average of influence-function
values. The fit then warns with `ParameterSpaceWarning`. A risk curve can also be non-monotone
across horizons. A mean of a non-absorbing outcome $Y_t$ is the end-of-study estimand on the panel
truncated after node $t$. Earlier $Y_s$ are covariates there (Kim et al. 2021, Theorem 3.1).
`IPSI` is Kennedy's odds tilt, and `lmtp::ipsi` is the risk-ratio tilt `RiskRatioTilt`.
[Incremental interventions over time](../technical-reference/longitudinal-tmle.md#incremental-interventions-over-time) states
the contract.

## Longitudinal MSM projections

`MSMProjection` can project regimen-specific longitudinal means onto a declared working model. The
regimen grid must identify the coefficient vector; a rank-deficient working design is refused.

A longitudinal MSM projection cross-fits by default, as every longitudinal fit does. Each fold
runs an untargeted recursion for every regimen, and one stacked fluctuation per node pools the
update. [MSM projections](../technical-reference/msm-projections.md#the-longitudinal-projection)
states the construction and its evidence.

## Diagnostics

Longitudinal results provide cumulative support, targeting-score, and nuisance-model reports by
node. Call `result.diagnostics.support()` for leverage and truncation by stage.

The nuisance report separates four roles. Each fitted treatment and censoring model appears once
per node. Outcome and pseudo-outcome models appear per fitted regimen, cause, horizon, and node.
Each row labels its loss as `out_of_fold` or `in_sample`.

The report uses weighted negative log likelihood for treatment and censoring. It uses weighted
Brier loss for a binary final target and weighted mean squared error otherwise. Earlier
pseudo-outcomes use weighted mean squared error. A complete-data fit records why no censoring
learner appears in `nuisance.omissions`.

Use an explicit grid to describe point-estimate movement across cumulative mechanism bounds:

```python
curve = result.diagnostics.truncation_curve(bounds=[0.01, (0.05, 0.95)])
```

A scalar `b` means `(b, 1)`, not the point-treatment shorthand `(b, 1 - b)`. The pair bounds each
cumulative treatment and censoring product after multiplication. The diagnostic keeps raw
mechanism predictions fixed and reruns the complete backward recursion at every pair. It refits
every bound-dependent outcome or pseudo-outcome regression and targeting update.

The curve is descriptive. It reports no standard error, confidence interval, preferred bound, or
pass threshold. It refuses `mechanism=True`, which is a point-treatment option. It also refuses a
learner it cannot replay.
[Replay-only unavailability](../technical-reference/scope-and-refusals.md#replay-only-unavailability)
states the `random_state` rule each outcome and pseudo-outcome learner must satisfy.

[Truncation stability](../technical-reference/validation-methods.md#truncation-stability) holds the
returned frame's columns and its two score-cell counting rules.

A combined report requires both the grid and permission for expensive refits:

```python
report = result.diagnostics.run_all(
    include_refits=True,
    arguments={"truncation_curve": {"bounds": [0.01, 0.05]}},
)
```

Point-only sensitivity formulas remain unavailable unless a longitudinal derivation exists.

Tan (2025) derives population sensitivity bounds for binary, static longitudinal strategies, but
no sample estimator for them. The
[roadmap](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) records that boundary.

## Persistence

`result.save()` and `cleverly.load()` carry a longitudinal result through a round trip. The
artifact keeps the folds, the fitted mechanisms, the sequential steps, the targeting state, the
causal metadata, and the truncation replay recipe. It keeps the plan that the fit used. A result
fitted on an adapted panel keeps its `PanelLayout` and `unit_ids`, so `to_long_panel()` works on
the loaded result.

`save()` writes the result with `cloudpickle`. A rule, a policy density, a map, or a learner that
you define inline is saved by value. That covers a lambda, a closure, and a function defined in a
notebook or in `__main__`. A loaded result evaluates each of them again. An importable function
or class is saved by reference, so its module must import on the machine that loads the artifact.

A by-value object loads only under the Python minor version that saved it. Load such an artifact
with the same minor version, for example 3.13 for an artifact that 3.13 wrote.
`tests/unit/test_long_panel.py::test_policies_defined_in_main_survive_a_save_in_a_fresh_interpreter`
saves a lambda rule, a closure density, and an `each_node` rule from `__main__`. It loads them in
a new interpreter and compares the assignments and the density.

[Persistence and replayability](results-assessment.md#persistence-and-replayability) states the
shared contract for every result, including the assessment cache and the capability rows that a
restored artifact refuses. `tests/unit/test_serialization.py`'s
`test_longitudinal_result_retains_the_complete_fitted_graph_and_assessment` checks the round trip
against one cross-fitted, weighted, censored fit.
