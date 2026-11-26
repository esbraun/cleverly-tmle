# Describe data over time

A longitudinal design records a treatment decided at several times (nodes), the covariates known
before each decision, censoring, and an outcome at the end or at every node. This page declares
those roles, reads a long panel with one row per person and time, and covers repeated outcomes,
clusters, survey weights and known treatment chances. [Plans and policies over
time](longitudinal-plans.md) covers what to compare, and [estimators over
time](longitudinal-estimators.md) how to fit it. Intervals, diagnostics and saving work as for any
fit, and their pages have a section for fits over time.

## Declare temporal roles

`LongitudinalTreatment` aligns treatment nodes, time-varying histories, censoring indicators, and
outcome processes. At treatment node `t`, a dynamic rule sees only history available by `t`.
For the reported influence-curve inference, the rule must be fixed before the fit and rowwise. A
callable may vectorize over the history frame, but it must not estimate a threshold from the
sample or otherwise make one row's assignment depend on other rows. The rule needs a declaration,
which [Declare a dynamic rule](longitudinal-plans.md#declare-a-dynamic-rule) shows.

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
keys. A regimen cannot look ahead or change interpretation between stages. A rule estimated from
the same sample is not a fixed rule, because its inference must account for the learning step; fit
it as a learned plan, `LearnedRegimen`, which [Declare a dynamic rule](longitudinal-plans.md#declare-a-dynamic-rule)
describes.

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
| `baseline=` | unit-level covariates, measured before the first decision. Name at least one column | the original name |
| `time_varying=` | covariates measured at that row, before its decision | `"eng@t"` |
| `censoring=` | `1` while the customer stays observed after the row's decision, `0` on the last row of a customer who left | `"active@t"` |
| `outcome=` | an end-of-study outcome, one value per customer, on any of its rows | the original name |
| `event=` | a survival event indicator on each row. A mapping of cause to column declares competing risks | `"event@t"` |
| `times=` | every decision time, in time order. Omit it when the times are numbers, dates or durations, and the reader sorts them | the node order |
| `cluster=` | an account or store identifier, constant within a customer ([Clustered units and survey weights](#clustered-units-and-survey-weights)) | `id=` of the cluster contract |
| `weights=` | a unit-level probability weight | `weights=` |
| `treatment_probabilities=`, `censoring_probabilities=` | the known mechanism, as `P(A_t = level \| past)` and `P(C_t = 1 \| past)` on each row ([Known treatment and dropout chances](#known-treatment-and-dropout-chances)). Beside `weights=`, also pass `mechanism_law=` | `"column@t"` |

These arguments take a value instead of a column name.

| argument | what it declares |
| --- | --- |
| `continuous_treatment=` | `True` when the treatment column holds a dose at every time point. The band goes in `treatment_support=`, where the wide example in [Shift a continuous dose over time](longitudinal-plans.md#shift-a-continuous-dose-over-time) maps each dose column to its band |
| `treatment_support=` | the band each dose lies in, as a `Support` |
| `dose_atoms=` | doses that many rows share exactly, such as a price of 0 |
| `weights_estimated=` | `True` when a model fitted to these same data produced the `weights=` column. It changes no number and marks the fit experimental |
| `outcome_family=` | the outcome's type: `"gaussian"` for a measured outcome or `"binomial"` for a yes-or-no one. The default `"auto"` reads it from the data |
| `eliminate=` | the competing events to prevent, with the mapping form of `event=`, as in [Ask what the risk would be had the other event been prevented](survival-competing-events.md#ask-what-the-risk-would-be-had-the-other-event-been-prevented) |

Declare exactly one of `outcome=` and `event=`. One customer trajectory is one observation for the
inference model. Repeated rows of one customer add no independent observations. Declare
`cluster=` when customers depend on each other, such as the customers of one account.

A generated name is `"column@t"`, where `t` is the node index and not the time label. The labels
stay in `data.panel`, a `PanelLayout`, which `data.panel.decision_times` lists. A rule at node `t`
reads `"column@s"` for `s <= t`. The summary of a fit on an adapted panel counts units, not
rows, and `summary(detail="full")` names the unit column, the decision times, and any column left
out at a node.
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
want the container first. The [sequential offers tutorial](../examples/sequential-offers.ipynb)
reads a long panel with a three-level offer and declared offer probabilities.

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
[Declare a dynamic rule](longitudinal-plans.md#declare-a-dynamic-rule) states.

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
never reads an earlier node instead. `tests/unit/test_long_panel.py` checks both refusals, and it
checks that `LTMLE` on the adapted panel equals the fit on the equivalent wide frame to a relative
difference of 1e-12.

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

Each refusal raises `DataError` before any learner. A message about one unit names the first such unit, and the decision time when one applies.

| the panel holds | why the reader refuses it |
| --- | --- |
| time labels that are not numbers, dates or durations, with no `times=` | the sort order of a label such as `"w10"` is not its time order. Declare `times=` in time order |
| a missing unit or time, a duplicated `(unit, time)` pair, a time that `times=` does not list | a long panel holds one row per unit and decision time. `times=` names every decision time and cannot coarsen a grid |
| a customer whose first row is after the first decision, or a gap between two rows | late entry is a different observation scheme. Intermittent observation is not censoring |
| rows after a customer left, or rows that stop with no censoring 0 and no event | a customer who is censored or has the event leaves for good. Declare `censoring=` |
| an event value on a row whose censoring is 0 | the order at a node is $A_t, C_t, Y_t$. Record an event as censoring 1 and event 1. Record a loss after an event-free interval as censoring 0 and a missing event |
| a `baseline`, `cluster` or `weights` column that varies within a customer, or is recorded only after the first row | a later value would let a rule at an earlier node read the future |
| an `outcome` that varies within a customer | an end-of-study outcome holds one value per unit. A per-row event indicator is `event=` |
| `outcome=Repeated([...])` | a long panel takes one outcome column. Declare a repeated outcome on a wide frame |
| one column in two roles, or a covariate that equals the outcome or the row's event | a rule could read the outcome under another name |
| a time-varying column with no value at a node after its first recorded node | the column would leave that node's adjustment set |
| `baseline=` with no column | the package fits no intercept-only nuisance design. The refusal has code `implementation_gap`, and [X63](../roadmap.md#x63-missing-data-intermediates-and-designs) part (c) tracks it. The wide layout refuses the same declaration |
| a treatment that never changes within a customer | that is one baseline decision held over every node. Build the wide frame and declare `treatment="A"` |
| a column named like a generated name, such as `eng@1` | the name collides with a generated column. Rename it |

The wide layout raises the same `DataError` for an event on a censored row.

### Long horizons

Most registered evidence for `LTMLE` and `SDR` covers two decision nodes. Three studies cover three to
five: [binary `LTMLE`](../technical-reference/method-evidence/long-horizon-longitudinal-tmle.md),
[three-level `LTMLE`](../technical-reference/method-evidence/long-horizon-categorical-longitudinal-tmle.md) and
[`SDR`](../technical-reference/method-evidence/long-horizon-sequentially-doubly-robust-estimation.md).
Every $L_t$ block accumulates into the history design, so a long panel adds columns
at each node and needs more customers. `markov_order=` limits each model to the most recent nodes,
as [Weight trimming, shorter histories, and node bounds](longitudinal-estimators.md#weight-trimming-shorter-histories-and-node-bounds)
describes. `times=` cannot coarsen a grid. To model fewer decisions, aggregate the frame to the
coarser decision times before you read it.

Long panels, many regimens and parallel workers also need memory. Before it fits any model,
`LTMLE`, `SDR` or `IPSI` estimates how much memory the whole fit will need and stops with an error
if that is more than the memory the machine has free. Pass `max_memory_bytes=` to the estimator,
or to `estimate()` on a study, to set the limit yourself, higher or lower. Where the system does not report its free memory, as on macOS, nothing
is checked unless you pass it. The estimate counts the arrays the package holds with simple
regression learners, so a fit that passes the check can still run out of memory with heavier
learners. [Memory](../technical-reference/longitudinal-tmle.md#memory) gives the details.

## Repeated outcomes

A repeated outcome is measured at every node and can return to zero. Declare it with
`Repeated([...])` as the `outcome`. A `RegimeMean` fit reports each plan's mean outcome at every
horizon, and a `RegimeContrast` fit, as here, reports the difference between plans.

```python
from cleverly import RegimeContrast, Repeated
from cleverly.datasets import make_longitudinal_repeated

frame, truth = make_longitudinal_repeated(n=1_500, seed=7)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome=Repeated(["Y1", "Y2", "Y3"]),
        treatment=("A1", "A2", "A3"),
        baseline=("W",),
        time_varying=((), ("L2",), ("L3",)),
        censoring=("C1", "C2", "C3"),
    ),
)
means = study.estimate(
    RegimeContrast({"never": 0, "always": 1}, reference="never"),
    outcome_learner=LogisticRegression(max_iter=1000),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
)
print(means.curve(scale="mean")[["estimand", "psi", "ci_lower", "ci_upper"]])
```

To draw the same rows over time, pass the fit to `cleverly.plotting.curve`:

```python
import cleverly.plotting

fig = cleverly.plotting.curve(means)
```

The figure shows the difference at each horizon with its interval and simultaneous band. This fit
estimates no plan's own mean, so no plan's curve is drawn; a `RegimeMean` fit draws one curve for
each plan. `cleverly.plotting.notes(fig)` says which parts of a figure are `experimental`; here the
curve and its band are.

| item | value |
| --- | --- |
| result keys | `ate_regimen[d vs ref @ t=h]` at every horizon `h` from `RegimeContrast`, as here, and `ey_regimen[d @ t=h]` from `RegimeMean` |
| `curve(scale="mean")` | one row for each reported key and horizon. It is the only scale |
| survival view | not available. A repeated outcome has no absorbing event |
| missing `Y_t` on an uncensored row | refused. Encode the loss as `C_t = 0` |
| `SDR` | fits. Each horizon is the end-of-study fit of the truncated panel |
| `IPSI` | fits. Each horizon is the end-of-study fit of the truncated panel |

`truth` holds the exact mean at each horizon, under the names the fit reports. In the generator, `W`
takes three values and every later node is binary. The fit can therefore check its curve against
`truth`.
[Repeated outcomes](../technical-reference/longitudinal-tmle.md#repeated-outcomes) gives the
estimator, its conditions, and its evidence. The
[repeated-outcome study](../technical-reference/method-evidence/repeated-outcome-longitudinal-tmle.md)
checks its intervals against exact truths. On the same 2,400 simulated datasets, given the same
treatment and censoring probabilities, the R package `ltmle` 1.3-0 gives the same estimates to
within six millionths.

The [repeated-outcome `IPSI` study](../technical-reference/method-evidence/repeated-outcome-ipsi.md)
does the same for `IPSI`. It has no comparator.

The [weekly orders tutorial](../examples/weekly-outcomes.ipynb) works a repeated outcome from start
to finish. It estimates the weekly share of customers ordering under three plans for a
free-shipping code and refits that curve with the sequentially doubly robust estimator. It also
compares doubled odds of a code with the current way of choosing who gets one.

## Clustered units and survey weights

Patients of one clinic, or customers of one account, often share a clinician, a policy or local
conditions that the covariates do not record, so their outcomes move together. Name the group
column in `cluster=`. The standard error then counts clinics rather than patients, and the fit
keeps each clinic whole in one fold when it splits the rows. In the data below, `id` is the
clinic label: 600 patients in 12 clinics of 50.

```python
from cleverly.datasets import make_longitudinal

clinics, _ = make_longitudinal(n=600, seed=3, cluster_size=50)
clinic_study = CausalStudy(
    clinics,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
        cluster="id",
    ),
)
clinic_learners = dict(
    outcome_learner=LogisticRegression(max_iter=1000),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    n_folds=2,
    random_state=0,
)
plans = RegimeContrast({"always": 1, "never": 0}, reference="never")
for variance in ("influence_curve", "jackknife"):
    clinic_fit = clinic_study.estimate(plans, cluster_variance=variance, **clinic_learners)
    difference = clinic_fit["ate_regimen[always vs never]"]
    print(variance, difference.ci, difference.reference_df, difference.support.label)
```

The default standard error gives an interval from 0.165 to 0.504 for the difference between
treating at both time points and at neither, and it reads `supported`. The third column is 10, the
degrees of freedom of the interval. Below 40 clusters the interval comes from a t distribution
with the cluster count minus 2 degrees of freedom, which widens it. Below 3 clusters the fit
reports no interval.

How few clusters an interval can rest on depends on the estimate. A simulation study measured this
difference, on a cross-fitted fit with two time points, dropout and parametric models such as these,
at every count from 4 clusters, so it reads `supported` here. With the default learner library, or
for any other estimate such as each plan's own mean from `RegimeMean`, the fit reads
`experimental` below 20 clusters.
[Few clusters](../technical-reference/inference.md#few-clusters) gives the floor for each kind of
fit, and the
[few-cluster floor study](../technical-reference/method-evidence/few-cluster-floor-cross-fitted-ltmle.md)
holds the measurements.

The second line uses `cluster_variance="jackknife"`, which refits the whole analysis once for each
clinic left out, here 12 times, and reads the standard error from the spread of those refits. Its
interval runs from 0.141 to 0.528. The few-cluster studies compared it with the default interval on
the same samples. Its intervals covered the true value no less often than the default's, within 3
percentage points, at every count they tried: from 10 clusters on a cross-fitted fit with five folds
and from 6 on a fit in sample. It still reads `experimental`, because the package has not yet marked
any jackknife setup as supported. Its justification holds only for models with a fixed set of
coefficients, so it accepts only parametric ones such as the logistic and linear models here, and
every fold must hold at least two clinics;
[Cluster jackknife](../technical-reference/inference.md#cluster-jackknife) lists every setting it
refuses.

`weights=` names a survey or sampling weight, one per person. Every model and the standard error use
it, as for a treatment given once ([data design](data-design.md#point-treatment)). Simulation
studies cover a weighted end-of-study fit
[in sample](../technical-reference/method-evidence/ordinary-weighted-end-of-study-longitudinal-tmle.md)
and
[cross-fitted](../technical-reference/method-evidence/cross-fitted-weighted-end-of-study-longitudinal-tmle.md).
When a model fitted to the same data produced the weights, such as a nonresponse model, also pass
`weights_estimated=True`. `cleverly` never sees the weight model and treats the weights as fixed
numbers, so the flag changes no number and the standard error still leaves out that model's error.
The fit then reads `experimental`.

The [discharge to readmission tutorial](../examples/discharge-to-readmission.ipynb) fits a
clustered program of two decisions at 100 care teams and compares its interval with a fit that
leaves `cluster=` out.

## Known treatment and dropout chances

In a sequentially randomized trial, the design sets each person's chance of each treatment at each
time point. A study may also know each person's chance of staying in follow-up. Pass these chances
in `treatment_probabilities=` and `censoring_probabilities=`, and the fit uses them in place of a
treatment model and a dropout model. The trial below gives the first treatment with chance 0.5
and the second with chance 0.6 to those who responded and 0.3 to the rest. Severe patients stay
in follow-up with chance 0.8 and the others with chance 0.9.

```python
smart_rng = np.random.default_rng(2)
severe = smart_rng.binomial(1, 0.5, 2_000)
first = smart_rng.binomial(1, 0.5, 2_000)
stay_chance = np.where(severe == 1, 0.8, 0.9)
stayed = smart_rng.binomial(1, stay_chance)
responded = smart_rng.binomial(1, 0.3 + 0.3 * first - 0.1 * severe)
second_chance = np.where(responded == 1, 0.6, 0.3)
second = smart_rng.binomial(1, second_chance)
chance = 0.2 + 0.2 * first + 0.15 * second + 0.2 * responded - 0.1 * severe
trial = pd.DataFrame(
    {
        "severe": severe,
        "first": first,
        "stayed": stayed,
        "responded": responded,
        "second": second,
        "stayed_2": 1,
        "recovered": smart_rng.binomial(1, chance),
        "first_0": 0.5,
        "first_1": 0.5,
        "second_0": 1 - second_chance,
        "second_1": second_chance,
        "stay_chance": stay_chance,
        "stay_2": 1.0,
    }
).astype(float)
trial.loc[stayed == 0, ["responded", "second", "stayed_2", "recovered"]] = np.nan
```

On your own trial you would already hold these columns, and the declaration is what changes.

```python
smart = LongitudinalTreatment(
    outcome="recovered",
    treatment=("first", "second"),
    baseline=("severe",),
    time_varying=((), ("responded",)),
    censoring=("stayed", "stayed_2"),
    treatment_probabilities={
        "first": {0: "first_0", 1: "first_1"},
        "second": {0: "second_0", 1: "second_1"},
    },
    censoring_probabilities={"stayed": "stay_chance", "stayed_2": "stay_2"},
)
smart_fit = CausalStudy(trial, design=smart).estimate(
    RegimeContrast({"always": 1, "never": 0}, reference="never"),
    outcome_learner=LogisticRegression(max_iter=1000),
    pseudo_learner=LinearRegression(),
    cross_fit=False,
    random_state=0,
)
smart_effect = smart_fit["ate_regimen[always vs never]"]
print(smart_effect.psi, smart_effect.ci, smart_effect.support.label)
```

Treating at both time points raises the recovery rate by 0.438 over treating at neither, with an
interval from 0.375 to 0.495, and the fit reads `supported`. The call passes only outcome models,
and `smart_fit.summary()` reads "known (declared)" on its treatment model and censoring model lines.

Each treatment maps its levels to columns that hold, on each row, the chance of that level given
the person's past. Each dropout column maps to a column with the chance of staying. A time point
you leave out of either mapping gets a model as usual. No one in this trial leaves after the
second treatment. The design still takes one dropout column per time point, so `stayed_2` is 1
for everyone still in the study, with a declared chance of 1.

The fit checks the chances against the data. It refuses a declared chance of zero for the treatment
a person took, and a plan that needs a treatment whose declared chance is zero for people it covers.
It also stops when the product of a person's declared chances over time points falls outside
`g_bounds`, the limits it puts on each person's chance of following the plan; the lower end is 0.01
by default. With many time points the product falls fast, so pass a lower end below the smallest
product, such as `g_bounds=(1e-8, 1.0)`. Beside weights that differ between people, also pass
`mechanism_law=`, as [Known chances beside weights](data-design.md#known-chances-beside-weights)
explains, and
[Known node mechanisms](../technical-reference/longitudinal-tmle.md#known-node-mechanisms) lists
every rule.

The [known-mechanism study](../technical-reference/method-evidence/known-node-mechanisms-ltmle.md)
fitted a trial built like this one in 1,000 samples of 2,000 rows, with every chance declared and
the fit in sample. Its interval for the difference covered the truth in 0.948 of the samples, and it
gave the same estimates as R `ltmle` given the same chances. The
[sequential offers tutorial](../examples/sequential-offers.ipynb) declares a three-level offer at
three time points beside an estimated consent model.
