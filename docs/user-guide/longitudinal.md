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
| `baseline=` | unit-level covariates, measured before the first decision. Name at least one column | the original name |
| `time_varying=` | covariates measured at that row, before its decision | `"eng@t"` |
| `censoring=` | `1` while the customer stays observed after the row's decision, `0` on the last row of a customer who left | `"active@t"` |
| `outcome=` | an end-of-study outcome, one value per customer, on any of its rows | the original name |
| `event=` | a survival event indicator on each row. A mapping of cause to column declares competing risks | `"event@t"` |
| `times=` | every decision time, in time order. Omit it when the times are numbers, dates or durations, and the reader sorts them | the node order |
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

The wide layout refuses the same event on a censored row. `LongitudinalData.from_frame` used to
drop it without a message, and it now raises `DataError`.

### Long horizons

Most registered evidence for `LTMLE` and `SDR` covers two decision nodes. Three studies cover three to
five: [binary `LTMLE`](../technical-reference/method-evidence/long-horizon-longitudinal-tmle.md),
[three-level `LTMLE`](../technical-reference/method-evidence/long-horizon-categorical-longitudinal-tmle.md) and
[`SDR`](../technical-reference/method-evidence/long-horizon-sequentially-doubly-robust-estimation.md).
Every $L_t$ block accumulates into the history design, so a long panel adds columns
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
states what each replicate resamples and refits. The percentile interval carries the support label
of the fit. A registered support row decides the label. The `bootstrap` kind of the row covers an
in-sample end-of-study or survival fit with static regimens. Any other fit prints
`[experimental]`.

The registered study measures correctly specified cell-mean nuisances on finite
binary laws, at $n = 1000$. A fit with fewer than 20 clusters prints `bootstrap sd` and a
percentile range, a diagnostic. `ratio` and `rmst` carry the bootstrap of their inputs.

### Bootstrap for sparse regimens

Three registered studies measured this feature at three to five decisions: [binary `LTMLE`](../technical-reference/method-evidence/long-horizon-longitudinal-tmle.md), [three-level `LTMLE`](../technical-reference/method-evidence/long-horizon-categorical-longitudinal-tmle.md) and [`SDR`](../technical-reference/method-evidence/long-horizon-sequentially-doubly-robust-estimation.md).
The label of a fit follows the support rows `G46`. A fit outside those rows prints `experimental`, and the label makes no coverage claim.

Use the refit bootstrap for `LTMLE` or `SDR` when few units follow a regimen, or when the
influence curve has heavy tails. The Wald interval reads its variance from those few rows. The
interval then under-covers, because the variance estimate is noisy and the sampling distribution is
skewed.

A Wald fit warns with `SparseInfluenceWarning` when this happens. A fit with a percentile
bootstrap warns too, because the percentile interval keeps the skew. A fit with BCa does not warn.
The bootstrap is opt in. A fit never turns it on.

| option | what it does | label today |
| --- | --- | --- |
| `n_bootstrap=B` | refits the whole estimator on `B` resamples and attaches an interval to each estimate | `supported` only for a covered fit, otherwise `experimental` |
| `bootstrap_interval="percentile"` | takes the quantiles of the `B` draws. This is the default | as above |
| `bootstrap_interval="bca"` | reads bias-corrected and accelerated limits from the draws. It needs `n_bootstrap` | `supported` for a single-fold fit that matches a long-horizon row, otherwise `experimental` |
| `simultaneous=True` | adds a studentized bootstrap sup-t band to the fit, for two or more estimates | `supported` for a three-level `LTMLE` fit at five decisions that matches `G46.tmle.categorical_k5`, otherwise `experimental` |
| `result.diagnostics.influence()` | reports how few rows carry each influence curve. Call it directly: `run_all()` does not include it | a diagnostic, with no label |

Pass these options to `LTMLE` or `SDR`. The `CausalStudy.estimate` route does not accept
`bootstrap_interval`.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE

frame, _ = make_longitudinal(n=300, seed=0)
result = LTMLE(
    {"always": 1, "never": 0},
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    n_folds=1,
    n_bootstrap=20,
    bootstrap_interval="bca",
    simultaneous=True,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
print(result.summary())
print(result.bootstrap_band.critical_value)
```

The example uses 20 resamples to run fast. Use at least 199 in an analysis. With `B` resamples a
BCa limit is an order statistic of the draws, so a small `B` gives a coarse limit.

**Percentile or BCa.** The percentile interval takes the 2.5% and 97.5% quantiles of the draws. It
assumes the draws are symmetric about the estimate. The BCa interval shifts both limits by a bias
correction, from the share of draws below the estimate, and by an acceleration, from the skew of
the influence curve. Choose BCa when `diagnostics.influence()` reports a large skewness. For a
symmetric curve the two intervals are close.

**The sup-t band.** Pass `simultaneous=True` with `n_bootstrap` to get `result.bootstrap_band`.
Each replicate records its own standard error. The band takes the largest studentized deviation
across the estimates in each replicate. Its critical value is a high order statistic of those
maxima.

The band is $\hat\psi_j \pm c^* \widehat{SE}_j$ for each estimate. The multiplier band in
`result.simultaneous` does not change. The summary lists both.

**The influence report.** `result.diagnostics.influence()` returns one row for each estimate.

| column | meaning |
| --- | --- |
| `kish` | the effective number of rows that carry the curve, $n / \hat\kappa$. With `id=` the unit is the cluster, and $n$ counts clusters |
| `skewness`, `kurtosis` | the third and fourth standardized moments of the curve. A normal curve has kurtosis 3 |
| `variance_rel_sd` | the relative standard deviation of the variance estimate |
| `edgeworth_loss` | the Wald coverage shortfall that the second-order Edgeworth term predicts. A positive value is a loss and a negative value is a predicted excess. The flag does not read this column |
| `bca_z0`, `bca_acceleration` | the BCa bias correction and acceleration. They are `nan` without a bootstrap |

The warning fires when `variance_rel_sd` is above 0.125, which is close to a Kish count of 64. The
warning lists the relative SD of each flagged row, with its Kish count and Edgeworth loss. A
flagged row can show a negative loss, because the skew term can predict an excess while the noisy
variance estimate still triggers the flag. The
X39 work set that threshold after it read the pilot runs, so it is not a pre-registered rule.
A cross-fitted fit, a hazard-model fit and an isotonic fit warn too, but the message does not
advise the bootstrap. For the cross-fitted fit the bootstrap over-covers, and the roadmap owner
`X20-bootstrap` in the roadmap holds that result.

**Cost.** Each resample refits every nuisance. A fit of 64,000 rows and five decisions takes about
0.2 seconds for each resample on the fast path, a development measurement that no committed
artifact records yet. The X39 studies declare `B = 199`, so one such fit costs about 40 seconds of
bootstrap time.

The fast path fits the distinct rows with the resample counts as weights. It
applies to an in-sample, unclustered, unweighted fit with declared treatment and censoring
mechanisms, categorical treatment, and learners that declare `frequency_exact`. Any other fit
refits each materialized resample, which costs much more. Set `n_jobs` to run resamples in
parallel.

**Evidence.** The three long-horizon studies fit the bootstrap with `B = 199`. Each study gates a BCa interval only where its Wald interval does not gate, that is where $n$ is below twice the exact-law tail size. The three-level `LTMLE` study gates the sup-t band over six plans at five decisions.

The `SDR` bootstrap band, the cross-fitted bootstrap and an MSM fit stay `experimental`. The studies carry Python property evidence. The R comparators (`ltmle` and `lmtp`) are deferred to [X59](../roadmap.md#x59-long-horizon-comparator-parity-for-ltmle-and-sdr). No study gates the warning threshold.

## Repeated outcomes

A repeated outcome is measured at every node and can return to zero. Declare it with
`Repeated([...])` as the `outcome`. The fit reports the mean of the outcome at every horizon.

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

| item | value |
| --- | --- |
| result keys | `ey_regimen[d @ t=h]` and `ate_regimen[d vs ref @ t=h]` at every horizon `h` |
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
checks its intervals against exact truths. Its comparison with R `ltmle` 1.3-0 is pending.

The [repeated-outcome `IPSI` study](../technical-reference/method-evidence/repeated-outcome-ipsi.md)
does the same for `IPSI`. It has no comparator.

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

## A survival curve that never decreases

The default survival fit targets each horizon in its own backward pass, so its curve can decrease
by a small amount. Two options give a non-decreasing curve.

| option | what it does | what it needs |
| --- | --- | --- |
| `survival_model="hazard", targeting="one_step"` | fits the cause-specific hazard at every node, and targets one set of hazards for every horizon together | one categorical treatment at baseline, held over every node, and no covariate after baseline |
| `isotonic=True` | projects each fitted curve and its band limits onto a non-decreasing curve | any survival or competing-risk fit |

Pass the hazard learner as `outcome_learner`. A hazard fit does not use `pseudo_learner`.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import CausalStudy, RegimeMean, TimeToEvent
from cleverly.datasets import make_point_survival

frame, truth = make_point_survival(n=1_000, seed=4, n_times=4)
study = CausalStudy(
    frame,
    design=TimeToEvent(
        time="time", event="event", treatment="A", baseline=("W1", "W2"), grid=(1, 2, 3, 4)
    ),
)
hazard_curve = study.estimate(
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

The risks of each arm do not decrease over the four grid times. Call `summary()` to see the walk: the
parameter count, the step count, the largest relative score, and whether the walk converged.

`targeting="iterative"`, the default, targets the hazards at each horizon on its own. That curve is
not guaranteed to be monotone.

To keep the iterated-mean fit and project its curve, pass `isotonic=True`.

```python
projected = study.estimate(
    RegimeMean({"arm1": 1, "arm0": 0}),
    isotonic=True,
    outcome_learner=LinearRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    cross_fit=False,
)
print(projected.isotonic_moves)
```

`isotonic_moves` lists each level the projection moved. On this sample it is empty, because the
fitted curve already does not decrease. The projection keeps every influence curve and standard
error. A pointwise interval at a moved level is valid only where the true curve strictly
increases. [The isotonic projection](../technical-reference/longitudinal-tmle.md#the-isotonic-projection)
states each claim and its condition.

The hazard model refuses a treatment at several nodes, a covariate after baseline, a continuous
treatment, and a modified treatment policy. Each message names the iterated-mean route.
[Hazard-based survival targeting](../technical-reference/longitudinal-tmle.md#hazard-based-survival-targeting)
gives the contract, both algorithms, and the evidence. No registered study measures the coverage
of a hazard fit or a projected fit yet.

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

## Report the sequential baselines

`SequentialGComputation` and `SequentialIPW` estimate the regimen levels and contrasts that
`LTMLE` estimates. Report them beside an `LTMLE` or `SDR` fit as reference estimators.

| estimator | the estimate reads | use it to |
| --- | --- | --- |
| `SequentialGComputation` | the outcome and pseudo-outcome regressions | check the targeted estimate against the plug-in |
| `SequentialIPW` | the treatment and censoring mechanisms | check the targeted estimate against the weighted mean |

```python
from cleverly.datasets import make_longitudinal
from cleverly.learners import GLM
from cleverly.longitudinal import SequentialGComputation, SequentialIPW

frame, _ = make_longitudinal(n=300, seed=0)
columns = dict(
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
regimens = {"always": 1, "never": 0}
gcomp = SequentialGComputation(regimens, reference="never", outcome_learner=GLM()).fit(
    frame, **columns
)
ipw = SequentialIPW(regimens, reference="never", treatment_learner=GLM()).fit(frame, **columns)
print(gcomp.summary())
print(ipw.summary())
```

`gcomp()` and `ipw()` are the one-call forms. Both take the same columns and `regimens=`.

The interval of each baseline holds only when every `GLM` that the estimate reads is correctly
specified. Under misspecification the interval covers the limit of the misspecified fit, not the
target. A targeted estimator has no such condition on one of its two models.

When a regression that the estimate reads uses a data-adaptive learner, the result reports the
status `adaptive_baseline_plugin`. It gives the point estimate only. The standard error is a
diagnostic under `plugin_std_error`. Fit `LTMLE` or `SDR` when you need an interval with
data-adaptive learners.

The answers can differ from `LTMLE` for three reasons.

| reason | effect |
| --- | --- |
| g-computation applies no fluctuation | its estimate reads the outcome regressions alone |
| IPW uses no outcome regression | its estimate reads the mechanisms alone, and a large weight moves it |
| both baselines fit `GLM` models by default | `LTMLE` fits a Super Learner by default |

With the same outcome learner, the g-computation estimate equals the untargeted first step of
`LTMLE`. [Sequential g-computation and IPW](../technical-reference/sequential-baselines.md) gives
the estimators, the stacked influence curve, the settings, and the refusals. The
[sequential offers tutorial](../examples/sequential-offers.ipynb) fits g-computation and `LTMLE`
with the same misspecified outcome model on one campaign.

## Estimate structural blips

`StructuralNestedMean` fits a linear structural nested mean model. Use it to report the effect of
each treatment node on the end-of-study outcome, and to let that effect vary with declared
features. It is a separate method. `LTMLE` and `SDR` fit no model for the effect.

| you want | fit |
| --- | --- |
| the effect of node $t$ with later treatment at the reference, as a coefficient matrix | `StructuralNestedMean` |
| the mean outcome under a static regimen, with no model for the effect | `LTMLE` or `SDR` |
| the value of a rule, a policy or an incremental intervention | `LTMLE`, `SDR` or `IPSI` |

Declare the features of each node in `blip_features`. A key is a node counted from one, or its
treatment column. A value lists columns. A tuple of columns declares a product. The intercept is
always a feature. A feature may read the baseline columns, the covariates through its node, and
the earlier treatments.

```python
from cleverly.datasets import make_longitudinal
from cleverly.learners import GLM
from cleverly.longitudinal import StructuralNestedMean

frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
result = StructuralNestedMean(
    {"always": 1, "never": 0},
    reference="never",
    blip_features={1: ["W1"], 2: ["W1", "A1"]},
    outcome_learner=GLM(family="binomial"),
    treatment_learner=GLM(family="binomial"),
    n_folds=2,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
)
print(result.summary())
print(result.blip(2))
```

`snmm()` is the one-call form. `result.blip(node)` returns the coefficients of a node as a matrix
with one row per feature and one column per coding component. It is the transpose of the matrix
$\Psi_t$ of the
[technical page](../technical-reference/structural-dynamic-effects.md#the-model). A categorical node has one column for each level
after the first sorted level. A continuous node has one column, the dose. The regimen levels and
contrasts follow from the blips and the mean outcome at the reference level `ey_reference`.

Pass `regression_learner=` when the outcome is binary and a node holds a dose or a feature is
continuous. A classifier takes no target outside $\{0, 1\}$, and `regression_learner` fits those
targets. It defaults to `outcome_learner`.

Read the result with this condition. The intervals are valid under the linear structural nested
mean model with the declared blip features. Every row of the summary and of `to_frame()` carries
that label. A blip that depends on an undeclared earlier treatment or covariate makes the
coefficients answer a different question.

The evidence label is "comparator-backed", against EconML 0.17.0 `DynamicDML`. The
[structural nested mean study](../technical-reference/method-evidence/structural-nested-mean.md)
covers cross-fitted fits with `GLM` nuisances on binary, three-level and dose nodes. It also
covers known weights and 40 or more clusters. A fit inside those rows reads `supported`. Any other fit, such as an in-sample fit, reads `experimental`. The summary
prints a `support` column and the reason.

Check the model against `LTMLE`.

```python
check = result.check_model()
print(check.summary())
```

`check_model()` fits `LTMLE` on the same rows and folds. A cross-fitted check refits both
estimators on `n_splits` seeded splits (default 5) and averages the difference. It reports a joint
Wald statistic on the difference of the contrasts. A rejection says that the model fails for these contrasts. A
non-rejection does not confirm the model.

| the fit | what it reports |
| --- | --- |
| `n_folds > 1` | the influence-curve intervals |
| `n_folds=1` and every learner a `GLM` | the influence-curve intervals, valid when every `GLM` is correct |
| `n_folds=1` and another learner | the point estimate, with a diagnostic standard error |

The estimator refuses a rule, a policy, censoring, survival and repeated outcomes. It refuses a
static level, a contrast and `check_model` when a feature reads a time-varying covariate.
[Structural dynamic effects](../technical-reference/structural-dynamic-effects.md#refusals) lists
each refusal and its alternative.

## Simulate the parametric g-formula

`ParametricGFormula` estimates the same regimen levels and contrasts as `LTMLE`. It fits a model for
each time-varying covariate and for the outcome. It then simulates each unit forward under each
regimen and averages the simulated outcomes. `SequentialGComputation` reaches the same target by
regression backward through the nodes.

| reason | choose |
| --- | --- |
| you need the forward-simulation form that `gfoRmula` and `pygformula` report, with an analytic interval | `ParametricGFormula` |
| a plan fixes a continuous dose at a node, or sets it by a known rule | `ParametricGFormula`. `LTMLE` and `SequentialGComputation` refuse both |
| you doubt the law of a time-varying covariate | `SequentialGComputation`, `LTMLE` or `SDR`. They model no covariate law |
| you want a fit that a misspecified model of one kind cannot bias | `LTMLE` or `SDR` |

The parametric g-formula needs a correct model for every covariate and for the outcome. It reads
more models than the backward form, and each one must be right.

```python
from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import ParametricGFormula

frame, _ = make_longitudinal(n=300, seed=0)
result = ParametricGFormula(
    {"always": 1, "never": 0},
    reference="never",
    covariate_types={"L2": "normal"},
    n_draws=20,
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
print(result.summary())
print(result.simulation.monte_carlo_share)
```

`gformula()` is the one-call form. The fit refuses an undeclared time-varying column that is not
coded 0/1, before any learner. Declare it in `covariate_types=` as `"binary"`, `"categorical"` or
`"normal"`.

The interval includes the simulation error. `result.simulation.monte_carlo_share` gives the share
of each variance that the simulation contributes, and the fit warns above 10%. Raise `n_draws=` to
lower it. A fit with the same `random_state=` repeats exactly.

A data-adaptive learner for any model that the estimate reads gives the status
`adaptive_baseline_plugin`: the point estimate only. A fixed number at a continuous treatment node
is a fixed dose, and a `DynamicRegimen` rule with `rule_kind="known"` there sets the dose from the
simulated history. The result then states its model reliance and reports a support diagnostic. No
published analytic variance exists for the simulation form, so the interval is a natural
extension. [Parametric g-formula](../technical-reference/parametric-gformula.md) gives the
estimator, the argument, the settings, the refusals and the validation.

## Estimate an incremental intervention over time

Pass `IncrementalEffect` to `identify()` and `method="ipsi"` to `estimate()`. Each
`Incremental(delta)` multiplies every unit's odds of treatment by `delta` at every node.
`Incremental(1.0)` is the natural course, and the example uses it as the reference. A node with
three or more levels takes one factor for each level, such as
`Incremental({"high": 2.0, "low": 0.5})`. A level that the mapping does not name keeps the
factor 1.

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
| tilt a node with three levels | `Incremental({"high": 2.0, "low": 0.5})`. A scalar there is refused before any learner, and so is a key that is not a level of the node |
| declare a known treatment mechanism | pass `treatment_probabilities=` to the design. The tilt is then a known policy |

A level can leave the outcome range, because the estimate is an average of influence-function
values. The fit then warns with `ParameterSpaceWarning`. A risk curve can also be non-monotone
across horizons. A mean of a non-absorbing outcome $Y_t$ is the end-of-study estimand on the panel
truncated after node $t$. Earlier $Y_s$ are covariates there (Kim et al. 2021, Theorem 3.1).

A fit with a node of three or more levels carries the label of the registered study
[categorical incremental policies](../technical-reference/method-evidence/categorical-incremental.md)
when it matches that study's rows. Any other such fit is labelled `experimental`. `IPSI` is Kennedy's odds tilt, and `lmtp::ipsi` is the risk-ratio tilt `RiskRatioTilt`.
[Incremental interventions over time](../technical-reference/longitudinal-tmle.md#incremental-interventions-over-time) states
the contract.

## Longitudinal MSM projections

`MSMProjection` can project regimen-specific longitudinal means onto a declared working model. The
regimen grid must identify the coefficient vector; a rank-deficient working design is refused.

A longitudinal MSM projection cross-fits by default, as every longitudinal fit does. Each fold
runs an untargeted recursion for every regimen, and one stacked fluctuation per node pools the
update. [MSM projections](../technical-reference/msm-projections.md#the-longitudinal-projection)
states the construction and its evidence.

## Repeat the fold draws

Pass `repeats=R` to `estimate()` to fit the whole estimator at `R` independent fold draws. The
result reports the median of the draws, and its variance adds the spread of the draws. Use it when
one split moves the estimate by more than you accept.

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
repeated = study.identify(RegimeContrast({"always": 1, "never": 0}, reference="never")).estimate(
    outcome_learner=LogisticRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(),
    censoring_learner=LogisticRegression(),
    n_folds=2,
    repeats=3,
    random_state=0,
)
assert repeated.n_repeats == 3
assert len({draw.seed for draw in repeated.repeats}) == 3
print(repeated.repeat_spread())
```

| task | how |
| --- | --- |
| read one draw | `repeated.repeats[r]` holds its folds, seed, fits and estimates |
| see how much the split moves each estimate | `repeated.repeat_spread()`, the standard deviation of the draws |
| contrast or ratio the levels | `contrast()` and `ratio()` run in every draw and report the median |
| set it on a method | `CrossFitting(repeats=3)` |

The fit refuses `repeats` above 1 with `n_folds=1`, because one fold has no split to draw again. It
also refuses `repeats` above 1 with `n_bootstrap` above 0. `SequentialGComputation` and
`SequentialIPW` refuse `repeats` above 1, because they fit in sample. A fit with `repeats` above 1
carries the `experimental` label until a registered study covers its configuration. With
`isotonic=True` each draw projects its levels and the fit builds no simultaneous band.

The rows of the support and nuisance reports describe draw 1. The support warning of `validate()` grades every draw and names the draw.
The score report has one row per node and draw. A repeated outcome, `Repeated([...])`, takes
`repeats` too.
[Repeated fold draws](../technical-reference/longitudinal-tmle.md#repeated-fold-draws) states the
rule, the seeds and the refusals.

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
