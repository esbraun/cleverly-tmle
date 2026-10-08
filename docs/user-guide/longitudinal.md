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
| Expect a risk of zero, with an interval of width zero, at a grid time before any follower of an arm had the event | the regression of a node with no event is zero, its maximum-likelihood hazard. `msm=` refuses such a cell |

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
artifact keeps the folds, the fitted mechanisms, the sequential steps, the targeting state, and
the causal metadata, and the truncation replay recipe. [Persistence and
replayability](results-assessment.md#persistence-and-replayability) states the shared contract for
every result, including the assessment cache and the capability rows a restored artifact refuses.
`tests/unit/test_serialization.py`'s
`test_longitudinal_result_retains_the_complete_fitted_graph_and_assessment` checks the round trip
against one cross-fitted, weighted, censored fit.
