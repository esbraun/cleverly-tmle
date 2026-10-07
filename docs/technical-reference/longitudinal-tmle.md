# Longitudinal TMLE

## What this solves

Treatment is given more than once, and what happens between the doses matters. A patient's lab
value at month three both responds to the first dose and decides the second. Adjusting for that lab
value blocks the confounding of the second dose and also blocks part of the first dose's effect.
Not adjusting for it leaves the second dose confounded. No single regression can do both.

This is time-varying confounding, and it is the reason a longitudinal question is not a
point-treatment question with more columns. Longitudinal TMLE estimates the mean outcome under a
treatment plan followed at every node, by iterating a regression backward through the nodes.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| repeated treatment with time-varying confounders | the mean outcome under a plan, identified by the g-formula and estimated as a plug-in | one regression per node per regimen, and positivity is now a statement about a *cumulative* product |
| units drop out over time | censoring enters the same cumulative product as treatment, and each node's regression uses only its uncensored followers | a censoring model per node |
| the plan depends on the history | fixed, rowwise dynamic rules receive the history available at their node, and no static plan can express them | the fixed rule is part of the estimand. A rule learned from the same sample needs additional inference that this estimator does not provide |
| the plan draws the arm at random | a known policy $q_t(a \mid H_t)$ at a node, with the integrated influence curve | the policy is part of the estimand and must be fixed before the fit |
| you want a survival curve | the same recursion, seeded at the horizon, reports cumulative risk at each horizon you name | each horizon is its own backward pass. The cost is quadratic in the node count |
| several causes of failure compete | cause-specific cumulative incidence, with the competing causes left alone | that is a *total* effect. Eliminating the competing event is a different question, and is refused by name |
| you want the effects summarised across regimens | a working model over regimen and horizon cells | see [MSM projections](msm-projections.md) |

Reach for [point-treatment TMLE](point-treatment-tmle.md) when the exposure is measured once.
Collaborative TMLE and DR-TMLE have no longitudinal derivation, and `available_methods()` says so
before any model is fitted.

A worked applied analysis is in the
[longitudinal tutorial](../examples/longitudinal-tmle.ipynb). It runs a point-treatment analysis
of the same data as a control, and that analysis fails in both available directions.
[Time-to-event outcomes](../examples/longitudinal-survival.ipynb) is the companion tutorial for
the event-process extensions below.

## The algorithm as implemented

### End-of-study regimen means

Let $L_t$ be the time-varying history, $A_t$ the treatment, $C_t$ the censoring indicator, and $Y$
the final outcome. For a regimen $g^*$, sequential regression works backward from $Q_{T+1}=Y$:

$$
Q_t(h_t)=E\{Q_{t+1}(H_{t+1})\mid H_t=h_t,\ A_t\sim g_t^*,\ C_t=0\}.
$$

The target is $\psi_{g^*}=E\{Q_0(W)\}$. Its efficient influence function is a telescoping sum

$$
D(P)(O)=\sum_{t=0}^{T} H_t(P)(O)\{Q_{t+1}(O)-Q_t(O)\} + Q_0(W)-\psi_{g^*},
$$

where $H_t$ is the cumulative product of the regimen-to-observed treatment and uncensoring density
ratios through node $t$.

Each regression is fitted on the units that followed the plan and stayed under observation through
$t$. It predicts for those that did so through $t-1$, which are *exactly* the units the previous
step is fitted on. That is what makes the recursion close, and a test asserts the two masks are the
same set rather than leaving it to be read off a paragraph.

The untargeted substitution estimator does not generally solve the efficient influence-curve
equation. `LTMLE` therefore gives each node a loss-weighted logistic intercept submodel,

$$
\operatorname{logit} Q_t(\epsilon) = \operatorname{logit} Q_t + \epsilon ,
$$

fitted with loss weight $H_t$. Its score is the $t$-th term of the sum above, so solving all $T$ of
them makes the fit solve $P_n D = 0$. The recursion carries the **targeted** prediction forward,
not the initial one, so a residual left by one node is regressed away by the next instead of
accumulating.

Bang and Robins (2005) supplies the sequential-regression foundation. Van der Laan and Gruber
(2012) gives longitudinal TMLE for multiple intervention points. Chaffee and van der Laan (2012)
covers fixed rowwise dynamic rules. See the
[longitudinal references](../references.md#longitudinal-survival-and-marginal-structural-models).
Implementation:
[`longitudinal/sequential.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/sequential.py),
[`longitudinal/regimen.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/regimen.py),
and
[`longitudinal/estimator.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/estimator.py).

### Finite-sample choices that are part of the algorithm

Four choices are algorithm rather than notation. Each one changes the finite-sample answer, and no
exact law can see any of them, because every fluctuation coefficient is zero at the truth.

| choice | what `cleverly` does | what the alternative would be |
| --- | --- | --- |
| where the bound is applied | the raw treatment and censoring factors are multiplied first, and each cumulative prefix is then truncated | bounding each factor before multiplication, which is a different regularisation |
| where the loss weight sits | $H_t$ is the loss weight of the logistic submodel | putting $H_t$ in the submodel instead, which solves the same score along a different path |
| what the recursion carries | the targeted prediction | the initial prediction, which accumulates residuals |
| how a categorical node is read | one probability per declared level, selected by the raw label assigned on that row | a binary complement shortcut, which is wrong above two arms |

Each node owns its own level set and dense encoding, so labels and level counts may differ over
time. History designs contain the dense codes. Plans, settings, and errors retain your labels.
Every mechanism training fold must contain every observed level. A missing level is refused before
fitting, because probability-column alignment cannot identify an arm that is absent from that
training law.

The split reads no treatment, so it cannot deliver that property. The fit checks the realized draw
instead. `preflight_mechanism_support` runs once, over every node, before the first learner. It
checks the first node at any level count under cross-fitting, and a later node from three levels
up.

`g_bounds` defaults to the explicit fixed pair `(0.01, 1.0)`, matching R `ltmle`. It is a
**heuristic convention**. It is not an automatic procedure and not a derived rate. It does not
depend on the row count, the effective sample size, the fitted probabilities, or the follow-up
depth. A cumulative path probability shrinks with depth, so falling below `0.01` does not by itself
prove a node-level positivity failure. Clipping can also replace every scored row and make the
clever covariate constant. `res.diagnostics.support().to_frame()["share_truncated"]` reports the
effect per regimen and node.

`res.diagnostics.truncation_curve(bounds=...)` repeats this finite-sample choice over an explicit
grid. A scalar `b` resolves to the cumulative lower-only pair `(b, 1)`. An explicit
`(lower, upper)` pair keeps both limits. It never uses the point-treatment scalar convention
`(b, 1 - b)`.

Each grid point keeps the fitted treatment and censoring predictions fixed. It recomputes their
bounded cumulative prefixes and reruns the complete backward recursion. It refits every
bound-dependent outcome or pseudo-outcome regression and every targeting update. Reusing an
earlier-node prediction would change the next regression's response and is outside this contract.

The result is a descriptive ordinary LTMLE point estimate at each fixed bound. It changes the
finite-sample estimator, not the causal estimand. It provides no standard error, interval,
preferred bound, selection correction, or positivity verdict. The
[source audit](../references.md#longitudinal-survival-and-marginal-structural-models) records the
algorithm provenance and the boundary of that claim.

This replay keeps the realized data, resolved plans, folds, weights, clusters, and parameter
structure fixed. A clustered replay keeps the whole-cluster folds of its fit, and its variance
stays cluster-summed.

A cross-fitted replay rebuilds the out-of-fold mechanism pair at each bound. The pooled
fluctuation divides by that pair alone, and the untargeted fold regressions read no mechanism. The
result must retain a replay recipe, and that
recipe's outcome and pseudo-outcome learners must be cloneable.
[Replay-only unavailability](scope-and-refusals.md#replay-only-unavailability) states the
`random_state` rule each learner must satisfy, and lists every code that makes the operation
unavailable. [Truncation stability](validation-methods.md#truncation-stability) holds the frame
schema and the score-cell counting rules.

Before evaluating the requested grid, replay at the fitted pair must reproduce every retained
estimate, regimen fit, and MSM fit exactly. This preflight also runs when the grid omits that pair.
A mismatch makes the operation unavailable instead of reporting a different fitted procedure.

### Survival and competing risks

[Time-to-event outcomes](../examples/longitudinal-survival.ipynb) works this section applied.

An outcome sequence represents an absorbing event process. **Which population each node's
regression is fitted on is the whole of what changes**, and it is the one thing here that is easy
to get backwards. The recursion is seeded at the horizon with $Q_{k+1}=0$ and carries back

$$
Z_t = Y_t + (1 - Y_t)\,Q^*_{t+1},
$$

fitted on the units at risk *entering* $t$, which means event-free through $t-1$. That is one node
earlier than the censoring factor runs to. A unit that has the event at $t$ **is** in node $t$'s
regression, because it is the observation that the event happened. It is not in node $t+1$'s. So
the identity the end-of-study recursion closed on generalises rather than holds:

```text
at_risk(t + 1) == following(t) & event-free at t
```

For competing risks the pseudo-outcome carried back is

$$
Z_t = \mathbb{1}\{\text{cause } j \text{ at } t\} + \mathbb{1}\{\text{no event at } t\}\,Q^*_{t+1},
$$

a **cause-specific numerator** against an **all-cause survival factor**. A unit that left through a
competing cause contributes a zero and carries nothing forward. It is no more available to have
this cause's event than one that already had it.

`curve(scale="survival")` reports the complement of a risk curve. It mirrors a level interval and
it negates a contrast interval. A fit that declares two or more causes refuses this view, because
one minus a cause-specific incidence is not all-cause survival. A fit that declares one cause
reports the view, because the sum over the causes is that one incidence. `incidence_total()` sums
the cause-specific influence curves and reports their joint standard error.

On a fit whose status supplies no inference, `incidence_total()` names its `std_err` column
`plugin_std_err`, a diagnostic. On such a fit, `curve()` renames its three spread columns as
`to_frame()` does, and it adds an `inference` column
([inference status](inference.md#inference-status)).

Two vocabularies describe a row of the curve. Each vocabulary gets its own column.

| column | values | what it says |
| --- | --- | --- |
| `scale` | `level`, `difference` | the word `to_frame()` uses for the same idea |
| `view` | `risk`, `survival` | the view the caller requested |
| `estimand` | a parameter name | the name of the quantity the row reports |
| `parameter` | a key of the result | the estimate the row derives from |

On the survival view a level row reports the name `survival_regimen[...]`. The fit holds no
estimate under that name. Read the `parameter` column to get the estimate behind any row.

What does **not** change is the positivity story. Being event-free is part of the history and not
an intervened node, so it enters the *indicator* of the clever covariate and never its denominator.
The cumulative product is still over the $2T$ treatment and censoring factors. The causes share
every nuisance fit and differ only in what is regressed, so $J$ causes cost $J$ backward passes and
one mechanism.

Two reductions pin these as generalisations rather than as second estimators. A fit whose event can
only happen at the last node reproduces the end-of-study fit **bit for bit**. A fit declaring a
single cause reproduces a single-event survival fit **bit for bit**. Stitelman, De Gruttola and van
der Laan (2012) is the survival implementation reference.

### Known stochastic policies

A plan node can hold a known policy density instead of a label or a rule. The unit then draws its
arm at that node with probability $q_t(a \mid H_t)$. Write the node as a
`Stochastic(q, name, density_kind="known")` inside a `DynamicRegimen`:

```python
import numpy as np
import pandas as pd

from cleverly.interventions import Stochastic
from cleverly.longitudinal import DynamicRegimen


LEVELS = ["high", "low", "standard"]  # each node's levels, sorted


def first(history):
    return np.tile([0.5, 0.25, 0.25], (len(history), 1))


def second(history):
    # Draw the first node's arm again with probability one half.
    drawn = pd.get_dummies(history["A1"]).reindex(columns=LEVELS, fill_value=0)
    drawn = drawn.to_numpy(dtype=float)
    return 0.5 * drawn + 0.25 * (1.0 - drawn)


mix = DynamicRegimen(
    "mix",
    (
        Stochastic(first, "first", density_kind="known"),
        Stochastic(second, "second", density_kind="known"),
    ),
)
```

The density function receives the policy frame: the history $[W, L_1, \ldots, L_t]$ and the
earlier treatments as labels. It never receives $A_t$, a censoring indicator, or a later column.
It returns one column per level in `treatment_levels[t - 1]` order, or a dataframe whose columns
are the levels. `resolve_plans` evaluates it once, before any learner. A density that is one-hot
on every reachable row resolves as the rule it equals, and `config.policy_point_mass_nodes` names
that node.

Write $\pi_t(a \mid h)$ for the node's intervention density. The table gives each part of the
estimator at the two kinds of node.

| part | label or rule node | policy node |
| --- | --- | --- |
| $\pi_t(a \mid h)$ | $1\{a = d_t(h)\}$ | $q_t(a \mid h)$ |
| rows that stay on the plan | the rows whose observed arm is the assigned arm | the rows whose observed arm has $q_t > 0$ |
| node regression | covariate history, plus the arms of earlier policy nodes, on the rows that stay on the plan | one fit with the current arm and the arms of earlier policy nodes as columns, predicted at every level |
| value carried to node $t - 1$ | $Q^*_t(H_t)$ | $\sum_j q_t(j \mid H_t)\, Q^*_t(j, H_t)$ |
| clever covariate | $1 / \operatorname{bound}(\prod_{s \le t} g_s c_s)$ | $\prod_{s \le t} \pi_s(A_s \mid H_s) / \operatorname{bound}(\prod_{s \le t} g_s c_s)$ |
| fluctuation | intercept, offset $Q_t$ at the observed arm, loss weight $w R_t$ | the same, and every level's prediction moves by the same $\epsilon_t$ |

The influence curve is the integrated one:

$$
D^*(O) = \bar Q^*_1(H_1) - \Psi + \sum_{t=1}^{T} R_t\,\{Z_t - Q^*_t(A_t, H_t)\},
$$

with $\bar Q^*_1$ the carried value at node 1 and $R_t$ the clever covariate. The fluctuation is
the `ltmle::UpdateQ` placement. It is also the `lmtp` placement in `R/tmle.R`, where the natural
prediction is the offset and the shifted prediction takes the same coefficient.

The mechanism is evaluated at the observed history at a policy node. `g_bounds` bounds the
cumulative denominator only, with its `CalcCumG` meaning. The numerator is at most one, so
$R_t \le 1/\text{lower}$. A side effect follows: when $\prod q$ and $\prod g c$ are both below the
lower bound, the bound shrinks a moderate ratio. Take $\prod q = \prod g c = 0.001$ and a lower
bound of 0.01. The ratio is then 1 unbounded and 0.1 bounded. `share_truncated` counts denominator truncation,
so a truncated cell need not carry a large ratio.

The base result is Díaz, Williams, Hoffman and Schenck (2023). The table gives each step of the
extension and the condition it reads.

| item | statement |
| --- | --- |
| base result | Section 2, journal page 849, admits a random regime $d(a_t, h_t, \varepsilon_t)$ whose randomizer is drawn independently across units and of $U$, with a law that does not depend on $P$. The section places $\varepsilon_t$ in $L_t$. Theorem 3, page 853, then covers the estimator that records the randomizer |
| step | the integrated estimator above. Its curve is $E[D_{aug} \mid O]$, the conditional expectation of the Theorem 3 curve given the data without the randomizer. Its remainder is the conditional expectation of the Theorem 3 remainder. The Eligibility rule lists this step as the chain rule with a known weight |
| conditions met by construction | no $Q_t$ reads a randomizer; each ratio reads its own node only; the randomizers are independent across nodes; $q_t$ reads $H_t$ only |
| identification | sequential exchangeability, consistency, and positivity $g_t c_t > 0$ wherever $q_t > 0$. The policy history excludes $A_t$, so Assumption 2 suffices and Assumption 3 is not needed |
| inherited conditions | the Theorem 3 rate condition, bounded ratios, and a policy density known and fixed before the fit (`density_kind="known"`) |

The same construction runs on every composition the fit supports. The table gives each one and
its evidence.

| composition | what changes | evidence |
| --- | --- | --- |
| cross-fitting | each fold's untargeted recursion carries the fold's policy mean. The parent stitches the held-out per-level predictions, and the pooled fluctuation moves them | `tests/unit/test_pooled_longitudinal_policy_targeting.py` recomputes the stitch by hand. Three mutations fail it or the pooled score: a per-fold fluctuation, an observed-arm carry in the fold recursion, and per-level blocks moved to other rows |
| censoring, survival, competing risks | the pseudo-outcome is unchanged and composes the carried policy mean | exact-law curves on the survival and competing-risk laws in `tests/unit/test_influence_gateaux_longitudinal_policy.py` |
| observation weights | the weight enters every regression and the loss weight, never $R_t$ | the weighted exact law in the same file |
| clusters | the split, the inner groups, the cluster-summed curve and the status are target-agnostic | `tests/e2e/test_ltmle_multivalue.py` |
| `msm=` over policy cells, in sample and cross-fitted | the stacked fluctuation moves each policy cell's per-level predictions along that cell's block of the design | `tests/unit/test_longitudinal_policy_msm.py`, at one and at five folds, under the identity and logit links, and over survival cells |
| contrasts, bands, `longitudinal_truncation_curve` | none; they read the stacked curves and the frozen plans | `tests/unit/test_longitudinal_policy_plumbing.py` replays a policy fit under truncation. The `simultaneous_coverage` cells of the [registered study](method-evidence/stochastic-categorical-longitudinal-tmle.md) measure the band and its pointwise control |

A level that the policy can draw at an at-risk row must also appear among the rows the node is
fitted on. Otherwise its design column is all zero, and the regression extrapolates where no
residual of the curve can see it. `LTMLE.fit` refuses that sample before any learner, per outer
training fold.

A deterministic rule receives the history frame, which holds no earlier treatment. A rule that
must read an arm drawn at an earlier policy node, such as "continue the arm drawn at the first
node", is a one-hot `Stochastic` node that reads the policy frame. It resolves as that rule.

Two neighbouring targets are outside this section. The table names where each one is covered.

| target | why it is not a known policy | owner |
| --- | --- | --- |
| a policy that reads the natural value $A_t$, such as a modified treatment policy | its ratio reads the fitted mechanism, and identification needs Assumption 3 | [modified treatment policies at a node](#modified-treatment-policies-at-a-node) |
| a policy density that depends on $P$, such as an incremental odds tilt | page 850 records that multiply robust estimation "is not generally possible" for such a regime; the curve needs a mechanism-derivative term | [X19](../roadmap.md#x19-incremental-interventions-over-time) |

`tests/unit/test_influence_gateaux_longitudinal_policy.py` holds the exact-law evidence. The point
estimate and the curve equal the g-formula of
`tests/discrete_law_longitudinal_policy.py` and its Gateaux derivative to $10^{-10}$, on mixed
plans in both orders and on a partial-support policy.

Mutations M1 to M7 each move the estimate or the curve away from the oracle by more than
$10^{-4}$. M8 and M9 each fail the targeting witness by more than $10^{-4}$: the carried mean
against its longhand, and the solved score. A misspecified initial fit gives every node a
coefficient above 0.05 and pins the per-level update. A one-hot policy is its rule bit for bit.
On an augmented law that records a uniform randomizer, the integrated curve is the mean of the
recorded curve over the randomizer, and its variance is smaller.

The registered study `stochastic-categorical-ltmle` pairs the in-sample fit with R `lmtp` 1.5.4.
`lmtp` takes one shifted value per unit, so `tests/canonical/lmtp_policy_adapter.R` realises each
policy by copying every unit four times. Copy $c$ takes its shifted arm from row $c$ of the node's
allocation table, and every policy probability is a multiple of one quarter. The pre-declaration
smoke run required the policy mean to pair within $10^{-6}$.

Over the 2,000 committed
replications its mean paired difference is $4.9 \times 10^{-11}$, and the largest is
$6.3 \times 10^{-10}$. The `low` mean and the two contrasts are not exactness pairs, because
`lmtp` pools label nodes over every arm. Their mean absolute paired difference is about
$9 \times 10^{-3}$, and the largest is 0.042. The
[study page](method-evidence/stochastic-categorical-longitudinal-tmle.md) gives every verdict.

### Modified treatment policies at a node

A node of a `DynamicRegimen` plan can be a modified treatment policy. That policy reads the
natural value $A_t$ and the history. The classes are those of the
[point-treatment section](point-treatment-tmle.md#modified-treatment-policies). Díaz, Williams,
Hoffman and Schenck (2023), Theorem 1, identifies the mean under Assumption 3. Equation (3)
gives the ratio at a continuous node, and the discrete formula gives it at a categorical node.

| node kind | how to declare it | ratio |
| --- | --- | --- |
| continuous dose | `LTMLE.fit(..., continuous_treatment=["A1", "A2"])` | a pooled-hazard density with `density_bins` bins, or `ratio="classifier"` |
| categorical | the default | the discrete formula over the fitted mechanism |
| vector of categorical columns | a list of columns as one entry of `treatment=` | the discrete formula over the joint levels |

`LTMLE.fit` refuses a vector node with a continuous component, before any learner. At a
continuous node the treatment factor of the cumulative product is the ratio itself. So
`g_bounds` bounds the censoring and categorical factors only, and the fit stores each node's
ratio as `node_ratio`. A cross-fitted fit (`n_folds` above one) uses the pooled
construction of the other plan kinds. `msm=` accepts modified treatment policy cells.

`regimens=` takes the policy as a plan node, and the `policies=` keyword stays refused by name:
`regimens={"+0.5": DynamicRegimen("+0.5", (Shift(0.5, cap=4.0),) * 2)}`.

Evidence: `tests/unit/test_influence_gateaux_longitudinal_mtp.py` checks the estimate and the
curve against the g-formula of `tests/discrete_law_longitudinal_mtp.py` and its Gateaux
derivative. `tests/unit/test_longitudinal_mtp_targeting.py` holds the mutation controls, and
`tests/unit/test_longitudinal_mtp_compositions.py` covers survival, `msm=`, cross-fitting and the
refusals. The registered study is `longitudinal-mtp`, and its run is pending.

### One baseline treatment held over the nodes

A point-treatment survival analysis has one treatment decision, at baseline, and an event process
over $K$ visits. Declare it with one column name: `treatment="A"` on `LTMLE.fit`,
`LongitudinalData.from_frame` and `LongitudinalTreatment`. Node 1 is the decision. Each later node
is an *identity node*.

| item | rule |
| --- | --- |
| identity node | its plan value is the observed node-1 value, and its factor is exactly one in the ratio numerator and in the denominator. It fits no model, it is never a policy or modified treatment policy node, and it adds no block to any mechanism or outcome design |
| node count | a survival outcome gives one node per event column. An end-of-study outcome takes the count from `censoring=`, else from `time_varying=`, else it is one. Lengths that disagree raise `DataError` |
| regimens | each regimen resolves on node 1, once, on the node-1 history. A static plan that repeats one label over the nodes reads as that label. Any other plan of more than one node raises `ValueError` before any learner |
| plans at node 1 | an arm, a known rule, a known stochastic policy, or a modified treatment policy (categorical or continuous), as for an ordinary node 1 |
| identification | without $L_t$, Benkeser, Carone and Gilbert (2018), Section 2.3. With $L_t$, Díaz, Williams, Hoffman and Schenck (2023), Theorem 1, with the identity policy $d_t(a_t, h_t) = a_t$ at $t \ge 2$ |
| influence curve | the shipped telescoping curve, with $H_t = 1\{A^d = A\}\,(q_1/g_1) \prod_{s \le t} 1\{C_s = 1\}/c_s$. For an arm or a rule, $q_1/g_1 = 1/g_1(d \mid W)$, which is $H_k$ of Benkeser et al. (2018), Section 4 |
| end-of-study outcome | the same held design with the shipped end-of-study recursion. Without $L_t$ it equals point TMLE with `delta=` and $\Delta = C_1 \cdots C_K$, because with saturated nuisances the curve keeps only its last term |
| report | `summary()` prints `treatment: one baseline decision 'A', held over K node(s)`. The nuisance report shows `LONGITUDINAL_HELD_DECISION` in place of a treatment row at each identity node. `bootstrap_design_kind` returns no kind, so the bootstrap is a diagnostic |

The wide layout with one treatment column per node gives the same fit when the later columns copy
node 1. With cell-mean learners at one fold the copied fit's later factor is exactly one, so the
two agree to the last bit. With a penalized treatment learner the copied fit's later factor is not
one, and the two fits differ.

`lmtp` with `trt` of length one evaluates every node's regression at the shifted treatment. For a
policy that is not idempotent, such as a stochastic draw or `a - 1`, that applies the policy once
per node, which is a different estimand. Copied treatment columns with the identity policy after
node 1 reproduce the held estimand in `lmtp` (`point-treatment-survival-policies`).

The evidence is in the table.

| test module | what it checks |
| --- | --- |
| `tests/unit/test_point_survival_law.py` | the estimate against the g-formula of `tests/discrete_law_point_survival.py`, the curve against its Gateaux derivative, and the held fit against the copied-column fit |
| `tests/unit/test_point_end_of_study_law.py` | the same three identities on the end-of-study law, and the point-TMLE reduction |
| `tests/unit/test_point_policy_mtp.py` | a policy and a modified treatment policy at node 1. A mutation that makes the identity nodes policy nodes fails the Gateaux check |
| `tests/unit/test_point_survival_mutations.py` | the other witnesses and mutations |

### Time-to-event input

`LongitudinalData.from_time_to_event` and the design `TimeToEvent` read one row per unit: a time,
an event code (`0` for censored), one baseline treatment and the baseline covariates. Node $k$
covers the interval $(g_{k-1}, g_k]$ of a grid $0 = g_0 < g_1 < \dots < g_K$. The rules follow the
visit structure of Benkeser, Carone and Gilbert (2018), Section 2.1.

| unit | nodes |
| --- | --- |
| an event of cause $j$ at $T \le g_K$ | the event at the node $k$ with $g_{k-1} < T \le g_k$ |
| censored at a grid time $g_j < g_K$ | observed through node $j$, then censored |
| an event or a censoring after $g_K$, or censored at $g_K$ | event-free and observed through node $K$, the administrative end. This is the `survtmle` `t0` convention (`makeDataList.R`) |

An event and a censoring at one grid time are ordered event first, as `survtmle`, `tmle3` and MOSS
order them. `grid=None` needs integer times and uses $g_k = k$ up to the largest event time. Each
regimen and cause costs $K(K+1)/2$ sequential regressions, so a default grid on day-level times can
hold thousands of nodes. `summary()` prints the grid. On a gridded fit, `horizons=` and the `tau`
of `rmst` take grid times, `curve()` reports the grid time in its `time` column, and `to_frame()`
adds a `time` column. A parameter name keeps the node index.

A censoring time strictly between two grid times is refused. Such a time cannot be ordered against
the events of its interval, and both conventions are biased. Declare the grid at the visit times.

Take one interval $(0, 1]$, an exponential event time with rate 0.3 and an independent exponential
censoring time with rate 0.2. The truth is $P(T \le 1) = 0.259$. The rule that censors the unit
inside the interval estimates 0.280. The rule that observes it through the interval estimates
0.236. `test_both_off_grid_censoring_conventions_are_biased` derives the three numbers in closed
form.

Each refusal of the input is raised before any learner.

| trigger | type |
| --- | --- |
| a time that is missing, not finite, or at or below zero | `DataError` |
| `grid=None` with a time that is not an integer | `DataError` |
| a grid that is not strictly increasing, positive and finite | `ValueError` |
| an event code that is missing, negative, not an integer, or not named by `causes=`; a cause with no event at or before $g_K$ | `DataError` |
| a horizon or an RMST time off the grid, or past $g_K$ on a declared grid | `ValueError` |
| `time_varying=` on `TimeToEvent` | `TypeError` |
| a censoring time between two grid times | `DataError` |

`to_time_to_event()` returns each unit's grid time and event code. It is the inverse map on a law
whose times sit on the grid. `tests/unit/test_time_to_event_input.py` checks one refusal per row
and the round trip on two grids. It also checks the long fit against the wide held fit. Floor
binning and the censor-first map are its mutations, and each one misses the truth.

### A censoring node with no censoring

A censoring node at which no eligible unit is censored has the fixed factor one and fits no
learner. The eligible rows are the rows at risk before the node with positive weight. On the long
layout node 1 always has this shape, because every time is at least $g_1$. A cross-fitted training
fold with no censored unit predicts one, its empirical rate. The nuisance report shows
`LONGITUDINAL_NO_CENSORING`, or `LONGITUDINAL_NO_CENSORING_IN_FOLD` beside the node's row.
`survtmle` sets `G_dC = 1` at `t = 1` and has a `noCens` branch for the same case. Before this rule a
standard classifier raised on the constant target. `tests/unit/test_no_censoring_node.py` removes
the rule and sees that error return.

## Functionals of a fitted result

Each method below reads the influence curves of the reported estimates. Each one is an exact
delta-method or linear-functional computation, so it needs no new theory. Each derived estimate
takes the inference status, the covariance rule and the cluster unit of its inputs.

| method | value | influence curve |
| --- | --- | --- |
| `ratio(a, b)` | $\psi_a / \psi_b$, interval on the log scale | $IC_a/\psi_a - IC_b/\psi_b$ |
| `ratio(a, b, kind="or")` | the ratio of the odds, interval on the log scale | $IC_a/(\psi_a(1-\psi_a)) - IC_b/(\psi_b(1-\psi_b))$ |
| `ratio(a, b, view="survival")` | the ratio of $1 - \psi_a$ and $1 - \psi_b$ | the same curves after $\psi \mapsto 1 - \psi$ and $IC \mapsto -IC$ |
| `rmst(d, tau)` | $\tau - \sum_{t=1}^{\tau-1} F_d(t)$ | $-\sum_{t=1}^{\tau-1} IC_{F_d(t)}$ |
| `rmtl(d, tau, cause)` | $\sum_{t=1}^{\tau-1} F_{d,j}(t)$ | $\sum_{t=1}^{\tau-1} IC_{F_{d,j}(t)}$ |

The ratio curves are those of `lmtp_contrast(type = "rr")` and `type = "or"` in `lmtp` 1.5.4
(`R/contrasts.R`, lines 19 to 57). A ratio of two `ey_regimen`, `risk_regimen` or `cif_regimen`
levels at one cause and horizon takes the name `rr_regimen[a vs b @ t=h]`. The survival view
takes `survival_rr_regimen[...]`. The view is refused on an end-of-study fit, and on a fit that
declares two or more causes, for the reason `curve()` gives. An odds ratio of end-of-study means
needs a binary outcome. `concrete` reports a relative risk of cumulative incidence with a
linear-scale interval and no arm covariance. This package uses the joint curve and the log scale,
as `lmtp` does.

Let $T$ be the node of the first event, with $T = K + 1$ for a unit that is event-free through
node $K$. Then $\mathrm{RMST}_d(\tau) = E[\min(T^d, \tau)]$ for $2 \le \tau \le K + 1$. The
restricted mean time lost to cause $j$ is $E[(\tau - \min(T^d, \tau))\,\mathbb 1\{J = j\}]$.
Royston and Parmar (2013) define the RMST. Andersen, Hansen and Klein (2004) define the time lost
to one cause.

The unit is the node index. For nodes $\Delta$ apart, the RMST in calendar time is
$\Delta \times \mathrm{RMST}$.

On a fit with a time grid, $\tau$ is a grid time $g_m$ with $2 \le m \le K$, and

$$
\mathrm{RMST}_d(g_m) = \sum_{k \le m} (g_k - g_{k-1})\,S_d(g_{k-1})
  = g_m - \sum_{k=2}^{m} (g_k - g_{k-1})\,F_d(g_{k-1}),
$$

with the curve $-\sum_{k=2}^{m} (g_k - g_{k-1})\,IC_{F_d(g_{k-1})}$. `rmtl` takes the same
spacings with a plus sign. This is $E[\min(\tilde T^d, g_m)]$, with $\tilde T$ the grid ceiling of
the event time. It equals $\int_0^{g_m} S_d(u)\,du$ only when the events sit on the grid, and for a
continuous event time it is larger. The name carries the node index $m$. On the unit grid
$g_k = k$ the time $\tau = K + 1$ is also accepted, and `rmst` equals the wide fit's
`rmst` (`test_on_the_unit_grid_rmst_is_the_wide_rmst`). On a competing-risk fit, $F_d$ is the sum of the cause-specific
incidences, so `rmst` needs every declared cause. `rmst` refuses a fit that omits a horizon below
$\tau$, an end-of-study fit, and a working-model fit.

`tests/unit/test_contrast_conveniences.py` checks each row on the exact laws. A four-node law in
`tests/studies/survival_grid_law.py` states the RMST truth by enumeration over the event times. It
sums no risk curve.

### The full-refit bootstrap

`LTMLE(n_bootstrap=B)` refits the whole estimator on `B` resamples. The contract is the table.

| question | contract |
| --- | --- |
| the unit | one row of the wide table: the whole trajectory of a unit, with its censoring and event nodes. A censored row and an event row resample as whole rows |
| a cluster | `bootstrap_resampling="auto"` resamples whole clusters when `id=` is declared. Each drawn occurrence gets its own cluster code. `"cluster"` without `id=` is refused before any learner |
| weights | the weights resample with their rows and are renormalized to mean one. They are not derived again |
| what a replicate refits | everything: the outer split, the treatment and censoring mechanisms, every backward regression and fluctuation for every regimen, cause and horizon, and the working-model projection. The replicate uses the same regimens, reference, horizons, bounds and learner specifications, with `simultaneous=False` |
| the outer split | a replicate draws its split from the fit's own `random_state`, as the point-treatment bootstrap does. Two copies of one unit can fall in different folds |
| what a replicate returns | the point estimate of every reported parameter |
| a failed replicate | it is dropped and counted in `n_failed`, and it is not drawn again. A resample that leaves a regimen with no followers fails |
| status | a replicate carries no status. The fit's status names the bootstrap columns. Under `few_cluster_plugin`, and on a fit with a Student $t$ reference below 40 clusters, the summary prints `bootstrap sd` and `percentile range` |
| replay | `truncation_curve()` does not rerun the bootstrap. Its check at the fitted bound compares the estimates without their bootstrap summaries |
| random draws | `run_bootstrap` spawns the stream from `random_state`. The stream draws nothing the fit uses, so every analytic field of the fit is unchanged. `tests/unit/test_ltmle_bootstrap.py` pins a committed `canonical-ltmle` row at `n_bootstrap=0` and at `n_bootstrap=2` |
| parallel layers | the replicates run in parallel over `n_jobs`, and each replicate fit runs with one worker |
| licensing scope | `bootstrap_design_kind` names a fit's kind: an end-of-study or single-cause survival outcome, fitted in sample, cross-fitted, clustered in sample (`cluster`) or clustered and cross-fitted (`cluster_cross_fit`), with static regimens, one treatment decision per node, a binary outcome, no weights and no working model. Any other fit has no kind, and a held baseline treatment is one such fit. A kind is licensed as inference only when it is in `LICENSED_BOOTSTRAP_DESIGNS`, and it enters after its cells in the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) are green. The registered run licensed `end_of_study/in_sample` and `survival/in_sample`. The cross-fitted and in-sample clustered end-of-study kinds stay diagnostic, with owner `X20-bootstrap`. No study cell measures `cluster_cross_fit`, so it stays diagnostic too. An unlicensed kind prints `bootstrap sd` and a percentile range. The study measures correctly specified cell-mean nuisances on finite binary laws, at $n = 1000$, and at 1,500 rows in 60 clusters for the cluster bootstrap. No result covers a data-adaptive nuisance. Cai and van der Laan (2020) is the warning for that case |
| derived estimates | `ratio`, `rmst`, `rmtl` and `contrast` apply the same function to each replicate's estimates, and attach the percentile interval of those values. The derived interval is licensed only when every input's interval is |

## Variations

| option | what it does |
| --- | --- |
| `regimens=` | static plans, fixed rowwise dynamic rules, categorical arms, or [known stochastic policies](#known-stochastic-policies). A plan is a sequence of arms, or one arm meaning that arm at every node. A plan with a callable node must be a `DynamicRegimen` declared `rule_kind="known"`. `LTMLE.fit` refuses an undeclared or `"estimated"` rule, and a callable written inline in a mapping, before any learner. Sample-adaptive thresholds and learned rules need additional inference and are outside this contract. The [scope page](scope-and-refusals.md#wrong-by-construction) gives the threshold witness |
| `reference=` | which regimen the contrasts are taken against. It is part of the estimand rather than a display setting |
| `horizons=` | which time points a survival fit reports cumulative risk at. `None` reports the whole curve. Name the horizons you will report: the cost is $T(T+1)/2$ regressions per regimen rather than $T$ |
| `msm=` | a working model over the regimen and horizon cells, in sample or cross-fitted. See [MSM projections](msm-projections.md) |
| four learner slots | `outcome_learner`, `pseudo_learner`, `treatment_learner`, `censoring_learner`. The pseudo learner fits the intermediate regressions, whose outcome is a bounded prediction rather than the outcome itself |
| `n_folds=`, `learner_folds=` | one outer split serves every node and regimen. The split is unstratified: `random_partition` draws it from the row count and the seed, and it balances no treatment node. Each fold fits the mechanism and an untargeted backward regression sequence on its training rows. One pooled fluctuation per node then targets the out-of-fold predictions, as [cross-fitting the recursion](#cross-fitting-the-recursion) states. The mechanism keeps the out-of-fold probabilities only, so its memory does not grow with $K$ |
| `g_bounds=`, `q_bounds=`, `alpha=` | cumulative truncation, outcome scaling, and the logistic shrink. Above one fold, a continuous outcome must declare `q_bounds` |
| `alpha_sig=`, `simultaneous=`, `n_multiplier=`, `multiplier_kind=` | interval level, and the simultaneous bands across the reported regimens |
| `n_bootstrap=`, `bootstrap_resampling=` | the full-refit bootstrap. [The bootstrap contract](#the-full-refit-bootstrap) states what each replicate resamples and refits |

### Cross-fitting the recursion

Above one fold, a per-regimen fit follows Section 5.2, Steps 1 to 4, of Díaz, Williams, Hoffman
and Schenck (2023). Those steps are on journal pages 852 and 853. arXiv:2006.01366 v4 uses the
same section and step numbers. The functions `_fit_regimen_crossfit` and `_pooled_targeting` in
[`longitudinal/sequential.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/sequential.py)
implement the steps.

| part of the fit | rows | what the fit does | Section 5.2 |
| --- | --- | --- | --- |
| fold regressions | the training rows of fold $k$ | fit the treatment and censoring mechanism, and run an untargeted backward regression sequence. Node $t$ regresses the fold's own untargeted prediction from node $t + 1$ | the cross-fitted nuisance estimates that Step 1 takes as its start |
| out-of-fold estimate | the held-out rows of fold $k$ | predict each node's regression and each mechanism factor. The $K$ folds give one out-of-fold prediction per row and node | Step 1 |
| loss weight | every follower of node $t$ | the observation weight times the inverse of the out-of-fold cumulative mechanism | Step 2 |
| pooled fluctuation | every follower of node $t$, from $t = T$ back to node 1 | fit one logistic fluctuation. Its offset is the out-of-fold prediction, and its outcome is the pooled targeted prediction from node $t + 1$ | Step 3, which fits "using all the data points in the sample" |
| estimate | every row | average the targeted node-1 prediction | Step 4 |

Each node's fluctuation solves its score over every follower. The fit therefore solves the
estimated efficient score equation, as a single-fold fit does.
`res.diagnostics.score_equations()` reports one `solver` row per node on both fits.

Theorem 3, on journal page 853, gives weak convergence at the nonparametric efficiency bound for
this construction. Its proof in the supplement, Section 6 of arXiv v4, conditions on each fold's
initial nuisance fit being fixed given its training data. The pooled coefficients depend on the
full sample and are handled as a low-dimensional fluctuation class. The untargeted fold
regressions meet the conditional requirement. Carrying a pooled coefficient back into a fold's
initial regression would not, so the package does not carry it back.

`tests/unit/test_pooled_longitudinal_targeting.py` recomputes each row of the table by hand. It
also carries four mutation controls. One fits the coefficient on one fold's training rows. The
others drop the loss weight, read a refit of one fold's mechanism model, or carry targeted values
back into the folds.

`n_folds=1` keeps the canonical single-fold recursion. Each node regresses the targeted prediction
from node $t + 1$, and each fluctuation solves over the rows its regression was fitted on.

The package also computes specializations and compositions that Section 5.2 does not state. The
[Eligibility rule](../roadmap.md#eligibility) admits a natural extension of a published result.
The table identifies a direct specialized source where one exists and otherwise gives the base,
step, and established argument.

| composition | base result | the step | the established argument | evidence |
| --- | --- | --- | --- | --- |
| cumulative risk at a horizon | Díaz, Hoffman, Hejazi and Williams (2024), corrected in 2025, with one cause | $Y$ is the indicator of an event by the horizon. The survival pseudo-outcome is $Z_t = Y_t + (1 - Y_t)\,\bar Q^*_{t+1}$ | Appendix E gives the cross-fitted pooled TMLE directly; a single event type is its one-cause reduction | [survival-curve study](method-evidence/cross-fitted-survival-curve-longitudinal-tmle.md) |
| cause-specific cumulative incidence | Díaz, Hoffman, Hejazi and Williams (2024), corrected in 2025 | $Y$ is the indicator of an event of one cause by the horizon. A competing event ends follow-up, and the regression is zero after it | Proposition 1 identifies the target and Appendix E gives the out-of-fold nuisances, all-row per-node fluctuation, pooled backward carry, and score argument | [competing-risk study](method-evidence/cross-fitted-competing-risk-longitudinal-tmle.md) |
| known observation weights | Theorem 3 under iid sampling | the target is a weighted mean of the node-1 regression, and every fluctuation carries the weight in its loss | the chain rule for influence functions, applied to a ratio of two means under iid draws of the observation and its known, bounded weight | [weighted study](method-evidence/cross-fitted-weighted-end-of-study-longitudinal-tmle.md), which fails two property cells and three paired comparisons that conclude underpowered, and publishes them under a `reporting` policy |
| categorical treatments and deterministic dynamic rules | Theorem 3 for a fixed modified treatment policy $d(a_t, h_t)$ that does not depend on $P$ | a fixed rowwise rule assigns one level from that unit's history | Section 2 lets a fixed policy depend on the unit's history, and Section 4, journal page 850, gives the intervention density for a discrete exposure. A sample-adaptive threshold or learned rule is outside this result | [categorical study](method-evidence/cross-fitted-categorical-longitudinal-tmle.md), and the fixed dynamic rule in the [end-of-study study](method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md) |
| several horizons, causes, regimens, and their contrasts | Theorem 3 for each parameter | each parameter has its own recursion on one shared split, and the report stacks their influence curves | a fixed-dimension stack by Cramér–Wold, then linearity or the delta method for each contrast | the survival-curve and competing-risk studies |
| ratios of regimen levels, RMST and RMTL | the stacked curves of the row above | a log ratio is a smooth function of two levels. RMST and RMTL are fixed linear combinations of the risks below the horizon | the delta method, and linearity | the ratio and RMST cells of the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) |
| the full-refit bootstrap | the estimator of each row above | refit the whole estimator on each resample | none for this estimator. The percentile interval is licensed only for a measured design kind whose cells are green, with correctly specified cell-mean nuisances on finite binary laws | the bootstrap cells of the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) |

Theorem 3 also requires every mechanism ratio and targeted sequential regression to be consistent,
the sum over nodes of their error products to be $o_P(n^{-1/2})$, and the density ratios to stay
bounded. Those are conditions on the targeted regressions. Appendix E of the competing-risk paper
shows that a sufficient condition stated in terms of the initial recursive learners can contain
cross-time products between a mechanism error at node $t$ and regression errors at later nodes.
The weighted row additionally needs weights that are known and bounded.

The split balances nothing. It used to balance the first treatment node, and the reviewed
longitudinal theorem defines a random near-balanced row partition instead. The audit in
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) found no source for the
first-node strata, so the package now draws the partition from the seed alone. Each fold must also
carry enough events for each declared cause, which
[scope and refusals](scope-and-refusals.md#how-to-read-a-refusal) states with its remedy.

One design is refused above one fold. A continuous outcome with `q_bounds=None` is refused,
because the sample outcome range would take the scale from held-out rows. The message names the
in-sample fit, which is `CrossFitting(enabled=False)`, or `n_folds=1` on the engine. A fit with
`id=` cross-fits with whole-cluster folds, as [clusters](#clusters) states.

Longitudinal `msm=` cross-fits the same way. Each fold runs the untargeted recursion of every
regimen and horizon cell. One stacked fluctuation per node then pools the update over every
follower of every cell. [MSM projections](msm-projections.md#the-longitudinal-projection) states the
construction and its evidence. The score report has one `solver` row per node and term.

`LTMLE` refuses each point-treatment keyword in `_REFUSED` in
[`longitudinal/estimator.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/estimator.py)
by name, with its own reason. Some keywords redirect to the argument that does the work.
`interventions=` redirects to `regimens=`. The table gives the refusals that are statements about
the question, the construction, or coverage.

| refused | kind | what it would need |
| --- | --- | --- |
| eliminating the competing events | a different question | what is reported is the cause-specific cumulative incidence with the competing causes *left alone*, so a competing event is part of the history. Removing it makes it an intervened node: a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions |
| `intermediate=` | a different question | a controlled direct effect fixes a mediator at one time point. Over a sequence, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| `mechanism=True` on `truncation_curve()` | a different question | a longitudinal fit holds one cumulative treatment-and-censoring bound, and it fits no separate observation mechanism to sweep. The option names a point-treatment axis, so the call raises `CapabilityError` rather than sweeping the cumulative bound under another name |
| a callable node that is not declared `"known"`, or that is declared `"estimated"` | wrong by construction | the fit treats each rule as fixed. A rule learned from the analysis sample needs a learned-policy estimand and its own inference. `LTMLE.fit` raises `CapabilityError` before any learner. Declare a rule fixed before the fit with `DynamicRegimen(label, plan, rule_kind="known")` |
| a callable written inline in a `regimens=` mapping | wrong by construction | an inline callable carries no declaration, so the fit refuses it before any learner. Write the plan as a `DynamicRegimen` declared `rule_kind="known"`, with `(rule,) * T` for one rule at every node |
| an outcome missing for a reason other than censoring | wrong by construction | left as it is, the probability of observing it is silently taken to be one. Encode it as a final censoring column, so it is estimated and enters the cumulative product |
| longitudinal sensitivity-bound estimation | not written yet | a sample estimator and sampling theory for its bound functionals. [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) holds the stop |
| `incremental=` | not written yet | Kennedy (2019), *Journal of the American Statistical Association* 114(526), treats incremental interventions on a time-varying treatment. The tilt is built from the mechanism, so it needs the product of tilted mechanisms and a mechanism submodel at every node. [X19](../roadmap.md#x19-incremental-interventions-over-time) holds the work |
| a continuous outcome with `q_bounds=None` above one fold | not written yet | with `q_bounds=None` the scale comes from every observed outcome, held-out rows included, and no shipped result covers that scale |

Two plan shapes raise `DataError` before any learner. Each one is a statement about the input, so
the table of kinds above does not list it.

| plan | what the refusal asks for |
| --- | --- |
| an iterator in `regimens=`, such as a generator | a tuple, or a `DynamicRegimen`, which reads its plan once and stores it as a tuple. The fit does not read the iterator: a fit reads `regimens=` at each call, and an iterator is empty after its first read |
| a `DynamicRegimen` plan that is one label or one callable | one entry per node. Write one rule for every node as `(rule,) * T` |

See [scope and refusals](scope-and-refusals.md#how-to-read-a-refusal) for what each `kind` means.

### Clusters

`LTMLE.fit(id=...)` treats the cluster as the unit. The cluster is the fold unit and the variance
unit, in sample and under cross-fitting. The table gives each part of a clustered fit.

| part | rule |
| --- | --- |
| outer split | `random_partition` draws whole clusters from the labels and `random_state`. `LTMLE.fit` checks the drawn split with `check_integrity` before any learner. A split cluster raises `DataError` |
| fold count | the requested `n_folds`, capped at the cluster count with a `UserWarning` |
| inner Super Learner folds | grouped on the same labels, at every mechanism and node regression |
| targeting | the pooled fluctuation over every follower of each node, as on an unclustered fit |
| curve and variance | the row curve summed within each cluster. The variance is $J\,\widehat{\mathrm{var}}(S_c)/n^2$, with $S_c = \sum_{i \in c} D_i$ and $J$ clusters |
| status, reference and bands | the rules of [clusters](inference.md#clusters) for a longitudinal fit. Below 40 positive-mass clusters the reference is Student $t$ with $J - 2$ degrees of freedom. Below 20 the fit takes `"few_cluster_plugin"`, in sample and cross-fitted, because 20 is the smallest count the registered studies measure. Simultaneous bands need 40 or more |
| bootstrap | the design kind `<outcome>/cluster_cross_fit`, which no study licenses, so the percentile output is a diagnostic. Each replicate redraws a grouped split over its resampled clusters, so two copies of one cluster can land in different folds |

The base result is Díaz, Williams, Hoffman and Schenck (2023), Section 5.2, journal page 852, and
Theorem 3, page 853. Theorem 3 takes $n$ iid units and a random partition of them.

Take the unit to be the cluster $O_c = (O_{c1}, \dots, O_{cN_c})$, with $\varphi_c = \sum_i \varphi(O_{ci})$. The
row estimating equation is then $\sum_c \{\varphi_c - N_c \hat\theta\} = 0$. That is a
cluster-level equation for $E\varphi_c / E N_c$. A partition of clusters is a random partition of
the iid units, so each training fold is independent of its validation fold. The remainder rates
then read $o_P(J^{-1/2})$, and the curve is $(\varphi_c - N_c\theta)/E(N)$.

The targeting pools every follower, so the curve is centred at the pooled estimate. The variance
therefore takes `ddof=1` over all $J$ cluster sums, and the fold count does not enter the degrees
of freedom. A fold-targeted or fold-evaluated `LTMLE` would need the centred rule of the
fold-evaluated point-treatment fit. The package ships neither.

The result inherits five conditions: independent clusters, no interference between units, a
bounded cluster size, the Theorem 3 rates and bounded density ratios in the cluster count, and the
unequal-size and few-cluster rules of [clusters](inference.md#clusters). Wang, Park, Small and Li
(2024), Remark 3, caution against complex learners at about 20 clusters. The registered evidence
uses parametric nuisances.

The cluster handling does not depend on the target. The split, the inner groups, the
cluster-summed curve and the status are the same for every target, a
[known policy](#known-stochastic-policies) included. A [modified treatment policy](#modified-treatment-policies-at-a-node)
node is included too. `incremental=` stays refused for its own reason, and it inherits this
handling when it ships.

Two side effects follow from the cross-fitted default. A fit with fewer than 10 clusters warns that
it reduces `n_folds` to the cluster count, and a fit with fewer than 20 reports no interval. At few clusters a training fold can lack a first-node
level, or hold one outcome value among a regimen's followers. The fit then raises
`LongitudinalError` after the draw, and the message names the in-sample fit.

`tests/unit/test_clustered_cross_fitted_ltmle.py` holds the fast evidence. Each witness has a
mutation control that fails it.

| claim | witness |
| --- | --- |
| every cluster lands in one fold | the fit's split equals the grouped draw. A draw without `cluster=`, or an overriding `_folds`, raises `DataError` before any learner |
| the inner folds are grouped | a recording learner receives the codes of exactly its training rows at every nuisance fit |
| a held-out cluster is unseen | a learner that memorizes a site cannot see its own cluster. Under row-level folds, the flip of a cluster mate moves the prediction |
| the curve is cluster-summed | the variance of every reported and derived estimate equals the cluster-sum variance of its curve to `rel=1e-12`. At 60 clusters of 10 to 70 rows it exceeds the row variance by a factor above 3 |
| the size term is carried | the variance reads $A_c - N_c\hat\psi$, which differs from $A_c$ by more than 10% at unequal sizes |
| every target kind | end of study (static, dynamic, categorical), survival, competing risks with `incidence_total()`, weights with a zero-mass cluster, ratios, RMST and RMTL, at 40 and 21 clusters |
| the floor | 20 clusters keep $t_{18}$ and 19 take `"few_cluster_plugin"`, cross-fitted and in sample. A fit that reads the point-treatment floor of 10 fails it |

Two registered studies measure the fit, and both publish under `reporting`. The
[clustered cross-fitted study](method-evidence/clustered-cross-fitted-longitudinal-tmle.md) pairs
the fit with R `lmtp` 1.5.4 at 100 clusters of 40 rows. Its four cluster-robust pairs and five
paired comparisons pass. Its band covers 0.933, below the 0.92 lower bound of its interval band,
and the `band-finite-sample` owner holds that cell. The
[few-cluster study](method-evidence/few-cluster-cross-fitted-longitudinal-tmle.md) measures the
$t$ reference at 20 and 30 clusters, and all 8 scored cells pass.

| limit | reason |
| --- | --- |
| no competing-risk study | `make_longitudinal_competing` has no cluster option. The fast tier pins the exact clustered identity for every incidence and for `incidence_total()` |
| no informative-size cell | the size enters no longitudinal study law. `clustered-unequal-cvtmle` measures the row-weighted target for point treatment |
| no bootstrap licence | the `cluster_cross_fit` kind has no study cell |
| the `lmtp` pair at equal sizes only | `ife` aggregates cluster means, which equal cluster sums at equal sizes only |
| 4 to 19 clusters | the fit takes `"few_cluster_plugin"` and reports no interval. No cross-fitted study measures that range: a probe of 2,000 draws at 10 clusters found 3 and 8 draws that admit no fit, and the harness refuses a cell that loses a replication. [F28](../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) owns it |

## Validation issues special to this method

**The reported standard error is the plug-in influence-curve variance.** It does not implement R
`ltmle`'s default recursive `variance.method="tmle"`, which takes the larger of that variance and a
robust estimate. R itself warns that influence-curve-only inference can be substantially
anti-conservative under positivity problems or rare outcomes, and its robust method has its own
availability restrictions. Active truncation is a reason to qualify the interval. It is not
evidence that the interval has absorbed truncation uncertainty.

**The exact laws cannot see the finite-sample choices, because every fluctuation coefficient is
zero there.** That is why the canonical R fixture exists, and why it is deliberately narrow. It
freezes R `ltmle` 1.3-0 with fixed numeric mechanism predictions, intercept-only outcome
regressions, no cross-fitting, and bounds of `(0.2, 0.99)`. One baseline stratum binds only at the
second cumulative prefix, and the deepest fluctuation coefficient exceeds 0.4 in magnitude. A
second, censoring-active variant exists so the `uncensored` and `trained_on` masks are not inferred
from a fixture in which every censoring indicator equals one. The fixture compares the estimate,
every row of the influence curve, the used cumulative probabilities, and the targeting
coefficients.

Its interpretation is equally narrow, and is predetermined. Disagreement means the implementation
choices differ and must be reconciled. Agreement is evidence for those choices only. The
independently derived law and the Gateaux checks remain the acceptance evidence for the parameter
and the influence curve.

**A point estimate can stay green while six influence-curve comparisons go red.** Evaluating the
mechanism at a constant arm does exactly that, because with an exact initial fit the fluctuation
coefficient is zero, the estimate is the plug-in, and no error in the mechanism can move it. That
mutation is one of two the dynamic-rule oracle carries.

**Tidying a `t-1` to a `t` reads like a correction and is not.** On the survival path it silently
drops every failure from its own node's regression, biases the risk downwards, and leaves every
score at `1e-16` and every convergence flag green. It is a deliberate mutation, and it turns 26 of
that module's 30 tests red.

**Writing the cause's own survival factor is wrong by exactly the mass that left through the other
causes.** On the competing-risks path that mutation takes 21 of 130 tests, every one of them at the
second horizon, because at the first horizon there is no survival factor to get wrong.

**The exact law is blind to how an earlier arm is *coded* into the mechanism's design.** Its
learners are saturated and partition by distinct design row, under which an ordinal code and a
drop-first indicator tuple are a bijection. Only a separate non-monotone `glm` witness separates
them, and it covers one node, one link, and one truth.

**A dynamic rule needs a quadrature truth, and a wider rule is not a better one.** An indicator puts
a step function into the integrand, where a Gauss-Hermite rule converges algebraically rather than
spectrally. The naive version moved by `1.7e-3` between 48 and 64 nodes, which is worse than the
Monte Carlo it exists to avoid. The axis is therefore integrated as two Gauss-Legendre panels
meeting at the jump, which makes the arm constant *within* a panel and the answer stable to `1e-13`
under refinement.

| where to read the evidence | what is there |
| --- | --- |
| [ordinary end-of-study longitudinal TMLE](method-evidence/ordinary-end-of-study-longitudinal-tmle.md) | against R `ltmle` 1.3-0, including a targeted-versus-unfluctuated pair that measures what the paired comparison cannot |
| [cross-fitted end-of-study longitudinal TMLE](method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md) | against R `lmtp` 1.5.4 on one exact five-fold assignment, with the mechanism supplied to both, plus held-out recursion checks and independent theory properties |
| [ordinary weighted end-of-study longitudinal TMLE](method-evidence/ordinary-weighted-end-of-study-longitudinal-tmle.md) | reporting evidence against weighted R `ltmle` on exact-size selected samples, with target-weight and learner-weight controls |
| [cross-fitted weighted end-of-study longitudinal TMLE](method-evidence/cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | reporting evidence against weighted R `lmtp` on identical five-fold assignments, with a dedicated weighted-learner adapter |
| [ordinary survival-curve longitudinal TMLE](method-evidence/ordinary-survival-curve-longitudinal-tmle.md) | against R `ltmle` 1.3-0 with `survivalOutcome=TRUE`, across two horizons and with survival-recursion controls |
| [cross-fitted survival-curve longitudinal TMLE](method-evidence/cross-fitted-survival-curve-longitudinal-tmle.md) | against R `lmtp` 1.5.4 across both horizons on one exact five-fold assignment, with held-out and survivor-only controls |
| [ordinary categorical longitudinal TMLE](method-evidence/ordinary-categorical-longitudinal-tmle.md) | against R `lmtp` 1.5.4 with exact categorical mechanisms, for three treatment levels at two nodes |
| [cross-fitted categorical longitudinal TMLE](method-evidence/cross-fitted-categorical-longitudinal-tmle.md) | against R `lmtp` 1.5.4 on the identical five-fold assignment, with a flexible-tree cross-fit control |
| [ordinary competing-risk longitudinal TMLE](method-evidence/ordinary-competing-risk-longitudinal-tmle.md) | against R `lmtp` 1.5.4 with the other cause in `compete=` and exact mechanisms supplied to both |
| [cross-fitted competing-risk longitudinal TMLE](method-evidence/cross-fitted-competing-risk-longitudinal-tmle.md) | against R `lmtp` 1.5.4 on the identical five-fold assignment, with a flexible-tree cross-fit pair |
| [longitudinal estimands outside the target registry](evidence.md#longitudinal-estimands-outside-the-target-registry) | the parameter and influence-curve oracle, the mutation witness, and the declared gaps, for each of the five longitudinal variants |
| [clustered cross-fitted end-of-study longitudinal TMLE](method-evidence/clustered-cross-fitted-longitudinal-tmle.md) | against R `lmtp` 1.5.4 with `ife` 0.2.3 at 100 clusters of 40 rows, on the realized whole-cluster folds, with four cluster-robust pairs and a cluster multiplier band |
| [cross-fitted clustered longitudinal TMLE at few clusters](method-evidence/few-cluster-cross-fitted-longitudinal-tmle.md) | the $t_{J-2}$ reference at 20 and 30 clusters of equal and unequal sizes, with an IID control |
| [the implementation validation grid](method-evidence/validation-grid.md) | the twelve registered longitudinal TMLE rows and each row's declared limits |

Competing-risk correctness also rests on the independent finite law, the Gateaux comparison, the
all-cause-versus-cause-specific mutation, and the one-cause reduction. The two competing-risk rows
add the R `lmtp` comparison. Both rows keep the competing event natural and supply the mechanisms.
Neither row covers weights, clustering, or an eliminated competing event.
