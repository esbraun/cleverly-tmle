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
| you want a survival curve that never decreases | `survival_model="hazard", targeting="one_step"` reads every horizon from one set of targeted hazards. `isotonic=True` projects any curve onto a non-decreasing one | the hazard model needs one categorical baseline treatment and no covariate after baseline. A projected level has a valid pointwise interval only where the true curve strictly increases |
| several causes of failure compete | cause-specific cumulative incidence, with the competing causes left alone | that is a *total* effect. Eliminating the competing event is a different question, and is refused by name |
| you want the effects summarised across regimens | a working model over regimen and horizon cells | see [MSM projections](msm-projections.md) |

Reach for [point-treatment TMLE](point-treatment-tmle.md) when the exposure is measured once.
Reach for [`IPSI`](#incremental-interventions-over-time) when the plan multiplies each node's odds of
treatment.
Reach for [`SequentialGComputation` and `SequentialIPW`](sequential-baselines.md) for the
plug-in and weighting baselines of the same target. Their intervals hold only under correctly
specified `GLM` models.
Reach for [`ParametricGFormula`](parametric-gformula.md) for the forward-simulation form of the same
target and for a fixed or rule-set continuous dose. Its interval holds only under correctly specified models.

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

### Repeated outcomes

A repeated outcome is measured at every node and is not absorbing. A unit with $Y_1 = 1$ can have
$Y_2 = 0$. Declare it with `outcome=Repeated(["Y1", "Y2", "Y3"])`. The node order is

$$
W,\; L_1, A_1, C_1, Y_1,\; L_2, A_2, C_2, Y_2,\; \ldots,\; L_T, A_T, C_T, Y_T.
$$

The parameter is the curve of means $E[Y_h^d]$, one for each plan $d$ and horizon $h$. A fit
reports `ey_regimen[d @ t=h]` and `ate_regimen[d vs ref @ t=h]`.

The construction is a reduction to the end-of-study fit. The container puts $Y_s$ first in the
covariate block of node $s + 1$, before the covariate screen. Each later treatment, censoring and
outcome regression can then read it. At horizon $h$ the fit seeds the recursion with the observed
$Y_h$ and runs the end-of-study recursion over nodes $1, \ldots, h$. That fit is the end-of-study
fit of the panel truncated after $h$, with $Y_h$ as its outcome. The base for each horizon is the
end-of-study derivation of Bang and Robins (2005) and van der Laan and Gruber (2012). Kim, Kennedy
and Naimi (2021, arXiv v3 preprint) discuss the repeated-outcome setting. This implementation does
not rely on that preprint.

| item | value |
| --- | --- |
| history at node $t$ | $W, L_1, Y_1, L_2, \ldots, Y_{t-1}, L_t$ and the earlier treatments |
| seed at horizon $h$ | the scaled $Y_h$ on the rows uncensored through $h$ |
| missing $Y_t$ on a row uncensored through $t$ | refused. Encode it as $C_t = 0$ |
| outcome scale | one scale over the observed values of every node, so a continuous curve has one scale |
| family | resolved over the observed values of every node |
| influence curve at $(d, h)$ | the end-of-study influence curve of the truncated panel |
| joint band | the multiplier band over the stacked curves of every reported $(d, h)$ |
| working model, `msm=` | a design over $(d, h)$ cells. Each cell seeds its own $Y_h$ |
| full-refit bootstrap | shown with the label `experimental`. No support row lists `bootstrap` for this outcome |

The remainder at each horizon is the end-of-study product remainder. The stack over horizons adds
no remainder term. The interval at $(d, h)$ is therefore the end-of-study interval of the truncated
panel, under the same inference statuses.

| composition | status |
| --- | --- |
| static, dynamic and known stochastic plans, modified treatment policies | fits. A rule may read $Y_{t-1}$ |
| cross-fitting, clusters, weights, a held treatment, known mechanisms | fits |
| `msm=` over $(d, h)$ cells | fits |
| `IPSI` | fits. Each horizon is the end-of-study fit of the truncated panel ([below](#ipsi-on-a-repeated-outcome)) |
| `SDR` | refused before any learner, until [X32b](../roadmap.md#x32b-repeated-outcomes-for-sdr-and-ipsi) |
| competing causes | not declarable. A cause mapping declares an absorbing outcome |
| `rmst()`, `rmtl()`, `incidence_total()`, `ratio(view="survival")` | refused. A mean of a repeated outcome is not a cumulative risk |
| `curve()` | `scale="mean"`, the default for this outcome. `"risk"` and `"survival"` are refused |

This construction does not cover a recurrent-event count process or truncation by death. If death
is encoded as censoring, the parameter is the mean in a population where death is prevented.

| estimator | status | source for horizon $h$ |
| --- | --- | --- |
| `LTMLE` | fits | Bang and Robins (2005); van der Laan and Gruber (2012) |
| `IPSI` | fits | Kim, Kennedy and Naimi (2021), Theorems 3.1, 4.1 and 5.1 |
| `SDR` | refused until X32b | none yet |

The allowlist `REPEATED_OUTCOME_KINDS` holds the kinds in this table that fit.
`tests/unit/test_repeated_outcome_subclasses.py` fits every estimator subclass and requires
either the reduction or the refusal.

A level of $E[Y_h^d]$ is checked against the range of its own horizon. The outcome scale pools
every node, so a level of $Y_2$ outside the range of $Y_2$ and inside the range of $Y_1$ would
pass the pooled range. The fit warns with `ParameterSpaceWarning` and names the range of
that horizon. A declared `q_bounds` replaces the derived range
(`tests/unit/test_repeated_outcome_container.py`, W16).

`tests/unit/test_repeated_outcome.py` checks the reduction at every horizon for 13 compositions
of `LTMLE`, and for 11 of those 13 compositions of `IPSI` plus six incremental ones: the estimate, the
influence curve, every node regression and the cumulative mechanism, at `rtol=1e-12`. The other two `IPSI` compositions refuse with the end-of-study message. `tests/unit/test_influence_gateaux_repeated.py` checks the influence curve against
the complex-step Gateaux derivative of the g-formula on two exact laws in
`tests/discrete_law_repeated.py`. One law is binary. The other has three values per node. Ten
mutation controls fail these tests, among them the absorbing seed $\max_{s \le h} Y_s$, a history
without $Y_{s}$, and one seed at $Y_T$ for every horizon.

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
| a policy density that depends on $P$, such as an incremental odds tilt | page 850 records that multiply robust estimation "is not generally possible" for such a regime; the curve needs a mechanism-derivative term | [incremental interventions over time](#incremental-interventions-over-time) |

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
| mixed vector, with categorical and continuous columns | a list of columns as one entry of `treatment=`, with a column of the list in `continuous_treatment=` | the joint ratio of a `JointPolicy`: a chain-rule density with `ratio="density"`, or `ratio="classifier"`. See [vector nodes with a continuous component](#vector-nodes-with-a-continuous-component) |

At a continuous node the treatment factor of the cumulative product is the ratio itself. So
`g_bounds` bounds the censoring and categorical factors only, and the fit stores each node's
ratio as `node_ratio`. A cross-fitted fit (`n_folds` above one) uses the pooled
construction of the other plan kinds. `msm=` accepts modified treatment policy cells.

**The bin count of the density route.** A binned density at a fixed bin count is not
consistent. A growing count is necessary, and the hazard learner must also be consistent. The bin edges are sample quantiles, so the tail bins are wide, and the binned ratio
errs most there. The table gives the error that one draw of the `longitudinal-mtp` law measured
with the exact bin probabilities, so the binning is the only error left.

| bins | spread of the influence curve over the efficient one, n = 2,000 | the same, n = 8,000 |
| ---: | --- | --- |
| 20 | 1.62 | 1.72 |
| 80 | 1.14 | 1.20 |
| 160 | 1.04 | 1.13 |
| 320 | 0.99 | 1.00 |

At a fixed count the excess spread stays as $n$ grows. It falls as the count grows: at
n = 2,000 it is 0.62 at 20 bins, 0.14 at 80 bins and 0.04 at 160 bins. So the
default `density_bins=None` grows the count with the sample: `max(20, ceil(2 n^(1/3)))`, the rate
of Scott (1979). An explicit `density_bins=` is used as given. `tests/unit/test_density_bins_consistency.py`
shows the tail error shrinking with $n$ under the default, and a mutation that holds the count at
20 fails it. The same file checks the default count through `fit_conditional_density` and
through a `TMLE` fit at n = 2,000.

**The memory of the density route.** The pooled hazard design holds about $n (K + 1) / 2$
records of the covariates and $K - 1$ bin columns, at 8 bytes each. At the default count it grows
as $n^{5/3}$. The table gives the design at one covariate column.

| n | default bins | design |
| ---: | ---: | ---: |
| 10,000 | 44 | 0.07 GiB |
| 100,000 | 93 | 3.2 GiB |
| 300,000 | 134 | 20 GiB |
| 1,000,000 | 200 | 150 GiB |

A fit counts the design before it allocates it. When the design exceeds the available memory, the
fit raises `MethodConfigurationError`. The message gives the size and the largest bin count whose
design fits in half of the memory. `cleverly.learners.density.hazard_design_bytes` gives the
estimate. A learner copies its design, so the peak can be a multiple of the table.

Two consequences follow. When the outcome regression is wrong, the bin error enters the bias at
first order, so double robustness through the density alone needs a ratio error of
$o(n^{-1/2})$, which a histogram does not give. When the outcome regression is right, a coarse
count inflates the influence curve, and the intervals are conservative. When efficiency matters,
use `ratio="classifier"` with a flexible classifier, or set `density_bins=` higher.

`regimens=` takes the policy as a plan node, and the `policies=` keyword stays refused by name:
`regimens={"+0.5": DynamicRegimen("+0.5", (Shift(0.5, cap=4.0),) * 2)}`.

Evidence: `tests/unit/test_influence_gateaux_longitudinal_mtp.py` checks the estimate and the
curve against the g-formula of `tests/discrete_law_longitudinal_mtp.py` and its Gateaux
derivative. `tests/unit/test_longitudinal_mtp_targeting.py` holds the mutation controls, and
`tests/unit/test_longitudinal_mtp_compositions.py` covers survival, `msm=`, cross-fitting and the
refusals. The registered study is
[`longitudinal-mtp`](method-evidence/longitudinal-modified-treatment-policies.md). Its
mechanism-only cells read oracle bins that grow as $n^{2/3}$, 320 at n = 2,000, for the reason
above. So they test the targeting with a nearly exact ratio and not the default density.

### Vector nodes with a continuous component

A node can hold several treatment columns. A list of columns is one entry of `treatment=`, and
`continuous_treatment=` names the columns that hold a dose. A node with a dose and a categorical
column is a *mixed node*. Díaz, Williams, Hoffman and Schenck (2023), Section 2, let the
treatment at a node be a vector $A_t = (C_t, D_t)$, and Theorem 1 identifies the mean for it.

```text
LTMLE(regimens, ratio="classifier", ...).fit(
    frame,
    outcome="Y",
    treatment=[["flag1", "dose1"], ["flag2", "dose2"]],
    baseline=["W"],
    continuous_treatment=["dose1", "dose2"],
)
```

`LTMLE` and `SDR` fit a mixed node. Each regimen sets the node with a `JointPolicy`, which
[the point-treatment section](point-treatment-tmle.md#vector-treatments) defines. The policy
frame at the node holds the history. Each map reads the natural values of the components as the
table of that section states.

| item | rule at a mixed node |
| --- | --- |
| ratio | the joint ratio $g^d / g$ at the observed vector. It sits in the numerator of the cumulative product, as the ratio of a continuous node does |
| `ratio="density"` | the chain rule: the categorical mechanism, then a pooled-hazard density for each dose in `order`. A sum over the categorical preimages and Equation (3) for each dose give the induced density. `g_bounds` bounds the categorical factor in the running denominator |
| `ratio="classifier"` | the stacked classification of Section 5.4. The design holds the history, an indicator for each categorical component and the raw doses. The classifier cannot split the ratio, so `g_bounds` reaches no factor of the node |
| history of a later node | both components of an earlier node, with a dose as a dose column and a categorical component as its own drop-first indicators |
| a flag with a dose | the dose map of one mixed node reads the natural flag. Two consecutive nodes, a flag node and then a dose node, read the flag that the first node assigned. With an identity flag map the two readings coincide, and the fits are equal (W10) |

**What composes.** Each row has a witness in `tests/unit/test_longitudinal_mixed_node.py`.

| composition | status |
| --- | --- |
| censoring, whose retention reads both components of an earlier node | supported (W9) |
| survival outcomes and one-cause competing risks | supported. A one-cause fit equals the survival fit |
| weights, clusters with `id=`, cross-fitting | supported. A cluster stays whole, and the folds split units |
| `msm=` over joint-policy plans | supported. A saturated working model equals the separate fits |
| truncation replay, bootstrap, `subset`, `dumps` and `loads` | supported |
| `SDR` | supported on both routes, in sample and with folds that are each an exact copy of the law |
| a known randomizer on the `JointPolicy` or on one component | supported (W11, `tests/unit/test_vector_component_randomizer.py`) |
| `RiskRatioTilt` on a binary component | supported. It joins the product randomizer |

[Scope and refusals](scope-and-refusals.md#vector-treatments) lists what a vector refuses and
gives the reason for each refusal.

**Evidence.** `tests/unit/test_longitudinal_mixed_node.py` checks the estimate and the influence
curve of every plan on the two-node mixed law of `tests/discrete_law_longitudinal_mtp.py`
against the g-formula and its complex-step Gateaux derivative (W3). The law gives each node a
flag and a dose on the grid $\{0, 1, 2\}$. W4 sets the flag and gates the dose on the natural
flag. Its law moves the estimand by more than $10^{-2}$ between that reading and the reading of
the assigned flag. W5 checks that the later history holds both components.

`tests/unit/test_longitudinal_mixed_node_mutations.py` holds the mutation controls, and each
control moves a witness by more than $10^{-4}$. They drop the flag or the dose from the
classifier, read the assigned flag, drop the earlier flag from the history, read the kind off the
first component only, and drop a component from the censoring or outcome design.
`tests/unit/test_vector_density_route.py` repeats the plans on the density route and adds an
ordered plan on a law with two doses per node. The registered study is
[`vector-mtp`](method-evidence/vector-modified-treatment-policies.md).

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
| report | `summary()` prints `treatment: one baseline decision 'A', held over K node(s)`. The nuisance report shows `LONGITUDINAL_HELD_DECISION` in place of a treatment row at each identity node. No support row lists a held baseline treatment with `bootstrap`, so the percentile interval is shown and carries the label `experimental` |

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
adds a `time` column. A parameter name keeps the node index. `continuous_treatment=True` declares
the treatment a continuous dose. Node 1 then fits one conditional density, and a regimen is a
modified treatment policy.

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
| `time_varying=` on `TimeToEvent` | `CapabilityError`, code `api_redirection` |
| a censoring time between two grid times | `DataError` |

`to_time_to_event()` returns each unit's grid time and event code. It is the inverse map on a law
whose times sit on the grid. `tests/unit/test_time_to_event_input.py` checks one refusal per row
and the round trip on two grids. It also checks the long fit against the wide held fit. Floor
binning and the censor-first map are its mutations, and each one misses the truth.

### Long-panel input

`LongitudinalData.from_long_panel` and the design `LongPanel` read one row per unit and decision
time. A row holds the covariates measured before its decision, the decision, and the censoring
indicator of the node. The reader reshapes the rows into the wide arrays and calls
`LongitudinalData.from_frame`, so the wide checks apply unchanged. The estimand, the recursion, and
the influence curve are those of the wide layout. One unit is one observation, and `cluster=` maps to the cluster
contract.

| item | rule |
| --- | --- |
| node | the $t$-th decision time, counted from 1. `times=` declares the grid in time order and cannot coarsen it. Without it the reader sorts the times, which must be numbers, dates or durations |
| generated column | `"column@t"` for the treatment, each time-varying column, the censoring column, each event column and each known probability column. `PanelLayout` keeps the original time labels |
| unit order | the sorted unit identifiers. Unsorted input gives the same container |
| history | the baseline columns and the time-varying columns of nodes 1 to $t$. Every $L_t$ block adds columns to the history design of every later node |
| leaving the study | censoring 0 on the last row of a unit that left, or the event row. A row after that row is refused |
| event and censoring | an event value on a row whose censoring is 0 is refused on both layouts, because the order at a node is $A_t, C_t, Y_t$ |
| leading missing values | a time-varying column with no value at the first nodes starts at its first recorded node. `PanelLayout.omitted` records the nodes left out |
| report | `summary()` adds a `long panel:` line from `PanelLayout.describe()` |
| inverse | `to_long_panel()` returns one row per unit and node in the study. It normalizes the weights to mean one, leaves a screened-out covariate missing, and omits the known probability columns |

A time-varying column is measured before the row's decision, and the declaration states it. A
rule at node $t$ reads the columns of nodes 1 to $t$ through `each_node`. The reader refuses what
the data can show: a unit-level column that varies, a covariate that copies the outcome, and a
time-varying column that stops being recorded.
The [scope and refusals](scope-and-refusals.md) page lists them. The registered evidence uses two
decision nodes. [X39](../roadmap.md#x39-longitudinal-evidence-over-three-to-five-decisions) plans three to five.

`tests/unit/test_long_panel.py` checks the round trip on irregular censoring and unsorted input,
one refusal per row of the table, and the equality of `LTMLE` and `SDR` fits on the adapted panel
and on the equivalent wide frame. The `LTMLE` comparison also checks a nonzero targeting update, so
the equality is not between two plug-in estimates.

### A censoring node with no censoring

A censoring node at which no eligible unit is censored has the fixed factor one and fits no
learner. The eligible rows are the rows at risk before the node with positive weight. On the long
layout node 1 always has this shape, because every censoring time is at least $g_1$. A cross-fitted training
fold with no censored unit predicts one, its empirical rate. The nuisance report shows
`LONGITUDINAL_NO_CENSORING`, or `LONGITUDINAL_NO_CENSORING_IN_FOLD` beside the node's row.
`survtmle` sets `G_dC = 1` at `t = 1` and has a `noCens` branch for the same case. Before this rule a
standard classifier raised on the constant target. `tests/unit/test_no_censoring_node.py` removes
the rule and sees that error return.

### A node with no event

A node regression whose fitted rows all hold 0, or all hold 1, is that value, and no learner runs
there. A grid node at which no follower of a regimen had the event has this shape, with hazard
zero. Zero is the maximum-likelihood hazard, and `survtmle` reads such a node the same way. The
fluctuation at the node is skipped, because the score is exactly zero at the initial fit. The rule
applies to every outcome and pseudo-outcome node, on both layouts.

| case | result |
| --- | --- |
| no follower had the event at every node through the horizon | risk zero and a zero influence curve. The parameter takes the `constant_node_plugin` status, so it reports no interval or p-value |
| a node of the recursion read a constant, at a horizon with events at other nodes | the point estimate stands. The node's term of the influence curve is zero, so the parameter takes the `constant_node_plugin` status too |
| the followers of one outer training fold all hold 0, or all hold 1 | that fold's regression is the value. The pooled fluctuation then bounds its predictions as it does any prediction |
| `msm=` with a cell whose followers all hold 0, or all hold 1 | `CapabilityError`. The pooled logistic fluctuation would move the value into its bounds |
| a pseudo-outcome that holds one value inside $(0, 1)$, such as an intercept-only regression carried back | not read by this rule. The learner fits it and the fluctuation moves it, so a fit that ran before the rule keeps its numbers |

A node's term of the influence curve is identically zero when the node reads a constant. The
true outcome variance there can still be positive, so the plug-in standard error leaves that node's variance
out. A risk of zero would report a standard error of zero, and a contrast against it would carry
the other arm's variance only. Each parameter whose recursion reads such a node therefore takes
the `constant_node_plugin` status
([inference status](inference.md#inference-status)). The other parameters of the fit keep
theirs.

| output | what it shows |
| --- | --- |
| the fit | one `DataWarning` with the count of such parameters, the first 10 names, and the reason. `constant_node_parameters(result)` lists them all |
| `summary()` | the estimate and a starred plug-in spread, with "not reported" for the interval and the p-value, and the reason under the table |
| `curve()` and `to_frame()` | the status in the `inference` column, NaN in the inferential spread columns, and the plug-in spread in the `plugin_` columns |
| `incidence_total()` | each total takes the status of the incidences it sums, with the same columns as `curve()` |
| a contrast, `rmst()` and `rmtl()` | the status, when they read such a parameter |
| the simultaneous band | the other parameters only. A `CausalStudy` band narrowed to `RegimeMean` or `RegimeContrast` leaves them out too |
| `assess()` | the nuisance item adds the status note, with the count of such parameters |

[X54](../roadmap.md#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) part (i) holds an interval at such a node at a fixed law, and
[F31](../roadmap.md#f31-inference-at-a-boundary-node-estimate) keeps the rare-event regime.
The nuisance report shows `LONGITUDINAL_CONSTANT_TARGET` in place of the node's row, with the
regimen, the cause and the horizon. A cross-fitted training fold that reads a constant while the
sample does not adds `LONGITUDINAL_CONSTANT_TARGET_IN_FOLD`, which names the folds, and the node's
row stays.

`tests/unit/test_event_free_node.py` checks the zero risk, the status of each flagged parameter,
the omission rows and the absent learner fits. One mutation removes the rule and sees the bounded
fluctuation move the risk off zero. Another removes the status and sees the interval $[0, 0]$
return. `tests/e2e/test_ltmle.py` checks the fold-level omission, and its mutation removes the
in-fold reading and sees the omission go.

### Hazard-based survival targeting

`survival_model="hazard"` models a survival or competing-risk curve through its discrete
cause-specific hazards. The iterated-mean recursion targets each horizon in its own backward pass,
so its curve can decrease. A hazard fit reads every horizon from targeted hazards. Its one-step
algorithm reports a curve that never decreases. The implementation is
[`longitudinal/hazard.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/hazard.py).

The contract is the table. The symbol $r$ indexes a regimen, $j$ and $l$ index a cause, $k$ is a
node, and $t$ is a reported horizon.

| item | contract |
| --- | --- |
| design | one baseline treatment held over every node (`treatment="A"` or `TimeToEvent`). The treatment is categorical, the outcome is a survival or competing-risk outcome, and no covariate is measured after baseline. Every other design is refused before any learner ([hazard and isotonic refusals](scope-and-refusals.md#hazard-and-isotonic-survival-refusals)) |
| regimens | a static arm, a known rule $d(W)$, or a known stochastic policy $q(a \mid W)$ at the baseline decision. A static arm is $q = \delta_a$, and a rule is $q = \delta_{d(W)}$. A modified treatment policy is refused |
| estimand | $\psi_{r,j}(t) = E \sum_a q_r(a \mid W) F_j(t \mid a, W)$, the estimand of the [held design](#one-baseline-treatment-held-over-the-nodes). Contrasts, ratios, the survival view, `rmst` and `rmtl` derive from it as on an iterated-mean fit. No parameter name changes |
| identification | no unmeasured confounding of $A$ given $W$, censoring at random given $(A, W)$, and positivity. Moore and van der Laan (2009), Section 2. Cai and van der Laan (2020), Section 2 |
| hazard | $\lambda_l(k \mid a, W) = P(\text{cause } l \text{ at } k \mid \text{uncensored through } k,\ \text{event-free before } k,\ A = a,\ W)$. Then $S(k) = \prod_{s \le k}(1 - \sum_l \lambda_l(s))$ and $F_j(t) = \sum_{k \le t} S(k - 1)\,\lambda_j(k)$ |
| initial fit | the sequential pseudo-hazards of `survtmle` (`hazards_estimate.R`). Cause $j$ at node $k$ regresses its event on the rows at risk with no event of an earlier cause. Then $\lambda_j = \rho_j (1 - \sum_{i < j} \lambda_i)$. `outcome_learner` fits each node and cause, and the fit predicts every treatment level for every unit |
| influence curve | $D_{r,j,t} = \sum_{k \le t} \sum_l h^{r,j,t}_l(k, A, W)\,R_k\,(dN_l(k) - \lambda_l(k \mid A, W)) + \sum_a q_r(a \mid W) F_j(t \mid a, W) - \psi_{r,j}(t)$. $R_k$ indicates a unit uncensored through $k$ and event-free before $k$ |
| clever covariate | $h^{r,j,t}_l(k, a, W) = \frac{q_r(a \mid W)}{g(a \mid W)\,G_k(a, W)}\,1\{k \le t\}\bigl[1\{l = j\} - \frac{F_j(t \mid a, W) - F_j(k \mid a, W)}{S(k \mid a, W)}\bigr]$, with the retention $G_k = \prod_{s \le k} c_s$. Moore and van der Laan (2009), (11) to (16), at one cause. Rytgaard and van der Laan (2024), Section 3.1 |
| submodel | the multinomial logistic fluctuation $\eta_l(\epsilon) = \eta_l + \sum_m \epsilon_m h^m_l$ of the softmax logits $\eta_l = \log(\lambda_l / \lambda_0)$. $\lambda_0$ is the probability of no event. At one cause the submodel is (10) of Moore and van der Laan (2009) |
| bound | the fit clips each logit into $[\operatorname{logit}(1 - \alpha), \operatorname{logit}(\alpha)]$ after every update, with `alpha=`. So $\sum_l \lambda_l < 1$ holds, and no renormalization follows a clip |
| inference | the influence-curve variance and the multiplier band over the reported parameters, as on an iterated-mean fit. With `id=`, the cluster sums and the fit statuses of [clusters](#clusters) apply |
| cross-fitting | with `n_folds > 1`, the hazards, $g$ and $G$ are out of fold. One pooled targeting then runs over every row |

`outcome_learner` is the hazard learner on a hazard fit. `pseudo_learner=` is refused, because the
fit regresses the event indicator at each node and fits no pseudo-outcome. Every arm is targeted
together, so the score of every contrast is solved.

#### Two algorithms

`targeting=` chooses the algorithm. Both stop on the relative score. That score is each target's
score divided by its scale $\operatorname{mean}(\lvert w \rvert \sum_l \lvert h_l \rvert)$ over the rows at risk.

| item | per horizon, `targeting="iterative"` (default) | one step, `targeting="one_step"` |
| --- | --- | --- |
| source | Moore and van der Laan (2009), Section 4.1. Benkeser, Carone and Gilbert (2018), Appendix. `survtmle` `method = "hazard"` | Cai and van der Laan (2020), Section 5, Algorithm 2. Rytgaard and van der Laan (2024), Sections 3.3 and 3.4 |
| targets | every regimen and cause at one horizon | every regimen, cause and reported horizon together |
| one update | a finite-difference Newton step on the target scores along the current clever covariates. This step departs from the source. When the step does not reduce the scores, the fit falls back to the maximum-likelihood fluctuation with $h$ held fixed (Moore and van der Laan 2009, (13)) | a step of length `step_size` along $\Sigma^{-1} s / \lVert s \rVert_\Sigma$, with $s = P_n D^{(1)}$. A step that raises $\lVert s \rVert_\Sigma$ is halved |
| after each update | the fit rebuilds $h$ | the fit rebuilds $h$ |
| metric | none | $\Sigma = \operatorname{diag}(\max(P_n D_m^2, 10^{-12}))$ at the initial fit, fixed for the whole walk (Rytgaard and van der Laan 2024, Remark 2) |
| stop | every relative score at most `tol`, or `max_iter` outer iterations | every relative score at most `tol`, 20,000 accepted steps, or a step below $10^{-14}$ |
| next horizon | restarts from the initial hazards, as `survtmle` refits for each `t0` | not applicable. Every horizon reads the one set of hazards |
| ignored settings | `step_size` is accepted and unused, as on point `TMLE` | `max_iter` |

A fit that stops short sets `converged` to false and raises `ConvergenceWarning`. Its record names
the failure class. The accepted $\Sigma$-norm trace of a walk never rises. The log-likelihood
along the walk falls by at most $10^{-8}$ per step, because a finite step carries a second-order
term. The test `test_e11_the_walk_trace` checks both traces and the fixed $\Sigma$.

| output | what it shows |
| --- | --- |
| `summary()` | `hazard targeting: one step over N parameter(s), S step(s), largest relative score x, converged`, or the per-horizon form with the horizon count and the largest iteration count |
| `result.fits[...].hazard` | the targeting record: the targets, the initial and final scores, the relative-score trace, and the fixed $\Sigma$ of a walk. A one-step fit shares one record between every reported parameter |
| `diagnostics.score_equations()` | one `solver` row per regimen, cause and horizon, at the horizon node |
| `diagnostics.nuisance_models()` | one `hazard` row per node and cause, which scores the pseudo-hazard $\rho_j$ the learner fitted |
| `truncation_curve()` | a rerun of the targeting at each bound from the stored initial hazards. It refits no learner |
| `n_bootstrap=` | shown with the label `experimental`. No support row lists `bootstrap` for a hazard fit |

#### Monotonicity

| fit | what holds | evidence |
| --- | --- | --- |
| one step | each level curve $\psi_{r,j}(t)$ is non-decreasing in $t$ in floating point, with no tolerance, for every regimen and cause | `test_w1_the_one_step_curve_never_decreases` and `test_w1_each_cause_and_their_sum_never_decrease` in `tests/unit/test_survival_monotone.py` |
| one step or per horizon | at one horizon, the cause-specific incidences sum to one minus the all-cause survival, to a relative tolerance of $10^{-12}$ | `test_e6_the_causes_and_survival_add_to_one` |
| per horizon | no claim across horizons, because each horizon reads its own targeted hazards. On the W1 witness the curve decreases by a rounding error | `test_w1_the_per_horizon_curve_decreases_in_floating_point` |
| iterated mean | no claim. On the W1 witness the curve of arm 1 decreases by about $10^{-3}$ between horizons 2 and 3 | `test_w1_the_iterated_mean_curve_decreases` asserts a decrease larger than $10^{-6}$ |
| a contrast | no claim | none |

The one-step guarantee has a floating-point condition. Each unit's $F_j(t)$ is a running sum of
non-negative terms, and a policy mixture of such sums is a sum of non-negative products. The fit
then reduces each horizon's column in one order that does not depend on the values. Rounding to
nearest is monotone, so a weighted mean of non-decreasing columns in that order is non-decreasing.
The W1 witness is
`make_point_survival(n=300, seed=3, layout="wide", hazard_scale={3: 0.1})` with logistic learners
that draw no random number.

#### Conditions the interval inherits

| condition | statement | source |
| --- | --- | --- |
| identification | the conditions of the contract table | Moore and van der Laan (2009), Cai and van der Laan (2020), and Rytgaard and van der Laan (2024), each in Section 2 |
| positivity | $g(a \mid W)\,G_k(a, W) > \delta$ wherever $q_r(a \mid W) > 0$, and $S(k \mid a, W) > 0$ for $k < t$. `g_bounds` bounds the cumulative product as on an iterated-mean fit, and `alpha` bounds the softmax logits | Rytgaard and van der Laan (2024), Assumption 1 |
| remainder | the second-order remainder is $o_P(n^{-1/2})$. Its terms are products of $\hat g - g_0$ or $\hat G - G_0$ with $\hat\lambda - \lambda_0$. Donsker classes or cross-fitting control the empirical-process term | Moore and van der Laan (2009), Section 4.1.2 and Appendix E. Cai and van der Laan (2020), Section 4.3 |
| grid | the reported horizons are a finite grid fixed before the sample, so the band is over a stack of fixed dimension | the refining-grid Theorem 1 of Rytgaard and van der Laan (2024), Section 4.2, is not claimed |
| solver | every relative score is at most `tol`. Otherwise `converged` is false, and the fit warns | the shipped solver rule |
| cross-fitting | the out-of-fold nuisances and one pooled targeting solve every component score | Zheng and van der Laan (2011), Section 2, applied to each algorithm's endpoint. The [cross-fitted hazard study](method-evidence/point-survival-curve-crossfit.md) measures the one-step walk and one per-horizon cell at five folds |

No registered study measures the coverage of a hazard fit yet. The evidence today is the exact-law
and mutation tests below.

#### A hazard cell with no event

A hazard cell is one node, one treatment level and one cause. Its followers are the rows at risk at
the node, with positive weight, that received the level. The fit holds a cell's hazard at exactly
zero for every unit when its followers exist and none has an event of that cause. Zero is the
maximum-likelihood value, and no fluctuation moves it.

The hold replaces the learner's prediction for every unit in the cell, whatever the learner.
A zero that holds only within a stratum of $W$ is not a held cell. The fit clips its logit to the
floor $\operatorname{logit}(1 - \alpha)$, and the fluctuation then moves it. That floor is a
hazard of about $5 \times 10^{-4}$ at the default $\alpha$, so it biases a fit whose true
per-node hazards lie below it.

A cell whose followers all have an event is *full*. The fit holds the all-cause hazard of a full
cell at exactly one, its maximum-likelihood value. The survival after a full cell is zero, and each ratio over that survival is zero there.

The hold is necessary. Without it, the fit clips the logit at its floor. The joint score equations
then need that hazard below the floor, so the one-step walk cannot solve them.
The mutation `test_m_a_clipped_cell_stalls_the_walk` removes the hold and sees the walk fail to converge.
The full-cell module has a mutation of the same name, and it clips the full cell instead.

Each parameter whose regimen reads a flagged cell at a node up to its horizon takes the
`constant_node_plugin` status. A regimen reads a level when it puts mass on that level for some
unit. `RegimenFit.hazard_constant_nodes` lists the flagged nodes, and `constant_nodes(fit)` reports
them. The status then works as [a node with no event](#a-node-with-no-event) states.

The empty-cell witness is
`make_point_survival(n=400, seed=36, layout="wide", hazard_scale={3: 0.05})`. Arm 1 has followers at
risk at node 3, and none has the event. Its risks at horizons 3 to 5 take the status, and its risks
at horizons 1 and 2 keep their intervals (`tests/unit/test_hazard_empty_cells.py`). In the
full-cell witness, every arm-1 follower at node 2 has the event. Its risk at horizon 2 is 1 and
takes the status (`tests/unit/test_hazard_full_cells.py`).

A node and cause whose training rows hold no event of that cause fits no learner. The nuisance
report shows `LONGITUDINAL_CONSTANT_TARGET` in place of its `hazard` row.

#### Evidence

| test module | what it checks |
| --- | --- |
| `tests/unit/test_hazard_survival_law.py` | on the exact law of `tests/discrete_law_point_survival.py`, both algorithms: the saturated estimate equals the g-formula, the influence curve equals the Gateaux derivative, and targeting repairs a wrong hazard fit with the exact mechanism. The maximum-likelihood fallback alone also repairs it. Also every score solved, one set of hazards per one-step fit, the competing-risk identity, policies, weights, the replay, and the walk trace |
| `tests/unit/test_hazard_survival_mutations.py` | M1 to M9, M11, M12 and M14: each deliberate mutation breaks the identity it should break. M14 reads the metric from the coefficients of each step. M4 (a sign flip under per-horizon targeting), M12 (the cause order) and M15 (the softmax clip) are positive controls |
| `tests/unit/test_survival_monotone.py` | the W1 witness of the monotonicity table |
| `tests/unit/test_hazard_empty_cells.py` | the hold, its mutation, and the status |
| `tests/unit/test_hazard_full_cells.py` | the hold of a full cell at one, its mutation, the status, and per-horizon convergence |
| `tests/unit/test_hazard_refusals.py` | each refusal H1 to H11, by type and message, before any learner |
| `tests/e2e/test_hazard_survival.py` | the `TimeToEvent` study path with three arms, two causes, a policy, weights, clusters and two folds. Also the replay, the bootstrap diagnostic, the summary, the nuisance report, and `save` and `load` |

The registered study [point survival hazard](method-evidence/point-survival-hazard.md) validates
the per-horizon algorithm on repeated samples of two exact laws. Its `survtmle` comparator runs
are pending.

The registered study [point survival curve](method-evidence/point-survival-curve.md) validates the
one-step walk on repeated samples of the `survival` law and on further exact laws in its property
cells. Its `tmle3` comparator runs are pending.

The registered study [point survival curve crossfit](method-evidence/point-survival-curve-crossfit.md)
validates the one-step walk at five folds on the same exact laws. It has no comparator.

### The isotonic projection

`isotonic=True` projects each survival or competing-risk curve onto a non-decreasing curve over the
reported horizons. It applies to an `LTMLE` fit with either survival model. Westling, van der Laan
and Carone (2020), Section 2.1, give the rule. On a grid in one dimension the projection is the
pool-adjacent-violators algorithm (PAVA).

| item | rule |
| --- | --- |
| levels | the fit projects the reported levels of each regimen and cause, in horizon order, by unit-weight PAVA (`scipy.optimize.isotonic_regression`). A curve that does not decrease is returned bit for bit |
| curve band limits | the fit builds the band at the unprojected levels. It then projects each curve's lower limits and upper limits by the same PAVA (Westling et al. 2020, Theorem 1 (iv) and (v)) |
| contrasts | each contrast is the difference or ratio of projected levels. Its band is kept as built |
| what does not move | every influence curve, every standard error, and the critical value |
| a parameter with no interval | a `constant_node_plugin` parameter is left out of the band, and so out of the band projection |
| record | `result.isotonic_moves` lists `(regimen, cause, horizon, before, after)` for each moved level. `summary()` prints `isotonic projection moved k level(s)` |
| identity | a one-step hazard fit, a fit with one reported horizon, and a curve that does not decrease |
| refused | an end-of-study outcome (H3), `msm=` (H5), and a point-treatment functional (H10) |
| bootstrap | shown with the label `experimental`. No support row lists `bootstrap` for a projected fit |

R `lmtp` 1.5.4 `lmtp_survival` projects over every grid time with `isotone::gpava` and unit
weights. `cleverly` projects over the reported horizons only. The two agree when the reported
horizons are every grid time. `test_e9_the_projection_is_pava_and_lmtps_transcription` transcribes
the `lmtp` arithmetic over all grid times and compares it with the fit.

The projection changes what the reported numbers mean. The table states each claim and its
condition.

| claim | condition | source |
| --- | --- | --- |
| the projected curve band covers the truth whenever the unprojected band covers it | any $n$ | Westling et al. (2020), Theorem 1 (iv). `test_e9b_the_projected_band_keeps_coverage_and_is_no_wider` |
| the projected curve band is no wider on average or at its maximum | any $n$ | Theorem 1 (v). The same test |
| the projected levels have a sup error over the reported horizons no larger than the unprojected levels | any $n$ | Theorem 1 (i) |
| a pointwise interval at a projected level, with the unprojected standard error, is valid | only when the true curve is strictly increasing at the reported horizons. On a fixed finite grid the projection is then the identity with probability tending to one | Theorem 2, and the sentence after Corollary 1 |
| the influence curve of a projected level is the unprojected one | a reporting convention, not an asymptotic statement. At a flat region the limit law of a projected level is not Gaussian | Theorem 2 and Corollary 1 cover the strictly increasing case only |

Two mutation controls guard the rule. The test
`test_m13_a_band_centred_on_the_projected_estimate_fails_e9b` centres a band on the projected
estimate with the unprojected half-widths, and that band loses coverage. The test
`test_m10_an_influence_curve_moved_with_its_level_fails_e9` moves the influence curve with the
level, and the identity check fails. No registered study measures a projected fit yet.

### Known node mechanisms

A sequentially randomized trial, such as a SMART, fixes its treatment probabilities by design. A
study can also know its retention probabilities. Declare either with `treatment_probabilities=`
and `censoring_probabilities=` in `LTMLE.fit` or in `LongitudinalData.from_frame`. In the study
API, declare them with `LongitudinalTreatment(treatment_probabilities=...,
censoring_probabilities=...)`. A time-to-event input declares its baseline mechanism as a level to
column mapping, in `LongitudinalData.from_time_to_event(treatment_probabilities=...)` or in
`TimeToEvent(treatment_probabilities=...)`. A declared node fits no learner, so the fold-support
check skips it.

| keyword | form | reading |
| --- | --- | --- |
| `treatment_probabilities` | `{"A1": {0: "p1_0", 1: "p1_1"}, ...}` | each node's levels mapped to frame columns of $P(A_t=a\mid\text{observed past})$. Nodes left out are estimated. This form is required when the nodes have different level sets |
| `treatment_probabilities` | `(n, T)` array | $P(A_t=\text{levels}_t[1]\mid\text{observed past})$, at binary nodes only |
| `treatment_probabilities` | `(n, T, K)` array | the full matrix, when every node shares $K$ levels |
| `censoring_probabilities` | `{"C1": "r1", ...}` or `(n, T)` array | $P(C_t=1\mid\text{observed past})$ |

A value on a row that is not at risk at its node is never read, and it may be missing. A missing
value on an at-risk row is refused with the row count. A node may give an arm probability one, as a
SMART does for the units it does not re-randomize. A declared zero for the arm an at-risk unit took
is refused, because the data contradict it. A declared retention of zero for a unit that stayed is
refused for the same reason.

A declared zero can also leave a regimen unidentified. The fit refuses such a regimen before any
learner. The check reads the rows that followed the plan through the previous node and are at risk
at this one.

| node | refused when, on a row at risk under the plan |
| --- | --- |
| label or rule | the declared probability of the assigned arm is zero |
| known stochastic policy | the policy puts mass on an arm whose declared probability is zero |
| modified treatment policy | the policy sends a unit to a level whose declared probability is zero |
| declared censoring | the declared retention of a unit that took the plan's arm is zero |

No unit with such a history can follow the regimen, so its mean there would extrapolate the
outcome regression. The message names the regimen, the node and the row count.
`tests/unit/test_known_node_mechanisms.py` checks a SMART whose responders keep their first arm with
certainty: the regimen that switches them is refused, and `always` is fitted.

A node's factor is the declared probability of the arm each row is *assigned*. The clever covariate
is nonzero only where the plan's history equals the observed one. The declaration describes the
observed history, so the fit reads it only where the two agree. That is the meaning of `ltmle`'s
numeric `gform`. On a held design the one decision is the one declared node.

With every treatment and censoring factor known, the remainder is zero and the curve is
$D^*(\bar Q_\infty,g_0)$ for any outcome regression. Theorem 2 of van der Laan and Gruber (2012)
states the double robustness. Their Section 4, page 19, states the curve at $g_n=g_0$. Petersen et
al. (2014), Appendix B, Corollary 1, give the identity $-P_0D^*(Q,g_0)=\Psi(Q)-\psi_0$.

With the treatment declared and the censoring estimated, the censoring factor must be consistent.
That is the common SMART analysis. With a wrong outcome regression, the interval is conservative
for a correctly specified parametric censoring model. It is not established for a flexible
censoring learner.

**Bounds.** The cumulative bound applies to the running product of the declared factors on every
row that follows a plan. A product outside `g_bounds` moves the estimate with no variance reason,
so the fit refuses before any learner. The message names `g_bounds` and the node where the product
first leaves the bounds. Products fall fast: 0.5 at each node reaches 0.01 near the seventh node.
Pass a lower bound below the smallest declared product, such as `g_bounds=(1e-8, 1.0)`.

`truncation_curve()` runs the same check on each pair of its grid before any replay. A pair that
the fit refuses is refused with a sentence that quotes the fit's condition, and with the
`invalid_declaration` code. A curve row therefore never truncates a declared product. With every factor declared, the curve over
admitted pairs is flat. With the censoring estimated, an admitted pair can still truncate the
estimated factor, and the curve reports that movement. `tests/unit/test_known_node_mechanisms.py`
pins the refusal, the flat curve and the estimated-censoring case.

| composition | result |
| --- | --- |
| static, dynamic and known stochastic plans, `msm=`, cross-fitting, clusters, survival and competing risks, held and time-to-event designs, `n_bootstrap=` | supported. A bootstrap replicate carries the declared values of its own rows |
| a modified treatment policy at a categorical declared node | supported. The ratio numerator $g^d_t(A_t\mid H_t)$ reads the declared matrix by the discrete formula. A declared node is the degenerate estimate $g_n=g_0$, so the remainder, a product of the ratio error and the outcome error, is zero |
| a continuous node | refused. A known density is not supported |

The nuisance report shows `LONGITUDINAL_KNOWN_MECHANISM` in place of each declared node's row.
`LongitudinalConfig.known_factors` names the declared treatment and censoring columns, and the
summary prints them on one line.
`tests/unit/test_known_node_mechanisms.py` checks a two-node SMART exact law: a wrong outcome
regression with every factor declared returns the truth, with and without declared censoring. It
also checks four compositions: a modified treatment policy, a held survival design, competing risks
and a time-to-event container. In each one, a declared factor and a learner that returns it give
one fit. The
registered [known-node-mechanisms study](method-evidence/known-node-mechanisms-ltmle.md) pairs
the fit with R `ltmle` at a numeric `gform`. The
[sequential offers tutorial](../examples/sequential-offers.ipynb) declares a three-level offer at
three nodes beside an estimated consent model.

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
| licensing scope | The `bootstrap` kind of a support row decides the label. A fit is covered when it has an end-of-study or single-cause survival outcome, is fitted in sample, uses static regimens, one treatment decision per node, a binary outcome, no weights and no working model. Any other fit, such as a held baseline treatment, matches no such row. The rows `G01` and `G11.nodes4` list it, and the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) measured them. A cross-fitted fit, a clustered fit and a clustered cross-fitted fit have no such row, so their percentile interval carries the label `experimental`. Owner `X20-bootstrap` holds the red cells. A fit with fewer than 20 clusters prints `bootstrap sd` and a percentile range, whatever its status. The study measures correctly specified cell-mean nuisances on finite binary laws, at $n = 1000$, and at 1,500 rows in 60 clusters for the cluster bootstrap. No result covers a data-adaptive nuisance. Cai and van der Laan (2020) is the warning for that case |
| derived estimates | `ratio`, `rmst`, `rmtl` and `contrast` apply the same function to each replicate's estimates, and attach the percentile interval of those values. The derived interval is licensed only when every input's interval is |

## Variations

| option | what it does |
| --- | --- |
| `regimens=` | static plans, fixed rowwise dynamic rules, categorical arms, or [known stochastic policies](#known-stochastic-policies). A plan is a sequence of arms, or one arm meaning that arm at every node. A plan with a callable node must be a `DynamicRegimen` declared `rule_kind="known"`. `LTMLE.fit` refuses an undeclared or `"estimated"` rule, and a callable written inline in a mapping, before any learner. Sample-adaptive thresholds and learned rules need additional inference and are outside this contract. The [scope page](scope-and-refusals.md#wrong-by-construction) gives the threshold witness |
| `reference=` | which regimen the contrasts are taken against. It is part of the estimand rather than a display setting |
| `horizons=` | which time points a survival fit reports cumulative risk at. `None` reports the whole curve. Name the horizons you will report: the cost is $T(T+1)/2$ regressions per regimen rather than $T$ |
| `survival_model=` | `"iterated_mean"` (default) runs one backward recursion per horizon. `"hazard"` fits and targets the cause-specific hazards, as [hazard-based survival targeting](#hazard-based-survival-targeting) states. The outcome learner is then the hazard learner |
| `targeting=`, `step_size=` | the hazard targeting algorithm: `"iterative"` (default) per horizon, or `"one_step"` over the whole curve. `step_size` is the first step of the one-step walk. `"one_step"` requires `survival_model="hazard"` |
| `isotonic=` | projects each curve's levels and band limits onto a non-decreasing curve, as [the isotonic projection](#the-isotonic-projection) states |
| `msm=` | a working model over the regimen and horizon cells, in sample or cross-fitted. See [MSM projections](msm-projections.md) |
| four learner slots | `outcome_learner`, `pseudo_learner`, `treatment_learner`, `censoring_learner`. The pseudo learner fits the intermediate regressions. Under `LTMLE` their outcome is a bounded prediction. Under [`SDR`](#the-sequentially-doubly-robust-estimator) it is a transform that can leave $[0, 1]$, so the pseudo learner must fit a real-valued target |
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
| the full-refit bootstrap | the estimator of each row above | refit the whole estimator on each resample | none for this estimator. The percentile interval is labelled `supported` only when a support row lists `bootstrap` and its cells are green, with correctly specified cell-mean nuisances on finite binary laws | the bootstrap cells of the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) |

Theorem 3 also requires every mechanism ratio and targeted sequential regression to be consistent,
the sum over nodes of their error products to be $o_P(n^{-1/2})$, and the density ratios to stay
bounded. On a continuous node a binned density meets the mechanism-ratio rate only if its bin
count grows with $n$ (see the bin count paragraph under modified treatment policies). Those are conditions on the targeted regressions. Appendix E of the competing-risk paper
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
`interventions=` redirects to `regimens=`. `incremental=` redirects to
[`IPSI`](#incremental-interventions-over-time), which fits the odds tilt over time. The table gives the
refusals that are statements about the question, the construction, or coverage.

| refused | code | what it would need |
| --- | --- | --- |
| eliminating the competing events | `different_estimand` | what is reported is the cause-specific cumulative incidence with the competing causes *left alone*, so a competing event is part of the history. Removing it makes it an intervened node: a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions |
| `intermediate=` | `different_estimand` | a controlled direct effect fixes a mediator at one time point. Over a sequence, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| a `truncation_curve()` pair that truncates a declared cumulative product | `invalid_declaration` | the fit refuses the same pair as `g_bounds`, because truncating a known product moves the estimate with no variance reason. A curve row at that pair would show a movement that no admitted fit has. Remove the pair from the grid. A pair `(lower, 1)` with `lower` below the smallest declared product is admitted |
| `mechanism=True` on `truncation_curve()` | `invalid_declaration` | a longitudinal fit holds one cumulative treatment-and-censoring bound, and it fits no separate observation mechanism to sweep. The option names a point-treatment axis, so the call raises `CapabilityError` rather than sweeping the cumulative bound under another name |
| a callable node that is not declared `"known"`, or that is declared `"estimated"` | `invalid_declaration` when undeclared, `implementation_gap` when `"estimated"` | the fit treats each rule as fixed. A rule learned from the analysis sample needs a learned-policy estimand and its own inference. `LTMLE.fit` raises `CapabilityError` before any learner. Declare a rule fixed before the fit with `DynamicRegimen(label, plan, rule_kind="known")` |
| a callable written inline in a `regimens=` mapping | `invalid_declaration` | an inline callable carries no declaration, so the fit refuses it before any learner. Write the plan as a `DynamicRegimen` declared `rule_kind="known"`, with `(rule,) * T` for one rule at every node |
| an outcome missing for a reason other than censoring | `api_redirection` | left as it is, the probability of observing it is silently taken to be one. Encode it as a final censoring column, so it is estimated and enters the cumulative product |
| longitudinal sensitivity-bound estimation | `implementation_gap` | a sample estimator and sampling theory for its bound functionals. [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) holds the stop |
| a continuous outcome with `q_bounds=None` above one fold | `invalid_declaration` | with `q_bounds=None` the scale comes from every observed outcome, held-out rows included, and no shipped result covers that scale |

Two plan shapes raise `DataError` before any learner. Each one is a statement about the input, so
the table of codes above does not list it.

| plan | what the refusal asks for |
| --- | --- |
| an iterator in `regimens=`, such as a generator | a tuple, or a `DynamicRegimen`, which reads its plan once and stores it as a tuple. The fit does not read the iterator: a fit reads `regimens=` at each call, and an iterator is empty after its first read |
| a `DynamicRegimen` plan that is one label or one callable | one entry per node. Write one rule for every node as `(rule,) * T` |

See [scope and refusals](scope-and-refusals.md#how-to-read-a-refusal) for what each code means.

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
| bootstrap | no support row lists `bootstrap` for a clustered cross-fitted fit, so the percentile interval is shown with the label `experimental`. Each replicate redraws a grouped split over its resampled clusters, so two copies of one cluster can land in different folds |

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
node is included too. [`IPSI`](#incremental-interventions-over-time) uses the same handling for an odds
tilt.

Two side effects follow from the cross-fitted default. A fit with fewer than 10 clusters warns that
it reduces `n_folds` to the cluster count, and a fit with fewer than 20 reports no interval. At few clusters a training fold can lack a first-node
level. The fit then raises `LongitudinalError` after the draw, and the message names the
in-sample fit. A training fold whose followers all hold one outcome value reads that value, as
[a node with no event](#a-node-with-no-event) states.

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
| 4 to 19 clusters | the fit takes `"few_cluster_plugin"` and reports no interval. No cross-fitted study measures that range: a probe of 2,000 draws at 10 clusters found 3 and 8 draws that admit no fit, and the harness refuses a cell that loses a replication. [X54](../roadmap.md#x54-small-sample-evidence-for-clustered-intervals-and-boundary-nodes) part (b) holds it |

## The sequentially doubly robust estimator

`SDR` estimates the same parameters as `LTMLE` with the sequentially doubly robust estimator of
Díaz, Williams, Hoffman and Schenck (2023), Section 5.3. It fits no fluctuation. Each node's
regression fits a transformed outcome, and the estimate is the mean of the first transform. The
estimator is consistent when, at each node, either the mechanism ratio or the regression of the
transform is consistent (Lemma 4). That property is sequential double robustness.

LTMLE does not have it. In `tests/unit/test_sdr_double_robustness.py`, node 1's mechanism and
node 2's regression are both wrong. SDR is exact on that law, and LTMLE is not.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import SDR

frame, _ = make_longitudinal(n=300, seed=0)
result = SDR(
    {"always": 1, "never": 0},
    reference="never",
    outcome_learner=LogisticRegression(),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(),
    censoring_learner=LogisticRegression(),
    n_folds=1,
    simultaneous=False,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
print(result.summary())
```

`sdr()` is the one-call form, and `CausalStudy(...).estimate(method="sdr")` routes a
longitudinal functional to the same engine. A point-treatment functional refuses
`method="sdr"`.

### The transform

Write $\pi_t$ for the intervention density at the observed arm. It is 1 at a label or rule node,
$q_t$ at a [known stochastic policy](#known-stochastic-policies) node, and the policy numerator at
a [modified treatment policy](#modified-treatment-policies-at-a-node) node. Write $p_t = g_t c_t$
for the treatment and censoring mechanism at the observed arm on the plan. Write $\check m_t$ for
the node regression. The transformed outcome is Equation (4) of the paper:

$$
\varphi_t = v_t + \sum_{s=t}^{h} W_{t,s} \bigl(Z_s - \check m_s(A_s, H_s)\bigr),
\qquad
W_{t,s} = \frac{\prod_{k=t}^{s} \pi_k}{\operatorname{bound}\bigl(\prod_{k=t}^{s} p_k\bigr)}.
$$

| term | definition |
| --- | --- |
| $v_t$ | the value node $t$ carries: the regression at the plan's arm, or the policy-weighted mean of the per-arm regressions at a policy node |
| $Z_s$ | $v_{s+1}$ composed with the event at node $s$, as the shipped pseudo-outcome composes it. $v_{h+1}$ is the scaled outcome, or 0 at a survival horizon |
| $W_{t,s}$ | the weight of the path from node $t$ to node $s$, on the rows that follow the plan through $s$, and 0 elsewhere |
| node-$t$ regression target | $\varphi_{t+1}$ composed with the node's event for $t < h$, and the outcome at the horizon |
| estimate | the weighted mean of $\varphi_1$ |
| influence curve | $w(\varphi_1 - \hat\psi)$. The standard error is the empirical standard deviation of this curve, the sentence below Theorem 4 |

`_recursion` in
[`longitudinal/sdr.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/sdr.py)
implements the table. `tests/unit/test_sdr_construction.py` compares it with an independent numpy
recursion to `1e-12`, in sample and at five folds.

### Contract

| item | contract |
| --- | --- |
| base result | Díaz, Williams, Hoffman and Schenck (2023), *Journal of the American Statistical Association* 118(542). Equation (4) defines $\varphi_t$. Theorem 2 gives the efficient influence function $\varphi_1 - \theta$. Section 5.3, journal page 854, gives the estimator, Lemma 3, Lemma 4 and Theorem 4 |
| rate condition | $\sum_t \lVert \hat r_t - r_t \rVert\, \lVert \check m_t - \tilde m_t \rVert = o_P(n^{-1/2})$, Theorem 4. Here $\tilde m_t$ is the regression of the estimated transform, not of the true one |
| efficiency | the variance reaches the bound only when every nuisance is consistent, and when $\lVert \varphi_1(\hat\eta) - \varphi_1(\eta) \rVert = o_P(1)$ |
| exact or known ratios | the remainder is 0 for any outcome learner. With the fold-measurable construction below and a GLM regression that converges to a fixed limit, the estimate is asymptotically normal with variance $\operatorname{Var} \varphi_1(r, \bar m)$ |
| bounded ratios | $P\{r_t < c\} = P\{\hat r_t < c\} = 1$ for some $c$. `g_bounds` gives the second |
| consistency without a rate | Lemma 4: at each node, the ratio is consistent or $\lVert \check m_t - \tilde m_t \rVert \to 0$. Lemma 4 gives no interval |
| in sample | `n_folds=1` rests on Donsker-type conditions (Luedtke et al. 2017, condition DC), not on Theorem 4, as for LTMLE |
| no interval claimed | multiply robust intervals. Page 854 states that neither SDR nor TMLE gives them |

### Cross-fitting

Section 5.3, Step 2 composes a training row's transform from the folds that hold that row out.
`lmtp` reads a training row's ratio from the out-of-fold mechanism. Neither transform is fixed
given the fold's training rows. The proof of Theorem 3 (supplement Section 6) conditions on that
fixed fit. Section 5.4 states that cross-fitting keeps the independences that the proofs of
Theorems 3 and 4 need. `SDR` therefore ships the training-fold recursion.

| part of the fit | rows | what the fit does |
| --- | --- | --- |
| fold mechanism | the training rows of fold $k$ | fit the treatment and censoring mechanism, and read it at every row |
| fold recursion | the training followers of fold $k$ | run the recursion of the transform table, with the fold's own regressions and its own mechanism in each training row's transform |
| held-out transform | the held-out rows of fold $k$ | compute $\varphi_1$ from fold $k$'s regressions and mechanism |
| estimate | every row | average the stitched $\varphi_1$ |

With ratios that do not depend on the fold, such as a known mechanism, the construction equals
`lmtp::cf_sdr`. Rotnitzky, Robins and Babino (2017), Section 5.2.1, leave the rate of an
in-fold transform's regression open. Theorem 4 assumes that rate, as the LTMLE contract assumes
its own. `tests/unit/test_sdr_construction.py` carries two mutation controls. The stitched Step 2
moves the estimate by more than `1e-4`, and so does a training row that reads the out-of-fold
mechanism.

### Choices that differ from LTMLE

| choice | `SDR` | reason |
| --- | --- | --- |
| fluctuation | none. `alpha=`, `max_iter=` and `tol=` raise `MethodConfigurationError` with code `invalid_declaration` | the transform replaces the targeting step |
| regression clip | none. The default `pseudo_learner` is `SuperLearner(task="regression", clip=None)` | a transform can leave $[0, 1]$, and `lmtp` predicts it unbounded. A clipping pseudo learner is accepted and moves the estimate away from `lmtp_sdr` (`tests/unit/test_sdr_bounds_and_clipping.py`) |
| `g_bounds=` | bounds each partial path probability $\prod_{k=t}^{s} p_k$ | at $t = 1$ this is the cumulative product that LTMLE divides by, equal to rounding. Where a bound binds, the bounded weight is a misspecified ratio on that path, and no SDR property is claimed there. `diagnostics.support()` reports the binding share per node, over the paths that the estimate reads. A cross-fitted fit counts the held-out rows of each fold |
| a level outside the outcome range | reported as computed, with `ParameterSpaceWarning` | an average of transforms can leave $[0, 1]$, or $[q_\text{lo}, q_\text{hi}]$ on a continuous outcome (page 854) |
| `ratio()` of such a level | `CapabilityError` on the log scale for a level at or below 0, and on the odds scale outside $(0, 1)$ | the scale is undefined there. The difference scale stays available |
| `msm=` | the identity link always. A logit or log link when its projection converges, and `CapabilityError` otherwise | a transform mean outside the link's range has no finite solution |
| full-refit bootstrap | runs, and reports `bootstrap sd` with no percentile inference | no registered study measures it |
| score equations | `diagnostics.score_equations()` is `not_applicable` | no node fits a fluctuation |

Clusters, observation weights, survival, competing risks, known policies, modified treatment
policies, contrasts, simultaneous bands and the truncation replay behave as they do for LTMLE.
Each composition reuses the step that the LTMLE contract states. `tests/unit/test_sdr_e2e.py`
runs each one end to end.

## Incremental interventions over time

An incremental intervention multiplies each unit's odds of treatment by a fixed $\delta$ at each
node. It shifts the treatment mechanism and does not assign an arm. Every unit at risk stays in
the plan, so the estimand needs no treatment positivity at a tilt node. A static plan such as
"always treat" needs positivity wherever it puts mass.

`IPSI` in `cleverly.longitudinal` fits this estimand on a `LongitudinalData` panel with a binary
treatment at each node. "Treatment" is the larger of the node's two sorted codes, so the
odds are those of that code. It implements the one-step estimator of Kennedy (2019), Algorithm 1. It
adds the dropout weights of Kim, Kennedy and Naimi (2021), Appendix A, Algorithm 1. It is not a
TMLE, and it fits no fluctuation.

| your situation | what `IPSI` gives | what it costs |
| --- | --- | --- |
| some histories almost never receive one arm | a plan mean with no treatment positivity at a tilt node | the treatment mechanism must be consistent at every tilt node, because the estimator is not doubly robust |
| you want a dose-response in the odds of treatment | one plan per $\delta$, with contrasts and a simultaneous band over the declared plans | the band covers the declared plans, not a continuum of $\delta$ |
| units drop out | the dropout weights of Kim et al. (2021) | a censoring model per node, and positivity of remaining under observation |
| the plan mixes a tilt with label, rule or known-policy nodes | one recursion that covers every node kind | positivity at each label and rule node |

`IPSI` is Kennedy's odds tilt. `lmtp::ipsi` is a risk-ratio tilt, which this package fits as
`RiskRatioTilt` with `LTMLE` or `SDR`. The point-treatment odds tilt is
[`TMLE(incremental=...)`](point-treatment-tmle.md#incremental-propensity-score-interventions).
Implementation:
[`longitudinal/ipsi.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/ipsi.py)
holds the recursion, and the `IPSI` class is in
[`longitudinal/estimator.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/estimator.py).

### The incremental estimand

Write $\pi_t(H_t) = P(A_t = 1 \mid H_t)$ for the treatment mechanism at node $t$ among the units
at risk. A tilt node draws the arm from Equation (1) of Kennedy (2019):

$$
q_t(1 \mid H_t) = \frac{\delta_t \pi_t(H_t)}{\delta_t \pi_t(H_t) + 1 - \pi_t(H_t)}.
$$

The target $\psi$ is the mean outcome when every node follows its plan. Theorem 1 of Kennedy
(2019) identifies it under consistency and sequential exchangeability (Assumptions 1 and 2). It
needs no positivity, for each $\delta_t$ in a compact subset of $(0, \infty)$.

Under dropout, Theorem 3.1 of Kim et al. (2021) adds missing at random given the history
(Assumption A2-M). It also adds dropout positivity (Assumption A3). Then $\psi$ is the plan mean
with dropout prevented. At $\delta = 1$ it is the natural course with dropout prevented, not the observed mean.

| plan | how to declare it |
| --- | --- |
| one $\delta$ at every node | `IPSI([Incremental(0.5), Incremental(1.0), Incremental(2.0)])`. Each `Incremental` is one plan, and its `name` is the label |
| one $\delta_t$ per node | `DynamicRegimen("ramp", (Incremental(1.5), Incremental(2.0)))` |
| a tilt beside label, rule or known-policy nodes | `DynamicRegimen("tilt then stop", (Incremental(2.0), 0))` |
| a plan with no tilt node | a static or dynamic plan, so a tilt can be contrasted with it on one fit |

`reference=` must name one of the declared plans. List `Incremental(1.0)` to contrast each plan
with the natural course. The parameter names are the longitudinal ones, such as
`ey_regimen[odds x2]` and `ate_regimen[odds x2 vs natural course]`.

A mean of a non-absorbing outcome $Y_t$ under a tilt is the end-of-study estimand on the panel
truncated after node $t$. `IPSI` fits it with `outcome=Repeated([...])`
([below](#ipsi-on-a-repeated-outcome)).

### The tilt recursion and its influence function

Each node regression $m_t(H_t, A_t)$ fits the target $Z_{t+1}$ on the rows at risk and under
observation through $t$. The node carries the tilted mean of its per-arm predictions,
$\bar m_t = \sum_a q_t(a \mid H_t)\, m_t(H_t, a)$. That is Remark 2 of Kennedy (2019). The
recursion runs from the horizon $h$ back to node 1.

| quantity | definition |
| --- | --- |
| $Z_{h+1}$ | the outcome, or the event indicator at a survival horizon |
| $Z_{t+1}$, survival | $Y_t + (1 - Y_t)\,\bar m_{t+1}$. A competing-risk fit uses one cause's indicator and the all-cause survival factor |
| tilt factor $r_t$ | $(\delta_t A_t + 1 - A_t)/(\delta_t \pi_t + 1 - \pi_t)$. The fit never computes it as $q_t / \pi_t$ |
| label, rule or policy factor | the plan's density at the observed arm over the treatment mechanism |
| censoring factor | $1\{C_t = 1\}/\omega_t$, with $\omega_t$ the probability of remaining under observation |
| weight $W_t$ | the product of every node factor through $t$, and 0 off the rows that follow the plan through $t$. $W_0 = 1$ |

The uncentred efficient influence function at the fitted nuisances is

$$
\varphi = \bar m_1(H_1) + \sum_{t=1}^{h} W_t \{Z_{t+1} - m_t(H_t, A_t)\}
  + \sum_{t \in \text{tilt}} W_{t-1}
    \frac{\delta_t \{m_t(H_t, 1) - m_t(H_t, 0)\}(A_t - \pi_t)}{(\delta_t \pi_t + 1 - \pi_t)^2}.
$$

The form follows Kennedy (2019), Theorem 2 and Appendix Lemmas 2 to 4. Kim et al. (2021),
Theorem 4.1, adds the censoring factors. The last sum is the *mechanism term*. It exists because
$q_t$ reads $\pi_t$. A known policy has no such term, and a label or rule node has none.

The estimate is the weighted mean $\hat\psi = \sum_i w_i \varphi_i / \sum_i w_i$. The influence
curve is $w_i(\varphi_i - \hat\psi)$, summed within a cluster under `id=`. The standard error is
the empirical standard deviation of that curve, as in Kennedy (2019), Theorem 3.

`tests/unit/test_ipsi_exact_laws.py` compares each estimate with the g-formula of
`tests/discrete_law_longitudinal_ipsi.py` to $10^{-12}$. It compares each curve with the
complex-step Gateaux derivative to $10^{-10}$. The oracle reads $\pi_t$ off the perturbed law, so
its derivative holds the mechanism term without being told. The cases are a censored two-node law,
a three-node law with a tilt at every node, survival, competing risks and a no-positivity law.
On two of these laws each mechanism term holds at least a tenth of the curve's variance.
`tests/unit/test_ipsi_construction.py` matches $\varphi$ to transcriptions of `npcausal` `ipsi.R`
and of Kim et al. (2021), Algorithm 1, to $10^{-12}$.

### Cross-fitting the tilt recursion

Kennedy (2019), Algorithm 1, constructs every nuisance "using only $D_0$", the training rows. A
training row's target is therefore built from the fold's own predictions. `IPSI` ships this
training-fold recursion.

| part of the fit | rows | what the fit does |
| --- | --- | --- |
| fold mechanism | the training rows of fold $k$ | fit the treatment and censoring mechanisms, and read them at every row |
| fold recursion | the training rows of fold $k$ | run the backward recursion, with the fold's own regressions and mechanism in each training target |
| held-out curve | the held-out rows of fold $k$ | compute $\varphi$ from fold $k$'s regressions and mechanism |
| estimate | every row | the weighted mean of the stitched $\varphi$ |

Algorithm 1 averages the $K$ fold means, and so does estimator (6) of Kim et al. (2021). The
package pools every row, $P_n \varphi(Z;
\hat\eta_{-S})$, which is the form of Theorem 3 and of `npcausal` `ipsi.R` line 208. The two
differ by $\sum_k (n_k/n - 1/K) P_n^k \varphi$, which is 0 at equal fold sizes. Whole-cluster
folds under `id=` can be unequal.

`n_folds=1` fits every nuisance in sample. That is Algorithm 2 of Kennedy (2019), Appendix 8.3,
which needs Donsker-type conditions. The default is `n_folds=10`.
`tests/unit/test_ipsi_construction.py` compares the cross-fitted engine with an independent
scikit-learn longhand to $10^{-12}$. Two rejected constructions move the estimate by more than
$10^{-4}$: a stitched training target, and the average of fold means on unequal folds.

### IPSI on a repeated outcome

`IPSI` accepts `outcome=Repeated(["Y1", "Y2", "Y3"])`. The fit reports one mean per tilt and horizon,
named `ey_regimen[odds x2 @ t=2]`, and the contrasts `ate_regimen[odds x2 vs natural course @ t=2]`.
Each horizon $h$ is the end-of-study fit of the panel truncated after node $h$, with $Y_h$ as the
outcome and each earlier $Y_s$ as a covariate of node $s + 1$. `IncrementalMean(horizons=[1, 3])`
and `IncrementalEffect(horizons=[1, 3])` take the same horizons in a `CausalStudy`.

The parameter is $\psi_h(\delta) = E[Y_h^{Q(\delta)}]$. The plan $Q(\delta)$ draws $A_t$ from
$q_t(1 \mid H_t)$ at every $t \le h$ and prevents dropout. Theorem 3.1 of Kim et al. (2021) at
$t = h$ identifies it under consistency, exchangeability with dropout missing at random
(Assumption A2-M), and dropout positivity (Assumption A3). A tilt needs no treatment
positivity for $\delta \in (0, \infty)$.

| source (arXiv v3) | statement | code | witness |
| --- | --- | --- | --- |
| Theorem 3.1 | $\psi_t(\delta)$ is an integral of $E[Y_t \mid H_t, A_t, R_{t+1} = 1]$ over the tilted arms | `tilt_propensities`, `ipsi._recursion`, `seed_carried(horizon=h)` | E4 compares the cell form with forward composition. E5 compares the fit with it |
| Theorem 4.1 | the uncentered influence function, summed over $s \le t$ | `ipsi._weights`, `ipsi._recursion` | E5 compares the fitted curve with the complex-step Gateaux derivative on an exact law |
| Theorem 5.1 | a Gaussian process over $(\delta, t)$ under conditions 1 to 4 | per-cell Wald interval, multiplier band over the stacked curves | E8 |

Kim's $X_t$ is $L_t$ here, and $R_{t+1}$ is $C_t$. The arXiv v3 text has four typographical
errors that this page and the code follow in their intended form: `Algorithm 1, step 4b` indexes
$\omega_s$ where it means $\omega_t$. Condition 3 of Theorem 5.1 lacks two minus signs. The
conclusion of Theorem 5.1 divides by $n$ where it means $\sqrt{n}$. The pointwise interval of
Section 5.2 prints $\hat\sigma^2$ where it means $\hat\sigma$.

**Single robustness.** `IPSI` is consistent when $\pi$ and $\omega$ are consistent, whatever $m$
is. It is not doubly robust. The tests below run on two exact laws with $T = 2$ in
`tests/discrete_law_repeated.py`. The law has a non-absorbing outcome, so $Y_2 = 0$ can follow
$Y_1 = 1$.

| witness | fit | result on the binary law at $h = 2$ |
| --- | --- | --- |
| E5 | saturated $m$, estimated $\pi$ and $\omega$ | the estimate equals the tilt functional, and the curve equals its Gateaux derivative, to $10^{-11}$ |
| E6 | $m$ on $W$ and the arm, estimated $\pi$ and $\omega$ | the estimate equals the truth. The plug-in of that $m$ misses by +0.0471 ($\delta = 0.5$) and −0.0293 ($\delta = 2$) |
| E6, known $\pi$ | the same, with `treatment_probabilities=` | the mechanism term is zero, and the estimate equals the truth |
| E7 | saturated $m$, $\pi$ and $\omega$ on $W$ alone | the estimate misses by +0.0008 ($\delta = 0.5$) and −0.0009 ($\delta = 2$). With $\pi$ on $W$ alone it misses by +0.0015 and −0.0003. With $\omega$ on $W$ alone it is exact |

The E7 bias is small, and no sampled family can power it at the studied sizes. The exact law
records it. `tests/unit/test_repeated_law_values.py` pins the same claim on the three-node law of
the study: a saturated $m$ with $\pi$ and $\omega$ on $W$ alone misses by +0.0028 at
$\delta = 0.5$ and $h = 3$.

**Dropout.** The censoring factor $1\{C_t = 1\}/\omega_t$ enters every weight $W_t$ up to $h$. With
a consistent $m$ the estimate does not read $\omega$, but the curve does. Replacing $1/\omega_t$
by one leaves the E5 estimate and moves its curve by at least 0.5 at every tilted cell, with $\delta = 1$ included. With a wrong $m$
it moves the estimate (E6): by +0.0413 ($\delta = 0.5$) and −0.0235 ($\delta = 2$) on the
binary law at $h = 2$.

**The band.** The default `simultaneous=True` band covers every declared $\delta$ and horizon, and
every contrast against the reference. It is the multiplier band over the stacked influence
curves.

| claim | condition |
| --- | --- |
| $\sqrt{n}(\hat\theta - \theta)$ is jointly normal | Kim conditions 1 to 4 at every declared $(\delta, h)$. The dimension is fixed, so the Cramér–Wold device applies. This is Theorem 5.1 restricted to a finite grid |
| the multiplier critical value is consistent | the stacked curves estimate their covariance consistently |
| the horizons are dependent | the horizons share units and mechanisms, so the band uses the joint covariance. E8 requires a nonzero correlation between the horizons of one tilt and rebuilds the critical value from the stacked curves |
| no band over a continuum of $\delta$ | the band covers the declared grid. It does not cover the points between the nodes of the grid |
| an inconsistent $m$ | no interval claim. The product $(\hat\pi - \pi)(\hat m - m)$ is first order, so the curve is not the influence function. $\psi(\delta)$ depends on $\pi$, so the projection argument of a doubly robust estimator does not apply |

The [registered study](method-evidence/repeated-outcome-ipsi.md) measures the coverage of this
curve on one three-node law. With $n$ = 8,000 in sample, the coverage of the 15 reported
parameters lies between 0.945 and 0.955, and the default band covers them jointly at 0.949. The
interval at $(\delta, h)$ is an asymptotic statement under the conditions above. The exact-law witnesses E4 to E8 and the
mutation controls in `tests/unit/test_repeated_outcome_ipsi_mutations.py` stand behind the point
estimate and the curve.

### The IPSI contract

| item | contract |
| --- | --- |
| base result, complete data | Kennedy (2019): Equation (1), Theorem 1, Theorem 2, Remark 2, Algorithm 1, Theorem 3 and Theorem 4 |
| base result, dropout | Kim, Kennedy and Naimi (2021): Theorem 3.1, Theorem 4.1, estimator (6), Theorem 5.1 and Appendix A, Algorithm 1. Appendix F, Lemmas F.2 to F.5, proves the Lemma 2 to 4 statements that Kennedy (2019) leaves unproved |
| rate condition | $\sup_\delta \bigl(\lVert \hat m_t - m_t \rVert + \lVert \hat\pi_t - \pi_t \rVert\bigr) \lVert \hat\pi_s - \pi_s \rVert = o_P(n^{-1/2})$ over the node pairs of Theorem 3, condition 4. Under dropout, Kim et al. (2021), Theorem 5.1, adds the $\lVert \hat\omega_t - \omega_t \rVert$ terms |
| other conditions | each $\delta_t$ in a compact subset of $(0, \infty)$; bounded $m_t$ and $\hat m_t$; consistency of $\varphi(\hat\eta)$ and of the variance (Theorem 3, conditions 1 to 3) |
| positivity | none at a tilt node. Positivity at each label and rule node, and dropout positivity (Kim et al. 2021, Assumption A3) |
| not doubly robust | every product in the rate condition holds a mechanism error, so the mechanism must be consistent at every tilt node (Kennedy 2019, Section 3.3) |
| in sample | `n_folds=1` rests on Donsker-type conditions (Algorithm 2), not on the sample-splitting argument of Theorem 3 |

`tests/unit/test_ipsi_exact_laws.py` witnesses the missing double robustness. With an exact
mechanism, a constant regression leaves no error. With an exact regression and a constant
mechanism, each tilt plan misses its truth by more than $10^{-3}$. A static plan on the same fit
stays exact to $10^{-12}$. The Taylor remainder along a path is second order and nonzero.

### IPSI compositions

| composition | behaviour | evidence |
| --- | --- | --- |
| censoring | the censoring factor enters every $W_t$. `g_bounds` bounds the cumulative censoring, label, rule and policy product | the censored exact laws; the Kim et al. (2021) transcription |
| survival horizons | one backward pass per horizon, with the survival target above | the survival exact law |
| competing risks | one cause at a time, with the all-cause survival factor | the competing exact law |
| observation weights | every regression and mechanism is weighted, and the estimate is the weighted mean of $\varphi$ | the weighted exact law. A second test shows that the weights move $\pi_t$ by more than 0.05 |
| clusters, `id=` | whole-cluster folds and a cluster-summed curve. The few-cluster rules of `LTMLE` apply: below 20 clusters the fit reports no interval | `tests/unit/test_ipsi_e2e.py` at 19 and 20 clusters |
| contrasts and bands | a fixed-dimension stack of the plan and horizon curves. The multiplier band covers the declared plans and horizons | `tests/unit/test_ipsi_e2e.py` |
| `msm=` | each cell's $\varphi$ is projected onto the working model. The identity link is supported. A logit or log link is accepted only when its projection converges | the exact-law projection linear in $\log\delta$; the logit refusal in `tests/unit/test_ipsi_e2e.py` |
| known treatment mechanism | `treatment_probabilities=` on `fit` declares $\pi_t$. The tilt is then a known policy, and the curve has no mechanism term (Kennedy 2019, Section 3.3) | the known-mechanism exact law |
| `CausalStudy` | `IncrementalMean` and `IncrementalEffect` on a `LongitudinalTreatment` design list `ipsi` as available. `tmle` and `sdr` list the reasons below | `tests/unit/test_ipsi_e2e.py` |

`LTMLE` also fits a tilt when every tilt node has a declared treatment mechanism. It translates
each tilt node to a known stochastic policy, which divides by the bounded mechanism. Where
`g_bounds` moves that mechanism, `LTMLE` refuses the fit before any learner. `IPSI` keeps the
cancelled tilt factor and needs no such bound.

### IPSI choices that differ from LTMLE

| choice | `IPSI` | reason |
| --- | --- | --- |
| fluctuation | none. `alpha=`, `max_iter=`, `tol=`, `targeting=` and `fluctuation=` raise `MethodConfigurationError` with code `invalid_declaration` | the estimator averages the influence function at the fitted nuisances |
| `g_bounds=` | bounds the cumulative censoring, label, rule and policy product. The tilt product multiplies in after the bound | the tilt factor lies in $[\min(\delta, 1/\delta), \max(\delta, 1/\delta)]$ and divides by no mechanism |
| regression clip | none. The default `pseudo_learner` is a regression Super Learner with `clip=None` | `npcausal` predicts the targets unbounded |
| a level outside the outcome range | reported as computed, with `ParameterSpaceWarning` | an average of $\varphi$ can leave $[0, 1]$, or the `q_bounds` range of a continuous outcome |
| `ratio()` of such a level | `CapabilityError` on the log scale at or below 0, and on the odds or logit scale outside $(0, 1)$ | the scale is undefined there. The difference scale stays available |
| full-refit bootstrap | runs, and reports `bootstrap sd` with no percentile inference | no registered study measures it |
| `truncation_curve()` | replays the bounded factors only | the tilt factor has no bound to sweep |
| score equations | `diagnostics.score_equations()` is `not_applicable` | no node fits a fluctuation |
| diagnostics | `diagnostics.support()` reports each tilt node's factor range, the mechanism term's share of the curve variance, and the share of $\hat\pi_t$ at exactly 0 or 1 | a tilt needs no positivity, so a degenerate $\hat\pi_t$ is reported and not warned |

A survival risk curve from this estimator can be non-monotone across horizons. Each horizon is its
own one-step average. The hazard model and the isotonic projection are `LTMLE` settings, and
`IPSIMethod` refuses them ([H12](scope-and-refusals.md#hazard-and-isotonic-survival-refusals)).

### IPSI refusals

| refusal | message source |
| --- | --- |
| `LTMLE` with an estimated mechanism at a tilt node, or `LTMLE(incremental=...)` | `LTMLE_TILT_REFUSAL` in `cleverly.methods` |
| `SDR` with a tilt node | `SDR_TILT_REFUSAL` in `cleverly.methods` |
| a tilt node with other than two levels | the shared point-treatment text, which names the node |
| an `IPSI` plan with a modified treatment policy or a `RiskRatioTilt` node | `tilt_mix_message` in `longitudinal/regimen.py` |
| an odds tilt at a continuous node, on `IPSI`, `LTMLE` or `SDR` | `tilt_continuous_message` in `longitudinal/regimen.py` |
| a fluctuation setting on `IPSI` or `IPSIMethod` | `IPSI_NO_FLUCTUATION` in `cleverly.methods` |
| a point-treatment container, or `IPSIMethod` on a point functional | `IPSI_POINT_REFUSAL` in `cleverly.methods` |
| a design that holds one baseline decision over every node | `IPSI` names `TMLE(incremental=[...])` on the baseline row |

[Scope and refusals](scope-and-refusals.md#not-written-yet) quotes the first, second and fourth
messages. `tests/unit/test_ipsi_e2e.py` pins each refusal before any learner runs.

### IPSI limitations

| limitation | detail |
| --- | --- |
| finite-sample bias at one fold | on the declared study laws, a one-fold fit carries a plug-in-order bias of about 0.2 standardized units at $n$ = 8,000. The design probe reads -0.18 on law `A` and -0.19 on law `C`. Source: `tests/diagnostics/x19_sizing/reading.csv`. Kennedy (2019), Algorithm 2 (Appendix 8.3), gives the one-fold interval under Donsker conditions. Saturated cell means on a finite history set form a Donsker class. Kim et al. (2021), Theorem 5.1, states the dropout result with sample splitting only, and the same Donsker argument extends it to one fold. Both studies gate the one-fold fits at sizes where this bias is small |
| finite-sample bias at five folds | the cross-fitted contrasts carry a second-order bias of order $1/n$. At $n$ = 8,000 the design probe reads 0.25 to 0.27 standardized units on law `A` at five folds, 0.35 at two folds, and 0.28 to 0.31 on law `C`. A sizing rule sets each gated cell before its run. A primary or rate cell passes with a probability of at least 0.95 at the projected bias, and a calibration cell with at least 0.90. The primary studies run at $n$ = 96,000 and 80,000 for this reason. Source: `tests/unit/test_longitudinal_incremental_design.py` |
| a band over the continuum of $\delta$ | not offered. Theorem 4 of Kennedy (2019) gives a band over a compact set of $\delta$, and the API represents a finite grid only |
| the no-effect test | not offered. Kennedy (2019), Section 4.4, gives it, and `npcausal` does not return it |
| a TMLE for the tilt over time | not offered. The package ships Kennedy's own estimator, which Theorem 3 covers with no extension |
| separate covariate sets for the mechanism and the regressions | not offered. Each nuisance reads the full history |

Two registered studies measure the estimator. The
[complete-data study](method-evidence/longitudinal-incremental-interventions.md) carries Python
property evidence. Its `npcausal` 0.1.0 comparator runs at commit `56a5ac11` are pending. The
[dropout study](method-evidence/longitudinal-incremental-interventions-dropout.md) calibrates
dropout, survival, weights and clusters against exact truths, with no comparator. This page
claims no result from either study.

The natural-course value is the sample mean of the outcome. A tilt of 1 sets every weight to 1,
so the correction terms cancel. Each contrast with the natural course is therefore the tilted
value minus that mean. Source: `test_the_fitted_natural_course_is_the_sample_mean` in
`tests/unit/test_longitudinal_incremental_design.py`.

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
| [ordinary end-of-study longitudinal TMLE](method-evidence/ordinary-end-of-study-longitudinal-tmle.md) | comparator-backed, against R `ltmle` 1.3-0, including a targeted-versus-unfluctuated pair that measures what the paired comparison cannot |
| [cross-fitted end-of-study longitudinal TMLE](method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md) | comparator-backed, against R `lmtp` 1.5.4 on one exact five-fold assignment, with the mechanism supplied to both, plus held-out recursion checks and independent theory properties |
| [ordinary weighted end-of-study longitudinal TMLE](method-evidence/ordinary-weighted-end-of-study-longitudinal-tmle.md) | comparator-backed reporting evidence against weighted R `ltmle` on exact-size selected samples, with target-weight and learner-weight controls |
| [cross-fitted weighted end-of-study longitudinal TMLE](method-evidence/cross-fitted-weighted-end-of-study-longitudinal-tmle.md) | comparator-backed reporting evidence against weighted R `lmtp` on identical five-fold assignments, with a dedicated weighted-learner adapter |
| [ordinary survival-curve longitudinal TMLE](method-evidence/ordinary-survival-curve-longitudinal-tmle.md) | comparator-backed, against R `ltmle` 1.3-0 with `survivalOutcome=TRUE`, across two horizons and with survival-recursion controls |
| [cross-fitted survival-curve longitudinal TMLE](method-evidence/cross-fitted-survival-curve-longitudinal-tmle.md) | comparator-backed, against R `lmtp` 1.5.4 across both horizons on one exact five-fold assignment, with held-out and survivor-only controls |
| [ordinary categorical longitudinal TMLE](method-evidence/ordinary-categorical-longitudinal-tmle.md) | comparator-backed, against R `lmtp` 1.5.4 with exact categorical mechanisms, for three treatment levels at two nodes |
| [cross-fitted categorical longitudinal TMLE](method-evidence/cross-fitted-categorical-longitudinal-tmle.md) | comparator-backed, against R `lmtp` 1.5.4 on the identical five-fold assignment, with a flexible-tree cross-fit control |
| [ordinary competing-risk longitudinal TMLE](method-evidence/ordinary-competing-risk-longitudinal-tmle.md) | comparator-backed, against R `lmtp` 1.5.4 with the other cause in `compete=` and exact mechanisms supplied to both |
| [cross-fitted competing-risk longitudinal TMLE](method-evidence/cross-fitted-competing-risk-longitudinal-tmle.md) | comparator-backed, against R `lmtp` 1.5.4 on the identical five-fold assignment, with a flexible-tree cross-fit pair |
| [longitudinal estimands outside the target registry](evidence.md#longitudinal-estimands-outside-the-target-registry) | the parameter and influence-curve oracle, the mutation witness, and the declared gaps, for each of the five longitudinal variants |
| [clustered cross-fitted end-of-study longitudinal TMLE](method-evidence/clustered-cross-fitted-longitudinal-tmle.md) | comparator-backed, against R `lmtp` 1.5.4 with `ife` 0.2.3 at 100 clusters of 40 rows, on the realized whole-cluster folds, with four cluster-robust pairs and a cluster multiplier band |
| [cross-fitted clustered longitudinal TMLE at few clusters](method-evidence/few-cluster-cross-fitted-longitudinal-tmle.md) | the $t_{J-2}$ reference at 20 and 30 clusters of equal and unequal sizes, with an IID control |
| [the implementation validation grid](method-evidence/validation-grid.md) | the twelve registered longitudinal TMLE rows and each row's declared limits |

Competing-risk correctness also rests on the independent finite law, the Gateaux comparison, the
all-cause-versus-cause-specific mutation, and the one-cause reduction. The two competing-risk rows
add the R `lmtp` comparison. Both rows keep the competing event natural and supply the mechanisms.
Neither row covers weights, clustering, or an eliminated competing event.
