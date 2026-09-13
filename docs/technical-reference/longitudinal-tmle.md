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
| the plan depends on the history | dynamic rules receive the history available at their node, and no static plan can express them | the rule is part of the estimand. Two rules are two parameters |
| you want a survival curve | the same recursion, seeded at the horizon, reports cumulative risk at each horizon you name | each horizon is its own backward pass. The cost is quadratic in the node count |
| several causes of failure compete | cause-specific cumulative incidence, with the competing causes left alone | that is a *total* effect. Eliminating the competing event is a different question, and is refused by name |
| you want the effects summarised across regimens | a working model over regimen and horizon cells | see [MSM projections](msm-projections.md) |

Reach for [point-treatment TMLE](point-treatment-tmle.md) when the exposure is measured once.
Collaborative TMLE and DR-TMLE have no longitudinal derivation, and `available_methods()` says so
before any model is fitted.

A worked applied analysis is in the
[longitudinal tutorial](../examples/longitudinal-tmle.md). It runs a point-treatment analysis
of the same data as a control, and that analysis fails in both available directions.
[Time-to-event outcomes](../examples/longitudinal-survival.md) is the companion tutorial for
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
covers dynamic rules. See the
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
structure fixed. A cross-fitted replay uses every complete fold-specific mechanism slab. It does
not read the stitched out-of-fold pair alone. The result must retain a replay recipe, and that
recipe's outcome and pseudo-outcome learners must be cloneable.
[Replay-only unavailability](scope-and-refusals.md#replay-only-unavailability) states the
`random_state` rule each learner must satisfy, and lists every code that makes the operation
unavailable. [Truncation stability](validation-methods.md#truncation-stability) holds the frame
schema and the score-cell counting rules.

Before evaluating the requested grid, replay at the fitted pair must reproduce every retained
estimate, regimen fit, and MSM fit exactly. This preflight also runs when the grid omits that pair.
A mismatch makes the operation unavailable instead of reporting a different fitted procedure.

The direct `stagewise()` method remains a compatibility alias.

### Survival and competing risks

[Time-to-event outcomes](../examples/longitudinal-survival.md) works this section applied.

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

## Variations

| option | what it does |
| --- | --- |
| `regimens=` | static plans, dynamic rules, or categorical arms. A plan is a sequence of arms, or one arm meaning that arm at every node |
| `reference=` | which regimen the contrasts are taken against. It is part of the estimand rather than a display setting |
| `horizons=` | which time points a survival fit reports cumulative risk at. `None` reports the whole curve. Name the horizons you will report: the cost is $T(T+1)/2$ regressions per regimen rather than $T$ |
| `msm=` | a working model over the regimen and horizon cells. It requires `n_folds=1`. See [MSM projections](msm-projections.md) |
| four learner slots | `outcome_learner`, `pseudo_learner`, `treatment_learner`, `censoring_learner`. The pseudo learner fits the intermediate regressions, whose outcome is a bounded prediction rather than the outcome itself |
| `n_folds=`, `learner_folds=` | one outer split serves every node and regimen. Each fold fits a complete mechanism, backward recursion, and targeting sequence on its training rows. The result stitches predictions only on held-out rows. The fit keeps one mechanism slab per fold, so the mechanism costs $K$ times the memory of a single-fold fit and the saved result grows by the same factor |
| `g_bounds=`, `q_bounds=`, `alpha=` | cumulative truncation, outcome scaling, and the logistic shrink |
| `alpha_sig=`, `simultaneous=`, `n_multiplier=`, `multiplier_kind=` | interval level, and the simultaneous bands across the reported regimens |

### What cross-fitting splits

The whole backward recursion is the unit of splitting for regimen means. Fold $k$ fits every
mechanism, regression, and fluctuation on its training complement. Only its held-out rows reach the
report. Each fold must also carry enough events for each declared cause, which
[scope and refusals](scope-and-refusals.md#how-to-read-a-refusal) states with its remedy.

Longitudinal `msm=` requires `n_folds=1`. A saturated identity shows that two constructions reduce
to the same regimen means. It does not validate an unsaturated coefficient projection under
cross-fitting. That composition needs a separate property and repeated-sampling study.

**A cross-fitted fit does not solve the pooled score equation, and is not meant to.** Fold $k$ fits
its fluctuation coefficient on the rows it does not report, so the score of the stitched fit is a
mean-zero residual rather than a solved equation. Two consequences follow.

- `res.diagnostics.score_equations()` reports two rows per node. The `solver` row asks whether each
  fold reached the root of its own equation, and that answer is at solver tolerance. The
  `stitching` row asks whether the pooled residual sits where sampling would leave it, and reports
  a $z$ statistic against the residual's own standard error.
- The reported standard error runs above the actual sampling spread. Measured over 300 replications
  of `make_longitudinal` at $n = 500$: the ratio of reported standard error to the spread of the
  estimates was 1.01 at one fold and 1.09 at ten, for `ate_regimen[always vs never]`. The intervals
  are conservative rather than invalid. Coverage was 0.960 at one fold and 0.967 at ten.

Seventeen point-treatment keywords are refused **by name** on a longitudinal design, each with its
own reason. The list is in `_REFUSED` in
[`longitudinal/estimator.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/estimator.py).
The refusals that are statements about the *question* rather than about coverage are these.

| refused | kind | what it would need |
| --- | --- | --- |
| eliminating the competing events | a different question | what is reported is the cause-specific cumulative incidence with the competing causes *left alone*, so a competing event is part of the history. Removing it makes it an intervened node: a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions |
| `intermediate=` | a different question | a controlled direct effect fixes a mediator at one time point. Over a sequence, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| a **stochastic** categorical policy at a node | a different question | a deterministic rule assigns one label per unit, and the clever covariate selects that label's probability. A policy that assigns a *distribution* replaces the intervention density itself, so the cumulative product carries a ratio rather than a selected column |
| a **continuous dose** at a node | a different question | there is no label to assign, so the intervention is a shift along a conditional density at every node. A numeric node with coarse support is accepted, and warns that its values became unordered arms |
| `mechanism=True` on `truncation_curve()` | a different question | a longitudinal fit holds one cumulative treatment-and-censoring bound, and it fits no separate observation mechanism to sweep. The option names a point-treatment axis, so the call raises `CapabilityError` rather than sweeping the cumulative bound under another name |
| an outcome missing for a reason other than censoring | wrong by construction | left as it is, the probability of observing it is silently taken to be one. Encode it as a final censoring column, so it is estimated and enters the cumulative product |
| the targeted bootstrap and longitudinal sensitivity-bound estimation | not written yet | the bootstrap needs a resampling and replay contract. Sensitivity-bound estimation needs a sample estimator and sampling theory for its bound functionals |

See [scope and refusals](scope-and-refusals.md#how-to-read-a-refusal) for what each `kind` means.

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
| [the implementation validation grid](method-evidence/validation-grid.md) | the ten registered longitudinal TMLE rows and each row's declared limits |

Competing-risk correctness also rests on the independent finite law, the Gateaux comparison, the
all-cause-versus-cause-specific mutation, and the one-cause reduction. The two competing-risk rows
add the R `lmtp` comparison. Both rows keep the competing event natural and supply the mechanisms.
Neither row covers weights, clustering, or an eliminated competing event.
