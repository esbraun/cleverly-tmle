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
structure fixed. A clustered fit runs at one fold alone, so a clustered replay is an in-sample
replay.

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
$\Delta \times \mathrm{RMST}$. On a competing-risk fit, $F_d$ is the sum of the cause-specific
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
| status | a replicate carries no status. The fit's status names the bootstrap columns. Under `few_cluster_plugin` the summary prints `bootstrap sd` and `percentile range` |
| replay | `truncation_curve()` does not rerun the bootstrap. Its check at the fitted bound compares the estimates without their bootstrap summaries |
| random draws | `run_bootstrap` spawns the stream from `random_state`. The stream draws nothing the fit uses, so every analytic field of the fit is unchanged. `tests/unit/test_ltmle_bootstrap.py` pins a committed `canonical-ltmle` row at `n_bootstrap=0` and at `n_bootstrap=2` |
| parallel layers | the replicates run in parallel over `n_jobs`, and each replicate fit runs with one worker |
| licensing scope | `bootstrap_design_kind` names a fit's kind: an end-of-study or single-cause survival outcome, fitted in sample, cross-fitted or clustered, with static regimens, a binary outcome, no weights and no working model. Any other fit has no kind. A kind is licensed as inference only when it is in `LICENSED_BOOTSTRAP_DESIGNS`, and it enters after its cells in the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) are green. The registered run licensed `end_of_study/in_sample` and `survival/in_sample`. The cross-fitted and clustered end-of-study kinds stay diagnostic, with owner `X20-bootstrap`. An unlicensed kind prints `bootstrap sd` and a percentile range. The study measures correctly specified cell-mean nuisances on finite binary laws, at $n = 1000$, and at 1,500 rows in 60 clusters for the cluster bootstrap. No result covers a data-adaptive nuisance. Cai and van der Laan (2020) is the warning for that case |
| derived estimates | `ratio`, `rmst`, `rmtl` and `contrast` apply the same function to each replicate's estimates, and attach the percentile interval of those values. The derived interval is licensed only when every input's interval is |

## Variations

| option | what it does |
| --- | --- |
| `regimens=` | static plans, fixed rowwise dynamic rules, or categorical arms. A plan is a sequence of arms, or one arm meaning that arm at every node. A plan with a callable node must be a `DynamicRegimen` declared `rule_kind="known"`. `LTMLE.fit` refuses an undeclared or `"estimated"` rule, and a callable written inline in a mapping, before any learner. Sample-adaptive thresholds and learned rules need additional inference and are outside this contract. The [scope page](scope-and-refusals.md#wrong-by-construction) gives the threshold witness |
| `reference=` | which regimen the contrasts are taken against. It is part of the estimand rather than a display setting |
| `horizons=` | which time points a survival fit reports cumulative risk at. `None` reports the whole curve. Name the horizons you will report: the cost is $T(T+1)/2$ regressions per regimen rather than $T$ |
| `msm=` | a working model over the regimen and horizon cells. It requires `n_folds=1`. See [MSM projections](msm-projections.md) |
| four learner slots | `outcome_learner`, `pseudo_learner`, `treatment_learner`, `censoring_learner`. The pseudo learner fits the intermediate regressions, whose outcome is a bounded prediction rather than the outcome itself |
| `n_folds=`, `learner_folds=` | one outer split serves every node and regimen. The split is unstratified: `random_partition` draws it from the row count and the seed, and it balances no treatment node. Each fold fits the mechanism and an untargeted backward regression sequence on its training rows. One pooled fluctuation per node then targets the out-of-fold predictions, as [cross-fitting the recursion](#cross-fitting-the-recursion) states. The mechanism fit keeps one prediction slab per fold, so the mechanism costs $K$ times the memory of a single-fold fit and the saved result grows by the same factor |
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
others drop the loss weight, read one fold's mechanism slab, or carry targeted values back into
the folds.

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

Two designs are refused above one fold. A fit with `id=` is refused, because the cluster-robust
variance of this targeted recursion under a grouped draw is not established. A continuous outcome
with `q_bounds=None` is refused, because the sample outcome range would take the scale from
held-out rows. Each message names the in-sample fit, which is `CrossFitting(enabled=False)`, or
`n_folds=1` on the engine.

Longitudinal `msm=` requires `n_folds=1`. A saturated identity shows that two constructions reduce
to the same regimen means. It does not validate an unsaturated coefficient projection under
cross-fitting. That composition needs a separate property and repeated-sampling study.

The engine keeps a fold-fluctuated working-model path for that identity. Its folds each solve their
own fluctuation, so its score report adds a `stitching` row per node. The public estimator refuses
the path above one fold.

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
| a **stochastic** categorical policy at a node | not written yet | a deterministic rule assigns one label per unit, and the clever covariate selects that label's probability. A policy that assigns a *distribution* replaces the intervention density itself, so the cumulative product carries a ratio rather than a selected column. Díaz, Williams, Hoffman and Schenck (2023), Section 2 and Theorem 3, cover a policy density that is known before the fit. [F1](../roadmap.md#f1-stochastic-categorical-policies-at-a-longitudinal-node) holds the work |
| longitudinal sensitivity-bound estimation | not written yet | a sample estimator and sampling theory for its bound functionals. [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) holds the stop |
| a **continuous dose** at a node, and `shifts=` | not written yet | Díaz, Williams, Hoffman and Schenck (2023), Theorem 3, journal page 853, covers a fixed modified treatment policy $d(a_t, h_t)$ on a continuous dose. The fit needs a conditional density of the dose at every node, and each node's density ratio enters the cumulative product. `LTMLE` estimates no such density, so it refuses `shifts=` by name. It reads a numeric node as unordered arms. It warns at 10 or more distinct values, and it raises `DataError` above 20. [X12](../roadmap.md#x12-modified-treatment-policies-beyond-the-additive-point-shift) part (b) holds the work |
| `incremental=` | not written yet | Kennedy (2019), *Journal of the American Statistical Association* 114(526), treats incremental interventions on a time-varying treatment. The tilt is built from the mechanism, so it needs the product of tilted mechanisms and a mechanism submodel at every node. [X19](../roadmap.md#x19-incremental-interventions-over-time) holds the work |
| `id=` above one fold | not written yet | a grouped draw keeps each cluster whole, and the cluster-robust variance of the targeted sequential recursion under one is not established. The package permits the in-sample clustered fit. Below 40 clusters with positive weight mass it takes `"few_cluster_plugin"` and reports no interval ([clusters](inference.md#clusters)) |
| a continuous outcome with `q_bounds=None` above one fold | not written yet | with `q_bounds=None` the scale comes from every observed outcome, held-out rows included, and no shipped result covers that scale |

Two plan shapes raise `DataError` before any learner. Each one is a statement about the input, so
the table of kinds above does not list it.

| plan | what the refusal asks for |
| --- | --- |
| an iterator in `regimens=`, such as a generator | a tuple, or a `DynamicRegimen`, which reads its plan once and stores it as a tuple. The fit does not read the iterator: a fit reads `regimens=` at each call, and an iterator is empty after its first read |
| a `DynamicRegimen` plan that is one label or one callable | one entry per node. Write one rule for every node as `(rule,) * T` |

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
