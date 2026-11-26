# Continuous-time longitudinal TMLE

`ContinuousLongitudinalTMLE` in `cleverly.survival` estimates the absolute risk of each cause of a
competing-risks outcome at declared times under a treatment regime, when the treatment and the
covariates change at visits that fall at irregular, random times. It is the TMLE of Rytgaard,
Gerds and van der Laan (2022) in continuous time, with no time grid. The registered study
[continuous-time longitudinal TMLE](method-evidence/continuous-time-longitudinal.md) measures its
default cross-fitted fit, and a risk or risk difference reads `supported` inside the configuration
that study measures (see [Labels](#labels)). With a treatment fixed at baseline, use
[continuous-time survival](continuous-time-survival.md); on a time grid, use the
[longitudinal TMLE](longitudinal-tmle.md#survival-and-competing-risks).

## Data

The fit reads one row per interval between visits (`ContinuousLongitudinalData.from_frame`). A
unit's first row starts at time 0, each later row starts at a visit time $s_k$, and the last row
ends at the unit's follow-up time with its event code, $0$ for censored. Each row carries the
baseline covariates $W$, the time-varying covariates $L(s_k)$ measured at the visit and the
treatment $A(s_k)$ decided there. A boolean `decision=` column marks the visits where the
treatment may change; at any other visit the treatment carries forward, and no treatment model
or rule is read there.

Within one time the order is death, then the visit ($L$, then $A$), then censoring. A visit
recorded at a unit's death time is read as following the death: the row is dropped with a
`UserWarning`, and `ContinuousLongitudinalData.dropped_visits` counts the dropped rows. Visits
at or after the last target time are not read.

## Estimand and identification

For regime $r$, cause $j$ and time $\tau$ the target is

$$
\psi_{r,j,\tau} = P^{G_r}(T \le \tau,\ \Delta = j),
$$

the risk had every unit followed $r$ at baseline and at every decision visit, with censoring
removed and the visit process left as it runs under the regime (Rytgaard et al. 2022, display
(8), with $Y = N^{d_j}(\tau)$). A regime is a treatment level held at every decision visit, a
`Static` intervention, or a deterministic rule: a `Rule` declared `rule_kind="known"`, whose
function takes a frame of the decision visits (`time`, `visit`, the covariates by name and
`previous_treatment`) and returns a level per row. A bare callable carries no declaration and is
refused, as everywhere in the package. The fit reports every $\psi_{r,j,\tau}$, each regime's risk difference from the
reference regime and their ratio. With one cause the names read `risk_regimen[r @ t=...]`, and
with several `cif_regimen[r, cause @ t=...]`.

The risk is identified under consistency, no unmeasured confounding of each treatment decision
given the observed history, censoring independent of the event process given the history, and
positivity $P^{G_r} \ll P^{G_0}$ (Rytgaard et al. 2022, Assumptions 1 and 2). No data check
verifies the first three.

**History window.** Every intensity and the treatment model read the history only through the
window of the last visit,

$$
x_k = \bigl(A(s_k),\ W,\ L(s_k),\ A(s_{k-1}),\ k,\ s_k\bigr), \qquad A(s_{-1}) = 0,
$$

so the fit identifies $\psi$ when the law is window-Markov in this sense. This is a model
restriction, stated in `ContinuousLongitudinalRecord.history_restriction`.

**Visit cap.** The fit models visits up to a cap $K^*_r$ per regime: the largest $K$ at which
every outer training fold holds at least five units (`VISIT_CAP_MIN_FOLLOWERS`) that followed
$r$ through visit $K - 1$ and visit a $K$-th time before the last target time. Rows after visit
$K^*_r$ merge into segment $K^*_r$, whose only exits are deaths. The fit then targets
$\psi^{(K^*)}_r$, the risk under $r$ at baseline and the first $K^*_r$ visits with the treatment
natural after them, and

$$
\lvert \psi^{(K^*)}_{r,j,\tau} - \psi_{r,j,\tau} \rvert \le P^{G_r}\{N^v(\tau) > K^*_r\}.
$$

`ContinuousLongitudinalRecord.past_cap_share` estimates the right side per regime and target
time, inverse weighted to the regime. The fit warns when it exceeds 0.05. Past the cap, censoring and the
hazards read the window of visit $K^*_r$, not that of the unit's later visits. The death terms of
node $K^*_r$ stay the efficient influence curve of $\psi^{(K^*)}_r$, but the fit identifies
$\psi^{(K^*)}_r$ only when censoring after visit $K^*_r + 1$ does not depend on the later $L$ or
$A$ given $x_{K^*_r}$; otherwise the inverse censoring weight on those paths is wrong, and the
bound above does not cover that error. $K^*_r$ is chosen once on the full sample, so it reads
every fold's held-out units: a discrete tuning choice, not fitted out of fold.

## The efficient influence curve

A node is a visit. Node $k$'s rows are the units that reached visit $k$ having followed $r$
through visit $k - 1$. Between $s_k$ and the next exit the history is fixed, and the exits of
segment $k$ form one multinomial cell per grid time:

$$
dE_{d_l}(u) = d\Lambda^{d_l}(u), \qquad
dE_v(u) = \Bigl(1 - \sum_l d\Lambda^{d_l}(u)\Bigr)\, d\Lambda^v(u),
$$

evaluated with $A(s_k)$ at the regime's value (the slot). The segment survival is
$S_k(u) = \prod_{w \le u}(1 - \sum_e dE_e(w))$. For target $p = (j, \tau)$ the value of each exit
is $v^p_{d_l}(u) = 1\{l = j\} 1\{u \le \tau\}$ and
$v^p_v(u) = 1\{u \le \tau\} F^\nu_j(\tau \mid u)$, where $F^\nu$ is the continuation: the law of
the event time after a visit at $u$, held in bins cut at the target times. The value of the
segment after $u$ is

$$
Z^p_k(u) = \frac{1}{S_k(u)} \sum_{w > u} S_k(w^-) \sum_e v^p_e(w)\, dE_e(w),
$$

with $Z^p_k(u) = v^p_v(u)$ where $S_k(u) = 0$ (the guard: a row that exits at $u$ scores the
same for any value there, and the guard makes the visit term vanish term by term). Node $k$'s
part of the efficient influence curve is

$$
\sum_u h_k(u) \sum_e \{v^p_e(u) - Z^p_k(u)\}\{dN_e(u) - Y_k(u)\, dE_e(u)\}
+ 1\{\text{visit } k + 1 \text{ at } u\}\, h_k(u)\, \{Z^p_{k+1}(s_{k+1}) - F^\nu_j(\tau \mid u)\},
$$

with $h_k = 1\{A(s_k) = d_r\}\, \tilde h_k$ and the slot weight

$$
\tilde h_k(u) = 1 \Big/ \max\Bigl\{\prod_{m < k} \pi_m\, S_{c,m} \cdot \pi(d_r \mid x_k)\, S_c(u^- \mid x_k),\ \delta_n\Bigr\},
$$

where the product runs over the observed history before visit $k$, $\pi$ is the treatment
probability at a decision visit (one elsewhere), $S_c$ the censoring survival and $\delta_n$ the
floor (`g_bounds`, by default $5 / (\sqrt n \log n)$). The baseline node adds
$Z^p_0(0) - \psi_p$. Saturated fits on an exact law reproduce the Gateaux derivative of the
full-history g-formula to $10^{-10}$, the asserted bound
(`tests/unit/test_continuous_longitudinal_exact.py::test_saturated_fit_equals_g_formula_and_gateaux`).

## Initial fits

| nuisance | rows | default |
| --- | --- | --- |
| each cause's hazard | the segment rows, entering at $s_k$ (`entry=`), event: that cause at the stop | the three Cox models of `default_library`, selected by cross-validated partial likelihood |
| the visit intensity | the segment rows before the cap; a death row stops at its time less $\delta$ | the same three |
| the censoring hazard | the segment rows, entering at $s_k - \delta$; a row that exits by death or visit stops at its time less $\delta$ | the same three |
| the treatment at baseline and each decision visit | the decision visits, design $x_k$ without $A(s_k)$ | a main-terms and an L1 logistic model, selected by cross-validated log loss, each on $[x_k, b, b\,x_k]$ with $b = 1\{k = 0\}$ |
| the continuation $F^\nu_k$ | node $k + 1$'s rows; label: node $k + 1$'s binned sub-distribution after its visit, targeted in sample; under cross-fitting, each fold's untargeted training pass fits it and the folds are stitched, as the discrete-time hazard LTMLE does | a main-terms and an L1 logistic model of each bin's event fraction, selected by cross-validated soft log loss; a regressor passed as `continuation_learner` replaces the pair |

$\delta$ is half the smallest gap between distinct observed times, so the tie order above holds in
every fit. `CoxHazard`, `StratifiedNelsonAalen`, `partial_log_likelihood` and `select_hazard`
take `entry=` for delayed entry. With `cross_fit=True` (the default) every nuisance is fitted out
of fold, with folds drawn per unit, so a unit's segment and visit rows share one outer fold and
one label in each library's inner cross-validation.

The baseline indicator $b$ gives the baseline treatment model coefficients of its own: without it,
one logistic model pooled over baseline and later visits is misspecified whenever the two mechanisms
differ, and on the study's law CT that made the influence curve 7 to 8% wider than the Monte Carlo
spread at every $n$ (`tests/canonical/continuous_time_longitudinal/pilot-diagnosis.md`). At baseline
$A(s_{k-1})$, $k$ and $s_k$ are zero, so the unpenalized member equals two separate fits; the L1
members penalize the baseline's departure from the visit coefficients. With no visit among the
training rows the design is $x_k$ alone, so the fit reduces to the baseline engine's. A supplied
`treatment_learner` reads $x_k$ as it is.

A visit jump of one or more is kept at one only at a time where every row at risk whose own
fitted jump reaches one exits there (a scheduled visit). Elsewhere it is bounded at
$1 - 10^{-8}$, as a combined death jump of one or more is, with a `PositivityWarning`; the record
keeps both shares and the estimates name the bound in their label.

## Targeting

Each regime runs one backward pass. Node $K^*_r$ is walked first. Its targeted curve gives the
value $Z_{k+1}$ of node $K^*_r - 1$'s covariate term and so on to the baseline node, each node
walked once after every later node is final. In sample, the targeted curve is also the label of
node $k$'s continuation, which is refitted on it before node $k$ walks. Under cross-fitting
(the default) the continuation stays the stitched out-of-fold fit of the untargeted training
passes, and node $k$'s walk tilts it but does not refit it. Either way the walk solves the
covariate term's score.

Node $k$'s walk tilts its exits along $\sum_p v_p \tilde h_k c^p_e$ and its continuation's bin hazards along
$\sum_p v_p \tilde h_k \kappa^p$, with the slot weight $\tilde h_k$ on every row, deviating or not.
A step is accepted when it shortens the node's score in the metric held at the initial fit; the
step halves on a rejection and doubles back after an acceptance, as in the baseline walk.

Node $k$'s scores read only its own intensities and continuation, the mechanism and the later
nodes' final values, so after the baseline node every term of $P_n D^*$ is solved. Each node walk
stops at $s_n / N$, with $N$ the regime's nodes, so every target's total standardized score
$\lvert P_n D^* \rvert / \hat\sigma$ is at most $s_n$ (`tol`, by default
$1 / (\sqrt n \log n)$). The record keeps each node's steps and final scores.

## Inference

Each estimate reads the influence-curve variance $P_n \hat D^2 / n$ (Rytgaard et al. 2022,
display (27)) and a Wald interval, on the logit scale for a risk. The switched variance of the
baseline engine does not carry over: its predictable variation of the covariate term needs the
law of $L(s)$, which this method does not model. A ratio reads the delta method on the log
scale. `simultaneous=True` adds a sup-t band over every risk and risk difference.

## Labels

`continuous_longitudinal_support` labels each estimate. A risk or a risk difference reads
`supported`, with the evidence `continuous-time-longitudinal`, when every condition below holds.
The constants are read from the study's committed `floors.csv`, and
`tests/unit/test_continuous_longitudinal_design.py` checks each against it.

| condition | supported when | constant |
| --- | --- | --- |
| nuisances | every library and learner is the default, cross-fitted, with the default settings (`norm`, `g_bounds`, `max_iter`, `step_size`, `tol`, `n_folds`, `learner_folds`, `alpha`) | `LONGITUDINAL_DEFAULTS` |
| causes | two | `LONGITUDINAL_CAUSES` |
| sample size | at least 1,000 units | `LONGITUDINAL_N` |
| regimes | static levels or declared rules; every rule row passed | |
| event share | between 9.2% and 24.4% of the units have an event of the estimate's cause by its time, the range of the gated rows' median shares | `LONGITUDINAL_EVENT_SHARE` |
| sparsity | the relative SD of the influence-curve variance, `variance_rel_sd`, is at most 0.224, the largest median at which the gated rows pass | `LONGITUDINAL_REL_SD` |
| followers | at least 49 units follow each regime of the estimate at every decision visit and are at risk at the last target time, the fewest any gated fit saw | `LONGITUDINAL_FOLLOWERS` |
| visit cap | each regime's estimated share of followers past $K^*_r$ before the estimate's time is at most 0.178, the largest any gated fit saw | `LONGITUDINAL_PAST_CAP_SHARE` |
| weight floor | the floor raised at most 0.18% of the weighted cells at risk, the largest share any gated fit saw | `LONGITUDINAL_BOUNDED_SHARE` |
| jump bounds | the engine bounded no hazard jump and no visit jump | |

Otherwise the estimate reads `experimental`, and the reason names every condition that fails. A
fit with a library or learner other than the default for any nuisance never reads supported, a
one-member Cox library included, because a study measures the default libraries only. The ratio,
smooth contrasts and the band read experimental with reasons of their own: no study measures
them. The study fits one law, so its limits travel with every label: about two visits per unit
over 30 days, two causes and censoring that depend on the covariate and the treatment, always and
never treat and one rule, and three target times fitted jointly.

## Summary, plots and diagnostics

`summary()` prints the question, the target times, the regimes, the method, each nuisance
model's selection, each regime's visit cap $K^*_r$, the accepted update steps and the
estimates; `summary(detail="full")` adds the assumptions and how the node walks ended. `plot()`
draws each regime's risk curves with a panel of the risk differences against the reference.
The diagnostics read `record` and refit nothing:

| call | returns | reads |
| --- | --- | --- |
| `diagnostics.support()` | `ContinuousLongitudinalSupport` | the weight floor and the share of followers' cells it raised (`bounded_share`), with the severity tiers of the point support report; the shares of node rows whose death or visit jump was bounded (`jump_bound_share`, `visit_bound_share`); each regime's $K^*_r$ (`caps`), its share past the cap at each target time (`past_cap_share`) and its followers at the last target time (`followers_at_last`) |
| `diagnostics.score_equations()` | `ContinuousLongitudinalScores` | each target's final $\lvert P_n D^* \rvert / \hat\sigma$ against $s_n$, each node's score (`node_scores`) and its accepted steps |
| `diagnostics.nuisance_models()` | `ContinuousLongitudinalNuisances` | each hazard's, the visit intensity's, the censoring hazard's and the treatment model's candidates, losses and choice; a cross-fitted fit records the first fold's |
| `diagnostics.influence()` | the influence report | the influence curves of the estimates |

`validate()` runs the score, support and nuisance checks, and fails when a node walk stopped short of its cutoff. `validate()` and `assess()` run as on a continuous-time survival fit, because every kind's battery runs in the capability sweep; the items that would replay the fit (the refutation and the truncation curve) and the sensitivity rows name their refusals. The three report classes live in
`cleverly.survival.longitudinal_diagnostics`, and `tests/unit/test_continuous_longitudinal.py`
checks each against the record.

## What is refused

| request | refusal |
| --- | --- |
| a known stochastic policy, a modified treatment policy, an odds tilt, a continuous treatment, or an intervention on the visit intensity | `CapabilityError`, implementation gap: the continuous-time submodel of these policies is not written. The discrete-time hazard LTMLE fits known stochastic policies |
| a treatment with more than two levels | `CapabilityError`, implementation gap: the treatment model at each visit is a binary logistic library |
| `rmtl()` or `rmst()` | `CapabilityError`, API redirection: the fit targets the risks at its target times only |
| a bare callable as a regime, or a `Rule` whose `rule_kind` is not `"known"` | `CapabilityError`, invalid declaration (an implementation gap for `"estimated"`), with the package's text: a callable can close over an estimate, so the declaration is the check |
| a rule that returns a level outside the treatment's levels, or a missing value | `DataError` naming the rule, the value and the count |
| a target time past the follow-up of the units following some regime at baseline | `DataError`, as the baseline engine |
| no event of a cause by the first target time among the units following some regime at baseline | `DataError`, as the baseline engine |
| a user learner for any nuisance with `cross_fit=False` | `MethodConfigurationError`: Lemma S8 of Rytgaard et al. (2022) needs nuisances of bounded variation, which a user learner fitted in sample does not guarantee |
| a cap $K^*_r < 1$ while some unit visits before the last target time | `DataError` naming the regime, the fold and the count |
| `refute()` and the truncation curve (`continuous_truncation_curve()`) | `CapabilityError`, implementation gap, roadmap [X105](../roadmap.md#x105-negative-controls-generated-outcomes-and-resampling-for-continuous-time-fits) part (d): no adapter replays a node pass. `validate()` and `assess()` run, and their refutation and truncation-curve items name this refusal |
| a sensitivity analysis, through `sensitivity` or `evalue()` | `CapabilityError`, inference gap: an E-value reads one point exposure's risk ratio, and a sustained regime is not one. The sensitivity rows of `assess()` name this refusal |
| `score_equations(tolerance=...)` | `CapabilityError`, invalid declaration: the node walks stop at $s_n / N$, and the report gates on $s_n$ |
| the overlap, weight, balance, calibration and repeat plots | `CapabilityError` or `ValueError` from the plot: the record keeps no per-row treatment probability, weight or repeat |

The fit warns (`UserWarning`) when fewer than 30 units follow a regime at every decision visit
and are at risk at the last target time, and when the estimated share past the cap exceeds
0.05. A predicted peak above `max_memory_bytes` refuses before any learner.

## Evidence

The registered study [continuous-time longitudinal TMLE](method-evidence/continuous-time-longitudinal.md)
fits the default cross-fitted fit at n = 1,000 on one law, in continuous time and on whole days,
with always treat, never treat and the rule "treat when $L(s) = 1$". Every one of its 48 gated
primary rows passes, as do the 12 double-robustness rows with the outcome side wrong, the
plug-in control and both calibration rows. One fit of 3,200 was lost to a defect since fixed, and
the [red-cell ledger](method-evidence/red-cells.md) accepts the two accounting rows it turns red
by a named exception. The fast tier checks the engine on two exact laws passed by weight,
`tests/discrete_law_continuous_longitudinal.py`: law PL (one random visit time) and law PL2 (two
visit times, certain and uncertain visits, a second visit node).

| check | test |
| --- | --- |
| saturated fits equal the g-formula and the Gateaux derivative, for static regimes and a rule | `test_continuous_longitudinal_exact.py` |
| wrong intensities and continuation with the right mechanism: the walk reaches the truth | `test_walk_repairs_wrong_intensities` |
| each score term, the guard, the order rule, the treatment factor at every visit, the censoring weight, the slot, the continuation at counterfactual visit times, the soft-outcome chain, the earlier treatment, the floor, the rule at decision visits, the non-followers and the slot-weight tilt each have a mutation that breaks the exact check, and keeps a control on which its term vanishes | `test_continuous_longitudinal_exact.py` |
| the fit at $K^*_r = 1$ on PL2 equals $\psi^{(K^*)}$ and its Gateaux derivative | `test_target_past_the_cap` |
| with no visit, the initial fits and influence curves equal the baseline engine's, and under saturated learners the estimates, standard errors and influence curves agree to $10^{-10}$ | `test_continuous_longitudinal.py::test_reduces_to_the_baseline_engine`, `test_saturated_fits_reduce_to_the_baseline_engine` |
| against the discrete-time hazard LTMLE on its law (M), shift-mapped into continuous time, saturated and in sample: on the exact law, where both engines sit at the truth and take no step, estimates agree to $10^{-9}$ and influence curves agree up to that engine's weight scaling (the efficient influence curve term by term); on a 20,000-unit draw with no step, at fitted nuisances, the untargeted estimates and influence curves agree to $10^{-12}$. Targeted estimates on a draw differ at second order, because the window does not hold the history the slot weight's prefix reads and each engine walks its own submodel | `test_x70_agreement_on_law_m`, `test_x70_agreement_on_a_draw_with_no_step` |
| on law PL, whose baseline and visit treatment mechanisms differ, the default-shaped stratified logistic fit gives the exact efficient influence curve to $10^{-8}$; the pooled fit misses it by more than $10^{-3}$ | `test_default_treatment_model_holds_a_baseline_mechanism_of_its_own` |
| a binding weight floor moves only the rare cell's influence curves, by the closed form $A + B / \delta$ of the floored weight | `test_weight_floor` |
| per-node cutoffs $s_n / N$ bound every target's total score by $s_n$, on PL2 and on a stub node problem with fixed linear scores | `test_node_cutoffs_bound_the_total`, `test_node_cutoffs_bound_the_total_on_a_stub` |
| the share past the cap equals the g-formula of a second visit before the target time | `test_past_cap_share_is_the_g_formula` |
| delayed entry in the hazard learners against a hand computation | `test_cox_entry.py` |

A `nol` walk reaches the truth to about $10^{-9}$, not to the requested $10^{-10}$: the walk's
smallest step, $10^{-8}$, bounds how far a step can shorten the score.
