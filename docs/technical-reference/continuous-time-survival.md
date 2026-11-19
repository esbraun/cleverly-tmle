# Continuous-time survival and competing risks

`ContinuousSurvivalTMLE` in `cleverly.survival` estimates the absolute risk of each cause of a
competing-risks outcome under each arm of a binary baseline treatment, at declared times, with
no time grid. It is the one-step TMLE of Rytgaard and van der Laan (2024). The study route reaches
it through `TimeToEvent(grid="continuous")` with `TMLEMethod`; the discrete-time engines are in
[longitudinal TMLE](longitudinal-tmle.md#survival-and-competing-risks).

## Estimand and identification

A unit has baseline covariates $L$, a treatment $A \in \{0, 1\}$, a follow-up time $\tilde T$ and
an event code $\Delta \in \{0, 1, \dots, J\}$, with $0$ for censored. For arm $a$, cause $j$ and
time $t$ the target is

$$
\psi_{a,j,t} = E_L\, F_j(t \mid a, L), \qquad
F_j(t \mid a, L) = \int_0^t S(u^- \mid a, L)\, \Lambda_j(du \mid a, L),
$$

where $\Lambda_j$ is the cause-specific cumulative hazard and $S$ the product limit of the summed
hazards. The fit reports every $\psi_{a,j,t}$, the risk differences $\psi_{1,j,t} - \psi_{0,j,t}$
and the risk ratios. With one cause the names read `risk_regimen[arm @ t=...]`, and with several
`cif_regimen[arm, cause @ t=...]`.

$\psi_{a,j,t}$ is the counterfactual risk under four conditions: consistency, no unmeasured
confounder of $A$ given $L$, censoring independent of the event times given $(A, L)$, and
positivity, $\pi(a \mid L)\, S_c(t^- \mid a, L) > 0$, with $\pi$ the treatment probability and
$S_c$ the censoring survival. No data check verifies the first three.

## Initial fits

Each cause hazard and the censoring hazard is chosen from a library of Cox models by
cross-validated partial likelihood, inside the training folds, and refitted with the Breslow
baseline. The default library holds three unstratified models: on the treatment alone, on main
terms, and on main terms with their treatment interactions. The treatment model is chosen between
a main-terms logistic model and L1 logistic models by cross-validated log loss. With
`cross_fit=True`, the default, every nuisance is fitted out of fold on ten folds stratified by
event code. A user treatment learner needs cross-fitting: in sample the interval assumes a
finite-dimensional model.

An event and a censoring at one time are ordered event first, so the censoring hazard leaves the
event out of its risk set. The weight $1 / \{\pi(a \mid L)\, S_c(u^- \mid a, L)\}$ is bounded by
raising its denominator to `g_bounds`, by default $5 / (\sqrt n \log n)$. Two checks read the
initial fits and raise a `DataError`, which a study records as a failed fit: a combined hazard jump
of one or more, and an initial survival below $10^{-250}$ before the grid's end, the last target
time or restricted horizon, where double precision no longer holds the clever covariate.

## The one-step walk

The efficient influence curve of $\psi_{a,j,t}$ (Rytgaard and van der Laan 2024, Section 3.1) is

$$
D_{a,j,t} = \frac{1\{A = a\}}{\pi(a \mid L)} \sum_{l=1}^J \int_0^t
\frac{c_{l}(u)}{S_c(u^- \mid a, L)} \bigl\{N_l(du) - Y(u)\, \Lambda_l(du \mid a, L)\bigr\}
+ F_j(t \mid a, L) - \psi_{a,j,t},
\qquad
c_l(u) = 1\{l = j\} - \frac{F_j(t \mid a, L) - F_j(u \mid a, L)}{S(u \mid a, L)},
$$

with $N_l$ the counting process of cause $l$ and $Y$ the at-risk indicator. The quotient in $c_l$
is the conditional risk of a cause-$j$ event in $(u, t]$ given survival past $u$,
$\sum_{u < s \le t} S(s^- \mid a, L)\, \Lambda_j(ds \mid a, L) / S(u \mid a, L)$. The engine
computes it in that form: it sums the incidence increments after $u$, each no larger than
$S(u)$, and divides by $S(u)$. The difference of two incidences near one loses every digit where
an extrapolating hazard fit takes the survival to $10^{-20}$; the sum keeps them, which
`test_the_conditional_risk_holds_every_digit_where_the_survival_is_tiny` in
`tests/unit/test_continuous_survival.py` checks against exact rational arithmetic.

The walk moves every cause hazard of both arms, for every unit, along the multinomial logistic
path

$$
\Lambda_l^{\varepsilon}(du) = \frac{\Lambda_l(du)\, e^{\varepsilon x_l(u)}}
{1 - \sum_k \Lambda_k(du) + \sum_k \Lambda_k(du)\, e^{\varepsilon x_k(u)}}, \qquad
x_l = \sum_p v_p\, \frac{c_{l,p}}{\pi\, S_c},
$$

on the observed jump times. The direction $v = \Sigma^{-1} P_n D / \lVert P_n D \rVert_{\Sigma}$
uses the covariance $\Sigma$ of the influence curves, held at the initial fit as Remark 2 holds
the metric. In
continuous time this is the source's exponential tilt; on observed jumps it keeps each cell's
summed jump below one, so the incidences and the survival add to one at every step. A step is
accepted when it lowers $\lVert P_n D \rVert_\Sigma$, and halved otherwise. The walk stops when
$\max_p |P_n D_p| / \hat\sigma_p \le 1 / (\sqrt n \log n)$ (Section 4.3), or at `max_iter`
accepted steps, at `2 * max_iter` rejected ones, or at a step below $10^{-8}$, with a
`ConvergenceWarning`.

Two departures from `concrete` 1.0.5 are deliberate. Each arm's hazard is tilted for every unit,
as the source's submodel tilts the hazard function; `concrete` tilts only the units whose received
arm is the target's arm, which a mutation test shows is not doubly robust. And a step that would
take some unit's survival at the grid's end below $10^{-8}$, or below the initial fit's smallest
survival there when that is lower, holds that (unit, arm) pair
still for that step and tries the step again; the pair moves on the next step. The record's
`frozen_share` counts the pairs ever held.

## Inference

Each interval is a Wald interval. A risk is a mean bounded by 0 and 1, so its interval is formed
on the logit scale and mapped back, as for every unclustered bounded mean ([small-sample
scale](inference.md#small-sample-scale)). A risk difference uses the difference of the two curves
on its own scale, and a risk ratio is built on the log scale. The study's committed rows were
written under the identity interval of a risk. On the same estimates and standard errors the logit
interval moves no group's coverage by more than 0.008 and changes no verdict
(`tests/canonical/interval-audit-screen.csv`).

### The switched variance

The default, `variance="switched"`, reports the influence-curve variance $\hat\sigma^2_{IC}$
unless the curve is sparse, and then the larger of it and the predictable variation
$\hat\sigma^2_{pv}$ of the curve's martingale part:

$$
\hat\sigma^2 =
\begin{cases}
  \max(\hat\sigma^2_{IC},\, g^\top \hat\Sigma_{pv}\, g) & \text{if } r > 0.125,\\
  \hat\sigma^2_{IC} & \text{otherwise,}
\end{cases}
$$

where $r = \sqrt{(\hat\kappa - 1)/n}$ is the relative standard deviation of the estimate's own
influence-curve variance (`variance_rel_sd` of `cleverly.inference.sparse.influence_row`) and $g$
is the estimate's gradient over the targeted levels. The influence-curve variance squares the
realized martingale increments $\int c(u)\,dM(u)$. The predictable variation integrates their
conditional variance under the fitted law instead:

$$
\hat\Sigma_{pv} = \widehat{\mathrm{Cov}}_n(\hat F) + P_n \sum_a \sum_{u_m}
  S(u_m^-)\, \pi_a S_c(u_m^-)\, w_a(u_m)^2\, V_a(u_m),
\qquad
V_{pq} = \sum_l c_{lp} c_{lq}\, d\Lambda_l - \Big(\sum_l c_{lp}\, d\Lambda_l\Big)\Big(\sum_l c_{lq}\, d\Lambda_l\Big),
$$

with $c_{lp}$ the clever covariate of target $p$ for cause $l$, $w_a = 1/\max(\pi_a S_c, \text{floor})$
and $V$ the covariance of one multinomial jump of the causes. The martingale block is zero
across arms. `predictable_covariance` in `cleverly.survival._targeting` computes the whole matrix
in one pass, and the result stores it as `result.predictable` on the scale of the influence-curve
variance (divided by $n - 1$). At the true nuisances $\hat\Sigma_{pv}$ and the influence curve's
covariance have the same expectation; `tests/unit/test_continuous_survival_variance.py` checks
the identity on an exact law to $10^{-10}$, with a mutation control for the multinomial square,
the left limit $S(u^-)$, one arm's martingale block, the floored weight and the off-diagonal
entries. A fit with $\tau = 1.5$ and a binding positivity floor checks that each stored variance
reads the walk's matrix in time units, over $n - 1$, with the floored weight.

Three properties follow. The switched variance is never below the influence-curve variance, so
its interval at the same centre contains the influence-curve interval and keeps its double
robustness. Because $r$ falls like $1/\sqrt n$, it equals the influence-curve variance with
probability tending to one. The threshold 0.125, close to a Kish count of 64, was chosen after
the X39 truth pilots were read (`cleverly.inference.sparse`); here it selects an estimator, and it
was not tuned on these laws. `variance="influence_curve"` reports the influence-curve variance
alone. Each estimate keeps both variances in `estimate.switched` and names the one it reports in
`estimate.variance_source`.

`contrast()`, `ratio()`, `rmtl()` and `rmst()` take the switched variance of a derived estimate
from the stored matrix, with the switch read on that estimate's own influence curve. A generic
`cleverly.inference.smooth_contrast` or `ratio_contrast` of switched estimates refuses, because it
holds no matrix; the refusal names `ContinuousSurvivalResult.contrast` and
`variance="influence_curve"`. The sup-t band takes its critical value from the curves
standardized by their influence-curve standard errors and its half-widths from the reported ones.

### Why the early-time interval under-covered

In the registered study's record written under the influence-curve variance, at the first
quartile ($t = 0.46$, n = 1,000, 7,586 fits), the estimates had the efficient spread, but treated
cause 2, with about 16 expected events, covered 0.881 with a mean standard error 8% below its
spread. Two mechanisms are at work. In a
sparse cell a few squared increments carry the influence-curve variance: its standard error varied
by 30% between draws and moved with the estimate, so the misses were one-sided. A smaller excess
spread appears at control cause 1, whose weights are heaviest.

The predictable variation repairs
the first: it varies less between draws and does not hinge on whether the few events occurred. Its
level is biased low by the fitted hazards in sparse cells (11% for treated cause 2 at the median in
1,040 exploratory fits), and for that reason the switched variance raises the influence-curve
variance and never replaces it with a smaller one. On exploratory fits its gain over the influence
curve came mostly from a wider interval in the rows where the curve is sparse, not from per-draw
information at matched width. The study
[continuous-time survival switched variance](method-evidence/continuous-survival-variance.md)
decided on five fresh laws that the switched variance stays the default; the floors it set are
stated under [evidence](#evidence). The registered study's record regenerated under the switched
default reads, at the first quartile, an SE ratio of 0.997 and coverage 0.932 for treated cause 2,
and SE ratios 0.990 to 1.038 and coverage 0.932 to 0.961 over the six targets (`early-time.csv`).
The first quartile still reads experimental, because 8 of the 30 first-quartile rows of the
switched-variance study fail its calibration rule.

With `simultaneous=True` the fit adds a sup-t band over every risk and risk difference. The record
reports the selected models, the steps, whether the stopping rule held, each target's final
standardized score, the plug-in before any step (`initial_estimates`), and four shares: the
weights raised to the positivity floor (`bounded_share`), the pairs below it at the last time
(`last_time_floor_share`), the pairs held by the walk (`frozen_share`), and the random times at
which the score equation also holds (`off_grid_share`).

The record also stores the factors of the positivity condition, for overlap plots and the
support diagnostic below. `p1` holds each row's $\hat\pi(1 \mid L_i)$, the probability of the
treated arm, with shape `(n,)`. `retention_before` holds $\hat S_c(t_k^- \mid a, L_i)$, the
probability of remaining uncensored just before each target time had the row received arm $a$,
with shape `(2, n, n_target_times)` and the arms in the order of `arms`. Both are predicted out of
fold when the fit is cross-fitted, in sample otherwise, and stored as fitted, before any bound.
The weights bound the product $\hat\pi(a \mid L) \hat S_c(u^- \mid a, L)$ for both arms and
never $\hat\pi$ alone, and `weight_floor` records that bound as resolved. `arm` holds each row's
received arm, `0` for the reference arm and `1` for the treated arm. No estimate reads these
arrays.

## Restricted means, ratios and the curve

The restricted mean time lost to cause $j$ up to $\tau$ and the restricted mean survival time are

$$
\mathrm{RMTL}_{a,j}(\tau) = E_L \int_0^\tau F_j(u \mid a, L)\, du, \qquad
\mathrm{RMST}_a(\tau) = \tau - \sum_{j=1}^J \mathrm{RMTL}_{a,j}(\tau).
$$

The fitted $F_j$ is a step function on the grid $u_1 < \dots < u_M$ of observed times, so its
integral is $\sum_{u_m \le \tau} F_j(u_m)\, \Delta_m(\tau)$ with
$\Delta_m(\tau) = \min(u_{m+1}, \tau) - u_m$, exactly. Every observed time up to $\tau$ is a
grid point, so $D_{a,j,u}$ is constant between grid points and the efficient influence curve of
the RMTL is $\sum_m \Delta_m(\tau)\, D_{a,j,u_m}$. A test checks it against the Gateaux derivative
of an exact law.

The walk solves the score equations of the risks at the target times only. In the registered
study the score equation also held at about 40% of 100 random times on average
(`tests/canonical/continuous_survival/targeting-readings.csv`, `mean_off_grid_share`), so the
integral of the targeted curve has no solved score. Rytgaard and van der Laan (2024), Theorem 1,
cover the whole curve only on a grid of mesh $O_P(n^{-1/3-\eta})$, which a user's few target times
do not meet. Each restriction time in `restricted_horizons=` therefore adds
$\mathrm{RMTL}_{a,j}(\tau)$, for both arms and every cause, to the targets of the walk.

A target
with grid weights $\omega_m$ has the clever covariate
$1\{l = j\}\, W(u) - \{\sum_{u_m \ge u} \omega_m F_j(u_m) - W(u) F_j(u)\} / S(u)$, with
$W(u) = \sum_{u_m \ge u} \omega_m$, divided by $\pi\, S_c$. Both sums are reverse cumulative sums,
so a step still costs $O(JnM)$, and a risk is the target whose weights pick one grid point. The RMST
is linear in the RMTLs, so its score is solved with theirs. The extra targets change the walk, so
the risks of a fit with restricted horizons differ slightly from those of a fit without them; a fit
without them is unchanged bit for bit.

The result reads only targeted estimates and smooth functions of them:

| method | reports |
| --- | --- |
| `rmtl(arm, tau, cause, versus=)` | a targeted $\mathrm{RMTL}_{a,j}(\tau)$, or the difference of the two arms' |
| `rmst(arm, tau, versus=)` | $\mathrm{RMST}_a(\tau)$, or the difference of the two arms' |
| `ratio(numerator, denominator, kind=, view=)` | the risk ratio or odds ratio of the two arms' risks at one cause and time, on the log scale; `view="survival"` takes the ratio of $1 - F$ on a fit with one event type |
| `contrast(function, names)` | a smooth function of reported estimates by the delta method |
| `curve(scale=, times=)` | the risks and risk differences at the target times, with the band columns when the fit has one; `scale="survival"` reports $1 - F$ on a fit with one event type |
| `to_dict()` | every estimate's record, the selected models, the walk's record and the settings, ready for JSON |

An RMTL and an RMST carry the bounds $(0, \tau)$, and their differences $(-\tau, \tau)$. A
level's interval is the range-logit interval on $(0, \tau)$ of [small-sample
scale](inference.md#small-sample-scale), so it stays inside its range, while its standard
error and curve stay in time units, as a risk's stay on the risk scale. A difference keeps the
identity scale, as a risk difference does.
`cleverly.sensitivity.evalue(result, (numerator, denominator))` reads the E-value of
`result.ratio(numerator, denominator)`, the two arms' risks at one cause and time. The bound needs
the unmeasured confounder to affect the treatment and the event only, and not censoring given
$(A, L)$. No coverage study registers its confidence-limit E-value, so the limit prints as
descriptive.

## Assessment

`ContinuousSurvivalResult` declares a capability row for every assessment operation, so
`result.assess()` reports each operation as run or refused and does not raise for a refused one.
The support, score, nuisance and influence diagnostics read the record and the influence curves
and refit nothing. The result also stores a copy of its estimator, with the seed resolved and the
outer fold labels recorded, and its data, so `result.estimator.refit(result.data,
random_state=result.estimator.random_state)` reproduces the fit bit for bit in one process. The
refutation, the truncation curve and the simulated common cause refit that copy.

| operation | status | what it reads, or why it refuses |
| --- | --- | --- |
| `diagnostics.support()` | runs, `completed`; `warning` when the floor raised more than 1% of the followed cells | the quantiles of $\hat\pi(1 \mid L)$ overall and by arm, its tail shares, the Kish ratio of $1 / \hat\pi(A \mid L)$ in each arm, the quantiles of $\hat S_c(t_k^- \mid A, L)$ at each target time, `bounded_share` (which grades the row on the point support report's 1% and 5% truncation tiers), `last_time_floor_share` and `frozen_share` |
| `diagnostics.score_equations()` | runs, `passed` or `failed` | each target's $\lvert P_n D^* \rvert / \hat\sigma$ against the walk's cutoff, the steps and halvings; an explicit `tolerance` is refused, because the cutoff is the fit's own stopping rule |
| `diagnostics.nuisance_models()` | runs, `completed` | each cause hazard's, the censoring hazard's and the treatment model's candidates, cross-validated losses and choice; a cross-fitted fit records the first fold's |
| `diagnostics.influence()` | runs, a direct call | the Kish count, skewness and kurtosis of each estimate's influence curve (`cleverly.inference.sparse`) |
| `diagnostics.corrections()` | `not_applicable` | the engine uses no correction system |
| `diagnostics.refute()` | runs, descriptive (`passed=None`) | the placebo on `ate_regimen` (null 0) and `rr_regimen` (null 1), `random_common_cause` and `subset` on any estimate, each refitting the stored estimator on a perturbed copy of the data; the continuous-time survival part of [refutation operations](validation-methods.md#refutation-operations) states the draws, nulls, limits and refusals |
| `diagnostics.truncation_curve()` | runs, descriptive | the walk rerun at each floor on $\hat\pi(a \mid L) \hat S_c(u^- \mid a, L)$ from refitted initial hazards, after an exact replay of the fit at its own floor; [truncation stability](validation-methods.md#truncation-stability) states the grid and columns |
| `sensitivity.simulated_confounding(estimand=..., grid=...)` | runs, descriptive; deferred without a grid and an estimand | refits the stored estimator once per cell on event and censoring times redrawn from the replay's own Cox fits, with each cause-specific log-hazard shifted by $\gamma U$ and censoring redrawn where the flip changed the treatment; the fit's fold plan is reused under its own seed. A cell that drives the initial survival to zero is a recorded failure. [Continuous-time survival replay](validation-methods.md#simulated-common-cause-stress-surface) states the law and its checks |
| `sensitivity.omitted_confounding()`, `robustness_value()`, `elements()`, `contour()`, `benchmark()` | refused, [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) | the point-treatment bound reads the Riesz representer of a mean of one outcome regression, and none is derived for a cause-specific risk under censoring |
| `sensitivity.marginal_bounds()` | refused, F16 | the point-treatment bound tilts the outcome regression of an arm mean, and no tilt of the cause-specific hazards is derived |
| `sensitivity.missingness()`, `tipping_gamma()` | refused, F16 | the tilt perturbs a missing-at-random outcome, and no tilt of independent censoring in continuous time is derived |
| `sensitivity.evalue(estimand=(numerator, denominator))` | runs, `passed`; deferred without an estimand | the E-value of `result.ratio(numerator, denominator)`, the two arms' risks at one cause and target time, as [above](#restricted-means-ratios-and-the-curve) |

Each refusal is a `CapabilityError` carrying the record its row declares. The free functions of
`cleverly.sensitivity` and `cleverly.validation` raise the same record before they read any
point-treatment attribute, and `cleverly.validation.refute` and
`cleverly.sensitivity.truncation_curve` run as the facade does.
`tests/unit/test_continuous_survival_assessment.py` checks every row and every refusal, and that
`influence()` reads each estimate's influence curve. The capability sweep
(`tests/unit/test_capability_row_sweep.py`) asks every row of a continuous-time fit whether its call
agrees, and requires `refute`, `truncation_curve` and `simulated_confounding` to answer.
`tests/unit/test_simulated_confounding_continuous.py` checks the replay.

A refute with the default tests and replicates costs 60 refits of the fit (40 on an arm risk), and the curve one refit
of the nuisances plus one walk per floor. A simulated common-cause surface costs one refit per cell
other than the anchor, so a 3 by 3 grid costs 8 refits. At n = 4,000 a cross-fitted default fit takes 60 to 120 s
(`tests/studies/continuous_survival.py`), and the curve holds one extra copy of the initial hazard
jumps and the weight product, about 600 MB, while it runs. No registered study calibrates the refutation readings on this
fit, so each prints its numbers without a verdict.

The test's witnesses are a failed walk that fails the score row, a binding floor that warns on the
support row, and a censoring model that depends on the arm. Under that model each arm's stored
censoring survival matches the model's prediction under that arm, recomputed as the product over the
observed times strictly before each target, and the two arms differ. One target sits on an observed
censoring time, so a lookup that includes that time's own jump fails. Under cross-fitting, a patched
treatment fit whose value encodes its training rows shows that each row stores the probability from
the model its own fold did not train. `tests/unit/test_continuous_survival_refute.py` holds the
witnesses and mutation controls of the stored estimator, the refutation and the curve: a refit at
another seed moves a cross-fitted estimate, an unseeded SAGA treatment learner breaks the replay
unless the fit seeds it, and the curve and refutation controls listed in the validation reference.

### Covariate balance

`covariate_balance(result)` returns a `BalanceOverTime` with three diagnostics, each over the
baseline covariates $W$, unweighted and weighted, with no interval. The general definitions and
the averages are in [covariate balance over time](longitudinal-tmle.md#covariate-balance-over-time).

| diagnostic | rows | compares | weight |
| --- | --- | --- | --- |
| `"confounding"` | every unit | the arms | $1 / \hat\pi(A \mid L)$, the point-treatment ATE weight |
| `"joint"`, at each $t_k$ | units whose status at $t_k$ is known: an event of any cause at $\tilde T \le t_k$, or $\tilde T \ge t_k$ | the arms | $1 / \max(\hat\pi(A \mid L)\,\hat S_c(\min(\tilde T, t_k)^- \mid A, L), \text{floor})$ |
| `"censoring"`, at each $t_k$ and arm $a$ | the arm's status-known units against the whole arm | known against all | $1 / \hat S_c(\min(\tilde T, t_k)^- \mid a, L)$ on the known units |

For a continuous all-cause $T$, $E[\mathbb 1\{\text{known}\} / S_c(\min(\tilde T, t_k)^-) \mid A, L]
= P(T \le t_k \mid A, L) + P(T \ge t_k \mid A, L) = 1$, so under correct models both weighted
differences are zero in expectation. The `"joint"` row is the weighted population the risk at
$t_k$ stands on, but it is an arm contrast and cannot see a censoring-model error that shifts both
arms alike; the `"censoring"` row can. Its standardized column divides by the arm's unweighted
standard deviation. A unit with an event before $t_k$ reads $\hat S_c(\tilde T^-)$ from
`record.retention_at_time`; the others read `record.retention_before`. A result built by hand with
no data is refused, and there is no feedback diagnostic, because the fit has no covariates after
baseline. `tests/unit/test_balance_over_time.py` checks both rows at large $n$ under the true
censoring model and under one that ignores $W$.

## Memory

The walk needs each unit's hazards only through means over units, so the engine walks blocks of
$c = \max(1, \lfloor 2^{18} / (J M) \rfloor)$ units, with $M$ the distinct observed times up to
the last target time or restricted horizon (`cleverly.survival._chunked`). A block holds both
arms' hazard jumps, survival and incidences on the walk's grid: every time where some initial
hazard jump is not zero, some event is observed or a risk's target time sits. At any other time
every jump is zero and the tilt keeps it zero, so the curves there repeat the last kept time's. A
restricted mean's weights also sit on the dropped times, so its sums run on the full grid and read
the block's curves through a forward-fill map. When some block is not kept, each pass over the
blocks evaluates up to four trial step sizes at once and takes them in the order the dense walk
would, holding and halving as it does; with every block kept, a pass tries one.

Beside the blocks the walk holds $(n, P)$ arrays: the initial and current influence curves and
each trial's score terms, with $P = 2 J Q$ targets. It keeps as many blocks between passes as the
budget allows, about $(33 J + 33)$ bytes per unit and kept time each, and rebuilds the others from
the fitted hazard models and the record of accepted steps. With no block kept, the peak grows with
$n P$ and the off-grid diagnostic's $(n, 200 J)$ array, not with $n M$: about 0.3 GB at
$n = 100{,}000$ with one cause, two target times and $M = 59{,}980$, against about 1 TB to hold every
unit's arrays at once. Keeping every block at that size would take about 400 GB, so a large fit
rebuilds most blocks on every pass. A block rebuilt in the pass after accepted step $s$ replays all
$s$ steps, so a walk of $S$ accepted steps replays about $S (S + 1) / 2$ steps per block not kept,
and a larger `max_iter` lengthens a long walk quadratically.

The switched variance (`variance="switched"`, the default) adds its predictable variation in the
final pass. Each block's targeted jumps are copied onto the full grid into a buffer of
$b = \max(1, \lfloor 6 \times 10^6 / (J^2 Q M) \rfloor)$ units, the blocks over which
`cleverly.survival._targeting.predictable_covariance` sums. When the buffer fills, its units'
survival, incidences, bounded weight and floored ratio $\min(1, \hat\pi \hat S_c / g_{\min})$
are built and its martingale term is added, and every unit's plug-ins, $(n, P)$, are kept for
the centred plug-in covariance. The pass holds $16 J b M$ bytes for the buffer and about
$8 b M (J^2 Q + 3 J Q + 8)$ more while a buffer's term is formed: at most about 0.4 GB with one
cause and two targets, whatever $n$. So the default's peak grows with $n P$ too, plus that bounded
term: about 0.7 GB at $n = 100{,}000$ with one cause, two target times and $M = 59{,}980$.
`variance="influence_curve"` skips the pass and needs the 0.3 GB above. A refusal whose peak is
this pass's names the final pass and `variance="influence_curve"` with its own prediction.

The buffer is cut where the sum's blocks are, so the
pass adds the same blocks whatever the walk's block size or the blocks kept, and the martingale
term equals the one-pass computation's bit for bit. The plug-ins' matrix products run over a
buffer's rows rather than all $n$, so for a restricted mean the plug-in covariance can differ in
the last bit. Over 61 study fits refitted both ways, every standard error, both variances and
every choice between them matched a pass over all units at once bit for bit, and one matrix entry
pair of one restricted-mean fit differed by $1.5 \times 10^{-16}$ relative.

Before fitting any nuisance, the fit predicts each phase's peak with no block kept and the walk on
the full grid (`cleverly.survival._memory.peak_bytes`), and refuses with
`MethodConfigurationError` when that prediction exceeds the budget. The budget is the memory
available at that moment (`cleverly.utils.memory.available_memory`, which reads the control
group's limit on Linux), or `max_memory_bytes=` on `ContinuousSurvivalTMLE`, which overrides it in
either direction. Where the platform reports no available memory, as on macOS, nothing is checked
and every block is kept unless `max_memory_bytes=` is passed. After the initial pass the walk
keeps blocks up to the budget, or up to three quarters of the memory available under the default,
less a 15% margin for memory the C library's allocator keeps after freeing block arrays (the
resident set read 7% to 9% above the predicted arrays at 8,000 to 32,000 units).
The budget sets how many blocks are kept, so how long the walk runs, and never what it returns.
Every refit of a result, and so every refutation, assessment replay and truncation curve, uses the
fit's `max_memory_bytes`, and a truncation curve holds no more than its fit.
`TimeToEvent(grid="continuous")` has no such setting; its refusal spells the
`ContinuousSurvivalTMLE(...).fit_data(study.data)` call with the study's settings, where the budget
can be set.

With `n_jobs=` above one on `ContinuousSurvivalTMLE` (or `Runtime(n_jobs=...)` on the study
route), each pass runs its blocks on that many threads. Each block writes only its own rows, and
the off-grid sums and any floating-point signal are taken in row order, at most $2 \times$
`n_jobs` blocks ahead, so every result is bit for bit the same at any `n_jobs`. A user learner is
called by one thread at a time, and the threads leave the BLAS thread settings alone. Each thread
holds one block's working arrays, which the prediction and the budget count. `-1` takes the CPUs
the process may use, capped by a control group's quota.

Rebuilding a block calls the hazard learner again, so a learner's `cumulative_hazard` must return
the same values for the same rows on every call, each row predicted from that row alone. The fit
predicts the first block's cause and censoring hazards twice and refuses a user learner whose
predictions differ; it cannot check that a learner predicts each row from that row alone. The shipped
learners are read from their linear predictors and baseline hazards, or their strata, without a
second call.

For risks with the shipped learners, every estimate, standard error, influence curve, band and
record field equals the dense walk's bit for bit, whatever the block size, the blocks kept, the
number of trial steps per pass or the budget. With a restricted mean or a user hazard learner the
matrix products run over a block's rows rather than all $n$, so BLAS adds in another order: every
step count, halving, convergence flag and share is unchanged, and floats agree to $10^{-8}$
relative (the largest difference over 200 replayed study fits was $1.6 \times 10^{-9}$, and the same
fit run as one block was bit for bit the dense walk's).
`tests/unit/test_continuous_survival_blocks.py` checks both against a transcription of the dense
walk run from the same nuisance fits, anchored to numbers the dense engine produced, with mutation
controls on the event-support grid, the replay, the trial ladder and the off-grid sums.

The prediction counts NumPy arrays the engine and the shipped hazard learners allocate. It does
not count a user learner's own allocations, which can only make the check miss a refusal.
`tests/unit/test_continuous_survival_memory.py` checks that each phase's `tracemalloc` peak stays at
or below its prediction in fits that keep every block, none or half, with one to three causes, a
restricted mean or two, cross-fitting, the factual tilt and both variances. The walk's and the final
pass's predictions stay within 10% of their peaks, and the nuisance phase's within 25% for an
interaction Cox model on eight covariates. Dropping one term from a phase's prediction falls below
its peak, and doubling $n$ with no block kept, under the default with a predictable-variation
buffer smaller than a block, at most multiplies the peak by 2.5. Coarsening the time grid would
lower $M$ but changes the analysis, so the engine never does it.

## What is refused

| request | why |
| --- | --- |
| a treatment with more than two levels, a rule, a stochastic policy | every source has two static arms |
| a target time before the first event of some cause in some arm | that risk's influence curve is zero and its interval undefined |
| a target time past one arm's follow-up | the risk there would be an extrapolation of the hazard model |
| a declared cause with no event | its risk is zero by construction |
| a user treatment learner with `cross_fit=False` | the in-sample interval assumes a finite-dimensional model |
| a method other than `TMLEMethod`, or a setting the engine does not read | the continuous-time engine runs this one estimator |
| `rmtl()` or `rmst()` at a $\tau$ not in `restricted_horizons=` | the risks at the target times do not solve the score of their integral |
| `curve()` or `ratio()` at a time that is not a target time | a risk between target times solves no score equation of the fit |
| `ratio()` of anything but the two arms' risks at one cause and one time | `contrast()` takes any other smooth function of reported estimates |
| a restricted horizon past one arm's follow-up, or before the first event of some cause in some arm | as for a target time |
| `scale="survival"` or `view="survival"` on a fit with several causes | event-free survival is one minus the sum of the incidences |
| a fit whose predicted peak memory with no block kept exceeds the memory available or `max_memory_bytes=` | the fit would not fit even rebuilding every block; see [Memory](#memory) |
| a user hazard learner whose predictions change between calls | a block's hazards are rebuilt from the model on every pass; see [Memory](#memory) |

The rows on an arm's first event of a cause and on an arm's follow-up read the drawn data, not the
request, so they raise `DataError`, and a repeated-sampling study records such a draw as a lost
replicate. The other rows raise `CapabilityError`.

## Evidence

The registered study [continuous-time survival TMLE](method-evidence/continuous-time-survival.md)
gates the default cross-fitted fit at n = 1,000 on one law with two causes, at the median and the
upper quartile of the observed event times. An estimate reads `supported` when its fit matches
that configuration: cross-fitted, the default libraries and settings, two causes (see below for
one and three), a risk or a
risk difference, at a time by which 40% to 85% of the observed events of any cause have happened
(`cleverly.survival.result.continuous_survival_support`), and the two floors of the switched-variance
study below. Its record was regenerated under the switched default: all 12 primary rows pass, with
coverage 0.942 to 0.952 and SE ratio 0.991 to 1.032 over 7,600 fits, and the six property cells
pass. At the lower quartile it covered 0.932 to 0.961, but an estimate at a time that early reads
experimental because the switched-variance study's first-quartile rows do not all pass. The time
condition is calibrated on this one law and does not read how many events each arm and cause has.

The registered study [continuous-time survival restricted means and risk
ratios](method-evidence/continuous-survival-functionals.md) gates the same configuration with
`restricted_horizons=(2.32,)`, the upper quartile and last target time: the risks and risk
differences of that fit, whose extra targets move them, the log risk ratios at the median and
upper quartile, and the RMTL of each arm and cause, the RMST of each arm and their differences
at $\tau$. Its record was regenerated under the switched default interval. All 25 rows pass,
with coverage 0.939 to 0.954, and the RMST difference meets the calibration bands (coverage 0.945,
SE ratio 0.996). A fit with one restricted horizon at or before
its last target time reads `supported` under the conditions above with at least 1,000 units
(`FUNCTIONALS_N`), the one size that study ran, and its restricted means read
`supported` only at a $\tau$ by which 67% to 85% of the observed events have happened
(`RESTRICTED_EVENT_SHARE`). The measured $\tau$ sits at about 75%, and a restricted mean at an
earlier $\tau$ weights the sparse early times, where the risks' standard error ran low, more
heavily.

A fit with several restricted horizons, or one past its last target time, whose grid
the walk then extends, reads experimental for every estimate: only the exact-law tests below
measure that extension. The risk ratio of a fit without restricted means, an odds ratio, the
survival view, a smooth contrast and the band read experimental. A risk-only walk on the
study's draws left the RMTL score above the cutoff in 25% to 51% of fits but was unbiased on
this law, so the claim that a restricted mean must be targeted rests on the score test below.

The registered study [continuous-time survival switched
variance](method-evidence/continuous-survival-variance.md) ships the switched variance as the
default: it passes its ship rule with 20 rows improved and none harmed, and every gated row of its
four gated cells passes at 6,000 replicates.

The switched-variance study sets three floors. A risk or risk difference whose `variance_rel_sd`
lies above 0.246 (`MEASURED_REL_SD`) reads experimental. The share floor stays at 0.40, because
not every first-quartile row passed. The study's gated 500-unit cell, which fits no restricted
horizon, passed every row, so a risk or risk difference of a fit without restricted horizons
needs 500 units (`MEASURED_N`), not the 1,000 the first study gated, and a fit with restricted
horizons keeps 1,000. A supported risk or risk difference names this study in its evidence beside
the study that measured its configuration.

A fit with one cause or with three reads experimental until the study
`continuous-survival-causes` passes for that number of causes; four or more causes always read
experimental. That study is declared and its pilot is committed; its declared run has not yet
been made. It fits the default cross-fitted configuration at n = 1,000 on two laws built from the
registered one: one cause, its cause-2 hazard removed, and three causes, with a third constant
hazard $h_3 = \exp(-2.5 + 0.5 X_1 + 0.5 X_3 + 0.5 Z)$ added. It gates every risk and risk
difference at each law's median and upper quartile (6 rows and 18), a calibration cell per law
with the shrunk-SE and noise controls, and per law the doubly robust cell with wrong hazards and
the right propensity beside two controls on the same draws, the initial plug-in and the factual
tilt, each of which must stay biased. The pilot sets 7,200 replicates per primary cell and 3,200 per
calibration cell (`tests/canonical/continuous_survival_causes/regenerate.py` records its readings).

A number of causes joins `MEASURED_CAUSES` only when every gated row and cell of its law passes,
and then needs at least 1,000 units (`CAUSES_N`), since the 500-unit floor was measured with two
causes. A fit with one or three causes and a restricted
horizon keeps reading experimental, because the functionals study measures two causes only.

`tests/unit/test_continuous_survival.py` checks the influence curve
against the Gateaux derivative of an exact law, the path and the stopping rule, and a mutation
control for each censoring, survival and tilting term. `tests/unit/test_continuous_survival_functionals.py`
checks the restricted mean's curve against the Gateaux derivative of the same law and the weighted
direction against a sum over grid points, with the term $W(u) F_j(u)$ dropped as a mutation. It
also checks that the walk removes the bias of a restricted mean under a wrong hazard, and that a
restricted mean left out of the direction stays unsolved and biased.
`tests/unit/test_continuous_survival_variance.py` checks the predictable-variation covariance
against the influence curve's on an exact law, the switch on both sides of its threshold, and the
derived estimates against the stored ones.
`tests/unit/test_continuous_survival_causes.py` builds an exact law with any number of causes and
checks, with one cause and with three, the influence curve against the Gateaux derivative, the
predictable-variation covariance against the influence curve's, and that the walk from hazards
that ignore the covariate reaches the truth with the incidences and the survival adding to one.
Its mutations freeze the third cause's tilt, drop the third cause from the survival or from the
multinomial cross term, drop the one-cause square $d\Lambda^2$, reorder the targets, and tilt only
the received arm; each fails.
