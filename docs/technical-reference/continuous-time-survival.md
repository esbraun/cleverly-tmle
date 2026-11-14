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
of one or more, and an initial survival that reaches zero before the grid's end, the last target
time or restricted horizon.

## The one-step walk

The efficient influence curve of $\psi_{a,j,t}$ (Rytgaard and van der Laan 2024, Section 3.1) is

$$
D_{a,j,t} = \frac{1\{A = a\}}{\pi(a \mid L)} \sum_{l=1}^J \int_0^t
\frac{c_{l}(u)}{S_c(u^- \mid a, L)} \bigl\{N_l(du) - Y(u)\, \Lambda_l(du \mid a, L)\bigr\}
+ F_j(t \mid a, L) - \psi_{a,j,t},
\qquad
c_l(u) = 1\{l = j\} - \frac{F_j(t \mid a, L) - F_j(u \mid a, L)}{S(u \mid a, L)},
$$

with $N_l$ the counting process of cause $l$ and $Y$ the at-risk indicator. The walk moves every
cause hazard of both arms, for every unit, along the multinomial logistic path

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
take some unit's survival at the grid's end below $10^{-8}$ holds that (unit, arm) pair
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

### Why the early-time interval under-covers

In the registered study at the first quartile ($t = 0.46$, n = 1,000, 7,586 fits) the estimates
had the efficient spread, but treated cause 2, with about 16 expected events, covered 0.881 with a
mean standard error 8% below its spread (`early-time.csv`). Two mechanisms are at work. In a
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
decides on fresh laws whether the switched variance stays the default. It is declared and has not
run.

With `simultaneous=True` the fit adds a sup-t band over every risk and risk difference. The record
reports the selected models, the steps, whether the stopping rule held, each target's final
standardized score, the plug-in before any step (`initial_estimates`), and four shares: the
weights raised to the positivity floor (`bounded_share`), the pairs below it at the last time
(`last_time_floor_share`), the pairs held by the walk (`frozen_share`), and the random times at
which the score equation also holds (`off_grid_share`).

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

## Evidence

The registered study [continuous-time survival TMLE](method-evidence/continuous-time-survival.md)
gates the default cross-fitted fit at n = 1,000 on one law with two causes, at the median and the
upper quartile of the observed event times. An estimate reads `supported` when its fit matches
that configuration: cross-fitted, the default libraries and settings, at least 1,000 units, two
causes, a risk or a risk difference, at a time by which 40% to 85% of the observed events of any
cause have happened (`cleverly.survival.result.continuous_survival_support`). At the lower
quartile the standard error ran up to 8% below the sampling spread and coverage fell to 0.88, so
an estimate at a time that early reads experimental. The time condition is calibrated on this one
law and does not read how many events each arm and cause has.

The registered study's rows were written under the influence-curve variance. On exploratory fits
of its law the switched variance widened the gated rows' intervals by 0% to 4.7%, and the study
regenerates under the switched default once the switched-variance study ships it. That study also
sets a floor on `variance_rel_sd`: a risk or risk difference whose curve is sparser than every
gated row that passed reads experimental. Until the study runs the floor is pending and labels no
estimate.

No registered study measures a risk ratio, an odds ratio, a restricted mean, a smooth contrast,
or any estimate of a fit with `restricted_horizons=`, whose extra targets move its risks, so each
of these reads experimental.

`tests/unit/test_continuous_survival.py` checks the influence curve against the Gateaux
derivative of an exact law, the path and the stopping rule, and a mutation control for each
censoring, survival and tilting term. `tests/unit/test_continuous_survival_functionals.py` checks
the restricted mean's curve against the Gateaux derivative of the same law and the weighted
direction against a sum over grid points, with the term $W(u) F_j(u)$ dropped as a mutation. It
also checks that the walk removes the bias of a restricted mean under a wrong hazard, and that a
restricted mean left out of the direction stays unsolved and biased.
`tests/unit/test_continuous_survival_variance.py` checks the predictable-variation covariance
against the influence curve's on an exact law, the switch on both sides of its threshold, and the
derived estimates against the stored ones.
