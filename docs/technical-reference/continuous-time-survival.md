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
of one or more, and an initial survival that reaches zero before the last target time.

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
take some unit's survival at the last target time below $10^{-8}$ holds that (unit, arm) pair
still for that step and tries the step again; the pair moves on the next step. The record's
`frozen_share` counts the pairs ever held.

## Inference

Each interval is a Wald interval with the empirical variance of the influence curve. A risk is
a mean bounded by 0 and 1, so its interval is formed on the logit scale and mapped back, as for
every unclustered bounded mean ([small-sample scale](inference.md#small-sample-scale)). A risk
difference uses the difference of the two curves on its own scale, and a risk ratio is built on
the log scale. The study's committed rows were written under the identity interval of a risk. On
the same estimates and standard errors the logit interval moves no group's coverage by more than
0.008 and changes no verdict (`tests/canonical/interval-audit-screen.csv`).

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
arrays, and storing them left every estimate of a fixed fit bit-identical.

## Assessment

`ContinuousSurvivalResult` declares a capability row for every assessment operation, so
`result.assess()` reports each operation as run or refused and does not raise for a refused one.
Each diagnostic that runs reads the record and the influence curves and refits nothing.

| operation | status | what it reads, or why it refuses |
| --- | --- | --- |
| `diagnostics.support()` | runs, `completed`; `warning` when the floor raised more than 1% of the followed cells | the quantiles of $\hat\pi(1 \mid L)$ overall and by arm, its tail shares, the Kish ratio of $1 / \hat\pi(A \mid L)$ in each arm, the quantiles of $\hat S_c(t_k^- \mid A, L)$ at each target time, `bounded_share` (which grades the row on the point support report's 1% and 5% truncation tiers), `last_time_floor_share` and `frozen_share` |
| `diagnostics.score_equations()` | runs, `passed` or `failed` | each target's $\lvert P_n D^* \rvert / \hat\sigma$ against the walk's cutoff, the steps and halvings; an explicit `tolerance` is refused, because the cutoff is the fit's own stopping rule |
| `diagnostics.nuisance_models()` | runs, `completed` | each cause hazard's, the censoring hazard's and the treatment model's candidates, cross-validated losses and choice; a cross-fitted fit records the first fold's |
| `diagnostics.influence()` | runs, a direct call | the Kish count, skewness and kurtosis of each estimate's influence curve (`cleverly.inference.sparse`) |
| `diagnostics.corrections()` | `not_applicable` | the engine uses no correction system |
| `diagnostics.refute()` | refused, [X101](../roadmap.md#x101-assessment-of-continuous-time-survival-fits) | each test refits the estimator on perturbed data, and the result keeps neither its estimator nor its data |
| `diagnostics.truncation_curve()` | refused, X101 | each point reruns the walk at another floor, and the result keeps no initial hazard jumps |
| `sensitivity.simulated_confounding()` | refused, X101 | the replay refits on simulated data, as `refute` does |
| `sensitivity.omitted_confounding()`, `robustness_value()`, `elements()`, `contour()`, `benchmark()` | refused, [F16](../roadmap.md#f16-longitudinal-sensitivity-bound-estimation) | the point-treatment bound reads the Riesz representer of a mean of one outcome regression, and none is derived for a cause-specific risk under censoring |
| `sensitivity.marginal_bounds()` | refused, F16 | the point-treatment bound tilts the outcome regression of an arm mean, and no tilt of the cause-specific hazards is derived |
| `sensitivity.missingness()`, `tipping_gamma()` | refused, F16 | the tilt perturbs a missing-at-random outcome, and no tilt of independent censoring in continuous time is derived |
| `sensitivity.evalue()` | refused, [X6](../roadmap.md#x6-continuous-time-survival-and-competing-risks) | no reading of the confidence-limit E-value of a continuous-time risk ratio is registered; the refusal names `evalue_from_rr` for a descriptive reading |

Each refusal is a `CapabilityError` carrying the record its row declares. The free functions of
`cleverly.sensitivity` and `cleverly.validation` raise the same record before they read any
point-treatment attribute. `tests/unit/test_continuous_survival_assessment.py` checks every row and
every refusal, and that `influence()` reads each estimate's influence curve. Its witnesses are a
failed walk that fails the score row, a binding floor that warns on the support row, and a
censoring model that depends on the arm. Under that model each arm's stored censoring survival
matches the model's prediction under that arm, and the two arms differ. The capability sweep
(`tests/unit/test_capability_row_sweep.py`) asks every row of a continuous-time fit whether its
call agrees.

## What is refused

| request | why |
| --- | --- |
| a treatment with more than two levels, a rule, a stochastic policy | every source has two static arms |
| a target time before the first event of some cause in some arm | that risk's influence curve is zero and its interval undefined |
| a target time past one arm's follow-up | the risk there would be an extrapolation of the hazard model |
| a declared cause with no event | its risk is zero by construction |
| a user treatment learner with `cross_fit=False` | the in-sample interval assumes a finite-dimensional model |
| a method other than `TMLEMethod`, or a setting the engine does not read | the continuous-time engine runs this one estimator |

## Evidence

The registered study [continuous-time survival TMLE](method-evidence/continuous-time-survival.md)
gates the default cross-fitted fit at n = 1,000 on one law with two causes, at the median and the
upper quartile of the observed event times. An estimate reads `supported` when its fit matches
that configuration: cross-fitted, the default libraries and settings, at least 1,000 units, two
causes, a risk or a risk difference, at a time by which 40% to 85% of the observed events of any
cause have happened
(`cleverly.survival.result.continuous_survival_support`). At the lower quartile the standard error
ran up to 8% below the sampling spread and coverage fell to 0.88, so an estimate at a time that
early reads experimental. The time condition is calibrated on this one law and does not read
how many events each arm and cause has. `tests/unit/test_continuous_survival.py` checks the influence curve
against the Gateaux derivative of an exact law, the path and the stopping rule, and a mutation
control for each censoring, survival and tilting term.
