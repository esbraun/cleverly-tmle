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
uses the covariance $\Sigma$ of the influence curves held at the initial fit (Remark 2). In
continuous time this is the source's exponential tilt; on observed jumps it keeps each cell's
summed jump below one, so the incidences and the survival add to one at every step. A step is
accepted when it lowers $\lVert P_n D \rVert_\Sigma$, and halved otherwise. The walk stops when
$\max_p |P_n D_p| / \hat\sigma_p \le 1 / (\sqrt n \log n)$ (Section 4.3), or at `max_iter`
accepted steps with a `ConvergenceWarning`.

Two departures from `concrete` 1.0.5 are deliberate. Each arm's hazard is tilted for every unit,
as the source's submodel tilts the hazard function; `concrete` tilts only the units whose received
arm is the target's arm, which a mutation test shows is not doubly robust. And a step that would
take some unit's survival at the last target time below $10^{-8}$ holds that (unit, arm) pair
still for that step and tries the step again; the pair moves on the next step. The record's
`frozen_share` counts the pairs ever held.

## Inference

Each interval is a Wald interval with the empirical variance of the influence curve. A risk
difference uses the difference of the two curves, and a risk ratio is built on the log scale. With
`simultaneous=True` the fit adds a sup-t band over every risk and risk difference. The record
reports the selected models, the steps, whether the stopping rule held, each target's final
standardized score, the plug-in before any step (`initial_estimates`), and four shares: the
weights raised to the positivity floor (`bounded_share`), the pairs below it at the last time
(`last_time_floor_share`), the pairs held by the walk (`frozen_share`), and the random times at
which the score equation also holds (`off_grid_share`).

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
that configuration: cross-fitted, the default libraries and settings, two causes, a risk or a risk
difference, at a time by which 40% to 85% of the observed events have happened
(`cleverly.survival.result.continuous_survival_support`). At the lower quartile the standard error
ran up to 8% below the sampling spread and coverage fell to 0.88, so an estimate at a time that
early reads experimental. `tests/unit/test_continuous_survival.py` checks the influence curve
against the Gateaux derivative of an exact law, the path and the stopping rule, and a mutation
control for each censoring, survival and tilting term.
