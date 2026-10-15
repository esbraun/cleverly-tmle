# Sequential g-computation and IPW

## What this solves

`SequentialGComputation` and `SequentialIPW` estimate the regimen levels and contrasts that
[`LTMLE`](longitudinal-tmle.md) estimates. They estimate the same target under the same
identification assumptions. They are the reference estimators that applied work reports beside a
targeted estimate.

| estimator | what the estimate reads | interval |
| --- | --- | --- |
| `SequentialGComputation` | the outcome and pseudo-outcome regressions | the stacked estimating equation of the `GLM` regressions |
| `SequentialIPW` | the treatment and censoring mechanisms | the stacked estimating equation of the `GLM` mechanisms |

Both intervals hold for the target only when every `GLM` that the estimate reads is correctly
specified. Under misspecification the interval covers the limit of the misspecified fit, which
differs from the target. A targeted estimator has no such condition on one of its two models.
Fit `LTMLE` or `SDR` when the models are not known to be correct.

```python
from cleverly.datasets import make_longitudinal
from cleverly.learners import GLM
from cleverly.longitudinal import SequentialGComputation, SequentialIPW

frame, _ = make_longitudinal(n=300, seed=0)
columns = dict(
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
gcomp = SequentialGComputation(
    {"always": 1, "never": 0}, reference="never", outcome_learner=GLM()
).fit(frame, **columns)
ipw = SequentialIPW({"always": 1, "never": 0}, reference="never", treatment_learner=GLM()).fit(
    frame, **columns
)
print(gcomp.summary())
print(ipw.summary())
```

`gcomp()` and `ipw()` are the one-call forms.

## The estimators

Write $\lambda_t$ for the cumulative ratio of the intervention density to the treatment and
censoring mechanism through node $t$. Write $R_t$ for the indicator that a unit follows the plan
through node $t$, is uncensored, and is event-free before $t$. The estimators then are:

| estimator | end of study | survival risk at horizon $k$ |
| --- | --- | --- |
| g-computation (ICE) | the mean of the first-node regression $\bar Q_1$ of the untargeted recursion | the same recursion, with the pseudo-outcome $Z_t = Y_t + (1 - Y_t) \bar Q_{t+1}$ |
| IPW, `normalization="hajek"` (default) | $P_n[w \lambda_T R_T Y] / P_n[w \lambda_T R_T]$ | $1 - \prod_{t \le k} (1 - h_t)$, with the IPW hazard $h_t = P_n[w \lambda_t R_t Y_t] / P_n[w \lambda_t R_t]$ |
| IPW, `normalization="horvitz_thompson"` | $P_n[w \lambda_T R_T Y]$ | $\sum_{t \le k} P_n[w \lambda_t R_t Y_t]$ |

A competing-risk fit replaces the hazard product by the Aalen-Johansen sum
$F_j(k) = \sum_{t \le k} S(t - 1)\, h_{jt}$, with $S(t) = \prod_{s \le t} (1 - \sum_j h_{js})$.

| item | source |
| --- | --- |
| the ICE recursion | Bang & Robins (2005), Section 4, p. 969, steps 1 to 5 |
| the shared recursion | `untargeted_recursion` in `longitudinal/sequential.py`. `LTMLE` fits it before its fluctuation |
| the Hajek form | `CalcIPTW` in R `ltmle` 1.3-0, and the delta method on a fixed number of means |
| the survival identity | the end-of-study identity at each node, summed over the nodes |

The g-computation estimate is the first-node mean of the recursion that `LTMLE` fits before its
fluctuation. `tests/unit/test_sequential_baselines.py` (W8) checks that the two agree bit for
bit. The Hajek and Horvitz-Thompson forms are two finite-sample estimators of one target. Hajek
stays in the outcome range. Both read a continuous outcome on its own scale, so `q_bounds=` does
not move them while the bounds contain the outcome.

## Inference

### The stacked curve

Each `GLM` solves its own score equation. The estimate solves one more equation. The influence
curve of the stacked system is the curve that the result reports (Stefanski & Boos 2002,
Section 2).

| estimator | curve |
| --- | --- |
| g-computation | $w (\bar Q_1 - \hat\psi) + \sum_t a_t^\top M_t^{-1} S_t$. $S_t$ is the node-$t$ score, $M_t$ its information over $n$, $a_1$ the derivative of the plug-in mean, and $a_{t+1}^\top = a_t^\top M_t^{-1} N_t$ |
| IPW | the weighted summand minus its mean, plus $b^\top I^{-1} S$ over every estimated mechanism factor. $b = -P_n[x\, \partial \log(g_s c_s)]$ for the summand $x$ |
| Hajek forms | the delta method through the two means, the hazards and the product |

$N_t$ is the derivative of the node-$t$ score in the later node's coefficients. It carries the
pseudo-outcome's dependence on the later fit. A prediction that a bound moves has derivative 0.

| check | test | tolerance |
| --- | --- | --- |
| the curve has mean 0 | W5 | `1e-10` relative |
| the curve is the derivative of the estimate in each observation weight | W6 | `1e-5` relative, at `n = 200` |
| the curve equals the efficient influence curve on a saturated law | `test_the_stacked_curve_is_the_gateaux_derivative_on_a_saturated_law` | `1e-10` |
| an independent numpy construction agrees | W10 | `1e-10` on points, `1e-8` on curves |

Lunceford & Davidian (2004), Section 3.1, eqs. (13) to (16), show the effect of the mechanism
term at a single time point. With a correct ML-fitted mechanism, the stacked IPW variance is at
most the fixed-mechanism variance.

### Status

| fit | status | what the result reports |
| --- | --- | --- |
| every regression that the estimate reads is a `GLM`, or a node with no learner (a declared known factor or no censored unit) | `influence_curve` | the stacked curve, with an interval |
| a regression that the estimate reads uses another learner | `adaptive_baseline_plugin` | the point estimate, and a diagnostic curve under `plugin_std_error` |
| `weights_estimated=True` | `estimated_weight_plugin` | the stacked curve as a diagnostic |
| a node that reads a constant target | `constant_node_plugin`, on that parameter | as for `LTMLE` |

G-computation reads the outcome and pseudo-outcome regressions only. IPW reads the treatment and
censoring mechanisms only. A g-computation fit with a data-adaptive `treatment_learner` therefore
keeps its interval. The mechanism of a g-computation fit enters only the diagnostics.

Under `adaptive_baseline_plugin`, the diagnostic curve of g-computation is the efficient
influence curve at the untargeted fits. That is the curve that R `ltmle` reports for
`gcomp=TRUE`. The diagnostic curve of IPW is the fixed-mechanism curve. No result gives the
asymptotic linearity of either estimator with data-adaptive nuisances over several time points.
Ertefaie, Hejazi & van der Laan (2023), Section 3, Theorem 1, cover IPW with an undersmoothed HAL
propensity at one time point only. [F32](../roadmap.md#f32-inference-for-sequential-baselines-with-data-adaptive-nuisances)
reopens the status when such a result exists.

The full-refit bootstrap runs and reports `bootstrap sd` with no percentile inference, as for
`SDR`. No registered study measures it for the baselines.

### Clusters

A cluster is the unit of the curve, as for `LTMLE`. The few-cluster floor of `LTMLE` applies
unchanged. No study measures that floor for the baselines.

## Settings

| setting | `SequentialGComputation` | `SequentialIPW` |
| --- | --- | --- |
| default learners | `GLM()` for every regression | `GLM()` for the treatment and censoring mechanisms |
| `n_folds` | `1`. `n_folds > 1` raises `CapabilityError` | the same |
| `alpha=`, `max_iter=`, `tol=` | `TypeError`. No fluctuation exists | the same |
| `g_bounds=` | moves only the diagnostics and the diagnostic curve | bounds each cumulative mechanism probability. A bounded prefix has derivative 0 |
| `treatment_learner=`, `censoring_learner=` | fit the diagnostic mechanism only | fit the estimate |
| `outcome_learner=`, `pseudo_learner=` | fit the estimate | `CapabilityError`. IPW fits no outcome regression |
| `normalization=` | none | `"hajek"` (default) or `"horvitz_thompson"` |

The default `GLM()` follows the `SL.library="glm"` default of R `ltmle`. `GLM(interactions=k)`
adds products of distinct columns.

## Refusals

| request | estimator | error |
| --- | --- | --- |
| `n_folds > 1` | both | `CapabilityError`. Neither baseline has a cross-fitted root-n result. Fit `LTMLE` or `SDR` |
| `msm=` | both | `CapabilityError`. `LTMLE(msm=...)` and `SDR(msm=...)` fit a working model |
| an odds tilt with an estimated mechanism | both | `CapabilityError`. Fit `IPSI` |
| a modified treatment policy | IPW | `CapabilityError`. IPW of a shifted law needs a density ratio |
| `ratio=`, `density_bins=` | IPW | `CapabilityError` |
| a continuous outcome, a binomial `GLM`, and `q_bounds=None` | g-computation | `CapabilityError`. The scale moves with the sample extremes. Declare `q_bounds=`, or pass `GLM(family="gaussian")` |
| `longitudinal_truncation_curve` | g-computation | `CapabilityError`. `g_bounds` does not move the estimate |
| a prediction through a `GLM` column that the followers alias | g-computation | `DataError`. The column has no coefficient, so the prediction assumes that the column has no effect |

A `GLM` drops a design column that is constant, or a combination of other columns, on its fit
rows. G-computation refuses the fit when a row at risk departs from that relation, because no
follower has that history. `test_a_prediction_through_an_aliased_column_is_refused` checks it.

No check finds other positivity failures. `preflight_mechanism_support` checks only that each
treatment level occurs among the rows at risk. Where no unit follows the regimen, the outcome
`GLM` fits extrapolate, and the interval holds only if they are correct there.

A diagnostic mechanism that cannot fit, for example under separation, does not stop a
g-computation fit. The fit warns with `PositivityWarning`, and the diagnostics report `nan`.

## Validation

| witness | what it checks | test |
| --- | --- | --- |
| W1, W2, W4 | each estimator equals the g-formula on exact binary, survival, competing-risk, three-arm and policy laws | `test_w1_w2_w4_every_estimate_is_the_g_formula` |
| W3 | a point-mass policy gives the regimen's estimate and curve | `test_w3_the_policy_path_on_a_point_mass_is_the_regimen` |
| W7 | the IPW mechanism term and each g-computation chain term move the standard error | `test_w7_each_stacked_term_moves_the_standard_error` |
| W6, bounds | the curve is the weight derivative where `g_bounds` binds, and where the Gaussian clip binds | `test_w6_holds_where_g_bounds_bind`, `test_w6_holds_where_the_gaussian_clip_binds` |
| W5, W6, three arms | the multinomial mechanism term, a known rule and a known policy, off a saturated law | `test_w5_w6_hold_on_multinomial_nodes` |
| scale | the Horvitz-Thompson estimate does not move with `q_bounds=` or with the outcome of a unit that does not follow the regimen | `test_horvitz_thompson_reads_the_outcome_scale` |
| status | a data-adaptive pseudo-outcome learner gives every cell the diagnostic curve | `test_the_status_and_every_cell_curve_agree` |

The registered study `sequential-baselines` adds repeated-sampling evidence. It is Python property
evidence. The [method-evidence page](method-evidence/sequential-baselines.md) reports 27 of 27
primary tests against known truth and 159 of 162 property cells. The three red cells are a declared
negative control on a law where every main-terms `GLM` is wrong. The R comparator parity is
fixture-level, not a registered comparator study. The tests `test_sequential_baselines_ltmle_r.py`
(R `ltmle`, witness W9) and `test_sequential_baselines_stremr.py` (R `stremr`) compare the estimate on
one panel each.
