# Conditional effects

## What this solves

`ConditionalContrast` reports how a contrast of two treatment arms varies with a declared set of
baseline modifiers $V \subseteq W$. With a finite basis $\varphi(V)$ declared before the fit, the
target is the projection of the conditional contrast onto $\operatorname{span}(\varphi)$, a
coefficient vector of fixed dimension with an efficient influence curve. TMLE targets it, and
the coefficients, the curve $\varphi(v)^\top\beta$ at declared points, a simultaneous band over
a declared grid, and a Wald test of the non-constant terms all carry intervals.

| output | interval | why |
| --- | --- | --- |
| projection coefficients on either scale, with or without missing outcomes | Wald, with the joint covariance | a pathwise-differentiable parameter of fixed dimension whose remainder is a product of nuisance errors |
| heterogeneity test | joint Wald chi-square, df the number of non-constant terms | a fixed-dimension stack of the coefficients |
| curve $\varphi(v)^\top\beta$ at declared points; band over a declared finite grid | pointwise Wald; multiplier band | linear in $\beta$; the shipped multiplier band over a fixed finite set |
| risk difference and calibration of a fixed external score (`assess_score`) | Wald | the efficient influence curve of the risk at a fixed function; the calibration is a projection onto $(1, s(V))$ |
| the conditional effect itself, with no basis (`model=None`) | refused | not pathwise differentiable |

The [user guide page](../user-guide/conditional-effects.md) gives the workflow.

## Estimands

For a binary treatment, or the declared pair $t, r$ of a multi-valued one, write
$\bar Q(a, W) = E[Y \mid A = a, W]$ and $\tau(W) = \bar Q(t, W) - \bar Q(r, W)$. The modifiers
are a subset $V$ of the adjustment columns $W$, and $\varphi(V) \in \mathbb{R}^p$ is a known
basis.

**CATE projection** (`scale="difference"`, reported as `cate_projection[term]`). With
$M = E[\varphi(V)\varphi(V)^\top]$ invertible,

$$
\beta_0 = M^{-1} E[\varphi(V)\,\tau(W)] = \arg\min_b E\big[\{E[\tau(W)\mid V] - \varphi(V)^\top b\}^2\big],
$$

the best linear projection of $E[\tau(W) \mid V]$ onto the basis. The working model need not be
correct. With $\varphi = (1)$ it is the ATE.

**log-CRR projection** (`scale="log_ratio"`, reported as `log_crr_projection[term]`).
$\beta_0$ solves

$$
U(\beta) = E\big[\varphi(V)\{(\bar Q_t + \bar Q_r)\,\operatorname{expit}(\varphi(V)^\top\beta) - \bar Q_t\}\big] = 0,
$$

the minimizer over $\{\varphi^\top b\}$ of the log-CRR risk
$E[(\bar Q_t + \bar Q_r)\log(1 + e^{\theta}) - \bar Q_t\,\theta]$, whose minimizer over all
functions of $V$ is $\log E[\bar Q_t \mid V] - \log E[\bar Q_r \mid V]$ (van der Laan, Carone and
Luedtke, Section 2.2, Example 2). The risk is strictly convex (their Lemma 16). With
$\varphi = (1)$, $\beta = \log(E\bar Q_t / E\bar Q_r)$, the marginal log risk ratio. It needs
$\bar Q(a, W) > 0$ for both arms.

On a continuous outcome with `q_bounds=(a, b)` the difference scale is reported on the outcome's
scale, $b - a$ times the contrast of the scaled outcome. The log ratio needs $a = 0$, because a
ratio of means is not invariant to a shift; `scale="log_ratio"` with a lower bound other than 0
is refused before any fit.

## Identification

The conditions are the shipped point-treatment ones: consistency, no interference, no unmeasured
confounding given $W$, and positivity for the two compared arms. Then
$E[Y^t - Y^r \mid V] = E[\tau(W) \mid V]$ and $E[Y^a \mid V] = E[\bar Q(a, W) \mid V]$, which
holds only when $V$ is part of $W$. A modifier outside the adjustment set is therefore refused:
adding it to $W$ is a different adjustment set, and it can break no unmeasured confounding. With
missing outcomes, add missingness at random given $(A, W)$ and $\Pi(a, W) > 0$, the shipped
missing-outcome conditions; $\bar Q$ is then the regression among observed rows. The source
states no identification list. `_CONDITIONAL_CONTRAST_ID` in
[`targets/builtin.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/targets/builtin.py)
records these conditions, with a known basis, an invertible $M$, a bounded outcome and strong
positivity.

## Influence curves

Write $s(A) = 1\{A = t\} - 1\{A = r\}$, which is 0 on a third arm, and

$$
\tilde D = \frac{s(A)\,\Delta}{g(A \mid W)\,\Pi(A, W)}\{Y - \bar Q(A, W)\} + \tau(W),
$$

with $\Delta = \Pi = 1$ when every outcome is observed.

**CATE projection.** $E[\varphi\tau]$ is a mean with the known weight $\varphi(V)$, so its
efficient influence curve is $\varphi\tilde D - E[\varphi\tau]$. $M$ is a mean of a known
function of $V$. The delta method on $(M, m) \mapsto M^{-1}m$ gives

$$
\mathrm{IC}_\beta = M^{-1}\Big[\varphi(V)\{\tau(W) - \varphi(V)^\top\beta\} + \varphi(V)\,\frac{s(A)\,\Delta}{g(A \mid W)\,\Pi(A, W)}\{Y - \bar Q(A, W)\}\Big].
$$

Van der Laan, Carone and Luedtke, Section 4, Example 1, identify this estimator, the debiasing
regression with a linear class of fixed size, as a TMLE of the best linear projection.

**log-CRR projection.** $U$ is linear in $\bar Q$. With $\eta = \varphi(V)^\top\beta$, its
influence curve at fixed $\beta$ is

$$
\mathrm{IC}_U = \varphi\{(\bar Q_t + \bar Q_r)\operatorname{expit}(\eta) - \bar Q_t\}
+ \varphi\,\frac{\Delta}{g(A \mid W)\,\Pi(A, W)}\big[1\{A = r\}\operatorname{expit}(\eta) - 1\{A = t\}\{1 - \operatorname{expit}(\eta)\}\big]\{Y - \bar Q(A, W)\} - U,
$$

and the implicit-function step gives $\mathrm{IC}_\beta = -J^{-1}\mathrm{IC}_U$ with
$J = E[\varphi\varphi^\top(\bar Q_t + \bar Q_r)\operatorname{expit}(\eta)\{1 - \operatorname{expit}(\eta)\}]$.

Both curves enter the engine through the MSM machinery: the clever covariate carries one column
per basis term, the fluctuation solves one score equation per coefficient, and on the log-ratio
scale the fluctuation and the projection alternate until both equations hold.

## Remainders

With cross-fitted nuisances and the pooled fluctuation, the CATE projection's second-order
remainder is

$$
M^{-1} P_0\Big[\varphi(V) \sum_{a \in \{t, r\}} s(a)\,\frac{(g_0\Pi_0 - g_n\Pi_n)(a, W)}{(g_n\Pi_n)(a, W)}\,(\bar Q_0 - \bar Q_n^*)(a, W)\Big],
$$

a product of the two nuisance errors. The submodel's features $s(A)\varphi(V)$ are fixed and
known, so the estimator does not regress on an estimated index, and the remainder is not of the
linear-remainder form. It is identically 0 with a known treatment mechanism and no missing
outcomes. The log-CRR remainder at fixed $\beta$ is the same product with weight
$\varphi(V)\{\operatorname{expit}(\eta) - 1\{a = t\}\}$, plus the $O(\lVert\beta_n - \beta_0\rVert^2)$
term of a Z-estimator. The interval holds when both nuisances converge at the product rate. One
consistent nuisance gives a consistent point estimate, not a valid interval.

## Curve, band and heterogeneity test

The curve at a declared point is $\varphi(v)^\top\hat\beta$, with standard error
$\{\varphi(v)^\top\hat\Sigma\varphi(v)\}^{1/2}$ from the joint covariance. Over a declared finite
grid $v_1, \dots, v_K$ the band is the shipped multiplier band of the curves
$\varphi(v_k)^\top\mathrm{IC}_\beta$, Rademacher by default. A conditional multiplier central limit theorem for a fixed number of jointly normal estimates
makes it valid over a fixed finite set, as the
[reference list](../references.md#point-treatment-and-stochastic-interventions) records for the
shipped band. A point
outside a declared domain is refused unless `Modifiers(outside="clip")` maps it to the nearest
end, for the fit and for prediction alike.

`heterogeneity_test()` tests $\beta_j = 0$ for every term except the intercept, a Wald
chi-square on as many degrees of freedom. It tests the absence of variation along the basis, not
the absence of heterogeneity.

## External scores

`assess_score` reads a column $s(V)$ computed from declared adjustment columns and estimates,
from the outcome and treatment fits of an ATE fit on the same study, two one-step functionals.

The **risk difference** compares the risk of the score with that of the best constant. On the
difference scale $R(s) = E[s^2 - 2s\tau]$, the best constant is the ATE with $R = -\mathrm{ATE}^2$,
and

$$
\mathrm{IC} = \{s^2 - 2s\tilde D - R(s)\} + 2\,\mathrm{ATE}\,(\tilde D - \mathrm{ATE}).
$$

The influence curve of the risk at a fixed function is van der Laan, Carone and Luedtke,
Theorem 1, as Section 3.1 specializes it. The derivative of $R(c)$ in $c$ vanishes at the best
constant, so estimating the constant adds no term. On the log-ratio scale, with
$L(\theta) = (\bar Q_t + \bar Q_r)\log(1 + e^\theta) - \bar Q_t\theta$ and its augmentation
$\Delta_L(\theta)$, the curve is $[L(s) + \Delta_L(s) - R(s)] - [L(c) + \Delta_L(c) - R(c)]$ with
$c = \log(E\bar Q_t / E\bar Q_r)$ held fixed.

The **calibration** is the projection of the contrast onto $(1, s(V))$, the CATE or log-CRR
projection with that basis. Both intervals need $s$ to be a fixed function, fitted on other
rows. A score fitted on this study's outcomes or treatments is an estimated index, and the
projection onto it has a remainder the interval ignores. `score_kind="external"` is the
declaration the interval rests on. A column equal to an in-sample curve of the same identified
effect, or named with the `effect_curve:` prefix, is refused, and that detection is best effort.

## Weights, clusters and missing outcomes

Weights define the weighted law, as for every point target. With `PointTreatment(cluster=...)` the cluster is the unit:
the variance sums the influence curve within clusters, with the $G/(G - 1)$ factor of the shipped
cluster-robust variance, and folds are grouped by cluster. With missing outcomes the clever
covariate carries $\Delta/\Pi$, and a cross-fitted fit meets the
[arm-indexed contract](scope-and-refusals.md#missing-outcome-arm-indexed-contract).

## Assessing a curve

`EffectCurveResult.diagnostics` reads the producing fit for `support()`, `nuisance_models()`
and `score_equations()`, and refits the producing estimator for four point readings with no
interval. The out-of-fold refits run on folds of independent units (clusters kept whole), and
each held-out prediction is compared with the doubly robust pseudo-outcome
$\tilde D = D_t - D_r$ of the producing fit's cross-fitted nuisances, which never saw that row.

- `calibration()`: the least-squares line of $\tilde D$ on $(1, \hat\tau_{-k})$, and the bin
  means of both over quantile bins of $\hat\tau_{-k}$. On the log-ratio scale the line
  minimises the one-step loss $(D_t + D_r)\log(1 + e^\eta) - D_t\eta$ and a bin reports
  $\log(\bar D_t / \bar D_r)$.
- `risk_comparison()`: the out-of-fold risk $P_n[s^2 - 2 s\tilde D]$ (on the log-ratio scale
  the one-step loss) of the curve, the training-fold constant, the T-learner and, for a
  difference, the DR-learner on the same basis. No log-ratio DR-learner exists to compare.
- `stability()`: the mean across rows of the SD of the refitted curves over 5 split seeds,
  over the SD of the curve; above 0.5 it warns. A cosine basis is also refitted as a B-spline
  of the same size, and the reverse.
- `modifier_stability()`: the out-of-fold risk with each modifier dropped from the basis.

`corrections`, `truncation_curve` and `refute` are declared unavailable, the last with R9.
`tests/unit/test_curve_assessment.py` checks the rows, that a held-out fold's predictions do
not move when its own outcomes are flipped (and that a refit on every row does move them), and
each reading against its formula.

## Refusals

| request | reason | refusal |
| --- | --- | --- |
| `model=None` with `method="tmle"` | the conditional effect is not pathwise differentiable | `invalid_declaration` |
| a basis that needs a domain or levels that were not declared, `outside="clip"` without a domain, or a user basis not declared known | a basis read off the sample is a functional of the observed law, and the influence curve has no term for it | `invalid_declaration` |
| `Sieve(basis)` with its size chosen by cross-validation | no reviewed result covers inference after the selection | `invalid_declaration` |
| `scale="log_ratio"` with `q_bounds` not starting at 0 | a ratio of means is not invariant to a shift | `invalid_declaration` |
| a modifier outside the adjustment set | the contrast given $V$ is identified only when $V \subseteq W$ | `invalid_declaration` |
| missing treatments, an intermediate, a continuous or vector treatment, `strata=`, `repeats` above 1, the full-refit bootstrap | the projection's clever covariate, covariance or study cell for the composition is not built | `implementation_gap`, roadmap P1 |
| a sensitivity analysis or refutation of a projection fit | each is defined for a mean or a contrast of means | `implementation_gap`, roadmap P1 |
| `heterogeneity_test()` on a basis without an intercept column | its coefficients being zero says the contrast is zero, not that it is constant | `invalid_declaration` |
| `method="ep"` | the EP learner is not shipped; for any other estimand it estimates a different target | `implementation_gap`; `different_estimand` |
| `assess_score` without `score_kind="external"`, with `computed_from` outside the adjustment set, or on an in-sample curve | the interval needs a fixed function of adjustment columns | `invalid_declaration` |

Each one is raised before any learner is fitted, and `tests/unit/test_conditional_contrast.py`
pins its text.

## Status and evidence

No support row covers these fits yet, so every projection fit reads `experimental`. The
`conditional-projection` study, declared in `tests/studies/conditional_projection.py` with its
verdicts in `tests/studies/conditional_projection_properties.py`, has not run. Its declared run
is `python -m tests.canonical.conditional_projection.regenerate`, whose docstring holds the
red-cell reading. The declaration fixes, before the run:

| family | groups | rule |
| --- | --- | --- |
| `interval_calibration` | `trial_2k`, `obs_1k`, `obs_4k`, `crr_1k`, `crr_4k`, `multi_4k`, `clustered_4k`, `obs_missing_4k`, `obs_missing_crr_4k`, `assess_score_2k`, the three ladder rungs (R = 4,800), and `weighted_sl_2k` with the ensemble learner and 5 outer folds (R = 3,600) | 99% bootstrap SE-ratio interval in [0.93, 1.07] and 99% Clopper-Pearson coverage interval in [0.92, 0.98] |
| `bias` | `slow_1k`, `slow_4k`, `slow_16k` (both nuisances off by $5 n^{-1/3}$ on the logit scale); `noweight_dr_4k` and `missing_dr_4k` (a declared wrong outcome regression) | 99% interval of the standardized bias inside $\pm 0.25$ |
| `type_i_error`, `power` | the df-2 heterogeneity test at slope 0 ($n = 2{,}000$, R = 2,400) and 0.10 ($n = 4{,}000$, R = 1,000; power 0.933, noncentrality 14.27) | rejection upper end at most 0.10; lower end at least 0.80 |

Each control shares its group's draws and must fail: the row-level SE on `clustered_4k` falls
below the SE-ratio band; the T-learner (the projection of $\bar Q_1 - \bar Q_0$), the IPW
projection, the one-step with unit weights in place of $1/g$, the one-step without $1/\Pi$,
and the fit that drops the sampling weights each have a bias interval wholly outside
$\pm 0.25$ of their own spread. `tests/unit/test_conditional_projection_design.py` computes
each calibration row's pass probability (above 0.999 at a calibrated SE, 0.99 for the family),
the power and type-I pass probabilities, and each control's detection probability (above 0.99),
from the laws by quadrature. The EP cell of the plan moves to the part (b) study.

The fast-tier instruments are:

- exact-law point and influence-curve checks against a Gâteaux derivative on the binary,
  missing-outcome, three-arm and weighted discrete laws, with clusters, both scales, and double
  robustness of the point under a wrong outcome regression
  (`tests/unit/test_effect_modification_exact.py`);
- the identity MSM `[1, a, W, aW]` cross-check, and the constant basis against the ATE and the
  log risk ratio;
- nonzero witnesses of the residual term, and mutation controls that the exact checks must
  fail: the arm sign replaced by its absolute value, a third arm given a nonzero weight, and the
  reference arm dropped from the log-ratio curve;
- the heterogeneity statistic on a two-modifier exact law, equal to the exact noncentrality on 2
  degrees of freedom, with clusters summed within each cluster, and the intercept-free basis
  refused because its test would reject a constant effect;
- the row weights reaching each candidate of the ensemble in every outer and inner fold, by a
  spy, with the weightless member fit as a mutation that must fail;
- Gâteaux checks of the score risk on both scales, on the complete and the missing-outcome law, with the risk curve that omits the term for the
  constant as a mutation that must fail, and the true CATE calibrated to intercept 0 and slope 1
  (`tests/unit/test_conditional_contrast.py`).

The [evidence manifest](evidence.md#the-table) lists them. The source is
van der Laan, Carone and Luedtke (2024; v2 dated 2026), arXiv:2402.01972v2, a preprint; see the
[reference list](../references.md#conditional-effects).
