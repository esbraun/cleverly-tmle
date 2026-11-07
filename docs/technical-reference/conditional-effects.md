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
| the conditional effect itself, with no basis (`model=None`), by `method="ep"` | none: `predict(interval=True)` refuses | not pathwise differentiable; the EP learner's theorems give rates, not a limit law |

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

## The EP learner

`method="ep"` (`EPMethod`, `cleverly/estimators/ep.py`) is the efficient plug-in learner of
van der Laan, Carone and Luedtke (arXiv:2402.01972v2, "EPL" below). It runs in three steps.

1. **Nuisances.** The cross-fitted $\bar Q_n(a, W)$, $g_n(a \mid W)$ and, with missing
   outcomes, $\Pi_n(a, W)$ of the shipped point fit, each row with its own out-of-fold fit,
   $g$ truncated at the resolved `g_bounds` and $\bar Q$ shrunk into the open unit interval
   on the scale `q_bounds` declares.
2. **Debiasing** (EPL Algorithm 2 and App C.1, Method 1). One pooled logistic regression of
   the scaled outcome on features $x$, with offset $\operatorname{logit}\bar Q_n(A, W)$ and
   weight $\Delta w / \{g_n(A \mid W)\Pi_n(A, W)\}$. On the difference scale
   $x = (1\{A = t\} - 1\{A = r\})\varphi_k(V)$; on the log-ratio scale
   $x = (1\{A \in \{t, r\}\}\varphi_k, 1\{A = t\}\varphi_k)$, which span the blocks
   $H_1 = 1$ and $H_2 = -a$ of EPL eq (3). Its score is zero at the fit, which is EPL's
   equations (6), and the updated $\bar Q^*$ stays inside $(0, 1)$. Rows of a third arm carry
   no feature.
3. **Final stage** (EPL Algorithm 3). On the difference scale, least squares of
   $\bar Q^*(t, W) - \bar Q^*(r, W)$ on $\varphi_k(V)$, times $b - a$ of the declared outcome
   range, or a declared `final_learner` on the modifier columns. On the log-ratio scale, the
   logistic regression of $\bar Q^*_t / (\bar Q^*_t + \bar Q^*_r)$ on $\varphi_k$ with weight
   $\bar Q^*_t + \bar Q^*_r$.

The sieve is `Sieve(basis, size=...)`, by default `Sieve.default(modifiers)`: additive cosines
of each continuous modifier and indicators of each categorical one, up to 6 terms per
component. A fixed size, or a schedule $n \mapsto k(n)$ declared before the fit, is used as
given. `size="cv"` chooses $k$ by EPL Algorithm 4: the cross-validated one-step risk
$P_n[\theta^2 - 2\theta(D_t - D_r)]$, or $P_n[(D_t + D_r)\log(1 + e^\theta) - D_t\theta]$ on
the log-ratio scale. Each training fold refits the nuisances by an inner cross-fit of 5 folds
(clusters kept whole) before it debiases and fits the curve at each size, and the held-out
fold's risk uses the outer out-of-fold pseudo-outcomes. No nuisance and no curve that is
evaluated on a fold saw that fold, which is EPL's own remedy for the leakage it notes after
Algorithm 4. The result records the sizes, the risks, the chosen size and the inner seeds.

What the theory claims, by configuration:

| configuration | claim |
| --- | --- |
| difference, fixed size $k$, default final stage | the projection onto $\operatorname{span}(\varphi_k)$, the target of the TMLE projection: the EP coefficients solve both parts of its efficient estimating equation, $P_n\varphi\{(2A - 1)(Y - \bar Q^*)/g\} = 0$ and $P_n\varphi(\tau^* - \varphi^\top\beta) = 0$. Consistent if either nuisance is; no interval |
| difference, a declared schedule $k(n)$ | EPL Theorems 4 and 5 under their conditions C1 to C6, which the user must judge, for empirical risk minimisation over a fixed, uniformly bounded convex class $\mathcal F$ that meets C4. The default final stage is least squares over the growing, unbounded span of $\varphi_{k(n)}$, so the user must also judge that it behaves as such a class. Then: a rate, and $\lVert\theta^*_n - \theta_{n0}\rVert = o_p(E\lVert\theta_{n0} - \theta_0\rVert)$ for the oracle DR-learner $\theta_{n0}$ |
| log ratio, a declared schedule $k(n)$ | EPL Theorems 4 and 5 under C1 to C6 for the log-CRR risk of eq (3), whose population risk is strictly convex (Lemma 16), with the same caveat on the final stage, a weighted logistic regression over the span of $\varphi_{k(n)}$. The one-step empirical risk need not be convex (Section 3.2) |
| `size="cv"` (the default) | none: EPL proves no theorem for the selected size (Section 5.1) |
| a declared `final_learner` | none: the theorems cover least squares over a declared convex class (Section 7.1) |
| log ratio, fixed size | the log-CRR projection's target, but the two-block debiasing does not solve that projection's efficient score $\varphi\{\operatorname{expit}(\eta) - 1\{A = t\}\}$, so no root-$n$ claim |
| any EP curve | no interval: `predict(interval=True)` raises the R1 refusal |

With `model=None` on the difference scale the curve is clipped to $[-(b - a), b - a]$, where
the target lies, and the result records the clipped share. With `model=` declared the target is
that projection and nothing is clipped.

An EP fit reads `supported` only in a composition a
registered study measured (`EP_MEASURED` in `cleverly/estimators/ep.py`): a declared schedule
with `model=None` on either scale (`ep-cate`, `ep-crr`), or a fixed size with `model=` declared
on the difference scale (`ep-cate`'s `ep_fixed_obs_4k` cells), each with one outcome regression
per arm, two arms, an estimated mechanism and no weights, clusters or missing outcomes. The
label carries no interval. A size chosen by cross-validation (the default) or a declared
`final_learner` lies outside EPL's theorems and stays `experimental` after any study. So
do a pooled outcome regression, a fixed size with `model=None`, the log-ratio fixed size (its
debiasing does not solve the projection's efficient score), more than two arms and declared
treatment probabilities. Weights, clusters and missing outcomes are implemented (the weighted
law, the cluster as the unit with grouped folds, the $\Delta/(g\Pi)$ weight) without a study
and read `experimental`. `diagnostics.score_equations()` reports each debiasing feature's weighted mean
score before and after the regression.

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

## Targeting curves

`IdentifiedEffect.targeting_curve(column, computed_from=..., score_kind="external")` asks how
much treating by a priority score $S = s(W)$ first gains over treating at random. With $u$ the
rank position by decreasing $S$ (ties broken by an independent uniform, so $u \sim U(0, 1)$),
it estimates, on the difference scale whatever the scale of $S$:

| quantity | definition | reference |
| --- | --- | --- |
| $\mathrm{TOC}(q)$ | $E[\tau \mid u \le q] - \mathrm{ATE}$ at each grid point; $\mathrm{TOC}(1) = 0$ | 0 |
| $G(q)$ | $q\,E[\tau \mid u \le q]$, so $G(q) - q\,\mathrm{ATE} = q\,\mathrm{TOC}(q)$ | $q\,\mathrm{ATE}$ |
| AUTOC | $\int_0^1 \mathrm{TOC}(q)\,dq$ | 0 |
| Qini | $\int_0^1 q\,\mathrm{TOC}(q)\,dq$ | 0 |
| group effect | $(G(q_k) - G(q_{k-1})) / (q_k - q_{k-1})$ for group edges on the grid | the ATE |

These are Yadlowsky et al. (2025), Definition 2 and Examples 2 and 3 (arXiv:2111.07966v2). In
the $n$ evaluation rows each is a weighted mean $n^{-1}\sum_i c(i)\Gamma_i$ of doubly robust
scores: $m_q(i)/q - 1$ for the TOC, with $m_q$ the row's fractional membership of the top
fraction ($\sum_i m_q(i) = nq$ exactly); $H_n - H_{k-1} - 1$ for the AUTOC and
$(n + 1 - 2k)/(2n)$ for the Qini at rank $k$, which `tests/unit/test_targeting_curves.py`
checks against the double sums of (2.9). A tied block takes the mean of its weights, which is
the average over all orderings within it (the 720 orderings of a six-row example are checked).

**The TMLE.** The weight columns of every score, with the ATE column, are reduced to an
orthonormal basis of their span by a thin SVD (singular values above $10^{-10}$ times the
largest are kept), so a score with few distinct values does not make the Newton Hessian
singular. One pooled fluctuation of the ATE fit's cross-fitted outcome regression runs along
that basis, with clever covariate $b(W)(1\{A = t\} - 1\{A = r\})/(g\Pi)$ and the fit's own
truncated $g$. Every reported value is the substitution estimate
$n^{-1}\sum_i c(i)(\bar Q^*_t - \bar Q^*_r)(W_i)$, and solving the score equations makes it equal
$n^{-1}\sum_i c(i)\Gamma^*_i$, with $\Gamma^*$ the AIPW score built from $\bar Q^*$; the unit test
checks the equality to $10^{-8}$ under a tight targeting tolerance, and the shipped stopping
rule bounds the gap by its own tolerance. On a binary outcome the substitution estimate keeps
the top-fraction mean in $[-1, 1]$, which the one-step score form can leave. `TargetingCurve.ate`
is the fluctuation's own ATE column, so $\mathrm{TOC}(q)$ is the top-fraction mean minus it
exactly.

**Inference.** The standard error is the SD of $B$ half-sample draws (2,000 by default) of
$\lfloor n/2 \rfloor$ rows, or whole clusters, without replacement and with no rescaling; the
interval is $\hat\theta \pm z_{1-\alpha/2}\,\sigma^*$ (Lemma 4, Corollary 5). Each draw keeps
$\Gamma^*$ fixed and recomputes the ranks, cuts, tie blocks and weights on its rows; the
nuisances are not refit. The areas are the published results (Theorem 3, Corollary 7 for the
Qini, Proposition 8 for the AUTOC). The rest rests on four natural extensions
([verdicts](natural-extension-verdicts.md)):

- D1, the TOC at a fixed $q$: asymptotically linear with influence function
  $1\{S \ge s_q\}\Gamma/q - E[\tau \mid S \ge s_q] - \mu(s_q)(1\{S \ge s_q\} - q)/q - (\Gamma - \mathrm{ATE})$,
  $\mu(s) = E[\tau \mid S = s]$, when $S$ has no atom at its $1 - q$ quantile $s_q$, that
  quantile is unique, $\mu$ is continuous at $s_q$ and Theorem 3's Assumption C holds. Only
  the first condition can be checked in the sample (the `tied_cut` column), approximately.
- D2, the sup-t band over the grid points below 1: simultaneous coverage at the grid points,
  not over continuous $q$, with the critical value of `studentized_critical_value`. It is
  withheld when any point's SE is 0 or withheld.
- D3, ties: a cut strictly inside an atom is the delta method on five sample means, and a cut
  at an atom's edge has a non-normal limit. A cut inside a sample tied block, or exactly on the
  boundary next to one (`tie_share` is then NaN), sets `tied_cut` and reads `experimental`; an
  atom edge the sample does not show cannot be detected. Areas on a score with finitely many values are smooth functions of block
  masses and sums. Areas on a continuous score with atoms (a block holding at least 1% of the
  rows, and more than $n/10$ distinct values) rest on the source's unproved statement and read `experimental`.
- D4, bootstrapping $\Gamma^*$: the targeting update is a fixed-dimension nuisance. With an
  inconsistent $\bar Q$ this holds only under a known mechanism (`treatment_probabilities=`),
  where the TMLE and the one-step have different influence functions and the draws carry the
  TMLE's. On the study's cell K (a main-terms outcome regression with the known mechanism) the
  TMLE's AUTOC draws are 1.25 to 1.3 times as wide as the one-step's, while at TOC(0.5) and the
  Qini the two agree to 2% (`k_untargeted_ratio` in `tests/studies/targeting_curves.py`).

The remainder is not of the linear-remainder form: $S$ is fixed given the evaluation rows, and
the cut's first-order term is part of $\psi$ and is redrawn with each draw.

**The bootstrap at the study's sizes.** With the true nuisances, so that only the half-sample
SE is measured, its ratio to the estimator's SD lies between 0.978 and 1.027 for every row the
study gates at $n = 2{,}000$, on 6,000 to 30,000 replicates per cell
(`tests/studies/targeting_curves_oracle.csv`). The AUTOC of cell H reads 1.002 (SE 0.004).
The exception is the paired difference of two correlated scores at TOC(0.1): 1.063 at
$n = 2{,}000$, 1.051 at 4,000 and 1.024 at 8,000. That interval is conservative by a
finite-sample amount that shrinks with $n$, so the study reports the row and does not gate it.

**Routes.** Route 1 takes a study column fitted on other rows, with the three checks of
`assess_score` on it and on a `compare=` column. Route 2,
`curve.diagnostics.targeting_curve(split=0.5)`, refits the producing estimator on a share of
the units, scores the other rows with that refit (kept as `TargetingCurve.score_curve`) and
fits the ATE on them only; the curve is that of the refit, not of the curve fitted on all rows.
A paired `compare=` column shares the draws, and the SE of a difference is the SD of the
per-draw differences.

**Guards.** A grid point with fewer than 50 rows in its top fraction (`TARGETING_MIN_TOP_ROWS`,
a declared guard with no source) keeps its point and withholds its interval and the band; 50 to
99 rows read `experimental`. Weights, clusters, missing outcomes and a multi-arm pair are
computed and read `experimental`; fewer than 40 clusters withhold the intervals. A row whose
half-sample draws do not vary (SE at most $10^{-12}$, as for every TOC of a score that ties all
rows) withholds its interval rather than report a width of rounding. With weights, every rank
weight reads the row mass normalised to mean one, so the weighted AUTOC is defined in those
units. The status reads the configuration only: the learners, the number of folds and $n$ are
not checked against the study's cells, and an area row carries no size guard of its own.

**Status.** The `targeting-curves` study is declared in `tests/studies/targeting_curves.py`
and has not run, so every curve reads `experimental` with that reason until it registers. It
runs 4,800 replicates per cell, the smallest multiple of 1,200 at which every gated row passes
together with probability 0.80 at its measured bootstrap ratio. The
study's cells are the measured configurations: a continuous and a binary outcome, an atomic
score, a known mechanism with a misspecified outcome regression, a declared split, a paired
second score, and null laws for the band's size.

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
  over the SD of the curve; above 0.5 it warns. Every refit uses the same sample, so the
  reading measures sensitivity to the cross-fitting split, not sampling noise, and a low value
  does not mean the curve is well determined (the `ep-assessment` pilot reads 0.09 on a curve
  of noise at a constant effect). For that, read the heterogeneity test and the projection's
  intervals. A cosine basis is also refitted as a B-spline of the same size, and the reverse.
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
| `method="ep"` for an estimand other than `ConditionalContrast` | it estimates a different target | `different_estimand` |
| `predict(interval=True)` on an EP curve, or a non-default `Inference` on `EPMethod` | EPL proves rates, not a limit law for the curve | `inference_gap` |
| `method="ep"` with `model=Sieve(basis)` sized by cross-validation | the model is a declared projection target; pass the sieve as `sieve=` with `model=None` | `invalid_declaration` |
| `method="ep"` with both `model=` and `sieve=`, or `final_learner=` with `model=` | two bases for one fit; with a model the final stage is least squares on it | `invalid_declaration` |
| `final_learner=` on the log-ratio scale | the weighted logistic final stage with a declared learner is not built | `implementation_gap`, roadmap P1 |
| `EPMethod` with `cross_fit=False`, `repeats` above 1, `cv_evaluation=True`, a non-default targeting scheme, or a fluctuation setting | the debiasing pools every row with its own out-of-fold nuisances and fits no fluctuation | `invalid_declaration` |
| `assess_score` without `score_kind="external"`, with `computed_from` outside the adjustment set, or on an in-sample curve | the interval needs a fixed function of adjustment columns | `invalid_declaration` |
| `targeting_curve` under the same three conditions, for the score or the `compare=` column | as `assess_score` | `invalid_declaration` |
| `curve.diagnostics.targeting_curve(split=...)` with a split that is not a fraction in $(0, 1)$ | a fold average of fold-specific curves is a property of the learning algorithm; its only source (Imai and Li 2023, Section 4) covers difference-in-means scores in a randomized experiment | `invalid_declaration` |
| a targeting curve with `repeats` above 1 | a median over splits of a whole curve has no reviewed joint law | `inference_gap` |
| a targeting curve with `target_weights=True` or a declared stopping rule | every row loads several weight columns, which the weighted submodel cannot carry; a declared stopping rule needs a stopping context no curve builds | `implementation_gap`, roadmap X92 |
| `compare=` on route 2 of an engine fit | the column is read from the producing study, which an engine fit does not keep | `invalid_declaration` |

Each one is raised before any learner is fitted, and `tests/unit/test_conditional_contrast.py`
pins its text.

## Status and evidence

The registered [conditional-projection study](method-evidence/conditional-projection.md)
passed every one of its 64 property cells and its three primary tests. Its support rows `T49`
(the CATE projection, family `P13`) and `T50` (the log-CRR projection, `P14`) label a fit
`supported` when its key is one the study measured, with one outcome regression per arm
(`outcome_by_arm=True`):

| row | composition |
| --- | --- |
| `T49`, `T50` | binary outcome and treatment, iid rows, complete outcomes, stacked cross-fitting |
| `T49.known` | the same with `treatment_probabilities=` |
| `T49.arms3` | three arms |
| `T49.weighted` | sampling weights |
| `T49.clustered` | equal clusters |
| `T49.missing`, `T50.missing` | outcomes missing at random |

A pooled outcome regression on this axis, a composition no group measured (weights with
clusters, for example), the simultaneous band and the external-score check read `experimental`.
The study was declared in `tests/studies/conditional_projection.py` with its verdicts in
`tests/studies/conditional_projection_properties.py`, and its declared run is
`python -m tests.canonical.conditional_projection.regenerate`, whose docstring holds the
red-cell reading. The declaration fixed, before the run:

| family | groups | rule |
| --- | --- | --- |
| `interval_calibration` | `trial_2k`, `obs_1k`, `obs_4k`, `crr_1k`, `crr_4k`, `multi_4k`, `clustered_4k`, `obs_missing_4k`, `obs_missing_crr_4k`, `assess_score_2k` (R = 4,800), and `weighted_sl_2k` with the ensemble learner and 5 outer folds (R = 3,600) | 99% bootstrap SE-ratio interval in [0.93, 1.07] and 99% Clopper-Pearson coverage interval in [0.92, 0.98] |
| `bias` | `slow_1k`, `slow_4k`, `slow_16k` (both nuisances off by $5 n^{-1/3}$ on the logit scale; R = 4,800, slope over the rungs reported); `noweight_dr_4k` and `missing_dr_4k` (a declared wrong outcome regression) | 99% interval of the standardized bias inside $\pm 0.25$ |
| `type_i_error`, `power` | the df-2 heterogeneity test at slope 0 ($n = 2{,}000$, R = 2,400) and 0.10 ($n = 4{,}000$, R = 1,000; power 0.933, noncentrality 14.27) | rejection upper end at most 0.10; lower end at least 0.80 |

The ladder claims bias and not interval calibration. With both nuisances wrong on purpose, the
plug-in variance overstates the spread at 1k (the re-review's pilot read SE ratios up to about
1.06), so the band would read a conservative interval as a defect. Its coverage is published
on the bias rows without a gate.

Each control shares its group's draws and must fail: the row-level SE of the intercept and the
`W1` slope on `clustered_4k`, where `W1` and the treatment are cluster level, falls below the
SE-ratio band. The `W3` slope varies within clusters, so its row-level SE is close to the
clustered one and carries no control row. The T-learner (the projection of
$\bar Q_1 - \bar Q_0$), the IPW projection, the one-step with unit weights in place of $1/g$,
the one-step without $1/\Pi$, and the spline-GLM fit that drops the sampling weights each have
a bias interval wholly outside $\pm 0.25$ of their own spread.
`tests/unit/test_conditional_projection_design.py` computes the power and type-I pass
probabilities and each control's detection probability (above 0.99) from the laws by quadrature.
The plan's fixed-size EP cell, `ep_fixed_obs_4k`, moved to the `ep-cate` study.

The calibration rows' pass probabilities are read at SE ratios from a pilot on its own seed
stream, committed as `tests/studies/conditional_projection_pilot.csv`: 2,400 fits of each GLM
group and 960 of `weighted_sl_2k`. At a calibrated SE the family of 33 rows passes with
probability 0.996. The piloted ratios of each learner spread no more than their noise
(Cochran's $Q$, $p = 0.79$ for the GLM rows and 0.86 for the ensemble rows), around common
ratios of 1.002 and 1.012. At those common ratios the family passes with probability 0.962.
Read row by row, the pilot's own noise counts as miscalibration and the family rate falls to
0.78, mostly from the three ensemble rows. The pilot also gives the row-level SE ratios on
`clustered_4k`: 0.78 and 0.77 for the intercept and `W1`, and 0.97 for `W3`.

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

The [`ep-cate` study](method-evidence/ep-cate.md), declared in `tests/studies/ep_cate.py` with
its verdicts in `tests/studies/ep_cate_properties.py` and run by
`python -m tests.canonical.ep_cate.regenerate`, passed every primary test and all 21 gated
cells. At 4,000 rows the upper bounds of the oracle comparison run from 0.033 to 0.076 and every
growth bound is at most 0.065. On the rescue laws EP reads 0.67 and 0.72 (upper bounds 0.72 and
0.77) while its T-learner reads 3.08 and 2.94. The fixed-size coefficients have standardized
biases of at most 0.030.

Its laws are hte3's code at `eac64673` (the d = 3 generator differs from EPL App B.2: noise SD 1, `positivity=TRUE` is the limited overlap,
and the simple CATE under moderate overlap is $1 + W_1$), with the noise truncated at $\pm 4$.
The two rescue laws are the complex d = 3 laws with App B.2's noise, SD 2 truncated at
$\pm 8$. Each gate reads the paired mean of
$\log \mathrm{ISE}(\text{EP}) - \log \mathrm{ISE}(\theta_{n0})$ over replicates, one-sided at
99%:

| gate | cells | rule |
| --- | --- | --- |
| oracle comparison (EPL Theorem 5) | every law but the rescue laws, the plan's ensemble, the schedule $k(n) = 3, 4, 5$ at $n = 1{,}000, 2{,}000, 4{,}000$ | upper bound at 4,000 below $\log 1.5$ |
| no growth | the same | upper bound of the change from 1,000 to 4,000 below $\log 1.25$ |
| DR rescue | the rescue laws, both overlaps, a per-arm main-terms linear outcome learner | upper bound at 4,000 below $\log 3$, and no growth |
| control that must fail | the T-learner at that learner | lower bound at 4,000 above $\log 3$ |
| fixed size, `ep_fixed_obs_4k` | the observational law of `conditional-projection`, `model=Linear(["W1", "W3"])`, R = 2,400 | each coefficient's standardized bias inside $\pm 0.25$ |

The rescue claim depends on the outcome noise. With a wrong outcome regression the EP curve
stays consistent when the propensity is right, but the residual $Y - \bar Q_n$ enters its
variance, and the oracle's does not carry it. When the outcome noise is small, that residual
dominates. In the pilot at hte3's noise SD of 1 the misspecified-learner EP has a mean log ratio
of about 1.45 at $n = 4{,}000$ under both overlaps, an ISE about four times the oracle's, against
$\log 3 = 1.10$. Under App B.2's SD of 2, the setting of the paper's rescue claim, it reads 0.71
(moderate overlap) and 0.64 (limited). The code-law reading stays in the study as an ungated
diagnostic row, and the gate runs at App B.2's noise (plan amendment rev 2.4).

The replicate count of each law comes from pilots on their own streams
(`tests/studies/ep_cate_pilot.csv`; the rescue laws' stream is separate) by the plan's sizing
rule, and `tests/unit/test_ep_cate_design.py` pins it: 300 for the d = 3 laws and the rescue
laws and 200 for the d = 20 laws. Each gate's pass probability is averaged over the pilot's
own uncertainty in its mean and spread, and every gate passes with probability at least 0.9;
the whole family passes with probability 0.93. A growth row red while its 1,000-row mean is
below 0 and the same estimator's 4,000-row gate is green is read before the run as EP
approaching the oracle from below, and is reported rather than treated as a defect; with that
reading the family is free of a defect suspect with probability 0.97. The 2,000-row d = 3 groups feed only the ungated rate report and run
100 replicates. The comparator, hte3 `Lrnr_cate_EP` with `targeting_style="dr"`, is declared
pending: its debiasing is the linear Method 2, so the two agree only to $o_p(n^{-1/2})$ and the
comparison is of different estimators.

The [`ep-crr` study](method-evidence/ep-crr.md) (`tests/studies/ep_crr.py`, run by
`python -m tests.canonical.ep_crr.regenerate`) passed every primary test and all 13 gated
cells, and every oracle Hessian was positive definite. At 4,000 rows EP's upper bounds run from
$-0.063$ to $-0.017$; the intercept-only learner's EP reads $-0.09$ against its T-learner's 0.92,
and the offset learner's EP 0.25 (upper bound 0.30) against 1.22.

Its laws are hte3's `generate_data_RR.R`: $W \sim U(-1, 1)^3$, a binary outcome with
$\operatorname{logit} \mu_0(0, w) = -1 + 0.3 \sum_k (w_k + \sin 4 w_k)$, the log CRR
$0.1(-1 + \sum_k w_k)$ (simple) or $0.1(-1 + \sum_k \{w_k + \sin 4w_k\})$ (complex), and the
two overlaps; $\mu(1, w)$ stays below 0.8, so nothing is clipped.

The `ep-crr` oracle minimises the
one-step log-CRR risk at the true nuisances on the same sieve. That risk is not convex in
general (EPL Section 3.2), so each replicate records whether the oracle's Hessian is positive
definite. A replicate where it is not is excluded from its group's comparisons, and a gated row
is red when more than 1% of its group's replicates were excluded.

The gates are as in
`ep-cate`, with the rescue margin at $\log 1.5$, on the complex law at limited overlap with two
wrong outcome learners, each with its T-learner as the control. The first ignores $W$. The
log-ratio debiasing features span a per-arm logistic model on the sieve and this law's
$\operatorname{logit} \mu_0$ is nearly additive, so even a debiasing without the $1/\pi$ weight
recovers the curve there: that cell shows that the debiasing corrects an intercept-only $\mu$,
and does not see the weight. The second is the true regression with the treated arm's logit
off by $2\{(\sum_k W_k)^2 - 1\}$, whose cross terms lie outside the additive sieve, so only the
weighted correction removes the error; its growth row is reported, not gated, because with
$\mu$ wrong the remainder is first order in the error of $\pi$ and no theorem bounds the
ratio's growth. No DR-learner is compared, since the ratio has no such pseudo-outcome.

In the `ep-crr` pilot
(`tests/studies/ep_crr_pilot.csv`) EP reads between $-0.08$ and $-0.02$ at 4,000 on every law,
and the intercept-learner EP $-0.08$ against its T-learner's $1.02$. A second pilot of 100
replicates at each end of the simple law at limited overlap
(`tests/studies/ep_crr_growth_pilot.csv`) reads $-0.14$ at 1,000 and $-0.08$ at 4,000. The
offset pilot (`tests/studies/ep_crr_offset_pilot.csv`, 31 replicates at each end) reads the
offset-learner EP at 0.25 (SD 0.34) at 4,000, its T-learner at 1.35, and EP with the weight
replaced by 1 at 1.72, so that mutation fails the rescue gate with probability above 0.999.
At R = 300 every gate passes with probability at least 0.95 averaged over the pilots'
uncertainty, and the family with probability 0.93. The test
`test_the_inverse_propensity_weight[log_ratio]` in `tests/unit/test_ep_exact.py` checks the
weight exactly on a small population. No oracle Hessian failed in 24,000 checks on the gated
laws and sizes (`tests/studies/ep_crr_hessian_pilot.csv`), so the 1% exclusion bound turns no
row red with probability above 0.9999 even at the 95% upper rate.

The [`ep-assessment` study](method-evidence/ep-assessment.md) (`tests/studies/ep_assessment.py`)
passed its three gated cells: the calibration slope reads 0.986 (99% interval 0.985 to 0.987),
the stretched control 0.657, and the risk difference has the right sign in all 300 strong-law
replicates. `relative_sd` reads 0.009 and 0.077, and no replicate warns. It reads the
diagnostics of an EP curve on the complex d = 3 law at moderate overlap and on a weak law with
the constant CATE 0.1, at $n = 2{,}000$ with the ensemble:

| reading | claim | control |
| --- | --- | --- |
| out-of-fold calibration slope, 300 replicates | the 99% interval of its mean inside $[0.85, 1.15]$ | the curve stretched by 1.5 about its mean on the same folds, whose slope is exactly $1/1.5$ of the curve's: its 99% upper bound below 0.85 |
| stability, `relative_sd` over 5 split seeds, 100 replicates per law | none: reported only, with the share of each law's replicates above 0.5 | none |
| risk comparison, EP against the constant | the share of complex-law replicates whose reported DR risk difference has the sign of its exact value at the true CATE, with the same bound | none; at the weak law the share is reported only |

The pilot (`tests/studies/ep_assessment_pilot.csv`) reads the slope at 0.987 (SD 0.005) and
the stretched slope at 0.658, so the calibration claim and its control pass with probability
1.0, as does the sign share. Stability is reported, not gated (plan amendment rev 2.4).
Refits under other split seeds reuse the same draw, and a curve's wiggle at a constant effect
comes from the draw, so `relative_sd` reads 0.088 at the weak law (0.009 at the complex law).
A claim that the weak law reads above 0.5 would fail at any R, and no control could make the
reading's warning fail.

The EP learner's fast-tier instruments are:

- the debiasing on finite-support populations with the curve indexed by part of $W$: the
  projection exact at a wrong outcome regression and at a wrong propensity, under missing
  outcomes and with a third arm, the saturated log ratio exact, and the unsaturated log ratio's
  gap reported (`tests/unit/test_ep_exact.py`);
- a mutation of each component that the exact checks must fail: the targeting sign, a dropped
  basis column, the contrast's direction, the third arm's feature, the offset, the $(2A - 1)$
  factor, the $1/g$ weight on both scales, the $1/\Pi$ weight, the log ratio's second block and
  its final-stage weight;
- the study layer at the oracle, the $b - a$ factor, the clip, the estimating equation against
  the TMLE projection, fold locality of the size selection (with a refit that sees its fold as
  the mutation), clusters kept whole in outer and inner folds, the row weights in every outer
  and inner outcome and treatment fit (with a fit that drops them as the mutation), and each
  refusal before any learner, a declared size above the family's largest member among them
  (`tests/unit/test_ep_contract.py`);
- frequency weights: on both scales, integer row weights fit the same curve as the rows
  repeated, to $10^{-10}$, and dropping the row weight from the debiasing weight or from the
  final stage breaks the equality;
- the cross-validated risk table recomputed by hand, on a weighted sample and both scales, from
  each fold's curves, the outer out-of-fold pseudo-outcomes at the held-out rows and the row
  weights, to $10^{-12}$: evaluating on the training rows, swapping $D_t$ and $D_r$, or dropping
  the weights moves it.

The [evidence manifest](evidence.md#the-table) lists them. The source is
van der Laan, Carone and Luedtke (2024; v2 dated 2026), arXiv:2402.01972v2, a preprint; see the
[reference list](../references.md#conditional-effects).
