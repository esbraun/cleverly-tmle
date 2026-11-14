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
| heterogeneity test | joint Wald chi-square, df the number of non-constant terms; below 40 clusters, the Hotelling $F$ reference | a fixed-dimension stack of the coefficients |
| curve $\varphi(v)^\top\beta$ at declared points; band over a declared finite grid | pointwise Wald; multiplier band | linear in $\beta$; the shipped multiplier band over a fixed finite set |
| risk difference and calibration of a fixed external score, and the paired risk difference of two (`assess_score`, `compare=`), on either scale | Wald, from one TMLE fit | the efficient influence curve of the risk at a fixed function, and its difference for two; the calibration is a projection onto $(1, s(V))$ |
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

The score assessment's parameters have the same product remainder with weight $c_a(W)$ (see
[External scores](#external-scores)), and it is not of the linear-remainder form either: the
fluctuation's features are known functions of the fixed scores, and the only estimated quantity
inside them, the log-ratio calibration $\beta_k$, enters quadratically. A single score's risk
difference adds the error of its best constant. On the difference scale that is
$(\widehat{\mathrm{ATE}} - \mathrm{ATE})^2$. On the log-ratio scale $c_n$ minimizes the estimated
risk among constants, so the row is biased upward by about
$\operatorname{Var}(R_n'(c_0))/(2R''(c_0))$, which is $O(1/n)$, and its curve evaluated at $c_n$
overstates the SE slightly. The [score-assessment study](method-evidence/score-assessment.md)
measured 0.09 to 0.13 SD of bias on the calibration groups' log-ratio rows at n = 4,000 (0.08
to 0.10 under a wrong outcome regression) and 0.03 to 0.05 SD on the difference scale at 2,000. The paired row carries neither term, because the constant cancels.

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
chi-square on as many degrees of freedom $q$. A clustered fit with $J < 40$ clusters, whose
coefficients read $t$ on $\nu = J - 2$ degrees of freedom, reads the statistic $T$ by the
Hotelling law instead: $T(\nu - q + 1)/(\nu q)$ against $F(q, \nu - q + 1)$, the square of
that $t$ at $q = 1$. A basis with $q > \nu$ has no such reference and is refused. The test is of
the absence of variation along the basis, not the absence of heterogeneity.

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

`assess_score` reads a column $s(V)$ computed from declared adjustment columns by a rule fitted
on other rows. From one TMLE fluctuation of an ATE fit on the same study it estimates three
parameters of the score, and with `compare=` the same three of a second score $s_2$ and the
paired risk difference of the two. Each carries a Wald interval from its influence curve.

**Difference scale.** The risk of a score is $R(s) = E[s^2 - 2s\tau] = E[(s - \tau)^2] - E[\tau^2]$.
The best constant is the ATE, with $R = -\mathrm{ATE}^2$, and the **risk difference**
$R(s) + \mathrm{ATE}^2$ is negative when the score predicts $\tau$ better than the constant does.
The **paired risk difference** is

$$
\psi_{12} = R(s_1) - R(s_2) = E[(s_1 - \tau)^2] - E[(s_2 - \tau)^2],
$$

the difference of the two scores' mean squared errors for the conditional effect, negative when
$s_1$ predicts better. When both scores read only $V$, the $E[\operatorname{Var}(\tau \mid V)]$
terms cancel and $\psi_{12}$ is the difference of mean squared errors for $E[\tau \mid V]$.

**Log-ratio scale.** With $\ell_t(\theta) = \log(1 + e^{-\theta})$ and
$\ell_r(\theta) = \log(1 + e^{\theta})$, the risk is $R_L(s) = E[\bar Q_t\ell_t(s) + \bar Q_r\ell_r(s)]$,
the loss of the log-CRR projection. The best constant is $c = \log(E\bar Q_t / E\bar Q_r)$, and
the paired risk is $R_L(s_1) - R_L(s_2)$. Since
$R_L(s) - R_L(\theta^*) = E[(\bar Q_t + \bar Q_r)\,\mathrm{KL}(\pi_t \,\|\, \operatorname{expit} s)]$
with $\pi_t = \bar Q_t / (\bar Q_t + \bar Q_r)$ and $\theta^* = \log(\bar Q_t / \bar Q_r)$, the
paired risk is a difference of weighted Bernoulli divergences from the true log-CRR.

The **calibration** is the projection of the contrast onto $(1, s(V))$, the CATE or log-CRR
projection with that basis: a slope of 1 and an intercept of 0 say the score is calibrated.

**Influence curves.** Each risk has the form $E[k(W) + c_t(W)\bar Q_t + c_r(W)\bar Q_r]$ with
known weight functions, for example $k = s^2$, $c_t = -2s$, $c_r = 2s$ on the difference scale
and $k = 0$, $c_t = \ell_t(s)$, $c_r = \ell_r(s)$ on the log-ratio scale. Its efficient influence
curve at a fixed $s$ is van der Laan, Carone and Luedtke, Theorem 1 (arXiv:2402.01972v2):

$$
k + c_t\bar Q_t + c_r\bar Q_r - R + \sum_{a \in \{t, r\}} \frac{1\{A = a\}\Delta}{g(a \mid W)\Pi(a, W)}\,c_a(W)\,(Y - \bar Q_a).
$$

The single-score risk difference adds the comparator's curve, $2\,\mathrm{ATE}(\tilde D - \mathrm{ATE})$
on the difference scale and the curve of $R_L(c)$ with $c$ held fixed on the log-ratio scale; the
derivative of $R(c)$ in $c$ vanishes at the best constant, so estimating it adds no first-order
term. The paired curve is the difference of the two single-score curves, and the comparator
terms cancel exactly:
$\mathrm{IC}_{12} = (s_1^2 - s_2^2) - 2(s_1 - s_2)\tilde D - \psi_{12}$ on the difference scale.
The two blocks and the paired row form one fixed-dimension stack on the same rows.

**The TMLE.** The ATE fit supplies cross-fitted $\bar Q^0$, $g$ and $\Pi$. On the difference
scale the columns $\{1, s_1, s_2\}$ are reduced to an orthonormal basis of their span by the thin
SVD of the targeting curves, and one logistic fluctuation of $\bar Q^0$ runs along
$H = \sum_a 1\{A = a\}B_a(W)/(g\Pi)(a, W)$ with numerators $B$ at arm $t$ and $-B$ at arm $r$.
Every parameter on this scale is linear in $\tau$ with features in that span, so one solve
targets all of them. On the log-ratio scale $c_t \ne -c_r$, so the numerators are arm specific:
$(1, 0)$ and $(0, 1)$ for the arm means that fix $c$, $(\ell_t(s_k), \ell_r(s_k))$ for each risk,
and the log-CRR projection's covariate for each calibration. The calibration columns read
$\beta_k$, so the fluctuation and the projection alternate until the relative shift in $\beta$
is within tolerance, and a fit that does not settle raises a `DataError`. Every reported value
is the substitution estimate from the targeted $\bar Q^*$, for example
$\hat\psi_{12} = P_n[s_1^2 - s_2^2] - 2P_n[(s_1 - s_2)(\bar Q^*_t - \bar Q^*_r)]$.

**Reduced-span retry.** Near a calibration slope of about 0 or 0.5, a score's log-ratio
numerators are collinear to second order, and the rounds can cycle. A fit whose full-span rounds
do not settle reruns from its best round along the span without its thinnest direction, then
without its two thinnest, at most one per score. It reports the first that settles with every
numerator within $10^{-3}$ of the reduced span and every original score equation, the dropped
ones included, inside the `scaled_var` threshold at the accepted $\bar Q^*$; otherwise it raises
the `DataError`. The span record's `reduced` counts the dropped directions, and such a fit reads
`experimental`. Coverage at a true slope near 0 or 0.5, where the retry keeps firing as $n$
grows, is not measured.

**Why a TMLE.** The TMLE and the one-step share the influence curve and the remainder, so
neither is more efficient. The substitution has three exact properties the one-step lacks. The
paired estimate on the difference scale is $P_n(s_1 - \tau^*)^2 - P_n(s_2 - \tau^*)^2$ against one
bounded $\tau^*$, and the single-score risk difference is at least $-\operatorname{Var}_n(\tau^*)$, so it never says a
score beats the constant by more than the estimated heterogeneity. The log-ratio calibration
projects $\bar Q^*$ under a strictly convex loss, so it is unique, where the one-step's weights
$D_t + D_r$ can be negative. Every estimate is a functional of one $\bar Q^*$. The one-step
$P_n[s^2 - 2s\tilde D]$ survives as an auxiliary: at the targeted fit it equals the substitution
to the solver's tolerance, which the tests check, and the `score-assessment` study publishes it
at the initial fit beside the TMLE.

With `compare=` the span grows from $\{1, s_1\}$ to $\{1, s_1, s_2\}$, so the first score's own
rows differ in finite samples from a call without it. Both are valid TMLEs with the same
influence curve and remainder.

**Conditions.** The remainder is the product of nuisance errors of
[Remainders](#remainders), with $c_a$ in place of $s(a)\varphi$, so the intervals need both
nuisances to converge at the product rate, strong positivity, a bounded outcome and bounded
scores. With a known mechanism, complete outcomes and a truncation that does not bind, any
outcome regression will do. A single score's risk difference also carries the best-constant term
of Remainders, and the paired row does not.

Every interval needs each score to be a fixed function, fitted on other rows. A score fitted on
this study's outcomes or treatments is an estimated index: the target becomes the random
$R(s_n)$, and the projection onto it has a remainder the interval ignores. Out-of-fold
predictions of a model trained on these rows are not external either, because the function
changes with the fold and depends on these rows' outcomes. `score_kind="external"` is the
declaration the interval rests on, for both columns. A column equal to an in-sample curve of the
same identified effect, or named with the `effect_curve:` prefix, is refused, and that detection
is best effort. The paired interval covers $R(s_1) - R(s_2)$ for the one pair named before the
fit; trying many pairs and reporting the best is a selection it does not cover.

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
SE is measured, its ratio to the estimator's SD lies between 0.987 and 1.027 for every row the
study gates at $n = 2{,}000$, on 18,000 to 30,000 replicates per cell
(`tests/studies/targeting_curves_oracle.csv`). The AUTOC of cell H reads 1.002 (SE 0.004).
The exception is the paired difference of two correlated scores at TOC(0.1): 1.063 at
$n = 2{,}000$, 1.051 at 4,000 and 1.024 at 8,000. That interval is conservative by a
finite-sample amount that shrinks with $n$. The study reports the row with a one-sided gate:
the lower ends of its 99% SE-ratio and coverage intervals must still reach 0.93 and 0.92, so
an interval that became too narrow would fail. Only a row that errs on the safe side can be
reported this way.

**Routes.** Route 1 takes a study column fitted on other rows, with the three checks of
`assess_score` on it and on a `compare=` column. Route 2,
`curve.diagnostics.targeting_curve(split=0.5)`, refits the producing estimator on a share of
the units, scores the other rows with that refit (kept as `TargetingCurve.score_curve`) and
fits the ATE on them only; the curve is that of the refit, not of the curve fitted on all rows.
A paired `compare=` column shares the draws, and the SE of a difference is the SD of the
per-draw differences.

**Guards.** A grid point with fewer than 50 rows in its top fraction (`TARGETING_MIN_TOP_ROWS`,
a declared guard with no source) keeps its point and withholds its interval and the band; 50 to
199 rows read `experimental`, because the study calibrates no interval with fewer than 200 top
rows. Weights, clusters, missing outcomes and a multi-arm pair are
computed and read `experimental`; fewer than 40 clusters withhold the intervals. A row whose
half-sample draws do not vary (SE at most $10^{-12}$, as for every TOC of a score that ties all
rows) withholds its interval rather than report a width of rounding. With weights, every rank
weight reads the row mass normalised to mean one, so the weighted AUTOC is defined in those
units. The status reads the configuration and the top-fraction rows only: the learners, the
number of folds and $n$ are not checked against the study's cells, and an area row carries no
size guard of its own.

**Status.** The registered [targeting-curves study](method-evidence/targeting-curves.md)
ran 6,000 replicates per cell at $n = 2{,}000$ (800 at $n = 1{,}000$ for the band's power), the
count its declaration fixed from the oracle and pilot ratios. Every gated cell passed, and no
fit failed. The 33 gated calibration rows have SE-ratio intervals inside 0.93 to 1.07 and
coverage intervals inside 0.92 to 0.98. The band covers the grid jointly in 0.956 of the
replications, with size 0.041 and 0.042 on the two null laws and power 0.874 at its declared
slope. The one red cell is the reporting row, the paired TOC(0.1) difference: SE ratio 1.060
(99% interval 1.035 to 1.086) and coverage 0.963 (0.956 to 0.969). It is conservative, as the
oracle computation predicted, and both lower ends clear its one-sided gate.

A curve reads `supported` when its configuration is one a cell measured, with a binary
treatment, iid unweighted rows, complete outcomes, one split and cross-fitted nuisances
(`TARGETING_MEASURED` in `cleverly.targeting_curves`):

| configuration | cells |
| --- | --- |
| an external score, a continuous outcome, an estimated mechanism | `H`, `N`, `P`, and the paired difference `C` |
| an external score, a binary outcome, an estimated mechanism | `B`, `U`, `D` |
| an external score, a continuous outcome, a known mechanism | `K` |
| a declared split, a continuous outcome, an estimated mechanism | `S` |

A paired difference reads `supported` in the first configuration only. Any other configuration,
a grid point with fewer than 200 top rows, a cut in a tied block and an area on a continuous
score with atoms read `experimental`. No comparator is registered; the `grf` comparison is
pending, and a missing comparator does not block the row.

## Weights, clusters and missing outcomes

Weights define the weighted law, as for every point target. With `PointTreatment(cluster=...)` the cluster is the unit:
the variance sums the influence curve within clusters, with the $G/(G - 1)$ factor of the shipped
cluster-robust variance, and folds are grouped by cluster. With missing outcomes the clever
covariate carries $\Delta/\Pi$, and a cross-fitted fit meets the
[arm-indexed contract](scope-and-refusals.md#missing-outcome-arm-indexed-contract).

`assess_score` inherits all three from its ATE fit, without a study. A clustered assessment
follows the library's few-cluster rule: with fewer than 40 clusters the interval takes a $t$
reference on $J - 2$ degrees of freedom, from 3 to 9 clusters it is labelled
`few_cluster_plugin`, and below 3 only the point estimate is reported.

## Missing treatments

With `PointTreatment(treatment_missingness=...)` an in-sample projection fit reads the
treatment through the composite indicator $C_a = \Delta_A \Delta 1\{A = a\}$, whose mechanism is
$g_{c,a}(W) = \pi_A(W)\, g(a \mid \Delta_A = 1, W)\, \Pi(a, W)$, with $\pi_A(W) = P(\Delta_A = 1
\mid W)$ and $\Pi = 1$ without `missingness=`. Identification needs the treatment and the outcome
missing at random given $(A, W)$ and $g_{c,a} > 0$; the target is the full-data projection, and
every row's $W$ enters the plug-in and the Gram matrix $M$. The clever covariate is the shipped
one with $1\{A = a\}/(g \Pi)$ replaced by $C_a / g_{c,a}$; the composite is clipped as a product
([DR-TMLE theorem](dr-tmle/theorem.md)). The remainder is the shipped product of the outcome error
and the error of $g_{c,a}$. A cross-fitted fit refuses: its fold-local product of three factors is
[X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) (c).
`tests/unit/test_projection_compositions.py` checks the point and the curve at oracle factors on
an exact law, the $\pi_A$ factor through a wrong outcome regression, the complete-case and
masking mutations, and the reduction to the missing-outcome projection.

## Baseline strata

With `strata=` the fit reports, beside the marginal coefficients `cate_projection[term]`, each
stratum's coefficients `cate_projection[term][S=s]`: the projection $\beta_s = M_s^{-1}
P_0[\phi(V)\tau(W) \mid S = s]$ under the law of $V$ within the stratum. The marginal and stratum
coefficients are related by $\beta = M^{-1} \sum_s p_s M_s \beta_s$. Each stratum's clever
covariate is the block $1\{S = s\} H_s / P_n(S = s)$, and its influence curve is the stratum's own
curve scaled by $n / n_s$. Every stratum fits its own nuisances, and a stratum whose working-model
Gram matrix is singular (a modifier constant within it, or an `Indicators` level absent from it)
refuses before any learner, and again inside each bootstrap replicate.

On the difference scale the stratified fluctuation solves the marginal score too, since the
marginal clever covariate is a linear combination of the blocks, so the marginal names come from
the same fluctuation. On the log-ratio scale the clever covariate reads $\beta$, and the blocks
at $\beta_s$ do not span the marginal covariate at $\beta$: the marginal names come from a second,
unstratified fluctuation, the shipped log-ratio projection, run as its own target group
(`effect_modification_ratio_strata` holds the stratum names). Each name comes from exactly one
group; a name two groups produce raises. The covariance across a marginal and a stratum name reads
the stored influence curves of both groups. `heterogeneity_test(stratum=)` and
`effect_curve(stratum=)` read one stratum's coefficients.

## Repeated splits

With `repeats` above 1 each repeat $r$ refits the projection on its own split and stores its
coefficients $\beta_r$, covariance $\Sigma_r$ and influence curves. The point is the coordinate-wise
median $\tilde\beta$, and the covariance is the matrix rule of Chernozhukov et al. (2018),
equation 3.14: among the candidates $\Sigma_r + d_r d_r'$ with $d_r = \beta_r - \tilde\beta$, the
one of median operator norm, which `covariance()` returns over projection names. Each name's
reported standard error is the scalar rule, the square root of the median of $\sigma_{j,r}^2 +
d_{j,r}^2$, which a one-name `covariance()` equals; `se_without_displacement` is
$\operatorname{median}_r \sigma_{j,r}$, the rule with $d_r = 0$, published beside it.

| consumer | what it reads on a repeated fit |
| --- | --- |
| point, standard error, interval | the median and the scalar rule, per name |
| `covariance(names)` | the matrix rule over projection names; other names keep the repeated point-fit refusal |
| `heterogeneity_test()` | $\tilde\beta' \tilde\Sigma^{-1} \tilde\beta$ over the tested coefficients, with $\tilde\Sigma$ the matrix rule |
| `effect_curve().predict(v, interval=True)` | the median over repeats of $\phi(v)'\beta_r$, with the median rule's variance |
| the simultaneous band | the median over repeats of the per-repeat bands (`median_bands`); it reads experimental |

A repeat that withholds inference withholds it in the report and in `covariance()`.

## Bootstrap

`n_bootstrap=` refits the whole estimator, nuisances included, on each resample of the units
(clusters when declared) and publishes the percentile interval of each projection coefficient.
For an in-sample fit with a fixed basis the coefficients are a finite-dimensional Z-estimator of
the targeted fit; the package's own reasoning, not a read source, is that the full-refit bootstrap
then has the Wald interval's limit under the conditions of the in-sample fit (roadmap
[X55](../roadmap.md#x55-bootstrap-extensions) (a) names Tang and Westling (2024), Theorem 3.4, as the base of the case with
finite-dimensional nuisances).
A cross-fitted fit's bootstrap reads experimental with the reason of
[F2](../roadmap.md#f2-targeted-bootstrap-inference). A replicate that raises, for example one
whose resample leaves a stratum's Gram matrix singular, is dropped and counted.
`heterogeneity_test()` and `effect_curve()` read the influence curves, not the replicates.

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
| an intermediate, a continuous or vector treatment | the projection's clever covariate for the composition is not built | `implementation_gap`, roadmap P1 |
| a missing treatment with cross-fitting | the fold-local product of the three factors of $g_{c,a}$ is not built | `implementation_gap`, roadmap X53 |
| `strata=` with a stratum whose working-model Gram matrix is singular | the stratum's coefficients are not identified | `DataError` |
| `method="ep"` with a missing treatment | the EP learner cross-fits its nuisances, and the cross-fitted composite is not built | `implementation_gap`, roadmap X53 |
| `method="ep"` with `strata=` | the EP curve has no stratified form; fit one curve per stratum, or declare the stratum a modifier | `api_redirection` |
| `assess_score` with a missing treatment or `strata=` | the score assessment fits one pooled fluctuation over the ATE fit's mechanism, and its composite and stratified forms are not built | `implementation_gap`, roadmap X98 |
| a sensitivity analysis or refutation of a projection fit | each is defined for a mean or a contrast of means | `implementation_gap`, roadmap P1 |
| `heterogeneity_test()` on a basis without an intercept column | its coefficients being zero says the contrast is zero, not that it is constant | `invalid_declaration` |
| `heterogeneity_test()` on a clustered fit with fewer than 40 clusters and more than $J - 2$ non-constant terms | the Hotelling reference on $J - 2$ degrees of freedom has no law for more terms | `inference_gap` |
| `method="ep"` for an estimand other than `ConditionalContrast` | it estimates a different target | `different_estimand` |
| `predict(interval=True)` on an EP curve, or a non-default `Inference` on `EPMethod` | EPL proves rates, not a limit law for the curve | `inference_gap` |
| `method="ep"` with `model=Sieve(basis)` sized by cross-validation | the model is a declared projection target; pass the sieve as `sieve=` with `model=None` | `invalid_declaration` |
| `method="ep"` with both `model=` and `sieve=`, or `final_learner=` with `model=` | two bases for one fit; with a model the final stage is least squares on it | `invalid_declaration` |
| `final_learner=` on the log-ratio scale | the weighted logistic final stage with a declared learner is not built | `implementation_gap`, roadmap P1 |
| `EPMethod` with `cross_fit=False`, `repeats` above 1, `cv_evaluation=True`, a non-default targeting scheme, or a fluctuation setting | the debiasing pools every row with its own out-of-fold nuisances and fits no fluctuation | `invalid_declaration` |
| `assess_score` without `score_kind="external"`, with `computed_from` outside the adjustment set, or on an in-sample curve | the interval needs a fixed function of adjustment columns | `invalid_declaration` |
| `assess_score` with `compare=` naming the score column, or a column with the same value on every row | the paired risk is 0 with no spread | `DataError` |
| `assess_score` with `compare=` a sequence of two or more columns | a joint interval over every pair of three or more scores is not built | `implementation_gap`, roadmap X98 |
| `assess_score` with `repeats` above 1 | a median over splits of jointly targeted risks has no reviewed joint covariance | `inference_gap` |
| `assess_score` with `n_bootstrap` above 0 | the refit bootstrap is not built; the influence-curve intervals are reported | `inference_gap` |
| `assess_score` with `target_weights=True`, a declared stopping rule, a non-pooled targeting scheme or `cv_evaluation=True` | every row loads several score columns, which the weighted submodel cannot carry; the fluctuation is one pooled solve under the relative stopping test | `implementation_gap`, roadmap X98 |
| `targeting_curve` without `score_kind="external"`, with `computed_from` outside the adjustment set, or on an in-sample curve, for the score or the `compare=` column | as `assess_score` | `invalid_declaration` |
| `curve.diagnostics.targeting_curve(split=...)` with a split that is not a fraction in $(0, 1)$ | a fold average of fold-specific curves is a property of the learning algorithm; its only source (Imai and Li 2023, Section 4) covers difference-in-means scores in a randomized experiment | `invalid_declaration` |
| a targeting curve with `repeats` above 1 | a median over splits of a whole curve has no reviewed joint law | `inference_gap` |
| a targeting curve with `target_weights=True` or a declared stopping rule | every row loads several weight columns, which the weighted submodel cannot carry; a declared stopping rule needs a stopping context no curve builds | `implementation_gap`, roadmap X92 |
| a targeting curve on an ATE fit with a non-pooled targeting scheme or `cv_evaluation=True` | the curve fits one pooled fluctuation over every evaluation row | `implementation_gap`, roadmap X92 |
| a targeting curve on an ATE fit with `n_bootstrap` above 0 | the curve's intervals come from its half-sample bootstrap | `inference_gap` |
| `compare=` on route 2 of an engine fit | the column is read from the producing study, which an engine fit does not keep | `invalid_declaration` |

Each one is raised before any learner is fitted. `tests/unit/test_conditional_contrast.py`
pins its text, the score assessment's among them, and `tests/unit/test_projection_compositions.py`
pins the rows of the four compositions above.

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
clusters, for example) and the simultaneous band read `experimental`. On this axis the
outcome-regression option reads the other way round from the point estimands: the measured fit
is the per-arm one, and the support key holds `outcome_by_arm`, so a pooled projection matches
no row. The key keeps that label now that the point study `option-outcome-by-arm` is registered and the
option's fit condition is deleted (`tests/unit/test_option_outcome_by_arm_witness.py`). The external-score check
has its own measured configurations, below.

The in-sample fit, a missing treatment, `strata=`, `repeats` above 1 and the bootstrap read
`experimental` too. The study that measures them, `projection-compositions`, is declared in
`tests/studies/projection_compositions.py`, and `tests/unit/test_projection_compositions_design.py`
checks its laws, cells and the design numbers that need no pilot. It has not run: its calibration
`R` is computed from pilots first, and its rows join this table when it is registered.

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

The registered [score-assessment study](method-evidence/score-assessment.md)
(`tests/studies/score_assessment.py`, run by
`python -m tests.canonical.score_assessment.regenerate`) passed its primary test and all 49
property cells, with no failed fit. Its 32 calibration rows, at R = 15,200, read SE ratios from
0.989 to 1.017 and coverage from 0.948 to 0.953. The paired Wald test rejects in 0.050 of the
replications at equal risks and in 0.896 at the declared difference. Under a declared wrong
outcome regression every TMLE risk difference stays within 0.11 SD of the truth, while the
untargeted plug-in misses by 4 to 49 SD. An estimate of `assess_score` reads `supported` when
its configuration is one a group measured, with a binary treatment, iid unweighted rows,
complete outcomes, cross-fitted nuisances and a fluctuation settled on the full span:

| scale, outcome, mechanism | single-score rows | paired row |
| --- | --- | --- |
| difference, binary, estimated | `conditional-projection` `assess_score_2k`; `score-assessment` `paired_2k` | `paired_2k` |
| difference, binary, known | `paired_trial_2k` | `paired_trial_2k` |
| difference, continuous, estimated | `paired_cont_2k` | `paired_cont_2k`, `equal_cont_2k`, `power_cont_2k` |
| log ratio, binary, estimated | `single_crr_4k`, `paired_crr_4k` | `paired_crr_4k` |

`cleverly.effect_curves.SCORE_ASSESSMENT_MEASURED` and `SCORE_PAIRED_MEASURED` hold the table,
and `tests/unit/test_score_assessment_tmle.py` checks that every group it names ran and passed.
Weights, clusters, missing outcomes, a pair of arms of a multi-arm treatment, an ATE fit without
cross-fitting, a reduced-span fit and every other configuration read `experimental`. The
difference-scale groups run at n = 2,000 and the log-ratio groups at 4,000: at 2,000 rows the
pilot's SE ratio of 1.047, the best-constant term plus an unexplained excess, predicted that no
replication count up to 50,000 meets the calibration rule, and no interval was measured there.
The study page gives that term's measured size, the reduced-span counts, and the limits of what
the study can see.

The score assessment's fast-tier instruments (`tests/unit/test_score_assessment_tmle.py`) are:

- the paired point against the functional and its curve against the Gâteaux derivative on a
  three-cell law, on both scales, with the sign following the order, and mutations of the
  second-moment term, the residual term and the cancelled comparator term, each with a witness
  that the term is not small;
- on a six-cell law where the fluctuation does not saturate, the paired row equal to the
  difference of the single rows of one fit; every estimate exact under a wrong outcome
  regression and the oracle mechanism, while three mutations of the numerators miss it; and the
  substitution equal to the one-step at the targeted fit;
- the targeted regressions in range where the one-step's weights turn negative, the log-ratio
  calibration as the unique minimizer of a convex loss, and a loud failure when the log-ratio
  alternation cannot settle;
- missing outcomes, observation weights and the three cluster cases, the support labels and
  their tables, and the report.

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
