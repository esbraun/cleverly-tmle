# Adaptive TMLE for a trial with external data

## What this solves

A randomized trial identifies its own treatment effect without bias but with a wide interval when
it is small. External rows on the same outcome, such as registry patients or the controls of an
earlier study, add information about the covariate-outcome relation. They also carry bias, because
nothing randomized their treatment or selected them like the trial. Adaptive TMLE (A-TMLE) of van
der Laan, Qiu, Tarp and van der Laan (J. Causal Inference 14(1):20240025,
doi:10.1515/jci-2024-0025; arXiv:2405.07186v2) learns that bias as a function of the covariates
and removes it, so that the external rows add information about the effect.

`TrialWithExternalData` declares the pooled frame, `TrialATE` names the effect, and
`AdaptiveTMLEMethod` (`method="atmle"`) estimates it. Every locator below reads arXiv:2405.07186v2.
The journal text keeps v2's section, lemma, theorem and appendix numbering, so the locators hold
for it.

| piece | object | what it fixes |
| --- | --- | --- |
| data | `TrialWithExternalData` | $O = (S, W, A, Y)$, plus $\Delta$ when outcomes can be missing |
| estimand | `TrialATE(population=)` | $\Psi$ over the pooled covariates, or $\Psi_2$ over the trial's |
| method | `AdaptiveTMLEMethod(form=)` | the adaptive CV-TMLE (default) or Algorithm 1 |
| result | `AdaptiveTMLEResult` | the estimate, the working models of every fold, and the diagnostics |

## Data and estimands

$S = 1$ marks a trial row and $S = 0$ an external row. $W$ is the declared adjustment set, $A$ a
binary treatment, $Y$ the outcome, and $\Delta = 1$ when $Y$ is recorded. The trial's conditional
effect is $\tau_1(W) = Q(1, W, 1) - Q(1, W, 0)$ with $Q(s, w, a) = E(Y \mid S = s, W = w, A = a)$.

| `population` | estimand | source |
| --- | --- | --- |
| `"pooled"` | $\Psi = E_W[\tau_1(W)]$, averaged over the covariates of every row | arXiv:2405.07186v2, Sec 2.4 |
| `"trial"` | $\Psi_2 = E[S\,\tau_1(W)] / P(S = 1)$, averaged over the trial's covariates | a natural extension, derived here |

The paper derives $\Psi$ only. $\Psi_2$ ships by the pointwise form of Lemma 2, the chain rule and
the delta method for the ratio. Its influence function is in the [influence functions
section](#influence-functions), and the exact-law tests check it by a Gateaux derivative. The two
estimands differ whenever the conditional effect varies with covariates that also predict trial
membership.

The difference scale is the only one derived. The contrast is the second sorted treatment level
against the first.

## Identification

| id | assumption | population |
| --- | --- | --- |
| A1 | in the trial, $Y^a$ is independent of $A$ given $W$ (randomization) | both |
| A2 | in the trial, $0 < P(A = 1 \mid S = 1, W) < 1$ | both |
| A3 | $P(S = 1 \mid W) > 0$ wherever the pooled covariates fall, so $Q(1, W, a)$ is defined at every pooled row | `"pooled"` only |
| MAR | with `missingness=`, $Y$ is independent of $\Delta$ given $(W, A)$ among trial rows | when declared |

Under A1 and A2, $E(Y^a \mid S = 1, W) = Q(1, W, a)$, so $\Psi_2$ is identified from the trial's
regressions alone. $\Psi$ also needs A3 for $Q(1, W, a)$ to exist on the pooled support of $W$. The
design assumes nothing about the external outcomes. The fit reports A3 as the estimated
$P(S = 1 \mid W)$ on the external rows and warns when mass lies below $0.01$; it does not report
A3 for `"trial"`, whose average the trial covers by construction.

The missing-outcome assumption is stated on $(W, A)$. Section B of the paper states independence of
$(Y^1, Y^0)$ and $\Delta$ given $(W, S = 1)$. Once $\Delta$ depends on $A$, neither statement implies
the other, and the fitted $P(\Delta = 1 \mid S, W, A)$ uses the $(W, A)$ form.

## Decomposition

Write $\tilde Q(W, a) = E(Y \mid W, A = a)$ on the pooled rows, $\Pi(1 \mid W, a) = P(S = 1 \mid W, A = a)$, and
$\tau_S(W, a) = Q(1, W, a) - Q(0, W, a)$ for the trial-minus-external difference. Lemma 2 gives

$$
\tau_1(W) = \tau_A(W) - \tau^{\#}(W),
\qquad
\tau^{\#}(W) = \Pi(0 \mid W, 0)\,\tau_S(W, 0) - \Pi(0 \mid W, 1)\,\tau_S(W, 1),
$$

with $\tau_A(W) = \tilde Q(W, 1) - \tilde Q(W, 0)$, the contrast of the pooled regression. The
pointwise form follows from $\tilde Q = \Pi\,Q(1, \cdot) + (1 - \Pi)\,Q(0, \cdot)$. Hence
$\Psi = \tilde\Psi - \Psi^{\#}$, the pooled effect minus the bias of the external rows, and
$\Psi_2 = E[S\,\tau_1(W)] / P(S = 1)$. With external controls only, $\Pi(0 \mid W, 1) = 0$, so
$\tau_S(W, 1)$ is neither identified nor needed, and the fit learns the bias on the control rows.

## Working models and the R-loss

Two working models describe the two unknown functions. Let $\phi(W)$ be a dictionary with
$\phi_0 = 1$ for the effect, and $\psi(W, A)$ a dictionary for the bias with $\psi_0 = 1$ and
$\psi_1 = A$. For selected columns, the projections are

$$
\beta_A = \arg\min_b E\big[(Y - \theta(W) - (A - g(W))\,b^\top\phi(W))^2\big],
\qquad
\beta_S = \arg\min_b E\big[(Y - \tilde Q(W, A) - (S - \Pi(W, A))\,b^\top\psi(W, A))^2\big],
$$

with $\theta(W) = E(Y \mid W)$ and $g(W) = P(A = 1 \mid W)$ on the pooled rows. These are
Robinson-type R-losses, so each solves a score that is Neyman orthogonal in its nuisances. The
working-model parameter is

$$
\Psi_w = E[\beta_A^\top\phi(W)] - E\big[\Pi(0 \mid W, 0)\,\beta_S^\top\psi(W, 0) - \Pi(0 \mid W, 1)\,\beta_S^\top\psi(W, 1)\big],
$$

and $\Psi_{2,w}$ is the same with the $S$-weighted average. When $\tau_A \in \mathrm{span}(\phi)$
and $\tau_S \in \mathrm{span}(\psi)$, $\Psi_w = \Psi$. That condition is sufficient and not
necessary. With $w = g(1 - g)$, the error $\tilde\Psi_w - \tilde\Psi$ is a product of the error of
$\tau_A$ and the error of the representer $1/w$ in the weighted projection, as in Theorem 4 of
van der Laan et al. (arXiv:2307.12544v2, Sec 4.3). The projection equals $\Psi$ whenever either factor
lies in the span, and the two truths coincide for any dictionary when $g(1 - g)$ and $\Pi(1 - \Pi)$
are nearly constant.

## The dictionary

The default dictionary is `WorkingBasis` followed by `WorkingDictionary`.

- A covariate with at least nine distinct values on the selection rows gets its standardized linear
  term, hinge terms $(z - k_j)_+$ at five interior quantile knots of the standardized value, and
  every pairwise product of standardized values. A covariate with fewer values gets one indicator
  for each level above the smallest. A constant covariate contributes nothing.
- The bias dictionary is $\psi = [1, A, \phi_{-0}, A\,\phi_{-0}]$. For external controls only it is
  $\phi$.
- A support screen drops a column unless it is nonzero on at least
  $k_{\min} = \max(10, \lceil \sqrt{n}/2 \rceil)$ rows in each of two groups. An effect column
  needs $k_{\min}$ treated and $k_{\min}$ control rows. A bias column needs $k_{\min}$ external
  and $k_{\min}$ trial rows, counting only treated rows for a column that carries $A$. The
  constant, $A$, the linear terms and their products with $A$ always pass the screen.
- A pivoted QR drops exactly collinear columns, so every refit has full column rank.

The screen exists because a hinge that is nonzero on a handful of external treated rows has a
coefficient the data cannot identify, and the refit then returns an extreme value for it. A transformer passed as `working_basis=`
goes through the same screen and rank steps. `working_basis=` must be a scikit-learn transformer
with `fit` and `transform`.

The lasso is the R-loss lasso, solved exactly by feature-sign search on `cv.glmnet`'s standardized
scale, with only the constant unpenalized. The penalty is the largest one attaining the smallest
cross-validated R-loss risk over a 100-point grid (`learner_folds` inner folds). Where the search
stalls short of the optimality conditions, cyclic coordinate descent continues from it, and the
quadratic is solved exactly on its support and signs. Coordinate descent converges on this convex
objective from any start ([Tseng 2001](../references.md)). The solver checks the
Karush-Kuhn-Tucker conditions and raises `CapabilityError` with a smaller-basis alternative if
they still fail.

The arm column $A$ of the bias dictionary is penalized by default; `force_arm=True` leaves it
unpenalized, as `atmle` does, and then `arm_hierarchy` has nothing to add. Forcing it in gives the external treated
and control arms separate free mean shifts, so the external rows then carry no information on the
level of the trial effect. On law `a0` (no bias, so the oracle bias selection is the constant), the exact-law
standard deviation of $D_w$ is $0.48$ times the trial-only efficient standard deviation, and
$1.00$ times with $A$ forced
(`test_the_a0_efficiency_gain_is_the_theory_and_a_forced_arm_column_removes_it`). The gain
therefore depends on the lasso dropping every column that shifts an external arm's mean, which
nothing guarantees at a given $n$.

A bias selection that holds a column carrying $A$, such as $A \cdot x_0$, also holds $A$
(`atmle.arm_hierarchy`). Without $A$, the product says the two external arms' biases agree where
the centred column is zero. The dictionary centres at the selection rows' mean, so that point is a
sample quantity, and the projection then misses $\Psi$ by the order of the sample mean's error at
every $n$. On law `a3` with the selection $\{1, A x_0\}$ the miss is 1.15 standard errors of
$D_w$ at $n = 4{,}000$. $A$ added, or centring at the population mean, removes it
(`test_an_arm_product_without_the_arm_column_misses_the_effect_by_a_sample_mean`).

With the rule, the one arm restriction a selection can make is $\tau_S(W, 1) = \tau_S(W, 0)$: the external bias
is the same in both arms. That restriction does not depend on how the columns are centred, and
the efficiency gain comes from it.

## The estimator

Nuisances are $g(W) = P(A = 1 \mid W)$, $\theta(W)$, $\tilde Q(W, A)$, $\Pi(W, A)$ and
$\pi(W) = P(S = 1 \mid W)$, all fitted by the declared learners on the pooled rows and clipped by
`Targeting.g_bounds`. The R-loss is fitted as least squares of the pseudo-response on
$(A - g)\phi$ or $(S - \Pi)\psi$ and never divides by $A - g$, so no denominator bound is needed.
Each lasso selection is followed by a relaxed (unpenalized least-squares) refit of the selected
columns.

### Adaptive CV-TMLE (default)

`form="cv"` follows Appendix C and Theorem 2. For outer fold $v$ with training rows $T_v$:

1. The nuisances are fitted on $T_v$. The pseudo-responses on $T_v$ need nuisance values on $T_v$'s
   own rows, so a row of fold $u \ne v$ takes nuisances fitted on the folds other than $u$ and $v$.
   This is why the form needs at least three folds.
2. The working models are selected on $T_v$ and refitted there.
3. On the validation rows, each coefficient vector moves along its target direction
   $I_v^{-1} m_v$, taken on $T_v$, by one step $\epsilon$ pooled over all validation folds. The
   step has a closed form because the score is linear in it.
4. $\Pi$ is fluctuated by one $\epsilon$ pooled over the validation folds, and the bias-model step
   alternates with it.
5. The estimate is the mean of the validation-fold plug-ins, with standard error
   $\sqrt{\bar\sigma^2 / n}$, where $\bar\sigma^2$ is the mean of the validation-fold variances of
   the influence function $D_v$ at the fold's targeted fit $P^*_{n,v}$.

The direction is fixed before the validation rows are read. A full refit of the coefficients on the
validation folds was dropped because it was unstable at $n = 2{,}000$ with five folds. A refit on
$T_v$ whose selected columns are rank deficient raises `CapabilityError` naming the fold.

The variance does not reuse the targeting's directions. Theorem 2 normalizes by
$\sigma^2_{w,n,v} = P_0 D^2_{M_{w,n,v}, P_0}$, whose coefficient terms carry
$d_0 = I_0^{-1} m_0$ for fold $v$'s working model, and App C gives no estimator for it. The fit
estimates $d_0$ on every row, with each row's targeted cross-fitted nuisances (the ones in $D_v$'s
residuals) and fold $v$'s selected columns (`atmle.variance_directions`). It then takes
$\hat\sigma^2_v$ as the centred variance of $D_v$ over the validation rows. The estimate, the score
equations and the stopping rule keep the $T_v$ directions, so the point estimate does not move.

The $T_v$ direction $d_T$ is computed with the pair-excluded selection nuisances. Applied with
coefficient 1 on $V_v$, it estimates the variance at $d_T$, which adds
$(d_T - d_0)^\top \mathrm{Var}(G)\,(d_T - d_0)$ for the coefficient score $G$, plus a cross term
that vanishes when the working model holds. The added term vanishes as $n$ grows, but with 37
selected bias columns it put the selective study's `s6_cv` SE ratio at 1.16.

Refits of 100 `s6_cv` draws with the study's seeds read SE ratios of 1.23 with $d_T$,
1.04 with $I$ and $m$ on the validation rows alone (the literal plug-in at $P^*_{n,v}$), and 0.99
on every row. The validation-row plug-in inflates the variance where the selected columns are nearly
collinear: on the separating law `e1` its SE ratio was 1.12 against 1.05 for $d_T$ and 1.00 for
every row (200 draws). Every row gives $n$ rather than $n / V$ rows to an estimate that reads no
outcome; most of those rows also chose the working model, a second-order effect on $I$ and $m$. The
covariate and $\Pi$ components hold no direction and are the same under each choice.

A working model whose Hessian the rows cannot estimate falls back to the $T_v$ direction for that
fold's variance, with a warning. It cannot be estimated when there are fewer than
`VARIANCE_ROWS_PER_COLUMN` (10) rows with a nonzero $(A - g)\phi$ or $(S - \Pi)\psi$ per selected
column, or when the columns are collinear by the refit's rank rule. With $K$ columns on $m$ rows the
inverse of an estimated Hessian is inflated by about $m / (m - K)$, so the floor keeps that within
about 11%. The fallback is consistent and leaves the estimate valid, so it is preferred to a
refusal. A ridge-stabilized inverse has no rule for its penalty here and biases the variance in a
direction that depends on the law. `WorkingModel.variance_fallback` records which component fell
back, and `WorkingModel.variance_directions` the directions the variance used.

### Algorithm 1 (pooled)

`form="pooled"` follows Algorithm 1 and Theorem 1, with cross-fitted nuisances. The lasso selection
runs on all rows, the relaxed refit solves $P_n U_A = P_n U_S = 0$, the $\Pi$ fluctuation
alternates with the bias refit, and the interval is the Wald interval with variance
$P_n D_w^2 / n$.

### Targeting $\Pi$ and the stopping rule

The paper states neither the submodel nor whether the step iterates, and Algorithm 1 leaves the
bias-model score unsolved after $\Pi$ moves. This implementation takes the logistic submodel
$\mathrm{logit}\,\Pi_\epsilon = \mathrm{logit}\,\Pi + \epsilon\,C$ with the log-likelihood loss of
$S$, whose score at $\epsilon = 0$ is the $\Pi$-component $C\,(S - \Pi)$ of the influence function.
Here $C(W, A) = \frac{A}{g(1 \mid W)}\,\tau_{S,\beta}(W, 1) - \frac{1 - A}{g(0 \mid W)}\,\tau_{S,\beta}(W, 0)$
is built from the current bias coefficients, and $g$ is the pooled treatment mechanism. For
$\Psi_2$ the covariate is multiplied by $\pi(W) / P(S = 1)$.

The fit alternates the $\Pi$ step with the bias-model step until
$|P_n D| \le \mathrm{sd}(D) / (\sqrt{n}\log n)$ or 50 passes. A fit that reaches 50 passes warns
with `ConvergenceWarning` and records the residual. With external controls only, $\Pi(W, 1) = 1$
is not fluctuated.

## Missing outcomes

With `missingness=`, $\Delta = 1$ when $Y$ is recorded, and $g^\Delta(S, W, A) = P(\Delta = 1 \mid S, W, A)$
is fitted on the pooled rows. Appendix B multiplies both R-losses and the $Y$-components of the
influence functions by $\Delta / g^\Delta$. That literal inverse weighting leaves a term

$$
E\Big[\Big(\tfrac{\Delta}{g^\Delta} - \tfrac{\Delta}{g^\Delta_0}\Big)
\big(Q(S, W, A) - \theta - (A - g)\,\beta_A^\top\phi\big)\Big]
$$

that is linear in the error of $g^\Delta$, because the bracketed residual is not a nuisance error.
This implementation therefore uses the augmented form: each residual $Y - \theta - \cdots$ becomes
$\frac{\Delta}{g^\Delta}(Y - Q(S, W, A)) + (Q(S, W, A) - \theta - (A - g)\beta_A^\top\phi)$, and
likewise for the bias residual and the R-loss responses. The remainder then gains only products
$(g^\Delta - g^\Delta_0)(Q - Q_0)$. The departure from the paper's literal form is deliberate, and
the exact-law tests run once with $\Delta$, including a deliberate IPCW mutation that fails.

The external rows need no missing-at-random assumption. The reweighted law keeps the trial's
regressions $Q(1, W, a)$, Lemma 2 holds pointwise for it, and any external missingness changes only
the bias the method estimates. $\theta$ and $\tilde Q$ are built from $Q$ fitted on recorded rows
as $\tilde Q = \sum_s \Pi(s \mid W, A)\,Q(s, W, A)$.

## Influence functions

These are the nonparametric-model gradients of the projection parameters. The model that knows the
trial randomization has a smaller tangent space and $D_w$ is not orthogonal to its scores, so $D_w$
is a gradient there and not that model's efficient influence function. Write
$r_A = Y - \theta - (A - g)\beta_A^\top\phi$, $r_S = Y - \tilde Q - (S - \Pi)\beta_S^\top\psi$,
$U_A = I_A^{-1}(A - g)\phi\,r_A$, $U_S = I_S^{-1}(S - \Pi)\psi\,r_S$, with
$I_A = E[g(1 - g)\phi\phi^\top]$ and $I_S = E[\Pi(1 - \Pi)\psi\psi^\top]$, and

$$
D_A = \beta_A^\top\phi - \tilde\Psi_w + m_A^\top U_A, \qquad
D_{\#} = \tau^{\#}_\beta - \Psi^{\#}_w + (S - \Pi)\,C + m_S^\top U_S, \qquad
D_w = D_A - D_{\#},
$$

with $m_A = E[\phi]$ and $m_S = E[\psi(W, 0)\Pi(0 \mid W, 0) - \psi(W, 1)\Pi(0 \mid W, 1)]$.
No $g$, $\theta$, $\tilde Q$ or $\Pi$ term appears from the coefficient equations because each
R-loss is Neyman orthogonal in its nuisances. The $(S - \Pi)\,C$ term is the only $\Pi$ term. For
$\Psi_2$ the $W$-average is over $S = 1$, which adds the factor $\pi(W) / P(S = 1)$ to the $\Pi$ term
and $S$-weighted means to the coefficient terms; the mean of the observed $S\,h(W)$ contributes
$S\,h(W) - E[S\,h(W)]$ and no $\Pi$ derivative.

The remainder is a sum of products of nuisance errors: $(g - g_0)(\theta - \theta_0)$,
$(g - g_0)^2\beta_A^\top\phi$, $(\Pi - \Pi_0)(\tilde Q - \tilde Q_0)$, $(\Pi - \Pi_0)^2\beta_S^\top\psi$,
$(\Pi - \Pi_0)(g - g_0)\tau_S / g$, and squared coefficient errors. No regression is fitted on an
estimated index, so the fit is not in the linear-remainder family and needs no restriction on the
index's regularity.

## Inference status

Every `atmle` estimate carries the status `working_model_projection`, under the experimental
label. The interval covers $\Psi_w$ at the working models the fit selected. It covers $\Psi$
when the oracle-approximation condition of Theorem 1 holds (Theorem 2 for the CV form): the
selected models approximate $\tau_A$ and $\tau_S$ well enough that the working-model bias is
$o_P(n^{-1/2})$. No data check confirms that condition.

The source supplies no theorem for the lasso selection itself. If the selected span converges, the
interval is for the projection onto its limit, which equals $\Psi$ when $\tau_A$
and $\tau_S$ lie in that span or when the representers of the previous section do.

[Inference](inference.md) lists the status in precedence order, and roadmap item
[F39](../roadmap.md#f39-adaptive-tmle-the-oracle-approximation-condition-and-the-underived-scope)
owns a testable oracle condition. The fit reports `result.working_models` for every fold so that a
reader can see which columns carried the bias.

## Exact reductions

Two exact reductions serve as comparators on the same rows.

- The trial-only TMLE of $\Psi_2$ is `PointTreatment` on the $S = 1$ rows.
- The regular TMLE of $\Psi$ (Lemma 1) is `PointTreatment(missingness=)` on the pooled rows with $Y$
  set missing outside the trial. Its shipped missing-outcome target is $E_W[E(Y \mid A = 1, \Delta = 1, W) - E(Y \mid A = 0, \Delta = 1, W)] = \Psi$ and its clever covariate
  $(2A - 1) / (g(A \mid W)\,P(\Delta = 1 \mid A, W))$ equals Lemma 1's weight. A test checks that
  the two influence functions agree on a finite law
  (`test_lemma1_regular_tmle_equals_the_shipped_missing_outcome_influence_curve`).

## Refusals

Each refusal is raised before any learner fits. [Scope and refusals](scope-and-refusals.md#adaptive-tmle-refusals)
lists the codes.

| request | refusal | why |
| --- | --- | --- |
| a treatment with more than two levels, a dose, a vector treatment | `implementation_gap`, [F39](../roadmap.md#f39-adaptive-tmle-the-oracle-approximation-condition-and-the-underived-scope) | Lemma 2 and the working models are stated for a binary treatment |
| a ratio scale | `implementation_gap`, F39 | the source gives no working model for a ratio |
| `method="atmle"` on `PointTreatment` | `invalid_declaration` | the method needs trial membership |
| `TrialATE` on any other design, or another estimand on `TrialWithExternalData` | `invalid_declaration` or `different_estimand` | only the trial effect is identified |
| any method other than `atmle` on the pooled design | `different_estimand` | the other methods read the pooled rows as one population |
| `CrossFitting(repeats > 1)` | `inference_gap` | no source combines adaptive working-model fits across splits |
| `form="cv"` with fewer than three folds, or with cross-fitting off | `invalid_declaration` | the form selects and targets on other folds |
| a setting the method does not read (bootstrap, simultaneous band, outcome fluctuation, `n_jobs`) | `invalid_declaration` | each is refused by name with its reason |
| `weights=`, `cluster=`, `strata=`, `intermediate=`, `treatment_missingness=` | not accepted by the design | no source derives them |
| the sensitivity analyses, refutation and truncation curves | `inference_gap` or `api_redirection` | their bounds assume a point-treatment fit; each fit refuses with its reason |

An arm with all outcomes missing, a trial membership column with a null, and a pooled frame with
no trial row, no external row or a trial of one arm are `DataError`s. A fit with external treated
rows at or below $\max(20, 1\%)$ of the external rows warns that the bias of those rows is learned from
few observations.

## Result surface

`AdaptiveTMLEResult` supports `estimate`, `estimates`, `to_frame`, `summary`, `covariance`,
`influence_curves`, `contrast`, `save` and `load`. `support_report()`, `score_report()`,
`nuisance_report()` and `corrections_report()` return the plain reports in
`cleverly.estimators.atmle_result`. `score_equations` gates $|P_n D|$ against the engine's own
stopping rule and refuses an explicit `tolerance=`. Replay is `summarize` and
`rerun_configuration` only.

## Differences from the published algorithm and the atmle package

The paper is van der Laan, Qiu, Tarp and van der Laan (arXiv:2405.07186v2; J. Causal Inference
14(1):20240025). The journal text keeps v2's sections, lemmas, theorems, Algorithm 1 and
appendices, and the locators here hold for both. The package is the R package `atmle` at commit
`8452808` (`845280889d2fb7d929adb494759b1588ab776f43`, the head of `tq21/atmle` main); the
functions `run`, `fit_cate`, `target_Pi`, `inference` and `get_tau_S_eic` are named below. A row is first order when it changes the estimand's projection, the limit of the estimator or
the reported variance, and second order or numerical when it changes remainders or arithmetic only.

| topic | paper | `atmle` @8452808 | here | order |
| --- | --- | --- | --- | --- |
| CV form | Appendix C: select on the training fold, update on the validation fold, average the folds | none | `form="cv"`, App C option (b): one pooled step per direction and one pooled $\Pi$ fluctuation | none to compare |
| bias-model arm column $A$ | Sec 7 allows "unpenalizing certain covariates or basis functions" | unpenalized by default (`fit_cate(force_A = TRUE)`, `penalty_factor[force_basis_idx] <- 0`) | penalized; `force_arm=True` reproduces `atmle`'s default and makes `arm_hierarchy` a no-op | first |
| refit of the coefficients | relaxed refit, "eliminates the need to further target" $\tilde\Psi$ (Sec 7) | default `target_method = "tmle"`: one step from the lasso coefficients along $I^{-1}E[\phi]$; `"relaxed"` refits | relaxed refit, as in the paper | first, for `atmle`'s default |
| dictionary | relaxed HAL on $W$ and on $(W, A)$ (Alg 1 lines 5, 11) | `hal9001::enumerate_basis`, degree 3, first-order splines, 20 knots, then `filter_hal_basis` at support $1/\sqrt{n}$ | linear terms, hinges at 5 quantile knots and pairwise linear products, with the $k_{\min}$ screen | first when the truth needs hinge-by-hinge interactions |
| missing outcomes | App B: IPCW factor $\Delta/g^\Delta$ on both losses and the $Y$-parts of the influence functions | none: $\Delta$ fixed at 1 and unrecorded $Y$ set to 0 in `get_tau_S_eic` | AIPCW, which equals App B's IPCW when the outcome regression is zero | first, for efficiency and robustness |
| $\Pi$ fluctuation | submodel and loss not stated; Lemma 8 gives the score along $C$ | two parameters, one per arm (`target_Pi`, covariates $\tau_S(W, a)\,1\{A = a\}$, weighted by $A/g + (1 - A)/g_0$ with `target_gwt = TRUE`) | one parameter along the paper's clever covariate $C$ | second |
| CV variance | App C gives none; Theorem 2 normalizes by $P_0 D^2$ at each fold's working model | none | centred variance of $D_v$ over the validation rows, with $I^{-1}m$ estimated on every row from the targeted cross-fitted nuisances; the $T_v$ direction where the rows cannot estimate the Hessian | first, for the reported variance |
| pooled variance | Alg 1 line 18: $P_n D^2 / n$, uncentred | `var(eic_A - eic_S) / n`, centred (`inference`) | uncentred | numerical, $(P_n D)^2$ |
| stopping rule | not stated | $10^{-4}\,\mathrm{sd}(D) / (\sqrt{n}\log n)$ or 50 passes (`run`) | $\mathrm{sd}(D) / (\sqrt{n}\log n)$ or 50 passes | numerical |
| outer folds | not stated | stratified by $S$, and by $S \times Y$ for a binary outcome; 3 to 20 folds by $n$ | five folds, stratified by $S$ and by $S \times Y$ for a binary outcome; a supplied `split_plan` is kept as given | second |

Four more differences are second order or numerical and kept on purpose. First, with one learner
`atmle` predicts its nuisances in sample, and with a stack the stack's predictions are out of fold;
the paper cross-fits, and so does this package always. Second, `atmle` clips $\Pi$ at
$[10^{-3}, 1 - 10^{-3}]$ and bounds the denominators of its pseudo-outcomes, because it divides by
$A - g$ and $S - \Pi$; here `g_bounds` clips the probabilities and the R-loss never divides.
Third, the information matrices are the empirical $E[(A - g)^2\phi\phi^\top]$ and
$E[(S - \Pi)^2\psi\psi^\top]$ in place of the model-based $E[g(1 - g)\phi\phi^\top]$ of Lemmas 4
and 7, which are equal in expectation given $W$ and make the score identity exact. Fourth, with
external controls only `atmle` fits $\Pi$ on every row and bounds $\Pi(W, 1)$ at $1 - 10^{-3}$,
where $\Pi(W, 1) = 1$ here.

**Why the defaults differ.** Forcing $A$ gives each external arm a free mean shift, so the external
rows carry no information on the effect's level and the efficiency bound is the trial-only one. On
law `a0` the exact-law standard deviation of $D_w$ is 0.48 of the trial-only bound with $A$
penalized and 1.00 with $A$ forced
(`tests/unit/test_adaptive_tmle_trial_design.py`). The paper permits either choice, so
`force_arm=False` is the default for the gain, and `force_arm=True` is the setting under which a
fit is comparable with `atmle`'s default. With external controls only the bias dictionary has no
arm column, and the setting has no effect.

**The dictionary.** The projection parameter $\Psi_w$ depends on the dictionary. When the true
conditional effect or bias needs hinge-by-hinge interactions that the linear, hinge and
pairwise-linear columns cannot span, this package and `atmle` estimate different projections, and
the oracle-approximation condition of Theorem 1 can fail here first. The default stays small for
stability at the sizes of the registered study. A highly adaptive lasso basis is outside this
library's scope. A user can pass a transformer of their own through `working_basis=` (the `basis=`
argument of the engine), which goes through the same support screen and rank steps.

**Reproducing `atmle`.** A comparison with the package must set `target_method = "relaxed"`,
`n_lambda = 1` and `force_A` equal to `force_arm`, and it must use a multi-learner stack or accept
that a single learner predicts in sample. Otherwise it measures `atmle`'s shrinkage or its
in-sample nuisances and not the method. The comparison is declared and not run, and it never
blocks the method.

## Evidence

Two registered studies and exact-law unit tests check the method. Study A, `adaptive-tmle-trial`,
has a committed result, recorded in the [method-evidence page](method-evidence/adaptive-tmle-trial.md).
Study S6, `adaptive-tmle-selective`, has a committed result, recorded in the
[study page](method-evidence/adaptive-tmle-selective.md). `s6_pooled` is inside every margin. `s6_cv`
covers in 0.937 of replications, and its bias is 0.0290 (0.20 error SDs), from a finite-sample term
of its targeting step with no defect. The repo owner accepted the study with that reading by a named
exception on 2026-11-10, and the study was not rerun. The label of every fit stays
experimental. No support row covers the method, and a row would not change the label: the
`working_model_projection` status keeps an estimate experimental inside any row until F39 closes
(see Inference status).

### Studies

`adaptive-tmle-trial` (`tests/studies/adaptive_tmle_trial.py`) draws $n = 4{,}000$ pooled rows (a
trial of 1,000 and 3,000 external rows, the paper's "3x") and fits `AdaptiveTMLEMethod` with
explicit spline learners, five outer and five inner folds. Each scenario except `b1y` gates one estimand with
the primary performance tests: a bias leg, a coverage floor and an SE sanity band.

| scenario | law | gated estimand |
| --- | --- | --- |
| `a0`, `a1a`, `a1b` | the paper's laws (a) with no bias and (b) with bias linear in $W$ | $\Psi$ |
| `a2c`, `a2d` | the paper's laws (c), (d), with bias outside the span | the working-model projection |
| `e1` | a separating law where the selected projection differs from $\Psi$ by at least three standard errors | the projection |
| `a3`, `a3_pi` | positivity stress; `a3_pi` misspecifies the $\Pi$ learner | $\Psi$ |
| `a4` | controls-only external data | $\Psi$ |
| `a5` | missing outcomes, with an external mechanism that depends on $Y$ | $\Psi$ |
| `b1y` | binary outcome, with a bias that is not in the span of the dictionary | none; a reported stress row |
| `h2_pooled`, `h2_trial` | a heterogeneous effect, so $\Psi \ne \Psi_2$ | $\Psi$, and $\Psi_2$ |
| `f1a`, `f1b` | `form="pooled"` on (a) and (b) | $\Psi$ |

`b1y` is declared as a reported stress row. The study publishes its coverage of $\Psi$, its
coverage of the projection, its bias and its SE ratio, and no verdict reads them. The guarantee of
the method is pointwise (Theorem 1 of the paper, Appendix C Theorem 2, and Theorems 5 to 7 of
van der Laan et al., arXiv:2307.12544v2): the interval covers $\Psi$ only when the working model approximates the
bias, the oracle condition. The external bias on this law passes through `expit`, which the
linear-plus-hinge dictionary does not span. The two external arms' biases differ by about one
standard error of that difference ($\Psi^\# = -0.0148$, about 1.3 standard errors). At
$n = 4{,}000$ the lasso then usually selects only the constant, and the projection misses $\Psi$ by
about the whole of $\Psi^\#$.

No uniformly valid interval can be much shorter than the robust one
at a zero difference: that is the Hodges and Le Cam super-efficiency limit, and it is the point
of the post-selection results of Leeb and Pötscher and of Armstrong and Kolesar (2018) on adaptive
intervals. A coverage shortfall on $\Psi$ under `b1y` is therefore the price of the narrower
interval where the bias is zero, not evidence of a defect. In the sizing pilot, `b1y` covered
$\Psi$ in 708 of 800 replicates, so a gate on it would have no replication budget that meets it.
The declared run covered $\Psi$ in 0.884 of 2,800 replicates and the projection in 0.942, with a
standardized bias of -0.46 and an SE ratio of 0.896.

The projection truth $\Psi_w(P_0)$ is computed per replicate by quadrature on each fold's stored
transformer and selected columns, so the record declares a truth that varies by replicate.

Controls ride on the same draws and must fail where they should:

| control | what it checks | expected |
| --- | --- | --- |
| efficiency contrast on `a0` | RMSE(A-TMLE) / RMSE(trial-only TMLE), upper bootstrap bound below one | gates. `a1a` and `a1b` report the ratio, because the gain needs the lasso to drop every arm-shift column |
| K1, naive pooled TMLE on `a1b` | treating external rows as exchangeable | $\Psi$ coverage below the floor |
| K2, unpenalized main-terms working model on `e1` | the status label | projection coverage passes, $\Psi$ coverage fails |
| K3, no $\Pi$ fluctuation on `a3_pi` | the targeting step | the targeted fit passes the coverage floor and the fit without the step falls wholly below it |
| K4, a ladder on `e1` at three sample sizes | the bias-to-SE ratio of $\Psi$ grows like $\sqrt{n}$ while the projection's stays bounded | reporting row |
| K5, $S/p$ replaced by one on `h2_pooled` | the $\Psi_2$ weights | $\Psi_2$ coverage fails |

`adaptive-tmle-selective` (`tests/studies/adaptive_tmle_selective.py`) is cell S6: selective
enrollment with a large wiggly bias, Super Learner nuisances (spline GLM, shallow boosting, lasso),
both forms, $n = 3{,}200$. Control K6, main-terms GLM nuisances on the same draws, is a reporting
row, because no cheap law is known on which main-terms nuisances fail at that size. Karim
(2026) reports the failure that S6 and K6 target: A-TMLE with main-terms GLM nuisances under
selective enrollment and a large wiggly bias undercovers, with coverage falling from 0.81 to 0.47
as the pooled sample grows from 500 to 3,200 rows, and flexible nuisances restore it. That is a
simulation caution from another paper, not a result of this library.

The replication count of each study is set by the repeated-sampling sizing rule at a pilot, and the
rule reads only gated cells. Study A uses an 800-replicate pilot of its 16 gated scenarios (0 lost
replicates, every control passed, the efficiency RMSE ratio on `a0` 0.703 with interval
[0.654, 0.754]), which gives 2,800 replications of each scenario. At 143.24 core-seconds per
replicate set that is about 111 core-hours. Study S6 uses a 1,200-replicate pilot (1,159 and
1,126 covered for the two forms, 0 lost), which gives 2,400 replications at 545.46 core-seconds per
set, about 364 core-hours. The declared red-cell reading is the one in the
[benchmarking strategy](../development/method-benchmarking.md#red-cells).

Study A ran as declared: 2,800 replications of each of 17 scenarios, no lost replicate. These are
the numbers of the fresh run of 2026-11-09 at c6f9e51c, made after the fix to the cross-validated
form's variance (the fixture README's "Declaration history" gives the reason).
All 16 gated tests passed, with coverage of the gated target from 0.931 to 0.953 and SE ratios from
0.946 to 1.023. Every control behaved as declared.

| item | result |
| --- | --- |
| study A, gated scenarios | 16 of 16 tests pass. Coverage 0.9314 to 0.9525, SE ratio 0.946 to 1.023, largest standardized bias 0.151 (the n = 1,000 ladder rung) |
| study A, reported row `b1y` | coverage of $\Psi$ 0.884, of the projection 0.942, standardized bias -0.46, SE ratio 0.896. Reported, not gated |
| efficiency contrast | RMSE ratio 0.677 on `a0` (99% interval 0.651 to 0.703), gated. 1.022 on `a1a` and 1.056 on `a1b`, reported |
| K1 to K3, K5 | naive pooled $\Psi$ coverage 0.846 on `a1b`; main-terms $\Psi$ coverage 0.0146 beside projection coverage 0.948 on `e1`; coverage 0.931 with the $\Pi$ step and 0.529 without it on `a3_pi`; pooled fit against $\Psi_2$ 0.0221 |
| K4 ladder on `e1` | bias over SE of $\Psi$ 1.80, 2.78 and 4.07 at n = 1,000, 2,000 and 4,000, as expected for a fixed projection offset, with projection coverage 0.935, 0.948 and 0.948 |
| study S6 | pending the declared run |
| R `atmle` pairing | pending; declared, not blocking |

The [study page](method-evidence/adaptive-tmle-trial.md) lists every row. The study declares no
property cell and no registered comparator, so the
[implementation validation grid](method-evidence/validation-grid.md) does not list it, and the
method has no grid row. Study S6 is declared the same way, under `reporting`. A property cell declared before a run,
or the R `atmle` pairing record, would give the method a row.

### Unit tests

| test | what it checks | what it cannot see |
| --- | --- | --- |
| `tests/unit/test_atmle_exact.py` | on finite laws with the dictionary outside the truth's span: Lemma 2, the saturated dictionary projecting to the trial effect, the influence function as a Gateaux derivative, each influence component being needed, the remainder as a second-order product, first-order remainders for a trial treatment mechanism in the covariate and for an inverse-weighted residual, Lemma 1, and the $\Pi$ fluctuation solving the source score | the truth at a fitted learner |
| `tests/unit/test_atmle_engine.py` | the lasso against its optimality conditions and an independent solver, the refit and fluctuation score equations, full rank and the support screen, the rows each fold reads, the penalized $A$ column with a nonzero witness, the arm column kept beside a selected arm product with a mutation control, the CV variance against a row-by-row computation of its directions and fold variances, with a mutation to the training directions that moves the standard error and not the estimate, and the guard's fallback | coverage |
| `tests/unit/test_atmle_refusals.py` | every refusal text and code, with a spy that proves no engine call | none |
| `tests/unit/test_atmle_workflow.py` | the fit through `CausalStudy` equals the engine on the same arrays, the status, the controls-only and few-treated cases, and `save` and `load` | none |
| `tests/unit/test_adaptive_tmle_trial_design.py` | every declared constant of the studies: truths, separations, control powers and sizing, recomputed without a run | estimator behaviour |

Each exact-law test pins a nonzero witness against a value that the test computes from the law, so
a deleted sign, mask or counterfactual block fails at least one of them.

## Reading list

- van der Laan, Qiu, Tarp and van der Laan, J. Causal Inference 14(1):20240025,
  doi:10.1515/jci-2024-0025 (arXiv:2405.07186v2): Sec 2.4 for $\Psi$, Lemma 2, Algorithm 1,
  Theorems 1 and 2, Appendix B (missing outcomes) and Appendix C (CV form).
- [References](../references.md) lists the entry with the other sources.
