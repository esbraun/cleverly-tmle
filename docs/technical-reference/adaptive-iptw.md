# Adaptive-propensity IPTW

## What this solves

`AdaptiveIPTW` reports the Horvitz-Thompson mean of each treatment arm. It tilts the fitted
propensity along the reduced regression of the outcome on that propensity. The estimator fits no
outcome regression. `AdaptiveIPTWMethod` selects it through `CausalStudy`.

The estimator reports a point estimate for every supported fit. It reports a standard error, an
interval and a p-value only when the treatment mechanism is known. [When an interval is
reported](#when-an-interval-is-reported) gives the reason.

| fit | point estimate | standard error, interval, p-value |
| --- | --- | --- |
| complete data, treatment probabilities declared with `treatment_probabilities=` | yes | yes, from the influence curve |
| complete data, treatment mechanism estimated | yes | no. The status is `estimated_mechanism_plugin` |
| missing outcome or missing treatment | yes | no. The response factor is estimated |
| complete-data `ey_obs` alone | yes, the sample mean | yes. The estimate reads no mechanism |

The example fits one estimator twice on one sample. The first fit estimates the mechanism. The
second fit declares it.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import AdaptiveIPTW

rng = np.random.default_rng(0)
n = 600
w = rng.normal(size=n)
g = 1 / (1 + np.exp(-0.8 * w))
a = rng.binomial(1, g)
y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * a + 0.7 * w))))
frame = pd.DataFrame({"W": w, "A": a, "Y": y})

estimator = AdaptiveIPTW(
    treatment_learner=LogisticRegression(),
    reduced_outcome_learner=LinearRegression(),
    estimands=["ey1", "ey0", "ate"],
    cross_fit=False,
    simultaneous=False,
    random_state=0,
)
columns = dict(outcome="Y", treatment="A", covariates=["W"])
estimated = estimator.fit(frame, **columns).single()
known = estimator.fit(
    frame, treatment_probabilities=np.column_stack([1 - g, g]), **columns
).single()
print(estimated.inference_status, known.inference_status)
```

The first fit prints `estimated_mechanism_plugin`. Its `std_error`, `ci` and `pvalue` raise
`CapabilityError`. The second fit prints `influence_curve`. Pass explicit learners in a fit. The
default learner library takes 30 to 120 seconds per fit.

## The estimator

A key is one targeting problem with an indicator $I$, a mechanism $g$ and an outcome $T$. Each key
has its own tilt, reduced regression and score equation. Baseline strata add one of each per
stratum.

| key kind | indicator $I$ | outcome $T$ | mechanism $g$ |
| --- | --- | --- | --- |
| arm | $1\{A = a\}$ | $Y$ | $g(a \mid W)$ |
| composite arm, with `delta=` or `treatment_delta=` | $\Delta_A \Delta 1\{A = a\}$ | $Y$ | $P(\Delta_A = 1 \mid W) P(A = a \mid \Delta_A = 1, W) \pi(a, W)$ |
| deterministic regime $d$ | $1\{A = d(W)\}$ | $Y$ | $\sum_a g(a \mid W) 1\{d(W) = a\}$ |
| stochastic regime, one key per regime and arm | $1\{A = a\}$ | $g^*(a \mid W) Y$ | $g(a \mid W)$ |
| natural course, with `delta=` | $\Delta$ | $Y$ | $\pi(A, W) = P(\Delta = 1 \mid A, W)$ |

A regime key needs a density that is known before the fit. A density that is 0 or 1 on every row
takes the deterministic key. Any other known density takes the stochastic key.

For each key, with weights $w$ and the fitted mechanism $g$ clipped to `g_bounds`, the loop runs
in this order.

1. Fit the reduced regression $Q_r = E(T \mid I = 1, g)$ on the rows with $I = 1$. Set
   $H = Q_r / g$ and $D_g = H (I - g)$.
2. Compute the one-step value $P_n[w I T / g] / P_n[w] - P_n[w D_g] / P_n[w]$ at these initial
   nuisances.
3. Tilt the mechanism by $\operatorname{logit} g^* = \operatorname{logit} g + \epsilon H$. Solve
   $\epsilon$ by weighted logistic maximum likelihood on $I$. The solver clips $g^*$ to `g_bounds`
   at every step.
4. Refit $Q_r$ on $g^*$. Repeat steps 3 and 4 until the relative score $P_n[w D_{g^*}]$ meets
   `tol`, or `max_iter` rounds pass.
5. Report $P_n[w I T / g^*] / P_n[w]$ with the influence curve at the refitted $Q_r$ and $g^*$.

`form="targeted"` reports the value of step 5. `form="one_step"` reports the value of step 2.
A fit computes both forms. `result.extra["adaptive_iptw"]` records the other form, the untargeted
Horvitz-Thompson value, the tilt steps, the rounds and the stop reason.

Contrasts, ratios, `ey_obs`, `par` and `paf` come from the shipped target machinery over the arm
means. A Horvitz-Thompson mean is not a substitution estimator. A binary-outcome mean can leave
$(0, 1)$.

A request for `rr`, `or` or `paf` raises `DataError` after targeting when a mean it
needs leaves its domain. The test has a margin of `tol` times the summed score scale of the arm's
keys. For an arm whose observed outcomes are all 1, the score is the mean minus 1. So the fit does
not resolve that mean closer to 1 than the margin. The default report and `estimands="all"` drop
that target instead, and the fit warns with `DataWarning`.

## Contract

| item | contract |
| --- | --- |
| base result | van der Laan (2012), WP302, Theorem 1, Section 2, pp. 5-6, with Lemmas 1 and 2, pp. 27-28. The treatment-specific mean is $\psi_0 = E_0 E_0(Y \mid A = 1, W)$. The IPTW functional is $\Psi(g) = P_n D(g)$ with $D(g) = Y I(A = 1) / g(1 \mid W)$ |
| reduced regression | $Q_{r,0} = E_0(Y \mid A = 1, g_0(W))$. The estimator $Q_{r,n}$ regresses $Y$ on $g_n(W)$ among the rows with $A = 1$, and treats $g_n$ as a fixed covariate |
| targeting | WP302 fits one logistic step with the covariate $H_n = Q_{r,n} / g_n$. The loop above refits $Q_r$ on $g^*$ and steps again. Lemma 2 states its result for any $H_n$ consistent for $H_0$ and any $g^*_n$ that solves $P_n D_{H_n}(g^*_n) = 0$. So the converged loop is covered |
| influence curve | $IC = D(g_0) - \psi_0 - H_0 (A - g_0)$, evaluated at the final nuisances. The sign of the last term is minus. [The three defects of WP302](#three-defects-of-wp302) record the printed sign |
| one-step form | $\psi_{os} = P_n D(g_n) - P_n D_{H_n}(g_n)$ at the initial $(g_n, Q_{r,n})$. WP302 does not state it. The `drtmle` vignette, lines 1317-1323, and `R/adaptive_iptw.R`, lines 208-215, give the form. The exact identity $P_0 D(g) - P_0 D_{Q_r/g}(g) - \psi_0 = P_0[(Q_{r,0,g} - Q_r)(g_0 - g)/g]$ holds, with $Q_{r,0,g} = E_0(Y \mid A = 1, g_0(W), g(W))$. The remainder is second order under H1, and identically 0 under H2. Otherwise it carries the linear term of [the validity condition](#the-validity-condition), so the one-step form takes the same H2 rule |
| data-adaptive choices | Theorem 1 takes $g_n$ as any estimator. It names the super learner with cross-validated weights as its example, on pp. 4-7. So learner selection inside $g_n$ supports the point estimate. The interval needs the condition below |

The steps from Theorem 1 to each key kind are standard steps of the roadmap
[Eligibility](../roadmap.md#eligibility) table. [Natural-extension
verdicts](natural-extension-verdicts.md) records each step. The table states the step, and the
consequence for the interval.

| step | statement | point estimate | interval |
| --- | --- | --- | --- |
| S1 | an indicator reduction to $1\{A = a\}$ for each arm | all arms | known mechanism only |
| S2 | a fixed-dimension stack of the arm curves. Contrasts, ratios, `par` and `paf` follow by linearity and the delta method. Complete-data `ey_obs` is the sample mean with curve $Y - EY$ | all targets | known mechanism only |
| S3 | the composite indicator $\Delta_A \Delta 1\{A = a\}$ with the composite mechanism, under the missing-at-random identification of the composite contract | arm keys, in sample and cross-fitted | none. The response factor is estimated |
| S4 | a known mechanism: Theorem 1 at $g_n = g_0$. The tilt is $O_P(n^{-1/2})$ and the remainders are $o_P(n^{-1/2})$ under C2 | complete data | yes |
| S5 | cross-fitting. Theorem 1 assumes a Donsker class. With `cross_fit=True`, fold $k$ fits its own $Q_r$ on designs from mechanism models that never saw fold $k$, and one pooled $\epsilon$ tilts every fold | all keys with a point estimate | known mechanism only, with the $\epsilon$-path condition below open |
| S6 | observation weights: Theorem 1 on the weight-tilted law, by the chain rule for a known weight. The curve is $(w / P_n w)(D - \psi - D_g)$ | all keys | known mechanism only |
| S7 | clusters: the same result with independent clusters as the units and cluster-summed curves | all keys | known mechanism only, with the shipped few-cluster rules |
| S8 | baseline strata: Theorem 1 inside each fixed stratum, then a stack | all keys | known mechanism only |
| S9 | a deterministic regime $d$: $B = 1\{A = d(W)\}$ is a binary treatment with mechanism $g_d(W)$. Theorem 1 applies to $(W, B, Y)$ | in sample and cross-fitted on complete data. In sample with missing data | known mechanism, complete data |
| S10 | the natural-course mean: $\Delta$ is the treatment, $(A, W)$ is the covariate, and $\pi(A, W)$ is the mechanism | with `delta=`, in sample and cross-fitted | none. $\pi$ is estimated |
| S11 | a known stochastic regime: one key per arm with $T = g^*(a \mid W) Y$, summed by linearity | as S9 | known mechanism, complete data |

S5 uses a fold-wise construction. Fold $k$ fits its reduced regression $Q_{r,k}$ on the rows of
the other folds. At a row of fold $m \ne k$, the design is $g^{(-k,-m)}(W)$, from a mechanism
model fitted without folds $k$ and $m$. At the rows of fold $k$, the design is the production
$g^{(-k)}(W)$. So no row of fold $k$ reaches $Q_{r,k}$ through a generated design. Term [B] of
[DR-TMLE targeting](dr-tmle/targeting.md) is the term that this design removes.

The targeting solves one pooled $\epsilon$ at each row's own fold. The step then moves each fold's
designs along that fold's own covariate $Q_{r,k} / g_k$, from the first step on. This is the
cross-validated TMLE shape: fold-specific initial fits and one pooled fluctuation (Zheng and van
der Laan (2011), Sections 2 and 2.1, pages 2 to 6).

After the first step, fold $k$ reaches
$Q_{r,k}$ only through the steps $\epsilon_1, \dots, \epsilon_r$, at most `max_iter` of them,
one coordinate for each step and block. The interval argument then needs the reduced fit to be
continuous in those steps for a fixed training set. That is condition (S) of the DR-TMLE page,
restricted to that finite-dimensional path. This page does not prove it, so it stays
open. A key with an estimated mechanism reports a point only. Point consistency needs the
mechanism and $Q_r$ to converge.

The leave-two-out designs cost $K(K - 1)$ further fits of each estimated mechanism factor, one for
each ordered pair of folds. The default `n_folds=10` gives 90. The construction needs at least
three folds, so a cross-fitted fit with `n_folds=2` is refused (R21).

A drawn split in which two
folds hold every trainable row of a class that an estimated factor needs is refused with a
`DataError` before any learner runs. The classes are each arm, both values of $\Delta$ and both
values of $\Delta_A$, counted over the rows outside the pair. More folds do not help, so use more
data or `cross_fit=False`. The X21 run-1 diagnosis
measured the pooled design that this replaced. On law WT at $n = 2{,}000$, its bias was about 0.4
SD. That diagnosis is planning evidence outside the repository.
`tests/unit/test_adaptive_iptw_foldwise.py` pins the construction.

The inherited conditions are as follows.

| id | condition |
| --- | --- |
| C1 | $D(g^*_n)$ and $D_{H_n}(g^*_n)$ lie in a $P_0$-Donsker class with probability tending to 1 in sample, or (E) and (S) hold under `cross_fit=True` |
| C2 | $P_0\{D(g^*_n) - D(g_0)\}^2 = o_P(1)$ and $P_0\{D_{H_n}(g^*_n) - D_{H_0}(g_0)\}^2 = o_P(1)$ |
| C3 | the three second-order terms of WP302 are $o_P(n^{-1/2})$. The middle term does not follow from any rate. [When an interval is reported](#when-an-interval-is-reported) treats it |
| C4 | the targeted score $P_n D_{H_n}(g^*_n)$ is $o_P(n^{-1/2})$. The fit's score check reports it |
| C5 | identification: consistency, no unmeasured confounding and positivity. S3 also needs the missing-at-random condition for $\Delta_A$ and $\Delta$ |

A wrong $g_n$ makes the estimator inconsistent, because the estimator has single robustness. Suppose
$Q_{r,n}$ does not converge to $Q_{r,0}$. A first-order term $P_0[(H_0 - H_\infty)(g_n - g_0)]$
then remains, and the reported curve omits it. The one-step standard error uses
the untargeted $g_n$, so its error is first order in $\lVert g_n - g_0 \rVert$.

`reduced_outcome_learner=` fits $Q_r$. When it is `None`, the fit uses `outcome_learner=` and then
the default regression Super Learner. A selecting learner is outside conditions (E) and (S) under
the default `cross_fit=True`. Use a fixed-basis learner of the one-dimensional covariate $g$. A
cross-fitted fit refits $Q_r$ once per fold, round and key. With an estimated mechanism, it also
fits the $K(K - 1)$ leave-two-out mechanism models of S5 once per fit.

## When an interval is reported

A fit reports `std_error`, `ci` and `pvalue` when the whole treatment mechanism of every reported
key is known. This is the H2 rule. A fit with any other key takes the status
`estimated_mechanism_plugin` for every estimate.

### The validity condition

Theorem 1 of WP302 carries a negligibility condition. It defines
$Q_{r,0,n} = E_0(Y \mid A = 1, g_0(W), g_n(W))$ at the foot of p. 5. The second of three displays
at the top of p. 6 assumes $P_0 (Q_{r,0,n} - Q_{r,0}) (g_0 - g_n) / g_0 = o_P(n^{-1/2})$.
WP302 numbers no equations. The
derivation in four steps shows that no convergence rate satisfies the condition.

1. For any fixed $g$, $P_0 D(g) - \psi_0 = P_0[Q_0 (g_0 - g) / g]$.
2. Linearize the targeted score at $g_0$. The $Q_{r,0}$ part of $Q_0$ cancels. The error is then
   $\psi^*_n - \psi_0 = (P_n - P_0) IC - P_0[(Q_0 - Q_{r,0})(g^*_n - g_0) / g_0] + R_n$. The
   remainder $R_n$ has order $\lVert g^*_n - g_0 \rVert^2 + \lVert g^*_n - g_0 \rVert \lVert H_n - H_0 \rVert$.
3. The ratio $(g_0 - g_n) / g_0$ is a function of $(g_0(W), g_n(W))$. So the term of WP302 equals
   exactly $P_0[(Q_0 - Q_{r,0})(g_0 - g_n) / g_0]$. This term is linear in $g_n - g_0$.
4. For $\operatorname{logit} g_n = \operatorname{logit} g_0 + \delta_n h$, the term equals
   $-\delta_n E[(Q_0 - Q_{r,0})(1 - g_0) h] + O(\delta_n^2)$.

A linear term is $o_P(n^{-1/2})$ only if it vanishes. The term vanishes in two cases.

| case | statement | who can guarantee it |
| --- | --- | --- |
| H1 | $Q_0 = Q_{r,0}$. The outcome regression is a function of the true $g_0$ | no fit can check it |
| H2 | $g_n$ is a function of $g_0$. The term is identically 0 when the whole mechanism is known | the design, when the fit declares the mechanism |

A parametric $g$ with design $X$ also satisfies the condition when
$c = E[(Q_0 - Q_{r,0})(1 - g_0) X] = 0$. No fit can check that. A saturated fit on a finite support
is a function of $W$, not of $g_0$, so it is not H2.

When the term does not vanish, the effect depends on the learner.

| mechanism | effect on the Theorem 1 interval | source |
| --- | --- | --- |
| correct parametric $g$ | the curve is $IC - c' I^{-1} S$, the residual after projection on the GLM score. The Theorem 1 standard error is conservative | a planning simulation outside the repository. The study cell `spread/parametric_q_ne_qr` (law VL1, logistic $g$) reports the same ratio |
| a learner whose error breaks the level sets of $g_0$, with error of order $r_n$ | a bias of order $r_n$ at first order | a planning simulation outside the repository. The study cell `bias/ey1__level_set_breaking` (law VL1) measures the bias as a control |

Williamson, Forbes and White (2014), Section 3.3, show that for plain IPTW with a parametric
propensity, a sandwich variance which ignores the estimated propensity is typically conservative. Ertefaie, Hejazi and van der Laan, arXiv:2005.11303v2, p. 2, say that the targeted IPW
estimator "has been shown to suffer from issues of irregularity".
Benkeser, Cai and van der Laan, arXiv:1901.05056v1, Appendix F, condition (v), treat the
mirror-image term of the doubly robust case. They assume smoothness for it explicitly.
[The references](../references.md#adaptive-propensity-iptw-adaptiveiptw) give the locators.

### The status of a fit

The estimator decides the status from the configuration and the names that a call reports. It
reads no fitted quantity. A key has a known mechanism only on complete data that declares the
treatment probabilities. These keys have an estimated mechanism.

| key | reason |
| --- | --- |
| a key with an estimated $g$ | the learner fits $g$ |
| a composite key, with or without a known $g$ | a known $g$ times an estimated response factor is not H2 |
| the natural-course key with `delta=` | $\pi$ is estimated |
| any key of a fit with `treatment_covariates=` | the mechanism fits on a subset of the covariates |

A complete-data `ey_obs` reads no key. A fit that reports `ey_obs` alone keeps `influence_curve`,
also with weights declared estimated. A fit that reports `ey_obs` beside an estimated key loses
its interval, because a fit has one status. A call to `retarget` decides the status from the names
that it reports.

A fit with an estimated mechanism and estimated weights takes
`estimated_mechanism_plugin`, because a withheld status outranks a shown one. A fit with a known
mechanism, complete data and weights declared estimated takes `estimated_weight_plugin`. Its
interval is shown under the `experimental` label. The reduced regression of the weighted law holds
the weights fixed. `few_cluster_plugin` resolves through the shipped precedence rule.

A user who wants the intervals of the known-mechanism keys fits those keys on their own. The
fit passes `treatment_probabilities=` and reports nothing that reads an estimated mechanism.

### What a withheld fit carries

The point estimates stand. Both forms and the untargeted value stay in
`result.extra["adaptive_iptw"]`. `std_error`, `ci` and `pvalue` raise `CapabilityError` with the
reason of the status. `plugin_std_error` and `plugin_interval` return the spread of the Theorem 1
curve. That spread is a diagnostic. It is not valid inference here, for the reasons in the table
above.

The summary prints the column `estimated-mechanism se`, and the fit builds no simultaneous
band. [Inference status](inference.md#inference-status) lists the status beside its siblings.
The roadmap item [X36](../roadmap.md#x36-projected-variance-for-adaptiveiptw-with-a-declared-parametric-mechanism)
holds the projected variance for a declared parametric mechanism.

### Three defects of WP302

| defect | where | correction |
| --- | --- | --- |
| the influence curve prints the term $H_0 (A - g_0)$ with a plus sign | the Theorem 1 display on p. 6 and Section 2.1 on p. 7 | the sign is minus. The proof preamble on p. 26 gives $P_0(D(g_n) - D(g_0)) \approx -(\Psi^r_1(g_n) - \Psi^r_1(g_0))$, and Lemma 2 gives the difference. WP302 p. 7 says the curve is the efficient influence function of $E Y(1)$ for the reduced data $(g_0(W), A, Y)$, and that function has the minus sign. `drtmle` `cov.t` uses `eval_Diptw - eval_Diptw_g`. `tests/unit/test_adaptive_iptw_exact.py::TestW3CurveSign` pins the fitted curve SD at the closed-form minus value (arm 1, 0.7623) and away from the plus value (1.228) |
| the gloss on p. 6 says that the second and third conditions again require $g_n$ to converge at rate $o(n^{-1/4})$ | p. 6, after the displays | false for the second condition. The term is linear in $g_n - g_0$, and no rate makes it small |
| Section 2.1 claims that a correct parametric model in the library makes all the consistency and second-order conditions hold | p. 7 | false when $Q_0 \ne Q_{r,0}$. The residual after projection on the score is not the Theorem 1 curve |

The `drtmle` documentation calls `iptw_os` "not recommended" in the docstring of `R/confint.R`,
lines 179-181. The vignette says only that `iptw_tmle` is expected to behave better in finite
samples, "though further study is needed" (lines 1345-1348).

The journal version of WP302 was not available to this project. This page cites the working paper
only.

## Result surface

A refused call raises `CapabilityError` with a `Refusal` record. The row `capability(operation)`
agrees with the call. `tests/unit/test_adaptive_iptw_surface.py` calls every public method of the
result on nine fits.

| surface | status |
| --- | --- |
| `estimate`, `estimates`, `influence_curves`, `covariance`, `contrast`, `to_frame`, `summary`, `save`, `load`, `inference_status` | supported |
| `nuisance`, `fluctuations` | supported. The outcome accessors raise a named `CapabilityError`: the fit has no outcome regression |
| `diagnostics.support`, `nuisance_models`, `score_equations` | supported. The outcome row reads "not fitted" |
| `sensitivity.evalue` | supported on a fit that reports inference. Refused on a fit with an estimated mechanism |
| `validate` | refused: `api_redirection` to `diagnostics.nuisance_models` |
| `diagnostics.truncation_curve` | refused: `implementation_gap`, [X81](../roadmap.md#x81-adaptiveiptw-follow-ups) |
| `diagnostics.refute` | refused: `implementation_gap`, [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) part I |
| `sensitivity.omitted_confounding`, `robustness_value`, `elements`, `benchmark`, `contour` | refused: `api_redirection` to TMLE. The bound needs an outcome regression |
| `sensitivity.simulated_confounding` | refused: `implementation_gap`, [X82](../roadmap.md#x82-refutation-and-sensitivity-checks-for-every-estimator) part I |
| `sensitivity.missingness`, `tipping_gamma` | refused: `api_redirection` to TMLE |

## Differences from `drtmle`

| id | `drtmle` 1.1.2 | `AdaptiveIPTW` |
| --- | --- | --- |
| D1 | one fluctuation per arm, so $g^*_0 + g^*_1 \ne 1$ | the same at every arm count |
| D2 | `tolg` floors an estimated $g$ and trims the offset. It does not clip a supplied $g$ | `g_bounds` clips both sides at every step. The per-key solver returns the unconstrained root bit for bit when no bound binds |
| D3 | stops at `max abs(P_n D_g) <= tolIC`, with `maxIter = 2` | the shipped relative score rule and `max_iter` |
| D4 | `stats::cov(D) / n`, with denominator $n - 1$ | the shipped `make_estimate` convention. The standard error ratio is $\sqrt{n / (n - 1)}$ |
| D5 | the `glm_Qr` branch keeps rows with `A == a & DeltaY == 1` | $\Delta_A \Delta 1\{A = a\}$. The two agree, because $A$ is missing where $\Delta_A = 0$ |
| D6 | the fluctuation retries without a start, and sets $\epsilon = 0$ if it fails | the shipped solver, which records the stop reason |
| D7 | a constant design gives $Q_r$ as the arm mean | the same. No learner runs |
| D8 | `ci` and `wald_test` report the Theorem 1 interval for an estimated `gn` | no interval unless the mechanism is known |

## Refusals

[Scope and refusals](scope-and-refusals.md#adaptiveiptw-refusals) lists each refusal with its
code. Every refusal is raised before any learner fits, except the range rule. A spy learner shows
zero `fit` calls in `tests/unit/test_adaptive_iptw_refusals.py`.

## Validation

A registered validation study of `AdaptiveIPTW` is pending. No row of the [implementation
validation grid](method-evidence/validation-grid.md) covers it. Every estimate carries the
`experimental` label, with the reason that no registered study covers its configuration. The
evidence today is Python property evidence from the exact-law tests, and the comparator runs
against `drtmle` are pending.

| instrument | what it checks | what it cannot see |
| --- | --- | --- |
| `tests/unit/test_adaptive_iptw_exact.py` | on exact finite laws, the shipped value, tilt steps and score equal values computed in the test from the law tables, for the arm keys, the one-step form, the weights, the bound and the constant design. A GLM reduced regression separates the refit from the one-step | the truth at a fitted learner |
| `tests/unit/test_adaptive_iptw_composite.py` | the composite mechanism and its targeting equal the test-side product, in sample with wrong factors and cross-fitted, and the cross-fitted training designs are leave-two-folds-out | the $\epsilon$-path condition of S5 |
| `tests/unit/test_adaptive_iptw_untrainable.py` | a pair of folds that holds every row of an arm, of a $\Delta$ value or of a $\Delta_A$ value is refused with zero learner fits | none |
| `tests/unit/test_adaptive_iptw_foldwise.py` | fold $k$'s reduced regression ignores fold $k$'s treatment and outcome, at round 0 and on a design grid in every round with the $\epsilon$ history replayed, for the arm keys and for the natural-course key with $\Delta$. The shipped loop equals a test-side fold-wise loop, and an adaptive mechanism learner trains on leave-two-out designs. A pooled design fails each check | the $\epsilon$-path condition of S5 |
| `tests/unit/test_adaptive_iptw_regimes.py` | the deterministic, stochastic and natural-course keys, the strata and the clusters equal test-side values | cross-fitted regimes with missing data, which are refused |
| `tests/unit/test_adaptive_iptw_status.py` | each status of a construction, the withheld accessors, and the validity identity with a nonzero term | the size of the term at a fitted nuisance |
| `tests/unit/test_adaptive_iptw_refusals.py`, `_api.py`, `_surface.py`, `_support.py` | each refusal text and code, the keyword inventory, every public call, and the support key | coverage |

Each test pins a nonzero witness against a value that the test computes itself, or against a
count. A deleted sign, factor, mask or tilt therefore fails at least one of them.
