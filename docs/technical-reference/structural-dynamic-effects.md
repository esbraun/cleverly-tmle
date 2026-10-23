# Structural dynamic effects

## What this solves

`StructuralNestedMean` fits a linear structural nested mean model (SNMM) by cross-fitted
g-estimation. It reports the effect of each treatment node on the end-of-study outcome. It reports
that effect as a coefficient matrix, and it lets the effect vary with declared features. It also
reports the mean outcome at the reference level, and the mean and contrast of declared static
regimens.

The estimator is a separate method, not a second route to the `LTMLE` estimand. The coefficients
are the parameters of a restricted model. A regimen level or contrast has the same estimand as
the one `LTMLE` reports only when that model holds.

| question | estimator | what it assumes |
| --- | --- | --- |
| What is the effect of a dose or an arm at node $t$, with later treatment at the reference? | `StructuralNestedMean` | the linear blip model of this page |
| What is the mean outcome under a static regimen, with no model for the effect? | `LTMLE`, `SDR` | no structural model |
| What is the value of a rule or a policy? | `LTMLE`, `SDR`, `IPSI` | no structural model |

```python
from cleverly.datasets import make_longitudinal
from cleverly.learners import GLM
from cleverly.longitudinal import snmm

frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
result = snmm(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    regimens={"always": 1, "never": 0},
    reference="never",
    blip_features={1: ["W1"], 2: ["W1", "A1"]},
    outcome_learner=GLM(family="binomial"),
    treatment_learner=GLM(family="binomial"),
    n_folds=2,
    random_state=0,
)
print(result.summary())
print(result.blip(2))
print(result.check_model().summary())
```

`snmm()` is the one-call form. `result.blip(node)` returns the coefficient matrix of one node.
`result.check_model()` compares the structural contrasts with the contrasts of `LTMLE`.

## The model

Write $H_t$ for the history before the treatment $A_t$ of node $t$. The history holds the baseline
columns, the covariates through node $t$, and the earlier treatments. Write $Y(\bar a_t, 0)$ for the
outcome with treatment $a$ through node $t$ and the reference level afterwards. The blip is

$$
\gamma_t(h_t, a_t) = E[Y(\bar a_{t-1}, a_t, 0) - Y(\bar a_{t-1}, 0, 0) \mid H_t = h_t, A_t = a_t]
= c(a_t)^\top \Psi_t\, \phi_t(h_t).
$$

| symbol | meaning |
| --- | --- |
| $c(a_t)$ | the coding of the treatment. A binary node uses the indicator. A categorical node uses the indicators of the levels after the first sorted level. A continuous node uses the dose |
| $\phi_t$ | the declared feature map of node $t$. The intercept is always a feature |
| $\Psi_t$ | the matrix of constants that the fit estimates, with one row per coding component and one column per feature. `result.blip(t)` returns $\Psi_t^\top$, with one row per feature |
| $Z_t$ | $\phi_t(H_t) \otimes c(A_t)$. The vector $\psi_t$ is $\mathrm{vec}(\Psi_t)$. It lists one feature at a time, and each feature lists its coding components |

A feature $\phi_t$ may read the baseline columns, the covariates through node $t$, the earlier
treatments, and products of those. Result names have the form `blip[<treatment>:<component>:<feature>]`. The names list one coding
component at a time, and each component lists its features. The order of the names therefore
differs from the order of $\psi_t$ when a node has two or more components and two or more
features.

Three assumptions apply. `LTMLE` shares the first. The estimator adds the second. The third is the
positivity condition in the form that this estimator needs.

| assumption | statement |
| --- | --- |
| sequential exchangeability | the potential outcome $Y(\bar A_{t-1}, 0)$ is independent of $A_t$ given $H_t$. `LTMLE` assumes the same |
| linear blip | the equation above holds with the declared features and constant $\Psi_t$. `LTMLE` assumes no such model |
| average positivity | $E[\tilde Z_{t,t} \tilde Z_{t,t}^\top]$ is positive definite, with the residuals of the estimating equation below. The estimator inverts no probability, so it truncates none |

The model is a restriction. A blip that depends on an undeclared earlier treatment or covariate
makes the coefficients answer a different question. Laws L and P in the validation table show the
size of that error.

## Identification of the static level and contrast

Take a static regimen $a = (a_1, \dots, a_K)$ and a feature map $\phi_t$ that reads the baseline
columns and the earlier treatments only.

| step | statement | reason |
| --- | --- | --- |
| 1 | $Y(a) - Y(0) = \sum_t [Y(\bar a_t, 0) - Y(\bar a_{t-1}, 0)]$ | the telescoping sum |
| 2 | $E[Y(\bar a_t, 0) - Y(\bar a_{t-1}, 0) \mid H_t] = \gamma_t(H_t, a_t)$ | exchangeability at node $t$ removes the conditioning on $A_t$ |
| 3 | the mean of the term is $E[\gamma_t(H_t(\bar a_{t-1}), a_t)]$ | the iterated expectation over the history under the regimen. This is the static case of Lewis and Syrgkanis (2021), supplement Lemma 6 |
| 4 | the mean is $c(a_t)^\top \Psi_t\, E[\phi_t(X_0, \bar a_{t-1})]$ | the earlier treatments are the constants $\bar a_{t-1}$, and treatment does not affect the baseline columns |
| 5 | $E[Y - \sum_t \gamma_t(H_t, A_t)] = E[Y(0)]$ | backward induction over the nodes with consistency and the blip definition |

The result is

$$
E[Y(a)] = E[Y(0)] + \sum_t c(a_t)^\top \Psi_t\, \bar\phi_t(a), \qquad
\bar\phi_t(a) = E[\phi_t(X_0, \bar a_{t-1})].
$$

A contrast is the difference of two such levels. A feature that reads a covariate $L_t$ needs
$E[\phi_t(H_t(\bar a))]$ in step 4. That mean depends on the law of $L_t$ under the regimen. The
estimator does not estimate that law. [X72](../roadmap.md#x72-static-regimen-values-from-structural-blips-with-time-varying-features)
tracks the gap. Lewis and Syrgkanis (2021), arXiv version, Corollary 9, gives the inference for
that route.

A static dose level is regular only under this model. The nonparametric model does not make a
point dose pathwise differentiable. The estimator therefore requires each static dose and the
reference dose 0 to lie in the observed range of its node.

## The estimating equation

Each fit cross-fits these nuisances on the full history $H_t$, one set per node $t$.

| nuisance | definition | learner |
| --- | --- | --- |
| $q_t$ | $E[Y \mid H_t]$ | `outcome_learner` as a classifier for a binary outcome. `regression_learner` as a regression for any other outcome |
| $m_{j,t}$ for $j \ge t$ | $E[Z_j \mid H_t]$ | `treatment_learner` as a classifier for a target in $\{0, 1\}$. `regression_learner` as a regression for any other target |

`regression_learner` defaults to `outcome_learner`. A binary outcome with a dose node or a
continuous feature needs it, because a classifier takes no target outside $\{0, 1\}$.

For $j > t$ the feature map $\phi_j$ reads treatments that $H_t$ does not hold. The estimator
splits each feature into the factors that $H_t$ determines and the factors it does not. It
regresses the product of the undetermined factors and the coding on $H_t$. It then multiplies the
determined factors back in.

The residuals are $\tilde Y_t = Y - q_t(H_t)$ and $\tilde Z_{j,t} = Z_j - m_{j,t}(H_t)$. The score
of node $t$ is

$$
s_t(\psi) = \tilde Z_{t,t}\Big(\tilde Y_t - \sum_{j \ge t} \tilde Z_{j,t}^\top \psi_j\Big).
$$

The estimator solves $E_n[s_t] = 0$ from the last node to the first. Node $t$ uses the solved
$\psi_j$ for $j > t$. The validity argument is short. At the true $\psi$, the quantity
$Y - \sum_{j \ge t} Z_j^\top \psi_j$ has mean $E[Y(\bar A_{t-1}, 0) \mid H_t, A_t]$. By
exchangeability that mean does not depend on $A_t$. The centered score therefore has mean zero
given $(H_t, A_t)$, and $\tilde Z_{t,t}$ is a function of $(H_t, A_t)$.

Every nuisance is a regression of an observed variable on $H_t$. No nuisance is fitted on an
estimated index. The linear-remainder pattern therefore does not apply to this estimator. The
remainder of node $t$ at the true $\psi$ is

$$
R_t = E[\delta_{t,t}\, \delta_q] - \sum_{j \ge t} E[\delta_{t,t}\, \delta_{j,t}^\top]\, \psi_j,
$$

where $\delta_q$ and $\delta_{j,t}$ are the out-of-fold errors of $q_t$ and $m_{j,t}$. Every
term holds the error of the own-node treatment nuisance. The point is consistent when every
own-node treatment nuisance converges, even if $q_t$ does not. A correct $q_t$ alone is not enough.

The instrument $\tilde Z_{t,t}$ is unweighted. The locally efficient estimator of the model weights
it by the conditional variance of the outcome residual. This estimator is not that one. This page
states no ranking of its variance against the variance of `LTMLE`.
[F36](../roadmap.md#f36-locally-efficient-g-estimation-of-the-structural-nested-mean-model)
tracks the weighted estimator.

## Influence curves

The Jacobian $G$ is block upper triangular. Block $(t, j)$ for $j \ge t$ is
$E_n[w\, \tilde Z_{t,t} \tilde Z_{j,t}^\top]$. It is the derivative of the score of node $t$ in
$\psi_j$. Write $S_i$ for the stacked scores of unit $i$.

| functional | estimate | influence curve of unit $i$ |
| --- | --- | --- |
| $\psi$ | the backward solution | $G^{-1} S_i$ |
| $\mu_0 = E[Y(0)]$ | $E_n[Y - \sum_t Z_t^\top \psi_t]$ | $(u_i - \mu_0) - \sum_t E_n[Z_t]^\top \mathrm{IC}_{\psi_t,i}$, with $u_i = Y_i - \sum_t Z_{t,i}^\top \psi_t$ |
| level of $a$ | $\mu_0 + \sum_t c(a_t)^\top \Psi_t \bar\phi_t(a)$ | the curve of $\mu_0$, plus $\sum_t c(a_t)^\top \Psi_t(\phi_{t,i} - \bar\phi_t(a))$, plus $\sum_t (\bar\phi_t(a) \otimes c(a_t))^\top \mathrm{IC}_{\psi_t,i}$ |
| contrast | a difference of two levels | the difference of the two level curves. The $\mu_0$ terms cancel |

The estimator divides the weights by their mean before it builds these curves. Write
$\bar w_i = w_i / E_n[w]$. A known weight enters each mean and each regression. The curve of unit
$i$ carries the factor $\bar w_i$. The feature term is $\bar w_i(\phi_{t,i} - \bar\phi_t(a))$. With
clusters, the variance sums the curve within a cluster, and the cluster is the unit.

The variance is $\mathrm{Var}_n(\mathrm{IC}) / n$. For $\psi$ this is the sandwich
$G^{-1} \hat\Sigma G^{-\top} / n$ with $\hat\Sigma = E_n[S S^\top]$. The sample variance divides by
$n - 1$. This differs from the $1/n$ of EconML by the factor $n / (n - 1)$.

Two terms vanish at special laws. The feature term is zero when $\phi_t$ is constant. The term in
$E_n[Z_t]$ is zero when $E_n[Z_t] = \bar\phi_t(a) \otimes c(a_t)$. The validation table lists the
laws and the mutation controls that keep a nonzero witness for each.

## Status

| fit | status | what the result reports |
| --- | --- | --- |
| cross-fitted (`n_folds > 1`) | `influence_curve` | the curve and the intervals |
| `n_folds=1` and every learner a `GLM` | `influence_curve` | the curve and the intervals |
| `n_folds=1` and another learner | `adaptive_baseline_plugin` | the point estimate, and a diagnostic curve under `plugin_std_error` |
| `weights_estimated=True` | `estimated_weight_plugin` | the plug-in interval with the weights held fixed, under the experimental label |
| 3 to 19 clusters | `few_cluster_plugin` | the plug-in interval on a Student t reference with $J - 2$ degrees of freedom, under the experimental label, as for `LTMLE` |
| fewer than 3 clusters | `no_cluster_variance_plugin` | the point estimate, and a diagnostic curve under `plugin_std_error` |

The `influence_curve` status for a cross-fitted fit rests on Lewis and Syrgkanis (2021),
supplement Theorem 8. It asks that each nuisance converge faster than $n^{-1/4}$ and that average
positivity hold. The arXiv version, Theorem 4 and Theorem 8, states the product-rate form and the
variance estimator $\hat J^{-1} \hat\Sigma \hat J^{-\top}$. A `GLM` fit in sample has an interval
only when every `GLM` is correct. The mechanism-score terms of an in-sample fit are not built.
No source read gives the condition for an in-sample data-adaptive fit.

Every estimate carries the label "valid under the linear structural nested mean model with the
declared blip features". The label appears in the summary, in the `model` column of
`to_frame()`, and on every level and contrast row. It survives `save()` and `load()`. The status
says the interval is an influence-curve interval. The label says what the interval is valid for.

The few-cluster floor of 20 is inherited from `LTMLE`. No study measures it for this estimator.

### Support label

Every estimate carries `.support`, as the estimates of the other estimators do. The label is
`supported` only when a support row of `src/cleverly/_support_rows.py` covers the key of the fit.
`structural_key` reads the key before the first nuisance fit. The engine is `("snmm",)`, and the
family holds `end_of_study`, `blips` and, with a declared regimen, `regimens` and `static`.

The rows `G42` name this engine. Each row covers one key that the
[structural nested mean study](method-evidence/structural-nested-mean.md) fits. Every row is
cross-fitted, with a Gaussian outcome and no declared bounds. Each row licenses the point estimate
and the Wald interval. The study fits no simultaneous band, so a band reads `experimental`.

| row | key | learners (`outcome` and `treatment`) |
| --- | --- | --- |
| `G42.two_nodes` | two binary nodes, with static regimens. Laws `S2`, `H`, `P` and `M` | `GLM` |
| `G42.three_nodes` | three binary nodes. Law `S3` | `GLM` |
| `G42.categorical` | two nodes of three levels. Law `K` | `GLM` |
| `G42.dose` | two dose nodes. Law `C` | `GLM` |
| `G42.covariate_feature` | a feature that reads a time-varying covariate, with no regimen. Law `L` | `GLM` |
| `G42.weighted` | two binary nodes with known weights. Law `S2W` | `GLM` |
| `G42.clustered` | two binary nodes with 40 or more clusters. Law `S2C` | `GLM` |

Each row also requires that the fit sets no `regression_learner`. A fit outside these rows reads
`experimental`. Examples are an in-sample fit, a learner other than `GLM`, any
`regression_learner`, estimated weights and fewer than 40 clusters.

Law `N` of the study has no row. A row names a learner class and not its settings. The study
measures one early-stopped booster. A booster of fixed size kept a bias of 1.8 standard errors at
$n$ = 512,000. A row for the class would cover a configuration that the study refutes.

The reason of the label names the nearest row and the axes that differ. The summary prints a
`support` column, one `experimental:` note for each distinct reason, and the marker
`[experimental]` after the band. The frame carries `support` and `support_reason`. The label
survives `save()` and `load()`, and so does the key in `config.support_key`.

The label is separate from the status. A status that shows its interval adds its own reason to the
label. A status that withholds its interval raises on `ci`, as for the other estimators.
`tests/unit/test_structural_support.py` checks that a fit inside a `G42` row reads `supported`.
A fit that differs on one axis reads `experimental`.

## The model check

`check_model()` fits `LTMLE` on the same rows, weights, clusters and folds. Both estimators are
asymptotically linear, so the difference of the two contrast vectors is asymptotically normal.
Its mean is zero under the structural model. The check reports a joint Wald statistic on that
difference, with one degree of freedom per contrast.

A cross-fitted check refits both estimators on `n_splits` seeded sample splits (default 5). The
fit's own split is the first. The statistic reads the mean difference over the splits. Its
covariance is the mean cross-split covariance of the curves plus the between-split covariance
divided by `n_splits` (Chernozhukov et al. 2018, Section 3.4).

One split leaves split noise out of
the covariance. At two folds and `n=1,000` that check rejected a correct model in 18% of 1,000
draws. With five splits it rejected in 2.2% of 500 draws. `tests/unit/test_structural_model_check.py`
holds the size checks. An in-sample check uses one fit.

The covariance is an estimate, and the p-value accounts for its error. The check computes the
Welch-Satterthwaite degrees of freedom `reference_df` of each diagonal entry (Satterthwaite 1946).
The first-order part uses the empirical variance of its per-unit terms. The between-split part has
`n_splits - 1` degrees of freedom.

The check takes the smallest value `nu` over the `k` contrasts.
It reads `T (nu - k + 1) / (nu k)` against `F(k, nu - k + 1)`, the Hotelling law. When `nu` is at
most `k - 1`, the check reports no test. Bell and McCaffrey (2002) and Imbens and Kolesar (2016)
apply the same correction to a robust variance with high-leverage rows.

Under weak positivity a few rows carry large inverse weights and most of the covariance. The
table gives the rejection rate at the 5% level on the law `SP` of the X44 study. The fits use
five splits, and the model is correct. The `r01` regimen has the cumulative probability
0.009 in the history `W = 1, L2 = 0`.

| n | folds | draws | chi-squared reference | Hotelling reference | median `nu` | `LTMLE` coverage of `r01` |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 10 | 294 | 18.4% | 2.7% | 5.5 | not recorded |
| 2,000 | 2 | 600 | 16.0% | 4.8% | 8.6 | 0.857 |
| 2,000 | 10 | 400 | 16.3% | 6.2% | 7.8 | 0.862 |
| 8,000 | 2 | 400 | 11.0% | 5.5% | 21.4 | 0.880 |
| 8,000 | 10 | 150 | 7.3% | 3.3% | 21.2 | not recorded |

At good positivity the Hotelling reference is more conservative. On `S2` and `H` at `n = 1,000` and two folds, the
Hotelling reference rejected 2.0% of 400 and 1.0% of 300 draws. The chi-squared reference rejected
3.0% and 3.7%. On the negative control `LU` at `n = 2,000` both references rejected every draw
at two folds (200) and at ten folds (150).

The excess of the chi-squared reference comes from the `LTMLE` side. Its `r01` interval
under-covers in every recorded row, and its Monte Carlo SD is 1.18 and 1.20 times its mean standard error.
The mean covariance entry of the `r01` difference is 0.65 of its Monte Carlo variance at both
sizes and two folds. The structural `r01` interval covers 0.95 and 0.93.

The Hotelling reference corrects the
reference law of the check. It does not correct the `LTMLE` interval. At few effective degrees of
freedom the check has little power, and `reference_df` reports how few.

| item on `SP` | value |
| --- | --- |
| model-check fit order | order 2 for the outcome and the treatment. The node-2 outcome regression is therefore inexact. The size null still holds by double robustness, because the treatment regression is exact |
| difference standard error | about 11% above its sampling SD. The ratio is 1.114 (SE 0.039) with outcome order 3 and 1.127 (SE 0.040) with outcome order 2. Each reading uses 400 draws at `n = 2,000` and two folds |
| reading | a non-rejection on `SP` carries less information than the nominal level suggests |

| item | rule |
| --- | --- |
| scope | discrete static regimens. The fit and the nonparametric fit must show an interval |
| continuous node | refused with `invalid_declaration`. A static dose has no nonparametric counterpart |
| feature that reads a covariate | refused with `implementation_gap` and X72 |
| rank | when the smallest eigenvalue of the difference covariance is below `rank_floor` times the largest eigenvalue of the two contrast covariances, the check reports no test and names the reason |
| degrees of freedom | when `reference_df` is at most the number of contrasts less one, the check reports no test and names the reason |
| label | the check carries `.support`, the combination of the support of the two sets of contrasts it compares. The check is `supported` only when every contrast is supported on one row. No row covers the `LTMLE` side of a check on a Gaussian outcome with complete data, so such a check reads `experimental`. A fit with a shown status, such as `few_cluster_plugin`, runs the check under the experimental label |
| reading | a rejection says the model fails for these contrasts. A non-rejection does not confirm the model |

## Refusals

The rule of RM42b for the [experimental label](scope-and-refusals.md#experimental-compositions) is
that a composition which the code computes and no study covers runs and reads `experimental`.
A refusal stays for an invalid declaration, a different estimand, a gap with no code, and a case
that returns a misleading number. The last column classifies each refusal of this estimator.

| request | code | alternative | why it stays a refusal |
| --- | --- | --- | --- |
| a rule, a modified treatment policy or a known stochastic plan | `api_redirection` | `LTMLE` or `SDR` with the regimen | different estimand. The value needs the law of the history under the policy |
| an incremental policy | `api_redirection` | `IPSI` | different estimand. The policy depends on the observed treatment law |
| a censoring column, or a censored outcome | `api_redirection` | `LTMLE` or `SDR` | no code. The curve of the weighted g-estimator has derivative terms that this estimator does not build |
| a survival or competing-risk outcome | `api_redirection` | the survival paths of `LTMLE` | different estimand. The blip is a mean difference |
| a repeated outcome | `api_redirection` | `LTMLE` | no code. The estimator fits one end-of-study outcome |
| one baseline decision held over every node | `api_redirection` | `LTMLE` or `SDR` | invalid design. The later blips have no treatment variation to read |
| a declared treatment mechanism | `api_redirection` | `LTMLE` | no code. The estimator fits every treatment nuisance |
| `msm=` | `api_redirection` | `LTMLE(regimens, msm=...)` | no code. The blips are the structural model |
| a declared `markov_order` | `api_redirection` | omit `markov_order=`, or `LTMLE` or `SDR` | misleading. The Markov restriction names the regimen outcome regressions and the mechanisms. The g-estimation means $E[Y \mid H_t]$ and $E[Z_j \mid H_t]$ condition on the full history |
| a vector node with a continuous component | `invalid_declaration` | `LTMLE` or `SDR`, or separate nodes | no code. The blip of a flag with a dose has no coding |
| a static level, a contrast or `check_model` when a feature reads a covariate | `implementation_gap`, X72 | `LTMLE` or `SDR`, or features that read baseline columns and earlier treatments | no code. The level needs the law of the covariate under the regimen |
| a static dose outside the observed range of its node | `invalid_declaration` | a dose inside the range | misleading. The level is regular under the model only inside the range |
| a dose node whose observed range excludes the reference dose 0 | `invalid_declaration` | recode the dose so that 0 is observed | invalid declaration. The blip is the effect against dose 0 |
| a diagonal Jacobian block below `min_eigenvalue` | `invalid_declaration` | drop a feature, merge levels, or check positivity | invalid design. The blip is not identified |
| a feature that reads a later covariate, a later treatment, or an unknown column | `invalid_declaration` | a feature of the history through node $t$ | invalid declaration |
| `check_model` on a status that withholds its interval | `inference_gap` | cross-fit | no interval to compare. A shown status runs the check under the experimental label |

No refusal of this estimator is a composition that the code computes and no study covers, so the
classification of a fit collects no condition. The `Preflight` of the other estimators has no
counterpart here. Known weights, clusters, an in-sample fit, a dose node and a declared
regimen all run. A fit that no `G42` row covers reads `experimental`, and the
[support label](#support-label) says so.

`tests/unit/test_structural_nested_mean.py` (W7) pins the code of each refusal.

## Assessment and refutation

The result declares the `structural` assessment family. `cleverly.assessment.assessment_capabilities`
returns one row per diagnostic operation, and `result.sensitivity` returns one row per sensitivity
operation. Each row runs, or refuses with its reason code. Each `implementation_gap` refusal names
its roadmap item. The attribute `result.diagnostics` stays the fit's own positivity and learner
record, so the support and nuisance rows redirect there.

`cleverly.validation.refute` runs the placebo, `random_common_cause`, `subset`,
`bootstrap_stability`, and `negative_control_outcome` tests. The placebo and the negative control
read every blip coefficient and every regimen contrast against 0. The [refutation operations](validation-methods.md#refutation-operations) page gives
each draw, reference, and witness. The cell SN1 of the
[longitudinal refutation study](method-evidence/refuter-calibration-longitudinal.md) calibrates
the placebo, subset and negative-control rules on the regimen contrast. Its scope is
intercept-only blips with GLM nuisances, five folds, two nodes, and $n \ge 1000$. Outside that
scope, every test prints its numbers with `passed=None`.

The interval of this estimator has no linear-remainder exposure. Every nuisance regresses an
observed variable on $H_t$, and the remainder $R_t$ of [the estimating
equation](#the-estimating-equation) is a sum of products of nuisance errors. So a reading that uses
the fit's standard error, such as `subset` or the negative control, inherits no first-order bias
from a nuisance fitted on an estimated index. The theory admits a verdict for those two tests within
a calibrated scope. Cell SN1 registers one such scope.

## Comparator

The comparator is EconML 0.17.0 `DynamicDML`. It solves the same estimating equation with three
differences.

| item | `DynamicDML` | `StructuralNestedMean` |
| --- | --- | --- |
| conditioning set | one period row, the Markov state | the full history $H_t$ |
| effect modifiers | the first-period features only | baseline columns, earlier treatments and covariates |
| weights, clusters, censoring | weights ignored, no clusters, no censoring | known weights and clusters, no censoring |

The library and the test suite never install EconML. `tests/canonical/econml_dynamic_dml/` holds a
container, a hash-pinned `requirements.txt`, the generator, and a committed fixture of seven
cases. The fixture came from a virtual environment whose installed distributions equal those pins,
not from the container. It records that route under `environment`. Nobody has built the container. `tests/unit/test_structural_econml_parity.py` (W2) refits each case on the folds and
nuisances that EconML used.

| quantity | result |
| --- | --- |
| coefficients, binary, categorical and dose nodes, with and without features | equal to $10^{-8}$. The largest gap in the fixture is $10^{-14}$ |
| coefficient covariance, scalar nodes | equal to $10^{-10}$ after the factor $n/(n-1)$ |
| contrast error, scalar nodes, no features | equal to $10^{-9}$ after the same factor |
| coefficient covariance, categorical nodes | differs. EconML fills the off-diagonal Jacobian blocks of a vector coding in the transposed order |
| contrast error, categorical case | `cleverly` 0.1967 (on the EconML scale), EconML 0.1769, and a fixed-residual bootstrap 0.1956 |

The generator recomputes both sandwiches from EconML's own residuals. The corrected block is
$E[\tilde Z_{t,t} \tilde Z_{j,t}^\top]$, the derivative of equation $t$ in $\psi_j$. The two
orientations agree for a scalar coding. `test_w2_applying_the_transposition_to_cleverly_reproduces_the_econml_covariance`
shows that applying the transposition to the `cleverly` source reproduces the EconML covariance.

The [structural nested mean study](method-evidence/structural-nested-mean.md) runs EconML on
1,600 samples of each paired law. The 21 paired points agree to 2.4e-14. On law `k` the mean SE
ratio of EconML over `cleverly` is 0.9747 and 1.0306 for the two contrasts, from the transposed
blocks.

## Validation

The evidence label is "comparator-backed", against EconML 0.17.0 `DynamicDML`. The
[structural nested mean study](method-evidence/structural-nested-mean.md) gives the
repeated-sampling evidence, and the tests below give the exact-law evidence.

| id | what it checks | test |
| --- | --- | --- |
| W1 | on exact finite laws S2, S3, K, C, H and P, the coefficients, `ey_reference`, the levels and the contrasts equal the g-formula to $10^{-9}$ | `test_w1_exact_law_returns_the_truth` |
| W1 | the product nuisances equal the conditional means of the law | `test_w1_product_nuisances_equal_the_conditional_means_of_the_law` |
| W2 | the EconML parity fixture | `tests/unit/test_structural_econml_parity.py` |
| W3 | each curve is the derivative of the estimator in the observation weights, with the nuisances held fixed | `test_w3_curves_are_the_derivative_of_the_estimator_in_the_weights` |
| W4 | with too few declared features the structural contrast misses the regimen truth by a closed-form amount (laws L and P) | `test_w4_baseline_only_features_on_law_p_have_a_closed_form_limit`, `test_w4_an_undeclared_covariate_blip_has_a_closed_form_limit` |
| W5 | on law M the second treatment reads a covariate the period state omits, and the full history recovers the truth | `test_w5_the_full_history_removes_the_markov_restriction` |
| W6 | the structural label survives `save()` and `load()` | `test_w6_the_structural_label_survives_save_and_load` |
| W7 | each refusal carries its reason code, including the redirected designs, the dose and reference refusals, and the refusals of `check_model` | the `test_w7_*` tests |
| W8 | each standard error and the simultaneous bands sum the curve within a cluster. The data share a cluster effect, so a variance that ignores the clusters differs from the hand calculation | `test_w8_the_variance_sums_the_curve_within_a_cluster`, `test_w8_the_bands_and_the_joint_covariance_read_the_clusters`. A control that drops `cluster=` fails both: `test_w8_dropping_the_cluster_from_the_variance_fails_the_witness` |
| W9 | `check_model` reports the closed-form gap between the structural and the nonparametric contrasts on law L, and its covariance, degrees of freedom and p-value follow from the curves, at the cluster level | `test_check_model_statistic_on_exact_law_l`, `test_check_model_covariance_degrees_of_freedom_and_clusters` |
| W10 | the share of rows with an extreme class probability equals the mass of the extreme cell | `test_the_extreme_share_is_the_mass_of_the_cells_near_zero_or_one` |
| W11 | a fit inside `G42.two_nodes` reads `supported` and its band reads `experimental`. A fit that no row covers and the model check read `experimental`. The key covers a fit when a row names it, and the shown and withheld statuses keep their intervals | `tests/unit/test_structural_support.py` |
| M1 | dropping the adjustment for later blips fails W1 on S3 | `test_m1_dropping_the_later_blip_adjustment_misses_the_truth_by_its_limit` |
| M2 | residualizing on the period state fails W5 | `test_m2_a_period_state_treatment_nuisance_misses_by_the_limit` |
| M3 | a wrong reference level fails W1 on law K | `test_m3_the_wrong_reference_moves_the_level_blips_and_the_reference_mean` |
| M4 | transposed Jacobian blocks fail W3 on law K. A scalar law cannot see the mutation | `test_m4_transposed_jacobian_blocks_fail_the_curve_check_on_a_vector_node` |
| M5 | dropping the feature deviation term fails the contrast curve on law H | `test_m5_dropping_the_feature_mean_term_fails_the_contrast_curve` |
| M6 | dropping the $E_n[Z_t]$ derivative fails the level curve on law H | `test_m6_dropping_the_mean_of_z_derivative_fails_the_level_curve_only` |
| M7 | using the treatment mean in place of the product nuisance fails W1 on law P | `test_m7_the_treatment_mean_in_place_of_the_product_nuisance_fails_on_law_p` |

The exact-law tests give each saturated `GLM` the exact probabilities as weights. A saturated
learner then returns the population conditional means, so the fit returns the population solution.
The truths come from an independent g-formula in `tests/discrete_law_structural.py`.

The [structural nested mean study](method-evidence/structural-nested-mean.md) is registered.
All 61 primary tests, all 21 EconML truth tests, all 21 paired tests and all 237 gated property
cells pass. The model check has size cells on `S2`, `H` and `SP` and power cells on `LU`. The
`G42` support rows cover the keys that the study fits.

The X44 law `N` fits its nuisances with a gradient booster. The booster stops early on a validation
split. No general proof gives its rate. The claim that the rate suffices rests on a probe of this law.
At `n = 128,000` and `n = 512,000`, with 24 draws each, the bias over the standard error of each `N`
estimand is consistent with 0.

## Sources

| item | source |
| --- | --- |
| the model and the sequential exchangeability assumption | Lewis and Syrgkanis (2021), supplement, Appendix A, Assumption 1 and Assumption 2 |
| the identification of a policy value from the blips | Lewis and Syrgkanis (2021), supplement, Lemma 6, after Robins (2004) |
| the normal limit, with a linear blip | Lewis and Syrgkanis (2021), supplement, Theorem 8 |
| the variance estimator and product rates | Lewis and Syrgkanis (2021), arXiv version, Theorem 4 and Theorem 8 |
| the policy value inference for a covariate feature | Lewis and Syrgkanis (2021), arXiv version, Corollary 9 |
| the degrees of freedom of the model-check covariance | Satterthwaite (1946), *Biometrics Bulletin* 2(6), 110-114 |
| the correction of a robust variance with high-leverage rows | Bell and McCaffrey (2002), *Survey Methodology* 28(2), 169-181; Imbens and Kolesar (2016), *Review of Economics and Statistics* 98(4), 701-712 |
