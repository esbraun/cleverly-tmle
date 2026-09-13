# How every method reports uncertainty

Each estimator reports a point estimate and an influence curve, and every interval on this page is
built from that curve. The intervals are valid where the estimator is asymptotically linear with
the curve it reports. Each method entry defines its curve and states the conditions the curve
requires.

This page describes what happens to the curve after that, once for all of them.

## Influence-curve variance

For an estimate with influence values $D_i$ and independent observations, the default rule is

$$
\widehat{\operatorname{Var}}(\hat\psi)=\frac{1}{n}\cdot\frac{1}{n-1}\sum_{i=1}^n(D_i-\bar D)^2 ,
$$

with the corresponding covariance matrix when a fit reports several parameters.

The curve is centered rather than assumed to be centered. Targeting drives $\bar D$ to
approximately zero. Reading the mean off the sample, instead of substituting zero, is what makes
the reported variance a statement about the curve that was actually computed.

## Covariance rules

Each `ParameterEstimate` declares a `covariance_rule`. `result.covariance()` and
`result.contrast()` apply that rule to the influence curves at every selection size. A contrast
inherits the rule of its inputs.

| rule | declared by | covariance entry $(j,k)$ | stored `variance` |
| --- | --- | --- | --- |
| `"centered"` | every estimate except the one below. This is the default | the sample covariance of the curves above, at the observation or cluster unit | the same centered value, except on a fold-evaluated fit or a repeated fit |
| `"second_moment"` | the stacked cross-fitted missing-outcome `NaturalCourseMean` | $n^{-2}\sum_i D_{ij}D_{ik}$, the raw second moment | the same raw second moment |

The stacked natural-course estimator reports $P_nD^2/n$ as its variance. Its curve has mean zero to
targeting tolerance, for two reasons:

| step | what it gives |
| --- | --- |
| the pooled fluctuation solves $P_n[\Delta\{Y-m^\star(X)\}/\pi(X)]=0$ | the residual term of $D$ has sample mean zero |
| the point estimate is $\hat\psi=P_n m^\star(X)$ | the plug-in term $m^\star(X)-\hat\psi$ has sample mean zero |

Here $m^\star$ is the targeted outcome prediction and $\pi$ is the response prediction, each from
the fold that holds the row. The centered rule gives the variance
$\{n(n-1)\}^{-1}\sum_i(D_i-\bar D)^2$. On a mean-zero curve, $P_nD^2/n$ equals that variance
times $(n-1)/n$. The two rules therefore have the same limit and differ only by that factor. The
registered [stacked study](method-evidence/stacked-missing-outcome-natural-course-cvtmle.md)
validates the second-moment rule, and its R `tmle` comparator reports the centered rule.
`tests/unit/test_natural_course_crossfit.py::test_the_stacked_curve_is_mean_zero_so_the_rules_differ_by_n_minus_one_over_n`
checks the zero mean and the factor at two and three folds.

The [point-treatment reference](point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects)
defines the curve. The estimator is scalar, so its rule applies to a one-name `covariance()` and
to a one-input smooth contrast.

Five selections raise `ValueError`. No fit produces any of these five inputs. The only
`"second_moment"` estimate is scalar, its fit refuses clusters and `repeats` above one, and a fit
builds bands only
over two or more estimates. A direct call to `simultaneous_bands` can still receive that estimate.

| function | selection | reason |
| --- | --- | --- |
| `covariance()`, `contrast()`, and the helpers in `cleverly.inference.results` | estimates that declare different rules | no derivation here supplies the cross-covariance between a centered curve and a raw second moment |
| the same helpers | a `"second_moment"` estimate on a clustered result | the raw second-moment rule is defined for independent rows only |
| `make_estimate` in `cleverly.inference.influence` | `covariance_rule="second_moment"` with clusters | the same reason |
| `median_estimates` in `cleverly.inference.influence` | repeats whose estimates declare different rules | a median over draws needs one rule |
| `simultaneous_bands` | any `"second_moment"` estimate | the multiplier draws center each influence curve. Centering matches the raw second moment only on a mean-zero curve, and `simultaneous_bands` does not check the mean |

A fold-evaluated fit, with `cv_evaluation=True`, stores the cross-validated variance from
uncentered fold second moments. Its estimates still declare `"centered"`. The covariance diagonal
therefore differs from the stored variance on that fit.
`tests/unit/test_cv_targeting.py::TestTheFoldEvaluatedCovarianceRule` checks that difference.
`tests/unit/test_inference.py::TestTheCovarianceRule` checks both rules, contrast inheritance, and
the five refusals. A repeated fit refuses `covariance()` and `contrast()` altogether, as the
[CV-TMLE reference](cv-tmle.md) states.

## Clusters

With $m$ independent clusters, the influence values are summed inside each cluster first:

$$
\widehat{\operatorname{Var}}(\hat\psi)=\frac{m}{n^2}\cdot\frac{1}{m-1}\sum_{c=1}^m(S_c-\bar S)^2,
\qquad S_c=\sum_{i\in c}D_i .
$$

The independent unit is then the cluster and not the row. `cluster=` changes the unit for the
covariance and for fold construction. It does not change the estimand.

A cluster stays intact in every split. Splitting a cluster across folds to buy more folds is
refused: the out-of-fold predictions stop being independent of the rows they are used on, and the
standard error shrinks in the exact direction the cluster role was declared to prevent.

## Transformed parameters

Risk ratios, odds ratios, and user contrasts propagate the joint influence curve by the delta
method. A ratio's curve is the delta-method transform of the levels' curves, and the technical
reference records that as an exact identity rather than as an approximation. Ratio intervals are
built on the log scale and exponentiated.

Implementation:
[`inference/influence.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/influence.py),
[`inference/delta.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/delta.py),
and
[`inference/cluster.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/cluster.py).

## Simultaneous bands

A fit that reports several correlated parameters needs error control over the family and not over
each interval separately. The multiplier bootstrap draws from the joint influence matrix and
estimates a familywise critical value.

| choice | what it does |
| --- | --- |
| `simultaneous=` | turns the familywise band on. It is on by default |
| `n_multiplier=` | the number of draws. `"auto"` resolves per engine, because the point path draws 1000 and the sequential path draws 2000 |
| `multiplier_kind=` | the multiplier distribution: `rademacher`, `mammen`, or `normal` |

Ordinary and cluster resampling both preserve the declared independent unit. Whole-cluster
resampling gives each sampled occurrence a distinct cluster code. Repeated draws of one source
cluster therefore remain separate for fold construction and variance estimation.

Bootstrap configuration is refused for engines that cannot implement it. An engine does not
accept and then discard this configuration.

`simultaneous_bands` refuses an estimate that declares the `"second_moment"` covariance rule. The
multiplier draws center each influence curve. Centering matches the raw second moment only on a
mean-zero curve, and `simultaneous_bands` does not check the mean.
[Covariance rules](#covariance-rules) lists this refusal with the others. A repeated fit also refuses bands, as the [CV-TMLE reference](cv-tmle.md#variations) states.

Implementation:
[`inference/multiplier.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/multiplier.py)
and
[`inference/bootstrap.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/bootstrap.py).
Benjamini and Hochberg (1995) is the reference for the FDR-adjusted reporting; see
[multiple testing](../references.md#multiple-testing).

## Reporting a subset of a family

When the public layer reports a subset of the parameters an engine computed, the inference it
reports is the inference for that subset. A joint band is a statement about a family. Narrowing the
family and keeping the critical value would assert a coverage property over parameters the result
no longer contains. `cleverly` recomputes from the retained influence curves under the same
significance level, draw count, multiplier distribution, seed, and cluster structure.

## What is not on this page

Two inference rules belong to one method each, and each method entry states its own.

| rule | where it is stated |
| --- | --- |
| the cross-validated variance under fold evaluation, and why it is not a fold-averaged second moment | [CV-TMLE](cv-tmle.md#variations) |
| the plug-in influence-curve variance for `LTMLE`, and what it does not absorb | [Longitudinal TMLE](longitudinal-tmle.md#validation-issues-special-to-this-method) |

The corrected curve `DRTMLE` reports is the estimator's own influence function rather than the
efficient one. [DR-TMLE](dr-tmle/index.md#what-this-solves) says what follows from that.

Evidence for everything above:
[`tests/unit/test_inference.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_inference.py)
pins the exact covariance identities, the weighted effective sample size, cluster aggregation, the
delta-method transformations, the multiplier critical value, and the simultaneous bands.
A repeated cross-fitted fit reports no covariance.
[`tests/unit/test_repeated_crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_repeated_crossfit.py)
pins that refusal, and pins the median point and the split-adjusted median variance the fit reports
instead.
