# How every method reports uncertainty

Each estimator reports a point estimate and an influence curve, and every interval on this page is
built from that curve. The intervals are valid where the estimator is asymptotically linear with
the curve it reports. Each method entry defines its curve and states the conditions the curve
requires.

This page describes what happens to the curve after that, once for all of them.

## Influence-curve variance

For an estimate with influence values $D_i$ and independent observations, `cleverly` reports

$$
\widehat{\operatorname{Var}}(\hat\psi)=\frac{1}{n}\cdot\frac{1}{n-1}\sum_{i=1}^n(D_i-\bar D)^2 ,
$$

with the corresponding covariance matrix when a fit reports several parameters.

The curve is centered rather than assumed to be centered. Targeting drives $\bar D$ to
approximately zero. Reading the mean off the sample, instead of substituting zero, is what makes
the reported variance a statement about the curve that was actually computed.

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

Ordinary and cluster resampling both preserve the declared independent unit. Bootstrap
configuration is refused for engines that cannot implement it, rather than accepted and then
discarded.

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
Repeated-cross-fit covariance is pinned in
[`tests/unit/test_repeated_crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/unit/test_repeated_crossfit.py).
