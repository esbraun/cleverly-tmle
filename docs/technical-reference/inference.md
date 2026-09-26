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

## Inference status

Each `ParameterEstimate` declares an `inference` status. The status says whether `cleverly`
supplies inference for the estimate. `supplies_inference` is `True` for `"influence_curve"` only.
Every other status is a non-inferential status.

| status | declared by | `std_error`, `ci`, and `pvalue` | `summary()` column | reopened by |
| --- | --- | --- | --- | --- |
| `"influence_curve"` | every estimate except the ones below. This is the constructor default | return the values on this page | `std_err` | not applicable |
| `"working_mechanism_plugin"` | a `CTMLE` fit with `strategy="greedy"`, `"ordered"`, or `"discrete"`. [Collaborative TMLE](collaborative-tmle.md) gives the reason | raise `CapabilityError` with the reason of the status | `working-mechanism se` | [F18](../roadmap.md#f18-selector-path-c-tmle-inference) |
| `"generated_design_plugin"` | every `CTMLE` fit with `strategy="oat"`, including a fit with `delta=` and a fit that requests one arm mean. [Collaborative TMLE](collaborative-tmle.md) gives the reason | raise `CapabilityError` with the reason of the status | `generated-design se` | [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| `"undeclared_function_plugin"` | a restored `TMLE` result whose `Rule`, user-written `Intervention`, `Stochastic` density, or written MSM design or weight is not declared `"known"`. A restored `LTMLE` result takes it when a callable regimen node is undeclared or its MSM lacks evidence of known source functions. New fits refuse such a function. A new fit with each function declared `"known"` reports an interval. [RM28](../roadmap.md#rm28-declared-densities-of-user-written-interventions) gives the load rule | raise `CapabilityError` with the reason of the status | `undeclared-function se` | [RM28](../roadmap.md#rm28-declared-densities-of-user-written-interventions) |
| `"estimated_weight_plugin"` | a `DRTMLE` fit with a non-empty `guard` and varying weights declared estimated (`weights_estimated=True`). A fit with `guard=()` keeps `"influence_curve"`. Constant weights fit the unweighted estimator, so they keep it too. [DR-TMLE supported estimands](dr-tmle/supported-estimands.md#refused-by-name) gives the reason | raise `CapabilityError` with the reason of the status | `fixed-weight se` | [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| `"cross_fitted_longitudinal_plugin"` | a saved cross-fitted `LTMLE` result with `id=`. New fits refuse this design. [RM29](../roadmap.md#rm29-saved-cross-fitted-clustered-longitudinal-results) gives the load rule | raise `CapabilityError` with the reason of the status | `grouped-longitudinal plug-in se` | [F22](../roadmap.md#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| `"stratified_fold_plugin"` | a saved cross-fitted `TMLE` or `DRTMLE` result with `stratify_folds="treatment"` or `"treatment+outcome"`, and a saved `LTMLE` result with more than one fold whose `Folds.origin` is `None`. Releases 0.1.0 and 0.1.1 drew the point-treatment split under `"treatment"` and the `LTMLE` split on the first node by default, and new fits refuse them. An in-sample result and a continuous-dose result keep `"influence_curve"` under this rule, because the fit drew no split, or its split read no treatment. A continuous-dose result on an undeclared outcome scale takes the next status. [RM31](../roadmap.md#rm31-inference-status-of-a-saved-stratified-cross-fitted-result) gives the load rule | raise `CapabilityError` with the reason of the status | `stratified-fold plug-in se` | [RM31](../roadmap.md#rm31-inference-status-of-a-saved-stratified-cross-fitted-result) |
| `"undeclared_scale_plugin"` | a saved cross-fitted continuous-dose `TMLE` result of a continuous outcome with `q_bounds=None`, and a copied `TMLE` or `DRTMLE` estimator on that scale. Releases 0.1.0 and 0.1.1 took that outcome scale from every observed outcome, held-out rows included, and new fits refuse it. Those releases stratified every saved cross-fitted discrete-treatment result, so such a result takes `"stratified_fold_plugin"` first. Release 0.1.1 `DRTMLE` refused a continuous treatment, so no saved `DRTMLE` result reaches this status. A declared `q_bounds`, an in-sample result, and a binary outcome keep `"influence_curve"` under this rule. [RM33](../roadmap.md#rm33-inference-status-of-a-saved-undeclared-scale-cross-fitted-result) gives the load rule | raise `CapabilityError` with the reason of the status | `undeclared-scale plug-in se` | [RM33](../roadmap.md#rm33-inference-status-of-a-saved-undeclared-scale-cross-fitted-result) |
| `"unequal_cluster_plugin"` | a cross-fitted `TMLE` or `DRTMLE` fit with `id=` whose clusters differ in row count or weight mass, overall or in one reported baseline stratum. This includes `cv_evaluation=True`. [Clusters](#clusters) gives the reason | raise `CapabilityError` with the reason of the status | `cluster-robust plug-in se` | [F22](../roadmap.md#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| `"few_cluster_plugin"` | a `TMLE`, `DRTMLE`, or `LTMLE` fit with `id=` and fewer than 40 clusters with positive weight mass in the fit or one reported baseline stratum. `LTMLE` takes `id=` in sample only. [Clusters](#clusters) gives the reason | raise `CapabilityError` with the reason of the status | `normal-reference se` | [F22](../roadmap.md#f22-grouped-cross-fitting-beyond-point-treatment-tmle) |
| `"unrecorded_status_plugin"` | a saved `ParameterEstimate` whose state has no `inference` key. The estimate inside a `VariableImportanceEntry` or a `CVTargeting` takes it the same way. Releases 0.1.0 and 0.1.1 wrote no such key. Such an estimate holds no estimator, data, or folds, so the load rule cannot read its configuration. A `TMLEResult` or `LongitudinalResult` that holds the estimate gives it the status of its own configuration. [RM34](../roadmap.md#rm34-inference-status-of-a-saved-estimate-outside-its-result) gives the load rule | raise `CapabilityError` with the reason of the status | `unrecorded-status plug-in se` | [RM34](../roadmap.md#rm34-inference-status-of-a-saved-estimate-outside-its-result) |

At every status, `plugin_std_error` and `plugin_interval` return the plug-in spread of the
reported curve. On `"influence_curve"` they return the numbers of `std_error` and `ci` under names
that claim no coverage. On a non-inferential status they return a diagnostic.

One table in `cleverly._inference_status`, `NON_INFERENTIAL`, holds the texts of each
non-inferential status. The refusal, the `summary()` paragraph, the assessment note, and the
E-value row each read that table. `tests/unit/test_inference_status_registry.py` checks that the
table, the `InferenceStatus` type, and the rows above list the same statuses.

A fit has one status. `TMLEResult.inference_status` and `LongitudinalResult.inference_status`
return it. The estimator decides the status from its configuration and the prepared data. It reads
no fitted quantity. When more than one non-inferential status applies, the fit takes the first one
in the table above. The rows are in that order.

A report that publishes a spread names its columns through `spread_name` in
`cleverly.inference.influence`. On a non-inferential status, `to_frame()` emits `inference`,
`plugin_std_err`, `plugin_interval_lower`, and `plugin_interval_upper` in place of `std_err`,
`ci_lower`, and `ci_upper`. It emits no `p_value`. `ParameterEstimate.spread_columns()` returns
those columns under the name that fits the status. The table below gives the other reports that
rename a number on a non-inferential status.

| report | inferential name | name on a non-inferential status |
| --- | --- | --- |
| `result.cv_targeting.to_frame()` | `cv_std_err`, `pooled_std_err` | `cv_plugin_std_err`, `pooled_plugin_std_err`, and an `inference` column |
| `result.cv_targeting.std_error` | the property | refused. `plugin_std_error` returns the same numbers |
| `omitted_variable_bounds(...).to_dict()` | `ci_lower`, `ci_upper`, `robustness_value_ci` | `plugin_interval_lower`, `plugin_interval_upper`, `robustness_value_plugin_interval`, and an `inference` key |
| `robustness_value(...)` | `rva` | `rv_plugin_interval`, and an `inference` key |
| `LongitudinalResult.curve()` | `std_err`, `ci_lower`, `ci_upper` | `plugin_std_err`, `plugin_interval_lower`, `plugin_interval_upper`, and an `inference` column |
| `LongitudinalResult.incidence_total()` | `std_err` | `plugin_std_err` |
| `VariableImportanceResult.to_frame()`, on a restored result only | `std_err`, `ci_lower`, `ci_upper`, `p_value`, `p_value_adjusted` | `plugin_std_err`, `plugin_interval_lower`, `plugin_interval_upper`, and an `inference` column. `adjusted_pvalue` reads `None` |

Under `nu2_estimator="plugin"` an omitted-variable bound refuses its limits at every status, for a
different reason. No derivation in a source this package cites gives their standard error. `to_dict()` then omits the three limit
keys and carries `nu2_estimator`, and `robustness_value()` omits `rva` under every name. The
[standard error of the omitted-variable bound](validation-methods.md#standard-error-of-the-omitted-variable-bound) gives the rule.

`CVTargeting.inference` reads the status from the two fold-level reports. The fit stamps those
reports where it stamps its own estimates. `SensitivityBounds.inference` carries the status of the
estimate that the bound adjusts.

The file `tests/unit/test_inference_status_reach.py` forces each non-inferential status on a
clustered point-treatment fit and checks its reports in this section. The file
`tests/unit/test_longitudinal_cluster_status.py` checks the `LongitudinalResult` reports on a fit
with 39 clusters. The file `tests/unit/test_saved_fold_policy_status.py` loads a saved stratified
result of each path and checks its reports. The file `tests/unit/test_saved_scale_status.py` loads
a saved result on an undeclared outcome scale and checks its reports. The file
`tests/unit/test_saved_bare_estimate_status.py` loads an estimate, an entry, a fold-level report,
and whole results that record no status.

A contrast inherits the status of its inputs, so a contrast of two diagnostic estimates refuses
`ci` as its inputs do. A simultaneous band refuses every non-inferential status with
`CapabilityError`. On such a fit, `TMLE` and `LTMLE` skip the band, so the default
`simultaneous=True` does not raise. An explicit `simultaneous=True` has the same behavior. Neither
case emits a warning; `summary()` states the omission on a fit with two or more estimates. The
constructor default does not distinguish an explicit request from the default.

Two selections that mix the statuses raise `ValueError`. No fit produces either input, because the
estimator stamps one status on every estimate it reports.

| function | selection | reason |
| --- | --- | --- |
| `contrast()`, and `smooth_contrast` in `cleverly.inference.results` | estimates that declare different statuses | an inferential status would give the refused input an interval |
| `median_estimates` in `cleverly.inference.influence` | repeats whose estimates declare different statuses | the median would report one draw's refusal under the other draw's name |

`tests/unit/test_inference.py::TestTheInferenceStatus` checks the inheritance, both `ValueError`
selections, and the band refusal.

## Clusters

With $m$ independent clusters, the influence values are summed inside each cluster first:

$$
\widehat{\operatorname{Var}}(\hat\psi)=\frac{m}{n^2}\cdot\frac{1}{m-1}\sum_{c=1}^m(S_c-\bar S)^2,
\qquad S_c=\sum_{i\in c}D_i .
$$

The independent unit is then the cluster and not the row. `cluster=` changes the unit for the
covariance and for fold construction. It does not change the estimand.

A cluster stays intact in every split. `random_partition` permutes the distinct cluster labels and
cuts them into near-equal parts. Splitting a cluster across folds to buy more folds is
refused under the package's fold contract. Such a split can couple nuisance training to the
evaluation rows. No derivation here covers its interval. Balkus, Laith and Hejazi (2026) show that
split correlated units can still remove an empirical-process term under their conditions. Their
result does not establish the variance of this package's estimators.

Two estimators refuse `cluster=` rather than draw that split. Collaborative TMLE refuses it at
every setting, and longitudinal TMLE refuses it above one fold. The
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) give the audit and both
messages.

Two clustered settings of `TMLE` and `DRTMLE` report no interval. Each takes a status from the
[status table](#inference-status). The status is determined from prepared cluster labels, strata,
and weights, without reading a fitted quantity.

| setting | status | reason |
| --- | --- | --- |
| cross-fitted, and the clusters hold different numbers of rows or weight mass, overall or within a reported baseline stratum | `"unequal_cluster_plugin"` | the [grouped folds](cv-tmle.md#grouped-folds) argument and registered study cover equal sizes and masses. The point estimator remains row weighted at unequal sizes, but its cross-fitted interval lacks a validation result there |
| fewer than 40 clusters with positive weight mass in the fit, or in one reported baseline stratum, in sample or cross-fitted | `"few_cluster_plugin"` | the package uses a normal reference. Nugent et al. (2024), Section 2.2, recommend a $t$ reference with $J - 2$ degrees of freedom below 40 clusters, where $J$ is the contributing cluster count. Benitez et al. (2023), Section 3.1.2, paragraph on inference, and Section 3.2.1, last paragraph, recommend it at every cluster count. No registered study covers few clusters |

The few-cluster setting applies to the in-sample `LTMLE` fit too. The data of a longitudinal fit
hold no baseline strata, so the count is that of the whole fit. The file
`tests/unit/test_longitudinal_cluster_status.py` holds a witness at 39 clusters, a control at 40,
and mutations of the rule.

The threshold is a reporting policy, not a coverage guarantee at 40 clusters. It counts clusters
with positive weight mass but does not measure weight concentration. A fit with 40 such clusters
can still put almost all weight on one. Inspect the weight report and overlap before using an
interval. A restored cross-fitted clustered `LTMLE` result takes
`"cross_fitted_longitudinal_plugin"` at every cluster count and size. New fits refuse that design.

The table gives the functions in `cleverly.inference.cluster` that apply each rule.

| function | what it reads |
| --- | --- |
| `unequal_cluster_sizes` | the row count and, on a weighted fit, the weight mass of each cluster. `cluster_inference_status` checks both measures again within each reported stratum. Two masses count as equal within `WEIGHT_MASS_RTOL`, 1e-9, of the largest |
| `fewest_clusters` | the distinct cluster count with positive weight mass. On a fit with baseline strata, it also counts those clusters within each stratum and returns the smallest count |

A fit has one status. So one stratum with fewer than 40 contributing clusters withholds the interval
of every estimate, the marginal estimates included. An in-sample fit at unequal sizes keeps its
interval when it has 40 or more contributing clusters in the fit and in each reported stratum.
Benitez et al. (2023),
Section 3.2.1, give the cluster-sum aggregation for that row-weighted estimand. When both settings
apply, the fit takes `"unequal_cluster_plugin"`, which comes first in the status table.
`tests/unit/test_cluster_status.py` holds a witness, a control, and a mutation for each rule.

The few-cluster rule counts contributing clusters, and it reads no row count. So a fit of 30 rows
with `id=` and one row in
each cluster takes `"few_cluster_plugin"`. The same rows without `id=` keep their interval. This
refusal is conservative, and the roadmap records it as a known over-refusal.

The facts block of `TMLEResult.summary()` prints the cluster count. It adds unequal row or
weight-mass ranges, including within-stratum ranges. It names clusters with positive weight mass
when some have zero mass, and the fewest contributing clusters in one stratum.
`LongitudinalResult.summary()` prints the cluster count too. It adds the count of clusters with
positive weight mass when some clusters have zero mass.

`FEW_CLUSTER_THRESHOLD` in `cleverly._inference_status` holds the threshold of 40.
[References](../references.md#grouped-folds-and-clustered-cross-fitting) gives both
sources. [F22](../roadmap.md#f22-grouped-cross-fitting-beyond-point-treatment-tmle) holds the
route that reopens each setting.

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
