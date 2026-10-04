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
| `"second_moment"` | the stacked cross-fitted missing-outcome `NaturalCourseMean`, requested alone | $n^{-2}\sum_i D_{ij}D_{ik}$, the raw second moment | the same raw second moment |

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
checks the zero mean and the factor at two and three folds. A stacked fit that reports
`ey_obs` beside arm targets declares the centered rule for every estimate, so one fit never mixes
the two rules ([CV-TMLE](cv-tmle.md#missing-outcome-natural-course-mean)).

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
uncentered fold second moments, or from centered fold cluster variances with `id=`
([CV-TMLE](cv-tmle.md#the-algorithm-as-implemented)). Its estimates still declare `"centered"`. The covariance diagonal
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
| `"working_mechanism_plugin"` | a `CTMLE` fit with `strategy="greedy"`, `"ordered"`, or `"discrete"`. A `discrete` fit with one declared candidate, equal to the full adjustment set, takes the TMLE status instead. [Collaborative TMLE](collaborative-tmle.md) gives the reason | raise `CapabilityError` with the reason of the status | `working-mechanism se` | [F18](../roadmap.md#f18-selector-path-c-tmle-inference) |
| `"generated_design_plugin"` | every `CTMLE` fit with `strategy="oat"`, including a fit with `delta=` and a fit that requests one arm mean. [Collaborative TMLE](collaborative-tmle.md) gives the reason | raise `CapabilityError` with the reason of the status | `generated-design se` | [F19](../roadmap.md#f19-outcome-adaptive-c-tmle-generated-design-inference) |
| `"estimated_weight_plugin"` | a `DRTMLE` fit with a non-empty `guard` and varying weights declared estimated (`weights_estimated=True`). A fit with `guard=()` keeps `"influence_curve"`. Constant weights fit the unweighted estimator, so they keep it too. [DR-TMLE supported estimands](dr-tmle/supported-estimands.md#refused-by-name) gives the reason | raise `CapabilityError` with the reason of the status | `fixed-weight se` | [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) |
| `"few_cluster_plugin"` | a `TMLE` or `DRTMLE` fit with `id=` and fewer than 10 clusters with positive weight mass in the fit or one reported baseline stratum, or an `LTMLE` fit with fewer than 20. `LTMLE` takes `id=` in sample only. [Clusters](#clusters) gives the reason | raise `CapabilityError` with the reason of the status | `normal-reference se` | [F28](../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) |

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
| `to_frame()` of a fit with `n_bootstrap` | `bootstrap_std_err`, `bootstrap_ci_lower`, `bootstrap_ci_upper` | `bootstrap_sd`, `bootstrap_range_lower`, `bootstrap_range_upper` |
| the bootstrap rows of `summary()` | `bootstrap se`, `percentile CI` | `bootstrap sd`, `percentile range` |

The bootstrap spread is the sample standard deviation of the replicate estimates. A standard error
is an inference claim, so a non-inferential status publishes that number as `bootstrap sd`.
`BootstrapSummary.inferential` records whether a registered study licenses the percentile interval
for the fit that produced it. When it is `False`, the bootstrap columns take the diagnostic names
at every status. `bootstrap_column` in `inference/influence.py` applies both conditions.
`BootstrapSummary` keeps the field names `std_error` and `ci` at every status. Its fields describe
the replicate distribution, and the status decides only the published name.
`TestTheBootstrapPublishesUnderTheStatusName` in
`tests/unit/test_summary_and_message_accuracy.py` checks both names on a selector fit and an
ordinary fit.

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
with 39 clusters.

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
covariance and for fold construction. It does not change the estimand. A composite-indicator fit
with a missing treatment follows the same rule: its point estimate does not move with `id=`, and
its variance comes from the cluster sums of its curve
(`tests/unit/test_composite_missing_data.py::test_e20_clusters_move_the_variance_and_not_the_estimate`).

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

The estimand is the row-weighted mean $\mu_I$ of Wang, Park, Small and Li (2024), Section 2,
with every row of a cluster in the analysed population. A row-level treatment keeps row-level
exchangeability given `W`. The cluster size enters only when it confounds, and then it must be a
covariate. To estimate the cluster-average mean $\mu_C$, pass `weights` equal to one over the
cluster size.

### Unequal cluster sizes

Every target builder writes the curve as $D_i=\phi_i-\hat\psi$. The cluster total is then
$S_j=A_j-N_j\hat\psi$, with $A_j=\sum_{i\in j}\phi_i$ and $N_j$ the row count of cluster $j$. The
point estimate is $\sum_j A_j/\sum_j N_j$, a ratio of two cluster means. The variance above is
the delta-method variance of that ratio at $\mu_I$, so it holds at unequal sizes. A weighted fit
replaces $N_j$ by the weight mass of the cluster.

At equal sizes the term $-N_j\hat\psi$ shifts
every total by one constant, and the centered variance removes it. A study at equal sizes cannot
see that term, so `tests/unit/test_cluster_ratio_variance.py` checks it at sizes 1 to 6.

| path | variance at unequal sizes |
| --- | --- |
| in sample, and the stacked cross-fitted report | the cluster-sum variance above |
| `cv_evaluation=True` | the centered cluster variance inside each validation fold ([CV-TMLE](cv-tmle.md#the-algorithm-as-implemented)). Each validation fold needs 2 clusters |
| `targeting_scheme="fold"`, `DRTMLE` cross-fitted, and `repeats` above 1 | the variance of the shipped construction. Each sums the curve within clusters |
| baseline strata | the stratum curve, embedded at $n/n_s$ and summed over the full cluster vector |

The pinned comparators aggregate a clustered curve in two ways. Only the cluster sum agrees with
this package at unequal sizes.

| implementation | cluster aggregation | agrees at unequal sizes | reference distribution |
| --- | --- | --- | --- |
| R `ltmle` 1.3-0 | the cluster sum times $J/n$ (`HouseholdIC`, `R/ltmle.R` lines 1025 to 1032) | yes. The variance is the same | Student $t$ with $J-1$ degrees of freedom below 100 clusters |
| R `tmle` 2.1.1 | the cluster mean (`R/tmle.R` lines 1566 to 1568) | no | normal |
| R `tmle3` at `ed72f8a` | the cluster mean (`R/utils.R` lines 44 to 48) | no | normal |
| R `lmtp` 1.5.4 with `ife` 0.2.3 | the cluster mean | no | normal |
| R `drtmle` 1.1.2 | no cluster argument | not applicable | not applicable |

### Few clusters

A clustered fit that reads $J$ clusters with positive weight mass, with $10 \le J < 40$ ($20 \le J < 40$ for `LTMLE`), reports
its intervals and p-values on a Student $t$ reference with $J-2$ degrees of freedom. Nugent et
al. (2024), Section 2.2, last paragraph, give the rule. At 40 clusters or more the normal
reference stays. `ParameterEstimate.reference_df` holds the degrees of freedom, or `None` for the
normal reference.

| estimate | `reference_df` |
| --- | --- |
| a marginal estimate of the fit | $J-2$, with $J$ the positive-mass cluster count of the fit |
| a baseline-stratum estimate | $J_s-2$, with $J_s$ the positive-mass cluster count of the stratum |
| a fold-evaluated estimate over $V$ validation folds: the `cv_evaluation=True` report and the `fold_evaluated` report of `cv_targeting` | $\min(J-2, J-V)$. Its variance centers the cluster totals in each fold, so it estimates one mean per fold and keeps $J-V$ degrees of freedom, the pooled within-group count |
| a derived estimate: `contrast()`, `ratio()`, and the median over `repeats` | the smallest `reference_df` of its inputs. `None` counts as infinite |
| an in-fit `rr` or `or` | the value of the arm means it reads |
| every estimate of a `"few_cluster_plugin"` fit | `None`. The diagnostic keeps the normal reference |

The stamp in `TMLE._retarget_detailed` computes the value from the rows each estimate reads, and
only when the fit supplies inference. `ci`, `pvalue`, `wald_test()`, `plugin_interval`, the
missingness tilt and the omitted-variable limits read it through `wald_ci` and
`reference_quantile` in `cleverly.inference.delta`. The summary prints a `df` column and a note.

| request on a fit with a $t$ reference | result |
| --- | --- |
| the default simultaneous band | skipped. The summary states the reason. No source gives a $t$-calibrated band |
| `simultaneous_bands()` | `CapabilityError` |
| the cluster bootstrap | printed as a percentile range with a diagnostic note. No result validates the cluster bootstrap below 40 clusters |

$J-2$ is the rule that Nugent et al. (2024) and Benitez et al. (2023), Sections 3.1.2 and 3.2.1,
state for cluster-randomized trials. An arm mean of this package is a one-sample mean of $J$
cluster totals, whose classical reference has $J-1$ degrees of freedom. R `ltmle` uses $J-1$. So
$J-2$ is conservative by one degree of freedom on the in-sample and stacked reports.

A fold-evaluated report has fewer degrees of freedom, $J-V$, and takes $\min(J-2, J-V)$. An
implementation review measured the difference at 2 clusters per fold: at $J=10$ and $V=5$,
$t_{J-2}$ covered 0.935 and $t_{J-V}$ covered 0.955 over 1,500 draws.

Wang et al. (2024), Remark 3, caution against complex nuisance learners at about 20 clusters. The
registered few-cluster evidence uses parametric nuisance learners only.

Below 10 clusters with positive weight mass, in the fit or in one reported baseline stratum, the
fit takes `"few_cluster_plugin"`. An `LTMLE` fit takes it below 20. Each floor is the smallest
count that the registered few-cluster study measures for that fit
(`MINIMUM_INTERVAL_CLUSTERS` and `MINIMUM_LONGITUDINAL_INTERVAL_CLUSTERS`), and
[F28](../roadmap.md#f28-finite-sample-limits-of-clustered-intervals) owns the counts below. The
status is
determined from prepared cluster labels, strata, and weights, without reading a fitted quantity.
A fit has one status, so one stratum below the floor withholds the interval of every estimate.
The rule applies to the in-sample `LTMLE` fit too, whose data hold no baseline strata.

The rule counts contributing clusters, and it reads no row count. So 30 rows with `id=` and one
row in each cluster report $t_{28}$ intervals, and the same rows without `id=` use the normal
reference. This is a known conservative reference.

The threshold is a reporting policy, not a coverage guarantee at 40 clusters. It counts clusters
with positive weight mass but does not measure weight concentration. A fit with 40 such clusters
can still put almost all weight on one. Inspect the weight report and overlap before using an
interval.

| function in `cleverly.inference.cluster` | what it reads |
| --- | --- |
| `fewest_clusters` | the distinct cluster count with positive weight mass. On a fit with baseline strata, it also counts those clusters within each stratum and returns the smallest count |
| `cluster_reference_df` | the positive-mass cluster count of the rows one estimate reads, as $J-2$, or `None` at 40 or more |
| `cluster_sizes`, `cluster_weight_mass`, `unequal_cluster_sizes` | the row count and the weight mass of each cluster, for the summary facts only |

The facts block of `TMLEResult.summary()` prints the cluster count. It adds unequal row or
weight-mass ranges, including within-stratum ranges. It names clusters with positive weight mass
when some have zero mass, and the fewest contributing clusters in one stratum.
`LongitudinalResult.summary()` prints the cluster count too. It adds the count of clusters with
positive weight mass when some clusters have zero mass.

`FEW_CLUSTER_THRESHOLD` and `MINIMUM_INTERVAL_CLUSTERS` in `cleverly._inference_status` hold
the counts 40 and 10. `tests/unit/test_cluster_status.py`,
`tests/unit/test_longitudinal_cluster_status.py` and `tests/unit/test_few_cluster_reference.py`
hold a witness, a control, and a mutation for each rule.
[References](../references.md#grouped-folds-and-clustered-cross-fitting) gives the sources.

## Transformed parameters

Risk ratios, odds ratios, and user contrasts propagate the joint influence curve by the delta
method. A ratio's curve is the delta-method transform of the levels' curves, and the technical
reference records that as an exact identity rather than as an approximation. Ratio intervals are
built on the log scale and exponentiated.

Let $h$ be a smooth function of the estimates $\hat\psi$, with gradient $\nabla h$ and joint curve
$IC$. Let $f$ be a monotone transform with derivative $f'$. The table gives each construction.

| construction | estimate | curve on the inference scale | interval |
| --- | --- | --- | --- |
| `contrast(h, names)` | $h(\hat\psi)$ | $\nabla h^\top IC$ | $h \pm z\,se$ |
| `contrast(h, names, scale="ratio")` | $v = h(\hat\psi) > 0$ | $\nabla h^\top IC / v$, the chain rule for $\log$ | $\exp(\log v \pm z\,se)$ |
| `contrast(h, names, transform=f)` | $h(\hat\psi)$ | $f'(h)\,\nabla h^\top IC$ | the sorted pair $f^{-1}(f(h) \pm z\,se)$ |
| `ratio(a, b)` | $\psi_a / \psi_b$ | $IC_a/\psi_a - IC_b/\psi_b$ | $\exp(\log\psi \pm z\,se)$ |

The transform row is `ci(contrast = list(f, f_inv, h, fh_grad))` of R `drtmle` 1.1.2
(`R/confint.R`, lines 146 to 167). `drtmle` takes the gradient of $f \circ h$. This package takes
$\nabla h$ and $f'$ and applies the chain rule. The limits are sorted after the inverse map, so a
decreasing $f$ gives an ordered interval. `drtmle` does not sort them. `Transform.log()` and
`scale="ratio"` give the same standard error, interval and p-value bit for bit. A `Transform` built
from lambdas does not pickle. A fit never stores a transformed estimate.

`estimate.wald_test(null=c)` reports $z = (\tilde\psi - \tilde c)/se$ and $p = 2\Phi(-|z|)$, with
both values on the inference scale. For a ratio $\tilde c = \log c$, so $c$ must be positive. For a
transformed estimate $\tilde c = f(c)$. This is `wald_test(null = )` of R `drtmle` 1.1.2
(`R/test.R`, lines 85 to 183). `pvalue` is `wald_test().pvalue` at the default null: 0 for a level,
difference or fraction, and 1 for a ratio. A non-inferential estimate refuses `wald_test` as it
refuses `pvalue`.

`tests/unit/test_contrast_conveniences.py` checks each row against a longhand statement. It also
runs one mutation control for each transform and gradient.

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
accept and then discard this configuration. `TMLE` and `LTMLE` both take `n_bootstrap=`, and each
replicate refits the whole estimator. The
[longitudinal bootstrap contract](longitudinal-tmle.md#the-full-refit-bootstrap) states what a
replicate resamples and refits. The
[full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md)
measures the percentile interval with correctly specified cell-mean nuisances on finite binary
laws, at $n = 1000$, and at 1,500 rows in 60 clusters for the cluster bootstrap. No result covers
a data-adaptive nuisance.

An `LTMLE` bootstrap is licensed as inference only for a design kind
in `LICENSED_BOOTSTRAP_DESIGNS`, which a kind enters after its registered cells are green. Every
other `LTMLE` bootstrap prints as a diagnostic. `POINT_BOOTSTRAP_INFERENTIAL` in
`estimators/tmle.py` holds the point-treatment rule.

`simultaneous_bands` refuses an estimate that declares the `"second_moment"` covariance rule. The
multiplier draws center each influence curve. Centering matches the raw second moment only on a
mean-zero curve, and `simultaneous_bands` does not check the mean.
[Covariance rules](#covariance-rules) lists this refusal with the others. A repeated fit also refuses bands, as the [CV-TMLE reference](cv-tmle.md#variations) states.

Each fit seeds its multiplier draws with its own `random_state`. Every fit that uses one seed
therefore draws the same multiplier sign matrix. A registered study that fixes `random_state`
measures the band as it ships.

Implementation:
[`inference/multiplier.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/multiplier.py)
and
[`inference/bootstrap.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/bootstrap.py).
Benjamini and Hochberg (1995) is the reference for the FDR-adjusted reporting; see
[multiple testing](../references.md#multiple-testing).

### Evidence for the default band

A joint-coverage cell reads the default band of a fit and the pointwise intervals of the same fit.
The band cell passes when its 99% joint-coverage interval lies inside $[0.92, 0.98]$. The control
cell reads the pointwise intervals jointly and passes when its 99% upper endpoint is below 0.95.
The control shows that the band's wider critical value does work.
`tests/unit/test_simultaneous_cell_design.py` checks before any run that each control can fail.
Each study page gives the measured coverage.

| registered study | shapes it measures |
| --- | --- |
| [stacked arm-indexed missing-outcome CV-TMLE](method-evidence/stacked-arm-indexed-missing-outcome-cvtmle.md) | the arm means and contrasts of four laws, through `CausalStudy` |
| [ordinary multi-arm TMLE](method-evidence/ordinary-multi-arm-tmle.md) | three arm means, two differences, two risk ratios and two odds ratios |
| [ordinary end-of-study longitudinal TMLE](method-evidence/ordinary-end-of-study-longitudinal-tmle.md) | three regimen means and two contrasts |
| [ordinary survival-curve longitudinal TMLE](method-evidence/ordinary-survival-curve-longitudinal-tmle.md) | the risk curve of one plan over two horizons, and all ten parameters of three plans |
| [ordinary competing-risk longitudinal TMLE](method-evidence/ordinary-competing-risk-longitudinal-tmle.md) | two causes at two horizons for one plan, and all ten parameters at horizon two |
| [stratified point-treatment TMLE](method-evidence/stratified-point-treatment-tmle.md) | marginal and stratum parameters, in sample and cross-fitted |
| [observational missing-data DR-TMLE](method-evidence/observational-missing-data-dr-tmle.md) | the arm means and contrasts of the composite indicator at two and three arms, with an observational missing outcome and with a missing treatment |
| [default simultaneous bands](method-evidence/default-simultaneous-bands.md) | 25 further shapes: the `TMLE()` default, weights, missing outcomes, DR-TMLE, the controlled direct effect, MSMs, clusters, shift and incremental grids, regimes, and the cross-fitted, weighted and categorical longitudinal fits |

These fit shapes publish no default band.

| shape | why it publishes no band |
| --- | --- |
| more than one repeat | refused with two or more estimands, as the [CV-TMLE reference](cv-tmle.md#variations) states |
| fold-targeted CV-TMLE, the learned-rule value, and the missing-outcome natural-course mean | one estimate per fit |
| the C-TMLE selector and outcome-adaptive paths | their status supplies no inference, so no band is built |

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
