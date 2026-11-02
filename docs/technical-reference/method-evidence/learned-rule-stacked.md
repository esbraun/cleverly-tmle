# Stacked report of the learned-rule value

This study validates the interval of the stacked report of the learned-rule value. The estimator
is `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=False, targeting_scheme="pooled",
g_bounds="auto")` with `repeats=1`. Each fold learns the plug-in rule $d_v$ of its outcome
regression on its training rows. One pooled fluctuation runs over every validation row with the
observation weights, and the report is $\hat\psi = \sum_v (W_v/W)\,\psi^*_{nv}$, where $W_v$ is
fold $v$'s weight mass (its row count without weights). The curve is
$\phi_i = w_i (Z_i - \hat\psi)$, centred at the pooled estimate.
[The stacked report](../point-treatment-tmle.md#learned-rules) gives the construction, its
source and the variance identity.

The target is data-adaptive: $\tilde\psi^{s}_{0n} = \sum_v (W_v/W)\,\Psi_{d_{nv}}(P_0)$, the value
of the rules the folds learned, each fold weighted by its share of the rows, or of the weight
mass under `weights=`. Each replication carries its own truth, and every statistic reads the
error, the estimate minus that truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** R `polle`'s Algorithm 4 (Nordland and Holst 2026)
is the construction's source, not a pinned comparator of the learned-rule value. A zero-row
equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | the RM30 laws of the [gated study](learned-rule-cvtmle.md#what-was-tested): $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W)$ |
| fit | `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=False, targeting_scheme="pooled", g_bounds="auto")` at nominal n = 2,000 with `repeats=1` and pointwise intervals |
| outcome learner | `Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), LogisticRegression(C=1e6, max_iter=5000, tol=1e-10))` on $(A, W_1, W_2)$ |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, a correct parametric model |
| truth | per replication: refit each fold's outcome learner on the fit's training rows with its weights and clusters, require the fit's rule on every validation row, integrate each rule's value by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$ (and 64 Gauss-Hermite nodes of the size-biased cluster effect), and weight fold $v$ by $W_v/W$ from the fit's own data and folds. The harness raises unless these weights equal the record's `fold_weights` to $10^{-12}$ |
| stored variance | rows and weights: $\{n(n-1)\}^{-1}\sum_i (\phi_i - \bar\phi)^2$, with $\bar\phi = 0$ to solver tolerance and mean-one weights. Clusters: $J\,\widehat{\mathrm{var}}(T_j)/n^2$ of the cluster totals $T_j$ with divisor $J - 1$, on the normal reference (the $t$ rule applies below 40 clusters). The interval is $\hat\psi \pm 1.96\,\mathrm{SE}$ |
| primary budget | 2,000 replications for each setting |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20263071` and `resampling_seed=20263072` |
| policy | `gated` |

| setting | law, $b(W)$ | what the fit adds | n |
| --- | --- | --- | ---: |
| `rows` | `non_exceptional`, $0.1 + W_1$ | nothing: ten equal folds, so $W_v/W = 1/V$ and the cell measures the pooled centring | 2,000 |
| `weighted` | `misspecified_limit`, $0.8 W_1 + W_1^2 - 0.3$ | the known weight $w = 1 + 1.5 \cdot 1\{W_1 > 0\}$ through `weights=`, the law of the [compositions study](learned-rule-compositions.md). The folds hold equal rows and unequal masses, so $W_v/W \neq 1/V$ | 2,000 |
| `clustered` | `non_exceptional` plus $1.0\, z_j$ on the logit scale, $z_j \sim N(0, 1)$ per cluster | `id=` with 200 clusters of size $N_j = 5 + \operatorname{Binomial}(10, \operatorname{expit}(1.5 z_j))$, so a cluster's size rises with its outcome effect. Folds move whole clusters | about 2,000 (1,856 to 2,179 realized) |

At the limit rule of the cluster law the correlation of a cluster's size with its mean curve is
0.891, and the curve's intra-cluster correlation under the row law is 0.077. Under clusters the
row-level $\bar Q_0$ is the size-biased marginal over $z$. From the law alone, before any run,
the ratio-bias oracle of `tests/unit/test_fold_evaluated_ratio_bias.py` put the fold-evaluated
point at -0.120 stacked SDs and the stacked point at about -0.01 SD at 200 clusters in 10 folds,
and at -0.29 and -0.03 SD at 40 clusters. `tests/unit/test_learned_rule_stacked_design.py` pins
the law's constants, the oracle, the seeds and the sizing.

The X11 plan, its review and the study design record fixed the settings, the learners, the
budgets, the seeds, the truth rule, the variance rules and the `unequal_clusters` rule before any
run. `tests/unit/test_learned_rule_stacked.py` holds the unit witnesses of the construction, and
`tests/unit/test_learned_rule_stacked_harness.py` checks the laws, the harness, the stored
variance rules and the variance terms.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.1 + W1 and a cluster effect of SD 1.0 on the logit scale, 200 clusters of size 5 + Binomial(10, expit(1.5 z)) that rises with the effect | `ey_learned_rule[learned rule]` | fold-mass-weighted average of the value of the rule learned on each fold's training rows | `cleverly` stacked report of the learned-rule value: one pooled fluctuation over every validation row, each fold weighted by its share of the weight mass, and the curve centred at the pooled estimate | -0.000563 to 0.0016 | 0.9565 | 1.0088 | pass |
| learned-rule law with the blip 0.1 + W1, rows only, ten equal folds | `ey_learned_rule[learned rule]` | fold-mass-weighted average of the value of the rule learned on each fold's training rows | `cleverly` stacked report of the learned-rule value: one pooled fluctuation over every validation row, each fold weighted by its share of the weight mass, and the curve centred at the pooled estimate | -0.000936 to 0.000759 | 0.9520 | 1.0153 | pass |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight 1 + 1.5 * 1{W1 > 0}, a misspecified outcome learner | `ey_learned_rule[learned rule]` | fold-mass-weighted average of the value of the rule learned on each fold's training rows | `cleverly` stacked report of the learned-rule value: one pooled fluctuation over every validation row, each fold weighted by its share of the weight mass, and the curve centred at the pooled estimate | -0.000423 to 0.0014 | 0.9515 | 1.0136 | pass |
<!-- /generated -->

## Theory properties

Every property cell fits nominal n = 2,000 with the declared learners.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each positive lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, with 8,800 replications of each setting | a shrunken-SE control multiplies each SE by 0.80, and a noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. On the clustered setting, `row_variance_control` reads the same estimate with the row-level centred SE of the same curve, which ignores the clusters (limit ratio 0.771). Each control must put its SE-ratio interval below 0.93 |
| `fold_evaluated_reporting` | how far the fold-evaluated report (`cv_evaluation=True`) of every clustered calibration draw sits from its own target, beside the stacked report on the same draw, folds and rules | nothing fails. The cell has the role `diagnostic`, and no margin is read |

The noise scale is the SD of $\sqrt{n}$ times the estimate at each setting's limit rule, and no
verdict reads it as an efficiency bound. The size was set before the run by the per-arm sizing
rule over the three positives on a 4,000-draw pilot at the declared size. The reporting fit uses
the same fold seed as the stacked fit, and the harness checks on every draw that the two fits
drew the same partition; its truth weights the same fold values by $1/V$.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `fold_evaluated_reporting` | `clustered__fold_evaluated` | diagnostic | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters whose size rises with the effect: the fold-evaluated fit (cv_evaluation=True) on every clustered calibration draw, with the same folds and rules, read against the 1/V average of the same fold values | none. The cell reports the fold-evaluated bias beside the stacked bias and their paired error difference | bias -0.0026 to -0.0015, coverage 0.9366 to 0.9495, paired error difference -0.0021 | reported |
| `interval_calibration` | `clustered__declared` | positive | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters whose size rises with the effect: the declared stacked fit with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9365 to 0.9494, SE ratio 0.9684 to 1.0066 | pass |
| `interval_calibration` | `clustered__noise_control` | control | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters whose size rises with the effect: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8191 to 0.8399, SE ratio 0.6933 to 0.7211 | pass |
| `interval_calibration` | `clustered__row_variance_control` | control | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters whose size rises with the effect: the same estimate with the row-level centred standard error that ignores the clusters | the SE-ratio interval must fall below the calibration band | coverage 0.8535 to 0.8725, SE ratio 0.7476 to 0.7771 | pass |
| `interval_calibration` | `clustered__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters whose size rises with the effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8652 to 0.8835, SE ratio 0.7743 to 0.8054 | pass |
| `interval_calibration` | `rows__declared` | positive | the learned-rule law with the blip 0.1 + W1 on rows, in ten equal folds: the declared stacked fit with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9400 to 0.9525, SE ratio 0.9740 to 1.0128 | pass |
| `interval_calibration` | `rows__noise_control` | control | the learned-rule law with the blip 0.1 + W1 on rows, in ten equal folds: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8263 to 0.8467, SE ratio 0.6978 to 0.7247 | pass |
| `interval_calibration` | `rows__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1 on rows, in ten equal folds: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8698 to 0.8878, SE ratio 0.7793 to 0.8109 | pass |
| `interval_calibration` | `weighted__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight 1 + 1.5 * 1{W1 > 0}: the declared stacked fit with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9547, SE ratio 0.9674 to 1.0042 | pass |
| `interval_calibration` | `weighted__noise_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight 1 + 1.5 * 1{W1 > 0}: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8145 to 0.8354, SE ratio 0.6887 to 0.7154 | pass |
| `interval_calibration` | `weighted__shrunken_se_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight 1 + 1.5 * 1{W1 > 0}: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8675 to 0.8857, SE ratio 0.7734 to 0.8040 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed. The 3 primary tests have a worst
standardized bias of 0.0309, coverage of 0.9515 to 0.9565, and SE ratios of 1.009 to 1.015. The 3
calibration positives have SE ratios of 0.9855 to 0.9930 and coverage of 0.9432 to 0.9489, and
every interval lies inside its band. `weighted` was the tightest positive, as the sizing pilot
predicted.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 3 of 3 | `performance-tests.csv` |
| gated property cells passed | 10 of 10 | `properties.csv` |
| diagnostic cells, reported with no verdict | 1 | `properties.csv`, `fold_evaluated_reporting` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |
| smallest calibration SE-ratio lower endpoint | 0.9674 | `properties.csv`, `weighted__declared` |
| smallest calibration coverage lower endpoint | 0.9365 | `properties.csv`, `clustered__declared` |

The `weighted` SE ratio below 1 is a property of its law. The fold-evaluated compositions study
read 0.969 on the same law, and the oracle SE over the empirical SD is already about 0.97 there.

### The cluster cell and the `unequal_clusters` rule

A learned-rule fit with clusters of unequal sizes keys the qualifier `unequal_clusters`, because
sizes that depend on the outcome put a ratio bias in the fold-evaluated point. The study declared
before any run the rule that decides whether a stacked fit drops it (`UNEQUAL_CLUSTERS_RULE` in
`tests/studies/learned_rule_stacked.py`). Every clause holds on the committed results.

| clause | reading |
| --- | --- |
| `clustered__declared` passes the calibration rule | SE ratio 0.9684 to 1.0066, coverage 0.9365 to 0.9494 |
| `clustered__declared` establishes its bias inside the margin | standardized bias 0.0024 |
| the row-variance, shrunken-SE and noise controls of the cell each fail | SE-ratio upper endpoints 0.7771, 0.8054 and 0.7211, against 0.93 |
| no clustered replication was lost beyond the failure bound | 0 failed of 8,800 |
| the reporting row's paired error difference is negative, with its upper 99% bound below zero | -0.00213 with MC SE 0.0000151 |

So the registered row `T42.clustered` declares `unequal_clusters`, and its limits state the
measured scope: about 200 clusters of size 5 to 15 in 10 folds, the size rising with the
cluster's outcome effect, with no weights. Fits with 40 to 199 clusters share the key, and the
interval was not measured there.

The fold-evaluated report keeps the qualifier, by theory and not by a failed verdict. Its ratio
bias is of order $V/J$ in absolute terms, so of order $V/\sqrt{J}$ in SDs, against $1/\sqrt{J}$
for the stacked point. On the same draws the fold-evaluated estimate sits 0.00213 below the
stacked one, about 0.11 stacked SDs, and its standardized bias is -0.107. That bias lies inside
the 0.25 margin, and its interval reads an SE ratio of 0.986 and coverage of 0.9433, so at 200
clusters it would pass the same bands. The oracle puts its bias at about -0.29 SD at 40 clusters.

### The variance terms

The stacked variance is not guaranteed to be at least the fold-evaluated second moment, because
the cross term of the variance identity can be negative. Over the primary replications the mean
cross term is -0.09% (`rows`), -0.11% (`weighted`) and -0.16% (`clustered`) of that second moment,
and the mean between-fold term +0.02% to +0.03%. Together they move the SE by under 0.1%, far
below the 99% SE-ratio half-width of about 0.02 at 8,800 replications, so no calibration reading
here is attributed to them. Both terms scale as $V/n$ and are unmeasured at a smaller n or a
larger V.

### What the verdicts cannot see

In these cells the $W_v/W$ target and the $1/V$ fold average differ by far less than the
resolution of any verdict. So no calibration verdict tells the declared target from the fold
average. The unit witnesses of `tests/unit/test_learned_rule_stacked.py` and the harness's
fold-weight check establish the target's weighting, and the reporting row's paired difference
establishes the estimator's.

The readings do not show the items in this list.

- Strata, fold targeting, a rule-learner loss or `repeats` above 1 under the stacked report. Each
  reads experimental.
- Weights with clusters, or fewer than 40 clusters on the $t$ reference.
- Another sample size, fold count or learner.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the SD of $\sqrt{n}$ times the estimate at the limit rule over $\sqrt{2000}$. They size the noise
controls only, and no verdict reads them as an efficiency bound. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications per setting |
| `n` | 2000 | nominal observations per primary replication |
| `independent_tests_total` | 3 | parameter tests against the truth |
| `independent_tests_passed` | 3 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 10 | gated repeated-sampling property cells |
| `property_cells_passed` | 10 | cells whose own and family verdicts pass |
| `property_cells_reported` | 1 | diagnostic cells, reported with no verdict |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0309 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9515 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9378 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9704 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0600 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-stacked/clustered/ey_learned_rule[learned rule]]:coverage` | 0.9565 | primary coverage, clustered |
| `performance[cleverly-learned-rule-stacked/clustered/ey_learned_rule[learned rule]]:se_ratio` | 1.0088 | primary SE ratio, clustered |
| `performance[cleverly-learned-rule-stacked/weighted/ey_learned_rule[learned rule]]:standardized_bias` | 0.0309 | the largest primary standardized bias |
| `summary[cleverly-learned-rule-stacked/clustered/ey_learned_rule[learned rule]]:truth_min` | 0.5943 | the smallest clustered truth |
| `summary[cleverly-learned-rule-stacked/clustered/ey_learned_rule[learned rule]]:truth_max` | 0.6187 | the largest clustered truth |
| `cross_term_mean_rows` | -0.000917 | mean cross term over the fold-evaluated second moment, rows |
| `cross_term_mean_weighted` | -0.0011 | the same, weighted |
| `cross_term_mean_clustered` | -0.0016 | the same, clustered |
| `between_term_mean_rows` | 0.000272 | mean between-fold term over the same second moment, rows |
| `between_term_mean_weighted` | 0.000224 | the same, weighted |
| `between_term_mean_clustered` | 0.000241 | the same, clustered |
| `properties[interval_calibration/rows__declared]:se_ratio` | 0.9930 | calibration SE ratio, rows |
| `properties[interval_calibration/rows__declared]:coverage` | 0.9465 | calibration coverage, rows |
| `properties[interval_calibration/weighted__declared]:se_ratio` | 0.9855 | calibration SE ratio, weighted |
| `properties[interval_calibration/weighted__declared]:se_ratio_ci_lower` | 0.9674 | its lower endpoint, against 0.93 |
| `properties[interval_calibration/weighted__declared]:coverage` | 0.9489 | calibration coverage, weighted |
| `properties[interval_calibration/clustered__declared]:se_ratio` | 0.9871 | calibration SE ratio, clustered |
| `properties[interval_calibration/clustered__declared]:se_ratio_ci_lower` | 0.9684 | its lower endpoint |
| `properties[interval_calibration/clustered__declared]:se_ratio_ci_upper` | 1.0066 | its upper endpoint |
| `properties[interval_calibration/clustered__declared]:coverage` | 0.9432 | calibration coverage, clustered |
| `properties[interval_calibration/clustered__declared]:coverage_ci_lower` | 0.9365 | its lower endpoint, against 0.92 |
| `properties[interval_calibration/clustered__declared]:coverage_ci_upper` | 0.9494 | its upper endpoint |
| `properties[interval_calibration/clustered__declared]:standardized_bias` | 0.0024 | calibration standardized bias, clustered |
| `properties[interval_calibration/clustered__declared]:failed_replicates` | 0 | clustered calibration fits lost |
| `properties[interval_calibration/clustered__row_variance_control]:se_ratio` | 0.7621 | row-variance control SE ratio |
| `properties[interval_calibration/clustered__row_variance_control]:se_ratio_ci_upper` | 0.7771 | its upper endpoint, against 0.93 |
| `properties[interval_calibration/clustered__shrunken_se_control]:se_ratio_ci_upper` | 0.8054 | shrunken-SE control upper endpoint, clustered |
| `properties[interval_calibration/clustered__noise_control]:se_ratio_ci_upper` | 0.7211 | noise control upper endpoint, clustered |
| `properties[interval_calibration/rows__shrunken_se_control]:se_ratio_ci_upper` | 0.8109 | the highest shrunken-SE control upper endpoint |
| `properties[interval_calibration/rows__noise_control]:se_ratio_ci_upper` | 0.7247 | the highest noise-control upper endpoint |
| `properties[fold_evaluated_reporting/clustered__fold_evaluated]:standardized_bias` | -0.1070 | the fold-evaluated report's standardized bias |
| `properties[fold_evaluated_reporting/clustered__fold_evaluated]:se_ratio` | 0.9858 | its SE ratio |
| `properties[fold_evaluated_reporting/clustered__fold_evaluated]:coverage` | 0.9433 | its coverage |
| `properties[fold_evaluated_reporting/clustered__fold_evaluated]:stacked_standardized_bias` | 0.0024 | the stacked report's standardized bias on the same draws |
| `properties[fold_evaluated_reporting/clustered__fold_evaluated]:paired_error_difference` | -0.0021 | fold-evaluated error minus stacked error |
| `properties[fold_evaluated_reporting/clustered__fold_evaluated]:paired_error_difference_se` | 0.000015 | its Monte Carlo SE |
| `bound:rows_standard_error` | 0.0149 | the limit-rule SD over the root of n, rows |
| `bound:weighted_standard_error` | 0.0160 | the same, weighted |
| `bound:clustered_standard_error` | 0.0191 | the same, clustered |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.8000 | factor the shrunken-SE control applies to each SE |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 20 | failed primary fits that bound allows per setting |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | R `polle`'s Algorithm 4 is the construction's source, not a pinned comparator. The row rests on accuracy against the truth and on the theory properties |
| One size, ten folds and one learner class | The study fits nominal n = 2,000 with 10 folds, `repeats=1`, pooled targeting, complete data and the RM30 logistic learners, with the plug-in rule. The support key holds neither the size, the fold count nor the learner, so another n, V or learner reads supported and was not measured |
| Compositions that were not measured | Strata, fold targeting, a rule-learner loss, `repeats` above 1 and weights with clusters under the stacked report match no row and read experimental |
| A data-adaptive target that weights folds by their share | The target weights each fold by its share of the rows, or of the weight mass under `weights=`, so at unequal folds it is not the fold-evaluated target. The calibration verdicts cannot tell the two targets apart in these cells. The weighting rests on the unit witnesses and the harness's fold-weight check |
| The stacked variance can sit below the fold-evaluated second moment | The cross term can be negative. Here it moved the SE by under 0.1%, which the study cannot resolve. The terms scale as $V/n$ and were not measured at a smaller n or a larger V |
| Unequal cluster sizes at the measured scope only | The cell draws about 200 clusters of size 5 to 15 in 10 folds, the size rising with the outcome effect (size-curve correlation 0.891, curve ICC 0.077), with no weights. Fits with 40 to 199 clusters share the key. The oracle puts the stacked ratio bias at about -0.03 SD at 40 clusters, and the interval was not measured there. A fit under the $t$ reference matches no row |
| The fold-evaluated report keeps `unequal_clusters` by theory | Its ratio bias grows as $V/\sqrt{J}$ in SDs: -0.11 stacked SDs here and about -0.29 SD at 40 clusters by the oracle. At 200 clusters its interval would pass the same bands, so the reporting row shows the size of that bias, not a failure |
| The weighted shortfall belongs to the law | The `weighted` SE ratio of 0.986 matches the fold-evaluated compositions cell of the same law, 0.969 |
| The omitted-variable bound refuses | The bound of a stacked fit refuses as part (n2) of [X11](../../roadmap.md#x11-learned-policy-follow-ups) |
| Binary treatment and outcome | Every setting has a binary treatment and a binary outcome |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_stacked/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_stacked/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_stacked/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_stacked/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_stacked/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_stacked/properties.csv)
carry every published row.
