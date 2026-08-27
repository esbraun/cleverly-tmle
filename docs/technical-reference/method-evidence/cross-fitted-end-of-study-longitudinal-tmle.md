# Cross-fitted end-of-study longitudinal TMLE

This study validates the five-fold end-of-study recursion. Each outer fold fits and targets a
complete recursion on its training rows. The estimator stitches predictions and influence curves
only on held-out rows.

The canonical comparison uses R [`lmtp`](https://github.com/nt-williams/lmtp) 1.5.4 at commit
`f04a2b4`, which is the maintained package that implements a cross-fitted sequential regression.
R `ltmle`, the comparator the two ordinary longitudinal rows use, has no cross-fitting at all, so
it cannot witness this construction.

Agreement with R is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_longitudinal.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_longitudinal.py).

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets and folds | 1,600 censored panels, each with one exact five-fold assignment | the identical rows and the identical assignment, read from the panel |
| plans | never treat, always treat, and continue after initial treatment when L2 is positive | the same three, as shifted treatment columns |
| contrasts | always-minus-never and dynamic-minus-never | the same, from the difference of the two rowwise influence curves so covariance is preserved |
| treatment and censoring mechanisms | the generating probabilities from the law | the same probabilities, supplied as exact per-node density ratios |
| sequential regressions | quasibinomial GLMs fitted within each outer training set | `SL.glm`, fitted within the same training sets |
| targeting | a complete fold-specific backward recursion | the same, through `cf_tmle` and `theta_dr` |
| cumulative-g bounds | nonbinding | nonbinding, `.trim = 1` |
| intervals | pointwise 95% identity-scale Wald intervals | the same, from the returned influence curve |

Both implementations receive the mechanism from the law rather than estimating it. `lmtp` has no
`gform` argument, so the adapter substitutes exact per-node density ratios into it, the same way
it substitutes the fold assignment. The substitution is checked on every run against `lmtp`'s own
estimate: the zero pattern must agree cell for cell, and the cumulative ratio must track it.

That choice is what makes the comparison a comparison. An earlier version let each side estimate
the mechanism its own way, and the result measured two unrelated pipelines. `lmtp` fits its ratio
with `SL.glm`, whose linear logit cannot represent the exact classifier log-odds `-log g` for a
deterministic regime, so the ratio came out shrunken. A shrunken clever covariate under-targets
and understates the influence curve: coverage ran from 0.75 to 0.91 and the SE ratio from 0.60 to
0.86, against 0.949 to 0.957 and 1.00 to 1.02 for `cleverly` on the same panels. The sequential
regressions stay misspecified on both sides, so targeting still has work to do.

The exact-law structural test fixes the fold assignment and repeats every support point in each
fold. It checks every regimen mean and correlated contrast against the functional and Gateaux EIF.
The leakage tests also require held-out outcome predictions at both recursion nodes.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` cross-fitted LTMLE | -0.0019 to 0.0022 | 0.9431 | 0.9839 | pass |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` | -0.0018 to 0.0023 | 0.9406 | 0.9831 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` cross-fitted LTMLE | -0.0014 to 0.0028 | 0.9525 | 0.9986 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` | -0.0015 to 0.0026 | 0.9506 | 1.0001 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` cross-fitted LTMLE | -0.000844 to 0.0016 | 0.9531 | 1.0156 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` | -0.000873 to 0.0015 | 0.9519 | 1.0171 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` cross-fitted LTMLE | -0.0014 to 0.0018 | 0.9506 | 0.9998 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` | -0.0015 to 0.0017 | 0.9469 | 1.0012 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` cross-fitted LTMLE | -0.000445 to 0.0022 | 0.9531 | 1.0028 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` | -0.000648 to 0.0019 | 0.9506 | 1.0048 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -0.000063 | 0.0132 | 1.0064 | 0.0025 | 0.0044 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 0.000120 | 0.0248 | 1.0131 | 0.0019 | 0.0049 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | 0.000034 | 0.0122 | 1.0092 | 0.0012 | 0.0059 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 0.000097 | 0.0256 | 1.0091 | 0.0037 | 0.0059 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | 0.000217 | 0.0718 | 1.0199 | 0.0025 | 0.0100 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_ltmle` | positive | five-fold end-of-study LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.1486 to 1.1965 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3463 to 0.3603 | pass |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0022 to 0.0015, margin 0.0063, SE ratio 1.0020 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0165 to 0.0206, margin 0.0069, SE ratio 1.0012 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0018 to 0.0022, margin 0.0067, SE ratio 0.9937 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0021 to 0.0017, margin 0.0063, SE ratio 1.0257 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0060 to 0.0026, margin 0.0144, SE ratio 1.0179 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0247 to -0.0162, margin 0.0143, SE ratio 0.6892 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0028 to 0.0058, margin 0.0145, SE ratio 1.0357 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0062 to 0.0029, margin 0.0153, SE ratio 0.6182 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9388 to 0.9619, SE ratio 0.9832 to 1.0640, empirical efficiency ratio 0.9530 to 1.0304, reported efficiency ratio 1.0106 to 1.0163 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8098 to 0.8497, SE ratio 0.6952 to 0.7478, empirical efficiency ratio 1.3560 to 1.4571, reported efficiency ratio 1.0106 to 1.0163 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8129 to 0.8525, SE ratio 0.6888 to 0.7445, empirical efficiency ratio 0.9533 to 1.0294, reported efficiency ratio 0.7074 to 0.7114 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9305 to 0.9552, SE ratio 0.9714 to 1.0481, empirical efficiency ratio 0.9740 to 1.0501, reported efficiency ratio 1.0177 to 1.0235 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8111 to 0.8509, SE ratio 0.6903 to 0.7437, empirical efficiency ratio 1.3730 to 1.4783, reported efficiency ratio 1.0177 to 1.0234 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8076 to 0.8477, SE ratio 0.6801 to 0.7337, empirical efficiency ratio 0.9741 to 1.0504, reported efficiency ratio 0.7124 to 0.7164 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9650, 0.9447 to 0.9796 | pass |
| `root_n_and_efficiency` | `dynamic__n_1000` | control | dynamic plan: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0015, coverage 0.9208 to 0.9637, SE ratio 0.9979 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000376, coverage 0.9223 to 0.9647, SE ratio 0.9995 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000036, coverage 0.9386 to 0.9757, SE ratio 1.0355 | pass |
| `root_n_and_efficiency` | `static__n_1000` | control | static plan: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0037, coverage 0.9121 to 0.9575, SE ratio 1.0067 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0038, coverage 0.9371 to 0.9747, SE ratio 1.0306 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000677, coverage 0.9267 to 0.9677, SE ratio 1.0022 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5730 to -0.4862 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5140 to -0.5074 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5582 to -0.4697 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5211 to -0.5144 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0020 to 0.0018, margin 0.0065 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0232 to 0.0274, margin 0.0070 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0042 to 0.0050, margin 0.0153 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0263 to -0.0173, margin 0.0151 | pass |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0425, 0.0263 to 0.0644 | pass |
<!-- /generated -->

The property study preserves the ordinary row's double-robustness, rate, efficiency, calibration,
null, power, and targeting instruments. It runs each positive estimator with five outer folds.

The overfitting pair uses identical nonlinear panels and a fully grown outcome tree. The positive
arm predicts held-out rows. The control predicts its training rows. The joint verdict requires an
SE ratio inside the declared band and a predeclared coverage gain over the control.

Read the direction of that ratio, not only the verdict. In-sample fitting understates the standard
error by a factor near three, at 0.3532. Cross-fitting removes the understatement and overshoots
it, at 1.1725. A noisy outcome model inflates the residual term of the influence curve, so a
conservative ratio is the expected direction here rather than an anomaly. The cell establishes
that cross-fitting restores honest inference. It does not establish calibration under a fully
grown tree, and the `interval_calibration` family, which does make a calibration claim, uses
correctly specified nuisances instead.

The gate is close. The 99% interval reaches 1.1965 against a ceiling of 1.2000, so a change of
learner, sample size, or fold count could move this cell across it. The replication budget is
8,000 rather than the shared 400 because the shared budget leaves the interval wider than the
remaining margin. That is legitimate for an equivalence-shaped gate, which more replications make
easier rather than harder, and it is recorded here so the margin cannot be read as comfortable.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_total` | 10 | implementation-estimand truth tests |
| `independent_tests_passed` | 10 | truth tests passing |
| `paired_tests_total` | 5 | paired estimand comparisons |
| `paired_tests_passed` | 5 | paired comparisons passing |
| `property_cells_total` | 32 | independent property cells |
| `property_cells_passed` | 32 | property cells passing |
| `max_standardized_bias` | 0.0425 | largest primary standardized bias |
| `min_coverage` | 0.9406 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9237 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9421 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0650 | highest primary SE-ratio endpoint |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:coverage` | 0.9754 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5081 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:coverage_gain_ci_lower` | 0.4526 | lower bound for the paired coverage gain |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:replicates` | 8000 | paired overfitting replications |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope upper bound |
| `margin:excluded_slope` | -0.2500 | rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

## Limitations

| limitation | what it means for use |
| --- | --- |
| Agreement with `lmtp` is distributional, not numerical | The paired claim is that the mean difference sits inside the similarity margin and that `cleverly` is no worse. The two ordinary rows agree with R `ltmle` to solver precision because both run the identical regression; here the sequential regressions are still fitted differently, so per-replication estimates differ at statistical scale |
| One fixed five-fold assignment is studied | The row does not validate repeated folds or time-respecting splits |
| The cross-fit overfitting cell passes near its ceiling | Its 99% SE-ratio interval reaches 1.1965 against a ceiling of 1.2000. The cell shows that cross-fitting restores honest inference under a fully grown tree. It does not show calibration under one |
| Flexible learning is an independent property instrument | The paired comparison uses one GLM learner on each side. The tree pair validates held-out prediction behavior, not parity for learner-library selection |
| The row reports one terminal mean per plan | Survival curves, competing risks, and longitudinal MSM projections have different parameters |
| Inference is pointwise | The row does not validate simultaneous bands, bootstrap intervals, weights, or clustering |
| The mechanism is supplied rather than estimated | Both implementations receive the generating probabilities, so the paired row says nothing about mechanism estimation or about a severe practical-positivity violation. The property study's double-robustness cells cover misspecified mechanisms separately |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring. Single-correct-nuisance cells establish
consistency only. Calibrated inference uses the cells where both nuisance sequences are correct.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle/manifest.json)
records the seeds, the configuration, the pinned `lmtp` version and source commit, the digest of
every study module and reference source, and the artifact hashes.
