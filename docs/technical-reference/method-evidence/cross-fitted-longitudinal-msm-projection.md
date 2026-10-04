# Cross-fitted longitudinal MSM projection

This study validates the cross-fitted identity-link projection over four censored two-time-point
treatment plans. The subject is the [ordinary longitudinal MSM projection](ordinary-longitudinal-msm-projection.md)
subject at five outer folds. Each fold runs the untargeted recursion of every plan on its training
rows, and one stacked fluctuation per node targets every follower of every plan. The
[longitudinal projection](../msm-projections.md#the-longitudinal-projection) states the
construction and its theory.

No pinned comparator cross-fits a longitudinal MSM. The R comparison fits each plan with pinned
`lmtp` 1.5.4 on the same realized folds, then applies the declared projection to the four estimates
and their joint influence curves. The design has no effect modifier, so that projection is the
same estimand.

## What was compared

| setting | `cleverly` | projected R `lmtp` fits |
| --- | --- | --- |
| datasets | 800 censored two-time-point samples of 2,500 rows | the identical rows |
| folds | five outer folds, from the row count and `random_state=0` | the realized fold column |
| plans | never, always, early-only, and a dynamic second-node rule | the same four plans |
| working model | identity-link intercept and treatment-duration terms | the same fixed projection after four cross-fitted fits |
| projection weights | 0.1, 10, 0.1, and 10 in declared plan order | the same weights |
| mechanisms | the generating treatment and censoring probabilities | the same, as exact per-node density ratios |
| targeting | one stacked fluctuation per node over every follower of every plan | one scalar fluctuation per plan, on each training fold |
| intervals | pointwise 95% Wald from the joint coefficient curve | the projected joint regimen curves |

The two constructions differ twice. `lmtp` fluctuates on each training fold, and it targets each
plan alone. Upstream commit `9996b04` classifies the 1.5.4 training-fold update as a bug, so the
paired verdicts compare two constructions and validate neither fold-local update.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring and four projected treatment plans, fitted with five outer folds | `msm_regimen[(intercept)]` | longitudinal regimen MSM projection intercept coefficient | `cleverly` cross-fitted longitudinal MSM projection | -0.000671 to 0.0047 | 0.9375 | 0.9761 | pass |
| two-time-point law with monotone censoring and four projected treatment plans, fitted with five outer folds | `msm_regimen[(intercept)]` | longitudinal regimen MSM projection intercept coefficient | projected R `lmtp` cross-fitted regimen fits | -0.000757 to 0.0045 | 0.9387 | 0.9822 | pass |
| two-time-point law with monotone censoring and four projected treatment plans, fitted with five outer folds | `msm_regimen[duration]` | longitudinal regimen MSM projection treatment-duration coefficient | `cleverly` cross-fitted longitudinal MSM projection | -0.0025 to 0.000337 | 0.9563 | 1.0158 | pass |
| two-time-point law with monotone censoring and four projected treatment plans, fitted with five outer folds | `msm_regimen[duration]` | longitudinal regimen MSM projection treatment-duration coefficient | projected R `lmtp` cross-fitted regimen fits | -0.0024 to 0.000370 | 0.9463 | 1.0252 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring and four projected treatment plans, fitted with five outer folds | `msm_regimen[(intercept)]` | longitudinal regimen MSM projection intercept coefficient | 0.000138 | 0.0314 | 1.0315 | -0.0012 | 0.0105 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring and four projected treatment plans, fitted with five outer folds | `msm_regimen[duration]` | longitudinal regimen MSM projection treatment-duration coefficient | -0.000064 | 0.0277 | 1.0345 | 0.0100 | 0.0247 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_msm` | positive | five-fold longitudinal MSM projection with fully grown outcome trees | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9840 to 1.0256 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3802 to 0.3963 | pass |
| `double_robustness` | `(intercept)__both_correct` | positive | intercept coefficient: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0044 to 0.0071, margin 0.0176, SE ratio 1.0224 | pass |
| `double_robustness` | `(intercept)__both_wrong` | control | intercept coefficient: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0459 to 0.0574, margin 0.0176, SE ratio 0.9170 | pass |
| `double_robustness` | `(intercept)__mechanism_correct` | positive | intercept coefficient: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000076 to 0.0118, margin 0.0182, SE ratio 1.0094 | pass |
| `double_robustness` | `(intercept)__outcome_correct` | positive | intercept coefficient: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0056 to 0.0063, margin 0.0181, SE ratio 0.8592 | pass |
| `double_robustness` | `duration__both_correct` | positive | treatment-duration coefficient: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0051 to 0.0039, margin 0.0137, SE ratio 1.0063 | pass |
| `double_robustness` | `duration__both_wrong` | control | treatment-duration coefficient: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0431 to -0.0342, margin 0.0136, SE ratio 0.7182 | pass |
| `double_robustness` | `duration__mechanism_correct` | positive | treatment-duration coefficient: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0080 to 0.000908, margin 0.0137, SE ratio 1.0261 | pass |
| `double_robustness` | `duration__outcome_correct` | positive | treatment-duration coefficient: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0056 to 0.0035, margin 0.0139, SE ratio 0.6871 | pass |
| `in_sample_agreement` | `(intercept)__in_sample_agreement` | diagnostic | intercept coefficient: the absolute difference from the in-sample coefficient on the same draw, over the cross-fitted standard error; mean_abs_difference_over_se is its mean | none; the row reports the statistic | mean absolute difference / SE 0.0287, in-sample estimate inside the cross-fitted interval 1 | reported |
| `in_sample_agreement` | `duration__in_sample_agreement` | diagnostic | treatment-duration coefficient: the absolute difference from the in-sample coefficient on the same draw, over the cross-fitted standard error; mean_abs_difference_over_se is its mean | none; the row reports the statistic | mean absolute difference / SE 0.0335, in-sample estimate inside the cross-fitted interval 1 | reported |
| `interval_calibration` | `(intercept)__correctly_specified` | positive | intercept coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9297 to 0.9493, SE ratio 0.9548 to 1.0099, empirical efficiency ratio 1.0054 to 1.0629, reported efficiency ratio 1.0134 to 1.0164 | pass |
| `interval_calibration` | `(intercept)__noise_control` | control | intercept coefficient: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8199 to 0.8503, SE ratio 0.6894 to 0.7297, empirical efficiency ratio 1.3910 to 1.4721, reported efficiency ratio 1.0134 to 1.0165 | pass |
| `interval_calibration` | `(intercept)__shrunken_se_control` | control | intercept coefficient: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8020 to 0.8337, SE ratio 0.6689 to 0.7081, empirical efficiency ratio 1.0037 to 1.0619, reported efficiency ratio 0.7094 to 0.7115 | pass |
| `interval_calibration` | `duration__correctly_specified` | positive | treatment-duration coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9275 to 0.9474, SE ratio 0.9545 to 1.0113, empirical efficiency ratio 1.0103 to 1.0689, reported efficiency ratio 1.0185 to 1.0232 | pass |
| `interval_calibration` | `duration__noise_control` | control | treatment-duration coefficient: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8181 to 0.8486, SE ratio 0.6765 to 0.7183, empirical efficiency ratio 1.4214 to 1.5082, reported efficiency ratio 1.0185 to 1.0231 | pass |
| `interval_calibration` | `duration__shrunken_se_control` | control | treatment-duration coefficient: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8010 to 0.8327, SE ratio 0.6686 to 0.7075, empirical efficiency ratio 1.0104 to 1.0682, reported efficiency ratio 0.7129 to 0.7163 | pass |
| `interval_calibration` | `duration_logit__correctly_specified` | positive | the treatment-duration coefficient of a logit working model: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9404 to 0.9585, SE ratio 0.9732 to 1.0306, empirical efficiency ratio 1.0000 to 1.0596, reported efficiency ratio 1.0286 to 1.0332 | pass |
| `interval_calibration` | `duration_logit__noise_control` | control | the treatment-duration coefficient of a logit working model: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8323 to 0.8618, SE ratio 0.7086 to 0.7505, empirical efficiency ratio 1.3733 to 1.4549, reported efficiency ratio 1.0286 to 1.0330 | pass |
| `interval_calibration` | `duration_logit__shrunken_se_control` | control | the treatment-duration coefficient of a logit working model: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8165 to 0.8472, SE ratio 0.6816 to 0.7211, empirical efficiency ratio 1.0007 to 1.0590, reported efficiency ratio 0.7200 to 0.7231 | pass |
| `power` | `duration__alternative` | positive | treatment-duration coefficient: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9938, 0.9824 to 0.9987 | pass |
| `projection_necessity` | `duration__declared_weights` | positive | treatment-duration coefficient: the working model uses its declared nonuniform projection weights | bias interval inside the equivalence margin | bias -0.0064 to 0.0025, margin 0.0137 | pass |
| `projection_necessity` | `duration__uniform_weights` | control | treatment-duration coefficient: the identical working model is projected under uniform weights | bias interval must fall entirely outside the margin | bias -0.1051 to -0.1002, margin 0.0075 | pass |
| `root_n_and_efficiency` | `(intercept)__n_1000` | control | intercept coefficient: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0029, coverage 0.9265 to 0.9699, SE ratio 1.0428 | pass |
| `root_n_and_efficiency` | `(intercept)__n_2000` | positive | intercept coefficient: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0023, coverage 0.9147 to 0.9619, SE ratio 0.9773 | pass |
| `root_n_and_efficiency` | `(intercept)__n_8000` | positive | intercept coefficient: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000142, coverage 0.9351 to 0.9755, SE ratio 1.0385 | pass |
| `root_n_and_efficiency` | `duration__n_1000` | control | treatment-duration coefficient: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias 0.0015, coverage 0.9214 to 0.9665, SE ratio 1.0182 | pass |
| `root_n_and_efficiency` | `duration__n_2000` | positive | treatment-duration coefficient: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000688, coverage 0.9231 to 0.9677, SE ratio 0.9991 | pass |
| `root_n_and_efficiency` | `duration__n_8000` | positive | treatment-duration coefficient: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000276, coverage 0.9369 to 0.9766, SE ratio 1.0192 | pass |
| `root_n_rate` | `(intercept)__empirical_sd` | positive | intercept coefficient: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5670 to -0.4661 | pass |
| `root_n_rate` | `(intercept)__reported_se` | positive | intercept coefficient: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5143 to -0.5097 | pass |
| `root_n_rate` | `duration__empirical_sd` | positive | treatment-duration coefficient: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5697 to -0.4687 | pass |
| `root_n_rate` | `duration__reported_se` | positive | treatment-duration coefficient: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5206 to -0.5133 | pass |
| `simultaneous_coverage` | `cross_fitted_msm__pointwise_joint_control` | control | the two terms of the cross-fitted longitudinal MSM projection: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.9120 to 0.9339 | pass |
| `simultaneous_coverage` | `cross_fitted_msm__simultaneous_band` | positive | the two terms of the cross-fitted longitudinal MSM projection: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9391 to 0.9573 | pass |
| `targeting_necessity` | `duration__targeted` | positive | treatment-duration coefficient: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0036 to 0.0056, margin 0.0140 | pass |
| `targeting_necessity` | `duration__untargeted` | control | treatment-duration coefficient: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0486 to -0.0394, margin 0.0141 | pass |
| `type_i_error` | `duration__sharp_null` | positive | treatment-duration coefficient: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0700, 0.0488 to 0.0965 | pass |
<!-- /generated -->

Every property family but two uses the exact censored binary support law of the ordinary study.
Its regimen truths and influence curves are derived independently before the fixed projection is
applied. The logit truth is a longhand Newton solve of the weighted logistic projection. The
overfitting pair and the band cell use the primary law. The untargeted arm is the projection of
the same fit's stitched initial fit.

`in_sample_agreement` is reported, not gated. Each row is
$\lvert\hat\beta_{\mathrm{cf}} - \hat\beta_{\mathrm{in}}\rvert / \widehat{SE}_{\mathrm{cf}}$ on one
draw of the n = 8,000 rung, and the summary publishes its mean.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2500 | observations per paired replication |
| `independent_tests_passed` | 4 | truth tests passing |
| `independent_tests_total` | 4 | truth tests reported |
| `paired_tests_passed` | 2 | paired comparisons passing |
| `paired_tests_total` | 2 | paired comparisons reported |
| `property_cells_passed` | 37 | property cells passing |
| `property_cells_total` | 37 | property cells reported |
| `max_standardized_bias` | 0.0696 | largest primary standardized bias |
| `min_coverage` | 0.9375 | lowest primary coverage |
| `max_margin_utilization` | 0.0314 | largest paired similarity-margin share |
| `properties[in_sample_agreement/(intercept)__in_sample_agreement]:mean_abs_difference_over_se` | 0.0287 | mean in-sample distance, intercept, in cross-fitted SEs |
| `properties[in_sample_agreement/duration__in_sample_agreement]:mean_abs_difference_over_se` | 0.0335 | mean in-sample distance, duration, in cross-fitted SEs |
| `properties[crossfit_overfitting/cross_fitted_msm]:coverage` | 0.9504 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5490 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_msm]:replicates` | 8000 | paired overfitting replications |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:alpha` | 0.0500 | test size |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:projection_displacement` | 0.2500 | minimum projection displacement |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

## Limits

| limit | what it means for use |
| --- | --- |
| one law, four plans and two time points | the evidence covers one censored binary law, with the projection weights, plan durations and terms fixed before fitting |
| no repeated-sampling cell for survival, competing-risk, weighted or clustered projections | these compositions have the fast-tier exact identities of `tests/unit/test_cross_fitted_longitudinal_msm.py` only |
| the n = 500 rung was dropped before the run | its failure-only probe found 5 failures in 2,000 draws: a training fold can hold only events among the followers of `always`. The n = 1,000 rung replaced it as the ladder's control rung |
| the comparator runs a different construction | `lmtp` 1.5.4 fluctuates on each training fold and targets each plan alone, so each paired verdict compares two constructions |
| the mechanism is supplied | both implementations receive the generating probabilities, so the paired row says nothing about mechanism estimation. The double-robustness cells cover misspecified mechanisms |
| one fixed five-fold assignment | the row does not validate repeated folds or time-respecting splits |
| learner caution | the tree pair validates held-out prediction with flexible learners, not learner-library selection. With `id=`, Wang et al. (2024), Remark 3, caution against complex learners at few clusters |
