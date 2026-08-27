# Ordinary longitudinal MSM projection

This study validates the ordinary identity-link projection over four censored two-time-point
treatment plans. The working model has an intercept and treatment-duration term, so four regimen
means are reduced to two coefficients under a fixed nonuniform measure. The R comparison fits each
plan with pinned `ltmle`, then applies the declared projection to the estimates and their joint
influence curves. Raw `ltmleMSM` coefficients are not compared because they use a quasibinomial
projection rather than this outcome-scale weighted least-squares parameter.

## What was compared

| setting | `cleverly` | projected R `ltmle` fits |
| --- | --- | --- |
| datasets | 800 censored two-time-point samples | the identical rows |
| plans | never, always, early-only, and a dynamic second-node rule | the same four plans |
| working model | identity-link intercept and treatment-duration terms | the same fixed projection after four ordinary fits |
| projection weights | 0.1, 10, 0.1, and 10 in declared plan order | the same weights |
| mechanisms | the generating treatment and censoring probabilities | the same |
| intervals | pointwise 95% Wald from the joint coefficient curve | the projected joint regimen curves |

Projecting the joint curves is material. Projecting four marginal standard errors would discard
the covariance created because every regimen fit uses the same sample.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring and four projected treatment plans | `msm_regimen[(intercept)]` | longitudinal regimen MSM projection intercept coefficient | `cleverly` | -0.0037 to 0.0015 | 0.9413 | 0.9944 | pass |
| two-time-point law with monotone censoring and four projected treatment plans | `msm_regimen[(intercept)]` | longitudinal regimen MSM projection intercept coefficient | projected R `ltmle` regimen fits | -0.0037 to 0.0015 | 0.9413 | 0.9945 | pass |
| two-time-point law with monotone censoring and four projected treatment plans | `msm_regimen[duration]` | longitudinal regimen MSM projection treatment-duration coefficient | `cleverly` | -0.000999 to 0.0019 | 0.9337 | 0.9805 | pass |
| two-time-point law with monotone censoring and four projected treatment plans | `msm_regimen[duration]` | longitudinal regimen MSM projection treatment-duration coefficient | projected R `ltmle` regimen fits | -0.000998 to 0.0019 | 0.9337 | 0.9807 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring and four projected treatment plans | `msm_regimen[(intercept)]` | longitudinal regimen MSM projection intercept coefficient | 0.000002 | 0.000370 | 1.0003 | 0 | 0.000171 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring and four projected treatment plans | `msm_regimen[duration]` | longitudinal regimen MSM projection treatment-duration coefficient | -8.634e-07 | 0.000362 | 1.0004 | 0 | 0.000181 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `(intercept)__both_correct` | positive | intercept coefficient: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0054 to 0.0067, margin 0.0185, SE ratio 0.9594 | pass |
| `double_robustness` | `(intercept)__both_wrong` | control | intercept coefficient: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0438 to 0.0555, margin 0.0180, SE ratio 0.8918 | pass |
| `double_robustness` | `(intercept)__mechanism_correct` | positive | intercept coefficient: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0058 to 0.0058, margin 0.0178, SE ratio 1.0374 | pass |
| `double_robustness` | `(intercept)__outcome_correct` | positive | intercept coefficient: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0110 to 0.000998, margin 0.0184, SE ratio 0.8304 | pass |
| `double_robustness` | `duration__both_correct` | positive | treatment-duration coefficient: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0057 to 0.0035, margin 0.0141, SE ratio 0.9628 | pass |
| `double_robustness` | `duration__both_wrong` | control | treatment-duration coefficient: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0426 to -0.0334, margin 0.0141, SE ratio 0.6868 | pass |
| `double_robustness` | `duration__mechanism_correct` | positive | treatment-duration coefficient: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0054 to 0.0034, margin 0.0136, SE ratio 1.0418 | pass |
| `double_robustness` | `duration__outcome_correct` | positive | treatment-duration coefficient: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000797 to 0.0085, margin 0.0143, SE ratio 0.6525 | pass |
| `interval_calibration` | `(intercept)__correctly_specified` | positive | intercept coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9383 to 0.9566, SE ratio 0.9569 to 1.0136, empirical efficiency ratio 0.9850 to 1.0434, reported efficiency ratio 0.9969 to 1.0000 | pass |
| `interval_calibration` | `(intercept)__noise_control` | control | intercept coefficient: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8175 to 0.8482, SE ratio 0.6816 to 0.7222, empirical efficiency ratio 1.3825 to 1.4644, reported efficiency ratio 0.9969 to 1.0000 | pass |
| `interval_calibration` | `(intercept)__shrunken_se_control` | control | intercept coefficient: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8023 to 0.8339, SE ratio 0.6701 to 0.7091, empirical efficiency ratio 0.9856 to 1.0432, reported efficiency ratio 0.6978 to 0.7000 | pass |
| `interval_calibration` | `duration__correctly_specified` | positive | treatment-duration coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9364 to 0.9550, SE ratio 0.9469 to 1.0023, empirical efficiency ratio 0.9944 to 1.0518, reported efficiency ratio 0.9939 to 0.9987 | pass |
| `interval_calibration` | `duration__noise_control` | control | treatment-duration coefficient: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8049 to 0.8363, SE ratio 0.6747 to 0.7152, empirical efficiency ratio 1.3933 to 1.4765, reported efficiency ratio 0.9940 to 0.9988 | pass |
| `interval_calibration` | `duration__shrunken_se_control` | control | treatment-duration coefficient: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7953 to 0.8274, SE ratio 0.6630 to 0.7013, empirical efficiency ratio 0.9945 to 1.0521, reported efficiency ratio 0.6958 to 0.6991 | pass |
| `power` | `duration__alternative` | positive | treatment-duration coefficient: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9975, 0.9885 to 0.9999 | pass |
| `projection_necessity` | `duration__declared_weights` | positive | treatment-duration coefficient: the working model uses its declared nonuniform projection weights | bias interval inside the equivalence margin | bias -0.0051 to 0.0039, margin 0.0137 | pass |
| `projection_necessity` | `duration__uniform_weights` | control | treatment-duration coefficient: the identical working model is projected under uniform weights | bias interval must fall entirely outside the margin | bias -0.1054 to -0.1006, margin 0.0073 | pass |
| `root_n_and_efficiency` | `(intercept)__n_2000` | positive | intercept coefficient: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0019, coverage 0.9047 to 0.9549, SE ratio 0.9893 | pass |
| `root_n_and_efficiency` | `(intercept)__n_500` | control | intercept coefficient: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0032, coverage 0.8769 to 0.9345, SE ratio 0.9130 | pass |
| `root_n_and_efficiency` | `(intercept)__n_8000` | positive | intercept coefficient: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0011, coverage 0.9164 to 0.9631, SE ratio 0.9871 | pass |
| `root_n_and_efficiency` | `duration__n_2000` | positive | treatment-duration coefficient: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0028, coverage 0.9147 to 0.9619, SE ratio 0.9725 | pass |
| `root_n_and_efficiency` | `duration__n_500` | control | treatment-duration coefficient: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0033, coverage 0.8625 to 0.9234, SE ratio 0.8638 | pass |
| `root_n_and_efficiency` | `duration__n_8000` | positive | treatment-duration coefficient: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000607, coverage 0.9265 to 0.9699, SE ratio 0.9920 | pass |
| `root_n_rate` | `(intercept)__empirical_sd` | positive | intercept coefficient: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5509 to -0.4817 | pass |
| `root_n_rate` | `(intercept)__reported_se` | positive | intercept coefficient: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4902 to -0.4844 | pass |
| `root_n_rate` | `duration__empirical_sd` | positive | treatment-duration coefficient: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5600 to -0.4904 | pass |
| `root_n_rate` | `duration__reported_se` | positive | treatment-duration coefficient: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4801 to -0.4710 | pass |
| `targeting_necessity` | `duration__targeted` | positive | treatment-duration coefficient: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0049 to 0.0046, margin 0.0145 | pass |
| `targeting_necessity` | `duration__untargeted` | control | treatment-duration coefficient: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0500 to -0.0403, margin 0.0148 | pass |
| `type_i_error` | `duration__sharp_null` | positive | treatment-duration coefficient: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0550, 0.0363 to 0.0792 | pass |
<!-- /generated -->

The property study uses the exact censored binary support law. Its regimen truths and influence
curves are derived independently before the fixed projection is applied. The targeting control is
a separately implemented backward recursion with every fluctuation removed.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2500 | observations per paired replication |
| `independent_tests_passed` | 4 | truth tests passing |
| `independent_tests_total` | 4 | truth tests reported |
| `paired_tests_passed` | 2 | paired comparisons passing |
| `paired_tests_total` | 2 | paired comparisons reported |
| `property_cells_passed` | 30 | property cells passing |
| `property_cells_total` | 30 | property cells reported |
| `max_standardized_bias` | 0.0376 | largest primary standardized bias |
| `min_coverage` | 0.9337 | lowest primary coverage |
| `max_margin_utilization` | 0.000370 | largest paired similarity-margin share |
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
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:projection_displacement` | 0.2500 | minimum projection displacement |

## Limits

- The evidence covers four plans and two time points on one censored binary law.
- Projection weights, plan durations, and the identity link are fixed before fitting.
- The row covers ordinary `n_folds=1` longitudinal MSM targeting only.
- Cross-fitted longitudinal coefficient inference remains refused.
- Intervals are pointwise Wald intervals rather than simultaneous regions.
- The mechanisms are supplied exactly in the primary comparison.
- Survival, competing risks, observation weights, and clustering are outside this row.

