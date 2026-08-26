# Point-treatment MSM projection

This study validates the ordinary identity-link MSM projection in `cleverly`. The working model
has an intercept, treatment, and baseline term against six counterfactual cells, so it is
deliberately unsaturated. Its fixed weights are nonuniform. The primary comparison uses the
Gaussian projection in pinned R `tmle3` `Param_MSM`; an independent finite-support functional and
Gateaux derivative supply the property truths and efficiency bounds.

## What was compared

| setting | `cleverly` | R `tmle3` |
| --- | --- | --- |
| datasets | 800 bounded continuous-outcome samples | the identical rows |
| working model | identity-link intercept, treatment, and baseline terms | `A + V`, transformed to the same coefficient basis |
| projection measure | fixed weight `1 + 0.5 A + 5 W` | the same custom weight |
| nuisance fits | exact conditional means and treatment probabilities | correctly specified Gaussian and logistic GLMs |
| intervals | pointwise 95% Wald, joint coefficient influence curve | the same |

The pinned `tmle3` release compares a documented custom weight function with string sentinels
before checking that it is a function. The runner gives that function a narrow equality method so
the package reaches its documented custom-weight branch without patching the pinned source.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded continuous-outcome law with an unsaturated working model | `msm[(intercept)]` | point-treatment MSM projection intercept coefficient | `cleverly` | -0.000478 to 0.000609 | 0.9425 | 0.9731 | pass |
| bounded continuous-outcome law with an unsaturated working model | `msm[(intercept)]` | point-treatment MSM projection intercept coefficient | R `tmle3` | -0.000471 to 0.000617 | 0.9437 | 0.9736 | pass |
| bounded continuous-outcome law with an unsaturated working model | `msm[W]` | point-treatment MSM projection baseline-covariate coefficient | `cleverly` | -0.000314 to 0.000374 | 0.9425 | 0.9710 | pass |
| bounded continuous-outcome law with an unsaturated working model | `msm[W]` | point-treatment MSM projection baseline-covariate coefficient | R `tmle3` | -0.000318 to 0.000371 | 0.9450 | 0.9708 | pass |
| bounded continuous-outcome law with an unsaturated working model | `msm[a]` | point-treatment MSM projection treatment coefficient | `cleverly` | -0.000558 to 0.000492 | 0.9463 | 0.9969 | pass |
| bounded continuous-outcome law with an unsaturated working model | `msm[a]` | point-treatment MSM projection treatment coefficient | R `tmle3` | -0.000565 to 0.000485 | 0.9513 | 0.9977 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded continuous-outcome law with an unsaturated working model | `msm[(intercept)]` | point-treatment MSM projection intercept coefficient | -0.000007 | 0.0081 | 1.0020 | -0.0012 | 0.0036 vs 0.0500 | equivalent |
| bounded continuous-outcome law with an unsaturated working model | `msm[W]` | point-treatment MSM projection baseline-covariate coefficient | 0.000004 | 0.0070 | 1.0012 | -0.0025 | 0.0040 vs 0.0500 | equivalent |
| bounded continuous-outcome law with an unsaturated working model | `msm[a]` | point-treatment MSM projection treatment coefficient | 0.000007 | 0.0082 | 1.0013 | -0.0050 | 0.0026 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `(intercept)__both_correct` | positive | intercept coefficient: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0031 to 0.000952, margin 0.0069 | pass |
| `double_robustness` | `(intercept)__both_wrong` | control | intercept coefficient: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0845 to 0.0882, margin 0.0062 | pass |
| `double_robustness` | `(intercept)__mechanism_correct` | positive | intercept coefficient: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0033 to 0.0012, margin 0.0077 | pass |
| `double_robustness` | `(intercept)__outcome_correct` | positive | intercept coefficient: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0012 to 0.0024, margin 0.0059 | pass |
| `double_robustness` | `W__both_correct` | positive | baseline-covariate coefficient: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0011 to 0.0016, margin 0.0046 | pass |
| `double_robustness` | `W__both_wrong` | control | baseline-covariate coefficient: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0214 to 0.0237, margin 0.0038 | pass |
| `double_robustness` | `W__mechanism_correct` | positive | baseline-covariate coefficient: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0014 to 0.0016, margin 0.0051 | pass |
| `double_robustness` | `W__outcome_correct` | positive | baseline-covariate coefficient: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0017 to 0.000645, margin 0.0039 | pass |
| `double_robustness` | `a__both_correct` | positive | treatment coefficient: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0016 to 0.0026, margin 0.0069 | pass |
| `double_robustness` | `a__both_wrong` | control | treatment coefficient: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.1899 to -0.1860, margin 0.0065 | pass |
| `double_robustness` | `a__mechanism_correct` | positive | treatment coefficient: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.000537 to 0.0038, margin 0.0072 | pass |
| `double_robustness` | `a__outcome_correct` | positive | treatment coefficient: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0017 to 0.0019, margin 0.0061 | pass |
| `interval_calibration` | `(intercept)__correctly_specified` | positive | intercept coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9321 to 0.9614, SE ratio 0.9405 to 1.0296, empirical efficiency ratio 0.9676 to 1.0588, reported efficiency ratio 0.9933 to 0.9990 | pass |
| `interval_calibration` | `(intercept)__noise_control` | control | intercept coefficient: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7960 to 0.8458, SE ratio 0.6590 to 0.7225, empirical efficiency ratio 1.3787 to 1.5111, reported efficiency ratio 0.9931 to 0.9988 | pass |
| `interval_calibration` | `(intercept)__shrunken_se_control` | control | intercept coefficient: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7973 to 0.8470, SE ratio 0.6579 to 0.7200, empirical efficiency ratio 0.9687 to 1.0595, reported efficiency ratio 0.6952 to 0.6992 | pass |
| `interval_calibration` | `W__correctly_specified` | positive | baseline-covariate coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9357 to 0.9641, SE ratio 0.9630 to 1.0556, empirical efficiency ratio 0.9432 to 1.0330, reported efficiency ratio 0.9918 to 0.9980 | pass |
| `interval_calibration` | `W__noise_control` | control | baseline-covariate coefficient: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8006 to 0.8500, SE ratio 0.6672 to 0.7301, empirical efficiency ratio 1.3634 to 1.4913, reported efficiency ratio 0.9918 to 0.9981 | pass |
| `interval_calibration` | `W__shrunken_se_control` | control | baseline-covariate coefficient: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7979 to 0.8476, SE ratio 0.6746 to 0.7380, empirical efficiency ratio 0.9444 to 1.0326, reported efficiency ratio 0.6943 to 0.6986 | pass |
| `interval_calibration` | `a__correctly_specified` | positive | treatment coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9357 to 0.9641, SE ratio 0.9537 to 1.0430, empirical efficiency ratio 0.9556 to 1.0439, reported efficiency ratio 0.9930 to 0.9985 | pass |
| `interval_calibration` | `a__noise_control` | control | treatment coefficient: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8137 to 0.8617, SE ratio 0.6820 to 0.7471, empirical efficiency ratio 1.3335 to 1.4605, reported efficiency ratio 0.9931 to 0.9985 | pass |
| `interval_calibration` | `a__shrunken_se_control` | control | treatment coefficient: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7966 to 0.8464, SE ratio 0.6667 to 0.7308, empirical efficiency ratio 0.9539 to 1.0453, reported efficiency ratio 0.6952 to 0.6989 | pass |
| `power` | `a__alternative` | positive | treatment coefficient: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `projection_necessity` | `W__declared_weights` | positive | baseline-covariate coefficient: the working model uses its declared nonuniform projection weights | bias interval inside the equivalence margin | bias -0.000906 to 0.0018, margin 0.0045 | pass |
| `projection_necessity` | `W__uniform_weights` | control | baseline-covariate coefficient: the identical working model is projected under uniform weights | bias interval must fall entirely outside the margin | bias -0.1150 to -0.1129, margin 0.0035 | pass |
| `root_n_and_efficiency` | `(intercept)__n_2000` | positive | intercept coefficient: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000144, coverage 0.9163 to 0.9534, SE ratio 0.9836 | pass |
| `root_n_and_efficiency` | `(intercept)__n_500` | control | intercept coefficient: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0022, coverage 0.9145 to 0.9520, SE ratio 0.9701 | pass |
| `root_n_and_efficiency` | `(intercept)__n_8000` | positive | intercept coefficient: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000836, coverage 0.9277 to 0.9620, SE ratio 0.9670 | pass |
| `root_n_and_efficiency` | `W__n_2000` | positive | baseline-covariate coefficient: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000309, coverage 0.9480 to 0.9765, SE ratio 1.0356 | pass |
| `root_n_and_efficiency` | `W__n_500` | control | baseline-covariate coefficient: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0017, coverage 0.9061 to 0.9455, SE ratio 0.9475 | pass |
| `root_n_and_efficiency` | `W__n_8000` | positive | baseline-covariate coefficient: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000550, coverage 0.9229 to 0.9584, SE ratio 0.9570 | pass |
| `root_n_and_efficiency` | `a__n_2000` | positive | treatment coefficient: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000476, coverage 0.9154 to 0.9527, SE ratio 0.9544 | pass |
| `root_n_and_efficiency` | `a__n_500` | control | treatment coefficient: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0028, coverage 0.9163 to 0.9534, SE ratio 0.9820 | pass |
| `root_n_and_efficiency` | `a__n_8000` | positive | treatment coefficient: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000500, coverage 0.9258 to 0.9606, SE ratio 0.9770 | pass |
| `root_n_rate` | `(intercept)__empirical_sd` | positive | intercept coefficient: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5233 to -0.4694 | pass |
| `root_n_rate` | `(intercept)__reported_se` | positive | intercept coefficient: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4993 to -0.4943 | pass |
| `root_n_rate` | `W__empirical_sd` | positive | baseline-covariate coefficient: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5271 to -0.4732 | pass |
| `root_n_rate` | `W__reported_se` | positive | baseline-covariate coefficient: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4999 to -0.4945 | pass |
| `root_n_rate` | `a__empirical_sd` | positive | treatment coefficient: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5236 to -0.4672 | pass |
| `root_n_rate` | `a__reported_se` | positive | treatment coefficient: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4993 to -0.4945 | pass |
| `targeting_necessity` | `a__targeted` | positive | treatment coefficient: the estimator fluctuates a constant outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0013 to 0.0030, margin 0.0071 | pass |
| `targeting_necessity` | `a__untargeted` | control | treatment coefficient: the identical backward recursion with no fluctuation at any node | bias interval must fall entirely outside the margin | bias -0.5208 to -0.5206, margin 0.000401 | pass |
| `type_i_error` | `a__sharp_null` | positive | treatment coefficient: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0413, 0.0253 to 0.0629 | pass |
<!-- /generated -->

The property study uses an exact binary finite-support law. It recomputes the nonuniform
projection from the law and differentiates that functional by complex step. The uniform-weight
control moves the baseline coefficient, and the untargeted control leaves a misspecified outcome
regression unadjusted.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `paired_tests_total` | 3 | paired comparisons reported |
| `property_cells_passed` | 42 | property cells passing |
| `property_cells_total` | 42 | property cells reported |
| `max_standardized_bias` | 0.0122 | largest primary standardized bias |
| `min_coverage` | 0.9425 | lowest primary coverage |
| `max_margin_utilization` | 0.0082 | largest paired similarity-margin share |
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
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:projection_displacement` | 0.2500 | minimum projection displacement |

## Limits

- The primary law is bounded continuous, and the property law is binary finite support.
- The design and projection weights are fixed rather than estimated.
- The study covers the identity link and ordinary, non-cross-fitted targeting.
- It reports pointwise Wald intervals, not simultaneous coefficient regions.
- It validates three terms and two treatment arms.
- It does not validate continuous-dose integration or data-adaptive projection measures.

