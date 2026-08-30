# Ordinary controlled direct-effect TMLE

This study validates ordinary, non-cross-fitted controlled direct-effect TMLE for a binary
intermediate and outcomes missing at random. It compares `cleverly` with R
[`tmle`](https://cran.r-project.org/package=tmle) 2.1.1 on identical samples, with exact outcome,
treatment, intermediate, and observation nuisance predictions supplied to both implementations.
Each replicate draws one sample from a single observed stochastic law. Both intervention levels
use that sample. The controlled effect is negative at level zero and positive at level one.

The comparison reports the two treatment-specific means, ATE, risk ratio, and odds ratio at each
fixed intermediate level, with pointwise 95% Wald intervals. Ratios are assessed on their log
inference scales.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ate` | average treatment effect | `cleverly` controlled direct-effect TMLE | -0.0034 to 0.0014 | 0.9350 | 0.9726 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ate` | average treatment effect | R `tmle` controlled direct-effect path | -0.0034 to 0.0014 | 0.9350 | 0.9726 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ey0` | counterfactual mean under no treatment | `cleverly` controlled direct-effect TMLE | -0.000657 to 0.0032 | 0.9216 | 0.9694 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ey0` | counterfactual mean under no treatment | R `tmle` controlled direct-effect path | -0.000657 to 0.0032 | 0.9216 | 0.9694 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ey1` | counterfactual mean under treatment | `cleverly` controlled direct-effect TMLE | -0.0011 to 0.0017 | 0.9519 | 1.0125 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ey1` | counterfactual mean under treatment | R `tmle` controlled direct-effect path | -0.0011 to 0.0017 | 0.9519 | 1.0125 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `or` | marginal odds ratio, reported on the log scale | `cleverly` controlled direct-effect TMLE | -0.0305 to -0.0061 | 0.9331 | 0.9629 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `or` | marginal odds ratio, reported on the log scale | R `tmle` controlled direct-effect path | -0.0305 to -0.0061 | 0.9331 | 0.9629 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `rr` | marginal risk ratio, reported on the log scale | `cleverly` controlled direct-effect TMLE | -0.0046 to 0.0027 | 0.9428 | 0.9802 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `rr` | marginal risk ratio, reported on the log scale | R `tmle` controlled direct-effect path | -0.0046 to 0.0027 | 0.9428 | 0.9802 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ate` | average treatment effect | `cleverly` controlled direct-effect TMLE | -0.0022 to 0.0021 | 0.9397 | 0.9795 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ate` | average treatment effect | R `tmle` controlled direct-effect path | -0.0022 to 0.0021 | 0.9397 | 0.9795 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ey0` | counterfactual mean under no treatment | `cleverly` controlled direct-effect TMLE | -0.0014 to 0.0012 | 0.9419 | 0.9906 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ey0` | counterfactual mean under no treatment | R `tmle` controlled direct-effect path | -0.0014 to 0.0012 | 0.9419 | 0.9906 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ey1` | counterfactual mean under treatment | `cleverly` controlled direct-effect TMLE | -0.0018 to 0.0015 | 0.9272 | 0.9779 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ey1` | counterfactual mean under treatment | R `tmle` controlled direct-effect path | -0.0018 to 0.0015 | 0.9272 | 0.9779 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `or` | marginal odds ratio, reported on the log scale | `cleverly` controlled direct-effect TMLE | -0.0032 to 0.0155 | 0.9409 | 0.9797 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `or` | marginal odds ratio, reported on the log scale | R `tmle` controlled direct-effect path | -0.0032 to 0.0155 | 0.9409 | 0.9797 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `rr` | marginal risk ratio, reported on the log scale | `cleverly` controlled direct-effect TMLE | -0.0025 to 0.0074 | 0.9459 | 0.9818 | pass |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `rr` | marginal risk ratio, reported on the log scale | R `tmle` controlled direct-effect path | -0.0025 to 0.0074 | 0.9459 | 0.9818 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ate` | average treatment effect | -3.126e-10 | 3.927e-08 | 1.0000 | 0 | 2.202e-09 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ey0` | counterfactual mean under no treatment | 3.204e-10 | 4.988e-08 | 1.0000 | 0 | 3.398e-09 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `ey1` | counterfactual mean under treatment | 7.840e-12 | 1.735e-09 | 1.0000 | 0 | 7.079e-11 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `or` | marginal odds ratio, reported on the log scale | -6.157e-10 | 3.543e-08 | 1.0000 | 0 | 3.092e-09 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to zero | `rr` | marginal risk ratio, reported on the log scale | -2.613e-10 | 2.896e-08 | 1.0000 | 0 | 1.672e-09 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ate` | average treatment effect | 5.451e-13 | 7.861e-11 | 1.0000 | 0 | 5.265e-10 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ey0` | counterfactual mean under no treatment | -9.426e-12 | 2.230e-09 | 1.0000 | 0 | 7.966e-10 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `ey1` | counterfactual mean under treatment | -8.881e-12 | 1.625e-09 | 1.0000 | 0 | 3.178e-10 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `or` | marginal odds ratio, reported on the log scale | 8.124e-11 | 6.819e-10 | 1.0000 | 0 | 6.197e-10 vs 0.0500 | equivalent |
| binary-outcome MAR observed law, intervention sets the intermediate to one | `rr` | marginal risk ratio, reported on the log scale | 6.593e-11 | 1.992e-09 | 1.0000 | 0 | 9.517e-10 vs 0.0500 | equivalent |
<!-- /generated -->

R `tmle` 2.1.1 constructs the observed outcome offset for its native second controlled-effect
result from `Q`, although its level-one counterfactual predictions come from `Q.Z1`. Supplying
exact, level-specific regressions therefore makes that native second result internally
inconsistent.

The registered adapter recodes each requested intermediate level to level zero,
transforms `Z` and `pZ1` together, and selects the first result. The
[`native-result2-defect.csv`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_cde/native-result2-defect.csv)
artifact freezes the upstream failure. The standard regeneration command recreates it from the
generated sample and truth. The manifest hashes both the probe and the artifact. The paired table
above validates the recoded result.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `cde_robustness` | `z0__all_correct` | positive | controlled direct effect at intermediate level zero: the outcome regression and all three mechanisms are correct | bias interval inside the equivalence margin | bias -0.0073 to 0.000619, margin 0.0133 | pass |
| `cde_robustness` | `z0__intermediate_wrong` | control | controlled direct effect at intermediate level zero: only the intermediate mechanism is wrong beside a wrong outcome regression | bias interval must fall entirely outside the margin | bias 0.1628 to 0.1717, margin 0.0149 | pass |
| `cde_robustness` | `z0__mechanisms_correct` | positive | controlled direct effect at intermediate level zero: all three mechanisms are correct and the outcome regression is wrong | bias interval inside the equivalence margin | bias -0.0071 to 0.0014, margin 0.0143 | pass |
| `cde_robustness` | `z0__observation_wrong` | control | controlled direct effect at intermediate level zero: only the observation mechanism is wrong beside a wrong outcome regression | bias interval must fall entirely outside the margin | bias -0.1860 to -0.1784, margin 0.0127 | pass |
| `cde_robustness` | `z0__outcome_correct` | positive | controlled direct effect at intermediate level zero: the outcome regression is correct and all three mechanisms are wrong | bias interval inside the equivalence margin | bias -0.0038 to 0.0011, margin 0.0082 | pass |
| `cde_robustness` | `z0__treatment_wrong` | control | controlled direct effect at intermediate level zero: only the treatment mechanism is wrong beside a wrong outcome regression | bias interval must fall entirely outside the margin | bias 0.2668 to 0.2753, margin 0.0142 | pass |
| `cde_robustness` | `z1__all_correct` | positive | controlled direct effect at intermediate level one: the outcome regression and all three mechanisms are correct | bias interval inside the equivalence margin | bias -0.0031 to 0.0039, margin 0.0117 | pass |
| `cde_robustness` | `z1__intermediate_wrong` | control | controlled direct effect at intermediate level one: only the intermediate mechanism is wrong beside a wrong outcome regression | bias interval must fall entirely outside the margin | bias -0.1665 to -0.1581, margin 0.0140 | pass |
| `cde_robustness` | `z1__mechanisms_correct` | positive | controlled direct effect at intermediate level one: all three mechanisms are correct and the outcome regression is wrong | bias interval inside the equivalence margin | bias 0.000193 to 0.0097, margin 0.0159 | pass |
| `cde_robustness` | `z1__observation_wrong` | control | controlled direct effect at intermediate level one: only the observation mechanism is wrong beside a wrong outcome regression | bias interval must fall entirely outside the margin | bias -0.1797 to -0.1727, margin 0.0118 | pass |
| `cde_robustness` | `z1__outcome_correct` | positive | controlled direct effect at intermediate level one: the outcome regression is correct and all three mechanisms are wrong | bias interval inside the equivalence margin | bias -0.0025 to 0.0037, margin 0.0105 | pass |
| `cde_robustness` | `z1__treatment_wrong` | control | controlled direct effect at intermediate level one: only the treatment mechanism is wrong beside a wrong outcome regression | bias interval must fall entirely outside the margin | bias 0.2137 to 0.2195, margin 0.0099 | pass |
| `interval_calibration` | `z0__correctly_specified` | positive | controlled direct effect at intermediate level zero: all required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9363 to 0.9474, SE ratio 0.9726 to 1.0056, empirical efficiency ratio 0.9841 to 1.0167, reported efficiency ratio 0.9873 to 0.9910 | pass |
| `interval_calibration` | `z0__noise_control` | control | controlled direct effect at intermediate level zero: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8249 to 0.8425, SE ratio 0.6937 to 0.7174, empirical efficiency ratio 1.3788 to 1.4259, reported efficiency ratio 0.9873 to 0.9911 | pass |
| `interval_calibration` | `z0__shrunken_se_control` | control | controlled direct effect at intermediate level zero: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8099 to 0.8281, SE ratio 0.6810 to 0.7039, empirical efficiency ratio 0.9839 to 1.0167, reported efficiency ratio 0.6911 to 0.6937 | pass |
| `interval_calibration` | `z1__correctly_specified` | positive | controlled direct effect at intermediate level one: all required nuisance functions are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9352 to 0.9463, SE ratio 0.9789 to 1.0133, empirical efficiency ratio 0.9767 to 1.0102, reported efficiency ratio 0.9876 to 0.9910 | pass |
| `interval_calibration` | `z1__noise_control` | control | controlled direct effect at intermediate level one: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8212 to 0.8389, SE ratio 0.6913 to 0.7147, empirical efficiency ratio 1.3845 to 1.4311, reported efficiency ratio 0.9875 to 0.9910 | pass |
| `interval_calibration` | `z1__shrunken_se_control` | control | controlled direct effect at intermediate level one: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8174 to 0.8353, SE ratio 0.6856 to 0.7098, empirical efficiency ratio 0.9759 to 1.0099, reported efficiency ratio 0.6913 to 0.6937 | pass |
| `power` | `z0__alternative` | positive | controlled direct effect at intermediate level zero: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9988, 0.9907 to 1.0000 | pass |
| `power` | `z1__alternative` | positive | controlled direct effect at intermediate level one: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `z0__n_2000` | positive | controlled direct effect at intermediate level zero: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000936, coverage 0.9160 to 0.9431, SE ratio 0.9664 | pass |
| `root_n_and_efficiency` | `z0__n_500` | control | controlled direct effect at intermediate level zero: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0034, coverage 0.8921 to 0.9229, SE ratio 0.9441 | pass |
| `root_n_and_efficiency` | `z0__n_8000` | positive | controlled direct effect at intermediate level zero: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000563, coverage 0.9402 to 0.9630, SE ratio 1.0157 | pass |
| `root_n_and_efficiency` | `z1__n_2000` | positive | controlled direct effect at intermediate level one: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000763, coverage 0.9196 to 0.9462, SE ratio 0.9613 | pass |
| `root_n_and_efficiency` | `z1__n_500` | control | controlled direct effect at intermediate level one: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0027, coverage 0.8868 to 0.9183, SE ratio 0.9120 | pass |
| `root_n_and_efficiency` | `z1__n_8000` | positive | controlled direct effect at intermediate level one: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000502, coverage 0.9462 to 0.9678, SE ratio 1.0324 | pass |
| `root_n_rate` | `z0__empirical_sd` | positive | controlled direct effect at intermediate level zero: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5302 to -0.4933 | pass |
| `root_n_rate` | `z0__reported_se` | positive | controlled direct effect at intermediate level zero: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4881 to -0.4817 | pass |
| `root_n_rate` | `z1__empirical_sd` | positive | controlled direct effect at intermediate level one: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5476 to -0.5104 | pass |
| `root_n_rate` | `z1__reported_se` | positive | controlled direct effect at intermediate level one: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4875 to -0.4815 | pass |
| `targeting_necessity` | `z0__targeted` | positive | controlled direct effect at intermediate level zero: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0051 to 0.0030, margin 0.0137 | pass |
| `targeting_necessity` | `z0__untargeted` | control | controlled direct effect at intermediate level zero: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0828 to -0.0819, margin 0.0016 | pass |
| `targeting_necessity` | `z1__targeted` | positive | controlled direct effect at intermediate level one: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias 0.000291 to 0.0097, margin 0.0158 | pass |
| `targeting_necessity` | `z1__untargeted` | control | controlled direct effect at intermediate level one: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0530 to -0.0518, margin 0.0020 | pass |
| `type_i_error` | `z0__sharp_null` | positive | controlled direct effect at intermediate level zero: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0537, 0.0353 to 0.0777 | pass |
| `type_i_error` | `z1__sharp_null` | positive | controlled direct effect at intermediate level one: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0550, 0.0363 to 0.0792 | pass |
<!-- /generated -->

The robustness grid checks the actual four-nuisance contract at both intermediate levels. A
correct outcome regression must rescue wrong treatment, intermediate, and observation mechanisms;
with a wrong outcome regression, all three mechanisms must be correct. Three separate controls
make exactly one mechanism wrong. Further cells test root-n contraction, exact-law efficiency,
interval calibration and its two controls, type-I error and power, and whether targeting materially
changes the plug-in.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 3200 | paired replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 20 | truth tests passing |
| `independent_tests_total` | 20 | truth tests reported |
| `paired_tests_passed` | 10 | paired comparisons passing |
| `paired_tests_total` | 10 | paired comparisons reported |
| `property_cells_passed` | 36 | property cells passing |
| `property_cells_total` | 36 | property cells reported |
| `max_standardized_bias` | 0.0685 | largest primary standardized bias |
| `min_coverage` | 0.9216 | lowest primary coverage |
| `max_margin_utilization` | 4.988e-08 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
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
| `bound:z0_standard_error` | 0.0524 | exact level-zero ATE efficiency SD |
| `bound:z1_standard_error` | 0.0457 | exact level-one ATE efficiency SD |

## Limits

- The study covers binary treatment, binary intermediate and outcome, one three-level baseline
  covariate, and an observational MAR response mechanism with no post-treatment confounder between
  treatment and the intermediate.
- Exact finite-support nuisances isolate targeting and inference. The study does not validate
  flexible learner wrappers or cross-fitting.
- The comparator requires level recoding because of the frozen native second-result defect in R
  `tmle` 2.1.1. The registered probe recreates the defect during standard regeneration. The study
  does not claim that the native result is valid with exact `Q.Z1` inputs.
- Inference is pointwise. The study does not cover simultaneous bands, active probability bounds,
  weights, clusters, multi-arm treatment, continuous intermediates, or MNAR outcomes.
- `ATT` and `ATC` retain exact-law Gateaux and remainder evidence but are absent here because R
  `tmle` does not report them on its controlled direct-effect path.
- This is not evidence for natural direct or indirect effects, nor for a general longitudinal
  intervention with treatment-induced intermediate confounding.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_cde/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_cde/manifest.json),
[native-result probe](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_cde/probe_native_result2.R),
[native-result artifact](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_cde/native-result2-defect.csv),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_cde/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_cde/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_cde/properties.csv)
carry the protocol, provenance, and every published row.
