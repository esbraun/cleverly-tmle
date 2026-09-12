# Ordinary missing-outcome natural-course TMLE

This study validates ordinary, non-cross-fitted TMLE for the observed outcome mean under MAR.
The target averages the outcome regression over the empirical joint distribution of treatment and
baseline covariates. Its score therefore uses the outcome and response nuisances, but no treatment
mechanism. Binary and bounded-continuous primary laws share the same conditional means and
observation process; the continuous outcome is beta distributed and uses fixed bounds `(0, 1)`.

No canonical implementation is compared. The repository's pinned R `tmle` 2.1.1 adapter covers
MAR intervention-arm means rather than this natural-course parameter. Díaz, Carone, and van der
Laan (2016), Section 2 and equations 1--5 give the estimator construction, but the source and
repository search found no maintained public implementation exposing a matching callable target.
Exact-law Gateaux and remainder checks elsewhere in the test suite and this independent
repeated-sampling study provide the validation evidence.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome observational natural-course law with MAR outcomes | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome natural-course TMLE | -0.000474 to 0.0029 | 0.9350 | 0.9468 | pass |
| bounded continuous-outcome observational natural-course law with MAR outcomes | `ey_obs` | observed outcome mean under the natural course | `cleverly` missing-outcome natural-course TMLE | -0.000306 to 0.000726 | 0.9375 | 0.9687 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

There is no canonical comparison for this construction. The committed `equivalence.csv` is empty
and schema-valid.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the outcome-response mechanism are correct | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0017 to 0.000937, margin 0.0044, SE ratio 0.9913 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0156 to -0.0135, margin 0.0034, SE ratio 1.5919 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000619 to 0.0012, margin 0.0031, SE ratio 1.5107 | pass |
| `double_robustness` | `response_correct` | positive | only the outcome-observation mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0010 to 0.0020, margin 0.0051, SE ratio 1.0152 | pass |
| `interval_calibration` | `ey_obs__correctly_specified` | positive | observed outcome mean under the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9440 to 0.9614, SE ratio 0.9745 to 1.0307, empirical efficiency ratio 0.9690 to 1.0250, reported efficiency ratio 0.9977 to 0.9996 | pass |
| `interval_calibration` | `ey_obs__noise_control` | control | observed outcome mean under the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8251 to 0.8551, SE ratio 0.6943 to 0.7341, empirical efficiency ratio 1.3604 to 1.4383, reported efficiency ratio 0.9977 to 0.9996 | pass |
| `interval_calibration` | `ey_obs__shrunken_se_control` | control | observed outcome mean under the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8113 to 0.8424, SE ratio 0.6820 to 0.7218, empirical efficiency ratio 0.9686 to 1.0248, reported efficiency ratio 0.6984 to 0.6997 | pass |
| `missingness_necessity` | `ey_obs__complete_case_control` | control | observed outcome mean under the natural course: the identical estimator silently discards unobserved outcomes and ignores selection | bias interval must fall entirely outside the margin | bias -0.0075 to -0.0052, margin 0.0040 | pass |
| `missingness_necessity` | `ey_obs__declared` | positive | observed outcome mean under the natural course: the observation indicator is declared, so correct mechanisms carry a wrong outcome model | bias interval inside the equivalence margin | bias -0.0013 to 0.0017, margin 0.0051 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000060, coverage 0.9356 to 0.9737, SE ratio 1.0358 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0012, coverage 0.9341 to 0.9727, SE ratio 1.0312 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000133, coverage 0.9121 to 0.9575, SE ratio 0.9658 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5092 to -0.4417 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5003 to -0.4973 | pass |
| `targeting_necessity` | `ey_obs__targeted` | positive | observed outcome mean under the natural course: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0019 to 0.0012, margin 0.0052 | pass |
| `targeting_necessity` | `ey_obs__untargeted` | control | observed outcome mean under the natural course: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0117 to 0.0123, margin 0.0011 | pass |
<!-- /generated -->

The property law tests the two-nuisance union model: either the outcome regression or the response
mechanism may be correct, and a both-wrong cell must discriminate the resulting bias. Separate
cells test three-size root-n behavior, exact-EIF efficiency, repeated-sampling interval calibration
and its two mutations, the targeting update, and the bias induced by silently analyzing complete
cases.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications per law |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 2 | truth tests passing |
| `independent_tests_total` | 2 | truth tests reported |
| `paired_tests_passed` | 0 | paired comparisons passing |
| `paired_tests_total` | 0 | paired comparisons reported |
| `property_cells_passed` | 16 | property cells passing |
| `property_cells_total` | 16 | property cells reported |
| `max_standardized_bias` | 0.0653 | largest primary standardized bias |
| `min_coverage` | 0.9350 | lowest primary coverage |
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
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:missingness_displacement` | 0.2500 | minimum complete-case displacement |
| `bound:ey_obs_standard_error` | 0.0173 | exact binary-law EIF standard error at primary n |

## Limits

- The primary study covers binary treatment, one three-level baseline covariate, binary and
  beta-distributed bounded-continuous outcomes, and one observational MAR response mechanism.
- The response probabilities and binary outcome regression are finite-law oracles. The continuous
  primary law uses a correctly specified affine outcome learner fit to the scaled observed
  outcomes. This does not validate flexible learner wrappers or cross-fitting.
- The exact efficiency comparison is for the binary property law; the continuous primary law is
  checked for truth recovery, coverage, and reported-SE calibration without claiming the same
  bound.
- There is no external parity comparison because the audited maintained comparator does not expose
  the same target.
- The study uses ordinary pointwise Wald intervals and excludes weights, clusters, missing
  treatment, multinomial treatment, MNAR outcomes, sensitivity analysis, and longitudinal data.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course/replicates.csv.gz),
[performance decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course/performance-tests.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course/properties.csv)
carry the protocol, provenance, and every published row.
