# Ordinary end-of-study longitudinal TMLE

This study validates `cleverly`'s ordinary, non-cross-fitted two-time-point regimen mean. The law
has monotone censoring and includes static and dynamic plans. The canonical comparison uses R
[`ltmle`](https://www.jstatsoft.org/article/view/v081i01) 1.3-0. The parameter, longitudinal
double robustness, and efficient influence curve follow Bang and Robins (2005), van der Laan and
Gruber (2012), and Petersen et al. (2014).

Agreement with R is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_longitudinal.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_longitudinal.py).

## What was compared

| setting | `cleverly` | R `ltmle` |
| --- | --- | --- |
| datasets | 1,600 censored samples generated in Python | the identical rows |
| plans | never treat, always treat, and "treat, then continue if L2 is positive" | the same three, invoked once per regimen |
| contrasts | always-minus-never and dynamic-minus-never | the same, from the difference of the two rowwise influence curves so covariance is preserved |
| mechanisms | the generating treatment and censoring probabilities | the same |
| sequential regressions | follower-stratified quasibinomial | the same |
| cumulative-g bounds | nonbinding | nonbinding |
| intervals | pointwise 95% identity-scale Wald, influence-curve variance | the same, `variance.method="ic"` |

The paired discrepancies are numerical-solver scale rather than statistical scale. Each
implementation is also tested against quadrature truth on its own, and both verdicts are carried
beside the paired one. A regeneration fails if either is false.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` | -0.0025 to 0.0015 | 0.9413 | 0.9890 | pass |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `ltmle` | -0.0025 to 0.0015 | 0.9413 | 0.9890 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` | -0.0025 to 0.0016 | 0.9387 | 0.9943 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `ltmle` | -0.0025 to 0.0016 | 0.9387 | 0.9943 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` | -0.0015 to 0.000862 | 0.9506 | 1.0130 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `ltmle` | -0.0015 to 0.000862 | 0.9506 | 1.0130 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` | -0.0015 to 0.0018 | 0.9450 | 0.9710 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `ltmle` | -0.0015 to 0.0018 | 0.9450 | 0.9710 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` | -0.0016 to 0.0010 | 0.9469 | 0.9854 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `ltmle` | -0.0016 to 0.0010 | 0.9469 | 0.9854 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | 2.101e-10 | 4.449e-08 | 1.0000 | 0 | 5.021e-10 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 6.175e-12 | 1.283e-09 | 1.0000 | 0 | 8.709e-09 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | 8.850e-11 | 3.180e-08 | 1.0000 | 0 | 9.006e-09 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | -1.216e-10 | 3.133e-08 | 1.0000 | 0 | 4.273e-10 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | -1.155e-10 | 3.766e-08 | 1.0000 | 0 | 1.753e-08 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0019 to 0.0017, margin 0.0062 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0146 to 0.0188, margin 0.0070 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0016 to 0.0022, margin 0.0064 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0020 to 0.0017, margin 0.0062 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0055 to 0.0030, margin 0.0142 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0288 to -0.0201, margin 0.0146 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0036 to 0.0053, margin 0.0149 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0040 to 0.0049, margin 0.0151 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9282 to 0.9533, SE ratio 0.9552 to 1.0280, empirical efficiency ratio 0.9729 to 1.0461, reported efficiency ratio 0.9967 to 1.0025 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8133 to 0.8529, SE ratio 0.6883 to 0.7404, empirical efficiency ratio 1.3502 to 1.4513, reported efficiency ratio 0.9967 to 1.0025 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8068 to 0.8469, SE ratio 0.6689 to 0.7186, empirical efficiency ratio 0.9737 to 1.0457, reported efficiency ratio 0.6976 to 0.7018 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9397 to 0.9626, SE ratio 0.9671 to 1.0399, empirical efficiency ratio 0.9591 to 1.0307, reported efficiency ratio 0.9942 to 1.0001 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8168 to 0.8560, SE ratio 0.6838 to 0.7375, empirical efficiency ratio 1.3532 to 1.4573, reported efficiency ratio 0.9943 to 1.0000 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8068 to 0.8469, SE ratio 0.6779 to 0.7281, empirical efficiency ratio 0.9590 to 1.0298, reported efficiency ratio 0.6960 to 0.7000 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9700, 0.9508 to 0.9833 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000406, coverage 0.9282 to 0.9688, SE ratio 0.9811 | pass |
| `root_n_and_efficiency` | `dynamic__n_500` | control | dynamic plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0020, coverage 0.8822 to 0.9353, SE ratio 0.8963 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000337, coverage 0.9326 to 0.9717, SE ratio 1.0063 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000474, coverage 0.9020 to 0.9502, SE ratio 0.9707 | pass |
| `root_n_and_efficiency` | `static__n_500` | control | static plan: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0011, coverage 0.8682 to 0.9244, SE ratio 0.8792 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000085, coverage 0.9208 to 0.9637, SE ratio 0.9884 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5581 to -0.4904 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4857 to -0.4776 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5482 to -0.4841 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4786 to -0.4694 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0014 to 0.0023, margin 0.0063 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0239 to 0.0282, margin 0.0073 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0018 to 0.0068, margin 0.0143 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0242 to -0.0156, margin 0.0144 | pass |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0700, 0.0488 to 0.0965 | pass |
<!-- /generated -->

The property study samples the exact binary support law rather than the continuous comparison law.
Its longhand functional supplies exact static and dynamic truths, and its Gateaux derivative
supplies the efficiency bound without reading either estimator.

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
| `property_cells_total` | 30 | independent property cells |
| `property_cells_passed` | 30 | property cells passing |
| `max_standardized_bias` | 0.0180 | largest primary standardized bias |
| `min_coverage` | 0.9387 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9216 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9290 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0641 | highest primary SE-ratio endpoint |
| `max_margin_utilization` | 4.449e-08 | largest share of paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0000 | largest paired RMSE-ratio bound |
| `min_coverage_difference_lower` | 0 | smallest paired coverage-difference bound |
| `max_calibration_excess_upper` | 1.969e-08 | largest paired calibration-excess bound |
| `properties[double_robustness/static__both_wrong]:standardized_bias` | -0.4195 | static both-wrong control |
| `properties[double_robustness/dynamic__both_wrong]:standardized_bias` | 0.5944 | dynamic both-wrong control |
| `properties[root_n_and_efficiency/static__n_500]:coverage` | 0.8988 | static small-sample control coverage |
| `properties[root_n_and_efficiency/dynamic__n_500]:coverage` | 0.9113 | dynamic small-sample control coverage |
| `properties[interval_calibration/static__correctly_specified]:efficiency_empirical_ratio` | 0.9948 | static empirical spread over exact EIF bound |
| `properties[interval_calibration/dynamic__correctly_specified]:efficiency_empirical_ratio` | 1.0093 | dynamic empirical spread over exact EIF bound |
| `properties[type_i_error/static__sharp_null]:rejection_rate` | 0.0700 | confounded-null rejection rate |
| `properties[type_i_error/static__sharp_null]:rejection_ci_upper` | 0.0965 | one-sided bound the type-I ceiling is checked against |
| `properties[power/static__alternative]:rejection_rate` | 0.9700 | alternative rejection rate |
| `properties[root_n_and_efficiency/static__n_2000]:coverage_ci_lower` | 0.9020 | tightest primary property coverage endpoint |
| `properties[targeting_necessity/static__targeted]:standardized_bias` | 0.0435 | static contrast as the estimator computes it |
| `properties[targeting_necessity/static__untargeted]:standardized_bias` | -0.3447 | the same recursion with no fluctuation |
| `properties[targeting_necessity/dynamic__untargeted]:standardized_bias` | 0.8970 | dynamic contrast with no fluctuation |
| `properties[targeting_necessity/static__targeted]:targeting_displacement` | 0.3909 | least-displaced contrast, in targeted standard deviations |
| `max_targeting_displacement` | 0.0938 | largest final-fluctuation move, in standard errors |
| `median_targeting_displacement` | 0.0117 | median final-fluctuation move, in standard errors |
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
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The static paired comparisons do not witness targeting | A plug-in built from the same two regressions with no fluctuation at either node still clears both acceptance gates on `ey_regimen[never]`, `ey_regimen[always]`, and `ate_regimen[always vs never]`. Only the two estimands carrying the dynamic rule separate it. The targeting evidence in this row is the dynamic plan and the `targeting_necessity` cells, not the count of paired comparisons. `tests/e2e/test_ltmle_targeting_slow.py` asserts exactly which estimands survive, so the limitation is measured on every run |
| `initial_estimate` measures the final fluctuation only | The earlier node's regression is of the *already targeted* later node. R's `fit$Q[[1]]` regresses the updated `Q.kplus1` and `cleverly`'s first step does the same. So `max_targeting_displacement` and `median_targeting_displacement` measure the last fluctuation rather than the whole targeting step |
| The reference's truth column is not independent of this codebase | The quadrature truth the R rows are scored against comes from `cleverly.datasets.longitudinal`. What makes it usable is that the quadrature is checked separately, against Monte Carlo and by node refinement, in `tests/unit/test_datasets_longitudinal.py` |
| The test over-rejects mildly under the harder null | The rate is 0.0700 at n=4,000 on 800 replications, about 2.6 Monte Carlo standard errors above nominal, and coverage in the same cell is 0.9300. The one-sided bound of 0.0965 clears the predeclared ceiling of 0.1000, so the cell passes, but with roughly three rejections to spare and a point estimate genuinely above 0.05. The influence-curve standard error is slightly optimistic here, and this is published rather than absorbed |
| The n=500 cells are controls whose inference the row does not claim | Each must *resolve*, placing its exact 99% coverage interval clear of nominal on one side or the other. Below the floor is a published small-sample limitation; at or above it is the estimator turning out to be adequate. Only a straddling interval fails, because it is the one outcome that says nothing |
| Positivity is comfortable throughout | The smallest cumulative mechanism product on the comparison law sits between 0.006 and 0.03, and the property law bounds every conditional into [0.25, 0.75]. No cell here speaks to near-positivity behaviour or to an active bound |
| The row is bounded to ordinary end-of-study estimation | Survival has its own row below. This row excludes competing risks, longitudinal MSMs, observation weights, clustering, simultaneous bands, flexible learning, cross-fitting, active truncation, and R parity for learned mechanisms. Those are different estimators or compositions and require their own studies |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring. Single-correct-nuisance cells establish
consistency only. Calibrated influence-curve inference is claimed where both nuisance sequences are
correct.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/manifest.json)
carries the package checksum, R image digest, source commit, formulas, seeds, and artifact hashes.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/replicates.csv.gz),
[paired verdicts](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle/properties.csv)
carry every published row.

The sharp-null law replaces the outcome probabilities on the cells the contrasted plans traverse
and nothing else, so it shares this law's treatment, censoring, and L2 mechanisms exactly. L2 still
moves the outcome from 0.25 to 0.75, censoring is still informative through it, and the first arm
still matters. A baseline-only standardisation returns -0.0088 rather than the truth, so the null
is one an estimator has to work for. Both contrasts are exactly zero under it. Only the static one
is registered as a cell, because a type-I cell needs a nonzero-effect power control and the dynamic
contrast's power at n=4,000 is near 0.43 against a floor of 0.80.
