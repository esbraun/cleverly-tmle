# Stochastic point-treatment regimes

This study validates an ordinary known stochastic regime over two treatment arms. It compares
`cleverly` with pinned R `lmtp` 1.5.4 on identical binary-outcome samples and exact treatment
probabilities.

The comparison runs through the shared point adapter rather than through `lmtp`'s shift interface.
That interface takes a function of the natural treatment value, and this regime ignores that
value. The adapter supplies the analytic density ratio and the shifted frame directly. The static
regime uses `mtp = FALSE`. The tilt draws one treatment value per unit from the declared density
and uses `mtp = TRUE`. Pinned `tmle3` stays unusable for this parameter, because `Param_TSM`
evaluates a counterfactual at one treatment value and integrates over no density.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with a known stochastic treatment density | `ate_regime[tilt vs never]` | difference in means under the regimes "draw from the known stochastic tilt" against "assign no treatment" | `cleverly` | -0.000970 to 0.000901 | 0.9437 | 1.0142 | pass |
| binary-outcome law with a known stochastic treatment density | `ate_regime[tilt vs never]` | difference in means under the regimes "draw from the known stochastic tilt" against "assign no treatment" | R `lmtp` | -0.000923 to 0.000980 | 0.9487 | 1.0295 | pass |
| binary-outcome law with a known stochastic treatment density | `ey_regime[never]` | mean under the regime assign no treatment | `cleverly` | -0.0020 to 0.000657 | 0.9387 | 0.9948 | pass |
| binary-outcome law with a known stochastic treatment density | `ey_regime[never]` | mean under the regime assign no treatment | R `lmtp` | -0.0020 to 0.000662 | 0.9375 | 0.9946 | pass |
| binary-outcome law with a known stochastic treatment density | `ey_regime[tilt]` | mean under the regime draw from the known stochastic tilt | `cleverly` | -0.0018 to 0.000390 | 0.9375 | 0.9993 | pass |
| binary-outcome law with a known stochastic treatment density | `ey_regime[tilt]` | mean under the regime draw from the known stochastic tilt | R `lmtp` | -0.0018 to 0.000482 | 0.9413 | 0.9976 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law with a known stochastic treatment density | `ate_regime[tilt vs never]` | difference in means under the regimes "draw from the known stochastic tilt" against "assign no treatment" | -0.000063 | 0.0403 | 0.9997 | -0.0050 | 0.0387 vs 0.0500 | equivalent |
| binary-outcome law with a known stochastic treatment density | `ey_regime[never]` | mean under the regime assign no treatment | -0.000001 | 0.000471 | 0.9981 | 0.0012 | 0.0013 vs 0.0500 | equivalent |
| binary-outcome law with a known stochastic treatment density | `ey_regime[tilt]` | mean under the regime draw from the known stochastic tilt | -0.000064 | 0.0348 | 0.9899 | -0.0037 | 0.0176 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `density_necessity` | `tilt__declared` | positive | known stochastic tilt: the estimator integrates over the declared covariate-dependent treatment density | bias interval inside the equivalence margin | bias -0.000592 to 0.000920, margin 0.0025 | pass |
| `density_necessity` | `tilt__uniform_control` | control | known stochastic tilt: the same fit replaces the declared density with a uniform distribution | bias interval must fall entirely outside the margin | bias 0.0269 to 0.0285, margin 0.0026 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.000926 to 0.000563, margin 0.0025 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.1967 to -0.1954, margin 0.0022 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.000754 to 0.000361, margin 0.0019 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin | bias -0.0011 to 0.000477, margin 0.0027 | pass |
| `interval_calibration` | `tilt__correctly_specified` | positive | known stochastic tilt: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9445 to 0.9619, SE ratio 0.9796 to 1.0372, empirical efficiency ratio 0.9625 to 1.0191, reported efficiency ratio 0.9971 to 0.9996 | pass |
| `interval_calibration` | `tilt__noise_control` | control | known stochastic tilt: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8238 to 0.8539, SE ratio 0.6954 to 0.7353, empirical efficiency ratio 1.3579 to 1.4358, reported efficiency ratio 0.9971 to 0.9997 | pass |
| `interval_calibration` | `tilt__shrunken_se_control` | control | known stochastic tilt: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8150 to 0.8457, SE ratio 0.6863 to 0.7265, empirical efficiency ratio 0.9620 to 1.0183, reported efficiency ratio 0.6979 to 0.6998 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9934 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000196, coverage 0.9223 to 0.9647, SE ratio 1.0015 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.000643, coverage 0.9208 to 0.9637, SE ratio 0.9871 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000148, coverage 0.9267 to 0.9677, SE ratio 0.9718 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5273 to -0.4583 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5019 to -0.4973 | pass |
| `targeting_necessity` | `tilt__targeted` | positive | known stochastic tilt: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000897 to 0.000709, margin 0.0027 | pass |
| `targeting_necessity` | `tilt__untargeted` | control | known stochastic tilt: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.2251 to -0.2250, margin 0.000208 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0512, 0.0333 to 0.0748 | pass |
<!-- /generated -->

The property study tests double robustness, exact-EIF calibration, a confounded sharp null, and
targeting. A paired control replaces the declared density with a uniform density.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `paired_tests_passed` | 3 | paired comparisons passing |
| `paired_tests_total` | 3 | paired comparisons reported |
| `property_cells_passed` | 18 | property cells passing |
| `property_cells_total` | 18 | property cells reported |
| `max_standardized_bias` | 0.0588 | largest primary standardized bias |
| `min_coverage` | 0.9375 | lowest primary coverage |
| `max_margin_utilization` | 0.0403 | largest paired similarity-margin share |
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
| `margin:paired_difference` | 0.1500 | paired-similarity margin |
| `margin:rmse_noninferiority` | 1.1000 | RMSE non-inferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage non-inferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration non-inferiority bound |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |
| `margin:necessity_displacement` | 0.2500 | minimum density-control displacement |

## Limits

- The laws use binary treatment, binary outcome, and one three-level baseline covariate.
- The primary working outcome model is logistic and the natural treatment mechanism is supplied exactly.
- The intervention density is known, covariate dependent, and supported on both arms.
- The study covers ordinary, non-cross-fitted targeting and pointwise Wald intervals.
- It excludes missing outcomes, weights, clusters, simultaneous bands, and flexible learners.
- It does not validate continuous-dose, incremental, or longitudinal stochastic interventions.
- The reference evaluates its shifted regression at one draw from the declared density. It does
  not integrate over that density, so it carries a Monte Carlo variance that `cleverly` does not.
- Both implementations fit a logistic outcome regression on the covariate as one numeric column.
  The paired comparison is therefore symmetric in its outcome model. The calibration claim for a
  correctly specified pair belongs to the `interval_calibration` cells, which use exact nuisances.
