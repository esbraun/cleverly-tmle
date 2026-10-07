# Point modified treatment policies

This study validates point-treatment TMLE under modified treatment policies beyond the additive
shift. The law has one dose, a normal truncated to $[0, 6]$ given two covariates, and a binary
outcome whose mean is quadratic in the dose. The
[modified treatment policies](../point-treatment-tmle.md#modified-treatment-policies) section
states the estimator, its conditions and its exact-law evidence.

The study pairs each fit with pinned `lmtp` 1.5.4. Both sides read the same density ratio, the
one this package computed, and the same policy dose. The adapter
`tests/canonical/lmtp_mtp_adapter.R` writes both into an `lmtp` task, so `lmtp` fits no ratio.

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,000 samples of 2,000 rows | the identical rows |
| policies | the natural course; `Scale(1.25, cap=5.5)`; a `Piecewise` policy that lowers doses above 3 by 0.5; the declared `ModifiedPolicy` "halve below 3", which moves a dose at or below 3 halfway toward 3 | the same policy doses, read from the replicate data |
| density ratio | Equation (3) of Díaz, Williams, Hoffman and Schenck (2023) on an oracle binned density with `ceil(320 (n / 2000)^(2/3))` bins (320 at n = 2,000) | the same ratio array, read from the replicate data |
| outcome regression | a logistic regression on the dose, its square and both covariates | `SL.glm` on the same design, through `SL.glm.quadratic` |
| targeting | the covariate submodel over the observed and policy doses | the intercept fluctuation weighted by the ratio |
| intervals | pointwise 95% Wald from the influence curve | the same, from `lmtp`'s influence curve |

The untargeted estimates agree to rounding. The two fluctuations differ by construction, so the
paired rows are read under the default margins and are not an exactness check.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[halve below 3 vs natural course]` | difference in means under the modified treatment policies "move a dose at or below 3 halfway toward 3" against "leave the observed treatment mechanism unchanged" | `cleverly` point-treatment TMLE under modified treatment policies | -0.000050 to 0.000822 | 0.9520 | 0.9931 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[halve below 3 vs natural course]` | difference in means under the modified treatment policies "move a dose at or below 3 halfway toward 3" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000058 to 0.000813 | 0.9510 | 0.9945 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[piecewise vs natural course]` | difference in means under the modified treatment policies "lower the dose by 0.5 above 3, and leave it unchanged at or below 3" against "leave the observed treatment mechanism unchanged" | `cleverly` point-treatment TMLE under modified treatment policies | -0.000197 to 0.000675 | 0.9550 | 1.0481 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[piecewise vs natural course]` | difference in means under the modified treatment policies "lower the dose by 0.5 above 3, and leave it unchanged at or below 3" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000199 to 0.000677 | 0.9540 | 1.0429 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[x1.25 vs natural course]` | difference in means under the modified treatment policies "multiply the dose by 1.25, holding doses whose product exceeds 5.5" against "leave the observed treatment mechanism unchanged" | `cleverly` point-treatment TMLE under modified treatment policies | -0.000575 to 0.000992 | 0.9510 | 1.0150 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[x1.25 vs natural course]` | difference in means under the modified treatment policies "multiply the dose by 1.25, holding doses whose product exceeds 5.5" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000589 to 0.000964 | 0.9560 | 1.0275 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[halve below 3]` | mean under the modified treatment policy move a dose at or below 3 halfway toward 3 | `cleverly` point-treatment TMLE under modified treatment policies | -0.000142 to 0.0018 | 0.9550 | 1.0256 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[halve below 3]` | mean under the modified treatment policy move a dose at or below 3 halfway toward 3 | R `lmtp` | -0.000150 to 0.0018 | 0.9550 | 1.0264 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | `cleverly` point-treatment TMLE under modified treatment policies | -0.000408 to 0.0013 | 0.9630 | 1.0373 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | R `lmtp` | -0.000408 to 0.0013 | 0.9630 | 1.0373 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[piecewise]` | mean under the modified treatment policy lower the dose by 0.5 above 3, and leave it unchanged at or below 3 | `cleverly` point-treatment TMLE under modified treatment policies | -0.000280 to 0.0017 | 0.9560 | 1.0267 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[piecewise]` | mean under the modified treatment policy lower the dose by 0.5 above 3, and leave it unchanged at or below 3 | R `lmtp` | -0.000281 to 0.0017 | 0.9590 | 1.0258 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[x1.25]` | mean under the modified treatment policy multiply the dose by 1.25, holding doses whose product exceeds 5.5 | `cleverly` point-treatment TMLE under modified treatment policies | -0.000507 to 0.0018 | 0.9490 | 1.0332 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[x1.25]` | mean under the modified treatment policy multiply the dose by 1.25, holding doses whose product exceeds 5.5 | R `lmtp` | -0.000521 to 0.0018 | 0.9540 | 1.0418 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[halve below 3 vs natural course]` | difference in means under the modified treatment policies "move a dose at or below 3 halfway toward 3" against "leave the observed treatment mechanism unchanged" | 0.000008 | 0.0104 | 1.0065 | 0.0010 | 0.0039 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[piecewise vs natural course]` | difference in means under the modified treatment policies "lower the dose by 0.5 above 3, and leave it unchanged at or below 3" against "leave the observed treatment mechanism unchanged" | 1.898e-07 | 0.000236 | 1.0014 | 0.0010 | 0.0061 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[x1.25 vs natural course]` | difference in means under the modified treatment policies "multiply the dose by 1.25, holding doses whose product exceeds 5.5" against "leave the observed treatment mechanism unchanged" | 0.000021 | 0.0147 | 1.0183 | -0.0050 | 0.0293 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[halve below 3]` | mean under the modified treatment policy move a dose at or below 3 halfway toward 3 | 0.000008 | 0.0046 | 1.0021 | 0 | 0.0023 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | -1.977e-10 | 1.249e-07 | 1.0000 | 0 | 1.439e-08 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[piecewise]` | mean under the modified treatment policy lower the dose by 0.5 above 3, and leave it unchanged at or below 3 | 1.896e-07 | 0.000106 | 1.0007 | -0.0030 | 0.0020 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[x1.25]` | mean under the modified treatment policy multiply the dose by 1.25, holding doses whose product exceeds 5.5 | 0.000021 | 0.0099 | 1.0117 | -0.0050 | 0.0174 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `x1_25__both_correct` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000902 to 0.000752, margin 0.0025, SE ratio 0.9623 | pass |
| `double_robustness` | `x1_25__both_wrong` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0293 to 0.0308, margin 0.0023, SE ratio 1.3549 | pass |
| `double_robustness` | `x1_25__density_correct` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: only the continuous-dose density ratio is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000541 to 0.0012, margin 0.0027, SE ratio 0.9778 | pass |
| `double_robustness` | `x1_25__outcome_correct` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000585 to 0.000905, margin 0.0023, SE ratio 1.2833 | pass |
| `interval_calibration` | `classifier_route__correctly_specified` | positive | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9255 to 0.9533, SE ratio 0.9431 to 1.0244, empirical efficiency ratio 0.9598 to 1.0421, reported efficiency ratio 0.9793 to 0.9860 | pass |
| `interval_calibration` | `classifier_route__noise_control` | control | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7916 to 0.8368, SE ratio 0.6667 to 0.7239, empirical efficiency ratio 1.3578 to 1.4732, reported efficiency ratio 0.9793 to 0.9861 | pass |
| `interval_calibration` | `classifier_route__shrunken_se_control` | control | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8010 to 0.8454, SE ratio 0.6605 to 0.7178, empirical efficiency ratio 0.9587 to 1.0408, reported efficiency ratio 0.6855 to 0.6902 | pass |
| `interval_calibration` | `halve__correctly_specified` | positive | the contrast of the declared policy halve below 3 against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9479 to 0.9709, SE ratio 1.0011 to 1.0881, empirical efficiency ratio 0.9208 to 1.0003, reported efficiency ratio 1.0003 to 1.0034 | **fail** |
| `interval_calibration` | `halve__noise_control` | control | the contrast of the declared policy halve below 3 against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8268 to 0.8685, SE ratio 0.6982 to 0.7588, empirical efficiency ratio 1.3202 to 1.4346, reported efficiency ratio 1.0004 to 1.0033 | pass |
| `interval_calibration` | `halve__shrunken_se_control` | control | the contrast of the declared policy halve below 3 against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8315 to 0.8728, SE ratio 0.7014 to 0.7624, empirical efficiency ratio 0.9196 to 0.9996, reported efficiency ratio 0.7002 to 0.7023 | pass |
| `interval_calibration` | `piecewise__correctly_specified` | positive | the contrast of the dose lowered by 0.5 above 3 against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9422 to 0.9665, SE ratio 0.9660 to 1.0414, empirical efficiency ratio 0.9608 to 1.0360, reported efficiency ratio 0.9993 to 1.0018 | pass |
| `interval_calibration` | `piecewise__noise_control` | control | the contrast of the dose lowered by 0.5 above 3 against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8068 to 0.8506, SE ratio 0.6751 to 0.7295, empirical efficiency ratio 1.3718 to 1.4819, reported efficiency ratio 0.9994 to 1.0018 | pass |
| `interval_calibration` | `piecewise__shrunken_se_control` | control | the contrast of the dose lowered by 0.5 above 3 against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8042 to 0.8482, SE ratio 0.6751 to 0.7302, empirical efficiency ratio 0.9591 to 1.0374, reported efficiency ratio 0.6996 to 0.7013 | pass |
| `interval_calibration` | `x1_25__correctly_specified` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9250 to 0.9529, SE ratio 0.9533 to 1.0320, empirical efficiency ratio 0.9548 to 1.0312, reported efficiency ratio 0.9804 to 0.9877 | pass |
| `interval_calibration` | `x1_25__noise_control` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8210 to 0.8634, SE ratio 0.6915 to 0.7495, empirical efficiency ratio 1.3139 to 1.4224, reported efficiency ratio 0.9805 to 0.9876 | pass |
| `interval_calibration` | `x1_25__shrunken_se_control` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7995 to 0.8440, SE ratio 0.6677 to 0.7225, empirical efficiency ratio 0.9542 to 1.0313, reported efficiency ratio 0.6863 to 0.6914 | pass |
| `inverse_necessity` | `halve__declared_inverse` | positive | the contrast of the declared policy halve below 3 against the natural course: the density ratio reads each piece at its declared inverse | bias interval inside the equivalence margin | bias -0.000694 to 0.000252, margin 0.0014 | pass |
| `inverse_necessity` | `halve__inverse_dropped_control` | control | the contrast of the declared policy halve below 3 against the natural course: the same fit reads each moving piece at the dose itself | bias interval must fall entirely outside the margin | bias -0.0149 to -0.0149, margin 0 | pass |
| `power` | `x1_25__alternative` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9617, 0.9367 to 0.9790 | pass |
| `root_n_and_efficiency` | `x1_25__n_2000` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000496, coverage 0.9204 to 0.9688, SE ratio 0.9897 | pass |
| `root_n_and_efficiency` | `x1_25__n_500` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000758, coverage 0.8833 to 0.9432, SE ratio 0.9515 | pass |
| `root_n_and_efficiency` | `x1_25__n_8000` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000072, coverage 0.9125 to 0.9636, SE ratio 1.0197 | pass |
| `root_n_rate` | `x1_25__empirical_sd` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5503 to -0.4760 | pass |
| `root_n_rate` | `x1_25__reported_se` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4922 to -0.4826 | pass |
| `targeting_necessity` | `x1_25__targeted` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000143 to 0.0015, margin 0.0026 | pass |
| `targeting_necessity` | `x1_25__untargeted` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0247 to -0.0247, margin 0 | pass |
| `type_i_error` | `x1_25__sharp_null` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0667, 0.0431 to 0.0973 | pass |
<!-- /generated -->

Every family reads the law with the declared policies. The truths come from Gauss-Legendre
quadrature over the dose and Gauss-Hermite quadrature over the continuous covariate, checked at
twice the order. The efficiency bounds are the standard deviations of the closed-form efficient
influence functions, over 400,000 draws.

`classifier_route` estimates the ratio by stacked classification. Its classifier is a logistic
regression on the true log ratio, so it is correctly specified. `inverse_necessity` fits "halve
below 3" twice on one sample. The positive arm reads Equation (3) at the declared inverse, and the
control reads it at the dose itself.

The oracle's bin count grows as $n^{2/3}$. The first run read a fixed 160 bins at every size.
After `longitudinal-mtp` found that a fixed count leaves the binned ratio inconsistent, this
study re-ran under a fresh declaration.

The `halve__correctly_specified` calibration cell is red on its SE-ratio band, in both runs. The
SE ratio is 1.0429, and its 99% interval is 1.0011 to 1.0881 against an upper bound of 1.07. It
was 1.0430 at 160 bins, so the bin count does not move it. Its coverage is 0.9605, and the
reported standard error matches the efficiency bound (ratio 1.0018). The influence curve with the
binned ratio predicts a ratio of 0.9998, and at 2,000 replications a calibrated cell crosses the
bound with a probability of about 5%.

The diagnosis finds no defect. By the red-cell rule the study moved to `reporting`, with the owner
row `mtp-point-calibration`, and the run repeated with no other change. The repeat reproduced
every artifact.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_passed` | 14 | truth tests passing |
| `independent_tests_total` | 14 | truth tests reported |
| `paired_tests_passed` | 7 | paired comparisons passing |
| `paired_tests_total` | 7 | paired comparisons reported |
| `property_cells_passed` | 26 | property cells passing |
| `property_cells_total` | 27 | property cells reported |
| `max_standardized_bias` | 0.0722 | largest primary standardized bias |
| `min_coverage` | 0.9490 | lowest primary coverage |
| `max_margin_utilization` | 0.0147 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
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
| `margin:targeting_displacement` | 0.1000 | minimum targeting displacement |
| `margin:inverse_displacement` | 0.1000 | minimum inverse displacement |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:minimum_power` | 0.8000 | power lower bound |

## Limits

| limit | what it means for use |
| --- | --- |
| one law, one dose and two covariates | the evidence covers one bounded dose law with policies fixed before the run |
| the density is an oracle | the ratio reads the generating density through a bin count that grows with $n$. The [bin count paragraph](../longitudinal-tmle.md#modified-treatment-policies-at-a-node) states why a fixed count is not consistent |
| one calibration cell is red under `reporting` | the `halve` cell's standard error is conservative by about 4%. The `mtp-point-calibration` owner holds it |
| the paired rows are not an exactness check | the two fluctuations differ by construction, so the pairs read under the default margins |
| a policy must be known and fixed | a policy learned from the sample, or one that depends on the law, is outside this evidence |
| no cell for a categorical treatment, a risk-ratio tilt or a randomized policy | these compositions have the exact-law and end-to-end evidence of the [point-treatment section](../point-treatment-tmle.md#modified-treatment-policies) only |
