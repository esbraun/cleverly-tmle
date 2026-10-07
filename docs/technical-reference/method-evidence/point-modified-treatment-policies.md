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
| density ratio | Equation (3) of Díaz, Williams, Hoffman and Schenck (2023) on an oracle binned density with 160 bins | the same ratio array, read from the replicate data |
| outcome regression | a logistic regression on the dose, its square and both covariates | `SL.glm` on the same design, through `SL.glm.quadratic` |
| targeting | the covariate submodel over the observed and policy doses | the intercept fluctuation weighted by the ratio |
| intervals | pointwise 95% Wald from the influence curve | the same, from `lmtp`'s influence curve |

The untargeted estimates agree to rounding. The two fluctuations differ by construction, so the
paired rows are read under the default margins and are not an exactness check.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[halve below 3 vs natural course]` | difference in means under the modified treatment policies "move a dose at or below 3 halfway toward 3" against "leave the observed treatment mechanism unchanged" | `cleverly` point-treatment TMLE under modified treatment policies | -0.000050 to 0.000823 | 0.9520 | 0.9926 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[halve below 3 vs natural course]` | difference in means under the modified treatment policies "move a dose at or below 3 halfway toward 3" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000057 to 0.000814 | 0.9510 | 0.9941 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[piecewise vs natural course]` | difference in means under the modified treatment policies "lower the dose by 0.5 above 3, and leave it unchanged at or below 3" against "leave the observed treatment mechanism unchanged" | `cleverly` point-treatment TMLE under modified treatment policies | -0.000200 to 0.000671 | 0.9520 | 1.0489 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[piecewise vs natural course]` | difference in means under the modified treatment policies "lower the dose by 0.5 above 3, and leave it unchanged at or below 3" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000202 to 0.000674 | 0.9560 | 1.0432 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[x1.25 vs natural course]` | difference in means under the modified treatment policies "multiply the dose by 1.25, holding doses whose product exceeds 5.5" against "leave the observed treatment mechanism unchanged" | `cleverly` point-treatment TMLE under modified treatment policies | -0.000590 to 0.000986 | 0.9500 | 1.0138 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[x1.25 vs natural course]` | difference in means under the modified treatment policies "multiply the dose by 1.25, holding doses whose product exceeds 5.5" against "leave the observed treatment mechanism unchanged" | R `lmtp` | -0.000605 to 0.000960 | 0.9570 | 1.0250 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[halve below 3]` | mean under the modified treatment policy move a dose at or below 3 halfway toward 3 | `cleverly` point-treatment TMLE under modified treatment policies | -0.000142 to 0.0018 | 0.9570 | 1.0257 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[halve below 3]` | mean under the modified treatment policy move a dose at or below 3 halfway toward 3 | R `lmtp` | -0.000149 to 0.0018 | 0.9560 | 1.0264 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | `cleverly` point-treatment TMLE under modified treatment policies | -0.000408 to 0.0013 | 0.9630 | 1.0373 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | R `lmtp` | -0.000408 to 0.0013 | 0.9630 | 1.0373 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[piecewise]` | mean under the modified treatment policy lower the dose by 0.5 above 3, and leave it unchanged at or below 3 | `cleverly` point-treatment TMLE under modified treatment policies | -0.000283 to 0.0017 | 0.9550 | 1.0272 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[piecewise]` | mean under the modified treatment policy lower the dose by 0.5 above 3, and leave it unchanged at or below 3 | R `lmtp` | -0.000284 to 0.0017 | 0.9590 | 1.0264 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[x1.25]` | mean under the modified treatment policy multiply the dose by 1.25, holding doses whose product exceeds 5.5 | `cleverly` point-treatment TMLE under modified treatment policies | -0.000519 to 0.0018 | 0.9500 | 1.0342 | pass |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[x1.25]` | mean under the modified treatment policy multiply the dose by 1.25, holding doses whose product exceeds 5.5 | R `lmtp` | -0.000532 to 0.0018 | 0.9540 | 1.0431 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[halve below 3 vs natural course]` | difference in means under the modified treatment policies "move a dose at or below 3 halfway toward 3" against "leave the observed treatment mechanism unchanged" | 0.000008 | 0.0098 | 1.0066 | 0.0010 | 0.0039 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[piecewise vs natural course]` | difference in means under the modified treatment policies "lower the dose by 0.5 above 3, and leave it unchanged at or below 3" against "leave the observed treatment mechanism unchanged" | -2.804e-07 | 0.000349 | 1.0012 | -0.0040 | 0.0063 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ate_policy[x1.25 vs natural course]` | difference in means under the modified treatment policies "multiply the dose by 1.25, holding doses whose product exceeds 5.5" against "leave the observed treatment mechanism unchanged" | 0.000021 | 0.0143 | 1.0169 | -0.0070 | 0.0272 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[halve below 3]` | mean under the modified treatment policy move a dose at or below 3 halfway toward 3 | 0.000008 | 0.0043 | 1.0022 | 0.0010 | 0.0024 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[natural course]` | mean under the modified treatment policy leave the observed treatment mechanism unchanged | -1.977e-10 | 1.249e-07 | 1.0000 | 0 | 1.439e-08 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[piecewise]` | mean under the modified treatment policy lower the dose by 0.5 above 3, and leave it unchanged at or below 3 | -2.806e-07 | 0.000157 | 1.0008 | -0.0040 | 0.0020 vs 0.0500 | equivalent |
| one truncated-normal dose on [0, 6] with an outcome quadratic in the dose | `ey_policy[x1.25]` | mean under the modified treatment policy multiply the dose by 1.25, holding doses whose product exceeds 5.5 | 0.000021 | 0.0096 | 1.0118 | -0.0040 | 0.0173 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `x1_25__both_correct` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000884 to 0.000772, margin 0.0025, SE ratio 0.9656 | pass |
| `double_robustness` | `x1_25__both_wrong` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0291 to 0.0306, margin 0.0023, SE ratio 1.3575 | pass |
| `double_robustness` | `x1_25__density_correct` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: only the continuous-dose density ratio is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000615 to 0.0011, margin 0.0027, SE ratio 0.9800 | pass |
| `double_robustness` | `x1_25__outcome_correct` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000597 to 0.000893, margin 0.0023, SE ratio 1.2923 | pass |
| `interval_calibration` | `classifier_route__correctly_specified` | positive | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9255 to 0.9533, SE ratio 0.9431 to 1.0244, empirical efficiency ratio 0.9598 to 1.0421, reported efficiency ratio 0.9793 to 0.9860 | pass |
| `interval_calibration` | `classifier_route__noise_control` | control | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7916 to 0.8368, SE ratio 0.6667 to 0.7239, empirical efficiency ratio 1.3578 to 1.4732, reported efficiency ratio 0.9793 to 0.9861 | pass |
| `interval_calibration` | `classifier_route__shrunken_se_control` | control | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8010 to 0.8454, SE ratio 0.6605 to 0.7178, empirical efficiency ratio 0.9587 to 1.0408, reported efficiency ratio 0.6855 to 0.6902 | pass |
| `interval_calibration` | `halve__correctly_specified` | positive | the contrast of the declared policy halve below 3 against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9490 to 0.9717, SE ratio 1.0015 to 1.0881, empirical efficiency ratio 0.9209 to 0.9999, reported efficiency ratio 1.0003 to 1.0033 | **fail** |
| `interval_calibration` | `halve__noise_control` | control | the contrast of the declared policy halve below 3 against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8262 to 0.8681, SE ratio 0.6978 to 0.7585, empirical efficiency ratio 1.3207 to 1.4354, reported efficiency ratio 1.0003 to 1.0033 | pass |
| `interval_calibration` | `halve__shrunken_se_control` | control | the contrast of the declared policy halve below 3 against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8336 to 0.8747, SE ratio 0.7015 to 0.7626, empirical efficiency ratio 0.9194 to 0.9995, reported efficiency ratio 0.7002 to 0.7023 | pass |
| `interval_calibration` | `piecewise__correctly_specified` | positive | the contrast of the dose lowered by 0.5 above 3 against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9428 to 0.9670, SE ratio 0.9648 to 1.0402, empirical efficiency ratio 0.9617 to 1.0371, reported efficiency ratio 0.9994 to 1.0019 | pass |
| `interval_calibration` | `piecewise__noise_control` | control | the contrast of the dose lowered by 0.5 above 3 against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8058 to 0.8497, SE ratio 0.6747 to 0.7291, empirical efficiency ratio 1.3723 to 1.4828, reported efficiency ratio 0.9994 to 1.0019 | pass |
| `interval_calibration` | `piecewise__shrunken_se_control` | control | the contrast of the dose lowered by 0.5 above 3 against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8005 to 0.8449, SE ratio 0.6743 to 0.7293, empirical efficiency ratio 0.9604 to 1.0385, reported efficiency ratio 0.6996 to 0.7013 | pass |
| `interval_calibration` | `x1_25__correctly_specified` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9272 to 0.9546, SE ratio 0.9565 to 1.0350, empirical efficiency ratio 0.9561 to 1.0335, reported efficiency ratio 0.9853 to 0.9927 | pass |
| `interval_calibration` | `x1_25__noise_control` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8226 to 0.8648, SE ratio 0.6944 to 0.7520, empirical efficiency ratio 1.3156 to 1.4236, reported efficiency ratio 0.9853 to 0.9927 | pass |
| `interval_calibration` | `x1_25__shrunken_se_control` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7974 to 0.8421, SE ratio 0.6693 to 0.7246, empirical efficiency ratio 0.9556 to 1.0339, reported efficiency ratio 0.6898 to 0.6949 | pass |
| `inverse_necessity` | `halve__declared_inverse` | positive | the contrast of the declared policy halve below 3 against the natural course: the density ratio reads each piece at its declared inverse | bias interval inside the equivalence margin | bias -0.000693 to 0.000253, margin 0.0014 | pass |
| `inverse_necessity` | `halve__inverse_dropped_control` | control | the contrast of the declared policy halve below 3 against the natural course: the same fit reads each moving piece at the dose itself | bias interval must fall entirely outside the margin | bias -0.0149 to -0.0149, margin 0 | pass |
| `power` | `x1_25__alternative` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9600, 0.9346 to 0.9777 | pass |
| `root_n_and_efficiency` | `x1_25__n_2000` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000508, coverage 0.9184 to 0.9675, SE ratio 0.9918 | pass |
| `root_n_and_efficiency` | `x1_25__n_500` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000752, coverage 0.8814 to 0.9418, SE ratio 0.9604 | pass |
| `root_n_and_efficiency` | `x1_25__n_8000` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000077, coverage 0.9105 to 0.9622, SE ratio 1.0279 | pass |
| `root_n_rate` | `x1_25__empirical_sd` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5510 to -0.4768 | pass |
| `root_n_rate` | `x1_25__reported_se` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4932 to -0.4835 | pass |
| `targeting_necessity` | `x1_25__targeted` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000221 to 0.0015, margin 0.0026 | pass |
| `targeting_necessity` | `x1_25__untargeted` | control | the contrast of the dose times 1.25, capped at 5.5, against the natural course: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0247 to -0.0247, margin 0 | pass |
| `type_i_error` | `x1_25__sharp_null` | positive | the contrast of the dose times 1.25, capped at 5.5, against the natural course: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0617, 0.0391 to 0.0915 | pass |
<!-- /generated -->

Every family reads the law with the declared policies. The truths come from Gauss-Legendre
quadrature over the dose and Gauss-Hermite quadrature over the continuous covariate, checked at
twice the order. The efficiency bounds are the standard deviations of the closed-form efficient
influence functions, over 400,000 draws.

`classifier_route` estimates the ratio by stacked classification. Its classifier is a logistic
regression on the true log ratio, so it is correctly specified. `inverse_necessity` fits "halve
below 3" twice on one sample. The positive arm reads Equation (3) at the declared inverse, and the
control reads it at the dose itself.

The `halve__correctly_specified` calibration cell is red on its SE-ratio band. The SE ratio is
1.043, and its 99% interval is 1.0015 to 1.088 against an upper bound of 1.07. Its coverage is
0.9615, inside the coverage band, so the reported standard error is conservative. The influence
curve with the binned ratio predicts a ratio of 0.9998 for this cell. At 2,000 replications a
calibrated cell crosses the bound with a probability of about 5%.

The study was declared `gated`. By its declared red-cell route it was re-registered under
`reporting`, with the owner row `mtp-point-calibration`, and the run repeated with no other
change. The repeat reproduced every artifact of the first run.

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
| `max_standardized_bias` | 0.0723 | largest primary standardized bias |
| `min_coverage` | 0.9500 | lowest primary coverage |
| `max_margin_utilization` | 0.0143 | largest paired similarity-margin share |
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
| the density is an oracle | the ratio reads the generating density through 160 bins. The [bin count paragraph](../longitudinal-tmle.md#modified-treatment-policies-at-a-node) states what a fixed bin count means for double robustness |
| one calibration cell is red under `reporting` | the `halve` cell's standard error is conservative by about 4%. The `mtp-point-calibration` owner holds it |
| the paired rows are not an exactness check | the two fluctuations differ by construction, so the pairs read under the default margins |
| a policy must be known and fixed | a policy learned from the sample, or one that depends on the law, is outside this evidence |
| no cell for a categorical treatment, a risk-ratio tilt or a randomized policy | these compositions have the exact-law and end-to-end evidence of the [point-treatment section](../point-treatment-tmle.md#modified-treatment-policies) only |
