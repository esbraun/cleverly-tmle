# Known-score ATT and ATC on point TMLE

This study validates point `TMLE` fitted with `conditional_form="known_score"` on data that
declares its treatment mechanism. With $g_0$ known, the ATT is the plug-in
$P_n\{g_0(1 \mid W)\tau^*(W)\}/P_n g_0(1 \mid W)$, the ATC weights by $g_0(0 \mid W)$, and the
reported influence curve is $D_{\text{KS}}$, the canonical gradient of the model in which $g_0$
is known (Chen, Hong and Tarozzi 2008, Theorem 2). The
[known treatment mechanism section](../point-treatment-tmle.md#known-treatment-mechanism) states
the estimator and its curve. At a consistent outcome regression the variance falls below that of
the shipped arm-mean form by $E[g_0(1-g_0)(\tau-\psi)^2]/p^2$, which is zero only when the effect
is constant where $0 < g_0 < 1$. With a declared mechanism the remainder is zero for any outcome
regression limit, so the estimate stays unbiased when the outcome regression is wrong.

No maintained package reports the known-score ATT with its own curve, so
no canonical implementation is compared. The study rests on exact truths and on
repeated-sampling properties.

## What was fitted

| law | covariates, mechanism and outcome | fits |
| --- | --- | --- |
| S | $W_1 \sim U(-\sqrt3, \sqrt3)$, $W_2 \sim \text{Bernoulli}(0.5)$; $g_0 = \operatorname{expit}(-0.3 + W_1 + 0.5 W_2)$; $Y = W_1 + 0.5 W_2 + A(1 + 1.5 W_1) + \varepsilon$, $\varepsilon \sim U(-\sqrt3, \sqrt3)$ | least squares on $(1, A, W_1, W_2, A W_1)$ (correct) or on $(1, A, W_2)$ (wrong); in sample, and stacked over 5 folds with declared `q_bounds=(-8, 8)`; the known-score and the arm-mean ATT and ATC on the same draws; n = 500, 2,000 and 8,000 |
| S0 | law S with $\tau \equiv 1$, where the two forms have the same efficiency bound | the correct outcome regression, both forms, n = 2,000 |
| B | the binary-outcome law of `tests/studies/known_mechanism_law.py`, two and three arms | a main-terms logistic outcome regression on $(A, W_1, W_3)$, which is wrong; the ATT and ATC against arm 0, n = 2,000 |

The laws, learners and exact truths are in `tests/studies/known_score_att_law.py`. Law S truths
and the exact SDs of $D_{\text{KS}}$ and $D_{\text{AM}}$ come from a 200-node Gauss-Legendre rule.
The control declares the constant mechanism 0.49 on law S, so its exact bias is the ATE minus the
ATT. `tests/unit/test_known_score_att_design.py` recomputes every truth, SD and size, and
`tests/unit/test_known_score_att.py` checks the curve as a complex-step Gateaux derivative on
exact finite laws, with mutation controls that restore the arm-mean term.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000: the declared mechanism, least squares on (1, A, W1, W2, A W1) (correct), in sample; the known-score ATT and ATC | `atc` | average effect on the untreated | `cleverly` | -0.0048 to 0.0021 | 0.9479 | 0.9839 | pass |
| law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000: the declared mechanism, least squares on (1, A, W1, W2, A W1) (correct), in sample; the known-score ATT and ATC | `att` | average effect on the treated | `cleverly` | -0.0025 to 0.0042 | 0.9492 | 0.9981 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `efficiency_gain` | `atc__arm_mean` | diagnostic | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the ATC: the arm-mean fit (conditional_form='arm_mean') on the same draws | none. The row is a diagnostic and states no verdict | spread ratio 0.9088, 0.8946 to 0.9231, boundary 0.9800 | reported |
| `efficiency_gain` | `atc__known_score` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the ATC: the known-score fit, paired with the arm-mean fit of the same draws | the 99% paired-bootstrap upper bound of SD(known score) / SD(arm mean) lies below 0.98 | spread ratio 0.9088, 0.8946 to 0.9231, boundary 0.9800 | pass |
| `efficiency_gain` | `atc_zero_gain__arm_mean` | diagnostic | law S with a constant effect (law S0), n = 2,000, the ATC: the arm-mean fit (conditional_form='arm_mean') on the same draws | none. The row is a diagnostic and states no verdict | spread ratio 0.9993, 0.9985 to 1.0001, boundary 0.9800 | reported |
| `efficiency_gain` | `atc_zero_gain__known_score` | control | law S with a constant effect (law S0), n = 2,000, the ATC: the known-score fit where the gain is zero, paired with the arm-mean fit | the 99% paired-bootstrap upper bound of SD(known score) / SD(arm mean) reaches 0.98 | spread ratio 0.9993, 0.9985 to 1.0001, boundary 0.9800, reached by the upper endpoint | pass |
| `efficiency_gain` | `att__arm_mean` | diagnostic | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the ATT: the arm-mean fit (conditional_form='arm_mean') on the same draws | none. The row is a diagnostic and states no verdict | spread ratio 0.9032, 0.8889 to 0.9174, boundary 0.9800 | reported |
| `efficiency_gain` | `att__known_score` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the ATT: the known-score fit, paired with the arm-mean fit of the same draws | the 99% paired-bootstrap upper bound of SD(known score) / SD(arm mean) lies below 0.98 | spread ratio 0.9032, 0.8889 to 0.9174, boundary 0.9800 | pass |
| `efficiency_gain` | `att_zero_gain__arm_mean` | diagnostic | law S with a constant effect (law S0), n = 2,000, the ATT: the arm-mean fit (conditional_form='arm_mean') on the same draws | none. The row is a diagnostic and states no verdict | spread ratio 1.0001, 0.9993 to 1.0009, boundary 0.9800 | reported |
| `efficiency_gain` | `att_zero_gain__known_score` | control | law S with a constant effect (law S0), n = 2,000, the ATT: the known-score fit where the gain is zero, paired with the arm-mean fit | the 99% paired-bootstrap upper bound of SD(known score) / SD(arm mean) reaches 0.98 | spread ratio 1.0001, 0.9993 to 1.0009, boundary 0.9800, reached by the upper endpoint | pass |
| `interval_calibration` | `atc_arm_mean_in_sample__correctly_specified` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the arm-mean ATC on the same draws, in sample: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9386 to 0.9554, SE ratio 0.9613 to 1.0147, empirical efficiency ratio 0.9837 to 1.0379, reported efficiency ratio 0.9967 to 0.9994 | pass |
| `interval_calibration` | `atc_arm_mean_in_sample__noise_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the arm-mean ATC on the same draws, in sample: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8154 to 0.8435, SE ratio 0.6796 to 0.7167, empirical efficiency ratio 1.3928 to 1.4684, reported efficiency ratio 0.9967 to 0.9993 | pass |
| `interval_calibration` | `atc_arm_mean_in_sample__shrunken_se_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the arm-mean ATC on the same draws, in sample: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8092 to 0.8377, SE ratio 0.6739 to 0.7104, empirical efficiency ratio 0.9837 to 1.0365, reported efficiency ratio 0.6977 to 0.6995 | pass |
| `interval_calibration` | `atc_known_score_in_sample__correctly_specified` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATC, in sample: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9364 to 0.9535, SE ratio 0.9609 to 1.0128, empirical efficiency ratio 0.9852 to 1.0381, reported efficiency ratio 0.9963 to 0.9992 | pass |
| `interval_calibration` | `atc_known_score_in_sample__noise_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATC, in sample: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8162 to 0.8443, SE ratio 0.6814 to 0.7180, empirical efficiency ratio 1.3898 to 1.4643, reported efficiency ratio 0.9963 to 0.9992 | pass |
| `interval_calibration` | `atc_known_score_in_sample__shrunken_se_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATC, in sample: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8064 to 0.8351, SE ratio 0.6726 to 0.7090, empirical efficiency ratio 0.9849 to 1.0384, reported efficiency ratio 0.6974 to 0.6994 | pass |
| `interval_calibration` | `atc_known_score_stacked__correctly_specified` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATC, stacked over five folds with q_bounds (-8, 8): both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9373 to 0.9543, SE ratio 0.9624 to 1.0149, empirical efficiency ratio 0.9858 to 1.0396, reported efficiency ratio 0.9989 to 1.0018 | pass |
| `interval_calibration` | `atc_known_score_stacked__noise_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATC, stacked over five folds with q_bounds (-8, 8): one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8079 to 0.8365, SE ratio 0.6753 to 0.7125, empirical efficiency ratio 1.4042 to 1.4814, reported efficiency ratio 0.9989 to 1.0018 | pass |
| `interval_calibration` | `atc_known_score_stacked__shrunken_se_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATC, stacked over five folds with q_bounds (-8, 8): the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8081 to 0.8367, SE ratio 0.6738 to 0.7109, empirical efficiency ratio 0.9850 to 1.0391, reported efficiency ratio 0.6992 to 0.7012 | pass |
| `interval_calibration` | `att_arm_mean_in_sample__correctly_specified` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the arm-mean ATT on the same draws, in sample: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9434 to 0.9595, SE ratio 0.9658 to 1.0194, empirical efficiency ratio 0.9788 to 1.0329, reported efficiency ratio 0.9965 to 0.9991 | pass |
| `interval_calibration` | `att_arm_mean_in_sample__noise_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the arm-mean ATT on the same draws, in sample: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8160 to 0.8441, SE ratio 0.6911 to 0.7286, empirical efficiency ratio 1.3693 to 1.4438, reported efficiency ratio 0.9966 to 0.9991 | pass |
| `interval_calibration` | `att_arm_mean_in_sample__shrunken_se_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the arm-mean ATT on the same draws, in sample: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8111 to 0.8395, SE ratio 0.6762 to 0.7135, empirical efficiency ratio 0.9790 to 1.0330, reported efficiency ratio 0.6976 to 0.6994 | pass |
| `interval_calibration` | `att_known_score_in_sample__correctly_specified` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATT, in sample: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9418 to 0.9581, SE ratio 0.9634 to 1.0161, empirical efficiency ratio 0.9814 to 1.0355, reported efficiency ratio 0.9961 to 0.9989 | pass |
| `interval_calibration` | `att_known_score_in_sample__noise_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATT, in sample: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8173 to 0.8453, SE ratio 0.6848 to 0.7204, empirical efficiency ratio 1.3845 to 1.4566, reported efficiency ratio 0.9961 to 0.9989 | pass |
| `interval_calibration` | `att_known_score_in_sample__shrunken_se_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATT, in sample: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8113 to 0.8397, SE ratio 0.6749 to 0.7114, empirical efficiency ratio 0.9818 to 1.0346, reported efficiency ratio 0.6973 to 0.6993 | pass |
| `interval_calibration` | `att_known_score_stacked__correctly_specified` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATT, stacked over five folds with q_bounds (-8, 8): both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9422 to 0.9585, SE ratio 0.9659 to 1.0174, empirical efficiency ratio 0.9832 to 1.0355, reported efficiency ratio 0.9987 to 1.0016 | pass |
| `interval_calibration` | `att_known_score_stacked__noise_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATT, stacked over five folds with q_bounds (-8, 8): one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8141 to 0.8423, SE ratio 0.6846 to 0.7209, empirical efficiency ratio 1.3874 to 1.4608, reported efficiency ratio 0.9987 to 1.0015 | pass |
| `interval_calibration` | `att_known_score_stacked__shrunken_se_control` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATT, stacked over five folds with q_bounds (-8, 8): the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8102 to 0.8387, SE ratio 0.6767 to 0.7131, empirical efficiency ratio 0.9819 to 1.0344, reported efficiency ratio 0.6991 to 0.7011 | pass |
| `known_score_accuracy` | `b2_atc__q_wrong` | positive | two-arm law B, n = 2,000, the known-score ATC: the declared mechanism with a wrong outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000439 to 0.0032, margin 0.0066, coverage 0.9339 to 0.9644, SE ratio 0.9994 | pass |
| `known_score_accuracy` | `b2_att__q_wrong` | positive | two-arm law B, n = 2,000, the known-score ATT: the declared mechanism with a wrong outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0014 to 0.0021, margin 0.0063, coverage 0.9363 to 0.9662, SE ratio 0.9980 | pass |
| `known_score_accuracy` | `b3_atc_1__q_wrong` | positive | three-arm law B, n = 2,000, the known-score ATC of arm 1: the declared mechanism with a wrong outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0014 to 0.0022, margin 0.0065, coverage 0.9339 to 0.9644, SE ratio 0.9963 | pass |
| `known_score_accuracy` | `b3_atc_2__q_wrong` | positive | three-arm law B, n = 2,000, the known-score ATC of arm 2: the declared mechanism with a wrong outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0022 to 0.0020, margin 0.0076, coverage 0.9225 to 0.9558, SE ratio 0.9895 | pass |
| `known_score_accuracy` | `b3_att_1__q_wrong` | positive | three-arm law B, n = 2,000, the known-score ATT of arm 1: the declared mechanism with a wrong outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012 to 0.0024, margin 0.0065, coverage 0.9355 to 0.9656, SE ratio 0.9931 | pass |
| `known_score_accuracy` | `b3_att_2__q_wrong` | positive | three-arm law B, n = 2,000, the known-score ATT of arm 2: the declared mechanism with a wrong outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0019 to 0.0023, margin 0.0076, coverage 0.9258 to 0.9583, SE ratio 0.9778 | pass |
| `known_score_accuracy` | `s_atc__q_wrong` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATC: the declared mechanism with a wrong outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0026 to 0.0117, margin 0.0260, coverage 0.9274 to 0.9595, SE ratio 0.9764 | pass |
| `known_score_accuracy` | `s_att__declared_constant` | control | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATT: the constant 0.49 declared as the mechanism, correct outcome regression | bias interval lies outside the margin | bias -0.6669 to -0.6587, margin 0.0148, coverage 0 to 0.0038, SE ratio 0.9435 | pass |
| `known_score_accuracy` | `s_att__q_wrong` | positive | law S (W1 uniform on (-sqrt 3, sqrt 3), W2 binary, g0 = expit(-0.3 + W1 + 0.5 W2), effect 1 + 1.5 W1, uniform unit-variance noise), n = 2,000, the known-score ATT: the declared mechanism with a wrong outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0063 to 0.0069, margin 0.0239, coverage 0.9169 to 0.9514, SE ratio 0.9657 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000698, coverage 0.9328 to 0.9619, SE ratio 0.9905 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0036, coverage 0.9279 to 0.9581, SE ratio 0.9873 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000239, coverage 0.9343 to 0.9630, SE ratio 0.9795 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5176 to -0.4718 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4993 to -0.4957 | pass |
<!-- /generated -->

The calibration cells read the efficiency bound of each form: the reported and the empirical
standard errors of the known-score fits are compared with the exact SD of $D_{\text{KS}}$, and
the arm-mean fits on the same draws with that of $D_{\text{AM}}$. The efficiency-gain cells read
the paired ratio of the two forms' spreads on those draws. Its exact value is 0.9014 for the ATT
and 0.9081 for the ATC, and a positive passes when the 99% upper bound of the ratio is below 0.98.
On law S0 the exact ratio is 1 and the controls must hold at the boundary. The boundary sits at
0.98 rather than 1 because on S0 the two forms differ by a term of order $1/n$ that the paired
interval would resolve.

`known_score_accuracy/s_att__declared_constant` is the gated control of the failure mode. A
declared mechanism that is wrong leaves the remainder whatever the outcome regression, and the
cell reads the bias the exact law predicts. The wrong-regression cells on law B measure bias and
coverage. They say nothing about efficiency, because on law B the gain is about 1% of the curve's
variance.

## Sizing

Every size was fixed before the run from the pilot in `tests/studies/known_score_att_pilot.py`,
on a stream no declared cell reads. Each sized instrument passes with at least its share of the
study-wide 0.95 in `tests/studies/repeated_sizing.py`. The calibration count, 4,800, is the
arm-pooled predictive rule at an 8,000-draw calibration pilot. The accuracy cells run 1,400, the
root-n rungs 1,600 and the primary phase 2,400. The design test recomputes each count and the
designed chance of a red cell, 0.025.

## Red-cell route

The study is registered under `reporting`, and no cell is red. The declared reading is in
`tests/canonical/known_score_att/regenerate.py`.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 2 | truth tests passing |
| `independent_tests_total` | 2 | truth tests reported |
| `property_cells_passed` | 36 | gated property cells passing |
| `property_cells_total` | 36 | gated property cells reported |
| `max_standardized_bias` | 0.0205 | largest primary standardized bias |
| `min_coverage` | 0.9479 | lowest primary coverage |
| `properties[interval_calibration/att_known_score_in_sample__correctly_specified]:se_ratio` | 0.9895 | SE ratio of the in-sample known-score ATT |
| `properties[interval_calibration/atc_known_score_in_sample__correctly_specified]:se_ratio` | 0.9864 | SE ratio of the in-sample known-score ATC |
| `properties[interval_calibration/att_known_score_stacked__correctly_specified]:se_ratio` | 0.9912 | SE ratio of the stacked known-score ATT |
| `properties[interval_calibration/atc_known_score_stacked__correctly_specified]:se_ratio` | 0.9879 | SE ratio of the stacked known-score ATC |
| `properties[efficiency_gain/att__known_score]:spread_ratio` | 0.9032 | paired spread ratio, known score over arm mean, ATT |
| `properties[efficiency_gain/atc__known_score]:spread_ratio` | 0.9088 | paired spread ratio, known score over arm mean, ATC |
| `properties[efficiency_gain/att_zero_gain__known_score]:spread_ratio` | 1.0001 | the same ratio on law S0, ATT |
| `properties[efficiency_gain/atc_zero_gain__known_score]:spread_ratio` | 0.9993 | the same ratio on law S0, ATC |
| `properties[known_score_accuracy/s_att__declared_constant]:standardized_bias` | -11.1829 | standardized bias of the declared-constant control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4946 | slope of the log empirical SD on log n |
| `properties[root_n_rate/reported_se]:slope` | -0.4975 | slope of the log reported SE on log n |
| `bound:att_standard_error` | 0.0643 | exact SD of the known-score curve, ATT (efficiency bound) |
| `bound:atc_standard_error` | 0.0648 | exact SD of the known-score curve, ATC (efficiency bound) |
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
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound on each calibration positive |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound on each calibration positive |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 24 | the same bound, in primary replications |

## Limitations

- **The declared mechanism is the whole claim.** The cells measure a mechanism declared by column
  and equal to the law's. A declared mechanism that is wrong biases the estimate whatever the
  outcome regression, as the control shows. With an estimated mechanism the option is refused.
- **Efficiency and calibration on one law.** Interval calibration and the efficiency gain were
  measured on the two-arm continuous law S with a correct outcome regression. The binary and
  three-arm cells measured bias and coverage with a wrong outcome regression only.
- **The population ATT.** The known-score form estimates the ATT of the population the declared
  mechanism describes. For the effect among the treated units in this sample, the arm-mean form is
  the natural estimator.
- **Families P3 only.** The `T55` rows cover the ATT and ATC. The option leaves every other
  estimand unchanged.
- **No comparator.** No maintained implementation reports this estimator with its own curve.
- **No weights, clusters, strata or missing outcomes.** Each is a separate composition and reads
  experimental, although the fast-tier exact-law tests cover the curve with weights, missing
  outcomes and strata.

## Reproduction

`tests/canonical/known_score_att/regenerate.py` regenerates the study and states the run form. It
runs no reference process, because the study has no comparator. The manifest records the seeds,
the margins, the estimator configuration and the source and result hashes, and the run log
records the commit, the clean tree and the wall time.
