# Longitudinal incremental propensity-score interventions under dropout

This study validates `IPSI` in `cleverly.longitudinal` with a censoring mechanism. `IPSI` is the
one-step estimator of Kim, Kennedy and Naimi (2021), Algorithm 1 and Theorem 5.1. The
[incremental interventions over time](../longitudinal-tmle.md#incremental-interventions-over-time)
section states the estimator, its conditions and its exact-law evidence.

No maintained package ships this construction with dropout.
So no canonical implementation is compared. The study rests on exact truths and on
repeated-sampling properties. The [complete-data study](longitudinal-incremental-interventions.md)
pairs the same estimator with R `npcausal`, where that package applies. Its comparator runs are
pending.

## What was fitted

| scenario | law and estimands | folds |
| --- | --- | --- |
| `dropout_folds_1` | law `C`: three nodes with a continuous end-of-study outcome and dropout after each arm. The three plan means and their two contrasts with the natural course | one |
| `dropout_folds_5` | law `C`, the same estimands | five |
| `survival_folds_1` | law `D`: three nodes with a binary absorbing event and the same dropout. The risk of each plan by horizons 1, 2 and 3, and the two contrasts at each horizon | one |
| `survival_folds_5` | law `D`, the same estimands | five |

Three plans tilt every node by 0.5, 1 (the natural course) and 2. The laws are in
`tests/discrete_law_longitudinal_incremental.py`. Law `C` is law `A` of the
[complete-data study](longitudinal-incremental-interventions.md) with dropout at random given the
history. The dropout probabilities $\omega_t$ lie in [0.85, 0.97]. Law `D` has event hazards in
[0.05, 0.30]. Every learner is a saturated cell mean.

The one-fold scenarios are primary rows. A one-fold fit with saturated cell means is the plug-in
g-formula on the empirical cell frequencies. Its tilt is nonlinear in $\hat\pi$, so it carries a
bias of order (cells)/n. The conditions for the one-fold interval are in the table below.

| source | what it states |
| --- | --- |
| Kennedy (2019), Algorithm 2 (Appendix 8.3) | the one-fold interval, under Donsker conditions |
| Kim et al. (2021), Theorem 5.1 | the dropout result, with sample splitting only |
| the argument of this study | saturated cell means on a finite history set form a finite-dimensional class. The remainder is a product of $O_P(n^{-1/2})$ errors. The Donsker argument therefore extends the dropout result to one fold |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-node law with dropout after each arm, one fold | `ate_regimen[odds x0.5 vs natural course]` | difference in mean outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | `cleverly` `IPSI` with a censoring mechanism | -0.000084 to 0.000487 | 0.9400 | 0.9614 | pass |
| three-node law with dropout after each arm, one fold | `ate_regimen[odds x2 vs natural course]` | difference in mean outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | `cleverly` `IPSI` with a censoring mechanism | -0.000472 to 0.000114 | 0.9510 | 0.9871 | pass |
| three-node law with dropout after each arm, one fold | `ey_regimen[natural course]` | mean outcome under the plan leave the observed treatment mechanism unchanged | `cleverly` `IPSI` with a censoring mechanism | -0.000933 to 0.000541 | 0.9490 | 0.9874 | pass |
| three-node law with dropout after each arm, one fold | `ey_regimen[odds x0.5]` | mean outcome under the plan multiply the treatment odds by 0.5 | `cleverly` `IPSI` with a censoring mechanism | -0.000717 to 0.000727 | 0.9540 | 0.9974 | pass |
| three-node law with dropout after each arm, one fold | `ey_regimen[odds x2]` | mean outcome under the plan multiply the treatment odds by two | `cleverly` `IPSI` with a censoring mechanism | -0.0011 to 0.000356 | 0.9440 | 0.9807 | pass |
| three-node law with dropout after each arm, five folds | `ate_regimen[odds x0.5 vs natural course]` | difference in mean outcome between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" | `cleverly` `IPSI` with a censoring mechanism | -0.000522 to 0.000020 | 0.9530 | 1.0133 | pass |
| three-node law with dropout after each arm, five folds | `ate_regimen[odds x2 vs natural course]` | difference in mean outcome between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" | `cleverly` `IPSI` with a censoring mechanism | 0.000009 to 0.000579 | 0.9420 | 1.0174 | pass |
| three-node law with dropout after each arm, five folds | `ey_regimen[natural course]` | mean outcome under the plan leave the observed treatment mechanism unchanged | `cleverly` `IPSI` with a censoring mechanism | -0.000723 to 0.000738 | 0.9500 | 0.9964 | pass |
| three-node law with dropout after each arm, five folds | `ey_regimen[odds x0.5]` | mean outcome under the plan multiply the treatment odds by 0.5 | `cleverly` `IPSI` with a censoring mechanism | -0.000970 to 0.000483 | 0.9450 | 0.9911 | pass |
| three-node law with dropout after each arm, five folds | `ey_regimen[odds x2]` | mean outcome under the plan multiply the treatment odds by two | `cleverly` `IPSI` with a censoring mechanism | -0.000415 to 0.0010 | 0.9490 | 1.0014 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `ate_regimen[odds x0.5 vs natural course @ t=1]` | difference in cumulative risk between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000030 to 0.000029 | 0.9430 | 0.9916 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `ate_regimen[odds x0.5 vs natural course @ t=2]` | difference in cumulative risk between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000071 to 0.000043 | 0.9350 | 0.9669 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `ate_regimen[odds x0.5 vs natural course @ t=3]` | difference in cumulative risk between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000096 to 0.000054 | 0.9440 | 0.9815 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `ate_regimen[odds x2 vs natural course @ t=1]` | difference in cumulative risk between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000036 to 0.000039 | 0.9420 | 0.9895 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `ate_regimen[odds x2 vs natural course @ t=2]` | difference in cumulative risk between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000037 to 0.000092 | 0.9440 | 0.9643 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `ate_regimen[odds x2 vs natural course @ t=3]` | difference in cumulative risk between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000051 to 0.000111 | 0.9430 | 0.9678 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[natural course @ t=1]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000094 to 0.000134 | 0.9500 | 0.9753 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[natural course @ t=2]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000127 to 0.000158 | 0.9480 | 1.0236 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[natural course @ t=3]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000141 to 0.000172 | 0.9550 | 0.9977 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[odds x0.5 @ t=1]` | cumulative risk under the plan multiply the treatment odds by 0.5 at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000105 to 0.000143 | 0.9440 | 0.9568 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[odds x0.5 @ t=2]` | cumulative risk under the plan multiply the treatment odds by 0.5 at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000156 to 0.000160 | 0.9510 | 1.0062 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[odds x0.5 @ t=3]` | cumulative risk under the plan multiply the treatment odds by 0.5 at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000182 to 0.000172 | 0.9500 | 0.9876 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[odds x2 @ t=1]` | cumulative risk under the plan multiply the treatment odds by two at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000090 to 0.000133 | 0.9590 | 1.0065 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[odds x2 @ t=2]` | cumulative risk under the plan multiply the treatment odds by two at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000108 to 0.000195 | 0.9550 | 1.0232 | pass |
| three-node survival law with dropout, horizons 1 to 3, one fold | `risk_regimen[odds x2 @ t=3]` | cumulative risk under the plan multiply the treatment odds by two at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000126 to 0.000218 | 0.9510 | 1.0023 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `ate_regimen[odds x0.5 vs natural course @ t=1]` | difference in cumulative risk between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000043 to 0.000016 | 0.9500 | 0.9960 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `ate_regimen[odds x0.5 vs natural course @ t=2]` | difference in cumulative risk between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000068 to 0.000046 | 0.9420 | 0.9661 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `ate_regimen[odds x0.5 vs natural course @ t=3]` | difference in cumulative risk between the plans "multiply the treatment odds by 0.5" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000059 to 0.000092 | 0.9430 | 0.9732 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `ate_regimen[odds x2 vs natural course @ t=1]` | difference in cumulative risk between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000019 to 0.000055 | 0.9490 | 0.9975 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `ate_regimen[odds x2 vs natural course @ t=2]` | difference in cumulative risk between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000056 to 0.000071 | 0.9420 | 0.9848 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `ate_regimen[odds x2 vs natural course @ t=3]` | difference in cumulative risk between the plans "multiply the treatment odds by two" against "leave the observed treatment mechanism unchanged" at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000079 to 0.000080 | 0.9440 | 0.9871 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[natural course @ t=1]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000065 to 0.000155 | 0.9480 | 1.0087 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[natural course @ t=2]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000139 to 0.000157 | 0.9480 | 0.9873 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[natural course @ t=3]` | cumulative risk under the plan leave the observed treatment mechanism unchanged at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000170 to 0.000146 | 0.9490 | 0.9900 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[odds x0.5 @ t=1]` | cumulative risk under the plan multiply the treatment odds by 0.5 at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000085 to 0.000148 | 0.9490 | 1.0181 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[odds x0.5 @ t=2]` | cumulative risk under the plan multiply the treatment odds by 0.5 at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000163 to 0.000158 | 0.9530 | 0.9927 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[odds x0.5 @ t=3]` | cumulative risk under the plan multiply the treatment odds by 0.5 at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000175 to 0.000184 | 0.9500 | 0.9731 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[odds x2 @ t=1]` | cumulative risk under the plan multiply the treatment odds by two at horizon t = 1 | `cleverly` `IPSI` with a censoring mechanism | -0.000050 to 0.000176 | 0.9480 | 0.9933 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[odds x2 @ t=2]` | cumulative risk under the plan multiply the treatment odds by two at horizon t = 2 | `cleverly` `IPSI` with a censoring mechanism | -0.000143 to 0.000175 | 0.9440 | 0.9753 | pass |
| three-node survival law with dropout, horizons 1 to 3, five folds | `risk_regimen[odds x2 @ t=3]` | cumulative risk under the plan multiply the treatment odds by two at horizon t = 3 | `cleverly` `IPSI` with a censoring mechanism | -0.000185 to 0.000161 | 0.9500 | 0.9963 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `censoring_complete_case` | `mean_x2__complete_case` | diagnostic | odds-x2 mean at every node, three-node law: a fit to the rows that never drop out, without a censoring model, beside the exact-law limit | reported only | bias 0.0705 to 0.0716, margin 0.0029, coverage 0.0000 | reported |
| `clustered_inference` | `mean_x2__cluster_robust` | positive | odds-x2 mean at every node, three-node law: five-fold point-treatment TMLE with cluster-robust ATE inference | SE-ratio and coverage intervals both stay inside their calibration bands | coverage 0.9451 to 0.9648, SE ratio 0.9809 to 1.0465, paired coverage gain 0.1530 to 0.1887 | pass |
| `clustered_inference` | `mean_x2__iid_control` | control | odds-x2 mean at every node, three-node law: the identical rows, point estimates, and influence curves treated as independent | the SE-ratio upper endpoint must not exceed the declared IID-control ceiling | coverage 0.7654 to 0.8043, SE ratio 0.6212 to 0.6625, paired coverage gain 0.1530 to 0.1887 | pass |
| `interval_calibration` | `competing_contrast_x2_h3__correctly_specified` | positive | odds-x2 contrast in the cause-1 incidence by t = 3: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9491 to 0.9681, SE ratio 1.0124 to 1.0830, empirical efficiency ratio 0.9239 to 0.9884, reported efficiency ratio 1.0005 to 1.0007 | **fail** |
| `interval_calibration` | `competing_contrast_x2_h3__noise_control` | control | odds-x2 contrast in the cause-1 incidence by t = 3: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8328 to 0.8667, SE ratio 0.6925 to 0.7421, empirical efficiency ratio 1.3483 to 1.4449, reported efficiency ratio 1.0005 to 1.0007 | pass |
| `interval_calibration` | `competing_contrast_x2_h3__shrunken_se_control` | control | odds-x2 contrast in the cause-1 incidence by t = 3: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8259 to 0.8604, SE ratio 0.7091 to 0.7584, empirical efficiency ratio 0.9235 to 0.9877, reported efficiency ratio 0.7003 to 0.7005 | pass |
| `interval_calibration` | `dropout_contrast_x2__correctly_specified` | positive | odds-x2 contrast at the end of study, with dropout: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9407 to 0.9612, SE ratio 0.9876 to 1.0548, empirical efficiency ratio 0.9489 to 1.0135, reported efficiency ratio 1.0007 to 1.0012 | pass |
| `interval_calibration` | `dropout_contrast_x2__noise_control` | control | odds-x2 contrast at the end of study, with dropout: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8141 to 0.8495, SE ratio 0.6886 to 0.7342, empirical efficiency ratio 1.3634 to 1.4534, reported efficiency ratio 1.0007 to 1.0012 | pass |
| `interval_calibration` | `dropout_contrast_x2__shrunken_se_control` | control | odds-x2 contrast at the end of study, with dropout: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8210 to 0.8559, SE ratio 0.6914 to 0.7379, empirical efficiency ratio 0.9496 to 1.0133, reported efficiency ratio 0.7005 to 0.7008 | pass |
| `interval_calibration` | `dropout_contrast_x2_in_sample__correctly_specified` | positive | odds-x2 contrast at the end of study, with dropout, one fold: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9388 to 0.9597, SE ratio 0.9877 to 1.0566, empirical efficiency ratio 0.9467 to 1.0126, reported efficiency ratio 0.9999 to 1.0004 | pass |
| `interval_calibration` | `dropout_contrast_x2_in_sample__noise_control` | control | odds-x2 contrast at the end of study, with dropout, one fold: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8182 to 0.8534, SE ratio 0.6880 to 0.7354, empirical efficiency ratio 1.3600 to 1.4537, reported efficiency ratio 0.9999 to 1.0004 | pass |
| `interval_calibration` | `dropout_contrast_x2_in_sample__shrunken_se_control` | control | odds-x2 contrast at the end of study, with dropout, one fold: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8186 to 0.8537, SE ratio 0.6923 to 0.7392, empirical efficiency ratio 0.9471 to 1.0112, reported efficiency ratio 0.7000 to 0.7003 | pass |
| `interval_calibration` | `msm_log_delta__correctly_specified` | positive | slope of the identity-link working model in log odds multiplier: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9451 to 0.9648, SE ratio 0.9792 to 1.0457, empirical efficiency ratio 0.9573 to 1.0222, reported efficiency ratio 1.0007 to 1.0012 | pass |
| `interval_calibration` | `msm_log_delta__noise_control` | control | slope of the identity-link working model in log odds multiplier: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8269 to 0.8613, SE ratio 0.6966 to 0.7467, empirical efficiency ratio 1.3405 to 1.4368, reported efficiency ratio 1.0007 to 1.0012 | pass |
| `interval_calibration` | `msm_log_delta__shrunken_se_control` | control | slope of the identity-link working model in log odds multiplier: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8207 to 0.8556, SE ratio 0.6848 to 0.7325, empirical efficiency ratio 0.9566 to 1.0232, reported efficiency ratio 0.7005 to 0.7009 | pass |
| `interval_calibration` | `survival_contrast_x2_h3__correctly_specified` | positive | odds-x2 contrast in the risk by t = 3, with dropout: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9429 to 0.9630, SE ratio 0.9881 to 1.0552, empirical efficiency ratio 0.9486 to 1.0131, reported efficiency ratio 1.0009 to 1.0012 | pass |
| `interval_calibration` | `survival_contrast_x2_h3__noise_control` | control | odds-x2 contrast in the risk by t = 3, with dropout: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8127 to 0.8483, SE ratio 0.6969 to 0.7437, empirical efficiency ratio 1.3461 to 1.4366, reported efficiency ratio 1.0009 to 1.0012 | pass |
| `interval_calibration` | `survival_contrast_x2_h3__shrunken_se_control` | control | odds-x2 contrast in the risk by t = 3, with dropout: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8168 to 0.8521, SE ratio 0.6918 to 0.7384, empirical efficiency ratio 0.9490 to 1.0128, reported efficiency ratio 0.7006 to 0.7008 | pass |
| `interval_calibration` | `survival_contrast_x2_h3_in_sample__correctly_specified` | positive | odds-x2 contrast in the risk by t = 3, with dropout, one fold: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9436 to 0.9636, SE ratio 0.9869 to 1.0539, empirical efficiency ratio 0.9484 to 1.0128, reported efficiency ratio 0.9993 to 0.9996 | pass |
| `interval_calibration` | `survival_contrast_x2_h3_in_sample__noise_control` | control | odds-x2 contrast in the risk by t = 3, with dropout, one fold: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8103 to 0.8460, SE ratio 0.6819 to 0.7277, empirical efficiency ratio 1.3735 to 1.4659, reported efficiency ratio 0.9993 to 0.9996 | pass |
| `interval_calibration` | `survival_contrast_x2_h3_in_sample__shrunken_se_control` | control | odds-x2 contrast in the risk by t = 3, with dropout, one fold: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8137 to 0.8492, SE ratio 0.6916 to 0.7377, empirical efficiency ratio 0.9483 to 1.0117, reported efficiency ratio 0.6995 to 0.6997 | pass |
| `interval_calibration` | `weighted_contrast_x0_5__correctly_specified` | positive | odds-x0.5 contrast on the law weighted by 1 + A1 + W/2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9425 to 0.9627, SE ratio 0.9666 to 1.0318, empirical efficiency ratio 0.9699 to 1.0352, reported efficiency ratio 1.0005 to 1.0009 | pass |
| `interval_calibration` | `weighted_contrast_x0_5__noise_control` | control | odds-x0.5 contrast on the law weighted by 1 + A1 + W/2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8186 to 0.8537, SE ratio 0.6963 to 0.7436, empirical efficiency ratio 1.3456 to 1.4369, reported efficiency ratio 1.0004 to 1.0009 | pass |
| `interval_calibration` | `weighted_contrast_x0_5__shrunken_se_control` | control | odds-x0.5 contrast on the law weighted by 1 + A1 + W/2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8127 to 0.8483, SE ratio 0.6774 to 0.7230, empirical efficiency ratio 0.9688 to 1.0340, reported efficiency ratio 0.7003 to 0.7007 | pass |
| `simultaneous_coverage` | `survival_grid__pointwise_joint_control` | control | three plans by three horizons, the risks and six contrasts: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6746 to 0.7181 | pass |
| `simultaneous_coverage` | `survival_grid__simultaneous_band` | positive | three plans by three horizons, the risks and six contrasts: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9370 to 0.9582 | pass |
<!-- /generated -->

One cell is red. `interval_calibration/competing_contrast_x2_h3__correctly_specified` fails its
SE-ratio band on the conservative side: the interval 1.0124 to 1.0830 crosses the upper bound of
1.07. Its coverage interval, 0.9491 to 0.9681, stays inside the band. The reported efficiency ratio
is 1.0006, so the reported standard error matches the exact efficient influence-function standard
deviation. The empirical spread sits 4.4% below that bound. A regular estimator does not stay below
the bound as n grows, so the diagnosis reads a finite-sample or Monte Carlo excursion and finds no
defect. The design probes on other draws read SE ratios of 0.948 at n = 32,000 and 0.980 at
n = 64,000 (`tests/diagnostics/x19_sizing/reading.csv`).

By the declared route the study moved to `reporting`. The roadmap row [`X19-competing-calibration`](../../roadmap.md#red-cell-owners) owns
the cell. No fit was repeated: `regenerate.py --reuse-subject --skip-properties` summarized the
committed rows again under the new policy. No property fit failed.

Every property fit uses five folds, saturated cell means at every slot and 3,000 draws. The
default size is n = 48,000. `x2` is the plan that doubles the odds of treatment at every node.
`contrast_x2` is its contrast with the natural course.
`tests/studies/longitudinal_incremental_dropout_properties.py` declares every cell.

| family | cells and fits | what must hold |
| --- | --- | --- |
| `interval_calibration` | seven labels with three cells each: the correctly specified fit and two derived controls | SE ratio and coverage intervals inside the calibration bands, and both efficiency-ratio intervals inside their band. The shrunken-SE control falls below the band. The noise control rises above the efficiency-ratio band |
| `simultaneous_coverage` | the band over the risks of three plans at horizons 1 to 3 and their six contrasts, on law `D` at n = 48,000, and its pointwise control | the band's joint coverage interval lies inside the calibration coverage band. The pointwise control's joint coverage interval lies below the nominal rate |
| `clustered_inference` | `ey_regimen[odds x2]` on law `Ck` in 800 clusters of 10 (n = 8,000): the cluster-robust standard error, and the standard error that treats each unit as independent (the control) | the cluster-robust arm is calibrated. The control's SE-ratio interval lies below 0.80. The coverage gain of the pair is at least 0.03 |
| `censoring_complete_case` | `ey_regimen[odds x2]` on its own draws of law `C`, fitted to the rows that never drop out, with no censoring model. 3,000 draws at n = 48,000 | no verdict. The cell prints its bias beside the exact complete-case limit (`complete_case_limit`) |

The calibration labels:

| label | law and estimand | n |
| --- | --- | --- |
| `dropout_contrast_x2` | `C`, `contrast_x2` at the end of study | 48,000 |
| `survival_contrast_x2_h3` | `D`, `contrast_x2` of the risk by horizon 3, read off the band fit | 48,000 |
| `dropout_contrast_x2_in_sample` | `C`, `contrast_x2`, one fold (Kennedy 2019, Algorithm 2) | 48,000 |
| `survival_contrast_x2_h3_in_sample` | `D`, `contrast_x2` of the risk by horizon 3, one fold, on the band draws | 48,000 |
| `competing_contrast_x2_h3` | `Dc`, `contrast_x2` of the cause-1 incidence by horizon 3 | 96,000 |
| `msm_log_delta` | `C`, the slope of an identity-link working model linear in $\log\delta$ over the three plans | 48,000 |
| `weighted_contrast_x0_5` | `C` with the informative weight $1 + A_1 + W/2$, `odds x0.5 vs natural course`. The truth is the weighted law's | 48,000 |

Law `Dc` adds a second, competing cause at each node. Law `Ck` is law `C` with a shared binary
cluster effect. The one-fold cells are gated like the cross-fitted ones. The
`censoring_complete_case` cell shows why the weights matter: a fit that ignores dropout sits away
from the target, and the cell reports how far.

## Sizing

The five-fold contrasts carry a second-order bias of order 1/n, so their standardized bias falls
as $1/\sqrt n$. The sizing probe in `tests/diagnostics/x19_sizing/` measured the bias at
n = 8,000. The size rule requires that every primary row passes the bias rule with probability at
least 0.95 at the projected bias. It requires 0.90 for each calibration cell.

| sizing input | value | source |
| --- | --- | --- |
| standardized bias at n = 8,000, dropout contrasts | 0.31 | `tests/studies/canonical_longitudinal_incremental_dropout.py` |
| one-fold standardized bias at n = 8,000 | about 0.2 | the same |
| primary n | 80,000 | the same |
| calibration replications | 3,000. At 2,000 an ideal cell passes with probability 0.935, and at 3,000 with 0.996 | `tests/studies/longitudinal_incremental_dropout_properties.py` |
| calibration n | 48,000. The calibration cells pass jointly with probability 0.93 | the same, with `tests/unit/test_longitudinal_incremental_design.py` |
| competing-risk n | 96,000. At n = 8,000 a training fold can hold one or two treated units of a rare history, and the censoring fit reaches its lower bound. The SE ratio reads 0.948 at 32,000 and 0.980 at 64,000 | `tests/studies/longitudinal_incremental_dropout_properties.py` |
| clustered cell | n = 8,000 and 3,000 draws. The calibration rule cannot pass at 1,000 | the same |
| weighted truth | the informative weight moves the truth by 9.5 standard errors at n = 48,000 | `tests/unit/test_longitudinal_incremental_design.py` |
| independent-unit control | the exact SE ratio is 0.60 and the exact coverage is 0.76 | the same |
| complete-case limit | 6.2 standard errors from the truth | the same |
| band control | the pointwise intervals cover jointly with exact probability 0.695. The control clears the control-power rule at that value plus 0.005 | the same |

The fallback-read bias bound holds far below these sizes. The study fixed every size,
replication count, margin, law, learner, fold count and seed before any run.

## Red-cell route

| event | route, declared before any run |
| --- | --- |
| a red primary truth row | blocks the merge and gets a diagnosis |
| a red property cell that reads as a finite-sample limit | is re-registered `reporting` with an owner row before any re-run |
| a replication that raises | is never redrawn. The failing cell drops to its owner with its failure count stated as a page limit, and the run repeats without it |
| any verdict | moves no budget, margin, law, learner, fold count or seed. Controls carry the verdict their family states |

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
exact standard errors of the efficient influence curve at the primary n. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 80000 | observations per primary replication |
| `independent_tests_passed` | 40 | truth tests passing |
| `independent_tests_total` | 40 | truth tests reported |
| `property_cells_passed` | 24 | property cells passing |
| `property_cells_total` | 25 | property cells reported |
| `max_standardized_bias` | 0.0843 | largest primary standardized bias |
| `min_coverage` | 0.9350 | lowest primary coverage |
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
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:iid_control_se_ceiling` | 0.8000 | independent-unit control SE-ratio ceiling |
| `margin:clustered_coverage_gain` | 0.0300 | paired coverage-gain floor |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `bound:ate_regimen[odds x0.5 vs natural course @ t=1]_standard_error` | 0.000359 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x0.5 vs natural course @ t=2]_standard_error` | 0.000675 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x0.5 vs natural course @ t=3]_standard_error` | 0.000901 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x0.5 vs natural course]_standard_error` | 0.0034 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x2 vs natural course @ t=1]_standard_error` | 0.000451 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x2 vs natural course @ t=2]_standard_error` | 0.000762 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x2 vs natural course @ t=3]_standard_error` | 0.000962 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ate_regimen[odds x2 vs natural course]_standard_error` | 0.0035 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ey_regimen[natural course]_standard_error` | 0.0089 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ey_regimen[odds x0.5]_standard_error` | 0.0088 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:ey_regimen[odds x2]_standard_error` | 0.0088 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[natural course @ t=1]_standard_error` | 0.0014 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[natural course @ t=2]_standard_error` | 0.0018 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[natural course @ t=3]_standard_error` | 0.0019 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x0.5 @ t=1]_standard_error` | 0.0015 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x0.5 @ t=2]_standard_error` | 0.0020 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x0.5 @ t=3]_standard_error` | 0.0021 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x2 @ t=1]_standard_error` | 0.0014 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x2 @ t=2]_standard_error` | 0.0019 | exact influence-curve standard error of this estimand at the primary n = 80,000 |
| `bound:risk_regimen[odds x2 @ t=3]_standard_error` | 0.0021 | exact influence-curve standard error of this estimand at the primary n = 80,000 |

## Limits

| limit | what it means for use |
| --- | --- |
| dropout at random given the history, after each arm | the laws set $\omega_t$ in [0.85, 0.97]. Heavier dropout and informative dropout are outside this evidence |
| saturated cell means at every slot | the study does not test a flexible learner. The one-fold claim rests on the finite-dimensional class |
| one-fold primary rows rest on an argument | Kim et al. (2021) state the dropout result with sample splitting only. The Donsker argument above is the extension, and no source states it |
| finite-sample bias is sized, not removed | at n = 8,000 the standardized bias reads 0.31 on the dropout contrasts. The rows pass at the sizes above, and smaller sizes are not claimed |
| three nodes and horizons 1 to 3 | longer panels and later horizons are not claimed |
| no comparator | no pinned implementation fits this design, so the study has no paired verdict |
| the clustered cell reads one mean | `ey_regimen[odds x2]` on law `Ck` in 800 clusters of 10. Fewer clusters are unmeasured |
| the complete-case cell states no verdict | it reports a bias beside the exact limit |

## Reproduction

`tests/canonical/longitudinal_incremental_dropout/regenerate.py` regenerates the study. It runs no
reference process, because the study has no comparator.
