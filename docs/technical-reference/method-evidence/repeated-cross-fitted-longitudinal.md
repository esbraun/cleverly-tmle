# Repeated cross-fitted longitudinal TMLE, SDR and IPSI

This study measures the repeated cross-fitted report of the longitudinal estimators. A fit with
`repeats=3` draws three outer splits and fits each draw completely. It reports the median of the
draw estimates and the split-adjusted median variance.

For draw-specific points $\hat\psi_r$ and variances $\hat\sigma_r^2$, the report is

$$
\widetilde\psi = \operatorname{median}_r(\hat\psi_r), \qquad
\widetilde\sigma^2 = \operatorname{median}_r\left\{
\hat\sigma_r^2 + (\hat\psi_r - \widetilde\psi)^2
\right\}.
$$

[Chernozhukov et al. (2018), Definition 3.3 and equation
(3.14)](https://academic.oup.com/ectj/article/21/1/C1/5056401) supply this fixed-repeat rule.

**No canonical implementation is compared.** The pinned `lmtp` and `npcausal` adapters accept a
supplied fold list. Their parity for one draw is the evidence of the single-draw rows. Neither
adapter takes a repeat option. A run of either one per draw, with this package's median applied
after it, compares the aggregation with itself. The zero-row equivalence artifact records the
absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| construction | five outer folds, three complete fold draws, two learner folds |
| nuisances | saturated cell means in every slot, refitted in every fold of every draw |
| bounds | `g_bounds` of 0.01 to 1; the smallest true cumulative product is 0.0352 on law D and 0.0401 on laws S and C |
| variance | median within-draw variance plus squared split displacement |
| law D | two censored decisions; `LTMLE` and `SDR`; regimens `never`, `always` and `treat_if_l2`; n = 8,000 |
| law I | incremental law with three nodes; `IPSI` at odds multipliers 0.5, 1 and 2; n = 80,000 |
| law S | survival over three nodes; `LTMLE` risks and risk differences at horizons 1 to 3; n = 8,000 |
| law C | clusters of 10 with `id=`; `LTMLE` static means and their contrast; n = 8,000 |
| law M | law D with an identity working model; the duration term is centred at 1.2304; n = 8,000 |
| Monte Carlo inference | 99% intervals around all declared endpoints |

The design test `tests/unit/test_longitudinal_repeated_crossfit_design.py` states every size and
every pass probability. It reads them from the sizing probe in `tests/diagnostics/x46_sizing/`.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clusters of ten units on a two-node law, `LTMLE` with `id=`, five folds and three fold draws | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0055 to 0.0029 | 0.9613 | 1.0463 | pass |
| clusters of ten units on a two-node law, `LTMLE` with `id=`, five folds and three fold draws | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0057 to 0.0025 | 0.9500 | 1.0122 | pass |
| clusters of ten units on a two-node law, `LTMLE` with `id=`, five folds and three fold draws | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0037 to 0.0031 | 0.9519 | 1.0162 | pass |
| two-node censored law, `LTMLE` with five folds, three fold draws and saturated cell means | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0022 to 0.0014 | 0.9563 | 1.0316 | pass |
| two-node censored law, `LTMLE` with five folds, three fold draws and saturated cell means | `ate_regimen[treat_if_l2 vs never]` | difference in mean outcome between the plans "treat first, then treat at the second time only where L2 equals one" against "treat at neither time" | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0012 to 0.000451 | 0.9556 | 0.9887 | pass |
| two-node censored law, `LTMLE` with five folds, three fold draws and saturated cell means | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0016 to 0.0015 | 0.9625 | 1.0492 | pass |
| two-node censored law, `LTMLE` with five folds, three fold draws and saturated cell means | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000697 to 0.0013 | 0.9531 | 1.0100 | pass |
| two-node censored law, `LTMLE` with five folds, three fold draws and saturated cell means | `ey_regimen[treat_if_l2]` | mean outcome under the plan treat first, then treat at the second time only where L2 equals one | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000958 to 0.000843 | 0.9500 | 1.0091 | pass |
| two-node censored law, `LTMLE` with five folds, three fold draws and saturated cell means | `rr_regimen[always vs never]` | risk ratio, with inference on the log scale, between the plans "treat at both times" against "treat at neither time" | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0042 to 0.0023 | 0.9594 | 1.0259 | pass |
| the same two-node censored law, `SDR` with five folds, three fold draws and saturated cell means | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0022 to 0.0014 | 0.9563 | 1.0317 | pass |
| the same two-node censored law, `SDR` with five folds, three fold draws and saturated cell means | `ate_regimen[treat_if_l2 vs never]` | difference in mean outcome between the plans "treat first, then treat at the second time only where L2 equals one" against "treat at neither time" | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0012 to 0.000451 | 0.9556 | 0.9887 | pass |
| the same two-node censored law, `SDR` with five folds, three fold draws and saturated cell means | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0016 to 0.0015 | 0.9625 | 1.0493 | pass |
| the same two-node censored law, `SDR` with five folds, three fold draws and saturated cell means | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000697 to 0.0013 | 0.9531 | 1.0100 | pass |
| the same two-node censored law, `SDR` with five folds, three fold draws and saturated cell means | `ey_regimen[treat_if_l2]` | mean outcome under the plan treat first, then treat at the second time only where L2 equals one | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000958 to 0.000843 | 0.9500 | 1.0091 | pass |
| the same two-node censored law, `SDR` with five folds, three fold draws and saturated cell means | `rr_regimen[always vs never]` | risk ratio, with inference on the log scale, between the plans "treat at both times" against "treat at neither time" | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0042 to 0.0023 | 0.9594 | 1.0260 | pass |
| the two-node law with an identity working model on treatment duration, `LTMLE`, five folds and three fold draws | `msm_regimen[(intercept)]` | longitudinal regimen MSM projection intercept coefficient | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0013 to 0.000293 | 0.9513 | 0.9980 | pass |
| the two-node law with an identity working model on treatment duration, `LTMLE`, five folds and three fold draws | `msm_regimen[duration]` | longitudinal regimen MSM projection treatment-duration coefficient | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0019 to 0.0016 | 0.9469 | 1.0105 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000752 to 0.000395 | 0.9469 | 0.9944 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000922 to 0.000973 | 0.9494 | 0.9843 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `ate_regimen[always vs never @ t=3]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 3 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0016 to 0.000871 | 0.9537 | 1.0176 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000616 to 0.000277 | 0.9537 | 1.0025 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000733 to 0.000718 | 0.9463 | 1.0174 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `risk_regimen[always @ t=3]` | cumulative risk under the plan treat at both times at horizon t = 3 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.0012 to 0.000700 | 0.9513 | 1.0243 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000346 to 0.000360 | 0.9519 | 1.0043 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000615 to 0.000552 | 0.9437 | 0.9801 | pass |
| three-node survival law with an absorbing event, `LTMLE` with five folds and three fold draws | `risk_regimen[never @ t=3]` | cumulative risk under the plan treat at neither time at horizon t = 3 | `cleverly` repeated cross-fitted longitudinal fit, three outer splits | -0.000696 to 0.000865 | 0.9519 | 1.0091 | pass |
<!-- /generated -->

## Theory properties

| family | what it measures |
| --- | --- |
| `interval_calibration` | the SE ratio and coverage of nine contrasts at 24,000 replications, each with a control that multiplies the reported SE by 0.70 |
| `simultaneous_coverage` | the joint coverage of the shipped band on each law, with a pointwise control |
| `repeat_increment` | the median SE ratio minus the first draw's SE ratio on law S; horizon 3 is gated at 0.03, with a control that multiplies the median SE by 1.05 |
| `repeat_stability` | the spread of the three-draw median over the spread of its first draw, on one fixed sample per law |
| `root_n_and_efficiency` and `root_n_rate` | the law-D static contrast at n = 2,000, 8,000 and 32,000; the slope rows carry the check, because the 2,000 rung cannot fail at 800 replications |
| `ipsi_primary` | law I's means and contrasts at n = 80,000, read by the primary rules |
| `single_draw_reading` and `split_displacement` | the first draw's calibration and the displacement share, reported only |

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `c_static__correctly_specified` | positive | `LTMLE` difference of always against never on the clustered law; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9494 to 0.9565, SE ratio 1.0046 to 1.0279 | pass |
| `interval_calibration` | `c_static__shrunken_se_control` | control | `LTMLE` difference of always against never on the clustered law, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8267 to 0.8391, SE ratio 0.7030 to 0.7200 | pass |
| `interval_calibration` | `d_ltmle_dynamic__correctly_specified` | positive | `LTMLE` difference of treat-if-L2 against never on the two-node law; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9464 to 0.9537, SE ratio 0.9992 to 1.0231 | pass |
| `interval_calibration` | `d_ltmle_dynamic__shrunken_se_control` | control | `LTMLE` difference of treat-if-L2 against never on the two-node law, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8300 to 0.8423, SE ratio 0.6997 to 0.7162 | pass |
| `interval_calibration` | `d_ltmle_static__correctly_specified` | positive | `LTMLE` risk difference of always against never on the two-node law; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9475 to 0.9547, SE ratio 0.9985 to 1.0221 | pass |
| `interval_calibration` | `d_ltmle_static__shrunken_se_control` | control | `LTMLE` risk difference of always against never on the two-node law, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8301 to 0.8424, SE ratio 0.6988 to 0.7156 | pass |
| `interval_calibration` | `d_sdr_dynamic__correctly_specified` | positive | `SDR` difference of treat-if-L2 against never on the two-node law; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9464 to 0.9537, SE ratio 0.9998 to 1.0235 | pass |
| `interval_calibration` | `d_sdr_dynamic__shrunken_se_control` | control | `SDR` difference of treat-if-L2 against never on the two-node law, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8299 to 0.8422, SE ratio 0.6994 to 0.7162 | pass |
| `interval_calibration` | `d_sdr_static__correctly_specified` | positive | `SDR` difference of always against never on the two-node law; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9474 to 0.9546, SE ratio 0.9980 to 1.0224 | pass |
| `interval_calibration` | `d_sdr_static__shrunken_se_control` | control | `SDR` difference of always against never on the two-node law, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8302 to 0.8425, SE ratio 0.6985 to 0.7157 | pass |
| `interval_calibration` | `i_contrast_x0_5__correctly_specified` | positive | `IPSI` contrast of odds multiplier 0.5 against 1 on the incremental law; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9457 to 0.9531, SE ratio 0.9886 to 1.0120 | pass |
| `interval_calibration` | `i_contrast_x0_5__shrunken_se_control` | control | `IPSI` contrast of odds multiplier 0.5 against 1 on the incremental law, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8210 to 0.8336, SE ratio 0.6917 to 0.7083 | pass |
| `interval_calibration` | `i_contrast_x2__correctly_specified` | positive | `IPSI` contrast of odds multiplier 2 against 1 on the incremental law; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9456 to 0.9530, SE ratio 0.9930 to 1.0168 | pass |
| `interval_calibration` | `i_contrast_x2__shrunken_se_control` | control | `IPSI` contrast of odds multiplier 2 against 1 on the incremental law, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8255 to 0.8380, SE ratio 0.6952 to 0.7118 | pass |
| `interval_calibration` | `m_duration__correctly_specified` | positive | `LTMLE` duration coefficient of the working model; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9489 to 0.9560, SE ratio 1.0070 to 1.0310 | pass |
| `interval_calibration` | `m_duration__shrunken_se_control` | control | `LTMLE` duration coefficient of the working model, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8317 to 0.8440, SE ratio 0.7048 to 0.7216 | pass |
| `interval_calibration` | `s_contrast_h2__correctly_specified` | positive | `LTMLE` risk difference of always against never at horizon 2 on the survival law; the median over three fold draws and its split-adjusted variance | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9463 to 0.9536, SE ratio 0.9930 to 1.0172 | pass |
| `interval_calibration` | `s_contrast_h2__shrunken_se_control` | control | `LTMLE` risk difference of always against never at horizon 2 on the survival law, with each reported SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8258 to 0.8383, SE ratio 0.6954 to 0.7119 | pass |
| `ipsi_primary` | `contrast_x0_5__repeated` | positive | `IPSI` at n = 80,000, the contrast of odds multiplier 0.5 against 1; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000558 to -0.000147, margin 0.000796, coverage 0.9258 to 0.9565, SE ratio 0.9384 to 1.0317 | pass |
| `ipsi_primary` | `contrast_x2__repeated` | positive | contrast of odds multiplier 2 against 1: `IPSI` at n = 80,000, the contrast of odds multiplier 2 against 1; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000182 to 0.000607, margin 0.000823, coverage 0.9300 to 0.9597, SE ratio 0.9589 to 1.0511 | pass |
| `ipsi_primary` | `mean_x0_5__repeated` | positive | `IPSI` at n = 80,000, the mean under odds multiplier 0.5; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000741 to 0.000320, margin 0.0021, coverage 0.9357 to 0.9641, SE ratio 0.9537 to 1.0431 | pass |
| `ipsi_primary` | `mean_x1__repeated` | positive | `IPSI` at n = 80,000, the mean under the natural course; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000406 to 0.000690, margin 0.0021, coverage 0.9350 to 0.9635, SE ratio 0.9466 to 1.0338 | pass |
| `ipsi_primary` | `mean_x2__repeated` | positive | mean under odds multiplier 2: `IPSI` at n = 80,000, the mean under odds multiplier 2; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000007 to 0.0011, margin 0.0021, coverage 0.9378 to 0.9657, SE ratio 0.9456 to 1.0337 | pass |
| `repeat_increment` | `h1__repeat_median` | diagnostic | survival risk difference at horizon 1, the SE ratio of the median over three draws | reported only; no verdict | increment 0.000152, -0.000199 to 0.000490, ceiling 0.0300 | reported |
| `repeat_increment` | `h1__single_draw` | diagnostic | survival risk difference at horizon 1, the SE ratio of the first draw alone | reported only; no verdict | increment 0.000152, -0.000199 to 0.000490, ceiling 0.0300 | reported |
| `repeat_increment` | `h2__repeat_median` | diagnostic | survival risk difference at horizon 2, the SE ratio of the median over three draws | reported only; no verdict | increment 0.000952, 0.000216 to 0.0017, ceiling 0.0300 | reported |
| `repeat_increment` | `h2__single_draw` | diagnostic | survival risk difference at horizon 2, the SE ratio of the first draw alone | reported only; no verdict | increment 0.000952, 0.000216 to 0.0017, ceiling 0.0300 | reported |
| `repeat_increment` | `h3__inflated_se_control` | control | the horizon-3 median with its reported SE multiplied by 1.05 | the increment must reach or exceed the ceiling | increment 0.0666, 0.0637 to 0.0698, ceiling 0.0300 | pass |
| `repeat_increment` | `h3__repeat_median` | positive | survival risk difference at horizon 3, the SE ratio of the median over three draws | the increment over the first draw stays under its ceiling | increment 0.0156, 0.0127 to 0.0185, ceiling 0.0300 | pass |
| `repeat_increment` | `h3__single_draw` | positive | survival risk difference at horizon 3, the SE ratio of the first draw alone | the first-draw ratio, the base of the increment | increment 0.0156, 0.0127 to 0.0185, ceiling 0.0300 | pass |
| `repeat_stability` | `d_ltmle_dynamic__one_repeat_control` | control | `LTMLE` difference of treat-if-L2 against never on the two-node law; the first draw alone on the same fixed sample | the spread ratio of the first draw against itself stays under the declared boundary | spread ratio 0.6616, 0.5781 to 0.7460, boundary 1 | pass |
| `repeat_stability` | `d_ltmle_dynamic__three_repeats` | positive | `LTMLE` difference of treat-if-L2 against never on the two-node law; the median over three fold draws on one fixed sample, over many fold seeds | the spread of the median over the spread of its first draw stays under the declared boundary, above the spread floor | spread ratio 0.6616, 0.5781 to 0.7460, boundary 1 | pass |
| `repeat_stability` | `d_ltmle_static__one_repeat_control` | control | `LTMLE` risk difference of always against never on the two-node law; the first draw alone on the same fixed sample | the spread ratio of the first draw against itself stays under the declared boundary | spread ratio 0.6886, 0.6094 to 0.7720, boundary 1 | pass |
| `repeat_stability` | `d_ltmle_static__three_repeats` | positive | `LTMLE` risk difference of always against never on the two-node law; the median over three fold draws on one fixed sample, over many fold seeds | the spread of the median over the spread of its first draw stays under the declared boundary, above the spread floor | spread ratio 0.6886, 0.6094 to 0.7720, boundary 1 | pass |
| `repeat_stability` | `d_sdr_dynamic__one_repeat_control` | control | `SDR` difference of treat-if-L2 against never on the two-node law; the first draw alone on the same fixed sample | the spread ratio of the first draw against itself stays under the declared boundary | spread ratio 0.6614, 0.5824 to 0.7486, boundary 1 | pass |
| `repeat_stability` | `d_sdr_dynamic__three_repeats` | positive | `SDR` difference of treat-if-L2 against never on the two-node law; the median over three fold draws on one fixed sample, over many fold seeds | the spread of the median over the spread of its first draw stays under the declared boundary, above the spread floor | spread ratio 0.6614, 0.5824 to 0.7486, boundary 1 | pass |
| `repeat_stability` | `d_sdr_static__one_repeat_control` | control | `SDR` difference of always against never on the two-node law; the first draw alone on the same fixed sample | the spread ratio of the first draw against itself stays under the declared boundary | spread ratio 0.6879, 0.6101 to 0.7689, boundary 1 | pass |
| `repeat_stability` | `d_sdr_static__three_repeats` | positive | `SDR` difference of always against never on the two-node law; the median over three fold draws on one fixed sample, over many fold seeds | the spread of the median over the spread of its first draw stays under the declared boundary, above the spread floor | spread ratio 0.6879, 0.6101 to 0.7689, boundary 1 | pass |
| `repeat_stability` | `i_contrast_x2__one_repeat_control` | control | `IPSI` contrast of odds multiplier 2 against 1 on the incremental law; the first draw alone on the same fixed sample | the spread ratio of the first draw against itself stays under the declared boundary | spread ratio 0.7026, 0.6420 to 0.7683, boundary 1 | pass |
| `repeat_stability` | `i_contrast_x2__three_repeats` | positive | `IPSI` contrast of odds multiplier 2 against 1 on the incremental law; the median over three fold draws on one fixed sample, over many fold seeds | the spread of the median over the spread of its first draw stays under the declared boundary, above the spread floor | spread ratio 0.7026, 0.6420 to 0.7683, boundary 1 | pass |
| `root_n_and_efficiency` | `ltmle_static__n_2000` | control | `LTMLE` static contrast on the two-node law at n = 2,000; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0022, coverage 0.9524 to 0.9843, SE ratio 1.1085 | pass |
| `root_n_and_efficiency` | `ltmle_static__n_32000` | positive | `LTMLE` static contrast on the two-node law at n = 32,000; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.000441, coverage 0.9267 to 0.9677, SE ratio 0.9911 | pass |
| `root_n_and_efficiency` | `ltmle_static__n_8000` | positive | `LTMLE` static contrast on the two-node law at n = 8,000; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.000222, coverage 0.9371 to 0.9747, SE ratio 1.0217 | pass |
| `root_n_and_efficiency` | `sdr_static__n_2000` | control | `SDR` static contrast on the two-node law at n = 2,000; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.0023, coverage 0.9524 to 0.9843, SE ratio 1.1074 | pass |
| `root_n_and_efficiency` | `sdr_static__n_32000` | positive | `SDR` static contrast on the two-node law at n = 32,000; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.000441, coverage 0.9267 to 0.9677, SE ratio 0.9911 | pass |
| `root_n_and_efficiency` | `sdr_static__n_8000` | positive | `SDR` static contrast on the two-node law at n = 8,000; the median over three fold draws | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.000224, coverage 0.9371 to 0.9747, SE ratio 1.0217 | pass |
| `root_n_rate` | `ltmle_static__empirical_sd` | positive | `LTMLE` static contrast; the log-log slope of the empirical SD over the three sizes | the slope interval lies inside the root-n band | slope -0.5293 to -0.4680 | pass |
| `root_n_rate` | `ltmle_static__reported_se` | positive | `LTMLE` static contrast; the log-log slope of the mean reported SE over the three sizes | the slope interval lies inside the root-n band | slope -0.5412 to -0.5354 | pass |
| `root_n_rate` | `sdr_static__empirical_sd` | positive | `SDR` static contrast; the log-log slope of the empirical SD over the three sizes | the slope interval lies inside the root-n band | slope -0.5304 to -0.4674 | pass |
| `root_n_rate` | `sdr_static__reported_se` | positive | `SDR` static contrast; the log-log slope of the mean reported SE over the three sizes | the slope interval lies inside the root-n band | slope -0.5419 to -0.5360 | pass |
| `simultaneous_coverage` | `c_ltmle__pointwise_joint_control` | control | the pointwise 95% intervals over the means and the contrast of `LTMLE` on the clustered law, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8785 to 0.8892 | pass |
| `simultaneous_coverage` | `c_ltmle__simultaneous_band` | positive | the shipped band over the means and the contrast of `LTMLE` on the clustered law | joint coverage interval inside the calibration coverage band | joint coverage 0.9502 to 0.9572 | pass |
| `simultaneous_coverage` | `d_ltmle__pointwise_joint_control` | control | the pointwise 95% intervals over every mean and contrast of `LTMLE` on the two-node law, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8209 to 0.8335 | pass |
| `simultaneous_coverage` | `d_ltmle__simultaneous_band` | positive | the shipped band over every mean and contrast of `LTMLE` on the two-node law | joint coverage interval inside the calibration coverage band | joint coverage 0.9481 to 0.9553 | pass |
| `simultaneous_coverage` | `d_sdr__pointwise_joint_control` | control | the pointwise 95% intervals over every mean and contrast of `SDR` on the two-node law, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8210 to 0.8336 | pass |
| `simultaneous_coverage` | `d_sdr__simultaneous_band` | positive | the shipped band over every mean and contrast of `SDR` on the two-node law | joint coverage interval inside the calibration coverage band | joint coverage 0.9482 to 0.9553 | pass |
| `simultaneous_coverage` | `i_ipsi__pointwise_joint_control` | control | the pointwise 95% intervals over every mean and contrast of `IPSI` on the incremental law, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8461 to 0.8580 | pass |
| `simultaneous_coverage` | `i_ipsi__simultaneous_band` | positive | the shipped band over every mean and contrast of `IPSI` on the incremental law | joint coverage interval inside the calibration coverage band | joint coverage 0.9451 to 0.9525 | pass |
| `simultaneous_coverage` | `m_ltmle__pointwise_joint_control` | control | the pointwise 95% intervals over both coefficients of the working model, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.9032 to 0.9128 | pass |
| `simultaneous_coverage` | `m_ltmle__simultaneous_band` | positive | the shipped band over both coefficients of the working model | joint coverage interval inside the calibration coverage band | joint coverage 0.9503 to 0.9573 | pass |
| `simultaneous_coverage` | `s_ltmle__pointwise_joint_control` | control | the pointwise 95% intervals over every risk and risk difference of `LTMLE` at horizons 1 to 3 on the survival law, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7232 to 0.7380 | pass |
| `simultaneous_coverage` | `s_ltmle__simultaneous_band` | positive | the shipped band over every risk and risk difference of `LTMLE` at horizons 1 to 3 on the survival law | joint coverage interval inside the calibration coverage band | joint coverage 0.9458 to 0.9531 | pass |
| `single_draw_reading` | `c_static__draw_0` | diagnostic | `LTMLE` difference of always against never on the clustered law; the first of the three draws alone | reported only; no verdict | coverage 0.9480 to 0.9552, SE ratio 0.9966 to 1.0202 | reported |
| `single_draw_reading` | `d_ltmle_dynamic__draw_0` | diagnostic | `LTMLE` difference of treat-if-L2 against never on the two-node law; the first of the three draws alone | reported only; no verdict | coverage 0.9460 to 0.9533, SE ratio 0.9948 to 1.0180 | reported |
| `single_draw_reading` | `d_ltmle_static__draw_0` | diagnostic | `LTMLE` risk difference of always against never on the two-node law; the first of the three draws alone | reported only; no verdict | coverage 0.9457 to 0.9531, SE ratio 0.9913 to 1.0151 | reported |
| `single_draw_reading` | `d_sdr_dynamic__draw_0` | diagnostic | `SDR` difference of treat-if-L2 against never on the two-node law; the first of the three draws alone | reported only; no verdict | coverage 0.9460 to 0.9533, SE ratio 0.9948 to 1.0186 | reported |
| `single_draw_reading` | `d_sdr_static__draw_0` | diagnostic | `SDR` difference of always against never on the two-node law; the first of the three draws alone | reported only; no verdict | coverage 0.9457 to 0.9530, SE ratio 0.9916 to 1.0150 | reported |
| `single_draw_reading` | `i_contrast_x0_5__draw_0` | diagnostic | `IPSI` contrast of odds multiplier 0.5 against 1 on the incremental law; the first of the three draws alone | reported only; no verdict | coverage 0.9458 to 0.9531, SE ratio 0.9889 to 1.0119 | reported |
| `single_draw_reading` | `i_contrast_x2__draw_0` | diagnostic | `IPSI` contrast of odds multiplier 2 against 1 on the incremental law; the first of the three draws alone | reported only; no verdict | coverage 0.9458 to 0.9531, SE ratio 0.9924 to 1.0163 | reported |
| `single_draw_reading` | `m_duration__draw_0` | diagnostic | `LTMLE` duration coefficient of the working model; the first of the three draws alone | reported only; no verdict | coverage 0.9482 to 0.9554, SE ratio 1.0011 to 1.0247 | reported |
| `single_draw_reading` | `s_contrast_h2__draw_0` | diagnostic | `LTMLE` risk difference of always against never at horizon 2 on the survival law; the first of the three draws alone | reported only; no verdict | coverage 0.9464 to 0.9536, SE ratio 0.9923 to 1.0159 | reported |
| `split_displacement` | `c_static__share` | diagnostic | `LTMLE` difference of always against never on the clustered law; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.0049 | reported |
| `split_displacement` | `d_ltmle_dynamic__share` | diagnostic | `LTMLE` difference of treat-if-L2 against never on the two-node law; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.0026 | reported |
| `split_displacement` | `d_ltmle_static__share` | diagnostic | `LTMLE` risk difference of always against never on the two-node law; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.0038 | reported |
| `split_displacement` | `d_sdr_dynamic__share` | diagnostic | `SDR` difference of treat-if-L2 against never on the two-node law; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.0027 | reported |
| `split_displacement` | `d_sdr_static__share` | diagnostic | `SDR` difference of always against never on the two-node law; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.0039 | reported |
| `split_displacement` | `i_contrast_x0_5__share` | diagnostic | `IPSI` contrast of odds multiplier 0.5 against 1 on the incremental law; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.000130 | reported |
| `split_displacement` | `i_contrast_x2__share` | diagnostic | `IPSI` contrast of odds multiplier 2 against 1 on the incremental law; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.000131 | reported |
| `split_displacement` | `m_duration__share` | diagnostic | `LTMLE` duration coefficient of the working model; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.0036 | reported |
| `split_displacement` | `s_contrast_h2__share` | diagnostic | `LTMLE` risk difference of always against never at horizon 2 on the survival law; the share of the reported variance that is the squared displacement of the draws | reported only; no verdict | displacement share 0.000838 | reported |
<!-- /generated -->

## Declared red-cell reading

The study module states this reading before the run, in `RED_CELL_POLICY`. Each red row is first
a defect suspect. The owner of a reported red row is `X46-finite-sample`. A row goes to the owner
only when its diagnosis finds no defect.

| red row | diagnosis | reported with the owner when |
| --- | --- | --- |
| calibration or band, SE-ratio interval contains 1 | the `single_draw_reading` row against the median row, the `split_displacement` share and the pilot reading | the coverage interval contains 0.95 and the diagnosis finds no defect |
| calibration, SE-ratio interval excludes 1 | the same diagnosis, and the unit witnesses of the split adjustment at the run's commit | the draw-0 row is off by the same amount, the share is below 0.01, and the pilot agrees within its Monte Carlo error |
| `ipsi_primary` | the law-I calibration rows at the same size, and the pilot bias of +0.067 and -0.077 SD | the draw-0 rows share the bias and the pilot agrees |
| `root_n_rate` | the per-rung rows and the pilot segment slope of -0.587 | the positive rungs and the pilot explain the slope |
| `repeat_stability` or `repeat_increment` | a defect suspect in the repeat layer | never |
| a cell that lost more than 1% of its replicates to recorded fitting failures, or whose clause fails at its worst case | the records in `property-failures.csv` and `property-failure-accounting.csv`; the summary is read again from the checkpoints | the records show no defect |
| a control that does not fail | a design defect | never |

A median red row whose draw-0 row is green is a defect suspect in the repeat layer. A cell that
lost 1% or less of its replicates keeps its verdict only when each declared clause holds at its
worst case. The cell publishes the count in `failed_replicates`.

The design test states the declared pass probability of each gated cell. The nine calibration
cells pass together with probability 1.00 at the pilot reading. With the pilot's correlated
errors integrated out, that probability is 0.89. A red calibration row has a probability of about
0.11, and this table reads it. Each band cell passes with probability 0.99 or more.

## Result

All 26 implementation-estimand tests pass, and all 58 gated property cells pass. The 22 reported
cells state no verdict. No row is red, so the red-cell reading above did not apply to this run.
Every control row meets its own rule, which means the instrument detects the declared defect in
that control.

The study reads these results as follows.

| reading | where to find it |
| --- | --- |
| Each calibration contrast reports an SE ratio near 1 and a coverage near 0.95 at 24,000 replications | `interval_calibration`, the `correctly_specified` rows |
| The shipped band covers the whole truth vector at the joint rate on all six laws | `simultaneous_coverage`, the `simultaneous_band` rows |
| The three-draw median has a smaller spread than its first draw on every stability label | `repeat_stability`, the `three_repeats` rows |
| The empirical SD contracts at the root-n rate, and the reported SE contracts a little faster | `root_n_rate` |
| Law I at n = 80,000 meets the bias, coverage and SE-ratio rules | `ipsi_primary` |

### Survival risk difference at horizon 3

The median raises the SE ratio of the survival risk difference at horizon 3 above the SE ratio of
the first draw. The increment stays under its ceiling of 0.03, and the inflated-SE control exceeds
it. Both SE ratios are at or above 1, and the coverage of both stays at or above 0.95. This sign
is conservative: the reported interval is slightly wide, not narrow.

A separate single-draw probe of cross-fitted `LTMLE` on the same survival law reads the same sign
(row 25a, not committed to this repository). At n = 8,000 the reported SE is 1.034 to 1.039 times
the Monte Carlo SD, and the coverage is 0.956 to 0.961. The Monte Carlo SD equals the efficient SE
within its Monte Carlo error from n = 8,000. The reported-SE excess shrinks about as 1/n between
n = 2,000 and n = 32,000. The reading is a finite-sample property of the variance estimate, not a
defect of the repeat layer. This study does not isolate its cause. It commits no
known-mechanism rows, so it names none.


## Measured values

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications per scenario |
| `n` | 8000 | observations per primary replication |
| `independent_tests_total` | 26 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 26 | of those, passing |
| `paired_tests_total` | 0 | paired comparisons with R `lmtp` |
| `paired_tests_passed` | 0 | of those, passing |
| `property_cells_total` | 58 | repeated-sampling property cells, reported cells excluded |
| `property_cells_passed` | 58 | cells whose own and family verdicts pass |
| `property_cells_reported` | 22 | cells reported and not gated |
| `max_standardized_bias` | 0.0404 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9437 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9377 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0999 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 16 | the same bound, in primary replications |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | The fast tier recomputes each verdict and does not fail on a red one |
| There is no comparator | No pinned implementation repeats a longitudinal fold draw. The single-draw rows rest on the parity of the cross-fitted longitudinal studies |
| The nuisances are saturated cell means on finite laws | The row does not measure flexible learners, continuous covariates or a misspecified nuisance |
| The repeat count is three | The row does not establish behavior at other repeat counts |
| Each draw uses one unstratified split | The row does not establish a stratified, grouped by design, or supplied split plan |
| The calibration, `ipsi_primary` and band cells cannot see the split adjustment | The pilot displacement share is 1.2e-4 to 0.0050 on the calibration labels, so the adjustment moves a reported SE by 0.25% or less. Only `repeat_stability` and `repeat_increment` at horizon 3 read the repeat layer. The unit witnesses of the library tests carry the exact check of the displacement term |
| Every calibration and primary fit uses `random_state = 0` | All replications share one set of three fold partitions. The folds do not depend on the data, so each replication stays valid |
| `SDR` equals `LTMLE` under saturated cell means | The pilot SE ratios of the two estimators correlate at 1.000. The `SDR` cells exercise the code path of `SDR`, not independent statistical evidence |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/longitudinal_repeated_crossfit/README.md)
gives the regeneration command and the commands that run the study in parts.
