# Cross-fitted competing-risk longitudinal TMLE

This study validates five-fold cumulative-incidence estimation for two competing causes. It covers
two horizons, monotone censoring, static plans, and one dynamic plan. The comparison uses R `lmtp`
1.5.4 with the same realized outer folds.

The study reports each unique parameter once. The dynamic plan equals the always-treated plan at
the first horizon. A structural test checks this identity outside repeated sampling.

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,600 panels from the same finite-support law | the identical rows |
| causes | relapse and death | each cause fitted as the event, with the other cause in `compete=` |
| folds | one balanced five-fold assignment | the identical serialized assignment |
| plans | never treat, always treat, and continue when L2 equals one | the same unique plan and horizon combinations |
| mechanisms | exact treatment and censoring probabilities | the same exact per-node density ratios |
| sequential regressions | fold-specific intercept-only regressions | fold-specific `SL.mean` regressions on the same histories |
| learner path | scikit-learn estimators, called directly | lmtp's internal `run_ensemble` replaced by the same single-learner fit, without SuperLearner's inner cross-validation. The adapter checks the two agree to 1e-10 before every run |
| intervals | pointwise 95% Wald intervals from held-out influence curves | the same, after transforming one minus the incidence back to incidence |

The outcome regressions are misspecified in this comparison. Therefore, targeting stays nonzero
and the competing-event risk mask affects the result. The exact mechanisms and identical folds
isolate those steps without making the comparison an oracle-law duplicate.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.0015 to 0.000965 | 0.9450 | 1.0027 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.0015 to 0.000965 | 0.9450 | 1.0027 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000189 to 0.0039 | 0.9537 | 1.0052 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.000189 to 0.0039 | 0.9537 | 1.0052 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.0019 to 0.000557 | 0.9444 | 0.9909 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.0019 to 0.000557 | 0.9444 | 0.9909 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0038 to 0.000153 | 0.9481 | 0.9971 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0038 to 0.000153 | 0.9481 | 0.9971 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.001000 to 0.0029 | 0.9444 | 0.9979 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.001000 to 0.0029 | 0.9444 | 0.9979 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0029 to 0.0010 | 0.9500 | 0.9998 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0029 to 0.0010 | 0.9500 | 0.9998 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.0010 to 0.000949 | 0.9469 | 1.0063 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.0010 to 0.000949 | 0.9469 | 1.0063 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000150 to 0.0034 | 0.9425 | 1.0090 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.000150 to 0.0034 | 0.9425 | 1.0090 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.0014 to 0.000704 | 0.9425 | 0.9759 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.0014 to 0.000704 | 0.9425 | 0.9759 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0028 to 0.000714 | 0.9413 | 0.9881 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.0028 to 0.000714 | 0.9413 | 0.9881 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000938 to 0.0024 | 0.9413 | 1.0046 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | R `lmtp` | -0.000938 to 0.0024 | 0.9413 | 1.0046 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0019 to 0.0016 | 0.9463 | 0.9975 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | R `lmtp` | -0.0019 to 0.0016 | 0.9463 | 0.9975 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.000521 to 0.000955 | 0.9437 | 0.9646 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.000521 to 0.000955 | 0.9437 | 0.9646 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0012 to 0.000793 | 0.9469 | 0.9714 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.0012 to 0.000793 | 0.9469 | 0.9714 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.000322 to 0.000932 | 0.9463 | 0.9962 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.000322 to 0.000932 | 0.9463 | 0.9962 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000183 to 0.0017 | 0.9394 | 0.9607 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.000183 to 0.0017 | 0.9394 | 0.9607 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 1.037e-10 | 3.673e-08 | 1.0000 | 0 | 3.534e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | -7.855e-11 | 1.661e-08 | 1.0000 | 0 | 3.105e-09 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 1.343e-09 | 4.698e-07 | 1.0000 | 0 | 1.298e-09 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | 9.396e-12 | 2.053e-09 | 1.0000 | 0 | 2.688e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | -1.766e-10 | 3.909e-08 | 1.0000 | 0 | 2.222e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | 3.492e-11 | 7.660e-09 | 1.0000 | 0 | 2.190e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | -4.741e-12 | 2.077e-09 | 1.0000 | 0 | 7.265e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | -1.017e-10 | 2.464e-08 | 1.0000 | 0 | 4.269e-09 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | -1.073e-10 | 4.308e-08 | 1.0000 | 0 | 1.858e-08 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | -9.479e-11 | 2.329e-08 | 1.0000 | 0 | 2.951e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | -1.998e-10 | 5.170e-08 | 1.0000 | 0 | 7.426e-09 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | -6.926e-11 | 1.721e-08 | 1.0000 | 0 | 2.182e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | -1.085e-10 | 6.320e-08 | 1.0000 | 0 | 1.049e-08 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | -2.312e-11 | 9.809e-09 | 1.0000 | 0 | 4.994e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | -1.450e-09 | 9.942e-07 | 1.0000 | 0 | 7.298e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | -1.042e-10 | 4.649e-08 | 1.0000 | 0 | 4.341e-10 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `competing_risk_recursion_necessity` | `death_always_t2__all_cause` | positive | always-treat death incidence at horizon two: the estimator removes every first-node event from the later risk set | bias interval inside the equivalence margin | bias -0.0041 to 0.000023, margin 0.0070 | pass |
| `competing_risk_recursion_necessity` | `death_always_t2__cause_specific_control` | control | always-treat death incidence at horizon two: the same recursion wrongly lets the competing cause remain at risk | bias interval must fall entirely outside the margin | bias 0.1475 to 0.1564, margin 0.0150 | pass |
| `competing_risk_recursion_necessity` | `relapse_always_t2__all_cause` | positive | always-treat relapse incidence at horizon two: the estimator removes every first-node event from the later risk set | bias interval inside the equivalence margin | bias -0.000538 to 0.0034, margin 0.0066 | pass |
| `competing_risk_recursion_necessity` | `relapse_always_t2__cause_specific_control` | control | always-treat relapse incidence at horizon two: the same recursion wrongly lets the competing cause remain at risk | bias interval must fall entirely outside the margin | bias 0.0621 to 0.0685, margin 0.0108 | pass |
| `crossfit_overfitting` | `cross_fitted_competing_ltmle` | positive | five-fold horizon-two competing-risk LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.0708 to 1.1141 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3940 to 0.4101 | pass |
| `double_robustness` | `death_static_t2__both_correct` | positive | static death contrast at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0032 to 0.0015, margin 0.0079, SE ratio 0.9945 | pass |
| `double_robustness` | `death_static_t2__both_wrong` | control | static death contrast at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0184 to -0.0140, margin 0.0074, SE ratio 0.7400 | pass |
| `double_robustness` | `death_static_t2__mechanism_correct` | positive | static death contrast at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000666 to 0.0040, margin 0.0079, SE ratio 1.0043 | pass |
| `double_robustness` | `death_static_t2__outcome_correct` | positive | static death contrast at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0029 to 0.0018, margin 0.0080, SE ratio 0.6907 | pass |
| `double_robustness` | `relapse_dynamic_t2__both_correct` | positive | dynamic relapse contrast at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0027 to 0.0017, margin 0.0073, SE ratio 1.0343 | pass |
| `double_robustness` | `relapse_dynamic_t2__both_wrong` | control | dynamic relapse contrast at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0279 to 0.0324, margin 0.0076, SE ratio 0.7487 | pass |
| `double_robustness` | `relapse_dynamic_t2__mechanism_correct` | positive | dynamic relapse contrast at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0033 to 0.0014, margin 0.0079, SE ratio 0.9596 | pass |
| `double_robustness` | `relapse_dynamic_t2__outcome_correct` | positive | dynamic relapse contrast at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0022, margin 0.0076, SE ratio 0.7438 | pass |
| `interval_calibration` | `death_static_t2__correctly_specified` | positive | static death contrast at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9417 to 0.9535, SE ratio 0.9756 to 1.0129, empirical efficiency ratio 0.9891 to 1.0270, reported efficiency ratio 1.0014 to 1.0023 | pass |
| `interval_calibration` | `death_static_t2__noise_control` | control | static death contrast at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8267 to 0.8463, SE ratio 0.6960 to 0.7222, empirical efficiency ratio 1.3872 to 1.4396, reported efficiency ratio 1.0014 to 1.0023 | pass |
| `interval_calibration` | `death_static_t2__shrunken_se_control` | control | static death contrast at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8181 to 0.8380, SE ratio 0.6829 to 0.7089, empirical efficiency ratio 0.9893 to 1.0269, reported efficiency ratio 0.7010 to 0.7016 | pass |
| `interval_calibration` | `relapse_dynamic_t2__correctly_specified` | positive | dynamic relapse contrast at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9438 to 0.9554, SE ratio 0.9745 to 1.0110, empirical efficiency ratio 0.9908 to 1.0278, reported efficiency ratio 1.0011 to 1.0020 | pass |
| `interval_calibration` | `relapse_dynamic_t2__noise_control` | control | dynamic relapse contrast at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8240 to 0.8436, SE ratio 0.6992 to 0.7259, empirical efficiency ratio 1.3796 to 1.4321, reported efficiency ratio 1.0011 to 1.0020 | pass |
| `interval_calibration` | `relapse_dynamic_t2__shrunken_se_control` | control | dynamic relapse contrast at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8168 to 0.8368, SE ratio 0.6827 to 0.7079, empirical efficiency ratio 0.9904 to 1.0270, reported efficiency ratio 0.7008 to 0.7014 | pass |
| `power` | `death_static_t2__alternative` | positive | static death contrast at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9956, 0.9893 to 0.9987 | pass |
| `power` | `relapse_dynamic_t2__alternative` | positive | dynamic relapse contrast at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9869, 0.9776 to 0.9931 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_32000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 32,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000005, coverage 0.9350 to 0.9635, SE ratio 1.0126 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_4000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0015, coverage 0.9258 to 0.9565, SE ratio 0.9989 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_8000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012, coverage 0.9371 to 0.9651, SE ratio 1.0149 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_32000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 32,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000071, coverage 0.9357 to 0.9641, SE ratio 1.0298 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_4000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000001, coverage 0.9364 to 0.9646, SE ratio 1.0103 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_8000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000929, coverage 0.9251 to 0.9559, SE ratio 0.9914 | pass |
| `root_n_rate` | `death_static_t2__empirical_sd` | positive | static death contrast at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5428 to -0.4811 | pass |
| `root_n_rate` | `death_static_t2__reported_se` | positive | static death contrast at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5072 to -0.5043 | pass |
| `root_n_rate` | `relapse_dynamic_t2__empirical_sd` | positive | dynamic relapse contrast at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5474 to -0.4869 | pass |
| `root_n_rate` | `relapse_dynamic_t2__reported_se` | positive | dynamic relapse contrast at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5065 to -0.5035 | pass |
| `targeting_necessity` | `death_static_t2__targeted` | positive | static death contrast at horizon two: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000210 to 0.0045, margin 0.0079 | pass |
| `targeting_necessity` | `death_static_t2__untargeted` | control | static death contrast at horizon two: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0155 to -0.0109, margin 0.0077 | pass |
| `targeting_necessity` | `relapse_dynamic_t2__targeted` | positive | dynamic relapse contrast at horizon two: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0023 to 0.0022, margin 0.0075 | pass |
| `targeting_necessity` | `relapse_dynamic_t2__untargeted` | control | dynamic relapse contrast at horizon two: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0262 to 0.0304, margin 0.0072 | pass |
| `type_i_error` | `death_static_t2__sharp_null` | positive | static death contrast at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0550, 0.0414 to 0.0714 | pass |
| `type_i_error` | `relapse_dynamic_t2__sharp_null` | positive | dynamic relapse contrast at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0638, 0.0490 to 0.0811 | pass |
<!-- /generated -->

The property study keeps the ordinary row's two-cause instruments. It also pairs a fully grown
outcome tree with the same fit without cross-fitting. The joint verdict requires cross-fitting to
restore standard-error scale and improve coverage.

At the second horizon a baseline-only standardisation misses the null by -0.0077 for death and
0.0091 for relapse. The null is therefore one an estimator has to be longitudinal to find. At the
first horizon that same analysis returns exactly zero, because no time-varying node precedes the
first event node. A crude comparison of arms is biased at both horizons, so the first-horizon
cells still test baseline and censoring adjustment.

## Measured values

Names beginning `margin:` are thresholds declared before the run. The other values come from the
committed results.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | paired replications |
| `n` | 4000 | observations per paired replication |
| `independent_tests_total` | 32 | implementation-estimand truth tests |
| `independent_tests_passed` | 32 | truth tests passing |
| `paired_tests_total` | 16 | paired estimand comparisons |
| `paired_tests_passed` | 16 | paired comparisons passing |
| `property_cells_total` | 38 | independent property cells |
| `property_cells_passed` | 38 | property cells passing |
| `max_standardized_bias` | 0.0595 | largest primary standardized bias |
| `min_coverage` | 0.9394 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9223 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9190 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0576 | highest primary SE-ratio endpoint |
| `properties[crossfit_overfitting/cross_fitted_competing_ltmle]:coverage` | 0.9646 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5650 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_competing_ltmle]:coverage_gain_ci_lower` | 0.3855 | lower bound for the paired coverage gain |
| `properties[crossfit_overfitting/cross_fitted_competing_ltmle]:replicates` | 8000 | paired overfitting replications |
| `properties[targeting_necessity/relapse_dynamic_t2__targeted]:targeting_displacement` | 0.4868 | least paired targeting displacement |
| `properties[competing_risk_recursion_necessity/relapse_always_t2__all_cause]:recursion_displacement` | 2.4121 | least competing-risk recursion displacement |
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
| `margin:paired_difference` | 0.1500 | paired similarity margin |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope upper bound |
| `margin:excluded_slope` | -0.2500 | rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least required targeting displacement |
| `margin:recursion_displacement` | 0.2500 | least required recursion displacement |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

## Limitations

| limitation | what it means for use |
| --- | --- |
| Agreement with `lmtp` is distributional | The paired claim tests mean similarity and non-inferiority, not rowwise numerical equality |
| One fixed five-fold assignment is studied | The row does not validate repeated folds or time-respecting splits |
| The comparison covers two causes and two horizons | Longer event processes and more causes need separate evidence |
| The first-horizon null is not a longitudinal null | No time-varying node precedes the first event node, so a baseline-only standardisation recovers that null exactly. The first-horizon type-I cells test baseline and censoring adjustment only |
| The inference is pointwise | The row does not validate simultaneous bands across causes, plans, or horizons |
| The mechanisms are supplied | The comparison does not test learned-mechanism parity or active truncation |
| Competing events remain natural | The row does not validate an estimand that eliminates a competing event |
| The data are independent and unweighted | The row does not validate observation weights or clustering |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, conditionally independent censoring, and correct competing-event coding.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle_competing_crossfit/README.md)
gives the commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle_competing_crossfit/manifest.json)
records the seeds, configuration, source digests, and artifact hashes.
