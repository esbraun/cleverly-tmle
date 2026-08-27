# Ordinary competing-risk longitudinal TMLE

This study validates ordinary cumulative-incidence estimation for two competing causes. It covers
two horizons, monotone censoring, static plans, and one dynamic plan. The comparison uses R `lmtp`
1.5.4 with the other cause supplied through `compete=`.

The study reports each unique parameter once. The dynamic plan equals the always-treated plan at
the first horizon. A structural test checks this identity outside repeated sampling.

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,600 panels from the same finite-support law | the identical rows |
| causes | relapse and death | each cause fitted as the event, with the other cause in `compete=` |
| horizons | cause-specific cumulative incidence at times one and two | a binary mean at time one and a survival fit at time two |
| plans | never treat, always treat, and continue when L2 equals one | the same unique plan and horizon combinations |
| mechanisms | exact treatment and censoring probabilities | the same exact per-node density ratios |
| sequential regressions | intercept-only working regressions | `SL.mean` working regressions on the same histories |
| learner path | scikit-learn estimators, called directly | lmtp's internal `run_ensemble` replaced by the same single-learner fit, without SuperLearner's inner cross-validation. The adapter checks the two agree to 1e-10 before every run |
| intervals | pointwise 95% Wald intervals from the influence curve | the same, after transforming one minus the incidence back to incidence |

The outcome regressions are misspecified in this comparison. Therefore, targeting stays nonzero
and the competing-event risk mask affects the result. Exact mechanisms isolate those two steps
without making the comparison an oracle-law duplicate.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` competing-risk LTMLE | -0.0017 to 0.000777 | 0.9381 | 0.9815 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.0017 to 0.000777 | 0.9381 | 0.9815 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.0032 to 0.000929 | 0.9519 | 0.9833 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0032 to 0.000929 | 0.9519 | 0.9833 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` competing-risk LTMLE | -0.000670 to 0.0018 | 0.9519 | 0.9919 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.000670 to 0.0018 | 0.9519 | 0.9919 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.000818 to 0.0032 | 0.9375 | 0.9784 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.000818 to 0.0032 | 0.9375 | 0.9784 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.0031 to 0.000878 | 0.9394 | 0.9632 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0031 to 0.000878 | 0.9394 | 0.9632 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.000864 to 0.0032 | 0.9425 | 0.9599 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.000864 to 0.0032 | 0.9425 | 0.9599 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | `cleverly` competing-risk LTMLE | -0.000993 to 0.0010 | 0.9413 | 0.9872 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.000993 to 0.0010 | 0.9413 | 0.9872 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.0022 to 0.0013 | 0.9406 | 0.9867 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.0022 to 0.0013 | 0.9406 | 0.9867 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | `cleverly` competing-risk LTMLE | -0.000540 to 0.0016 | 0.9481 | 0.9952 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.000540 to 0.0016 | 0.9481 | 0.9952 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.000574 to 0.0029 | 0.9369 | 0.9866 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.000574 to 0.0029 | 0.9369 | 0.9866 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.0022 to 0.0013 | 0.9325 | 0.9594 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | R `lmtp` | -0.0022 to 0.0013 | 0.9325 | 0.9594 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.000615 to 0.0029 | 0.9369 | 0.9675 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | R `lmtp` | -0.000615 to 0.0029 | 0.9369 | 0.9675 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | `cleverly` competing-risk LTMLE | -0.000249 to 0.0012 | 0.9494 | 0.9935 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.000249 to 0.0012 | 0.9494 | 0.9935 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.000333 to 0.0017 | 0.9363 | 0.9813 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.000333 to 0.0017 | 0.9363 | 0.9813 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | `cleverly` competing-risk LTMLE | -0.000673 to 0.000580 | 0.9500 | 0.9956 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.000673 to 0.000580 | 0.9500 | 0.9956 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | `cleverly` competing-risk LTMLE | -0.000946 to 0.000931 | 0.9387 | 0.9809 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.000946 to 0.000931 | 0.9387 | 0.9809 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 1.253e-10 | 4.346e-08 | 1.0000 | 0 | 4.917e-09 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | -8.519e-11 | 1.783e-08 | 1.0000 | 0 | 4.289e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 1.364e-09 | 4.781e-07 | 1.0000 | 0 | 1.690e-09 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | 2.470e-11 | 5.336e-09 | 1.0000 | 0 | 6.692e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | -1.918e-10 | 4.141e-08 | 1.0000 | 0 | 6.677e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | 5.143e-11 | 1.092e-08 | 1.0000 | 0 | 5.014e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | -4.066e-12 | 1.750e-09 | 1.0000 | 0 | 3.228e-11 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | -1.084e-10 | 2.607e-08 | 1.0000 | 0 | 4.629e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | -8.801e-11 | 3.605e-08 | 1.0000 | 0 | 2.009e-08 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | -8.995e-11 | 2.225e-08 | 1.0000 | 0 | 6.593e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | -2.150e-10 | 5.387e-08 | 1.0000 | 0 | 7.877e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | -6.323e-11 | 1.538e-08 | 1.0000 | 0 | 3.586e-10 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | -1.293e-10 | 7.763e-08 | 1.0000 | 0 | 1.505e-08 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | -2.319e-11 | 9.968e-09 | 1.0000 | 0 | 1.702e-09 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | -1.452e-09 | 9.960e-07 | 1.0000 | 0 | 1.031e-09 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | -1.147e-10 | 5.250e-08 | 1.0000 | 0 | 7.086e-10 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `competing_risk_recursion_necessity` | `death_always_t2__all_cause` | positive | always-treat death incidence at horizon two: the estimator removes every first-node event from the later risk set | bias interval inside the equivalence margin | bias -0.0032 to 0.0010, margin 0.0070 | pass |
| `competing_risk_recursion_necessity` | `death_always_t2__cause_specific_control` | control | always-treat death incidence at horizon two: the same recursion wrongly lets the competing cause remain at risk | bias interval must fall entirely outside the margin | bias 0.1488 to 0.1581, margin 0.0156 | pass |
| `competing_risk_recursion_necessity` | `relapse_always_t2__all_cause` | positive | always-treat relapse incidence at horizon two: the estimator removes every first-node event from the later risk set | bias interval inside the equivalence margin | bias -0.000904 to 0.0031, margin 0.0067 | pass |
| `competing_risk_recursion_necessity` | `relapse_always_t2__cause_specific_control` | control | always-treat relapse incidence at horizon two: the same recursion wrongly lets the competing cause remain at risk | bias interval must fall entirely outside the margin | bias 0.0608 to 0.0673, margin 0.0109 | pass |
| `double_robustness` | `death_static_t2__both_correct` | positive | static death contrast at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0046 to -0.000077, margin 0.0077 | pass |
| `double_robustness` | `death_static_t2__both_wrong` | control | static death contrast at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias -0.0175 to -0.0129, margin 0.0077 | pass |
| `double_robustness` | `death_static_t2__mechanism_correct` | positive | static death contrast at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0026 to 0.0022, margin 0.0080 | pass |
| `double_robustness` | `death_static_t2__outcome_correct` | positive | static death contrast at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0028 to 0.0019, margin 0.0079 | pass |
| `double_robustness` | `relapse_dynamic_t2__both_correct` | positive | dynamic relapse contrast at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin | bias -0.0014 to 0.0031, margin 0.0076 | pass |
| `double_robustness` | `relapse_dynamic_t2__both_wrong` | control | dynamic relapse contrast at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin | bias 0.0260 to 0.0302, margin 0.0071 | pass |
| `double_robustness` | `relapse_dynamic_t2__mechanism_correct` | positive | dynamic relapse contrast at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin | bias -0.0020 to 0.0025, margin 0.0076 | pass |
| `double_robustness` | `relapse_dynamic_t2__outcome_correct` | positive | dynamic relapse contrast at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin | bias -0.0012 to 0.0033, margin 0.0076 | pass |
| `interval_calibration` | `death_static_t2__correctly_specified` | positive | static death contrast at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9395 to 0.9515, SE ratio 0.9700 to 1.0066, empirical efficiency ratio 0.9913 to 1.0289, reported efficiency ratio 0.9975 to 0.9984 | pass |
| `interval_calibration` | `death_static_t2__noise_control` | control | static death contrast at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8207 to 0.8405, SE ratio 0.6896 to 0.7160, empirical efficiency ratio 1.3935 to 1.4470, reported efficiency ratio 0.9975 to 0.9984 | pass |
| `interval_calibration` | `death_static_t2__shrunken_se_control` | control | static death contrast at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8168 to 0.8368, SE ratio 0.6794 to 0.7045, empirical efficiency ratio 0.9915 to 1.0281, reported efficiency ratio 0.6982 to 0.6989 | pass |
| `interval_calibration` | `relapse_dynamic_t2__correctly_specified` | positive | dynamic relapse contrast at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9421 to 0.9539, SE ratio 0.9732 to 1.0098, empirical efficiency ratio 0.9888 to 1.0261, reported efficiency ratio 0.9981 to 0.9991 | pass |
| `interval_calibration` | `relapse_dynamic_t2__noise_control` | control | dynamic relapse contrast at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8198 to 0.8397, SE ratio 0.6908 to 0.7167, empirical efficiency ratio 1.3933 to 1.4455, reported efficiency ratio 0.9982 to 0.9991 | pass |
| `interval_calibration` | `relapse_dynamic_t2__shrunken_se_control` | control | dynamic relapse contrast at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8133 to 0.8335, SE ratio 0.6818 to 0.7073, empirical efficiency ratio 0.9882 to 1.0253, reported efficiency ratio 0.6987 to 0.6994 | pass |
| `power` | `death_static_t2__alternative` | positive | static death contrast at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9969, 0.9912 to 0.9993 | pass |
| `power` | `relapse_dynamic_t2__alternative` | positive | dynamic relapse contrast at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9869, 0.9776 to 0.9931 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_32000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 32,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000294, coverage 0.9471 to 0.9726, SE ratio 1.0323 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_4000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000180, coverage 0.9147 to 0.9477, SE ratio 0.9633 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_8000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000635, coverage 0.9216 to 0.9532, SE ratio 0.9755 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_32000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 32,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000220, coverage 0.9307 to 0.9603, SE ratio 1.0189 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_4000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000270, coverage 0.9140 to 0.9471, SE ratio 0.9388 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_8000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000841, coverage 0.9300 to 0.9597, SE ratio 0.9871 | pass |
| `root_n_rate` | `death_static_t2__empirical_sd` | positive | static death contrast at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5570 to -0.4995 | pass |
| `root_n_rate` | `death_static_t2__reported_se` | positive | static death contrast at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4954 to -0.4924 | pass |
| `root_n_rate` | `relapse_dynamic_t2__empirical_sd` | positive | dynamic relapse contrast at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5614 to -0.5036 | pass |
| `root_n_rate` | `relapse_dynamic_t2__reported_se` | positive | dynamic relapse contrast at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4972 to -0.4942 | pass |
| `targeting_necessity` | `death_static_t2__targeted` | positive | static death contrast at horizon two: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0043 to 0.000449, margin 0.0080 | pass |
| `targeting_necessity` | `death_static_t2__untargeted` | control | static death contrast at horizon two: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0193 to -0.0147, margin 0.0078 | pass |
| `targeting_necessity` | `relapse_dynamic_t2__targeted` | positive | dynamic relapse contrast at horizon two: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0018 to 0.0027, margin 0.0075 | pass |
| `targeting_necessity` | `relapse_dynamic_t2__untargeted` | control | dynamic relapse contrast at horizon two: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0264 to 0.0309, margin 0.0074 | pass |
| `type_i_error` | `death_static_t2__sharp_null` | positive | static death contrast at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0600, 0.0457 to 0.0770 | pass |
| `type_i_error` | `relapse_dynamic_t2__sharp_null` | positive | dynamic relapse contrast at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0531, 0.0397 to 0.0693 | pass |
<!-- /generated -->

The property study checks one relapse contrast and one death contrast. It measures double
robustness, root-n behavior, exact-EIF efficiency, calibration, type-I error, and power. Separate
controls remove targeting and replace all-cause survival with target-cause survival.

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
| `property_cells_total` | 36 | independent property cells |
| `property_cells_passed` | 36 | property cells passing |
| `max_standardized_bias` | 0.0432 | largest primary standardized bias |
| `min_coverage` | 0.9325 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9147 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9172 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0427 | highest primary SE-ratio endpoint |
| `properties[targeting_necessity/relapse_dynamic_t2__targeted]:targeting_displacement` | 0.4698 | least paired targeting displacement |
| `properties[competing_risk_recursion_necessity/relapse_always_t2__all_cause]:recursion_displacement` | 2.3623 | least competing-risk recursion displacement |
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
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least required targeting displacement |
| `margin:recursion_displacement` | 0.2500 | least required recursion displacement |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The comparison covers two causes and two horizons | Longer event processes and more causes need separate evidence |
| The first-horizon null is not a longitudinal null | No time-varying node precedes the first event node, so a baseline-only standardisation recovers that null exactly. The first-horizon type-I cells test baseline and censoring adjustment only |
| The inference is pointwise | The row does not validate simultaneous bands across causes, plans, or horizons |
| The mechanisms are supplied | The comparison does not test learned-mechanism parity or active truncation |
| The fit is ordinary | Flexible learning and cross-fitting belong to the separate cross-fitted row |
| Competing events remain natural | The row does not validate an estimand that eliminates a competing event |
| The data are independent and unweighted | The row does not validate observation weights, clustering, or time-respecting splits |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, conditionally independent censoring, and correct competing-event coding.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle_competing/README.md)
gives the commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle_competing/manifest.json)
records the seeds, configuration, source digests, and artifact hashes.
