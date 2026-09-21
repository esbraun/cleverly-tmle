# Cross-fitted competing-risk longitudinal TMLE

This study validates five-fold cumulative-incidence estimation for two competing causes. It covers
two horizons, monotone censoring, static plans, and one dynamic plan. The comparison uses R `lmtp`
1.5.4 with the same realized outer folds.

Each fold runs an untargeted backward regression sequence on its training rows. One pooled
fluctuation per node then targets the out-of-fold predictions over every follower. The
[cross-fitting section](../longitudinal-tmle.md#cross-fitting-the-recursion) of the technical
reference gives the construction. The pinned comparator instead fits each fold's fluctuation on
that fold's training rows and carries it into the fold's next regression. Upstream `lmtp` commit
[`9996b04`](https://github.com/nt-williams/lmtp/commit/9996b04dcbb3ae0b1ef8862097c36d95e9f2fcf9)
classifies that update as a bug. Each paired margin stays as registered.

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
| sequential regressions | fold-specific untargeted intercept-only regressions | fold-specific `SL.mean` regressions on the same histories |
| targeting | one pooled fluctuation per node over every follower | a fluctuation on each fold's training rows, carried into that fold's next regression |
| learner path | scikit-learn estimators, called directly | lmtp's internal `run_ensemble` replaced by the same single-learner fit, without SuperLearner's inner cross-validation. The adapter checks the two agree to 1e-10 before every run |
| intervals | pointwise 95% Wald intervals from held-out influence curves | the same, after transforming one minus the incidence back to incidence |

The outcome regressions are misspecified in this comparison. Therefore, targeting stays nonzero
and the competing-event risk mask affects the result. The exact mechanisms and identical folds
isolate those steps without making the comparison an oracle-law duplicate.

The two targeting steps now differ, and the paired table shows it. Before the pooled update, both
implementations ran the same fold-local update, and the largest paired mean difference was below
2e-9 in absolute value. Under the pooled update, the largest one is -0.000195, for
`cif_regimen[continue_if_l2, relapse @ t=2]`, against a margin of 0.004029. Every paired comparison
still concludes equivalent.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.0014 to 0.000982 | 0.9463 | 1.0030 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.0015 to 0.000965 | 0.9450 | 1.0027 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000199 to 0.0039 | 0.9537 | 1.0044 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.000189 to 0.0039 | 0.9537 | 1.0052 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.0019 to 0.000525 | 0.9444 | 0.9908 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.0019 to 0.000557 | 0.9444 | 0.9909 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0037 to 0.000193 | 0.9475 | 0.9947 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0038 to 0.000153 | 0.9481 | 0.9971 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000981 to 0.0029 | 0.9437 | 0.9946 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.001000 to 0.0029 | 0.9444 | 0.9979 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0031 to 0.000864 | 0.9481 | 0.9966 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0029 to 0.0010 | 0.9500 | 0.9998 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.0010 to 0.000955 | 0.9456 | 1.0066 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.0010 to 0.000949 | 0.9469 | 1.0063 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000165 to 0.0034 | 0.9431 | 1.0085 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.000150 to 0.0034 | 0.9425 | 1.0090 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.0015 to 0.000674 | 0.9425 | 0.9762 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.0014 to 0.000704 | 0.9425 | 0.9759 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0028 to 0.000726 | 0.9419 | 0.9869 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.0028 to 0.000714 | 0.9413 | 0.9881 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000926 to 0.0024 | 0.9413 | 1.0000 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | R `lmtp` | -0.000938 to 0.0024 | 0.9413 | 1.0046 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0021 to 0.0014 | 0.9431 | 0.9932 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | R `lmtp` | -0.0019 to 0.0016 | 0.9463 | 0.9975 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.000532 to 0.000943 | 0.9437 | 0.9647 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.000521 to 0.000955 | 0.9437 | 0.9646 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.0012 to 0.000787 | 0.9469 | 0.9712 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.0012 to 0.000793 | 0.9469 | 0.9714 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | `cleverly` cross-fitted competing-risk LTMLE | -0.000320 to 0.000934 | 0.9469 | 0.9959 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.000322 to 0.000932 | 0.9463 | 0.9962 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | `cleverly` cross-fitted competing-risk LTMLE | -0.000210 to 0.0017 | 0.9363 | 0.9600 | pass |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.000183 to 0.0017 | 0.9394 | 0.9607 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=1]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 0.000018 | 0.0062 | 1.0007 | 0.0013 | 0.000953 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | -0.000010 | 0.0021 | 1.0037 | 0 | 0.0048 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=1]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | -0.000031 | 0.0110 | 1.0011 | 0 | 0.000925 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[always vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | 0.000037 | 0.0081 | 1.0049 | -0.000625 | 0.0027 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, death @ t=2]` | difference in cumulative incidence of death between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | 0.000023 | 0.0051 | 1.0051 | -0.000625 | 0.0028 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `ate_regimen[continue_if_l2 vs never, relapse @ t=2]` | difference in cumulative incidence of relapse between the plans "treat first, then continue if L2 equals one" against "treat at neither time" at horizon t = 2 | -0.000169 | 0.0370 | 1.0047 | -0.0019 | 0.0025 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=1]` | cumulative incidence of death under the plan treat at both times at horizon t = 1 | 0.000006 | 0.0028 | 1.0009 | -0.0012 | 0.0011 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, death @ t=2]` | cumulative incidence of death under the plan treat at both times at horizon t = 2 | -0.000016 | 0.0038 | 1.0037 | 0.000625 | 0.0043 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 1 | -0.000029 | 0.0117 | 1.0009 | 0 | 0.0012 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[always, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at both times at horizon t = 2 | 0.000011 | 0.0026 | 1.0039 | 0.000625 | 0.0032 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, death @ t=2]` | cumulative incidence of death under the plan treat first, then continue if L2 equals one at horizon t = 2 | 0.000017 | 0.0044 | 1.0065 | 0 | 0.0121 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[continue_if_l2, relapse @ t=2]` | cumulative incidence of relapse under the plan treat first, then continue if L2 equals one at horizon t = 2 | -0.000195 | 0.0484 | 1.0057 | -0.0031 | 0.0027 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=1]` | cumulative incidence of death under the plan treat at neither time at horizon t = 1 | -0.000011 | 0.0065 | 1.0004 | 0 | 0.000557 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, death @ t=2]` | cumulative incidence of death under the plan treat at neither time at horizon t = 2 | -0.000006 | 0.0026 | 1.0019 | 0 | 0.0020 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=1]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 1 | 0.000002 | 0.0015 | 1.0008 | 0.000625 | 0.000476 vs 0.0500 | equivalent |
| two-time-point, two-cause competing-risk law with monotone censoring | `cif_regimen[never, relapse @ t=2]` | cumulative incidence of relapse under the plan treat at neither time at horizon t = 2 | -0.000026 | 0.0118 | 1.0022 | -0.0031 | 0.0019 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `competing_risk_recursion_necessity` | `death_always_t2__all_cause` | positive | always-treat death incidence at horizon two: the estimator removes every first-node event from the later risk set | bias interval inside the equivalence margin | bias -0.0041 to 0.000037, margin 0.0070 | pass |
| `competing_risk_recursion_necessity` | `death_always_t2__cause_specific_control` | control | always-treat death incidence at horizon two: the same recursion wrongly lets the competing cause remain at risk | bias interval must fall entirely outside the margin | bias 0.1475 to 0.1564, margin 0.0150 | pass |
| `competing_risk_recursion_necessity` | `relapse_always_t2__all_cause` | positive | always-treat relapse incidence at horizon two: the estimator removes every first-node event from the later risk set | bias interval inside the equivalence margin | bias -0.000219 to 0.0037, margin 0.0067 | pass |
| `competing_risk_recursion_necessity` | `relapse_always_t2__cause_specific_control` | control | always-treat relapse incidence at horizon two: the same recursion wrongly lets the competing cause remain at risk | bias interval must fall entirely outside the margin | bias 0.0621 to 0.0685, margin 0.0108 | pass |
| `crossfit_overfitting` | `cross_fitted_competing_ltmle` | positive | five-fold horizon-two competing-risk LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9682 to 1.0094 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3940 to 0.4101 | pass |
| `double_robustness` | `death_static_t2__both_correct` | positive | static death contrast at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0028 to 0.0019, margin 0.0080, SE ratio 0.9861 | pass |
| `double_robustness` | `death_static_t2__both_wrong` | control | static death contrast at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0184 to -0.0139, margin 0.0074, SE ratio 0.7384 | pass |
| `double_robustness` | `death_static_t2__mechanism_correct` | positive | static death contrast at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000715 to 0.0040, margin 0.0079, SE ratio 1.0031 | pass |
| `double_robustness` | `death_static_t2__outcome_correct` | positive | static death contrast at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0027 to 0.0020, margin 0.0079, SE ratio 0.6928 | pass |
| `double_robustness` | `relapse_dynamic_t2__both_correct` | positive | dynamic relapse contrast at horizon two: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0026 to 0.0018, margin 0.0074, SE ratio 1.0309 | pass |
| `double_robustness` | `relapse_dynamic_t2__both_wrong` | control | dynamic relapse contrast at horizon two: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0279 to 0.0324, margin 0.0075, SE ratio 0.7503 | pass |
| `double_robustness` | `relapse_dynamic_t2__mechanism_correct` | positive | dynamic relapse contrast at horizon two: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0035 to 0.0012, margin 0.0079, SE ratio 0.9574 | pass |
| `double_robustness` | `relapse_dynamic_t2__outcome_correct` | positive | dynamic relapse contrast at horizon two: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0023 to 0.0023, margin 0.0076, SE ratio 0.7391 | pass |
| `interval_calibration` | `death_static_t2__correctly_specified` | positive | static death contrast at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9421 to 0.9539, SE ratio 0.9749 to 1.0124, empirical efficiency ratio 0.9896 to 1.0277, reported efficiency ratio 1.0014 to 1.0023 | pass |
| `interval_calibration` | `death_static_t2__noise_control` | control | static death contrast at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8265 to 0.8461, SE ratio 0.6958 to 0.7220, empirical efficiency ratio 1.3876 to 1.4398, reported efficiency ratio 1.0014 to 1.0023 | pass |
| `interval_calibration` | `death_static_t2__shrunken_se_control` | control | static death contrast at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8164 to 0.8364, SE ratio 0.6826 to 0.7083, empirical efficiency ratio 0.9902 to 1.0275, reported efficiency ratio 0.7010 to 0.7016 | pass |
| `interval_calibration` | `relapse_dynamic_t2__correctly_specified` | positive | dynamic relapse contrast at horizon two: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9431 to 0.9548, SE ratio 0.9734 to 1.0097, empirical efficiency ratio 0.9919 to 1.0289, reported efficiency ratio 1.0011 to 1.0020 | pass |
| `interval_calibration` | `relapse_dynamic_t2__noise_control` | control | dynamic relapse contrast at horizon two: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8235 to 0.8432, SE ratio 0.6987 to 0.7254, empirical efficiency ratio 1.3807 to 1.4331, reported efficiency ratio 1.0011 to 1.0020 | pass |
| `interval_calibration` | `relapse_dynamic_t2__shrunken_se_control` | control | dynamic relapse contrast at horizon two: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8169 to 0.8369, SE ratio 0.6820 to 0.7072, empirical efficiency ratio 0.9914 to 1.0282, reported efficiency ratio 0.7008 to 0.7014 | pass |
| `power` | `death_static_t2__alternative` | positive | static death contrast at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9950, 0.9884 to 0.9984 | pass |
| `power` | `relapse_dynamic_t2__alternative` | positive | dynamic relapse contrast at horizon two: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9869, 0.9776 to 0.9931 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_32000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 32,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000006, coverage 0.9336 to 0.9624, SE ratio 1.0121 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_4000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0011, coverage 0.9210 to 0.9526, SE ratio 0.9947 | pass |
| `root_n_and_efficiency` | `death_static_t2__n_8000` | positive | static death contrast at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012, coverage 0.9357 to 0.9641, SE ratio 1.0091 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_32000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 32,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000064, coverage 0.9350 to 0.9635, SE ratio 1.0298 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_4000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 4,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000059, coverage 0.9364 to 0.9646, SE ratio 1.0089 | pass |
| `root_n_and_efficiency` | `relapse_dynamic_t2__n_8000` | positive | dynamic relapse contrast at horizon two: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000896, coverage 0.9203 to 0.9521, SE ratio 0.9904 | pass |
| `root_n_rate` | `death_static_t2__empirical_sd` | positive | static death contrast at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5445 to -0.4829 | pass |
| `root_n_rate` | `death_static_t2__reported_se` | positive | static death contrast at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5073 to -0.5043 | pass |
| `root_n_rate` | `relapse_dynamic_t2__empirical_sd` | positive | dynamic relapse contrast at horizon two: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5484 to -0.4876 | pass |
| `root_n_rate` | `relapse_dynamic_t2__reported_se` | positive | dynamic relapse contrast at horizon two: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5065 to -0.5035 | pass |
| `targeting_necessity` | `death_static_t2__targeted` | positive | static death contrast at horizon two: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.000266 to 0.0044, margin 0.0079 | pass |
| `targeting_necessity` | `death_static_t2__untargeted` | control | static death contrast at horizon two: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0155 to -0.0109, margin 0.0077 | pass |
| `targeting_necessity` | `relapse_dynamic_t2__targeted` | positive | dynamic relapse contrast at horizon two: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0025 to 0.0020, margin 0.0074 | pass |
| `targeting_necessity` | `relapse_dynamic_t2__untargeted` | control | dynamic relapse contrast at horizon two: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0262 to 0.0304, margin 0.0072 | pass |
| `type_i_error` | `death_static_t2__sharp_null` | positive | static death contrast at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0581, 0.0441 to 0.0749 | pass |
| `type_i_error` | `relapse_dynamic_t2__sharp_null` | positive | dynamic relapse contrast at horizon two: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0612, 0.0468 to 0.0784 | pass |
<!-- /generated -->

The property study keeps the ordinary row's two-cause instruments. It also pairs a fully grown
outcome tree with the same fit without cross-fitting. The joint verdict requires cross-fitting to
restore standard-error scale and improve coverage. Under the pooled update the cross-fitted ratio is
0.987990, with a 99% interval from 0.968219 to 1.009389. The fold-fluctuated construction gave
1.092332 on the same seeds.

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
| `min_coverage` | 0.9363 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9189 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9187 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0576 | highest primary SE-ratio endpoint |
| `properties[crossfit_overfitting/cross_fitted_competing_ltmle]:coverage` | 0.9460 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5650 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_competing_ltmle]:coverage_gain_ci_lower` | 0.3673 | lower bound for the paired coverage gain |
| `properties[crossfit_overfitting/cross_fitted_competing_ltmle]:replicates` | 8000 | paired overfitting replications |
| `properties[targeting_necessity/relapse_dynamic_t2__targeted]:targeting_displacement` | 0.4859 | least paired targeting displacement |
| `properties[competing_risk_recursion_necessity/relapse_always_t2__all_cause]:recursion_displacement` | 2.3835 | least competing-risk recursion displacement |
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
| Agreement with `lmtp` is distributional | The paired claim tests mean similarity and non-inferiority, not rowwise numerical equality. `lmtp` 1.5.4 targets on each fold's training rows, and `cleverly` runs one pooled fluctuation per node |
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
