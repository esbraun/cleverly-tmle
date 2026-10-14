# Repeated-outcome longitudinal TMLE

This study validates `LTMLE` with `outcome=Repeated(["Y1", "Y2", "Y3"])`. A repeated outcome is
measured at every node and is not absorbing. The parameter is the mean $E[Y_h^d]$ for each plan $d$
and horizon $h$. The section [repeated outcomes](../longitudinal-tmle.md#repeated-outcomes) states
the estimator and its exact-law evidence. The canonical comparison uses R
[`ltmle`](https://www.jstatsoft.org/article/view/v081i01) 1.3-0 with `survivalOutcome = FALSE`.

The study reports each unique parameter once. The dynamic plan equals the always-treated plan at the
first node, so its mean and contrast at $t = 1$ duplicate those of `always`. A fast test checks that
identity. The duplicates are not cells of this study.

## What was compared

| setting | `cleverly` | R `ltmle` |
| --- | --- | --- |
| datasets | 1,600 censored panels of 2,000 rows from `make_longitudinal_repeated`, with three nodes | the identical rows |
| fit | one fit with `outcome=Repeated(["Y1", "Y2", "Y3"])` | one fit of the prefix through $Y_h$ for each horizon $h$. The earlier $Y_s$ are covariates |
| plans | `never`, `always`, and a rule that treats at node 1 and then treats exactly when the previous outcome was positive | the same three plans |
| reported parameters | the mean of each plan at $t = 1, 2, 3$, and the contrasts of the other two plans against `never` | the same, 13 unique cells |
| mechanism | the generating treatment and censoring probabilities, read through a declared design width, with `g_bounds` from the study constant | a numeric `gform` with the same probabilities and bounds |
| sequential regressions | follower-stratified quasibinomial regressions on the history at each node | the same family, with a main-terms `Qform` on the same history at each block |
| fitting | in sample, `n_folds = 1` | in sample |
| intervals | pointwise 95% Wald from the influence curve | `variance.method = "ic"` |
| initial estimate | the first node's regression of the targeted second node | a second call with `gcomp = TRUE`, the sequential regression with no fluctuation |

The two initial estimates differ by design, and no verdict reads them. W is binary, so the first
node's regression is saturated in W and its fluctuation is zero. That value equals the targeted
estimate. The R plug-in is the value a targeting step must move away from.

Each truth is an exact enumeration of the binary law (`cleverly.datasets.repeated_truth`).
`tests/studies/ltmle_repeated_law.py` repeats the enumeration on the observed-data law.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 1 | `cleverly` | -0.0023 to 0.000383 | 0.9494 | 1.0131 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 1 | R `ltmle` | -0.0023 to 0.000383 | 0.9494 | 1.0131 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 2 | `cleverly` | -0.0030 to 0.0014 | 0.9469 | 0.9959 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 2 | R `ltmle` | -0.0030 to 0.0014 | 0.9469 | 0.9959 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 3 | `cleverly` | -0.0037 to 0.0032 | 0.9456 | 0.9984 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 3 | R `ltmle` | -0.0037 to 0.0032 | 0.9456 | 0.9984 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 2 | `cleverly` | -0.0032 to 0.0016 | 0.9463 | 0.9987 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 2 | R `ltmle` | -0.0032 to 0.0016 | 0.9463 | 0.9987 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 3 | `cleverly` | -0.0063 to 0.0021 | 0.9375 | 0.9611 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 3 | R `ltmle` | -0.0063 to 0.0021 | 0.9375 | 0.9611 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=1]` | mean of the repeated outcome under the plan treat at every time at horizon t = 1 | `cleverly` | -0.0016 to 0.000438 | 0.9506 | 1.0106 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=1]` | mean of the repeated outcome under the plan treat at every time at horizon t = 1 | R `ltmle` | -0.0016 to 0.000438 | 0.9506 | 1.0106 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=2]` | mean of the repeated outcome under the plan treat at every time at horizon t = 2 | `cleverly` | -0.0022 to 0.0011 | 0.9400 | 0.9823 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=2]` | mean of the repeated outcome under the plan treat at every time at horizon t = 2 | R `ltmle` | -0.0022 to 0.0011 | 0.9400 | 0.9823 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=3]` | mean of the repeated outcome under the plan treat at every time at horizon t = 3 | `cleverly` | -0.0033 to 0.0017 | 0.9356 | 1.0014 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=3]` | mean of the repeated outcome under the plan treat at every time at horizon t = 3 | R `ltmle` | -0.0033 to 0.0017 | 0.9356 | 1.0014 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=1]` | mean of the repeated outcome under the plan treat at no time at horizon t = 1 | `cleverly` | -0.000511 to 0.0013 | 0.9487 | 1.0097 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=1]` | mean of the repeated outcome under the plan treat at no time at horizon t = 1 | R `ltmle` | -0.000511 to 0.0013 | 0.9487 | 1.0097 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=2]` | mean of the repeated outcome under the plan treat at no time at horizon t = 2 | `cleverly` | -0.0013 to 0.0018 | 0.9469 | 0.9843 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=2]` | mean of the repeated outcome under the plan treat at no time at horizon t = 2 | R `ltmle` | -0.0013 to 0.0018 | 0.9469 | 0.9843 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=3]` | mean of the repeated outcome under the plan treat at no time at horizon t = 3 | `cleverly` | -0.0031 to 0.0019 | 0.9375 | 0.9605 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=3]` | mean of the repeated outcome under the plan treat at no time at horizon t = 3 | R `ltmle` | -0.0031 to 0.0019 | 0.9375 | 0.9605 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=2]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 2 | `cleverly` | -0.0024 to 0.0013 | 0.9450 | 0.9750 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=2]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 2 | R `ltmle` | -0.0024 to 0.0013 | 0.9450 | 0.9750 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=3]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 3 | `cleverly` | -0.0062 to 0.000799 | 0.9200 | 0.9254 | pass |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=3]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 3 | R `ltmle` | -0.0062 to 0.000799 | 0.9200 | 0.9254 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=1]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 1 | 1.076e-10 | 3.428e-08 | 1.0000 | 0 | 8.411e-09 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 2 | 4.027e-11 | 7.763e-09 | 1.0000 | 0 | 6.224e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[always vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at every time" against "treat at no time" at horizon t = 3 | 1.099e-10 | 1.371e-08 | 1.0000 | 0 | 3.476e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=2]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 2 | -3.794e-11 | 6.804e-09 | 1.0000 | 0 | 7.294e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ate_regimen[treat then treat after an outcome vs never @ t=3]` | difference in the mean of the repeated outcome between the plans "treat at the first time, then treat at each later time after an outcome of one" against "treat at no time" at horizon t = 3 | -3.449e-08 | 0.000004 | 1.0000 | 0 | 0.000001 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=1]` | mean of the repeated outcome under the plan treat at every time at horizon t = 1 | -5.831e-12 | 2.483e-09 | 1.0000 | 0 | 9.993e-11 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=2]` | mean of the repeated outcome under the plan treat at every time at horizon t = 2 | -1.534e-10 | 4.071e-08 | 1.0000 | 0 | 4.998e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[always @ t=3]` | mean of the repeated outcome under the plan treat at every time at horizon t = 3 | -7.381e-11 | 1.279e-08 | 1.0000 | 0 | 2.845e-09 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=1]` | mean of the repeated outcome under the plan treat at no time at horizon t = 1 | -1.134e-10 | 5.384e-08 | 1.0000 | 0 | 1.735e-08 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=2]` | mean of the repeated outcome under the plan treat at no time at horizon t = 2 | -1.937e-10 | 5.273e-08 | 1.0000 | 0 | 9.166e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[never @ t=3]` | mean of the repeated outcome under the plan treat at no time at horizon t = 3 | -1.837e-10 | 3.193e-08 | 1.0000 | 0 | 5.202e-10 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=2]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 2 | -2.316e-10 | 5.321e-08 | 1.0000 | 0 | 1.100e-09 vs 0.0500 | equivalent |
| three-time-point law with a repeated binary outcome and monotone censoring | `ey_regimen[treat then treat after an outcome @ t=3]` | mean of the repeated outcome under the plan treat at the first time, then treat at each later time after an outcome of one at horizon t = 3 | -3.468e-08 | 0.000004 | 1.0000 | 0 | 0.000001 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `always_t3__both_wrong` | control | mean under always at horizon t = 3, repeated binary outcome: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0242 to -0.0186, margin 0.0085, SE ratio 1.1531 | pass |
| `double_robustness` | `always_t3__mechanism_correct` | positive | mean under always at horizon t = 3, repeated binary outcome: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0031 to 0.0034, margin 0.0100, SE ratio 0.9908 | pass |
| `double_robustness` | `always_vs_never_t3__both_wrong` | control | contrast of always against never at horizon t = 3, repeated binary outcome: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0580 to -0.0497, margin 0.0127, SE ratio 1.1281 | pass |
| `double_robustness` | `always_vs_never_t3__mechanism_correct` | positive | contrast of always against never at horizon t = 3, repeated binary outcome: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0042 to 0.0047, margin 0.0137, SE ratio 1.0061 | pass |
| `double_robustness` | `never_t3__both_wrong` | control | mean under never at horizon t = 3, repeated binary outcome: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0293 to 0.0357, margin 0.0098, SE ratio 1.0704 | pass |
| `double_robustness` | `never_t3__mechanism_correct` | positive | mean under never at horizon t = 3, repeated binary outcome: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0030 to 0.0029, margin 0.0091, SE ratio 1.0574 | pass |
| `double_robustness` | `rule_t3__mechanism_correct` | positive | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0054 to 0.0032, margin 0.0132, SE ratio 1.0284 | pass |
| `double_robustness` | `rule_vs_never_t3__mechanism_correct` | positive | contrast of the rule against never at horizon t = 3, repeated binary outcome: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0064 to 0.0043, margin 0.0164, SE ratio 1.0117 | pass |
| `interval_calibration` | `always_t1__estimated_mechanisms` | positive | mean under always at horizon t = 1, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9454 to 0.9645, SE ratio 0.9794 to 1.0451 | pass |
| `interval_calibration` | `always_t2__estimated_mechanisms` | positive | always-treat risk at horizon two: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9382 to 0.9586, SE ratio 0.9668 to 1.0316 | pass |
| `interval_calibration` | `always_t3__continuous_gaussian` | positive | mean under always at horizon t = 3, repeated binary outcome: n = 8,000, the three-value outcome law, least-squares outcome regressions, main-terms logistic mechanisms estimated | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9372 to 0.9577, SE ratio 0.9740 to 1.0408 | pass |
| `interval_calibration` | `always_t3__crossfit_boosted` | positive | mean under always at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, a depth-2 booster of 50 rounds with an L2 penalty for both outcome learners | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9430 to 0.9625, SE ratio 0.9808 to 1.0461 | pass |
| `interval_calibration` | `always_t3__crossfit_glm` | positive | mean under always at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9433 to 0.9628, SE ratio 0.9810 to 1.0476 | pass |
| `interval_calibration` | `always_t3__estimated_mechanisms` | positive | mean under always at horizon t = 3, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9365 to 0.9571, SE ratio 0.9726 to 1.0371 | pass |
| `interval_calibration` | `always_t3__noise_control` | control | mean under always at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8226 to 0.8563, SE ratio 0.6952 to 0.7415 | pass |
| `interval_calibration` | `always_t3__shrunken_se_control` | control | mean under always at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8129 to 0.8474, SE ratio 0.6805 to 0.7249 | pass |
| `interval_calibration` | `always_vs_never_t1__estimated_mechanisms` | positive | contrast of always against never at horizon t = 1, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9358 to 0.9566, SE ratio 0.9710 to 1.0352 | pass |
| `interval_calibration` | `always_vs_never_t2__estimated_mechanisms` | positive | contrast of always against never at horizon t = 2, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9379 to 0.9583, SE ratio 0.9662 to 1.0315 | pass |
| `interval_calibration` | `always_vs_never_t3__continuous_gaussian` | positive | contrast of always against never at horizon t = 3, repeated binary outcome: n = 8,000, the three-value outcome law, least-squares outcome regressions, main-terms logistic mechanisms estimated | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9365 to 0.9571, SE ratio 0.9675 to 1.0354 | pass |
| `interval_calibration` | `always_vs_never_t3__crossfit_boosted` | positive | contrast of always against never at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, a depth-2 booster of 50 rounds with an L2 penalty for both outcome learners | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9409 to 0.9608, SE ratio 0.9750 to 1.0377 | pass |
| `interval_calibration` | `always_vs_never_t3__crossfit_glm` | positive | contrast of always against never at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9382 to 0.9586, SE ratio 0.9812 to 1.0459 | pass |
| `interval_calibration` | `always_vs_never_t3__estimated_mechanisms` | positive | contrast of always against never at horizon t = 3, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9409 to 0.9608, SE ratio 0.9594 to 1.0235 | pass |
| `interval_calibration` | `always_vs_never_t3__noise_control` | control | contrast of always against never at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8041 to 0.8393, SE ratio 0.6750 to 0.7192 | pass |
| `interval_calibration` | `always_vs_never_t3__shrunken_se_control` | control | contrast of always against never at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8019 to 0.8372, SE ratio 0.6710 to 0.7150 | pass |
| `interval_calibration` | `never_t1__estimated_mechanisms` | positive | mean under never at horizon t = 1, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9355 to 0.9563, SE ratio 0.9554 to 1.0187 | pass |
| `interval_calibration` | `never_t2__estimated_mechanisms` | positive | mean under never at horizon t = 2, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9420 to 0.9617, SE ratio 0.9710 to 1.0370 | pass |
| `interval_calibration` | `never_t3__continuous_gaussian` | positive | mean under never at horizon t = 3, repeated binary outcome: n = 8,000, the three-value outcome law, least-squares outcome regressions, main-terms logistic mechanisms estimated | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9406 to 0.9605, SE ratio 0.9736 to 1.0367 | pass |
| `interval_calibration` | `never_t3__crossfit_boosted` | positive | mean under never at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, a depth-2 booster of 50 rounds with an L2 penalty for both outcome learners | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9338 to 0.9549, SE ratio 0.9701 to 1.0362 | pass |
| `interval_calibration` | `never_t3__crossfit_glm` | positive | mean under never at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9409 to 0.9608, SE ratio 0.9724 to 1.0359 | pass |
| `interval_calibration` | `never_t3__estimated_mechanisms` | positive | mean under never at horizon t = 3, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9372 to 0.9577, SE ratio 0.9562 to 1.0189 | pass |
| `interval_calibration` | `never_t3__noise_control` | control | mean under never at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8032 to 0.8384, SE ratio 0.6797 to 0.7243 | pass |
| `interval_calibration` | `never_t3__shrunken_se_control` | control | mean under never at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8006 to 0.8360, SE ratio 0.6699 to 0.7139 | pass |
| `interval_calibration` | `rule_t2__estimated_mechanisms` | positive | mean under the rule that treats after an outcome at horizon t = 2, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9430 to 0.9625, SE ratio 0.9705 to 1.0358 | pass |
| `interval_calibration` | `rule_t3__continuous_gaussian` | positive | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: n = 8,000, the three-value outcome law, least-squares outcome regressions, main-terms logistic mechanisms estimated | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9294 to 0.9512, SE ratio 0.9431 to 1.0049 | pass |
| `interval_calibration` | `rule_t3__crossfit_boosted` | positive | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, a depth-2 booster of 50 rounds with an L2 penalty for both outcome learners | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9365 to 0.9571, SE ratio 0.9785 to 1.0442 | pass |
| `interval_calibration` | `rule_t3__crossfit_glm` | positive | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9334 to 0.9546, SE ratio 0.9709 to 1.0366 | pass |
| `interval_calibration` | `rule_t3__estimated_mechanisms` | positive | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9348 to 0.9557, SE ratio 0.9685 to 1.0309 | pass |
| `interval_calibration` | `rule_t3__noise_control` | control | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8074 to 0.8423, SE ratio 0.6904 to 0.7362 | pass |
| `interval_calibration` | `rule_t3__shrunken_se_control` | control | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8129 to 0.8474, SE ratio 0.6777 to 0.7229 | pass |
| `interval_calibration` | `rule_vs_never_t2__estimated_mechanisms` | positive | contrast of the rule against never at horizon t = 2, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9375 to 0.9580, SE ratio 0.9638 to 1.0250 | pass |
| `interval_calibration` | `rule_vs_never_t3__continuous_gaussian` | positive | contrast of the rule against never at horizon t = 3, repeated binary outcome: n = 8,000, the three-value outcome law, least-squares outcome regressions, main-terms logistic mechanisms estimated | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9328 to 0.9540, SE ratio 0.9531 to 1.0179 | pass |
| `interval_calibration` | `rule_vs_never_t3__crossfit_boosted` | positive | contrast of the rule against never at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, a depth-2 booster of 50 rounds with an L2 penalty for both outcome learners | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9406 to 0.9605, SE ratio 0.9742 to 1.0388 | pass |
| `interval_calibration` | `rule_vs_never_t3__crossfit_glm` | positive | contrast of the rule against never at horizon t = 3, repeated binary outcome: n = 8,000, five folds, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9423 to 0.9619, SE ratio 0.9790 to 1.0441 | pass |
| `interval_calibration` | `rule_vs_never_t3__estimated_mechanisms` | positive | contrast of the rule against never at horizon t = 3, repeated binary outcome: n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome regressions, every horizon | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9341 to 0.9552, SE ratio 0.9491 to 1.0115 | pass |
| `interval_calibration` | `rule_vs_never_t3__noise_control` | control | contrast of the rule against never at horizon t = 3, repeated binary outcome: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8074 to 0.8423, SE ratio 0.6715 to 0.7157 | pass |
| `interval_calibration` | `rule_vs_never_t3__shrunken_se_control` | control | contrast of the rule against never at horizon t = 3, repeated binary outcome: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8012 to 0.8366, SE ratio 0.6639 to 0.7075 | pass |
| `outcome_history_necessity` | `always_t3__history_free` | control | mean under always at horizon t = 3, repeated binary outcome: an end-of-study fit with no earlier outcome in any history, cell-mean outcome regressions | bias interval outside the margin, and inside it around the coarse g-formula | bias -0.0300 to -0.0246, margin 0.0083 | pass |
| `outcome_history_necessity` | `always_t3__repeated` | positive | mean under always at horizon t = 3, repeated binary outcome: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.0057 to 0.000801, margin 0.0099 | pass |
| `outcome_history_necessity` | `always_vs_never_t3__history_free` | control | contrast of always against never at horizon t = 3, repeated binary outcome: an end-of-study fit with no earlier outcome in any history, cell-mean outcome regressions | bias interval outside the margin, and inside it around the coarse g-formula | bias -0.0711 to -0.0629, margin 0.0126 | pass |
| `outcome_history_necessity` | `always_vs_never_t3__repeated` | positive | contrast of always against never at horizon t = 3, repeated binary outcome: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.0090 to 0.000074, margin 0.0138 | pass |
| `outcome_history_necessity` | `never_t3__history_free` | control | mean under never at horizon t = 3, repeated binary outcome: an end-of-study fit with no earlier outcome in any history, cell-mean outcome regressions | bias interval outside the margin, and inside it around the coarse g-formula | bias 0.0366 to 0.0428, margin 0.0095 | pass |
| `outcome_history_necessity` | `never_t3__repeated` | positive | mean under never at horizon t = 3, repeated binary outcome: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.0011 to 0.0051, margin 0.0095 | pass |
| `repeated_outcome_necessity` | `always_t2__absorbing_reading` | control | always-treat risk at horizon two: an end-of-study fit of the maximum of the outcomes through the horizon, cell means with a GLM on an unseen cell | bias interval outside the margin, and inside it around the maximum's own truth | bias 0.1589 to 0.1605, margin 0.0025 | pass |
| `repeated_outcome_necessity` | `always_t2__repeated` | positive | always-treat risk at horizon two: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.000708 to 0.0014, margin 0.0032 | pass |
| `repeated_outcome_necessity` | `always_t3__absorbing_reading` | control | mean under always at horizon t = 3, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, cell means with a GLM on an unseen cell | bias interval outside the margin, and inside it around the maximum's own truth | bias 0.2967 to 0.2985, margin 0.0027 | pass |
| `repeated_outcome_necessity` | `always_t3__repeated` | positive | mean under always at horizon t = 3, repeated binary outcome: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.0014 to 0.0018, margin 0.0050 | pass |
| `repeated_outcome_necessity` | `never_t2__absorbing_reading` | control | mean under never at horizon t = 2, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, cell means with a GLM on an unseen cell | bias interval outside the margin, and inside it around the maximum's own truth | bias 0.0871 to 0.0891, margin 0.0030 | pass |
| `repeated_outcome_necessity` | `never_t2__repeated` | positive | mean under never at horizon t = 2, repeated binary outcome: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.0014 to 0.000615, margin 0.0030 | pass |
| `repeated_outcome_necessity` | `never_t3__absorbing_reading` | control | mean under never at horizon t = 3, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, cell means with a GLM on an unseen cell | bias interval outside the margin, and inside it around the maximum's own truth | bias 0.1964 to 0.1993, margin 0.0044 | pass |
| `repeated_outcome_necessity` | `never_t3__repeated` | positive | mean under never at horizon t = 3, repeated binary outcome: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.0015 to 0.0016, margin 0.0047 | pass |
| `repeated_outcome_necessity` | `rule_t2__absorbing_reading` | control | mean under the rule that treats after an outcome at horizon t = 2, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, cell means with a GLM on an unseen cell | bias interval outside the margin, and inside it around the maximum's own truth | bias 0.1583 to 0.1603, margin 0.0030 | pass |
| `repeated_outcome_necessity` | `rule_t2__repeated` | positive | mean under the rule that treats after an outcome at horizon t = 2, repeated binary outcome: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.0012 to 0.0011, margin 0.0036 | pass |
| `repeated_outcome_necessity` | `rule_t3__absorbing_reading` | control | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: an end-of-study fit of the maximum of the outcomes through the horizon, cell means with a GLM on an unseen cell | bias interval outside the margin, and inside it around the maximum's own truth | bias 0.2725 to 0.2754, margin 0.0045 | pass |
| `repeated_outcome_necessity` | `rule_t3__repeated` | positive | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: the repeated-outcome fit, mechanisms estimated, GLM outcome regressions | bias interval inside the equivalence margin | bias -0.0029 to 0.0015, margin 0.0066 | pass |
| `rule_necessity` | `rule_t2__declared` | positive | mean under the rule that treats after an outcome at horizon t = 2, repeated binary outcome: the declared covariate-dependent rule assigns both treatment arms | bias interval inside the equivalence margin | bias -0.0036 to 0.0011, margin 0.0071 | pass |
| `rule_necessity` | `rule_t2__reversed_rule` | control | mean under the rule that treats after an outcome at horizon t = 2, repeated binary outcome: the same fit treats after a zero outcome instead of after an outcome of one | bias interval must fall entirely outside the margin | bias 0.0996 to 0.1031, margin 0.0053 | pass |
| `rule_necessity` | `rule_t3__declared` | positive | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: the declared covariate-dependent rule assigns both treatment arms | bias interval inside the equivalence margin | bias -0.0028 to 0.0061, margin 0.0136 | pass |
| `rule_necessity` | `rule_t3__reversed_rule` | control | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: the same fit treats after a zero outcome instead of after an outcome of one | bias interval must fall entirely outside the margin | bias 0.1151 to 0.1199, margin 0.0073 | pass |
| `simultaneous_coverage` | `all_reported__pointwise_joint_control` | control | every parameter the calibration fit reports: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5831 to 0.6279 | pass |
| `simultaneous_coverage` | `all_reported__simultaneous_band` | positive | every parameter the calibration fit reports: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9314 to 0.9529 | pass |
| `small_sample_calibration` | `always_t3__continuous_gaussian` | diagnostic | mean under always at horizon t = 3, repeated binary outcome: the in-sample continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9189 to 0.9962 | reported |
| `small_sample_calibration` | `always_t3__estimated_mechanisms` | diagnostic | mean under always at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9224 to 1.0001 | reported |
| `small_sample_calibration` | `always_vs_never_t3__continuous_gaussian` | diagnostic | contrast of always against never at horizon t = 3, repeated binary outcome: the in-sample continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9178 to 0.9951 | reported |
| `small_sample_calibration` | `always_vs_never_t3__estimated_mechanisms` | diagnostic | contrast of always against never at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9221 to 1.0035 | reported |
| `small_sample_calibration` | `never_t3__continuous_gaussian` | diagnostic | mean under never at horizon t = 3, repeated binary outcome: the in-sample continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9251 to 1.0048 | reported |
| `small_sample_calibration` | `never_t3__estimated_mechanisms` | diagnostic | mean under never at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9561 to 1.0423 | reported |
| `small_sample_calibration` | `rule_t3__continuous_gaussian` | diagnostic | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: the in-sample continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9102 to 0.9880 | reported |
| `small_sample_calibration` | `rule_t3__estimated_mechanisms` | diagnostic | mean under the rule that treats after an outcome at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.8728 to 0.9445 | reported |
| `small_sample_calibration` | `rule_vs_never_t3__continuous_gaussian` | diagnostic | contrast of the rule against never at horizon t = 3, repeated binary outcome: the in-sample continuous calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.9121 to 0.9894 | reported |
| `small_sample_calibration` | `rule_vs_never_t3__estimated_mechanisms` | diagnostic | contrast of the rule against never at horizon t = 3, repeated binary outcome: the in-sample binary calibration arm at n = 2,000 | reported, no verdict | SE ratio 0.8870 to 0.9629 | reported |
<!-- /generated -->

The property families draw from their own streams of `make_longitudinal_repeated`. Their draws
differ from the primary draws. The table above lists every cell. This section states what each
family tests.

| family | what it tests | size |
| --- | --- | --- |
| `interval_calibration` | pointwise coverage and SE ratio inside the calibration bands. The cells are in-sample with estimated mechanisms, cross-fitted GLM, cross-fitted boosted trees, and a continuous outcome. The in-sample cells cover every horizon, $t = 1, 2, 3$. A shrunken-SE control and a noise control must fail the same bands | $n$ = 8,000, 3,200 replications |
| `small_sample_calibration` | the in-sample and continuous arms refitted at $n$ = 2,000. The role is `diagnostic`. The table reports these cells and no rule gates them | $n$ = 2,000, 2,000 replications |
| `double_robustness` | bias inside the margin when the outcome regression omits every $Y_{t-1}$ and $L_t$ term and the mechanisms are correct. A static plan also has a both-wrong control, which must show bias | $n$ = 2,000, 1,000 replications |
| `simultaneous_coverage` | the default band over all 15 reported parameters, against its pointwise control | $n$ = 8,000, 3,200 replications |
| `rule_necessity` | the declared rule against its reverse, which treats after a zero outcome. The reversed rule must miss the declared rule's truth | $n$ = 2,000, 1,000 replications |
| `repeated_outcome_necessity` (N1) | an absorbing reading of the outcome, which fits $\max_{s \le h} Y_s$. The control must miss the truth and reach the cumulative-maximum limit. Its outcome learner is a cell-means regression with a GLM fallback on an unseen cell | $n$ = 8,000, 1,000 replications |
| `outcome_history_necessity` (N2) | a fit that leaves every $Y_s$ out of the history. The control must miss the truth and reach the coarse g-formula limit | $n$ = 2,000, 1,000 replications |

The in-sample cells use $n$ = 8,000. At $n$ = 2,000 the standard error runs a few percent below
the sampling spread for the rule. The ratio moves toward 1 at the $1/n$ rate of in-sample fitting. The module `tests/studies/ltmle_repeated_properties.py` records the pilot behind that
choice. The `small_sample_calibration` cells report the shortfall at $n$ = 2,000.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 26 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 26 | of those, passing |
| `paired_tests_total` | 13 | paired comparisons with R `ltmle` |
| `paired_tests_passed` | 13 | of those, passing |
| `property_cells_total` | 70 | repeated-sampling property cells, reported cells excluded |
| `property_cells_passed` | 70 | cells whose own and family verdicts pass |
| `property_cells_reported` | 10 | cells reported and not gated, as the `small_sample_calibration` diagnostics |
| `max_standardized_bias` | 0.0496 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9200 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.8852 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0629 | highest bootstrap primary SE-ratio endpoint |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear. No cell of this study reads it |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit. No cell of this study reads it |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit. No cell of this study reads it |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit. No cell of this study reads it |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control multiplies the standard error by |
| `margin:necessity_displacement` | 0.2500 | smallest paired displacement of a necessity control, in empirical standard deviations |

## Limitations

| limit | what it means for use |
| --- | --- |
| one law with three nodes and a binary outcome | the continuous outcome has one calibration arm, and the rate in $n$ is not measured |
| the paired comparison fixes the mechanisms | it isolates the sequential regressions, targeting and inference. It does not claim parity for learned treatment or censoring models |
| the in-sample interval is mildly anticonservative at $n$ = 2,000 | the `small_sample_calibration` cells report the shortfall and no rule gates them. The gated cells draw $n$ = 8,000 |
| the cross-fitted learners are one GLM and one shallow booster | the study does not cover a larger learner library or a different fold count |
| the absorbing and history-free controls read exact limits of this law | they show that the repeated reading and the outcome history matter here. They do not bound the difference on another law |
| positivity is comfortable | no cell validates active truncation or near-positivity behavior |
| no recurrent-event or truncation-by-death law | the parameter is the mean in a population where censoring is prevented |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/manifest.json)
records the pinned R package, source commit, image digest, formulas, seeds and artifact hashes.
The [replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/replicates.csv.gz),
[paired verdicts](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ltmle_repeated/properties.csv)
carry every published row.
