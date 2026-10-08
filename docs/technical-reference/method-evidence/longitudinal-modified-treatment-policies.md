# Longitudinal modified treatment policies

This study validates longitudinal TMLE under modified treatment policies at its nodes. The main
law has two nodes, a dose at each node, a covariate between them, and a binary outcome. The
[modified treatment policies at a node](../longitudinal-tmle.md#modified-treatment-policies-at-a-node)
section states the estimator, its conditions and its exact-law evidence.

The study pairs each fit with pinned `lmtp` 1.5.4. Both sides read the per-node density ratio
that this package computed (`node_ratio`) and the same policy values. The adapter
`tests/canonical/lmtp_mtp_adapter.R` writes both into an `lmtp` task, so `lmtp` fits no ratio.

## What was compared

| scenario | law and plans | pairing |
| --- | --- | --- |
| `mtp_continuous` | a truncated-normal dose at both nodes; `up` (a capped shift at both nodes), `scale at 2`, and `up then history`, whose second node reads the first dose | exact, at one fold |
| `mtp_continuous_crossfit` | the same plans at five folds | `reporting`: the two sides run different fluctuation constructions |
| `mtp_categorical` | six integer levels at both nodes; `minus one` at both nodes and an `L2`-gated variant | exact, at one fold |
| `rr_tilt` | a binary treatment at both nodes; `RiskRatioTilt(0.25)` at both nodes against `lmtp::ipsi(0.25)`, through four copies of each unit | exact, at one fold |

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,000 samples of 2,000 rows per scenario | the identical rows |
| density ratio | an oracle binned density at a continuous node, with `ceil(320 (n / 2000)^(2/3))` bins (320 at n = 2,000), and the generating probabilities at a categorical node | the same ratio array, read from the replicate data |
| node regressions | a quasibinomial GLM at the last node and least squares at the first | `SL.glm` on the same columns |
| intervals | pointwise 95% Wald from the influence curve | the same, from `lmtp`'s influence curve |

At five folds `lmtp` fits each node's fluctuation on the training rows of each fold, and this
package solves one pooled fluctuation per node. The cross-fitted pairs were declared `reporting`
for that reason. All of them pass.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-node law with six integer treatment levels at both nodes | `ate_regimen[gated vs natural]` | difference in mean outcome between the plans "lower the second level by one where L2 equals one" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.000178 to 0.000568 | 0.9420 | 0.9673 | pass |
| two-node law with six integer treatment levels at both nodes | `ate_regimen[gated vs natural]` | difference in mean outcome between the plans "lower the second level by one where L2 equals one" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000178 to 0.000568 | 0.9420 | 0.9673 | pass |
| two-node law with six integer treatment levels at both nodes | `ate_regimen[minus one vs natural]` | difference in mean outcome between the plans "lower the level by one at both times wherever it stays at least one" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.0011 to 0.000543 | 0.9600 | 1.0383 | pass |
| two-node law with six integer treatment levels at both nodes | `ate_regimen[minus one vs natural]` | difference in mean outcome between the plans "lower the level by one at both times wherever it stays at least one" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0011 to 0.000543 | 0.9600 | 1.0383 | pass |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[gated]` | mean outcome under the plan lower the second level by one where L2 equals one | `cleverly` LTMLE under modified treatment policies | -0.000791 to 0.0012 | 0.9510 | 0.9998 | pass |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[gated]` | mean outcome under the plan lower the second level by one where L2 equals one | R `lmtp` | -0.000791 to 0.0012 | 0.9510 | 0.9998 | pass |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[minus one]` | mean outcome under the plan lower the level by one at both times wherever it stays at least one | `cleverly` LTMLE under modified treatment policies | -0.0015 to 0.000938 | 0.9570 | 1.0348 | pass |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[minus one]` | mean outcome under the plan lower the level by one at both times wherever it stays at least one | R `lmtp` | -0.0015 to 0.000938 | 0.9570 | 1.0348 | pass |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` LTMLE under modified treatment policies | -0.000893 to 0.000896 | 0.9540 | 1.0170 | pass |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.000893 to 0.000896 | 0.9540 | 1.0170 | pass |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.0010 to 0.000494 | 0.9530 | 1.0304 | pass |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0010 to 0.000494 | 0.9530 | 1.0304 | pass |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.0010 to 0.000372 | 0.9480 | 0.9833 | pass |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0010 to 0.000372 | 0.9480 | 0.9833 | pass |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.0012 to 0.000169 | 0.9540 | 1.0206 | pass |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0012 to 0.000169 | 0.9540 | 1.0206 | pass |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` LTMLE under modified treatment policies | -0.000486 to 0.0014 | 0.9380 | 0.9888 | pass |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.000486 to 0.0014 | 0.9380 | 0.9888 | pass |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | `cleverly` LTMLE under modified treatment policies | -0.0011 to 0.0014 | 0.9410 | 0.9671 | pass |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | R `lmtp` | -0.0011 to 0.0014 | 0.9410 | 0.9671 | pass |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | `cleverly` LTMLE under modified treatment policies | -0.0010 to 0.0013 | 0.9440 | 0.9801 | pass |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | R `lmtp` | -0.0010 to 0.0013 | 0.9440 | 0.9801 | pass |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | `cleverly` LTMLE under modified treatment policies | -0.0013 to 0.0011 | 0.9380 | 0.9914 | pass |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | R `lmtp` | -0.0013 to 0.0011 | 0.9380 | 0.9914 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.000431 to 0.0012 | 0.9460 | 0.9969 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000400 to 0.0012 | 0.9510 | 1.0048 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.0012 to 0.000141 | 0.9530 | 0.9984 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0012 to 0.000166 | 0.9520 | 1.0021 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.0012 to 0.000260 | 0.9510 | 1.0034 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.0012 to 0.000281 | 0.9530 | 1.0066 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` LTMLE under modified treatment policies | -0.0013 to 0.000411 | 0.9560 | 1.0605 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.0013 to 0.000427 | 0.9560 | 1.0599 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | `cleverly` LTMLE under modified treatment policies | -0.0012 to 0.0011 | 0.9450 | 1.0346 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | R `lmtp` | -0.0012 to 0.0011 | 0.9470 | 1.0377 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | `cleverly` LTMLE under modified treatment policies | -0.0021 to 0.000115 | 0.9590 | 1.0334 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | R `lmtp` | -0.0020 to 0.000156 | 0.9590 | 1.0348 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | `cleverly` LTMLE under modified treatment policies | -0.0020 to 0.000224 | 0.9520 | 1.0318 | pass |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | R `lmtp` | -0.0020 to 0.000262 | 0.9520 | 1.0329 | pass |
| two-node law with a binary treatment at both nodes | `ate_regimen[rr 0.25 vs natural]` | difference in mean outcome between the plans "keep each treated unit treated with probability 0.25 at both times" against "leave the observed dose unchanged at both times" | `cleverly` LTMLE under modified treatment policies | -0.000933 to 0.0012 | 0.9540 | 1.0076 | pass |
| two-node law with a binary treatment at both nodes | `ate_regimen[rr 0.25 vs natural]` | difference in mean outcome between the plans "keep each treated unit treated with probability 0.25 at both times" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000933 to 0.0012 | 0.9540 | 1.0076 | pass |
| two-node law with a binary treatment at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` LTMLE under modified treatment policies | -0.0014 to 0.000463 | 0.9630 | 0.9996 | pass |
| two-node law with a binary treatment at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.0014 to 0.000463 | 0.9630 | 0.9996 | pass |
| two-node law with a binary treatment at both nodes | `ey_regimen[rr 0.25]` | mean outcome under the plan keep each treated unit treated with probability 0.25 at both times | `cleverly` LTMLE under modified treatment policies | -0.0017 to 0.0011 | 0.9570 | 1.0125 | pass |
| two-node law with a binary treatment at both nodes | `ey_regimen[rr 0.25]` | mean outcome under the plan keep each treated unit treated with probability 0.25 at both times | R `lmtp` | -0.0017 to 0.0011 | 0.9570 | 1.0125 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-node law with six integer treatment levels at both nodes | `ate_regimen[gated vs natural]` | difference in mean outcome between the plans "lower the second level by one where L2 equals one" against "leave the observed dose unchanged at both times" | -1.412e-12 | 2.061e-09 | 1.0000 | 0 | 3.612e-10 vs 0.0500 | equivalent |
| two-node law with six integer treatment levels at both nodes | `ate_regimen[minus one vs natural]` | difference in mean outcome between the plans "lower the level by one at both times wherever it stays at least one" against "leave the observed dose unchanged at both times" | -1.196e-10 | 8.052e-08 | 1.0000 | 0 | 1.324e-09 vs 0.0500 | equivalent |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[gated]` | mean outcome under the plan lower the second level by one where L2 equals one | 1.181e-12 | 6.508e-10 | 1.0000 | 0 | 7.578e-11 vs 0.0500 | equivalent |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[minus one]` | mean outcome under the plan lower the level by one at both times wherever it stays at least one | -1.170e-10 | 5.294e-08 | 1.0000 | 0 | 7.665e-10 vs 0.0500 | equivalent |
| two-node law with six integer treatment levels at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | 2.593e-12 | 1.577e-09 | 1.0000 | 0 | 1.848e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | 8.156e-12 | 5.752e-09 | 1.0000 | 0 | 9.125e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | 7.251e-11 | 5.712e-08 | 1.0000 | 0 | 7.418e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | 8.362e-11 | 6.420e-08 | 1.0000 | 0 | 1.373e-09 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | -8.757e-12 | 5.182e-09 | 1.0000 | 0 | 1.533e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | -6.011e-13 | 2.610e-10 | 1.0000 | 0 | 1.932e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | 6.376e-11 | 2.988e-08 | 1.0000 | 0 | 5.327e-10 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | 7.486e-11 | 3.459e-08 | 1.0000 | 0 | 4.525e-10 vs 0.0500 | equivalent |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | -0.000026 | 0.0178 | 1.0089 | -0.0050 | 0.0119 vs 0.0500 | equivalent |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | -0.000027 | 0.0216 | 1.0055 | 0.0010 | 0.0061 vs 0.0500 | equivalent |
| the two-node continuous-dose law, cross-fitted over five folds | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | -0.000023 | 0.0174 | 1.0051 | -0.0020 | 0.0082 vs 0.0500 | equivalent |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | -0.000015 | 0.0096 | 1.0010 | 0 | 0.0016 vs 0.0500 | equivalent |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | -0.000042 | 0.0192 | 1.0044 | -0.0020 | 0.0069 vs 0.0500 | equivalent |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | -0.000042 | 0.0208 | 1.0033 | 0 | 0.0037 vs 0.0500 | equivalent |
| the two-node continuous-dose law, cross-fitted over five folds | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | -0.000038 | 0.0184 | 1.0029 | 0 | 0.0032 vs 0.0500 | equivalent |
| two-node law with a binary treatment at both nodes | `ate_regimen[rr 0.25 vs natural]` | difference in mean outcome between the plans "keep each treated unit treated with probability 0.25 at both times" against "leave the observed dose unchanged at both times" | -1.735e-11 | 8.713e-09 | 1.0000 | 0 | 9.760e-10 vs 0.0500 | equivalent |
| two-node law with a binary treatment at both nodes | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | -6.071e-13 | 3.622e-10 | 1.0000 | 0 | 7.463e-11 vs 0.0500 | equivalent |
| two-node law with a binary treatment at both nodes | `ey_regimen[rr 0.25]` | mean outcome under the plan keep each treated unit treated with probability 0.25 at both times | -1.796e-11 | 7.226e-09 | 1.0000 | 0 | 1.140e-09 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_mtp_ltmle` | positive | five-fold policy LTMLE with interpolating extra-trees outcome learners | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.1460 to 1.1877 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.6634 to 0.6882 | pass |
| `double_robustness` | `up__both_correct` | positive | the contrast of the shift up at both nodes against the natural course: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000553 to 0.000877, margin 0.0022, SE ratio 1.0070 | pass |
| `double_robustness` | `up__both_wrong` | control | the contrast of the shift up at both nodes against the natural course: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0343 to 0.0363, margin 0.0031, SE ratio 1.1475 | pass |
| `double_robustness` | `up__mechanism_correct` | positive | the contrast of the shift up at both nodes against the natural course: only the mechanisms are correctly specified, read from 320 oracle bins so the binned density's own first-order error stays below the margin | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000664 to 0.000930, margin 0.0024, SE ratio 0.9574 | pass |
| `double_robustness` | `up__outcome_correct` | positive | the contrast of the shift up at both nodes against the natural course: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000912 to 0.000722, margin 0.0025, SE ratio 1.3318 | pass |
| `interval_calibration` | `categorical_mtp__correctly_specified` | positive | the minus-one policy contrast on the six-level law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9289 to 0.9560, SE ratio 0.9432 to 1.0214, empirical efficiency ratio 1.0746 to 1.1643, reported efficiency ratio 1.0948 to 1.1006 | **fail** |
| `interval_calibration` | `categorical_mtp__noise_control` | control | the minus-one policy contrast on the six-level law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8162 to 0.8591, SE ratio 0.6851 to 0.7431, empirical efficiency ratio 1.4775 to 1.6020, reported efficiency ratio 1.0949 to 1.1005 | pass |
| `interval_calibration` | `categorical_mtp__shrunken_se_control` | control | the minus-one policy contrast on the six-level law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8021 to 0.8463, SE ratio 0.6603 to 0.7135, empirical efficiency ratio 1.0765 to 1.1638, reported efficiency ratio 0.7664 to 0.7704 | pass |
| `interval_calibration` | `classifier_route__correctly_specified` | positive | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9233 to 0.9515, SE ratio 0.9312 to 1.0108, empirical efficiency ratio 0.9887 to 1.0730, reported efficiency ratio 0.9954 to 1.0034 | pass |
| `interval_calibration` | `classifier_route__noise_control` | control | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7864 to 0.8321, SE ratio 0.6529 to 0.7083, empirical efficiency ratio 1.4118 to 1.5297, reported efficiency ratio 0.9956 to 1.0033 | pass |
| `interval_calibration` | `classifier_route__shrunken_se_control` | control | the primary policy contrast, its ratio estimated by a classifier on the true log ratio: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7896 to 0.8349, SE ratio 0.6521 to 0.7065, empirical efficiency ratio 0.9905 to 1.0729, reported efficiency ratio 0.6968 to 0.7025 | pass |
| `interval_calibration` | `msm_mtp__correctly_specified` | positive | the dose coefficient of a working model over three continuous plans: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9333 to 0.9595, SE ratio 0.9681 to 1.0510, empirical efficiency ratio 0.9629 to 1.0444, reported efficiency ratio 1.0068 to 1.0164 | pass |
| `interval_calibration` | `msm_mtp__noise_control` | control | the dose coefficient of a working model over three continuous plans: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8199 to 0.8624, SE ratio 0.6894 to 0.7494, empirical efficiency ratio 1.3493 to 1.4661, reported efficiency ratio 1.0068 to 1.0163 | pass |
| `interval_calibration` | `msm_mtp__shrunken_se_control` | control | the dose coefficient of a working model over three continuous plans: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7995 to 0.8440, SE ratio 0.6780 to 0.7351, empirical efficiency ratio 0.9638 to 1.0438, reported efficiency ratio 0.7048 to 0.7113 | pass |
| `interval_calibration` | `randomized_mtp__correctly_specified` | positive | the contrast of a randomized node-2 shift against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9316 to 0.9582, SE ratio 0.9566 to 1.0371, empirical efficiency ratio 0.9722 to 1.0534, reported efficiency ratio 1.0059 to 1.0110 | pass |
| `interval_calibration` | `randomized_mtp__noise_control` | control | the contrast of a randomized node-2 shift against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8189 to 0.8615, SE ratio 0.6907 to 0.7482, empirical efficiency ratio 1.3480 to 1.4598, reported efficiency ratio 1.0059 to 1.0111 | pass |
| `interval_calibration` | `randomized_mtp__shrunken_se_control` | control | the contrast of a randomized node-2 shift against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7979 to 0.8425, SE ratio 0.6692 to 0.7255, empirical efficiency ratio 0.9736 to 1.0542, reported efficiency ratio 0.7041 to 0.7078 | pass |
| `interval_calibration` | `survival_mtp_h2__correctly_specified` | positive | the cumulative risk at t = 2 of a policy plan on the survival law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9361 to 0.9617, SE ratio 0.9636 to 1.0422, empirical efficiency ratio 0.9590 to 1.0377, reported efficiency ratio 0.9979 to 1.0015 | pass |
| `interval_calibration` | `survival_mtp_h2__noise_control` | control | the cumulative risk at t = 2 of a policy plan on the survival law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8178 to 0.8605, SE ratio 0.6821 to 0.7414, empirical efficiency ratio 1.3482 to 1.4658, reported efficiency ratio 0.9979 to 1.0014 | pass |
| `interval_calibration` | `survival_mtp_h2__shrunken_se_control` | control | the cumulative risk at t = 2 of a policy plan on the survival law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8052 to 0.8492, SE ratio 0.6737 to 0.7285, empirical efficiency ratio 0.9601 to 1.0388, reported efficiency ratio 0.6985 to 0.7010 | pass |
| `interval_calibration` | `up__correctly_specified` | positive | the contrast of the shift up at both nodes against the natural course: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9366 to 0.9622, SE ratio 0.9850 to 1.0690, empirical efficiency ratio 0.9481 to 1.0271, reported efficiency ratio 1.0077 to 1.0169 | pass |
| `interval_calibration` | `up__noise_control` | control | the contrast of the shift up at both nodes against the natural course: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8189 to 0.8615, SE ratio 0.6960 to 0.7556, empirical efficiency ratio 1.3411 to 1.4535, reported efficiency ratio 1.0076 to 1.0169 | pass |
| `interval_calibration` | `up__shrunken_se_control` | control | the contrast of the shift up at both nodes against the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8157 to 0.8586, SE ratio 0.6880 to 0.7481, empirical efficiency ratio 0.9474 to 1.0296, reported efficiency ratio 0.7053 to 0.7119 | pass |
| `interval_calibration` | `vector_node__correctly_specified` | positive | the vector-node policy contrast on two binary components: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9266 to 0.9542, SE ratio 0.9469 to 1.0295, empirical efficiency ratio 0.9723 to 1.0577, reported efficiency ratio 0.9984 to 1.0042 | pass |
| `interval_calibration` | `vector_node__noise_control` | control | the vector-node policy contrast on two binary components: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8026 to 0.8468, SE ratio 0.6822 to 0.7362, empirical efficiency ratio 1.3602 to 1.4679, reported efficiency ratio 0.9985 to 1.0043 | pass |
| `interval_calibration` | `vector_node__shrunken_se_control` | control | the vector-node policy contrast on two binary components: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8021 to 0.8463, SE ratio 0.6626 to 0.7193, empirical efficiency ratio 0.9740 to 1.0576, reported efficiency ratio 0.6990 to 0.7029 | pass |
| `inverse_necessity` | `history__declared_inverse` | positive | the contrast of the history-reading plan against the natural course: the density ratio reads each piece at its declared inverse, at 320 oracle bins | bias interval inside the equivalence margin | bias -0.000089 to 0.0014, margin 0.0022 | pass |
| `inverse_necessity` | `history__inverse_dropped_control` | control | the contrast of the history-reading plan against the natural course: the same fit reads each moving piece at the dose itself, at 320 oracle bins | bias interval must fall entirely outside the margin | bias -0.0176 to -0.0166, margin 0.0015 | pass |
| `power` | `up__alternative` | positive | the contrast of the shift up at both nodes against the natural course: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9912 to 1 | pass |
| `root_n_and_efficiency` | `up__n_2000` | positive | the contrast of the shift up at both nodes against the natural course: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000381, coverage 0.9046 to 0.9582, SE ratio 0.9769 | pass |
| `root_n_and_efficiency` | `up__n_500` | control | the contrast of the shift up at both nodes against the natural course: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.000868, coverage 0.9007 to 0.9555, SE ratio 0.9537 | pass |
| `root_n_and_efficiency` | `up__n_8000` | positive | the contrast of the shift up at both nodes against the natural course: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000215, coverage 0.9224 to 0.9701, SE ratio 1.0040 | pass |
| `root_n_rate` | `up__empirical_sd` | positive | the contrast of the shift up at both nodes against the natural course: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5454 to -0.4720 | pass |
| `root_n_rate` | `up__reported_se` | positive | the contrast of the shift up at both nodes against the natural course: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4945 to -0.4835 | pass |
| `simultaneous_coverage` | `primary__pointwise_joint_control` | control | every primary estimand of the study: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.7906 to 0.8359 | pass |
| `simultaneous_coverage` | `primary__simultaneous_band` | positive | every primary estimand of the study: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9266 to 0.9542 | pass |
| `targeting_necessity` | `up__targeted` | positive | the contrast of the shift up at both nodes against the natural course: the targeted fit with only the mechanisms correct, at 320 oracle bins | bias interval inside the equivalence margin | bias -0.000990 to 0.000540, margin 0.0023 | pass |
| `targeting_necessity` | `up__untargeted` | control | the contrast of the shift up at both nodes against the natural course: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0781 to -0.0781, margin 0 | pass |
| `type_i_error` | `up__sharp_null` | positive | the contrast of the shift up at both nodes against the natural course: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0517, 0.0312 to 0.0796 | pass |
<!-- /generated -->

The continuous truths come from Gauss-Legendre quadrature, checked at twice the order and against
a simulation of Definition 1 of Díaz, Williams, Hoffman and Schenck (2023). The finite-law
truths and efficiency bounds come from the exact laws and their complex-step derivatives.

The oracle's bin count grows as $n^{2/3}$. A first run at a fixed 80 bins found that a binned
density ratio is not consistent at a fixed count, so the package default now grows with $n$
and this study re-ran under a fresh declaration. At a growing count the ratio error falls
faster than $n^{-1/2}$, which the cells with a wrong outcome regression need. The
[bin count paragraph](../longitudinal-tmle.md#modified-treatment-policies-at-a-node) gives the
measurements.

`classifier_route` estimates the ratio by stacked classification, with a logistic regression on
the true log ratio, so the classifier is correctly specified. `inverse_necessity` fits `up then
history` twice on one sample. The positive arm reads Equation (3) at the declared inverse, and
the control reads it at the dose itself.

One property cell fails its own rule, and the diagnosis finds no defect. The owner row
`mtp-longitudinal-limits` holds it.

| cell | measured | reading |
| --- | --- | --- |
| `interval_calibration/categorical_mtp__correctly_specified` | efficiency ratios 1.120 and 1.098 against a band of 0.9 to 1.1; coverage 0.9435; SE ratio 0.980 | finite-sample, sparse cells |

The cell fits saturated cell probabilities for the six-level mechanism. Both outcome regressions
are misspecified, although the cell's stream is named `correctly_specified`. At node 2 a logistic
GLM fits a linear probability. At node 1 least squares fits a regression that is nonlinear in the
first dose. The estimate does not depend on them: with a saturated mechanism on a finite law, the
fit equals the NPMLE whatever the outcome learner. A diagnostic refitted the cell on fresh draws
([`tests/diagnostics/longitudinal_mtp_categorical_efficiency/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/longitudinal_mtp_categorical_efficiency)).

| fit | reported efficiency ratio, n = 2,000 | the same, n = 8,000 |
| --- | ---: | ---: |
| in sample, as declared | 1.103 | 1.023 |
| five folds | 1.389 | 1.048 |

The reported ratio's excess falls about four times as $n$ grows four times. That is the order of
the cost of estimating sparse cell probabilities. At larger sizes the reported ratio reads 1.019
at n = 8,000, 1.005 at n = 32,000 and 1.003 at n = 128,000, on 20 draws each (`large_n.log` in
the same directory). A wrong bound or an inefficient curve would keep its excess as $n$ grows.

The empirical ratio, 1.107 and 1.052 on the diagnostic's draws, is too noisy to show this alone.
It reads the same estimates as an NPMLE, which is efficient at this bound. The cell read the same
in runs 2 and 3.

**The overfitting pair.** Run 2 published the pair red. A single fully grown tree returns one
training outcome at a shifted dose, so the in-sample curve kept the outcome noise, and the
control could not reach its margins. Run 3 fits an interpolating extra-trees ensemble in both
arms, with the margins unchanged. The control reads an SE ratio of 0.675 and the cross-fitted arm
1.166, and the pair passes. The pre-run probe and the design screen are in
[`tests/diagnostics/longitudinal_mtp_overfit_design/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/longitudinal_mtp_overfit_design).

Part of the control's coverage loss comes from its bias, about 0.89 of its empirical SD, because
the in-sample fit is the ensemble's plug-in. The SE-ratio clause measures the understated
standard error directly.

Before the run, the probe put the cross-fitted arm's pass probability at about 0.23, or about 0.4
with the screen's draws. Its SE ratio sits near the top of the 0.8 to 1.2 band, and more
replications cannot move a structurally conservative ratio. The declaration routed a red there
to `reporting`. The run read 1.166, inside the band.

A fully grown tree cannot learn the dose hazard of the pair. On two draws, trees with leaves of 5
to 200 rows gave cross-fitted standard errors of 1e17 or more, and a tree with leaves of one row
gave none (`checks.jsonl`). The in-sample control's standard error stayed between 0.0206 and
0.0211, because the control multiplies the ratio by zero residuals. So learning the hazard could
not make the control discriminate.

The study was declared `gated`. By the red-cell rule it moved to `reporting` before a repeat run
of run 2. Run 3 kept the categorical cell's declared route.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_passed` | 44 | truth tests passing |
| `independent_tests_total` | 44 | truth tests reported |
| `paired_tests_passed` | 22 | paired comparisons passing |
| `paired_tests_total` | 22 | paired comparisons reported |
| `property_cells_passed` | 39 | property cells passing |
| `property_cells_total` | 40 | property cells reported |
| `max_standardized_bias` | 0.0731 | largest primary standardized bias |
| `min_coverage` | 0.9380 | lowest primary coverage |
| `max_margin_utilization` | 0.0216 | largest paired similarity-margin share |
| `properties[crossfit_overfitting/cross_fitted_mtp_ltmle]:coverage` | 0.9743 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.6455 | in-sample tree coverage |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
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
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

## Limits

| limit | what it means for use |
| --- | --- |
| one continuous law, one categorical law and one binary law, each with two nodes | the evidence covers these laws with policies fixed before the run |
| the density is an oracle | a continuous node reads the generating density through a bin count that grows with $n$. The default estimated density is not tested here |
| the cross-fitted pairs compare two constructions | `lmtp` 1.5.4 fluctuates on the training rows of each fold. The pairs pass, but they are not exactness checks |
| one property cell is red under `reporting` | a finite-sample categorical calibration cell with sparse cells. The `mtp-longitudinal-limits` owner holds it |
| the tilt pairing copies each unit four times | the pairing needs every branch weight on a grid of one quarter |
| a vector node with a continuous component is refused | [X31](../../roadmap.md#x31-continuous-vector-components-at-a-node) owns it |
| a policy must be known and fixed | a policy learned from the sample, or one that depends on the law, is outside this evidence |
