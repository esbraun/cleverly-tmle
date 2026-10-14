# Sequentially doubly robust estimation

This study validates `SDR`, the sequentially doubly robust estimator of Díaz, Williams, Hoffman
and Schenck (2023), Equation (4) and Section 5.3. The
[sequentially doubly robust estimator](../longitudinal-tmle.md#the-sequentially-doubly-robust-estimator)
section states the estimator, its conditions and its exact-law evidence.

The study pairs each fit with pinned `lmtp` 1.5.4: `cf_sdr` and `theta_dr(is_sdr = TRUE)`. Both
sides read the same shifted values, per-node ratios and fold assignment. The study writes them
beside each replicate, so `lmtp` fits no ratio.

## What was compared

| scenario | law and plans | pairing |
| --- | --- | --- |
| `sdr_end_of_study` | the censored two-node law; `never`, `always` and a dynamic plan | one fold |
| `sdr_end_of_study_crossfit` | the same law and plans at five folds | five folds, identical fold assignment |
| `sdr_survival` | the two-node survival law; the same three plans at horizons 1 and 2 | one fold |
| `sdr_mtp` | a truncated-normal dose at both nodes; `up`, `scale at 2` and `up then history` | one fold, exact |
| `sdr_policy` | a categorical treatment under three known stochastic policies, through four copies of each unit | one fold; `mix` is exact |

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 1,000 samples of 2,000 rows per scenario | the identical rows |
| per-node ratio | the generating mechanism, and the oracle binned density of the longitudinal modified-treatment-policy study at a continuous node | the same ratio array, read from the replicate data |
| node regressions | a quasibinomial GLM at the last node and least squares at the earlier node, unclipped | `SL.glm` on the same columns |
| cross-fitting | the training-fold recursion: fold $k$ fits every regression on its training rows | `cf_sdr` on the same folds |
| intervals | pointwise 95% Wald from the influence curve | the same, from `lmtp`'s influence curve |

At five folds the two constructions differ. This package composes each fold's transforms from
that fold's own regressions. `lmtp` stitches each row's later transform from the fold that holds
the row out. The pairing gates (`gates.csv`) therefore hold each scenario to one of two rules.
The `sdr_mtp` rows and `ey_regimen[mix]` are exact pairs, and their largest estimate gap is
3.2e-12. Every other row is a margins pair. All 180 gate rows pass.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-node law with monotone censoring, three plans, one fold | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0014 to 0.0038 | 0.9460 | 0.9826 | pass |
| two-node law with monotone censoring, three plans, one fold | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` | -0.0016 to 0.0036 | 0.9470 | 0.9872 | pass |
| two-node law with monotone censoring, three plans, one fold | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000346 to 0.0049 | 0.9420 | 0.9981 | pass |
| two-node law with monotone censoring, three plans, one fold | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` | -0.000724 to 0.0045 | 0.9510 | 1.0046 | pass |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000780 to 0.0023 | 0.9410 | 1.0122 | pass |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` | -0.000814 to 0.0022 | 0.9450 | 1.0155 | pass |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0025 to 0.0016 | 0.9510 | 0.9909 | pass |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` | -0.0024 to 0.0017 | 0.9570 | 0.9973 | pass |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | 0.000113 to 0.0035 | 0.9410 | 0.9723 | pass |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` | -0.000117 to 0.0032 | 0.9450 | 0.9795 | pass |
| the same law and plans, five folds | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0036 to 0.0017 | 0.9410 | 0.9603 | pass |
| the same law and plans, five folds | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` | -0.0034 to 0.0019 | 0.9400 | 0.9655 | pass |
| the same law and plans, five folds | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0035 to 0.0020 | 0.9410 | 0.9616 | pass |
| the same law and plans, five folds | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` | -0.0034 to 0.0021 | 0.9460 | 0.9681 | pass |
| the same law and plans, five folds | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0016 to 0.0017 | 0.9370 | 0.9435 | pass |
| the same law and plans, five folds | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` | -0.0015 to 0.0019 | 0.9360 | 0.9384 | pass |
| the same law and plans, five folds | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0012 to 0.0031 | 0.9390 | 0.9659 | pass |
| the same law and plans, five folds | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` | -0.0012 to 0.0031 | 0.9410 | 0.9691 | pass |
| the same law and plans, five folds | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0014 to 0.0019 | 0.9500 | 0.9950 | pass |
| the same law and plans, five folds | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` | -0.0014 to 0.0020 | 0.9480 | 0.9922 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000357 to 0.0012 | 0.9520 | 1.0253 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000357 to 0.0012 | 0.9520 | 1.0253 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000551 to 0.000799 | 0.9340 | 1.0070 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000551 to 0.000799 | 0.9340 | 1.0070 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000398 to 0.0010 | 0.9420 | 1.0169 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | R `lmtp` | -0.000398 to 0.0010 | 0.9420 | 1.0169 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000841 to 0.000981 | 0.9590 | 0.9974 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | R `lmtp` | -0.000841 to 0.000981 | 0.9590 | 0.9974 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000689 to 0.0017 | 0.9540 | 1.0287 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | R `lmtp` | -0.000689 to 0.0017 | 0.9540 | 1.0287 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000914 to 0.0013 | 0.9590 | 1.0270 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | R `lmtp` | -0.000914 to 0.0013 | 0.9590 | 1.0270 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000748 to 0.0015 | 0.9580 | 1.0315 | pass |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | R `lmtp` | -0.000748 to 0.0015 | 0.9580 | 1.0315 | pass |
| the three-level law with the plans low, mix and taper | `ate_regimen[mix vs low]` | difference in mean outcome between the plans "draw each arm from a known policy at both times" against "assign the low arm at both times" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0025 to 0.0032 | 0.9350 | 0.9815 | pass |
| the three-level law with the plans low, mix and taper | `ate_regimen[mix vs low]` | difference in mean outcome between the plans "draw each arm from a known policy at both times" against "assign the low arm at both times" | R `lmtp` | -0.0025 to 0.0033 | 0.9410 | 0.9923 | pass |
| the three-level law with the plans low, mix and taper | `ate_regimen[taper vs low]` | difference in mean outcome between the plans "assign the high arm first, then draw from a known policy" against "assign the low arm at both times" | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0037 to 0.0031 | 0.9420 | 0.9809 | pass |
| the three-level law with the plans low, mix and taper | `ate_regimen[taper vs low]` | difference in mean outcome between the plans "assign the high arm first, then draw from a known policy" against "assign the low arm at both times" | R `lmtp` | -0.0034 to 0.0035 | 0.9510 | 0.9926 | pass |
| the three-level law with the plans low, mix and taper | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0035 to 0.0020 | 0.9350 | 0.9868 | pass |
| the three-level law with the plans low, mix and taper | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | R `lmtp` | -0.0037 to 0.0020 | 0.9370 | 0.9948 | pass |
| the three-level law with the plans low, mix and taper | `ey_regimen[mix]` | mean outcome under the plan draw each arm from a known policy at both times | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0016 to 0.000747 | 0.9530 | 1.0011 | pass |
| the three-level law with the plans low, mix and taper | `ey_regimen[mix]` | mean outcome under the plan draw each arm from a known policy at both times | R `lmtp` | -0.0016 to 0.000747 | 0.9530 | 1.0011 | pass |
| the three-level law with the plans low, mix and taper | `ey_regimen[taper]` | mean outcome under the plan assign the high arm first, then draw from a known policy | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0030 to 0.000884 | 0.9520 | 0.9944 | pass |
| the three-level law with the plans low, mix and taper | `ey_regimen[taper]` | mean outcome under the plan assign the high arm first, then draw from a known policy | R `lmtp` | -0.0027 to 0.0011 | 0.9520 | 1.0043 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0012 to 0.0019 | 0.9580 | 1.0086 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | R `lmtp` | -0.0012 to 0.0019 | 0.9580 | 1.0078 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0013 to 0.0035 | 0.9410 | 0.9714 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0012 to 0.0035 | 0.9450 | 0.9817 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0012 to 0.0036 | 0.9460 | 0.9799 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | R `lmtp` | -0.0011 to 0.0037 | 0.9470 | 0.9910 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0013 to 0.000623 | 0.9500 | 1.0081 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | R `lmtp` | -0.0013 to 0.000617 | 0.9520 | 1.0099 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000840 to 0.0021 | 0.9440 | 0.9797 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | R `lmtp` | -0.000805 to 0.0021 | 0.9480 | 0.9830 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0020 to 0.000503 | 0.9450 | 1.0029 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | R `lmtp` | -0.0020 to 0.000509 | 0.9470 | 1.0023 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.0024 to 0.0014 | 0.9450 | 0.9780 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | R `lmtp` | -0.0024 to 0.0014 | 0.9480 | 0.9945 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | `cleverly` sequentially doubly robust (SDR) longitudinal estimator | -0.000754 to 0.0022 | 0.9470 | 1.0169 | pass |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | R `lmtp` | -0.000714 to 0.0022 | 0.9510 | 1.0210 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-node law with monotone censoring, three plans, one fold | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | 0.000180 | 0.0379 | 1.0077 | -0.0010 | 0.0047 vs 0.0500 | equivalent |
| two-node law with monotone censoring, three plans, one fold | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 0.000396 | 0.0829 | 1.0160 | -0.0090 | 0.0148 vs 0.0500 | equivalent |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[always]` | mean outcome under the plan treat at both times | 0.000036 | 0.0129 | 1.0085 | -0.0040 | 0.0114 vs 0.0500 | equivalent |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[never]` | mean outcome under the plan treat at neither time | -0.000144 | 0.0380 | 1.0093 | -0.0060 | 0.0050 vs 0.0500 | equivalent |
| two-node law with monotone censoring, three plans, one fold | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | 0.000252 | 0.0816 | 1.0252 | -0.0040 | 0.0102 vs 0.0500 | equivalent |
| the same law and plans, five folds | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -0.000219 | 0.0447 | 1.0164 | 0.0010 | 0.0067 vs 0.0500 | equivalent |
| the same law and plans, five folds | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | -0.000092 | 0.0183 | 1.0208 | -0.0050 | 0.0070 vs 0.0500 | equivalent |
| the same law and plans, five folds | `ey_regimen[always]` | mean outcome under the plan treat at both times | -0.000165 | 0.0544 | 1.0052 | 0.0010 | 0.0070 vs 0.0500 | equivalent |
| the same law and plans, five folds | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 0.000054 | 0.0138 | 1.0126 | -0.0020 | 0.0056 vs 0.0500 | equivalent |
| the same law and plans, five folds | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | -0.000038 | 0.0123 | 1.0191 | 0.0020 | 0.0140 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[scale at 2 vs natural]` | difference in mean outcome between the plans "multiply the second dose by 1.25, capped at 5.5" against "leave the observed dose unchanged at both times" | -3.687e-16 | 2.577e-13 | 1.0000 | 0 | 3.175e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[up then history vs natural]` | difference in mean outcome between the plans "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3" against "leave the observed dose unchanged at both times" | -2.780e-16 | 2.241e-13 | 1.0000 | 0 | 2.764e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes and four plans | `ate_regimen[up vs natural]` | difference in mean outcome between the plans "add 0.5 to the dose at both times, capped at 5.5" against "leave the observed dose unchanged at both times" | -2.735e-16 | 2.091e-13 | 1.0000 | 0 | 2.265e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[natural]` | mean outcome under the plan leave the observed dose unchanged at both times | 0 | 0 | 1 | 0 | 3.331e-16 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[scale at 2]` | mean outcome under the plan multiply the second dose by 1.25, capped at 5.5 | -3.920e-16 | 1.806e-13 | 1.0000 | 0 | 2.021e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[up then history]` | mean outcome under the plan add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | -2.846e-16 | 1.397e-13 | 1.0000 | 0 | 1.554e-14 vs 0.0500 | equivalent |
| two-node law with a truncated-normal dose at both nodes and four plans | `ey_regimen[up]` | mean outcome under the plan add 0.5 to the dose at both times, capped at 5.5 | -2.820e-16 | 1.355e-13 | 1.0000 | 0 | 1.643e-14 vs 0.0500 | equivalent |
| the three-level law with the plans low, mix and taper | `ate_regimen[mix vs low]` | difference in mean outcome between the plans "draw each arm from a known policy at both times" against "assign the low arm at both times" | -0.000080 | 0.0152 | 0.9893 | -0.0060 | 0.0182 vs 0.0500 | equivalent |
| the three-level law with the plans low, mix and taper | `ate_regimen[taper vs low]` | difference in mean outcome between the plans "assign the high arm first, then draw from a known policy" against "assign the low arm at both times" | -0.000324 | 0.0515 | 0.9955 | -0.0090 | 0.0164 vs 0.0500 | equivalent |
| the three-level law with the plans low, mix and taper | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | 0.000080 | 0.0155 | 0.9763 | -0.0020 | 0.0194 vs 0.0500 | equivalent |
| the three-level law with the plans low, mix and taper | `ey_regimen[mix]` | mean outcome under the plan draw each arm from a known policy at both times | -1.272e-13 | 5.993e-11 | 1.0000 | 0 | 6.405e-11 vs 0.0500 | equivalent |
| the three-level law with the plans low, mix and taper | `ey_regimen[taper]` | mean outcome under the plan assign the high arm first, then draw from a known policy | -0.000244 | 0.0693 | 1.0243 | 0 | 0.0150 vs 0.0500 | equivalent |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[always vs never @ t=1]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 1 | 0.000010 | 0.0034 | 1.0004 | 0 | 0.0013 vs 0.0500 | equivalent |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[always vs never @ t=2]` | difference in cumulative risk between the plans "treat at both times" against "treat at neither time" at horizon t = 2 | -0.000046 | 0.0104 | 1.0135 | -0.0040 | 0.0053 vs 0.0500 | equivalent |
| censored survival law, risks at horizons 1 and 2, one fold | `ate_regimen[treat then continue if l2 positive vs never @ t=2]` | difference in cumulative risk between the plans "treat, then continue only if L2 is positive" against "treat at neither time" at horizon t = 2 | -0.000051 | 0.0114 | 1.0151 | -0.0010 | 0.0054 vs 0.0500 | equivalent |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[always @ t=1]` | cumulative risk under the plan treat at both times at horizon t = 1 | 0.000005 | 0.0027 | 1.0033 | -0.0020 | 0.0049 vs 0.0500 | equivalent |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[always @ t=2]` | cumulative risk under the plan treat at both times at horizon t = 2 | -0.000030 | 0.0112 | 1.0060 | -0.0040 | 0.0028 vs 0.0500 | equivalent |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[never @ t=1]` | cumulative risk under the plan treat at neither time at horizon t = 1 | -0.000005 | 0.0021 | 1.0011 | -0.0020 | 0.0019 vs 0.0500 | equivalent |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[never @ t=2]` | cumulative risk under the plan treat at neither time at horizon t = 2 | 0.000016 | 0.0045 | 1.0208 | -0.0030 | 0.0066 vs 0.0500 | equivalent |
| censored survival law, risks at horizons 1 and 2, one fold | `risk_regimen[treat then continue if l2 positive @ t=2]` | cumulative risk under the plan treat, then continue only if L2 is positive at horizon t = 2 | -0.000035 | 0.0129 | 1.0091 | -0.0040 | 0.0115 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `clustered_calibration` | `clusters_200__known_ratio` | positive | contrast of always against never, 200 clusters of 10 rows: grouped folds, the known ratio and GLM outcome regressions | SE ratio and coverage intervals both inside their calibration bands | SE ratio 0.9699 to 1.0249, coverage 0.9348 to 0.9536 | pass |
| `clustered_calibration` | `clusters_200__noise_control` | control | contrast of always against never, 200 clusters of 10 rows: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | SE ratio 0.6845 to 0.7254, coverage 0.8075 to 0.8388 | pass |
| `clustered_calibration` | `clusters_200__shrunken_se_control` | control | contrast of always against never, 200 clusters of 10 rows: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | SE ratio 0.6777 to 0.7172, coverage 0.8067 to 0.8380 | pass |
| `clustered_calibration` | `clusters_20__known_ratio` | positive | contrast of always against never, 20 clusters of 100 rows: grouped folds, the known ratio and GLM outcome regressions | SE ratio and coverage intervals both inside their calibration bands | SE ratio 0.9511 to 1.0091, coverage 0.9375 to 0.9559 | pass |
| `clustered_calibration` | `clusters_20__noise_control` | control | contrast of always against never, 20 clusters of 100 rows: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | SE ratio 0.6950 to 0.7389, coverage 0.8204 to 0.8508 | pass |
| `clustered_calibration` | `clusters_20__shrunken_se_control` | control | contrast of always against never, 20 clusters of 100 rows: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | SE ratio 0.6660 to 0.7063, coverage 0.7969 to 0.8289 | pass |
| `construction_diagnostic` | `estimated_flexible_mechanism` | diagnostic | the shipped construction with a tree mechanism estimated in each fold, under the default bounds (0.01, 1) | reported only | bias -0.0034 to 0.0015, coverage 0.9650, SE ratio 1.0441 | reported |
| `construction_diagnostic` | `stitched_cross_fit` | diagnostic | the stitched construction: each row's node 2 transform comes from the fold that holds it out, and node 1 regresses the stitched transforms | reported only | bias -0.0029 to 0.0013, coverage 0.9483, SE ratio 0.9856 | reported |
| `crossfit_overfitting` | `cross_fitted_sdr` | positive | five-fold SDR with fully grown outcome trees and the known mechanism | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9739 to 1.0132 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3477 to 0.3622 | pass |
| `interval_calibration` | `competing_h2__correctly_specified` | positive | difference of always and never in the death incidence at t = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9307 to 0.9502, SE ratio 0.9617 to 1.0199, empirical efficiency ratio 1.0135 to 1.0732, reported efficiency ratio 1.0297 to 1.0359 | pass |
| `interval_calibration` | `competing_h2__noise_control` | control | difference of always and never in the death incidence at t = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8235 to 0.8537, SE ratio 0.7040 to 0.7453, empirical efficiency ratio 1.3863 to 1.4659, reported efficiency ratio 1.0297 to 1.0360 | pass |
| `interval_calibration` | `competing_h2__shrunken_se_control` | control | difference of always and never in the death incidence at t = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8059 to 0.8373, SE ratio 0.6736 to 0.7140, empirical efficiency ratio 1.0129 to 1.0727, reported efficiency ratio 0.7207 to 0.7252 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9396 to 0.9578, SE ratio 0.9806 to 1.0394, empirical efficiency ratio 0.9686 to 1.0263, reported efficiency ratio 1.0049 to 1.0077 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8217 to 0.8520, SE ratio 0.6934 to 0.7357, empirical efficiency ratio 1.3679 to 1.4513, reported efficiency ratio 1.0049 to 1.0077 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8139 to 0.8448, SE ratio 0.6863 to 0.7281, empirical efficiency ratio 0.9675 to 1.0261, reported efficiency ratio 0.7034 to 0.7054 | pass |
| `interval_calibration` | `msm_identity__correctly_specified` | positive | the duration coefficient of the identity-link working model: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9350 to 0.9539, SE ratio 0.9683 to 1.0276, empirical efficiency ratio 0.9923 to 1.0523, reported efficiency ratio 1.0170 to 1.0217 | pass |
| `interval_calibration` | `msm_identity__noise_control` | control | the duration coefficient of the identity-link working model: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8245 to 0.8546, SE ratio 0.6990 to 0.7413, empirical efficiency ratio 1.3757 to 1.4584, reported efficiency ratio 1.0171 to 1.0218 | pass |
| `interval_calibration` | `msm_identity__shrunken_se_control` | control | the duration coefficient of the identity-link working model: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8150 to 0.8457, SE ratio 0.6777 to 0.7196, empirical efficiency ratio 0.9919 to 1.0527, reported efficiency ratio 0.7119 to 0.7152 | pass |
| `interval_calibration` | `msm_logit__correctly_specified` | positive | the duration coefficient of the logit-link working model: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9386 to 0.9569, SE ratio 0.9692 to 1.0294, empirical efficiency ratio 1.0011 to 1.0634, reported efficiency ratio 1.0282 to 1.0326 | pass |
| `interval_calibration` | `msm_logit__noise_control` | control | the duration coefficient of the logit-link working model: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8240 to 0.8542, SE ratio 0.6950 to 0.7376, empirical efficiency ratio 1.3970 to 1.4825, reported efficiency ratio 1.0282 to 1.0327 | pass |
| `interval_calibration` | `msm_logit__shrunken_se_control` | control | the duration coefficient of the logit-link working model: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8163 to 0.8469, SE ratio 0.6786 to 0.7201, empirical efficiency ratio 1.0020 to 1.0632, reported efficiency ratio 0.7198 to 0.7228 | pass |
| `interval_calibration` | `mtp_up__correctly_specified` | positive | the shift up at both nodes against the natural course, continuous dose: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9359 to 0.9546, SE ratio 0.9676 to 1.0238, empirical efficiency ratio 1.0026 to 1.0606, reported efficiency ratio 1.0232 to 1.0296 | pass |
| `interval_calibration` | `mtp_up__noise_control` | control | the shift up at both nodes against the natural course, continuous dose: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8276 to 0.8575, SE ratio 0.7004 to 0.7431, empirical efficiency ratio 1.3815 to 1.4652, reported efficiency ratio 1.0232 to 1.0296 | pass |
| `interval_calibration` | `mtp_up__shrunken_se_control` | control | the shift up at both nodes against the natural course, continuous dose: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8090 to 0.8402, SE ratio 0.6767 to 0.7169, empirical efficiency ratio 1.0016 to 1.0619, reported efficiency ratio 0.7162 to 0.7208 | pass |
| `interval_calibration` | `policy_mix__correctly_specified` | positive | contrast of the known policy mix against low, three-level law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9440 to 0.9614, SE ratio 0.9921 to 1.0521, empirical efficiency ratio 0.9617 to 1.0194, reported efficiency ratio 1.0098 to 1.0131 | pass |
| `interval_calibration` | `policy_mix__noise_control` | control | contrast of the known policy mix against low, three-level law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8219 to 0.8522, SE ratio 0.6949 to 0.7358, empirical efficiency ratio 1.3750 to 1.4555, reported efficiency ratio 1.0098 to 1.0131 | pass |
| `interval_calibration` | `policy_mix__shrunken_se_control` | control | contrast of the known policy mix against low, three-level law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8230 to 0.8532, SE ratio 0.6945 to 0.7363, empirical efficiency ratio 0.9619 to 1.0193, reported efficiency ratio 0.7069 to 0.7092 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9318 to 0.9511, SE ratio 0.9637 to 1.0208, empirical efficiency ratio 1.0005 to 1.0589, reported efficiency ratio 1.0189 to 1.0233 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8227 to 0.8530, SE ratio 0.6903 to 0.7336, empirical efficiency ratio 1.3921 to 1.4787, reported efficiency ratio 1.0189 to 1.0233 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8077 to 0.8390, SE ratio 0.6742 to 0.7144, empirical efficiency ratio 1.0011 to 1.0598, reported efficiency ratio 0.7132 to 0.7163 | pass |
| `interval_calibration` | `survival_h2__correctly_specified` | positive | difference of always and never in the risk at t = 2, survival law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9377 to 0.9562, SE ratio 0.9772 to 1.0353, empirical efficiency ratio 1.0022 to 1.0608, reported efficiency ratio 1.0342 to 1.0402 | pass |
| `interval_calibration` | `survival_h2__noise_control` | control | difference of always and never in the risk at t = 2, survival law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8178 to 0.8484, SE ratio 0.7041 to 0.7451, empirical efficiency ratio 1.3925 to 1.4726, reported efficiency ratio 1.0340 to 1.0401 | pass |
| `interval_calibration` | `survival_h2__shrunken_se_control` | control | difference of always and never in the risk at t = 2, survival law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8095 to 0.8407, SE ratio 0.6847 to 0.7251, empirical efficiency ratio 1.0021 to 1.0603, reported efficiency ratio 0.7239 to 0.7281 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9637, 0.9431 to 0.9786 | pass |
| `root_n_and_efficiency` | `static__n_1000` | positive | static plan: bias, coverage and SE calibration at n = 1,000 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000706, coverage 0.9282 to 0.9711, SE ratio 1.0228 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000667, coverage 0.9214 to 0.9665, SE ratio 0.9922 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.0011, coverage 0.9214 to 0.9665, SE ratio 1.0002 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5565 to -0.4633 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5221 to -0.5150 | pass |
| `sdr_tmle_agreement` | `static` | diagnostic | the absolute difference of SDR and LTMLE over the SE of SDR, for always against never, on the calibration draws | reported only | mean absolute difference / SE 0.0087 | reported |
| `sequential_double_robustness` | `P_A__n_2000` | positive | both outcome regressions are correct and both mechanisms are wrong, at n = 2,000 | absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio inside the union-model band | absolute bias + 2 MCSE 0.0069, SE ratio 0.7200, coverage 0.8420 | pass |
| `sequential_double_robustness` | `P_A__n_8000` | positive | both outcome regressions are correct and both mechanisms are wrong, at n = 8,000 | absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio inside the union-model band | absolute bias + 2 MCSE 0.0023, SE ratio 0.7498, coverage 0.8730 | pass |
| `sequential_double_robustness` | `P_B__n_2000` | positive | node 1 outcome regression wrong, node 2 mechanism wrong, at n = 2,000 | absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio inside the union-model band | absolute bias + 2 MCSE 0.0073, SE ratio 0.6910, coverage 0.8160 | pass |
| `sequential_double_robustness` | `P_B__n_8000` | positive | node 1 outcome regression wrong, node 2 mechanism wrong, at n = 8,000 | absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio inside the union-model band | absolute bias + 2 MCSE 0.0023, SE ratio 0.6722, coverage 0.8260 | pass |
| `sequential_double_robustness` | `P_C__n_2000` | positive | both mechanisms are correct and both outcome regressions are wrong, at n = 2,000 | bias inside the margin, SE ratio inside the union-model band, coverage clears the floor | bias -0.0064 to 0.0074, margin 0.0210, coverage 0.9100 to 0.9518, SE ratio 0.9868 | pass |
| `sequential_double_robustness` | `P_C__n_8000` | positive | both mechanisms are correct and both outcome regressions are wrong, at n = 8,000 | bias inside the margin, SE ratio inside the union-model band, coverage clears the floor | bias -0.0014 to 0.0053, margin 0.0103, coverage 0.9307 to 0.9669, SE ratio 1.0106 | pass |
| `sequential_double_robustness` | `P_D__n_2000` | positive | node 1 mechanism wrong, node 2 outcome regression wrong, the pattern LTMLE misses, at n = 2,000 | absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio inside the union-model band | absolute bias + 2 MCSE 0.0083, SE ratio 0.6023, coverage 0.7620 | pass |
| `sequential_double_robustness` | `P_D__n_8000` | positive | node 1 mechanism wrong, node 2 outcome regression wrong, the pattern LTMLE misses, at n = 8,000 | absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio inside the union-model band | absolute bias + 2 MCSE 0.0036, SE ratio 0.5927, coverage 0.7650 | pass |
| `sequential_double_robustness` | `both_correct__n_2000` | positive | every outcome regression and mechanism is correct, at n = 2,000 | bias inside the margin, SE ratio inside the union-model band, coverage clears the floor | bias 0.000011 to 0.0112, margin 0.0172, coverage 0.8853 to 0.9327, SE ratio 0.9459 | **fail** |
| `sequential_double_robustness` | `both_correct__n_8000` | positive | every outcome regression and mechanism is correct, at n = 8,000 | bias inside the margin, SE ratio inside the union-model band, coverage clears the floor | bias -0.0018 to 0.0033, margin 0.0077, coverage 0.9295 to 0.9661, SE ratio 0.9994 | pass |
| `sequential_double_robustness` | `ltmle_P_D_control` | control | LTMLE in pattern P_D, where its recursion needs the node 2 outcome regression | bias interval at least 3 empirical standard deviations from zero, SE ratio inside the union-model band | bias 0.2158 to 0.2235, empirical SD 0.0468, SE ratio 1.0199 | pass |
| `sequential_double_robustness` | `node2_both_wrong_control` | control | SDR with the outcome regression and the mechanism both wrong at node 2 | bias interval at least 3 empirical standard deviations from zero, SE ratio inside the union-model band | bias 0.2158 to 0.2233, empirical SD 0.0456, SE ratio 1.0690 | pass |
| `simultaneous_coverage` | `six_regimens__pointwise_joint_control` | control | the six regimen means and five contrasts of the two-node law: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.6739 to 0.7271 | pass |
| `simultaneous_coverage` | `six_regimens__simultaneous_band` | positive | the six regimen means and five contrasts of the two-node law: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9085 to 0.9394 | **fail** |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0600, 0.0404 to 0.0850 | pass |
| `weighted_calibration` | `selection_weighted__correctly_specified` | positive | contrast of always against never on the selected law, with weights: every nuisance is correct; no efficiency ratio is claimed (RM18-fixed-weights) | SE ratio and coverage intervals both inside their calibration bands | SE ratio 0.9681 to 1.0272, coverage 0.9273 to 0.9472 | pass |
| `weighted_calibration` | `selection_weighted__noise_control` | control | contrast of always against never on the selected law, with weights: one efficiency-bound unit of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | SE ratio 0.6900 to 0.7323, coverage 0.8093 to 0.8404 | pass |
| `weighted_calibration` | `selection_weighted__shrunken_se_control` | control | contrast of always against never on the selected law, with weights: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | SE ratio 0.6777 to 0.7180, coverage 0.8038 to 0.8354 | pass |
<!-- /generated -->

The `sequential_double_robustness` family reads the two-node law of
`tests/discrete_law_longitudinal_sdr.py`. Each of the patterns P_A to P_D makes one nuisance
wrong at each node, and the pattern `both_correct` makes none wrong. The patterns `both_correct`
and `P_C` keep the ratios exact, so they gate the bias and the coverage at both sizes. The
patterns `P_A`, `P_B` and `P_D` gate the absolute bias at n = 2,000 and n = 8,000, and report their
coverage only. Theorem 4 needs the rate product for their intervals. Both controls make the same
nuisances wrong at one node, and their bias clears the floor of 3 empirical SDs.

The `construction_diagnostic` and `sdr_tmle_agreement` cells state no verdict. The stitched cell
fits Step 2 of Section 5.3 as printed. The tree cell estimates the mechanism with trees of at
least 50 rows a leaf, under the default bounds (0.01, 1).

Two property cells fail their own rule, and the diagnosis finds no defect in `SDR`. The
[`X4-finite-sample`](../../roadmap.md#red-cell-owners) owner holds the coverage cell. The
[`band-finite-sample`](../../roadmap.md#red-cell-owners) owner holds the band.

| cell | measured | reading |
| --- | --- | --- |
| `sequential_double_robustness/both_correct__n_2000` | coverage 0.911, 99% interval 0.885 to 0.933, against a floor of 0.90; SE ratio 0.946 | finite-sample, shared with LTMLE |
| `simultaneous_coverage/six_regimens__simultaneous_band` | joint coverage 0.925, 99% interval 0.909 to 0.939, against a band of 0.92 to 0.98 | finite-sample, pointwise shortfall inherited |

**The `both_correct` cell at n = 2,000.** The law's smallest path probability is near 0.012, so
an inverse weight can reach about 85. At n = 2,000 the standard error then varies with the error
it measures. A diagnostic refitted the cell's own fit and `LTMLE` with the same learners on the
same 1,000 fresh draws
([`tests/diagnostics/x4_both_correct/`](https://github.com/esbraun/cleverly-tmle/tree/main/tests/diagnostics/x4_both_correct)).

| engine | coverage | SE ratio | coefficient of variation of the SE | correlation of SE and error |
| --- | ---: | ---: | ---: | ---: |
| `SDR` | 0.903 | 0.918 | 0.186 | 0.301 |
| `LTMLE` | 0.903 | 0.917 | 0.187 | 0.321 |

The two engines cover equally on the same draws. The same cell at n = 8,000 covers 0.950, with
an SE ratio of 0.999. A defect in the `SDR` curve would not be shared with `LTMLE`, and would not
disappear as $n$ grows.

**The band.** The band reads every regimen mean and contrast of the six-plan law. The oracle band
at the exact-law critical value of 2.693 covers 0.931 on the same replications. The package band
covers 0.925 at a mean critical value of 2.683. The shortfall therefore comes from the pointwise
intervals. The other cells that the `band-finite-sample` owner holds read the same way.
`tests/unit/test_sdr_shortfall_reading.py` rebuilds the coverage reading from the committed rows.
`tests/unit/test_band_shortfall_reading.py` rebuilds the band reading.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_passed` | 60 | truth tests passing |
| `independent_tests_total` | 60 | truth tests reported |
| `paired_tests_passed` | 30 | paired comparisons passing |
| `paired_tests_total` | 30 | paired comparisons reported |
| `property_cells_passed` | 54 | property cells passing |
| `property_cells_total` | 56 | property cells reported |
| `property_cells_reported` | 3 | diagnostic cells with no verdict |
| `max_standardized_bias` | 0.0871 | largest primary standardized bias |
| `min_coverage` | 0.9340 | lowest primary coverage |
| `max_margin_utilization` | 0.0829 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
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
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:minimum_power` | 0.8000 | power lower bound |

## Limits

| limit | what it means for use |
| --- | --- |
| finite-support laws with two nodes, and one continuous-dose law | the evidence covers these laws, with plans and policies fixed before the run |
| the mechanism is known in every primary scenario | both sides read the generating ratio. An estimated mechanism appears in the tree diagnostic cell only, which states no verdict |
| parametric node regressions in the primary scenarios | a quasibinomial GLM and least squares. The overfitting family adds fully grown trees with a known mechanism |
| the cross-fitted pairs compare two constructions | `lmtp` stitches Step 2 across folds, and this package runs the training-fold recursion. The pairs are margins pairs, not exactness checks |
| the one-correct patterns report coverage only | an interval under `P_A`, `P_B` or `P_D` needs the Theorem 4 rate product, which a wrong nuisance does not meet |
| two property cells are red under `reporting` | a finite-sample coverage cell at n = 2,000 shared with LTMLE, and a band that inherits its pointwise shortfall. The `X4-finite-sample` and `band-finite-sample` owners hold them |
| clusters, weights and an MSM are calibration cells only | the `clustered_calibration`, `weighted_calibration` and `msm_*` cells have no `lmtp` pairing |
| no multiply robust interval | page 854 of the source states that neither SDR nor TMLE gives one |
