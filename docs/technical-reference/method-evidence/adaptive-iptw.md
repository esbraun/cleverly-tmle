# Adaptive IPTW

This study validates `AdaptiveIPTW`, the Horvitz-Thompson mean whose propensity is targeted along
$H = Q_r / g$. The reduced regression $Q_r$ is the regression of the outcome on the propensity.
The [contract page](../adaptive-iptw.md) states the estimator, its conditions, and the
[rule for an interval](../adaptive-iptw.md#when-an-interval-is-reported). The estimator reports an
interval only when the whole treatment mechanism is declared known. An estimated mechanism gives a
point estimate with the status `estimated_mechanism_plugin`.

So the study reads two kinds of cell. A **calibration** cell reads a known-mechanism fit against
the SE-ratio and coverage bands. A **bias** cell reads an estimated-mechanism fit against its
standardized-bias margin. A bias row records the plug-in standard error and interval in the
`std_error` and `covered` columns. No verdict outside the `spread` cell reads those columns.

The declared comparator is R [`drtmle`](https://github.com/benkeser/drtmle) 1.1.2, function
`adaptive_iptw`. Its runs are pending. The separate study `adaptive-iptw-pairing` will run them on
its own scenarios and seeds, so this page holds Python property evidence, with comparator runs
pending.

## What was compared

The R column states the declared design of the pending comparator runs.

| setting | `cleverly` | R `drtmle` |
| --- | --- | --- |
| datasets | 2,000 samples for each known-mechanism cell, and 400 to 3,000 samples for each bias cell, from the laws below. Every scene draws its own stream | the identical rows, in the pairing study |
| folds | `cross_fit=True` with 10 folds (5 on law M), drawn by the package's random partition from the replication seed. The in-sample cells use no folds | `cvFolds = 10` with the folds that R draws and the pairing study passes to Python |
| reduced regression | an unpenalized polynomial of $\operatorname{logit} g$ with a degree for each law, a logistic fit for an indicator key and least squares for a stochastic-regime key. Each fold fits it on leave-two-folds-out designs | `glm_Qr` with the same formula |
| targeting | the relative score tolerance $10^{-8}$ and at most 100 rounds. $g$ is clipped to $[0.01, 0.99]$ | `tolIC = 1e-8` and `maxIter = 100` |
| intervals | pointwise 95% Wald from the influence curve, for a known mechanism only | the same, for a known `gn` |

The laws are exact, so every truth is a finite sum or a quadrature.

| law | construction | $Q_0 = Q_{r,0}$ | fits |
| --- | --- | --- | --- |
| P | Benkeser, Carone, van der Laan and Gilbert (2017): $W_1 \sim U(-2, 2)$, $W_2 \sim \mathrm{Bern}(0.5)$, $\operatorname{logit} g = -W_1 + 2 W_1 W_2$, $\operatorname{logit} Q = 0.2 A - W_1 + 2 W_1 W_2$ | yes, both arms | cross-fitted and in-sample point fits with a correct, a `rate`, a wrong and a gradient-boosting mechanism, at $n$ = 3,000. A `rate` ladder at $n$ = 500, 1,500 and 4,500 |
| PD | law P's covariates and mechanism, with $\operatorname{logit} Q(a, W) = -W_1 + 2 W_1 W_2 + a (0.2 + W_1)$ | arm 0 only | the known mechanism on the arm and regime axes, and a `rate_scaled` mechanism on the regime axis, at $n$ = 2,000. The regimes are never treat, the rule $1\{W_1 > 0\}$ and the stochastic regime $\operatorname{expit}(W_1)$ |
| M | the three-arm law of the multi-arm studies, five folds | no | a correct and a shrunk multinomial mechanism, at $n$ = 2,000. The shrinkage keeps each arm's level sets |
| C | the binary and three-arm composite laws with a missing outcome. The oracle mechanisms are declared as estimated | yes | cross-fitted and in-sample fits, at $n$ = 2,000 |
| CN, RM | the binary composite law: the natural course, and a rule and a stochastic regime with a missing outcome | yes | the natural-course keys at $n$ = 2,000, and the regimes in sample at $n$ = 3,000 |
| MA | a binary law with a missing treatment and a missing outcome | yes | in sample at $n$ = 2,000, and cross-fitted at $n$ = 7,000 |
| WT, CL, ST | fixed known design weights, unequal clusters (200 clusters), and declared baseline strata | yes | the known mechanism, and the correct estimated mechanism on the first 400 draws |
| VL1, VL0 | $W_1, W_2 \sim N(0, 1)$ and $g_0 = \operatorname{expit}(0.6 W_1 + 0.3 W_2)$. VL1 has $Q(1, W) = \operatorname{expit}(-0.2 - W_1 + 2 W_2)$ and VL0 has $Q(1, W) = \operatorname{expit}(0.1 + 1.5 (0.6 W_1 + 0.3 W_2))$ | VL1 no, VL0 yes | a logistic mechanism, and a mechanism that adds $n^{-1/3} W_2$ to the logit and so breaks the level sets of $g_0$, at $n$ = 2,000 |

## Accuracy against known truth

The primary cells fit law PD with the known mechanism and read the arm means and the ATE.

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| effect-modified two-arm law with a continuous and a binary covariate; the cross-fitted targeted fit of the arm means and the ATE with the treatment mechanism declared known | `ate` | average treatment effect | `cleverly` `AdaptiveIPTW`, the Horvitz-Thompson mean with a targeted propensity | -0.000984 to 0.0017 | 0.9530 | 1.0175 | pass |
| effect-modified two-arm law with a continuous and a binary covariate; the cross-fitted targeted fit of the arm means and the ATE with the treatment mechanism declared known | `ey0` | counterfactual mean under no treatment | `cleverly` `AdaptiveIPTW`, the Horvitz-Thompson mean with a targeted propensity | -0.0015 to 0.000390 | 0.9510 | 1.0000 | pass |
| effect-modified two-arm law with a continuous and a binary covariate; the cross-fitted targeted fit of the arm means and the ATE with the treatment mechanism declared known | `ey1` | counterfactual mean under treatment | `cleverly` `AdaptiveIPTW`, the Horvitz-Thompson mean with a targeted propensity | -0.0012 to 0.000794 | 0.9590 | 1.0309 | pass |
<!-- /generated -->

No replication raised an error, so no cell lost a replicate.

## Agreement with the canonical implementation

Python property evidence, with comparator runs pending. The R runners of the pairing study are
written with that study and have not run. No paired verdict is published, and `equivalence.csv` has
no rows. The study that runs them pins the `drtmle` image and adds the paired verdicts and the
nuisance gate.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bias` | `ate__clustered` | positive | average treatment effect: an estimated mechanism on the clustered law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0041 to 0.0058, margin 0.0096 | pass |
| `bias` | `ate__composite_l1` | positive | average treatment effect: the composite mechanism of treatment and response on a finite two-arm law, cross-fitted | bias interval inside the equivalence margin | bias -0.0058 to 0.0029, margin 0.0084 | pass |
| `bias` | `ate__correct` | positive | average treatment effect: the correct logistic mechanism, cross-fitted, on the law with a correct reduced regression | bias interval inside the equivalence margin | bias -0.0019 to 0.0025, margin 0.0043 | pass |
| `bias` | `ate__flexible` | positive | average treatment effect: a gradient-boosting mechanism, cross-fitted | bias interval inside the equivalence margin; pre-routed to reporting if it is red, because the learner's rate is not shown | bias 0.0204 to 0.0254, margin 0.0048 | **fail** |
| `bias` | `ate__insample_composite_l1` | positive | average treatment effect: the composite mechanism on a finite two-arm law, fitted in sample | bias interval inside the equivalence margin | bias -0.0062 to 0.0026, margin 0.0085 | pass |
| `bias` | `ate__insample_correct` | positive | average treatment effect: the correct logistic mechanism, fitted in sample | bias interval inside the equivalence margin | bias -0.0030 to 0.0019, margin 0.0048 | pass |
| `bias` | `ate__insample_missing_treatment` | positive | average treatment effect: the arm fit with a missing treatment and a missing outcome, fitted in sample | bias interval inside the equivalence margin | bias -0.0020 to 0.0075, margin 0.0091 | pass |
| `bias` | `ate__missing_treatment_crossfit` | positive | average treatment effect: the arm fit with a missing treatment and a missing outcome, cross-fitted | bias interval inside the equivalence margin | bias -0.0024 to 0.0029, margin 0.0051 | pass |
| `bias` | `ate__rate` | positive | average treatment effect: a mechanism whose logit is off by 2 n^(-1/3) (1 + W2), cross-fitted, on the law with a correct reduced regression | bias interval inside the equivalence margin | bias 0.000882 to 0.0046, margin 0.0044 | **fail** |
| `bias` | `ate__stratified` | positive | average treatment effect: an estimated mechanism on the stratified law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0041 to 0.000171, margin 0.0041 | pass |
| `bias` | `ate__weighted` | positive | average treatment effect: an estimated mechanism on the weighted law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0022 to 0.0051, margin 0.0070 | pass |
| `bias` | `ate__wrong_mechanism` | control | average treatment effect: a main-terms logistic mechanism that omits the W1 W2 interaction of the true mechanism | bias interval must fall entirely outside the margin | bias 0.2364 to 0.2405, margin 0.0040 | pass |
| `bias` | `ate_d__regime` | positive | mean under the deterministic rule minus the mean if no one is treated: the regime fit with the mechanism logit scaled by 1 + 2 n^(-1/3), on the effect-modified law, where the scaling keeps the level sets | bias interval inside the equivalence margin | bias -0.0012 to 0.0030, margin 0.0040 | pass |
| `bias` | `ate_low__composite_l3` | positive | mean under the low arm minus the mean under the high arm: the composite mechanism on a finite three-arm law, cross-fitted | bias interval inside the equivalence margin | bias -0.0072 to 0.0025, margin 0.0094 | pass |
| `bias` | `ate_low__insample_composite_l3` | positive | mean under the low arm minus the mean under the high arm: the composite mechanism on a finite three-arm law, fitted in sample | bias interval inside the equivalence margin | bias -0.0043 to 0.0058, margin 0.0097 | pass |
| `bias` | `ate_low__rate_reduced` | positive | mean under the low arm minus the mean under the high arm: a multinomial mechanism shrunk toward uniform by 2 n^(-1/3), five folds, on a law where the reduced regression is not exact; each arm probability stays a function of the true one, so the level sets survive | bias interval inside the equivalence margin | bias -0.0045 to 0.0028, margin 0.0070 | pass |
| `bias` | `ate_low__three_arm` | positive | mean under the low arm minus the mean under the high arm: the correct multinomial mechanism on a three-arm law, five folds | bias interval inside the equivalence margin | bias -0.0023 to 0.0044, margin 0.0064 | pass |
| `bias` | `ate_medium__rate_reduced` | positive | mean under the medium arm minus the mean under the high arm: a multinomial mechanism shrunk toward uniform by 2 n^(-1/3), five folds, on a law where the reduced regression is not exact; each arm probability stays a function of the true one, so the level sets survive | bias interval inside the equivalence margin | bias -0.0015 to 0.0059, margin 0.0072 | pass |
| `bias` | `ate_medium__three_arm` | positive | mean under the medium arm minus the mean under the high arm: the correct multinomial mechanism on a three-arm law, five folds | bias interval inside the equivalence margin | bias -0.0014 to 0.0053, margin 0.0065 | pass |
| `bias` | `ate_mid__composite_l3` | positive | mean under the middle arm minus the mean under the high arm: the composite mechanism on a finite three-arm law, cross-fitted | bias interval inside the equivalence margin | bias -0.0033 to 0.0063, margin 0.0093 | pass |
| `bias` | `ate_mid__insample_composite_l3` | positive | mean under the middle arm minus the mean under the high arm: the composite mechanism on a finite three-arm law, fitted in sample | bias interval inside the equivalence margin | bias -0.0035 to 0.0062, margin 0.0094 | pass |
| `bias` | `ate_s__regime` | positive | mean under the stochastic rule minus the mean if no one is treated: the regime fit with the mechanism logit scaled by 1 + 2 n^(-1/3), on the effect-modified law, where the scaling keeps the level sets | bias interval inside the equivalence margin | bias -0.0011 to 0.0021, margin 0.0031 | pass |
| `bias` | `ate_s_vs_d__insample_regime_missing` | positive | mean under the stochastic rule minus the mean under the deterministic rule: the regime fit with a missing outcome, fitted in sample | bias interval inside the equivalence margin | bias -0.000149 to 0.000830, margin 0.0013 | pass |
| `bias` | `ate_v0__stratified` | positive | average treatment effect in stratum V = 0: an estimated mechanism on the stratified law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0059 to 0.000622, margin 0.0063 | pass |
| `bias` | `ate_v1__stratified` | positive | average treatment effect in stratum V = 1: an estimated mechanism on the stratified law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0044 to 0.0031, margin 0.0073 | pass |
| `bias` | `ate_v2__stratified` | positive | average treatment effect in stratum V = 2: an estimated mechanism on the stratified law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0053 to 0.0013, margin 0.0064 | pass |
| `bias` | `ey0__clustered` | positive | arm-0 mean: an estimated mechanism on the clustered law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0017 to 0.0038, margin 0.0054 | pass |
| `bias` | `ey0__composite_l1` | positive | arm-0 mean: the composite mechanism of treatment and response on a finite two-arm law, cross-fitted | bias interval inside the equivalence margin | bias -0.0014 to 0.0046, margin 0.0058 | pass |
| `bias` | `ey0__correct` | positive | arm-0 mean: the correct logistic mechanism, cross-fitted, on the law with a correct reduced regression | bias interval inside the equivalence margin | bias -0.0022 to 0.0013, margin 0.0034 | pass |
| `bias` | `ey0__flexible` | positive | arm-0 mean: a gradient-boosting mechanism, cross-fitted | bias interval inside the equivalence margin; pre-routed to reporting if it is red, because the learner's rate is not shown | bias -0.0120 to -0.0083, margin 0.0035 | **fail** |
| `bias` | `ey0__insample_composite_l1` | positive | arm-0 mean: the composite mechanism on a finite two-arm law, fitted in sample | bias interval inside the equivalence margin | bias -0.0018 to 0.0047, margin 0.0063 | pass |
| `bias` | `ey0__insample_correct` | positive | arm-0 mean: the correct logistic mechanism, fitted in sample | bias interval inside the equivalence margin | bias -0.0016 to 0.0020, margin 0.0035 | pass |
| `bias` | `ey0__insample_missing_treatment` | positive | arm-0 mean: the arm fit with a missing treatment and a missing outcome, fitted in sample | bias interval inside the equivalence margin | bias -0.0069 to 0.000348, margin 0.0070 | pass |
| `bias` | `ey0__missing_treatment_crossfit` | positive | arm-0 mean: the arm fit with a missing treatment and a missing outcome, cross-fitted | bias interval inside the equivalence margin | bias -0.0033 to 0.000950, margin 0.0041 | pass |
| `bias` | `ey0__natural_course` | positive | arm-0 mean: the natural-course fit with a response mechanism off by a multiple of n^(-1/3) | bias interval inside the equivalence margin | bias -0.0034 to 0.0026, margin 0.0058 | pass |
| `bias` | `ey0__rate` | positive | arm-0 mean: a mechanism whose logit is off by 2 n^(-1/3) (1 + W2), cross-fitted, on the law with a correct reduced regression | bias interval inside the equivalence margin | bias -0.0032 to -0.000466, margin 0.0033 | pass |
| `bias` | `ey0__weighted` | positive | arm-0 mean: an estimated mechanism on the weighted law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0019 to 0.0022, margin 0.0039 | pass |
| `bias` | `ey0__wrong_mechanism` | control | arm-0 mean: a main-terms logistic mechanism that omits the W1 W2 interaction of the true mechanism | bias interval must fall entirely outside the margin | bias -0.1216 to -0.1184, margin 0.0031 | pass |
| `bias` | `ey1__clustered` | positive | arm-1 mean: an estimated mechanism on the clustered law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0014 to 0.0052, margin 0.0064 | pass |
| `bias` | `ey1__composite_l1` | positive | arm-1 mean: the composite mechanism of treatment and response on a finite two-arm law, cross-fitted | bias interval inside the equivalence margin | bias -0.0029 to 0.0032, margin 0.0059 | pass |
| `bias` | `ey1__correct` | positive | arm-1 mean: the correct logistic mechanism, cross-fitted, on the law with a correct reduced regression | bias interval inside the equivalence margin | bias -0.0018 to 0.0016, margin 0.0033 | pass |
| `bias` | `ey1__flexible` | positive | arm-1 mean: a gradient-boosting mechanism, cross-fitted | bias interval inside the equivalence margin; pre-routed to reporting if it is red, because the learner's rate is not shown | bias 0.0107 to 0.0147, margin 0.0038 | **fail** |
| `bias` | `ey1__insample_composite_l1` | positive | arm-1 mean: the composite mechanism on a finite two-arm law, fitted in sample | bias interval inside the equivalence margin | bias -0.0033 to 0.0027, margin 0.0058 | pass |
| `bias` | `ey1__insample_correct` | positive | arm-1 mean: the correct logistic mechanism, fitted in sample | bias interval inside the equivalence margin | bias -0.0022 to 0.0015, margin 0.0035 | pass |
| `bias` | `ey1__insample_missing_treatment` | positive | arm-1 mean: the arm fit with a missing treatment and a missing outcome, fitted in sample | bias interval inside the equivalence margin | bias -0.0037 to 0.0027, margin 0.0061 | pass |
| `bias` | `ey1__level_set_breaking` | control | arm-1 mean: a mechanism whose error moves across the level sets of the true mechanism (it adds n^(-1/3) W2 to the logit), on a law where the reduced regression is not exact | bias interval must fall entirely outside the margin | bias -0.0113 to -0.0091, margin 0.0037 | pass |
| `bias` | `ey1__level_set_breaking_h1` | positive | arm-1 mean: the same mechanism error on a law where the reduced regression is exact | bias interval inside the equivalence margin | bias -0.0012 to 0.0028, margin 0.0038 | pass |
| `bias` | `ey1__missing_treatment_crossfit` | positive | arm-1 mean: the arm fit with a missing treatment and a missing outcome, cross-fitted | bias interval inside the equivalence margin | bias -0.0027 to 0.000794, margin 0.0034 | pass |
| `bias` | `ey1__parametric_q_ne_qr` | positive | arm-1 mean: the correct logistic mechanism on a law where the reduced regression is not exact | bias interval inside the equivalence margin | bias -0.0010 to 0.0012, margin 0.0037 | pass |
| `bias` | `ey1__rate` | positive | arm-1 mean: a mechanism whose logit is off by 2 n^(-1/3) (1 + W2), cross-fitted, on the law with a correct reduced regression | bias interval inside the equivalence margin | bias -0.000589 to 0.0024, margin 0.0035 | pass |
| `bias` | `ey1__weighted` | positive | arm-1 mean: an estimated mechanism on the weighted law, on the draws of the known-mechanism cell | bias interval inside the equivalence margin | bias -0.0015 to 0.0047, margin 0.0059 | pass |
| `bias` | `ey1__wrong_mechanism` | control | arm-1 mean: a main-terms logistic mechanism that omits the W1 W2 interaction of the true mechanism | bias interval must fall entirely outside the margin | bias 0.1170 to 0.1200, margin 0.0030 | pass |
| `bias` | `ey_d__insample_regime_missing` | positive | mean outcome under the deterministic rule that treats where the rule says so: the regime fit with a missing outcome, fitted in sample | bias interval inside the equivalence margin | bias -0.0025 to 0.0010, margin 0.0048 | pass |
| `bias` | `ey_d__insample_regime_missing_treatment` | positive | mean outcome under the deterministic rule that treats where the rule says so: the regime fit with a missing treatment and a missing outcome, fitted in sample | bias interval inside the equivalence margin | bias -0.0036 to 0.0025, margin 0.0059 | pass |
| `bias` | `ey_d__regime` | positive | mean outcome under the deterministic rule that treats where the rule says so: the regime fit with the mechanism logit scaled by 1 + 2 n^(-1/3), on the effect-modified law, where the scaling keeps the level sets | bias interval inside the equivalence margin | bias -0.000687 to 0.0037, margin 0.0043 | pass |
| `bias` | `ey_high__composite_l3` | positive | mean outcome under the high arm: the composite mechanism on a finite three-arm law, cross-fitted | bias interval inside the equivalence margin | bias -0.0052 to 0.0018, margin 0.0068 | pass |
| `bias` | `ey_high__insample_composite_l3` | positive | mean outcome under the high arm: the composite mechanism on a finite three-arm law, fitted in sample | bias interval inside the equivalence margin | bias -0.0038 to 0.0032, margin 0.0068 | pass |
| `bias` | `ey_high__rate_reduced` | positive | mean outcome under the high arm: a multinomial mechanism shrunk toward uniform by 2 n^(-1/3), five folds, on a law where the reduced regression is not exact; each arm probability stays a function of the true one, so the level sets survive | bias interval inside the equivalence margin | bias -0.0037 to 0.0016, margin 0.0051 | pass |
| `bias` | `ey_high__three_arm` | positive | mean outcome under the high arm: the correct multinomial mechanism on a three-arm law, five folds | bias interval inside the equivalence margin | bias -0.0026 to 0.0022, margin 0.0047 | pass |
| `bias` | `ey_low__composite_l3` | positive | mean outcome under the low arm: the composite mechanism on a finite three-arm law, cross-fitted | bias interval inside the equivalence margin | bias -0.0075 to -0.000620, margin 0.0066 | **fail** |
| `bias` | `ey_low__insample_composite_l3` | positive | mean outcome under the low arm: the composite mechanism on a finite three-arm law, fitted in sample | bias interval inside the equivalence margin | bias -0.0033 to 0.0041, margin 0.0071 | pass |
| `bias` | `ey_low__rate_reduced` | positive | mean outcome under the low arm: a multinomial mechanism shrunk toward uniform by 2 n^(-1/3), five folds, on a law where the reduced regression is not exact; each arm probability stays a function of the true one, so the level sets survive | bias interval inside the equivalence margin | bias -0.0044 to 0.000537, margin 0.0048 | pass |
| `bias` | `ey_low__three_arm` | positive | mean outcome under the low arm: the correct multinomial mechanism on a three-arm law, five folds | bias interval inside the equivalence margin | bias -0.0016 to 0.0033, margin 0.0048 | pass |
| `bias` | `ey_medium__rate_reduced` | positive | mean outcome under the medium arm: a multinomial mechanism shrunk toward uniform by 2 n^(-1/3), five folds, on a law where the reduced regression is not exact; each arm probability stays a function of the true one, so the level sets survive | bias interval inside the equivalence margin | bias -0.0013 to 0.0035, margin 0.0047 | pass |
| `bias` | `ey_medium__three_arm` | positive | mean outcome under the medium arm: the correct multinomial mechanism on a three-arm law, five folds | bias interval inside the equivalence margin | bias -0.000741 to 0.0043, margin 0.0049 | pass |
| `bias` | `ey_mid__composite_l3` | positive | mean outcome under the middle arm: the composite mechanism on a finite three-arm law, cross-fitted | bias interval inside the equivalence margin | bias -0.0038 to 0.0034, margin 0.0069 | pass |
| `bias` | `ey_mid__insample_composite_l3` | positive | mean outcome under the middle arm: the composite mechanism on a finite three-arm law, fitted in sample | bias interval inside the equivalence margin | bias -0.0023 to 0.0044, margin 0.0065 | pass |
| `bias` | `ey_never__regime` | positive | mean outcome if no one is treated: the regime fit with the mechanism logit scaled by 1 + 2 n^(-1/3), on the effect-modified law, where the scaling keeps the level sets | bias interval inside the equivalence margin | bias -0.0014 to 0.0026, margin 0.0039 | pass |
| `bias` | `ey_obs__natural_course` | positive | observed outcome mean under the natural course: the natural-course fit with a response mechanism off by a multiple of n^(-1/3) | bias interval inside the equivalence margin | bias -0.0031 to 0.0014, margin 0.0043 | pass |
| `bias` | `ey_s__insample_regime_missing` | positive | mean outcome under the stochastic rule that treats with the declared probability: the regime fit with a missing outcome, fitted in sample | bias interval inside the equivalence margin | bias -0.0019 to 0.0010, margin 0.0040 | pass |
| `bias` | `ey_s__regime` | positive | mean outcome under the stochastic rule that treats with the declared probability: the regime fit with the mechanism logit scaled by 1 + 2 n^(-1/3), on the effect-modified law, where the scaling keeps the level sets | bias interval inside the equivalence margin | bias -0.000669 to 0.0029, margin 0.0034 | pass |
| `bias` | `paf__natural_course` | positive | population attributable fraction: the natural-course fit with a response mechanism off by a multiple of n^(-1/3) | bias interval inside the equivalence margin | bias -0.0044 to 0.0041, margin 0.0082 | pass |
| `bias` | `par__natural_course` | positive | population attributable risk: the natural-course fit with a response mechanism off by a multiple of n^(-1/3) | bias interval inside the equivalence margin | bias -0.0024 to 0.0016, margin 0.0039 | pass |
| `interval_calibration` | `ate__known_arms__correctly_specified` | positive | average treatment effect: the targeted arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9389 to 0.9639, SE ratio 0.9847 to 1.0667 | pass |
| `interval_calibration` | `ate__known_clustered__correctly_specified` | positive | average treatment effect: the targeted fit on the clustered law with unequal cluster sizes with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9305 to 0.9573, SE ratio 0.9443 to 1.0249 | pass |
| `interval_calibration` | `ate__known_stratified__correctly_specified` | positive | average treatment effect: the targeted fit on the stratified law, one target per stratum with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9311 to 0.9578, SE ratio 0.9488 to 1.0334 | pass |
| `interval_calibration` | `ate__known_weighted__correctly_specified` | positive | average treatment effect: the targeted fit on the weighted law with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9394 to 0.9643, SE ratio 0.9479 to 1.0275 | pass |
| `interval_calibration` | `ate__one_step_known_arms__correctly_specified` | positive | average treatment effect: the one-step arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9400 to 0.9648, SE ratio 0.9849 to 1.0679 | pass |
| `interval_calibration` | `ate_d__known_regimes__correctly_specified` | positive | mean under the deterministic rule minus the mean if no one is treated: the targeted regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9311 to 0.9578, SE ratio 0.9511 to 1.0336 | pass |
| `interval_calibration` | `ate_d__one_step_known_regimes__correctly_specified` | positive | mean under the deterministic rule minus the mean if no one is treated: the one-step regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9300 to 0.9569, SE ratio 0.9492 to 1.0304 | pass |
| `interval_calibration` | `ate_s__known_regimes__correctly_specified` | positive | mean under the stochastic rule minus the mean if no one is treated: the targeted regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9378 to 0.9630, SE ratio 0.9793 to 1.0656 | pass |
| `interval_calibration` | `ate_s__one_step_known_regimes__correctly_specified` | positive | mean under the stochastic rule minus the mean if no one is treated: the one-step regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9361 to 0.9617, SE ratio 0.9789 to 1.0666 | pass |
| `interval_calibration` | `ate_v0__known_stratified__correctly_specified` | positive | average treatment effect in stratum V = 0: the targeted fit on the stratified law, one target per stratum with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9272 to 0.9546, SE ratio 0.9468 to 1.0249 | pass |
| `interval_calibration` | `ate_v1__known_stratified__correctly_specified` | positive | average treatment effect in stratum V = 1: the targeted fit on the stratified law, one target per stratum with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9327 to 0.9591, SE ratio 0.9563 to 1.0394 | pass |
| `interval_calibration` | `ate_v2__known_stratified__correctly_specified` | positive | average treatment effect in stratum V = 2: the targeted fit on the stratified law, one target per stratum with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9372 to 0.9626, SE ratio 0.9568 to 1.0329 | pass |
| `interval_calibration` | `ey0__known_arms__correctly_specified` | positive | arm-0 mean: the targeted arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9300 to 0.9569, SE ratio 0.9491 to 1.0294 | pass |
| `interval_calibration` | `ey0__known_clustered__correctly_specified` | positive | arm-0 mean: the targeted fit on the clustered law with unequal cluster sizes with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9394 to 0.9643, SE ratio 0.9902 to 1.0722 | **fail** |
| `interval_calibration` | `ey0__known_weighted__correctly_specified` | positive | arm-0 mean: the targeted fit on the weighted law with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9272 to 0.9546, SE ratio 0.9390 to 1.0180 | pass |
| `interval_calibration` | `ey0__one_step_known_arms__correctly_specified` | positive | arm-0 mean: the one-step arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9266 to 0.9542, SE ratio 0.9479 to 1.0269 | pass |
| `interval_calibration` | `ey1__known_arms__correctly_specified` | positive | arm-1 mean: the targeted arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9372 to 0.9626, SE ratio 0.9743 to 1.0645 | pass |
| `interval_calibration` | `ey1__known_clustered__correctly_specified` | positive | arm-1 mean: the targeted fit on the clustered law with unequal cluster sizes with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9383 to 0.9635, SE ratio 0.9375 to 1.0205 | pass |
| `interval_calibration` | `ey1__known_weighted__correctly_specified` | positive | arm-1 mean: the targeted fit on the weighted law with its declared mechanism | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9350 to 0.9608, SE ratio 0.9615 to 1.0443 | pass |
| `interval_calibration` | `ey1__one_step_known_arms__correctly_specified` | positive | arm-1 mean: the one-step arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9366 to 0.9622, SE ratio 0.9752 to 1.0667 | pass |
| `interval_calibration` | `ey_d__known_regimes__correctly_specified` | positive | mean outcome under the deterministic rule that treats where the rule says so: the targeted regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9300 to 0.9569, SE ratio 0.9406 to 1.0178 | pass |
| `interval_calibration` | `ey_d__one_step_known_regimes__correctly_specified` | positive | mean outcome under the deterministic rule that treats where the rule says so: the one-step regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9250 to 0.9529, SE ratio 0.9384 to 1.0143 | pass |
| `interval_calibration` | `ey_never__known_regimes__correctly_specified` | positive | mean outcome if no one is treated: the targeted regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9300 to 0.9569, SE ratio 0.9500 to 1.0290 | pass |
| `interval_calibration` | `ey_never__one_step_known_regimes__correctly_specified` | positive | mean outcome if no one is treated: the one-step regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9266 to 0.9542, SE ratio 0.9483 to 1.0287 | pass |
| `interval_calibration` | `ey_s__known_regimes__correctly_specified` | positive | mean outcome under the stochastic rule that treats with the declared probability: the targeted regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9278 to 0.9551, SE ratio 0.9329 to 1.0126 | pass |
| `interval_calibration` | `ey_s__one_step_known_regimes__correctly_specified` | positive | mean outcome under the stochastic rule that treats with the declared probability: the one-step regime fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9272 to 0.9546, SE ratio 0.9315 to 1.0111 | pass |
| `interval_calibration` | `or__known_arms__correctly_specified` | positive | marginal odds ratio: the targeted arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9400 to 0.9648, SE ratio 0.9853 to 1.0674 | pass |
| `interval_calibration` | `or__one_step_known_arms__correctly_specified` | positive | marginal odds ratio: the one-step arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9417 to 0.9661, SE ratio 0.9866 to 1.0679 | pass |
| `interval_calibration` | `rr__known_arms__correctly_specified` | positive | marginal risk ratio: the targeted arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9378 to 0.9630, SE ratio 0.9834 to 1.0645 | pass |
| `interval_calibration` | `rr__one_step_known_arms__correctly_specified` | positive | marginal risk ratio: the one-step arm fit with the declared mechanism of the effect-modified two-arm law, so the validity term is zero and the interval is published | SE-ratio and coverage intervals both inside their calibration bands | coverage 0.9411 to 0.9657, SE ratio 0.9839 to 1.0663 | pass |
| `root_n_points` | `n_1500` | positive | the average treatment effect at n = 1,500 under a mechanism off by a multiple of n^(-1/3) | bias interval inside the equivalence margin | bias 0.0022 to 0.0065, margin 0.0066 | pass |
| `root_n_points` | `n_4500` | positive | the average treatment effect at n = 4,500 under a mechanism off by a multiple of n^(-1/3) | bias interval inside the equivalence margin | bias 0.000373 to 0.0029, margin 0.0039 | pass |
| `root_n_points` | `n_500` | positive | the average treatment effect at n = 500 under a mechanism off by a multiple of n^(-1/3) | bias interval inside the equivalence margin | bias 0.0013 to 0.0085, margin 0.0109 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5086 to -0.4320 | pass |
| `simultaneous_coverage` | `known_band__simultaneous_band` | positive | the simultaneous band over the arm means, the effect, the risk ratio and the odds ratio, with the declared mechanism | the band covers the whole truth vector at the nominal joint rate | joint coverage 0.9289 to 0.9560 | pass |
| `spread` | `parametric_q_ne_qr` | diagnostic | the plug-in standard error over the empirical spread of the estimates, for the correct logistic mechanism on a law where the reduced regression is not exact | reported only: the ratio and its interval, with no verdict | plug-in SE ratio 1.0843 to 1.2002, coverage 0.9620 to 0.9858 | reported |
| `targeting_necessity` | `cn_ey_obs__targeted` | positive | observed outcome mean under the natural course: the targeted estimate of the fit | bias interval inside the equivalence margin | bias -0.0031 to 0.0014, margin 0.0043 | pass |
| `targeting_necessity` | `cn_ey_obs__untargeted` | control | observed outcome mean under the natural course: the Horvitz-Thompson mean of the same fit at the estimated mechanism, before the targeting step, on the same draws | the displacement from the targeted estimate must reach one standard deviation of the targeted estimate, and the bias interval must fall entirely outside the margin | bias -0.0573 to -0.0534, margin 0.0038 | pass |
| `targeting_necessity` | `p_ate__targeted` | positive | average treatment effect: the targeted estimate of the fit | bias interval inside the equivalence margin | bias 0.000882 to 0.0046, margin 0.0044 | **fail** |
| `targeting_necessity` | `p_ate__untargeted` | control | average treatment effect: the Horvitz-Thompson mean of the same fit at the estimated mechanism, before the targeting step, on the same draws | the displacement from the targeted estimate must reach one standard deviation of the targeted estimate, and the bias interval must fall entirely outside the margin | bias -0.1145 to -0.1103, margin 0.0049 | pass |
| `targeting_necessity` | `p_ey0__targeted` | positive | arm-0 mean: the targeted estimate of the fit | bias interval inside the equivalence margin | bias -0.0032 to -0.000466, margin 0.0033 | pass |
| `targeting_necessity` | `p_ey0__untargeted` | control | arm-0 mean: the Horvitz-Thompson mean of the same fit at the estimated mechanism, before the targeting step, on the same draws | the displacement from the targeted estimate must reach one standard deviation of the targeted estimate, and the bias interval must fall entirely outside the margin | bias 0.0724 to 0.0761, margin 0.0043 | pass |
| `targeting_necessity` | `p_ey1__targeted` | positive | arm-1 mean: the targeted estimate of the fit | bias interval inside the equivalence margin | bias -0.000589 to 0.0024, margin 0.0035 | pass |
| `targeting_necessity` | `p_ey1__untargeted` | control | arm-1 mean: the Horvitz-Thompson mean of the same fit at the estimated mechanism, before the targeting step, on the same draws | the displacement from the targeted estimate must reach one standard deviation of the targeted estimate, and the bias interval must fall entirely outside the margin | bias -0.0394 to -0.0368, margin 0.0032 | pass |
| `targeting_necessity` | `pd_d__targeted` | positive | mean under the deterministic rule: the targeted estimate of the fit | bias interval inside the equivalence margin | bias -0.000687 to 0.0037, margin 0.0043 | pass |
| `targeting_necessity` | `pd_d__untargeted` | control | mean under the deterministic rule: the Horvitz-Thompson mean of the same fit at the estimated mechanism, before the targeting step, on the same draws | the displacement from the targeted estimate must reach one standard deviation of the targeted estimate, and the bias interval must fall entirely outside the margin | bias 0.0407 to 0.0462, margin 0.0053 | pass |
| `targeting_necessity` | `pd_s__targeted` | positive | mean under the stochastic rule: the targeted estimate of the fit | bias interval inside the equivalence margin | bias -0.000669 to 0.0029, margin 0.0034 | pass |
| `targeting_necessity` | `pd_s__untargeted` | control | mean under the stochastic rule: the Horvitz-Thompson mean of the same fit at the estimated mechanism, before the targeting step, on the same draws | the displacement from the targeted estimate must reach one standard deviation of the targeted estimate, and the bias interval must fall entirely outside the margin | bias 0.0380 to 0.0424, margin 0.0043 | pass |
| `targeting_necessity` | `rm_d__targeted` | positive | mean under the deterministic rule, with a missing outcome: the targeted estimate of the fit | bias interval inside the equivalence margin | bias -0.0025 to 0.0010, margin 0.0048 | pass |
| `targeting_necessity` | `rm_d__untargeted` | control | mean under the deterministic rule, with a missing outcome: the Horvitz-Thompson mean of the same fit at the estimated mechanism, before the targeting step, on the same draws | the displacement from the targeted estimate must reach one standard deviation of the targeted estimate, and the bias interval must fall entirely outside the margin | bias -0.0244 to -0.0211, margin 0.0046 | pass |
<!-- /generated -->

| family | what it tests | size |
| --- | --- | --- |
| `interval_calibration` | the SE ratio and coverage of the known-mechanism fits of laws PD (arms and regimes, targeted and one-step), WT, CL and ST. Every cell is a correctly specified H2 fit | $n$ from 200 clusters to 3,000, 2,000 replications |
| `simultaneous_coverage` | the simultaneous band of the known-mechanism arm fit of law PD | $n$ = 2,000, 2,000 replications |
| `bias` | the standardized bias of every estimated-mechanism cell against the margin of 0.25. A positive cell must fall inside the margin. A control must fall outside it | $n$ from 200 clusters to 7,000, 400 to 1,200 replications |
| `root_n_points` and `root_n_rate` | the bias of the `rate` fit of law P at three sizes, and the slope of its empirical SD on $n$ | $n$ = 500 to 4,500, 1,000 replications for each size |
| `targeting_necessity` | the targeted estimate beside the untargeted Horvitz-Thompson mean of the same fit, paired by replicate. The untargeted arm must move away from the truth by at least the displacement floor | seven pairs, 400 to 800 replications |
| `spread` | the plug-in standard error over the empirical SD of the parametric fit of law VL1. It reports and states no verdict | $n$ = 2,000, 1,200 replications |

Three controls must fail. The wrong mechanism of law P leaves the bias margin. The
level-set-breaking mechanism of law VL1 leaves it. The untargeted mean of every necessity pair
moves away from the truth. All three do.

An estimated-mechanism cell reads bias only. The estimator withholds `std_error`, `ci` and `pvalue`
there, so no coverage or SE-ratio claim is made for it.

## Red cells of this study

The study publishes under the `reporting` policy. Twenty of 122 gated cells are red, and one cell
is reported without a verdict. Every red cell is a positive cell, and every control behaves. The
first run of the study found a leak in the pooled reduced regression, where fold $k$'s treatment
and outcome entered the regression for fold $k$. The estimator now fits each fold's reduced
regression on leave-two-folds-out designs, and this run is the repeat. Each red cell below has an
owner in the [roadmap](../../roadmap.md#red-cell-owners) and a place in the
[red-cell ledger](red-cells.md).

| owner | cells | reading |
| --- | --- | --- |
| `adaptive-iptw-flexible-learner` | `bias/ate__flexible`, `bias/ey0__flexible`, `bias/ey1__flexible` | declared `reporting` before the run. Law P is H1, so the validity term vanishes for the gradient-boosting mechanism. The standardized biases are 1.182, -0.718 and 0.832. The learner's own error terms cause them. The key of a fit reads the mechanism learner family. A boosted fit matches no row and carries the `experimental` label. The three cells are outside the configuration of every row |
| `adaptive-iptw-finite-sample` | `bias/ate__rate` and the 14 `targeting_necessity` rows | the `rate` ATE reads a standardized bias of 0.155, and its 99% interval in SD units is 0.050 to 0.261. The design predicts 0.086, and the interval contains it. The `root_n_points` rungs pass. The necessity family reads the same estimates, and the family clause turns its other 13 rows red. The diagnosis finds no defect |
| `adaptive-iptw-finite-sample` | `interval_calibration/ey0__known_clustered__correctly_specified` | 200 clusters. The SE ratio is 1.030, and its 99% interval ends at 1.072 against an upper bound of 1.07. Coverage is 0.953, and its interval contains 0.95. The in-sample fit on the same draws reads 1.027. The diagnosis finds no defect |
| `adaptive-iptw-composite-l3` | `bias/ey_low__composite_l3` | 400 draws read a standardized bias of -0.153, and the interval does not contain the predicted bias of 0. The in-sample fit on the same draws reads -0.151. A diagnostic runs the same scene on 4,000 draws with seeds that the study never uses. It reads -0.011, with a 99% interval of -0.052 to 0.030. The diagnosis finds no defect, and the 400-draw reading does not reproduce |

The diagnostic of the last row is `tests/diagnostics/adaptive_iptw_composite_l3/`. The test
`tests/unit/test_adaptive_iptw_composite_l3_reading.py` rebuilds its reading from the committed
rows. The study changes no seed, size, margin, law or learner after the verdict.

## Convergence

Every property row records the largest number of tilt rounds of the fit it reads and whether every
fluctuation met the tolerance. A fit that stops at the 100-round cap keeps its rows. No verdict
reads either column, so a cell can pass with a nonconverged fit, and `properties.csv` reports
`nonconverged_fits` and `max_rounds` for each cell.

Ten of the 106,400 property rows come from nonconverged fits. They are the ten regime cells of law
PD, targeted and one-step, on one draw (replicate 1985), where the loop reached the cap of 100
rounds. Each of those cells has 2,000 replicates and passes. Every other cell stops at 23 rounds or
fewer. The mean number of rounds of a cell is between 2.4 and 7.6.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 3 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 3 | of those, passing |
| `paired_tests_total` | 0 | paired comparisons with R `drtmle` |
| `paired_tests_passed` | 0 | of those, passing |
| `property_cells_total` | 122 | repeated-sampling property cells, reported cells excluded |
| `property_cells_passed` | 102 | cells whose own and family verdicts pass |
| `property_cells_reported` | 1 | cells reported and not gated |
| `property_rows_nonconverged` | 10 | property rows with at least one nonconverged fit |
| `property_nonconverged_fits` | 10 | nonconverged fits, counted once for each cell that reads them |
| `property_max_rounds` | 100 | most tilt rounds of any property row |
| `max_standardized_bias` | 0.0344 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9510 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9616 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0741 | highest bootstrap primary SE-ratio endpoint |
| `properties[bias/ate__rate]:standardized_bias` | 0.1553 | standardized bias of the `rate` ATE |
| `properties[bias/ate__rate]:replicates` | 600 | replications of that cell |
| `properties[interval_calibration/ey0__known_clustered__correctly_specified]:se_ratio` | 1.0300 | SE ratio of the clustered known-mechanism arm mean |
| `properties[interval_calibration/ey0__known_clustered__correctly_specified]:se_ratio_ci_upper` | 1.0722 | upper end of its 99% interval |
| `properties[interval_calibration/ey0__known_clustered__correctly_specified]:coverage` | 0.9530 | its coverage |
| `properties[bias/ey_low__composite_l3]:standardized_bias` | -0.1529 | standardized bias of the three-arm composite cell |
| `properties[bias/ate__flexible]:standardized_bias` | 1.1821 | standardized bias of the boosted ATE |
| `properties[bias/ey0__flexible]:standardized_bias` | -0.7177 | standardized bias of the boosted arm 0 mean |
| `properties[bias/ey1__flexible]:standardized_bias` | 0.8316 | standardized bias of the boosted arm 1 mean |
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
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
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
| `margin:targeting_displacement` | 1 | least displacement of the untargeted mean in a necessity pair, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 20 | the same bound, in primary replications |

## Support label

A fit carries the label `supported` when a row of `cleverly._support_rows` covers its
configuration with the kind of inference that the fit reports. The rows `AI.*` cite the cells of
this study. Each red cell that a row cites carries its disposition in
`tests/studies/evidence/support_dispositions.py`. Any other fit carries the label `experimental`.

The key of a fit reads two learner families (`cleverly._support.learner_family`). A learner is
parametric when its exact class is `LogisticRegression`, `LinearRegression` or `GLM`. A learner is
also parametric when its class sets `cleverly_learner_family = "parametric"`. Every other learner
is adaptive, and this includes the default Super Learner.

| option | value | meaning |
| --- | --- | --- |
| `mechanism_learner` | `known` | the data declare the whole mechanism |
| `mechanism_learner` | `parametric` | every mechanism learner that the fit uses is parametric |
| `mechanism_learner` | `adaptive` | any other mechanism learner |
| `reduced_learner` | `fixed_basis` | the reduced regression is parametric |
| `reduced_learner` | `selecting` | any other reduced regression, and the default |

Every `AI.known_*` row needs `known` and `fixed_basis`. Every `AI.points*` row needs `parametric`
and `fixed_basis`. These are the families that the cells fitted.

| rows | configuration | kinds | cells |
| --- | --- | --- | --- |
| `AI.known_arms.targeted` | declared mechanism, arm targets, cross-fitted | point, Wald, band, derived | `interval_calibration/*known_arms*` and `simultaneous_coverage/known_band` |
| `AI.known_arms.one_step` | the same, one-step form | point, Wald, derived | `interval_calibration/*one_step_known_arms*` |
| `AI.known_regimes.targeted`, `.one_step` | declared mechanism, regime targets | point, Wald | `interval_calibration/*known_regimes*` and `*one_step_known_regimes*` |
| `AI.known_weighted`, `.known_clustered`, `.known_stratified` | declared mechanism with fixed weights, clusters or strata | point, Wald | `interval_calibration/*known_weighted*`, `*known_clustered*` and `*known_stratified*` |
| `AI.points.*`, `AI.points_three_arm`, `AI.points_regime` | estimated mechanism, complete data | point | the `bias` cells, `root_n_points` and `targeting_necessity` |
| `AI.points_missing_outcome.*`, `AI.points_composite_three_arm.*`, `AI.points_regime_missing` | estimated mechanism, missing outcome | point | the composite, natural-course and missing-data `bias` cells |
| `AI.points_missing_treatment.*` | estimated mechanism, missing treatment | point | the missing-treatment `bias` cells |
| `AI.points_weighted`, `.points_clustered`, `.points_stratified` | estimated mechanism with fixed weights, clusters or strata | point | the weighted, clustered and stratified `bias` cells |

An estimated mechanism reports no interval, so an estimated-mechanism row licenses the point
estimate only. The plug-in standard error and interval of such a fit are a diagnostic. Each row
names the pending `drtmle` comparator and the learner families that its cells fitted. Each
cross-fitted row also names the open $\epsilon$-path condition of S5 in
[AdaptiveIPTW](../adaptive-iptw.md#contract). These configurations stay `experimental`:

| configuration | reason |
| --- | --- |
| the one-step form with an estimated mechanism | no bias cell reads it |
| a declared mechanism fitted in sample | the calibration cells are cross-fitted |
| a declared mechanism with a missing outcome | no cell |
| `rr` or `or` with an estimated mechanism | no bias cell reads them |
| an in-sample natural-course fit | no in-sample natural-course cell |
| a treatment adjustment, or declared-estimated weights with a declared mechanism | a group-A condition or a shown status |
| fewer than 40 clusters | the study has no cell below 200 clusters |
| an estimated mechanism with an adaptive learner, such as gradient boosting or the default | outside the configuration of every row. The `bias/*__flexible` cells read a standardized bias of -0.718 (`ey0`), 0.832 (`ey1`) and 1.182 (`ate`) |
| a selecting reduced regression, such as the default | outside the configuration of every row. No cell fits one |

Two decisions come from the red cells. The row `AI.known_clustered` is supported with a stated
limit, because the cell `ey0__known_clustered` reads an SE ratio whose interval ends at 1.072.
The three flexible-learner cells are outside the configuration of every row.

## Limits

| limit | what it means for use |
| --- | --- |
| the comparator runs are pending | the claim rests on exact truths, the calibration cells and the controls. The pairing study adds a paired check against R `drtmle` |
| an estimated mechanism has no interval | the bias cells measure the point estimate only. The plug-in standard error and interval in the rows are a diagnostic, and the package does not publish them as inference |
| the validity term vanishes by design in the calibration cells only | the intervals are valid where the whole mechanism is declared known (H2). Laws PD, WT, CL and ST are the cells. Law P with a fitted mechanism is H1, so the bias cells there do not measure the validity term. VL1 and the level-set-breaking control measure it |
| a flexible mechanism learner is red | the `bias/flexible` cells show a bias of 0.7 to 1.2 SD with a gradient-boosting mechanism. The key reads the mechanism learner family, so a boosted or default fit is `experimental` |
| the reduced regression is a fixed polynomial of $\operatorname{logit} g$ | the study fits no selecting reduced learner and no default super learner. A selecting learner is outside conditions (E) and (S), and the key labels its fit `experimental` |
| the composite laws use the oracle factor learners declared as estimated | the cells measure the composite targeting and the fold-wise reduced regression. They do not measure a fitted response model |
| the natural-course and regime cells | an in-sample natural-course fit and a cross-fitted regime with missing data have no cell. The estimator refuses the second one |
| one fold draw per replicate | `repeats > 1` is refused, so the study has no repeat aggregate |
| `rr` and `or` | the known-mechanism cells read them on the log scale. No bias cell reads them |
| strata, clusters and weights | the keys of the WT, CL and ST laws are covered. The regime and natural-course keys under weights, clusters and strata have exact-law witnesses and no dedicated cell |

The causal interpretation requires consistency, conditional exchangeability given the covariates,
and positivity. The missing-data cells also require that the missingness is at random given the
covariates and the treatment.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_iptw/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_iptw/manifest.json)
records the pending comparator, the seeds and the artifact hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_iptw/property-replicates.parquet)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/adaptive_iptw/properties.csv)
carry every published row.
