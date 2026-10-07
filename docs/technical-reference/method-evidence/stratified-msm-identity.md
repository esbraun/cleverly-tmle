# Stratified identity-link MSM

This study validates the identity-link marginal structural model (MSM) of ordinary
point-treatment TMLE with baseline strata, `strata=["V"]`. The fit solves one score block
$I(V=s) H_s / P_n(V=s)$ for each stratum in one pooled fluctuation. Each stratum's coefficients
are the projection of the outcome regression on the working model under the law given $V = s$.
[MSM projections](../msm-projections.md) states the construction.

## What was tested

| setting | declaration |
| --- | --- |
| law | L1 of `tests/studies/stratified_alternating_law.py`, the [baseline-strata law](stratified-point-treatment-tmle.md), with a bounded outcome $Y \sim \operatorname{Beta}(24 Q, 24 (1 - Q))$. $E[Y \mid A, W, V]$ is L1's $Q$, so every truth is exact |
| working model | `MSM.linear(modifiers=("W",), interaction=False)`: $(1, a, W)$ with uniform weights |
| fit | in sample, a quasi-binomial outcome GLM on $(A, W, V)$ and a main-terms logistic treatment regression on $(W, V)$, with `g_bounds=(0.01, 0.99)` |
| primary scenario | `stratified_msm_identity`: the nine stratum coefficients, $n = 2000$, 1,000 replications |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

The comparator is R `tmle3` `Param_MSM` at commit `ed72f8a`. `tmle3` has no stratified MSM, so
the runner calls `Param_MSM` once per stratum subset, as the marginal `tmle3_msm` runner builds
it, and transforms the arm-indicator coefficients to the $(1, a, W)$ basis. The two
implementations differ in one declared way: `tmle3` fits its nuisances inside each stratum subset
with `Lrnr_glm`, and `cleverly` pools the nuisances over the strata and solves one block for each
stratum. A subset curve is on the subset's scale, so the runner's standard error is
$\operatorname{sd} / \sqrt{n_s}$, the number the stratified fit's embedded curve gives.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=0]` | point-treatment MSM projection intercept coefficient, in stratum V=0 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000735 to 0.000059 | 0.9430 | 0.9763 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=0]` | point-treatment MSM projection intercept coefficient, in stratum V=0 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000742 to 0.000053 | 0.9410 | 0.9756 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=1]` | point-treatment MSM projection intercept coefficient, in stratum V=1 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000820 to 0.000312 | 0.9460 | 0.9894 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=1]` | point-treatment MSM projection intercept coefficient, in stratum V=1 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000814 to 0.000319 | 0.9480 | 1.0067 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=2]` | point-treatment MSM projection intercept coefficient, in stratum V=2 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000316 to 0.000973 | 0.9410 | 0.9975 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=2]` | point-treatment MSM projection intercept coefficient, in stratum V=2 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000337 to 0.000954 | 0.9470 | 1.0332 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=0]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=0 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000439 to 0.000227 | 0.9450 | 0.9719 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=0]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=0 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000432 to 0.000234 | 0.9470 | 0.9799 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=1]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=1 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000349 to 0.000374 | 0.9610 | 1.0260 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=1]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=1 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000345 to 0.000378 | 0.9700 | 1.0691 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=2]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=2 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000567 to 0.000187 | 0.9360 | 0.9732 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=2]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=2 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000563 to 0.000193 | 0.9480 | 1.0220 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=0]` | point-treatment MSM projection treatment coefficient, in stratum V=0 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000134 to 0.000880 | 0.9480 | 0.9931 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=0]` | point-treatment MSM projection treatment coefficient, in stratum V=0 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000130 to 0.000885 | 0.9450 | 0.9940 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=1]` | point-treatment MSM projection treatment coefficient, in stratum V=1 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000514 to 0.000660 | 0.9490 | 0.9875 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=1]` | point-treatment MSM projection treatment coefficient, in stratum V=1 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000523 to 0.000649 | 0.9510 | 0.9939 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=2]` | point-treatment MSM projection treatment coefficient, in stratum V=2 | `cleverly` identity-link MSM TMLE with baseline strata | -0.000651 to 0.000411 | 0.9460 | 0.9825 | pass |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=2]` | point-treatment MSM projection treatment coefficient, in stratum V=2 | R `tmle3` `Param_MSM`, once per stratum subset | -0.000627 to 0.000438 | 0.9450 | 0.9850 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=0]` | point-treatment MSM projection intercept coefficient, in stratum V=0 | 0.000006 | 0.0082 | 1.0032 | 0.0020 | 0.0043 vs 0.0500 | equivalent |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=1]` | point-treatment MSM projection intercept coefficient, in stratum V=1 | -0.000006 | 0.0060 | 1.0056 | -0.0020 | 0.0173 vs 0.0500 | equivalent |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[(intercept)][V=2]` | point-treatment MSM projection intercept coefficient, in stratum V=2 | 0.000020 | 0.0172 | 1.0070 | -0.0060 | 0.0675 vs 0.0500 **>** | equivalent |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=0]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=0 | -0.000007 | 0.0119 | 1.0047 | -0.0020 | 0.0037 vs 0.0500 | equivalent |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=1]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=1 | -0.000004 | 0.0060 | 1.0077 | -0.0090 | 0.0519 vs 0.0500 **>** | equivalent |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[W][V=2]` | point-treatment MSM projection baseline-covariate coefficient, in stratum V=2 | -0.000006 | 0.0081 | 1.0047 | -0.0120 | 0.0476 vs 0.0500 | **inconclusive** |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=0]` | point-treatment MSM projection treatment coefficient, in stratum V=0 | -0.000005 | 0.0048 | 1.0027 | 0.0030 | 0.0028 vs 0.0500 | equivalent |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=1]` | point-treatment MSM projection treatment coefficient, in stratum V=1 | 0.000010 | 0.0089 | 1.0059 | -0.0020 | 0.0038 vs 0.0500 | equivalent |
| baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q)) | `msm[a][V=2]` | point-treatment MSM projection treatment coefficient, in stratum V=2 | -0.000025 | 0.0260 | 1.0037 | 0.0010 | 0.0063 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `identity_marginal_a__correctly_specified` | positive | treatment coefficient of the identity MSM (1, a, W), marginal: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9278 to 0.9551, SE ratio 0.9350 to 1.0150, empirical efficiency ratio 0.9853 to 1.0695, reported efficiency ratio 0.9990 to 1.0011 | pass |
| `interval_calibration` | `identity_marginal_a__noise_control` | control | treatment coefficient of the identity MSM (1, a, W), marginal: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8131 to 0.8563, SE ratio 0.6893 to 0.7446, empirical efficiency ratio 1.3429 to 1.4509, reported efficiency ratio 0.9990 to 1.0011 | pass |
| `interval_calibration` | `identity_marginal_a__shrunken_se_control` | control | treatment coefficient of the identity MSM (1, a, W), marginal: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8037 to 0.8478, SE ratio 0.6551 to 0.7111, empirical efficiency ratio 0.9849 to 1.0689, reported efficiency ratio 0.6994 to 0.7008 | pass |
| `interval_calibration` | `identity_marginal_intercept__correctly_specified` | positive | intercept coefficient of the identity MSM (1, a, W), marginal: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9406 to 0.9652, SE ratio 0.9791 to 1.0599, empirical efficiency ratio 0.9441 to 1.0221, reported efficiency ratio 0.9993 to 1.0017 | pass |
| `interval_calibration` | `identity_marginal_intercept__noise_control` | control | intercept coefficient of the identity MSM (1, a, W), marginal: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8294 to 0.8709, SE ratio 0.6972 to 0.7582, empirical efficiency ratio 1.3197 to 1.4341, reported efficiency ratio 0.9993 to 1.0017 | pass |
| `interval_calibration` | `identity_marginal_intercept__shrunken_se_control` | control | intercept coefficient of the identity MSM (1, a, W), marginal: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8115 to 0.8549, SE ratio 0.6844 to 0.7426, empirical efficiency ratio 0.9428 to 1.0232, reported efficiency ratio 0.6995 to 0.7012 | pass |
| `interval_calibration` | `identity_marginal_w__correctly_specified` | positive | baseline-covariate coefficient of the identity MSM (1, a, W), marginal: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9417 to 0.9661, SE ratio 0.9698 to 1.0526, empirical efficiency ratio 0.9508 to 1.0317, reported efficiency ratio 0.9995 to 1.0017 | pass |
| `interval_calibration` | `identity_marginal_w__noise_control` | control | baseline-covariate coefficient of the identity MSM (1, a, W), marginal: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8010 to 0.8454, SE ratio 0.6678 to 0.7261, empirical efficiency ratio 1.3785 to 1.4981, reported efficiency ratio 0.9994 to 1.0018 | pass |
| `interval_calibration` | `identity_marginal_w__shrunken_se_control` | control | baseline-covariate coefficient of the identity MSM (1, a, W), marginal: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8173 to 0.8601, SE ratio 0.6799 to 0.7375, empirical efficiency ratio 0.9499 to 1.0303, reported efficiency ratio 0.6996 to 0.7012 | pass |
| `interval_calibration` | `identity_v0_a__correctly_specified` | positive | treatment coefficient of the identity MSM (1, a, W), in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9316 to 0.9582, SE ratio 0.9347 to 1.0142, empirical efficiency ratio 0.9857 to 1.0701, reported efficiency ratio 0.9984 to 1.0016 | pass |
| `interval_calibration` | `identity_v0_a__noise_control` | control | treatment coefficient of the identity MSM (1, a, W), in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7911 to 0.8364, SE ratio 0.6584 to 0.7115, empirical efficiency ratio 1.4054 to 1.5182, reported efficiency ratio 0.9983 to 1.0016 | pass |
| `interval_calibration` | `identity_v0_a__shrunken_se_control` | control | treatment coefficient of the identity MSM (1, a, W), in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7937 to 0.8387, SE ratio 0.6548 to 0.7108, empirical efficiency ratio 0.9847 to 1.0689, reported efficiency ratio 0.6989 to 0.7011 | pass |
| `interval_calibration` | `identity_v0_intercept__correctly_specified` | positive | intercept coefficient of the identity MSM (1, a, W), in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9327 to 0.9591, SE ratio 0.9348 to 1.0190, empirical efficiency ratio 0.9830 to 1.0714, reported efficiency ratio 0.9995 to 1.0036 | pass |
| `interval_calibration` | `identity_v0_intercept__noise_control` | control | intercept coefficient of the identity MSM (1, a, W), in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8068 to 0.8506, SE ratio 0.6714 to 0.7299, empirical efficiency ratio 1.3725 to 1.4915, reported efficiency ratio 0.9995 to 1.0035 | pass |
| `interval_calibration` | `identity_v0_intercept__shrunken_se_control` | control | intercept coefficient of the identity MSM (1, a, W), in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7995 to 0.8440, SE ratio 0.6540 to 0.7125, empirical efficiency ratio 0.9835 to 1.0721, reported efficiency ratio 0.6996 to 0.7025 | pass |
| `interval_calibration` | `identity_v0_w__correctly_specified` | positive | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9278 to 0.9551, SE ratio 0.9461 to 1.0233, empirical efficiency ratio 0.9780 to 1.0574, reported efficiency ratio 0.9979 to 1.0033 | pass |
| `interval_calibration` | `identity_v0_w__noise_control` | control | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8115 to 0.8549, SE ratio 0.6787 to 0.7365, empirical efficiency ratio 1.3581 to 1.4739, reported efficiency ratio 0.9979 to 1.0032 | pass |
| `interval_calibration` | `identity_v0_w__shrunken_se_control` | control | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7963 to 0.8411, SE ratio 0.6623 to 0.7169, empirical efficiency ratio 0.9770 to 1.0571, reported efficiency ratio 0.6985 to 0.7022 | pass |
| `interval_calibration` | `identity_v1_a__correctly_specified` | positive | treatment coefficient of the identity MSM (1, a, W), in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9305 to 0.9573, SE ratio 0.9387 to 1.0184, empirical efficiency ratio 0.9810 to 1.0640, reported efficiency ratio 0.9969 to 1.0013 | pass |
| `interval_calibration` | `identity_v1_a__noise_control` | control | treatment coefficient of the identity MSM (1, a, W), in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7901 to 0.8354, SE ratio 0.6627 to 0.7188, empirical efficiency ratio 1.3897 to 1.5075, reported efficiency ratio 0.9969 to 1.0012 | pass |
| `interval_calibration` | `identity_v1_a__shrunken_se_control` | control | treatment coefficient of the identity MSM (1, a, W), in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8005 to 0.8449, SE ratio 0.6582 to 0.7120, empirical efficiency ratio 0.9825 to 1.0623, reported efficiency ratio 0.6979 to 0.7009 | pass |
| `interval_calibration` | `identity_v1_intercept__correctly_specified` | positive | intercept coefficient of the identity MSM (1, a, W), in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9255 to 0.9533, SE ratio 0.9440 to 1.0247, empirical efficiency ratio 0.9771 to 1.0601, reported efficiency ratio 0.9984 to 1.0038 | pass |
| `interval_calibration` | `identity_v1_intercept__noise_control` | control | intercept coefficient of the identity MSM (1, a, W), in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8178 to 0.8605, SE ratio 0.6884 to 0.7471, empirical efficiency ratio 1.3402 to 1.4553, reported efficiency ratio 0.9984 to 1.0038 | pass |
| `interval_calibration` | `identity_v1_intercept__shrunken_se_control` | control | intercept coefficient of the identity MSM (1, a, W), in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7979 to 0.8425, SE ratio 0.6604 to 0.7170, empirical efficiency ratio 0.9774 to 1.0611, reported efficiency ratio 0.6989 to 0.7027 | pass |
| `interval_calibration` | `identity_v1_w__correctly_specified` | positive | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9473 to 0.9704, SE ratio 0.9723 to 1.0507, empirical efficiency ratio 0.9544 to 1.0308, reported efficiency ratio 0.9993 to 1.0058 | pass |
| `interval_calibration` | `identity_v1_w__noise_control` | control | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8037 to 0.8478, SE ratio 0.6726 to 0.7292, empirical efficiency ratio 1.3751 to 1.4912, reported efficiency ratio 0.9994 to 1.0056 | pass |
| `interval_calibration` | `identity_v1_w__shrunken_se_control` | control | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8121 to 0.8553, SE ratio 0.6812 to 0.7358, empirical efficiency ratio 0.9538 to 1.0300, reported efficiency ratio 0.6996 to 0.7040 | pass |
| `interval_calibration` | `identity_v2_a__correctly_specified` | positive | treatment coefficient of the identity MSM (1, a, W), in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9233 to 0.9515, SE ratio 0.9264 to 1.0042, empirical efficiency ratio 0.9925 to 1.0766, reported efficiency ratio 0.9937 to 1.0005 | **fail** |
| `interval_calibration` | `identity_v2_a__noise_control` | control | treatment coefficient of the identity MSM (1, a, W), in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8021 to 0.8463, SE ratio 0.6742 to 0.7302, empirical efficiency ratio 1.3648 to 1.4795, reported efficiency ratio 0.9937 to 1.0005 | pass |
| `interval_calibration` | `identity_v2_a__shrunken_se_control` | control | treatment coefficient of the identity MSM (1, a, W), in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7885 to 0.8340, SE ratio 0.6484 to 0.7032, empirical efficiency ratio 0.9930 to 1.0758, reported efficiency ratio 0.6955 to 0.7003 | pass |
| `interval_calibration` | `identity_v2_intercept__correctly_specified` | positive | intercept coefficient of the identity MSM (1, a, W), in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9305 to 0.9573, SE ratio 0.9462 to 1.0239, empirical efficiency ratio 0.9762 to 1.0558, reported efficiency ratio 0.9943 to 1.0038 | pass |
| `interval_calibration` | `identity_v2_intercept__noise_control` | control | intercept coefficient of the identity MSM (1, a, W), in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7937 to 0.8387, SE ratio 0.6543 to 0.7087, empirical efficiency ratio 1.4101 to 1.5271, reported efficiency ratio 0.9945 to 1.0040 | pass |
| `interval_calibration` | `identity_v2_intercept__shrunken_se_control` | control | intercept coefficient of the identity MSM (1, a, W), in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8021 to 0.8463, SE ratio 0.6614 to 0.7175, empirical efficiency ratio 0.9753 to 1.0566, reported efficiency ratio 0.6962 to 0.7026 | pass |
| `interval_calibration` | `identity_v2_w__correctly_specified` | positive | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9311 to 0.9578, SE ratio 0.9466 to 1.0276, empirical efficiency ratio 0.9692 to 1.0513, reported efficiency ratio 0.9891 to 1.0009 | pass |
| `interval_calibration` | `identity_v2_w__noise_control` | control | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8016 to 0.8459, SE ratio 0.6653 to 0.7209, empirical efficiency ratio 1.3808 to 1.4947, reported efficiency ratio 0.9890 to 1.0009 | pass |
| `interval_calibration` | `identity_v2_w__shrunken_se_control` | control | baseline-covariate coefficient of the identity MSM (1, a, W), in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8010 to 0.8454, SE ratio 0.6624 to 0.7199, empirical efficiency ratio 0.9683 to 1.0508, reported efficiency ratio 0.6924 to 0.7007 | pass |
<!-- /generated -->

The property grid calibrates each coefficient of the stratified fit, marginal and in each stratum,
against its exact efficiency bound under the bounded outcome. Each cell has a shrunken-SE control
and a noise control.

Two results are red, both in stratum 2, which holds about 400 of the 2,000 rows. The
`interval_calibration/identity_v2_a__correctly_specified` cell has an SE-ratio interval of 0.926
to 1.004 against a floor of 0.93. On the primary draws the same coefficient reads 0.983 in
`cleverly` and 0.985 in R. The paired `msm[W][V=2]` row is inconclusive on its calibration leg
alone. The estimates differ by 6e-6 on average with the same spread, and R reports a standard
error 2.2% above its own spread. That is consistent with the declared nuisance difference: R
fits a Gaussian `Lrnr_glm` inside each subset.

The owner `X8-identity-small-stratum` in the [roadmap](../../roadmap.md#red-cell-owners) records
the reading. The study declared `gated` and moved to `reporting` before a re-run at the same
seeds.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | paired replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 18 | truth tests passing |
| `independent_tests_total` | 18 | truth tests reported |
| `paired_tests_passed` | 8 | paired comparisons passing |
| `paired_tests_total` | 9 | paired comparisons reported |
| `property_cells_passed` | 35 | property cells passing |
| `property_cells_total` | 36 | property cells reported |
| `max_standardized_bias` | 0.0706 | largest primary standardized bias |
| `min_coverage` | 0.9360 | lowest primary coverage |
| `max_margin_utilization` | 0.0260 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
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
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `bound:msm[(intercept)][V=0]_standard_error` | 0.0048 | exact bounded-outcome efficiency-bound standard error at primary n |
| `bound:msm[(intercept)][V=1]_standard_error` | 0.0069 | exact bounded-outcome efficiency-bound standard error at primary n |
| `bound:msm[(intercept)][V=2]_standard_error` | 0.0079 | exact bounded-outcome efficiency-bound standard error at primary n |
| `bound:msm[W][V=0]_standard_error` | 0.0040 | exact bounded-outcome efficiency-bound standard error at primary n |
| `bound:msm[W][V=1]_standard_error` | 0.0045 | exact bounded-outcome efficiency-bound standard error at primary n |
| `bound:msm[W][V=2]_standard_error` | 0.0045 | exact bounded-outcome efficiency-bound standard error at primary n |
| `bound:msm[a][V=0]_standard_error` | 0.0062 | exact bounded-outcome efficiency-bound standard error at primary n |
| `bound:msm[a][V=1]_standard_error` | 0.0071 | exact bounded-outcome efficiency-bound standard error at primary n |
| `bound:msm[a][V=2]_standard_error` | 0.0064 | exact bounded-outcome efficiency-bound standard error at primary n |

## Limits

- The study covers one finite-support law with one three-level stratum and a bounded outcome.
- The fits are in sample. A cross-fitted stratified MSM is not measured here.
- Linked and continuous-dose stratified MSMs are measured in the
  [stratified incremental MSM study](stratified-incremental-msm.md).

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_stratified_msm/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_stratified_msm/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_stratified_msm/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_stratified_msm/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_stratified_msm/properties.csv)
carry the protocol, provenance, and every published row.
