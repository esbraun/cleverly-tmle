# Stacked point-treatment CV-TMLE study

[← Method evidence grid](../../methodology.md#method-evidence-grid)

This study applies stacked point-treatment CV-TMLE to repeated samples from laws with known
parameter values. Cleverly and the pinned R reference receive the same realized samples and the
same treatment-stratified ten-fold assignments.

## Study setup

| setting | registered value |
| --- | --- |
| cleverly method | out-of-fold nuisance predictions, one stacked targeting update, whole-sample plug-in evaluation |
| external reference | R `tmle3` 0.2.0 and `sl3` at pinned commits, using `tmle3_Update(cvtmle = TRUE)` |
| laws | binary outcome; bounded continuous outcome |
| estimands | `ey1`, `ey0`, `ate`, `att`, `atc`, `ey_obs`, `par`; plus `paf`, `rr`, and `or` for the binary law |
| primary run | 1,600 replications per law; 1,000 observations per replication |
| folds | one shared treatment-stratified ten-fold assignment per sample |
| inference | pointwise 95% Wald intervals; RR and OR use the log scale |
| propensity bounds | 0.025–0.975 |
| committed artifacts | [performance tests](../../../tests/canonical/tmle3_cvtmle/performance-tests.csv), [paired tests](../../../tests/canonical/tmle3_cvtmle/equivalence.csv), [property tests](../../../tests/canonical/tmle3_cvtmle/properties.csv), [manifest](../../../tests/canonical/tmle3_cvtmle/manifest.json) |
| reproduction | [`tests/canonical/tmle3_cvtmle/README.md`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3_cvtmle/README.md) |

The PAF implementations target the same parameter but report uncertainty on different native
scales. Their point performance and coverage are compared; raw standard-error calibration is
therefore marked `N/A` in the paired table.

## How to read the tests

The reported 99% intervals quantify Monte Carlo uncertainty across simulation replications; they
are not the estimator's nominal 95% confidence intervals. Bias is computed on the inference scale
shown in the results table. **Pass** records only whether the predeclared decision rule was met.

| test | statistic | Monte Carlo uncertainty | decision rule |
| --- | --- | --- | --- |
| bias equivalence | mean error `mean(estimate − truth)`; the margin is 0.25 times the empirical SD of the estimates | two-sided 99% Student-t interval for mean error | the complete interval lies inside the displayed ± margin |
| coverage validity | mean of the indicators that the nominal 95% interval contains truth | two-sided exact 99% Clopper–Pearson interval | the lower endpoint is at least 0.90 |
| SE ratio | mean reported SE divided by the empirical SD of the estimates; 1 denotes equal scales | 99% percentile interval from 10,000 rowwise bootstrap resamples | the complete interval lies inside 0.80–1.20 |
| paired similarity | mean cleverly-minus-reference estimate on identical simulated samples | two-sided 99% paired Student-t interval; the margin is 0.15 times the pooled empirical SD | the complete interval lies inside the displayed ± margin |
| RMSE non-inferiority | cleverly RMSE divided by reference RMSE, both against known truth | one-sided 99% upper percentile bound from 10,000 paired bootstrap resamples | the upper bound is at most 1.10 |
| coverage non-inferiority | cleverly coverage minus reference coverage on paired samples | one-sided 99% lower percentile bound from the paired bootstrap | the lower bound is at least −0.025 |
| calibration non-inferiority | excess absolute deviation of the cleverly SE ratio from 1 relative to the reference | one-sided 99% upper percentile bound from the paired bootstrap | the upper bound is at most 0.05; `N/A` means native SE scales differ |

A performance row passes only if its bias, coverage, and SE-ratio rules all pass. A paired row
passes only if similarity and every applicable non-inferiority rule pass. A control marked
**Pass** means the control moved in the required opposing direction; it does not mean the
deliberately in-sample or misspecified fit was valid.

### Scientific-property test meanings

| test | design and statistic | uncertainty and decision rule |
| --- | --- | --- |
| double robustness | ATE bias with both nuisances correct, outcome only correct, treatment only correct, and both wrong | the first three Student-t bias intervals must lie inside ±0.25 empirical SD; the `both_wrong` interval must lie wholly outside its margin |
| root-n and efficiency | ATE bias, nominal-interval coverage, and mean-SE/empirical-SD ratio at `n` = 500, 2,000, and 8,000 | at each size: bias equivalence, exact coverage lower bound at least 0.90, and the point SE ratio in 0.80–1.20 |
| root-n rate | OLS slope of log empirical SD on log `n`, and separately of log mean reported SE on log `n` | 99% percentile intervals from 10,000 within-size bootstrap resamples must lie in −0.5 ± 0.125 and exclude −0.25 |
| interval calibration | coverage and SE ratio where both nuisance models are correctly specified | the exact 99% coverage interval must lie in 0.92–0.98 and the 10,000-resample 99% SE-ratio interval in 0.93–1.07 |
| type-I error | Wald rejection rate for the ATE under a confounded sharp null, paired with interval coverage | the exact 99% rejection upper bound must be at most 0.10 and the exact coverage lower bound at least 0.90 |
| power | Wald rejection rate under the registered nonzero ATE alternative | the exact 99% rejection lower bound must be at least 0.80 |
| cross-fit overfitting | coverage and SE ratio for held-out tree nuisances and a deliberately in-sample tree on the same 400 samples of size 500 | 10,000-resample 99% intervals require held-out SE ratio in 0.85–1.20, control SE-ratio upper bound at most 0.75, and paired coverage-gain lower bound at least 0.15 |

## Test-by-test results

The tables below are generated from the committed artifacts. Parentheses in one-sided comparison
columns contain the applicable 99% bound; square brackets contain two-sided 99% intervals.

<!-- BEGIN GENERATED STUDY RESULTS -->
### Performance versus truth

#### Binary outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly-stacked-cvtmle | `ey1` | identity | -0.0014 [-0.0028, 0.000051] | ±0.0056 | Pass | 0.9375 [0.9203, 0.9521] | Pass | 0.9906 [0.9461, 1.0421] | Pass | Pass |
| cleverly-stacked-cvtmle | `ey0` | identity | 0.000715 [-0.000696, 0.0021] | ±0.0055 | Pass | 0.9556 [0.9406, 0.9678] | Pass | 1.0076 [0.9639, 1.0552] | Pass | Pass |
| cleverly-stacked-cvtmle | `ate` | identity | -0.0021 [-0.0040, -0.000199] | ±0.0074 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0163 [0.9735, 1.0637] | Pass | Pass |
| cleverly-stacked-cvtmle | `att` | identity | -0.0019 [-0.0039, 0.000098] | ±0.0077 | Pass | 0.9519 [0.9364, 0.9646] | Pass | 1.0216 [0.9771, 1.0697] | Pass | Pass |
| cleverly-stacked-cvtmle | `atc` | identity | -0.0023 [-0.0043, -0.000378] | ±0.0076 | Pass | 0.9456 [0.9293, 0.9592] | Pass | 1.0139 [0.9696, 1.0635] | Pass | Pass |
| cleverly-stacked-cvtmle | `ey_obs` | identity | -0.000235 [-0.0013, 0.000802] | ±0.0040 | Pass | 0.9456 [0.9293, 0.9592] | Pass | 0.9815 [0.9368, 1.0305] | Pass | Pass |
| cleverly-stacked-cvtmle | `par` | identity | -0.000950 [-0.0020, 0.000061] | ±0.0039 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0209 [0.9772, 1.0703] | Pass | Pass |
| cleverly-stacked-cvtmle | `paf` | identity | -0.0017 [-0.0039, 0.000426] | ±0.0084 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0234 [0.9791, 1.0745] | Pass | Pass |
| cleverly-stacked-cvtmle | `rr` | log | -0.0035 [-0.0078, 0.000826] | ±0.0168 | Pass | 0.9475 [0.9314, 0.9608] | Pass | 1.0206 [0.9778, 1.0670] | Pass | Pass |
| cleverly-stacked-cvtmle | `or` | log | -0.0071 [-0.0151, 0.000844] | ±0.0309 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0167 [0.9725, 1.0646] | Pass | Pass |
| tmle3-cvtmle | `ey1` | identity | -0.0014 [-0.0028, 0.000051] | ±0.0056 | Pass | 0.9375 [0.9203, 0.9521] | Pass | 0.9906 [0.9458, 1.0404] | Pass | Pass |
| tmle3-cvtmle | `ey0` | identity | 0.000715 [-0.000696, 0.0021] | ±0.0055 | Pass | 0.9556 [0.9406, 0.9678] | Pass | 1.0076 [0.9640, 1.0577] | Pass | Pass |
| tmle3-cvtmle | `ate` | identity | -0.0021 [-0.0040, -0.000201] | ±0.0074 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 1.0164 [0.9720, 1.0636] | Pass | Pass |
| tmle3-cvtmle | `att` | identity | -0.0019 [-0.0039, 0.000081] | ±0.0077 | Pass | 0.9525 [0.9371, 0.9651] | Pass | 1.0257 [0.9812, 1.0764] | Pass | Pass |
| tmle3-cvtmle | `atc` | identity | -0.0023 [-0.0042, -0.000352] | ±0.0075 | Pass | 0.9469 [0.9307, 0.9603] | Pass | 1.0195 [0.9757, 1.0696] | Pass | Pass |
| tmle3-cvtmle | `ey_obs` | identity | -0.000235 [-0.0013, 0.000802] | ±0.0040 | Pass | 0.9456 [0.9293, 0.9592] | Pass | 0.9815 [0.9393, 1.0284] | Pass | Pass |
| tmle3-cvtmle | `par` | identity | -0.000950 [-0.0020, 0.000061] | ±0.0039 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 1.0212 [0.9774, 1.0709] | Pass | Pass |
| tmle3-cvtmle | `paf` | negative_log_complement | -0.0013 [-0.0040, 0.0014] | ±0.0105 | Pass | 0.9525 [0.9371, 0.9651] | Pass | 1.0233 [0.9801, 1.0725] | Pass | Pass |
| tmle3-cvtmle | `rr` | log | -0.0035 [-0.0078, 0.000826] | ±0.0168 | Pass | 0.9475 [0.9314, 0.9608] | Pass | 1.0205 [0.9783, 1.0680] | Pass | Pass |
| tmle3-cvtmle | `or` | log | -0.0071 [-0.0151, 0.000844] | ±0.0309 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0166 [0.9738, 1.0645] | Pass | Pass |

#### Continuous outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly-stacked-cvtmle | `ey1` | identity | -0.000069 [-0.000593, 0.000456] | ±0.0020 | Pass | 0.9544 [0.9392, 0.9668] | Pass | 1.0142 [0.9697, 1.0612] | Pass | Pass |
| cleverly-stacked-cvtmle | `ey0` | identity | 0.000014 [-0.000446, 0.000473] | ±0.0018 | Pass | 0.9469 [0.9307, 0.9603] | Pass | 0.9851 [0.9424, 1.0329] | Pass | Pass |
| cleverly-stacked-cvtmle | `ate` | identity | -0.000082 [-0.000502, 0.000337] | ±0.0016 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0260 [0.9811, 1.0750] | Pass | Pass |
| cleverly-stacked-cvtmle | `att` | identity | -0.000028 [-0.000483, 0.000426] | ±0.0018 | Pass | 0.9444 [0.9279, 0.9581] | Pass | 0.9887 [0.9475, 1.0356] | Pass | Pass |
| cleverly-stacked-cvtmle | `atc` | identity | -0.000102 [-0.000556, 0.000353] | ±0.0018 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 0.9995 [0.9563, 1.0469] | Pass | Pass |
| cleverly-stacked-cvtmle | `ey_obs` | identity | 0.000047 [-0.000462, 0.000556] | ±0.0020 | Pass | 0.9431 [0.9265, 0.9570] | Pass | 0.9862 [0.9451, 1.0332] | Pass | Pass |
| cleverly-stacked-cvtmle | `par` | identity | 0.000033 [-0.000261, 0.000327] | ±0.0011 | Pass | 0.9387 [0.9216, 0.9532] | Pass | 0.9762 [0.9326, 1.0239] | Pass | Pass |
| tmle3-cvtmle | `ey1` | identity | -0.000068 [-0.000592, 0.000456] | ±0.0020 | Pass | 0.9556 [0.9406, 0.9678] | Pass | 1.0159 [0.9732, 1.0633] | Pass | Pass |
| tmle3-cvtmle | `ey0` | identity | 0.000021 [-0.000438, 0.000480] | ±0.0018 | Pass | 0.9463 [0.9300, 0.9597] | Pass | 0.9853 [0.9433, 1.0325] | Pass | Pass |
| tmle3-cvtmle | `ate` | identity | -0.000082 [-0.000500, 0.000335] | ±0.0016 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 1.0391 [0.9934, 1.0901] | Pass | Pass |
| tmle3-cvtmle | `att` | identity | -0.000762 [-0.0012, -0.000308] | ±0.0018 | Pass | 0.9406 [0.9237, 0.9548] | Pass | 0.9904 [0.9487, 1.0379] | Pass | Pass |
| tmle3-cvtmle | `atc` | identity | 0.000613 [0.000163, 0.0011] | ±0.0017 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0061 [0.9621, 1.0542] | Pass | Pass |
| tmle3-cvtmle | `ey_obs` | identity | 0.000047 [-0.000462, 0.000556] | ±0.0020 | Pass | 0.9431 [0.9265, 0.9570] | Pass | 0.9862 [0.9437, 1.0345] | Pass | Pass |
| tmle3-cvtmle | `par` | identity | 0.000004 [-0.000289, 0.000297] | ±0.0011 | Pass | 0.9394 [0.9223, 0.9537] | Pass | 0.9852 [0.9429, 1.0327] | Pass | Pass |

### Cross-implementation tests

#### Binary outcome law

| estimand | paired difference (99% CI) | similarity margin | similarity | RMSE ratio (99% upper) | RMSE NI | coverage difference (99% lower) | coverage NI | calibration excess (99% upper) | calibration NI | overall |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `ey1` | 0.000000 [-0.000000, 0.000000] | ±0.0034 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000050) | Pass | Pass |
| `ey0` | 0.000000 [-0.000000, 0.000000] | ±0.0033 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000044 (0.000055) | Pass | Pass |
| `ate` | 0.000002 [-0.000002, 0.000006] | ±0.0045 | Pass | 1.0000 (1.0001) | Pass | -0.0013 (-0.0037) | Pass | 0 (0.000177) | Pass | Pass |
| `att` | 0.000010 [-0.000037, 0.000056] | ±0.0046 | Pass | 1.0041 (1.0054) | Pass | -0.000625 (-0.0037) | Pass | 0 (0.0045) | Pass | Pass |
| `atc` | -0.000037 [-0.000085, 0.000012] | ±0.0045 | Pass | 1.0056 (1.0070) | Pass | -0.0012 (-0.0044) | Pass | 0 (0.0062) | Pass | Pass |
| `ey_obs` | -0.000000 [-0.000000, -0.000000] | ±0.0024 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000000) | Pass | Pass |
| `par` | -0.000000 [-0.000003, 0.000003] | ±0.0024 | Pass | 1.0004 (1.0007) | Pass | -0.000625 (-0.0025) | Pass | 0 (0.000400) | Pass | Pass |
| `paf` | -0.000000 [-0.000006, 0.000006] | ±0.0051 | Pass | 1.0004 (1.0007) | Pass | -0.0025 (-0.0081) | Pass | N/A | N/A | Pass |
| `rr` | -0.000000 [-0.000001, 0.000001] | ±0.0151 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000061 (0.000073) | Pass | Pass |
| `or` | 0.000000 [-0.000004, 0.000004] | ±0.0398 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000061 (0.000073) | Pass | Pass |

#### Continuous outcome law

| estimand | paired difference (99% CI) | similarity margin | similarity | RMSE ratio (99% upper) | RMSE NI | coverage difference (99% lower) | coverage NI | calibration excess (99% upper) | calibration NI | overall |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `ey1` | -0.000001 [-0.000016, 0.000015] | ±0.0012 | Pass | 1.0020 (1.0039) | Pass | -0.0012 (-0.0044) | Pass | 0 (0.0030) | Pass | Pass |
| `ey0` | -0.000007 [-0.000021, 0.000006] | ±0.0011 | Pass | 1.0013 (1.0031) | Pass | 0.000625 (0) | Pass | 0.000277 (0.0018) | Pass | Pass |
| `ate` | 0.000000 [-0.000026, 0.000026] | ±0.000974 | Pass | 1.0045 (1.0082) | Pass | -0.0013 (-0.0056) | Pass | 0 (0.0116) | Pass | Pass |
| `att` | 0.000733 [0.000704, 0.000762] | ±0.0011 | Pass | 0.9963 (1.0042) | Pass | 0.0037 (-0.0037) | Pass | 0.0017 (0.0056) | Pass | Pass |
| `atc` | -0.000715 [-0.000749, -0.000681] | ±0.0011 | Pass | 1.0052 (1.0125) | Pass | -0.000625 (-0.0063) | Pass | 0 (0.0098) | Pass | Pass |
| `ey_obs` | -0.000000 [-0.000000, 0.000000] | ±0.0012 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000000) | Pass | Pass |
| `par` | 0.000029 [0.000016, 0.000042] | ±0.000683 | Pass | 1.0039 (1.0072) | Pass | -0.000625 (-0.0044) | Pass | 0.0090 (0.0114) | Pass | Pass |

### Scientific-property and control tests

| test | cell | n | replications | observed result | uncertainty | decision rule | status |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Double robustness | `both_correct` | 700 | 1,200 | bias -0.0031 | bias CI [-0.0100, 0.0038] | 99% bias CI inside ±0.0231 | Pass |
| Double robustness | `outcome_correct` | 700 | 1,200 | bias 0.000149 | bias CI [-0.0064, 0.0067] | 99% bias CI inside ±0.0219 | Pass |
| Double robustness | `treatment_correct` | 700 | 1,200 | bias -0.0196 | bias CI [-0.0305, -0.0086] | 99% bias CI inside ±0.0366 | Pass |
| Double robustness | `both_wrong` | 700 | 1,200 | bias -0.3348 | bias CI [-0.3437, -0.3259] | 99% bias CI outside ±0.0299 | Pass |
| Root-n and efficiency | `n_500` | 500 | 800 | bias -0.000654; coverage 0.9487; SE ratio 1.0128 | bias CI [-0.0090, 0.0077]; coverage CI [0.9252, 0.9667] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_2000` | 2,000 | 800 | bias -0.000506; coverage 0.9475; SE ratio 0.9823 | bias CI [-0.0048, 0.0037]; coverage CI [0.9238, 0.9657] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_8000` | 8,000 | 800 | bias 0.000293; coverage 0.9550; SE ratio 1.0087 | bias CI [-0.0018, 0.0024]; coverage CI [0.9326, 0.9717] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n rate | `empirical_sd` | 8,000 | 2,400 | slope -0.5053 | slope CI [-0.5376, -0.4731] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Root-n rate | `reported_se` | 8,000 | 2,400 | slope -0.5067 | slope CI [-0.5080, -0.5054] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Interval calibration | `correctly_specified` | 2,000 | 2,400 | coverage 0.9550; SE ratio 1.0077 | coverage CI [0.9430, 0.9652]; SE-ratio CI [0.9731, 1.0453] | coverage CI in [0.9200, 0.9800]; SE-ratio CI in [0.9300, 1.0700] | Pass |
| Type-I error | `sharp_null` | 1,000 | 400 | rejection 0.0325; coverage 0.9675 | rejection upper 0.0627; coverage lower 0.9373 | rejection upper ≤ 0.1000; coverage lower ≥ 0.9000 | Pass |
| Power | `alternative` | 1,000 | 400 | rejection 1 | rejection CI [0.9868, 1] | 99% rejection lower ≥ 0.8000 | Pass |
| Cross-fit overfitting | `stacked_cvtmle` | 500 | 400 | coverage 0.8950; SE ratio 0.9880 | SE-ratio CI [0.9076, 1.0852]; coverage-gain CI [0.1850, 0.3050] | cross-fit SE-ratio CI in [0.8500, 1.2000]; control upper ≤ 0.7500; coverage-gain lower ≥ 0.1500 | Pass |
| Cross-fit overfitting | `in_sample_control` | 500 | 400 | coverage 0.6500; SE ratio 0.5792 | SE-ratio CI [0.5299, 0.6424]; coverage-gain CI [0.1850, 0.3050] | cross-fit SE-ratio CI in [0.8500, 1.2000]; control upper ≤ 0.7500; coverage-gain lower ≥ 0.1500 | Pass |
<!-- END GENERATED STUDY RESULTS -->

## Study boundary

The registered study covers the two complete-outcome point-treatment laws, corresponding GLM
nuisance regressions, one shared ten-fold split, a pooled update, whole-sample evaluation, the
stated bounds, and pointwise Wald inference. It does not cover repeated or nested cross-fitting,
fold-evaluated or fold-specific-epsilon CV-TMLE, simultaneous or bootstrap intervals, missing
outcomes, weights, clusters, strata, multi-valued treatment, broad learner-library selection, or
severe practical-positivity settings.
