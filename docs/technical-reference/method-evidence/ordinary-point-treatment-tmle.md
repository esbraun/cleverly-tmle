# Ordinary point-treatment TMLE study

[← Method evidence grid](../../methodology.md#method-evidence-grid)

This study applies ordinary, non-cross-fitted point-treatment TMLE to repeated samples from laws
with known parameter values. The same realized samples are also fitted with the pinned R `tmle3`
reference where a comparison is registered.

## Study setup

| setting | registered value |
| --- | --- |
| cleverly method | ordinary point-treatment TMLE without nuisance cross-fitting |
| external reference | R `tmle3` 0.2.0 at commit [`ed72f8a`](https://github.com/tlverse/tmle3/tree/ed72f8a20e64c914ab25ffe015d865f7a9963d27) |
| laws | binary outcome; bounded continuous outcome |
| estimands | `ey1`, `ey0`, `ate`, `att`, `atc`, `ey_obs`, `par`; plus `paf`, `rr`, and `or` for the binary law |
| primary run | 1,600 replications per law; 1,000 observations per replication |
| inference | pointwise 95% Wald intervals; RR and OR use the log scale |
| propensity bounds | 0.01–0.99 |
| committed artifacts | [performance tests](../../../tests/canonical/tmle3/performance-tests.csv), [paired tests](../../../tests/canonical/tmle3/equivalence.csv), [property tests](../../../tests/canonical/tmle3/properties.csv), [manifest](../../../tests/canonical/tmle3/manifest.json) |
| reproduction | [`tests/canonical/tmle3/README.md`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle3/README.md) |

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
deliberately misspecified fit was valid.

### Scientific-property test meanings

| test | design and statistic | uncertainty and decision rule |
| --- | --- | --- |
| double robustness | ATE bias with both nuisances correct, outcome only correct, treatment only correct, and both wrong | the first three Student-t bias intervals must lie inside ±0.25 empirical SD; the `both_wrong` interval must lie wholly outside its margin |
| root-n and efficiency | ATE bias, nominal-interval coverage, and mean-SE/empirical-SD ratio at `n` = 500, 2,000, and 8,000 | at each size: bias equivalence, exact coverage lower bound at least 0.90, and the point SE ratio in 0.80–1.20 |
| root-n rate | OLS slope of log empirical SD on log `n`, and separately of log mean reported SE on log `n` | 99% percentile intervals from 10,000 within-size bootstrap resamples must lie in −0.5 ± 0.125 and exclude −0.25 |
| interval calibration | coverage and SE ratio where both nuisance models are correctly specified | the exact 99% coverage interval must lie in 0.92–0.98 and the 10,000-resample 99% SE-ratio interval in 0.93–1.07 |
| type-I error | Wald rejection rate for the ATE under a confounded sharp null, paired with interval coverage | the exact 99% rejection upper bound must be at most 0.10 and the exact coverage lower bound at least 0.90 |
| power | Wald rejection rate under the registered nonzero ATE alternative | the exact 99% rejection lower bound must be at least 0.80 |

## Test-by-test results

The tables below are generated from the committed artifacts. Parentheses in one-sided comparison
columns contain the applicable 99% bound; square brackets contain two-sided 99% intervals.

<!-- BEGIN GENERATED STUDY RESULTS -->
### Performance versus truth

#### Binary outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly | `ey1` | identity | -0.000144 [-0.0016, 0.0013] | ±0.0056 | Pass | 0.9425 [0.9258, 0.9565] | Pass | 0.9861 [0.9406, 1.0365] | Pass | Pass |
| cleverly | `ey0` | identity | 0.000593 [-0.000802, 0.0020] | ±0.0054 | Pass | 0.9531 [0.9378, 0.9657] | Pass | 1.0102 [0.9647, 1.0588] | Pass | Pass |
| cleverly | `ate` | identity | -0.000737 [-0.0026, 0.0012] | ±0.0074 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0067 [0.9620, 1.0547] | Pass | Pass |
| cleverly | `att` | identity | -0.000833 [-0.0028, 0.0012] | ±0.0077 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 1.0088 [0.9645, 1.0593] | Pass | Pass |
| cleverly | `atc` | identity | -0.000635 [-0.0026, 0.0013] | ±0.0076 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 0.9997 [0.9552, 1.0475] | Pass | Pass |
| cleverly | `ey_obs` | identity | 0.000168 [-0.000858, 0.0012] | ±0.0040 | Pass | 0.9450 [0.9286, 0.9586] | Pass | 0.9919 [0.9489, 1.0395] | Pass | Pass |
| cleverly | `par` | identity | -0.000425 [-0.0014, 0.000586] | ±0.0039 | Pass | 0.9469 [0.9307, 0.9603] | Pass | 1.0082 [0.9640, 1.0565] | Pass | Pass |
| cleverly | `paf` | identity | -0.000817 [-0.0030, 0.0013] | ±0.0084 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0118 [0.9672, 1.0607] | Pass | Pass |
| cleverly | `rr` | log | -0.000975 [-0.0053, 0.0033] | ±0.0167 | Pass | 0.9556 [0.9406, 0.9678] | Pass | 1.0117 [0.9674, 1.0599] | Pass | Pass |
| cleverly | `or` | log | -0.0015 [-0.0095, 0.0064] | ±0.0309 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0074 [0.9622, 1.0580] | Pass | Pass |
| tmle3 | `ey1` | identity | -0.000144 [-0.0016, 0.0013] | ±0.0056 | Pass | 0.9425 [0.9258, 0.9565] | Pass | 0.9861 [0.9389, 1.0362] | Pass | Pass |
| tmle3 | `ey0` | identity | 0.000593 [-0.000802, 0.0020] | ±0.0054 | Pass | 0.9531 [0.9378, 0.9657] | Pass | 1.0101 [0.9641, 1.0606] | Pass | Pass |
| tmle3 | `ate` | identity | -0.000736 [-0.0026, 0.0012] | ±0.0074 | Pass | 0.9487 [0.9328, 0.9619] | Pass | 1.0066 [0.9617, 1.0561] | Pass | Pass |
| tmle3 | `att` | identity | -0.000811 [-0.0028, 0.0012] | ±0.0077 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0141 [0.9705, 1.0638] | Pass | Pass |
| tmle3 | `atc` | identity | -0.000658 [-0.0026, 0.0013] | ±0.0076 | Pass | 0.9544 [0.9392, 0.9668] | Pass | 1.0049 [0.9613, 1.0545] | Pass | Pass |
| tmle3 | `ey_obs` | identity | 0.000168 [-0.000858, 0.0012] | ±0.0040 | Pass | 0.9450 [0.9286, 0.9586] | Pass | 0.9919 [0.9489, 1.0401] | Pass | Pass |
| tmle3 | `par` | identity | -0.000424 [-0.0014, 0.000587] | ±0.0039 | Pass | 0.9469 [0.9307, 0.9603] | Pass | 1.0081 [0.9635, 1.0578] | Pass | Pass |
| tmle3 | `paf` | negative_log_complement | -0.000138 [-0.0029, 0.0026] | ±0.0105 | Pass | 0.9487 [0.9328, 0.9619] | Pass | 1.0114 [0.9675, 1.0594] | Pass | Pass |
| tmle3 | `rr` | log | -0.000974 [-0.0053, 0.0033] | ±0.0167 | Pass | 0.9556 [0.9406, 0.9678] | Pass | 1.0116 [0.9666, 1.0619] | Pass | Pass |
| tmle3 | `or` | log | -0.0015 [-0.0095, 0.0064] | ±0.0309 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0074 [0.9633, 1.0562] | Pass | Pass |

#### Continuous outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly | `ey1` | identity | 0.000264 [-0.000260, 0.000788] | ±0.0020 | Pass | 0.9519 [0.9364, 0.9646] | Pass | 1.0107 [0.9673, 1.0571] | Pass | Pass |
| cleverly | `ey0` | identity | 0.000298 [-0.000147, 0.000743] | ±0.0017 | Pass | 0.9531 [0.9378, 0.9657] | Pass | 1.0146 [0.9706, 1.0630] | Pass | Pass |
| cleverly | `ate` | identity | -0.000034 [-0.000459, 0.000392] | ±0.0017 | Pass | 0.9481 [0.9321, 0.9614] | Pass | 1.0015 [0.9577, 1.0511] | Pass | Pass |
| cleverly | `att` | identity | -0.000144 [-0.000602, 0.000314] | ±0.0018 | Pass | 0.9394 [0.9223, 0.9537] | Pass | 0.9706 [0.9279, 1.0184] | Pass | Pass |
| cleverly | `atc` | identity | 0.000054 [-0.000400, 0.000508] | ±0.0018 | Pass | 0.9463 [0.9300, 0.9597] | Pass | 0.9893 [0.9441, 1.0361] | Pass | Pass |
| cleverly | `ey_obs` | identity | 0.000272 [-0.000234, 0.000779] | ±0.0020 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 0.9908 [0.9476, 1.0398] | Pass | Pass |
| cleverly | `par` | identity | -0.000026 [-0.000327, 0.000276] | ±0.0012 | Pass | 0.9344 [0.9168, 0.9493] | Pass | 0.9451 [0.9041, 0.9906] | Pass | Pass |
| tmle3 | `ey1` | identity | 0.000279 [-0.000244, 0.000802] | ±0.0020 | Pass | 0.9519 [0.9364, 0.9646] | Pass | 1.0122 [0.9693, 1.0619] | Pass | Pass |
| tmle3 | `ey0` | identity | 0.000298 [-0.000147, 0.000742] | ±0.0017 | Pass | 0.9537 [0.9385, 0.9662] | Pass | 1.0149 [0.9707, 1.0639] | Pass | Pass |
| tmle3 | `ate` | identity | -0.000018 [-0.000442, 0.000406] | ±0.0016 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0125 [0.9671, 1.0626] | Pass | Pass |
| tmle3 | `att` | identity | -0.000888 [-0.0013, -0.000431] | ±0.0018 | Pass | 0.9400 [0.9230, 0.9543] | Pass | 0.9726 [0.9289, 1.0210] | Pass | Pass |
| tmle3 | `atc` | identity | 0.000837 [0.000387, 0.0013] | ±0.0017 | Pass | 0.9475 [0.9314, 0.9608] | Pass | 0.9957 [0.9531, 1.0427] | Pass | Pass |
| tmle3 | `ey_obs` | identity | 0.000272 [-0.000234, 0.000779] | ±0.0020 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 0.9908 [0.9462, 1.0400] | Pass | Pass |
| tmle3 | `par` | identity | -0.000024 [-0.000325, 0.000277] | ±0.0012 | Pass | 0.9375 [0.9203, 0.9521] | Pass | 0.9526 [0.9106, 0.9991] | Pass | Pass |

### Cross-implementation tests

#### Binary outcome law

| estimand | paired difference (99% CI) | similarity margin | similarity | RMSE ratio (99% upper) | RMSE NI | coverage difference (99% lower) | coverage NI | calibration excess (99% upper) | calibration NI | overall |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `ey1` | -0.000000 [-0.000000, 0.000000] | ±0.0034 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000040) | Pass | Pass |
| `ey0` | 0.000000 [-0.000000, 0.000000] | ±0.0032 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000040 (0.000049) | Pass | Pass |
| `ate` | -0.000001 [-0.000004, 0.000003] | ±0.0044 | Pass | 0.9999 (1.0000) | Pass | 0.000625 (0) | Pass | 0.000032 (0.000129) | Pass | Pass |
| `att` | -0.000022 [-0.000070, 0.000026] | ±0.0046 | Pass | 1.0054 (1.0067) | Pass | 0.000625 (-0.0031) | Pass | 0 (0.0062) | Pass | Pass |
| `atc` | 0.000023 [-0.000025, 0.000071] | ±0.0046 | Pass | 1.0052 (1.0065) | Pass | -0.0037 (-0.0081) | Pass | 0 (0.0061) | Pass | Pass |
| `ey_obs` | -0.000000 [-0.000000, -0.000000] | ±0.0024 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000000) | Pass | Pass |
| `par` | -0.000000 [-0.000002, 0.000002] | ±0.0024 | Pass | 1.0001 (1.0002) | Pass | 0 (0) | Pass | 0.000041 (0.000146) | Pass | Pass |
| `paf` | -0.000001 [-0.000005, 0.000003] | ±0.0050 | Pass | 1.0001 (1.0002) | Pass | 0.000625 (-0.0044) | Pass | N/A | N/A | Pass |
| `rr` | -0.000000 [-0.000001, 0.000000] | ±0.0151 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000052 (0.000059) | Pass | Pass |
| `or` | -0.000001 [-0.000002, 0.000001] | ±0.0400 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000051 (0.000058) | Pass | Pass |

#### Continuous outcome law

| estimand | paired difference (99% CI) | similarity margin | similarity | RMSE ratio (99% upper) | RMSE NI | coverage difference (99% lower) | coverage NI | calibration excess (99% upper) | calibration NI | overall |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `ey1` | -0.000015 [-0.000029, -0.000001] | ±0.0012 | Pass | 1.0017 (1.0033) | Pass | 0 (-0.0019) | Pass | 0 (0.0025) | Pass | Pass |
| `ey0` | -0.000000 [-0.000013, 0.000013] | ±0.0010 | Pass | 1.0013 (1.0030) | Pass | -0.000625 (-0.0025) | Pass | 0 (0.0015) | Pass | Pass |
| `ate` | -0.000016 [-0.000039, 0.000008] | ±0.000989 | Pass | 1.0030 (1.0062) | Pass | -0.0019 (-0.0063) | Pass | 0 (0.0128) | Pass | Pass |
| `att` | 0.000744 [0.000717, 0.000772] | ±0.0011 | Pass | 0.9948 (1.0019) | Pass | -0.000625 (-0.0069) | Pass | 0.0021 (0.0054) | Pass | Pass |
| `atc` | -0.000783 [-0.000814, -0.000751] | ±0.0011 | Pass | 1.0009 (1.0087) | Pass | -0.0012 (-0.0088) | Pass | 0.0064 (0.0097) | Pass | Pass |
| `ey_obs` | -0.000000 [-0.000000, -0.000000] | ±0.0012 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000000) | Pass | Pass |
| `par` | -0.000001 [-0.000014, 0.000011] | ±0.000701 | Pass | 1.0023 (1.0045) | Pass | -0.0031 (-0.0069) | Pass | 0.0075 (0.0095) | Pass | Pass |

### Scientific-property and control tests

| test | cell | n | replications | observed result | uncertainty | decision rule | status |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Double robustness | `both_correct` | 700 | 1,200 | bias -0.0034 | bias CI [-0.0104, 0.0037] | 99% bias CI inside ±0.0236 | Pass |
| Double robustness | `outcome_correct` | 700 | 1,200 | bias 0.000306 | bias CI [-0.0062, 0.0069] | 99% bias CI inside ±0.0220 | Pass |
| Double robustness | `treatment_correct` | 700 | 1,200 | bias -0.0182 | bias CI [-0.0292, -0.0072] | 99% bias CI inside ±0.0369 | Pass |
| Double robustness | `both_wrong` | 700 | 1,200 | bias -0.3332 | bias CI [-0.3420, -0.3244] | 99% bias CI outside ±0.0295 | Pass |
| Root-n and efficiency | `n_500` | 500 | 800 | bias -0.000770; coverage 0.9463; SE ratio 0.9913 | bias CI [-0.0091, 0.0076]; coverage CI [0.9223, 0.9647] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_2000` | 2,000 | 800 | bias -0.000513; coverage 0.9437; SE ratio 0.9777 | bias CI [-0.0048, 0.0037]; coverage CI [0.9194, 0.9627] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_8000` | 8,000 | 800 | bias 0.000285; coverage 0.9525; SE ratio 1.0077 | bias CI [-0.0018, 0.0023]; coverage CI [0.9297, 0.9698] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n rate | `empirical_sd` | 8,000 | 2,400 | slope -0.5045 | slope CI [-0.5361, -0.4719] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Root-n rate | `reported_se` | 8,000 | 2,400 | slope -0.4986 | slope CI [-0.4998, -0.4974] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Interval calibration | `correctly_specified` | 2,000 | 2,400 | coverage 0.9542; SE ratio 1.0027 | coverage CI [0.9420, 0.9645]; SE-ratio CI [0.9679, 1.0383] | coverage CI in [0.9200, 0.9800]; SE-ratio CI in [0.9300, 1.0700] | Pass |
| Type-I error | `sharp_null` | 1,000 | 400 | rejection 0.0325; coverage 0.9675 | rejection upper 0.0627; coverage lower 0.9373 | rejection upper ≤ 0.1000; coverage lower ≥ 0.9000 | Pass |
| Power | `alternative` | 1,000 | 400 | rejection 1 | rejection CI [0.9868, 1] | 99% rejection lower ≥ 0.8000 | Pass |
<!-- END GENERATED STUDY RESULTS -->

## Study boundary

The registered study covers the two complete-outcome point-treatment laws, corresponding GLM
nuisance regressions, ordinary targeting, the stated bounds, and pointwise Wald inference. It does
not cover nuisance cross-fitting, CV-TMLE evaluation, repeated folds, simultaneous or bootstrap
intervals, missing outcomes, weights, clusters, strata, multi-valued treatment, flexible learner
libraries, or severe practical-positivity settings.
