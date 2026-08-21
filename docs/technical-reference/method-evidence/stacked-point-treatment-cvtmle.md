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

Bias is computed on the inference scale shown in the results table. **Pass** records only whether
the predeclared decision rule was met. The rules themselves are not restated here: the
comparison rules are generated into [Decision rules applied](#decision-rules-applied) below, and
each scientific-property row carries its own rule in the results table, so a threshold cannot be
moved in the code and left standing in this prose.

| test | statistic | Monte Carlo uncertainty |
| --- | --- | --- |
| bias equivalence | mean error `mean(estimate − truth)`, against a margin scaled by the empirical SD of the estimates | two-sided 99% Student-t interval for mean error |
| coverage validity | mean of the indicators that the nominal 95% interval contains truth | two-sided exact 99% Clopper–Pearson interval |
| SE ratio | mean reported SE divided by the empirical SD of the estimates; 1 denotes equal scales | 99% percentile interval from 10,000 rowwise bootstrap resamples |
| paired similarity | mean cleverly-minus-reference estimate on identical simulated samples, against a margin scaled by the pooled empirical SD | two-sided 99% paired Student-t interval |
| RMSE non-inferiority | cleverly RMSE divided by reference RMSE, both against known truth | one-sided 99% upper percentile bound from 10,000 paired bootstrap resamples |
| coverage non-inferiority | cleverly coverage minus reference coverage on paired samples | one-sided 99% lower percentile bound from the paired bootstrap |
| calibration non-inferiority | excess absolute deviation of the cleverly SE ratio from 1 relative to the reference | one-sided 99% upper percentile bound from the paired bootstrap |

A performance row passes only if its bias, coverage, and SE-ratio rules all pass. A paired row
passes only if similarity and every applicable non-inferiority rule pass. A control marked
**Pass** means the control moved in the required opposing direction; it does not mean the
deliberately in-sample or misspecified fit was valid.

### Scientific-property test meanings

| test | design and statistic | Monte Carlo uncertainty |
| --- | --- | --- |
| double robustness | ATE bias with both nuisances correct, outcome only correct, treatment only correct, and both wrong | two-sided Student-t bias intervals; the positive cells must land inside their margin and the `both_wrong` control wholly outside it, so the same instrument is required to speak in both directions |
| root-n and efficiency | ATE bias, nominal-interval coverage, and mean-SE/empirical-SD ratio at `n` = 500, 2,000, and 8,000 | per size: a Student-t bias interval, an exact Clopper-Pearson coverage interval, and the point SE ratio |
| root-n rate | OLS slope of log empirical SD on log `n`, and separately of log mean reported SE on log `n` | within-size bootstrap percentile intervals for the fitted slope. The reported-SE row is the weaker of the two: an influence-curve standard error is sigma-hat over sqrt(n), so it returns the right exponent for any estimator that divides by the right power of `n`, consistent or not. It catches a standard error carrying the wrong power of `n` and nothing else; the empirical-SD row is the one that is evidence about the sampling distribution |
| interval calibration | coverage and SE ratio where both nuisance models are correctly specified | an exact coverage interval and a bootstrap SE-ratio interval, both two-sided and both required: a standard error inflated by a constant keeps coverage inside its band while failing the ratio, and a curve right on average but wrong replication by replication does the reverse |
| type-I error | Wald rejection rate for the ATE under a confounded sharp null, paired with interval coverage | exact one-sided bounds on the rejection rate and on coverage. One-sided because a test that over-rejects is invalid while one that under-rejects is merely conservative, which is what the power cell below exists to rule out |
| power | Wald rejection rate under the registered nonzero ATE alternative | an exact one-sided lower bound on the rejection rate, so a rejection indicator that never fires cannot pass the type-I cell by being inert |
| cross-fit overfitting | coverage and SE ratio for held-out tree nuisances and a deliberately in-sample tree on the same 400 samples of size 500 | bootstrap SE-ratio intervals for each arm and a paired bootstrap interval for the coverage difference on the same samples. Three statements, two per-arm and one paired; the results table gives each row its own |

## Test-by-test results

The tables below are generated from the committed artifacts. Parentheses in one-sided comparison
columns contain the applicable 99% bound; square brackets contain two-sided 99% intervals.

<!-- BEGIN GENERATED STUDY RESULTS -->
### Decision rules applied

Every interval below is a 99% Monte Carlo interval across simulation
replications; none of them is the estimator's nominal 95% confidence interval.
Equivalence margins are scaled by an observed spread, so they are printed per row.

| test | rule |
| --- | --- |
| bias equivalence | the whole 99% bias interval lies inside ± 0.2500 times the empirical SD of the estimates |
| coverage validity | the exact 99% Clopper-Pearson lower endpoint is at least 0.9000 |
| SE ratio | the whole 99% interval, from 10,000 rowwise bootstrap resamples, lies inside [0.8000, 1.2000] |
| paired similarity | the whole 99% paired interval lies inside ± 0.1500 times the pooled empirical SD |
| RMSE non-inferiority | the one-sided 99% upper bound on the RMSE ratio is at most 1.1000 |
| coverage non-inferiority | the one-sided 99% lower bound on the paired coverage difference is at least -0.0250 |
| calibration non-inferiority | the one-sided 99% upper bound on the excess absolute SE-ratio deviation is at most 0.0500; `N/A` means the native SE scales differ |

A performance row passes only if its bias, coverage and SE-ratio rules all pass. A
paired row passes only if similarity and every applicable non-inferiority rule pass.

### Performance versus truth

#### Binary outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly-stacked-cvtmle | `ey1` | identity | -0.0014 [-0.0028, 0.000051] | ±0.0056 | Pass | 0.9375 [0.9203, 0.9521] | Pass | 0.9906 [0.9459, 1.0405] | Pass | Pass |
| cleverly-stacked-cvtmle | `ey0` | identity | 0.000715 [-0.000696, 0.0021] | ±0.0055 | Pass | 0.9556 [0.9406, 0.9678] | Pass | 1.0076 [0.9620, 1.0577] | Pass | Pass |
| cleverly-stacked-cvtmle | `ate` | identity | -0.0021 [-0.0040, -0.000199] | ±0.0074 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0163 [0.9732, 1.0650] | Pass | Pass |
| cleverly-stacked-cvtmle | `att` | identity | -0.0019 [-0.0039, 0.000098] | ±0.0077 | Pass | 0.9519 [0.9364, 0.9646] | Pass | 1.0216 [0.9781, 1.0700] | Pass | Pass |
| cleverly-stacked-cvtmle | `atc` | identity | -0.0023 [-0.0043, -0.000378] | ±0.0076 | Pass | 0.9456 [0.9293, 0.9592] | Pass | 1.0139 [0.9701, 1.0624] | Pass | Pass |
| cleverly-stacked-cvtmle | `ey_obs` | identity | -0.000235 [-0.0013, 0.000802] | ±0.0040 | Pass | 0.9456 [0.9293, 0.9592] | Pass | 0.9815 [0.9379, 1.0297] | Pass | Pass |
| cleverly-stacked-cvtmle | `par` | identity | -0.000950 [-0.0020, 0.000061] | ±0.0039 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0209 [0.9778, 1.0693] | Pass | Pass |
| cleverly-stacked-cvtmle | `paf` | identity | -0.0017 [-0.0039, 0.000426] | ±0.0084 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0234 [0.9790, 1.0719] | Pass | Pass |
| cleverly-stacked-cvtmle | `rr` | log | -0.0035 [-0.0078, 0.000826] | ±0.0168 | Pass | 0.9475 [0.9314, 0.9608] | Pass | 1.0206 [0.9787, 1.0682] | Pass | Pass |
| cleverly-stacked-cvtmle | `or` | log | -0.0071 [-0.0151, 0.000844] | ±0.0309 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0167 [0.9739, 1.0630] | Pass | Pass |
| tmle3-cvtmle | `ey1` | identity | -0.0014 [-0.0028, 0.000051] | ±0.0056 | Pass | 0.9375 [0.9203, 0.9521] | Pass | 0.9906 [0.9461, 1.0400] | Pass | Pass |
| tmle3-cvtmle | `ey0` | identity | 0.000715 [-0.000696, 0.0021] | ±0.0055 | Pass | 0.9556 [0.9406, 0.9678] | Pass | 1.0076 [0.9635, 1.0563] | Pass | Pass |
| tmle3-cvtmle | `ate` | identity | -0.0021 [-0.0040, -0.000201] | ±0.0074 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 1.0164 [0.9732, 1.0643] | Pass | Pass |
| tmle3-cvtmle | `att` | identity | -0.0019 [-0.0039, 0.000081] | ±0.0077 | Pass | 0.9525 [0.9371, 0.9651] | Pass | 1.0257 [0.9815, 1.0742] | Pass | Pass |
| tmle3-cvtmle | `atc` | identity | -0.0023 [-0.0042, -0.000352] | ±0.0075 | Pass | 0.9469 [0.9307, 0.9603] | Pass | 1.0195 [0.9750, 1.0686] | Pass | Pass |
| tmle3-cvtmle | `ey_obs` | identity | -0.000235 [-0.0013, 0.000802] | ±0.0040 | Pass | 0.9456 [0.9293, 0.9592] | Pass | 0.9815 [0.9377, 1.0300] | Pass | Pass |
| tmle3-cvtmle | `par` | identity | -0.000950 [-0.0020, 0.000061] | ±0.0039 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 1.0212 [0.9783, 1.0711] | Pass | Pass |
| tmle3-cvtmle | `paf` | negative_log_complement | -0.0013 [-0.0040, 0.0014] | ±0.0105 | Pass | 0.9525 [0.9371, 0.9651] | Pass | 1.0233 [0.9797, 1.0732] | Pass | Pass |
| tmle3-cvtmle | `rr` | log | -0.0035 [-0.0078, 0.000826] | ±0.0168 | Pass | 0.9475 [0.9314, 0.9608] | Pass | 1.0205 [0.9769, 1.0680] | Pass | Pass |
| tmle3-cvtmle | `or` | log | -0.0071 [-0.0151, 0.000844] | ±0.0309 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0166 [0.9742, 1.0642] | Pass | Pass |

#### Continuous outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly-stacked-cvtmle | `ey1` | identity | -0.000069 [-0.000593, 0.000456] | ±0.0020 | Pass | 0.9544 [0.9392, 0.9668] | Pass | 1.0142 [0.9712, 1.0617] | Pass | Pass |
| cleverly-stacked-cvtmle | `ey0` | identity | 0.000014 [-0.000446, 0.000473] | ±0.0018 | Pass | 0.9469 [0.9307, 0.9603] | Pass | 0.9851 [0.9430, 1.0317] | Pass | Pass |
| cleverly-stacked-cvtmle | `ate` | identity | -0.000082 [-0.000502, 0.000337] | ±0.0016 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 1.0260 [0.9828, 1.0744] | Pass | Pass |
| cleverly-stacked-cvtmle | `att` | identity | -0.000028 [-0.000483, 0.000426] | ±0.0018 | Pass | 0.9444 [0.9279, 0.9581] | Pass | 0.9887 [0.9461, 1.0346] | Pass | Pass |
| cleverly-stacked-cvtmle | `atc` | identity | -0.000102 [-0.000556, 0.000353] | ±0.0018 | Pass | 0.9494 [0.9336, 0.9624] | Pass | 0.9995 [0.9549, 1.0482] | Pass | Pass |
| cleverly-stacked-cvtmle | `ey_obs` | identity | 0.000047 [-0.000462, 0.000556] | ±0.0020 | Pass | 0.9431 [0.9265, 0.9570] | Pass | 0.9862 [0.9423, 1.0350] | Pass | Pass |
| cleverly-stacked-cvtmle | `par` | identity | 0.000033 [-0.000261, 0.000327] | ±0.0011 | Pass | 0.9387 [0.9216, 0.9532] | Pass | 0.9762 [0.9329, 1.0232] | Pass | Pass |
| tmle3-cvtmle | `ey1` | identity | -0.000068 [-0.000592, 0.000456] | ±0.0020 | Pass | 0.9556 [0.9406, 0.9678] | Pass | 1.0159 [0.9732, 1.0650] | Pass | Pass |
| tmle3-cvtmle | `ey0` | identity | 0.000021 [-0.000438, 0.000480] | ±0.0018 | Pass | 0.9463 [0.9300, 0.9597] | Pass | 0.9853 [0.9429, 1.0326] | Pass | Pass |
| tmle3-cvtmle | `ate` | identity | -0.000082 [-0.000500, 0.000335] | ±0.0016 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 1.0391 [0.9957, 1.0884] | Pass | Pass |
| tmle3-cvtmle | `att` | identity | -0.000762 [-0.0012, -0.000308] | ±0.0018 | Pass | 0.9406 [0.9237, 0.9548] | Pass | 0.9904 [0.9494, 1.0361] | Pass | Pass |
| tmle3-cvtmle | `atc` | identity | 0.000613 [0.000163, 0.0011] | ±0.0017 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0061 [0.9625, 1.0554] | Pass | Pass |
| tmle3-cvtmle | `ey_obs` | identity | 0.000047 [-0.000462, 0.000556] | ±0.0020 | Pass | 0.9431 [0.9265, 0.9570] | Pass | 0.9862 [0.9438, 1.0324] | Pass | Pass |
| tmle3-cvtmle | `par` | identity | 0.000004 [-0.000289, 0.000297] | ±0.0011 | Pass | 0.9394 [0.9223, 0.9537] | Pass | 0.9852 [0.9411, 1.0344] | Pass | Pass |

### Cross-implementation tests

#### Binary outcome law

| estimand | paired difference (99% CI) | similarity margin | similarity | RMSE ratio (99% upper) | RMSE NI | coverage difference (99% lower) | coverage NI | calibration excess (99% upper) | calibration NI | overall |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `ey1` | 0.000000 [-0.000000, 0.000000] | ±0.0034 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000049) | Pass | Pass |
| `ey0` | 0.000000 [-0.000000, 0.000000] | ±0.0033 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000044 (0.000056) | Pass | Pass |
| `ate` | 0.000002 [-0.000002, 0.000006] | ±0.0045 | Pass | 1.0000 (1.0001) | Pass | -0.0013 (-0.0037) | Pass | 0 (0.000179) | Pass | Pass |
| `att` | 0.000010 [-0.000037, 0.000056] | ±0.0046 | Pass | 1.0041 (1.0054) | Pass | -0.000625 (-0.0037) | Pass | 0 (0.0046) | Pass | Pass |
| `atc` | -0.000037 [-0.000085, 0.000012] | ±0.0045 | Pass | 1.0056 (1.0070) | Pass | -0.0012 (-0.0044) | Pass | 0 (0.0062) | Pass | Pass |
| `ey_obs` | -0.000000 [-0.000000, -0.000000] | ±0.0024 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000000) | Pass | Pass |
| `par` | -0.000000 [-0.000003, 0.000003] | ±0.0024 | Pass | 1.0004 (1.0007) | Pass | -0.000625 (-0.0025) | Pass | 0 (0.000389) | Pass | Pass |
| `paf` | -0.000000 [-0.000006, 0.000006] | ±0.0051 | Pass | 1.0004 (1.0007) | Pass | -0.0025 (-0.0081) | Pass | N/A | N/A | Pass |
| `rr` | -0.000000 [-0.000001, 0.000001] | ±0.0151 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000061 (0.000074) | Pass | Pass |
| `or` | 0.000000 [-0.000004, 0.000004] | ±0.0398 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0.000061 (0.000073) | Pass | Pass |

#### Continuous outcome law

| estimand | paired difference (99% CI) | similarity margin | similarity | RMSE ratio (99% upper) | RMSE NI | coverage difference (99% lower) | coverage NI | calibration excess (99% upper) | calibration NI | overall |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `ey1` | -0.000001 [-0.000016, 0.000015] | ±0.0012 | Pass | 1.0020 (1.0039) | Pass | -0.0012 (-0.0044) | Pass | 0 (0.0030) | Pass | Pass |
| `ey0` | -0.000007 [-0.000021, 0.000006] | ±0.0011 | Pass | 1.0013 (1.0032) | Pass | 0.000625 (0) | Pass | 0.000277 (0.0018) | Pass | Pass |
| `ate` | 0.000000 [-0.000026, 0.000026] | ±0.000974 | Pass | 1.0045 (1.0082) | Pass | -0.0013 (-0.0056) | Pass | 0 (0.0116) | Pass | Pass |
| `att` | 0.000733 [0.000704, 0.000762] | ±0.0011 | Pass | 0.9963 (1.0041) | Pass | 0.0037 (-0.0037) | Pass | 0.0017 (0.0056) | Pass | Pass |
| `atc` | -0.000715 [-0.000749, -0.000681] | ±0.0011 | Pass | 1.0052 (1.0128) | Pass | -0.000625 (-0.0063) | Pass | 0 (0.0097) | Pass | Pass |
| `ey_obs` | -0.000000 [-0.000000, 0.000000] | ±0.0012 | Pass | 1.0000 (1.0000) | Pass | 0 (0) | Pass | 0 (0.000000) | Pass | Pass |
| `par` | 0.000029 [0.000016, 0.000042] | ±0.000683 | Pass | 1.0039 (1.0072) | Pass | -0.000625 (-0.0037) | Pass | 0.0090 (0.0114) | Pass | Pass |

### Scientific-property and control tests

A **control** row states that the estimator fails in the direction its property
predicts; its rule is the positive cells' rule reversed, and passing one is not a
claim that the deliberately misspecified or in-sample fit was valid. Where a
property needs more than one cell to establish, the shared clause is marked
*joint* in the rule and is reported once in the row beneath the table.

| test | cell | role | n | replications | observed result | uncertainty | decision rule | status |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| Double robustness | `both_correct` | positive | 700 | 1,200 | bias -0.0031 | bias CI [-0.0100, 0.0038] | 99% bias CI inside ±0.0231 | Pass |
| Double robustness | `outcome_correct` | positive | 700 | 1,200 | bias 0.000149 | bias CI [-0.0064, 0.0067] | 99% bias CI inside ±0.0219 | Pass |
| Double robustness | `treatment_correct` | positive | 700 | 1,200 | bias -0.0196 | bias CI [-0.0305, -0.0086] | 99% bias CI inside ±0.0366 | Pass |
| Double robustness | `both_wrong` | control | 700 | 1,200 | bias -0.3348 | bias CI [-0.3437, -0.3259] | 99% bias CI outside ±0.0299 | Pass (control broke as required) |
| Root-n and efficiency | `n_500` | positive | 500 | 800 | bias -0.000654; coverage 0.9487; SE ratio 1.0128 | bias CI [-0.0090, 0.0077]; coverage CI [0.9252, 0.9667] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_2000` | positive | 2,000 | 800 | bias -0.000506; coverage 0.9475; SE ratio 0.9823 | bias CI [-0.0048, 0.0037]; coverage CI [0.9238, 0.9657] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_8000` | positive | 8,000 | 800 | bias 0.000293; coverage 0.9550; SE ratio 1.0087 | bias CI [-0.0018, 0.0024]; coverage CI [0.9326, 0.9717] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n rate | `empirical_sd` | positive | 500 / 2,000 / 8,000 | 800 each | slope -0.5053 | slope CI [-0.5384, -0.4737] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Root-n rate | `reported_se` | positive | 500 / 2,000 / 8,000 | 800 each | slope -0.5067 | slope CI [-0.5080, -0.5055] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Interval calibration | `correctly_specified` | positive | 2,000 | 2,400 | coverage 0.9550; SE ratio 1.0077 | coverage CI [0.9430, 0.9652]; SE-ratio CI [0.9738, 1.0450] | coverage CI in [0.9200, 0.9800]; SE-ratio CI in [0.9300, 1.0700] | Pass |
| Type-I error | `sharp_null` | positive | 1,000 | 400 | rejection 0.0325; coverage 0.9675 | rejection upper 0.0627; coverage lower 0.9373 | rejection upper ≤ 0.1000; coverage lower ≥ 0.9000 | Pass |
| Power | `alternative` | positive | 1,000 | 400 | rejection 1 | rejection CI [0.9868, 1] | 99% rejection lower ≥ 0.8000 | Pass |
| Cross-fit overfitting | `stacked_cvtmle` | positive | 500 | 400 | coverage 0.8950; SE ratio 0.9880 | SE-ratio CI [0.9094, 1.0843]; coverage-gain CI [0.1875, 0.3050] | cross-fit SE-ratio CI in [0.8500, 1.2000] (joint: paired coverage-gain lower ≥ 0.1500) | Pass |
| Cross-fit overfitting | `in_sample_control` | control | 500 | 400 | coverage 0.6500; SE ratio 0.5792 | SE-ratio CI [0.5298, 0.6400]; coverage-gain CI [0.1875, 0.3050] | control SE-ratio CI upper ≤ 0.7500, i.e. the in-sample fit must understate its own spread (joint: paired coverage-gain lower ≥ 0.1500) | Pass (control broke as required) |

- **Double robustness** overall, every cell and joint clause together: Pass
- **Root-n and efficiency** overall, every cell and joint clause together: Pass
- **Root-n rate** overall, every cell and joint clause together: Pass
- **Cross-fit overfitting** overall, every cell and joint clause together: Pass
<!-- END GENERATED STUDY RESULTS -->

## Study boundary

The registered study covers the two complete-outcome point-treatment laws, corresponding GLM
nuisance regressions, one shared ten-fold split, a pooled update, whole-sample evaluation, the
stated bounds, and pointwise Wald inference. It does not cover repeated or nested cross-fitting,
fold-evaluated or fold-specific-epsilon CV-TMLE, simultaneous or bootstrap intervals, missing
outcomes, weights, clusters, strata, multi-valued treatment, broad learner-library selection, or
severe practical-positivity settings.
