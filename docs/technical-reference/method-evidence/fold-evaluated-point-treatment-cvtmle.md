# Fold-evaluated point-treatment CV-TMLE study

[← Method evidence grid](../../methodology.md#method-evidence-grid)

This study applies cleverly's fold-evaluated CV-TMLE report to repeated samples from laws with
known parameter values. It evaluates equal-fold plug-in means and the cross-validated influence-
curve variance after a pooled targeting update.

## Study setup

| setting | registered value |
| --- | --- |
| cleverly method | out-of-fold nuisance predictions, pooled targeting update, equal-fold plug-in evaluation, cross-validated variance |
| external reference | none registered |
| laws | binary outcome; bounded continuous outcome |
| estimands | `ey1`, `ey0`, `ate`, `att`, and `atc` |
| primary run | 1,600 replications per law; 1,000 observations per replication |
| folds | one treatment-stratified ten-fold assignment per sample |
| inference | pointwise 95% identity-scale Wald intervals |
| propensity bounds | 0.025–0.975 |
| committed artifacts | [performance tests](../../../tests/canonical/cvtmle_fold/performance-tests.csv), [registered zero-comparison artifact](../../../tests/canonical/cvtmle_fold/equivalence.csv), [property tests](../../../tests/canonical/cvtmle_fold/properties.csv), [manifest](../../../tests/canonical/cvtmle_fold/manifest.json) |
| reproduction | [`tests/canonical/cvtmle_fold/README.md`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/README.md) |

## How to read the tests

The reported 99% intervals quantify Monte Carlo uncertainty across simulation replications; they
are not the estimator's nominal 95% confidence intervals. Bias is computed on the inference scale
shown in the results table. **Pass** records only whether the predeclared decision rule was met.

| test | statistic | Monte Carlo uncertainty | decision rule |
| --- | --- | --- | --- |
| bias equivalence | mean error `mean(estimate − truth)`; the margin is 0.25 times the empirical SD of the estimates | two-sided 99% Student-t interval for mean error | the complete interval lies inside the displayed ± margin |
| coverage validity | mean of the indicators that the nominal 95% interval contains truth | two-sided exact 99% Clopper–Pearson interval | the lower endpoint is at least 0.90 |
| SE ratio | mean reported SE divided by the empirical SD of the estimates; 1 denotes equal scales | 99% percentile interval from 10,000 rowwise bootstrap resamples | the complete interval lies inside 0.80–1.20 |
| cross-implementation tests | paired comparison with a separately maintained implementation | no comparator is registered and the committed comparison artifact has zero rows | no comparison verdict is reported |

A performance row passes only if its bias, coverage, and SE-ratio rules all pass. A control marked
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

The tables below are generated from the committed artifacts. Square brackets contain two-sided
99% intervals.

<!-- BEGIN GENERATED STUDY RESULTS -->
### Performance versus truth

#### Binary outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly-fold-evaluated-cvtmle | `ey1` | identity | 0.000346 [-0.0011, 0.0018] | ±0.0055 | Pass | 0.9575 [0.9428, 0.9694] | Pass | 1.0155 [0.9701, 1.0642] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `ey0` | identity | -0.000472 [-0.0019, 0.000936] | ±0.0055 | Pass | 0.9525 [0.9371, 0.9651] | Pass | 1.0093 [0.9627, 1.0608] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `ate` | identity | 0.000818 [-0.0011, 0.0028] | ±0.0075 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 0.9997 [0.9536, 1.0502] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `att` | identity | 0.000533 [-0.0015, 0.0026] | ±0.0078 | Pass | 0.9550 [0.9399, 0.9673] | Pass | 1.0048 [0.9576, 1.0568] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `atc` | identity | 0.0011 [-0.000879, 0.0031] | ±0.0077 | Pass | 0.9456 [0.9293, 0.9592] | Pass | 0.9987 [0.9543, 1.0465] | Pass | Pass |

#### Continuous outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly-fold-evaluated-cvtmle | `ey1` | identity | -0.000085 [-0.000603, 0.000433] | ±0.0020 | Pass | 0.9537 [0.9385, 0.9662] | Pass | 1.0243 [0.9797, 1.0751] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `ey0` | identity | -0.000250 [-0.000702, 0.000203] | ±0.0018 | Pass | 0.9519 [0.9364, 0.9646] | Pass | 0.9967 [0.9533, 1.0441] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `ate` | identity | 0.000165 [-0.000264, 0.000593] | ±0.0017 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0034 [0.9579, 1.0520] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `att` | identity | 0.000274 [-0.000186, 0.000735] | ±0.0018 | Pass | 0.9375 [0.9203, 0.9521] | Pass | 0.9756 [0.9312, 1.0264] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `atc` | identity | 0.000099 [-0.000360, 0.000559] | ±0.0018 | Pass | 0.9513 [0.9357, 0.9641] | Pass | 0.9866 [0.9445, 1.0320] | Pass | Pass |

### Cross-implementation tests

No external comparison is registered for this study; its committed comparison artifact therefore contains zero tests.

### Scientific-property and control tests

| test | cell | n | replications | observed result | uncertainty | decision rule | status |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Double robustness | `both_correct` | 700 | 1,200 | bias -0.0031 | bias CI [-0.0100, 0.0038] | 99% bias CI inside ±0.0231 | Pass |
| Double robustness | `outcome_correct` | 700 | 1,200 | bias 0.000149 | bias CI [-0.0064, 0.0067] | 99% bias CI inside ±0.0219 | Pass |
| Double robustness | `treatment_correct` | 700 | 1,200 | bias -0.0196 | bias CI [-0.0305, -0.0086] | 99% bias CI inside ±0.0366 | Pass |
| Double robustness | `both_wrong` | 700 | 1,200 | bias -0.3348 | bias CI [-0.3437, -0.3259] | 99% bias CI outside ±0.0299 | Pass |
| Root-n and efficiency | `n_500` | 500 | 800 | bias -0.000654; coverage 0.9487; SE ratio 1.0139 | bias CI [-0.0090, 0.0077]; coverage CI [0.9252, 0.9667] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_2000` | 2,000 | 800 | bias -0.000506; coverage 0.9475; SE ratio 0.9825 | bias CI [-0.0048, 0.0037]; coverage CI [0.9238, 0.9657] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_8000` | 8,000 | 800 | bias 0.000293; coverage 0.9550; SE ratio 1.0088 | bias CI [-0.0018, 0.0024]; coverage CI [0.9326, 0.9717] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n rate | `empirical_sd` | 8,000 | 2,400 | slope -0.5053 | slope CI [-0.5375, -0.4725] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Root-n rate | `reported_se` | 8,000 | 2,400 | slope -0.5071 | slope CI [-0.5084, -0.5058] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Interval calibration | `correctly_specified` | 2,000 | 2,400 | coverage 0.9550; SE ratio 1.0080 | coverage CI [0.9430, 0.9652]; SE-ratio CI [0.9739, 1.0446] | coverage CI in [0.9200, 0.9800]; SE-ratio CI in [0.9300, 1.0700] | Pass |
| Type-I error | `sharp_null` | 1,000 | 400 | rejection 0.0325; coverage 0.9675 | rejection upper 0.0627; coverage lower 0.9373 | rejection upper ≤ 0.1000; coverage lower ≥ 0.9000 | Pass |
| Power | `alternative` | 1,000 | 400 | rejection 1 | rejection CI [0.9868, 1] | 99% rejection lower ≥ 0.8000 | Pass |
| Cross-fit overfitting | `fold_evaluated_cvtmle` | 500 | 400 | coverage 0.8950; SE ratio 0.9910 | SE-ratio CI [0.9094, 1.0874]; coverage-gain CI [0.1875, 0.3050] | cross-fit SE-ratio CI in [0.8500, 1.2000]; control upper ≤ 0.7500; coverage-gain lower ≥ 0.1500 | Pass |
| Cross-fit overfitting | `in_sample_control` | 500 | 400 | coverage 0.6500; SE ratio 0.5792 | SE-ratio CI [0.5294, 0.6409]; coverage-gain CI [0.1875, 0.3050] | cross-fit SE-ratio CI in [0.8500, 1.2000]; control upper ≤ 0.7500; coverage-gain lower ≥ 0.1500 | Pass |
<!-- END GENERATED STUDY RESULTS -->

## Study boundary

The registered study covers the five named estimands, two complete-outcome point-treatment laws,
corresponding GLM nuisance regressions, one ten-fold split, a pooled update, equal-fold evaluation,
the stated bounds, and pointwise Wald inference. It does not establish external parity or cover
repeated or nested cross-fitting, fold-specific targeting epsilons, simultaneous or bootstrap
intervals, missing outcomes, weights, clusters, strata, multi-valued treatment, ratio or observed-
risk estimands, or severe practical-positivity settings.
