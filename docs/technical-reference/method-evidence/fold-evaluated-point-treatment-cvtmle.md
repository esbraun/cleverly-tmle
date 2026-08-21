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
| cross-implementation tests | paired comparison with a separately maintained implementation | no comparator is registered and the committed comparison artifact has zero rows |

A performance row passes only if its bias, coverage, and SE-ratio rules all pass. A control marked
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

The tables below are generated from the committed artifacts. Square brackets contain two-sided
99% intervals.

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
| cleverly-fold-evaluated-cvtmle | `ey1` | identity | 0.000346 [-0.0011, 0.0018] | ±0.0055 | Pass | 0.9575 [0.9428, 0.9694] | Pass | 1.0155 [0.9707, 1.0656] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `ey0` | identity | -0.000472 [-0.0019, 0.000936] | ±0.0055 | Pass | 0.9525 [0.9371, 0.9651] | Pass | 1.0093 [0.9631, 1.0594] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `ate` | identity | 0.000818 [-0.0011, 0.0028] | ±0.0075 | Pass | 0.9506 [0.9350, 0.9635] | Pass | 0.9997 [0.9550, 1.0481] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `att` | identity | 0.000533 [-0.0015, 0.0026] | ±0.0078 | Pass | 0.9550 [0.9399, 0.9673] | Pass | 1.0048 [0.9579, 1.0549] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `atc` | identity | 0.0011 [-0.000879, 0.0031] | ±0.0077 | Pass | 0.9456 [0.9293, 0.9592] | Pass | 0.9987 [0.9536, 1.0471] | Pass | Pass |

#### Continuous outcome law

| implementation | estimand | scale | bias (99% CI) | bias margin | bias test | coverage (99% CI) | coverage test | SE ratio (99% CI) | SE test | overall |
| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| cleverly-fold-evaluated-cvtmle | `ey1` | identity | -0.000085 [-0.000603, 0.000433] | ±0.0020 | Pass | 0.9537 [0.9385, 0.9662] | Pass | 1.0243 [0.9801, 1.0737] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `ey0` | identity | -0.000250 [-0.000702, 0.000203] | ±0.0018 | Pass | 0.9519 [0.9364, 0.9646] | Pass | 0.9967 [0.9523, 1.0450] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `ate` | identity | 0.000165 [-0.000264, 0.000593] | ±0.0017 | Pass | 0.9500 [0.9343, 0.9630] | Pass | 1.0034 [0.9604, 1.0521] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `att` | identity | 0.000274 [-0.000186, 0.000735] | ±0.0018 | Pass | 0.9375 [0.9203, 0.9521] | Pass | 0.9756 [0.9303, 1.0258] | Pass | Pass |
| cleverly-fold-evaluated-cvtmle | `atc` | identity | 0.000099 [-0.000360, 0.000559] | ±0.0018 | Pass | 0.9513 [0.9357, 0.9641] | Pass | 0.9866 [0.9437, 1.0328] | Pass | Pass |

### Cross-implementation tests

No external comparison is registered for this study; its committed comparison artifact therefore contains zero tests.

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
| Root-n and efficiency | `n_500` | positive | 500 | 800 | bias -0.000654; coverage 0.9487; SE ratio 1.0139 | bias CI [-0.0090, 0.0077]; coverage CI [0.9252, 0.9667] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_2000` | positive | 2,000 | 800 | bias -0.000506; coverage 0.9475; SE ratio 0.9825 | bias CI [-0.0048, 0.0037]; coverage CI [0.9238, 0.9657] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n and efficiency | `n_8000` | positive | 8,000 | 800 | bias 0.000293; coverage 0.9550; SE ratio 1.0088 | bias CI [-0.0018, 0.0024]; coverage CI [0.9326, 0.9717] | bias equivalent; coverage lower ≥ 0.9000; SE ratio in [0.8000, 1.2000] | Pass |
| Root-n rate | `empirical_sd` | positive | 500 / 2,000 / 8,000 | 800 each | slope -0.5053 | slope CI [-0.5374, -0.4732] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Root-n rate | `reported_se` | positive | 500 / 2,000 / 8,000 | 800 each | slope -0.5071 | slope CI [-0.5084, -0.5058] | 99% slope CI inside [-0.6250, -0.3750] and excluding -0.2500 | Pass |
| Interval calibration | `correctly_specified` | positive | 2,000 | 2,400 | coverage 0.9550; SE ratio 1.0080 | coverage CI [0.9430, 0.9652]; SE-ratio CI [0.9730, 1.0456] | coverage CI in [0.9200, 0.9800]; SE-ratio CI in [0.9300, 1.0700] | Pass |
| Type-I error | `sharp_null` | positive | 1,000 | 400 | rejection 0.0325; coverage 0.9675 | rejection upper 0.0627; coverage lower 0.9373 | rejection upper ≤ 0.1000; coverage lower ≥ 0.9000 | Pass |
| Power | `alternative` | positive | 1,000 | 400 | rejection 1 | rejection CI [0.9868, 1] | 99% rejection lower ≥ 0.8000 | Pass |
| Cross-fit overfitting | `fold_evaluated_cvtmle` | positive | 500 | 400 | coverage 0.8950; SE ratio 0.9910 | SE-ratio CI [0.9091, 1.0908]; coverage-gain CI [0.1875, 0.3050] | cross-fit SE-ratio CI in [0.8500, 1.2000] (joint: paired coverage-gain lower ≥ 0.1500) | Pass |
| Cross-fit overfitting | `in_sample_control` | control | 500 | 400 | coverage 0.6500; SE ratio 0.5792 | SE-ratio CI [0.5308, 0.6408]; coverage-gain CI [0.1875, 0.3050] | control SE-ratio CI upper ≤ 0.7500, i.e. the in-sample fit must understate its own spread (joint: paired coverage-gain lower ≥ 0.1500) | Pass (control broke as required) |

- **Double robustness** overall, every cell and joint clause together: Pass
- **Root-n and efficiency** overall, every cell and joint clause together: Pass
- **Root-n rate** overall, every cell and joint clause together: Pass
- **Cross-fit overfitting** overall, every cell and joint clause together: Pass
<!-- END GENERATED STUDY RESULTS -->

## Study boundary

The registered study covers the five named estimands, two complete-outcome point-treatment laws,
corresponding GLM nuisance regressions, one ten-fold split, a pooled update, equal-fold evaluation,
the stated bounds, and pointwise Wald inference. It does not establish external parity or cover
repeated or nested cross-fitting, fold-specific targeting epsilons, simultaneous or bootstrap
intervals, missing outcomes, weights, clusters, strata, multi-valued treatment, ratio or observed-
risk estimands, or severe practical-positivity settings.
