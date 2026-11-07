# Assessment readings of an EP curve

This study validates the held-out readings of `curve.diagnostics` on an EP curve of the CATE:
the out-of-fold calibration slope of `calibration()`, the sign of the doubly robust risk
difference of `risk_comparison()` against the training-fold constant, and the split
sensitivity of `stability()`. The curve is `effect.estimate("ep", ...)` at the
[ep-cate](ep-cate.md) schedule, $k = 4$ cosines per modifier at n = 2,000. None of these
readings has an interval. [Conditional effects](../conditional-effects.md#assessing-a-curve)
defines each one.

**No canonical implementation is compared.** No maintained package computes these readings of
an EP curve. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | `strong`: ep-cate's complex d = 3 law at moderate overlap (CATE $1 + \sum_k W_k + \sum_k \sin 5W_k$, noise SD 1). `weak`: the same law with the constant CATE 0.1 |
| fit | `ep_cate.fit_ep` with the ensemble learners, 5 outer folds, `outcome_by_arm=True` and the schedule size 4 at n = 2,000; the readings use 5 out-of-fold refits |
| readings | the calibration slope of the difference-scale pseudo-outcome on the out-of-fold curve; the same slope of the curve stretched by 1.5 about its mean on the same folds; the reported EP-minus-constant DR risk difference, beside its exact value $\overline{(\hat\theta - \tau_0)^2} - \overline{(c - \tau_0)^2}$ at the true CATE; `relative_sd` over 5 split seeds |
| primary rows | the projection TMLE of conditional-projection's `obs_4k` law on this study's own seed, 2,400 replications at n = 4,000: the readings have no interval for the harness's primary rows to read |
| budgets | 300 replicates of `strong` for calibration and the risk comparison, of which the first 100 also read stability; 100 replicates of `weak` |
| seeds | `seed=20261125` and `resampling_seed=20261126`; every group draws from its own labelled stream |
| policy | `gated` |

The laws, readings, thresholds and seeds were fixed before the run in the P1 plan (section 5.4
and amendments rev 2.4 and 2.5) and `tests/studies/ep_assessment.py`. Every claim's pass
probability was computed from a pilot on its own stream (`tests/studies/ep_assessment_pilot.csv`),
and `tests/unit/test_ep_assessment_design.py` pins those design numbers.

## Accuracy against known truth

The primary rows are the projection TMLE of the shared observational law, as in
[ep-cate](ep-cate.md).

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[(intercept)]` | the intercept of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-assessment study: the EP curve has no interval for the primary rows to read | -0.000955 to 0.000679 | 0.9554 | 0.9996 | pass |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[W1]` | the W1 slope of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-assessment study: the EP curve has no interval for the primary rows to read | -0.0015 to 0.0013 | 0.9454 | 0.9978 | pass |
| conditional-projection's observational law with a CATE that is not linear in W3, spline GLM learners, n = 4,000, on this study's own seed | `cate_projection[W3]` | the W3 slope of the least-squares projection of the CATE on (1, W1, W3) | `cleverly` projection TMLE of the CATE on (1, W1, W3), the primary phase of the ep-assessment study: the EP curve has no interval for the primary rows to read | -0.000907 to 0.0020 | 0.9479 | 0.9807 | pass |
<!-- /generated -->

## Theory properties

| family | what it asks | how it fails |
| --- | --- | --- |
| `calibration` | at the strong law, is the two-sided 99% interval of the mean out-of-fold calibration slope inside 0.85 to 1.15 | the stretched curve's slope is exactly $1/1.5$ of the curve's on every replicate, so its 99% upper bound must fall below 0.85. It is a deterministic resolution check of the reading, not an independent failure |
| `risk_comparison` | at the strong law, does the share of replicates whose reported risk difference has the sign of its exact value have a 99% Clopper-Pearson lower bound of at least 0.80 | none; at the weak law the true difference is of the order of its estimation SD, so the share is reported only |
| `stability` | the mean `relative_sd` and the share of replicates above 0.5, at both laws | reported with no verdict |

Stability is reported, not gated (plan amendment rev 2.4). Every refit under another split seed
reuses the same draw, so the reading measures sensitivity to the cross-fitting split and not
sampling noise. A curve of noise at a constant effect then reads low, and a claim that the weak
law warns would fail at any number of replicates.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `calibration` | `strong_2k__slope` | positive | ep-cate's complex d = 3 law at moderate overlap, n = 2,000: the mean out-of-fold calibration slope of the EP curve at the schedule size 4 | two-sided 99% interval inside 0.85 to 1.15 | reading 0.9857, 99% interval 0.9849 to 0.9865 | pass |
| `calibration` | `strong_2k__stretched_slope` | control | ep-cate's complex d = 3 law at moderate overlap, n = 2,000: the same slope of the curve stretched by 1.5 about its mean, on the same folds | one-sided 99% upper bound below 0.85: the control must fail | reading 0.6571, 99% interval 0.6566 to 0.6577, threshold 0.8500 | pass |
| `calibration` | `weak_2k__slope` | diagnostic | the same law with the constant CATE 0.1, n = 2,000: the mean out-of-fold calibration slope of the EP curve at the schedule size 4 | reported with no verdict | reading -0.1572, 99% interval -0.2839 to -0.0306 | reported |
| `risk_comparison` | `strong_2k__exact_difference` | diagnostic | ep-cate's complex d = 3 law at moderate overlap, n = 2,000: the mean exact EP-minus-constant risk difference at the true CATE | reported with no verdict | reading -1.9672, 99% interval -1.9768 to -1.9576 | reported |
| `risk_comparison` | `strong_2k__risk_difference` | diagnostic | ep-cate's complex d = 3 law at moderate overlap, n = 2,000: the mean reported EP-minus-constant DR risk difference | reported with no verdict | reading -1.9671, 99% interval -1.9897 to -1.9445 | reported |
| `risk_comparison` | `strong_2k__sign_agrees` | positive | ep-cate's complex d = 3 law at moderate overlap, n = 2,000: the share of replicates whose reported EP-minus-constant DR risk difference has the sign of the exact difference at the true CATE | one-sided 99% Clopper-Pearson lower bound at least 0.80 | reading 1, 99% lower 0.9848, threshold 0.8000 | pass |
| `risk_comparison` | `weak_2k__exact_difference` | diagnostic | the same law with the constant CATE 0.1, n = 2,000: the mean exact EP-minus-constant risk difference at the true CATE | reported with no verdict | reading 0.0256, 99% interval 0.0231 to 0.0281 | reported |
| `risk_comparison` | `weak_2k__risk_difference` | diagnostic | the same law with the constant CATE 0.1, n = 2,000: the mean reported EP-minus-constant DR risk difference | reported with no verdict | reading 0.0224, 99% interval 0.0186 to 0.0263 | reported |
| `risk_comparison` | `weak_2k__sign_agrees` | diagnostic | the same law with the constant CATE 0.1, n = 2,000: the share of replicates whose reported EP-minus-constant DR risk difference has the sign of the exact difference at the true CATE | reported with no verdict | reading 0.9600, 99% lower 0.8883 | reported |
| `stability` | `strong_2k__relative_sd` | diagnostic | ep-cate's complex d = 3 law at moderate overlap, n = 2,000: the mean relative SD of the curve over five split seeds on the same draw | reported with no verdict | reading 0.0095, 99% interval 0.0091 to 0.0098 | reported |
| `stability` | `strong_2k__warns` | diagnostic | ep-cate's complex d = 3 law at moderate overlap, n = 2,000: the share of replicates whose relative SD exceeds 0.5 and warns | reported with no verdict | reading 0, 99% lower 0 | reported |
| `stability` | `weak_2k__relative_sd` | diagnostic | the same law with the constant CATE 0.1, n = 2,000: the mean relative SD of the curve over five split seeds on the same draw | reported with no verdict | reading 0.0773, 99% interval 0.0720 to 0.0827 | reported |
| `stability` | `weak_2k__warns` | diagnostic | the same law with the constant CATE 0.1, n = 2,000: the share of replicates whose relative SD exceeds 0.5 and warns | reported with no verdict | reading 0, 99% lower 0 | reported |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. The calibration
slope reads 0.986 with a 99% interval of 0.985 to 0.987, and the stretched control reads 0.657.
The reported risk difference has the sign of its exact value in all 300 strong-law replicates,
and its mean, $-1.967$, matches the exact mean.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 3 of 3 | `performance-tests.csv` |
| gated property cells passed | 3 of 3 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |

At the weak law the reported risk difference is positive, 0.022 against an exact 0.026: the
curve's wiggle costs risk against the constant, and the reading says so in 96 of 100
replicates. Its calibration slope reads $-0.16$, which carries no information at a constant
effect. `relative_sd` reads 0.009 at the strong law and 0.077 at the weak law, and no replicate
of either warns. The low weak-law value is the reason the warning cannot certify a curve: read
`heterogeneity_test()` and the projection's intervals for that.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 4000 | observations per replication |
| `independent_tests_total` | 3 | parameter tests against the truth |
| `independent_tests_passed` | 3 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 3 | gated repeated-sampling property cells |
| `property_cells_passed` | 3 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0197 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9454 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9324 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9450 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0369 | highest bootstrap primary SE-ratio endpoint |
| `properties[calibration/strong_2k__slope]:reading` | 0.9857 | mean calibration slope |
| `properties[calibration/strong_2k__slope]:reading_lower` | 0.9849 | its 99% lower end |
| `properties[calibration/strong_2k__slope]:reading_upper` | 0.9865 | its 99% upper end |
| `properties[calibration/strong_2k__stretched_slope]:reading_upper` | 0.6577 | stretched control's 99% upper end |
| `properties[risk_comparison/strong_2k__sign_agrees]:reading_lower` | 0.9848 | sign share's 99% lower end |
| `properties[risk_comparison/weak_2k__sign_agrees]:reading` | 0.9600 | weak-law sign share, reported |
| `properties[stability/strong_2k__relative_sd]:reading` | 0.0095 | strong-law `relative_sd` |
| `properties[stability/weak_2k__relative_sd]:reading` | 0.0773 | weak-law `relative_sd` |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:subject_failures_allowed` | 24 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| No interval | The readings are point readings. The intervals on this axis come from the projection TMLE and `assess_score` |
| One law, one size, one learner | Each claim is measured on one law at n = 2,000 with the ensemble and the schedule size. Another law, size or learner rests on the readings' construction alone |
| Stability is not a claim | `relative_sd` measures sensitivity to the cross-fitting split on the same data; a low value does not mean the curve is well determined |
| The stretched control is mechanical | Its slope is $1/1.5$ of the curve's by construction, so it fails whenever the calibration claim passes. It checks that the reading resolves a miscalibrated curve, not an independent failure |
| The sign claim is coarse | The strong law's exact risk difference is about $-2$, far from 0, so the sign claim shows that the reading points the right way when the gap is large. At the weak law, with a gap near the estimation SD, it is reported only |

## Reproduction

The declared run is `python -m tests.canonical.ep_assessment.regenerate`, whose docstring gives
the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_assessment/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_assessment/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_assessment/replicates.parquet)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ep_assessment/properties.csv)
carry every published row.
