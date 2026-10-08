# TMLE on a declared known treatment mechanism

This study validates point-treatment TMLE on data that declares its treatment mechanism with
`treatment_probabilities=`. The fit divides by the design's $g_0$ and fits no treatment learner.
The section
[known treatment mechanism](../point-treatment-tmle.md#known-treatment-mechanism) states the
estimator, its curve and its exact-law evidence.

The study pairs the fit with R `tmle` 2.1.1, which takes the same mechanism as `g1W`. Both sides
read this package's initial outcome regression. The pairs therefore test the targeting step at a
declared mechanism, not two outcome fits.

## What was compared

| setting | `cleverly` | R `tmle` |
| --- | --- | --- |
| datasets | 1,000 samples of 2,000 rows. The two scenarios read the same samples | the identical rows |
| law | $W_1 \sim N(0, 1)$, the randomization stratum $W_2 \sim \operatorname{Bernoulli}(0.5)$ and $W_3 \sim U(-1, 1)$. The mechanism is $g_0 = 0.25$ at $W_2 = 0$ and $0.70$ at $W_2 = 1$. The outcome is binary, with $W_1^2$ and two arm interactions | the same |
| scenarios | `binary_q_correct`: a logistic outcome regression on the true terms. `binary_q_wrong`: a main-terms logistic regression on $(A, W_1, W_3)$, which omits $W_2$, $W_1^2$ and both interactions | the same regressions, passed as `Q` |
| reported parameters | at the correct regression, the arm means, the ATE, RR, OR, ATT and ATC. At the wrong regression, the first five | the same |
| mechanism | the declared $g_0$, read from two probability columns | `g1W = g0`, with `gbound = 0.1` |
| targeting | the logistic fluctuation of the outcome regression | the same |
| intervals | pointwise 95% Wald from the influence curve, with RR and OR on the log scale | the same |

No bound binds on either side, because every $g_0$ lies in $[0.25, 0.70]$. Each truth is a
quadrature over the law. R `tmle` does not hold a known mechanism fixed for the ATT and the ATC.
It recalibrates `g1W` and fluctuates it beside the outcome regression. That construction is
equivalent to the known-mechanism ATT only at a correct outcome regression, so the study pairs
the ATT and the ATC at `binary_q_correct` only.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| randomized by W2 with a known mechanism, correct outcome regression | `atc` | average effect on the untreated | `cleverly` TMLE on a declared known mechanism | -0.0013 to 0.0028 | 0.9420 | 1.0045 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `atc` | average effect on the untreated | R `tmle` with the known mechanism as `g1W` | -0.0014 to 0.0027 | 0.9430 | 1.0036 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `ate` | average treatment effect | `cleverly` TMLE on a declared known mechanism | -0.0010 to 0.0027 | 0.9420 | 0.9899 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `ate` | average treatment effect | R `tmle` with the known mechanism as `g1W` | -0.0010 to 0.0027 | 0.9420 | 0.9899 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `att` | average effect on the treated | `cleverly` TMLE on a declared known mechanism | -0.0011 to 0.0029 | 0.9440 | 0.9910 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `att` | average effect on the treated | R `tmle` with the known mechanism as `g1W` | -0.0010 to 0.0029 | 0.9420 | 0.9902 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `ey0` | counterfactual mean under no treatment | `cleverly` TMLE on a declared known mechanism | -0.000673 to 0.0019 | 0.9540 | 1.0000 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `ey0` | counterfactual mean under no treatment | R `tmle` with the known mechanism as `g1W` | -0.000673 to 0.0019 | 0.9540 | 1.0000 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `ey1` | counterfactual mean under treatment | `cleverly` TMLE on a declared known mechanism | 0.000103 to 0.0028 | 0.9510 | 1.0208 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `ey1` | counterfactual mean under treatment | R `tmle` with the known mechanism as `g1W` | 0.000103 to 0.0028 | 0.9510 | 1.0208 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | `cleverly` TMLE on a declared known mechanism | -0.0034 to 0.0123 | 0.9420 | 0.9899 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | R `tmle` with the known mechanism as `g1W` | -0.0034 to 0.0123 | 0.9420 | 0.9899 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | `cleverly` TMLE on a declared known mechanism | -0.0029 to 0.0056 | 0.9410 | 0.9869 | pass |
| randomized by W2 with a known mechanism, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | R `tmle` with the known mechanism as `g1W` | -0.0029 to 0.0056 | 0.9410 | 0.9869 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ate` | average treatment effect | `cleverly` TMLE on a declared known mechanism | -0.000838 to 0.0031 | 0.9390 | 0.9860 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ate` | average treatment effect | R `tmle` with the known mechanism as `g1W` | -0.000838 to 0.0031 | 0.9390 | 0.9860 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ey0` | counterfactual mean under no treatment | `cleverly` TMLE on a declared known mechanism | -0.000805 to 0.0019 | 0.9430 | 0.9909 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ey0` | counterfactual mean under no treatment | R `tmle` with the known mechanism as `g1W` | -0.000805 to 0.0019 | 0.9430 | 0.9909 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ey1` | counterfactual mean under treatment | `cleverly` TMLE on a declared known mechanism | 0.000239 to 0.0031 | 0.9570 | 1.0258 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ey1` | counterfactual mean under treatment | R `tmle` with the known mechanism as `g1W` | 0.000239 to 0.0031 | 0.9570 | 1.0258 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `or` | marginal odds ratio, reported on the log scale | `cleverly` TMLE on a declared known mechanism | -0.0026 to 0.0139 | 0.9420 | 0.9860 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `or` | marginal odds ratio, reported on the log scale | R `tmle` with the known mechanism as `g1W` | -0.0026 to 0.0139 | 0.9420 | 0.9860 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `rr` | marginal risk ratio, reported on the log scale | `cleverly` TMLE on a declared known mechanism | -0.0026 to 0.0064 | 0.9330 | 0.9806 | pass |
| randomized by W2 with a known mechanism, outcome regression without W2 | `rr` | marginal risk ratio, reported on the log scale | R `tmle` with the known mechanism as `g1W` | -0.0026 to 0.0064 | 0.9330 | 0.9806 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| randomized by W2 with a known mechanism, correct outcome regression | `atc` | average effect on the untreated | 0.000049 | 0.0130 | 1.0002 | -0.0010 | 0.0025 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, correct outcome regression | `ate` | average treatment effect | 6.679e-11 | 1.963e-08 | 1.0000 | 0 | 3.417e-09 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, correct outcome regression | `att` | average effect on the treated | -0.000053 | 0.0143 | 1.0004 | 0.0020 | 0.0037 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, correct outcome regression | `ey0` | counterfactual mean under no treatment | -4.906e-11 | 2.062e-08 | 1.0000 | 0 | 8.523e-10 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, correct outcome regression | `ey1` | counterfactual mean under treatment | 1.773e-11 | 7.083e-09 | 1.0000 | 0 | 1.911e-09 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, correct outcome regression | `or` | marginal odds ratio, reported on the log scale | 7.194e-10 | 1.893e-08 | 1.0000 | 0 | 3.535e-09 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, correct outcome regression | `rr` | marginal risk ratio, reported on the log scale | 2.659e-10 | 2.025e-08 | 1.0000 | 0 | 4.669e-09 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ate` | average treatment effect | 8.830e-10 | 2.463e-07 | 1.0000 | 0 | 1.168e-08 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ey0` | counterfactual mean under no treatment | -8.081e-10 | 3.285e-07 | 1.0000 | 0 | 9.502e-09 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, outcome regression without W2 | `ey1` | counterfactual mean under treatment | 7.485e-11 | 2.896e-08 | 1.0000 | 0 | 7.739e-09 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, outcome regression without W2 | `or` | marginal odds ratio, reported on the log scale | 1.006e-08 | 2.514e-07 | 1.0000 | 0 | 1.209e-08 vs 0.0500 | equivalent |
| randomized by W2 with a known mechanism, outcome regression without W2 | `rr` | marginal risk ratio, reported on the log scale | 4.010e-09 | 2.907e-07 | 1.0000 | 0 | 1.348e-08 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `bootstrap_coverage` | `ate__known_g__q_wrong__percentile` | positive | average treatment effect: declared mechanism, outcome regression without W2, 200-replicate percentile interval | coverage interval clears the floor | coverage 0.9242 to 0.9747 | pass |
| `interval_calibration` | `ate__known_g_q_wrong` | positive | average treatment effect: declared mechanism, outcome regression without W2 | SE-ratio and coverage intervals inside the calibration bands | coverage 0.9411 to 0.9657, SE ratio 0.9857 to 1.0664 | pass |
| `interval_calibration` | `ate__noise_control` | control | average treatment effect: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8094 to 0.8530, SE ratio 0.6720 to 0.7300 | pass |
| `interval_calibration` | `ate__shrunken_se_control` | control | average treatment effect: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8252 to 0.8671, SE ratio 0.6906 to 0.7468 | pass |
| `known_mechanism_accuracy` | `atc__known_g__q_wrong` | positive | average effect on the untreated: declared mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0012 to 0.0030, margin 0.0065, coverage 0.9283 to 0.9652, SE ratio 1.0003 | pass |
| `known_mechanism_accuracy` | `ate_1_vs_0__known_g__multi_arm__q_wrong` | positive | difference of arm 1 and arm 0, three arms: declared three-arm mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0017 to 0.0028, margin 0.0068, coverage 0.9295 to 0.9661, SE ratio 0.9831 | pass |
| `known_mechanism_accuracy` | `ate_2_vs_0__known_g__multi_arm__q_wrong` | positive | difference of arm 2 and arm 0, three arms: declared three-arm mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0015 to 0.0031, margin 0.0070, coverage 0.9318 to 0.9677, SE ratio 1.0034 | pass |
| `known_mechanism_accuracy` | `ate__estimated_g_wrong__q_wrong` | control | average treatment effect: intercept-only estimated mechanism, outcome regression without W2 | bias interval lies outside the margin | bias 0.0773 to 0.0807, margin 0.0051, coverage 0.0171 to 0.0456, SE ratio 0.9892 | pass |
| `known_mechanism_accuracy` | `ate__known_g__q_correct` | positive | average treatment effect: declared mechanism, correct outcome regression | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0019 to 0.0018, margin 0.0056, coverage 0.9237 to 0.9619, SE ratio 0.9979 | pass |
| `known_mechanism_accuracy` | `ate__known_g__q_wrong` | positive | average treatment effect: declared mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0016 to 0.0022, margin 0.0059, coverage 0.9214 to 0.9602, SE ratio 0.9948 | pass |
| `known_mechanism_accuracy` | `ate__known_g__q_wrong__cv` | positive | average treatment effect: declared mechanism, outcome regression without W2, stacked CV-TMLE over five folds | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0018 to 0.0020, margin 0.0059, coverage 0.9214 to 0.9602, SE ratio 0.9958 | pass |
| `known_mechanism_accuracy` | `att__known_g__q_wrong` | positive | average effect on the treated: declared mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0024 to 0.0017, margin 0.0063, coverage 0.9307 to 0.9669, SE ratio 1.0055 | pass |
| `known_mechanism_accuracy` | `ey0__known_g__multi_arm__q_wrong` | positive | arm-0 mean, three arms: declared three-arm mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0026 to 0.000639, margin 0.0049, coverage 0.9283 to 0.9652, SE ratio 1.0255 | pass |
| `known_mechanism_accuracy` | `ey1__known_g__multi_arm__q_wrong` | positive | arm-1 mean, three arms: declared three-arm mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0020 to 0.0011, margin 0.0047, coverage 0.9191 to 0.9586, SE ratio 0.9602 | pass |
| `known_mechanism_accuracy` | `ey2__known_g__multi_arm__q_wrong` | positive | arm-2 mean, three arms: declared three-arm mechanism, outcome regression without W2 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0018 to 0.0015, margin 0.0051, coverage 0.9260 to 0.9636, SE ratio 0.9733 | pass |
| `known_mechanism_accuracy` | `ey_ipsi__known_g__incremental` | positive | incremental mean at odds multiplier 2: declared mechanism, outcome regression without W2, the known stochastic regime of the tilt | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000557 to 0.0013, margin 0.0028, coverage 0.9389 to 0.9726, SE ratio 1.0212 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0014, coverage 0.9353 to 0.9702, SE ratio 1.0119 | pass |
| `root_n_and_efficiency` | `n_500` | control | bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias 0.0019, coverage 0.9123 to 0.9535, SE ratio 0.9596 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000149, coverage 0.9330 to 0.9685, SE ratio 0.9976 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5399 to -0.4837 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4991 to -0.4970 | pass |
| `variance_direction` | `ate__known_mechanism` | diagnostic | average treatment effect: declared mechanism, outcome regression without W2 | reported, no verdict | estimated-over-declared spread ratio 0.9888, 0.9803 to 0.9975 | reported |
| `variance_direction` | `ate__parametric_mechanism` | diagnostic | average treatment effect: mechanism estimated by a logistic regression on W2, outcome regression without W2 | reported, no verdict | estimated-over-declared spread ratio 0.9888, 0.9803 to 0.9975 | reported |
<!-- /generated -->

The accuracy cells use the wrong outcome regression, except `ate__known_g__q_correct`. At the
declared mechanism, the remainder is zero for any outcome regression, so each positive cell must
be unbiased with a calibrated interval. The control `ate__estimated_g_wrong__q_wrong` replaces
the declaration with an intercept-only mechanism. Its population bias is 0.0790, and the cell
must establish a bias outside the margin.

The `variance_direction` rows have no verdict. They compare the declared mechanism with a
logistic regression on $W_2$ on the same draws. Moore and van der Laan (2009), Section 7.3, and
Petersen et al. (2014), Section 3.7, predict a smaller spread for the estimated mechanism. The
row reports the ratio of the two spreads with a 99% bootstrap interval.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications per scenario |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 24 | implementation-estimand tests against the truth |
| `independent_tests_passed` | 24 | of those, passing |
| `paired_tests_total` | 12 | paired comparisons with R `tmle` |
| `paired_tests_passed` | 12 | of those, passing |
| `property_cells_total` | 21 | repeated-sampling property cells |
| `property_cells_passed` | 21 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0955 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9330 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9251 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0889 | highest bootstrap primary SE-ratio endpoint |
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
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed. No cell of this study reads it |

## Limitations

| limit | what it means for use |
| --- | --- |
| parametric outcome regressions only | a flexible outcome learner has fast-tier evidence only |
| the ATT and the ATC are paired at the correct outcome regression only | at a wrong outcome regression, R `tmle` runs a different construction. The property cells measure the ATT and the ATC against the truth there |
| one binary and one three-arm law, each with a mechanism that depends on one stratum | a mechanism that depends on a continuous covariate has exact-law evidence only |
| the variance direction is reported, not gated | the row states the measured ratio. It does not test a margin |

## Reproduction

The [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_mechanism/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. Run `python -m tests.canonical.known_mechanism.regenerate` to regenerate the artifacts.
The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_mechanism/replicates.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/known_mechanism/properties.csv)
carry every published row.
