# Repeated point-treatment cross-fitted TMLE

This study validates `cleverly`'s repeated stacked CV-TMLE report: five-fold nuisance fitting,
one pooled targeting update per draw, whole-sample plug-in evaluation, median aggregation over
three independent fold draws, and the split-adjusted median variance. Risk ratios and odds ratios
are aggregated on their log inference scale.

For draw-specific points $\hat\psi_r$ and variances $\hat\sigma_r^2$, the report is

$$
\widetilde\psi = \operatorname{median}_r(\hat\psi_r), \qquad
\widetilde\sigma^2 = \operatorname{median}_r\left\{
\hat\sigma_r^2 + (\hat\psi_r - \widetilde\psi)^2
\right\}.
$$

[Chernozhukov et al. (2018), Definition 3.3 and equation
(3.13)](https://academic.oup.com/ectj/article/21/1/C1/5056401) supply this fixed-repeat rule.
[zEpid at `16a0f96`](https://github.com/pzivich/zEpid/blob/16a0f96f8b2c65df8715085801f21757d1478e1e/zepid/causal/doublyrobust/crossfit.py#L1602-L1641)
independently implements the same point and variance calculation for repeated cross-fit TMLE.
**No canonical implementation is compared.** zEpid is an aggregation-level comparator, not a
canonical implementation of the complete estimator. It trains each nuisance on one partition and
targets separately inside validation partitions. `cleverly` trains on the complement and makes one
stacked pooled targeting update. The zero-row equivalence artifact records the absence of a
full-method comparator.

## What was tested

| setting | declaration |
| --- | --- |
| primary construction | five folds, three complete fold draws, pooled targeting and whole-sample evaluation |
| primary variance | median within-draw variance plus squared split displacement |
| primary estimands | arm means, ATE, ATT, ATC, observed mean, PAR, PAF, RR, and OR where defined by the law |
| primary laws | binary and bounded-continuous point-treatment laws with exact truth |
| Monte Carlo inference | 99% intervals around all declared endpoints |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` repeated stacked CV-TMLE | -0.0017 to 0.0038 | 0.9475 | 1.0131 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` repeated stacked CV-TMLE | -0.0015 to 0.0040 | 0.9537 | 1.0101 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` repeated stacked CV-TMLE | -0.0014 to 0.0043 | 0.9513 | 1.0125 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` repeated stacked CV-TMLE | -0.0027 to 0.0014 | 0.9513 | 0.9925 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` repeated stacked CV-TMLE | -0.0014 to 0.0026 | 0.9525 | 1.0256 | pass |
| binary-outcome law | `ey_obs` | observed outcome mean under the natural course | `cleverly` repeated stacked CV-TMLE | -0.0014 to 0.0016 | 0.9387 | 0.9705 | pass |
| binary-outcome law | `or` | marginal odds ratio, reported on the log scale | `cleverly` repeated stacked CV-TMLE | -0.0045 to 0.0182 | 0.9550 | 1.0100 | pass |
| binary-outcome law | `paf` | population attributable fraction | `cleverly` repeated stacked CV-TMLE | -0.0014 to 0.0048 | 0.9500 | 1.0089 | pass |
| binary-outcome law | `par` | population attributable risk | `cleverly` repeated stacked CV-TMLE | -0.000726 to 0.0022 | 0.9525 | 1.0106 | pass |
| binary-outcome law | `rr` | marginal risk ratio, reported on the log scale | `cleverly` repeated stacked CV-TMLE | -0.0024 to 0.0101 | 0.9513 | 1.0025 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` repeated stacked CV-TMLE | -0.000320 to 0.000947 | 0.9550 | 1.0178 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` repeated stacked CV-TMLE | -0.000325 to 0.000857 | 0.9600 | 1.0324 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` repeated stacked CV-TMLE | -0.000315 to 0.000952 | 0.9500 | 1.0062 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` repeated stacked CV-TMLE | -0.000865 to 0.000471 | 0.9363 | 0.9596 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` repeated stacked CV-TMLE | -0.000669 to 0.000816 | 0.9563 | 1.0151 | pass |
| bounded continuous-outcome law with effect modification | `ey_obs` | observed outcome mean under the natural course | `cleverly` repeated stacked CV-TMLE | -0.000874 to 0.000607 | 0.9337 | 0.9607 | pass |
| bounded continuous-outcome law with effect modification | `par` | population attributable risk | `cleverly` repeated stacked CV-TMLE | -0.000347 to 0.000474 | 0.9513 | 0.9911 | pass |
<!-- /generated -->

## Theory properties

The double-robustness cells use a bounded nonlinear confounded law with exact ATE 1.75.
Its treatment mechanism stays between 0.076 and 0.924, so the configured bounds do not clip it.
The wrong main-effects outcome regression imposes a constant contrast, while the true contrast
varies with `W1` and `I(W2 > 0)`. The treatment-correct cell uses n = 2,000. The other three cells
use n = 700. Each cell uses 1,200 replications and the existing predeclared margin.

The repeat-stability cells hold one binary-law sample of 1,000 rows fixed across 400 fold-seed
trials. Each control uses the first actual fold draw from its paired three-repeat fit. The 99%
paired-bootstrap interval compares the three-repeat ATE spread with the control spread.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0043 to 0.0088, margin 0.0218, SE ratio 1.0001 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.3029 to -0.2844, margin 0.0310, SE ratio 1.0007 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0058 to 0.0076, margin 0.0225, SE ratio 0.9424 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0087 to 0.0032, margin 0.0200, SE ratio 0.9994 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9430 to 0.9652, SE ratio 0.9760 to 1.0468 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `repeat_stability` | `one_repeat_control` | control | the ATE from each repeated fit's first actual fold draw | the paired spread-ratio upper endpoint must fall below the declared boundary | spread ratio 0.5841, 0.5161 to 0.6592, boundary 1 | pass |
| `repeat_stability` | `three_repeats` | positive | the median ATE over three fold draws across labelled base seeds | the paired spread-ratio upper endpoint must fall below the declared boundary | spread ratio 0.5841, 0.5161 to 0.6592, boundary 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000524, coverage 0.9208 to 0.9637, SE ratio 0.9822 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000832, coverage 0.9238 to 0.9657, SE ratio 1.0184 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000276, coverage 0.9297 to 0.9698, SE ratio 1.0093 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5375 to -0.4732 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5098 to -0.5072 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0300, 0.0125 to 0.0594 | pass |
<!-- /generated -->

## Result

The median repeated estimator and every declared repeated-sampling property passed. The
repeat-stability ratio was 0.5841, with a 99% interval from 0.5161 to 0.6592. Mean aggregation is
not a public option. This record tests the method that the library ships.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications per law |
| `n` | 1000 | observations per primary replication |
| `independent_tests_total` | 17 | estimand-law tests against truth |
| `independent_tests_passed` | 17 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 14 | repeated-sampling property cells |
| `property_cells_passed` | 14 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0560 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9337 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9078 | lowest exact 99% primary coverage endpoint |
| `min_se_ratio_ci_lower` | 0.8960 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.1016 | highest bootstrap primary SE-ratio endpoint |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size the one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the root-n interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:repeat_spread_ratio` | 1 | largest accepted upper endpoint for the paired repeat-stability spread ratio |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | Every declared cell is green. The `reporting` policy does not assert that, so the fast tier recomputes each verdict and does not fail on a red one |
| There is no full-method cross-implementation evidence | zEpid corroborates the aggregation formula, not the complement-trained stacked pooled estimator |
| The study does not claim median superiority | Median is the source-backed reporting standard, not an option chosen because this law makes it beat a mean alternative |
| The stability law and repeat budget are fixed | Spread reduction is measured on one fixed binary-law sample at one versus three fold draws. It does not establish behavior for other samples, laws, or arbitrary repeat counts |
| Inference is marginal | Coordinatewise medians do not supply joint covariance or post-fit contrasts; the central-draw curve does not support simultaneous bands for the split-adjusted estimator |
| The scientific scope is point treatment | The row does not validate clustering, observation weights, missing outcomes, longitudinal data, bootstrap inference, or severe positivity violations outside the declared law |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/repeated_crossfit/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/repeated_crossfit/manifest.json)
records the seeds, margins, exact estimator configuration, source hashes, and result hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/repeated_crossfit/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/repeated_crossfit/properties.csv)
carry every published row.
