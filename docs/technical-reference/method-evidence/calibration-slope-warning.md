# Calibration-slope warning

This study validates the calibration-slope rule of the nuisance diagnostics. The rule reads the
calibration slope of each probability report, which is a logistic recalibration of the label on
$\operatorname{logit} \hat p$ with one intercept for each validation fold, and its sandwich
standard error. It warns when the Bonferroni interval over the tested models lies above 0 and
excludes 1. The [calibration-slope rule](../validation-methods.md#calibration-slope-rule) section
of the validation reference defines the statistic and the rule.

The study asks three questions. Is the slope's standard error calibrated when the propensity is
known? How often does the rule warn on a law where a warning is false? Does it detect a learner
whose predictions are too extreme or too moderate?

**No canonical implementation is compared.** No maintained implementation computes this rule. The
zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| fit | a three-fold `TMLE` at $n = 2000$, with `simultaneous=False` and the ATE only |
| outcome | $Y \sim \operatorname{Bernoulli}(\operatorname{expit}(0.5 A + 0.5 W_1 - 0.25))$. The outcome learner returns this known regression, so the outcome report is calibrated and each fit tests two models |
| covariates | four standard normal covariates, except the `correct_weak` law, which has one |
| primary scenarios | the known propensity at a weak signal, $\operatorname{logit} g_0 = 0.15 W_1$, and at a strong signal, $\operatorname{logit} g_0 = W_1 - 0.5 W_2$ |
| primary estimand | the calibration slope of the propensity report. Its truth is 1, because the treatment learner returns the true propensity |
| primary interval | the slope plus or minus 1.959964 standard errors |
| `warning_rate` laws | the two known propensities; a correct unpenalized logistic model on one weak covariate; the same law with four covariates and the model on all four; and a randomized law, $g_0 = 1/2$, with the model on four covariates |
| `fixed_band` controls | the same fits, read with the rule before RM15: the pooled one-intercept slope of the propensity or the outcome lies outside [0.7, 1.4] |
| `power` laws | an unpenalized logistic model on $W_1, W_2$ with its logit doubled, at $\operatorname{logit} g_0 = 0.4 W_1 - 0.2 W_2$, and the same model with its logit halved, at $\operatorname{logit} g_0 = W_1 - 0.5 W_2$. Their limit slopes are 1/2 and 2 |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

The RM15 plan in the [roadmap](../../roadmap.md#rm15-calibration-slope-warning-rule) fixed the
laws, the budget, the seeds, the margins and the red-cell route before any run.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| known propensity, logit g0 = W1 - 0.5 W2 | `propensity_calibration_slope` | calibration slope of the propensity report, one intercept per fold; truth 1 | `cleverly` calibration-slope rule | 0.0017 to 0.0046 | 0.9484 | 0.9976 | pass |
| known propensity, logit g0 = 0.15 W1 | `propensity_calibration_slope` | calibration slope of the propensity report, one intercept per fold; truth 1 | `cleverly` calibration-slope rule | -0.0017 to 0.0138 | 0.9503 | 1.0018 | pass |
<!-- /generated -->

## Theory properties

A `warning_rate` positive cell must bound the rule's rate at or below the type-I ceiling. A
finding on either tested model counts. A `fixed_band` control reads the same fits with the band
the package applied before RM15, and its rate must lie above the ceiling. A `power` cell must
detect the tempered learner in at least 80% of fits. Each law draws its own samples, and a control
reads the fits of its positive cell.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `power` | `overconfident_moderate__rule` | positive | a logistic model with its logit doubled, at a moderate signal: the calibration-slope rule, on a learner whose limit slope is not 1 | rejection lower bound clears the minimum power | rejection 1, 0.9995 to 1 | pass |
| `power` | `underconfident_strong__rule` | positive | a logistic model with its logit halved, at a strong signal: the calibration-slope rule, on a learner whose limit slope is not 1 | rejection lower bound clears the minimum power | rejection 1, 0.9995 to 1 | pass |
| `warning_rate` | `calibrated_strong__rule` | positive | the known propensity at a strong signal: the calibration-slope rule, which warns when the Bonferroni interval over the tested models lies above 0 and excludes 1 | rejection upper bound at or below the type-I ceiling | rejection 0.0539, 0.0482 to 0.0600 | pass |
| `warning_rate` | `calibrated_weak__fixed_band` | control | the known propensity at a weak signal: the band before RM15 on the same fits, which warns when the pooled one-intercept slope of the propensity or the outcome lies outside 0.7 to 1.4 | rejection lower bound above the type-I ceiling | rejection 0.2543, 0.2432 to 0.2657 | pass |
| `warning_rate` | `calibrated_weak__rule` | positive | the known propensity at a weak signal: the calibration-slope rule, which warns when the Bonferroni interval over the tested models lies above 0 and excludes 1 | rejection upper bound at or below the type-I ceiling | rejection 0.0364, 0.0317 to 0.0415 | pass |
| `warning_rate` | `correct_nested_weak__fixed_band` | control | a correct logistic model on four covariates, with a weak signal in one: the band before RM15 on the same fits, which warns when the pooled one-intercept slope of the propensity or the outcome lies outside 0.7 to 1.4 | rejection lower bound above the type-I ceiling | rejection 0.7353, 0.7238 to 0.7466 | pass |
| `warning_rate` | `correct_nested_weak__rule` | positive | a correct logistic model on four covariates, with a weak signal in one: the calibration-slope rule, which warns when the Bonferroni interval over the tested models lies above 0 and excludes 1 | rejection upper bound at or below the type-I ceiling | rejection 0.0403, 0.0354 to 0.0456 | pass |
| `warning_rate` | `correct_weak__fixed_band` | control | a correct logistic model on one weak covariate: the band before RM15 on the same fits, which warns when the pooled one-intercept slope of the propensity or the outcome lies outside 0.7 to 1.4 | rejection lower bound above the type-I ceiling | rejection 0.3603, 0.3480 to 0.3728 | pass |
| `warning_rate` | `correct_weak__rule` | positive | a correct logistic model on one weak covariate: the calibration-slope rule, which warns when the Bonferroni interval over the tested models lies above 0 and excludes 1 | rejection upper bound at or below the type-I ceiling | rejection 0.0225, 0.0189 to 0.0266 | pass |
| `warning_rate` | `randomized__fixed_band` | control | a logistic model on four covariates, for a randomized treatment: the band before RM15 on the same fits, which warns when the pooled one-intercept slope of the propensity or the outcome lies outside 0.7 to 1.4 | rejection lower bound above the type-I ceiling | rejection 0.9869, 0.9837 to 0.9896 | pass |
| `warning_rate` | `randomized__rule` | positive | a logistic model on four covariates, for a randomized treatment: the calibration-slope rule, which warns when the Bonferroni interval over the tested models lies above 0 and excludes 1 | rejection upper bound at or below the type-I ceiling | rejection 0.0228, 0.0191 to 0.0269 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The rule stays below the type-I ceiling on all
five laws where a warning is false, including the true propensity, a correct weak-signal model and
a randomized law. Four of those laws have a band control. On the same fits, the band's rate lies
above the ceiling on each of the four. The rule detects both tempered learners in every fit.

The `calibrated_strong` rate, 0.0539, is the largest of the five positive cells. Its 99% interval,
0.0482 to 0.0600, contains the family level of 0.05. The rate of a finding on either tested model
is therefore consistent with 0.05. The study bounds it at the type-I ceiling of 0.10.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 10000 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | primary tests against truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 11 | repeated-sampling property cells |
| `property_cells_passed` | 11 | cells whose own and family verdicts pass |
| `max_standardized_bias` | 0.0572 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9484 | lowest measured primary-study coverage |
| `min_se_ratio_ci_lower` | 0.9790 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0199 | highest bootstrap primary SE-ratio endpoint |
| `properties[warning_rate/calibrated_weak__rule]:rejection_rate` | 0.0364 | rule rate, known weak propensity |
| `properties[warning_rate/calibrated_strong__rule]:rejection_rate` | 0.0539 | rule rate, known strong propensity |
| `properties[warning_rate/calibrated_strong__rule]:rejection_ci_upper` | 0.0600 | its 99% upper endpoint |
| `properties[warning_rate/correct_weak__rule]:rejection_rate` | 0.0225 | rule rate, correct model on one weak covariate |
| `properties[warning_rate/correct_nested_weak__rule]:rejection_rate` | 0.0403 | rule rate, correct model on four covariates |
| `properties[warning_rate/randomized__rule]:rejection_rate` | 0.0228 | rule rate, randomized law |
| `properties[warning_rate/calibrated_weak__fixed_band]:rejection_rate` | 0.2543 | band rate, known weak propensity |
| `properties[warning_rate/correct_weak__fixed_band]:rejection_rate` | 0.3603 | band rate, correct model on one weak covariate |
| `properties[warning_rate/correct_nested_weak__fixed_band]:rejection_rate` | 0.7353 | band rate, correct model on four covariates |
| `properties[warning_rate/randomized__fixed_band]:rejection_rate` | 0.9869 | band rate, randomized law |
| `properties[warning_rate/correct_nested_weak__rule]:mean_estimate` | 0.5843 | mean out-of-fold slope of the correct model on four covariates |
| `properties[warning_rate/randomized__rule]:mean_estimate` | -0.1451 | mean out-of-fold slope on the randomized law |
| `properties[power/overconfident_moderate__rule]:rejection_ci_lower` | 0.9995 | detection lower endpoint, doubled logit |
| `properties[power/underconfident_strong__rule]:rejection_ci_lower` | 0.9995 | detection lower endpoint, halved logit |
| `properties[warning_rate/correct_weak__rule]:se_ratio` | 1.4612 | mean SE over empirical SD, correct model on one weak covariate |
| `properties[warning_rate/correct_nested_weak__rule]:se_ratio` | 0.9435 | mean SE over empirical SD, correct model on four covariates |
| `properties[warning_rate/randomized__rule]:se_ratio` | 0.7693 | mean SE over empirical SD, randomized law |
| `properties[power/overconfident_moderate__rule]:se_ratio` | 4.2290 | mean SE over empirical SD, doubled logit |
| `properties[power/underconfident_strong__rule]:se_ratio` | 8.0270 | mean SE over empirical SD, halved logit |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals, and the family level of the rule |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:type_i_ceiling` | 0.1000 | the rate a positive cell must bound and a control must exceed |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power cell must clear |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit. No cell of this study reads it |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit. No cell of this study reads it |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit. No cell of this study reads it |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |

## Limitations

| limitation | what it means for use |
| --- | --- |
| The row publishes under the reporting policy, not gated | Every declared cell is green. The `reporting` policy does not assert that, so the fast tier recomputes each verdict and does not fail on a red one |
| There is no cross-implementation evidence | No maintained implementation computes this rule |
| No weights and no clusters | The rule reads both through the sandwich standard error. The exact witnesses in `tests/unit/test_calibration_slope_rule.py` pin that arithmetic. No repeated-sampling cell covers it |
| A binary propensity only | The rule also tests `propensity[<arm>]`, `missingness`, `intermediate` and a binary `outcome`. The study covers the binary propensity, and its outcome model is known |
| Three folds and one sample size | The fold intercepts and the cross-fit dependence change with the fold count and $n$ |
| Known and unpenalized logistic learners only | A flexible learner is not covered. A planning probe of 40 gradient-boosting fits is not evidence |
| The standard error treats the fold models as fixed | The sandwich takes each fold's predictions as given. For a fitted learner, the ratio of the mean standard error to the empirical standard deviation of the slope is 1.4612, 0.9435 and 0.7693 on the `correct_weak`, `correct_nested_weak` and `randomized` laws. It is 4.2290 and 8.0270 for the doubled and the halved logit. The primary SE ratios of the known propensity are 0.9976 and 1.0018. Read the rule's rates, not its interval, as the evidence for a fitted learner |
| Detection only for a scaled logit at moderate and strong signal | At a weak signal a doubled logit has an interval that often reaches 0. When the interval reaches 0, the rule gives no finding |
| No prediction at 0 or 1 | A prediction clipped at the probability bounds has a logit near $\pm 27.6$, which can dominate the regression and move the slope toward 0. When the interval reaches 0, the rule gives no finding |
| Point treatment and out-of-fold predictions only | The longitudinal report and an in-sample fit carry the slope and no rule |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/calibration_slope_warning/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/calibration_slope_warning/manifest.json)
records the seeds, margins, laws, source hashes, and result hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/calibration_slope_warning/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/calibration_slope_warning/properties.csv)
carry every published row.
