# Learned-rule value, mean over splits

This study validates the interval of `LearnedRuleValue` when the fit repeats its ten-fold split.
The estimator is `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True,
repeats=5)`, the subject of the [gated learned-rule study](learned-rule-cvtmle.md) with five
independent splits. Each split learns its own fold rules on its training rows, and the fit reports
the mean of the five split estimates. The variance is $n^{-2} \sum_i \bar\phi_i^2$, the second
moment of the mean $\bar\phi_i$ of the five split curves.
[The mean over splits](../point-treatment-tmle.md#learned-rules) gives the contract and the
argument that the mean of asymptotically linear splits is asymptotically linear.

The target is data-adaptive, so each replication carries its own truth: the mean over the five
splits of each split's fold-average value of the rules it learned. The record sets
`truth_varies_by_replicate`, and every statistic reads the error, the estimate minus the
replication's truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** The gated study names the candidates and why each
fails for the base estimator. None of them reports a mean over independent splits of the
fold-average target either. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | the RM30 laws of the [gated study](learned-rule-cvtmle.md#what-was-tested): $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W)$. `non_exceptional` has $b(W) = 0.1 + W_1$ and `misspecified_limit` has $b(W) = 0.8 W_1 + W_1^2 - 0.3$ |
| fit | `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto", repeats=5)` at n = 2,000, on rows only: no weights and no `id=` |
| outcome learner | `Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), LogisticRegression(C=1e6, max_iter=5000, tol=1e-10))` on $(A, W_1, W_2)$ |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, a correct parametric model |
| primary budget | 2,000 replications for each law |
| truth | each replication refits, for each split, each fold's outcome learner on the split's training rows. The refit rule must equal the split's rule on every validation row. The split's value is the fold average of $E_0[\bar Q_0(d_{nv}(W), W)]$ by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$, and the truth is the mean of the five split values |
| oracle SE | the mean over the splits of each split's row-curve oracle SE. It is a descriptive column of `harness.csv.gz`, and no rule reads it |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20263021` and `resampling_seed=20263022` |
| policy | `gated` |

The engine draws the five split seeds from the replication's fold seed. Two guards stop a
replication whose reported point differs from the mean of the split points, or whose reported
variance exceeds the mean of the split variances, each at relative 1e-12. The second is Jensen's
inequality for $n^{-2} \sum_i \bar\phi_i^2$ and holds on every draw. The X11 plan and its study
design record fixed the laws, the learners, the budgets, the seeds and the truth rule before any
run. `tests/unit/test_learned_rule_split_mean_design.py` computes the design constants again, and
`tests/unit/test_learned_rule_split_mean_harness.py` checks the harness and the controls.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, a misspecified outcome learner | `ey_learned_rule[learned rule]` | mean over five splits of the fold average of the value of the rule learned on each fold's training rows | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, as the mean over five independent ten-fold splits | -0.0010 to 0.000740 | 0.9440 | 0.9753 | pass |
| learned-rule law with the blip 0.1 + W1, correctly specified learners | `ey_learned_rule[learned rule]` | mean over five splits of the fold average of the value of the rule learned on each fold's training rows | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, as the mean over five independent ten-fold splits | -0.000725 to 0.000965 | 0.9500 | 1.0050 | pass |
<!-- /generated -->

## Theory properties

Every property cell uses the declared learners at its law.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each positive cell lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, at 6,400 replications. The positives are the two laws at n = 2,000 | a shrunken-SE control multiplies each SE by 0.70. A noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. An independent-splits control keeps the estimate and reports the mean single-split SE over $\sqrt{5}$, which treats the five splits as independent samples. Each control's SE-ratio interval must lie below 0.93 |

The noise scale is the asymptotic SD of $\sqrt{n}$ times the estimate at the law's limit rule,
which a quadrature of the learner's population projection computes. Under C3 every split shares
that limit, so the mean curve has the SD of one split's. These values size the noise controls
only, and no verdict reads them as an efficiency bound.

The independent-splits control fails by construction. The splits reuse every row, so the control's
SE is about 0.45 of the reported SE on every draw, and it can reach 0.93 only where the positive
has already failed. It shows the size of the mistake and is a sanity control, like the two derived
controls. The evidence that the variance accounts for the correlation between splits is the
positive cell passing its band.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `misspecified_limit__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term: the declared fit with five splits and the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9348 to 0.9499, SE ratio 0.9533 to 0.9981 | pass |
| `interval_calibration` | `misspecified_limit__independent_splits_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term: the same estimate with the mean single-split standard error over the square root of five, which treats the five splits as independent samples | the SE-ratio interval must fall below the calibration band | coverage 0.5996 to 0.6311, SE ratio 0.4330 to 0.4536 | pass |
| `interval_calibration` | `misspecified_limit__noise_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8116 to 0.8363, SE ratio 0.6781 to 0.7094 | pass |
| `interval_calibration` | `misspecified_limit__shrunken_se_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8116 to 0.8363, SE ratio 0.6677 to 0.6993 | pass |
| `interval_calibration` | `non_exceptional__declared` | positive | the learned-rule law with the blip 0.1 + W1, whose logistic outcome learner is correct: the declared fit with five splits and the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9421 to 0.9563, SE ratio 0.9714 to 1.0162 | pass |
| `interval_calibration` | `non_exceptional__independent_splits_control` | control | the learned-rule law with the blip 0.1 + W1, whose logistic outcome learner is correct: the same estimate with the mean single-split standard error over the square root of five, which treats the five splits as independent samples | the SE-ratio interval must fall below the calibration band | coverage 0.6028 to 0.6342, SE ratio 0.4398 to 0.4606 | pass |
| `interval_calibration` | `non_exceptional__noise_control` | control | the learned-rule law with the blip 0.1 + W1, whose logistic outcome learner is correct: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8144 to 0.8389, SE ratio 0.6783 to 0.7103 | pass |
| `interval_calibration` | `non_exceptional__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, whose logistic outcome learner is correct: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8102 to 0.8349, SE ratio 0.6799 to 0.7110 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The interval covers the mean-over-splits target
at both laws with the declared logistic outcome learner, and the SE ratio sits inside the
calibration band without a correction for the degrees of freedom that overlapping splits lose.

| reading | value | source |
| --- | --- | --- |
| primary coverage, `non_exceptional` | 0.9500 | `performance-tests.csv` |
| primary coverage, `misspecified_limit` | 0.9440 | `performance-tests.csv` |
| calibration coverage, `non_exceptional` | 0.9421 to 0.9563 | `properties.csv`, `non_exceptional__declared` |
| calibration coverage, `misspecified_limit` | 0.9348 to 0.9499 | `properties.csv`, `misspecified_limit__declared` |
| SE ratio of the independent-splits control | 0.4398 to 0.4606 at `non_exceptional`, 0.4330 to 0.4536 at `misspecified_limit` | `properties.csv`, `independent_splits_control` |
| rule rows that the truth harness checked | 10,000 in each of 4,000 primary replications | `harness.csv.gz`, `rule_rows_checked` |
| solver warnings, primary | 4, on 2 replications | `harness.csv.gz`, `solver_warnings` |
| solver warnings, property | 8, on 4 draws of `interval_calibration` | `property-replicates.parquet`, `solver_warnings` |

Each replication checks every row once per split, so 10,000 rows is five times n. The 4 property
draws that warned give 16 property rows, because a draw's controls reuse its fit. The RM30 rule L4
counts a solver warning and does not treat it as a failed fit, and no rule reads the count. A
harness refit whose rule differed from the fit's rule would have stopped the run, and the run
completed.

### What the band cannot separate

The calibration band is plus or minus 7%, and close alternatives sit inside it. In a pilot of
4,000 draws per law, the mean of the split variances, the Jensen upper bound of the reported
variance, gave an SE 1.2% and 1.6% above the reported one. The first split's own SE gave ratios of
1.019 and 1.001. The per-split truth differs from the first split's truth by an SD of 6.5e-5 and
1.2e-4, against an error SD of 0.0145, so a harness that read one split for all five would give
the same ratios to three decimals. The positive cells therefore establish that the mean-over-splits
report is calibrated. The exact variance formula and the per-split truth rest on the unit witnesses
in `tests/unit/test_learned_rule_split_mean.py` and
`tests/unit/test_learned_rule_split_mean_harness.py`, which rebuild every split's value from a
single-split fit at that split's seed and check deliberate mutations that pair a split's folds with
another split's rule.

### What the mean buys

The summary carries two descriptive columns that no verdict reads. The first split's error SD is
the SD of the first split's estimate against its own target. The median columns are the error of
the median of the five split estimates, read against the mean target. They show what the refused
median would have reported, and the median estimates no defined parameter.

| reading | value | source |
| --- | --- | --- |
| error SD of the mean over splits, `non_exceptional` | 0.0148 | `properties.csv`, `empirical_se` |
| error SD of the first split alone, `non_exceptional` | 0.0151 | `properties.csv`, `first_split_error_sd` |
| error SD of the mean over splits, `misspecified_limit` | 0.0151 | `properties.csv`, `empirical_se` |
| error SD of the first split alone, `misspecified_limit` | 0.0154 | `properties.csv`, `first_split_error_sd` |
| bias of the median against the mean target, `non_exceptional` | -0.000005 | `properties.csv`, `median_aggregate_bias` |
| bias of the median against the mean target, `misspecified_limit` | 0.000205 | `properties.csv`, `median_aggregate_bias` |

Five splits lower the error SD by about 1.6% to 1.7% relative to one split, so the gain is small
and the study does not claim more. The median's bias against the mean target is within the
bias margin at both laws. That fits the argument that the two differ by $o_P(n^{-1/2})$, and it is
no reason to report the median, which has no target.

The readings do not show the items in this list.

- A repeat count other than 5. The linearity argument covers any fixed $R$, and the degrees of
  freedom that overlapping splits lose, about $(2V - V/R - 1)/n$ or 0.85% of the variance here, are
  ignored.
- Weights, fold targeting, or both together with `repeats` above 1. Each such fit reads experimental.
- Clusters with `repeats` above 1, which refuse ([X11](../../roadmap.md#x11-learned-policy-follow-ups) part (i2)).
- The exceptional and weak-blip laws that the [boundary study](learned-rule-cvtmle-boundary.md) reads.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the asymptotic SD at the limit rule over $\sqrt{2000}$, keyed by law. They size the noise
controls only, and no verdict reads them as an efficiency bound. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | primary replications per law |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 2 | law tests against the replication truth |
| `independent_tests_passed` | 2 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 8 | repeated-sampling property cells |
| `property_cells_passed` | 8 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0088 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9440 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9294 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9335 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0447 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-split-mean/non_exceptional/ey_learned_rule[learned rule]]:coverage` | 0.9500 | primary coverage at `non_exceptional` |
| `performance[cleverly-learned-rule-split-mean/misspecified_limit/ey_learned_rule[learned rule]]:coverage` | 0.9440 | primary coverage at `misspecified_limit` |
| `performance[cleverly-learned-rule-split-mean/non_exceptional/ey_learned_rule[learned rule]]:se_ratio` | 1.0050 | primary SE ratio at `non_exceptional` |
| `performance[cleverly-learned-rule-split-mean/misspecified_limit/ey_learned_rule[learned rule]]:se_ratio` | 0.9753 | primary SE ratio at `misspecified_limit` |
| `properties[interval_calibration/non_exceptional__declared]:se_ratio_ci_lower` | 0.9714 | `non_exceptional` calibration SE-ratio lower endpoint, against 0.93 |
| `properties[interval_calibration/non_exceptional__declared]:se_ratio_ci_upper` | 1.0162 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/non_exceptional__declared]:coverage_ci_lower` | 0.9421 | `non_exceptional` coverage lower endpoint, against 0.92 |
| `properties[interval_calibration/non_exceptional__declared]:coverage_ci_upper` | 0.9563 | its upper endpoint, against 0.98 |
| `properties[interval_calibration/misspecified_limit__declared]:se_ratio_ci_lower` | 0.9533 | `misspecified_limit` calibration SE-ratio lower endpoint, against 0.93 |
| `properties[interval_calibration/misspecified_limit__declared]:se_ratio_ci_upper` | 0.9981 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/misspecified_limit__declared]:coverage_ci_lower` | 0.9348 | `misspecified_limit` coverage lower endpoint, against 0.92 |
| `properties[interval_calibration/misspecified_limit__declared]:coverage_ci_upper` | 0.9499 | its upper endpoint, against 0.98 |
| `properties[interval_calibration/non_exceptional__independent_splits_control]:se_ratio_ci_upper` | 0.4606 | independent-splits control upper endpoint at `non_exceptional`, which must lie below 0.93 |
| `properties[interval_calibration/misspecified_limit__independent_splits_control]:se_ratio_ci_upper` | 0.4536 | independent-splits control upper endpoint at `misspecified_limit` |
| `properties[interval_calibration/non_exceptional__shrunken_se_control]:se_ratio_ci_upper` | 0.7110 | shrunken-SE control upper endpoint at `non_exceptional` |
| `properties[interval_calibration/misspecified_limit__noise_control]:se_ratio_ci_upper` | 0.7094 | noise control upper endpoint at `misspecified_limit` |
| `properties[interval_calibration/non_exceptional__declared]:empirical_se` | 0.0148 | error SD of the mean over splits at `non_exceptional` |
| `properties[interval_calibration/non_exceptional__declared]:first_split_error_sd` | 0.0151 | error SD of the first split alone at `non_exceptional` |
| `properties[interval_calibration/misspecified_limit__declared]:empirical_se` | 0.0151 | error SD of the mean over splits at `misspecified_limit` |
| `properties[interval_calibration/misspecified_limit__declared]:first_split_error_sd` | 0.0154 | error SD of the first split alone at `misspecified_limit` |
| `properties[interval_calibration/non_exceptional__declared]:median_aggregate_bias` | -0.000005 | bias of the median against the mean target at `non_exceptional` |
| `properties[interval_calibration/misspecified_limit__declared]:median_aggregate_bias` | 0.000205 | bias of the median against the mean target at `misspecified_limit` |
| `mean_oracle_se_non_exceptional` | 0.0149 | mean oracle SE of the fixed fold rules at `non_exceptional`, from `harness.csv.gz`. No rule reads it |
| `mean_oracle_se_misspecified_limit` | 0.0149 | mean oracle SE at `misspecified_limit` |
| `bound:non_exceptional_standard_error` | 0.0149 | the asymptotic SD at the limit rule of `non_exceptional`, at n = 2,000 |
| `bound:misspecified_limit_standard_error` | 0.0149 | the asymptotic SD at the limit rule of `misspecified_limit`, at n = 2,000 |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | factor the shrunken-SE control applies to each SE |
| `margin:type_i_ceiling` | 0.1000 | largest size a one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical SDs |
| `margin:rmse_noninferiority` | 1.1000 | largest external-comparison RMSE ratio bound |
| `margin:coverage_noninferiority` | -0.0250 | smallest external-comparison coverage difference bound |
| `margin:calibration_noninferiority` | 0.0500 | largest external-comparison calibration excess bound |
| `margin:minimum_power` | 0.8000 | rejection lower bound a power control must clear |
| `margin:root_n_slope` | -0.5000 | contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted root-n slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted root-n slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate a root-n interval must exclude |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 20 | failed primary fits that bound allows per setting |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package reports the mean over independent splits of the fold-average learned-rule value. The row rests on accuracy against the replication truth and on the theory properties |
| The independent-splits control fails by construction | Its SE is about 0.45 of the reported SE on every draw, so it shows the size of the independent-splits mistake and cannot show that the variance is right. The positive cells passing their band carry that claim |
| The band cannot separate close variance formulas | The mean of the split variances, one split's own SE and a harness that reads one split's truth all sit inside the band. The exact formula and the per-split truth rest on the unit witnesses |
| Five splits of ten folds | The study measures `repeats=5` and `n_folds=10`. The argument covers any fixed repeat count, and the fit ignores the loss of about $(2V - V/R - 1)/n$ in degrees of freedom, 0.85% of the variance here |
| Rows only, pooled targeting | The study fits unweighted rows with `targeting_scheme="pooled"` and no `id=`. A fit with weights or fold targeting at `repeats` above 1 reads experimental, and `id=` refuses |
| The mean has no efficient-gain claim | Five splits lower the error SD by about 1.6% to 1.7% against one split. The study gates calibration only, and no verdict reads the gain or the median columns |
| Two laws that meet C3 | Both laws have a fixed limit rule. The exceptional and weak-blip laws are not tested with `repeats` above 1 |
| One primary size | The primary cells run at n = 2,000, and this study has no root-n ladder. The [gated study](learned-rule-cvtmle.md) measures the rate of the pooled fit |
| A binary treatment and a binary outcome | A continuous outcome with declared `q_bounds` is not tested |
| The truth harness needs deterministic learners | The truth refits each fold's learner for each split. A learner that draws its own randomness cannot give the same rule again |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_split_mean/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_split_mean/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_split_mean/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_split_mean/replicates.parquet),
the [harness columns](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_split_mean/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_split_mean/properties.csv)
carry every published row.
