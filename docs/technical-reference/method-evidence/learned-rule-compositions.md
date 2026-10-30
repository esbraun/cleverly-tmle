# Learned-rule value with fold targeting, weights and clusters

This study validates the interval of `LearnedRuleValue` under three settings of the fold-evaluated
fit: fold targeting, known observation weights, and clusters. The estimator is
`TMLE(learned_rule=LearnedRule(), cv_evaluation=True)`, the subject of the
[gated learned-rule study](learned-rule-cvtmle.md), with `targeting_scheme="fold"`, `weights=` or
`id=`. Each fold learns the plug-in rule of its outcome regression on its training rows, and the
report averages the fold values with the weight $1/V$.
[Learned rules](../point-treatment-tmle.md#learned-rules) gives the contract, the conditions C1 to
C6, and how each setting changes them.

The target is a data-adaptive parameter, so each replication carries its own truth. The record
sets `truth_varies_by_replicate`, and every statistic reads the error, the estimate minus the
replication's truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** The gated study names the candidates and why each
fails for the base estimator. None of them fits the fold-average target with fold targeting,
observation weights or clusters either. A zero-row equivalence artifact records the absence of a
comparator.

## What was tested

| setting | declaration |
| --- | --- |
| laws | the RM30 laws of the [gated study](learned-rule-cvtmle.md#what-was-tested): $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$, $\operatorname{logit} g_0(1 \mid W) = 0.3 W_1 - 0.2 W_2$, and $\operatorname{logit} \bar Q_0(a, W) = 0.2 + 0.5 W_1 - 0.3 W_2 + a\, b(W)$ |
| fit | `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, g_bounds="auto")` at n = 2,000, so each fold holds 200 rows |
| outcome learner | `Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), LogisticRegression(C=1e6, max_iter=5000, tol=1e-10))` on $(A, W_1, W_2)$ |
| treatment learner | `LogisticRegression(C=1e6, max_iter=1000)` on $(W_1, W_2)$, a correct parametric model |
| primary budget | 4,000 replications for each setting |
| truth | each replication refits each fold's outcome learner on the fit's own training rows with the fit's weights. The refit rule must equal the fit's rule on every validation row. The truth is $(1/V) \sum_v E_0[w\, \bar Q_0(d_{nv}(W), W)] / E_0[w]$, by the trapezoid rule on 4,001 points of $W_1$ for each $W_2$, with $w = 1$ outside the weighted setting |
| oracle SE | the SD of the row-level curve at each fold rule, by the same quadrature. It is a descriptive column of `harness.csv.gz`, row-level for the clustered setting, and no rule reads it |
| primary verdicts | the framework defaults on the error: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20263011` and `resampling_seed=20263012` |
| policy | `gated` |

The table gives the three primary settings and the few-cluster property cell.

| setting | law, $b(W)$ | what the fit adds | n |
| --- | --- | --- | ---: |
| `fold_targeted` | `non_exceptional`, $0.1 + W_1$ | `targeting_scheme="fold"`: each validation fold fits its own fluctuation on its own rows | 2,000 |
| `weighted` | `misspecified_limit`, $0.8 W_1 + W_1^2 - 0.3$ | the known weight $w = 1 + 1.5 \cdot 1\{W_1 > 0\}$ through `weights=`. The rule is the plug-in of the weighted outcome regression, and the target is the value in the weighted population | 2,000 |
| `clustered` | `non_exceptional` plus a cluster effect $u \sim N(0, 1.5^2)$ on the logit scale, shared by the rows of a cluster | `id=` with 200 clusters of 10 rows. Folds move whole clusters, and the variance sums the curve within clusters | 2,000 |
| `clustered_few` | as `clustered` | 30 clusters of 34 rows, so the fit reports a t interval with $\min(J - 2, J - V) = 20$ degrees of freedom | 1,020 |

The outcome learner has no $W_1^2$ term, so at `misspecified_limit` the weighted and the
unweighted rules have different limits. `tests/unit/test_learned_rule_compositions.py` checks the
two limiting rules by population quadrature and checks that the fit's rule is the weighted one.
Under clusters the row-level $\bar Q_0$ is the marginal over $u$, by Gauss-Hermite quadrature with
64 nodes. A main-terms learner is then misspecified for it, and inference rests on condition C4
with the treatment model correct. The intra-cluster correlation of the curve is 0.144.

The X11 plan and its study design record fixed the settings, the learners, the budgets, the
seeds, the truth rule and the run form before any run.
`tests/unit/test_learned_rule_compositions_design.py` computes the design constants again.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned-rule law with the blip 0.1 + W1 and a cluster effect of SD 1.5 on the logit scale, 200 clusters of 10 rows | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, with fold targeting, known weights or clusters | -0.0013 to 0.000572 | 0.9445 | 0.9926 | pass |
| learned-rule law with the blip 0.1 + W1, fold targeting | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, with fold targeting, known weights or clusters | -0.000597 to 0.000644 | 0.9427 | 0.9746 | pass |
| learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight 1 + 1.5 * 1{W1 > 0}, a misspecified outcome learner | `ey_learned_rule[learned rule]` | fold average of the value of the rule learned on each fold's training rows from its outcome regression | `cleverly` fold-evaluated CV-TMLE of the learned-rule value, with fold targeting, known weights or clusters | -0.0011 to 0.000245 | 0.9537 | 0.9916 | pass |
<!-- /generated -->

## Theory properties

Every property cell uses the declared learners at its setting's law, unless the table names
another learner.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of each positive cell lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98, at 8,400 replications. The positives are the three primary settings at n = 2,000 and `clustered_few` at n = 1,020 | a shrunken-SE control multiplies each SE by 0.70. A noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. The clustered setting adds a row-variance control, the same estimate with the row-level SE that ignores the clusters. Each control's SE-ratio interval must lie below 0.93 |
| `targeting_necessity` | is the fold-targeted fit unbiased when the outcome learner omits $W_1$ | the same fit without its fluctuations, the fold average of the initial plug-ins at the fit's own rules, must miss the bias margin, and the paired displacement must reach 0.5 SD. 2,655 replications |

The noise scale of each control is the asymptotic SD of $\sqrt{n}$ times the estimate at the
setting's limit rule, which a quadrature of the learner's population projection computes. For
the clustered cells it adds the within-cluster covariance, $\sqrt{\operatorname{Var} D + (m - 1)
\operatorname{Cov}}$ for clusters of $m$ rows. These values are not semiparametric efficiency
bounds, and they size the noise controls only.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `interval_calibration` | `clustered__declared` | positive | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters of 10 rows: the declared fit with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9384 to 0.9513, SE ratio 0.9725 to 1.0111 | pass |
| `interval_calibration` | `clustered__noise_control` | control | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters of 10 rows: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8268 to 0.8476, SE ratio 0.6984 to 0.7272 | pass |
| `interval_calibration` | `clustered__row_variance_control` | control | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters of 10 rows: the same estimate with the row-level standard error that ignores the clusters | the SE-ratio interval must fall below the calibration band | coverage 0.7923 to 0.8147, SE ratio 0.6443 to 0.6695 | pass |
| `interval_calibration` | `clustered__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters of 10 rows: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8163 to 0.8376, SE ratio 0.6803 to 0.7079 | pass |
| `interval_calibration` | `clustered_few__declared` | positive | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 30 clusters of 34 rows (n = 1,020), with the t reference at 20 degrees of freedom: the declared fit with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9400 to 0.9528, SE ratio 0.9685 to 1.0092 | pass |
| `interval_calibration` | `clustered_few__noise_control` | control | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 30 clusters of 34 rows (n = 1,020), with the t reference at 20 degrees of freedom: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8420 to 0.8621, SE ratio 0.6907 to 0.7194 | pass |
| `interval_calibration` | `clustered_few__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 30 clusters of 34 rows (n = 1,020), with the t reference at 20 degrees of freedom: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8312 to 0.8518, SE ratio 0.6782 to 0.7063 | pass |
| `interval_calibration` | `fold_targeted__declared` | positive | the learned-rule law with the blip 0.1 + W1, fitted with fold targeting: the declared fit with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9425 to 0.9550, SE ratio 0.9739 to 1.0125 | pass |
| `interval_calibration` | `fold_targeted__noise_control` | control | the learned-rule law with the blip 0.1 + W1, fitted with fold targeting: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8231 to 0.8441, SE ratio 0.6933 to 0.7205 | pass |
| `interval_calibration` | `fold_targeted__shrunken_se_control` | control | the learned-rule law with the blip 0.1 + W1, fitted with fold targeting: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8153 to 0.8367, SE ratio 0.6813 to 0.7082 | pass |
| `interval_calibration` | `weighted__declared` | positive | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight 1 + 1.5 * 1{W1 > 0}: the declared fit with the study's logistic learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9343 to 0.9476, SE ratio 0.9498 to 0.9889 | pass |
| `interval_calibration` | `weighted__noise_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight 1 + 1.5 * 1{W1 > 0}: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8222 to 0.8433, SE ratio 0.6872 to 0.7158 | pass |
| `interval_calibration` | `weighted__shrunken_se_control` | control | the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight 1 + 1.5 * 1{W1 > 0}: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8021 to 0.8242, SE ratio 0.6647 to 0.6921 | pass |
| `targeting_necessity` | `fold_targeted__targeted` | positive | the learned-rule law with the blip 0.1 + W1, fitted with fold targeting: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0010 to 0.000789, margin 0.0045 | pass |
| `targeting_necessity` | `fold_targeted__untargeted` | control | the learned-rule law with the blip 0.1 + W1, fitted with fold targeting: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0204 to 0.0218, margin 0.0036 | pass |
<!-- /generated -->

## Result

Every primary test and every property cell passed. The interval covers the fold-average target
under fold targeting, known weights, and equal-size clusters at 200 and at 30 clusters, with the
declared logistic outcome learner.

| reading | value | source |
| --- | --- | --- |
| primary coverage, `fold_targeted` | 0.9427 | `performance-tests.csv` |
| primary coverage, `weighted` | 0.9537 | `performance-tests.csv` |
| primary coverage, `clustered` | 0.9445 | `performance-tests.csv` |
| coverage of the t interval at 30 clusters | 0.9400 to 0.9528 | `properties.csv`, `clustered_few__declared` |
| coverage with the row-level SE under clusters | 0.7923 to 0.8147 | `properties.csv`, `clustered__row_variance_control` |
| rule rows that the truth harness checked | 2,000 in each of 12,000 primary replications | `harness.csv.gz`, `rule_rows_checked` |
| solver warnings, primary | 0 | `harness.csv.gz`, `solver_warnings` |
| solver warnings, property | 18, on 9 rows of `interval_calibration` and `targeting_necessity` | `property-replicates.parquet`, `solver_warnings` |

The 9 property rows come from four draws. Calibration replicate 1516 of `fold_targeted` warned,
and its two controls reuse its rows. `targeting_necessity` replicates 270, 1521 and 2301 warned,
and each pair of arms shares one fit. The RM30 rule L4 counts a solver warning and does not treat
it as a failed fit, and no rule reads the count. A harness refit whose rule differed from the
fit's rule would have stopped the run, and the run completed.

The targeted arm of `targeting_necessity` uses an outcome learner that omits $W_1$. It covers
0.9186, with a 99% interval of 0.9040 to 0.9317 and an SE ratio of 0.8978, over 2,655
replications. The family gates bias only, so these numbers are descriptive. They are not a
calibration claim for a misspecified outcome learner.

The readings do not show the items in this list.

- Coverage at unequal cluster sizes, with estimated weights, or with weights and clusters
  together. Each such fit reads experimental or refuses.
- Coverage of these settings at the exceptional and weak-blip laws that the
  [boundary study](learned-rule-cvtmle-boundary.md) reads.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Names beginning `bound:` are
the asymptotic SD at the limit rule over $\sqrt{2000}$, keyed by setting. They size the noise
controls only, and no verdict reads them as an efficiency bound. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 4000 | primary replications per setting |
| `n` | 2000 | observations per primary replication |
| `independent_tests_total` | 3 | setting tests against the replication truth |
| `independent_tests_passed` | 3 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 15 | repeated-sampling property cells |
| `property_cells_passed` | 15 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0256 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9427 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9326 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9480 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0227 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-rule-compositions/fold_targeted/ey_learned_rule[learned rule]]:coverage` | 0.9427 | primary coverage at `fold_targeted` |
| `performance[cleverly-learned-rule-compositions/weighted/ey_learned_rule[learned rule]]:coverage` | 0.9537 | primary coverage at `weighted` |
| `performance[cleverly-learned-rule-compositions/clustered/ey_learned_rule[learned rule]]:coverage` | 0.9445 | primary coverage at `clustered` |
| `performance[cleverly-learned-rule-compositions/clustered/ey_learned_rule[learned rule]]:se_ratio` | 0.9926 | primary SE ratio at `clustered` |
| `properties[interval_calibration/fold_targeted__declared]:se_ratio_ci_lower` | 0.9739 | fold-targeted calibration SE-ratio lower endpoint, against 0.93 |
| `properties[interval_calibration/fold_targeted__declared]:se_ratio_ci_upper` | 1.0125 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/weighted__declared]:se_ratio_ci_lower` | 0.9498 | weighted calibration SE-ratio lower endpoint, against 0.93 |
| `properties[interval_calibration/weighted__declared]:se_ratio_ci_upper` | 0.9889 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/clustered__declared]:se_ratio_ci_lower` | 0.9725 | clustered calibration SE-ratio lower endpoint, against 0.93 |
| `properties[interval_calibration/clustered__declared]:se_ratio_ci_upper` | 1.0111 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/clustered_few__declared]:se_ratio_ci_lower` | 0.9685 | few-cluster calibration SE-ratio lower endpoint, against 0.93 |
| `properties[interval_calibration/clustered_few__declared]:se_ratio_ci_upper` | 1.0092 | its upper endpoint, against 1.07 |
| `properties[interval_calibration/clustered_few__declared]:coverage_ci_lower` | 0.9400 | few-cluster coverage lower endpoint, against 0.92 |
| `properties[interval_calibration/clustered_few__declared]:coverage_ci_upper` | 0.9528 | few-cluster coverage upper endpoint, against 0.98 |
| `properties[interval_calibration/clustered__row_variance_control]:se_ratio_ci_upper` | 0.6695 | row-variance control upper endpoint, which must lie below 0.93 |
| `properties[interval_calibration/clustered__row_variance_control]:coverage` | 0.8037 | coverage of the row-variance control |
| `properties[interval_calibration/fold_targeted__shrunken_se_control]:se_ratio_ci_upper` | 0.7082 | shrunken-SE control upper endpoint at `fold_targeted`, which must lie below 0.93 |
| `properties[interval_calibration/clustered_few__noise_control]:se_ratio_ci_upper` | 0.7194 | noise control upper endpoint at `clustered_few`, which must lie below 0.93 |
| `properties[targeting_necessity/fold_targeted__untargeted]:standardized_bias` | 1.4800 | bias of the fit without its fluctuations, in error SDs |
| `properties[targeting_necessity/fold_targeted__targeted]:targeting_displacement` | 1.1774 | paired displacement, against 0.5 |
| `properties[targeting_necessity/fold_targeted__targeted]:coverage` | 0.9186 | coverage of the targeted arm, which no verdict reads |
| `properties[targeting_necessity/fold_targeted__targeted]:coverage_ci_lower` | 0.9040 | its 99% lower endpoint |
| `properties[targeting_necessity/fold_targeted__targeted]:coverage_ci_upper` | 0.9317 | its 99% upper endpoint |
| `properties[targeting_necessity/fold_targeted__targeted]:se_ratio` | 0.8978 | SE ratio of the targeted arm, which no verdict reads |
| `mean_oracle_se_fold_targeted` | 0.0149 | mean oracle SE of the fixed fold rules at `fold_targeted`, from `harness.csv.gz`. No rule reads it |
| `mean_oracle_se_weighted` | 0.0160 | mean oracle SE at `weighted` |
| `mean_oracle_se_clustered` | 0.0151 | mean row-level oracle SE at `clustered` |
| `summary[cleverly-learned-rule-compositions/clustered/ey_learned_rule[learned rule]]:mean_std_error` | 0.0229 | mean reported SE at `clustered` |
| `bound:fold_targeted_standard_error` | 0.0149 | the asymptotic SD at the limit rule of `fold_targeted`, at n = 2,000 |
| `bound:weighted_standard_error` | 0.0160 | the asymptotic SD at the limit rule of `weighted`, at n = 2,000 |
| `bound:clustered_standard_error` | 0.0228 | the cluster-level asymptotic SD at the limit rule of `clustered`, at n = 2,000 |
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
| `margin:targeting_displacement` | 0.5000 | smallest paired displacement of `targeting_necessity`, in positive-arm error SDs |
| `margin:subject_failure_bound` | 0.0100 | the failed-fit bound of the committed run |
| `margin:subject_failures_allowed` | 40 | failed primary fits that bound allows per setting |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits this construction. The row rests on accuracy against the replication truth and on the theory properties |
| One law for each setting | Fold targeting and clusters run at `non_exceptional`, and weights at `misspecified_limit`. Each law meets condition C3. The exceptional and weak-blip laws are not tested with these settings |
| Equal cluster sizes only | The study draws 200 clusters of 10 and 30 clusters of 34. A learned-rule fit with unequal sizes reads experimental, because sizes that depend on the outcome put a ratio bias in the fold-evaluated point |
| Known weights only | The weight is a fixed function of $W_1$. Estimated weights refuse, and a fit with both weights and clusters reads experimental |
| One learner class for coverage | The primary and calibration cells use a logistic outcome model with treatment interactions. The learner that omits $W_1$ enters a bias verdict only, so the row does not claim coverage with it |
| Two cluster counts in ten folds | The t reference is measured at 30 clusters with 20 degrees of freedom, and the normal reference at 200 clusters. Fewer clusters are not tested |
| One primary size | The primary settings run at n = 2,000, and this study has no root-n ladder. The [gated study](learned-rule-cvtmle.md) measures the rate of the pooled fit |
| A binary treatment and a binary outcome | A continuous outcome with declared `q_bounds` is not tested. The package refuses more than two arms, missing outcomes, strata, repeats and the full-refit bootstrap for the learned rule |
| The truth harness needs deterministic learners | The truth refits each fold's learner with the fit's weights. A learner that draws its own randomness cannot give the same rule again |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_compositions/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_compositions/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_compositions/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_compositions/replicates.parquet),
the [harness columns](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_compositions/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_rule_compositions/properties.csv)
carry every published row.
