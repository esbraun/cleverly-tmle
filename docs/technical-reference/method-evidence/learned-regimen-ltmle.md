# Learned two-node regimens

This study validates the interval of the cross-validated longitudinal TMLE of a learned two-node
regimen, `LTMLE(LearnedRegimen(), n_folds=10)`, on a censored panel with a binary treatment at
each node and an end-of-study binary outcome. Each training fold $v$ learns its rule pair
$d_{nv} = (d_{nv,1}, d_{nv,2})$ by backward plug-in Q-learning inside its own untargeted
recursion, each validation row is evaluated under its own fold's pair, and one pooled logistic
fluctuation targets each node. [Learned regimens](../longitudinal-tmle.md#learned-regimens) gives
the construction and its conditions.

The target is data-adaptive: $\tilde\psi_{0n} = \sum_v (n_v / n)\, \Psi_{d_{nv}}(P_0)$, the
stacked value of the rule pairs the folds learned. That is not the value of the optimal pair.
Each replication carries its own truth, and every statistic reads the error, the estimate minus
that truth.
[A truth that varies by replication](../../development/method-benchmarking.md#a-truth-that-varies-by-replication)
gives the rule.

**No canonical implementation is compared.** No maintained package fits a learned two-node regimen
by cross-validated TMLE. A zero-row equivalence artifact records the absence of a comparator.

## What was tested

| setting | declaration |
| --- | --- |
| law | $W_1 \sim U(-1, 1)$, $W_2 \sim \operatorname{Bernoulli}(0.5)$; $\operatorname{logit} g_1 = 0.8 W_1 - 0.3 W_2$; $\operatorname{logit} P(C_1 = 1) = 2.5 + 0.3 W_1$; $L_2 \sim N(0.5 W_1 + 0.2 W_2 - 0.8 A_1, 1)$; $\operatorname{logit} g_2 = -0.2 + 0.3 L_2 + 0.4 A_1$; $\operatorname{logit} P(C_2 = 1) = 2.5 - 0.2 L_2$; $\operatorname{logit} \bar Q_0 = -0.2 + 0.6 W_1 - 0.2 W_2 + 0.3 L_2 + a_2 (-0.1 + 1.2 L_2 - 0.6 a_1) + a_1 (0.3 + 1.2 W_1)$ |
| optimal pair | node 2 treats where $L_2 > (0.1 + 0.6 a_1) / 1.2$ (34% of the rows that follow node 1), and node 1 is a threshold in $W_1$ for each $W_2$ (36% treated). The optimal value is 0.557923 and the best static regimen's 0.4600. Both boundaries have a margin: 5.6% of rows have a node-2 blip within 0.1 logit of zero, and 4.0% a node-1 value gap below 0.01 |
| fit | `LTMLE(LearnedRegimen(), n_folds=10, random_state=<fold seed>)` with the learners below and every other setting at its default, `repeats=1` |
| node-2 learner | `LogisticRegression(C=1e6)` on $(W_1, W_2, L_2)$. In the Q-learning that learns the rules it is fitted per stratum $(A_1, A_2)$, where it is a correct model. The estimator's own node-2 regression pools the followers' strata with no treatment column, so there it is an approximation |
| node-1 learners | `declared`: a cubic polynomial in $(W_1, W_2)$ by least squares, an approximation. `squared_w1`: least squares on $(W_1^2, W_2)$, which keeps the confounder $W_1$ of $A_1$ only through its square |
| treatment and censoring learners | main-terms logistic models, correct at both nodes, fitted in every cell |
| support key | regimen kind `learned`, splitting `("cross_fitted",)`, missing `("censoring",)`, two binary nodes, an end-of-study binary outcome and every option at its default, pinned by `tests/unit/test_learned_regimen_ltmle_design.py` |
| truth | per replication: refit each fold's Q-learning on the fit's training rows with scikit-learn alone, require both rules to equal the stored ones on every row, and integrate each fold's pair by the trapezoid rule on 1,001 points of $W_1$ for each $W_2$ and by Gauss-Legendre over $L_2$, split at the fold's node-2 threshold |
| stored variance | $\frac{1}{n(n-1)}\sum_i (Z_i - \hat\psi)^2$ of the stacked curve centred at the pooled estimate. The harness rebuilds the curve from the stored node steps and stops the run if its variance differs from the reported one by more than a relative $10^{-8}$ |
| primary budget | 1,200 replications at n = 2,000 |
| primary verdicts | the framework defaults: the 99% Student bias interval inside 0.25 SD, the lower end of the 99% Clopper-Pearson coverage interval at or above 0.90, and the SE-ratio interval inside 0.80 to 1.20 |
| seeds | `seed=20271431` and `resampling_seed=20271432` |
| policy | `gated` |

The law, the learners, the budgets, the seeds and the support-row conditions were fixed before any
run, in the X11 plan, its implementation review and the study design record with its review.
`tests/unit/test_learned_regimen_ltmle_design.py` pins the law's values by quadrature, the sizing
and the seeds. `tests/unit/test_learned_regimen_ltmle_harness.py` checks the fits, the key, the
truth, the rule refit, the exactness gate and the property path.
`tests/unit/test_longitudinal_learned_regimen.py` holds the unit witnesses of the construction.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| censored two-node panel with binary treatments and an end-of-study binary outcome, the declared learners, correct treatment and censoring models, ten folds | `ey_regimen[learned regimen]` | the fold-size-weighted average of the values of the rule pairs learned on each training fold, sum_v (n_v / n) Psi_{d_nv}(P_0) | `cleverly` cross-validated longitudinal TMLE of a learned two-node regimen: each fold's rule pair learned by backward plug-in Q-learning on its training rows, each validation row evaluated under its own fold's pair, one pooled fluctuation per node | -0.0013 to 0.0017 | 0.9542 | 1.0168 | pass |
<!-- /generated -->

## Theory properties

Every property cell fits n = 2,000 on the one law.

| family | what it asks | how it fails |
| --- | --- | --- |
| `interval_calibration` | does the SE ratio of the declared fit (9,200 replications) lie inside 0.93 to 1.07, and its coverage inside 0.92 to 0.98 | a shrunken-SE control multiplies each SE by 0.80, and a noise control adds noise of one asymptotic SD over $\sqrt{n}$ to each estimate. Each control's SE-ratio interval must fall below the band |
| `conservative_coverage` | with the misspecified `squared_w1` node-1 learner and correct treatment and censoring models, is the bias interval inside the margin and the coverage interval's lower end at or above 0.92, with 2,400 replications | the control multiplies each SE by 0.80, and its coverage interval must lie wholly below 0.92. No verdict reads the SE ratio |
| `targeting_necessity` | does the fluctuation remove the plug-in's bias, for `squared_w1` and for the declared learners, with 1,200 replications each | the control is the same fit's untargeted plug-in, the mean of its stitched node-1 initial values. The positive's bias interval must lie inside the margin, the control's outside it, and the smaller displacement over the two labels must be at least 0.5 positive-arm SDs |
| `fold_locality` | is the fit, with each fold's pair learned on its training rows, unbiased for the value of its own pairs, with 1,200 replications | the control fits the same sample and folds with each fold's pair learned on its own validation rows, read against those pairs' value. Its bias interval must lie outside the margin, and the displacement must be at least 0.5 |

The noise scale is the SD of $\sqrt{n}$ times the estimate at the optimal pair, 0.894934 by
quadrature, and no verdict reads it as an efficiency bound. The `squared_w1` limit rule treats
50% at node 1 and is worth 0.5046, 0.053 below the optimum, and it crosses its boundary with a
nonzero slope, so the cell keeps a margin. Its Q-learning plug-in is biased at its limit by 2.17
SEs at n = 2,000. The declared learners' plug-in bias at n = 2,000 is a finite-sample bias whose
limit is about 0. The design record gives the sizing.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `conservative_coverage` | `squared_w1__declared` | positive | the censored two-node learned-regimen law with a misspecified node-1 regression on (W1^2, W2), correct logistic treatment and censoring models, n = 2,000: the declared fit with that node-1 learner and the correct treatment and censoring models | bias interval inside the margin, and the coverage interval's lower end at or above 0.92; the SE ratio is read by no verdict | bias -0.0022 to 0.000272, margin 0.0060, coverage 0.9264 to 0.9518, SE ratio 0.9594 | pass |
| `conservative_coverage` | `squared_w1__shrunken_coverage_control` | control | the censored two-node learned-regimen law with a misspecified node-1 regression on (W1^2, W2), correct logistic treatment and censoring models, n = 2,000: the declared fit's standard errors multiplied by 0.80 | the coverage interval must lie wholly below 0.92 | coverage 0.8457 to 0.8820, SE ratio 0.7675 | pass |
| `fold_locality` | `two_node__fold_local` | positive | the censored two-node learned-regimen law with the declared learners: a logistic node-2 regression on (W1, W2, L2) and a cubic node-1 regression on (W1, W2), correct logistic treatment and censoring models, n = 2,000: the declared fit, each fold's rule pair learned on the fold's training rows | bias interval inside the equivalence margin | bias -0.0020 to 0.0011, margin 0.0053 | pass |
| `fold_locality` | `two_node__validation_rows` | control | the censored two-node learned-regimen law with the declared learners: a logistic node-2 regression on (W1, W2, L2) and a cubic node-1 regression on (W1, W2), correct logistic treatment and censoring models, n = 2,000: the same sample and folds, with each fold's rule pair learned on the fold's own validation rows, read against the value of those pairs | bias interval must fall entirely outside the margin | bias 0.0884 to 0.0916, margin 0.0052 | pass |
| `interval_calibration` | `two_node__declared` | positive | the censored two-node learned-regimen law with the declared learners: a logistic node-2 regression on (W1, W2, L2) and a cubic node-1 regression on (W1, W2), correct logistic treatment and censoring models, n = 2,000: the declared fit with the study's learners | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9319 to 0.9449, SE ratio 0.9601 to 0.9976 | pass |
| `interval_calibration` | `two_node__noise_control` | control | the censored two-node learned-regimen law with the declared learners: a logistic node-2 regression on (W1, W2, L2) and a cubic node-1 regression on (W1, W2), correct logistic treatment and censoring models, n = 2,000: a declared scale of independent noise is added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8235 to 0.8436, SE ratio 0.6894 to 0.7158 | pass |
| `interval_calibration` | `two_node__shrunken_se_control` | control | the censored two-node learned-regimen law with the declared learners: a logistic node-2 regression on (W1, W2, L2) and a cubic node-1 regression on (W1, W2), correct logistic treatment and censoring models, n = 2,000: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8661 to 0.8840, SE ratio 0.7682 to 0.7983 | pass |
| `targeting_necessity` | `squared_w1__targeted` | positive | the censored two-node learned-regimen law with a misspecified node-1 regression on (W1^2, W2), correct logistic treatment and censoring models, n = 2,000: the declared fit with those learners, so targeting does the adjusting | bias interval inside the equivalence margin | bias -0.0024 to 0.0011, margin 0.0058 | pass |
| `targeting_necessity` | `squared_w1__untargeted` | control | the censored two-node learned-regimen law with a misspecified node-1 regression on (W1^2, W2), correct logistic treatment and censoring models, n = 2,000: the same fit's untargeted plug-in, the mean of its stitched node-1 initial values, with no fluctuation | bias interval must fall entirely outside the margin | bias 0.0585 to 0.0620, margin 0.0058 | pass |
| `targeting_necessity` | `two_node__targeted` | positive | the censored two-node learned-regimen law with the declared learners: a logistic node-2 regression on (W1, W2, L2) and a cubic node-1 regression on (W1, W2), correct logistic treatment and censoring models, n = 2,000: the declared fit with those learners, so targeting does the adjusting | bias interval inside the equivalence margin | bias -0.0023 to 0.000891, margin 0.0053 | pass |
| `targeting_necessity` | `two_node__untargeted` | control | the censored two-node learned-regimen law with the declared learners: a logistic node-2 regression on (W1, W2, L2) and a cubic node-1 regression on (W1, W2), correct logistic treatment and censoring models, n = 2,000: the same fit's untargeted plug-in, the mean of its stitched node-1 initial values, with no fluctuation | bias interval must fall entirely outside the margin | bias 0.0142 to 0.0171, margin 0.0049 | pass |
<!-- /generated -->

## Result

Every primary test and every gated property cell passed, and no fit failed. The primary test
reads coverage of 0.954 and an SE ratio of 1.017 over 1,200 draws. The calibration positive reads
an SE ratio of 0.979, with a 99% interval of 0.960 to 0.998, and coverage of 0.939 over 9,200
draws. Each control fails as declared.

| reading | value | source |
| --- | --- | --- |
| primary tests passed | 1 of 1 | `performance-tests.csv` |
| gated property cells passed | 11 of 11 | `properties.csv` |
| failed fits | 0 | `manifest.json`, `properties.csv` and the run log |
| calibration SE-ratio interval | 0.9601 to 0.9976 | `properties.csv`, `two_node__declared` |
| calibration coverage interval | 0.9319 to 0.9449 | `properties.csv`, `two_node__declared` |

The support row was declared before the run. It needed the primary bias leg, coverage floor and
SE sanity, every `interval_calibration`, `conservative_coverage`, `targeting_necessity` and
`fold_locality` verdict to pass. They did, so the registered row `G47` covers the learned
two-node regimen under the cross-fitted censored fit with every option at its default. Node
bounds, `ratio_trim=`, a stopping rule, a known mechanism, an uncensored panel and a continuous
outcome with `q_bounds=` stay experimental. The support key records each of them apart from the
row, so a learned-regimen fit with a stopping rule matches no row: the stopping-rule rows `G55`
cover static regimens only.

### The calibration positive

The SE ratio reads 0.979, and its interval sits below 1 and inside the band. The coverage
interval, 0.932 to 0.945, also sits below 0.95 and inside the band. The primary cell reads an SE
ratio of 1.017 and coverage of 0.954 at 1,200 draws, so the two readings bracket 1. Coverage of the
learned regimen rests on the correct treatment and censoring models: the reported curve omits the
$D_g$ term, which makes it conservative when those models are correct.

### The controls

The shrunken-SE and noise controls read SE ratios of 0.783 and 0.702. With `squared_w1`, the
targeted fit covers at 0.940 and its control at 0.865. The untargeted plug-in is biased by 0.060
with `squared_w1` and by 0.0157 with the declared learners, 2.60 and 0.80 of their own SDs, and
the targeted fits' bias intervals sit inside the margin. With each fold's pair learned on its
validation rows, the fit is biased by 0.090 against the value of those pairs, 4.30 of its own SDs,
so the instrument resolves rules learned on the rows that evaluate them.

No primary or property draw raised a solver warning, and the rule refit found the fit's rules on
every row of every draw.

## Measured values

Names beginning `margin:` are thresholds declared before the run. The name beginning `bound:` is
the SD of $\sqrt{n}$ times the estimate at the optimal pair over $\sqrt{2000}$. It sizes the noise
control only, and no verdict reads it as an efficiency bound. Everything else is measured from the
committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1200 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 1 | parameter tests against the truth |
| `independent_tests_passed` | 1 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 11 | gated repeated-sampling property cells |
| `property_cells_passed` | 11 | cells whose own and family verdicts pass |
| `subject_failures` | 0 | failed primary fits |
| `max_standardized_bias` | 0.0108 | largest absolute primary bias in error SDs |
| `min_coverage` | 0.9542 | lowest measured primary coverage |
| `min_coverage_ci_lower` | 0.9363 | lowest exact 99% primary coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9686 | lowest bootstrap primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0686 | highest bootstrap primary SE-ratio endpoint |
| `performance[cleverly-learned-regimen-ltmle/two_node/ey_regimen[learned regimen]]:coverage` | 0.9542 | primary coverage |
| `performance[cleverly-learned-regimen-ltmle/two_node/ey_regimen[learned regimen]]:se_ratio` | 1.0168 | primary SE ratio |
| `mean_oracle_se_two_node` | 0.0202 | mean oracle SE at the primary fits' fold rule pairs |
| `properties[interval_calibration/two_node__declared]:se_ratio` | 0.9786 | calibration SE ratio |
| `properties[interval_calibration/two_node__declared]:se_ratio_ci_lower` | 0.9601 | its interval, lower end |
| `properties[interval_calibration/two_node__declared]:se_ratio_ci_upper` | 0.9976 | its interval, upper end |
| `properties[interval_calibration/two_node__declared]:coverage` | 0.9386 | calibration coverage |
| `properties[interval_calibration/two_node__shrunken_se_control]:se_ratio` | 0.7829 | shrunken-SE control SE ratio |
| `properties[interval_calibration/two_node__noise_control]:se_ratio` | 0.7022 | noise control SE ratio |
| `properties[conservative_coverage/squared_w1__declared]:coverage` | 0.9400 | `squared_w1` coverage |
| `properties[conservative_coverage/squared_w1__declared]:coverage_ci_lower` | 0.9264 | its interval, lower end, against 0.92 |
| `properties[conservative_coverage/squared_w1__shrunken_coverage_control]:coverage` | 0.8646 | coverage control |
| `properties[conservative_coverage/squared_w1__shrunken_coverage_control]:coverage_ci_upper` | 0.8820 | its interval, upper end, against 0.92 |
| `properties[targeting_necessity/squared_w1__untargeted]:bias` | 0.0602 | untargeted plug-in bias, `squared_w1` |
| `properties[targeting_necessity/squared_w1__untargeted]:standardized_bias` | 2.5963 | in its own SDs |
| `properties[targeting_necessity/two_node__untargeted]:bias` | 0.0157 | untargeted plug-in bias, declared learners |
| `properties[targeting_necessity/two_node__untargeted]:standardized_bias` | 0.8014 | in its own SDs |
| `properties[targeting_necessity/squared_w1__targeted]:standardized_bias` | -0.0271 | targeted standardized bias, `squared_w1` |
| `properties[targeting_necessity/two_node__targeted]:standardized_bias` | -0.0327 | targeted standardized bias, declared learners |
| `properties[targeting_necessity/two_node__targeted]:targeting_displacement` | 0.7678 | the smaller displacement over the two labels, in positive-arm SDs |
| `properties[fold_locality/two_node__validation_rows]:bias` | 0.0900 | validation-row control bias |
| `properties[fold_locality/two_node__validation_rows]:standardized_bias` | 4.2984 | in its own SDs |
| `properties[fold_locality/two_node__fold_local]:standardized_bias` | -0.0202 | fold-local standardized bias |
| `properties[fold_locality/two_node__fold_local]:fold_locality_displacement` | 4.2717 | the fold-locality displacement |
| `bound:two_node_standard_error` | 0.0200 | the SD at the optimal pair over the root of n |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval |
| `margin:alpha` | 0.0500 | nominal size of the reported intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in error SDs |
| `margin:coverage_floor` | 0.9000 | validity floor the primary coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:shrunken_se_factor` | 0.8000 | factor the shrunken-SE control applies to each SE |
| `margin:targeting_displacement` | 0.5000 | smallest displacement the necessity pair must show |
| `margin:fold_locality_displacement` | 0.5000 | smallest displacement the fold-locality pair must show |
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
| `margin:subject_failures_allowed` | 12 | failed primary fits that bound allows |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | No maintained package fits a learned two-node regimen by cross-validated TMLE. The row rests on accuracy against the truth and on the theory properties |
| The interval targets the learned pairs | The interval covers $\sum_v (n_v / n)\, \Psi_{d_{nv}}(P_0)$, the stacked value of the fit's own learned rule pairs, not the value of the optimal pair |
| One law, one size, ten folds | The study fits n = 2,000 on one law with two binary nodes, censoring at each node, an end-of-study binary outcome and ten folds. Neither n nor the fold count is in the key, so another n, another fold count or another law reads supported and rests on the theory alone |
| Coverage rests on the treatment and censoring models | Those models are correct in every cell. The reported curve omits the $D_g$ term, so it is conservative when they are correct maximum-likelihood fits (Corollary 3 of van der Laan and Luedtke 2015). A misspecified treatment or censoring model was not measured |
| The outcome regressions are approximations | The node-2 learner is a correct model of each stratum in the Q-learning that learns the rules. The estimator's own outcome regressions pool the followers' strata with no treatment column, so the logistic and the cubic node-1 learner are approximations there. `squared_w1` adds a misspecified node-1 learner with $g$ correct, and claims coverage only |
| Targeting necessity | Targeting removes the plug-in's bias for `squared_w1`, a limit bias, and for the declared learners, a finite-sample bias at n = 2,000 whose limit is about 0 |
| A margin at both nodes | The law's limit rules have a margin. A learner whose limit rule has no margin gives a nonregular target whose interval under-covers, and no cell measures that regime |
| Compositions outside the row | Node bounds, `ratio_trim=`, a stopping rule, `treatment_probabilities=`, an uncensored panel and a continuous outcome with `q_bounds=` read experimental. More than two nodes, `repeats` above 1, `n_folds=1`, weights, clusters, a selecting node-1 learner and the refit surfaces refuse |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_regimen_ltmle/README.md)
gives the run command and the run form. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_regimen_ltmle/manifest.json)
records the seeds, the margins, the estimator configuration, the source hashes and the result
hashes. The [run log](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_regimen_ltmle/run.log)
records the commit, the clean tree, the runtime and the wall time. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_regimen_ltmle/replicates.parquet),
the [truth harness](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_regimen_ltmle/harness.csv.gz)
and the [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/learned_regimen_ltmle/properties.csv)
carry every published row.
