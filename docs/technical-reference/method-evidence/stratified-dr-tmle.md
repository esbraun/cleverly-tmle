# Stratified DR-TMLE

This study validates `DRTMLE(guard=("Q", "g"))` with baseline strata, `strata=["V"]`. The fit
solves every equation of the doubly robust alternation with one block per stratum. It fits every
reduced regression inside each stratum. Each stratum's estimate is then the DR-TMLE of the law
given $V = s$. [DR-TMLE estimands](../dr-tmle/supported-estimands.md#baseline-strata) states the
construction.

## What was tested

| setting | declaration |
| --- | --- |
| law | the complete-data binary law of Benkeser et al. (2017), Section 5.1, with a stratum $V \in \{0, 1, 2\}$ (probabilities 0.5, 0.3 and 0.2) that shifts both intercepts and the treatment effect. `tests/studies/stratified_alternating_law.py` computes each truth by one-dimensional quadrature |
| nuisances | a logistic outcome GLM on $(A, W_1, W_2, W_1 W_2, V, A V)$ and a logistic treatment GLM on $(W_1, W_2, W_1 W_2, V)$, both correct. A wrong GLM omits $W_1 W_2$ |
| fit | cross-fitted on ten folds, `reduction="univariate"`, Gaussian and binomial reduced GLMs, `g_bounds=(0.01, 0.99)` |
| primary scenario | `stratified_both_correct`: the two arm means and the ATE in each stratum, $n = 2000$, 800 replications |
| Monte Carlo inference | exact 99% intervals around every rate, and 99% intervals around every primary endpoint |

The comparator is R `drtmle` 1.1.2 at commit `538a3a2`. `drtmle` has no stratified form, so the
runner calls it once per stratum subset. Both sides read the same rows, the same ten-fold
assignment, and the subject's initial nuisance arrays. Inside a stratum, `drtmle` fits its
reduced regressions on the stratum's rows, which is the construction the subject uses.

| difference | effect |
| --- | --- |
| the subject's outer loop stops when every stratum's equations meet the stop rule. R `drtmle` stops on each subset alone | a stratum can take more rounds in the subject than in R |
| the marginal estimate of a stratified fit reduces on $(g_n, V)$. A marginal `drtmle` call reduces on $g_n$ alone | the three marginal names are not paired. The property cells measure the marginal ATE against its exact truth |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=0]` | average treatment effect, in stratum V=0 | `cleverly` DR-TMLE with baseline strata | -0.0012 to 0.0046 | 0.9587 | 1.0690 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=0]` | average treatment effect, in stratum V=0 | R `drtmle`, once per stratum subset | -0.0015 to 0.0043 | 0.9525 | 1.0630 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=1]` | average treatment effect, in stratum V=1 | `cleverly` DR-TMLE with baseline strata | -0.000489 to 0.0072 | 0.9450 | 1.0096 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=1]` | average treatment effect, in stratum V=1 | R `drtmle`, once per stratum subset | -0.0012 to 0.0066 | 0.9437 | 1.0057 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=2]` | average treatment effect, in stratum V=2 | `cleverly` DR-TMLE with baseline strata | 0.0022 to 0.0123 | 0.9287 | 0.9848 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=2]` | average treatment effect, in stratum V=2 | R `drtmle`, once per stratum subset | 0.0021 to 0.0122 | 0.9387 | 0.9727 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=0]` | counterfactual mean under treatment arm '0', in stratum V=0 | `cleverly` DR-TMLE with baseline strata | -0.0038 to 0.0012 | 0.9537 | 1.0364 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=0]` | counterfactual mean under treatment arm '0', in stratum V=0 | R `drtmle`, once per stratum subset | -0.0035 to 0.0014 | 0.9450 | 1.0302 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=1]` | counterfactual mean under treatment arm '0', in stratum V=1 | `cleverly` DR-TMLE with baseline strata | -0.0054 to 0.000269 | 0.9500 | 0.9912 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=1]` | counterfactual mean under treatment arm '0', in stratum V=1 | R `drtmle`, once per stratum subset | -0.0053 to 0.000332 | 0.9513 | 0.9952 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=2]` | counterfactual mean under treatment arm '0', in stratum V=2 | `cleverly` DR-TMLE with baseline strata | -0.0033 to 0.0022 | 0.9587 | 1.0685 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=2]` | counterfactual mean under treatment arm '0', in stratum V=2 | R `drtmle`, once per stratum subset | -0.0032 to 0.0023 | 0.9575 | 1.0714 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=0]` | counterfactual mean under treatment arm '1', in stratum V=0 | `cleverly` DR-TMLE with baseline strata | -0.0015 to 0.0023 | 0.9575 | 1.0317 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=0]` | counterfactual mean under treatment arm '1', in stratum V=0 | R `drtmle`, once per stratum subset | -0.0016 to 0.0022 | 0.9587 | 1.0330 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=1]` | counterfactual mean under treatment arm '1', in stratum V=1 | `cleverly` DR-TMLE with baseline strata | -0.0021 to 0.0037 | 0.9587 | 1.0100 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=1]` | counterfactual mean under treatment arm '1', in stratum V=1 | R `drtmle`, once per stratum subset | -0.0027 to 0.0032 | 0.9537 | 1.0052 | pass |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=2]` | counterfactual mean under treatment arm '1', in stratum V=2 | `cleverly` DR-TMLE with baseline strata | 0.0023 to 0.0112 | 0.9000 | 0.9388 | **fail** |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=2]` | counterfactual mean under treatment arm '1', in stratum V=2 | R `drtmle`, once per stratum subset | 0.0022 to 0.0111 | 0.8925 | 0.9205 | **fail** |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=0]` | average treatment effect, in stratum V=0 | 0.000339 | 0.0715 | 1.0052 | 0.0062 | 0.0121 vs 0.0500 | superior |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=1]` | average treatment effect, in stratum V=1 | 0.000652 | 0.1025 | 1.0093 | 0.0012 | 0.0118 vs 0.0500 | equivalent |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ate[V=2]` | average treatment effect, in stratum V=2 | 0.000169 | 0.0203 | 1.0251 | -0.0100 | 0.0404 vs 0.0500 | equivalent |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=0]` | counterfactual mean under treatment arm '0', in stratum V=0 | -0.000248 | 0.0608 | 1.0073 | 0.0088 | 0.0135 vs 0.0500 | equivalent |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=1]` | counterfactual mean under treatment arm '0', in stratum V=1 | -0.000066 | 0.0141 | 1.0146 | -0.0013 | 0.0112 vs 0.0500 | equivalent |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[0][V=2]` | counterfactual mean under treatment arm '0', in stratum V=2 | -0.000126 | 0.0276 | 1.0098 | 0.0012 | 0.0105 vs 0.0500 | equivalent |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=0]` | counterfactual mean under treatment arm '1', in stratum V=0 | 0.000091 | 0.0292 | 1.0050 | -0.0012 | 0.0047 vs 0.0500 | equivalent |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=1]` | counterfactual mean under treatment arm '1', in stratum V=1 | 0.000587 | 0.1212 | 1.0089 | 0.0050 | 0.0123 vs 0.0500 | equivalent |
| Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances | `ey[1][V=2]` | counterfactual mean under treatment arm '1', in stratum V=2 | 0.000043 | 0.0058 | 1.0238 | 0.0075 | 0.0277 vs 0.0500 | equivalent |
<!-- /generated -->

## Repeated-sampling properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `marginal_ate__both_correct` | positive | average treatment effect over every stratum, from the stratified fit: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0013 to 0.0042, margin 0.0062, SE ratio 0.9629 | pass |
| `double_robustness` | `marginal_ate__both_wrong` | control | average treatment effect over every stratum, from the stratified fit: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.2197 to 0.2237, margin 0.0055, SE ratio 2.4268 | pass |
| `double_robustness` | `marginal_ate__outcome_correct` | positive | average treatment effect over every stratum, from the stratified fit: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000440 to 0.0048, margin 0.0059, SE ratio 1.1031 | pass |
| `double_robustness` | `marginal_ate__treatment_correct` | positive | average treatment effect over every stratum, from the stratified fit: only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0136 to 0.0185, margin 0.0067, SE ratio 0.9902 | **fail** |
| `double_robustness` | `v0_ate__both_correct` | positive | average treatment effect in stratum V = 0: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0011 to 0.0029, margin 0.0087, SE ratio 0.9659 | pass |
| `double_robustness` | `v0_ate__both_wrong` | control | average treatment effect in stratum V = 0: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.2267 to 0.2323, margin 0.0077, SE ratio 2.0577 | pass |
| `double_robustness` | `v0_ate__outcome_correct` | positive | average treatment effect in stratum V = 0: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000165 to 0.0060, margin 0.0084, SE ratio 1.0270 | pass |
| `double_robustness` | `v0_ate__treatment_correct` | positive | average treatment effect in stratum V = 0: only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0094 to 0.0162, margin 0.0093, SE ratio 0.9505 | **fail** |
| `double_robustness` | `v1_ate__both_correct` | positive | average treatment effect in stratum V = 1: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000831 to 0.0059, margin 0.0109, SE ratio 0.9792 | pass |
| `double_robustness` | `v1_ate__both_wrong` | control | average treatment effect in stratum V = 1: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.2218 to 0.2289, margin 0.0097, SE ratio 1.8668 | pass |
| `double_robustness` | `v1_ate__outcome_correct` | positive | average treatment effect in stratum V = 1: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0015 to 0.0061, margin 0.0104, SE ratio 1.0494 | pass |
| `double_robustness` | `v1_ate__treatment_correct` | positive | average treatment effect in stratum V = 1: only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0123 to 0.0214, margin 0.0124, SE ratio 0.9364 | **fail** |
| `double_robustness` | `v2_ate__both_correct` | positive | average treatment effect in stratum V = 2: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0031 to 0.0097, margin 0.0143, SE ratio 0.9544 | pass |
| `double_robustness` | `v2_ate__both_wrong` | control | average treatment effect in stratum V = 2: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.1924 to 0.2010, margin 0.0118, SE ratio 1.7752 | pass |
| `double_robustness` | `v2_ate__outcome_correct` | positive | average treatment effect in stratum V = 2: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0027 to 0.0071, margin 0.0134, SE ratio 1.2492 | pass |
| `double_robustness` | `v2_ate__treatment_correct` | positive | average treatment effect in stratum V = 2: only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.0179 to 0.0286, margin 0.0147, SE ratio 1.0627 | **fail** |
| `interval_calibration` | `marginal_ate__correctly_specified` | positive | average treatment effect over every stratum, from the stratified fit: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9355 to 0.9613, SE ratio 0.9735 to 1.0565, empirical efficiency ratio 0.9755 to 1.0587, reported efficiency ratio 1.0259 to 1.0383 | pass |
| `interval_calibration` | `marginal_ate__noise_control` | control | average treatment effect over every stratum, from the stratified fit: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8157 to 0.8586, SE ratio 0.6843 to 0.7457, empirical efficiency ratio 1.3832 to 1.5074, reported efficiency ratio 1.0258 to 1.0387 | pass |
| `interval_calibration` | `marginal_ate__shrunken_se_control` | control | average treatment effect over every stratum, from the stratified fit: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8141 to 0.8572, SE ratio 0.6818 to 0.7407, empirical efficiency ratio 0.9751 to 1.0585, reported efficiency ratio 0.7181 to 0.7269 | pass |
| `interval_calibration` | `v0_ate__correctly_specified` | positive | average treatment effect in stratum V = 0: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9305 to 0.9573, SE ratio 0.9682 to 1.0534, empirical efficiency ratio 0.9672 to 1.0520, reported efficiency ratio 1.0144 to 1.0226 | pass |
| `interval_calibration` | `v0_ate__noise_control` | control | average treatment effect in stratum V = 0: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8147 to 0.8577, SE ratio 0.6875 to 0.7483, empirical efficiency ratio 1.3617 to 1.4803, reported efficiency ratio 1.0145 to 1.0227 | pass |
| `interval_calibration` | `v0_ate__shrunken_se_control` | control | average treatment effect in stratum V = 0: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8115 to 0.8549, SE ratio 0.6766 to 0.7378, empirical efficiency ratio 0.9663 to 1.0532, reported efficiency ratio 0.7103 to 0.7160 | pass |
| `interval_calibration` | `v1_ate__correctly_specified` | positive | average treatment effect in stratum V = 1: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9266 to 0.9542, SE ratio 0.9471 to 1.0298, empirical efficiency ratio 0.9951 to 1.0817, reported efficiency ratio 1.0201 to 1.0291 | pass |
| `interval_calibration` | `v1_ate__noise_control` | control | average treatment effect in stratum V = 1: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8131 to 0.8563, SE ratio 0.6884 to 0.7467, empirical efficiency ratio 1.3734 to 1.4887, reported efficiency ratio 1.0202 to 1.0291 | pass |
| `interval_calibration` | `v1_ate__shrunken_se_control` | control | average treatment effect in stratum V = 1: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8005 to 0.8449, SE ratio 0.6633 to 0.7202, empirical efficiency ratio 0.9966 to 1.0809, reported efficiency ratio 0.7141 to 0.7204 | pass |
| `interval_calibration` | `v2_ate__correctly_specified` | positive | average treatment effect in stratum V = 2: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9069 to 0.9380, SE ratio 0.9294 to 1.0112, empirical efficiency ratio 1.0348 to 1.1237, reported efficiency ratio 1.0296 to 1.0653 | **fail** |
| `interval_calibration` | `v2_ate__noise_control` | control | average treatment effect in stratum V = 2: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7984 to 0.8430, SE ratio 0.6702 to 0.7304, empirical efficiency ratio 1.4368 to 1.5556, reported efficiency ratio 1.0294 to 1.0659 | pass |
| `interval_calibration` | `v2_ate__shrunken_se_control` | control | average treatment effect in stratum V = 2: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7843 to 0.8302, SE ratio 0.6501 to 0.7072, empirical efficiency ratio 1.0361 to 1.1246, reported efficiency ratio 0.7207 to 0.7458 | pass |
<!-- /generated -->

The property grid calibrates the marginal ATE and each stratum ATE of the both-correct fit, each
with a shrunken-SE control and a noise control. The robustness family fits the same targets under
four nuisance configurations. The marginal cells read the same fits as the stratum cells.

Seven results are red. The four `treatment_correct` robustness cells carry a positive bias of
0.34 to 0.60 of their spread. The owner `X8-drtmle-one-sided-bias` in the
[roadmap](../../roadmap.md#red-cell-owners) records a declared diagnostic.

Inside each stratum, the shipped unstratified `DRTMLE` and R `drtmle`, handed the same initial
arrays, carry a positive bias too. The direction is shared in every stratum, and the size mostly is. In stratum 2 the
package's bias, 0.0237, exceeds R's, 0.0157, on the same arrays: the paired difference is 0.0080
with a standard error of 0.0018. The shipped unstratified fit on the stratum's rows has the
package's bias there to 2e-5, so the excess belongs to `DRTMLE` at about 400 rows and not to the
strata. This is the one-sided bias that `RM18-one-sided-bias` records for the unstratified study
on the same law.

The marginal estimate is the mixture of the stratum estimates, so it inherits their bias. Its
bias is 0.56 of its spread at n = 2,000 (99% interval 0.43 to 0.69), and 0.43 at n = 8,000 (0.25
to 0.61). The two intervals overlap, so the diagnostic does not establish that the bias contracts
faster than the spread.

The other three reds sit in stratum 2, which holds about 400 of the 2,000 rows. The two truth rows
of `ey[1][V=2]` cover 0.900 and 0.8925 in the two implementations on the same draws. The
`v2_ate__correctly_specified` cell has an SE-ratio interval that ends at 0.9294 against a floor of
0.93. The owner `X8-drtmle-small-stratum` records the reading. The study was declared under
`reporting`.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | paired replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 16 | truth tests passing |
| `independent_tests_total` | 18 | truth tests reported |
| `paired_tests_passed` | 9 | paired comparisons passing |
| `paired_tests_total` | 9 | paired comparisons reported |
| `property_cells_passed` | 23 | property cells passing |
| `property_cells_total` | 28 | property cells reported |
| `max_standardized_bias` | 0.1375 | largest primary standardized bias |
| `min_coverage` | 0.8925 | lowest primary coverage |
| `max_margin_utilization` | 0.1212 | largest paired similarity-margin share |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:efficiency_ratio_lower` | 0.9000 | efficiency-ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | efficiency-ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `bound:marginal_ate_standard_error` | 0.0233 | efficiency-bound standard error by quadrature at primary n |
| `bound:v0_ate_standard_error` | 0.0330 | efficiency-bound standard error by quadrature at primary n |
| `bound:v1_ate_standard_error` | 0.0417 | efficiency-bound standard error by quadrature at primary n |
| `bound:v2_ate_standard_error` | 0.0526 | efficiency-bound standard error by quadrature at primary n |

## Limits

- The study covers one law with one three-level stratum and a complete binary outcome.
- The study fits only the univariate reduction at `guard=("Q", "g")`. The exact-law tests in
  `tests/unit/test_stratified_drtmle_exact.py` cover the other guards, the bivariate reduction,
  three arms, and the missing-outcome and composite routes.
- The marginal estimate has no paired comparator.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_stratified/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_stratified/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_stratified/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_stratified/equivalence.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/drtmle_stratified/properties.csv)
carry the protocol, provenance, and every published row.
