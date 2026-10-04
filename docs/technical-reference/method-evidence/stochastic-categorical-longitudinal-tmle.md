# Stochastic categorical longitudinal TMLE

This study validates ordinary longitudinal TMLE under known stochastic categorical policies. The
law has two nodes and three treatment levels. A policy node draws each unit's arm from a density
that is fixed before the fit. The
[known stochastic policies](../longitudinal-tmle.md#known-stochastic-policies) section states the
estimator, its conditions, and its exact-law evidence.

Pinned `lmtp` 1.5.4 takes one shifted value per unit. The adapter
`tests/canonical/lmtp_policy_adapter.R` therefore realises each policy exactly by copying every
unit four times. Copy $c$ takes its shifted arm from row $c$ of the node's allocation table, and
every policy probability is a multiple of one quarter.

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets | 2,000 samples of 2,000 rows from the two-node three-level law | the identical rows |
| folds | one fold, in sample | one fold |
| plans | `low` at both nodes; `mix`, a policy at both nodes; `taper`, `high` then a policy | the same plans; `mix` and `taper` through four copies of each unit |
| mechanism | the generating probabilities | the same, as the exact per-node ratio $q / g$ at the natural arm |
| node regressions | a quasibinomial GLM at the last node and least squares at the first | `SL.glm`: binomial at the last node and gaussian at the first |
| policy-node design | the covariate history, the current arm, and the arms of earlier policy nodes | the full history with the current arm replaced |
| intervals | pointwise 95% Wald from the influence curve | the unit mean of the per-copy `eif` |

For `mix` the two designs span the same columns and the learners match, so the pair is an
exactness witness. For `low` and `taper`, this package fits the label nodes on the units that
follow the plan, while `lmtp` pools over every arm. These pairs and the two contrasts therefore
differ by up to about $9 \times 10^{-3}$ per sample. They are read under the default paired
margins, as in the [categorical study](ordinary-categorical-longitudinal-tmle.md).

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the three-level law with known stochastic policies at both nodes | `ate_regimen[mix vs low]` | difference in mean outcome between the plans "draw each arm from a known policy at both times" against "assign the low arm at both times" | `cleverly` ordinary LTMLE under known stochastic categorical policies | -0.0023 to 0.0018 | 0.9360 | 0.9691 | pass |
| the three-level law with known stochastic policies at both nodes | `ate_regimen[mix vs low]` | difference in mean outcome between the plans "draw each arm from a known policy at both times" against "assign the low arm at both times" | R `lmtp` | -0.0018 to 0.0024 | 0.9385 | 0.9791 | pass |
| the three-level law with known stochastic policies at both nodes | `ate_regimen[taper vs low]` | difference in mean outcome between the plans "assign the high arm first, then draw from a known policy" against "assign the low arm at both times" | `cleverly` ordinary LTMLE under known stochastic categorical policies | -0.0037 to 0.0011 | 0.9480 | 0.9759 | pass |
| the three-level law with known stochastic policies at both nodes | `ate_regimen[taper vs low]` | difference in mean outcome between the plans "assign the high arm first, then draw from a known policy" against "assign the low arm at both times" | R `lmtp` | -0.0027 to 0.0023 | 0.9430 | 0.9825 | pass |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | `cleverly` ordinary LTMLE under known stochastic categorical policies | -0.0021 to 0.0018 | 0.9355 | 0.9731 | pass |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | R `lmtp` | -0.0028 to 0.0013 | 0.9385 | 0.9857 | pass |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[mix]` | mean outcome under the plan draw each arm from a known policy at both times | `cleverly` ordinary LTMLE under known stochastic categorical policies | -0.0012 to 0.000400 | 0.9520 | 0.9950 | pass |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[mix]` | mean outcome under the plan draw each arm from a known policy at both times | R `lmtp` | -0.0012 to 0.000400 | 0.9520 | 0.9950 | pass |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[taper]` | mean outcome under the plan assign the high arm first, then draw from a known policy | `cleverly` ordinary LTMLE under known stochastic categorical policies | -0.0028 to -0.000064 | 0.9450 | 0.9820 | pass |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[taper]` | mean outcome under the plan assign the high arm first, then draw from a known policy | R `lmtp` | -0.0023 to 0.000400 | 0.9500 | 0.9870 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| the three-level law with known stochastic policies at both nodes | `ate_regimen[mix vs low]` | difference in mean outcome between the plans "draw each arm from a known policy at both times" against "assign the low arm at both times" | -0.000593 | 0.1109 | 0.9860 | -0.0025 | 0.0144 vs 0.0500 | equivalent |
| the three-level law with known stochastic policies at both nodes | `ate_regimen[taper vs low]` | difference in mean outcome between the plans "assign the high arm first, then draw from a known policy" against "assign the low arm at both times" | -0.0011 | 0.1691 | 0.9884 | 0.0050 | 0.0129 vs 0.0500 | equivalent |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[low]` | mean outcome under the plan assign the low arm at both times | 0.000593 | 0.1136 | 0.9776 | -0.0030 | 0.0148 vs 0.0500 | equivalent |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[mix]` | mean outcome under the plan draw each arm from a known policy at both times | -4.892e-11 | 2.285e-08 | 1.0000 | 0 | 3.839e-10 vs 0.0500 | equivalent |
| the three-level law with known stochastic policies at both nodes | `ey_regimen[taper]` | mean outcome under the plan assign the high arm first, then draw from a known policy | -0.000478 | 0.1343 | 1.0195 | -0.0050 | 0.0069 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_policy_ltmle` | positive | five-fold policy LTMLE with fully grown outcome trees | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9764 to 0.9948 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.2722 to 0.2772 | pass |
| `double_robustness` | `mix__both_correct` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0018 to 0.0032, margin 0.0084, SE ratio 0.9957 | pass |
| `double_robustness` | `mix__both_wrong` | control | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0298 to 0.0349, margin 0.0084, SE ratio 0.9604 | pass |
| `double_robustness` | `mix__mechanism_correct` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0030 to 0.0023, margin 0.0088, SE ratio 1.0065 | pass |
| `double_robustness` | `mix__outcome_correct` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0022 to 0.0029, margin 0.0086, SE ratio 0.9063 | pass |
| `interval_calibration` | `mix__correctly_specified` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9364 to 0.9550, SE ratio 0.9672 to 1.0281, empirical efficiency ratio 0.9630 to 1.0231, reported efficiency ratio 0.9879 to 0.9913 | pass |
| `interval_calibration` | `mix__noise_control` | control | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8173 to 0.8479, SE ratio 0.6828 to 0.7227, empirical efficiency ratio 1.3693 to 1.4490, reported efficiency ratio 0.9878 to 0.9913 | pass |
| `interval_calibration` | `mix__shrunken_se_control` | control | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8175 to 0.8482, SE ratio 0.6781 to 0.7193, empirical efficiency ratio 0.9632 to 1.0211, reported efficiency ratio 0.6915 to 0.6939 | pass |
| `interval_calibration` | `msm_policy__correctly_specified` | positive | the dose coefficient of a working model over low, mix and taper: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9340 to 0.9530, SE ratio 0.9547 to 1.0127, empirical efficiency ratio 0.9780 to 1.0368, reported efficiency ratio 0.9889 to 0.9912 | pass |
| `interval_calibration` | `msm_policy__noise_control` | control | the dose coefficient of a working model over low, mix and taper: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8106 to 0.8417, SE ratio 0.6750 to 0.7154, empirical efficiency ratio 1.3842 to 1.4663, reported efficiency ratio 0.9889 to 0.9912 | pass |
| `interval_calibration` | `msm_policy__shrunken_se_control` | control | the dose coefficient of a working model over low, mix and taper: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8093 to 0.8404, SE ratio 0.6686 to 0.7088, empirical efficiency ratio 0.9776 to 1.0362, reported efficiency ratio 0.6922 to 0.6938 | pass |
| `interval_calibration` | `policy_risk_h2__correctly_specified` | positive | the cumulative risk at t = 2 of a known policy on the survival law: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9273 to 0.9472, SE ratio 0.9333 to 0.9871, empirical efficiency ratio 1.0029 to 1.0604, reported efficiency ratio 0.9876 to 0.9920 | pass |
| `interval_calibration` | `policy_risk_h2__noise_control` | control | the cumulative risk at t = 2 of a known policy on the survival law: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.7930 to 0.8252, SE ratio 0.6658 to 0.7037, empirical efficiency ratio 1.4067 to 1.4864, reported efficiency ratio 0.9876 to 0.9920 | pass |
| `interval_calibration` | `policy_risk_h2__shrunken_se_control` | control | the cumulative risk at t = 2 of a known policy on the survival law: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.7912 to 0.8235, SE ratio 0.6528 to 0.6913, empirical efficiency ratio 1.0026 to 1.0607, reported efficiency ratio 0.6913 to 0.6944 | pass |
| `policy_necessity` | `mix__declared_policy` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the recursion integrates over the declared policy at each node | bias interval inside the equivalence margin | bias -0.0017 to 0.0033, margin 0.0084 | pass |
| `policy_necessity` | `mix__uniform_control` | control | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the same fit replaces each declared policy with a uniform distribution | bias interval must fall entirely outside the margin | bias 0.0160 to 0.0208, margin 0.0079 | pass |
| `power` | `mix__alternative` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.5575, 0.5114 to 0.6029 | **fail** |
| `randomizer_projection` | `mix__integrated` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the integrated estimator, with the policy-weighted recursion and the ratio q / g | bias interval inside the equivalence margin; the sd ratio bound below one | bias -0.000562 to 0.0013, margin 0.0032, sd ratio 0.3343, 99% upper 0.3563 | pass |
| `randomizer_projection` | `mix__recorded_randomizer` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the same sample with a recorded uniform randomizer per node and the rule that reads it | bias interval inside the equivalence margin | bias -0.0028 to 0.0028, margin 0.0095, sd ratio 0.3343, 99% upper 0.3563 | pass |
| `root_n_and_efficiency` | `mix__n_2000` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0015, coverage 0.9282 to 0.9688, SE ratio 0.9981 | pass |
| `root_n_and_efficiency` | `mix__n_500` | control | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: bias, coverage and SE calibration at n = 500 | coverage interval lies below nominal or clears the declared floor | bias -0.0030, coverage 0.9194 to 0.9627, SE ratio 0.9611 | pass |
| `root_n_and_efficiency` | `mix__n_8000` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000889, coverage 0.9121 to 0.9575, SE ratio 0.9813 | pass |
| `root_n_rate` | `mix__empirical_sd` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5297 to -0.4623 | pass |
| `root_n_rate` | `mix__reported_se` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4912 to -0.4855 | pass |
| `simultaneous_coverage` | `primary__pointwise_joint_control` | control | the five primary estimands of the known-policy study: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.8210 to 0.8634 | pass |
| `simultaneous_coverage` | `primary__simultaneous_band` | positive | the five primary estimands of the known-policy study: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9255 to 0.9533 | pass |
| `targeting_necessity` | `mix__targeted` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0021 to 0.0032, margin 0.0088 | pass |
| `targeting_necessity` | `mix__untargeted` | control | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0640 to 0.0688, margin 0.0080 | pass |
| `type_i_error` | `mix__sharp_null` | positive | the known policy mix, as its contrast against low, and as its mean in randomizer_projection: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0425, 0.0263 to 0.0644 | pass |
<!-- /generated -->

Every end-of-study family uses the three-level law with the declared policies. Its truths come
from the finite-support g-formula in `tests/discrete_law_longitudinal_policy.py`. The survival
calibration cell (`policy_risk_h2`) uses the binary censored survival law with a known policy at
both nodes. The working-model cell (`msm_policy`) uses an identity-link projection over `low`,
`mix` and `taper`, and its truth is the projection of the exact policy means.

`policy_necessity` scores a fit with uniform policies against the declared truth.
`randomizer_projection` fits the same sample twice. The integrated estimator is the shipped one.
The recorded estimator reads a fresh uniform randomizer at each node, which is Theorem 3 as
stated. The one-sided 99% bootstrap upper bound of the ratio of their spreads must lie below one.
These two cells read `ey_regimen[mix]`. Every other `mix` cell reads `ate_regimen[mix vs low]`.

The `power` cell is red. The contrast is -0.0488, and its reported standard error at n = 4,000
is about 0.0237. A two-sided 5% test therefore has power near 0.54 at the declared size, and the
cell measured 0.5575. Its bias is inside the margin and its coverage is 0.9475.

The study was
declared `gated`. By its declared red-cell route it was re-registered under `reporting`, with the
owner row `F1-power-design`, and the run repeated with no other change. The repeat reproduced every
artifact of the first run.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2000 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_passed` | 10 | truth tests passing |
| `independent_tests_total` | 10 | truth tests reported |
| `paired_tests_passed` | 5 | paired comparisons passing |
| `paired_tests_total` | 5 | paired comparisons reported |
| `property_cells_passed` | 29 | property cells passing |
| `property_cells_total` | 30 | property cells reported |
| `max_standardized_bias` | 0.0603 | largest primary standardized bias |
| `min_coverage` | 0.9355 | lowest primary coverage |
| `max_margin_utilization` | 0.1691 | largest paired similarity-margin share |
| `properties[crossfit_overfitting/cross_fitted_policy_ltmle]:coverage` | 0.9404 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.4041 | in-sample tree coverage |
| `properties[randomizer_projection/mix__integrated]:projection_sd_ratio` | 0.3343 | spread ratio, integrated over recorded |
| `properties[randomizer_projection/mix__integrated]:projection_sd_ratio_ci_upper` | 0.3563 | one-sided 99% upper bound of the spread ratio |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:alpha` | 0.0500 | test size |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:targeting_displacement` | 0.1000 | minimum targeting displacement |
| `margin:policy_displacement` | 0.1000 | minimum policy displacement |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound |
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

## Limits

| limit | what it means for use |
| --- | --- |
| one law, two nodes and three levels | the evidence covers one finite law with policies fixed before the run |
| the power cell is red by design | its declared size gives a power near 0.54. The `F1-power-design` owner holds it, and the `type_i_error` cell of the same law passes |
| no study cell for weights, clusters, competing risks, a continuous outcome or fold repeats | these compositions have the exact-law and end-to-end evidence of the [known stochastic policies](../longitudinal-tmle.md#known-stochastic-policies) section only |
| `low`, `taper` and both contrasts pair to about $9 \times 10^{-3}$ only | `lmtp` pools label nodes over every arm, and this package fits them on the followers. Only `ey_regimen[mix]` is an exactness pair |
| the comparator copies each unit four times | the pairing needs every policy probability on a grid of one quarter. A policy off that grid has no exact `lmtp` pair |
| the mechanism is supplied | both implementations receive the generating probabilities. The double-robustness cells cover misspecified mechanisms |
| a policy must be known and fixed | a policy learned from the sample, a policy that reads the natural treatment, and a policy that depends on the law are outside this evidence |
| learner caution | the tree pair validates held-out prediction with flexible learners, not learner-library selection |
