# Cross-fitted end-of-study longitudinal TMLE

This study validates the five-fold end-of-study recursion. Each outer fold fits and targets a
complete recursion on its training rows. The estimator stitches predictions and influence curves
only on held-out rows.

The canonical comparison uses R [`lmtp`](https://github.com/nt-williams/lmtp) 1.5.4 at commit
`f04a2b4`, which is the maintained package that implements a cross-fitted sequential regression.
R `ltmle`, the comparator the two ordinary longitudinal rows use, has no cross-fitting at all, so
it cannot witness this construction.

Agreement with R is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_longitudinal.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_longitudinal.py).

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets and folds | 1,600 censored panels, each with one exact five-fold assignment | the identical rows and the identical assignment, read from the panel |
| plans | never treat, always treat, and continue after initial treatment when L2 is positive | the same three, as shifted treatment columns |
| contrasts | always-minus-never and dynamic-minus-never | the same, from the difference of the two rowwise influence curves so covariance is preserved |
| treatment and censoring mechanisms | the generating probabilities from the law | the same probabilities, supplied as exact per-node density ratios |
| sequential regressions | quasibinomial GLMs fitted within each outer training set | `SL.glm`, fitted within the same training sets |
| targeting | a complete fold-specific backward recursion | the same, through `cf_tmle` and `theta_dr` |
| cumulative-g bounds | nonbinding | nonbinding, `.trim = 1` |
| intervals | pointwise 95% identity-scale Wald intervals | the same, from the returned influence curve |

Both implementations receive the mechanism from the law rather than estimating it. `lmtp` has no
`gform` argument, so the adapter substitutes exact per-node density ratios into it, the same way
it substitutes the fold assignment. The substitution is checked on every run against `lmtp`'s own
estimate: the zero pattern must agree cell for cell, and the cumulative ratio must track it.

That choice is what makes the comparison a comparison. An earlier version let each side estimate
the mechanism its own way, and the result measured two unrelated pipelines. `lmtp` fits its ratio
with `SL.glm`, whose linear logit cannot represent the exact classifier log-odds `-log g` for a
deterministic regime, so the ratio came out shrunken. A shrunken clever covariate under-targets
and understates the influence curve: that earlier probe measured coverage from 0.75 to 0.91 and an
SE ratio from 0.60 to 0.86. No committed artifact holds its rows, because supplying the mechanism
retired it. This row's own committed rows give `cleverly` coverage from 0.9431 to 0.9531 and an SE
ratio from 0.9839 to 1.0164. The sequential regressions stay misspecified on both sides, so
targeting still has work to do.

The exact-law structural test fixes the fold assignment and repeats every support point in each
fold. It checks every regimen mean and correlated contrast against the functional and Gateaux EIF.
The leakage tests also require held-out outcome predictions at both recursion nodes.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` cross-fitted LTMLE | -0.0019 to 0.0022 | 0.9431 | 0.9839 | pass |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` | -0.0018 to 0.0023 | 0.9437 | 0.9832 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` cross-fitted LTMLE | -0.0014 to 0.0028 | 0.9519 | 0.9982 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` | -0.0015 to 0.0026 | 0.9506 | 0.9999 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` cross-fitted LTMLE | -0.000849 to 0.0016 | 0.9531 | 1.0164 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` | -0.000871 to 0.0015 | 0.9525 | 1.0169 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` cross-fitted LTMLE | -0.0014 to 0.0019 | 0.9487 | 0.9991 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` | -0.0015 to 0.0017 | 0.9463 | 1.0009 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` cross-fitted LTMLE | -0.000419 to 0.0022 | 0.9531 | 1.0035 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` | -0.000644 to 0.0019 | 0.9506 | 1.0044 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -0.000088 | 0.0183 | 1.0066 | -0.000625 | 0.0042 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | 0.000124 | 0.0258 | 1.0135 | 0.0012 | 0.0048 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | 0.000026 | 0.0094 | 1.0082 | 0.000625 | 0.0049 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 0.000114 | 0.0300 | 1.0095 | 0.0025 | 0.0053 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | 0.000239 | 0.0788 | 1.0190 | 0.0025 | 0.0087 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_ltmle` | positive | five-fold end-of-study LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 1.1533 to 1.2016 | **fail** |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3463 to 0.3602 | pass |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0022 to 0.0015, margin 0.0063, SE ratio 1.0025 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0165 to 0.0206, margin 0.0069, SE ratio 1.0020 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0018 to 0.0022, margin 0.0067, SE ratio 0.9935 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0021 to 0.0017, margin 0.0063, SE ratio 1.0243 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0059 to 0.0027, margin 0.0144, SE ratio 1.0168 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0247 to -0.0162, margin 0.0143, SE ratio 0.6897 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0029 to 0.0057, margin 0.0145, SE ratio 1.0360 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0062 to 0.0030, margin 0.0154, SE ratio 0.6180 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9406 to 0.9634, SE ratio 0.9850 to 1.0657, empirical efficiency ratio 0.9526 to 1.0298, reported efficiency ratio 1.0118 to 1.0175 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8159 to 0.8552, SE ratio 0.6962 to 0.7490, empirical efficiency ratio 1.3550 to 1.4564, reported efficiency ratio 1.0117 to 1.0174 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8133 to 0.8529, SE ratio 0.6897 to 0.7457, empirical efficiency ratio 0.9528 to 1.0293, reported efficiency ratio 0.7082 to 0.7123 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9314 to 0.9559, SE ratio 0.9722 to 1.0488, empirical efficiency ratio 0.9736 to 1.0500, reported efficiency ratio 1.0185 to 1.0242 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8129 to 0.8525, SE ratio 0.6909 to 0.7443, empirical efficiency ratio 1.3725 to 1.4777, reported efficiency ratio 1.0185 to 1.0241 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8059 to 0.8461, SE ratio 0.6811 to 0.7344, empirical efficiency ratio 0.9741 to 1.0500, reported efficiency ratio 0.7130 to 0.7169 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9625, 0.9416 to 0.9776 | pass |
| `root_n_and_efficiency` | `dynamic__n_1000` | control | dynamic plan: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0016, coverage 0.9136 to 0.9585, SE ratio 0.9998 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000389, coverage 0.9179 to 0.9616, SE ratio 0.9987 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000035, coverage 0.9371 to 0.9747, SE ratio 1.0351 | pass |
| `root_n_and_efficiency` | `static__n_1000` | control | static plan: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0032, coverage 0.9165 to 0.9606, SE ratio 1.0049 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0038, coverage 0.9416 to 0.9776, SE ratio 1.0320 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000695, coverage 0.9252 to 0.9667, SE ratio 1.0013 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5716 to -0.4853 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5138 to -0.5073 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5566 to -0.4686 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5198 to -0.5131 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0020 to 0.0018, margin 0.0065 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0232 to 0.0274, margin 0.0070 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0041 to 0.0050, margin 0.0153 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0263 to -0.0173, margin 0.0151 | pass |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0425, 0.0263 to 0.0644 | pass |
<!-- /generated -->

The property study preserves the ordinary row's double-robustness, rate, efficiency, calibration,
null, power, and targeting instruments. It runs each positive estimator with five outer folds.

The overfitting pair uses identical nonlinear panels and a fully grown outcome tree. The positive
arm predicts held-out rows. The control predicts its training rows. The joint verdict requires an
SE ratio inside the declared band and a predeclared coverage gain over the control.

Read the direction of that ratio, not only the verdict. In-sample fitting understates the standard
error by a factor near three, at 0.3532. Cross-fitting removes the understatement and overshoots
it, at 1.1767. A noisy outcome model inflates the residual term of the influence curve, so a
conservative ratio is the expected direction here rather than an anomaly. The cell establishes
that cross-fitting restores honest inference. It does not establish calibration under a fully
grown tree, and the `interval_calibration` family, which does make a calibration claim, uses
correctly specified nuisances instead.

The gate no longer holds. The 99% interval reaches 1.201555 against a ceiling of 1.2000, so the
cell misses by 0.001555. The earlier row measured 1.196518 on a treatment-stratified first-node
split and recorded the margin as uncomfortable. The limitations table gives the attribution and
the direction.

The replication budget stays at 8,000 rather than the shared 400, because the
shared budget leaves the interval wider than the remaining margin. That budget was sized before
the first run of this study. Raising it now would buy a pass on an equivalence-shaped gate, which
more replications make easier rather than harder, so this study publishes the red cell instead.

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | paired replications |
| `n` | 2000 | observations per paired replication |
| `independent_tests_total` | 10 | implementation-estimand truth tests |
| `independent_tests_passed` | 10 | truth tests passing |
| `paired_tests_total` | 5 | paired estimand comparisons |
| `paired_tests_passed` | 5 | paired comparisons passing |
| `property_cells_total` | 32 | independent property cells |
| `property_cells_passed` | 30 | property cells passing |
| `max_standardized_bias` | 0.0438 | largest primary standardized bias |
| `min_coverage` | 0.9431 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9265 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9422 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0649 | highest primary SE-ratio endpoint |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:coverage` | 0.9760 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5081 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:se_ratio_ci_upper` | 1.201555 | cross-fitted tree SE-ratio upper endpoint |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:coverage_gain_ci_lower` | 0.4534 | lower bound for the paired coverage gain |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:replicates` | 8000 | paired overfitting replications |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
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
| `margin:minimum_power` | 0.8000 | power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope upper bound |
| `margin:excluded_slope` | -0.2500 | rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:shrunken_se_factor` | 0.7000 | deliberate SE mutation factor |
| `margin:targeting_displacement` | 0.2500 | least the fluctuation must move the estimate |
| `margin:overfit_se_floor` | 0.8500 | cross-fitted tree SE-ratio lower bound |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample tree SE-ratio upper bound |
| `margin:overfit_coverage_gain` | 0.1500 | minimum paired coverage gain |

## Limitations

| limitation | what it means for use |
| --- | --- |
| Agreement with `lmtp` is distributional, not numerical | The paired claim is that the mean difference sits inside the similarity margin and that `cleverly` is no worse. The two ordinary rows agree with R `ltmle` to solver precision because both run the identical regression; here the sequential regressions are still fitted differently, so per-replication estimates differ at statistical scale |
| One fixed five-fold assignment is studied | The row does not validate repeated folds or time-respecting splits |
| The cross-fit overfitting cell misses its ceiling | This row publishes under the `reporting` policy, which was declared before the run that measured this cell. The 99% SE-ratio interval reaches 1.201555 against a ceiling of 1.2000, so the cell misses by 0.001555. A controlled 2x2 over the same 8,000 registered seeds attributes the whole move to the first-node fold policy. The stratified arm gives 1.172543 and an upper endpoint of 1.196538, which reproduces the committed row to 2e-5 relative. The single-fold control gives 0.353193 under both policies, because it has no outer split for a policy to change. The breach is in the conservative direction. Coverage rose from 0.975375 to 0.976000. The cell still shows that cross-fitting restores honest inference under a fully grown tree. It does not show calibration under one. The [fold and outcome-scale rules](../cv-tmle.md#fold-and-outcome-scale-rules) record that the reviewed longitudinal theorem defines random near-balanced row folds, and that the theorem's targeting construction differs from the shipped fold-local recursion. The roadmap's Remediation section carries the open question this residual belongs to |
| Flexible learning is an independent property instrument | The paired comparison uses one GLM learner on each side. The tree pair validates held-out prediction behavior, not parity for learner-library selection |
| The row reports one terminal mean per plan | Survival curves, competing risks, and longitudinal MSM projections have different parameters |
| Inference is pointwise | The row does not validate simultaneous bands, bootstrap intervals, weights, or clustering |
| The mechanism is supplied rather than estimated | Both implementations receive the generating probabilities, so the paired row says nothing about mechanism estimation or about a severe practical-positivity violation. The property study's double-robustness cells cover misspecified mechanisms separately |

The causal interpretation requires consistency, sequential exchangeability, longitudinal
positivity, and conditionally independent censoring. Single-correct-nuisance cells establish
consistency only. Calibrated inference uses the cells where both nuisance sequences are correct.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle/README.md)
gives the regeneration commands. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/lmtp_ltmle/manifest.json)
records the seeds, the configuration, the pinned `lmtp` version and source commit, the digest of
every study module and reference source, and the artifact hashes.
