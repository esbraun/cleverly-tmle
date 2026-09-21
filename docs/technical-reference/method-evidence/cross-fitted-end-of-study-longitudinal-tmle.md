# Cross-fitted end-of-study longitudinal TMLE

This study validates the five-fold end-of-study recursion. Each outer fold runs an untargeted
backward regression sequence on its training rows. The held-out predictions of the folds form one
out-of-fold initial estimate per node. One pooled fluctuation per node then targets that estimate
over every follower. The [cross-fitting section](../longitudinal-tmle.md#cross-fitting-the-recursion)
of the technical reference gives the construction and its source.

The canonical comparison uses R [`lmtp`](https://github.com/nt-williams/lmtp) 1.5.4 at commit
`f04a2b4`, which is the maintained package that implements a cross-fitted sequential regression.
R `ltmle`, the comparator the two ordinary longitudinal rows use, has no cross-fitting at all, so
it cannot witness this construction.

The pinned comparator runs a different construction. `lmtp` 1.5.4 fits each fold's fluctuation on
that fold's training rows, and it carries the targeted prediction into the fold's next regression.
Upstream `lmtp` commit
[`9996b04`](https://github.com/nt-williams/lmtp/commit/9996b04dcbb3ae0b1ef8862097c36d95e9f2fcf9)
later classified that behavior as a bug because the EIF was not mean zero. The patch moves the
fluctuation to validation rows and adds a mean-EIF test. Neither version runs the pooled update.
Each paired margin stays as registered. The paired verdicts therefore compare two constructions,
and they do not validate either fold-local update.

Agreement with R is secondary to the finite-support functional and Gateaux EIF in
[`tests/discrete_law_longitudinal.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_longitudinal.py).

## What was compared

| setting | `cleverly` | R `lmtp` |
| --- | --- | --- |
| datasets and folds | 1,600 censored panels, each with one exact five-fold assignment | the identical rows and the identical assignment, read from the panel |
| plans | never treat, always treat, and continue after initial treatment when L2 is positive | the same three, as shifted treatment columns |
| contrasts | always-minus-never and dynamic-minus-never | the same, from the difference of the two rowwise influence curves so covariance is preserved |
| treatment and censoring mechanisms | the generating probabilities from the law | the same probabilities, supplied as exact per-node density ratios |
| sequential regressions | untargeted quasibinomial GLMs fitted within each outer training set | `SL.glm`, fitted within the same training sets |
| targeting | one pooled fluctuation per node over every follower, with the out-of-fold predictions as offset | a fluctuation on each fold's training rows, carried into that fold's next regression, through `cf_tmle` and `theta_dr` |
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
retired it. This row's own committed rows give `cleverly` coverage from 0.9431 to 0.9537 and an SE
ratio from 0.9797 to 1.0130. The sequential regressions stay misspecified on both sides, so
targeting still has work to do.

The exact-law structural test fixes the fold assignment and repeats every support point in each
fold. It checks every regimen mean and correlated contrast against the functional and Gateaux EIF.
The leakage tests also require held-out outcome predictions at both recursion nodes.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | `cleverly` cross-fitted LTMLE | -0.0020 to 0.0021 | 0.9431 | 0.9797 | pass |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | R `lmtp` | -0.0018 to 0.0023 | 0.9437 | 0.9832 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | `cleverly` cross-fitted LTMLE | -0.0017 to 0.0025 | 0.9537 | 0.9954 | pass |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | R `lmtp` | -0.0015 to 0.0026 | 0.9506 | 0.9999 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | `cleverly` cross-fitted LTMLE | -0.000892 to 0.0015 | 0.9525 | 1.0130 | pass |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | R `lmtp` | -0.000871 to 0.0015 | 0.9525 | 1.0169 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | `cleverly` cross-fitted LTMLE | -0.0014 to 0.0019 | 0.9481 | 0.9955 | pass |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | R `lmtp` | -0.0015 to 0.0017 | 0.9463 | 1.0009 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | `cleverly` cross-fitted LTMLE | -0.000659 to 0.0020 | 0.9500 | 1.0010 | pass |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | R `lmtp` | -0.000644 to 0.0019 | 0.9506 | 1.0044 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-time-point law with monotone censoring | `ate_regimen[always vs never]` | difference in mean outcome between the plans "treat at both times" against "treat at neither time" | -0.000155 | 0.0323 | 1.0117 | -0.000625 | 0.0047 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ate_regimen[treat then continue if l2 positive vs never]` | difference in mean outcome between the plans "treat, then continue only if L2 is positive" against "treat at neither time" | -0.000140 | 0.0289 | 1.0164 | 0.0031 | 0.0053 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[always]` | mean outcome under the plan treat at both times | -0.000014 | 0.0049 | 1.0120 | 0 | 0.0110 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[never]` | mean outcome under the plan treat at neither time | 0.000141 | 0.0371 | 1.0136 | 0.0019 | 0.0062 vs 0.0500 | equivalent |
| two-time-point law with monotone censoring | `ey_regimen[treat then continue if l2 positive]` | mean outcome under the plan treat, then continue only if L2 is positive | 0.000002 | 0.000578 | 1.0214 | -0.000625 | 0.0133 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `cross_fitted_ltmle` | positive | five-fold end-of-study LTMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9961 to 1.0370 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.3463 to 0.3603 | pass |
| `double_robustness` | `dynamic__both_correct` | positive | dynamic plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0022 to 0.0015, margin 0.0063, SE ratio 0.9987 | pass |
| `double_robustness` | `dynamic__both_wrong` | control | dynamic plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0165 to 0.0206, margin 0.0070, SE ratio 0.9998 | pass |
| `double_robustness` | `dynamic__mechanism_correct` | positive | dynamic plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0018 to 0.0022, margin 0.0067, SE ratio 0.9913 | pass |
| `double_robustness` | `dynamic__outcome_correct` | positive | dynamic plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0021 to 0.0016, margin 0.0063, SE ratio 1.0220 | pass |
| `double_robustness` | `static__both_correct` | positive | static plan: both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0059 to 0.0027, margin 0.0145, SE ratio 1.0147 | pass |
| `double_robustness` | `static__both_wrong` | control | static plan: both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0248 to -0.0162, margin 0.0144, SE ratio 0.6872 | pass |
| `double_robustness` | `static__mechanism_correct` | positive | static plan: only the treatment and censoring mechanisms are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0062, margin 0.0145, SE ratio 1.0325 | pass |
| `double_robustness` | `static__outcome_correct` | positive | static plan: only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0063 to 0.0029, margin 0.0154, SE ratio 0.6173 | pass |
| `interval_calibration` | `dynamic__correctly_specified` | positive | dynamic plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9365 to 0.9600, SE ratio 0.9812 to 1.0622, empirical efficiency ratio 0.9553 to 1.0332, reported efficiency ratio 1.0118 to 1.0174 | pass |
| `interval_calibration` | `dynamic__noise_control` | control | dynamic plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8124 to 0.8521, SE ratio 0.6950 to 0.7474, empirical efficiency ratio 1.3579 to 1.4592, reported efficiency ratio 1.0117 to 1.0174 | pass |
| `interval_calibration` | `dynamic__shrunken_se_control` | control | dynamic plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8172 to 0.8564, SE ratio 0.6877 to 0.7440, empirical efficiency ratio 0.9546 to 1.0320, reported efficiency ratio 0.7082 to 0.7122 | pass |
| `interval_calibration` | `static__correctly_specified` | positive | static plan: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9301 to 0.9548, SE ratio 0.9690 to 1.0444, empirical efficiency ratio 0.9777 to 1.0536, reported efficiency ratio 1.0184 to 1.0242 | pass |
| `interval_calibration` | `static__noise_control` | control | static plan: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8081 to 0.8481, SE ratio 0.6897 to 0.7431, empirical efficiency ratio 1.3749 to 1.4805, reported efficiency ratio 1.0185 to 1.0241 | pass |
| `interval_calibration` | `static__shrunken_se_control` | control | static plan: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8020 to 0.8425, SE ratio 0.6783 to 0.7314, empirical efficiency ratio 0.9785 to 1.0539, reported efficiency ratio 0.7129 to 0.7169 | pass |
| `power` | `static__alternative` | positive | static plan: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 0.9625, 0.9416 to 0.9776 | pass |
| `root_n_and_efficiency` | `dynamic__n_1000` | control | dynamic plan: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0014, coverage 0.9223 to 0.9647, SE ratio 0.9989 | pass |
| `root_n_and_efficiency` | `dynamic__n_2000` | positive | dynamic plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000324, coverage 0.9194 to 0.9627, SE ratio 0.9977 | pass |
| `root_n_and_efficiency` | `dynamic__n_8000` | positive | dynamic plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000035, coverage 0.9341 to 0.9727, SE ratio 1.0358 | pass |
| `root_n_and_efficiency` | `static__n_1000` | control | static plan: bias, coverage and SE calibration at n = 1,000 | coverage interval lies below nominal or clears the declared floor | bias -0.0036, coverage 0.9078 to 0.9544, SE ratio 0.9939 | pass |
| `root_n_and_efficiency` | `static__n_2000` | positive | static plan: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.0036, coverage 0.9356 to 0.9737, SE ratio 1.0217 | pass |
| `root_n_and_efficiency` | `static__n_8000` | positive | static plan: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000622, coverage 0.9252 to 0.9667, SE ratio 1.0042 | pass |
| `root_n_rate` | `dynamic__empirical_sd` | positive | dynamic plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5726 to -0.4852 | pass |
| `root_n_rate` | `dynamic__reported_se` | positive | dynamic plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5138 to -0.5073 | pass |
| `root_n_rate` | `static__empirical_sd` | positive | static plan: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5636 to -0.4759 | pass |
| `root_n_rate` | `static__reported_se` | positive | static plan: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5199 to -0.5132 | pass |
| `targeting_necessity` | `dynamic__targeted` | positive | dynamic plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0021 to 0.0018, margin 0.0065 | pass |
| `targeting_necessity` | `dynamic__untargeted` | control | dynamic plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias 0.0232 to 0.0274, margin 0.0070 | pass |
| `targeting_necessity` | `static__targeted` | positive | static plan: the estimator fluctuates a misspecified outcome model, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0040 to 0.0052, margin 0.0153 | pass |
| `targeting_necessity` | `static__untargeted` | control | static plan: the identical fit with every fluctuation step removed | bias interval must fall entirely outside the margin | bias -0.0263 to -0.0173, margin 0.0151 | pass |
| `type_i_error` | `static__sharp_null` | positive | static plan: a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0450, 0.0283 to 0.0674 | pass |
<!-- /generated -->

The property study preserves the ordinary row's double-robustness, rate, efficiency, calibration,
null, power, and targeting instruments. It runs each positive estimator with five outer folds.

The overfitting pair uses identical nonlinear panels and a fully grown outcome tree. The positive
arm predicts held-out rows. The control predicts its training rows. The joint verdict requires an
SE ratio inside the declared band and a predeclared coverage gain over the control.

Read the ratio as well as the verdict. In-sample fitting understates the standard error by a factor
near three, at 0.3532. Cross-fitting removes the understatement. The cross-fitted ratio is 1.0161,
and its 99% interval runs from 0.996092 to 1.036997 against the 1.2000 ceiling.

The cell establishes
that cross-fitting restores honest inference. It does not establish calibration under a fully
grown tree. The `interval_calibration` family makes the calibration claim, and it uses correctly
specified nuisances.

The fold-fluctuated construction that the pooled update replaced gave a ratio of 1.176650 on the
same cell. Its 99% interval reached 1.201555, so the cell missed the ceiling by 0.001555. The two
runs share the law, the learner, the seeds, the budget, and the margins.
[RM18](../../roadmap.md#rm18-red-property-cells-after-the-fold-scale-and-law-changes) records the
before and after values. Its
[code-by-runtime diagnostic](../../roadmap.md#what-the-runtime-isolation-found) attributes the
verdict change to the pooled code, not to the Python and SciPy change.

The single-fold control has no outer split, and its ratio moved only in the sixth decimal, from
0.353193 to 0.353196. In the RM18 diagnostic, both code states give 0.353193 under Python 3.11.13
and 0.353196 under Python 3.13.7.

The replication budget stays at 8,000 rather than the shared 400. That budget was declared before
the first run of this study, and the pooled update ran at it unchanged. The study keeps the
`reporting` policy that the registry declares for it. Every verdict now passes, and a move to
`gated` is a separate registry change that this regeneration did not make.

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
| `property_cells_passed` | 32 | property cells passing |
| `max_standardized_bias` | 0.0324 | largest primary standardized bias |
| `min_coverage` | 0.9431 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9265 | lowest primary coverage lower endpoint |
| `min_se_ratio_ci_lower` | 0.9393 | lowest primary SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0649 | highest primary SE-ratio endpoint |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:coverage` | 0.9510 | cross-fitted tree coverage |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.5081 | in-sample tree coverage |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:se_ratio_ci_upper` | 1.036997 | cross-fitted tree SE-ratio upper endpoint |
| `properties[crossfit_overfitting/cross_fitted_ltmle]:coverage_gain_ci_lower` | 0.4281 | lower bound for the paired coverage gain |
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
| Agreement with `lmtp` is distributional, not numerical | The paired claim is that the mean difference sits inside the similarity margin and that `cleverly` is no worse. The two ordinary rows agree with R `ltmle` to solver precision because both run the identical regression. Here the sequential regressions are fitted differently and targeted differently, so per-replication estimates differ at statistical scale |
| One fixed five-fold assignment is studied | The row does not validate repeated folds or time-respecting splits |
| The pinned comparator runs a different construction | `lmtp` 1.5.4 fits a fluctuation on each fold's training rows and carries it into that fold's next regression. `cleverly` runs one pooled fluctuation per node. Upstream commit `9996b04` classifies the 1.5.4 update as a bug. The paired margins stay as registered, so each paired verdict compares two constructions and validates neither fold-local update |
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
