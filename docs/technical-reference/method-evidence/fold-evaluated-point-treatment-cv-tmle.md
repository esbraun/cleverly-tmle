# Fold-evaluated point-treatment CV-TMLE

This study validates `cleverly`'s fold-evaluated CV-TMLE report: unstratified ten-fold
nuisance fitting, one pooled targeting update, equal-fold plug-in evaluation, and the
cross-validated influence-curve variance. It is separate from the stacked row above because
averaging fold reports, rather than evaluating the updated regression over the whole sample, is a
genuine finite-sample method choice. The construction and its source boundary are mapped in
[CV-TMLE and cross-fitting](../cv-tmle.md#the-algorithm-as-implemented).

**No canonical implementation is compared.** No maintained package pairs a pooled targeting update
with fold plug-in evaluation. The study rests on the accuracy and theory-property questions
alone. A zero-row equivalence artifact records that absence rather than borrowing the stacked R
comparison. Python `zEpid` targets inside each fold, which is a different estimator. The separate
[fold-targeted study](fold-targeted-point-treatment-cv-tmle.md) compares that construction.

## What was compared

| setting | `cleverly` |
| --- | --- |
| construction | ten-fold nuisance fitting, pooled update, equal-fold plug-in evaluation |
| variance | cross-validated influence curve |
| estimands | `ey1`, `ey0`, `ate`, `att`, `atc` |
| laws | the same binary and bounded-continuous laws the ordinary TMLE study uses |
| nuisance learners | corresponding logistic and linear GLM |
| folds | unstratified ten-fold, drawn from the estimator's own seed |
| continuous outcome scale | declared `q_bounds` of 0 to 1 |
| binary outcome scale | identity, and no declaration |
| propensity bounds | 0.025 to 0.975 |
| intervals | pointwise 95% identity-scale Wald |

The split reads neither the treatment nor the outcome. A split that reads either one makes each
fold's training rows a function of the rows it predicts. The fit asserts the realized scheme
rather than trusting the keyword it was given.

The continuous law draws a proportion, so 0 to 1 is the law's own support and the study declares
it. Inferring the scale from every observed outcome would make each fold's training predictions
depend on the rows they are predicting, so the package refuses a cross-fitted continuous outcome
that declares none. The binary law needs no declaration, because a binary outcome already sits on
the unit interval.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law | `atc` | average effect on the untreated | `cleverly` fold-evaluated CV-TMLE | -0.000783 to 0.0032 | 0.9506 | 1.0098 | pass |
| binary-outcome law | `ate` | average treatment effect | `cleverly` fold-evaluated CV-TMLE | -0.0012 to 0.0027 | 0.9506 | 1.0015 | pass |
| binary-outcome law | `att` | average effect on the treated | `cleverly` fold-evaluated CV-TMLE | -0.0017 to 0.0024 | 0.9556 | 1.0200 | pass |
| binary-outcome law | `ey0` | counterfactual mean under no treatment | `cleverly` fold-evaluated CV-TMLE | -0.0019 to 0.000956 | 0.9537 | 1.0128 | pass |
| binary-outcome law | `ey1` | counterfactual mean under treatment | `cleverly` fold-evaluated CV-TMLE | -0.0011 to 0.0017 | 0.9550 | 1.0157 | pass |
| bounded continuous-outcome law with effect modification | `atc` | average effect on the untreated | `cleverly` fold-evaluated CV-TMLE | -0.000291 to 0.000628 | 0.9513 | 1.0021 | pass |
| bounded continuous-outcome law with effect modification | `ate` | average treatment effect | `cleverly` fold-evaluated CV-TMLE | -0.000257 to 0.000598 | 0.9544 | 1.0053 | pass |
| bounded continuous-outcome law with effect modification | `att` | average effect on the treated | `cleverly` fold-evaluated CV-TMLE | -0.000242 to 0.000687 | 0.9425 | 0.9872 | pass |
| bounded continuous-outcome law with effect modification | `ey0` | counterfactual mean under no treatment | `cleverly` fold-evaluated CV-TMLE | -0.000716 to 0.000190 | 0.9531 | 0.9961 | pass |
| bounded continuous-outcome law with effect modification | `ey1` | counterfactual mean under treatment | `cleverly` fold-evaluated CV-TMLE | -0.000610 to 0.000425 | 0.9531 | 1.0253 | pass |
<!-- /generated -->

## Theory properties

Every cell here samples an outcome in the open interval 0 to 1, drawn as a beta variate around
the law's conditional mean. A proportion has a known support, so each cell declares `q_bounds` of
0 to 1 and the outcome scaler is the identity. Each law is a bounded twin of the Gaussian law the
ordinary point-treatment row samples, and it keeps that law's treatment mechanism unchanged.
`tests/studies/bounded_cv_laws.py` records the pilot measurement behind every declared constant.

The double-robustness cells use a bounded nonlinear confounded law with exact ATE 0.2368749. Its
treatment mechanism stays between 0.076 and 0.924, so the configured bounds do not clip it. Its
conditional mean stays between 0.088 and 0.924 on the quadrature grid, so both beta shapes stay
away from zero.

The wrong main-effects outcome regression imposes a constant contrast, while the true contrast
varies with `W1` and `I(W2 > 0)`. The treatment-correct cell uses n = 2,000. The other three
cells use n = 700. Each cell uses 1,200 replications and the existing predeclared margin.

The root-n ladder and the calibration cell sample a bounded linear law with exact ATE 0.1072047.
A quasibinomial solver recovers that law's six coefficients to 1.4e-15 on a quadrature grid, so
both nuisances are correctly specified there.

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `fold_evaluated_cvtmle` | positive | fold-evaluated CV-TMLE with a flexible learner | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9349 to 1.1226 | pass |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.4453 to 0.5382 | pass |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000589 to 0.0011, margin 0.0029, SE ratio 0.9670 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0358 to -0.0335, margin 0.0040, SE ratio 1.0115 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000840 to 0.000725, margin 0.0026, SE ratio 1.0505 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000825 to 0.000659, margin 0.0025, SE ratio 0.9719 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9346 to 0.9585, SE ratio 0.9660 to 1.0419 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000046, coverage 0.9121 to 0.9575, SE ratio 0.9712 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias -0.000300, coverage 0.9401 to 0.9767, SE ratio 1.0205 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000056, coverage 0.9238 to 0.9657, SE ratio 0.9997 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5330 to -0.4702 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5096 to -0.5073 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0400, 0.0191 to 0.0725 | pass |
<!-- /generated -->

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1600 | replications per law |
| `n` | 1000 | observations per replication |
| `independent_tests_total` | 10 | estimand-law tests against truth |
| `independent_tests_passed` | 10 | of those, passing |
| `paired_tests_total` | 0 | external comparisons declared |
| `paired_tests_passed` | 0 | external comparisons passing |
| `property_cells_total` | 14 | repeated-sampling property cells |
| `property_cells_passed` | 14 | of those, passing |
| `max_standardized_bias` | 0.0393 | largest absolute bias in empirical standard deviations |
| `min_coverage` | 0.9425 | lowest measured primary-study coverage |
| `min_coverage_ci_lower` | 0.9258 | lowest exact 99% coverage endpoint, against 0.90 |
| `min_se_ratio_ci_lower` | 0.9405 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0752 | highest bootstrap SE-ratio endpoint |
| `properties[double_robustness/outcome_correct]:bias` | -0.000057 | bias with only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:bias` | -0.000083 | bias with only the treatment nuisance correct |
| `properties[double_robustness/both_wrong]:bias` | -0.0347 | both-wrong negative control |
| `properties[root_n_rate/empirical_sd]:slope` | -0.5011 | fitted log-log sampling-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5330 | its 99% lower endpoint |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4702 | its 99% upper endpoint |
| `properties[root_n_rate/reported_se]:slope` | -0.5085 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 1.0022 | SE calibration where both nuisances are correct |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9660 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0419 | its 99% upper endpoint |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9475 | coverage of the same cell |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0400 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.0725 | its 99% upper endpoint, against 0.10 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the positive control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage` | 0.9325 | coverage with cross-fitted tree predictions |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:se_ratio` | 1.0193 | SE calibration with cross-fitting |
| `properties[crossfit_overfitting/in_sample_control]:coverage` | 0.4875 | coverage with the deliberately in-sample tree |
| `properties[crossfit_overfitting/in_sample_control]:se_ratio` | 0.4874 | SE calibration of that control |
| `properties[crossfit_overfitting/fold_evaluated_cvtmle]:coverage_gain_ci_lower` | 0.3775 | paired 99% lower bound for coverage gained over the control |
| `margin:confidence_level` | 0.9900 | confidence level of every Monte Carlo interval below |
| `margin:alpha` | 0.0500 | nominal size of the estimator's own intervals |
| `margin:nominal_coverage` | 0.9500 | nominal coverage those intervals claim |
| `margin:bootstrap_replicates` | 10000 | resamples behind every bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin, in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor the exact coverage lower endpoint must clear |
| `margin:over_coverage_ceiling` | 0.9900 | above this, coverage is conservative rather than invalid |
| `margin:se_ratio_sanity_lower` | 0.8000 | SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration-cell SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration-cell SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration-cell coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration-cell coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | largest size the one-sided type-I bound may establish |
| `margin:paired_difference` | 0.1500 | paired similarity margin, in pooled empirical standard deviations |
| `margin:rmse_noninferiority` | 1.1000 | largest RMSE ratio the one-sided upper bound may reach |
| `margin:coverage_noninferiority` | -0.0250 | smallest coverage difference the one-sided lower bound may reach |
| `margin:calibration_noninferiority` | 0.0500 | largest excess SE-calibration error the upper bound may reach |
| `margin:minimum_power` | 0.8000 | rejection lower bound the power control must clear |
| `margin:root_n_slope` | -0.5000 | the contraction rate root-n asymptotics predict |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | the slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:overfit_se_floor` | 0.8500 | SE ratio the cross-fit arm must restore |
| `margin:overfit_control_ceiling` | 0.7500 | ceiling the in-sample control's upper bound must stay below |
| `margin:overfit_coverage_gain` | 0.1500 | coverage cross-fitting must buy over the in-sample control |

## Limitations

| limitation | what it means for use |
| --- | --- |
| There is no cross-implementation evidence | The row rests on accuracy against known truth and on the theory properties. It is not parity evidence for stacked R CV-TMLE, and it does not inherit the stacked row's comparison |
| The cross-fit overfitting cells are relative evidence | A fully grown regression tree is fitted on the nonlinear law twice, once with held-out predictions and once in sample, on the identical 400 samples of size 500. The cross-fitted cell's evidence is restored SE calibration and a load-bearing improvement over the control. The primary GLM study carries the absolute coverage gate |
| The continuous evidence needs a known outcome support | Every continuous cell here draws a proportion and declares `q_bounds` of 0 to 1. The row says nothing about a continuous outcome whose support the analyst does not know, because the package refuses that composition under cross-fitting |
| The row is bounded to one unstratified ten-fold assignment per sample | It does not establish repeated or nested cross-fitting, a fold-specific targeting epsilon, simultaneous or bootstrap intervals, missing outcomes, weights, clusters, multi-valued treatment, ratio estimands, observed-risk functionals, or behaviour under severe practical-positivity violations |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/README.md)
gives the regeneration command. The
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/manifest.json)
records the primary and control samples, every margin and seed, the exact estimator configuration,
and the source and result hashes. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/replicates.csv.gz)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/cvtmle_fold/properties.csv)
carry every published row.
