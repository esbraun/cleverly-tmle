# Stacked missing-outcome natural-course CV-TMLE

This study validates the narrow RM9 estimator for the observed outcome mean under MAR: one
package-generated unstratified ten-fold split, respondent-only outcome-regression fitting on each
training complement, a separately fitted response mechanism, one pooled logistic fluctuation, and
whole-sample plug-in and influence-curve evaluation. The primary rows fit separate data-adaptive
classification trees for both nuisances through the public `TMLEMethod` and `CrossFitting` API.

No canonical implementation is compared. The maintained R `tmle` 2.1.1 API was rejected because
its MAR path reports intervention-arm means; `tmle3` was rejected because no maintained callable
task was identified for this exact missing-outcome pooled stacked natural-course construction; and
zEpid was rejected because its cross-fit TMLE targets treatment contrasts. The committed
`equivalence.csv` therefore has zero rows. Díaz, Carone and van der Laan (2016), Section 2,
Equations (1)–(5), supplies the natural-course MAR functional, efficient influence curve, targeting
update, and two-nuisance remainder; Levy (2018), Section 3.1, supplies the stacked CV-TMLE
construction and whole-sample influence-curve variance.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome observational natural-course law with learned MAR nuisances | `ey_obs` | observed outcome mean under the natural course | `cleverly` stacked missing-outcome natural-course CV-TMLE | -0.0018 to 0.0014 | 0.9537 | 0.9943 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

There is no canonical comparison for this exact construction. The committed `equivalence.csv` is
empty and schema-valid.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.4918 to 0.5870 | pass |
| `crossfit_overfitting` | `stacked_mar_natural_course_cvtmle` | positive | stacked MAR natural-course CV-TMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9297 to 1.1005 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0156 to -0.0135, margin 0.0034, SE ratio 1.5963 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000595 to 0.0013, margin 0.0032, SE ratio 1.4842 | pass |
| `double_robustness` | `response_correct` | positive | only the outcome-observation mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0012 to 0.0019, margin 0.0053, SE ratio 0.9666 | pass |
| `interval_calibration` | `ey_obs__flexible_learning` | positive | observed outcome mean under the natural course: separate data-adaptive trees learn the outcome and response nuisances out of fold | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9422 to 0.9665, SE ratio 0.9807 to 1.0602 | pass |
| `interval_calibration` | `ey_obs__shrunken_se_control` | control | observed outcome mean under the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8126 to 0.8558, SE ratio 0.6864 to 0.7411 | pass |
<!-- /generated -->

The flexible-learning cell jointly checks empirical coverage and reported-SE calibration. Its
derived shrunken-SE arm verifies that the calibration instrument detects invalid inference. A
paired experiment fits the identical fully grown outcome tree on identical samples, changing only
whether its predictions are cross-fitted. The union-model cells separately make the outcome
regression or response mechanism correct, while the both-wrong control must retain detectable
bias.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 1 | truth tests passing |
| `independent_tests_total` | 1 | truth tests reported |
| `paired_tests_passed` | 0 | paired comparisons passing |
| `paired_tests_total` | 0 | paired comparisons reported |
| `property_cells_passed` | 7 | property cells passing |
| `property_cells_total` | 7 | property cells reported |
| `max_standardized_bias` | 0.0105 | largest primary standardized bias |
| `min_coverage` | 0.9537 | lowest primary coverage |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | nominal test size |
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
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:overfit_se_floor` | 0.8500 | cross-fit SE-ratio floor |
| `margin:overfit_control_ceiling` | 0.7500 | in-sample SE-ratio ceiling |
| `margin:overfit_coverage_gain` | 0.1500 | paired coverage-gain floor |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `bound:ey_obs_standard_error` | 0.0173 | exact binary-law EIF standard error at primary n |

## Limits

- The evidence covers one observational binary-treatment, binary-outcome MAR law with one
  three-level baseline covariate.
- It covers exactly one package-generated unstratified ten-fold draw, pooled targeting,
  whole-sample evaluation, pointwise Wald inference, and bounded nuisance predictions.
- The flexible primary learners are single classification trees. The study does not establish
  performance for every learner library or tuning procedure.
- The exact EIF bound describes the finite law, but the learned-nuisance calibration cell does not
  claim efficiency-bound attainment.
- There is no external parity claim; the audited candidates do not expose the same target and
  estimator construction.
- The study excludes repeated splits, stratified or supplied folds, fold targeting or evaluation,
  continuous or multinomial treatment, continuous outcomes, weights, clusters, baseline strata,
  intermediates, missing treatment, MNAR outcomes, and simultaneous intervals.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/replicates.csv.gz),
[performance decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/performance-tests.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/properties.csv)
carry the protocol, provenance, and every published row.
