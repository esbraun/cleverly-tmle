# Stacked missing-outcome natural-course CV-TMLE

This study validates the RM9 estimator of the observed outcome mean under MAR. The table gives the
studied construction. The primary rows declare it through the public `TMLEMethod` and
`CrossFitting` API.

| component | studied setting |
| --- | --- |
| partition | one package-generated, unstratified ten-fold split |
| outcome regression | fitted on the respondents in each training complement |
| response mechanism | fitted separately on each training complement |
| targeting | one pooled logistic fluctuation over the stacked held-out rows |
| evaluation | plug-in and influence curve over the whole sample |
| primary learners | separate classification trees with depth five and minimum leaf size 25 |

The law has six `(A, W)` covariate cells. The smallest respondent cell has an expected 72 rows in
an 1,800-row training complement, which exceeds the minimum leaf size. Each depth-five tree can
therefore fit the saturated model, and that model is correct for this law. The primary and
flexible-learning rows thus do not test calibration under a misspecified data-adaptive learner.
The paired overfitting arm supplies the data-adaptive evidence. It fits a fully grown outcome tree
with eight noise covariates beside the exact response mechanism.

The study compares R `tmle` 2.1.1 through its population-mean path. The adapter supplies the same
stitched out-of-fold outcome and response predictions that `cleverly` targets. It passes a
constant synthetic treatment and duplicates each realized-arm prediction across the two required
input columns. The original treatment remains beside `W`, preserving the conditioning set of the
supplied predictions. R then fits one pooled respondent-only fluctuation with clever covariate
`1 / pi` and reports `EY1` as the population mean.

This comparison is conditional on the supplied nuisance predictions. It validates the pooled
targeting, plug-in, and influence-curve path. It does not validate R-native cross-fit training.
The table gives the status of the other candidates. [`docs/references.md`](../../references.md)
records each source locator.

| candidate | disposition |
| --- | --- |
| R `tmle` 2.1.1 | compared through its population-mean path with the same stitched out-of-fold outcome and response predictions |
| `tmle3` at commit `ed72f8a` | its generic treatment-specific outcome fit can use the `Delta = 0` pseudo-outcomes when it predicts under `Delta = 1`. `cleverly` fits that regression on respondents only |
| zEpid 0.9.1 | its cross-fit TMLE targets inside each fold rather than with one pooled fluctuation |
| Newey and Robins (2018) | the construction fits the outcome and inverse response regressions on distinct subsamples. That is a different estimator |

| source | locator | what it supplies |
| --- | --- | --- |
| Díaz, Carone and van der Laan (2016) | Section 2, Equations (1)–(5) | the MAR natural-course functional, efficient influence curve, targeting update, and two-nuisance remainder |
| Levy (2018) | abstract | the stacked validation update and the whole-sample plug-in |
| Levy (2018) | Section 3.1 | the asymptotic expansion after the pooled score is solved, which supports the influence-curve interval |
| Zheng and van der Laan (2011) | Sections 2 and 2.1 | the training-complement nuisance fits and the untargeted empirical-distribution component |
| Zheng and van der Laan (2011) | Theorem 2 | the partial-targeting expansion and its conditions |

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome observational natural-course law with learned MAR nuisances | `ey_obs` | observed outcome mean under the natural course | `cleverly` stacked missing-outcome natural-course CV-TMLE | -0.0018 to 0.0014 | 0.9537 | 0.9943 | pass |
| binary-outcome observational natural-course law with learned MAR nuisances | `ey_obs` | observed outcome mean under the natural course | R `tmle` population-mean path | -0.0018 to 0.0014 | 0.9537 | 0.9946 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome observational natural-course law with learned MAR nuisances | `ey_obs` | observed outcome mean under the natural course | -1.770e-11 | 6.736e-09 | 1.0000 | 0 | 0.000001 vs 0.0500 | equivalent |
<!-- /generated -->

R `tmle` uses the centered, `n - 1` divisor sample variance `var(IC) / n`. `cleverly` uses the
uncentered second moment `mean(IC^2) / n`. The two rules differ by the finite-sample factor
`n / (n - 1)` when the targeted score is zero. R also bounds its binary interval to `[0, 1]`,
while `cleverly` reports an unbounded Wald interval. This law stays away from the bounds.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `crossfit_overfitting` | `in_sample_control` | control | the same flexible learner fitted in sample, with no cross-fitting | SE ratio must fall below the overfitting ceiling | SE ratio 0.4918 to 0.5870 | pass |
| `crossfit_overfitting` | `stacked_mar_natural_course_cvtmle` | positive | stacked MAR natural-course CV-TMLE with a fully grown outcome tree | SE ratio clears the overfitting floor and stays inside the sanity band | SE ratio 0.9297 to 1.1005 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias -0.0156 to -0.0135, margin 0.0034, SE ratio 1.5963 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000595 to 0.0013, margin 0.0032, SE ratio 1.4842 | pass |
| `double_robustness` | `response_correct` | positive | only the outcome-observation mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0012 to 0.0019, margin 0.0053, SE ratio 0.9666 | pass |
| `interval_calibration` | `ey_obs__flexible_learning` | positive | observed outcome mean under the natural course: separate depth-five trees fit the outcome and response nuisances out of fold; on this six-cell law each tree fits the saturated, correct model | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9422 to 0.9665, SE ratio 0.9807 to 1.0602 | pass |
| `interval_calibration` | `ey_obs__shrunken_se_control` | control | observed outcome mean under the natural course: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8126 to 0.8558, SE ratio 0.6864 to 0.7411 | pass |
<!-- /generated -->

The flexible-learning cell checks empirical coverage and reported-SE calibration together. Its
trees fit the correct saturated model, as the first section states. The derived shrunken-SE arm
multiplies each reported standard error by 0.70 and must fall below the calibration band.

The paired overfitting experiment is the data-adaptive check. It draws 400 samples of 500 rows and
adds eight noise covariates to each. Both arms fit the same fully grown outcome tree and the exact
response mechanism on each sample. Only cross-fitting differs between the arms. Coverage is 0.9500
with cross-fitting and 0.6950 in sample.

The union-model cells make either the outcome regression or the response mechanism correct. The
both-wrong control passes on bias discrimination. Its 99% bias interval, -0.0156 to -0.0135, lies
outside the 0.0034 margin. Its standardized bias is -1.0593. Its interval coverage stays at 0.9858,
because its reported standard error is 1.5963 times the empirical spread. The control therefore
shows detectable bias rather than an interval failure.

## Measured values and declared margins

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | primary replications |
| `n` | 2000 | observations per primary replication |
| `independent_tests_passed` | 2 | truth tests passing |
| `independent_tests_total` | 2 | truth tests reported |
| `paired_tests_passed` | 1 | paired comparisons passing |
| `paired_tests_total` | 1 | paired comparisons reported |
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
- The primary and flexible-learning rows use depth-five trees that fit the saturated, correct
  model on this six-cell law. Those rows do not test calibration under nuisance misspecification.
  The overfitting arm is the data-adaptive evidence, and it uses the exact response mechanism. The
  study does not establish performance for every learner library or tuning procedure.
- The exact EIF bound describes the finite law. The flexible-learning cell does not claim
  efficiency-bound attainment.
- The external comparison conditions on the stitched nuisance predictions from `cleverly`. It
  does not compare nuisance training implementations.
- The R variance and interval conventions differ as the agreement section states. The study does
  not claim bitwise interval equality.
- The study excludes repeated splits, stratified or supplied folds, and fold targeting or
  evaluation. It excludes continuous or multinomial treatment, continuous outcomes, weights,
  clusters, and baseline strata. It also excludes intermediates, missing treatment, MNAR outcomes,
  and simultaneous intervals.

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/README.md),
[manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/manifest.json),
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/replicates.csv.gz),
[performance decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/performance-tests.csv),
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/tmle_mar_natural_course_cvtmle/properties.csv)
carry the protocol, provenance, and every published row.
