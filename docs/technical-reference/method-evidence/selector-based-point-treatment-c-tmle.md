# Selector-based point-treatment C-TMLE

This study validates `cleverly`'s greedy, ordered, and discrete C-TMLE selectors. The canonical
comparison uses R [`ctmle`](https://github.com/jucheng1992/ctmle) 0.1.2 at pinned commit
[`18de559`](https://github.com/jucheng1992/ctmle/tree/18de559f47dc1286617350a0668391e80e1dbf7c).
That package is the maintained comparator for these selector entry points. It is not a tlverse
package, and the tlverse comparison applies only to the outcome-adaptive study below. The theory
is van der Laan and Gruber (2010).

## What was compared

| setting | `cleverly` | R `ctmle` |
| --- | --- | --- |
| strategies | `greedy`, `ordered`, `discrete` | `ctmleGeneral`, and `ctmleDiscrete(preOrder = TRUE)` for both ordered and discrete |
| datasets | binary-outcome samples generated in Python with their exact ATE | the identical rows and all three DGP covariates |
| selector folds | unstratified five-fold, taken off the `cleverly` fit that produced the subject row | the same assignment, asserted to be a partition that holds both arms before selecting |
| nuisance regressions | corresponding logistic GLM | corresponding logistic GLM |
| propensity bounds | 0.025 to 0.975 | 0.025 to 0.975 |
| selector loss | unpenalized | unpenalized |
| cross-fitting | disabled for the comparison | not applicable |

The fit is not cross-fitted, and it still draws a split. The selector scores its candidate path
over five selection folds. That split reads no label here, and the fit asserts the realized
partition. R selects against the same assignment, so the two sides compare one split rather than
two.

`cleverly`'s default penalty follows the published trace-plus-bias criterion. It is not presented
as numerical parity with R's implementation-specific adjustment. The property study exercises a
public nested cross-fit configuration instead: five outer folds, three selection folds, and two
inner folds, with the penalty on. Every property cell is cross-fitted, so every property cell
samples a proportion and declares an outcome support of 0 to 1.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, discrete selector | `ate` | average treatment effect | `cleverly` selector-based C-TMLE | -0.0021 to 0.0018 | 0.9463 | 0.9437 | pass |
| binary-outcome law, discrete selector | `ate` | average treatment effect | R `ctmle` | -0.0022 to 0.0018 | 0.9437 | 0.9427 | pass |
| binary-outcome law, greedy selector | `ate` | average treatment effect | `cleverly` selector-based C-TMLE | -0.0018 to 0.0021 | 0.9563 | 0.9571 | pass |
| binary-outcome law, greedy selector | `ate` | average treatment effect | R `ctmle` | -0.0018 to 0.0021 | 0.9475 | 0.9452 | pass |
| binary-outcome law, ordered selector | `ate` | average treatment effect | `cleverly` selector-based C-TMLE | -0.0010 to 0.0028 | 0.9400 | 0.9643 | pass |
| binary-outcome law, ordered selector | `ate` | average treatment effect | R `ctmle` | -0.0011 to 0.0028 | 0.9387 | 0.9596 | pass |
<!-- /generated -->

## Agreement with the canonical implementation

<!-- generated: agreement -->
| law | estimand | what was compared | paired difference | share of margin used | RMSE ratio bound | coverage difference | calibration resolution | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary-outcome law, discrete selector | `ate` | average treatment effect | 0.000055 | 0.0170 | 1.0069 | 0.0025 | 0.0052 vs 0.0500 | equivalent |
| binary-outcome law, greedy selector | `ate` | average treatment effect | 0.000039 | 0.0121 | 1.0059 | 0.0088 | 0.0218 vs 0.0500 | superior |
| binary-outcome law, ordered selector | `ate` | average treatment effect | 0.000034 | 0.0105 | 1.0034 | 0.0012 | 0.0118 vs 0.0500 | equivalent |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `both_correct` | positive | both the outcome regression and the treatment mechanism are correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000634 to 0.000925, margin 0.0026, SE ratio 0.9850 | pass |
| `double_robustness` | `both_wrong` | control | both nuisances are misspecified | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0072 to 0.0091, margin 0.0032, SE ratio 0.9940 | pass |
| `double_robustness` | `outcome_correct` | positive | only the outcome regression is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000559 to 0.000978, margin 0.0026, SE ratio 0.9937 | pass |
| `double_robustness` | `treatment_correct` | positive | only the treatment mechanism is correctly specified | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias 0.000508 to 0.0017, margin 0.0020, SE ratio 0.9670 | pass |
| `interval_calibration` | `correctly_specified` | positive | both nuisances are correctly specified | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9273 to 0.9526, SE ratio 0.9368 to 1.0070 | pass |
| `power` | `alternative` | positive | the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9868 to 1 | pass |
| `root_n_and_efficiency` | `n_2000` | positive | bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000184, coverage 0.9165 to 0.9606, SE ratio 0.9746 | pass |
| `root_n_and_efficiency` | `n_500` | positive | bias, coverage and SE calibration at n = 500 | bias inside the margin, coverage clears the floor, SE ratio inside the band | bias 0.000242, coverage 0.9326 to 0.9717, SE ratio 0.9954 | pass |
| `root_n_and_efficiency` | `n_8000` | positive | bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000224, coverage 0.9252 to 0.9667, SE ratio 0.9783 | pass |
| `root_n_rate` | `empirical_sd` | positive | log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5314 to -0.4616 | pass |
| `root_n_rate` | `reported_se` | positive | the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5043 to -0.5021 | pass |
| `selector_necessity` | `collaborative` | positive | the selector chooses its own mechanism path | bias interval inside the equivalence margin | bias 0.0015 to 0.0037, margin 0.0030, RMSE ratio 0.2410 | **fail** |
| `selector_necessity` | `empty_control` | control | the selector is forced to stop at an empty path | bias interval must fall entirely outside the margin | bias 0.0495 to 0.0511, margin 0.0022, RMSE ratio 0.2410 | pass |
| `type_i_error` | `sharp_null` | positive | a confounded law whose true contrast is exactly zero | one-sided rejection bound stays under the declared type-I ceiling | rejection 0.0700, 0.0412 to 0.1095 | **fail** |
<!-- /generated -->

## Measured values

Names beginning `margin:` are thresholds declared before the run. Everything else is measured from
the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 800 | replications per selector strategy |
| `n` | 2000 | observations per replication |
| `independent_tests_total` | 6 | implementation-strategy tests against truth |
| `independent_tests_passed` | 6 | of those, passing |
| `paired_tests_total` | 3 | paired selector-strategy tests |
| `paired_tests_passed` | 3 | of those, passing |
| `property_cells_total` | 14 | independent property cells |
| `property_cells_passed` | 11 | of those, passing |
| `max_standardized_bias` | 0.0429 | largest absolute primary bias in empirical standard deviations |
| `min_coverage` | 0.9387 | lowest primary coverage |
| `min_coverage_ci_lower` | 0.9136 | lowest exact 99% coverage endpoint |
| `min_se_ratio_ci_lower` | 0.8891 | lowest bootstrap SE-ratio endpoint |
| `max_se_ratio_ci_upper` | 1.0348 | highest bootstrap SE-ratio endpoint |
| `max_margin_utilization` | 0.0170 | largest share of the paired similarity margin used |
| `max_rmse_ratio_upper` | 1.0069 | largest paired RMSE-ratio upper bound |
| `min_coverage_difference_lower` | -0.0063 | smallest paired coverage-difference lower bound |
| `max_calibration_excess_upper` | 0.0100 | largest paired SE-calibration-excess upper bound |
| `properties[double_robustness/both_correct]:standardized_bias` | 0.0139 | both nuisances correct |
| `properties[double_robustness/outcome_correct]:standardized_bias` | 0.0203 | only the outcome nuisance correct |
| `properties[double_robustness/treatment_correct]:standardized_bias` | 0.1390 | only the treatment nuisance correct |
| `properties[double_robustness/treatment_correct]:n` | 2000 | observations that leg needs to resolve its remainder |
| `properties[double_robustness/both_wrong]:standardized_bias` | 0.6332 | both-wrong negative control |
| `properties[selector_necessity/collaborative]:rmse_ratio` | 0.2410 | collaborative RMSE divided by the empty-path control RMSE |
| `properties[selector_necessity/collaborative]:se_ratio` | 0.8011 | reported SE over empirical spread in that cell; see below |
| `properties[selector_necessity/collaborative]:standardized_bias` | 0.2173 | forced-selection bias, in empirical standard deviations |
| `properties[selector_necessity/collaborative]:bias_ci_upper` | 0.0037 | its 99% upper endpoint, against the margin below |
| `properties[selector_necessity/collaborative]:bias_margin` | 0.0030 | the equivalence margin that cell answers to |
| `properties[root_n_rate/empirical_sd]:slope` | -0.4969 | fitted empirical-spread rate |
| `properties[root_n_rate/empirical_sd]:slope_ci_lower` | -0.5314 | its 99% lower endpoint, against a band of -0.625 to -0.375 |
| `properties[root_n_rate/empirical_sd]:slope_ci_upper` | -0.4616 | its 99% upper endpoint, which must also exclude -0.25 |
| `properties[root_n_rate/reported_se]:slope` | -0.5032 | fitted reported-SE rate |
| `properties[interval_calibration/correctly_specified]:coverage` | 0.9408 | calibration-cell coverage |
| `properties[interval_calibration/correctly_specified]:se_ratio` | 0.9711 | calibration-cell SE ratio |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_lower` | 0.9368 | its 99% lower endpoint, against a band of 0.93--1.07 |
| `properties[interval_calibration/correctly_specified]:se_ratio_ci_upper` | 1.0070 | its 99% upper endpoint |
| `properties[type_i_error/sharp_null]:rejection_rate` | 0.0700 | rejection under the confounded sharp null |
| `properties[type_i_error/sharp_null]:rejection_ci_upper` | 0.1095 | its 99% upper endpoint, against 0.05 + 0.05 |
| `properties[type_i_error/sharp_null]:coverage` | 0.9300 | coverage under the confounded sharp null |
| `properties[type_i_error/sharp_null]:coverage_ci_lower` | 0.8905 | its exact 99% coverage endpoint, against a floor of 0.90 |
| `properties[power/alternative]:rejection_rate` | 1 | rejection under the power control |
| `margin:confidence_level` | 0.9900 | confidence level of Monte Carlo intervals |
| `margin:alpha` | 0.0500 | nominal estimator size |
| `margin:nominal_coverage` | 0.9500 | nominal estimator coverage |
| `margin:bootstrap_replicates` | 10000 | resamples per bootstrap interval |
| `margin:standardized_bias` | 0.2500 | bias equivalence margin in empirical standard deviations |
| `margin:coverage_floor` | 0.9000 | validity floor for the exact coverage lower endpoint |
| `margin:over_coverage_ceiling` | 0.9900 | coverage above this is labeled conservative |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio screen, lower limit |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio screen, upper limit |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio band, lower limit |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio band, upper limit |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage band, lower limit |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage band, upper limit |
| `margin:type_i_ceiling` | 0.1000 | one-sided type-I error ceiling |
| `margin:paired_difference` | 0.1500 | paired mean-difference margin in pooled SDs |
| `margin:rmse_noninferiority` | 1.1000 | paired RMSE-ratio upper limit |
| `margin:coverage_noninferiority` | -0.0250 | paired coverage-difference lower limit |
| `margin:calibration_noninferiority` | 0.0500 | paired calibration-excess upper limit |
| `margin:minimum_power` | 0.8000 | power-control rejection lower bound |
| `margin:root_n_slope` | -0.5000 | predicted root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | accepted slope band, lower limit |
| `margin:root_n_slope_upper` | -0.3750 | accepted slope band, upper limit |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:selector_rmse_ratio` | 0.5000 | maximum collaborative-to-empty RMSE ratio |

## Limitations

| limitation | what it means for use |
| --- | --- |
| Two of the three strategies reach the same R entry point | R `ctmle` has one pre-ordered selector, so `ordered` and `discrete` are both compared against `ctmleDiscrete(preOrder = TRUE)`. The correspondence is earned: the `discrete` candidate list is exactly the nested prefix ladder that mode enumerates. An arbitrary candidate list therefore has no reference here, and the row carries two reference constructions on three separate draws |
| The two sides select differently even where they agree | `cleverly` refits the outcome regression inside two nested folds within each selection fold. R scores every fold against one full-sample `Q`. Agreement here is evidence about the C-TMLE machinery on a law where the selector's choice is stable, not about the selection rule. That the search is load-bearing is established by `selector_necessity` and by the unit tests |
| No cell asks for calibrated inference while selection is load-bearing | The forced-selection cell claims the RMSE ratio and only that. Its reported standard error is `properties[selector_necessity/collaborative]:se_ratio` of the empirical spread. That ratio is below one, so the reported interval is anti-conservative here rather than conservative. It sits just above `margin:se_ratio_sanity_lower`, which is the screen the size cells answer to. This cell reports the ratio and does not gate on it. The earlier row reported 1.2539 on a treatment-stratified selection split. An unstratified split reconciles this cell with [RM12](../../roadmap.md#rm12-collaborative-intervals-at-an-inconsistent-working-mechanism)'s own probe of the instrument law, which measured a ratio of 0.844 and coverage of 0.92. [F18](../../roadmap.md#f18-selector-path-c-tmle-inference) records that no result derives an influence curve after the shipped stopping-index selection. `interval_calibration` is measured where both nuisances are correct and the search has nothing to do, so the gap is real |
| One robustness cell is sized differently from its siblings | `treatment_correct` runs at `properties[double_robustness/treatment_correct]:n` observations where the other three run at 700. It is the leg that leans on inverse weighting, so its `O(n^-1)` remainder is the largest of the four arms. The size was declared on the Gaussian linear law this row used before, where 700 left that remainder too close to the fixed margin to be told apart from first-order bias. It carries over to the bounded twin with the budget and the seed. It was not remeasured on the bounded law, because this is a positive cell and its own bias is the verdict statistic. The margin was not moved after seeing it |
| The forced-selection cell is red | This row publishes under the `reporting` policy, which was declared before the run that measured this cell. `selector_necessity/collaborative` reports a standardized bias of `properties[selector_necessity/collaborative]:standardized_bias`. Its 99% bias interval reaches `properties[selector_necessity/collaborative]:bias_ci_upper`, above the `properties[selector_necessity/collaborative]:bias_margin` the equivalence margin allows. The RMSE ratio still clears `margin:selector_rmse_ratio`, so the search is doing work on this law. What is not established is that the reported interval is valid while it does. [F18](../../roadmap.md#f18-selector-path-c-tmle-inference) records the missing result. `empty_control` passes its own control rule, and the family verdict is the joint one, so the family is red |
| The null-size cell is red | On the confounded bounded sharp null the rejection rate is `properties[type_i_error/sharp_null]:rejection_rate`. Its 99% upper endpoint is `properties[type_i_error/sharp_null]:rejection_ci_upper`, above the predeclared ceiling of `margin:type_i_ceiling`. Coverage is `properties[type_i_error/sharp_null]:coverage`, and its exact 99% lower endpoint is `properties[type_i_error/sharp_null]:coverage_ci_lower`, below the floor of `margin:coverage_floor`. The same law, size and budget are green in four committed cross-fitted TMLE rows, and red in every binary C-TMLE row that measured this null. That is a pattern across rows rather than a controlled experiment: this cell also uses five outer folds and its own seed stream. [F18](../../roadmap.md#f18-selector-path-c-tmle-inference) records the missing result. No margin, budget, size or seed moved after the run |
| The default ordering and two strategies lack their own property evidence | The `ordered` cell pins an explicit covariate order, so the default `preorder="logistic"` ordering is exercised by neither half of this row. The property cells all run the default `greedy` search, so `ordered` and `discrete` have parity evidence without repeated-sampling evidence |
| The property cells need a known outcome support | Every cell here is cross-fitted, so every cell draws a proportion and declares `q_bounds` of 0 to 1. The row says nothing about a continuous outcome whose support the analyst does not know, because the package refuses that composition under cross-fitting |
| One unstratified split does the selecting | The selection folds read neither the treatment nor the outcome, and the fit asserts the realized partition. The row does not establish a stratified split, a grouped split, or a supplied plan, and it does not establish that a stranded arm is impossible at another size |
| The parity claim is narrow | It is binary, two-arm, complete-outcome, GLM, non-cross-fitted, unpenalized ATE only. The row does not establish parity for the default penalty or nested cross-fitting, ratios or arm means, missing outcomes, weights, clusters, strata, multi-valued treatment, flexible learner libraries, simultaneous or bootstrap intervals, or severe practical-positivity behaviour |

## Reproduction

The [fixture README](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/README.md)
and [manifest](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/manifest.json)
record the provenance and the regeneration commands. The
[replications](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/replicates.csv.gz),
[paired decisions](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/equivalence.csv)
and [property results](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/ctmle_selector/properties.csv)
carry every published row.
